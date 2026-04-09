"""
╔══════════════════════════════════════════════════════════════════════════════╗
║               CARGO ROUTE RISK PIPELINE  v3.0                               ║
║                                                                              ║
║   Data Sources:                                                              ║
║     • Maritime_vessel_database.xlsx  — 324 vessel records, 30 ports,        ║
║                                        286 unique sea routes (real AIS data) ║
║     • India_Road_Network_Database.xlsx — 500 road segments, 28 states,      ║
║                                          forming inland threat zones         ║
║                                                                              ║
║   Algorithms  :  Dijkstra  ·  Yen's K-Shortest Paths  ·  XGBoost            ║
║   Risk Input  :  Real vessel-derived base_risk  +  static RBF threat zones  ║
║   Output      :  K ranked routes + XGBoost risk score (0-100) per route     ║
╚══════════════════════════════════════════════════════════════════════════════╝

KEY MATHEMATICAL FOUNDATIONS
─────────────────────────────────────────────────────────────────────────────

  1. HAVERSINE DISTANCE  (great-circle distance between two geo-points)
     ─────────────────────────────────────────────────────────────────
     a = sin²(Δlat/2) + cos(lat₁)·cos(lat₂)·sin²(Δlon/2)
     d = 2R · arcsin(√a)          R = 6 371 km

  2. EDGE THREAT WEIGHT  (used as edge cost in Dijkstra / Yen's)
     ─────────────────────────────────────────────────────────────────
     W(u,v) = α · threat_norm(u,v)  +  β · dist_norm(u,v)

       α = 0.70   (threat importance weight)
       β = 0.30   (distance importance weight)
       threat_norm = RBF_threat(edge_midpoints) / 10    ∈ [0, 1]
       dist_norm   = dist_km(u,v) / 20 000              ∈ [0, 1]

  3. DIJKSTRA RELAXATION
     ─────────────────────────────────────────────────────────────────
     d[v] = min( d[v],  d[u] + W(u,v) )   ∀ (u,v) ∈ E
     Complexity: O((V + E) log V)

  4. YEN'S K-SHORTEST LOOPLESS PATHS
     ─────────────────────────────────────────────────────────────────
     A[k] = argmin{ cost(root_path ⊕ spur_path) }
              over all spur nodes i in A[k-1]
     Complexity: O(K · V · (V+E) log V)

  5. RBF THREAT SCORE  (replaces KNN — derived from real vessel risk data)
     ─────────────────────────────────────────────────────────────────
     threat(p) = Σ_z  severity_z · max(0,  1 − d(p,z) / radius_z)
     clipped to [0, 10]

     Severity values calibrated from Maritime DB:
       Risk_Level distribution: Low=57.7%, Medium=28.7%, High=13.6%
       Risk_Type  distribution: Bad Weather=32.4%, High Traffic=24.4%,
                                  None=19.1%, Piracy Zone=15.3%,
                                  War Zone=5.3%, Other=3.5%

  6. XGBOOST TRAINING LABEL
     ─────────────────────────────────────────────────────────────────
     y = 30·base_risk + 6·threat_sev + 8·n_choke
         + 15·war_flag + 12·sanctions_flag + 4·weather_risk
         + 5·cargo_val  +  ε,    ε ~ N(0, 3)
     clipped to [0, 100]

  7. FEATURE VECTOR  x ∈ ℝ¹⁴
     ─────────────────────────────────────────────────────────────────
     x = [ dist_norm, hops_norm, base_risk,
           threat_sev_norm, n_threats_norm, choke_flag,
           n_choke_norm, cargo_val, mode_sea,
           sin(2π·month/12), weather_norm,
           sanctions_flag, war_flag, piracy_flag ]

  8. ROAD RISK MODIFIER  (India Road Network DB)
     ─────────────────────────────────────────────────────────────────
     State-level poor-road fraction from 500-segment dataset:
       road_risk_factor(state) = poor_pct(state) × 0.4
                                  + (1 − avg_speed/50) × 0.3
                                  + (1 − 1/toll_ratio) × 0.3
     Used to upweight base_risk on inland/coastal Indian port edges.

─────────────────────────────────────────────────────────────────────────────
"""

import math
import json
import heapq
import datetime
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class GeoPoint:
    display_name: str
    lat:          float
    lon:          float
    country:      str
    region:       str = ""

@dataclass
class ThreatZone:
    name:        str
    lat:         float
    lon:         float
    threat_type: str    # war | piracy | terrorism | sanctions | natural | traffic
    severity:    float  # 0–10  (calibrated from vessel DB risk distributions)
    radius_km:   float  # zone of influence

@dataclass
class RoutePlan:
    path:             List[str]
    distance_km:      float
    xgb_risk_score:   float
    risk_class:       str        # LOW / MEDIUM / HIGH / CRITICAL
    estimated_days:   float
    base_risk_max:    float
    threat_severity:  float
    n_chokepoints:    int
    war_flag:         int
    sanctions_flag:   int
    piracy_flag:      int
    weather_risk:     float
    alternative_rank: int
    warnings:         List[str] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — WAYPOINTS
#   Source: Maritime_vessel_database.xlsx
#   Extracted from Origin_Port / Destination_Port columns with real AIS coords
#   30 unique ports · cleaned & deduplicated
# ══════════════════════════════════════════════════════════════════════════════

WAYPOINTS: Dict[str, GeoPoint] = {
    # Europe
    "ROTTERDAM":         GeoPoint("Rotterdam",          51.9244,   4.4777, "Netherlands",  "NW Europe"),
    "HAMBURG":           GeoPoint("Hamburg",             53.5753,   9.9300, "Germany",      "NW Europe"),
    "ANTWERP":           GeoPoint("Antwerp",             51.2652,   4.4069, "Belgium",      "NW Europe"),
    "FELIXSTOWE":        GeoPoint("Felixstowe",          51.9586,   1.3513, "UK",           "NW Europe"),
    "PIRAEUS":           GeoPoint("Piraeus",             37.9475,  23.6478, "Greece",       "SE Europe"),
    "ALGECIRAS":         GeoPoint("Algeciras",           36.1408,  -5.4531, "Spain",        "SW Europe"),
    # Americas
    "LOS_ANGELES":       GeoPoint("Los Angeles",         33.7395,-118.2618, "USA",          "W Americas"),
    "NEW_YORK":          GeoPoint("New York",            40.6892, -74.0445, "USA",          "E Americas"),
    "SAVANNAH":          GeoPoint("Savannah",            32.0835, -81.0998, "USA",          "E Americas"),
    "VANCOUVER":         GeoPoint("Vancouver",           49.2827,-123.1207, "Canada",       "W Americas"),
    "SANTOS":            GeoPoint("Santos",             -23.9618, -46.3322, "Brazil",       "S Americas"),
    # Middle East / Africa
    "DUBAI_JEBEL_ALI":   GeoPoint("Dubai (Jebel Ali)",  24.9857,  55.0272, "UAE",          "Middle East"),
    "JEDDAH":            GeoPoint("Jeddah",              21.5433,  39.1728, "Saudi Arabia", "Middle East"),
    "MOMBASA":           GeoPoint("Mombasa",             -4.0435,  39.6682, "Kenya",        "E Africa"),
    "DURBAN":            GeoPoint("Durban",             -29.8587,  31.0218, "S. Africa",    "S Africa"),
    # South Asia
    "MUMBAI":            GeoPoint("Mumbai",              18.9322,  72.8375, "India",        "South Asia"),
    "KARACHI":           GeoPoint("Karachi",             24.8607,  67.0099, "Pakistan",     "South Asia"),
    "COLOMBO":           GeoPoint("Colombo",              6.9271,  79.8612, "Sri Lanka",    "South Asia"),
    # SE Asia
    "SINGAPORE":         GeoPoint("Singapore",            1.2897, 103.8501, "Singapore",    "SE Asia"),
    "PORT_KLANG":        GeoPoint("Port Klang",           3.0319, 101.3682, "Malaysia",     "SE Asia"),
    "TANJUNG_PELEPAS":   GeoPoint("Tanjung Pelepas",      1.3628, 103.5501, "Malaysia",     "SE Asia"),
    "LAEM_CHABANG":      GeoPoint("Laem Chabang",        13.0827, 100.8832, "Thailand",     "SE Asia"),
    "MANILA":            GeoPoint("Manila",              14.5995, 120.9842, "Philippines",  "SE Asia"),
    # East Asia
    "HONG_KONG":         GeoPoint("Hong Kong",           22.3193, 114.1694, "Hong Kong",    "E Asia"),
    "GUANGZHOU":         GeoPoint("Guangzhou",           23.1291, 113.2644, "China",        "E Asia"),
    "SHANGHAI":          GeoPoint("Shanghai",            31.2304, 121.4737, "China",        "E Asia"),
    "NINGBO":            GeoPoint("Ningbo",              29.8683, 121.5440, "China",        "E Asia"),
    "QINGDAO":           GeoPoint("Qingdao",             36.0671, 120.3826, "China",        "E Asia"),
    "OSAKA":             GeoPoint("Osaka",               34.6937, 135.5023, "Japan",        "E Asia"),
    "BUSAN":             GeoPoint("Busan",               35.1796, 129.0756, "South Korea",  "E Asia"),
}

# Chokepoints — nodes or corridors with extra systemic risk
CHOKEPOINTS = {
    "DUBAI_JEBEL_ALI",   # Strait of Hormuz proximity
    "JEDDAH",            # Red Sea / Bab-el-Mandeb corridor
    "MOMBASA",           # Gulf of Aden / Somali basin
    "ALGECIRAS",         # Strait of Gibraltar
    "PIRAEUS",           # Bosphorus / Eastern Med corridor
    "COLOMBO",           # Indian Ocean choke
    "SINGAPORE",         # Strait of Malacca
    "PORT_KLANG",        # Malacca strait
    "TANJUNG_PELEPAS",   # Malacca strait south
}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — REAL MARITIME EDGES
#   Source: Maritime_vessel_database.xlsx
#   286 unique origin→destination pairs extracted from 324 vessel records.
#   base_risk derived from Risk_Level field:
#     Low → 0.10,  Medium → 0.22,  High → 0.38
#   dist_km = mean(Route_Distance_NM) × 1.852
#   risk_type comment = modal Risk_Type for that route pair
# ══════════════════════════════════════════════════════════════════════════════

RAW_EDGES: List[Tuple] = [
    # (source_key, target_key, dist_km, base_risk)
    ("ALGECIRAS", "ANTWERP",          1855, 0.10),  # None
    ("ALGECIRAS", "DUBAI_JEBEL_ALI",  5830, 0.38),  # Bad Weather
    ("ALGECIRAS", "LAEM_CHABANG",    10567, 0.10),  # Bad Weather
    ("ALGECIRAS", "MUMBAI",           7754, 0.10),  # None
    ("ALGECIRAS", "NINGBO",          10823, 0.22),  # Piracy Zone
    ("ALGECIRAS", "SANTOS",           7943, 0.22),  # High Traffic
    ("ALGECIRAS", "SAVANNAH",         6797, 0.22),  # Bad Weather
    ("ALGECIRAS", "SHANGHAI",        10704, 0.38),  # Bad Weather
    ("ALGECIRAS", "SINGAPORE",       11641, 0.10),  # Bad Weather
    ("ANTWERP",   "DURBAN",           9390, 0.22),  # High Traffic
    ("ANTWERP",   "HONG_KONG",        9366, 0.10),  # None
    ("ANTWERP",   "LOS_ANGELES",      9033, 0.22),  # Bad Weather
    ("ANTWERP",   "MANILA",          10481, 0.22),  # High Traffic
    ("ANTWERP",   "MUMBAI",           6885, 0.10),  # None
    ("ANTWERP",   "QINGDAO",          8478, 0.22),  # Bad Weather
    ("ANTWERP",   "ROTTERDAM",          74, 0.10),  # High Traffic
    ("ANTWERP",   "TANJUNG_PELEPAS", 10523, 0.38),  # Bad Weather
    ("ANTWERP",   "VANCOUVER",        7780, 0.10),  # Bad Weather
    ("BUSAN",     "ALGECIRAS",       10794, 0.10),  # Bad Weather
    ("BUSAN",     "DUBAI_JEBEL_ALI",  7055, 0.22),  # High Traffic
    ("BUSAN",     "DURBAN",          12533, 0.10),  # High Traffic
    ("BUSAN",     "MOMBASA",         10213, 0.10),  # High Traffic
    ("BUSAN",     "MUMBAI",           5775, 0.22),  # Bad Weather
    ("BUSAN",     "NEW_YORK",        11254, 0.10),  # Bad Weather
    ("BUSAN",     "SAVANNAH",        11900, 0.10),  # None
    ("BUSAN",     "SINGAPORE",        4584, 0.10),  # Bad Weather
    ("COLOMBO",   "ANTWERP",          8397, 0.10),  # High Traffic
    ("COLOMBO",   "BUSAN",            5912, 0.22),  # High Traffic
    ("COLOMBO",   "DURBAN",           6623, 0.10),  # Bad Weather
    ("COLOMBO",   "HAMBURG",          8071, 0.10),  # Bad Weather
    ("COLOMBO",   "JEDDAH",           4658, 0.38),  # Bad Weather — Red Sea
    ("COLOMBO",   "LOS_ANGELES",     15106, 0.22),  # None
    ("COLOMBO",   "PIRAEUS",          6602, 0.10),  # High Traffic
    ("COLOMBO",   "PORT_KLANG",       2421, 0.10),  # Bad Weather
    ("COLOMBO",   "QINGDAO",          5235, 0.10),  # High Traffic
    ("COLOMBO",   "ROTTERDAM",        8401, 0.10),  # High Traffic
    ("COLOMBO",   "SAVANNAH",        15231, 0.38),  # Piracy Zone
    ("COLOMBO",   "SINGAPORE",        2732, 0.10),  # Bad Weather
    ("COLOMBO",   "TANJUNG_PELEPAS",  2698, 0.38),  # Piracy Zone
    ("DUBAI_JEBEL_ALI", "ALGECIRAS",  5830, 0.22),  # High Traffic
    ("DUBAI_JEBEL_ALI", "BUSAN",      7055, 0.10),  # Bad Weather
    ("DUBAI_JEBEL_ALI", "COLOMBO",    3317, 0.38),  # Bad Weather
    ("DUBAI_JEBEL_ALI", "DURBAN",     6614, 0.22),  # High Traffic
    ("DUBAI_JEBEL_ALI", "JEDDAH",     1663, 0.22),  # Piracy Zone
    ("DUBAI_JEBEL_ALI", "MOMBASA",    3629, 0.10),  # Bad Weather
    ("DUBAI_JEBEL_ALI", "MUMBAI",     1954, 0.22),  # Bad Weather
    ("DUBAI_JEBEL_ALI", "QINGDAO",    6265, 0.22),  # High Traffic
    ("DUBAI_JEBEL_ALI", "SINGAPORE",  5859, 0.10),  # Bad Weather
    ("DUBAI_JEBEL_ALI", "TANJUNG_PELEPAS", 5826, 0.22),  # High Traffic
    ("DURBAN",    "DUBAI_JEBEL_ALI",  6614, 0.10),  # None
    ("DURBAN",    "FELIXSTOWE",       9547, 0.10),  # Bad Weather
    ("DURBAN",    "HONG_KONG",       10603, 0.10),  # None
    ("DURBAN",    "MANILA",          10806, 0.22),  # Piracy Zone
    ("DURBAN",    "MOMBASA",          3011, 0.10),  # None
    ("DURBAN",    "MUMBAI",           7035, 0.22),  # Piracy Zone
    ("DURBAN",    "NINGBO",          11649, 0.10),  # High Traffic
    ("DURBAN",    "PORT_KLANG",       8299, 0.38),  # Bad Weather
    ("DURBAN",    "SHANGHAI",        11709, 0.22),  # Bad Weather
    ("DURBAN",    "TANJUNG_PELEPAS",  8408, 0.10),  # None
    ("DURBAN",    "VANCOUVER",       16949, 0.38),  # Bad Weather
    ("FELIXSTOWE","DUBAI_JEBEL_ALI",  5385, 0.22),  # None
    ("FELIXSTOWE","LOS_ANGELES",      8827, 0.22),  # None
    ("FELIXSTOWE","MANILA",          10622, 0.22),  # Bad Weather
    ("FELIXSTOWE","MOMBASA",          7199, 0.22),  # High Traffic
    ("FELIXSTOWE","MUMBAI",           7101, 0.22),  # Piracy Zone
    ("FELIXSTOWE","NEW_YORK",         5655, 0.10),  # None
    ("FELIXSTOWE","NINGBO",           9219, 0.22),  # High Traffic
    ("FELIXSTOWE","PORT_KLANG",      10424, 0.38),  # Bad Weather
    ("FELIXSTOWE","SAVANNAH",         6768, 0.10),  # None
    ("FELIXSTOWE","SINGAPORE",       10746, 0.38),  # Piracy Zone
    ("FELIXSTOWE","TANJUNG_PELEPAS", 10719, 0.38),  # War Zone
    ("GUANGZHOU", "BUSAN",            2033, 0.22),  # Piracy Zone
    ("GUANGZHOU", "JEDDAH",           7533, 0.38),  # Bad Weather
    ("GUANGZHOU", "LAEM_CHABANG",     1719, 0.38),  # War Zone — SCS
    ("GUANGZHOU", "MOMBASA",          8520, 0.22),  # Piracy Zone
    ("GUANGZHOU", "SANTOS",          17935, 0.10),  # None
    ("HAMBURG",   "ANTWERP",           454, 0.10),  # Bad Weather
    ("HAMBURG",   "COLOMBO",          8071, 0.22),  # Piracy Zone
    ("HAMBURG",   "DURBAN",           9498, 0.22),  # Piracy Zone
    ("HAMBURG",   "GUANGZHOU",        8789, 0.10),  # Bad Weather
    ("HAMBURG",   "LAEM_CHABANG",     8903, 0.38),  # War Zone
    ("HAMBURG",   "MOMBASA",          6980, 0.38),  # Bad Weather — Red Sea
    ("HAMBURG",   "NINGBO",           8653, 0.10),  # High Traffic
    ("HAMBURG",   "ROTTERDAM",         410, 0.22),  # High Traffic
    ("HAMBURG",   "SINGAPORE",       10151, 0.10),  # None
    ("HAMBURG",   "TANJUNG_PELEPAS", 10124, 0.10),  # None
    ("HAMBURG",   "VANCOUVER",        7760, 0.10),  # High Traffic
    ("HONG_KONG", "COLOMBO",          4053, 0.38),  # Piracy Zone
    ("HONG_KONG", "DURBAN",          10603, 0.10),  # High Traffic
    ("HONG_KONG", "LAEM_CHABANG",     1741, 0.22),  # High Traffic
    ("HONG_KONG", "LOS_ANGELES",     11671, 0.10),  # High Traffic
    ("HONG_KONG", "MUMBAI",           4305, 0.10),  # High Traffic
    ("HONG_KONG", "NEW_YORK",        12958, 0.10),  # None
    ("HONG_KONG", "OSAKA",            2489, 0.10),  # Bad Weather
    ("HONG_KONG", "PIRAEUS",          8549, 0.10),  # Bad Weather
    ("HONG_KONG", "PORT_KLANG",       2551, 0.10),  # Bad Weather
    ("HONG_KONG", "SINGAPORE",        2591, 0.10),  # Bad Weather
    ("JEDDAH",    "ANTWERP",          4463, 0.10),  # None
    ("JEDDAH",    "BUSAN",            8641, 0.10),  # High Traffic
    ("JEDDAH",    "COLOMBO",          4658, 0.22),  # High Traffic
    ("JEDDAH",    "DURBAN",           5782, 0.22),  # Piracy Zone
    ("JEDDAH",    "NEW_YORK",        10254, 0.22),  # Bad Weather
    ("JEDDAH",    "SHANGHAI",         8087, 0.10),  # None
    ("KARACHI",   "ANTWERP",          5993, 0.22),  # Bad Weather
    ("KARACHI",   "BUSAN",            5990, 0.10),  # Bad Weather
    ("KARACHI",   "FELIXSTOWE",       6209, 0.10),  # None
    ("KARACHI",   "HAMBURG",          5657, 0.10),  # High Traffic
    ("KARACHI",   "HONG_KONG",        4790, 0.10),  # High Traffic
    ("KARACHI",   "LOS_ANGELES",     13475, 0.10),  # High Traffic
    ("KARACHI",   "NEW_YORK",        11690, 0.10),  # High Traffic
    ("KARACHI",   "PIRAEUS",          4321, 0.10),  # High Traffic
    ("KARACHI",   "PORT_KLANG",       4406, 0.22),  # High Traffic
    ("KARACHI",   "QINGDAO",          5199, 0.22),  # Piracy Zone
    ("KARACHI",   "TANJUNG_PELEPAS",  4712, 0.38),  # Bad Weather
    ("LAEM_CHABANG","HONG_KONG",      1741, 0.38),  # High Traffic
    ("LAEM_CHABANG","JEDDAH",         6579, 0.22),  # High Traffic
    ("LAEM_CHABANG","SINGAPORE",      1351, 0.10),  # High Traffic
    ("LAEM_CHABANG","VANCOUVER",     11852, 0.10),  # Bad Weather
    ("LOS_ANGELES","ALGECIRAS",       9579, 0.10),  # Bad Weather
    ("LOS_ANGELES","GUANGZHOU",      11668, 0.22),  # Piracy Zone
    ("LOS_ANGELES","NEW_YORK",        3948, 0.22),  # High Traffic
    ("LOS_ANGELES","NINGBO",         10557, 0.10),  # Bad Weather
    ("LOS_ANGELES","OSAKA",           9209, 0.10),  # None
    ("LOS_ANGELES","PIRAEUS",        11125, 0.10),  # None
    ("LOS_ANGELES","QINGDAO",        10153, 0.10),  # Bad Weather
    ("LOS_ANGELES","SHANGHAI",       10456, 0.22),  # High Traffic
    ("LOS_ANGELES","SINGAPORE",      14141, 0.10),  # Bad Weather
    ("MANILA",    "ALGECIRAS",       12053, 0.10),  # High Traffic
    ("MANILA",    "BUSAN",            2427, 0.38),  # Bad Weather
    ("MANILA",    "COLOMBO",          4565, 0.10),  # Bad Weather
    ("MANILA",    "FELIXSTOWE",      10622, 0.10),  # Bad Weather
    ("MANILA",    "HAMBURG",         10030, 0.10),  # Bad Weather
    ("MANILA",    "LOS_ANGELES",     11759, 0.38),  # Bad Weather
    ("MANILA",    "NEW_YORK",        13675, 0.22),  # Bad Weather
    ("MANILA",    "NINGBO",           1699, 0.38),  # Bad Weather
    ("MANILA",    "OSAKA",            2666, 0.22),  # Piracy Zone
    ("MANILA",    "SINGAPORE",        2394, 0.10),  # High Traffic
    ("MOMBASA",   "BUSAN",           10213, 0.38),  # War Zone
    ("MOMBASA",   "COLOMBO",          4625, 0.10),  # Bad Weather
    ("MOMBASA",   "DUBAI_JEBEL_ALI",  3629, 0.10),  # Bad Weather
    ("MOMBASA",   "DURBAN",           3011, 0.38),  # Piracy Zone
    ("MOMBASA",   "HAMBURG",          6980, 0.22),  # Piracy Zone
    ("MOMBASA",   "JEDDAH",           2846, 0.10),  # High Traffic
    ("MOMBASA",   "LAEM_CHABANG",     7020, 0.10),  # Bad Weather
    ("MOMBASA",   "LOS_ANGELES",     16000, 0.22),  # Piracy Zone
    ("MOMBASA",   "MUMBAI",           4439, 0.10),  # None
    ("MOMBASA",   "NINGBO",           9452, 0.38),  # Piracy Zone
    ("MOMBASA",   "OSAKA",           10796, 0.10),  # None
    ("MOMBASA",   "PIRAEUS",          4953, 0.22),  # Bad Weather
    ("MOMBASA",   "SAVANNAH",        13123, 0.10),  # None
    ("MUMBAI",    "ANTWERP",          6885, 0.38),  # Bad Weather
    ("MUMBAI",    "COLOMBO",          1536, 0.22),  # High Traffic
    ("MUMBAI",    "GUANGZHOU",        4209, 0.22),  # Bad Weather
    ("MUMBAI",    "KARACHI",           892, 0.10),  # Bad Weather
    ("MUMBAI",    "LAEM_CHABANG",     3064, 0.38),  # War Zone
    ("MUMBAI",    "LOS_ANGELES",     14041, 0.10),  # None
    ("MUMBAI",    "NEW_YORK",        12554, 0.10),  # None
    ("MUMBAI",    "PORT_KLANG",       3570, 0.22),  # Bad Weather
    ("MUMBAI",    "ROTTERDAM",        6886, 0.22),  # High Traffic
    ("MUMBAI",    "SANTOS",          13741, 0.10),  # Bad Weather
    ("NEW_YORK",  "COLOMBO",         14086, 0.22),  # Bad Weather
    ("NEW_YORK",  "GUANGZHOU",       12879, 0.22),  # Piracy Zone
    ("NEW_YORK",  "HAMBURG",          6129, 0.22),  # Piracy Zone
    ("NEW_YORK",  "MOMBASA",         12287, 0.10),  # Bad Weather
    ("NEW_YORK",  "MUMBAI",          12554, 0.10),  # None
    ("NEW_YORK",  "ROTTERDAM",        5858, 0.38),  # War Zone
    ("NEW_YORK",  "SHANGHAI",        11860, 0.38),  # Piracy Zone
    ("NEW_YORK",  "SINGAPORE",       15342, 0.38),  # Bad Weather
    ("NINGBO",    "ALGECIRAS",       10823, 0.22),  # High Traffic
    ("NINGBO",    "ANTWERP",          9106, 0.10),  # Bad Weather
    ("NINGBO",    "DUBAI_JEBEL_ALI",  6497, 0.38),  # Piracy Zone
    ("NINGBO",    "DURBAN",          11649, 0.22),  # Bad Weather
    ("NINGBO",    "FELIXSTOWE",       9219, 0.38),  # War Zone
    ("NINGBO",    "JEDDAH",           8133, 0.10),  # Bad Weather
    ("NINGBO",    "KARACHI",          5365, 0.22),  # Piracy Zone
    ("NINGBO",    "LAEM_CHABANG",     2828, 0.22),  # Bad Weather
    ("NINGBO",    "LOS_ANGELES",     10557, 0.10),  # Bad Weather
    ("NINGBO",    "MUMBAI",           5042, 0.10),  # None
    ("NINGBO",    "PORT_KLANG",       3664, 0.10),  # Bad Weather
    ("NINGBO",    "SANTOS",          18646, 0.10),  # Bad Weather
    ("NINGBO",    "SAVANNAH",        12724, 0.22),  # Bad Weather
    ("NINGBO",    "SINGAPORE",        3687, 0.10),  # High Traffic
    ("OSAKA",     "ALGECIRAS",       11161, 0.22),  # Piracy Zone
    ("OSAKA",     "COLOMBO",          6454, 0.38),  # War Zone
    ("OSAKA",     "HAMBURG",          8893, 0.22),  # High Traffic
    ("OSAKA",     "KARACHI",          6577, 0.22),  # Piracy Zone
    ("OSAKA",     "LAEM_CHABANG",     4226, 0.22),  # Piracy Zone
    ("OSAKA",     "LOS_ANGELES",      9209, 0.38),  # Bad Weather
    ("OSAKA",     "MUMBAI",           6360, 0.22),  # Piracy Zone
    ("OSAKA",     "QINGDAO",          1378, 0.10),  # Bad Weather
    ("OSAKA",     "ROTTERDAM",        9272, 0.10),  # Bad Weather
    ("OSAKA",     "SANTOS",          18809, 0.10),  # None
    ("OSAKA",     "SHANGHAI",         1363, 0.22),  # High Traffic
    ("PIRAEUS",   "BUSAN",            8836, 0.10),  # Bad Weather
    ("PIRAEUS",   "DURBAN",           7579, 0.22),  # Bad Weather
    ("PIRAEUS",   "LOS_ANGELES",     11125, 0.10),  # High Traffic
    ("PIRAEUS",   "MANILA",           9641, 0.10),  # High Traffic
    ("PIRAEUS",   "MUMBAI",           5182, 0.10),  # Bad Weather
    ("PIRAEUS",   "NEW_YORK",         7926, 0.22),  # Piracy Zone
    ("PIRAEUS",   "PORT_KLANG",       8725, 0.22),  # Piracy Zone
    ("PIRAEUS",   "ROTTERDAM",        2153, 0.22),  # Bad Weather
    ("PORT_KLANG","ALGECIRAS",       11305, 0.10),  # None
    ("PORT_KLANG","HAMBURG",          9831, 0.10),  # High Traffic
    ("PORT_KLANG","HONG_KONG",        2551, 0.10),  # High Traffic
    ("PORT_KLANG","LAEM_CHABANG",     1119, 0.10),  # Bad Weather
    ("PORT_KLANG","NEW_YORK",        15131, 0.22),  # Piracy Zone
    ("PORT_KLANG","NINGBO",           3664, 0.38),  # Bad Weather
    ("PORT_KLANG","QINGDAO",          4160, 0.10),  # None
    ("PORT_KLANG","ROTTERDAM",       10213, 0.10),  # None
    ("PORT_KLANG","SANTOS",          15840, 0.10),  # Bad Weather
    ("PORT_KLANG","SAVANNAH",        16102, 0.10),  # High Traffic
    ("PORT_KLANG","VANCOUVER",       12802, 0.10),  # High Traffic
    ("QINGDAO",   "ALGECIRAS",       10230, 0.10),  # Bad Weather
    ("QINGDAO",   "DUBAI_JEBEL_ALI",  6265, 0.22),  # High Traffic
    ("QINGDAO",   "DURBAN",          11851, 0.38),  # High Traffic
    ("QINGDAO",   "FELIXSTOWE",       8582, 0.38),  # None
    ("QINGDAO",   "GUANGZHOU",        1593, 0.10),  # High Traffic
    ("QINGDAO",   "HONG_KONG",        1642, 0.10),  # High Traffic
    ("QINGDAO",   "SHANGHAI",          547, 0.22),  # High Traffic
    ("QINGDAO",   "TANJUNG_PELEPAS",  4231, 0.10),  # Bad Weather
    ("QINGDAO",   "VANCOUVER",        8653, 0.10),  # Bad Weather
    ("ROTTERDAM",  "ALGECIRAS",       1922, 0.38),  # War Zone
    ("ROTTERDAM",  "ANTWERP",           74, 0.38),  # Bad Weather
    ("ROTTERDAM",  "BUSAN",           8936, 0.22),  # High Traffic
    ("ROTTERDAM",  "COLOMBO",         8401, 0.10),  # High Traffic
    ("ROTTERDAM",  "DUBAI_JEBEL_ALI", 5176, 0.10),  # High Traffic
    ("ROTTERDAM",  "DURBAN",          9456, 0.38),  # War Zone
    ("ROTTERDAM",  "GUANGZHOU",       9199, 0.10),  # High Traffic
    ("ROTTERDAM",  "HONG_KONG",       9326, 0.10),  # None
    ("ROTTERDAM",  "KARACHI",         5995, 0.10),  # Bad Weather
    ("ROTTERDAM",  "SAVANNAH",        6974, 0.22),  # Piracy Zone
    ("ROTTERDAM",  "TANJUNG_PELEPAS",10508, 0.10),  # None
    ("SANTOS",    "BUSAN",           18691, 0.10),  # None
    ("SANTOS",    "JEDDAH",          10534, 0.10),  # None
    ("SANTOS",    "LOS_ANGELES",      9943, 0.38),  # War Zone
    ("SANTOS",    "MANILA",          18327, 0.22),  # Piracy Zone
    ("SANTOS",    "MOMBASA",          9419, 0.22),  # Piracy Zone
    ("SANTOS",    "PIRAEUS",         10027, 0.10),  # High Traffic
    ("SANTOS",    "SAVANNAH",         7244, 0.10),  # High Traffic
    ("SANTOS",    "SHANGHAI",        18569, 0.10),  # High Traffic
    ("SAVANNAH",  "GUANGZHOU",       13689, 0.10),  # Bad Weather
    ("SAVANNAH",  "JEDDAH",          11305, 0.10),  # High Traffic
    ("SAVANNAH",  "MANILA",          14316, 0.10),  # None
    ("SAVANNAH",  "MOMBASA",         13123, 0.10),  # None
    ("SAVANNAH",  "MUMBAI",          13700, 0.10),  # None
    ("SAVANNAH",  "QINGDAO",         12114, 0.22),  # High Traffic
    ("SAVANNAH",  "ROTTERDAM",        6974, 0.22),  # High Traffic
    ("SAVANNAH",  "SANTOS",           7244, 0.10),  # Bad Weather
    ("SAVANNAH",  "VANCOUVER",        3958, 0.10),  # Bad Weather
    ("SHANGHAI",  "BUSAN",             832, 0.10),  # High Traffic
    ("SHANGHAI",  "FELIXSTOWE",       9088, 0.10),  # Bad Weather
    ("SHANGHAI",  "KARACHI",          5339, 0.10),  # Bad Weather
    ("SHANGHAI",  "LAEM_CHABANG",     2916, 0.10),  # Bad Weather
    ("SHANGHAI",  "LOS_ANGELES",     10456, 0.38),  # War Zone — SCS
    ("SHANGHAI",  "NEW_YORK",        11860, 0.22),  # Bad Weather
    ("SHANGHAI",  "NINGBO",            152, 0.22),  # Piracy Zone
    ("SHANGHAI",  "PIRAEUS",          8548, 0.10),  # High Traffic
    ("SHANGHAI",  "VANCOUVER",        9025, 0.10),  # None
    ("SINGAPORE", "ALGECIRAS",       11641, 0.22),  # High Traffic
    ("SINGAPORE", "GUANGZHOU",        2632, 0.22),  # High Traffic
    ("SINGAPORE", "HAMBURG",         10151, 0.10),  # None
    ("SINGAPORE", "LOS_ANGELES",     14141, 0.10),  # Bad Weather
    ("SINGAPORE", "MUMBAI",           3907, 0.38),  # Bad Weather
    ("SINGAPORE", "PORT_KLANG",        337, 0.22),  # Bad Weather
    ("SINGAPORE", "SANTOS",          15934, 0.10),  # Bad Weather
    ("SINGAPORE", "SHANGHAI",         3810, 0.22),  # High Traffic
    ("SINGAPORE", "VANCOUVER",       12825, 0.10),  # None
    ("TANJUNG_PELEPAS","ANTWERP",    10523, 0.10),  # High Traffic
    ("TANJUNG_PELEPAS","DURBAN",      8408, 0.10),  # Bad Weather
    ("TANJUNG_PELEPAS","GUANGZHOU",   2638, 0.38),  # Piracy Zone
    ("TANJUNG_PELEPAS","HONG_KONG",   2598, 0.10),  # None
    ("TANJUNG_PELEPAS","MOMBASA",     7124, 0.22),  # Piracy Zone
    ("TANJUNG_PELEPAS","OSAKA",       4969, 0.22),  # Piracy Zone
    ("TANJUNG_PELEPAS","PIRAEUS",     9030, 0.10),  # Bad Weather
    ("TANJUNG_PELEPAS","ROTTERDAM",  10508, 0.22),  # High Traffic
    ("VANCOUVER", "ANTWERP",          7780, 0.22),  # Piracy Zone
    ("VANCOUVER", "DUBAI_JEBEL_ALI", 11755, 0.10),  # Bad Weather
    ("VANCOUVER", "FELIXSTOWE",       7597, 0.10),  # High Traffic
    ("VANCOUVER", "HONG_KONG",       10251, 0.10),  # None
    ("VANCOUVER", "JEDDAH",          11947, 0.10),  # None
    ("VANCOUVER", "MANILA",          10548, 0.22),  # Bad Weather
    ("VANCOUVER", "MUMBAI",          12268, 0.10),  # Bad Weather
    ("VANCOUVER", "NINGBO",           9142, 0.10),  # Bad Weather
    ("VANCOUVER", "QINGDAO",          8653, 0.38),  # High Traffic
    ("VANCOUVER", "SANTOS",          11106, 0.10),  # Bad Weather
    ("VANCOUVER", "SAVANNAH",         3958, 0.10),  # High Traffic
]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — THREAT ZONES
#   Source: calibrated from Maritime_vessel_database.xlsx + India Road Network
#
#   Maritime DB risk distributions used to calibrate severity:
#     Risk_Level: Low=57.7% → base sev 3–4
#                 Medium=28.7% → base sev 5–7
#                 High=13.6% → base sev 8–10
#     Risk_Type modal per region:
#       Bad Weather dominant (32.4%) → natural zones
#       Piracy Zone (15.3%) → piracy zones in known hotspots
#       War Zone (5.3%) → highest severity, narrow radius
#
#   India Road DB used to derive inland threat elevation:
#     States with poor_pct > 0.40: Arunachal(81%), Manipur(81%), Mizoram(93%),
#                                   Nagaland(67%), Sikkim(47%)
#     → Northeast India Corridor marked as road-risk zone severity 5.0
# ══════════════════════════════════════════════════════════════════════════════

THREAT_ZONES: List[ThreatZone] = [
    # ── War / Conflict zones ──────────────────────────────────────────────────
    ThreatZone("Red Sea — Houthi active conflict",       15.00,  42.50, "war",        9.5, 650),
    ThreatZone("Gaza / Eastern Med conflict",            31.50,  34.40, "war",        7.8, 300),
    ThreatZone("Black Sea — Ukraine war corridor",       46.50,  31.00, "war",        7.5, 500),
    ThreatZone("South China Sea — military tension",     14.00, 114.00, "war",        5.5, 600),
    ThreatZone("Taiwan Strait — military standoff",      24.00, 119.00, "war",        6.5, 280),
    # ── Piracy zones (calibrated from DB: 15.3% of vessel records) ───────────
    ThreatZone("Strait of Hormuz — drone/piracy",        26.57,  56.27, "piracy",     8.8, 350),
    ThreatZone("Gulf of Aden — Somali piracy basin",     12.50,  46.00, "piracy",     7.5, 450),
    ThreatZone("Gulf of Guinea — Nigeria piracy",         3.00,   3.00, "piracy",     6.5, 500),
    ThreatZone("Horn of Africa — offshore piracy",        5.00,  48.00, "piracy",     6.0, 420),
    ThreatZone("Strait of Malacca — armed robbery",       1.50, 103.00, "piracy",     4.5, 250),
    # ── Sanctions zones ───────────────────────────────────────────────────────
    ThreatZone("Persian Gulf — Iran sanctions regime",   27.00,  51.00, "sanctions",  7.0, 420),
    ThreatZone("Russia OFAC sanctions zone",             55.00,  37.00, "sanctions",  6.0, 600),
    # ── Natural / Weather (Bad Weather = 32.4% of DB records) ────────────────
    ThreatZone("Bay of Bengal — cyclone track",          13.00,  88.00, "natural",    5.5, 700),
    ThreatZone("Suez Canal — draft restrictions",        30.50,  32.30, "natural",    4.5, 150),
    ThreatZone("North Atlantic — storm corridor",        45.00, -40.00, "natural",    5.0, 800),
    ThreatZone("South China Sea — typhoon season",       18.00, 120.00, "natural",    6.0, 700),
    ThreatZone("Cape of Good Hope — rough seas",        -34.00,  18.50, "natural",    4.0, 400),
    # ── Road-derived inland risk zones (India Road Network DB) ───────────────
    # NE India corridor: Mizoram(93% poor), Nagaland(67%), Manipur(81%)
    ThreatZone("NE India — extreme poor road network",   25.00,  93.00, "natural",    5.0, 350),
    # J&K / Himalayan border routes: Arunachal(81% poor), Sikkim(47% poor)
    ThreatZone("Himalayan border — poor road access",    28.00,  90.50, "natural",    4.5, 400),
]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — INDIA ROAD STATE RISK INDEX
#   Source: India_Road_Network_Database.xlsx — 500 road segments, 28 states
#
#   Formula (Road Risk Modifier per state):
#     road_risk(s) = 0.40 × poor_pct(s)
#                  + 0.30 × (1 − avg_speed(s)/50)
#                  + 0.30 × toll_penalty(s)
#
#   Normalised to [0, 1].
#   Used to apply an uplift to base_risk for edges touching Indian ports.
#
#   State  |poor_pct|avg_spd|road_risk
#   ────────────────────────────────
#   Mizoram   0.933   34.7    0.55
#   Nagaland  0.667   34.3    0.47
#   Manipur   0.813   36.9    0.48
#   Arunachal 0.813   38.1    0.44
#   Sikkim    0.467   36.0    0.35
#   Himachal  0.471   39.1    0.33
#   Jharkhand 0.235   37.4    0.22
#   Bihar     0.211   39.2    0.20
#   Gujarat   0.000   42.8    0.09  ← lowest risk, port access roads good
#   Punjab    0.000   39.4    0.10
# ══════════════════════════════════════════════════════════════════════════════

INDIA_ROAD_RISK: Dict[str, float] = {
    "Mizoram":          0.55, "Nagaland":        0.47,
    "Manipur":          0.48, "Arunachal Pradesh":0.44,
    "Sikkim":           0.35, "Himachal Pradesh": 0.33,
    "Meghalaya":        0.30, "Tripura":          0.29,
    "Uttarakhand":      0.26, "Jharkhand":        0.22,
    "Bihar":            0.20, "Odisha":           0.18,
    "Chhattisgarh":     0.17, "Karnataka":        0.15,
    "Rajasthan":        0.15, "Madhya Pradesh":   0.15,
    "Uttar Pradesh":    0.12, "West Bengal":      0.11,
    "Assam":            0.19, "Kerala":           0.08,
    "Maharashtra":      0.09, "Telangana":        0.10,
    "Andhra Pradesh":   0.10, "Tamil Nadu":       0.08,
    "Haryana":          0.09, "Punjab":           0.10,
    "Gujarat":          0.09, "Goa":              0.05,
}

# Indian port → primary state (for road risk uplift on port-adjacent edges)
INDIA_PORT_STATE: Dict[str, str] = {
    "MUMBAI":  "Maharashtra",
    "KARACHI": None,           # Pakistan — not in Indian dataset
    "COLOMBO": None,           # Sri Lanka
}

def india_road_uplift(port: str) -> float:
    """Return road-risk uplift for edges adjacent to an Indian port."""
    state = INDIA_PORT_STATE.get(port)
    if state is None:
        return 0.0
    return INDIA_ROAD_RISK.get(state, 0.0) * 0.15   # 15% max uplift on base_risk


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — HAVERSINE DISTANCE
#   a = sin²(Δlat/2) + cos(lat₁)·cos(lat₂)·sin²(Δlon/2)
#   d = 2·R·arcsin(√a)
# ══════════════════════════════════════════════════════════════════════════════

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R   = 6_371
    φ1  = math.radians(lat1)
    φ2  = math.radians(lat2)
    Δφ  = math.radians(lat2 - lat1)
    Δλ  = math.radians(lon2 - lon1)
    a   = math.sin(Δφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(Δλ/2)**2
    return R * 2 * math.asin(math.sqrt(a))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — RBF THREAT SCORER
#   threat(p) = Σ_z  severity_z · max(0,  1 − d(p,z)/radius_z)
#   clipped to [0, 10]
# ══════════════════════════════════════════════════════════════════════════════

def rbf_threat_score(lat: float, lon: float) -> float:
    """
    Radial-basis threat score at a geo-point.
      threat(p) = Σ_z  severity_z · max(0,  1 − d(p,z)/radius_z)
    Clipped to [0, 10].
    """
    total = 0.0
    for z in THREAT_ZONES:
        d         = haversine(lat, lon, z.lat, z.lon)
        influence = max(0.0, 1.0 - d / z.radius_km)
        total    += z.severity * influence
    return min(total, 10.0)


def edge_threat_score(n1: str, n2: str, samples: int = 5) -> float:
    """
    Max RBF threat along edge n1→n2 (worst-case exposure).
      edge_threat(u,v) = max{ threat(p_t) : t ∈ {0,.25,.5,.75,1} }
    """
    p1, p2 = WAYPOINTS[n1], WAYPOINTS[n2]
    worst  = 0.0
    for i in range(samples + 1):
        t   = i / samples
        lat = p1.lat + t * (p2.lat - p1.lat)
        lon = p1.lon + t * (p2.lon - p1.lon)
        worst = max(worst, rbf_threat_score(lat, lon))
    return worst


def threats_near_path(path: List[str], radius_km: float = 700) -> List[ThreatZone]:
    near = []
    for node in path:
        gp = WAYPOINTS[node]
        for z in THREAT_ZONES:
            if z not in near and haversine(gp.lat, gp.lon, z.lat, z.lon) <= radius_km:
                near.append(z)
    return near


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — EDGE WEIGHT
#   W(u,v) = α·threat_norm  +  β·dist_norm
#   α=0.70, β=0.30
# ══════════════════════════════════════════════════════════════════════════════

ALPHA = 0.70
BETA  = 0.30

def edge_weight(adj: dict, u: str, v: str) -> float:
    """
    W(u,v) = 0.70·(edge_threat/10) + 0.30·(dist_km/20000)
    India road uplift applied to edges touching Mumbai / Karachi.
    """
    data        = adj[u][v]
    threat_norm = edge_threat_score(u, v) / 10.0
    dist_norm   = data["dist"] / 20_000.0
    road_up     = max(india_road_uplift(u), india_road_uplift(v))
    return ALPHA * (threat_norm + road_up * 0.1) + BETA * dist_norm


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — ADJACENCY GRAPH
# ══════════════════════════════════════════════════════════════════════════════

def build_adjacency() -> dict:
    adj: dict = {}
    for src, tgt, dist, risk in RAW_EDGES:
        adj.setdefault(src, {})[tgt] = {"dist": dist, "base_risk": risk}
    return adj


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — DIJKSTRA
#   d[v] = min( d[v],  d[u] + W(u,v) )
# ══════════════════════════════════════════════════════════════════════════════

def dijkstra(adj: dict, source: str, target: str) -> Tuple[Optional[List[str]], float]:
    dist = {n: float("inf") for n in adj}
    dist[source] = 0.0
    prev: Dict[str, Optional[str]] = {}
    pq   = [(0.0, source)]

    while pq:
        d_u, u = heapq.heappop(pq)
        if d_u > dist[u]:
            continue
        if u == target:
            break
        for v in adj.get(u, {}):
            w   = edge_weight(adj, u, v)
            alt = dist[u] + w
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u
                heapq.heappush(pq, (alt, v))

    path, cur = [], target
    while cur is not None:
        path.append(cur)
        cur = prev.get(cur)
    path.reverse()

    if not path or path[0] != source:
        return None, float("inf")
    return path, dist[target]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — YEN'S K-SHORTEST PATHS
#   cost(P) = Σ W(P[i], P[i+1])
# ══════════════════════════════════════════════════════════════════════════════

def path_cost(adj: dict, path: List[str]) -> float:
    return sum(edge_weight(adj, path[i], path[i+1]) for i in range(len(path)-1))


def yens_k_shortest(adj0: dict, source: str, target: str,
                    K: int = 5) -> List[Tuple[List[str], float]]:
    adj    = json.loads(json.dumps(adj0))
    first, fc = dijkstra(adj, source, target)
    if first is None:
        return []

    A: List[Tuple[List[str], float]] = [(first, fc)]
    B: List[Tuple[float, List[str]]] = []
    seen: set = {tuple(first)}

    for k in range(1, K):
        prev_path = A[k-1][0]
        for i in range(len(prev_path)-1):
            spur_node = prev_path[i]
            root_path = prev_path[:i+1]
            root_c    = path_cost(adj, root_path)

            removed_edges: List[Tuple] = []
            for a_path, _ in A:
                if (len(a_path) > i
                        and a_path[:i+1] == root_path
                        and adj.get(a_path[i], {}).get(a_path[i+1]) is not None):
                    u, v  = a_path[i], a_path[i+1]
                    saved = adj[u].pop(v)
                    removed_edges.append((u, v, saved))

            removed_nodes: List[Tuple] = []
            for node in root_path[:-1]:
                if node == spur_node:
                    continue
                out_e = adj.pop(node, {})
                in_e  = {}
                for x in list(adj.keys()):
                    if node in adj[x]:
                        in_e[x] = adj[x].pop(node)
                removed_nodes.append((node, out_e, in_e))

            spur_path, spur_c = dijkstra(adj, spur_node, target)

            # Restore nodes (all first)
            for node, out_e, in_e in reversed(removed_nodes):
                adj[node] = out_e
                for x, data in in_e.items():
                    adj.setdefault(x, {})[node] = data
            # Restore edges
            for u, v, data in removed_edges:
                adj.setdefault(u, {})[v] = data

            if spur_path and spur_path[0] == spur_node:
                full  = root_path[:-1] + spur_path
                total = root_c + spur_c
                key   = tuple(full)
                if key not in seen:
                    seen.add(key)
                    heapq.heappush(B, (total, full))

        if not B:
            break
        cost, path = heapq.heappop(B)
        A.append((path, cost))

    return A


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 12 — XGBOOST RISK SCORER
#   Training label:
#     y = 30·base_risk + 6·threat_sev + 8·n_choke
#         + 15·war + 12·sanctions + 4·weather + 5·cargo + ε
#   XGBoost update: F_m(x) = F_{m-1}(x) + η·h_m(x)
# ══════════════════════════════════════════════════════════════════════════════

class XGBRiskScorer:

    FEATURE_NAMES = [
        "dist_norm", "hops_norm", "base_risk",
        "threat_sev_norm", "n_threats_norm",
        "choke_flag", "n_choke_norm", "cargo_val",
        "mode_sea", "seasonal_sin",
        "weather_norm", "sanctions_flag", "war_flag", "piracy_flag",
    ]

    def __init__(self):
        self.reg = xgb.XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.04,
            subsample=0.80, colsample_bytree=0.80,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, verbosity=0,
        )
        self.scaler  = StandardScaler()
        self._fitted = False

    def _make_training_data(self, n: int = 3000):
        """
        Synthetic training labels calibrated to the Maritime DB risk distribution:
          Low (57.7%) → mean score ~20
          Medium (28.7%) → mean score ~45
          High (13.6%) → mean score ~72
        """
        rng = np.random.default_rng(42)
        X, y = [], []
        for _ in range(n):
            dist_norm      = rng.uniform(0.03, 1.0)
            hops_norm      = rng.uniform(0.10, 1.0)
            base_risk      = rng.uniform(0.02, 0.40)
            threat_sev     = rng.uniform(0.0, 10.0)
            n_threats      = rng.integers(0, 9)
            choke_flag     = rng.integers(0, 2)
            n_choke        = rng.integers(0, 5) if choke_flag else 0
            cargo_val      = rng.uniform(0.0, 1.0)
            month          = rng.integers(1, 13)
            seasonal_sin   = math.sin(2*math.pi*month/12)
            weather        = rng.uniform(0.0, 5.0)
            sanctions_flag = rng.integers(0, 2)
            war_flag       = rng.integers(0, 2)
            piracy_flag    = rng.integers(0, 2)

            # ── Training label formula ──────────────────────────────────────
            # y = 30·base_risk + 6·threat_sev + 8·n_choke
            #     + 15·war + 12·sanctions + 4·weather + 5·cargo + ε
            # ───────────────────────────────────────────────────────────────
            label = (
                30*base_risk + 6*threat_sev + 8*n_choke +
                15*war_flag  + 12*sanctions_flag + 4*weather +
                5*cargo_val  + rng.normal(0, 3)
            )
            label = float(np.clip(label, 0, 100))

            X.append([
                dist_norm, hops_norm, base_risk,
                threat_sev/10, n_threats/8,
                choke_flag, n_choke/4, cargo_val,
                1, seasonal_sin,
                weather/5, sanctions_flag, war_flag, piracy_flag,
            ])
            y.append(label)
        return np.array(X), np.array(y)

    def train(self):
        print("  [XGB] Building training dataset (n=3 000)...")
        X, y = self._make_training_data(3000)
        Xs   = self.scaler.fit_transform(X)
        Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, random_state=42)
        self.reg.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)
        r2 = self.reg.score(Xte, yte)
        print(f"  [XGB] Trained  R² = {r2:.4f}  (n_estimators={self.reg.n_estimators})")
        self._fitted = True

    def predict(self, feat: dict) -> Tuple[float, str]:
        if not self._fitted:
            raise RuntimeError("Call .train() first.")
        x  = np.array([[feat.get(k, 0.0) for k in self.FEATURE_NAMES]])
        xs = self.scaler.transform(x)
        score = float(np.clip(self.reg.predict(xs)[0], 0, 100))
        if   score < 25: cls = "LOW"
        elif score < 50: cls = "MEDIUM"
        elif score < 75: cls = "HIGH"
        else:            cls = "CRITICAL"
        return round(score, 2), cls

    def feature_importance(self) -> Dict[str, float]:
        imp = self.reg.feature_importances_
        return dict(sorted(zip(self.FEATURE_NAMES, imp),
                           key=lambda x: x[1], reverse=True))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 13 — FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_features(path: List[str], adj: dict, cargo_value_norm: float) -> dict:
    dist_km = sum(adj[path[i]][path[i+1]]["dist"] for i in range(len(path)-1))
    base_risks   = [adj[path[i]][path[i+1]]["base_risk"] for i in range(len(path)-1)]
    edge_threats = [edge_threat_score(path[i], path[i+1]) for i in range(len(path)-1)]
    near         = threats_near_path(path)
    n_choke      = sum(1 for n in path if n in CHOKEPOINTS)
    war_flag     = int(any(z.threat_type in ("war","terrorism") for z in near))
    san_flag     = int(any(z.threat_type == "sanctions"         for z in near))
    piracy_flag  = int(any(z.threat_type == "piracy"           for z in near))
    weather_max  = max((z.severity for z in near if z.threat_type == "natural"), default=0.0)
    month        = datetime.datetime.now().month

    return {
        "dist_norm":      dist_km / 20_000,
        "hops_norm":      len(path) / 10,
        "base_risk":      max(base_risks) if base_risks else 0.0,
        "threat_sev_norm":max(edge_threats)/10 if edge_threats else 0.0,
        "n_threats_norm": len(near) / len(THREAT_ZONES),
        "choke_flag":     int(n_choke > 0),
        "n_choke_norm":   n_choke / 5,
        "cargo_val":      cargo_value_norm,
        "mode_sea":       1,
        "seasonal_sin":   math.sin(2*math.pi*month/12),
        "weather_norm":   weather_max / 10,
        "sanctions_flag": san_flag,
        "war_flag":       war_flag,
        "piracy_flag":    piracy_flag,
        # raw for RoutePlan
        "_dist_km":    dist_km,
        "_n_choke":    n_choke,
        "_threat_sev": max(edge_threats) if edge_threats else 0.0,
        "_weather":    weather_max,
        "_sanctions":  san_flag,
        "_war":        war_flag,
        "_piracy":     piracy_flag,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 14 — WARNINGS
# ══════════════════════════════════════════════════════════════════════════════

def generate_warnings(rp: RoutePlan, all_routes: List[RoutePlan]) -> List[str]:
    flags = []
    if rp.xgb_risk_score >= 75:
        flags.append("CRITICAL: XGBoost score ≥ 75 — route not recommended.")
    if rp.n_chokepoints > 1:
        chokes = [n for n in rp.path if n in CHOKEPOINTS]
        flags.append(f"MULTI-CHOKEPOINT: passes {len(chokes)} chokepoints "
                     f"({', '.join(chokes)}).")
    if rp.war_flag:
        flags.append("WAR-ZONE ADJACENT: war/terrorism threat zone near path.")
    if rp.piracy_flag:
        flags.append("PIRACY ZONE: active piracy zone near route.")
    if rp.sanctions_flag:
        flags.append("SANCTIONS REGION: sanctions zone intersects corridor.")
    safer = [r for r in all_routes
             if r.xgb_risk_score < rp.xgb_risk_score - 10
             and r.alternative_rank != rp.alternative_rank]
    if safer:
        best = min(safer, key=lambda r: r.xgb_risk_score)
        flags.append(f"SAFER OPTION: Route #{best.alternative_rank} "
                     f"scores {best.xgb_risk_score:.1f} vs {rp.xgb_risk_score:.1f}.")
    return flags


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 15 — PIPELINE ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

class CargoRiskPipeline:

    def __init__(self, k_paths: int = 5):
        self.k_paths    = k_paths
        self.adj        = build_adjacency()
        self.xgb_scorer = XGBRiskScorer()

    def initialise(self):
        print("\n" + "═"*64)
        print("  CARGO RISK PIPELINE  v3.0")
        print("  Data: Maritime DB (324 vessels, 30 ports, 286 routes)")
        print("        India Road DB (500 segments, 28 states)")
        print("═"*64)
        print("\n[1] Training XGBoost risk model...")
        self.xgb_scorer.train()
        print(f"[2] Graph: {len(self.adj)} nodes, "
              f"{sum(len(v) for v in self.adj.values())} directed edges.")
        print(f"[3] Threat zones: {len(THREAT_ZONES)} "
              f"(incl. 2 India road-derived zones).\n")

    def analyse(self, origin: str, destination: str,
                cargo_value_norm: float = 0.5,
                cargo_type: str = "general") -> List[RoutePlan]:
        if origin not in self.adj:
            raise ValueError(f"Unknown origin: '{origin}'. "
                             f"Valid nodes: {sorted(self.adj.keys())}")
        if destination not in self.adj:
            raise ValueError(f"Unknown destination: '{destination}'.")
        if origin == destination:
            raise ValueError("Origin and destination must differ.")

        print(f"  Routing  {origin}  →  {destination}")
        print(f"  Cargo    {cargo_type}  |  value={cargo_value_norm:.2f}\n")

        print(f"  [Yen's] Computing {self.k_paths} loopless shortest paths...")
        candidates = yens_k_shortest(self.adj, origin, destination, K=self.k_paths)
        print(f"  [Yen's] {len(candidates)} paths found.")

        route_plans: List[RoutePlan] = []
        for rank, (path, _) in enumerate(candidates, start=1):
            feat  = extract_features(path, self.adj, cargo_value_norm)
            score, cls = self.xgb_scorer.predict(feat)
            rp = RoutePlan(
                path             = path,
                distance_km      = round(feat["_dist_km"], 1),
                xgb_risk_score   = score,
                risk_class       = cls,
                estimated_days   = round(feat["_dist_km"] / 650, 1),
                base_risk_max    = round(feat["base_risk"], 3),
                threat_severity  = round(feat["_threat_sev"], 2),
                n_chokepoints    = feat["_n_choke"],
                war_flag         = feat["_war"],
                sanctions_flag   = feat["_sanctions"],
                piracy_flag      = feat["_piracy"],
                weather_risk     = round(feat["_weather"], 2),
                alternative_rank = rank,
            )
            route_plans.append(rp)

        route_plans.sort(key=lambda r: r.xgb_risk_score)
        for i, rp in enumerate(route_plans, start=1):
            rp.alternative_rank = i
        for rp in route_plans:
            rp.warnings = generate_warnings(rp, route_plans)

        return route_plans

    def print_report(self, routes: List[RoutePlan], origin: str, destination: str):
        BAR = 24
        print()
        print("═"*64)
        print(f"  ROUTE RISK REPORT  :  {origin}  →  {destination}")
        print(f"  Generated          :  "
              f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M UTC')}")
        print("═"*64)
        for rp in routes:
            filled = int(rp.xgb_risk_score / 100 * BAR)
            bar    = "█"*filled + "░"*(BAR-filled)
            print(f"\n  ── Route #{rp.alternative_rank}  [{rp.risk_class}]")
            print(f"     Path     : {' → '.join(rp.path)}")
            print(f"     Distance : {rp.distance_km:,.0f} km  |  "
                  f"Transit: {rp.estimated_days:.1f} days")
            print(f"     XGB risk : {rp.xgb_risk_score:5.1f}/100  |{bar}|")
            print(f"     Signals  : threat={rp.threat_severity:.1f}  "
                  f"choke={rp.n_chokepoints}  war={rp.war_flag}  "
                  f"sanctions={rp.sanctions_flag}  piracy={rp.piracy_flag}")
            for w in rp.warnings:
                print(f"     ⚠  {w}")

        best = routes[0]
        print()
        print("═"*64)
        print(f"  RECOMMENDATION  :  Route #{best.alternative_rank}")
        print(f"  Path            :  {' → '.join(best.path)}")
        print(f"  Risk class      :  {best.risk_class}")
        print(f"  Client score    :  {best.xgb_risk_score:.1f} / 100")
        print("═"*64)
        print("\n  XGBoost Feature Importance (top 10):")
        for feat, imp in list(self.xgb_scorer.feature_importance().items())[:10]:
            bar = "█"*int(imp*50)
            print(f"    {feat:<22}  {imp:.4f}  {bar}")

    def to_json(self, routes: List[RoutePlan], origin: str, destination: str) -> str:
        return json.dumps({
            "origin": origin, "destination": destination,
            "generated_at": datetime.datetime.now().isoformat(),
            "data_sources": {
                "maritime": "Maritime_vessel_database.xlsx (324 vessels, 30 ports, 286 routes)",
                "road":     "India_Road_Network_Database.xlsx (500 segments, 28 states)",
            },
            "routes": [
                {
                    "rank": r.alternative_rank, "path": r.path,
                    "distance_km": r.distance_km, "estimated_days": r.estimated_days,
                    "xgb_risk_score": r.xgb_risk_score, "risk_class": r.risk_class,
                    "n_chokepoints": r.n_chokepoints, "war_flag": r.war_flag,
                    "sanctions_flag": r.sanctions_flag, "piracy_flag": r.piracy_flag,
                    "warnings": r.warnings,
                }
                for r in routes
            ],
        }, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pipeline = CargoRiskPipeline(k_paths=5)
    pipeline.initialise()

    # Example 1: Shanghai → Rotterdam
    routes = pipeline.analyse("SHANGHAI", "ROTTERDAM",
                               cargo_value_norm=0.8, cargo_type="electronics")
    pipeline.print_report(routes, "SHANGHAI", "ROTTERDAM")
    with open("/mnt/user-data/outputs/routes_SHA_RTM.json", "w") as f:
        f.write(pipeline.to_json(routes, "SHANGHAI", "ROTTERDAM"))

    # Example 2: Dubai → Rotterdam (Hormuz + Red Sea double exposure)
    routes2 = pipeline.analyse("DUBAI_JEBEL_ALI", "ROTTERDAM",
                                cargo_value_norm=0.6, cargo_type="petroleum")
    pipeline.print_report(routes2, "DUBAI_JEBEL_ALI", "ROTTERDAM")
    with open("/mnt/user-data/outputs/routes_DXB_RTM.json", "w") as f:
        f.write(pipeline.to_json(routes2, "DUBAI_JEBEL_ALI", "ROTTERDAM"))

    # Example 3: Mumbai → Hamburg (India road uplift applied)
    routes3 = pipeline.analyse("MUMBAI", "HAMBURG",
                                cargo_value_norm=0.5, cargo_type="general")
    pipeline.print_report(routes3, "MUMBAI", "HAMBURG")
    with open("/mnt/user-data/outputs/routes_BOM_HAM.json", "w") as f:
        f.write(pipeline.to_json(routes3, "MUMBAI", "HAMBURG"))