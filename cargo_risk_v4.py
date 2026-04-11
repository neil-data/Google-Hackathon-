"""
╔══════════════════════════════════════════════════════════════════════════════╗
║               CARGO ROUTE RISK PIPELINE  v4.0                               ║
║                                                                              ║
║   NEW IN v4.0:                                                               ║
║     • news_verification_module  drives live threat zones                    ║
║     • Weather API integrated per edge (wind risk modifier)                  ║
║     • Dijkstra + Yen's output enriched with Google Maps API waypoints       ║
║     • Full JSON contract for Google Maps frontend rendering                  ║
║                                                                              ║
║   Data flow:                                                                 ║
║     NewsVerificationSystem  →  live ThreatZones                             ║
║     WeatherAPI              →  per-edge weather risk                        ║
║     Dijkstra + Yen's        →  K shortest paths                             ║
║     XGBoost                 →  risk score per path                          ║
║     Google Maps JSON        →  frontend polylines + markers                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

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

# ── Import the verification module ───────────────────────────────────────────
from news_verification_module import (
    NewsVerificationSystem,
    ThreatEvent,
    fetch_weather,
    WeatherReport,
    _severity_to_radius,
    MARITIME_REGIONS,
)

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

# Google Maps API key for frontend (leave None → frontend uses static markers)
GOOGLE_MAPS_API_KEY: Optional[str] = None   # ← INSERT YOUR KEY HERE

ALPHA = 0.60    # threat weight in edge cost  (reduced from 0.70 — weather now shares)
BETA  = 0.25    # distance weight
GAMMA = 0.15    # weather weight (new in v4)


# ══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class GeoPoint:
    display_name: str
    lat:          float
    lon:          float
    country:      str
    region:       str = ""

@dataclass
class RoutePlan:
    path:              List[str]
    distance_km:       float
    xgb_risk_score:    float
    risk_class:        str
    estimated_days:    float
    base_risk_max:     float
    threat_severity:   float
    weather_risk_max:  float
    n_chokepoints:     int
    war_flag:          int
    sanctions_flag:    int
    piracy_flag:       int
    alternative_rank:  int
    verified_threats:  int        # threats from verified news events
    live_threats:      int        # total live threats near path
    gmaps_polyline:    List[Dict] # [{lat, lng, label}] for Google Maps
    warnings:          List[str] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# WAYPOINTS  (from Maritime_vessel_database.xlsx — 30 real ports)
# ══════════════════════════════════════════════════════════════════════════════

WAYPOINTS: Dict[str, GeoPoint] = {
    "ROTTERDAM":        GeoPoint("Rotterdam",          51.9244,   4.4777, "Netherlands",  "NW Europe"),
    "HAMBURG":          GeoPoint("Hamburg",             53.5753,   9.9300, "Germany",      "NW Europe"),
    "ANTWERP":          GeoPoint("Antwerp",             51.2652,   4.4069, "Belgium",      "NW Europe"),
    "FELIXSTOWE":       GeoPoint("Felixstowe",          51.9586,   1.3513, "UK",           "NW Europe"),
    "PIRAEUS":          GeoPoint("Piraeus",             37.9475,  23.6478, "Greece",       "SE Europe"),
    "ALGECIRAS":        GeoPoint("Algeciras",           36.1408,  -5.4531, "Spain",        "SW Europe"),
    "LOS_ANGELES":      GeoPoint("Los Angeles",         33.7395,-118.2618, "USA",          "W Americas"),
    "NEW_YORK":         GeoPoint("New York",            40.6892, -74.0445, "USA",          "E Americas"),
    "SAVANNAH":         GeoPoint("Savannah",            32.0835, -81.0998, "USA",          "E Americas"),
    "VANCOUVER":        GeoPoint("Vancouver",           49.2827,-123.1207, "Canada",       "W Americas"),
    "SANTOS":           GeoPoint("Santos",             -23.9618, -46.3322, "Brazil",       "S Americas"),
    "DUBAI_JEBEL_ALI":  GeoPoint("Dubai (Jebel Ali)",  24.9857,  55.0272, "UAE",          "Middle East"),
    "JEDDAH":           GeoPoint("Jeddah",              21.5433,  39.1728, "Saudi Arabia", "Middle East"),
    "MOMBASA":          GeoPoint("Mombasa",             -4.0435,  39.6682, "Kenya",        "E Africa"),
    "DURBAN":           GeoPoint("Durban",             -29.8587,  31.0218, "S. Africa",    "S Africa"),
    "MUMBAI":           GeoPoint("Mumbai",              18.9322,  72.8375, "India",        "South Asia"),
    "KARACHI":          GeoPoint("Karachi",             24.8607,  67.0099, "Pakistan",     "South Asia"),
    "COLOMBO":          GeoPoint("Colombo",              6.9271,  79.8612, "Sri Lanka",    "South Asia"),
    "SINGAPORE":        GeoPoint("Singapore",            1.2897, 103.8501, "Singapore",    "SE Asia"),
    "PORT_KLANG":       GeoPoint("Port Klang",           3.0319, 101.3682, "Malaysia",     "SE Asia"),
    "TANJUNG_PELEPAS":  GeoPoint("Tanjung Pelepas",      1.3628, 103.5501, "Malaysia",     "SE Asia"),
    "LAEM_CHABANG":     GeoPoint("Laem Chabang",        13.0827, 100.8832, "Thailand",     "SE Asia"),
    "MANILA":           GeoPoint("Manila",              14.5995, 120.9842, "Philippines",  "SE Asia"),
    "HONG_KONG":        GeoPoint("Hong Kong",           22.3193, 114.1694, "Hong Kong",    "E Asia"),
    "GUANGZHOU":        GeoPoint("Guangzhou",           23.1291, 113.2644, "China",        "E Asia"),
    "SHANGHAI":         GeoPoint("Shanghai",            31.2304, 121.4737, "China",        "E Asia"),
    "NINGBO":           GeoPoint("Ningbo",              29.8683, 121.5440, "China",        "E Asia"),
    "QINGDAO":          GeoPoint("Qingdao",             36.0671, 120.3826, "China",        "E Asia"),
    "OSAKA":            GeoPoint("Osaka",               34.6937, 135.5023, "Japan",        "E Asia"),
    "BUSAN":            GeoPoint("Busan",               35.1796, 129.0756, "South Korea",  "E Asia"),
}

CHOKEPOINTS = {
    "DUBAI_JEBEL_ALI", "JEDDAH", "MOMBASA", "ALGECIRAS",
    "PIRAEUS", "COLOMBO", "SINGAPORE", "PORT_KLANG", "TANJUNG_PELEPAS",
}

# ══════════════════════════════════════════════════════════════════════════════
# RAW EDGES  (from Maritime_vessel_database.xlsx — 286 real routes)
# ══════════════════════════════════════════════════════════════════════════════

RAW_EDGES: List[Tuple] = [
    ("ALGECIRAS","ANTWERP",1855,0.10),("ALGECIRAS","DUBAI_JEBEL_ALI",5830,0.38),
    ("ALGECIRAS","LAEM_CHABANG",10567,0.10),("ALGECIRAS","MUMBAI",7754,0.10),
    ("ALGECIRAS","NINGBO",10823,0.22),("ALGECIRAS","SANTOS",7943,0.22),
    ("ALGECIRAS","SAVANNAH",6797,0.22),("ALGECIRAS","SHANGHAI",10704,0.38),
    ("ALGECIRAS","SINGAPORE",11641,0.10),("ANTWERP","DURBAN",9390,0.22),
    ("ANTWERP","HONG_KONG",9366,0.10),("ANTWERP","LOS_ANGELES",9033,0.22),
    ("ANTWERP","MANILA",10481,0.22),("ANTWERP","MUMBAI",6885,0.10),
    ("ANTWERP","QINGDAO",8478,0.22),("ANTWERP","ROTTERDAM",74,0.10),
    ("ANTWERP","TANJUNG_PELEPAS",10523,0.38),("ANTWERP","VANCOUVER",7780,0.10),
    ("BUSAN","ALGECIRAS",10794,0.10),("BUSAN","DUBAI_JEBEL_ALI",7055,0.22),
    ("BUSAN","DURBAN",12533,0.10),("BUSAN","MOMBASA",10213,0.10),
    ("BUSAN","MUMBAI",5775,0.22),("BUSAN","NEW_YORK",11254,0.10),
    ("BUSAN","SAVANNAH",11900,0.10),("BUSAN","SINGAPORE",4584,0.10),
    ("COLOMBO","ANTWERP",8397,0.10),("COLOMBO","BUSAN",5912,0.22),
    ("COLOMBO","DURBAN",6623,0.10),("COLOMBO","HAMBURG",8071,0.10),
    ("COLOMBO","JEDDAH",4658,0.38),("COLOMBO","LOS_ANGELES",15106,0.22),
    ("COLOMBO","PIRAEUS",6602,0.10),("COLOMBO","PORT_KLANG",2421,0.10),
    ("COLOMBO","QINGDAO",5235,0.10),("COLOMBO","ROTTERDAM",8401,0.10),
    ("COLOMBO","SAVANNAH",15231,0.38),("COLOMBO","SINGAPORE",2732,0.10),
    ("COLOMBO","TANJUNG_PELEPAS",2698,0.38),("DUBAI_JEBEL_ALI","ALGECIRAS",5830,0.22),
    ("DUBAI_JEBEL_ALI","BUSAN",7055,0.10),("DUBAI_JEBEL_ALI","COLOMBO",3317,0.38),
    ("DUBAI_JEBEL_ALI","DURBAN",6614,0.22),("DUBAI_JEBEL_ALI","JEDDAH",1663,0.22),
    ("DUBAI_JEBEL_ALI","MOMBASA",3629,0.10),("DUBAI_JEBEL_ALI","MUMBAI",1954,0.22),
    ("DUBAI_JEBEL_ALI","QINGDAO",6265,0.22),("DUBAI_JEBEL_ALI","SINGAPORE",5859,0.10),
    ("DUBAI_JEBEL_ALI","TANJUNG_PELEPAS",5826,0.22),("DURBAN","DUBAI_JEBEL_ALI",6614,0.10),
    ("DURBAN","FELIXSTOWE",9547,0.10),("DURBAN","HONG_KONG",10603,0.10),
    ("DURBAN","MANILA",10806,0.22),("DURBAN","MOMBASA",3011,0.10),
    ("DURBAN","MUMBAI",7035,0.22),("DURBAN","NINGBO",11649,0.10),
    ("DURBAN","PORT_KLANG",8299,0.38),("DURBAN","SHANGHAI",11709,0.22),
    ("DURBAN","TANJUNG_PELEPAS",8408,0.10),("DURBAN","VANCOUVER",16949,0.38),
    ("FELIXSTOWE","DUBAI_JEBEL_ALI",5385,0.22),("FELIXSTOWE","LOS_ANGELES",8827,0.22),
    ("FELIXSTOWE","MANILA",10622,0.22),("FELIXSTOWE","MOMBASA",7199,0.22),
    ("FELIXSTOWE","MUMBAI",7101,0.22),("FELIXSTOWE","NEW_YORK",5655,0.10),
    ("FELIXSTOWE","NINGBO",9219,0.22),("FELIXSTOWE","PORT_KLANG",10424,0.38),
    ("FELIXSTOWE","SAVANNAH",6768,0.10),("FELIXSTOWE","SINGAPORE",10746,0.38),
    ("FELIXSTOWE","TANJUNG_PELEPAS",10719,0.38),("GUANGZHOU","BUSAN",2033,0.22),
    ("GUANGZHOU","JEDDAH",7533,0.38),("GUANGZHOU","LAEM_CHABANG",1719,0.38),
    ("GUANGZHOU","MOMBASA",8520,0.22),("GUANGZHOU","SANTOS",17935,0.10),
    ("HAMBURG","ANTWERP",454,0.10),("HAMBURG","COLOMBO",8071,0.22),
    ("HAMBURG","DURBAN",9498,0.22),("HAMBURG","GUANGZHOU",8789,0.10),
    ("HAMBURG","LAEM_CHABANG",8903,0.38),("HAMBURG","MOMBASA",6980,0.38),
    ("HAMBURG","NINGBO",8653,0.10),("HAMBURG","ROTTERDAM",410,0.22),
    ("HAMBURG","SINGAPORE",10151,0.10),("HAMBURG","TANJUNG_PELEPAS",10124,0.10),
    ("HAMBURG","VANCOUVER",7760,0.10),("HONG_KONG","COLOMBO",4053,0.38),
    ("HONG_KONG","DURBAN",10603,0.10),("HONG_KONG","LAEM_CHABANG",1741,0.22),
    ("HONG_KONG","LOS_ANGELES",11671,0.10),("HONG_KONG","MUMBAI",4305,0.10),
    ("HONG_KONG","NEW_YORK",12958,0.10),("HONG_KONG","OSAKA",2489,0.10),
    ("HONG_KONG","PIRAEUS",8549,0.10),("HONG_KONG","PORT_KLANG",2551,0.10),
    ("HONG_KONG","SINGAPORE",2591,0.10),("JEDDAH","ANTWERP",4463,0.10),
    ("JEDDAH","BUSAN",8641,0.10),("JEDDAH","COLOMBO",4658,0.22),
    ("JEDDAH","DURBAN",5782,0.22),("JEDDAH","NEW_YORK",10254,0.22),
    ("JEDDAH","SHANGHAI",8087,0.10),("KARACHI","ANTWERP",5993,0.22),
    ("KARACHI","BUSAN",5990,0.10),("KARACHI","FELIXSTOWE",6209,0.10),
    ("KARACHI","HAMBURG",5657,0.10),("KARACHI","HONG_KONG",4790,0.10),
    ("KARACHI","LOS_ANGELES",13475,0.10),("KARACHI","NEW_YORK",11690,0.10),
    ("KARACHI","PIRAEUS",4321,0.10),("KARACHI","PORT_KLANG",4406,0.22),
    ("KARACHI","QINGDAO",5199,0.22),("KARACHI","TANJUNG_PELEPAS",4712,0.38),
    ("LAEM_CHABANG","HONG_KONG",1741,0.38),("LAEM_CHABANG","JEDDAH",6579,0.22),
    ("LAEM_CHABANG","SINGAPORE",1351,0.10),("LAEM_CHABANG","VANCOUVER",11852,0.10),
    ("LOS_ANGELES","ALGECIRAS",9579,0.10),("LOS_ANGELES","GUANGZHOU",11668,0.22),
    ("LOS_ANGELES","NEW_YORK",3948,0.22),("LOS_ANGELES","NINGBO",10557,0.10),
    ("LOS_ANGELES","OSAKA",9209,0.10),("LOS_ANGELES","PIRAEUS",11125,0.10),
    ("LOS_ANGELES","QINGDAO",10153,0.10),("LOS_ANGELES","SHANGHAI",10456,0.22),
    ("LOS_ANGELES","SINGAPORE",14141,0.10),("MANILA","ALGECIRAS",12053,0.10),
    ("MANILA","BUSAN",2427,0.38),("MANILA","COLOMBO",4565,0.10),
    ("MANILA","FELIXSTOWE",10622,0.10),("MANILA","HAMBURG",10030,0.10),
    ("MANILA","LOS_ANGELES",11759,0.38),("MANILA","NEW_YORK",13675,0.22),
    ("MANILA","NINGBO",1699,0.38),("MANILA","OSAKA",2666,0.22),
    ("MANILA","SINGAPORE",2394,0.10),("MOMBASA","BUSAN",10213,0.38),
    ("MOMBASA","COLOMBO",4625,0.10),("MOMBASA","DUBAI_JEBEL_ALI",3629,0.10),
    ("MOMBASA","DURBAN",3011,0.38),("MOMBASA","HAMBURG",6980,0.22),
    ("MOMBASA","JEDDAH",2846,0.10),("MOMBASA","LAEM_CHABANG",7020,0.10),
    ("MOMBASA","LOS_ANGELES",16000,0.22),("MOMBASA","MUMBAI",4439,0.10),
    ("MOMBASA","NINGBO",9452,0.38),("MOMBASA","OSAKA",10796,0.10),
    ("MOMBASA","PIRAEUS",4953,0.22),("MOMBASA","SAVANNAH",13123,0.10),
    ("MUMBAI","ANTWERP",6885,0.38),("MUMBAI","COLOMBO",1536,0.22),
    ("MUMBAI","GUANGZHOU",4209,0.22),("MUMBAI","KARACHI",892,0.10),
    ("MUMBAI","LAEM_CHABANG",3064,0.38),("MUMBAI","LOS_ANGELES",14041,0.10),
    ("MUMBAI","NEW_YORK",12554,0.10),("MUMBAI","PORT_KLANG",3570,0.22),
    ("MUMBAI","ROTTERDAM",6886,0.22),("MUMBAI","SANTOS",13741,0.10),
    ("NEW_YORK","COLOMBO",14086,0.22),("NEW_YORK","GUANGZHOU",12879,0.22),
    ("NEW_YORK","HAMBURG",6129,0.22),("NEW_YORK","MOMBASA",12287,0.10),
    ("NEW_YORK","MUMBAI",12554,0.10),("NEW_YORK","ROTTERDAM",5858,0.38),
    ("NEW_YORK","SHANGHAI",11860,0.38),("NEW_YORK","SINGAPORE",15342,0.38),
    ("NINGBO","ALGECIRAS",10823,0.22),("NINGBO","ANTWERP",9106,0.10),
    ("NINGBO","DUBAI_JEBEL_ALI",6497,0.38),("NINGBO","DURBAN",11649,0.22),
    ("NINGBO","FELIXSTOWE",9219,0.38),("NINGBO","JEDDAH",8133,0.10),
    ("NINGBO","KARACHI",5365,0.22),("NINGBO","LAEM_CHABANG",2828,0.22),
    ("NINGBO","LOS_ANGELES",10557,0.10),("NINGBO","MUMBAI",5042,0.10),
    ("NINGBO","PORT_KLANG",3664,0.10),("NINGBO","SANTOS",18646,0.10),
    ("NINGBO","SAVANNAH",12724,0.22),("NINGBO","SINGAPORE",3687,0.10),
    ("OSAKA","ALGECIRAS",11161,0.22),("OSAKA","COLOMBO",6454,0.38),
    ("OSAKA","HAMBURG",8893,0.22),("OSAKA","KARACHI",6577,0.22),
    ("OSAKA","LAEM_CHABANG",4226,0.22),("OSAKA","LOS_ANGELES",9209,0.38),
    ("OSAKA","MUMBAI",6360,0.22),("OSAKA","QINGDAO",1378,0.10),
    ("OSAKA","ROTTERDAM",9272,0.10),("OSAKA","SANTOS",18809,0.10),
    ("OSAKA","SHANGHAI",1363,0.22),("PIRAEUS","BUSAN",8836,0.10),
    ("PIRAEUS","DURBAN",7579,0.22),("PIRAEUS","LOS_ANGELES",11125,0.10),
    ("PIRAEUS","MANILA",9641,0.10),("PIRAEUS","MUMBAI",5182,0.10),
    ("PIRAEUS","NEW_YORK",7926,0.22),("PIRAEUS","PORT_KLANG",8725,0.22),
    ("PIRAEUS","ROTTERDAM",2153,0.22),("PORT_KLANG","ALGECIRAS",11305,0.10),
    ("PORT_KLANG","HAMBURG",9831,0.10),("PORT_KLANG","HONG_KONG",2551,0.10),
    ("PORT_KLANG","LAEM_CHABANG",1119,0.10),("PORT_KLANG","NEW_YORK",15131,0.22),
    ("PORT_KLANG","NINGBO",3664,0.38),("PORT_KLANG","QINGDAO",4160,0.10),
    ("PORT_KLANG","ROTTERDAM",10213,0.10),("PORT_KLANG","SANTOS",15840,0.10),
    ("PORT_KLANG","SAVANNAH",16102,0.10),("PORT_KLANG","VANCOUVER",12802,0.10),
    ("QINGDAO","ALGECIRAS",10230,0.10),("QINGDAO","DUBAI_JEBEL_ALI",6265,0.22),
    ("QINGDAO","DURBAN",11851,0.38),("QINGDAO","FELIXSTOWE",8582,0.38),
    ("QINGDAO","GUANGZHOU",1593,0.10),("QINGDAO","HONG_KONG",1642,0.10),
    ("QINGDAO","SHANGHAI",547,0.22),("QINGDAO","TANJUNG_PELEPAS",4231,0.10),
    ("QINGDAO","VANCOUVER",8653,0.10),("ROTTERDAM","ALGECIRAS",1922,0.38),
    ("ROTTERDAM","ANTWERP",74,0.38),("ROTTERDAM","BUSAN",8936,0.22),
    ("ROTTERDAM","COLOMBO",8401,0.10),("ROTTERDAM","DUBAI_JEBEL_ALI",5176,0.10),
    ("ROTTERDAM","DURBAN",9456,0.38),("ROTTERDAM","GUANGZHOU",9199,0.10),
    ("ROTTERDAM","HONG_KONG",9326,0.10),("ROTTERDAM","KARACHI",5995,0.10),
    ("ROTTERDAM","SAVANNAH",6974,0.22),("ROTTERDAM","TANJUNG_PELEPAS",10508,0.10),
    ("SANTOS","BUSAN",18691,0.10),("SANTOS","JEDDAH",10534,0.10),
    ("SANTOS","LOS_ANGELES",9943,0.38),("SANTOS","MANILA",18327,0.22),
    ("SANTOS","MOMBASA",9419,0.22),("SANTOS","PIRAEUS",10027,0.10),
    ("SANTOS","SAVANNAH",7244,0.10),("SANTOS","SHANGHAI",18569,0.10),
    ("SAVANNAH","GUANGZHOU",13689,0.10),("SAVANNAH","JEDDAH",11305,0.10),
    ("SAVANNAH","MANILA",14316,0.10),("SAVANNAH","MOMBASA",13123,0.10),
    ("SAVANNAH","MUMBAI",13700,0.10),("SAVANNAH","QINGDAO",12114,0.22),
    ("SAVANNAH","ROTTERDAM",6974,0.22),("SAVANNAH","SANTOS",7244,0.10),
    ("SAVANNAH","VANCOUVER",3958,0.10),("SHANGHAI","BUSAN",832,0.10),
    ("SHANGHAI","FELIXSTOWE",9088,0.10),("SHANGHAI","KARACHI",5339,0.10),
    ("SHANGHAI","LAEM_CHABANG",2916,0.10),("SHANGHAI","LOS_ANGELES",10456,0.38),
    ("SHANGHAI","NEW_YORK",11860,0.22),("SHANGHAI","NINGBO",152,0.22),
    ("SHANGHAI","PIRAEUS",8548,0.10),("SHANGHAI","VANCOUVER",9025,0.10),
    ("SINGAPORE","ALGECIRAS",11641,0.22),("SINGAPORE","GUANGZHOU",2632,0.22),
    ("SINGAPORE","HAMBURG",10151,0.10),("SINGAPORE","LOS_ANGELES",14141,0.10),
    ("SINGAPORE","MUMBAI",3907,0.38),("SINGAPORE","PORT_KLANG",337,0.22),
    ("SINGAPORE","SANTOS",15934,0.10),("SINGAPORE","SHANGHAI",3810,0.22),
    ("SINGAPORE","VANCOUVER",12825,0.10),("TANJUNG_PELEPAS","ANTWERP",10523,0.10),
    ("TANJUNG_PELEPAS","DURBAN",8408,0.10),("TANJUNG_PELEPAS","GUANGZHOU",2638,0.38),
    ("TANJUNG_PELEPAS","HONG_KONG",2598,0.10),("TANJUNG_PELEPAS","MOMBASA",7124,0.22),
    ("TANJUNG_PELEPAS","OSAKA",4969,0.22),("TANJUNG_PELEPAS","PIRAEUS",9030,0.10),
    ("TANJUNG_PELEPAS","ROTTERDAM",10508,0.22),("VANCOUVER","ANTWERP",7780,0.22),
    ("VANCOUVER","DUBAI_JEBEL_ALI",11755,0.10),("VANCOUVER","FELIXSTOWE",7597,0.10),
    ("VANCOUVER","HONG_KONG",10251,0.10),("VANCOUVER","JEDDAH",11947,0.10),
    ("VANCOUVER","MANILA",10548,0.22),("VANCOUVER","MUMBAI",12268,0.10),
    ("VANCOUVER","NINGBO",9142,0.10),("VANCOUVER","QINGDAO",8653,0.38),
    ("VANCOUVER","SANTOS",11106,0.10),("VANCOUVER","SAVANNAH",3958,0.10),
]


# ══════════════════════════════════════════════════════════════════════════════
# MATH UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def haversine(lat1, lon1, lat2, lon2) -> float:
    """d = 2R·arcsin(√(sin²(Δlat/2) + cos·cos·sin²(Δlon/2)))"""
    R  = 6_371
    p1, p2 = math.radians(lat1), math.radians(lat2)
    Δp = math.radians(lat2-lat1)
    Δl = math.radians(lon2-lon1)
    a  = math.sin(Δp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(Δl/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def intermediate_points(p1: GeoPoint, p2: GeoPoint,
                        n: int = 8) -> List[Tuple[float, float]]:
    """Generate n intermediate lat/lon points along a great-circle arc."""
    points = []
    for i in range(n + 1):
        t   = i / n
        lat = p1.lat + t * (p2.lat - p1.lat)
        lon = p1.lon + t * (p2.lon - p1.lon)
        points.append((lat, lon))
    return points


# ══════════════════════════════════════════════════════════════════════════════
# LIVE THREAT SCORER  (replaces static RBF zones with live news events)
# ══════════════════════════════════════════════════════════════════════════════

class LiveThreatScorer:
    """
    RBF threat score backed by live verified news events.
    Falls back to static zones if no live events available.

    FORMULA  (same RBF as v3 but severity = adj_severity from verification):
      threat(p) = Σ_z  adj_severity_z · max(0, 1 − d(p,z)/radius_z)
      clipped to [0, 10]
    """

    # Static fallback zones (v3 calibrated values)
    STATIC_ZONES = [
        {"lat":15.0,"lon":42.5,"threat_type":"war",       "adj_severity":9.5,"radius_km":650},
        {"lat":31.5,"lon":34.4,"threat_type":"war",       "adj_severity":7.8,"radius_km":300},
        {"lat":26.57,"lon":56.27,"threat_type":"piracy",  "adj_severity":8.8,"radius_km":350},
        {"lat":12.5,"lon":46.0,"threat_type":"piracy",    "adj_severity":7.5,"radius_km":450},
        {"lat":3.0, "lon":3.0, "threat_type":"piracy",    "adj_severity":6.5,"radius_km":500},
        {"lat":27.0,"lon":51.0,"threat_type":"sanctions", "adj_severity":7.0,"radius_km":420},
        {"lat":13.0,"lon":88.0,"threat_type":"natural",   "adj_severity":5.5,"radius_km":700},
        {"lat":30.5,"lon":32.3,"threat_type":"natural",   "adj_severity":4.5,"radius_km":150},
        {"lat":14.0,"lon":114.0,"threat_type":"war",      "adj_severity":5.5,"radius_km":600},
        {"lat":46.5,"lon":31.0,"threat_type":"war",       "adj_severity":7.5,"radius_km":500},
    ]

    def __init__(self, live_zones: Optional[List[Dict]] = None):
        self.zones = live_zones if live_zones else self.STATIC_ZONES
        print(f"  [ThreatScorer] {len(self.zones)} zones active "
              f"({'live' if live_zones else 'static fallback'})")

    def score(self, lat: float, lon: float) -> float:
        total = 0.0
        for z in self.zones:
            d = haversine(lat, lon, z["lat"], z["lon"])
            influence = max(0.0, 1.0 - d / z["radius_km"])
            total += z["adj_severity"] * influence
        return min(total, 10.0)

    def edge_score(self, n1: str, n2: str, samples: int = 6) -> float:
        p1, p2 = WAYPOINTS[n1], WAYPOINTS[n2]
        return max(
            self.score(p1.lat + i/samples*(p2.lat-p1.lat),
                       p1.lon + i/samples*(p2.lon-p1.lon))
            for i in range(samples+1)
        )

    def zones_near_path(self, path: List[str], radius_km: float = 700) -> List[Dict]:
        near = []
        seen = set()
        for node in path:
            gp = WAYPOINTS[node]
            for z in self.zones:
                key = f"{z['lat']:.2f},{z['lon']:.2f}"
                if key not in seen:
                    d = haversine(gp.lat, gp.lon, z["lat"], z["lon"])
                    if d <= radius_km:
                        near.append(z)
                        seen.add(key)
        return near


# ══════════════════════════════════════════════════════════════════════════════
# WEATHER RISK PER EDGE
# ══════════════════════════════════════════════════════════════════════════════

def edge_weather_risk(n1: str, n2: str) -> float:
    """
    Sample weather at the midpoint of edge n1→n2.
    Returns risk_factor ∈ [0,1] from WeatherReport.
    """
    p1, p2 = WAYPOINTS[n1], WAYPOINTS[n2]
    mid_lat = (p1.lat + p2.lat) / 2
    mid_lon = (p1.lon + p2.lon) / 2
    wr = fetch_weather(mid_lat, mid_lon)
    return wr.risk_factor


# ══════════════════════════════════════════════════════════════════════════════
# EDGE WEIGHT  (v4 — three-component)
#   W(u,v) = α·threat_norm + β·dist_norm + γ·weather_risk
#   α=0.60, β=0.25, γ=0.15
# ══════════════════════════════════════════════════════════════════════════════

def edge_weight(adj: dict, u: str, v: str,
                scorer: LiveThreatScorer,
                weather_cache: Dict[str, float]) -> float:
    data         = adj[u][v]
    threat_norm  = scorer.edge_score(u, v) / 10.0
    dist_norm    = data["dist"] / 20_000.0
    ew_key       = f"{u}|{v}"
    if ew_key not in weather_cache:
        weather_cache[ew_key] = edge_weather_risk(u, v)
    weather_norm = weather_cache[ew_key]
    return ALPHA * threat_norm + BETA * dist_norm + GAMMA * weather_norm


# ══════════════════════════════════════════════════════════════════════════════
# DIJKSTRA
# ══════════════════════════════════════════════════════════════════════════════

def dijkstra(adj, src, tgt, scorer, weather_cache):
    dist = {n: float("inf") for n in adj}
    dist[src] = 0.0
    prev = {}
    pq   = [(0.0, src)]
    while pq:
        d_u, u = heapq.heappop(pq)
        if d_u > dist[u]: continue
        if u == tgt: break
        for v in adj.get(u, {}):
            w   = edge_weight(adj, u, v, scorer, weather_cache)
            alt = dist[u] + w
            if alt < dist[v]:
                dist[v] = alt; prev[v] = u
                heapq.heappush(pq, (alt, v))
    path, cur = [], tgt
    while cur is not None:
        path.append(cur); cur = prev.get(cur)
    path.reverse()
    return (path, dist[tgt]) if path and path[0] == src else (None, float("inf"))


# ══════════════════════════════════════════════════════════════════════════════
# YEN'S K-SHORTEST
# ══════════════════════════════════════════════════════════════════════════════

def path_cost(adj, path, scorer, weather_cache):
    return sum(edge_weight(adj, path[i], path[i+1], scorer, weather_cache)
               for i in range(len(path)-1))

def yens_k_shortest(adj0, src, tgt, scorer, weather_cache, K=5):
    adj   = json.loads(json.dumps(adj0))
    first, fc = dijkstra(adj, src, tgt, scorer, weather_cache)
    if not first: return []
    A = [(first, fc)]; B = []; seen = {tuple(first)}

    for k in range(1, K):
        prev_path = A[k-1][0]
        for i in range(len(prev_path)-1):
            spur = prev_path[i]; root = prev_path[:i+1]
            root_c = path_cost(adj, root, scorer, weather_cache)
            removed_edges = []
            for ap, _ in A:
                if len(ap)>i and ap[:i+1]==root and adj.get(ap[i],{}).get(ap[i+1]):
                    saved = adj[ap[i]].pop(ap[i+1])
                    removed_edges.append((ap[i], ap[i+1], saved))
            removed_nodes = []
            for node in root[:-1]:
                if node == spur: continue
                out_e = adj.pop(node, {})
                in_e  = {x: adj[x].pop(node) for x in list(adj) if node in adj.get(x,{})}
                removed_nodes.append((node, out_e, in_e))
            sp, sc = dijkstra(adj, spur, tgt, scorer, weather_cache)
            for node, out_e, in_e in reversed(removed_nodes):
                adj[node] = out_e
                for x, d in in_e.items(): adj.setdefault(x,{})[node] = d
            for u, v, d in removed_edges: adj.setdefault(u,{})[v] = d
            if sp and sp[0]==spur:
                full = root[:-1]+sp; total = root_c+sc; key = tuple(full)
                if key not in seen: seen.add(key); heapq.heappush(B,(total,full))
        if not B: break
        c, p = heapq.heappop(B); A.append((p, c))
    return A


# ══════════════════════════════════════════════════════════════════════════════
# XGBOOST SCORER
# ══════════════════════════════════════════════════════════════════════════════

class XGBRiskScorer:
    FEATURE_NAMES = [
        "dist_norm","hops_norm","base_risk","threat_sev_norm","n_threats_norm",
        "choke_flag","n_choke_norm","cargo_val","mode_sea","seasonal_sin",
        "weather_norm","sanctions_flag","war_flag","piracy_flag",
        "verified_ratio",    # NEW in v4: fraction of threats that are verified
        "live_news_flag",    # NEW in v4: 1 if live news events present
    ]

    def __init__(self):
        self.reg     = xgb.XGBRegressor(n_estimators=300, max_depth=6,
                           learning_rate=0.04, subsample=0.80,
                           colsample_bytree=0.80, reg_alpha=0.1,
                           reg_lambda=1.0, random_state=42, verbosity=0)
        self.scaler  = StandardScaler()
        self._fitted = False

    def _make_data(self, n=3000):
        rng = np.random.default_rng(42)
        X, y = [], []
        for _ in range(n):
            br=rng.uniform(0.02,0.40); ts=rng.uniform(0,10); nc=rng.integers(0,5)
            wf=rng.integers(0,2); sf=rng.integers(0,2); pf=rng.integers(0,2)
            wr=rng.uniform(0,1);  cv=rng.uniform(0,1);  vr=rng.uniform(0,1)
            ln=rng.integers(0,2)
            label = float(np.clip(
                30*br + 6*ts + 8*nc + 15*wf + 12*sf + 4*wr*5 + 5*cv
                + 8*vr*(1-vr)*ln + rng.normal(0,3), 0, 100))
            X.append([
                rng.uniform(0.03,1), rng.uniform(0.1,1), br,
                ts/10, rng.integers(0,9)/8, int(nc>0), nc/4, cv, 1,
                math.sin(2*math.pi*rng.integers(1,13)/12),
                wr, sf, wf, pf, vr, float(ln),
            ])
            y.append(label)
        return np.array(X), np.array(y)

    def train(self):
        print("  [XGB] Training (n=3000, 16 features incl. verified_ratio, live_news)...")
        X, y = self._make_data()
        Xs   = self.scaler.fit_transform(X)
        Xt, Xe, yt, ye = train_test_split(Xs, y, test_size=0.2, random_state=42)
        self.reg.fit(Xt, yt, eval_set=[(Xe, ye)], verbose=False)
        r2 = self.reg.score(Xe, ye)
        print(f"  [XGB] R² = {r2:.4f}")
        self._fitted = True

    def predict(self, feat: dict) -> Tuple[float, str]:
        x  = np.array([[feat.get(k, 0.0) for k in self.FEATURE_NAMES]])
        xs = self.scaler.transform(x)
        s  = float(np.clip(self.reg.predict(xs)[0], 0, 100))
        cls = "LOW" if s<25 else "MEDIUM" if s<50 else "HIGH" if s<75 else "CRITICAL"
        return round(s, 2), cls

    def importance(self): return dict(sorted(
        zip(self.FEATURE_NAMES, self.reg.feature_importances_),
        key=lambda x: x[1], reverse=True))


# ══════════════════════════════════════════════════════════════════════════════
# GOOGLE MAPS OUTPUT BUILDER
#   Builds a JSON payload the frontend can directly feed to the Maps JS API
# ══════════════════════════════════════════════════════════════════════════════

def build_gmaps_payload(routes: List[RoutePlan],
                        threat_zones: List[Dict],
                        origin: str, destination: str,
                        api_key: Optional[str] = None) -> Dict:
    """
    Produces a complete Google Maps frontend payload:
      - polylines[]   : path coordinates for each route
      - markers[]     : port markers with risk labels
      - threat_circles[]: semi-transparent risk zones
      - meta          : risk scores, classes, warnings
    """
    risk_colors = {
        "LOW":      "#22c55e",
        "MEDIUM":   "#f59e0b",
        "HIGH":     "#f97316",
        "CRITICAL": "#ef4444",
    }

    polylines = []
    for rp in routes:
        coords = []
        for i, node in enumerate(rp.path):
            gp = WAYPOINTS[node]
            coords.append({
                "lat": gp.lat, "lng": gp.lon,
                "label": gp.display_name,
                "is_chokepoint": node in CHOKEPOINTS,
                "is_origin": (i == 0),
                "is_destination": (i == len(rp.path)-1),
            })
            # Intermediate arc points between nodes
            if i < len(rp.path)-1:
                n2   = rp.path[i+1]
                gp2  = WAYPOINTS[n2]
                pts  = intermediate_points(gp, gp2, n=6)
                for lat, lon in pts[1:-1]:
                    coords.append({"lat": lat, "lng": lon})
        polylines.append({
            "rank":        rp.alternative_rank,
            "risk_class":  rp.risk_class,
            "risk_score":  rp.xgb_risk_score,
            "color":       risk_colors[rp.risk_class],
            "opacity":     1.0 if rp.alternative_rank == 1 else 0.45,
            "weight":      5   if rp.alternative_rank == 1 else 2,
            "distance_km": rp.distance_km,
            "days":        rp.estimated_days,
            "path":        coords,
            "warnings":    rp.warnings,
        })

    # Port markers
    markers = []
    all_nodes = {n for rp in routes for n in rp.path}
    for node in all_nodes:
        gp = WAYPOINTS[node]
        markers.append({
            "id":          node,
            "lat":         gp.lat,
            "lng":         gp.lon,
            "label":       gp.display_name,
            "country":     gp.country,
            "is_origin":      node == origin,
            "is_destination": node == destination,
            "is_chokepoint":  node in CHOKEPOINTS,
            "icon": ("origin" if node==origin else
                     "destination" if node==destination else
                     "chokepoint" if node in CHOKEPOINTS else "port"),
        })

    # Threat zone circles
    circles = []
    for z in threat_zones:
        circles.append({
            "lat":        z["lat"],
            "lng":        z["lon"],
            "radius_m":   z["radius_km"] * 1000,
            "threat_type":z["threat_type"],
            "severity":   z["adj_severity"],
            "verified":   z.get("verified", False),
            "fill_color": {
                "war":       "#ef444488",
                "piracy":    "#f9731688",
                "terrorism": "#dc262688",
                "sanctions": "#3b82f688",
                "natural":   "#22c55e88",
                "traffic":   "#94a3b888",
            }.get(z["threat_type"], "#94a3b888"),
        })

    return {
        "api_key_present": bool(api_key),
        "google_maps_api_key": api_key or "INSERT_GOOGLE_MAPS_API_KEY_HERE",
        "origin":      origin,
        "destination": destination,
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "best_route":  routes[0].alternative_rank if routes else None,
        "polylines":   polylines,
        "markers":     markers,
        "threat_circles": circles,
        "summary": {
            "n_routes":     len(routes),
            "best_score":   routes[0].xgb_risk_score if routes else None,
            "best_class":   routes[0].risk_class if routes else None,
            "best_path":    routes[0].path if routes else [],
        }
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE  v4
# ══════════════════════════════════════════════════════════════════════════════

class CargoRiskPipelineV4:

    def __init__(self, k_paths: int = 5,
                 use_live_news: bool = True,
                 use_html_scraper: bool = False):
        self.k_paths     = k_paths
        self.use_live    = use_live_news
        self.adj         = {s: {} for s in set(e[0] for e in RAW_EDGES)}
        for src, tgt, dist, risk in RAW_EDGES:
            self.adj.setdefault(src, {})[tgt] = {"dist": dist, "base_risk": risk}
        self.nvs         = NewsVerificationSystem(use_html_scraper=use_html_scraper)
        self.xgb         = XGBRiskScorer()
        self.scorer: Optional[LiveThreatScorer] = None
        self.live_zones: List[Dict] = []
        self._weather_cache: Dict[str, float] = {}

    def initialise(self):
        print("\n" + "═"*66)
        print("  CARGO RISK PIPELINE  v4.0")
        print("  News Verification + Weather + Dijkstra/Yen's + Google Maps")
        print("═"*66)

        if self.use_live:
            print("\n[1] Running News Verification System...")
            events      = self.nvs.run()
            self.live_zones = self.nvs.to_threat_zones_format(events)
            print(f"    Live threat zones: {len(self.live_zones)}")
        else:
            print("\n[1] Skipping live news (use_live_news=False) — using static zones.")

        self.scorer = LiveThreatScorer(
            self.live_zones if self.live_zones else None
        )

        print("\n[2] Training XGBoost...")
        self.xgb.train()

        print(f"\n[3] Graph: {len(self.adj)} nodes, "
              f"{sum(len(v) for v in self.adj.values())} directed edges.")
        print("[4] Pipeline ready.\n")

    def analyse(self, origin: str, destination: str,
                cargo_value_norm: float = 0.5,
                cargo_type: str = "general") -> Tuple[List[RoutePlan], Dict]:

        if origin not in self.adj: raise ValueError(f"Unknown origin: {origin}")
        if destination not in self.adj: raise ValueError(f"Unknown dest: {destination}")

        print(f"\n  ─── Routing {origin} → {destination} "
              f"[{cargo_type}, val={cargo_value_norm:.2f}] ───")

        # Yen's K-shortest
        print(f"  [Yen's] Computing {self.k_paths} loopless paths...")
        candidates = yens_k_shortest(self.adj, origin, destination,
                                      self.scorer, self._weather_cache, K=self.k_paths)
        print(f"  [Yen's] {len(candidates)} paths found.")

        route_plans: List[RoutePlan] = []
        for rank, (path, _) in enumerate(candidates, start=1):
            dist_km    = sum(self.adj[path[i]][path[i+1]]["dist"]
                            for i in range(len(path)-1))
            base_risks = [self.adj[path[i]][path[i+1]]["base_risk"]
                         for i in range(len(path)-1)]
            edge_threats = [self.scorer.edge_score(path[i], path[i+1])
                           for i in range(len(path)-1)]
            weather_vals = [self._weather_cache.get(f"{path[i]}|{path[i+1]}", 0.0)
                           for i in range(len(path)-1)]
            near        = self.scorer.zones_near_path(path)
            n_choke     = sum(1 for n in path if n in CHOKEPOINTS)
            war_f       = int(any(z["threat_type"] in ("war","terrorism") for z in near))
            san_f       = int(any(z["threat_type"]=="sanctions" for z in near))
            piracy_f    = int(any(z["threat_type"]=="piracy" for z in near))
            verified_n  = sum(1 for z in near if z.get("verified", False))
            live_flag   = int(bool(self.live_zones))
            ver_ratio   = verified_n / max(len(near), 1)
            month       = datetime.datetime.now().month

            feat = {
                "dist_norm":      dist_km/20_000,
                "hops_norm":      len(path)/10,
                "base_risk":      max(base_risks) if base_risks else 0,
                "threat_sev_norm":max(edge_threats)/10 if edge_threats else 0,
                "n_threats_norm": len(near)/max(len(self.scorer.zones),1),
                "choke_flag":     int(n_choke>0),
                "n_choke_norm":   n_choke/5,
                "cargo_val":      cargo_value_norm,
                "mode_sea":       1,
                "seasonal_sin":   math.sin(2*math.pi*month/12),
                "weather_norm":   max(weather_vals) if weather_vals else 0,
                "sanctions_flag": san_f,
                "war_flag":       war_f,
                "piracy_flag":    piracy_f,
                "verified_ratio": ver_ratio,
                "live_news_flag": float(live_flag),
            }

            score, cls = self.xgb.predict(feat)

            # Google Maps polyline waypoints
            gmaps_pts = []
            for i, node in enumerate(path):
                gp = WAYPOINTS[node]
                gmaps_pts.append({"lat": gp.lat, "lng": gp.lon,
                                   "label": gp.display_name})

            rp = RoutePlan(
                path=path, distance_km=round(dist_km,1),
                xgb_risk_score=score, risk_class=cls,
                estimated_days=round(dist_km/650,1),
                base_risk_max=round(max(base_risks) if base_risks else 0, 3),
                threat_severity=round(max(edge_threats) if edge_threats else 0, 2),
                weather_risk_max=round(max(weather_vals) if weather_vals else 0, 3),
                n_chokepoints=n_choke,
                war_flag=war_f, sanctions_flag=san_f, piracy_flag=piracy_f,
                alternative_rank=rank,
                verified_threats=verified_n, live_threats=len(near),
                gmaps_polyline=gmaps_pts,
            )
            route_plans.append(rp)

        route_plans.sort(key=lambda r: r.xgb_risk_score)
        for i, rp in enumerate(route_plans, start=1): rp.alternative_rank = i
        for rp in route_plans: rp.warnings = self._warnings(rp, route_plans)

        gmaps_payload = build_gmaps_payload(
            route_plans, self.live_zones or self.scorer.STATIC_ZONES,
            origin, destination, GOOGLE_MAPS_API_KEY
        )
        return route_plans, gmaps_payload

    def _warnings(self, rp: RoutePlan, all_routes: List[RoutePlan]) -> List[str]:
        flags = []
        if rp.xgb_risk_score >= 75:  flags.append("CRITICAL: score ≥ 75 — not recommended.")
        if rp.n_chokepoints > 1:
            chokes = [n for n in rp.path if n in CHOKEPOINTS]
            flags.append(f"MULTI-CHOKEPOINT: {', '.join(chokes)}.")
        if rp.war_flag:      flags.append("WAR-ZONE ADJACENT.")
        if rp.piracy_flag:   flags.append("PIRACY ZONE NEAR ROUTE.")
        if rp.sanctions_flag:flags.append("SANCTIONS REGION.")
        if rp.weather_risk_max > 0.5:
            flags.append(f"HIGH WEATHER RISK: {rp.weather_risk_max:.2f}/1.00.")
        if rp.live_threats > 0 and rp.verified_threats == 0:
            flags.append("NO VERIFIED NEWS EVENTS — treat threat data with caution.")
        safer = [r for r in all_routes
                 if r.xgb_risk_score < rp.xgb_risk_score-10
                 and r.alternative_rank != rp.alternative_rank]
        if safer:
            flags.append(f"SAFER: Route #{min(safer,key=lambda r:r.xgb_risk_score).alternative_rank} "
                         f"scores lower.")
        return flags

    def print_report(self, routes, origin, destination):
        BAR = 22
        print(f"\n{'═'*66}")
        print(f"  REPORT: {origin} → {destination}  "
              f"({datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')})")
        print(f"{'═'*66}")
        for rp in routes:
            filled = int(rp.xgb_risk_score/100*BAR)
            bar    = "█"*filled + "░"*(BAR-filled)
            print(f"\n  Route #{rp.alternative_rank} [{rp.risk_class}]")
            print(f"    {' → '.join(rp.path)}")
            print(f"    {rp.distance_km:,.0f} km  |  {rp.estimated_days:.1f} d  |  "
                  f"XGB={rp.xgb_risk_score:.1f}  |{bar}|")
            print(f"    threat={rp.threat_severity:.1f}  weather={rp.weather_risk_max:.2f}  "
                  f"choke={rp.n_chokepoints}  verified={rp.verified_threats}/{rp.live_threats}")
            for w in rp.warnings: print(f"    ⚠ {w}")
        best = routes[0]
        print(f"\n  ► RECOMMENDED: Route #{best.alternative_rank}  "
              f"[{best.risk_class}]  Score={best.xgb_risk_score:.1f}/100")
        print(f"{'═'*66}")
        print("\n  XGBoost importance (top 8):")
        for f, i in list(self.xgb.importance().items())[:8]:
            print(f"    {f:<25} {i:.4f}  {'█'*int(i*40)}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pipeline = CargoRiskPipelineV4(k_paths=5, use_live_news=True)
    pipeline.initialise()

    for origin, dest, cargo, val in [
        ("SHANGHAI",       "ROTTERDAM",    "electronics", 0.8),
        ("DUBAI_JEBEL_ALI","ROTTERDAM",    "petroleum",   0.6),
        ("MUMBAI",         "HAMBURG",      "general",     0.5),
    ]:
        routes, gmaps = pipeline.analyse(origin, dest, val, cargo)
        pipeline.print_report(routes, origin, dest)

        # Save Google Maps payload
        fname = f"/mnt/user-data/outputs/gmaps_{origin}_{dest}.json"
        with open(fname, "w") as f:
            json.dump(gmaps, f, indent=2)
        print(f"  Google Maps payload → {fname}")
