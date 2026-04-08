"""
╔══════════════════════════════════════════════════════════════════════════════╗
║               CARGO ROUTE RISK PIPELINE  v2.0                               ║
║                                                                              ║
║   Algorithms  :  Dijkstra  ·  Yen's K-Shortest Paths  ·  XGBoost            ║
║   Risk Input  :  Static threat table + geo-distance weighting                ║
║   Output      :  K ranked routes + XGBoost risk score (0-100) per route      ║
╚══════════════════════════════════════════════════════════════════════════════╝

KEY MATHEMATICAL FOUNDATIONS
─────────────────────────────────────────────────────────────────────────────

  1. HAVERSINE DISTANCE  (great-circle distance between two geo-points)
     ─────────────────────────────────────────────────────────────────
     a = sin²(Δlat/2) + cos(lat₁)·cos(lat₂)·sin²(Δlon/2)
     d = 2R · arcsin(√a)          R = 6 371 km

  2. EDGE THREAT WEIGHT  (used as edge cost in Dijkstra / Yen's)
     ─────────────────────────────────────────────────────────────────
     W(u,v) = α · threat(u,v)  +  β · dist_norm(u,v)

     where   α = 0.70  (threat importance)
             β = 0.30  (distance importance)
             threat(u,v)  = base_risk(u,v)              ∈ [0, 1]
             dist_norm(u,v) = dist_km(u,v) / 15 000     ∈ [0, 1]

  3. DIJKSTRA  (single-source shortest path on threat-weighted graph)
     ─────────────────────────────────────────────────────────────────
     Greedy relaxation:
       d[v] = min( d[v] ,  d[u] + W(u,v) )   ∀ (u,v) ∈ E

     Complexity:  O((V + E) log V)  with a min-heap priority queue

  4. YEN'S K-SHORTEST LOOPLESS PATHS
     ─────────────────────────────────────────────────────────────────
     For k = 1 … K:
       For each spur node  i  in A[k-1]:
         root_path   = A[k-1][0 … i]
         spur_path   = Dijkstra( G - removed_edges - root_nodes,
                                 spur_node → target )
         candidate   = root_path ⊕ spur_path
         cost        = Σ W(e)  for e in candidate
       A[k] = lowest-cost candidate not already in A

     Complexity:  O(K · V · (V + E) log V)

  5. XGBOOST RISK SCORING
     ─────────────────────────────────────────────────────────────────
     Gradient Boosted Trees:
       F_m(x) = F_{m-1}(x) + η · h_m(x)

     where h_m is a regression tree fitted to the negative gradient
     (pseudo-residuals) of the loss L:
       r_i = -[ ∂L(y_i, F(x_i)) / ∂F(x_i) ]    for MSE: r_i = y_i - Ŷ_i

     Final risk score  Ŷ ∈ [0, 100]  passed through:
       Risk class = LOW    if Ŷ < 25
                    MEDIUM if 25 ≤ Ŷ < 50
                    HIGH   if 50 ≤ Ŷ < 75
                    CRITICAL if Ŷ ≥ 75

  6. COMPOSITE RISK GROUND-TRUTH  (training label formula)
     ─────────────────────────────────────────────────────────────────
     y = 30·base_risk + 6·threat_severity + 8·n_chokepoints
         + 15·war_flag + 12·sanctions_flag + 4·weather_risk
         + 5·cargo_value  +  ε,    ε ~ N(0, 3)

     clipped to [0, 100]

  7. FEATURE VECTOR  x ∈ ℝ¹⁴  fed into XGBoost
     ─────────────────────────────────────────────────────────────────
     x = [ dist_norm,  hops_norm,  base_risk,
            threat_sev_norm,  n_threats_norm,  chokepoint_flag,
            n_choke_norm,  cargo_val,  mode_sea,
            sin(2π·month/12),  weather_norm,
            sanctions_flag,  war_flag,  piracy_flag ]

─────────────────────────────────────────────────────────────────────────────
"""

import math
import json
import heapq
import datetime
import warnings
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

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
    name:    str
    lat:     float
    lon:     float
    country: str
    region:  str = ""

@dataclass
class ThreatZone:
    """
    A static geo-threat zone loaded from a pre-defined table.
    Replaces the news-feed + KNN pipeline with a clean, auditable threat table.
    """
    name:        str
    lat:         float
    lon:         float
    threat_type: str    # war | piracy | terrorism | sanctions | natural
    severity:    float  # 0–10
    radius_km:   float  # zone of influence

@dataclass
class RoutePlan:
    path:              List[str]
    distance_km:       float
    xgb_risk_score:    float       # 0–100 from XGBoost regressor
    risk_class:        str         # LOW / MEDIUM / HIGH / CRITICAL
    estimated_days:    float
    base_risk_max:     float
    threat_severity:   float
    n_chokepoints:     int
    war_flag:          int
    sanctions_flag:    int
    piracy_flag:       int
    weather_risk:      float
    alternative_rank:  int
    warnings:          List[str] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — WAYPOINT & EDGE DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

WAYPOINTS: Dict[str, GeoPoint] = {
    # East Asia
    "SHANGHAI":    GeoPoint("Shanghai Port",        31.23,  121.47, "China",        "East Asia"),
    "HONG_KONG":   GeoPoint("Hong Kong Port",        22.30,  114.17, "Hong Kong",    "East Asia"),
    "BANGKOK":     GeoPoint("Laem Chabang",          13.08,  100.88, "Thailand",     "SE Asia"),
    # SE Asia / Indian Ocean
    "MALACCA":     GeoPoint("Strait of Malacca",      1.50,  103.00, "International","Strait"),
    "SINGAPORE":   GeoPoint("Singapore Port",          1.29,  103.85, "Singapore",    "SE Asia"),
    "COLOMBO":     GeoPoint("Colombo Port",            6.93,   79.85, "Sri Lanka",    "South Asia"),
    # South Asia / Middle East
    "MUMBAI":      GeoPoint("Mumbai Port",            18.96,   72.82, "India",        "South Asia"),
    "KARACHI":     GeoPoint("Karachi Port",           24.85,   67.00, "Pakistan",     "South Asia"),
    "DUBAI":       GeoPoint("Jebel Ali",              24.98,   55.07, "UAE",          "Middle East"),
    "HORMUZ":      GeoPoint("Strait of Hormuz",       26.57,   56.27, "International","Strait"),
    # Red Sea / East Africa
    "ADEN":        GeoPoint("Port of Aden",           12.78,   45.04, "Yemen",        "Middle East"),
    "BABS_MANDEB": GeoPoint("Bab-el-Mandeb",         12.60,   43.40, "International","Strait"),
    "DJIBOUTI":    GeoPoint("Port of Djibouti",       11.59,   43.13, "Djibouti",     "E Africa"),
    "MOMBASA":     GeoPoint("Port of Mombasa",        -4.06,   39.67, "Kenya",        "E Africa"),
    # Suez / Mediterranean
    "PORT_SAID":   GeoPoint("Port Said",              31.26,   32.28, "Egypt",        "N Africa"),
    "SUEZ":        GeoPoint("Suez Canal",             30.58,   32.27, "Egypt",        "Canal"),
    "PIRAEUS":     GeoPoint("Port of Piraeus",        37.94,   23.64, "Greece",       "SE Europe"),
    "BOSPHORUS":   GeoPoint("Bosphorus Strait",       41.13,   29.05, "Turkey",       "Strait"),
    # NW Europe
    "ROTTERDAM":   GeoPoint("Port of Rotterdam",      51.90,    4.48, "Netherlands",  "NW Europe"),
    "HAMBURG":     GeoPoint("Port of Hamburg",        53.55,    9.99, "Germany",      "NW Europe"),
    "ANTWERP":     GeoPoint("Port of Antwerp",        51.23,    4.40, "Belgium",      "NW Europe"),
    # Africa
    "CAPE_TOWN":   GeoPoint("Port of Cape Town",     -33.90,   18.42, "S. Africa",   "S Africa"),
    # Americas
    "LOS_ANGELES": GeoPoint("Port of LA",            33.73, -118.27, "USA",          "W Americas"),
    "NEW_YORK":    GeoPoint("Port of New York",       40.68,  -74.04, "USA",          "E Americas"),
    "HOUSTON":     GeoPoint("Port of Houston",        29.73,  -94.98, "USA",          "Gulf"),
    "SANTOS":      GeoPoint("Port of Santos",        -23.95,  -46.33, "Brazil",       "S Americas"),
}

# (source, target, distance_km, base_risk_0_to_1)
# base_risk encodes lane-level static risk (piracy history, congestion, political)
RAW_EDGES: List[Tuple] = [
    ("SHANGHAI",    "HONG_KONG",     1270,  0.05),
    ("HONG_KONG",   "MALACCA",       2700,  0.07),
    ("HONG_KONG",   "BANGKOK",       1800,  0.06),
    ("MALACCA",     "SINGAPORE",      350,  0.06),
    ("SINGAPORE",   "COLOMBO",       1720,  0.08),
    ("COLOMBO",     "MUMBAI",         930,  0.07),
    ("MUMBAI",      "KARACHI",        460,  0.10),
    ("MUMBAI",      "DUBAI",         1930,  0.08),
    ("DUBAI",       "HORMUZ",         180,  0.15),
    ("HORMUZ",      "ADEN",          1470,  0.20),   # Arabian Sea
    ("ADEN",        "BABS_MANDEB",     70,  0.38),   # Red Sea gateway — HIGH RISK
    ("BABS_MANDEB", "DJIBOUTI",        30,  0.32),
    ("DJIBOUTI",    "PORT_SAID",     1960,  0.30),   # Red Sea — war zone adjacent
    ("PORT_SAID",   "SUEZ",            20,  0.10),
    ("SUEZ",        "PIRAEUS",       2100,  0.09),
    ("PIRAEUS",     "BOSPHORUS",      750,  0.12),
    ("PIRAEUS",     "ROTTERDAM",     3200,  0.06),
    ("PIRAEUS",     "HAMBURG",       3400,  0.06),
    ("PIRAEUS",     "ANTWERP",       3100,  0.07),
    ("ROTTERDAM",   "HAMBURG",        500,  0.04),
    ("ROTTERDAM",   "ANTWERP",        100,  0.03),
    ("HAMBURG",     "ANTWERP",        400,  0.03),
    ("COLOMBO",     "CAPE_TOWN",     6750,  0.09),   # Cape route (bypasses Red Sea)
    ("CAPE_TOWN",   "ROTTERDAM",     9700,  0.08),
    ("CAPE_TOWN",   "HAMBURG",       9900,  0.08),
    ("CAPE_TOWN",   "ANTWERP",       9800,  0.08),
    ("CAPE_TOWN",   "SANTOS",        6800,  0.07),
    ("CAPE_TOWN",   "MOMBASA",       3900,  0.09),
    ("DJIBOUTI",    "MOMBASA",       1100,  0.10),
    ("MOMBASA",     "CAPE_TOWN",     3900,  0.09),
    ("LOS_ANGELES", "SHANGHAI",      9900,  0.06),
    ("LOS_ANGELES", "HONG_KONG",     9650,  0.06),
    ("NEW_YORK",    "ROTTERDAM",     5850,  0.05),
    ("NEW_YORK",    "HAMBURG",       6200,  0.05),
    ("HOUSTON",     "ROTTERDAM",     8400,  0.05),
    ("SANTOS",      "ROTTERDAM",     9600,  0.06),
    ("SHANGHAI",    "LOS_ANGELES",   9900,  0.06),
]

# Chokepoints — nodes that carry extra systemic risk when transited
CHOKEPOINTS = {"MALACCA", "HORMUZ", "BOSPHORUS", "SUEZ", "BABS_MANDEB", "ADEN"}

# ──────────────────────────────────────────────────────────────────────────────
# STATIC THREAT ZONE TABLE  (replaces news feed + KNN)
# Loaded once at startup; severity 0–10, radius_km = zone of influence
# ──────────────────────────────────────────────────────────────────────────────
THREAT_ZONES: List[ThreatZone] = [
    ThreatZone("Red Sea — Houthi conflict",     15.0,  42.5, "war",        9.5, 600),
    ThreatZone("Strait of Hormuz — Iran risk",  26.5,  56.5, "terrorism",  8.8, 350),
    ThreatZone("Gulf of Aden — piracy",         12.5,  46.0, "piracy",     7.5, 400),
    ThreatZone("Gaza / E. Med conflict",        31.5,  34.4, "war",        7.8, 300),
    ThreatZone("Black Sea — Ukraine war",       46.5,  31.0, "war",        7.5, 500),
    ThreatZone("Persian Gulf — sanctions",      27.0,  51.0, "sanctions",  7.0, 400),
    ThreatZone("Gulf of Guinea — piracy",        3.0,   3.0, "piracy",     6.5, 500),
    ThreatZone("South China Sea — tension",     14.0, 114.0, "war",        5.5, 600),
    ThreatZone("Malacca — low congestion",       1.5, 103.0, "natural",    2.0, 200),
    ThreatZone("Suez Canal — draft limits",     30.5,  32.3, "natural",    4.5, 150),
    ThreatZone("Horn of Africa — piracy",        5.0,  48.0, "piracy",     6.0, 450),
    ThreatZone("Taiwan Strait — military",      24.0, 119.0, "war",        6.5, 300),
]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — HAVERSINE DISTANCE
#
#   FORMULA:
#     a = sin²(Δlat/2) + cos(lat₁) · cos(lat₂) · sin²(Δlon/2)
#     d = 2 · R · arcsin(√a)
# ══════════════════════════════════════════════════════════════════════════════

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Great-circle distance in km.

        a = sin²(Δlat/2) + cos(lat₁)·cos(lat₂)·sin²(Δlon/2)
        d = 2R·arcsin(√a)   where R = 6 371 km
    """
    R    = 6_371
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δφ      = math.radians(lat2 - lat1)
    Δλ      = math.radians(lon2 - lon1)

    a = math.sin(Δφ / 2)**2 + math.cos(φ1) * math.cos(φ2) * math.sin(Δλ / 2)**2
    return R * 2 * math.asin(math.sqrt(a))


def point_to_geopoint(lat: float, lon: float) -> GeoPoint:
    return GeoPoint("_tmp", lat, lon, "")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — STATIC THREAT SCORER
#
#   Replaces KNN.  For each graph node or edge midpoint we query
#   ALL threat zones and apply an inverse-distance weighted sum:
#
#   FORMULA:
#     threat_score(p) = Σ [ severity_z · max(0, 1 - d(p,z)/radius_z) ]
#                        z ∈ THREAT_ZONES
#
#   This is a simple radial basis function (RBF) influence model:
#     influence(d) = max(0,  1 - d / R_z)       (linear decay inside radius)
#     contribution = severity_z · influence(d_z)
#
#   Result is clipped to [0, 10].
# ══════════════════════════════════════════════════════════════════════════════

def static_threat_score(lat: float, lon: float) -> float:
    """
    Radial-basis threat score at a geo-point.

        threat(p) = Σ_z  severity_z · max(0,  1 - d(p,z) / radius_z)

    Clipped to [0, 10].
    """
    total = 0.0
    for z in THREAT_ZONES:
        d = haversine(lat, lon, z.lat, z.lon)
        influence = max(0.0, 1.0 - d / z.radius_km)   # linear decay
        total += z.severity * influence
    return min(total, 10.0)


def edge_threat_score(n1: str, n2: str, samples: int = 5) -> float:
    """
    Sample `samples` points along the edge n1→n2, compute static_threat_score
    at each, return the MAXIMUM (worst-case exposure along the edge).

        edge_threat(u,v) = max{ threat(p_t) : t ∈ [0,1], samples evenly spaced }

    where  p_t = lat1 + t·(lat2-lat1),  lon1 + t·(lon2-lon1)   (linear interp.)
    """
    p1, p2 = WAYPOINTS[n1], WAYPOINTS[n2]
    worst  = 0.0
    for i in range(samples + 1):
        t   = i / samples
        lat = p1.lat + t * (p2.lat - p1.lat)
        lon = p1.lon + t * (p2.lon - p1.lon)
        worst = max(worst, static_threat_score(lat, lon))
    return worst


def threats_near_path(path: List[str], radius_km: float = 700) -> List[ThreatZone]:
    """Return all threat zones within radius_km of ANY node on the path."""
    near = []
    for node in path:
        gp = WAYPOINTS[node]
        for z in THREAT_ZONES:
            d = haversine(gp.lat, gp.lon, z.lat, z.lon)
            if d <= radius_km and z not in near:
                near.append(z)
    return near


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — EDGE WEIGHT (DIJKSTRA / YEN'S COST FUNCTION)
#
#   FORMULA:
#     W(u,v) = α · threat_norm(u,v)  +  β · dist_norm(u,v)
#
#     α = 0.70   (weight on threat — safety-first routing)
#     β = 0.30   (weight on distance)
#
#     threat_norm(u,v) = edge_threat_score(u,v) / 10       ∈ [0, 1]
#     dist_norm(u,v)   = dist_km(u,v)           / 15 000   ∈ [0, 1]
#
#   Lower W → more preferred by Dijkstra.
# ══════════════════════════════════════════════════════════════════════════════

ALPHA = 0.70   # threat weight
BETA  = 0.30   # distance weight

def edge_weight(adj: dict, u: str, v: str) -> float:
    """
    Combined edge cost for Dijkstra / Yen's:

        W(u,v) = α · (edge_threat / 10)  +  β · (dist_km / 15 000)

    α = 0.70,  β = 0.30
    """
    data         = adj[u][v]
    threat_norm  = edge_threat_score(u, v) / 10.0
    dist_norm    = data["dist"] / 15_000.0
    return ALPHA * threat_norm + BETA * dist_norm


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — ADJACENCY GRAPH
# ══════════════════════════════════════════════════════════════════════════════

def build_adjacency() -> dict:
    """Build a bidirectional adjacency dict from RAW_EDGES."""
    adj: dict = {}
    for src, tgt, dist, risk in RAW_EDGES:
        for s, t in [(src, tgt), (tgt, src)]:
            adj.setdefault(s, {})
            adj[s][t] = {"dist": dist, "base_risk": risk}
    return adj


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — DIJKSTRA'S ALGORITHM
#
#   FORMULA:
#     d[v] = min( d[v],  d[u] + W(u,v) )    ∀ (u,v) ∈ E
#
#   Uses a binary min-heap (heapq) for O((V+E) log V) performance.
# ══════════════════════════════════════════════════════════════════════════════

def dijkstra(adj: dict, source: str, target: str) -> Tuple[Optional[List[str]], float]:
    """
    Single-source shortest path.

        Relaxation:  d[v] = min( d[v],  d[u] + W(u,v) )

    Returns (path, total_cost) or (None, inf) if unreachable.
    """
    dist = {n: float("inf") for n in adj}
    dist[source] = 0.0
    prev: Dict[str, Optional[str]] = {n: None for n in adj}
    pq   = [(0.0, source)]

    while pq:
        d_u, u = heapq.heappop(pq)
        if d_u > dist[u]:
            continue                        # stale entry
        if u == target:
            break
        for v, _ in adj.get(u, {}).items():
            w   = edge_weight(adj, u, v)
            alt = dist[u] + w               # relaxation
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u
                heapq.heappush(pq, (alt, v))

    # Reconstruct path
    path, cur = [], target
    while cur is not None:
        path.append(cur)
        cur = prev.get(cur)
    path.reverse()

    if not path or path[0] != source:
        return None, float("inf")
    return path, dist[target]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — YEN'S K-SHORTEST LOOPLESS PATHS
#
#   ALGORITHM SUMMARY:
#     A[0]  = shortest path from Dijkstra
#     For k = 1 … K-1:
#       For each spur node i in A[k-1]:
#         1. root_path   = A[k-1][0:i+1]
#         2. Remove edges that create the same root as previous A paths
#         3. Remove root_path nodes (except spur) to force new spur
#         4. spur_path   = Dijkstra(modified graph, spur_node → target)
#         5. candidate   = root_path[:-1] ⊕ spur_path
#         6. Push candidate into min-heap B if not already seen
#       A[k] = pop lowest-cost from B
#
#   PATH COST:
#     cost(path) = Σ W(path[i], path[i+1])   for i = 0 … len-2
# ══════════════════════════════════════════════════════════════════════════════

def path_cost(adj: dict, path: List[str]) -> float:
    """
    Total path cost:  cost(P) = Σ W(P[i], P[i+1])
    """
    return sum(edge_weight(adj, path[i], path[i + 1])
               for i in range(len(path) - 1))


def yens_k_shortest(adj0: dict, source: str, target: str,
                    K: int = 5) -> List[Tuple[List[str], float]]:
    """
    Yen's algorithm for K loopless shortest paths.

    Path cost:  cost(P) = Σ_{i} W(P[i], P[i+1])

    Returns list of (path, cost) sorted ascending by cost.
    """
    import json as _json

    adj = _json.loads(_json.dumps(adj0))          # deep copy

    first, fc = dijkstra(adj, source, target)
    if first is None:
        return []

    A: List[Tuple[List[str], float]] = [(first, fc)]
    B: List[Tuple[float, List[str]]] = []
    seen: set = {tuple(first)}

    for k in range(1, K):
        prev_path = A[k - 1][0]

        for i in range(len(prev_path) - 1):
            spur_node  = prev_path[i]
            root_path  = prev_path[: i + 1]
            root_c     = path_cost(adj, root_path)

            # --- Remove edges that duplicate the root prefix in confirmed paths
            removed_edges: List[Tuple] = []
            for a_path, _ in A:
                if (len(a_path) > i
                        and a_path[: i + 1] == root_path
                        and adj.get(a_path[i], {}).get(a_path[i + 1]) is not None):
                    u, v  = a_path[i], a_path[i + 1]
                    saved = adj[u].pop(v)
                    removed_edges.append((u, v, saved))

            # --- Temporarily remove root nodes (except spur_node)
            removed_nodes: List[Tuple] = []
            for node in root_path[:-1]:
                if node == spur_node:
                    continue
                out_edges = adj.pop(node, {})
                in_edges  = {}
                for x in list(adj.keys()):
                    if node in adj[x]:
                        in_edges[x] = adj[x].pop(node)
                removed_nodes.append((node, out_edges, in_edges))

            # --- Spur Dijkstra on the modified graph
            spur_path, spur_c = dijkstra(adj, spur_node, target)

            # --- Restore removed nodes
            for node, out_edges, in_edges in reversed(removed_nodes):
                adj[node] = out_edges
                for x, data in in_edges.items():
                    adj.setdefault(x, {})[node] = data

            # --- Restore removed edges
            for u, v, data in removed_edges:
                adj.setdefault(u, {})[v] = data

            # --- Record candidate if valid and new
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
# SECTION 9 — XGBOOST RISK SCORER
#
#   Training-label formula (composite risk ground truth):
#
#     y = 30·base_risk  +  6·threat_sev  +  8·n_choke
#         + 15·war_flag  + 12·sanctions_flag  +  4·weather_risk
#         + 5·cargo_val  +  ε,    ε ~ N(0, 3)
#     y = clip(y, 0, 100)
#
#   XGBoost gradient update (additive tree ensemble):
#
#     F_m(x) = F_{m-1}(x)  +  η · h_m(x)
#
#     pseudo-residuals  r_i = y_i − F_{m-1}(x_i)   (for MSE loss)
#     h_m fitted to r on training set each round.
#
#   Feature vector  x ∈ ℝ¹⁴:
#     [dist_norm, hops_norm, base_risk, threat_sev_norm, n_threats_norm,
#      choke_flag, n_choke_norm, cargo_val, mode_sea, seasonal_sin,
#      weather_norm, sanctions_flag, war_flag, piracy_flag]
# ══════════════════════════════════════════════════════════════════════════════

RISK_CLASSES = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

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
            n_estimators   = 300,
            max_depth      = 6,
            learning_rate  = 0.04,     # η in F_m = F_{m-1} + η·h_m
            subsample      = 0.80,
            colsample_bytree = 0.80,
            reg_alpha      = 0.1,      # L1 regularisation
            reg_lambda     = 1.0,      # L2 regularisation
            random_state   = 42,
            verbosity      = 0,
        )
        self.scaler  = StandardScaler()
        self._fitted = False

    # ──────────────────────────────────────────────────────────────────────────
    # SYNTHETIC TRAINING DATA
    # Ground-truth label:
    #   y = 30·base_risk + 6·threat_sev + 8·n_choke
    #       + 15·war + 12·sanctions + 4·weather + 5·cargo  + ε
    # ──────────────────────────────────────────────────────────────────────────
    def _make_training_data(self, n: int = 3000):
        rng = np.random.default_rng(42)
        X, y = [], []

        for _ in range(n):
            dist_norm      = rng.uniform(0.03, 1.0)
            hops_norm      = rng.uniform(0.10, 1.0)
            base_risk      = rng.uniform(0.02, 0.40)
            threat_sev     = rng.uniform(0.0,  10.0)
            n_threats      = rng.integers(0, 9)
            choke_flag     = rng.integers(0, 2)
            n_choke        = rng.integers(0, 5) if choke_flag else 0
            cargo_val      = rng.uniform(0.0,  1.0)
            mode_sea       = 1
            month          = rng.integers(1, 13)
            seasonal_sin   = math.sin(2 * math.pi * month / 12)
            weather        = rng.uniform(0.0,  5.0)
            sanctions_flag = rng.integers(0, 2)
            war_flag       = rng.integers(0, 2)
            piracy_flag    = rng.integers(0, 2)

            # ── Ground-truth label formula ──────────────────────────────────
            # y = 30·base_risk + 6·threat_sev + 8·n_choke
            #     + 15·war + 12·sanctions + 4·weather + 5·cargo + ε
            # ────────────────────────────────────────────────────────────────
            label = (
                30  * base_risk      +
                6   * threat_sev     +
                8   * n_choke        +
                15  * war_flag       +
                12  * sanctions_flag +
                4   * weather        +
                5   * cargo_val      +
                rng.normal(0, 3)     # ε ~ N(0, 3)
            )
            label = float(np.clip(label, 0, 100))

            X.append([
                dist_norm, hops_norm, base_risk,
                threat_sev / 10, n_threats / 8,
                choke_flag, n_choke / 4, cargo_val,
                mode_sea, seasonal_sin,
                weather / 5, sanctions_flag, war_flag, piracy_flag,
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
        print(f"  [XGB] Trained — R² = {r2:.4f}  (n_trees={self.reg.n_estimators})")
        self._fitted = True

    def predict(self, feat: dict) -> Tuple[float, str]:
        """
        Returns (risk_score ∈ [0,100], risk_class).

        risk_class:
          [0,25)  → LOW
          [25,50) → MEDIUM
          [50,75) → HIGH
          [75,100]→ CRITICAL
        """
        if not self._fitted:
            raise RuntimeError("Call .train() first.")

        x = np.array([[feat.get(k, 0.0) for k in self.FEATURE_NAMES]])
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
# SECTION 10 — ROUTE FEATURE EXTRACTION
#
#   Given a path (list of node names) and cargo metadata,
#   compute the 14-dimensional feature vector for XGBoost.
# ══════════════════════════════════════════════════════════════════════════════

def extract_features(path: List[str], adj: dict,
                     cargo_value_norm: float,
                     cargo_type: str = "general") -> dict:
    """
    Build the XGBoost feature vector for a single route.

    Feature definitions:
      dist_norm      = total_km / 15 000
      hops_norm      = len(path) / 10
      base_risk      = max edge base_risk along path          ∈ [0,1]
      threat_sev_norm= max static threat score / 10           ∈ [0,1]
      n_threats_norm = count(threat zones near path) / 12     ∈ [0,1]
      choke_flag     = 1 if any chokepoint on path
      n_choke_norm   = count(chokepoints on path) / 5         ∈ [0,1]
      cargo_val      = caller-supplied                        ∈ [0,1]
      mode_sea       = 1 (all routes are sea)
      seasonal_sin   = sin(2π·month/12)                       ∈ [-1,1]
      weather_norm   = max natural-type threat / 10           ∈ [0,1]
      sanctions_flag = 1 if sanctions zone near path
      war_flag       = 1 if war/terrorism zone near path
      piracy_flag    = 1 if piracy zone near path
    """
    dist_km = sum(
        adj[path[i]][path[i + 1]]["dist"]
        for i in range(len(path) - 1)
    )
    base_risks = [
        adj[path[i]][path[i + 1]]["base_risk"]
        for i in range(len(path) - 1)
    ]
    # Max edge threat score along the route
    edge_threats = [edge_threat_score(path[i], path[i + 1])
                    for i in range(len(path) - 1)]

    near          = threats_near_path(path)
    n_choke       = sum(1 for n in path if n in CHOKEPOINTS)
    war_flag      = int(any(z.threat_type in ("war", "terrorism") for z in near))
    sanctions_flag= int(any(z.threat_type == "sanctions"          for z in near))
    piracy_flag   = int(any(z.threat_type == "piracy"             for z in near))
    weather_max   = max((z.severity for z in near
                         if z.threat_type == "natural"), default=0.0)
    month         = datetime.datetime.now().month

    return {
        "dist_norm":      dist_km / 15_000,
        "hops_norm":      len(path) / 10,
        "base_risk":      max(base_risks) if base_risks else 0.0,
        "threat_sev_norm":max(edge_threats) / 10 if edge_threats else 0.0,
        "n_threats_norm": len(near) / 12,
        "choke_flag":     int(n_choke > 0),
        "n_choke_norm":   n_choke / 5,
        "cargo_val":      cargo_value_norm,
        "mode_sea":       1,
        "seasonal_sin":   math.sin(2 * math.pi * month / 12),
        "weather_norm":   weather_max / 10,
        "sanctions_flag": sanctions_flag,
        "war_flag":       war_flag,
        "piracy_flag":    piracy_flag,
        # keep raw values for RoutePlan fields
        "_dist_km":       dist_km,
        "_n_choke":       n_choke,
        "_threat_sev":    max(edge_threats) if edge_threats else 0.0,
        "_weather":       weather_max,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — WARNING GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def generate_warnings(rp: RoutePlan, all_routes: List["RoutePlan"]) -> List[str]:
    flags = []
    if rp.xgb_risk_score >= 75:
        flags.append("CRITICAL: XGBoost score ≥ 75 — route not recommended.")
    if rp.n_chokepoints > 1:
        chokes = [n for n in rp.path if n in CHOKEPOINTS]
        flags.append(f"MULTI-CHOKEPOINT: passes {len(chokes)} chokepoints "
                     f"({', '.join(chokes)}).")
    if rp.war_flag:
        flags.append("WAR-ZONE ADJACENT: one or more war/terrorism zones near path.")
    if rp.piracy_flag:
        flags.append("PIRACY ZONE: piracy threat active near this route.")
    if rp.sanctions_flag:
        flags.append("SANCTIONS REGION: sanctions zone intersects route corridor.")
    # Check if a safer alternative exists
    safer = [r for r in all_routes
             if r.xgb_risk_score < rp.xgb_risk_score - 10
             and r.alternative_rank != rp.alternative_rank]
    if safer:
        best_alt = min(safer, key=lambda r: r.xgb_risk_score)
        flags.append(f"SAFER OPTION: Route #{best_alt.alternative_rank} scores "
                     f"{best_alt.xgb_risk_score:.1f} vs {rp.xgb_risk_score:.1f}.")
    return flags


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 12 — PIPELINE ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

class CargoRiskPipeline:
    """
    End-to-end pipeline:
      1. Build adjacency graph
      2. Train XGBoost on synthetic data (or load pre-trained model)
      3. Run Dijkstra + Yen's K-shortest on threat-weighted graph
      4. Score each route with XGBoost
      5. Generate per-route warnings
      6. Return sorted RoutePlan list
    """

    def __init__(self, k_paths: int = 5):
        self.k_paths    = k_paths
        self.adj        = build_adjacency()
        self.xgb_scorer = XGBRiskScorer()

    def initialise(self):
        print("\n" + "═" * 62)
        print("  CARGO RISK PIPELINE  v2.0  (XGBoost + Dijkstra + Yen's)")
        print("═" * 62)
        print("\n[1] Training XGBoost risk model...")
        self.xgb_scorer.train()
        print("[2] Graph ready —",
              len(self.adj), "nodes,",
              sum(len(v) for v in self.adj.values()), "directed edges.")
        print()

    def analyse(self, origin: str, destination: str,
                cargo_value_norm: float = 0.5,
                cargo_type: str = "general") -> List[RoutePlan]:
        """
        Returns K RoutePlan objects sorted by XGBoost risk score (lowest first).
        """
        if origin not in self.adj:
            raise ValueError(f"Unknown origin: {origin}")
        if destination not in self.adj:
            raise ValueError(f"Unknown destination: {destination}")
        if origin == destination:
            raise ValueError("Origin and destination must differ.")

        print(f"  Routing  {origin}  →  {destination}")
        print(f"  Cargo    {cargo_type}  |  value index = {cargo_value_norm:.2f}")
        print()

        # Step A — Yen's K-shortest
        print(f"  [Yen's]  Computing {self.k_paths} loopless shortest paths...")
        candidates = yens_k_shortest(self.adj, origin, destination, K=self.k_paths)
        print(f"  [Yen's]  {len(candidates)} paths found.")

        # Step B — XGBoost scoring
        route_plans: List[RoutePlan] = []
        for rank, (path, _) in enumerate(candidates, start=1):
            feat = extract_features(path, self.adj, cargo_value_norm, cargo_type)
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
                war_flag         = feat["war_flag"],
                sanctions_flag   = feat["sanctions_flag"],
                piracy_flag      = feat["piracy_flag"],
                weather_risk     = round(feat["_weather"], 2),
                alternative_rank = rank,
            )
            route_plans.append(rp)

        # Step C — Sort by XGBoost score ascending
        route_plans.sort(key=lambda r: r.xgb_risk_score)
        for i, rp in enumerate(route_plans, start=1):
            rp.alternative_rank = i

        # Step D — Generate warnings
        for rp in route_plans:
            rp.warnings = generate_warnings(rp, route_plans)

        return route_plans

    # ──────────────────────────────────────────────────────────────────────────
    # REPORTING
    # ──────────────────────────────────────────────────────────────────────────
    def print_report(self, routes: List[RoutePlan],
                     origin: str, destination: str):
        BAR = 24
        print()
        print("═" * 62)
        print(f"  ROUTE RISK REPORT  :  {origin}  →  {destination}")
        print(f"  Generated          :  "
              f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M UTC')}")
        print("═" * 62)

        for rp in routes:
            filled = int(rp.xgb_risk_score / 100 * BAR)
            bar    = "█" * filled + "░" * (BAR - filled)
            print(f"\n  ── Route #{rp.alternative_rank}  [{rp.risk_class}]")
            print(f"     Path     : {' → '.join(rp.path)}")
            print(f"     Distance : {rp.distance_km:,.0f} km"
                  f"   |   Transit : {rp.estimated_days:.1f} days")
            print(f"     XGB risk : {rp.xgb_risk_score:5.1f}/100  |{bar}|")
            print(f"     Threat   : sev={rp.threat_severity:.1f}  "
                  f"choke={rp.n_chokepoints}  "
                  f"war={rp.war_flag}  sanctions={rp.sanctions_flag}  "
                  f"piracy={rp.piracy_flag}")
            if rp.warnings:
                for w in rp.warnings:
                    print(f"     ⚠  {w}")

        best = routes[0]
        print()
        print("═" * 62)
        print(f"  RECOMMENDATION  :  Route #{best.alternative_rank}")
        print(f"  Path            :  {' → '.join(best.path)}")
        print(f"  Risk class      :  {best.risk_class}")
        print(f"  Client score    :  {best.xgb_risk_score:.1f} / 100")
        print("═" * 62)

        # Feature importance
        print("\n  XGBoost Feature Importance (F-score, top 10):")
        for feat, imp in list(self.xgb_scorer.feature_importance().items())[:10]:
            bar = "█" * int(imp * 50)
            print(f"    {feat:<22}  {imp:.4f}  {bar}")

    def to_json(self, routes: List[RoutePlan],
                origin: str, destination: str) -> str:
        return json.dumps({
            "origin":       origin,
            "destination":  destination,
            "generated_at": datetime.datetime.now().isoformat(),
            "routes": [
                {
                    "rank":           r.alternative_rank,
                    "path":           r.path,
                    "distance_km":    r.distance_km,
                    "estimated_days": r.estimated_days,
                    "xgb_risk_score": r.xgb_risk_score,
                    "risk_class":     r.risk_class,
                    "n_chokepoints":  r.n_chokepoints,
                    "war_flag":       r.war_flag,
                    "sanctions_flag": r.sanctions_flag,
                    "piracy_flag":    r.piracy_flag,
                    "warnings":       r.warnings,
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

    # ── Example 1: Shanghai → Rotterdam (Red Sea exposure)
    routes = pipeline.analyse("SHANGHAI", "ROTTERDAM",
                               cargo_value_norm=0.8, cargo_type="electronics")
    pipeline.print_report(routes, "SHANGHAI", "ROTTERDAM")

    with open("/mnt/user-data/outputs/routes_SH_RTM.json", "w") as f:
        f.write(pipeline.to_json(routes, "SHANGHAI", "ROTTERDAM"))

    # ── Example 2: Dubai → Rotterdam (Hormuz + Red Sea double exposure)
    routes2 = pipeline.analyse("DUBAI", "ROTTERDAM",
                                cargo_value_norm=0.6, cargo_type="petroleum")
    pipeline.print_report(routes2, "DUBAI", "ROTTERDAM")

    with open("/mnt/user-data/outputs/routes_DXB_RTM.json", "w") as f:
        f.write(pipeline.to_json(routes2, "DUBAI", "ROTTERDAM"))