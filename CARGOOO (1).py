"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        CARGO RISK INTELLIGENCE PIPELINE                                     ║
║  XGBoost Risk Scoring · Dijkstra · Yen's K-Shortest · KNN Geo-Threat        ║
║  Google News API data feed · Verification Layer · Client Risk Score         ║
╚══════════════════════════════════════════════════════════════════════════════╝

ARCHITECTURE:
  [Google News API] ──► [NLP Threat Extractor] ──► [KNN Geo-Threat Mapper]
                                                          │
  [Route Graph]  ──► [Dijkstra + Yen's K-Paths] ──► [XGBoost Risk Scorer]
                                                          │
                                              [Verification Layer]
                                                          │
                                              [Client Risk Report]
"""

import math
import json
import heapq
import hashlib
import datetime
import warnings
import itertools
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict

import numpy as np
import networkx as nx
import requests

# ML libraries
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import xgboost as xgb

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# 0.  CONSTANTS & CONFIG
# ──────────────────────────────────────────────────────────────────────────────

GOOGLE_NEWS_API_KEY = "YOUR_GOOGLE_NEWS_API_KEY"   # ← replace in production
GOOGLE_NEWS_ENDPOINT = "https://newsapi.org/v2/everything"

# Threat keywords for NLP classification
THREAT_KEYWORDS = {
    "war":          ["war", "military strike", "bombing", "airstrike", "combat", "invasion",
                     "missile", "artillery", "troops", "offensive", "frontline", "shelling"],
    "piracy":       ["piracy", "pirate", "hijack", "hijacking", "armed robbery at sea",
                     "vessel seized", "crew kidnapped"],
    "sanctions":    ["sanctions", "embargo", "blockade", "trade ban", "export control",
                     "port closure", "restricted trade"],
    "natural":      ["hurricane", "typhoon", "cyclone", "earthquake", "tsunami", "flood",
                     "storm", "port damage", "infrastructure damage"],
    "terrorism":    ["terrorism", "terrorist attack", "explosion", "bomb", "ied",
                     "insurgency", "extremist", "militant"],
    "civil_unrest": ["protest", "riot", "coup", "civil war", "unrest", "strike",
                     "port workers strike", "blockade road"],
    "pandemic":     ["quarantine", "lockdown", "port closed", "border closed", "health emergency"],
}

THREAT_SEVERITY = {
    "war":          10,
    "piracy":       8,
    "terrorism":    9,
    "sanctions":    7,
    "civil_unrest": 5,
    "natural":      6,
    "pandemic":     4,
}

# ──────────────────────────────────────────────────────────────────────────────
# 1.  DATA CLASSES
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GeoPoint:
    name: str
    lat: float
    lon: float
    country: str
    region: str = ""

@dataclass
class ThreatEvent:
    title: str
    description: str
    lat: float
    lon: float
    threat_type: str
    severity: float          # 0–10
    published_at: str
    source: str
    verified: bool = False
    confidence: float = 0.0  # 0–1, set by verification layer

@dataclass
class RouteEdge:
    source: str
    target: str
    distance_km: float
    base_risk: float         # 0–1
    transport_mode: str      # sea / land / air
    waypoints: List[Tuple[float, float]] = field(default_factory=list)

@dataclass
class RoutePlan:
    path: List[str]
    total_distance_km: float
    total_risk_score: float       # 0–100 (XGBoost output)
    threat_events_near_path: List[ThreatEvent]
    knn_geo_threat_score: float   # KNN-derived proximity threat
    xgb_risk_score: float         # final XGB risk score
    client_risk_class: str        # LOW / MEDIUM / HIGH / CRITICAL
    estimated_days: float
    alternative_rank: int          # 1 = best (Yen's ordering)
    verification_flags: List[str] = field(default_factory=list)

# ──────────────────────────────────────────────────────────────────────────────
# 2.  WORLD ROUTE GRAPH (sample — extend for full global network)
# ──────────────────────────────────────────────────────────────────────────────

WAYPOINTS: Dict[str, GeoPoint] = {
    # Asia
    "SHANGHAI":     GeoPoint("Shanghai Port",         31.23,  121.47, "China",        "East Asia"),
    "SINGAPORE":    GeoPoint("Singapore Port",          1.29,  103.85, "Singapore",    "SE Asia"),
    "MUMBAI":       GeoPoint("Mumbai Port",            18.96,   72.82, "India",        "South Asia"),
    "DUBAI":        GeoPoint("Jebel Ali Port",         24.98,   55.07, "UAE",          "Middle East"),
    "KARACHI":      GeoPoint("Karachi Port",           24.85,   67.00, "Pakistan",     "South Asia"),
    "COLOMBO":      GeoPoint("Colombo Port",            6.93,   79.85, "Sri Lanka",    "South Asia"),
    "BANGKOK":      GeoPoint("Laem Chabang Port",      13.08,  100.88, "Thailand",     "SE Asia"),
    "HONG_KONG":    GeoPoint("Hong Kong Port",         22.30,  114.17, "Hong Kong",    "East Asia"),
    # Europe
    "ROTTERDAM":    GeoPoint("Port of Rotterdam",      51.90,    4.48, "Netherlands",  "NW Europe"),
    "HAMBURG":      GeoPoint("Port of Hamburg",        53.55,    9.99, "Germany",      "NW Europe"),
    "ANTWERP":      GeoPoint("Port of Antwerp",        51.23,    4.40, "Belgium",      "NW Europe"),
    "PIRAEUS":      GeoPoint("Port of Piraeus",        37.94,   23.64, "Greece",       "SE Europe"),
    # Middle East / Africa
    "PORT_SAID":    GeoPoint("Port Said (Suez)",       31.26,   32.28, "Egypt",        "N Africa"),
    "ADEN":         GeoPoint("Port of Aden",           12.78,   45.04, "Yemen",        "Middle East"),
    "DJIBOUTI":     GeoPoint("Port of Djibouti",       11.59,   43.13, "Djibouti",     "E Africa"),
    "MOMBASA":      GeoPoint("Port of Mombasa",        -4.06,   39.67, "Kenya",        "E Africa"),
    "CAPE_TOWN":    GeoPoint("Port of Cape Town",     -33.90,   18.42, "S. Africa",    "S Africa"),
    # Americas
    "LOS_ANGELES":  GeoPoint("Port of Los Angeles",   33.73, -118.27, "USA",          "W Americas"),
    "NEW_YORK":     GeoPoint("Port of New York",       40.68,  -74.04, "USA",          "E Americas"),
    "HOUSTON":      GeoPoint("Port of Houston",        29.73,  -94.98, "USA",          "E Americas"),
    "SANTOS":       GeoPoint("Port of Santos",        -23.95,  -46.33, "Brazil",       "S Americas"),
    # Chokepoints / Straits
    "MALACCA":      GeoPoint("Strait of Malacca",       1.50,  103.00, "International","Strait"),
    "HORMUZ":       GeoPoint("Strait of Hormuz",       26.57,   56.27, "International","Strait"),
    "BOSPHORUS":    GeoPoint("Bosphorus Strait",       41.13,   29.05, "Turkey",       "Strait"),
    "SUEZ":         GeoPoint("Suez Canal",             30.58,   32.27, "Egypt",        "Canal"),
    "BABS_MANDEB":  GeoPoint("Bab-el-Mandeb",         12.60,   43.40, "International","Strait"),
}

# Edges: (source, target, distance_km, base_risk, mode)
RAW_EDGES = [
    ("SHANGHAI",    "HONG_KONG",    1270,  0.05, "sea"),
    ("HONG_KONG",   "MALACCA",      2700,  0.07, "sea"),
    ("MALACCA",     "SINGAPORE",     350,  0.06, "sea"),
    ("SINGAPORE",   "COLOMBO",      1720,  0.08, "sea"),
    ("COLOMBO",     "MUMBAI",        930,  0.07, "sea"),
    ("MUMBAI",      "KARACHI",       460,  0.10, "sea"),
    ("MUMBAI",      "DUBAI",        1930,  0.08, "sea"),
    ("DUBAI",       "HORMUZ",        180,  0.15, "sea"),
    ("HORMUZ",      "ADEN",         1470,  0.20, "sea"),  # Gulf of Oman / Arabian Sea
    ("ADEN",        "BABS_MANDEB",    70,  0.35, "sea"),  # Red Sea approach
    ("BABS_MANDEB", "DJIBOUTI",       30,  0.30, "sea"),
    ("DJIBOUTI",    "PORT_SAID",    1960,  0.28, "sea"),  # Red Sea
    ("PORT_SAID",   "SUEZ",           20,  0.10, "sea"),
    ("SUEZ",        "PIRAEUS",      2100,  0.09, "sea"),  # Med
    ("PIRAEUS",     "BOSPHORUS",     750,  0.12, "sea"),
    ("PIRAEUS",     "ROTTERDAM",    3200,  0.06, "sea"),
    ("ROTTERDAM",   "HAMBURG",       500,  0.04, "sea"),
    ("ROTTERDAM",   "ANTWERP",       100,  0.03, "sea"),
    ("COLOMBO",     "CAPE_TOWN",    6750,  0.09, "sea"),  # Cape route
    ("CAPE_TOWN",   "SANTOS",       6800,  0.07, "sea"),
    ("CAPE_TOWN",   "ROTTERDAM",    9700,  0.08, "sea"),  # Cape of Good Hope bypass
    ("SINGAPORE",   "BANGKOK",      1450,  0.07, "sea"),
    ("LOS_ANGELES", "SHANGHAI",    10400,  0.06, "sea"),
    ("LOS_ANGELES", "HONG_KONG",    9650,  0.06, "sea"),
    ("NEW_YORK",    "ROTTERDAM",    5850,  0.05, "sea"),
    ("NEW_YORK",    "HAMBURG",      6200,  0.05, "sea"),
    ("HOUSTON",     "ROTTERDAM",    8400,  0.05, "sea"),
    ("SANTOS",      "ROTTERDAM",    9600,  0.06, "sea"),
    ("DJIBOUTI",    "MOMBASA",      1100,  0.10, "sea"),
    ("MOMBASA",     "CAPE_TOWN",    3900,  0.09, "sea"),
    ("SHANGHAI",    "LOS_ANGELES",  9900,  0.06, "sea"),  # Trans-Pacific
]

# ──────────────────────────────────────────────────────────────────────────────
# 3.  GRAPH BUILDER
# ──────────────────────────────────────────────────────────────────────────────

def haversine(p1: GeoPoint, p2: GeoPoint) -> float:
    """Great-circle distance between two geo-points in km."""
    R = 6371
    lat1, lon1 = math.radians(p1.lat), math.radians(p1.lon)
    lat2, lon2 = math.radians(p2.lat), math.radians(p2.lon)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def build_graph() -> nx.DiGraph:
    G = nx.DiGraph()
    for name, gp in WAYPOINTS.items():
        G.add_node(name, geopoint=gp)
    for src, tgt, dist, risk, mode in RAW_EDGES:
        G.add_edge(src, tgt, distance=dist, base_risk=risk, mode=mode)
        G.add_edge(tgt, src, distance=dist, base_risk=risk, mode=mode)
    return G

# ──────────────────────────────────────────────────────────────────────────────
# 4.  GOOGLE NEWS API — THREAT INGESTION
# ──────────────────────────────────────────────────────────────────────────────

# Geo-lookup table for known regions in news
REGION_GEO_MAP = {
    "red sea":       (20.0,  38.0), "gulf of aden":  (12.0,  46.0),
    "strait of hormuz": (26.5, 56.5), "persian gulf": (27.0,  51.0),
    "south china sea": (14.0, 114.0), "taiwan strait": (24.0, 119.0),
    "ukraine":       (49.0,  32.0), "russia":        (55.0,  37.0),
    "israel":        (31.5,  35.0), "gaza":          (31.5,  34.4),
    "yemen":         (15.5,  48.5), "iran":          (32.0,  53.5),
    "somalia":       ( 5.0,  46.0), "gulf of guinea": (3.0,   3.0),
    "malacca":       ( 1.5, 103.0), "suez":          (30.0,  32.0),
    "pakistan":      (30.0,  70.0), "myanmar":       (19.0,  96.0),
    "taiwan":        (23.5, 121.0), "north korea":   (40.0, 127.0),
    "venezuela":     ( 8.0, -66.0), "nigeria":       ( 9.0,   8.0),
}


def classify_threat(text: str) -> Tuple[str, float]:
    """Classify news text into a threat type and return severity."""
    text_lower = text.lower()
    scores = {}
    for threat_type, keywords in THREAT_KEYWORDS.items():
        hit_count = sum(1 for kw in keywords if kw in text_lower)
        if hit_count:
            scores[threat_type] = hit_count * THREAT_SEVERITY[threat_type]
    if not scores:
        return "none", 0.0
    best = max(scores, key=scores.get)
    # Normalise to 0–10
    raw = scores[best]
    severity = min(10.0, raw / 2.0)
    return best, round(severity, 2)


def extract_geo(text: str) -> Optional[Tuple[float, float]]:
    """Very lightweight geo-extraction from text using region map."""
    text_lower = text.lower()
    for region, coords in REGION_GEO_MAP.items():
        if region in text_lower:
            return coords
    return None


def fetch_news_threats (query: str, max_articles: int = 90) -> List[ThreatEvent]:
    """
    Fetch news from Google News API (NewsAPI.org) and convert to ThreatEvents.
    Falls back to a synthetic dataset if API key not set.
    """
    if GOOGLE_NEWS_API_KEY == "YOUR_GOOGLE_NEWS_API_KEY":
        print("  [NEWS] API key not set — using synthetic threat dataset for demo.")
        return _synthetic_threats()

    try:
        params = {
            "q":        query,
            "language": "en",
            "sortBy":   "publishedAt",
            "pageSize": max_articles,
            "apiKey":   GOOGLE_NEWS_API_KEY,
        }
        resp = requests.get(GOOGLE_NEWS_ENDPOINT, params=params, timeout=10)
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
    except Exception as e:
        print(f"  [NEWS] API error: {e} — falling back to synthetic dataset.")
        return _synthetic_threats()

    threats = []
    for art in articles:
        combined = f"{art.get('title','')} {art.get('description','')}"
        threat_type, severity = classify_threat(combined)
        if threat_type == "none" or severity < 1:
            continue
        geo = extract_geo(combined)
        if not geo:
            continue
        threats.append(ThreatEvent(
            title=art.get("title", "")[:100],
            description=art.get("description", "")[:200],
            lat=geo[0], lon=geo[1],
            threat_type=threat_type,
            severity=severity,
            published_at=art.get("publishedAt", ""),
            source=art.get("source", {}).get("name", "Unknown"),
        ))
    print(f"  [NEWS] Fetched {len(threats)} threat events from NewsAPI.")
    return threats


def _synthetic_threats() -> List[ThreatEvent]:
    """Realistic synthetic threat dataset for demonstration."""
    return [
        ThreatEvent("Houthi attacks escalate in Red Sea shipping lanes",
                    "Multiple container ships targeted near Bab-el-Mandeb strait.",
                    12.6, 43.4, "war", 9.5, "2025-04-01T08:00:00Z", "Reuters"),
        ThreatEvent("Drone strikes hit oil tanker near Strait of Hormuz",
                    "Armed drones reported near Iranian territorial waters.",
                    26.5, 56.5, "terrorism", 8.8, "2025-04-01T10:00:00Z", "BBC"),
        ThreatEvent("Somali pirates seize fishing vessel off coast of Djibouti",
                    "Armed pirates boarded vessel 80nm east of Djibouti.",
                    11.8, 44.5, "piracy", 7.5, "2025-03-30T14:00:00Z", "Lloyd's List"),
        ThreatEvent("Military conflict intensifies in Gaza – port operations disrupted",
                    "Ash Dod port reports operational delays due to security alerts.",
                    31.5, 34.4, "war", 8.0, "2025-04-01T06:00:00Z", "Al Jazeera"),
        ThreatEvent("Typhoon Krathon approaching South China Sea shipping lanes",
                    "Category 4 typhoon forecast to hit Taiwan Strait region.",
                    22.0, 120.0, "natural", 6.5, "2025-04-02T09:00:00Z", "WMO"),
        ThreatEvent("Iran sanctions tightened — Hormuz transit restrictions possible",
                    "US Treasury expands sanctions targeting Iranian shipping entities.",
                    27.0, 56.0, "sanctions", 7.0, "2025-03-28T12:00:00Z", "Bloomberg"),
        ThreatEvent("Piracy incidents rise in Gulf of Guinea",
                    "Six incidents reported in Q1 2025 off Nigerian coast.",
                    3.0, 3.0, "piracy", 6.5, "2025-03-25T11:00:00Z", "IMB"),
        ThreatEvent("Ukraine Black Sea shipping corridor under threat",
                    "Russian naval assets reported near Odessa export corridor.",
                    46.5, 31.0, "war", 7.5, "2025-04-01T07:00:00Z", "Reuters"),
        ThreatEvent("Suez Canal capacity reduced due to low water levels",
                    "Draft restrictions now in place; 20% capacity reduction.",
                    30.5, 32.3, "natural", 4.5, "2025-03-20T10:00:00Z", "Splash247"),
        ThreatEvent("Malacca Strait — no incidents reported, traffic nominal",
                    "Normal transit operations reported by MPA Singapore.",
                    1.5, 103.0, "none", 0.5, "2025-04-01T00:00:00Z", "MPA"),
    ]

# ──────────────────────────────────────────────────────────────────────────────
# 5.  KNN GEO-THREAT MAPPER
# ──────────────────────────────────────────────────────────────────────────────

class KNNGeoThreatMapper:
    """
    Uses K-Nearest Neighbours to estimate the threat intensity at any
    geo-coordinate based on known ThreatEvents.
    """

    def __init__(self, k: int = 5, radius_km: float = 500):
        self.k = k
        self.radius_km = radius_km
        self.regressor = Pipeline([
            ("scaler", StandardScaler()),
            ("knn",    KNeighborsRegressor(n_neighbors=k, weights="distance",
                                            metric="euclidean")),
        ])
        self._fitted = False

    def fit(self, threats: List[ThreatEvent]):
        if not threats:
            return
        X = np.array([[t.lat, t.lon] for t in threats])
        y = np.array([t.severity for t in threats])
        self.regressor.fit(X, y)
        self._fitted = True
        print(f"  [KNN] Fitted on {len(threats)} threat events (k={self.k}).")

    def predict_threat_score(self, lat: float, lon: float) -> float:
        """Return estimated threat intensity (0–10) at a given location."""
        if not self._fitted:
            return 0.0
        score = self.regressor.predict([[lat, lon]])[0]
        return float(np.clip(score, 0, 10))

    def edge_threat_score(self, p1: GeoPoint, p2: GeoPoint, samples: int = 5) -> float:
        """
        Estimate threat along an edge by sampling points between p1 and p2.
        Returns the maximum KNN threat score along that edge segment.
        """
        scores = []
        for i in range(samples + 1):
            t = i / samples
            lat = p1.lat + t * (p2.lat - p1.lat)
            lon = p1.lon + t * (p2.lon - p1.lon)
            scores.append(self.predict_threat_score(lat, lon))
        return max(scores)

# ──────────────────────────────────────────────────────────────────────────────
# 6.  XGBOOST RISK SCORER
# ──────────────────────────────────────────────────────────────────────────────

class XGBRouteRiskScorer:
    """
    XGBoost model that takes route-level features and outputs a risk score (0–100)
    and a risk class (LOW / MEDIUM / HIGH / CRITICAL).
    """

    RISK_CLASSES = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

    def __init__(self):
        self.xgb_regressor = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        )
        self.xgb_classifier = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            num_class=4,
            objective="multi:softprob",
            random_state=42,
            verbosity=0,
        )
        self.scaler = StandardScaler()
        self._fitted = False

    def _generate_training_data(self, n: int = 2000):
        """
        Generate synthetic but realistic training data for the XGB models.
        In production this would be replaced with historical route risk data.
        """
        np.random.seed(42)
        X, y_reg, y_cls = [], [], []

        for _ in range(n):
            # Features
            route_distance        = np.random.uniform(500, 15000)
            num_hops              = np.random.randint(2, 8)
            base_edge_risk        = np.random.uniform(0.02, 0.40)
            knn_threat_max        = np.random.uniform(0, 10)
            knn_threat_mean       = knn_threat_max * np.random.uniform(0.3, 0.9)
            num_threats_near      = np.random.randint(0, 8)
            max_threat_severity   = np.random.uniform(0, 10)
            passes_chokepoint     = np.random.randint(0, 2)
            num_chokepoints       = np.random.randint(0, 4) if passes_chokepoint else 0
            cargo_value_norm      = np.random.uniform(0, 1)    # 0=low value, 1=high value
            transport_mode_sea    = np.random.randint(0, 2)
            time_of_year          = np.random.uniform(1, 12)   # month
            weather_risk          = np.random.uniform(0, 5)
            sanctions_region      = np.random.randint(0, 2)
            war_zone_adjacent     = np.random.randint(0, 2)

            feats = [
                route_distance / 15000,    # normalised
                num_hops / 8,
                base_edge_risk,
                knn_threat_max / 10,
                knn_threat_mean / 10,
                num_threats_near / 8,
                max_threat_severity / 10,
                passes_chokepoint,
                num_chokepoints / 4,
                cargo_value_norm,
                transport_mode_sea,
                math.sin(2 * math.pi * time_of_year / 12),   # seasonal
                weather_risk / 5,
                sanctions_region,
                war_zone_adjacent,
            ]

            # Risk score formula (ground truth for training)
            risk = (
                base_edge_risk * 30 +
                knn_threat_max * 4.5 +
                max_threat_severity * 3.0 +
                num_threats_near * 2.0 +
                passes_chokepoint * 8 +
                num_chokepoints * 5 +
                sanctions_region * 12 +
                war_zone_adjacent * 15 +
                weather_risk * 2 +
                np.random.normal(0, 3)    # noise
            )
            risk = float(np.clip(risk, 0, 100))

            if risk < 25:   cls = 0  # LOW
            elif risk < 50: cls = 1  # MEDIUM
            elif risk < 75: cls = 2  # HIGH
            else:           cls = 3  # CRITICAL

            X.append(feats)
            y_reg.append(risk)
            y_cls.append(cls)

        return np.array(X), np.array(y_reg), np.array(y_cls)

    def train(self):
        print("  [XGB] Generating training data and fitting models...")
        X, y_reg, y_cls = self._generate_training_data(2000)
        X_scaled = self.scaler.fit_transform(X)

        X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te = train_test_split(
            X_scaled, y_reg, y_cls, test_size=0.2, random_state=42)

        self.xgb_regressor.fit(X_tr, yr_tr,
                                eval_set=[(X_te, yr_te)],
                                verbose=False)
        self.xgb_classifier.fit(X_tr, yc_tr,
                                 eval_set=[(X_te, yc_te)],
                                 verbose=False)
        self._fitted = True

        r2 = self.xgb_regressor.score(X_te, yr_te)
        print(f"  [XGB] Regressor R² = {r2:.4f}")
        print(f"  [XGB] Classifier trained ({len(self.RISK_CLASSES)} classes).")

    def predict(self, features: Dict[str, float]) -> Tuple[float, str]:
        """
        features must match the training feature order below.
        Returns (risk_score_0_100, risk_class_label).
        """
        if not self._fitted:
            raise RuntimeError("Model not trained. Call .train() first.")

        feat_vec = np.array([[
            features.get("route_distance_norm", 0),
            features.get("num_hops_norm", 0),
            features.get("base_edge_risk", 0),
            features.get("knn_threat_max_norm", 0),
            features.get("knn_threat_mean_norm", 0),
            features.get("num_threats_near_norm", 0),
            features.get("max_threat_severity_norm", 0),
            features.get("passes_chokepoint", 0),
            features.get("num_chokepoints_norm", 0),
            features.get("cargo_value_norm", 0),
            features.get("transport_mode_sea", 1),
            features.get("seasonal_signal", 0),
            features.get("weather_risk_norm", 0),
            features.get("sanctions_region", 0),
            features.get("war_zone_adjacent", 0),
        ]])

        feat_scaled = self.scaler.transform(feat_vec)
        risk_score  = float(np.clip(self.xgb_regressor.predict(feat_scaled)[0], 0, 100))
        cls_idx     = int(self.xgb_classifier.predict(feat_scaled)[0])
        risk_class  = self.RISK_CLASSES[cls_idx]

        return round(risk_score, 2), risk_class

    def get_feature_importance(self) -> Dict[str, float]:
        feature_names = [
            "route_distance", "num_hops", "base_edge_risk",
            "knn_threat_max", "knn_threat_mean", "num_threats_near",
            "max_threat_severity", "passes_chokepoint", "num_chokepoints",
            "cargo_value", "transport_mode", "seasonality",
            "weather_risk", "sanctions_region", "war_zone_adjacent",
        ]
        importances = self.xgb_regressor.feature_importances_
        return dict(sorted(zip(feature_names, importances),
                           key=lambda x: x[1], reverse=True))

# ──────────────────────────────────────────────────────────────────────────────
# 7.  DIJKSTRA — WEIGHTED SHORTEST PATH
# ──────────────────────────────────────────────────────────────────────────────

CHOKEPOINTS = {"MALACCA", "HORMUZ", "BOSPHORUS", "SUEZ", "BABS_MANDEB", "ADEN"}

def compute_edge_weight(G: nx.DiGraph, src: str, tgt: str,
                        knn_mapper: KNNGeoThreatMapper,
                        threat_weight: float = 0.7,
                        distance_weight: float = 0.90) -> float:
    """
    Combined edge weight for Dijkstra: blends KNN threat proximity score
    with physical distance. Lower = better.
    """
    data = G[src][tgt]
    p1 = G.nodes[src]["geopoint"]
    p2 = G.nodes[tgt]["geopoint"]

    knn_score   = knn_mapper.edge_threat_score(p1, p2) / 100.0   # 0–1
    base_risk   = data["base_risk"]                              # 0–1
    dist_norm   = min(data["distance"] / 15000, 100.0)            # 0–1

    threat_component   = (knn_score * 0.6 + base_risk * 0.4) * threat_weight
    distance_component = dist_norm * distance_weight

    return threat_component + distance_component


def dijkstra_best_route(G: nx.DiGraph, source: str, target: str,
                        knn_mapper: KNNGeoThreatMapper) -> Tuple[List[str], float]:
    """Standard Dijkstra on the threat-weighted graph."""
    dist = {n: float("inf") for n in G.nodes}
    prev = {n: None for n in G.nodes}
    dist[source] = 0
    pq = [(0.0, source)]

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v in G.successors(u):
            w = compute_edge_weight(G, u, v, knn_mapper)
            alt = dist[u] + w
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u
                heapq.heappush(pq, (alt, v))

    # Reconstruct path
    path, cur = [], target
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()

    if path[0] != source:
        return [], float("inf")    # no path found
    return path, dist[target]

# ──────────────────────────────────────────────────────────────────────────────
# 8.  YEN'S K-SHORTEST PATHS ALGORITHM
# ──────────────────────────────────────────────────────────────────────────────

def yens_k_shortest(G: nx.DiGraph, source: str, target: str,
                    knn_mapper: KNNGeoThreatMapper, K: int = 5) -> List[Tuple[List[str], float]]:
    """
    Yen's algorithm for K-shortest loopless paths in a directed weighted graph.
    Returns list of (path, cost) tuples, sorted by cost ascending.
    """
    def path_cost(path: List[str]) -> float:
        return sum(compute_edge_weight(G, path[i], path[i+1], knn_mapper)
                   for i in range(len(path) - 1))

    # First shortest path via Dijkstra
    first_path, first_cost = dijkstra_best_route(G, source, target, knn_mapper)
    if not first_path:
        return []

    A = [(first_path, first_cost)]   # confirmed k-shortest paths
    B = []                           # candidate paths (heap)
    seen = {tuple(first_path)}

    for k in range(1, K):
        prev_path = A[k - 1][0]

        for i in range(len(prev_path) - 1):
            spur_node     = prev_path[i]
            root_path     = prev_path[:i + 1]
            root_cost     = path_cost(root_path)

            # Temporarily remove edges that are part of previous k-shortest paths
            removed_edges = []
            for a_path, _ in A:
                if len(a_path) > i and a_path[:i+1] == root_path:
                    u, v = a_path[i], a_path[i+1]
                    if G.has_edge(u, v):
                        removed_edges.append((u, v, G[u][v].copy()))
                        G.remove_edge(u, v)

            # Remove root_path nodes (except spur_node) from graph temporarily
            removed_nodes = []
            for node in root_path[:-1]:
                if node != spur_node:
                    nbrs_out = list(G.successors(node))
                    nbrs_in  = list(G.predecessors(node))
                    removed_nodes.append((node, nbrs_out, nbrs_in,
                                          {n: G[node][n].copy() for n in nbrs_out},
                                          {n: G[n][node].copy() for n in nbrs_in}))
                    G.remove_node(node)

            # Find spur path
            spur_path, spur_cost = dijkstra_best_route(G, spur_node, target, knn_mapper)

            # Restore nodes first (all of them)
            for node, nbrs_out, nbrs_in, out_data, in_data in removed_nodes:
                G.add_node(node, geopoint=WAYPOINTS[node] if node in WAYPOINTS else GeoPoint(node, 0, 0, ""))
            # Then restore edges (all nodes exist now)
            for node, nbrs_out, nbrs_in, out_data, in_data in removed_nodes:
                for n in nbrs_out:
                    if G.has_node(n):
                        G.add_edge(node, n, **out_data[n])
                for n in nbrs_in:
                    if G.has_node(n):
                        G.add_edge(n, node, **in_data[n])
            for u, v, data in removed_edges:
                G.add_edge(u, v, **data)

            if spur_path and spur_path[0] == spur_node:
                full_path = root_path[:-1] + spur_path
                total_cost = root_cost + spur_cost
                t = tuple(full_path)
                if t not in seen:
                    seen.add(t)
                    heapq.heappush(B, (total_cost, full_path))

        if not B:
            break

        cost, path = heapq.heappop(B)
        A.append((path, cost))

    return A

# ──────────────────────────────────────────────────────────────────────────────
# 9.  VERIFICATION LAYER
# ──────────────────────────────────────────────────────────────────────────────

class VerificationLayer:
    """
    Multi-signal verification layer that cross-checks threat events
    and route risk scores for confidence and consistency.
    """

    # Trusted source multipliers
    SOURCE_TRUST = {
        "Reuters":    1.0, "BBC":        1.0, "Bloomberg":   1.0,
        "AP":         1.0, "AFP":         0.9, "Al Jazeera":  0.85,
        "Lloyd's List":1.0, "IMB":        1.0, "Splash247":  0.85,
        "WMO":        1.0, "MPA":         1.0,
    }

    def verify_threat_events(self, threats: List[ThreatEvent]) -> List[ThreatEvent]:
        """Score each threat event for confidence and mark as verified."""
        verified = []
        for t in threats:
            conf = 0.5   # base confidence

            # Source trust
            for source, trust in self.SOURCE_TRUST.items():
                if source.lower() in t.source.lower():
                    conf = min(conf + 0.3 * trust, 1.0)
                    break

            # Recency bonus (more recent = higher confidence)
            try:
                pub = datetime.datetime.fromisoformat(t.published_at.replace("Z", "+00:00"))
                age_hours = (datetime.datetime.now(datetime.timezone.utc) - pub).total_seconds() / 3600
                if age_hours < 6:    conf = min(conf + 0.15, 1.0)
                elif age_hours < 24: conf = min(conf + 0.10, 1.0)
                elif age_hours > 72: conf = max(conf - 0.10, 0.0)
            except Exception:
                pass

            # Severity plausibility check
            if t.severity > 9 and conf < 0.6:
                conf *= 0.8   # high severity from low-trust source — discount

            t.confidence = round(conf, 3)
            t.verified   = conf > 0.65
            verified.append(t)

        return verified

    def verify_route_risk(self, route: RoutePlan,
                           all_routes: List["RoutePlan"]) -> List[str]:
        """
        Cross-check route risk against alternatives and generate warning flags.
        Returns list of flag strings.
        """
        flags = []

        if route.xgb_risk_score > 90 :
            flags.append("CRITICAL_RISK: Score exceeds 75 — route not recommended.")

        if route.xgb_risk_score > 50 and route.alternative_rank == 1:
            alternatives_lower = [r for r in all_routes
                                  if r.xgb_risk_score < route.xgb_risk_score - 10
                                  and r.alternative_rank != 1]
            if alternatives_lower:
                flags.append(f"SAFER_ALTERNATIVE_EXISTS: Route #{alternatives_lower[0].alternative_rank} "
                             f"has risk {alternatives_lower[0].xgb_risk_score:.1f} vs {route.xgb_risk_score:.1f}.")

        unverified_threats = [t for t in route.threat_events_near_path
                              if not t.verified and t.severity > 6]
        if unverified_threats:
            flags.append(f"UNVERIFIED_HIGH_SEVERITY: {len(unverified_threats)} high-severity threats "
                         f"from unverified sources — treat with caution.")

        high_knn = route.knn_geo_threat_score > 7
        if high_knn:
            flags.append(f"GEO_THREAT_CLUSTER: KNN proximity threat score {route.knn_geo_threat_score:.1f}/10 "
                         f"— dense threat cluster near route.")

        chokepoints_on_path = [n for n in route.path if n in CHOKEPOINTS]
        if len(chokepoints_on_path) > 1:
            flags.append(f"MULTI_CHOKEPOINT: Route passes {len(chokepoints_on_path)} chokepoints: "
                         f"{', '.join(chokepoints_on_path)}.")

        return flags

# ──────────────────────────────────────────────────────────────────────────────
# 10. PIPELINE ORCHESTRATOR
# ──────────────────────────────────────────────────────────────────────────────

class CargoRiskPipeline:

    def __init__(self, k_paths: int = 5, news_query: str = "shipping war piracy sea route"):
        self.k_paths     = k_paths
        self.news_query  = news_query
        self.G           = build_graph()
        self.knn         = KNNGeoThreatMapper(k=5, radius_km=500)
        self.xgb_scorer  = XGBRouteRiskScorer()
        self.verifier    = VerificationLayer()
        self.threats: List[ThreatEvent] = []

    def initialise(self):
        print("\n" + "═"*60)
        print("  CARGO RISK PIPELINE — INITIALISING")
        print("═"*60)

        # 1. Fetch and classify news threats
        print("\n[STEP 1] Fetching threat intelligence from news sources...")
        raw_threats = fetch_news_threats(self.news_query)

        # 2. Verify threats
        print("[STEP 2] Running verification layer on threat events...")
        self.threats = self.verifier.verify_threat_events(raw_threats)
        verified_count = sum(1 for t in self.threats if t.verified)
        print(f"  [VERIFY] {verified_count}/{len(self.threats)} threats verified.")

        # 3. Fit KNN
        print("[STEP 3] Fitting KNN Geo-Threat Mapper...")
        self.knn.fit(self.threats)

        # 4. Train XGBoost
        print("[STEP 4] Training XGBoost Risk Scorer...")
        self.xgb_scorer.train()

        print("\n  Pipeline ready.\n")

    def _threats_near_path(self, path: List[str],
                            radius_km: float = 600) -> List[ThreatEvent]:
        """Find threat events within radius_km of any node on the path."""
        near = []
        for node in path:
            gp = self.G.nodes[node]["geopoint"]
            for threat in self.threats:
                d = haversine(gp, GeoPoint("t", threat.lat, threat.lon, ""))
                if d <= radius_km and threat not in near:
                    near.append(threat)
        return near

    def _path_distance(self, path: List[str]) -> float:
        total = 0.0
        for i in range(len(path) - 1):
            total += self.G[path[i]][path[i+1]]["distance"]
        return total

    def _path_base_risk(self, path: List[str]) -> float:
        risks = [self.G[path[i]][path[i+1]]["base_risk"]
                 for i in range(len(path) - 1)]
        return max(risks) if risks else 0.0

    def _path_knn_score(self, path: List[str]) -> float:
        """Maximum KNN threat score along all edges on the path."""
        scores = []
        for i in range(len(path) - 1):
            p1 = self.G.nodes[path[i]]["geopoint"]
            p2 = self.G.nodes[path[i+1]]["geopoint"]
            scores.append(self.knn.edge_threat_score(p1, p2))
        return max(scores) if scores else 0.0

    def analyse(self, origin: str, destination: str,
                 cargo_value_norm: float = 0.5,
                 cargo_type: str = "general") -> List[RoutePlan]:
        """
        Main entry point. Returns list of RoutePlan objects (best first).

        Args:
            origin:           Node name from WAYPOINTS
            destination:      Node name from WAYPOINTS
            cargo_value_norm: 0=low-value, 1=high-value (affects risk tolerance)
            cargo_type:       'general' | 'hazmat' | 'perishable' | 'bulk'
        """
        print(f"\n{'═'*60}")
        print(f"  ANALYSIS: {origin} → {destination}")
        print(f"  Cargo: {cargo_type.upper()} | Value Index: {cargo_value_norm:.2f}")
        print(f"{'═'*60}")

        if origin not in self.G or destination not in self.G:
            raise ValueError(f"Unknown node(s): {origin}, {destination}")

        # A. Yen's K-shortest paths
        print(f"\n[A] Computing Yen's {self.k_paths} shortest paths (Dijkstra-based)...")
        k_paths = yens_k_shortest(self.G, origin, destination, self.knn, K=self.k_paths)
        print(f"  Found {len(k_paths)} candidate routes.")

        now_month = datetime.datetime.now().month
        route_plans = []

        for rank, (path, _cost) in enumerate(k_paths, start=1):
            dist        = self._path_distance(path)
            base_risk   = self._path_base_risk(path)
            knn_max     = self._path_knn_score(path)
            knn_mean    = knn_max * 0.65
            nearby      = self._threats_near_path(path)
            n_threats   = len(nearby)
            max_sev     = max((t.severity for t in nearby), default=0.0)
            n_choke     = sum(1 for n in path if n in CHOKEPOINTS)
            war_adj     = int(any(t.threat_type in ("war","terrorism") for t in nearby))
            sanctions   = int(any(t.threat_type == "sanctions" for t in nearby))
            weather     = max((t.severity for t in nearby
                               if t.threat_type == "natural"), default=0.0)

            features = {
                "route_distance_norm":   dist / 15000,
                "num_hops_norm":         len(path) / 8,
                "base_edge_risk":        base_risk,
                "knn_threat_max_norm":   knn_max / 10,
                "knn_threat_mean_norm":  knn_mean / 10,
                "num_threats_near_norm": min(n_threats / 8, 1.0),
                "max_threat_severity_norm": max_sev / 10,
                "passes_chokepoint":     int(n_choke > 0),
                "num_chokepoints_norm":  n_choke / 4,
                "cargo_value_norm":      cargo_value_norm,
                "transport_mode_sea":    1,
                "seasonal_signal":       math.sin(2 * math.pi * now_month / 12),
                "weather_risk_norm":     weather / 5,
                "sanctions_region":      sanctions,
                "war_zone_adjacent":     war_adj,
            }

            xgb_score, risk_class = self.xgb_scorer.predict(features)
            estimated_days        = dist / 650   # ~650 km/day average container ship

            plan = RoutePlan(
                path=path,
                total_distance_km=round(dist, 1),
                total_risk_score=round(xgb_score, 2),
                threat_events_near_path=nearby,
                knn_geo_threat_score=round(knn_max, 2),
                xgb_risk_score=round(xgb_score, 2),
                client_risk_class=risk_class,
                estimated_days=round(estimated_days, 1),
                alternative_rank=rank,
            )
            route_plans.append(plan)

        # B. Verify routes
        print("[B] Running verification layer on route plans...")
        for plan in route_plans:
            plan.verification_flags = self.verifier.verify_route_risk(plan, route_plans)

        # Sort: best first (lowest XGB risk)
        route_plans.sort(key=lambda r: r.xgb_risk_score)
        for i, rp in enumerate(route_plans, start=1):
            rp.alternative_rank = i

        return route_plans

    def print_report(self, routes: List[RoutePlan], origin: str, destination: str):
        print(f"\n{'═'*60}")
        print(f"  ROUTE RISK REPORT: {origin} → {destination}")
        print(f"  Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"{'═'*60}")

        for rp in routes:
            bar = "█" * int(rp.xgb_risk_score / 5) + "░" * (20 - int(rp.xgb_risk_score / 5))
            print(f"\n  Route #{rp.alternative_rank}  [{rp.client_risk_class}]")
            print(f"  Path: {' → '.join(rp.path)}")
            print(f"  Distance:    {rp.total_distance_km:,.0f} km")
            print(f"  Est. Transit:{rp.estimated_days:.1f} days")
            print(f"  XGB Score:   {rp.xgb_risk_score:5.1f}/100  |{bar}|")
            print(f"  KNN Threat:  {rp.knn_geo_threat_score:.1f}/10")
            print(f"  Threats:     {len(rp.threat_events_near_path)} events near path")
            if rp.threat_events_near_path:
                for t in sorted(rp.threat_events_near_path,
                                key=lambda x: x.severity, reverse=True)[:3]:
                    verified_str = "✓" if t.verified else "?"
                    print(f"    [{verified_str}] [{t.threat_type.upper():12}] sev={t.severity:.1f} | {t.title[:60]}")
            if rp.verification_flags:
                print(f"  Flags:")
                for flag in rp.verification_flags:
                    print(f"    ⚠  {flag}")

        print(f"\n{'═'*60}")
        best = routes[0]
        print(f"  RECOMMENDATION: Route #{best.alternative_rank} ({' → '.join(best.path)})")
        print(f"  Risk Class: {best.client_risk_class} | Score: {best.xgb_risk_score:.1f}/100")
        print(f"  Client Risk Score: {best.xgb_risk_score:.1f} / 100")
        print(f"{'═'*60}\n")

        # Feature importance
        print("  XGBoost Feature Importance (Top 10):")
        for feat, imp in list(self.xgb_scorer.get_feature_importance().items())[:10]:
            bar = "█" * int(imp * 40)
            print(f"    {feat:<25} {imp:.4f}  {bar}")

    def to_json(self, routes: List[RoutePlan], origin: str, destination: str) -> str:
        """Serialise results to JSON for API / frontend consumption."""
        return json.dumps({
            "origin":       origin,
            "destination":  destination,
            "generated_at": datetime.datetime.now().isoformat(),
            "routes": [
                {
                    "rank":              r.alternative_rank,
                    "path":              r.path,
                    "distance_km":       r.total_distance_km,
                    "estimated_days":    r.estimated_days,
                    "xgb_risk_score":    r.xgb_risk_score,
                    "knn_threat_score":  r.knn_geo_threat_score,
                    "risk_class":        r.client_risk_class,
                    "num_threats":       len(r.threat_events_near_path),
                    "verification_flags": r.verification_flags,
                    "threats": [
                        {
                            "title":       t.title,
                            "type":        t.threat_type,
                            "severity":    t.severity,
                            "verified":    t.verified,
                            "confidence":  t.confidence,
                            "lat":         t.lat,
                            "lon":         t.lon,
                        }
                        for t in sorted(r.threat_events_near_path,
                                        key=lambda x: x.severity, reverse=True)
                    ],
                }
                for r in routes
            ],
        }, indent=2)


# ──────────────────────────────────────────────────────────────────────────────
# 11.  MAIN — DEMO RUN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pipeline = CargoRiskPipeline(k_paths=5, news_query="shipping war piracy red sea")
    pipeline.initialise()

    # Example 1: Shanghai → Rotterdam (classic Asia–Europe route, Red Sea risk)
    routes = pipeline.analyse(
        origin="SHANGHAI",
        destination="ROTTERDAM",
        cargo_value_norm=0.8,
        cargo_type="electronics",
    )
    pipeline.print_report(routes, "SHANGHAI", "ROTTERDAM")

    # Dump JSON (for API / dashboard integration)
    out_json = pipeline.to_json(routes, "SHANGHAI", "ROTTERDAM")
    with open("/mnt/user-data/outputs/route_analysis.json", "w") as f:
        f.write(out_json)
    print("  JSON output saved to route_analysis.json")

    # Example 2: Dubai → Rotterdam (Hormuz / Red Sea exposure)
    routes2 = pipeline.analyse(
        origin="DUBAI",
        destination="ROTTERDAM",
        cargo_value_norm=0.6,
        cargo_type="petroleum",
    )
    pipeline.print_report(routes2, "DUBAI", "ROTTERDAM")