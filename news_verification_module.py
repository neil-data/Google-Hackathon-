"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         NEWS VERIFICATION MODULE  v1.0                                      ║
║                                                                              ║
║  Pipeline:                                                                   ║
║    [Web Crawler (RSS/HTML)] ──┐                                              ║
║    [Google News API]  ────────┼──► [Aggregator] ──► [NLP Classifier]        ║
║    [◄ INSERT MODULE ►] ───────┘         │                                   ║
║                                         ▼                                   ║
║                              [Verification Engine]                           ║
║                               ├─ Cross-source corroboration                 ║
║                               ├─ Temporal decay scoring                     ║
║                               ├─ Source trust weighting                     ║
║                               ├─ Geo-entity confidence                      ║
║                               └─ Consensus voting                            ║
║                                         │                                   ║
║                              [Verified ThreatEvent list]                    ║
║                                         │                                   ║
║                              [Weather API integration]                       ║
║                                         │                                   ║
║                         ──► cargo_risk_v3 XGB pipeline ◄──                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

VERIFICATION ALGORITHM — Mathematical Summary
─────────────────────────────────────────────────────────────────────────────

  1. SOURCE TRUST SCORE  T(s) ∈ [0,1]
     Pre-assigned per news source based on editorial track record.

  2. TEMPORAL DECAY FACTOR  τ(t) = e^(−λ·Δt)
     λ = 0.15 per hour  →  50% decay at ~4.6 hours, near-zero at 48 hrs.

  3. GEO-ENTITY CONFIDENCE  G(a) ∈ [0,1]
     Fraction of region-name tokens in article that match known maritime zones.

  4. CORROBORATION SCORE  C(event)
     C = min(1,  n_sources / 3)   where n_sources = independent sources
                                   reporting the same event type in same region.

  5. COMPOSITE CONFIDENCE  Ψ(a)
     Ψ(a) = w₁·T(s) + w₂·τ(t) + w₃·G(a) + w₄·C(event)
             w₁=0.30  w₂=0.25  w₃=0.20  w₄=0.25

  6. VERIFICATION THRESHOLD  θ = 0.60
     event.verified = True   iff  Ψ(a) ≥ θ
     Severity is discounted:  sev_adj = sev × (0.5 + 0.5·Ψ(a))

─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import re
import json
import math
import time
import hashlib
import logging
import datetime
import threading
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
import feedparser

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("NewsVerification")


# ══════════════════════════════════════════════════════════════════════════════
# 1.  DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RawArticle:
    """Raw scraped article before classification or verification."""
    uid:          str           # SHA-1 of title+source
    title:        str
    body:         str
    url:          str
    source_name:  str
    fetched_at:   str           # ISO UTC
    region_hints: List[str] = field(default_factory=list)

@dataclass
class ThreatEvent:
    """Classified, verified threat event output to the cargo pipeline."""
    uid:          str
    title:        str
    summary:      str
    threat_type:  str           # war|piracy|terrorism|sanctions|natural|traffic
    raw_severity: float         # 0–10 before confidence adjustment
    adj_severity: float         # 0–10 after confidence discount
    lat:          float
    lon:          float
    region:       str
    source_name:  str
    source_url:   str
    published_at: str
    # Verification scores
    trust_score:         float = 0.0   # T(s)
    temporal_decay:      float = 0.0   # τ(t)
    geo_confidence:      float = 0.0   # G(a)
    corroboration:       float = 0.0   # C(event)
    composite_confidence: float = 0.0  # Ψ(a)
    verified:            bool  = False
    n_corroborating:     int   = 0


# ══════════════════════════════════════════════════════════════════════════════
# 2.  CONFIGURATION & CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# ── Source trust scores  T(s) ────────────────────────────────────────────────
SOURCE_TRUST: Dict[str, float] = {
    # International wire services
    "reuters":           1.00,
    "ap":                1.00,
    "afp":               0.95,
    "bloomberg":         0.95,
    "bbc":               0.92,
    # Shipping & maritime specialists
    "lloyds list":       1.00,
    "splash247":         0.90,
    "tradewinds":        0.88,
    "maritime executive":0.87,
    "gcaptain":          0.85,
    "imb":               1.00,   # International Maritime Bureau
    "unosat":            0.95,
    # Regional news (lower trust for geopolitical claims)
    "al jazeera":        0.80,
    "xinhua":            0.60,
    "tass":              0.55,
    "rt":                0.40,
    # Weather / environmental
    "wmo":               1.00,
    "noaa":              1.00,
    "jtwc":              1.00,   # Joint Typhoon Warning Center
    "ecmwf":             0.98,
    # Default for unknown sources
    "_default":          0.45,
}

# ── Threat keyword lexicon ────────────────────────────────────────────────────
THREAT_LEXICON: Dict[str, Dict] = {
    "war": {
        "keywords": [
            "war","military strike","bombing","airstrike","airstrike","combat",
            "invasion","missile","artillery","troops","offensive","frontline",
            "shelling","armed conflict","naval clash","warship","houthi",
            "drone attack","rocket fire","naval bombardment",
        ],
        "base_severity": 9.0,
    },
    "piracy": {
        "keywords": [
            "piracy","pirate","hijack","hijacking","armed robbery at sea",
            "vessel seized","crew kidnapped","armed boarding","ransom",
            "skiff attack","suspicious vessel","high-risk area",
        ],
        "base_severity": 7.5,
    },
    "terrorism": {
        "keywords": [
            "terrorism","terrorist attack","explosion","bomb","ied",
            "insurgency","extremist","militant","suicide bomb","blast",
            "sabotage","port attack",
        ],
        "base_severity": 8.5,
    },
    "sanctions": {
        "keywords": [
            "sanctions","embargo","blockade","trade ban","export control",
            "port closure","restricted trade","ofac","blacklist","asset freeze",
            "trade restriction","vessel ban",
        ],
        "base_severity": 6.5,
    },
    "natural": {
        "keywords": [
            "hurricane","typhoon","cyclone","earthquake","tsunami","flood",
            "storm","port damage","infrastructure damage","severe weather",
            "rough sea","force 10","gale warning","tropical storm",
        ],
        "base_severity": 5.5,
    },
    "traffic": {
        "keywords": [
            "port congestion","vessel queue","berth delay","canal closure",
            "traffic jam","anchorage full","waiting time","backlog",
        ],
        "base_severity": 3.0,
    },
}

# ── Known maritime regions with approximate geo-centres ──────────────────────
MARITIME_REGIONS: Dict[str, Tuple[float, float]] = {
    "red sea":              (20.0,  38.0),
    "gulf of aden":         (12.0,  46.0),
    "strait of hormuz":     (26.5,  56.5),
    "persian gulf":         (27.0,  51.0),
    "arabian sea":          (15.0,  65.0),
    "south china sea":      (14.0, 114.0),
    "taiwan strait":        (24.0, 119.0),
    "malacca strait":       ( 1.5, 103.0),
    "strait of malacca":    ( 1.5, 103.0),
    "bab el mandeb":        (12.6,  43.4),
    "bab-el-mandeb":        (12.6,  43.4),
    "suez canal":           (30.5,  32.3),
    "black sea":            (43.0,  34.0),
    "ukraine":              (49.0,  32.0),
    "russia":               (55.0,  37.0),
    "gaza":                 (31.5,  34.4),
    "israel":               (31.5,  35.0),
    "yemen":                (15.5,  48.5),
    "somalia":              ( 5.0,  46.0),
    "gulf of guinea":       ( 3.0,   3.0),
    "nigeria":              ( 9.0,   8.0),
    "horn of africa":       ( 5.0,  48.0),
    "mediterranean":        (36.0,  18.0),
    "north atlantic":       (45.0, -40.0),
    "bay of bengal":        (13.0,  88.0),
    "indian ocean":         (-5.0,  75.0),
    "cape of good hope":   (-34.0,  18.5),
    "mozambique channel":   (-18.0,  37.0),
    "iran":                 (32.0,  53.5),
}

# ── RSS feed sources (free, no API key needed) ───────────────────────────────
RSS_FEEDS: List[Dict] = [
    # Maritime / shipping specialists
    {"url": "https://splash247.com/feed/",            "source": "Splash247",         "trust": 0.90},
    {"url": "https://www.gcaptain.com/feed/",         "source": "gCaptain",          "trust": 0.85},
    {"url": "https://maritimeexecutive.com/rss.xml",  "source": "Maritime Executive", "trust": 0.87},
    # General news with maritime relevance
    {"url": "https://feeds.reuters.com/reuters/topNews","source": "Reuters",          "trust": 1.00},
    {"url": "https://feeds.bbci.co.uk/news/world/rss.xml","source":"BBC",            "trust": 0.92},
    # Piracy / security
    {"url": "https://www.icc-ccs.org/rss/piracy.xml", "source": "IMB (ICC-CCS)",     "trust": 1.00},
]

# ── Scraping targets (HTML pages with shipping news) ─────────────────────────
SCRAPE_TARGETS: List[Dict] = [
    {
        "url":    "https://splash247.com/category/sector/safety/",
        "source": "Splash247-Safety",
        "trust":  0.90,
        "article_selector": "h2.entry-title a",
        "max_articles": 5,
    },
    {
        "url":    "https://www.gcaptain.com/maritime-news/",
        "source": "gCaptain-News",
        "trust":  0.85,
        "article_selector": "h2.entry-title a",
        "max_articles": 5,
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# 3.  GOOGLE NEWS API CONNECTOR
#     ─────────────────────────────────────────────────────────────────────────
#     To activate: set GOOGLE_NEWS_API_KEY to your NewsAPI.org key
#     Leave as None → module silently skips and uses crawler data only.
# ══════════════════════════════════════════════════════════════════════════════

GOOGLE_NEWS_API_KEY: Optional[str] = None       # ← INSERT YOUR KEY HERE
GOOGLE_NEWS_ENDPOINT = "https://newsapi.org/v2/everything"
GOOGLE_NEWS_QUERIES  = [
    "shipping war piracy red sea",
    "maritime cargo risk route",
    "port closure sanctions embargo",
    "typhoon cyclone shipping disruption",
    "strait of hormuz vessel",
    "south china sea military",
]


def fetch_google_news(query: str, max_articles: int = 20) -> List[RawArticle]:
    """
    ┌─────────────────────────────────────────────────────────────┐
    │  GOOGLE NEWS API INSERTION POINT                            │
    │                                                             │
    │  To enable:                                                 │
    │    1. Get a free key at https://newsapi.org                 │
    │    2. Set GOOGLE_NEWS_API_KEY = "your_key_here"             │
    │    3. This function becomes active automatically.           │
    │                                                             │
    │  Returns: List[RawArticle]  (same schema as crawler output) │
    │  Falls back gracefully to [] when key is absent.            │
    └─────────────────────────────────────────────────────────────┘
    """
    if not GOOGLE_NEWS_API_KEY:
        return []       # ← silent no-op until key is provided

    articles: List[RawArticle] = []
    try:
        params = {
            "q":        query,
            "language": "en",
            "sortBy":   "publishedAt",
            "pageSize": max_articles,
            "apiKey":   GOOGLE_NEWS_API_KEY,
        }
        resp = requests.get(GOOGLE_NEWS_ENDPOINT, params=params, timeout=12)
        resp.raise_for_status()
        for art in resp.json().get("articles", []):
            text  = f"{art.get('title','')} {art.get('description','')}"
            uid   = hashlib.sha1(text.encode()).hexdigest()[:12]
            articles.append(RawArticle(
                uid=uid,
                title=art.get("title", "")[:200],
                body=art.get("description", "")[:800],
                url=art.get("url", ""),
                source_name=art.get("source", {}).get("name", "Unknown"),
                fetched_at=datetime.datetime.utcnow().isoformat() + "Z",
            ))
        log.info(f"[GoogleNewsAPI] '{query}' → {len(articles)} articles")
    except Exception as e:
        log.warning(f"[GoogleNewsAPI] Error: {e}")
    return articles


# ══════════════════════════════════════════════════════════════════════════════
# 4.  CUSTOM API / DATA SOURCE INSERTION MODULE
#     ─────────────────────────────────────────────────────────────────────────
#     Drop any additional data source here.  Return List[RawArticle].
#     The aggregator auto-merges results.
# ══════════════════════════════════════════════════════════════════════════════

class CustomSourceInsertion:
    """
    ┌──────────────────────────────────────────────────────────────┐
    │  CUSTOM SOURCE INSERTION POINT                               │
    │                                                              │
    │  Implement fetch() to plug in any proprietary data feed:     │
    │   • Lloyd's Intelligence API                                 │
    │   • MarineTraffic incident feed                              │
    │   • Dryad Global risk API                                    │
    │   • BIMCO safety alerts                                      │
    │   • Internal corporate incident database                     │
    │   • Any REST / GraphQL / WebSocket source                    │
    │                                                              │
    │  Schema contract: return List[RawArticle]                    │
    └──────────────────────────────────────────────────────────────┘
    """

    def fetch(self) -> List[RawArticle]:
        # ── INSERTION POINT ──────────────────────────────────────
        # Example skeleton:
        #
        #   resp = requests.get("https://your-api.com/incidents",
        #                       headers={"Authorization": "Bearer YOUR_TOKEN"})
        #   return [
        #       RawArticle(
        #           uid=item["id"],
        #           title=item["headline"],
        #           body=item["details"],
        #           url=item["link"],
        #           source_name="YourSource",
        #           fetched_at=item["timestamp"],
        #       )
        #       for item in resp.json()
        #   ]
        # ─────────────────────────────────────────────────────────
        return []       # ← returns empty until you implement above


# ══════════════════════════════════════════════════════════════════════════════
# 5.  WEB CRAWLER
# ══════════════════════════════════════════════════════════════════════════════

class WebCrawler:
    """
    Two-mode crawler:
      Mode A) RSS/Atom feeds  — fast, structured, preferred
      Mode B) HTML scraping   — fallback for sites without RSS
    """

    HEADERS = {
        "User-Agent": ("Mozilla/5.0 (compatible; CargoRiskBot/1.0; "
                       "+https://example.com/cargorisk)"),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    REQUEST_TIMEOUT = 12
    MAX_BODY_CHARS  = 1200

    def __init__(self):
        self._seen: set = set()

    def crawl_rss(self, feed: Dict) -> List[RawArticle]:
        """Parse an RSS/Atom feed and return RawArticle list."""
        articles: List[RawArticle] = []
        try:
            parsed = feedparser.parse(feed["url"])
            for entry in parsed.entries[:10]:
                title = entry.get("title", "").strip()
                body  = BeautifulSoup(
                    entry.get("summary", entry.get("description", "")),
                    "html.parser"
                ).get_text(" ", strip=True)[:self.MAX_BODY_CHARS]

                uid = hashlib.sha1(
                    f"{title}{feed['source']}".encode()
                ).hexdigest()[:12]
                if uid in self._seen:
                    continue
                self._seen.add(uid)

                articles.append(RawArticle(
                    uid=uid,
                    title=title,
                    body=body,
                    url=entry.get("link", feed["url"]),
                    source_name=feed["source"],
                    fetched_at=datetime.datetime.utcnow().isoformat() + "Z",
                ))
        except Exception as e:
            log.debug(f"[RSS] {feed['source']}: {e}")
        return articles

    def crawl_html(self, target: Dict) -> List[RawArticle]:
        """
        Scrape an HTML news listing page, follow article links,
        extract title + first 3 paragraphs.
        """
        articles: List[RawArticle] = []
        try:
            resp = requests.get(target["url"], headers=self.HEADERS,
                                timeout=self.REQUEST_TIMEOUT)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            links = soup.select(target["article_selector"])
            for a_tag in links[:target.get("max_articles", 5)]:
                href = a_tag.get("href", "")
                if not href.startswith("http"):
                    href = urljoin(target["url"], href)
                title = a_tag.get_text(strip=True)
                uid   = hashlib.sha1(
                    f"{title}{target['source']}".encode()
                ).hexdigest()[:12]
                if uid in self._seen:
                    continue
                self._seen.add(uid)

                # Fetch article body
                body = self._fetch_article_body(href)
                articles.append(RawArticle(
                    uid=uid,
                    title=title,
                    body=body,
                    url=href,
                    source_name=target["source"],
                    fetched_at=datetime.datetime.utcnow().isoformat() + "Z",
                ))
                time.sleep(0.8)   # polite crawl delay

        except Exception as e:
            log.debug(f"[HTML] {target['source']}: {e}")
        return articles

    def _fetch_article_body(self, url: str) -> str:
        """Fetch article URL and extract readable text."""
        try:
            resp = requests.get(url, headers=self.HEADERS,
                                timeout=self.REQUEST_TIMEOUT)
            soup = BeautifulSoup(resp.text, "html.parser")
            # Try common article containers
            for selector in ["article", ".entry-content", ".post-content",
                              "main", ".article-body"]:
                container = soup.select_one(selector)
                if container:
                    text = container.get_text(" ", strip=True)
                    return text[:self.MAX_BODY_CHARS]
            # Fallback: all paragraphs
            paras = soup.find_all("p")
            return " ".join(p.get_text(strip=True) for p in paras[:6])[:self.MAX_BODY_CHARS]
        except Exception:
            return ""

    def crawl_all(self, use_html_scraper: bool = False) -> List[RawArticle]:
        """
        Crawl all RSS feeds and (optionally) HTML targets.
        Returns de-duplicated list of RawArticles.
        """
        all_articles: List[RawArticle] = []

        log.info(f"[Crawler] Starting RSS crawl ({len(RSS_FEEDS)} feeds)...")
        for feed in RSS_FEEDS:
            arts = self.crawl_rss(feed)
            all_articles.extend(arts)
            log.info(f"  {feed['source']}: {len(arts)} articles")

        if use_html_scraper:
            log.info(f"[Crawler] Starting HTML scrape ({len(SCRAPE_TARGETS)} targets)...")
            for target in SCRAPE_TARGETS:
                arts = self.crawl_html(target)
                all_articles.extend(arts)
                log.info(f"  {target['source']}: {len(arts)} articles")

        log.info(f"[Crawler] Total raw articles: {len(all_articles)}")
        return all_articles


# ══════════════════════════════════════════════════════════════════════════════
# 6.  NLP CLASSIFIER
#     Keyword matching + region extraction + severity estimation
# ══════════════════════════════════════════════════════════════════════════════

class NLPClassifier:

    STOPWORDS = {
        "the","a","an","and","or","but","in","on","at","to","for",
        "of","with","by","from","that","this","is","was","are","were",
    }

    def classify(self, article: RawArticle) -> Optional[Dict]:
        """
        Returns dict with:
          threat_type, raw_severity, lat, lon, region, matched_keywords
        or None if no threat detected.
        """
        text  = f"{article.title} {article.body}".lower()
        tokens = set(re.sub(r"[^a-z\s]", " ", text).split()) - self.STOPWORDS

        # ── Threat type detection ────────────────────────────────────────────
        best_type, best_score, best_sev = None, 0, 0.0
        for t_type, conf in THREAT_LEXICON.items():
            hits = sum(1 for kw in conf["keywords"] if kw in text)
            if hits > best_score:
                best_score = hits
                best_type  = t_type
                best_sev   = conf["base_severity"]

        if best_score == 0:
            return None

        # Severity modifier based on keyword density
        word_count = max(len(text.split()), 1)
        density    = best_score / word_count
        sev_adj    = min(best_sev * (1 + density * 20), 10.0)

        # ── Geo extraction ───────────────────────────────────────────────────
        region, lat, lon = self._extract_geo(text)
        if lat is None:
            return None           # discard articles with no geo signal

        return {
            "threat_type":      best_type,
            "raw_severity":     round(sev_adj, 2),
            "lat":              lat,
            "lon":              lon,
            "region":           region,
            "n_keyword_hits":   best_score,
        }

    def _extract_geo(self, text: str) -> Tuple[str, Optional[float], Optional[float]]:
        """
        Match text against MARITIME_REGIONS (longest match first).
        Returns (region_name, lat, lon) or (None, None, None).
        """
        for region in sorted(MARITIME_REGIONS.keys(),
                             key=lambda r: -len(r)):
            if region in text:
                lat, lon = MARITIME_REGIONS[region]
                return region, lat, lon
        return "unknown", None, None


# ══════════════════════════════════════════════════════════════════════════════
# 7.  VERIFICATION ENGINE
#     Implements the 5-signal composite confidence formula Ψ(a)
# ══════════════════════════════════════════════════════════════════════════════

class VerificationEngine:
    """
    Ψ(a) = w₁·T(s) + w₂·τ(t) + w₃·G(a) + w₄·C(event)
    w = [0.30, 0.25, 0.20, 0.25]   θ = 0.60
    """

    W1, W2, W3, W4 = 0.30, 0.25, 0.20, 0.25
    THETA           = 0.60      # verification threshold
    LAMBDA_DECAY    = 0.15      # per-hour exponential decay λ

    # ── Signal 1: Source trust T(s) ──────────────────────────────────────────
    def trust_score(self, source_name: str) -> float:
        """T(s) — pre-assigned trust per source."""
        s_low = source_name.lower()
        for key, score in SOURCE_TRUST.items():
            if key in s_low:
                return score
        return SOURCE_TRUST["_default"]

    # ── Signal 2: Temporal decay τ(t) = e^(−λ·Δt) ────────────────────────────
    def temporal_decay(self, fetched_at: str) -> float:
        """τ(t) = exp(−λ·Δt_hours)  where λ = 0.15 /hour"""
        try:
            dt_str = fetched_at.replace("Z", "+00:00")
            pub_dt = datetime.datetime.fromisoformat(dt_str)
            now    = datetime.datetime.now(datetime.timezone.utc)
            delta_hours = (now - pub_dt).total_seconds() / 3600
            return math.exp(-self.LAMBDA_DECAY * max(delta_hours, 0))
        except Exception:
            return 0.5

    # ── Signal 3: Geo-entity confidence G(a) ─────────────────────────────────
    def geo_confidence(self, region: str, article_text: str) -> float:
        """
        G(a) = fraction of known maritime region tokens found in article.
        Rewards specificity: 'Red Sea' > 'Middle East'.
        """
        if region == "unknown":
            return 0.0
        all_regions  = list(MARITIME_REGIONS.keys())
        text_lower   = article_text.lower()
        found        = sum(1 for r in all_regions if r in text_lower)
        specificity  = 1.0 if len(region) >= 8 else 0.6
        return min(found / max(len(all_regions) * 0.05, 1), 1.0) * specificity

    # ── Signal 4: Corroboration C(event) ─────────────────────────────────────
    def corroboration_score(self, event_type: str, region: str,
                             all_classified: List[Dict]) -> Tuple[float, int]:
        """
        C = min(1,  n_independent_sources / 3)
        Two articles from the same source do not double-count.
        """
        same_region_type = [
            c for c in all_classified
            if c.get("threat_type") == event_type
            and c.get("region") == region
        ]
        unique_sources = len(set(c.get("source_name") for c in same_region_type))
        score          = min(1.0, unique_sources / 3)
        return score, unique_sources

    # ── Composite Ψ(a) ───────────────────────────────────────────────────────
    def compute_confidence(self, article: RawArticle, classified: Dict,
                           all_classified: List[Dict]) -> ThreatEvent:
        """
        Full verification pipeline for one article → ThreatEvent.
        Ψ(a) = 0.30·T(s) + 0.25·τ(t) + 0.20·G(a) + 0.25·C(event)
        """
        T   = self.trust_score(article.source_name)
        tau = self.temporal_decay(article.fetched_at)
        G   = self.geo_confidence(
                  classified["region"],
                  f"{article.title} {article.body}"
              )
        C, n_corr = self.corroboration_score(
            classified["threat_type"],
            classified["region"],
            [{**c, "source_name": c.get("source_name", "")}
             for c in all_classified]
        )

        psi   = self.W1*T + self.W2*tau + self.W3*G + self.W4*C
        psi   = round(min(psi, 1.0), 4)
        verified = psi >= self.THETA

        # Confidence-adjusted severity:  sev_adj = sev × (0.5 + 0.5·Ψ)
        raw_sev = classified["raw_severity"]
        adj_sev = round(raw_sev * (0.5 + 0.5 * psi), 2)

        return ThreatEvent(
            uid=article.uid,
            title=article.title[:140],
            summary=article.body[:300],
            threat_type=classified["threat_type"],
            raw_severity=raw_sev,
            adj_severity=adj_sev,
            lat=classified["lat"],
            lon=classified["lon"],
            region=classified["region"],
            source_name=article.source_name,
            source_url=article.url,
            published_at=article.fetched_at,
            trust_score=round(T,4),
            temporal_decay=round(tau,4),
            geo_confidence=round(G,4),
            corroboration=round(C,4),
            composite_confidence=psi,
            verified=verified,
            n_corroborating=n_corr,
        )

    def verify_batch(self, articles: List[RawArticle],
                     classified_pairs: List[Tuple[RawArticle, Dict]]
                     ) -> List[ThreatEvent]:
        """
        Verify all classified articles.
        Corroboration is computed across the full batch for cross-referencing.
        """
        all_classified_meta = [
            {**cls, "source_name": art.source_name}
            for art, cls in classified_pairs
        ]
        events = []
        for article, classified in classified_pairs:
            ev = self.compute_confidence(article, classified, all_classified_meta)
            events.append(ev)
        return events


# ══════════════════════════════════════════════════════════════════════════════
# 8.  WEATHER API INTEGRATION
#     Uses OpenWeatherMap free tier (no key → synthetic fallback)
# ══════════════════════════════════════════════════════════════════════════════

OPENWEATHER_API_KEY: Optional[str] = None  # ← INSERT YOUR KEY HERE
OPENWEATHER_ENDPOINT = "https://api.openweathermap.org/data/2.5/weather"

@dataclass
class WeatherReport:
    lat:         float
    lon:         float
    location:    str
    condition:   str     # Clear / Cloudy / Storm / Cyclone / ...
    wind_speed:  float   # m/s
    wave_height: float   # estimated from wind (Beaufort proxy)
    risk_factor: float   # 0–1 weather risk for cargo
    source:      str

def fetch_weather(lat: float, lon: float) -> WeatherReport:
    """
    Fetch live weather at (lat, lon).
    Falls back to a deterministic risk estimate if no API key.

    FORMULA (risk_factor):
        wind_risk  = min(wind_speed_ms / 28, 1.0)      [28 m/s = Storm force]
        wave_proxy = (wind_speed_ms ** 2) / (9.8 * 10)  [simplified JONSWAP]
        risk_factor = 0.6·wind_risk + 0.4·min(wave_proxy, 1)
    """
    if OPENWEATHER_API_KEY:
        try:
            resp = requests.get(
                OPENWEATHER_ENDPOINT,
                params={"lat": lat, "lon": lon,
                        "appid": OPENWEATHER_API_KEY, "units": "metric"},
                timeout=8,
            )
            data = resp.json()
            wind   = data.get("wind", {}).get("speed", 0.0)
            cond   = data.get("weather", [{}])[0].get("description", "unknown")
            name   = data.get("name", f"{lat:.2f},{lon:.2f}")
            return _build_weather_report(lat, lon, name, wind, cond, "OpenWeatherMap")
        except Exception as e:
            log.debug(f"[Weather] API error: {e}")

    # ── Synthetic fallback (deterministic, lat/lon seeded) ───────────────────
    seed    = abs(int(lat * 100 + lon * 10)) % 7
    wind_ms = [2.0, 5.0, 8.0, 12.0, 16.0, 22.0, 28.0][seed]
    cond    = ["Clear","Light Swell","Moderate Sea","Fresh Breeze",
               "Near Gale","Storm","Storm"][seed]
    return _build_weather_report(lat, lon, f"{lat:.1f},{lon:.1f}",
                                  wind_ms, cond, "Synthetic (no API key)")

def _build_weather_report(lat, lon, name, wind_ms, cond, source) -> WeatherReport:
    """Build WeatherReport applying the risk formula."""
    wind_risk   = min(wind_ms / 28.0, 1.0)
    wave_proxy  = (wind_ms**2) / (9.8 * 10)
    risk_factor = round(0.6*wind_risk + 0.4*min(wave_proxy, 1.0), 3)
    wave_height = round(0.026 * wind_ms**2, 2)   # empirical deep-water formula
    return WeatherReport(
        lat=lat, lon=lon, location=name,
        condition=cond, wind_speed=round(wind_ms, 1),
        wave_height=wave_height, risk_factor=risk_factor, source=source,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 9.  MASTER AGGREGATOR — glues all modules together
# ══════════════════════════════════════════════════════════════════════════════

class NewsVerificationSystem:
    """
    Orchestrates: Crawler → Google News API → Custom Source
                → NLP → Verification → Weather enrichment
    Returns List[ThreatEvent] for injection into cargo_risk_v3.
    """

    def __init__(self, use_html_scraper: bool = False):
        self.crawler    = WebCrawler()
        self.classifier = NLPClassifier()
        self.verifier   = VerificationEngine()
        self.custom_src = CustomSourceInsertion()
        self.use_html   = use_html_scraper
        self._cache: List[ThreatEvent] = []
        self._last_run: Optional[datetime.datetime] = None

    def run(self, force_refresh: bool = False,
            cache_minutes: int = 30) -> List[ThreatEvent]:
        """
        Full pipeline run.  Results cached for `cache_minutes` minutes.
        """
        # Cache guard
        if (not force_refresh and self._cache and self._last_run and
                (datetime.datetime.utcnow() - self._last_run).seconds < cache_minutes * 60):
            log.info(f"[NVS] Using cached {len(self._cache)} events.")
            return self._cache

        log.info("══════════════════════════════════════════")
        log.info("  NEWS VERIFICATION SYSTEM — STARTING")
        log.info("══════════════════════════════════════════")

        # ── Step 1: Gather raw articles ───────────────────────────────────────
        all_raw: List[RawArticle] = []

        # A. Web crawler (RSS + optional HTML)
        crawler_articles = self.crawler.crawl_all(use_html_scraper=self.use_html)
        all_raw.extend(crawler_articles)

        # B. Google News API (active only if API key set)
        for query in GOOGLE_NEWS_QUERIES:
            gn_articles = fetch_google_news(query, max_articles=15)
            all_raw.extend(gn_articles)
        if GOOGLE_NEWS_API_KEY:
            log.info(f"[GoogleAPI] Active — {len(GOOGLE_NEWS_QUERIES)} queries fired.")
        else:
            log.info("[GoogleAPI] Inactive — set GOOGLE_NEWS_API_KEY to enable.")

        # C. Custom insertion module
        custom_articles = self.custom_src.fetch()
        if custom_articles:
            log.info(f"[CustomSrc] {len(custom_articles)} articles from custom source.")
        all_raw.extend(custom_articles)

        log.info(f"[Aggregator] {len(all_raw)} raw articles collected.")

        # ── Step 2: NLP classification ─────────────────────────────────────
        classified_pairs: List[Tuple[RawArticle, Dict]] = []
        for art in all_raw:
            result = self.classifier.classify(art)
            if result:
                classified_pairs.append((art, result))

        log.info(f"[NLP] {len(classified_pairs)} articles with threat signals.")

        # ── Step 3: Verification ───────────────────────────────────────────
        events = self.verifier.verify_batch(all_raw, classified_pairs)
        verified  = [e for e in events if e.verified]
        unverified = [e for e in events if not e.verified]
        log.info(f"[Verify] {len(verified)} verified  |  {len(unverified)} below threshold.")

        # ── Step 4: Weather enrichment (sample key waypoints) ─────────────
        # Weather is fetched per-route in the pipeline; here we just test one
        log.info("[Weather] Module ready — weather fetched per route on demand.")

        self._cache     = events
        self._last_run  = datetime.datetime.utcnow()

        log.info(f"[NVS] Done. {len(events)} total events, {len(verified)} verified.")
        return events

    def summary(self, events: List[ThreatEvent]) -> Dict:
        return {
            "total":      len(events),
            "verified":   sum(1 for e in events if e.verified),
            "by_type":    {t: sum(1 for e in events if e.threat_type == t)
                           for t in THREAT_LEXICON},
            "avg_confidence": round(
                sum(e.composite_confidence for e in events) / max(len(events), 1), 3),
            "high_severity": [
                {"title": e.title[:60], "region": e.region,
                 "type": e.threat_type, "severity": e.adj_severity,
                 "confidence": e.composite_confidence, "verified": e.verified}
                for e in sorted(events, key=lambda x: x.adj_severity, reverse=True)
                if e.adj_severity >= 5.0
            ][:10],
        }

    def to_threat_zones_format(self, events: List[ThreatEvent]) -> List[Dict]:
        """
        Convert verified ThreatEvents to the ThreatZone-compatible dict
        that cargo_risk_v3 can directly consume (replaces static THREAT_ZONES).
        """
        zones = []
        for e in events:
            if e.verified or e.composite_confidence >= 0.45:
                zones.append({
                    "name":        f"{e.source_name}: {e.title[:60]}",
                    "lat":         e.lat,
                    "lon":         e.lon,
                    "threat_type": e.threat_type,
                    "severity":    e.adj_severity,
                    "radius_km":   _severity_to_radius(e.adj_severity),
                    "confidence":  e.composite_confidence,
                    "verified":    e.verified,
                })
        return zones

def _severity_to_radius(sev: float) -> float:
    """Map severity score to influence radius in km."""
    if sev >= 9:   return 700
    elif sev >= 7: return 500
    elif sev >= 5: return 350
    elif sev >= 3: return 200
    return 100


# ══════════════════════════════════════════════════════════════════════════════
# 10.  STANDALONE TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "═"*60)
    print("  NEWS VERIFICATION MODULE — STANDALONE TEST")
    print("═"*60)

    nvs    = NewsVerificationSystem(use_html_scraper=False)
    events = nvs.run()
    summary = nvs.summary(events)

    print(f"\n  Total events:   {summary['total']}")
    print(f"  Verified:       {summary['verified']}")
    print(f"  Avg confidence: {summary['avg_confidence']}")
    print(f"  By type:        {summary['by_type']}")

    if summary["high_severity"]:
        print("\n  HIGH SEVERITY EVENTS:")
        for ev in summary["high_severity"]:
            flag = "✓" if ev["verified"] else "?"
            print(f"  [{flag}] [{ev['type']:11}] sev={ev['severity']:.1f} "
                  f"conf={ev['confidence']:.2f}  {ev['title']}")

    # Test weather
    print("\n  WEATHER TEST (Red Sea midpoint):")
    w = fetch_weather(15.0, 42.5)
    print(f"  Condition: {w.condition}  Wind: {w.wind_speed} m/s  "
          f"Wave: {w.wave_height} m  Risk: {w.risk_factor:.3f}  [{w.source}]")

    # Export threat zones
    zones = nvs.to_threat_zones_format(events)
    print(f"\n  Threat zones exported for pipeline: {len(zones)}")
    with open("/mnt/user-data/outputs/live_threat_zones.json", "w") as f:
        json.dump(zones, f, indent=2)
    print("  Saved → live_threat_zones.json")
