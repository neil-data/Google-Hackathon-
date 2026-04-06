import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime
from database import SessionLocal, engine, Base
from models import Shipment, RiskScore

MOCK_SHIPMENTS = [
    {"id": "SHP-001", "origin": [31.2, 121.5], "dest": [51.5, -0.1],   "risk": 91, "threat_type": "war"},
    {"id": "SHP-002", "origin": [1.3, 103.8],  "dest": [40.7, -74.0],  "risk": 72, "threat_type": "piracy"},
    {"id": "SHP-003", "origin": [22.3, 114.2], "dest": [48.9, 2.3],    "risk": 45, "threat_type": "sanctions"},
    {"id": "SHP-004", "origin": [35.7, 139.7], "dest": [52.5, 13.4],   "risk": 20, "threat_type": "natural"},
    {"id": "SHP-005", "origin": [37.6, 126.9], "dest": [41.9, 12.5],   "risk": 85, "threat_type": "war"},
    {"id": "SHP-006", "origin": [13.8, 100.5], "dest": [53.3, -6.3],   "risk": 33, "threat_type": "other"},
    {"id": "SHP-007", "origin": [23.1, 72.6],  "dest": [40.4, -3.7],   "risk": 67, "threat_type": "piracy"},
    {"id": "SHP-008", "origin": [1.3, 103.8],  "dest": [34.1, -118.2], "risk": 55, "threat_type": "sanctions"},
    {"id": "SHP-009", "origin": [30.6, 114.3], "dest": [43.7, -79.4],  "risk": 78, "threat_type": "war"},
    {"id": "SHP-010", "origin": [22.3, 114.2], "dest": [37.8, -122.4], "risk": 12, "threat_type": "natural"},
]


def seed():
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    inserted = 0
    skipped = 0

    try:
        for entry in MOCK_SHIPMENTS:
            ship_id = entry["id"]

            existing = db.query(Shipment).filter(Shipment.id == ship_id).first()
            if existing:
                print(f"  [SKIP] {ship_id} — already exists.")
                skipped += 1
                continue

            shipment = Shipment(
                id=ship_id,
                origin_lat=entry["origin"][0],
                origin_lon=entry["origin"][1],
                dest_lat=entry["dest"][0],
                dest_lon=entry["dest"][1],
                cargo_type="general",
                status="in_transit",
                created_at=datetime.utcnow(),
            )
            db.add(shipment)
            db.flush()

            confidence = round(1.0 - (entry["risk"] / 200.0), 2)
            risk_score = RiskScore(
                shipment_id=ship_id,
                score=float(entry["risk"]),
                threat_type=entry["threat_type"],
                confidence=confidence,
                route_taken=None,
                scored_at=datetime.utcnow(),
            )
            db.add(risk_score)

            print(f"  [INSERT] {ship_id} — risk={entry['risk']}, threat={entry['threat_type']}")
            inserted += 1

        db.commit()
        print(f"\nSeed complete. Inserted: {inserted} | Skipped: {skipped}")

    except Exception as exc:
        db.rollback()
        print(f"\n[ERROR] Seed failed: {exc}")
        raise

    finally:
        db.close()


if __name__ == "__main__":
    print("ChainGuard — Seeding database...\n")
    seed()