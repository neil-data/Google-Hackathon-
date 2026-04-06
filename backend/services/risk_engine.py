import random


def compute_risk(shipment_id: str) -> dict:
    threat_types = ["war", "piracy", "sanctions", "natural", "other"]

    score = round(random.uniform(0.0, 100.0), 2)
    threat_type = random.choice(threat_types)
    confidence = round(random.uniform(0.5, 1.0), 4)
    route_taken = None

    return {
        "score": score,
        "threat_type": threat_type,
        "confidence": confidence,
        "route_taken": route_taken,
    }