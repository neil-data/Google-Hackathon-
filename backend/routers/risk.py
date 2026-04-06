from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from database import get_db
import models
import schemas
from services.risk_engine import compute_risk

router = APIRouter(
    prefix="/risk",
    tags=["risk"],
)


@router.post(
    "/score",
    response_model=schemas.RiskScoreResponse,
    status_code=status.HTTP_201_CREATED,
)
def score_shipment(
    shipment_id: str,
    db: Session = Depends(get_db),
):
    shipment = db.query(models.Shipment).filter(models.Shipment.id == shipment_id).first()
    if not shipment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Shipment '{shipment_id}' not found. Cannot score a non-existent shipment.",
        )

    result = compute_risk(shipment_id)

    risk_record = models.RiskScore(
        shipment_id=shipment_id,
        score=result["score"],
        threat_type=result["threat_type"],
        confidence=result["confidence"],
        route_taken=result["route_taken"],
    )
    db.add(risk_record)
    db.commit()
    db.refresh(risk_record)
    return risk_record


@router.get(
    "/scores/{shipment_id}",
    response_model=List[schemas.RiskScoreResponse],
)
def get_scores(shipment_id: str, db: Session = Depends(get_db)):
    shipment = db.query(models.Shipment).filter(models.Shipment.id == shipment_id).first()
    if not shipment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Shipment '{shipment_id}' not found.",
        )

    scores = (
        db.query(models.RiskScore)
        .filter(models.RiskScore.shipment_id == shipment_id)
        .order_by(models.RiskScore.scored_at.asc())
        .all()
    )
    return scores