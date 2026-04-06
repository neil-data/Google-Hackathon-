from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from database import get_db
import models
import schemas

router = APIRouter(
    prefix="/shipments",
    tags=["shipments"],
)


@router.post(
    "/",
    response_model=schemas.ShipmentResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_shipment(
    payload: schemas.ShipmentCreate,
    db: Session = Depends(get_db),
):
    existing = db.query(models.Shipment).filter(models.Shipment.id == payload.id).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Shipment with id '{payload.id}' already exists.",
        )

    shipment = models.Shipment(
        id=payload.id,
        origin_lat=payload.origin_lat,
        origin_lon=payload.origin_lon,
        dest_lat=payload.dest_lat,
        dest_lon=payload.dest_lon,
        cargo_type=payload.cargo_type,
        status=payload.status or "in_transit",
    )
    db.add(shipment)
    db.commit()
    db.refresh(shipment)
    return shipment


@router.get(
    "/",
    response_model=List[schemas.ShipmentResponse],
)
def list_shipments(db: Session = Depends(get_db)):
    return db.query(models.Shipment).all()


@router.get(
    "/{shipment_id}",
    response_model=schemas.ShipmentResponse,
)
def get_shipment(shipment_id: str, db: Session = Depends(get_db)):
    shipment = db.query(models.Shipment).filter(models.Shipment.id == shipment_id).first()
    if not shipment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Shipment '{shipment_id}' not found.",
        )
    return shipment


@router.delete(
    "/{shipment_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_shipment(shipment_id: str, db: Session = Depends(get_db)):
    shipment = db.query(models.Shipment).filter(models.Shipment.id == shipment_id).first()
    if not shipment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Shipment '{shipment_id}' not found.",
        )
    db.delete(shipment)
    db.commit()