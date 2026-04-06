from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List

from database import get_db
import models
import schemas

router = APIRouter(prefix="/maritime", tags=["maritime"])

@router.get("/", response_model=List[schemas.MaritimeVesselResponse])
def get_maritime_vessels(db: Session = Depends(get_db)):
    return db.query(models.MaritimeVessel).all()
