from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List

from database import get_db
import models
import schemas

router = APIRouter(prefix="/roads", tags=["roads"])

@router.get("/", response_model=List[schemas.RoadNetworkResponse])
def get_road_networks(db: Session = Depends(get_db)):
    return db.query(models.RoadNetwork).all()
