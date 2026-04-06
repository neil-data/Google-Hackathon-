from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List

from database import get_db
import models
import schemas

router = APIRouter(prefix="/aviation", tags=["aviation"])

@router.get("/", response_model=List[schemas.AviationRouteResponse])
def get_aviation_routes(db: Session = Depends(get_db)):
    return db.query(models.AviationRoute).all()
