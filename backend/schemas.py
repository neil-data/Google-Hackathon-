from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


class RiskScoreCreate(BaseModel):
    shipment_id: str = Field(..., description="ID of the shipment being scored")
    score: float = Field(..., ge=0.0, le=100.0, description="Risk score between 0 and 100")
    threat_type: str = Field(..., description="Category of threat: war/piracy/sanctions/natural/other")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence between 0 and 1")
    route_taken: Optional[str] = Field(None, description="JSON-encoded list of waypoints")


class RiskScoreResponse(BaseModel):
    id: int
    shipment_id: str
    score: float
    threat_type: str
    confidence: float
    route_taken: Optional[str] = None
    scored_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ShipmentCreate(BaseModel):
    id: str = Field(..., description="Unique shipment ID, e.g. 'SHP-001'")
    origin_lat: float = Field(..., description="Origin latitude")
    origin_lon: float = Field(..., description="Origin longitude")
    dest_lat: float = Field(..., description="Destination latitude")
    dest_lon: float = Field(..., description="Destination longitude")
    cargo_type: Optional[str] = Field(None, description="Type of cargo being transported")
    status: Optional[str] = Field("in_transit", description="Shipment status")


class ShipmentResponse(BaseModel):
    id: str
    origin_lat: float
    origin_lon: float
    dest_lat: float
    dest_lon: float
    cargo_type: Optional[str] = None
    status: str
    created_at: datetime
    risk_scores: List[RiskScoreResponse] = []

    model_config = ConfigDict(from_attributes=True)