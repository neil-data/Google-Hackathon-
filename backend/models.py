from datetime import datetime
from sqlalchemy import Column, String, Float, Integer, DateTime, ForeignKey
from sqlalchemy.orm import relationship

from database import Base


class Shipment(Base):
    __tablename__ = "shipments"

    id = Column(String, primary_key=True, index=True)
    origin_lat = Column(Float, nullable=False)
    origin_lon = Column(Float, nullable=False)
    dest_lat = Column(Float, nullable=False)
    dest_lon = Column(Float, nullable=False)
    cargo_type = Column(String, nullable=True)
    status = Column(String, default="in_transit")
    created_at = Column(DateTime, default=datetime.utcnow)

    risk_scores = relationship(
        "RiskScore",
        back_populates="shipment",
        cascade="all, delete-orphan",
    )


class RiskScore(Base):
    __tablename__ = "risk_scores"

    id = Column(Integer, primary_key=True, autoincrement=True)
    shipment_id = Column(String, ForeignKey("shipments.id"), nullable=False, index=True)
    score = Column(Float, nullable=False)
    threat_type = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    route_taken = Column(String, nullable=True)
    scored_at = Column(DateTime, default=datetime.utcnow)

    shipment = relationship("Shipment", back_populates="risk_scores")