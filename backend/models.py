from datetime import datetime
from sqlalchemy import Column, String, Float, Integer, DateTime, ForeignKey, Boolean
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


class AviationRoute(Base):
    __tablename__ = "aviation_routes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    aircraft_id = Column(String)
    flight_number = Column(String)
    airline_name = Column(String)
    airline_icao_code = Column(String)
    aircraft_type = Column(String)
    aircraft_registration = Column(String)
    timestamp_utc = Column(String)
    latitude = Column(Float)
    longitude = Column(Float)
    altitude_ft = Column(Integer)
    speed_kmh = Column(Float)
    heading_deg = Column(Float)
    flight_status = Column(String)
    origin_airport_name = Column(String)
    origin_iata = Column(String)
    origin_icao = Column(String)
    origin_latitude = Column(Float)
    origin_longitude = Column(Float)
    origin_country = Column(String)
    departure_terminal = Column(String)
    departure_gate = Column(String)
    scheduled_departure = Column(String)
    actual_departure = Column(String)
    destination_airport_name = Column(String)
    destination_iata = Column(String)
    destination_icao = Column(String)
    destination_latitude = Column(Float)
    destination_longitude = Column(Float)
    destination_country = Column(String)
    arrival_terminal = Column(String)
    arrival_gate = Column(String)
    scheduled_arrival = Column(String)
    eta = Column(String)
    ata = Column(String)
    route_id = Column(String)
    route_distance_km = Column(Float)
    flight_duration_min = Column(Float)
    number_of_stops = Column(Integer)
    route_path = Column(String)
    passenger_capacity = Column(Integer)
    passengers_onboard = Column(Integer)
    load_factor_pct = Column(Float)
    cargo_weight_kg = Column(Float)
    economy_class = Column(Integer)
    business_class = Column(Integer)
    first_class = Column(Integer)
    daily_arrivals_at_destination = Column(Integer)
    airport_traffic_level = Column(String)
    runway_assigned = Column(String)
    ground_handling_time_min = Column(Float)
    route_risk_level = Column(String)
    risk_type = Column(String)
    weather_condition = Column(String)
    delay_duration_min = Column(Float)
    cancellation_status = Column(String)
    emergency_indicator = Column(String)


class RoadNetwork(Base):
    __tablename__ = "road_networks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    road_id = Column(String)
    state_name = Column(String)
    road_name = Column(String)
    road_type = Column(String)
    route_start_location = Column(String)
    route_end_location = Column(String)
    major_cities_covered = Column(String)
    districts_covered = Column(String)
    region = Column(String)
    total_distance_km = Column(Float)
    estimated_travel_time_hours = Column(Float)
    average_truck_speed_kmph = Column(Float)
    fuel_efficiency_km_per_litre = Column(Float)
    total_fuel_consumption_litres = Column(Float)
    fuel_price_per_litre = Column(Float)
    total_fuel_cost_inr = Column(Float)
    road_condition = Column(String)
    toll_applicable = Column(String)
    category_alphabetical_index = Column(String)


class MaritimeVessel(Base):
    __tablename__ = "maritime_vessels"

    id = Column(Integer, primary_key=True, autoincrement=True)
    mmsi = Column(String)
    imo_number = Column(String)
    ship_name = Column(String)
    call_sign = Column(String)
    flag_country = Column(String)
    vessel_type = Column(String)
    length_m = Column(Float)
    width_m = Column(Float)
    deadweight_tonnage_dwt = Column(Float)
    timestamp_utc = Column(String)
    latitude = Column(Float)
    longitude = Column(Float)
    speed_over_ground_knots = Column(Float)
    course_over_ground_deg = Column(Float)
    heading_deg = Column(Float)
    navigation_status = Column(String)
    origin_port = Column(String)
    origin_latitude = Column(Float)
    origin_longitude = Column(Float)
    destination_port = Column(String)
    destination_latitude = Column(Float)
    destination_longitude = Column(Float)
    route_distance_nm = Column(Float)
    route_status = Column(String)
    eta_utc = Column(String)
    ata_utc = Column(String)
    etd_utc = Column(String)
    port_call_id = Column(String)
    berth_terminal = Column(String)
    cargo_type = Column(String)
    cargo_weight_tons = Column(Float)
    container_count_teu = Column(Float)
    hazard_class = Column(String)
    shipment_id = Column(String)
    shipper_name = Column(String)
    consignee_name = Column(String)
    risk_level = Column(String)
    risk_type = Column(String)
    weather_condition = Column(String)
    ais_signal_gap = Column(String)
    geofence_violation = Column(String)