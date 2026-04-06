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


class AviationRouteResponse(BaseModel):
    id: int
    aircraft_id: Optional[str]
    flight_number: Optional[str]
    airline_name: Optional[str]
    airline_icao_code: Optional[str]
    aircraft_type: Optional[str]
    aircraft_registration: Optional[str]
    timestamp_utc: Optional[str]
    latitude: Optional[float]
    longitude: Optional[float]
    altitude_ft: Optional[int]
    speed_kmh: Optional[float]
    heading_deg: Optional[float]
    flight_status: Optional[str]
    origin_airport_name: Optional[str]
    origin_iata: Optional[str]
    origin_icao: Optional[str]
    origin_latitude: Optional[float]
    origin_longitude: Optional[float]
    origin_country: Optional[str]
    departure_terminal: Optional[str]
    departure_gate: Optional[str]
    scheduled_departure: Optional[str]
    actual_departure: Optional[str]
    destination_airport_name: Optional[str]
    destination_iata: Optional[str]
    destination_icao: Optional[str]
    destination_latitude: Optional[float]
    destination_longitude: Optional[float]
    destination_country: Optional[str]
    arrival_terminal: Optional[str]
    arrival_gate: Optional[str]
    scheduled_arrival: Optional[str]
    eta: Optional[str]
    ata: Optional[str]
    route_id: Optional[str]
    route_distance_km: Optional[float]
    flight_duration_min: Optional[float]
    number_of_stops: Optional[int]
    route_path: Optional[str]
    passenger_capacity: Optional[int]
    passengers_onboard: Optional[int]
    load_factor_pct: Optional[float]
    cargo_weight_kg: Optional[float]
    economy_class: Optional[int]
    business_class: Optional[int]
    first_class: Optional[int]
    daily_arrivals_at_destination: Optional[int]
    airport_traffic_level: Optional[str]
    runway_assigned: Optional[str]
    ground_handling_time_min: Optional[float]
    route_risk_level: Optional[str]
    risk_type: Optional[str]
    weather_condition: Optional[str]
    delay_duration_min: Optional[float]
    cancellation_status: Optional[str]
    emergency_indicator: Optional[str]
    model_config = ConfigDict(from_attributes=True)


class RoadNetworkResponse(BaseModel):
    id: int
    road_id: Optional[str]
    state_name: Optional[str]
    road_name: Optional[str]
    road_type: Optional[str]
    route_start_location: Optional[str]
    route_end_location: Optional[str]
    major_cities_covered: Optional[str]
    districts_covered: Optional[str]
    region: Optional[str]
    total_distance_km: Optional[float]
    estimated_travel_time_hours: Optional[float]
    average_truck_speed_kmph: Optional[float]
    fuel_efficiency_km_per_litre: Optional[float]
    total_fuel_consumption_litres: Optional[float]
    fuel_price_per_litre: Optional[float]
    total_fuel_cost_inr: Optional[float]
    road_condition: Optional[str]
    toll_applicable: Optional[str]
    category_alphabetical_index: Optional[str]
    model_config = ConfigDict(from_attributes=True)


class MaritimeVesselResponse(BaseModel):
    id: int
    mmsi: Optional[str]
    imo_number: Optional[str]
    ship_name: Optional[str]
    call_sign: Optional[str]
    flag_country: Optional[str]
    vessel_type: Optional[str]
    length_m: Optional[float]
    width_m: Optional[float]
    deadweight_tonnage_dwt: Optional[float]
    timestamp_utc: Optional[str]
    latitude: Optional[float]
    longitude: Optional[float]
    speed_over_ground_knots: Optional[float]
    course_over_ground_deg: Optional[float]
    heading_deg: Optional[float]
    navigation_status: Optional[str]
    origin_port: Optional[str]
    origin_latitude: Optional[float]
    origin_longitude: Optional[float]
    destination_port: Optional[str]
    destination_latitude: Optional[float]
    destination_longitude: Optional[float]
    route_distance_nm: Optional[float]
    route_status: Optional[str]
    eta_utc: Optional[str]
    ata_utc: Optional[str]
    etd_utc: Optional[str]
    port_call_id: Optional[str]
    berth_terminal: Optional[str]
    cargo_type: Optional[str]
    cargo_weight_tons: Optional[float]
    container_count_teu: Optional[float]
    hazard_class: Optional[str]
    shipment_id: Optional[str]
    shipper_name: Optional[str]
    consignee_name: Optional[str]
    risk_level: Optional[str]
    risk_type: Optional[str]
    weather_condition: Optional[str]
    ais_signal_gap: Optional[str]
    geofence_violation: Optional[str]
    model_config = ConfigDict(from_attributes=True)