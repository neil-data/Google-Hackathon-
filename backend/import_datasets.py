import os
import sys
import pandas as pd
import io

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import SessionLocal, engine, Base
from models import AviationRoute, RoadNetwork, MaritimeVessel
import math

Base.metadata.create_all(bind=engine)

def sanitize_float(val):
    if pd.isna(val) or val == 'N/A' or val == '':
        return None
    try:
        return float(val)
    except:
        return None

def sanitize_int(val):
    if pd.isna(val) or val == 'N/A' or val == '':
        return None
    try:
        return int(float(val))
    except:
        return None

def sanitize_str(val):
    if pd.isna(val) or val == 'N/A':
        return None
    s = str(val).strip()
    return s if s else None


def import_aviation(file_path):
    print("Importing Aviation data...")
    df_raw = pd.read_excel(file_path)
    col = df_raw.columns[0]
    data = [col] + df_raw[col].tolist()
    csv_str = '\n'.join(str(x) for x in data)
    df = pd.read_csv(io.StringIO(csv_str))

    db = SessionLocal()
    inserted = 0
    try:
        for idx, row in df.iterrows():
            r = AviationRoute(
                aircraft_id=sanitize_str(row.get('Aircraft ID (ICAO 24-bit)')),
                flight_number=sanitize_str(row.get('Flight Number')),
                airline_name=sanitize_str(row.get('Airline Name')),
                airline_icao_code=sanitize_str(row.get('Airline ICAO Code')),
                aircraft_type=sanitize_str(row.get('Aircraft Type')),
                aircraft_registration=sanitize_str(row.get('Aircraft Registration')),
                timestamp_utc=sanitize_str(row.get('Timestamp (UTC)')),
                latitude=sanitize_float(row.get('Latitude')),
                longitude=sanitize_float(row.get('Longitude')),
                altitude_ft=sanitize_int(row.get('Altitude (ft)')),
                speed_kmh=sanitize_float(row.get('Speed (km/h)')),
                heading_deg=sanitize_float(row.get('Heading (°)')),
                flight_status=sanitize_str(row.get('Flight Status')),
                origin_airport_name=sanitize_str(row.get('Origin Airport Name')),
                origin_iata=sanitize_str(row.get('Origin IATA')),
                origin_icao=sanitize_str(row.get('Origin ICAO')),
                origin_latitude=sanitize_float(row.get('Origin Latitude')),
                origin_longitude=sanitize_float(row.get('Origin Longitude')),
                origin_country=sanitize_str(row.get('Origin Country')),
                departure_terminal=sanitize_str(row.get('Departure Terminal')),
                departure_gate=sanitize_str(row.get('Departure Gate')),
                scheduled_departure=sanitize_str(row.get('Scheduled Departure')),
                actual_departure=sanitize_str(row.get('Actual Departure')),
                destination_airport_name=sanitize_str(row.get('Destination Airport Name')),
                destination_iata=sanitize_str(row.get('Destination IATA')),
                destination_icao=sanitize_str(row.get('Destination ICAO')),
                destination_latitude=sanitize_float(row.get('Destination Latitude')),
                destination_longitude=sanitize_float(row.get('Destination Longitude')),
                destination_country=sanitize_str(row.get('Destination Country')),
                arrival_terminal=sanitize_str(row.get('Arrival Terminal')),
                arrival_gate=sanitize_str(row.get('Arrival Gate')),
                scheduled_arrival=sanitize_str(row.get('Scheduled Arrival')),
                eta=sanitize_str(row.get('ETA')),
                ata=sanitize_str(row.get('ATA')),
                route_id=sanitize_str(row.get('Route ID')),
                route_distance_km=sanitize_float(row.get('Route Distance (km)')),
                flight_duration_min=sanitize_float(row.get('Flight Duration (min)')),
                number_of_stops=sanitize_int(row.get('Number of Stops')),
                route_path=sanitize_str(row.get('Route Path')),
                passenger_capacity=sanitize_int(row.get('Passenger Capacity')),
                passengers_onboard=sanitize_int(row.get('Passengers Onboard')),
                load_factor_pct=sanitize_float(row.get('Load Factor (%)')),
                cargo_weight_kg=sanitize_float(row.get('Cargo Weight (kg)')),
                economy_class=sanitize_int(row.get('Economy Class')),
                business_class=sanitize_int(row.get('Business Class')),
                first_class=sanitize_int(row.get('First Class')),
                daily_arrivals_at_destination=sanitize_int(row.get('Daily Arrivals at Destination')),
                airport_traffic_level=sanitize_str(row.get('Airport Traffic Level')),
                runway_assigned=sanitize_str(row.get('Runway Assigned')),
                ground_handling_time_min=sanitize_float(row.get('Ground Handling Time (min)')),
                route_risk_level=sanitize_str(row.get('Route Risk Level')),
                risk_type=sanitize_str(row.get('Risk Type')),
                weather_condition=sanitize_str(row.get('Weather Condition')),
                delay_duration_min=sanitize_float(row.get('Delay Duration (min)')),
                cancellation_status=sanitize_str(row.get('Cancellation Status')),
                emergency_indicator=sanitize_str(row.get('Emergency Indicator'))
            )
            db.add(r)
            inserted += 1
        db.commit()
        print(f"  -> Inserted {inserted} aviation routes.")
    except Exception as e:
        db.rollback()
        print(f"Error importing aviation: {e}")
    finally:
        db.close()


def import_roads(file_path):
    print("Importing Roads data...")
    df = pd.read_excel(file_path)

    db = SessionLocal()
    inserted = 0
    try:
        for idx, row in df.iterrows():
            r = RoadNetwork(
                road_id=sanitize_str(row.get('Road_ID')),
                state_name=sanitize_str(row.get('State_Name')),
                road_name=sanitize_str(row.get('Road_Name')),
                road_type=sanitize_str(row.get('Road_Type')),
                route_start_location=sanitize_str(row.get('Route_Start_Location')),
                route_end_location=sanitize_str(row.get('Route_End_Location')),
                major_cities_covered=sanitize_str(row.get('Major_Cities_Covered')),
                districts_covered=sanitize_str(row.get('Districts_Covered')),
                region=sanitize_str(row.get('Region')),
                total_distance_km=sanitize_float(row.get('Total_Distance_km')),
                estimated_travel_time_hours=sanitize_float(row.get('Estimated_Travel_Time_hours')),
                average_truck_speed_kmph=sanitize_float(row.get('Average_Truck_Speed_kmph')),
                fuel_efficiency_km_per_litre=sanitize_float(row.get('Fuel_Efficiency_km_per_litre')),
                total_fuel_consumption_litres=sanitize_float(row.get('Total_Fuel_Consumption_litres')),
                fuel_price_per_litre=sanitize_float(row.get('Fuel_Price_per_Litre')),
                total_fuel_cost_inr=sanitize_float(row.get('Total_Fuel_Cost_INR')),
                road_condition=sanitize_str(row.get('Road_Condition')),
                toll_applicable=sanitize_str(row.get('Toll_Applicable')),
                category_alphabetical_index=sanitize_str(row.get('Category_Alphabetical_Index'))
            )
            db.add(r)
            inserted += 1
        db.commit()
        print(f"  -> Inserted {inserted} road networks.")
    except Exception as e:
        db.rollback()
        print(f"Error importing roads: {e}")
    finally:
        db.close()

def import_maritime(file_path):
    print("Importing Maritime data...")
    df_raw = pd.read_excel(file_path)
    header_str = df_raw.iloc[0, 0]
    lines = [header_str]
    for idx, row in df_raw.iloc[1:].iterrows():
        lines.append(str(row.iloc[0]))
    csv_str = '\n'.join(lines)
    df = pd.read_csv(io.StringIO(csv_str))

    db = SessionLocal()
    inserted = 0
    try:
        for idx, row in df.iterrows():
            r = MaritimeVessel(
                mmsi=sanitize_str(row.get('MMSI')),
                imo_number=sanitize_str(row.get('IMO Number')),
                ship_name=sanitize_str(row.get('Ship Name')),
                call_sign=sanitize_str(row.get('Call Sign')),
                flag_country=sanitize_str(row.get('Flag Country')),
                vessel_type=sanitize_str(row.get('Vessel Type')),
                length_m=sanitize_float(row.get('Length (m)')),
                width_m=sanitize_float(row.get('Width (m)')),
                deadweight_tonnage_dwt=sanitize_float(row.get('Deadweight Tonnage (DWT)')),
                timestamp_utc=sanitize_str(row.get('Timestamp (UTC)')),
                latitude=sanitize_float(row.get('Latitude')),
                longitude=sanitize_float(row.get('Longitude')),
                speed_over_ground_knots=sanitize_float(row.get('Speed Over Ground (knots)')),
                course_over_ground_deg=sanitize_float(row.get('Course Over Ground (°)')),
                heading_deg=sanitize_float(row.get('Heading (°)')),
                navigation_status=sanitize_str(row.get('Navigation Status')),
                origin_port=sanitize_str(row.get('Origin Port')),
                origin_latitude=sanitize_float(row.get('Origin Latitude')),
                origin_longitude=sanitize_float(row.get('Origin Longitude')),
                destination_port=sanitize_str(row.get('Destination Port')),
                destination_latitude=sanitize_float(row.get('Destination Latitude')),
                destination_longitude=sanitize_float(row.get('Destination Longitude')),
                route_distance_nm=sanitize_float(row.get('Route Distance (NM)')),
                route_status=sanitize_str(row.get('Route Status')),
                eta_utc=sanitize_str(row.get('ETA (UTC)')),
                ata_utc=sanitize_str(row.get('ATA (UTC)')),
                etd_utc=sanitize_str(row.get('ETD')),
                port_call_id=sanitize_str(row.get('Port Call ID')),
                berth_terminal=sanitize_str(row.get('Berth/Terminal')),
                cargo_type=sanitize_str(row.get('Cargo Type')),
                cargo_weight_tons=sanitize_float(row.get('Cargo Weight (tons)')),
                container_count_teu=sanitize_float(row.get('Container Count (TEU)')),
                hazard_class=sanitize_str(row.get('Hazard Class')),
                shipment_id=sanitize_str(row.get('Shipment ID')),
                shipper_name=sanitize_str(row.get('Shipper Name')),
                consignee_name=sanitize_str(row.get('Consignee Name')),
                risk_level=sanitize_str(row.get('Risk Level')),
                risk_type=sanitize_str(row.get('Risk Type')),
                weather_condition=sanitize_str(row.get('Weather Condition')),
                ais_signal_gap=sanitize_str(row.get('AIS Signal Gap')),
                geofence_violation=sanitize_str(row.get('Geofence Violation'))
            )
            db.add(r)
            inserted += 1
        db.commit()
        print(f"  -> Inserted {inserted} maritime vessels.")
    except Exception as e:
        db.rollback()
        print(f"Error importing maritime: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    aviation = r'c:\Users\yashv_p8oa5q\Desktop\chainguard\Global_Aviation_Route_Database.xlsx'
    road = r'c:\Users\yashv_p8oa5q\Desktop\chainguard\India_Road_Network_Database.xlsx'
    maritime = r'c:\Users\yashv_p8oa5q\Desktop\chainguard\Maritime_vessel_database.xlsx'
    import_aviation(aviation)
    import_roads(road)
    import_maritime(maritime)
