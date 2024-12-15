import os
import requests
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

API_KEY = os.getenv("NOAA_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
BASE_URL = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"

STATIONS = {
    "west_tn": {
        "id": "GHCND:USW00013893",
        "name": "Memphis International Airport"
    },
    "central_tn": {
        "id": "GHCND:USW00013897",
        "name": "Nashville International Airport"
    },
    "east_tn": {
        "id": "GHCND:USW00013891",
        "name": "Knoxville McGhee Tyson Airport"
    },
    "north_al": {
        "id": "GHCND:USW00003856",
        "name": "Huntsville International Airport"
    },
    "north_ms": {
        "id": "GHCND:USW00093862",
        "name": "Tupelo Regional Airport"
    }
}

def fetch_weather_data(station_id, date, datatype):
    time.sleep(0.25)
    
    params = {
        "datasetid": "GHCND",
        "stationid": station_id,
        "startdate": date,
        "enddate": date,
        "datatypeid": datatype,
        "limit": 1000
    }
    
    response = requests.get(
        BASE_URL,
        headers={"token": API_KEY},
        params=params
    )
    
    if response.status_code == 200:
        return response.json().get('results', [])
    print(f"Error {response.status_code}: {response.text}")
    return []

def convert_temp(celsius):
    return round((celsius / 10) * 9/5 + 32)

def convert_precip(mm):
    return round(mm / 25.4, 2)

def get_daily_data(date):
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    regions = {}
    
    print(f"Collecting data for {date}")
    
    for region, station_info in STATIONS.items():
        station_id = station_info["id"]
        print(f"Processing {station_info['name']}...")
        
        high_temp_data = fetch_weather_data(station_id, date, "TMAX")
        low_temp_data = fetch_weather_data(station_id, date, "TMIN")
        precip_data = fetch_weather_data(station_id, date, "PRCP")
        
        regions[region] = {
            "station_name": station_info["name"],
            "station_id": station_id,
            "high_temp": convert_temp(high_temp_data[0]['value']) if high_temp_data else None,
            "low_temp": convert_temp(low_temp_data[0]['value']) if low_temp_data else None,
            "precipitation": convert_precip(precip_data[0]['value']) if precip_data else 0
        }
    
    try:
        data = {
            "date": date,
            "regions": regions
        }
        result = supabase.table("weather_data").upsert(data).execute()
        print(f"Saved data for {date}")
        print("Data:", regions)
    except Exception as e:
        print(f"Error saving data: {e}")

def main():
    start_date = datetime.strptime("2022-05-09", "%Y-%m-%d")
    end_date = datetime.strptime("2024-01-01", "%Y-%m-%d")
    current_date = start_date
    
    while current_date <= end_date:
        get_daily_data(current_date.strftime("%Y-%m-%d"))
        current_date += timedelta(days=1)

if __name__ == "__main__":
    main()