import os
import requests
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Supabase credentials
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# EIA credentials
EIA_API_KEY = os.getenv("EIA_API_KEY")
EIA_API_URL = "https://api.eia.gov/v2/electricity/rto/daily-region-data/data/"

EIA_API_PARAMS = {
    "frequency": "daily",
    "data[0]": "value",
    "facets[respondent][]": "TVA",
    "facets[timezone][]": "Central",
    "start": "2019-01-01",
    "end": "2024-01-01",
    "sort[0][column]": "period",
    "sort[0][direction]": "desc",
    "offset": 0,
    "length": 7500,
    "api_key": EIA_API_KEY,
}


# Function to fetch data
def fetch_data():
    response = requests.get(EIA_API_URL, params=EIA_API_PARAMS)
    if response.status_code == 200:
        data = response.json()
        return data.get("response", {}).get("data", [])  # Extract 'data' field
    else:
        raise Exception(f"Failed to fetch data: {response.status_code} - {response.text}")


# Function to insert data into Supabase
def insert_data(data):
    for entry in data:
        try:
            # Insert data into Supabase directly
            response = supabase.table("daily_power_data").insert({
                "period": entry["period"],
                "respondent": entry["respondent"],
                "respondent_name": entry["respondent-name"],
                "type": entry["type"],
                "type_name": entry["type-name"],
                "value": float(entry["value"]),
                "value_units": entry["value-units"],
            }).execute()
            
            # Simply print response for each entry
            print(f"Inserted: {entry['period']}")
        except Exception as e:
            print(f"Error inserting data: {e}")

# Main function
def main():
    try:
        print("Starting process...")
        # Fetch the data
        data = fetch_data()
        print(f"Fetched {len(data)} records.")
        # Insert the data into Supabase
        insert_data(data)
        print("Data has been successfully added to Supabase!")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
