# utils/db_handler.py
import os
from dotenv import load_dotenv
from supabase import create_client
import pandas as pd

load_dotenv()

class DatabaseHandler:
    def __init__(self):
        self.supabase = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_KEY")
        )

    def get_demand_data(self, start_date, end_date):
        # First get demand data
        demand_data = self.supabase.table('daily_power_data') \
            .select('period, type_name, value') \
            .or_('type_name.eq.Demand,type_name.eq.Day-ahead demand forecast') \
            .filter('period', 'gte', start_date) \
            .filter('period', 'lte', end_date) \
            .order('period') \
            .execute()

        df = pd.DataFrame(demand_data.data)
        if not df.empty:
            df['period'] = pd.to_datetime(df['period'])
            # Pivot the data to create separate columns for demand and forecast
            df = df.pivot(index='period', columns='type_name', values='value').reset_index()
            df.columns.name = None  # Remove the columns name
            # Rename columns for clarity
            df = df.rename(columns={
                'Demand': 'actual_demand',
                'Day-ahead demand forecast': 'day_ahead_forecast'
            })
            df = df.sort_values('period')
        return df

    def get_weather_data(self, start_date, end_date):
        # Keep exact execution style
        weather_data = self.supabase.table('weather_data') \
            .select('date, regions') \
            .filter('date', 'gte', start_date) \
            .filter('date', 'lte', end_date) \
            .order('date') \
            .execute()

        # Process weather data from JSONB regions
        weather_rows = []
        for row in weather_data.data:
            date = row['date']
            regions = row['regions']

            # Collect temperature data
            high_temps = []
            low_temps = []

            for region in regions.values():
                if region['high_temp'] is not None:
                    high_temps.append(region['high_temp'])
                if region['low_temp'] is not None:
                    low_temps.append(region['low_temp'])

            if high_temps and low_temps:
                weather_rows.append({
                    'date': date,
                    'avg_high': sum(high_temps) / len(high_temps),
                    'avg_low': sum(low_temps) / len(low_temps)
                })

        df = pd.DataFrame(weather_rows)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
        return df

    def print_data_info(self, start_date, end_date):
        demand_data = self.get_demand_data(start_date, end_date)
        weather_data = self.get_weather_data(start_date, end_date)
        
        print(f"Found {len(demand_data)} demand records")
        print(f"Found {len(weather_data)} weather records")