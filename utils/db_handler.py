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
        demand_query = f'''
            select period, value
            from daily_power_data
            where type_name = 'Demand'
            and period between '{start_date}' and '{end_date}'
            order by period
        '''
        # Keep exact execution style
        demand_data = self.supabase.table('daily_power_data') \
            .select('period, value') \
            .filter('type_name', 'eq', 'Demand') \
            .filter('period', 'gte', start_date) \
            .filter('period', 'lte', end_date) \
            .order('period') \
            .execute()

        df = pd.DataFrame(demand_data.data)
        if not df.empty:
            df['period'] = pd.to_datetime(df['period'])
            df = df.sort_values('period')
        return df

    def get_weather_data(self, start_date, end_date):
        weather_query = f'''
            select date, regions
            from weather_data
            where date between '{start_date}' and '{end_date}'
            order by date
        '''
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