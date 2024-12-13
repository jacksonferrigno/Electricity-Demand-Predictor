import os
from dotenv import load_dotenv
from supabase import create_client
import matplotlib.pyplot as plt
import pandas as pd

# Load environment variables
load_dotenv()

# Initialize Supabase client
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

# Fetch demand data for 2021
demand_query = '''
    select period, value
    from daily_power_data
    where type_name = 'Demand'
    and period between '2021-01-01' and '2021-12-31'
    order by period
'''
demand_data = supabase.table('daily_power_data').select('period, value').filter('type_name', 'eq', 'Demand').filter('period', 'gte', '2021-01-01').filter('period', 'lte', '2021-12-31').order('period').execute()

# Fetch weather data for 2021
weather_query = '''
    select date, regions
    from weather_data
    where date between '2021-01-01' and '2021-12-31'
    order by date
'''
weather_data = supabase.table('weather_data').select('date, regions').filter('date', 'gte', '2021-01-01').filter('date', 'lte', '2021-12-31').order('date').execute()

print(f"Found {len(demand_data.data)} demand records")
print(f"Found {len(weather_data.data)} weather records")

# Process demand data into a DataFrame
demand_df = pd.DataFrame(demand_data.data)
if not demand_df.empty:
    demand_df['period'] = pd.to_datetime(demand_df['period'])
    demand_df = demand_df.sort_values('period')
    print("Demand date range:", demand_df['period'].min(), "to", demand_df['period'].max())
else:
    print("No demand data available for 2021")

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

weather_df = pd.DataFrame(weather_rows)
if not weather_df.empty:
    weather_df['date'] = pd.to_datetime(weather_df['date'])
    weather_df = weather_df.sort_values('date')
    print("Weather date range:", weather_df['date'].min(), "to", weather_df['date'].max())
else:
    print("No weather data available for 2021")

# Create plots for demand and weather data
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10))

# Plot demand data
if not demand_df.empty:
    ax1.plot(demand_df['period'], demand_df['value'], color='blue', label='Demand')
    ax1.set_title('TVA Demand (2021)')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Megawatthours')
    ax1.grid(True)
    ax1.tick_params(axis='x', rotation=1)
    ax1.legend()
else:
    print("No demand data to plot.")

# Plot weather data
if not weather_df.empty:
    ax2.plot(weather_df['date'], weather_df['avg_high'], color='red', label='Average High Temp')
    ax2.plot(weather_df['date'], weather_df['avg_low'], color='blue', label='Average Low Temp')
    ax2.set_title('Regional Average Temperatures (2021)')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Temperature (°F)')
    ax2.grid(True)
    ax2.tick_params(axis='x', rotation=1)
    ax2.legend()
else:
    print("No weather data to plot.")

# Overlay graph of normalized demand and max temperature
if not demand_df.empty and not weather_df.empty:
    # Merge dataframes on closest matching dates
    merged_df = pd.merge_asof(demand_df.rename(columns={'period': 'date'}), weather_df, on='date')
    
    # Normalize the values between 0 and 1
    merged_df['normalized_demand'] = (merged_df['value'] - merged_df['value'].min()) / (merged_df['value'].max() - merged_df['value'].min())
    merged_df['normalized_temp'] = (merged_df['avg_high'] - merged_df['avg_high'].min()) / (merged_df['avg_high'].max() - merged_df['avg_high'].min())
    
    # Plot the normalized values
    ax3.plot(merged_df['date'], merged_df['normalized_demand'], color='blue', label='Normalized Demand')
    ax3.plot(merged_df['date'], merged_df['normalized_temp'], color='red', label='Normalized Max Temp')
    
    ax3.set_title('TVA Demand vs Max Temperature (Normalized, 2021)')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Normalized Value')
    ax3.grid(True)
    ax3.tick_params(axis='x', rotation=1)
    ax3.legend()
else:
    print("Not enough data to plot overlay graph.")
    
    
    
plt.tight_layout()
plt.show()
