from utils.db_handler import DatabaseHandler
import matplotlib.pyplot as plt
import pandas as pd

# Initialize database handler
db = DatabaseHandler()

# Set date range
START_DATE = '2020-07-29'
END_DATE = '2023-12-31'

# Get data
demand_df = db.get_demand_data(START_DATE, END_DATE)
weather_df = db.get_weather_data(START_DATE, END_DATE)

print(f"Found {len(demand_df)} demand records")
print(f"Found {len(weather_df)} weather records")

# Process demand data into a DataFrame
if not demand_df.empty:
    print("Demand date range:", demand_df['period'].min(), "to", demand_df['period'].max())
else:
    print("No demand data available for 2020-2024")

# Process weather data
if not weather_df.empty:
    print("Weather date range:", weather_df['date'].min(), "to", weather_df['date'].max())
else:
    print("No weather data available for 2020-2024")

# Create plots for demand and weather data
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10))

# Plot demand data

ax1.plot(demand_df['period'], demand_df['value'], color='blue', label='Demand')
ax1.set_title('TVA Demand (2020-2024)')
ax1.set_xlabel('Date')
ax1.set_ylabel('Megawatthours')
ax1.grid(True)
ax1.tick_params(axis='x', rotation=1)
ax1.legend()


# Plot weather data
ax2.plot(weather_df['date'], weather_df['avg_high'], color='red', label='Average High Temp')
ax2.plot(weather_df['date'], weather_df['avg_low'], color='blue', label='Average Low Temp')
ax2.set_title('Regional Average Temperatures (2020-2024)')
ax2.set_xlabel('Date')
ax2.set_ylabel('Temperature (°F)')
ax2.grid(True)
ax2.tick_params(axis='x', rotation=1)
ax2.legend()


# Overlay graph of normalized demand and max temperature
# Merge dataframes on closest matching dates
merged_df = pd.merge_asof(demand_df.rename(columns={'period': 'date'}), weather_df, on='date')

# Normalize the values between 0 and 1
merged_df['normalized_demand'] = (merged_df['value'] - merged_df['value'].min()) / (merged_df['value'].max() - merged_df['value'].min())
merged_df['normalized_temp'] = (merged_df['avg_high'] - merged_df['avg_high'].min()) / (merged_df['avg_high'].max() - merged_df['avg_high'].min())

# Plot the normalized values
ax3.plot(merged_df['date'], merged_df['normalized_demand'], color='blue', label='Normalized Demand')
ax3.plot(merged_df['date'], merged_df['normalized_temp'], color='red', label='Normalized Max Temp')

ax3.set_title('TVA Demand vs Max Temperature (Normalized, 2020-2024)')
ax3.set_xlabel('Date')
ax3.set_ylabel('Normalized Value')
ax3.grid(True)
ax3.tick_params(axis='x', rotation=1)
ax3.legend()


plt.tight_layout()
plt.show()