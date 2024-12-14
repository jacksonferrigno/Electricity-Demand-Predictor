from utils.db_handler import DatabaseHandler
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# Configuration and data loading
START = '2020-07-29'
END = '2023-12-31'

# Load data
db = DatabaseHandler()
demand_df = db.get_demand_data(START, END)
weather_df = db.get_weather_data(START, END)
merged_df = pd.merge_asof(
    demand_df.rename(columns={'period': 'date'}),
    weather_df,
    on='date'
)

# Prepare features and target
X = merged_df[['avg_high']].values
y = merged_df['value'].values

# Create and fit polynomial regression model
poly_model = make_pipeline(
    PolynomialFeatures(2),
    LinearRegression()
)
poly_model.fit(X, y)

# Generate smooth points for plotting
X_smooth = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
y_poly_smooth = poly_model.predict(X_smooth)

# Calculate metrics
y_pred = poly_model.predict(X)
r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))

# Create plot
plt.figure(figsize=(12, 6))

# Plot data points and polynomial regression line
plt.scatter(X, y, alpha=0.5, label='Actual Data', color='blue', s=20)
plt.plot(X_smooth, y_poly_smooth, color='red', 
         label='Polynomial Regression', linewidth=2)

# Add labels and title
plt.xlabel('Average Maximum Temperature (°F)')
plt.ylabel('Demand (Megawatthours)')
plt.title('TVA Demand vs Temperature with Polynomial Regression')
plt.legend()
plt.grid(True, alpha=0.3)

# Add metrics to plot
metrics_text = (
    f'Polynomial Regression:\n'
    f'R² = {r2:.3f}\n'
    f'RMSE = {rmse:.2f} MWh'
)

plt.text(0.02, 0.98, metrics_text,
         transform=plt.gca().transAxes,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.show()