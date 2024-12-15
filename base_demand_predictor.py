# base_demand_predictor.py

from utils.db_handler import DatabaseHandler
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class DemandPredictor:
    def __init__(self, start_date, end_date, n_clusters=3):
        self.start_date = start_date
        self.end_date = end_date
        self.n_clusters = n_clusters
        self.data = None
        self.X = None
        self.y = None
        self.poly_model = None
        self.X_smooth = None
        self.y_smooth = None
        self.metrics = None
        self.clusters = None
        self.centers = None
        self.cluster_stats = None

    def load_data(self):
        """Load and merge demand and weather data."""
        db = DatabaseHandler()
        demand_df = db.get_demand_data(self.start_date, self.end_date)
        weather_df = db.get_weather_data(self.start_date, self.end_date)
        
        self.data = pd.merge_asof(
            demand_df.rename(columns={'period': 'date'}),
            weather_df,
            on='date'
        )
        
        self.X = self.data[['avg_high']].values
        self.y = self.data['value'].values
        return self.data

    def fit_polynomial_regression(self):
        """Perform polynomial regression analysis."""
        self.poly_model = make_pipeline(
            PolynomialFeatures(2),
            LinearRegression()
        )
        self.poly_model.fit(self.X, self.y)
        
        self.X_smooth = np.linspace(self.X.min(), self.X.max(), 300).reshape(-1, 1)
        self.y_smooth = self.poly_model.predict(self.X_smooth)
        
        y_pred = self.poly_model.predict(self.X)
        self.metrics = {
            'r2': r2_score(self.y, y_pred),
            'rmse': np.sqrt(mean_squared_error(self.y, y_pred))
        }

    def fit_clusters(self):
        """Perform clustering analysis on temperature and demand."""
        data_for_clustering = np.column_stack([self.X, self.y])
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_for_clustering)
        
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.clusters = kmeans.fit_predict(data_scaled)
        
        centers_scaled = kmeans.cluster_centers_
        self.centers = scaler.inverse_transform(centers_scaled)

    def analyze_clusters(self):
        """Analyze characteristics of each cluster."""
        data = pd.DataFrame({
            'temperature': self.X.flatten(),
            'demand': self.y,
            'cluster': self.clusters
        })
        
        cluster_analysis = []
        for i in range(len(self.centers)):
            cluster_data = data[data['cluster'] == i]
            analysis = {
                'cluster_number': i + 1,
                'size': len(cluster_data),
                'avg_temp': cluster_data['temperature'].mean(),
                'avg_demand': cluster_data['demand'].mean(),
                'min_demand': cluster_data['demand'].min(),
                'max_demand': cluster_data['demand'].max()
            }
            cluster_analysis.append(analysis)
        
        self.cluster_stats = pd.DataFrame(cluster_analysis)
        return self.cluster_stats

    def plot_analysis(self):
        """Create visualization of regression and clustering results."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Plot 1: Polynomial Regression
        ax1.scatter(self.X, self.y, alpha=0.5, label='Actual Data', color='blue', s=20)
        ax1.plot(self.X_smooth, self.y_smooth, color='red', label='Polynomial Regression', linewidth=2)
        
        ax1.set_xlabel('Average Maximum Temperature (°F)')
        ax1.set_ylabel('Demand (Megawatthours)')
        ax1.set_title('TVA Demand vs Temperature: Polynomial Regression')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add metrics to regression plot
        metrics_text = (
            f'Polynomial Regression:\n'
            f'R² = {self.metrics["r2"]:.3f}\n'
            f'RMSE = {self.metrics["rmse"]:.2f} MWh'
        )
        ax1.text(0.02, 0.98, metrics_text,
                 transform=ax1.transAxes,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot 2: Clustering Results
        cluster_colors = ['blue', 'green', 'red']
        for i in range(len(self.centers)):
            mask = self.clusters == i
            ax2.scatter(self.X[mask], self.y[mask], c=cluster_colors[i], 
                       label=f'Cluster {i+1}', alpha=0.5, s=20)
        
        ax2.scatter(self.centers[:, 0], self.centers[:, 1], c='black', marker='x', 
                    s=200, linewidths=3, label='Cluster Centers')
        
        ax2.set_xlabel('Average Maximum Temperature (°F)')
        ax2.set_ylabel('Demand (Megawatthours)')
        ax2.set_title('TVA Demand vs Temperature: Demand Clusters')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

    def fit(self):
        """Fit all models and perform analysis."""
        self.load_data()
        self.fit_polynomial_regression()
        self.fit_clusters()
        self.analyze_clusters()
        return self

    def get_metrics(self):
        """Get all analysis metrics."""
        if self.metrics is None:
            raise ValueError("Models not fitted yet. Call fit() first.")
        return {
            'regression_metrics': self.metrics,
            'cluster_stats': self.cluster_stats
        }