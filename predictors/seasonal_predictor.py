from predictors.base_demand_predictor import DemandPredictor
import pandas as pd
import matplotlib.pyplot as plt

class SeasonalPredictor(DemandPredictor):
    def __init__(self, start_date, end_date, n_clusters=3):
        super().__init__(start_date, end_date, n_clusters)
        self.seasonal_data = {}
        self.seasonal_results = {}
        
    def load_data(self):
        """Override load data func to include seasonal info"""
        super().load_data()
        
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data['season'] = self.data['date'].dt.month.map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
        
        for season in ['Winter', 'Spring', 'Summer', 'Fall']:
            season_mask = self.data['season'] == season
            self.seasonal_data[season] = self.data[season_mask]
            
        return self.data
    
    def fit_season(self, season):
        """Fit model for a specific season"""
        season_data = self.seasonal_data[season]
        
        # Update instance with seasonal data
        self.data = season_data
        self.X = season_data[['avg_high']].values
        self.y = season_data['value'].values
        
        # Use parent's fit method
        super().fit()
        
        # Store results
        self.seasonal_results[season] = {
            'data': season_data,
            'metrics': self.metrics,
            'cluster_stats': self.cluster_stats
        }
        
        return self
    
    def fit_all_seasons(self):
        """Fit model for all seasons"""
        if not self.seasonal_data:
            self.load_data()
        print("Fitting model for all seasons...")
        for season in ['Winter', 'Spring', 'Summer', 'Fall']:
            print(f"Processing {season}...")
            self.fit_season(season)
        return self
    
    def plot_analysis(self, season=None):
        """Override plot_analysis to handle seasonal plotting"""
        if season:
            # Plot single season
            self.data = self.seasonal_data[season]
            self.X = self.data[['avg_high']].values
            self.y = self.data['value'].values
            
            self.fit_polynomial_regression()
            self.fit_clusters()
            
            fig = super().plot_analysis()
            plt.suptitle(f'{season} Analysis', y=1.02, size=16)
            return fig
            
        # Plot all seasons in a grid
        fig, axes = plt.subplots(4, 2, figsize=(20, 32))
        for idx, season in enumerate(['Winter', 'Spring', 'Summer', 'Fall']):
            self.data = self.seasonal_data[season]
            self.X = self.data[['avg_high']].values
            self.y = self.data['value'].values
            
            self.fit_polynomial_regression()
            self.fit_clusters()
            
            self._plot_regression(axes[idx, 0])
            self._plot_clustering(axes[idx, 1])
            
            axes[idx, 0].set_title(f'{season} - Polynomial Regression')
            axes[idx, 1].set_title(f'{season} - Demand Clusters')
        
        plt.tight_layout()
        return fig
    """     
    def get_metrics(self):
        
        if self.metrics is None:
            raise ValueError("Models not fitted yet. Call fit() first.")
        return {
            season: {
                'regression_metrics': self.metrics,
                'cluster_stats': self.cluster_stats
            }[season]
            for season in self.seasonal_results
        }
    """  