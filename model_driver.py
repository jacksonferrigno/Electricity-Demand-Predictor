from base_demand_predictor import DemandPredictor
from seasonal_predictor import SeasonalPredictor
import matplotlib.pyplot as plt
from LTSM_predictor import LTSMDemandPredictor

if __name__ == "__main__":
    #config params
    START ='2020-07-29'
    END = '2023-12-31'
    print("\n running LSTM analysis...")
    lstm_predictor = LTSMDemandPredictor(START,END)
    
    lstm_predictor.fit(epochs=200,batch_size=16)
    
    lstm_metrics = lstm_predictor.get_metrics()
    
    print("\n=======LSTM performance=========")
    print(f"r2 score {lstm_metrics['r2']:.3f}")
    
    if 'training_history' in lstm_metrics:
        print("\n==== Training History ====")
        print(f"Final loss: {lstm_metrics['training_history']['final_loss']:.4f}")
        print(f"Best validation loss: {lstm_metrics['training_history']['best_val_loss']:.4f}")
        print(f"our model outperformed the TVA's {lstm_metrics['model_win_rate']:.2f}% of the time")
    
    # Plot LSTM analysis
    lstm_predictor.plot_analysis()
    plt.show()
    #base analysis 
    """
    print("\n running base analysis...")
    base_predictor = DemandPredictor(START,END)
    base_predictor.fit()
    
    #metrics 
    base_metrics = base_predictor.get_metrics()
    print("\n ==== Regression performance====")
    print(f"R2 score {base_metrics['regression_metrics']['r2']:.3f}")
    print("===== Cluster Analysis =====")
    print(base_metrics['cluster_stats'].to_string())
    
    base_predictor.plot_analysis()
    plt.show()
    #seasonal
    print("\n running seasonal analysis...")
    season_predictor= SeasonalPredictor(START, END)
    season_predictor.load_data()
    season_predictor.fit_all_seasons()
    
    #plot 
    #season_metrics= season_predictor.get_metrics()
    for season in ['Winter', 'Spring', 'Summer', 'Fall']:     
        season_predictor.plot_analysis(season)
        plt.show()
        """
    
        
    