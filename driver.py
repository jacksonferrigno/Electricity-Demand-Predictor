from predictors.LTSM_predictor import LTSMDemandPredictor
from grid_drl.electrical_grid import LauderdaleGrid
import tensorflow as tf
import matplotlib.pyplot as plt
##############
##############
GPKG_PATH = "grid_drl/Electric-Power-Transmission-Lines.gpkg"
MODEL_PATH = "predictors/model/best_model.keras"
START = '2021-01-01'
END = '2021-01-31'
##############
##############
def main():
    # Initialize LSTM Predictor
    lstm_pred = LTSMDemandPredictor(START, END)
    lstm_pred.model = tf.keras.models.load_model(MODEL_PATH,
                                               custom_objects={'custom_demand_loss': LTSMDemandPredictor.custom_demand_loss})
    
    # Load and prepare data 
    lstm_pred.load_data()
    X, _ = lstm_pred.prepare_sequences()
    predicted_demand = lstm_pred.model.predict(X)
    predicted_demand = lstm_pred.demand_scaler.inverse_transform(predicted_demand)

    # Short-term demand (last 7 days)
    short_term_demand = predicted_demand[-7:]
    short_term_sequences = X[-7:]  # Get corresponding input sequences

    # Initialize Lauderdale Grid and create network
    grid = LauderdaleGrid(GPKG_PATH)
    grid.create_grid()

    # Add loads based on predictions
    for day, (demand, sequence) in enumerate(zip(short_term_demand, short_term_sequences)):
        print(f"\nDAY {day + 1} ANALYSIS")
        print("=" * 50)
        grid.add_loads(predictor=lstm_pred, 
                      input_sequence=sequence.reshape(1, -1, sequence.shape[-1]),
                      scale_factor=0.015)
        performance= grid.run_power_flow()
        if performance['success']:
            grid.plot_performance_summary(performance)
            plt.show()

    # Visualize the grid
    #grid.plot_grid()

if __name__ == "__main__":
    main()

