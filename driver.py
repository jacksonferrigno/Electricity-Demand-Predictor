from predictors.LTSM_predictor import LTSMDemandPredictor
from grid_drl.electrical_grid import LauderdaleGrid
import tensorflow as tf
import numpy as np
from grid_drl.drl import DQNAgent

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
   day_sequence = X[-1:]

   # Initialize grid and create network
   grid = LauderdaleGrid(GPKG_PATH)
   network = grid.create_network()  
   # Add loads based on predictions
   print(f"\nDAY ANALYSIS")
   print("=" * 50)
   grid.add_loads(predictor=lstm_pred, 
                 input_sequence=day_sequence.reshape(1, -1, day_sequence.shape[-1]),
                 scale_factor=0.012)


if __name__ == "__main__":
   main()