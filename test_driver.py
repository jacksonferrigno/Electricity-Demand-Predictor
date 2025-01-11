import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.metrics import r2_score, mean_squared_error
from LTSM_predictor import LTSMDemandPredictor
import tensorflow as tf
import logging 
import os

class ModelTester:
    def __init__(self, start_date="2020-01-01", end_date="2024-12-31", output_file='test_results_v1.csv'):
        self.output_file = output_file
        self.start_date = start_date
        self.end_date = end_date
        self.setup_logging()
        
        # Initialize predictor 
        self.predictor = LTSMDemandPredictor(start_date, end_date)
        
        # Load data for prediction
        self.predictor.load_data()
        self.X, self.y = self.predictor.prepare_sequences()
        
        # Load saved model
        if os.path.exists('best_model.keras'):
            logging.info("Loading saved model...")
            
            # Build model first to get loss function
            self.predictor.model = self.predictor.build_attention_model(n_features=self.X.shape[2])
            custom_loss = self.predictor.model.loss
            
            # Load the trained weights
            self.predictor.model = tf.keras.models.load_model(
                'best_model.keras',
                custom_objects={'custom_demand_loss': custom_loss}
            )
            
            # Calculate initial metrics
            y_pred = self.predictor.model.predict(self.X)
            y_pred = self.predictor.demand_scaler.inverse_transform(y_pred)
            y_true = self.predictor.demand_scaler.inverse_transform(self.y)
            
            self.predictor.metrics = {
                'r2': r2_score(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred))
            }
        else:
            raise FileNotFoundError("no model found")
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def log_test_results(self, test_name, description, metrics):
        timestamp= datetime.now().strftime('%Y-%m-%d %H:%M')
        
        result={
            'timestamp': timestamp,
            'description': description,
            'r2_score': metrics['r2'],
            'rmse': metrics['rmse'],
            'win_rate': metrics.get('model_win_rate',0),
            'model_version': 'LSTM_V2.0'
        }
                # Add training history if available
        if 'training_history' in metrics:
            result.update({
                'final_loss': metrics['training_history']['final_loss'],
                'final_val_loss': metrics['training_history']['final_val_loss'],
                'best_val_loss': metrics['training_history']['best_val_loss']
            })
        df = pd.DataFrame([result])
        
        if not os.path.exists(self.output_file):
            df.to_csv(self.output_file,index=False)
        else:
            df.to_csv(self.output_file,mode='a',header=False,index=False)
            
        logging.info(f"test complete {test_name}")
        logging.info(f"Metrics: r2= {metrics['r2']:.3f} & win rate= {metrics.get('model_win_rate',0):.2f}%")
    def run_basic_test(self):
        logging.info("Starting basic model test....")
        metrics = self.predictor.get_metrics()
        self.log_test_results(
            "Basic test",
            "Full dataset performance eval",
            metrics
        )
        
        fig = self.predictor.plot_analysis()
        fig.savefig('model_plot.png')
        logging.info("saved analysis image to 'model_plot.png")
        
        
if __name__ == "__main__":
    try:
        # Initialize tester
        tester = ModelTester()
        
        # Run tests
        logging.info("Starting testing...")
        tester.run_basic_test()
        
    except Exception as e:
        logging.error(f"Error during testing: {str(e)}")
            
            
        
    