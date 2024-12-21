from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional, Concatenate, LayerNormalization
from tensorflow.keras.layers import MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.losses import MSE, MAE  
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

from base_demand_predictor import DemandPredictor

class LTSMDemandPredictor(DemandPredictor):
    def __init__(self, start_date, end_date, sequence_length=3, n_clusters=3):
        super().__init__(start_date, end_date, n_clusters)
        self.sequence_length= sequence_length
        self.temp_scaler= MinMaxScaler() # normalize temp ** easier to seperate to send back to actual value normalizing gets it 0-1
        self.demand_scaler =MinMaxScaler() # normalize demand
        self.model = None  #will hold model once built
        self.history=None # will store training data
        self.attention_weights = None # stores attention weights for model
        
    def prepare_sequences(self):
        """ preps sequences for feat engineering
        
        *What this does*
        
        Input Sequence (X):
        [Day 1 features, Day 2 features, ... Day 7 features] → Output (y): Day 8 demand
        [Day 2 features, Day 3 features, ... Day 8 features] → Output (y): Day 9 demand
        """
        # inital data check and scaling
        if self.data is None:
            self.load_data()
                
        #scale data 
        scaled_temps = self.temp_scaler.fit_transform(self.X) # scale temp 0-1
        scaled_demands = self.demand_scaler.fit_transform(self.y.reshape(-1,1)) # make sures data is in right format
        forecast_scaler= MinMaxScaler()
        scaled_forecasts = forecast_scaler.fit_transform(self.data['day_ahead_forecast'].values.reshape(-1,1))
        
        
        #feat engineering- historical patterns
        df = pd.DataFrame(self.data)
        df['scaled_temp'] = scaled_temps  # Add scaled temp to dataframe
        df['scaled_demand'] = scaled_demands  # Add scaled demand to dataframe
        df['scaled_forecast']=scaled_forecasts.flatten()
        
        
        # Calculate features using scaled temperature
        df['temp_rolling_mean_3d'] = df['scaled_temp'].rolling(window=3).mean() # 3 day temp avg short term trend
        df['temp_rolling_std'] = df['scaled_temp'].rolling(window=3).std() # captures temp volatility
        df['temp_change']= df['scaled_temp'].diff() # day to day shift
        df['demand_lag1']= df['scaled_demand'].shift(1) # prev day demand to help predict next day
        
        df['forecast_error']=df['scaled_demand']-df['scaled_forecast'].shift(1)
        df['forecast_bias']= df['forecast_error'].rolling(window=7).mean()
                    
        # cyclical time feats
        df['day_of_year']= df['date'].dt.dayofyear
        df['day_sin'] = np.sin(2*np.pi * df['day_of_year']/365)
        df['day_cos']= np.cos(2 *np.pi *df['day_of_year']/365 ) #these two functions make the days roll ( round ribbon)
        
        # Fill NaN values
        df = df.bfill()
            
        feature_columns = [
            'scaled_temp', 'scaled_demand', 'scaled_forecast',
            'temp_rolling_mean_3d', 
            'temp_rolling_std', 'temp_change', 'demand_lag1',
            'forecast_error', 'forecast_bias',
            'day_sin', 'day_cos'
        ]
        
        # creating sequences 
        X, y = [], []
        for i in range(len(df)- self.sequence_length): #create sliding window of data
            feature_sequence = df[feature_columns].iloc[i:(i+self.sequence_length)].values
            X.append(feature_sequence) # seq of feats for each window
            y.append(scaled_demands[i+self.sequence_length]) # demand value for day after each 
            
        return np.array(X), np.array(y)
    def build_attention_model(self, n_features):
        """ Build model with bidrectional LSTM and mul head attention"""
        #I/P layer
        inputs = Input(shape=(self.sequence_length, n_features))
        # first bidirectional layer
        x= Bidirectional(LSTM(128, return_sequences=True))(inputs) # processes 64 feats forward and backwards
        x= LayerNormalization()(x) # stablize training
        x= Dropout(0.2)(x) # prevent overfitting by randomly dropping 20% of connections
        
        #mul head attention block ( like looking thru data with multiple lenses)
        #4 heads to look at different aspects
        #key_dim = dimensions of attention comp
        attention_output1 = MultiHeadAttention(
            num_heads=8, key_dim=32 
        )(x,x,x)
        x= LayerNormalization()(attention_output1+x) # skip connection
        
        #second layer
        x= Bidirectional(LSTM(64, return_sequences=True))(x) # processes 32 (smaller than first) feats forward and backwards
        x= LayerNormalization()(x) # stablize training
        x= Dropout(0.2)(x) # prevent overfitting by randomly dropping 20% of connections
        
        attention_output2= MultiHeadAttention(
            num_heads=8, key_dim=32
        )(x,x,x)
        x=LayerNormalization()(attention_output2+x)
        
        # reduce seqeunce dimesnsion
        x= GlobalAveragePooling1D()(x)
        
        #final prediction layer
        x= Dense(128, activation='relu')(x)
        x=Dropout(0.2)(x)
        x= Dense(64, activation='relu')(x)
        x= Dropout(0.2)(x)
        outputs = Dense(1)(x)
        
        # create the model 
        model = Model(inputs=inputs, outputs=outputs)
        
        
        #loss function with demand penalty 
        def custom_demand_loss(y_true, y_pred):
            """extra penalty for under predicting demand"""
            mse = tf.keras.losses.MSE(y_true,y_pred)
            high_demand_mask = tf.cast(y_true>tf.reduce_mean(y_true), tf.float32)
            low_demand_mask = 1.0 - high_demand_mask
            
            under_pred_penalty = tf.maximum(0.0, y_true -y_pred) *high_demand_mask
            
            over_pred_penalty = tf.maximum(0.0,y_pred-y_true)* low_demand_mask
            penalty = 0.5 * tf.reduce_mean(under_pred_penalty) + 0.3 * tf.reduce_mean(over_pred_penalty)
            return mse +penalty
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=custom_demand_loss,
            metrics=['mae', 'mse']
        )
        self.model=model
        return model
    
    def fit(self, epochs =50, batch_size =32, validation_split=0.2):
        """training process"""
        # load and prep data
        super().load_data()
        X, y= self.prepare_sequences()
        
        #split data maintains time order
        train_size = int(len(X)*(1-validation_split))
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        #build the model 
        self.build_attention_model(n_features=X.shape[2])
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                min_delta=0.0001
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=10,
                min_lr=0.00001,
                min_delta=0.0001
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'best_model.keras',
                monitor='val_loss',
                save_best_only=True
            )
        ]
        #train the model 
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val,y_val),
            callbacks=callbacks,
            verbose=1
        )
        #calculate metrics 
        y_pred = self.model.predict(X)
        y_pred=self.demand_scaler.inverse_transform(y_pred)
        y_true=self.demand_scaler.inverse_transform(y)
        
        self.metrics={
            'r2': r2_score(y_true,y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true,y_pred))
        }
        
        return self
    
    def predict(self, input_sequence):
        """Make predictions with confidence estimation"""
        if self.model is None:
            raise ValueError("Call fit() first.")
        
        # Prepare input sequence
        processed_sequence = self._prepare_prediction_sequence(input_sequence)
        
        # Make prediction
        prediction = self.model.predict(processed_sequence)
        prediction = self.demand_scaler.inverse_transform(prediction)
        
        return prediction
    
    def _prepare_prediction_sequence(self, input_sequence):
        if len(input_sequence) != self.sequence_length:
            raise ValueError(f"Input sequence must have length {self.sequence_length}")
        
        # Create DataFrame with features
        df = pd.DataFrame({
            'avg_high': input_sequence['temperature'],
            'day_ahead_forecast': input_sequence['forecast']  # Add forecast to input
        })
        
        # Scale features
        scaled_temps = self.temp_scaler.transform(df[['avg_high']])
        scaled_forecasts = self.demand_scaler.transform(df[['day_ahead_forecast']])
        
        # Calculate all features
        df['scaled_temp'] = scaled_temps
        df['scaled_forecast'] = scaled_forecasts
        df['temp_rolling_mean_3d'] = df['scaled_temp'].rolling(window=3).mean()
        df['temp_rolling_std'] = df['scaled_temp'].rolling(window=3).std()
        df['temp_change'] = df['scaled_temp'].diff()
        
        # Add forecast features
        df['forecast_error'] = 0  # For prediction, we don't have actual values
        df['forecast_bias'] = 0   # For prediction, we don't have historical errors
        
        # Fill NaN values
        df = df.fillna(method='bfill')
        
        # Add cyclical features
        df['day_sin'] = input_sequence['day_sin']
        df['day_cos'] = input_sequence['day_cos']
        
        feature_columns = [
            'scaled_temp', 'scaled_forecast',
            'temp_rolling_mean_3d', 
            'temp_rolling_std', 'temp_change',
            'forecast_error', 'forecast_bias',
            'day_sin', 'day_cos'
        ]
        
        sequence = df[feature_columns].values
        return np.expand_dims(sequence, axis=0)
    def plot_analysis(self):
        """Create visualization of LSTM results and training"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Get predictions for entire dataset
        X, y = self.prepare_sequences()
        y_pred = self.model.predict(X)
        
        # Inverse transform the scaled values
        y_pred = self.demand_scaler.inverse_transform(y_pred)
        y_true = self.demand_scaler.inverse_transform(y)
        
        # Plot actual vs predicted
        ax.scatter(y_true, y_pred, alpha=0.5, color='blue', s=20, label='Predictions')
        
        # Plot perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        # Add labels and title
        ax.set_xlabel('Actual Demand (Megawatthours)')
        ax.set_ylabel('Predicted Demand (Megawatthours)')
        ax.set_title('LSTM Model: Actual vs Predicted Demand')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add metrics text
        metrics_text = (
            f'R² = {self.metrics["r2"]:.3f}\n'
            f'RMSE = {self.metrics["rmse"]:.2f} MWh'
        )
        ax.text(0.02, 0.98, metrics_text,
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    
    def get_metrics(self):
        """Get all analysis metrics including training history"""
        if self.metrics is None:
            raise ValueError("Models not fitted yet. Call fit() first.")
            
        metrics_dict = {
            'r2': self.metrics['r2'],
            'rmse': self.metrics['rmse']
        }
        
        if self.history is not None:
            metrics_dict['training_history'] = {
                'final_loss': self.history.history['loss'][-1],
                'final_val_loss': self.history.history['val_loss'][-1],
                'best_val_loss': min(self.history.history['val_loss'])
            }
        
        return metrics_dict