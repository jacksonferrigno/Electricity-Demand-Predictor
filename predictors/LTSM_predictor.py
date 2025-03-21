from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional, Concatenate, LayerNormalization
from tensorflow.keras.layers import MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.losses import MSE, MAE  
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import xgboost as xgb
import numpy as np
import pandas as pd

from  predictors.base_demand_predictor import DemandPredictor

class LTSMDemandPredictor(DemandPredictor):
    def __init__(self, start_date, end_date, sequence_length=3, n_clusters=3):
        super().__init__(start_date, end_date, n_clusters)
        self.sequence_length= sequence_length
        self.temp_scaler= StandardScaler() # normalize temp ** easier to seperate to send back to actual value normalizing gets it 0-1
        self.demand_scaler =StandardScaler() # normalize demand
        self.precip_scaler = StandardScaler() 
        self.model = None  #will hold model once built
        self.history=None # will store training data
        self.attention_weights = None # stores attention weights for model
        
    @staticmethod
    def custom_demand_loss(y_true, y_pred):
        """extra penalty for under predicting demand"""
        mse = tf.keras.losses.MSE(y_true,y_pred)
        high_demand_mask = tf.cast(y_true>tf.reduce_mean(y_true), tf.float32)
        low_demand_mask = 1.0 - high_demand_mask
        
        under_pred_penalty = tf.maximum(0.0, y_true -y_pred) *high_demand_mask
        over_pred_penalty = tf.maximum(0.0,y_pred-y_true)* low_demand_mask
        penalty = 1.9 * tf.reduce_mean(under_pred_penalty) + 1.6 * tf.reduce_mean(over_pred_penalty)
        
        return mse + penalty 
    def prepare_sequences(self):
        """ preps sequences for feat engineering
        
        *What this does*
        
        Input Sequence (X):
        [Day 1 features, Day 2 features, ... Day 7 features] → Output (y): Day 8 demand
        [Day 2 features, Day 3 features, ... Day 8 features] → Output (y): Day 9 demand
        """
        # Initial data check and scaling
        if self.data is None:
            self.load_data()
                
        scaled_temps = self.temp_scaler.fit_transform(self.X)
        scaled_demands = self.demand_scaler.fit_transform(self.y.reshape(-1,1))
        forecast_scaler = StandardScaler()
        scaled_forecasts = forecast_scaler.fit_transform(self.data['day_ahead_forecast'].values.reshape(-1,1))
        scaled_precip = self.precip_scaler.fit_transform(self.data['avg_precip'].values.reshape(-1,1))
        
        # Create DataFrame and add basic scaled features
        df = pd.DataFrame(self.data)
        df['date'] = pd.to_datetime(df['date'])
        df['scaled_temp'] = scaled_temps
        df['scaled_forecast'] = scaled_forecasts.flatten()
        df['scaled_precip'] = scaled_precip.flatten()
        
        # rolling mean features
        df['temp_rolling_mean_7d'] = df['scaled_temp'].rolling(window=7, min_periods=1).mean()
        df['temp_rolling_mean_3d'] = df['scaled_temp'].rolling(window=3, min_periods=1).mean()
        
        # demand lag features
        df['demand_lag1']= df['scaled_forecast'].shift(1)
        df['demand_lag7'] = df['scaled_forecast'].shift(7)
        df['demand_lag14'] = df['scaled_forecast'].shift(14)
        
        # demand change and weekly mean
        df['demand_change'] = df['scaled_forecast'].diff()
        df['demand_weekly_mean'] = df['scaled_forecast'].rolling(window=7, min_periods=1).mean()
        
        #  temperature volatility
        df['temp_volatility'] = df['scaled_temp'].rolling(window=7, min_periods=1).std()
        
        # interaction feature
        df['temp_demand_interaction'] = df['temp_rolling_mean_7d'] * df['demand_weekly_mean']
        
        # extreme temperature flag
        df['extreme_temp_flag'] = ((df['scaled_temp'] > 2.0) | (df['scaled_temp'] < -2.0)).astype(float)
        
        # season forcast deviations
        df['month'] = pd.to_datetime(df['date']).dt.month
        mon_means= df.groupby('month')['scaled_forecast'].transform('mean')
        mon_stds=df.groupby('month')['scaled_forecast'].transform('std')
        df['seasonal_forecast_deviation']= (df['scaled_forecast']-mon_means)/mon_stds
        
        # Add cyclical time features
        df['day_of_year'] = df['date'].dt.dayofyear
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year']/365).astype(float)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year']/365).astype(float)
        # dynamic forcast adjustments 
        # Fill NaN values
        df = df.bfill()
        
        # Define feature columns
        feature_columns = [
            'scaled_temp', 'scaled_forecast',
            'temp_rolling_mean_7d', 'temp_rolling_mean_3d',
            'demand_weekly_mean',   
            'temp_volatility', 'demand_lag1',     
            'demand_lag7', 'demand_lag14',
            'temp_demand_interaction', 'extreme_temp_flag',
            'demand_change', 'scaled_precip', 
            'day_sin', 'day_cos',
            'seasonal_forecast_deviation'
        ]
        
        # Ensure all features are float32
        df[feature_columns] = df[feature_columns].astype('float32')
        
        # Create sequences
        X, y = [], []
        for i in range(len(df) - self.sequence_length):
            feature_sequence = df[feature_columns].iloc[i:(i+self.sequence_length)].values
            X.append(feature_sequence)
            y.append(scaled_demands[i+self.sequence_length])
        
        return np.array(X, dtype='float32'), np.array(y, dtype='float32')
    
    def build_attention_model(self, n_features):
        """ Build model with bidrectional LSTM and mul head attention"""
        #I/P layer
        inputs = Input(shape=(self.sequence_length, n_features))
        # first bidirectional layer
        x= Bidirectional(LSTM(128, return_sequences=True))(inputs) # processes 128 feats forward and backwards
        x= LayerNormalization()(x) # stablize training
        x= Dropout(0.2)(x) # prevent overfitting by randomly dropping 20% of connections
        
        #mul head attention block ( like looking thru data with multiple lenses)
        #8 heads to look at different aspects
        #key_dim = dimensions of attention comp
        attention_output1 = MultiHeadAttention(
            num_heads=8, key_dim=32 
        )(x,x,x)
        x= LayerNormalization()(attention_output1+x) # skip connection
        
        #second layer
        x= Bidirectional(LSTM(64, return_sequences=True))(x) # processes 64 (smaller than first) feats forward and backwards
        x= LayerNormalization()(x) # stablize training
        x= Dropout(0.2)(x) # prevent overfitting by randomly dropping 20% of connections
        
        attention_output2= MultiHeadAttention(
            num_heads=8, key_dim=32
        )(x,x,x)
        x=LayerNormalization()(attention_output2+x)
        
        # reduce seqeunce dimesnsion
        x= GlobalAveragePooling1D()(x)
        #final prediction layer
        x = Dense(128, activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(0.02))(x)  
        x = Dropout(0.3)(x)  # Increased dropout
        x = Dense(64, activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(0.02))(x)
        x = Dropout(0.3)(x)
        # Add a constraint layer to help with extreme values
        x = Dense(32, activation='relu')(x)
        outputs = Dense(1, activation='linear')(x) 
        # create the model 
        model = Model(inputs=inputs, outputs=outputs)
        
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=self.custom_demand_loss,
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
                patience=25,
                restore_best_weights=True,
                min_delta=0.00005
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=12,
                min_lr=0.00001,
                min_delta=0.00005
            ),
            tf.keras.callbacks.ModelCheckpoint(
                '/model/best_model.keras',
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
            'day_ahead_forecast': input_sequence['forecast'],
            'avg_precip': input_sequence.get('precipitation', 0)
        })
        
        # Scale features
        scaled_temps = self.temp_scaler.transform(df[['avg_high']])
        scaled_forecasts = self.demand_scaler.transform(df[['day_ahead_forecast']])
        scaled_precip = self.precip_scaler.transform(df[['avg_precip']].values.reshape(-1, 1))
        
        # basic scaled features
        df['scaled_temp'] = scaled_temps
        df['scaled_forecast'] = scaled_forecasts.flatten()
        df['scaled_precip'] = scaled_precip.flatten()
        
        # rolling mean features
        df['temp_rolling_mean_7d'] = df['scaled_temp'].rolling(window=7, min_periods=1).mean()
        df['temp_rolling_mean_3d'] = df['scaled_temp'].rolling(window=3, min_periods=1).mean()
        
        # demand lag features
        df['demand_lag1']= df['scaled_forcast'].shift(1)
        df['demand_lag7'] = df['scaled_forecast'].shift(7)
        df['demand_lag14'] = df['scaled_forecast'].shift(14)
        
        # demand change and weekly mean
        df['demand_change'] = df['scaled_forecast'].diff()
        df['demand_weekly_mean'] = df['scaled_forecast'].rolling(window=7, min_periods=1).mean()
        
        # temperature volatility
        df['temp_volatility'] = df['scaled_temp'].rolling(window=7, min_periods=1).std()
        
        # interaction feature
        df['temp_demand_interaction'] = df['temp_rolling_mean_7d'] * df['demand_weekly_mean']
        
        # extreme  flags
        df['extreme_temp_flag'] = ((df['scaled_temp'] > 2.0) | (df['scaled_temp'] < -2.0)).astype(float)

        #forecast deviations 
        df['month'] = df['date'].dt.month
        monthly_means = df.groupby('month')['scaled_forecast'].transform('mean')
        monthly_stds = df.groupby('month')['scaled_forecast'].transform('std')
        df['seasonal_forecast_deviation'] = (df['scaled_forecast'] - monthly_means) / monthly_stds
        
        # Add cyclical features
        df['day_sin'] = input_sequence['day_sin']
        df['day_cos'] = input_sequence['day_cos']
        
        # Fill NaN values
        df = df.bfill()
        
        # Use exactly the same feature columns in same order as prepare_sequences
        feature_columns = [
            'scaled_temp', 'scaled_forecast',
            'temp_rolling_mean_7d', 'temp_rolling_mean_3d',
            'demand_weekly_mean',   
            'temp_volatility', 'demand_lag1',     
            'demand_lag7', 'demand_lag14',
            'temp_demand_interaction', 'extreme_temp_flag',
            'demand_change', 'scaled_precip', 
            'day_sin', 'day_cos',
            'seasonal_forecast_deviation'
            ]    
        # Ensure all features are float32
        df[feature_columns] = df[feature_columns].astype('float32')
        
        sequence = df[feature_columns].values
        return np.expand_dims(sequence, axis=0)
    def plot_analysis(self):
        """Create visualization of LSTM results and training"""
        fig, ((ax1,ax2),(ax3,ax4))=  plt.subplots(2, 2, figsize=(15, 12))
        
        # Get predictions for entire dataset
        X, y = self.prepare_sequences()
        y_pred = self.model.predict(X)
        
        # Inverse transform the scaled values
        y_pred = self.demand_scaler.inverse_transform(y_pred)
        y_true = self.demand_scaler.inverse_transform(y)
        
        # Plot1 actual vs predicted
        ax1.scatter(y_true, y_pred, alpha=0.5, color='blue', s=20, label='Predictions')
        
        # Plot perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        # Add labels and title
        ax1.set_xlabel('Actual Demand (Megawatthours)')
        ax1.set_ylabel('Predicted Demand (Megawatthours)')
        ax1.set_title('LSTM Model: Actual vs Predicted Demand')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        #plot 2 error distrubtion 
        errors= y_pred.flatten() - y_true.flatten()
        ax2.hist(errors, bins=50,color='skyblue',edgecolor='black')
        ax2.axvline(x=0,color='r',linestyle='dashed')
        ax2.set_title('Prediction Error Distribution')
        ax2.set_xlabel('Prediction Error (MWh)')
        ax2.set_ylabel('Frequency')
        
        #plot 3 training history
        if self.history:
            ax3.plot(self.history.history['loss'], label='Training Loss')
            ax3.plot(self.history.history['val_loss'],label='Validation Loss')
            ax3.set_title('Training History')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Loss')
            ax3.legend()
            
        #plot 4 residuals vs predicted
        ax4.scatter(y_pred,errors,alpha=0.5,color='purple',s=20)
        ax4.axhline(y=0,color='r',linestyle='dashed')
        ax4.set_title('Residuals vs Predicted VaLues')
        ax4.set_xlabel('Predicted Demand (MWh)')
        ax4.set_ylabel('Residual Error (MWh)')
        
        
        
        # Add metrics text
        metrics_text = (
        f'Model Performance Metrics:\n'
        f'R² = {self.metrics["r2"]:.3f}\n'
        f'RMSE = {self.metrics["rmse"]:.2f} MWh\n'
        f'Mean Error = {np.mean(errors):.2f} MWh\n'
        f'Std Error = {np.std(errors):.2f} MWh\n'
    )
        fig.text(0.02, 0.90, metrics_text, 
             fontsize=10, 
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
        
        plt.tight_layout()
        return fig
    
    
    def get_metrics(self):
        """Get all analysis metrics including training history"""
        if self.metrics is None:
            raise ValueError("Models not fitted yet. Call fit() first.")
        X, y= self.prepare_sequences()
        lstm_pred= self.model.predict(X)
        lstm_pred = self.demand_scaler.inverse_transform(lstm_pred)
        actual_demands=self.demand_scaler.inverse_transform(y)
        tva_forecasts= self.data['day_ahead_forecast'].values[self.sequence_length-1:-1][:len(lstm_pred)]
        tva_forecasts= tva_forecasts.reshape(-1,1)
        
        
        #calc how often we beat the tva
        lstm_errors= np.abs(lstm_pred-actual_demands)
        tva_errors=np.abs(tva_forecasts-actual_demands)
        lstm_wins= (lstm_errors<tva_errors).sum()
        total_predictions=float(len(lstm_errors))
            
        metrics_dict = {
            'r2': self.metrics['r2'],
            'rmse': self.metrics['rmse'],
            'model_win_rate': (lstm_wins/total_predictions)*100
        }
        
        if self.history is not None:
            metrics_dict['training_history'] = {
                'final_loss': self.history.history['loss'][-1],
                'final_val_loss': self.history.history['val_loss'][-1],
                'best_val_loss': min(self.history.history['val_loss'])
            }
        
            return metrics_dict