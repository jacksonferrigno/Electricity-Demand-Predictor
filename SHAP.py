import numpy as np
import pandas as pd
import shap
from stable_baselines3 import PPO
from grid_drl.electrical_grid import LauderdaleGrid
from grid_drl.drl import PowerGridEnv
import tensorflow as tf
from predictors.LTSM_predictor import LTSMDemandPredictor
import matplotlib.pyplot as plt

# Load trained PPO agent
model = PPO.load("ppo_powergrid_model.zip")

GPKG_PATH = "grid_drl/Electric-Power-Transmission-Lines.gpkg"
MODEL_PATH = "predictors/model/best_model.keras"
START = '2021-01-01'
END = '2021-01-31'
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
   
env = PowerGridEnv(grid=grid)

#SHAP wrapper
def predict_actions(obs):
    actions, _ = model.predict(obs, deterministic=True)
    return actions


#collect background observations
background = []
obs, _ = env.reset()
for _ in range(100):
    action, _ = model.predict(obs)
    obs,_,done,_,_ = env.step(action)
    background.append(obs)
    if done:
        obs, _ = env.reset()
background = np.array(background)

#SHAP values for an exmaple

explainer = shap.KernelExplainer(predict_actions, background)
test_obs= background[0:1]
shap_values = explainer.shap_values(test_obs, nsamples=100) 


feature_names=(
    [f"Generator_{i}_output" for i in range(env.num_generators)] +
    [f"Load_{i}_demand" for i in range(env.num_loads)] +
    [f"Generator_{i}_marginal_cost" for i in range(env.num_generators)] +
    [f"Transformer_{i}_tap" for i in range(env.num_transformers)]
)   
# === add the plotting logic=====

action_idx =0 

single_shap_values = shap_values[0,:, action_idx]

shap_df=pd.DataFrame({
    "Feature": feature_names,
    "Feature Value": test_obs[0],
    "SHAP Value": single_shap_values,
})

#sort by shap vlaue
shap_df["abs(SHAP Value)"] = shap_df["SHAP Value"].abs()
shap_df_sorted = shap_df.sort_values(by="abs(SHAP Value)", ascending=False)

print(shap_df_sorted.head(10))

plt.figure(figsize=(10, 8))
plt.barh(
    shap_df_sorted["Feature"].iloc[:15][::-1],
    shap_df_sorted["SHAP Value"].iloc[:15][::-1],
    color='skyblue'
)
plt.xlabel("SHAP Value (Impact on Action)")
plt.title("Top 15 Features Impacting Action")
plt.tight_layout()
plt.show()