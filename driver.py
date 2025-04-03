import os
from predictors.LTSM_predictor import LTSMDemandPredictor
from grid_drl.electrical_grid import LauderdaleGrid
import tensorflow as tf
import numpy as np
from grid_drl.drl import PowerGridEnv

from stable_baselines3 import PPO
from  stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import ProgressBarCallback, EvalCallback, CheckpointCallback

import gymnasium as gym
from gymnasium.envs.registration import register 
##############
##############

GPKG_PATH = "grid_drl/Electric-Power-Transmission-Lines.gpkg"
MODEL_PATH = "predictors/model/best_model.keras"
START = '2021-01-01'
END = '2021-01-31'
##############
##############
# register env with gym
register(
   id='PowerGridEnv-v0',
   entry_point='grid_drl.drl:PowerGridEnv',
   max_episode_steps=100
)

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
   # define a function that returns a fresh instance of custom env
   raw_env = lambda: gym.make('PowerGridEnv-v0', grid=grid)
   
   # wrap single env into vectorized interface -> gives a consistent interface for collecting and batch
   env = DummyVecEnv([raw_env])
    # wrap dummy vec into vec normalized obersation and reward for stabalized performance
   env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_reward=10.0)
   
   # === eval callback setup ===
   eval_env = DummyVecEnv([raw_env])
   eval_env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_reward=10.0)
   eval_env.obs_rms =env.obs_rms
   eval_env.ret_rms = env.ret_rms
   eval_env.training =False
   
   eval_callback= EvalCallback(
      eval_env,
      eval_freq=2048,
      n_eval_episodes=5,
      log_path="./ppo_logs_vect/eval",
      best_model_save_path="./ppo_logs_vect/best_model/",
      deterministic=True,
      render=False
   )
   
   # === checkpoint callback setup ===
   checkpoint_callback = CheckpointCallback(
    save_freq=25000,
    save_path="./checkpoints/",
    name_prefix="ppo_checkpoint"
      )
   

   
   model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_logs_vect")
   
   # === loading model logic ===
   #print("***Loaded model***")
   #model = PPO.load("ppo_powergrid_model", env=env)
      
   model.learn(total_timesteps=400000,callback= [ProgressBarCallback(), eval_callback, checkpoint_callback])
   
   # save model
   model.save("ppo_powergrid_model")
   env.save("vec_normalize.pkl")
   print("training complete")

if __name__ == "__main__":
   main()
   