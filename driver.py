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
EPISODES = 1000
BATCH_SIZE= 32
MAX_TIMESTEPS= 24
REWARD_T=-1000000
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
   #set up dqn
   state_size = len(grid.get_state())
   action_size = len(grid.actions)  # Get actions from grid instead
   agent = DQNAgent(state_size=state_size, action_size=action_size, grid=grid)
   

   
   # Training loop 
   for episode in range(EPISODES):
       print(f"\nEpisode {episode + 1}/{EPISODES}")
       grid = LauderdaleGrid(GPKG_PATH)
       grid.create_network()
       input_sequence= day_sequence + np.random.normal(0,0.01,day_sequence.shape)
       scale_factor = 0.012 *np.random.uniform(0.8,1.2)
       
       grid.add_loads(predictor=lstm_pred, 
                     input_sequence=day_sequence.reshape(1, -1, day_sequence.shape[-1]),
                     scale_factor=scale_factor)      
       state = grid.get_state()
       state = np.reshape(state, [1, state_size])
       
       # Run an episode
       done = False
       timestep = 0  # Track the number of timesteps per episode
       total_reward = 0  # Track cumulative reward for the episode

       while not done:
           # Ensure we don't exceed max timesteps
           if timestep >= MAX_TIMESTEPS:
               print(f"Episode terminated after reaching max timesteps ({MAX_TIMESTEPS}).")
               break

           # Choose action using epsilon-greedy policy
           action = agent.act(state)  
           grid.apply_action(action)

           # Get reward and next state
           reward = grid.reward()
           total_reward += reward
           next_state = grid.get_state()
           next_state = np.reshape(next_state, [1, state_size])
           
           # Store and remember in buffer
           agent.remember(state, action, reward, next_state, done)
           state = next_state

           # Perform experience replay
           agent.replay(BATCH_SIZE)

           # Print reward at each timestep
           print(f"Timestep {timestep + 1}, Reward: {reward:.2f}")

           # End condition based on reward threshold
           if reward < REWARD_T:
               print(f"Episode terminated early due to excessive penalty (reward: {reward:.2f}).")
               done = True

           timestep += 1  # Increment timestep
       
       # Update target model every 10 episodes
       if episode % 10 == 0:
           agent.update_target_model()
           print("Updated target model.")

       print(f"Episode {episode + 1} completed with total reward: {total_reward:.2f}")

if __name__ == "__main__":
   main()