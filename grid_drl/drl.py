import gymnasium as gym
from gymnasium import spaces
import numpy as np

class PowerGridEnv(gym.Env):
    def __init__(self, grid, max_steps=100, action_limit=0.1):
        super(PowerGridEnv, self).__init__()
        
        self.grid_model = grid
        self.grid = grid.network
        self.max_steps = max_steps
        self.action_limit = action_limit
        
        # Action space: Generator output adjustments
        self.num_generators = len(self.grid.generators)
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(self.num_generators,),
            dtype=np.float32
        )
        
        # State space: Generator outputs + load demands
        self.state_size = self.num_generators + len(self.grid.loads)
        self.observation_space = spaces.Box(
            low=-1,
            high=1,
            shape=(self.state_size,),
            dtype=np.float32
        )
        
        self.current_step = 0
        self.state = self._get_state()
        self.previous_violations = 0

    def step(self, action):
        if not isinstance(action, np.ndarray):
            action = np.array(action)
            
        self._apply_action(action)
        self.grid.lpf()
        
        self.state = self._get_state()
        reward = self._calculate_reward()
        terminated=self._check_done()
        truncated=False
        
        self.current_step += 1
        
        info = {
            "total_demand": self.grid.loads["p_set"].sum(),
            "total_generation": self.grid.generators["p_nom"].sum()
        }
        
        return self.state, reward, terminated,truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = self.grid_model.network
        self.current_step = 0
        self.previous_violations = 0
        self.state = self._get_state().astype(np.float32)
        return self.state,{}

    def _apply_action(self, action):
        scaled_action = action *self.action_limit
        for i, adjustment in enumerate(scaled_action):
            generator_name = self.grid.generators.index[i]
            current_p_nom = self.grid.generators.at[generator_name, "p_nom"]
            new_p_nom = current_p_nom * (1 + adjustment)
            new_p_nom = max(0, new_p_nom)  # Prevent negative output
            self.grid.generators.at[generator_name, "p_nom"] = new_p_nom

    def _get_state(self):
        generator_outputs = self.grid.generators["p_nom"].values
        load_demands = self.grid.loads["p_set"].values

        # Avoid division by zero by setting a safe normalization factor
        max_gen_output = max(np.max(generator_outputs), 1e-6)
        max_load = max(np.max(load_demands), 1e-6)

        # Normalize to [-1, 1] range
        generator_outputs = (generator_outputs / max_gen_output) * 2 - 1
        load_demands = (load_demands / max_load) * 2 - 1

        # Ensure correct sizes by padding missing values with zeros
        generator_outputs = np.pad(generator_outputs, (0, self.num_generators - len(generator_outputs)), mode='constant')
        load_demands = np.pad(load_demands, (0, (self.state_size - self.num_generators) - len(load_demands)), mode='constant')

        # Concatenate and ensure float32 dtype
        state = np.concatenate([generator_outputs, load_demands]).astype(np.float32)

        # Debugging: Check the new state values
        if np.any(state < -1) or np.any(state > 1):
            print(" Warning: State values are out of bounds!", state)

        return state

    def _calculate_reward(self):
        # Demand-supply mismatch penalty
        total_demand = self.grid.loads["p_set"].sum()
        total_generation = self.grid.generators["p_nom"].sum()
        mismatch = abs(total_generation - total_demand)
        reward = -mismatch * 0.01
        
        # Thermal violations penalty
        if "loading" in self.grid.lines.columns:
            thermal_violations = sum(self.grid.lines["loading"] > 100)
            reward -= (thermal_violations *0.5)
            
            # Reward for reducing violations
            violation_change = thermal_violations - self.previous_violations
            reward += max(0, -violation_change) * 3
            self.previous_violations = thermal_violations
            
        if mismatch<50: #mismatch is small : positive reward
            reward +=5*(0.4 *mismatch) # better reward as it gets closer to 0
            
        #reward for grid stability 
        gen_variability = np.std(self.grid.generators["p_nom"].values)
        reward -=gen_variability *0.01 # small penalty for instability
        
        if gen_variability<5:
            reward+=2500
            
        return float(reward)

    def _check_done(self):
        if self.current_step >= self.max_steps:
            return True
            
        if "loading" in self.grid.lines.columns:
            if any(self.grid.lines["loading"] > 150):  # Severe overload
                return True
                
        return False