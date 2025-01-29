import gym # type: ignore
from gym import spaces # type: ignore
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
            low=-self.action_limit,
            high=self.action_limit,
            shape=(self.num_generators,),
            dtype=np.float32
        )
        
        # State space: Generator outputs + load demands
        self.state_size = self.num_generators + len(self.grid.loads)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
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
        done = self._check_done()
        
        self.current_step += 1
        
        info = {
            "total_demand": self.grid.loads["p_set"].sum(),
            "total_generation": self.grid.generators["p_nom"].sum()
        }
        
        return self.state, reward, done, info

    def reset(self):
        self.grid = self.grid_model.create_network()
        self.current_step = 0
        self.previous_violations = 0
        self.state = self._get_state()
        return self.state

    def _apply_action(self, action):
        for i, adjustment in enumerate(action):
            generator_name = self.grid.generators.index[i]
            current_p_nom = self.grid.generators.at[generator_name, "p_nom"]
            new_p_nom = current_p_nom * (1 + adjustment)
            new_p_nom = max(0, new_p_nom)  # Prevent negative output
            self.grid.generators.at[generator_name, "p_nom"] = new_p_nom

    def _get_state(self):
        generator_outputs = self.grid.generators["p_nom"].values
        load_demands = self.grid.loads["p_set"].values
        return np.concatenate([generator_outputs, load_demands])

    def _calculate_reward(self):
        # Demand-supply mismatch penalty
        total_demand = self.grid.loads["p_set"].sum()
        total_generation = self.grid.generators["p_nom"].sum()
        mismatch = abs(total_generation - total_demand)
        reward = -mismatch * 0.1
        
        # Thermal violations penalty
        if "loading" in self.grid.lines.columns:
            thermal_violations = sum(self.grid.lines["loading"] > 100)
            reward -= thermal_violations * 5
            
            # Reward for reducing violations
            violation_change = thermal_violations - self.previous_violations
            reward += max(0, -violation_change) * 2
            self.previous_violations = thermal_violations
            
        return float(reward)

    def _check_done(self):
        if self.current_step >= self.max_steps:
            return True
            
        if "loading" in self.grid.lines.columns:
            if any(self.grid.lines["loading"] > 150):  # Severe overload
                return True
                
        return False