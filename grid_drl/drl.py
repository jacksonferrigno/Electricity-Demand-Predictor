import gymnasium as gym
from gymnasium import spaces
import numpy as np
import copy

class PowerGridEnv(gym.Env):
    def __init__(self, grid, max_steps=100, action_limit=0.1):
        super(PowerGridEnv, self).__init__()
        
        self.grid_model = grid
        self.grid = grid.network
        self.max_steps = max_steps
        self.action_limit = action_limit
        self.grid.loads["p_set"] = self.grid.loads["p_set"].astype(np.float64)
        self.initial_demand = self.grid.loads["p_set"].copy()
        
        #counts from the grid 
        self.num_generators = len(self.grid.generators)
        self.num_transformers= len(self.grid.transformers) if "transformers" in self.grid.__dict__ or not self.grid.transformers.empty else 0
        self.num_loads = len(self.grid.loads)
        
        # Action space: Generator output adjustments
        self.total_actions = self.num_generators +self.num_transformers + self.num_loads
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(self.total_actions,),
            dtype=np.float32
        )
        
        # State space: Generator outputs + load demands
        self.state_size = 2* self.num_generators + self.num_loads+self.num_transformers
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
        
        # Reset the grid from the original model.
        self.grid = copy.deepcopy(self.grid_model.network)
        
        print("Loads in reset:", self.grid.loads["p_set"])
        self.current_step = 0
        self.previous_violations = 0
        self.state = self._get_state().astype(np.float32)
        return self.state, {}




    def _apply_action(self, action):
        """Action vector updates: gen output, transformer tap positions, and load shedding 
                First: generator adjustments
                Second: Transformer tap adjustments
                Third: load shedding adjustments
        """
        # segment the vector
        gen_actions = action[:self.num_generators]
        trans_actions = action[self.num_generators:self.num_generators +self.num_transformers]
        shed_actions= action[self.num_generators+self.num_transformers:]
        
        # GEN ADJUSTMENTS
        scaled_gen_actions = gen_actions *self.action_limit
        for i, adj in enumerate(scaled_gen_actions):
            # get the gnerator name 
            gen_name= self.grid.generators.index[i]
            # current output of generator
            current = self.grid.generators.at[gen_name, "p_nom"]
            #ramp limit (rate of change per step)
            ramp = self.grid.generators.at[gen_name, "ramp_limit_up"]
            # determine the capacity 
            capacity = (self.grid.generators.at[gen_name, "p_nom_max"]
                        if "p_nom_max" in self.grid.generators.columns
                        else self.grid.generators.at[gen_name,"p_nom"])
            # calculate the min allowed in output based on a % of capacity 
            min_output = self.grid.generators.at[gen_name, "p_min_pu"]*capacity
            
            # calculate the change  allowed in output based on a % of capacity 
            delta= current* adj
            # limit change by ramp limit 
            max_delta= ramp *capacity
            delta=np.clip(delta,-max_delta,max_delta)
            # compute input and make sure we didnt make a mistake
            new_output= np.clip(current+delta, min_output, capacity)
            self.grid.generators.at[gen_name,"p_nom"] = new_output
            
        # TRANSFORMER TAP ADJUSTMENTS
        for i, tap_adj in enumerate(trans_actions):
            #get transmformer name by index
            trans_name= self.grid.transformers.index[i]
            # current tap position
            current_tap = self.grid.transformers.at[trans_name, "tap_pos"]
            # lower and upper limits of what we can do 
            tap_min = self.grid.transformers.at[trans_name, "tap_min"]
            tap_max = self.grid.transformers.at[trans_name, "tap_max"]
            # determine the adjustment if we dont have it just default to 2.5
            tap_step = (self.grid.transformers.at[trans_name, "tap_step_percent"]
                        if "tap_step_percent" in self.grid.transformers.columns
                        else 2.5)
            #ensure we stay within range
            new_tap = np.clip(current_tap+tap_adj *tap_step,tap_min,tap_max)
            #update 
            self.grid.transformers.at[trans_name,"tap_pos"]= new_tap
            
        # LOAD SHEDDING (up to 20%)
        max_shedding =0.2
        for i, shed_act in enumerate(shed_actions):
            #name by index
            load_name=self.grid.loads.index[i]
            # map action from [-1,1] to [0,max_shedding]
            shedding_fractions= ((shed_act+1)/2) *max_shedding
            #store base load if we dont have it yet
            if "p_base" not in self.grid.loads.columns:
                self.grid.loads["p_base"] = self.grid.loads["p_set"]
            base_load = self.grid.loads.at[load_name, "p_base"]
            self.grid.loads.at[load_name,"p_set"]= base_load*(1-shedding_fractions)
            
        
        
        
        
    def _get_state(self):
        generator_outputs = self.grid.generators["p_nom"].values
        load_demands = self.grid.loads["p_set"].values
        marginal_cost= self.grid.generators["marginal_cost"].values


        # Avoid division by zero by setting a safe normalization factor
        max_gen_output = max(np.max(generator_outputs), 1e-6)
        max_load = max(np.max(load_demands), 1e-6)

        # Normalize to [-1, 1] range
        generator_outputs = (generator_outputs / max_gen_output) * 2 - 1
        load_demands = (load_demands / max_load) * 2 - 1
        marginal_cost=(marginal_cost/100.0)*2-1
        
        
        #normalize transformer tap postions
        if self.num_transformers:
            tap_norm_list=[]
            for name in self.grid.transformers.index:
                tap = self.grid.transformers.at[name, "tap_pos"]
                tap_min = self.grid.transformers.at[name, "tap_min"]
                tap_max = self.grid.transformers.at[name, "tap_max"]
                norm_value= 2* ((tap-tap_min)/(tap_max-tap_min))-1
                tap_norm_list.append(norm_value)
            transformer_tap_norm= np.array(tap_norm_list)
        else:
            transformer_tap_norm=np.array([])
            
        #concatinate shape
        state = np.concatenate([generator_outputs, load_demands, marginal_cost, transformer_tap_norm]).astype(np.float32)

        print(f"Final state shape (should match observation space {self.state_size}): {state.shape}")

        return state
    def _calculate_reward(self):
        # Demand-supply mismatch penalty
        prize=0
        punishment=0
        reward=0
        
        total_demand = self.grid.loads["p_set"].sum()
        total_generation = self.grid.generators["p_nom"].sum()
        mismatch = abs(total_generation - total_demand)
        #supply and demand balance -> reward
        oversupply=(total_generation-total_demand)
        if 0< oversupply < 200:
            prize+=1500 
        elif mismatch<50:
            prize+=1000
        if oversupply >=205:
            punishment+=(oversupply-205)*.1
            
        #punishment if imbalance 
        if mismatch>500:
            punishment+=200
        
        
        # Thermal violations penalty harsh penalty 
        if "loading" in self.grid.lines.columns:
            thermal_violations = sum(self.grid.lines["loading"] > 100)
            punishment+=thermal_violations*4.5 
            

        #punishment for load shedding ---more load shedding =bad --- 
        if "p_base" in self.grid.loads.columns:
            shed_total = sum(self.grid.loads["p_base"]-self.grid.loads["p_set"])
            punishment += shed_total *0.01 # weight on shedding loads

            
        #punishment for grid stability 
        gen_variability = np.std(self.grid.generators["p_nom"].values)
        punishment +=gen_variability *0.025 #  penalty for instability
        
        if gen_variability<5:
            prize+=175
        
        #marginal cost penalty
        gen_cost= sum(
            self.grid.generators.loc[gen,"p_nom"] * self.grid.generators.loc[gen,"marginal_cost"]
            for gen in self.grid.generators.index
        )
        punishment+= 0.00002 * gen_cost
        # Additional bonus for minimal load shedding.
        if "p_base" in self.grid.loads.columns:
            if (self.grid.loads["p_base"] - self.grid.loads["p_set"]).sum() < 5:
                prize += 225

        # Bonus if there are no thermal overloads.
        if "loading" in self.grid.lines.columns:
            if (self.grid.lines["loading"] > 100).sum() == 0:
                prize += 200
        reward = prize-punishment
        print(f"=======PRIZE REPORT=======")
        print(f"Reward {reward:.2f} Demand: {total_demand:.2f}, Supply {total_generation:.2f}")
        print("="*25)
        return float(reward)

    def _check_done(self):
        if self.current_step >= self.max_steps:
            return True
            
        if "loading" in self.grid.lines.columns:
            if any(self.grid.lines["loading"] > 150):  # Severe overload
                return True
                
        return False