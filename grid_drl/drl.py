import random
import numpy as np
from collections import deque

import tensorflow as tf

class DQNAgent:
    def __init__(self,state_size, action_size, grid):
        """_summary_

        Args:
            state_size : Demension of state space (vector)
            action_size : Number of possible actions
        """
        self.state_size = state_size
        self.action_size =action_size
        
        # replay buffer - deque that stores past expereinces for training network
        self.memory = deque(maxlen=2000)
        
        #params for training 
        self.gamma =0.95 #determines how much future reward is worth compared to immediate reward
        
        self.epsilon =1.0  #exploration rate (prob of random events)
        self.epsilon_decay =0.995 # factor that reduces epsilon over time
        self.epsilon_min = 0.01 #min value for epsilon
        self.learning_rate =0.001 # learning rate for optimzer
        
        self.model =self._build_model() #main q-network
        self.target_model =self._build_model() # target q network 
        self.update_target_model()
        #pass thru our grid
        self.grid =grid
        
        
    def _build_model(self):
        """ Builds the NN model for approx Q-values
        """
        model = tf.keras.Sequential([
            #input layer with state size
            tf.keras.layers.Dense(64, input_dim=self.state_size, activation='relu'),
            #hidden layer
            tf.keras.layers.Dense(64,activation='relu'),
            #output layer with action size
            tf.keras.Dense(self.action_size,activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss='mse')
        return model
    def update_target_model(self):
        """Updates weight of target network to match main network
        """
        self.target_model.set_weights(self.model.get_weights())
        
        
    
    def remember(self,state,action,reward,next_state,done):
        """Stores a single experience in buffer

        Args:
            state - Current state of enviro
            action - action taken by agent
            reward - Reward received after taking action
            next_state - next state of environment
            done - bool indicating if episode is finished
        """
        #oldest experience is auto removed 
        self.memory.append((state,action,reward,next_state,done))
        
        
    def replay(self,batch_size):
        """Train network using random batchg of past expereinces from buffer

        Args:
            batch_size -Num of expereinces to sample from buffer
        """
        # do we have enough 
        if len(self.memory) < batch_size:
            return 
        
        #random batch 
        minibatch = random.sample(self.memory,batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            #predict q values
            target = self.model.predict(state)
            if done:
                #terminal state: target q is just reward
                target[0][action]=reward
            else:
                #non terminal state: use Bellman equation to calc target Q-value
                t=self.target_model.predict(next_state)[0]
                target[0][action]=reward +self.gamma *np.amax(t)
            #train model on update target
            self.model.fit(state,target,epochs=1,verbose=0)
            
        #decay epsilon after training 
        if self.epsilon>self.epsilon_min:
            self.epsilon*=self.epsilon_decay
            
            

    def get_state(self):
        """gets current state of the grid
        """
        #line flows as % of thermal limits 
        line_flows= [
            abs(self.grid.network.lines_t.p0[line.name].iloc[-1])/line.s_nom
                if line.name in self.grid.network.lines_t.p0.columns and line.s_nom>0 else 0 
                for _, line in self.grid.network.lines.iterrows()
        ]
        # gen outputs as % of capacity 
        generator_outputs=[
            self.grid.network.generators_t.p[gen.name].iloc[-1]/gen.p_nom
            if gen.name in self.grid.network.generators_t.p.columns and gen.p_nom > 0 else 0
            for _, gen in self.grid.network.generators.iterrows()
        ]
        #total demand and supply
        total_demand = self.grid.network.loads.p_set.sum().sum()
        total_supply =self.grid.network.generators_t.p.sum().sum()
        
        #unmet demand
        unmet_demand = max(0,total_demand-total_supply)
        # Combine state components into a single vector
        state = np.array(line_flows + generator_outputs + [total_demand, total_supply, unmet_demand])
        return state
        
                
            
    