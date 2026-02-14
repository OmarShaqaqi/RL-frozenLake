import numpy as np
import random
class QLearning:
    
    def __init__(self,state_size,action_size,learning_rate,gamma):
        
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.reset_qtable()
        
        
    def reset_qtable(self) : 
        """
        initializing Q-table
        """
        self.q_table = np.zeros((self.state_size,self.action_size))
        
    def update(self,state, action, reward, new_state) : 
        """
        Updating Q-table value
        """
        self.q_table[state,action] = self.q_table[state,action]  \
                                     + self.learning_rate \
                                     * (reward + self.gamma * np.max(self.q_table[new_state,:]) - self.q_table[state,action] )
                                     
class EpsilonGreedy:
    
    def __init__(self,epsilon):
        self.epsilon = epsilon
        
    def choose_action(self, action_space, state, qtable):
        """Choose an action `a` in the current world state (s)."""
        # First we randomize a number
        explor_exploit_tradeoff = random.random()

        # Exploration
        if explor_exploit_tradeoff < self.epsilon:
            action = action_space.sample()

        # Exploitation (taking the biggest Q-value for this state)
        else:
            # Break ties randomly
            # If all actions are the same for this state we choose a random one
            # (otherwise `np.argmax()` would always take the first one)
            if np.all(qtable[state, :]) == qtable[state, 0]:
                action = action_space.sample()
            else:
                action = np.argmax(qtable[state, :])
        return action