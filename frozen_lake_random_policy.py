import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random

from QLearning import QLearning, EpsilonGreedy

STATE_SIZE = 16
ACTION_SIZE = 4
LEARNING_RATE = 0.1
GAMMA = 0.9
EPISODES = 100
RENDER_MODE = "rgb_array"



learner = QLearning(STATE_SIZE,ACTION_SIZE,LEARNING_RATE,GAMMA)
explorer = EpsilonGreedy(1)



def run(num_epochs) : 
    env = gym.make("FrozenLake-v1", render_mode= RENDER_MODE)
    
    success = 0
    
    
    
    
    for _ in range(num_epochs) :
        state = env.reset()[0]
        terminated, truncated = False, False
        
        while not terminated and not truncated :
            action = explorer.choose_action(env.action_space, state, learner.q_table)
            new_state, reward, terminated, truncated, _ = env.step(action)
            learner.update(state, action, reward, new_state)
            
            if reward == 1 :
                success += 1
            
            state = new_state
            env.render()
            
    print(f"Success rate : {success/num_epochs}")
        
        
        
run(EPISODES)