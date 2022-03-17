# Import the Necessary Packages
import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from doubleDuelingDQN_agent import Agent
from qnet import QNetwork

from pyvirtualdisplay import Display

env = gym.make('LunarLander-v2')
env.seed(0);

agent = Agent(state_size=8, action_size=4, seed=0)


# load the weights from file
agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

for i in range(5):
    state = env.reset()
    img = plt.imshow(env.render(mode='rgb_array'))
    for j in range(600):
        action = agent.act(state)
        img.set_data(env.render(mode='rgb_array')) 
        plt.axis('off')
        state, reward, done, _ = env.step(action)
        if done:
            break 
            
env.close()