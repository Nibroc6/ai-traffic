from gym import Env
from gym.spaces import MultiBinary, Dict
import numpy as np
import random
from main import *

class TrafficEnv(Env):
    def __init__(self):
        self.action_space = MultiBinary([len(nodes), 2])
        self.observation_space = Dict()
    def step(self):
        pass
    def render(self):
        pass
    def reset(self):
        pass
    
tenv = TrafficEnv()
print(tenv.observation_space.sample())