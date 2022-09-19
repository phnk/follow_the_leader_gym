import gym
from gym import spaces
import numpy as np

class FollowTheLeaderEnv(gym.Env):

    # Main API
    def __init__(self, render_mode=None):
        pass

    def reset(self, seed=None, options=None):
        pass

    def step(self, action):
        pass

    def render(self):
        pass

    def close(self):
        pass

    # Helpers
    def _get_obs(self):
        pass

    def _get_info(self):
        pass
