import gym
from gym import spaces
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class FollowTheLeaderEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    # Main API
    def __init__(self, n=2, render_mode=None):
        self.MIN = 0 # min value of x
        self.MAX = 100 # max value of x
        self.SPEED = 1 # how fast we can move
        self.DISTANCE = 100
        self.WINDOW_SIZE = 0.1

        assert n > 0

        self.num_beams = n # number of beams
        self.leader_x = None
        self.follower_x = None
        self.leader_circle = None
        self.follower_circle = None

        # action space: left or right, discrete
        self.action_space = spaces.Discrete(2)

        # observation space array of length n with 1s or 0s.
        self.observation_space = spaces.MultiBinary(2 * self.num_beams + 1)

        self.fig, self.ax = plt.subplots()

    def reset(self, seed=None, options=None):
        self._seed(seed)

        self.leader_x = np.random.randint(self.MIN+1, self.MAX)
        self.current_leader_direction = np.random.choice([-1, 1])
        self.follower_x = np.random.randint(self.MIN+1, self.MAX)

        self.leader_circle = None
        self.follower_circle = None

        self.ax.clear()

        self.ax.set_xlim([0, 100])
        self.ax.set_ylim([0, 100])

        # caluclate the angle for our beams. Assumption: equally spaced out. We always have 1 beam straight forward.
        incr = 90/(self.num_beams+1)

        lesser_angles = []
        for i in range(1, self.num_beams + 1):
            lesser_angles.append(i * incr)

        larger_angles = [x + 90 for x in lesser_angles.copy()]

        self.rad_list = [math.radians(x) for x in lesser_angles + [90.0] + larger_angles]

        obs = self._get_obs()

        info = self._get_info()

        return obs, info

    def step(self, action):
        # process the action of the network
        direction = 1 if action == 1 else -1
        self._move_follower(direction)

        # calculate reward
        reward = self._get_reward()

        # move the leader
        self._move_leader()

        obs = self._get_obs()

        info = self._get_info()

        # get done?
        done = False

        return obs, reward, done, info

    def render(self):
        if self.leader_circle is None:
            self.leader_circle = patches.Circle((self.leader_x, 80), radius=2, color="r")
        if self.follower_circle is None:
            self.follower_circle = patches.Circle((self.follower_x, 20), radius=2, color="b")

            self.ax.add_patch(self.leader_circle)
            self.ax.add_patch(self.follower_circle)


        self.leader_circle.center = (self.leader_x, 80)
        self.follower_circle.center = (self.follower_x, 20)
        plt.draw()
        plt.pause(0.00001)

    def close(self):
        pass

    # Helpers
    def _get_obs(self):
        obs = [0] * (2 * self.num_beams + 1)

        for i, rad in enumerate(self.rad_list):
            beam_x = self.DISTANCE * math.tan(rad) + self.follower_x

            if abs(self.leader_x - beam_x) < self.WINDOW_SIZE or (math.degrees(rad) == 90 and abs(self.leader_x - self.follower_x) < self.WINDOW_SIZE):
                obs[i] = 1
                break

        assert len(obs) == 2*self.num_beams + 1
        return np.array(obs, dtype=np.int8)

    def _move_follower(self, direction):
        # -1 = left, +1 = right
        if self.follower_x >= self.MAX and direction == 1:
            self.follower_x = self.MAX
        elif self.follower_x <= self.MIN and direction == -1:
            self.follower_x = self.MIN
        else:
            self.follower_x += direction * self.SPEED

    def _move_leader(self):
        if self.leader_x <= self.MIN and self.current_leader_direction == -1:
            self.current_leader_direction = 1

        if self.leader_x >= self.MAX and self.current_leader_direction == 1:
            self.current_leader_direction = -1

        self.leader_x += self.SPEED * self.current_leader_direction

    def _get_reward(self):
        if abs(self.leader_x - self.follower_x) <= 1:
            return 1
        else:
            return 1/abs(self.leader_x - self.follower_x)

    def _get_info(self):
        return {}

    def _seed(self, seed):
        if seed != None:
            spaces.Space.seed(seed)
            np.random.seed(seed)
