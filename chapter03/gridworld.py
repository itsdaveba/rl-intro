from itertools import product

import numpy as np
import gymnasium as gym
from gymnasium import spaces, register


class GridWorld(gym.Env):
    def __init__(self, shape, reward_dynamics=None):
        self.shape = shape
        self.observation_space = spaces.MultiDiscrete(self.shape)
        self.action_space = spaces.Discrete(4)

        base_rewards = [0, -1]
        rd_keys = list(reward_dynamics.keys()) if reward_dynamics is not None else []
        rd_from = [reward_dynamics[key]["from"] for key in rd_keys]
        self.rewards = base_rewards + [reward_dynamics[key][1] for key in rd_keys]
        self.prob = np.zeros(self.shape + (self.action_space.n,) + self.shape + (len(self.rewards),), dtype=np.float32)

        actions = np.array([[0, 1], [-1, 0], [0, -1], [1, 0]], dtype=np.int32)
        for state in product(*[range(i) for i in self.shape]):
            for action in range(self.action_space.n):
                if state in rd_from:
                    self.prob[state][action][reward_dynamics[state][0]][rd_keys.index(state) + len(base_rewards)] = 1.0
                else:
                    new_state = state + actions[action]
                    if np.any(new_state // self.shape):
                        self.prob[state][action][state][1] = 1.0
                    else:
                        self.prob[state][action][tuple(new_state)][0] = 1.0
        self.state = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.observation_space.seed(seed)
        self.state = self.observation_space.sample()
        return self.state, {}

    def step(self, action):
        prob = self.prob[tuple(self.state)][action]
        index = self.np_random.choice(prob.size, p=prob.flatten())
        state, reward_index = np.divmod(index, len(self.rewards))
        reward = self.rewards[reward_index]
        self.state = np.array(np.divmod(state, self.shape[1]))
        return self.state, reward, False, False, {"prob": prob[tuple(self.state)][reward_index]}

    def render(self):
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                print("x" if np.all(self.state == (i, j)) else ".", end=" ")
            print()


register(
    id="GridWorld-v0",
    entry_point=GridWorld
)
