from itertools import product

import numpy as np
import gymnasium as gym
from gymnasium import spaces, register


class GridWorld(gym.Env):
    def __init__(self, shape, reward_dynamics=None):
        self.shape = shape
        self.observation_space = spaces.MultiDiscrete(self.shape)
        self.action_space = spaces.Discrete(4)

        self.rd_keys = list(reward_dynamics.keys()) if reward_dynamics is not None else []
        self.rd_from = [reward_dynamics[key]["from"] for key in self.rd_keys]
        self.rd_to = {reward_dynamics[key]["from"]: reward_dynamics[key]["to"] for key in self.rd_keys}
        self.rd_rw = {reward_dynamics[key]["from"]: reward_dynamics[key]["reward"] for key in self.rd_keys}
        self.prob = np.zeros(self.shape + (self.action_space.n,) + self.shape, dtype=np.float32)  # transition probability
        self.rewards = np.zeros(self.shape + (self.action_space.n,), dtype=np.float32)  # expected rewards

        self.actions = np.array([[0, 1], [-1, 0], [0, -1], [1, 0]], dtype=np.int32)
        for state in product(*[range(i) for i in self.shape]):
            for action in range(self.action_space.n):
                if state in self.rd_from:
                    self.prob[state][action][self.rd_to[state]] = 1.0
                    self.rewards[state][action] = self.rd_rw[state]
                else:
                    new_state = state + self.actions[action]
                    if np.any(new_state // self.shape):
                        self.prob[state][action][state] = 1.0
                        self.rewards[state][action] = -1.0
                    else:
                        self.prob[state][action][tuple(new_state)] = 1.0
        self.state = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.np_random.integers(self.shape)
        return self.state, {}

    def step(self, action):
        assert action in self.action_space

        if tuple(self.state) in self.rd_from:
            new_state = np.array(self.rd_to[tuple(self.state)])
            reward = self.rd_rw[tuple(self.state)]
        else:
            new_state = self.state + self.actions[action]
            if np.any(new_state // self.shape):
                new_state = self.state
                reward = -1.0
            else:
                reward = 0.0
        self.state = new_state

        return self.state, reward, False, False, {}

    def render(self):
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                k = 0
                if tuple(self.state) == (i, j):
                    print("x", end=" ")
                else:
                    for rd_from in self.rd_from:
                        if np.all(rd_from == (i, j)):
                            print(self.rd_keys[self.rd_from.index(rd_from)], end=" ")
                            break
                        if np.all(self.rd_to[rd_from] == (i, j)):
                            print(f"{self.rd_keys[self.rd_from.index(rd_from)]}'", end="")
                            break
                        k += 1
                if k == len(self.rd_keys) and not tuple(self.state) == (i, j):
                    print(".", end=" ")
            print()


register(
    id="GridWorld-v0",
    entry_point=GridWorld
)
