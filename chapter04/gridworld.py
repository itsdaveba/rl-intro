from itertools import product

import numpy as np
import gymnasium as gym
from gymnasium import spaces, register


class GridWorld(gym.Env):
    def __init__(self, shape, terminal_states=None):
        self.shape = shape
        self.terminal_states = terminal_states if terminal_states is not None else []
        self.observation_space = spaces.MultiDiscrete(self.shape)
        self.action_space = spaces.Discrete(4)

        self.prob = np.zeros(self.shape + (self.action_space.n,) + self.shape, dtype=np.float32)
        self.rewards = np.zeros(self.shape + (self.action_space.n,) + self.shape, dtype=np.float32)  # expected rewards

        self.actions = np.array([[0, 1], [-1, 0], [0, -1], [1, 0]], dtype=np.int32)
        for state in product(*[range(i) for i in self.shape]):
            for action in range(self.action_space.n):
                new_state = state + self.actions[action]
                if state in self.terminal_states:
                    self.prob[state][action][state] = 1.0
                    self.rewards[state][action][state] = 0.0
                elif np.any(new_state // self.shape):
                    self.prob[state][action][state] = 1.0
                    self.rewards[state][action][state] = -1.0
                else:
                    self.prob[state][action][tuple(new_state)] = 1.0
                    self.rewards[state][action][tuple(new_state)] = -1.0
        self.state = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.np_random.integers(self.shape)
        return self.state, {}

    def step(self, action):
        assert action in self.action_space

        new_state = self.state + self.actions[action]
        if tuple(self.state) in self.terminal_states or np.any(new_state // self.shape):
            new_state = self.state
        reward = 0.0 if tuple(self.state) in self.terminal_states else -1.0
        terminated = tuple(new_state) in self.terminal_states
        self.state = new_state

        return self.state, reward, terminated, False, {}

    def render(self):
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if tuple(self.state) == (i, j):
                    print("x", end=" ")
                elif self.terminal_states is not None and (i, j) in self.terminal_states:
                    print("T", end=" ")
                else:
                    print(".", end=" ")
            print()


register(
    id="GridWorld-v1",
    entry_point=GridWorld
)
