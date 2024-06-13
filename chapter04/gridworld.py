from itertools import product

import numpy as np
import gymnasium as gym
from gymnasium import spaces, register


class GridWorld(gym.Env):
    def __init__(self, shape, terminal_states=None):
        self.shape = shape
        self.terminal_states = terminal_states
        self.observation_space = spaces.MultiDiscrete(self.shape)
        self.action_space = spaces.Discrete(4)

        base_reward = [0, -1]
        self.rewards = base_reward
        self.prob = np.zeros(self.shape + (self.action_space.n,) + self.shape + (len(self.rewards),), dtype=np.float32)

        actions = np.array([[0, 1], [-1, 0], [0, -1], [1, 0]], dtype=np.int32)
        for state in product(*[range(i) for i in self.shape]):
            for action in range(self.action_space.n):
                new_state = state + actions[action]
                if state in self.terminal_states:
                    self.prob[state][action][state][0] = 1.0
                elif np.any(new_state // self.shape):
                    self.prob[state][action][state][1] = 1.0
                else:
                    self.prob[state][action][tuple(new_state)][1] = 1.0
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
        terminated = tuple(self.state) in self.terminal_states if self.terminal_states is not None else False
        return self.state, reward, terminated, False, {"prob": prob[tuple(self.state)][reward_index]}

    def render(self):
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                k = 0
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
