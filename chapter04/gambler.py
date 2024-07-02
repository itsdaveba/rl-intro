import numpy as np
import gymnasium as gym
from gymnasium import spaces, register

GOAL = 100


class Gambler(gym.Env):
    def __init__(self, prob_heads):
        self.prob_heads = prob_heads
        self.terminal_states = [0, GOAL]
        self.observation_space = spaces.Discrete(GOAL + 1)
        self.action_space = spaces.Discrete(GOAL // 2)

        self.prob = np.zeros((self.observation_space.n, self.action_space.n, self.observation_space.n), dtype=np.float32)  # transition probability
        self.rewards = np.zeros((self.observation_space.n, self.action_space.n, self.observation_space.n), dtype=np.float32)  # expected rewards

        for state in range(GOAL + 1):
            if state in self.terminal_states:
                self.prob[state, :, state] = 1.0
            a_max = min(state, GOAL - state)
            for action in range(a_max):
                new_states = state + np.array([action + 1, -action - 1])
                self.prob[state][action][new_states] = (self.prob_heads, 1.0 - self.prob_heads)
                if new_states[0] == GOAL:
                    self.rewards[state][action][new_states[0]] = self.prob_heads
        self.rewards = np.divide(self.rewards, self.prob, out=np.zeros_like(self.prob), where=self.prob != 0.0)
        self.state = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.np_random.integers(GOAL + 1)
        while self.state in self.terminal_states:
            self.state = self.np_random.integers(GOAL + 1)
        return self.state, {}

    def step(self, action):
        if self.state in self.terminal_states:
            return self.state, 0.0, True, False, {}

        a_max = min(self.state, GOAL - self.state)
        assert action >= 0 and action < a_max

        heads = self.np_random.random() < self.prob_heads
        self.state += action + 1 if heads else -action - 1
        reward = 1.0 if self.state == GOAL else 0.0
        terminated = self.state in self.terminal_states

        return self.state, reward, terminated, False, {}

    def render(self):
        pass


register(
    id="Gambler-v0",
    entry_point=Gambler
)
