import numpy as np


class BaseAgent:
    def __init__(self, k, epsilon, num_envs=1):
        assert num_envs > 0
        self.k = k
        self.epsilon = epsilon
        self.num_envs = num_envs
        self.np_random = None
        self.Q = None

    def reset(self, *, seed=None):
        if self.np_random is None or seed is not None:
            self.np_random = np.random.default_rng(seed)
        self.Q = np.zeros((self.num_envs, self.k))

    def predict(self, observation=None):
        action = np.where(
            self.np_random.random(size=self.num_envs) < self.epsilon,
            self.np_random.choice(self.k, size=self.num_envs),
            np.argmax(self.Q, axis=1)
        )
        return action

    def update(self):
        raise NotImplementedError


class SampleAverageAgent(BaseAgent):
    def __init__(self, k, epsilon, num_envs=1):
        super().__init__(k, epsilon, num_envs)
        self.N = None

    def reset(self, *, seed=None):
        super().reset(seed=seed)
        self.N = np.zeros((self.num_envs, self.k), dtype=int)

    def update(self, action, reward):
        action = np.expand_dims(action, axis=1)
        reward = np.expand_dims(reward, axis=1)
        Q = np.take_along_axis(self.Q, action, axis=1)
        N = np.take_along_axis(self.N, action, axis=1)
        np.put_along_axis(self.N, action, N + 1, axis=1)
        np.put_along_axis(self.Q, action, Q + (reward - Q) / (N + 1), axis=1)


class StepSizeAgent(BaseAgent):
    def __init__(self, k, epsilon, alpha, num_envs=1):
        super().__init__(k, epsilon, num_envs)
        self.alpha = alpha

    def update(self, action, reward):
        action = np.expand_dims(action, axis=1)
        reward = np.expand_dims(reward, axis=1)
        Q = np.take_along_axis(self.Q, action, axis=1)
        np.put_along_axis(self.Q, action, Q + self.alpha * (reward - Q), axis=1)
