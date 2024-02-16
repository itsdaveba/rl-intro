import numpy as np

np.seterr(divide="ignore", invalid="ignore")


class BaseAgent:
    def __init__(self, k, num_envs):
        assert num_envs > 0
        self.k = k
        self.num_envs = num_envs
        self.np_random = None
        self.Q = None

    def reset(self, *, seed=None):
        if self.np_random is None or seed is not None:
            self.np_random = np.random.default_rng(seed)
        self.Q = np.zeros((self.num_envs, self.k))

    def predict(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError


class EpsilonGreedyAgent(BaseAgent):
    def __init__(self, k, num_envs, epsilon):
        super().__init__(k, num_envs)
        self.epsilon = epsilon

    def predict(self):
        action = np.where(
            self.np_random.random(size=self.num_envs) < self.epsilon,
            self.np_random.choice(self.k, size=self.num_envs),
            np.argmax(self.Q, axis=1)
        )
        return action


class SampleAverageAgent(EpsilonGreedyAgent):
    def __init__(self, k, num_envs, epsilon):
        super().__init__(k, num_envs, epsilon)
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


class ConstantStepSizeAgent(EpsilonGreedyAgent):
    def __init__(self, k, num_envs, epsilon, alpha, init_val=0.0):
        super().__init__(k, num_envs, epsilon)
        self.alpha = alpha
        self.init_val = init_val

    def reset(self, *, seed=None):
        super().reset(seed=seed)
        self.Q += self.init_val

    def update(self, action, reward):
        action = np.expand_dims(action, axis=1)
        reward = np.expand_dims(reward, axis=1)
        Q = np.take_along_axis(self.Q, action, axis=1)
        np.put_along_axis(self.Q, action, Q + self.alpha * (reward - Q), axis=1)


class UpperConfidenceBoundAgent(SampleAverageAgent):
    def __init__(self, k, num_envs, c):
        super().__init__(k, num_envs, None)
        self.c = c
        self.t = None

    def reset(self, *, seed=None):
        super().reset(seed=seed)
        self.t = 1

    def predict(self):
        exploration = np.sqrt(np.log(self.t) / self.N)
        action = np.argmax(self.Q + self.c * exploration, axis=1)
        self.t += 1
        return action


class GradientAgent(BaseAgent):
    def __init__(self, k, num_envs, alpha, use_baseline=False):
        super().__init__(k, num_envs)
        self.alpha = alpha
        self.use_baseline = use_baseline
        self.pi = None
        self.baseline = None
        self.n = None

    def reset(self, *, seed=None):
        super().reset(seed=seed)
        self.pi = np.ones((self.num_envs, self.k)) / self.k
        self.baseline = np.zeros((self.num_envs, 1))
        self.n = 0

    def predict(self):
        cumpi = np.cumsum(self.pi, axis=1)
        rand = self.np_random.random(size=(self.num_envs, 1))
        action = np.argmax(rand < cumpi, axis=1)
        return action

    def update(self, action, reward):
        reward = np.expand_dims(reward, axis=1)
        if self.use_baseline:
            self.n += 1
            self.baseline += (reward - self.baseline) / self.n  # baseline should not include last reward
        update = (reward - self.baseline) * (np.eye(self.k)[action] - self.pi)
        self.Q += self.alpha * update
        self.pi = np.exp(self.Q) / np.sum(np.exp(self.Q), axis=1, keepdims=True)
