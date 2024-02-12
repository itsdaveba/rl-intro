import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium import spaces, register


class ArmedBanditEnv(gym.Env):
    def __init__(self, k, stationary=True, mean=0.0):
        self.k = k
        self.stationary = stationary
        self.mean = mean
        self.observation_space = spaces.Discrete(1)
        self.action_space = spaces.Discrete(self.k)
        self.q_star = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.stationary:
            self.q_star = self.np_random.normal(size=self.k)
        else:
            self.q_star = np.zeros(self.k)
        self.q_star += self.mean
        return 0, {"argmax": np.argmax(self.q_star)}

    def step(self, action):
        assert action in self.action_space
        if not self.stationary:
            self.q_star += self.np_random.normal(scale=0.01, size=self.k)
        reward = self.np_random.normal(self.q_star[action])
        return 0, reward, False, False, {"argmax": np.argmax(self.q_star)}

    def render(self):
        samples = self.np_random.normal(self.q_star, size=(10000, self.k))
        plt.violinplot(samples, showmeans=True)
        plt.xlabel("Action")
        plt.xticks(range(1, self.k + 1))
        plt.ylabel("Reward distribution")
        plt.grid(True, axis="y")
        plt.show()


class ArmedBanditVectorEnv(gym.vector.VectorEnv):
    def __init__(self, num_envs, max_episode_steps, k, stationary=True, mean=0.0):
        assert num_envs > 0
        observation_space = spaces.Discrete(1)
        action_space = spaces.Discrete(k)
        super().__init__(num_envs, observation_space, action_space)
        self.max_episode_steps = max_episode_steps
        self.k = k
        self.stationary = stationary
        self.mean = mean
        self.q_star = None
        self._argmax = None
        self.elapsed_steps = None

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        if self.stationary:
            self.q_star = self.np_random.normal(size=(self.num_envs, self.k))
        else:
            self.q_star = np.zeros((self.num_envs, self.k))
        self.q_star += self.mean
        self._argmax = np.ones(self.num_envs, dtype=bool)
        self.elapsed_steps = 0
        return np.zeros(self.num_envs, dtype=int), {
            "argmax": np.argmax(self.q_star, axis=1),
            "_argmax": self._argmax
        }

    def step(self, action):
        assert action in self.action_space
        action = np.expand_dims(action, axis=1)
        if not self.stationary:
            self.q_star += self.np_random.normal(scale=0.01, size=(self.num_envs, self.k))
        loc = np.take_along_axis(self.q_star, action, axis=1)
        reward = self.np_random.normal(loc)
        self.elapsed_steps += 1
        if self.elapsed_steps >= self.max_episode_steps:
            truncated = np.ones(self.num_envs, dtype=bool)
        else:
            truncated = np.zeros(self.num_envs, dtype=bool)
        return (
            np.zeros(self.num_envs, dtype=int),
            reward.squeeze(axis=1),
            np.zeros(self.num_envs, dtype=bool),
            truncated,
            {"argmax": np.argmax(self.q_star, axis=1), "_argmax": self._argmax}
        )

    def render(self, env_index=0):
        samples = self.np_random.normal(self.q_star[env_index], size=(10000, self.k))
        plt.violinplot(samples, showmeans=True)
        plt.xlabel("Action")
        plt.xticks(range(1, self.k + 1))
        plt.ylabel("Reward distribution")
        plt.grid(True, axis="y")
        plt.show()


def run_episode(env, agent, seed=None):
    env.reset(seed=seed)
    agent.reset(seed=seed)
    rewards = []
    optimals = []
    while True:
        action = agent.predict()
        _, reward, terminated, truncated, info = env.step(action)
        agent.update(action, reward)
        rewards.append(reward)
        optimals.append(action == info["argmax"])
        if np.any(terminated) or np.any(truncated):
            break
    return np.array(rewards), np.array(optimals)


register(
    id="ArmedBanditTestbed-v0",
    entry_point=ArmedBanditEnv,
    max_episode_steps=1000,
    vector_entry_point=ArmedBanditVectorEnv
)
