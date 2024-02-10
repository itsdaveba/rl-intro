import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium import spaces, register


class ArmedBanditEnv(gym.Env):
    def __init__(self, k):
        self.k = k
        self.observation_space = spaces.Discrete(1)
        self.action_space = spaces.Discrete(self.k)
        self.q_star = None
        self.argmax = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.q_star = self.np_random.normal(size=self.k)
        self.argmax = np.argmax(self.q_star)
        return 0, {"argmax": self.argmax}

    def step(self, action):
        assert action in self.action_space
        reward = self.np_random.normal(self.q_star[action])
        return 0, reward, False, False, {"argmax": self.argmax}

    def render(self):
        samples = self.np_random.normal(self.q_star, size=(10000, self.k))
        plt.violinplot(samples, showmeans=True)
        plt.xlabel("Action")
        plt.xticks(range(1, self.k + 1))
        plt.ylabel("Reward distribution")
        plt.grid(True, axis="y")
        plt.show()


class ArmedBanditVectorEnv(gym.vector.VectorEnv):
    def __init__(self, num_envs, max_episode_steps, k):
        assert num_envs > 0
        self.max_episode_steps = max_episode_steps
        self.k = k
        observation_space = spaces.Discrete(1)
        action_space = spaces.Discrete(self.k)
        super().__init__(num_envs, observation_space, action_space)
        self.q_star = None
        self.argmax = None
        self.elapsed_steps = None

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        self.q_star = self.np_random.normal(size=(self.num_envs, self.k))
        self.argmax = np.argmax(self.q_star, axis=1)
        self._argmax = np.ones_like(self.argmax, dtype=bool)
        self.elapsed_steps = 0
        return np.zeros(self.num_envs, dtype=int), {
            "argmax": self.argmax,
            "_argmax": self._argmax
        }

    def step(self, action):
        assert action in self.action_space
        action = np.expand_dims(action, axis=1)
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
            {"argmax": self.argmax, "_argmax": self._argmax}
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
    id="ArmedBandit-v0",
    entry_point=ArmedBanditEnv,
    max_episode_steps=1000,
    vector_entry_point=ArmedBanditVectorEnv
)
