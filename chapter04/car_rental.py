from itertools import product

import numpy as np
import gymnasium as gym
from gymnasium import spaces, register
from scipy.stats import poisson

RENT_REWARD = 10.0
ACTION_COST = 2.0

LAMBDA_RENTALS = [3, 4]
LAMBDA_RETURNS = [3, 2]

MAX_CARS = 20
MAX_ACTION = 5

MAX_RENTALS = 12
MAX_RETURNS = 12


class CarRental(gym.Env):
    def __init__(self):
        self.shape = (MAX_CARS + 1, MAX_CARS + 1)
        self.observation_space = spaces.MultiDiscrete(self.shape)
        self.action_space = spaces.Discrete(2 * MAX_ACTION + 1, start=-MAX_ACTION)

        self.prob = np.zeros(self.shape + (self.action_space.n,) + self.shape, dtype=np.float32)
        self.rewards = np.zeros(self.shape + (self.action_space.n,) + self.shape, dtype=np.float32)  # expected rewards

        rentals_pmf = poisson.pmf([[i, i] for i in range(MAX_RENTALS + 1)], mu=LAMBDA_RENTALS)
        returns_pmf = poisson.pmf([[i, i] for i in range(MAX_RETURNS + 1)], mu=LAMBDA_RETURNS)

        for state in product(*[range(i) for i in self.shape]):
            a_min = max(-state[1], state[0] - MAX_CARS)
            a_max = min(state[0], MAX_CARS - state[1])
            for action in range(max(-MAX_ACTION, a_min), min(MAX_ACTION, a_max) + 1):
                state_ = np.array(state) + [-action, action]
                reward_ = -ACTION_COST * abs(action)
                for rentals in product(*[range(MAX_RENTALS + 1)] * 2):
                    prob_rent = np.prod(np.diag(rentals_pmf[list(rentals)]))
                    rent = np.minimum(rentals, state_)
                    state__ = state_ - rent
                    reward = reward_ + RENT_REWARD * rent.sum()
                    for returns in product(*[range(MAX_RETURNS + 1)] * 2):
                        prob = prob_rent * np.prod(np.diag(returns_pmf[list(returns)]))
                        ret = np.minimum(returns, MAX_CARS - state__)
                        new_state = state__ + ret
                        self.prob[state][action][tuple(new_state)] += prob
                        self.rewards[state][action][tuple(new_state)] += prob * reward
        self.rewards = np.divide(self.rewards, self.prob, out=np.zeros_like(self.prob), where=self.prob != 0.0)
        self.state = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.observation_space.seed(seed)
        self.state = self.observation_space.sample()
        return self.state, {}

    def step(self, action):
        prob = self.prob[tuple(self.state)][action]
        index = self.np_random.choice(prob.size, p=prob.flatten())
        new_state = np.array(np.divmod(index, self.shape[1]))
        reward = self.rewards[tuple(self.state)][action][tuple(new_state)]
        self.state = new_state
        return new_state, reward, False, False, {"prob": prob[tuple(new_state)]}


register(
    id="CarRental-v0",
    entry_point=CarRental
)
