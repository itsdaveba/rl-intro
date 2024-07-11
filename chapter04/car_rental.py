from itertools import product

import numpy as np
import gymnasium as gym
from gymnasium import spaces, register
from scipy.stats import poisson

RENT_REWARD = 10.0
ACTION_COST = 2.0
PARKING_COST = 4.0

LAMBDA_RENTALS = [3, 4]
LAMBDA_RETURNS = [3, 2]

MAX_CARS = 20
MAX_ACTION = 5
MAX_PARKING = 10

MAX_RENTALS = 12
MAX_RETURNS = 12


class CarRental(gym.Env):
    def __init__(self, modified=False):
        self.modified = modified
        self.shape = (MAX_CARS + 1, MAX_CARS + 1)
        self.observation_space = spaces.MultiDiscrete(self.shape)
        self.action_space = spaces.Discrete(2 * MAX_ACTION + 1, start=-MAX_ACTION)

        self.prob = np.zeros(self.shape + (self.action_space.n,) + self.shape, dtype=np.float32)  # transition probability
        self.rewards = np.zeros(self.shape + (self.action_space.n,), dtype=np.float32)  # expected rewards

        rentals_pmf = poisson.pmf([[i, i] for i in range(MAX_RENTALS + 1)], mu=LAMBDA_RENTALS)
        returns_pmf = poisson.pmf([[i, i] for i in range(MAX_RETURNS + 1)], mu=LAMBDA_RETURNS)

        for state in product(*[range(i) for i in self.shape]):
            a_min = max(-state[1], state[0] - MAX_CARS)
            a_max = min(state[0], MAX_CARS - state[1])
            for action in range(max(-MAX_ACTION, a_min), min(MAX_ACTION, a_max) + 1):
                _state = np.array(state) + [-action, action]
                if self.modified and action > 0:
                    _reward = -ACTION_COST * (action - 1)
                else:
                    _reward = -ACTION_COST * abs(action)
                if self.modified:
                    _reward -= PARKING_COST * (np.maximum(_state - 1, 0) // MAX_PARKING).sum()
                for rentals in product(*[range(MAX_RENTALS + 1)] * 2):
                    prob_rent = np.prod(np.diag(rentals_pmf[list(rentals)]))
                    rent = np.minimum(rentals, _state)
                    __state = _state - rent
                    reward = _reward + RENT_REWARD * rent.sum()
                    for returns in product(*[range(MAX_RETURNS + 1)] * 2):
                        prob = prob_rent * np.prod(np.diag(returns_pmf[list(returns)]))
                        ret = np.minimum(returns, MAX_CARS - __state)
                        new_state = __state + ret
                        self.prob[state][action][tuple(new_state)] += prob
                        self.rewards[state][action] += prob * reward
        self.state = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.np_random.integers(self.shape)
        return self.state, {}

    def step(self, action):
        a_min = max(-self.state[1], self.state[0] - MAX_CARS)
        a_max = min(self.state[0], MAX_CARS - self.state[1])
        assert action >= max(-MAX_ACTION, a_min) and action <= min(MAX_ACTION, a_max)

        self.state += [-action, action]
        if self.modified and action > 0:
            reward = -ACTION_COST * (action - 1)
        else:
            reward = -ACTION_COST * abs(action)
        if self.modified:
            reward -= PARKING_COST * (np.maximum(self.state - 1, 0) // MAX_PARKING).sum()

        rentals = self.np_random.poisson(lam=LAMBDA_RENTALS)
        rent = np.minimum(rentals, self.state)
        self.state -= rent
        reward += RENT_REWARD * rent.sum()

        returns = self.np_random.poisson(lam=LAMBDA_RETURNS)
        ret = np.minimum(returns, MAX_CARS - self.state)
        self.state += ret

        return self.state, reward, False, False, {"rentals": rentals, "returns": returns}

    def render(self):
        pass


register(
    id="CarRental-v0",
    entry_point=CarRental
)
