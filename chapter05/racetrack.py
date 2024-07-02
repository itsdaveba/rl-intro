import gymnasium as gym
from gymnasium import spaces, register


class Racetrack(gym.Env):
    def __init__(self):
        self.shape = (3, 3)
        self.observation_space = spaces.MultiDiscrete(self.shape)
        self.action_space = spaces.MultiDiscrete((3, 3))
        self.state = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.np_random.integers(self.shape)
        return self.state, {}

    def step(self, action):
        assert action in self.action_space
        return self.state, 0.0, False, False, {}

    def render(self):
        pass


register(
    id="Racetrack-v0",
    entry_point=Racetrack
)
