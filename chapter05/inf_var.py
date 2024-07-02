import gymnasium as gym
from gymnasium import spaces, register


class InfiniteVar(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Discrete(2)
        self.action_space = spaces.Discrete(2)

        self.state = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = 0
        return self.state, {}

    def step(self, action):
        assert action in self.action_space

        reward = 0.0
        terminated = True
        if self.state != 1:
            if action:  # right
                self.state = 1
            else:  # left
                if self.np_random.random() < 0.1:
                    self.state = 1
                    reward = 1.0
                else:
                    terminated = False

        return self.state, reward, terminated, False, {}

    def render(self):
        pass


register(
    id="InfiniteVar-v0",
    entry_point=InfiniteVar
)
