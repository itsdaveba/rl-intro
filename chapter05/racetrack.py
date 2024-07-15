import numpy as np
import gymnasium as gym
from gymnasium import spaces, register
import matplotlib.pyplot as plt

MAX_VEL = 4

TRACK = 0
START = 1
FINISH = 2
NO_TRACK = 3
PATH = 4


def get_init_state(np_random, map):
    state = np_random.choice(np.argwhere(map == START))
    vel = np.array([0, 0], dtype=int)
    return state, vel


class Racetrack(gym.Env):
    def __init__(self, map_file):
        with open(map_file, mode="r") as file:
            lines = file.readlines()
        self.map = np.array([[int(c) for c in line.split()] for line in lines])

        obs_nvec = self.map.shape + (2 * MAX_VEL + 1, 2 * MAX_VEL + 1)
        act_nvec = (3, 3)
        self.observation_space = spaces.MultiDiscrete(obs_nvec, start=[0, 0, -MAX_VEL, -MAX_VEL])
        self.action_space = spaces.MultiDiscrete(act_nvec, start=(-1, -1))

        self.padded_map = np.pad(self.map, MAX_VEL, constant_values=NO_TRACK)

        self.intersection_mask = {}
        for velx in range(MAX_VEL + 1):
            for vely in range(MAX_VEL + 1):
                mask = np.zeros((vely + 1, velx + 1), dtype=int)
                mask[0, 0] = 1
                x, y = (0, 0)
                for i in range(vely):
                    x += velx
                    mask[i + 1, int(np.floor(x / vely))] = 1
                    mask[i, int(np.ceil(x / vely))] = 1
                for j in range(velx):
                    y += vely
                    mask[int(np.floor(y / velx)), j + 1] = 1
                    mask[int(np.ceil(y / velx)), j] = 1
                self.intersection_mask[(velx, vely)] = mask

        self.state = None
        self.vel = None
        self.trajectory = None
        self.noise = None

    def _get_obs(self):
        return np.concatenate([self.state, self.vel])

    def reset(self, *, seed=None, options={}):
        super().reset(seed=seed)
        self.state, self.vel = get_init_state(self.np_random, self.map)
        self.trajectory = [self.state]
        self.noise = options.get("noise", False)
        return self._get_obs(), {}

    def step(self, action):
        assert action in self.action_space

        reward = -1.0
        terminated = False
        if self.map[tuple(self.state)] == FINISH:
            reward = 0.0
            terminated = True
        else:
            if not self.noise or self.np_random.random() > 0.1:
                new_vel = np.clip(self.vel + action, -MAX_VEL, MAX_VEL)
                if np.any(new_vel):
                    self.vel = new_vel
                elif not np.any(self.vel):
                    action = self.np_random.integers(self.action_space.nvec) + self.action_space.start
                    self.vel += action
            velx, vely = self.vel
            new_state = self.state + [-vely, velx]
            rangey, rangex = [np.sort(range) for range in zip(self.state, new_state)]
            yfrom, yto = rangey + MAX_VEL
            xfrom, xto = rangex + MAX_VEL
            grid = self.padded_map[yfrom:yto + 1, xfrom:xto + 1]
            mask = self.intersection_mask[abs(velx), abs(vely)]
            mask = mask[::1 if vely <= 0 else -1, ::1 if velx >= 0 else -1]
            intersection = grid * mask
            if np.any(intersection == FINISH):
                int_from = np.array([max(0, vely), max(0, -velx)])
                int_to = self.np_random.choice(np.argwhere(intersection == FINISH))
                vely, velx = (int_to - int_from) * [-1, 1]
                (yfrom, yto), (xfrom, xto) = [np.sort(range) for range in zip(int_from, int_to)]
                intersection = intersection[yfrom:yto + 1, xfrom:xto + 1]
                if np.any(intersection == NO_TRACK):
                    self.state, self.vel = get_init_state(self.np_random, self.map)
                else:
                    terminated = True
                    self.vel = np.array([velx, vely], dtype=int)
                    self.state += [-vely, velx]
            elif np.any(intersection == NO_TRACK):
                self.state, self.vel = get_init_state(self.np_random, self.map)
            else:
                self.state = new_state
            self.trajectory.append(self.state.copy())

        return self._get_obs(), reward, terminated, False, {}

    def render(self):
        velx, vely = self.vel
        origin = (self.state - [-vely, velx])[::-1]
        render_map = self.map.copy()
        for state in self.trajectory:
            render_map[tuple(state)] = PATH
        plt.imshow(render_map, cmap="plasma")
        plt.arrow(*origin, velx, -vely, head_width=0.2)
        plt.xticks([])
        plt.yticks([])
        plt.show()


register(
    id="Racetrack-v0",
    entry_point=Racetrack
)
