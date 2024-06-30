import numpy as np
import gymnasium as gym
from gymnasium import spaces, register

PLAYER_MIN = 11

PLAYER_MAX = 21
UNIQUE_CARDS = 10
DEALER_SFROM = 17
FACE_CARDS = 4
DECK = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]


class Blackjack(gym.Env):
    def __init__(self):
        self.nvec = (PLAYER_MAX - PLAYER_MIN + 2, UNIQUE_CARDS + 1, 2)
        self.observation_space = spaces.MultiDiscrete(self.nvec)
        self.action_space = spaces.Discrete(2)

        self.prob = np.zeros(self.nvec + (self.action_space.n,) + self.nvec, dtype=np.float32)
        self.rewards = np.zeros(self.nvec + (self.action_space.n,) + self.nvec, dtype=np.float32)  # expected rewards

        # rows 17, 18, 19, 20, 21, bust
        prob_hard = np.zeros((PLAYER_MAX - DEALER_SFROM + 2, DEALER_SFROM + UNIQUE_CARDS - 1), dtype=np.float32)
        prob_soft = np.zeros((PLAYER_MAX - DEALER_SFROM + 2, DEALER_SFROM), dtype=np.float32)
        for i in range(UNIQUE_CARDS):
            prob_hard[min(i, PLAYER_MAX - DEALER_SFROM + 1)][i + DEALER_SFROM - 1] = 1.0
            if i < PLAYER_MAX - DEALER_SFROM + 2:
                prob_soft[i % (PLAYER_MAX - DEALER_SFROM + 1)][i + DEALER_SFROM - UNIQUE_CARDS - 1 + (DEALER_SFROM + UNIQUE_CARDS - PLAYER_MAX - 1) * (i // (PLAYER_MAX - DEALER_SFROM + 1))] = 1.0
        for i in range(PLAYER_MAX - DEALER_SFROM + 2):
            for j in range(DEALER_SFROM - 2, -1, -1):
                prob_hard[i, j] = (prob_soft[i, j + 1] + prob_hard[i, j + 2:j + UNIQUE_CARDS].sum() + FACE_CARDS * prob_hard[i, j + UNIQUE_CARDS]) / len(DECK)
                if j > PLAYER_MAX - UNIQUE_CARDS - 1:
                    prob_soft[i, j] = prob_hard[i, j]
                elif j < DEALER_SFROM - UNIQUE_CARDS - 1:
                    prob_soft[i, j] = (prob_soft[i, j + 1:j + UNIQUE_CARDS].sum() + FACE_CARDS * prob_soft[i, j + UNIQUE_CARDS]) / len(DECK)
            prob_hard[i, 0] = prob_soft[i, 0]

        outcome = -np.ones((PLAYER_MAX - DEALER_SFROM + 2, PLAYER_MAX - PLAYER_MIN + 1), dtype=np.float32)
        for i in range(PLAYER_MAX - DEALER_SFROM + 2):
            for j in range(PLAYER_MAX - PLAYER_MIN + 1):
                if i == PLAYER_MAX - DEALER_SFROM + 1:
                    outcome[i, j] = 1.0
                diag = PLAYER_MAX - PLAYER_MIN - j + i
                if diag <= PLAYER_MAX - DEALER_SFROM:
                    outcome[i, j] = 0.0 if diag == (PLAYER_MAX - DEALER_SFROM) else 1.0

        for player in range(PLAYER_MIN, PLAYER_MAX + 2):
            for dealer in range(UNIQUE_CARDS + 1):
                for usable in range(2):
                    state = (player - PLAYER_MIN, dealer, usable)
                    for action in range(2):
                        if player > PLAYER_MAX or dealer == 0:  # terminal
                            self.prob[state][action][state] = 1.0
                        else:
                            if action:  # hit
                                for card in DECK:
                                    if card == 1 and not usable:
                                        if player + card + UNIQUE_CARDS <= PLAYER_MAX:
                                            new_state = (player + card + UNIQUE_CARDS - PLAYER_MIN, dealer, 1)
                                        else:
                                            new_state = (player + card - PLAYER_MIN, dealer, 0)
                                    elif player + card > PLAYER_MAX and usable:
                                        new_state = (player + card - UNIQUE_CARDS - PLAYER_MIN, dealer, 0)
                                    else:
                                        new_state = (min(player + card, PLAYER_MAX + 1) - PLAYER_MIN, dealer, usable)
                                    self.prob[state][action][new_state] += 1 / len(DECK)
                                    if new_state[0] > PLAYER_MAX - PLAYER_MIN:
                                        self.rewards[state][action][new_state] -= 1 / len(DECK)
                            else:  # stand
                                new_state = (player - PLAYER_MIN, 0, usable)
                                self.prob[state][action][new_state] = 1.0
                                self.rewards[state][action][new_state] = (outcome[:, player - PLAYER_MIN] * prob_hard[:, dealer - 1]).sum()
        self.rewards = np.divide(self.rewards, self.prob, out=np.zeros_like(self.prob), where=self.prob != 0.0)

        self.player = None
        self.dealer = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return (0, 0, 0), {"player": self.player, "dealer": self.dealer}

    def step(self, action):
        if action:  # hit
            terminated = True
        else:  # stick
            terminated = False

        return self.state, 0.0, terminated, False, {}

    def render(self):
        pass


register(
    id="Blackjack-v2",
    entry_point=Blackjack
)
