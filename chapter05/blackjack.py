import numpy as np
import gymnasium as gym
from gymnasium import spaces, register

NUM_ACTIONS = 2
PLAYER_MIN = 11
PLAYER_MAX = 21
SOFT_DIFF = 10
UNIQUE_CARDS = 10
DEALER_SFROM = 17
FACE_CARDS = 4
DECK = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]


def sum_hand(hand):
    sum = np.sum(hand)
    soft_hand = 1 in hand and sum + SOFT_DIFF <= PLAYER_MAX
    if soft_hand:
        sum += SOFT_DIFF
    return sum, soft_hand


def is_natural(hand):
    return sorted(hand) == [1, 10]


class Blackjack(gym.Env):
    def __init__(self):
        nvec = (PLAYER_MAX - PLAYER_MIN + 2, UNIQUE_CARDS + 1, NUM_ACTIONS)
        self.observation_space = spaces.MultiDiscrete(nvec, start=[PLAYER_MIN, 0, 0])
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        self.prob = np.zeros(nvec + (self.action_space.n,) + nvec, dtype=np.float32)  # transition probability
        self.rewards = np.zeros(nvec + (self.action_space.n,) + nvec, dtype=np.float32)  # expected rewards

        # rows: 17, 18, 19, 20, 21, bust
        prob_dealer = np.zeros((PLAYER_MAX - DEALER_SFROM + 2, DEALER_SFROM + UNIQUE_CARDS - 1), dtype=np.float32)
        prob_dealer_soft = np.zeros((PLAYER_MAX - DEALER_SFROM + 2, DEALER_SFROM), dtype=np.float32)
        for i in range(UNIQUE_CARDS):
            prob_dealer[min(i, PLAYER_MAX - DEALER_SFROM + 1)][i + DEALER_SFROM - 1] = 1.0
            if i < PLAYER_MAX - DEALER_SFROM + 2:
                prob_dealer_soft[i % (PLAYER_MAX - DEALER_SFROM + 1)][i + DEALER_SFROM - SOFT_DIFF - 1 + (DEALER_SFROM + UNIQUE_CARDS - PLAYER_MAX - 1) * (i // (PLAYER_MAX - DEALER_SFROM + 1))] = 1.0
        for i in range(PLAYER_MAX - DEALER_SFROM + 2):
            for j in range(DEALER_SFROM - 2, -1, -1):
                prob_dealer[i, j] = (prob_dealer_soft[i, j + 1] + prob_dealer[i, j + 2:j + UNIQUE_CARDS].sum() + FACE_CARDS * prob_dealer[i, j + UNIQUE_CARDS]) / len(DECK)
                if j > PLAYER_MAX - SOFT_DIFF - 1:
                    prob_dealer_soft[i, j] = prob_dealer[i, j]
                elif j < DEALER_SFROM - SOFT_DIFF - 1:
                    prob_dealer_soft[i, j] = (prob_dealer_soft[i, j + 1:j + UNIQUE_CARDS].sum() + FACE_CARDS * prob_dealer_soft[i, j + UNIQUE_CARDS]) / len(DECK)
            prob_dealer[i, 0] = prob_dealer_soft[i, 0]

        # outcome factor
        outcome = -np.ones((PLAYER_MAX - DEALER_SFROM + 2, PLAYER_MAX - PLAYER_MIN + 1), dtype=np.float32)
        for i in range(PLAYER_MAX - DEALER_SFROM + 2):
            for j in range(PLAYER_MAX - PLAYER_MIN + 1):
                if i == PLAYER_MAX - DEALER_SFROM + 1:
                    outcome[i, j] = 1.0
                diag = PLAYER_MAX - PLAYER_MIN - j + i
                if diag <= PLAYER_MAX - DEALER_SFROM:
                    outcome[i, j] = 0.0 if diag == PLAYER_MAX - DEALER_SFROM else 1.0

        for player_sum in range(PLAYER_MIN, PLAYER_MAX + 2):
            for dealer_card in range(UNIQUE_CARDS + 1):
                for soft_hand in range(2):
                    state = (player_sum - PLAYER_MIN, dealer_card, soft_hand)
                    for action in range(NUM_ACTIONS):
                        if player_sum > PLAYER_MAX or dealer_card == 0:  # terminal
                            self.prob[state][action][state] = 1.0
                        else:
                            if action:  # hit
                                for card in DECK:
                                    if card == 1 and not soft_hand:
                                        if player_sum + card + SOFT_DIFF <= PLAYER_MAX:
                                            _player_sum = player_sum + card + SOFT_DIFF
                                            _soft_hand = 1
                                        else:
                                            _player_sum = player_sum + card
                                            _soft_hand = 0
                                    elif player_sum + card > PLAYER_MAX and soft_hand:
                                        _player_sum = player_sum + card - SOFT_DIFF
                                        _soft_hand = 0
                                    else:
                                        _player_sum = player_sum + card
                                        _soft_hand = soft_hand
                                    new_state = (min(_player_sum, PLAYER_MAX + 1) - PLAYER_MIN, dealer_card, _soft_hand)
                                    self.prob[state][action][new_state] += 1 / len(DECK)
                                    if new_state[0] + PLAYER_MIN > PLAYER_MAX:
                                        self.rewards[state][action][new_state] -= 1 / len(DECK)
                            else:  # stand
                                new_state = (player_sum - PLAYER_MIN, 0, soft_hand)
                                self.prob[state][action][new_state] = 1.0
                                self.rewards[state][action][new_state] = (outcome[:, player_sum - PLAYER_MIN] * prob_dealer[:, dealer_card - 1]).sum()
        self.rewards = np.divide(self.rewards, self.prob, out=np.zeros_like(self.prob), where=self.prob != 0.0)

        self.player = None
        self.dealer = None

    def _get_info(self, terminated):
        player_sum, _ = sum_hand(self.player)
        dealer_sum, _ = sum_hand(self.dealer)
        dealer_card = self.dealer[0]
        if terminated:
            return {"player": self.player, "player_sum": player_sum,
                    "dealer": self.dealer if dealer_card else self.dealer[1:], "dealer_sum": dealer_sum}
        else:
            return {"player": self.player, "player_sum": player_sum, "dealer_card": dealer_card}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.player = list(self.np_random.choice(DECK, size=2))
        self.dealer = list(self.np_random.choice(DECK, size=2))

        player_sum, soft_hand = sum_hand(self.player)
        while player_sum < PLAYER_MIN:
            self.player.append(self.np_random.choice(DECK))
            player_sum, soft_hand = sum_hand(self.player)
        dealer_card = self.dealer[0]
        state = np.array([player_sum, dealer_card, soft_hand])

        return state, self._get_info(False)

    def step(self, action):
        assert action in self.action_space

        player_sum, soft_hand = sum_hand(self.player)
        dealer_card = self.dealer[0]

        if player_sum > PLAYER_MAX or dealer_card == 0:  # terminal
            reward = 0.0
            terminated = True
            reward = 0.0

        elif action:  # hit
            self.player.append(self.np_random.choice(DECK))
            player_sum, soft_hand = sum_hand(self.player)
            terminated = player_sum > PLAYER_MAX
            reward = -1.0 if terminated else 0.0

        else:  # stand
            dealer_sum, _ = sum_hand(self.dealer)
            while dealer_sum < DEALER_SFROM:
                self.dealer.append(self.np_random.choice(DECK))
                dealer_sum, _ = sum_hand(self.dealer)

            terminated = True
            reward = float(player_sum > dealer_sum) - float(dealer_sum > player_sum)
            if dealer_sum > PLAYER_MAX or (is_natural(self.player) and not is_natural(self.dealer)):
                reward = 1.0

            dealer_card = 0
            self.dealer = [dealer_card] + self.dealer

        state = np.array((min(player_sum, PLAYER_MAX + 1), dealer_card, soft_hand))

        return state, reward, terminated, False, self._get_info(terminated)

    def render(self):
        pass


register(
    id="Blackjack-v2",
    entry_point=Blackjack
)
