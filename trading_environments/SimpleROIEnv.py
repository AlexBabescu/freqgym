from enum import Enum

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding


class Actions(Enum):
    Hold = 0
    Buy = 1

class SimpleROIEnv(gym.Env):

    def __init__(
        self,
        data,
        prices,
        window_size,
        required_startup,
        minimum_roi=0.02,
        roi_candles=24,
        punish_holding_amount=0,
        punish_missed_buy=True):

        self.data = data
        self.window_size = window_size
        self.prices = prices

        self.required_startup = required_startup
        self.minimum_roi = minimum_roi
        self.roi_candles = roi_candles
        self.punish_holding_amount = punish_holding_amount
        assert self.punish_holding_amount <= 0, "`punish_holding_amount` should be less or equal to 0"
        self.punish_missed_buy = punish_missed_buy

        _, number_of_features = self.data.shape
        self.shape = (self.window_size, number_of_features)

        # spaces
        self.action_space = spaces.Discrete(len(Actions))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self._total_reward = 0.
        self._current_tick = self.required_startup + self.window_size + 1
        self._end_tick = len(self.data) - self.roi_candles - 1

        return self._get_observation()

    def step(self, action):
        done = False

        reward = 0

        if self._current_tick + self.roi_candles >= self._end_tick:
            done = True

        current_price = self.prices['close'][self._current_tick]
        future_close_price = self.prices['close'][self._current_tick + self.roi_candles]
        future_highest_high = self.prices['high'][self._current_tick:self._current_tick + self.roi_candles].max()

        future_highest_high_diff = _pct_change(current_price, future_highest_high)
        future_close_price_diff = _pct_change(current_price, future_close_price)

        if action == Actions.Buy.value:
            self._current_tick += self.roi_candles
            reward = -abs(future_close_price_diff)
            if future_highest_high_diff >= self.minimum_roi:
                reward = future_highest_high_diff

        if action == Actions.Hold.value:
            reward = self.punish_holding_amount
            # Missed buying opportunity
            if self.punish_missed_buy:
                if future_highest_high_diff >= self.minimum_roi:
                    reward = -future_highest_high_diff

        self._current_tick += 1

        observation = self._get_observation()

        return observation, reward, done, {}

    def _get_observation(self):
        return self.data[(self._current_tick-self.window_size):self._current_tick].to_numpy()

def _pct_change(a, b):
    return (b - a) / a
