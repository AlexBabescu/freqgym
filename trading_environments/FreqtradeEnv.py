
from enum import Enum

import gym
import numpy as np
from freqtrade.persistence import Trade
from gym import spaces
from gym.utils import seeding

# Based on https://github.com/hugocen/freqtrade-gym/blob/master/freqtradegym.py

class Actions(Enum):
    Hold = 0
    Buy = 1
    Sell = 2

class FreqtradeEnv(gym.Env):
    """A freqtrade trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human', 'system', 'none']}

    def __init__(
        self,
        data,
        prices,
        window_size,
        pair,
        stake_amount,
        punish_holding_amount=0,
        fee=0.005
        ):

        self.data = data
        self.window_size = window_size
        self.prices = prices
        self.pair = pair
        self.stake_amount = stake_amount
        self.punish_holding_amount = punish_holding_amount
        assert self.punish_holding_amount <= 0, "`punish_holding_amount` should be less or equal to 0"
        self.fee = fee

        self.opened_trade = None
        self.trades = []

        self._reward = 0
        self.total_reward = 0

        _, number_of_features = self.data.shape
        self.shape = (self.window_size, number_of_features)

        self.action_space = spaces.Discrete(len(Actions))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float32)

        self.seed()

    def _get_observation(self):
        return self.data[(self._current_tick-self.window_size):self._current_tick].to_numpy()

    def _take_action(self, action):
        if action == Actions.Hold.value:
            self._reward = self.punish_holding_amount
            return

        if action == Actions.Buy.value:
            if self.opened_trade == None:
                self.opened_trade = Trade(
                    pair=self.pair,
                    open_rate=self.prices.loc[self._current_tick].open,
                    open_date=self.prices.loc[self._current_tick].date,
                    stake_amount=self.stake_amount,
                    amount=self.stake_amount / self.prices.loc[self._current_tick].open,
                    fee_open=self.fee,
                    fee_close=self.fee,
                    is_open=True,
                )
                self.trades.append({
                    "step": self._current_tick,
                    "type": 'buy',
                    "total": self.prices.loc[self._current_tick].open
                })
            return

        if action == Actions.Sell.value:
            if self.opened_trade != None:
                profit_percent = self.opened_trade.calc_profit_ratio(rate=self.prices.loc[self._current_tick].open)
                self.opened_trade = None
                self._reward = profit_percent

                self.trades.append({
                    "step": self._current_tick,
                    "type": 'sell',
                    "total": self.prices.loc[self._current_tick].open
                })
            return

    def step(self, action):
        # Execute one time step within the environment
        done = False

        self._reward = 0

        if self._current_tick >= self._end_tick:
            done = True

        self._take_action(action)

        self._current_tick += 1

        self.total_reward += self._reward

        observation = self._get_observation()

        return observation, self._reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.opened_trade = None
        self.trades = []

        self._reward = 0
        self.total_reward = 0

        self._current_tick = self.window_size + 1
        self._end_tick = len(self.data) - 1

        return self._get_observation()

def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
