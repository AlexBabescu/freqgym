import datetime
from os import access
from pathlib import Path

import mpu
import tensortrade.env.default as default
import torch as th
from freqtrade.configuration import Configuration, TimeRange
from freqtrade.data import history
from freqtrade.data.dataprovider import DataProvider
from freqtrade.exchange import Exchange as FreqtradeExchange
from freqtrade.resolvers import StrategyResolver
from gym.spaces import Discrete, Space

from stable_baselines3.common.monitor import Monitor

from tensortrade.env.default.actions import BSH, TensorTradeActionScheme
from tensortrade.env.default.rewards import PBR, RiskAdjustedReturns, SimpleProfit, TensorTradeRewardScheme
from tensortrade.env.generic import ActionScheme, TradingEnv
from tensortrade.feed.core import DataFeed, NameSpace, Stream
from tensortrade.oms.exchanges import Exchange, ExchangeOptions
from tensortrade.oms.instruments import BTC, ETH, LTC, USD, Instrument
from tensortrade.oms.orders import proportion_order
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.wallets import Portfolio, Wallet

from tb_callbacks import SaveOnStepCallback
from trading_environments import FreqtradeEnv, GymAnytrading, SimpleROIEnv

import os

import ray
from ray import tune
from ray.tune.registry import register_env
import ray.rllib.agents.ppo as ppo
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.optuna import OptunaSearch
from ray.tune.logger import pretty_print


"""Settings"""
TRAINING_PAIR = "ADA/USDT"
EVALUATION_PAIR = "XLM/USDT"
TRAINING_RANGE = "20210901-20211231"
WINDOW_SIZE = 10
LOAD_PREPROCESSED_DATA = False  # useful if you have to calculate a lot of features
SAVE_PREPROCESSED_DATA = True
LEARNING_TIME_STEPS = int(1e9)
LOG_DIR = "./logs/"
TENSORBOARD_LOG = "./tensorboard/"
MODEL_DIR = "./models/"
USER_DATA = Path(__file__).parent / "user_data"
"""End of settings"""

freqtrade_config = Configuration.from_files([str(USER_DATA / "config.json")])
_preprocessed_data_file = "preprocessed_data.pickle"

TRAINING_PAIR_FILE = str(Path(__file__).parent / "TRAINING_PAIR.pickle")
EVALUATION_PAIR_FILE = str(Path(__file__).parent / "EVALUATION_PAIR.pickle")

from gym.spaces import Discrete, Space


class BuySellHold(TensorTradeActionScheme):
    """A simple discrete action scheme where the only options are to buy, sell,
    or hold.

    Parameters
    ----------
    cash : `Wallet`
        The wallet to hold funds in the base instrument.
    asset : `Wallet`
        The wallet to hold funds in the quote instrument.
    """

    registered_name = "bsh"

    def __init__(self, cash: "Wallet", asset: "Wallet"):
        super().__init__()
        self.cash = cash
        self.asset = asset
        self.action = 2

        self.listeners = []

    @property
    def action_space(self):
        return Discrete(3)

    def attach(self, listener):
        self.listeners += [listener]
        return self

    def get_orders(self, action: int, portfolio: "Portfolio"):
        order = None

        if action == 2:  # Hold
            return []

        if action == 0:  # Buy
            if self.cash.balance == 0:
                return []
            order = proportion_order(portfolio, self.cash, self.asset, 1.0)
            self.action = 0

        if action == 1:  # Sell
            if self.asset.balance == 0:
                return []
            order = proportion_order(portfolio, self.asset, self.cash, 1.0)
            self.action = 1

        for listener in self.listeners:
            listener.on_action(action)

        return [order]

    def reset(self):
        super().reset()
        self.action = 2


def create_env(config):
    pair_data = mpu.io.read(config["data_file"])

    price_data = pair_data[["date", "open", "close", "high", "low", "volume"]].copy()

    pair_data.drop(columns=["date", "open", "close", "high", "low", "volume"], inplace=True)
    pair_data.fillna(0, inplace=True)

    ADA = Instrument("ADA", 3, "Cardano")

    price = Stream.source(list(price_data["close"]), dtype="float").rename("USD-ADA")

    exchange_options = ExchangeOptions(commission=0.0035)
    binance = Exchange("binance", service=execute_order, options=exchange_options)(price)

    cash = Wallet(binance, 1000 * USD)
    asset = Wallet(binance, 0 * ADA)

    portfolio = Portfolio(USD, [cash, asset])

    features = [Stream.source(list(pair_data[c]), dtype="float").rename(c) for c in pair_data.columns]

    feed = DataFeed(features)
    feed.compile()

    # reward_scheme = PBR(price)
    reward_scheme = PBR(price=price)

    action_scheme = BuySellHold(cash=cash, asset=asset).attach(reward_scheme)

    renderer_feed = DataFeed(
        [
            Stream.source(list(price_data["date"])).rename("date"),
            Stream.source(list(price_data["open"]), dtype="float").rename("open"),
            Stream.source(list(price_data["high"]), dtype="float").rename("high"),
            Stream.source(list(price_data["low"]), dtype="float").rename("low"),
            Stream.source(list(price_data["close"]), dtype="float").rename("close"),
            Stream.source(list(price_data["volume"]), dtype="float").rename("volume"),
            Stream.sensor(action_scheme, lambda s: s.action, dtype="float").rename("action"),
        ]
    )

    trading_env = default.create(
        portfolio=portfolio,
        action_scheme=action_scheme,
        reward_scheme=reward_scheme,
        feed=feed,
        renderer_feed=renderer_feed,
        renderer=[
            default.renderers.PlotlyTradingChart(),
        ],
        window_size=config["window_size"],
        max_allowed_loss=config["max_allowed_loss"],
    )

    # trading_env = Monitor(trading_env, LOG_DIR, info_keywords=('net_worth',))

    return trading_env


def main():

    # strategy = StrategyResolver.load_strategy(freqtrade_config)
    # strategy.dp = DataProvider(freqtrade_config, FreqtradeExchange(freqtrade_config), None)
    # required_startup = strategy.startup_candle_count
    # timeframe = freqtrade_config.get('timeframe')

    # data = _load_data(freqtrade_config, timeframe, TRAINING_RANGE)
    # training_data = strategy.advise_all_indicators({TRAINING_PAIR:data[TRAINING_PAIR]})[TRAINING_PAIR][required_startup:].reset_index(drop=True)
    # evaluation_data = strategy.advise_all_indicators({EVALUATION_PAIR:data[EVALUATION_PAIR]})[EVALUATION_PAIR][required_startup:].reset_index(drop=True)

    # mpu.io.write(TRAINING_PAIR_FILE, training_data)
    # mpu.io.write(EVALUATION_PAIR_FILE, evaluation_data)

    ray.init(include_dashboard=True, address=None, dashboard_host="127.0.0.1", num_cpus=12, num_gpus=0)

    register_env("MyTrainingEnv", create_env)

    import json
    config_file = Path(__file__).parent / "best_config.json"

    config = json.loads(config_file.read_text())

    # Restore agent
    agent = ppo.PPOTrainer(
        env="MyTrainingEnv",
        config=config,
    )

    # for i in range(100):
    #     # Perform one iteration of training the policy with PPO
    #     result = agent.train()
    #     print(pretty_print(result))

    #     if i % 10 == 0:
    #         checkpoint = agent.save()
    #         print("checkpoint saved at", checkpoint)

    print()


def _load_data(config, timeframe, timerange):
    timerange = TimeRange.parse_timerange(timerange)

    return history.load_data(
        datadir=config["datadir"],
        pairs=config["pairs"],
        timeframe=timeframe,
        timerange=timerange,
        startup_candles=config["startup_candle_count"],
        fail_without_data=True,
        data_format=config.get("dataformat_ohlcv", "json"),
    )


if __name__ == "__main__":
    main()
