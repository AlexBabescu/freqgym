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

    FC_SIZE = tune.grid_search(
        [[256, 256], [1024], [128, 64, 32]]
    )  # Those are the alternatives that ray.tune will try...
    # LEARNING_RATE = tune.grid_search([0.001, 0.0005, 0.00001])  # ... and they will be combined with these ones ...
    MINIBATCH_SIZE = tune.grid_search([5, 10, 20])  # ... and these ones, in a cartesian product.

    LR = tune.loguniform(1e-5, 1e-2)
    GAMMA = tune.uniform(0.8, 0.9999)
    LAMBDA = tune.uniform(0.1, 0.8)
    VF_LOSS_COEFF = tune.uniform(0.01, 1.0)
    ENTROPY_COEFF = tune.uniform(1e-8, 1e-1)

    ray.init(include_dashboard=True, address=None, dashboard_host="127.0.0.1", num_cpus=12, num_gpus=0)

    search_alg = OptunaSearch()
    search_alg = ConcurrencyLimiter(search_alg, max_concurrent=4)

    scheduler = ASHAScheduler()

    register_env("MyTrainingEnv", create_env)

    # Specific configuration keys that will be used during training
    env_config_training = {
        "window_size": 14,  # We want to look at the last 14 samples (hours)
        "reward_window_size": 7,  # And calculate reward based on the actions taken in the next 7 hours
        "max_allowed_loss": 0.10,  # If it goes past 10% loss during the iteration, we don't want to waste time on a "loser".
        "data_file": TRAINING_PAIR_FILE,  # The variable that will be used to differentiate training and validation datasets
    }

    # Specific configuration keys that will be used during evaluation (only the overridden ones)
    env_config_evaluation = {
        "max_allowed_loss": 1.00,  # During validation runs we want to see how bad it would go. Even up to 100% loss.
        "data_file": EVALUATION_PAIR_FILE,  # The variable that will be used to differentiate training and validation datasets
    }

    import time

    start = time.time()

    # fmt: off
    analysis = tune.run(
        run_or_experiment="PPO",  # We'll be using the builtin PPO agent in RLLib
        name="MyExperiment1",
        metric='episode_reward_mean',
        mode='max',
        stop={
            "training_iteration": 10  # Let's do 5 steps for each hyperparameter combination
        },
        config={
            "log_sys_usage": True,
            "env": "MyTrainingEnv",
            "env_config": env_config_training,  # The dictionary we built before
            "log_level": "INFO",
            "framework": "torch",
            "ignore_worker_failures": True,
            "num_workers": 11,  # One worker per agent. You can increase this but it will run fewer parallel trainings.
            "num_envs_per_worker": 2,
            "num_gpus": 0,  # I yet have to understand if using a GPU is worth it, for our purposes, but I think it's not. This way you can train on a non-gpu enabled system.
            "clip_rewards": True,
            "lr": LR,  # Hyperparameter grid search defined above
            "lr_schedule": [
                [0, 1e-1],
                [int(1e2), 1e-2],
                [int(1e3), 1e-3],
                [int(1e4), 1e-4],
                [int(1e5), 1e-5],
                [int(1e6), 1e-6],
                [int(1e7), 1e-7]
            ],
            "gamma": GAMMA,  # This can have a big impact on the result and needs to be properly tuned (range is 0 to 1)
            "lambda": LAMBDA,
            "observation_filter": "MeanStdFilter",
            "model": {
                "_use_default_native_models": True,
                "use_attention": True,
                "max_seq_len": 10,
                "attention_num_transformer_units": 1,
                "attention_dim": 32,
                "attention_memory_inference": 10,
                "attention_memory_training": 10,
                "attention_num_heads": 1,
                "attention_head_dim": 32,
                "attention_position_wise_mlp_dim": 32,
                # "fcnet_hiddens": FC_SIZE,  # Hyperparameter grid search defined above
                "vf_share_layers": True,
            },
            "vf_loss_coeff": VF_LOSS_COEFF,
            "entropy_coeff": ENTROPY_COEFF,
            "num_sgd_iter": 10,
            # "sgd_minibatch_size": MINIBATCH_SIZE,  # Hyperparameter grid search defined above
            "evaluation_interval": 1,  # Run evaluation on every iteration
            "evaluation_config": {
                "env_config": env_config_evaluation,  # The dictionary we built before (only the overriding keys to use in evaluation)
                "explore": False,  # We don't want to explore during evaluation. All actions have to be repeatable.
            },
        },
        search_alg=search_alg,
        scheduler=scheduler,
        num_samples=10,  # Have one sample for each hyperparameter combination. You can have more to average out randomness.
        keep_checkpoints_num=10,  # Keep the last 2 checkpoints
        checkpoint_freq=1,  # Do a checkpoint on each iteration (slower but you can pick more finely the checkpoint to use later)
        resume="AUTO",
    )
    # fmt: on

    taken = time.time() - start
    print(f"Time taken: {taken:.2f} seconds.")
    print(f"Best config: {analysis.best_config}")
    import json
    output = Path(__file__).parent / "best_config.json"
    output.write_text(json.dumps(analysis.best_config))


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
