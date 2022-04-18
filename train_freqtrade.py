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
from stable_baselines3.a2c.a2c import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.ppo.ppo import PPO
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

"""Settings"""
PAIR = "ADA/USDT"
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

        if action == 1:  # Sell
            if self.asset.balance == 0:
                return []
            order = proportion_order(portfolio, self.asset, self.cash, 1.0)

        for listener in self.listeners:
            listener.on_action(action)

        return [order]

    def reset(self):
        super().reset()


def main():

    strategy = StrategyResolver.load_strategy(freqtrade_config)
    strategy.dp = DataProvider(freqtrade_config, FreqtradeExchange(freqtrade_config), None)
    required_startup = strategy.startup_candle_count
    timeframe = freqtrade_config.get("timeframe")
    data = dict()

    if LOAD_PREPROCESSED_DATA:
        assert Path(_preprocessed_data_file).exists(), "Unable to load preprocessed data!"
        data = mpu.io.read(_preprocessed_data_file)
        assert PAIR in data, f"Loaded preprocessed data does not contain pair {PAIR}!"
    else:
        data = _load_data(freqtrade_config, timeframe, TRAINING_RANGE)
        data = strategy.advise_all_indicators({PAIR: data[PAIR]})
        if SAVE_PREPROCESSED_DATA:
            mpu.io.write(_preprocessed_data_file, data)

    pair_data = data[PAIR][required_startup:].copy()
    pair_data.reset_index(drop=True, inplace=True)

    del data

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

    renderer_feed = DataFeed(
        [
            Stream.source(list(price_data["date"])).rename("date"),
            Stream.source(list(price_data["open"]), dtype="float").rename("open"),
            Stream.source(list(price_data["high"]), dtype="float").rename("high"),
            Stream.source(list(price_data["low"]), dtype="float").rename("low"),
            Stream.source(list(price_data["close"]), dtype="float").rename("close"),
            Stream.source(list(price_data["volume"]), dtype="float").rename("volume"),
        ]
    )

    action_scheme = BuySellHold(cash=cash, asset=asset)

    # reward_scheme = PBR(price)
    reward_scheme = SimpleProfit(window_size=8)

    trading_env = default.create(
        portfolio=portfolio,
        action_scheme=action_scheme,
        reward_scheme=reward_scheme,
        feed=feed,
        renderer_feed=renderer_feed,
        window_size=WINDOW_SIZE,
        max_allowed_loss=0.50,
    )

    trading_env = Monitor(trading_env, LOG_DIR, info_keywords=("net_worth",))

    # Optional policy_kwargs
    # see https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html?highlight=policy_kwargs#custom-network-architecture
    # policy_kwargs = dict(activation_fn=th.nn.ReLU,
    #                  net_arch=[dict(pi=[32, 32], vf=[32, 32])])
    # policy_kwargs = dict(activation_fn=th.nn.Tanh, net_arch=[32, dict(pi=[64,  64], vf=[64, 64])])
    policy_kwargs = dict(net_arch=[128, dict(pi=[128, 128], vf=[128, 128])])

    start_date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    model = PPO(  # See https://stable-baselines3.readthedocs.io/en/master/guide/algos.html for other algos with discrete action space
        "MlpPolicy",  # MlpPolicy MultiInputPolicy
        trading_env,
        verbose=0,
        device="cuda",
        tensorboard_log=TENSORBOARD_LOG,
        # n_steps = len(pair_data),
        # batch_size = 1000,
        # n_epochs = 20,
        policy_kwargs=policy_kwargs,
    )

    base_name = f"{strategy.get_strategy_name()}_TensorTrade_{model.__class__.__name__}_{start_date}"

    tb_callback = SaveOnStepCallback(
        check_freq=10000, save_name=f"best_model_{base_name}", save_dir=MODEL_DIR, log_dir=LOG_DIR, verbose=1
    )

    print(f"You can run tensorboard with: 'tensorboard --logdir {Path(TENSORBOARD_LOG).absolute()}'")
    print("Learning started.")

    model.learn(total_timesteps=LEARNING_TIME_STEPS, callback=tb_callback)
    model.save(f"{MODEL_DIR}final_model_{base_name}")


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
