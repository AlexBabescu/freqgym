import datetime
from pathlib import Path

import mpu
import torch as th
from freqtrade.configuration import Configuration, TimeRange
from freqtrade.data import history
from freqtrade.resolvers import StrategyResolver
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.a2c.a2c import A2C

from tb_callbacks import SaveOnStepCallback
from trading_environments import FreqtradeEnv, SimpleROIEnv, GymAnytrading

"""Settings"""
PAIR = "BTC/USDT"
TRAINING_RANGE = "20210601-20210901"
WINDOW_SIZE = 10
LOAD_PREPROCESSED_DATA = False  # useful if you have to calculate a lot of features
SAVE_PREPROCESSED_DATA = True
LEARNING_TIME_STEPS = int(1e+6)
LOG_DIR = "./logs/"
TENSORBOARD_LOG = "./tensorboard/"
MODEL_DIR = "./models/"
"""End of settings"""

freqtrade_config = Configuration.from_files(['user_data/config.json'])
_preprocessed_data_file = "preprocessed_data.pickle"


def main():

    strategy = StrategyResolver.load_strategy(freqtrade_config)
    required_startup = strategy.startup_candle_count
    timeframe = freqtrade_config.get('timeframe')
    data = dict()

    if LOAD_PREPROCESSED_DATA:
        assert Path(_preprocessed_data_file).exists(), "Unable to load preprocessed data!"
        data = mpu.io.read(_preprocessed_data_file)
        assert PAIR in data, f"Loaded preprocessed data does not contain pair {PAIR}!"
    else:
        data = _load_data(freqtrade_config, PAIR, timeframe, TRAINING_RANGE, WINDOW_SIZE)
        data = strategy.advise_all_indicators(data)
        if SAVE_PREPROCESSED_DATA:
            mpu.io.write(_preprocessed_data_file, data)

    pair_data = data[PAIR][required_startup:].copy()
    pair_data.reset_index(drop=True, inplace=True)

    del data

    price_data = pair_data[['date', 'open', 'close', 'high', 'low', 'volume']].copy()

    pair_data.drop(columns=['date', 'open', 'close', 'high', 'low', 'volume'], inplace=True)
    pair_data.fillna(0, inplace=True)


    # trading_env = FreqtradeEnv(
    #     data=pair_data,
    #     prices=price_data,
    #     window_size=WINDOW_SIZE,  # how many past candles should it use as features
    #     pair=PAIR,
    #     stake_amount=freqtrade_config['stake_amount'],
    #     punish_holding_amount=0,
    #     )

    # trading_env = SimpleROIEnv(
    #     data=pair_data,
    #     prices=price_data,
    #     window_size=WINDOW_SIZE,  # how many past candles should it use as features
    #     required_startup=required_startup,
    #     minimum_roi=0.02,  # 2% target ROI
    #     roi_candles=24,  # 24 candles * 5m = 120 minutes
    #     punish_holding_amount=0,
    #     punish_missed_buy=True
    #     )

    trading_env = GymAnytrading(
        signal_features=pair_data,
        prices=price_data.close,
        window_size=WINDOW_SIZE,  # how many past candles should it use as features
        )

    trading_env = Monitor(trading_env, LOG_DIR)

    # Optional policy_kwargs
    # see https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html?highlight=policy_kwargs#custom-network-architecture
    # policy_kwargs = dict(activation_fn=th.nn.ReLU,
    #                  net_arch=[dict(pi=[32, 32], vf=[32, 32])])
    # policy_kwargs = dict(activation_fn=th.nn.Tanh, net_arch=[32, dict(pi=[64,  64], vf=[64, 64])])
    policy_kwargs = dict(net_arch=[32, dict(pi=[64,  64], vf=[64, 64])])

    start_date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    model = PPO(  # See https://stable-baselines3.readthedocs.io/en/master/guide/algos.html for other algos with discrete action space
        "MlpPolicy",
        trading_env,
        verbose=0,
        device='auto',
        tensorboard_log=TENSORBOARD_LOG,
        # policy_kwargs=policy_kwargs
    )

    base_name = f"{strategy.get_strategy_name()}_{trading_env.env.__class__.__name__}_{model.__class__.__name__}_{start_date}"

    tb_callback = SaveOnStepCallback(
        check_freq=5000,
        save_name=f"best_model_{base_name}",
        save_dir=MODEL_DIR,
        log_dir=LOG_DIR,
        verbose=1)

    print(f"You can run tensorboard with: 'tensorboard --logdir {Path(TENSORBOARD_LOG).absolute()}'")
    print("Learning started.")

    model.learn(
        total_timesteps=LEARNING_TIME_STEPS,
        callback=tb_callback
    )
    model.save(f"{MODEL_DIR}final_model_{base_name}")


def _load_data(config, pair, timeframe, timerange, window_size):
    timerange = TimeRange.parse_timerange(timerange)

    return history.load_data(
        datadir=config['datadir'],
        pairs=[pair],
        timeframe=timeframe,
        timerange=timerange,
        startup_candles=window_size + 1,
        fail_without_data=True,
        data_format=config.get('dataformat_ohlcv', 'json'),
    )

if __name__ == "__main__":
    main()
