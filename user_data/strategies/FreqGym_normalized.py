# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement

import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np  # noqa
import pandas as pd  # noqa
import talib.abstract as ta
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.a2c.a2c import A2C
from freqtrade.strategy import merge_informative_pair
import time

class FreqGym_normalized(IStrategy):
    # # If you've used SimpleROIEnv then use this minimal_roi
    # minimal_roi = {"770": -10, "600": 0.00001, "60": 0.01, "30": 0.02, "0": 0.03}

    minimal_roi = {
        # "1440": -10
        "0": 10,
    }

    stoploss = -0.10

    # Trailing stop:
    trailing_stop = False
    trailing_stop_positive = 0.001
    trailing_stop_positive_offset = 0.017
    trailing_only_offset_is_reached = True

    timeframe = "15m"
    informative_timeframe = "1h"

    use_sell_signal = True

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    startup_candle_count: int = 200

    model = None
    window_size = None

    freqtrade_columns = ["date", "open", "close", "high", "low", "volume", "buy", "sell", "buy_tag", "exit_tag"]
    informative_freqtrade_columns = [f"{c}_1h" for c in freqtrade_columns]
    btc_freqtrade_columns = [f"{c}_btc_1h" for c in freqtrade_columns]
    indicators = None

    try:
        model = PPO.load("models/best_model")  # Note: Make sure you use the same policy as the one used to train
        window_size = model.observation_space.shape[0]
    except Exception:
        pass

    timeperiods = [8, 16, 32]

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]
        informative_pairs += [("BTC/USDT", self.informative_timeframe)]
        return informative_pairs

    def generate_features(self, df):

        dataframe = df.copy()
        # Plus Directional Indicator / Movement
        dataframe["plus_di"] = normalize(ta.PLUS_DI(dataframe), 0, 100)

        # # Minus Directional Indicator / Movement
        dataframe["minus_di"] = normalize(ta.MINUS_DI(dataframe), 0, 100)

        # Ultimate Oscillator
        dataframe["uo"] = normalize(ta.ULTOSC(dataframe), 0, 100)

        # Hilbert Transform Indicator - SineWave
        hilbert = ta.HT_SINE(dataframe)
        dataframe["htsine"] = normalize(hilbert["sine"], -1, 1)
        dataframe["htleadsine"] = normalize(hilbert["leadsine"], -1, 1)

        # BOP                  Balance Of Power
        dataframe["bop"] = normalize(ta.BOP(dataframe), -1, 1)

        # STOCH - Stochastic
        stoch = ta.STOCH(dataframe)
        dataframe["slowk"] = normalize(stoch["slowk"], 0, 100)
        dataframe["slowd"] = normalize(stoch["slowd"], 0, 100)

        # STOCHF - Stochastic Fast
        stochf = ta.STOCHF(dataframe)
        dataframe["fastk"] = normalize(stochf["fastk"], 0, 100)
        dataframe["fastk"] = normalize(stochf["fastk"], 0, 100)

        # Bollinger Bands
        bollinger2 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        bollinger3 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=3)

        dataframe["bb2_lower_gt_close"] = bollinger2["lower"].gt(dataframe["close"]).astype("int")
        dataframe["bb3_lower_gt_close"] = bollinger3["lower"].gt(dataframe["close"]).astype("int")

        for period in self.timeperiods:
            # ADX                  Average Directional Movement Index
            dataframe[f"adx_{period}"] = normalize(ta.ADX(dataframe, timeperiod=period), 0, 100)

            # Aroon, Aroon Oscillator
            aroon = ta.AROON(dataframe, timeperiod=period)
            dataframe[f"aroonup_{period}"] = normalize(aroon["aroonup"], 0, 100)
            dataframe[f"aroondown_{period}"] = normalize(aroon["aroondown"], 0, 100)
            dataframe[f"aroonosc_{period}"] = normalize(ta.AROONOSC(dataframe, timeperiod=period), -100, 100)

            # CMO                  Chande Momentum Oscillator
            dataframe[f"cmo_{period}"] = normalize(ta.CMO(dataframe, timeperiod=period), -100, 100)

            # DX                   Directional Movement Index
            dataframe[f"dx_{period}"] = normalize(ta.DX(dataframe, timeperiod=period), 0, 100)

            # MFI                  Money Flow Index
            dataframe[f"mfi_{period}"] = normalize(ta.MFI(dataframe, timeperiod=period), 0, 100)

            # MINUS_DI             Minus Directional Indicator
            dataframe[f"minus_di_{period}"] = normalize(ta.MINUS_DI(dataframe, timeperiod=period), 0, 100)

            # PLUS_DI              Plus Directional Indicator
            dataframe[f"plus_di_{period}"] = normalize(ta.PLUS_DI(dataframe, timeperiod=period), 0, 100)

            # Williams %R
            dataframe[f"willr_{period}"] = normalize(ta.WILLR(dataframe, timeperiod=period), -100, 0)

            # RSI
            dataframe[f"rsi_{period}"] = normalize(ta.RSI(dataframe, timeperiod=period), 0, 100)

            # Inverse Fisher transform on RSI: values [-1.0, 1.0] (https://goo.gl/2JGGoy)
            rsi = 0.1 * (dataframe[f"rsi_{period}"] - 50)
            dataframe[f"fisher_rsi_{period}"] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)
            dataframe[f"fisher_rsi_{period}"] = normalize(dataframe[f"fisher_rsi_{period}"], -1, 1)

            # STOCHRSI - Stochastic Relative Strength Index
            stoch_rsi = ta.STOCHRSI(dataframe, timeperiod=period)
            dataframe[f"stochrsi_k_{period}"] = normalize(stoch_rsi["fastk"], 0, 100)
            dataframe[f"stochrsi_d_{period}"] = normalize(stoch_rsi["fastd"], 0, 100)

            # # CORREL - Pearson's Correlation Coefficient (r)
            # dataframe[f'correl_{period}'] = normalize(ta.CORREL(dataframe, timeperiod=period), -1, 1)  # this is buggy

            # LINEARREG_ANGLE - Linear Regression Angle
            dataframe[f"linangle_{period}"] = normalize(ta.LINEARREG_ANGLE(dataframe, timeperiod=period), -90, 90)

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Raw data from the exchange and parsed by parse_ticker_dataframe()
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """
        dataframe = self.generate_features(dataframe)

        informative_df = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe=self.informative_timeframe)
        informative_df = self.generate_features(informative_df)
        dataframe = merge_informative_pair(
            dataframe, informative_df, self.timeframe, self.informative_timeframe, ffill=True
        )
        dataframe = dataframe[dataframe.columns[~dataframe.columns.isin(self.informative_freqtrade_columns)]]

        informative_btc = self.dp.get_pair_dataframe("BTC/USDT", timeframe=self.informative_timeframe)
        informative_btc = self.generate_features(informative_btc)
        informative_btc.rename(columns=lambda s: s if s == "date" else f"{s}_btc", inplace=True)
        dataframe = merge_informative_pair(
            dataframe, informative_btc, self.timeframe, self.informative_timeframe, ffill=True
        )
        dataframe = dataframe[
            dataframe.columns[~dataframe.columns.isin(self.btc_freqtrade_columns + self.informative_freqtrade_columns)]
        ]

        dataframe.fillna(0, inplace=True)

        self.indicators = dataframe[dataframe.columns[~dataframe.columns.isin(self.freqtrade_columns)]].to_numpy()

        assert (
            self.indicators.max() < 1.00001 and self.indicators.min() > -0.00001
        ), "Error, values are not normalized!"

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        # dataframe['buy'] = self.rl_model_predict(dataframe)
        action = self.rl_model_predict(dataframe)
        dataframe["buy"] = (action == 0).astype("int")

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """

        action = self.rl_model_predict(dataframe)
        dataframe["sell"] = (action == 1).astype("int")

        return dataframe

    def rl_model_predict(self, dataframe):

        output = pd.DataFrame(np.full((len(dataframe), 1), 2))

        # start_time = time.time()
        def _predict(window):
            observation = self.indicators[window.index]
            res, _ = self.model.predict(observation, deterministic=True)
            return res

        output = output.rolling(window=self.window_size).apply(_predict)
        # print("--- rolling %s seconds ---" % (time.time() - start_time))

        # start_time = time.time()
        # for window in range(self.startup_candle_count, len(dataframe)):
        #     start = window - self.window_size + 1
        #     end = window
        #     observation = self.indicators[start:end+1]
        #     res, _ = self.model.predict(observation, deterministic=True)
        #     output.iloc[end] = res
        # print("--- forloop %s seconds ---" % (time.time() - start_time))

        return output


class FreqGym_normalizedDCA(FreqGym_normalized):
    # DCA
    position_adjustment_enable = True
    max_entry_position_adjustment = 2
    max_dca_multiplier = 1.25
    dca_stake_multiplier = 1.25

    # This is called when placing the initial order (opening trade)
    def custom_stake_amount(
        self,
        pair: str,
        current_time,
        current_rate: float,
        proposed_stake: float,
        min_stake: float,
        max_stake: float,
        **kwargs,
    ) -> float:

        if (self.config["position_adjustment_enable"] == True) and (self.config["stake_amount"] == "unlimited"):
            return self.wallets.get_total_stake_amount() / self.config["max_open_trades"] / self.max_dca_multiplier
        else:
            return proposed_stake

    def adjust_trade_position(
        self,
        trade,
        current_time,
        current_rate: float,
        current_profit: float,
        min_stake: float,
        max_stake: float,
        **kwargs,
    ):
        if current_profit > -0.05:
            return None

        # Obtain pair dataframe (just to show how to access it)
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        # Only buy when not actively falling price.
        last_candle = dataframe.iloc[-1].squeeze()
        previous_candle = dataframe.iloc[-2].squeeze()
        # if previous_candle['close'] < last_candle['close']:
        #     return None

        filled_buys = trade.select_filled_orders("buy")
        count_of_buys = len(filled_buys)

        if 0 < count_of_buys <= self.max_entry_position_adjustment:
            try:
                # This returns first order stake size
                stake_amount = filled_buys[0].cost
                # This then calculates current safety order size
                stake_amount = stake_amount * self.dca_stake_multiplier
                return stake_amount
            except Exception as exception:
                return None

        return None


def normalize(data, min_value, max_value):
    return (data - min_value) / (max_value - min_value)
