import pandas as pd
import numpy as np


class MovingAverageConvergenceDivergence:
    def __init__(self, window_slow: int, window_fast: int):
        self.window_slow = window_slow
        self.window_fast = window_fast

    def _buy_sell(self, hist, macd, signal):
        Buy = []
        Sell = []
        flag = -1

        for i in range(0, len(signal)):
            if macd.iloc[i] > signal.iloc[i]:
                Sell.append(np.nan)
                if flag != 1:
                    Buy.append(hist.iloc[i])
                    flag = 1
                else:
                    Buy.append(np.nan)
            elif macd.iloc[i] < signal.iloc[i]:
                Buy.append(np.nan)
                if flag != 0:
                    Sell.append(hist.iloc[i])
                    flag = 0
                else:
                    Sell.append(np.nan)
            else:
                Buy.append(np.nan)
                Sell.append(np.nan)

        return (Buy, Sell)

    def fit(self, hist: pd.Series):
        slow_sma = hist.ewm(self.window_slow, adjust=False).mean()
        fast_sma = hist.ewm(self.window_fast, adjust=False).mean()

        macd = fast_sma - slow_sma
        signal = macd.ewm(span=9, adjust=False).mean()

        buy_sell = self._buy_sell(hist, macd, signal)
        buy, sell = buy_sell[0], buy_sell[1]

        is_invested = np.where(sell == True, False, True)
        log_ret = np.log1p(hist.pct_change(1))
        log_ret = is_invested * log_ret

        self.buy = buy
        self.sell = sell
        self.log_ret = np.exp(log_ret) - 1

