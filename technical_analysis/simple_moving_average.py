import pandas as pd
import numpy as np


class SimpleMovingAverage:
    def __init__(self, window_slow: int, window_fast: int):
        self.window_slow = window_slow
        self.window_fast = window_fast

    def fit(self, hist: pd.Series):
        slow_sma = hist.rolling(self.window_slow).mean()
        fast_sma = hist.rolling(self.window_fast).mean()

        signal = np.where(fast_sma >= slow_sma, 1, 0)
        signal = pd.Series(signal)
        prev_signal = signal.shift(1)

        buy = (prev_signal == 0) & (signal == 1)  # Fast < Slow --> Fast > Slow
        sell = (prev_signal == 1) & (signal == 0)  # Fast > Slow --> Fast < Slow

        is_invested = np.where(sell == True, False, True)
        log_ret = np.log1p(hist.pct_change(1))
        log_ret = is_invested * log_ret

        self.buy = buy
        self.sell = sell
        self.log_ret = np.exp(log_ret) - 1

