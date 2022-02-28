import pandas as pd
import numpy as np


class StochasticOscillator:
    def __init__(self, window_low: int=14, window_high: int=14):
        self.window_low = window_low
        self.window_high = window_high

    def fit(self, hist_high: pd.Series, hist_low: pd.Series, hist: pd.Series):
        c = hist.shift(1)
        l14 = hist_low.rolling(self.window_low).min()
        h14 = hist_high.rolling(self.window_high).max()
        k = ((c - l14) / (h14 - l14)) * 100

        buy = np.where(k < 20, hist, np.nan)
        sell = np.where(k > 80, hist, np.nan)

        is_invested = np.where(sell > 0, False, True)
        log_ret = np.log1p(hist.pct_change(1))
        log_ret = is_invested * log_ret

        self.buy = buy
        self.sell = sell
        self.log_ret = np.exp(log_ret) - 1

