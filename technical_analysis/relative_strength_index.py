import pandas as pd
import numpy as np


class RelativeStrengthIndex:
    def __init__(self, window: int=14, threshold_upper: int=70, threshold_lower: int=30):
        self.window = window
        self.threshold_upper = threshold_upper
        self.threshold_lower = threshold_lower

    def fit(self, hist: pd.Series):
        avg_gain = pd.Series(np.where(hist > 0, hist, 0)).rolling(self.window).mean()
        avg_loss = pd.Series(np.where(hist < 0, abs(hist), 0)).rolling(self.window).mean()
        rsi = 100 - (100 / (1 + avg_gain / avg_loss))

        sell = np.where(hist > self.threshold_upper, True, False)
        buy = np.where(hist < self.threshold_lower, True, False)

        is_invested = np.where(sell == True, False, True)
        log_ret = np.log1p(hist.pct_change(1))
        log_ret = is_invested * log_ret

        self.buy = buy
        self.sell = sell
        self.log_ret = np.exp(log_ret) - 1

