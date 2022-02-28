import pandas as pd
import numpy as np


class BollingerBand:
    def __init__(self, window: int=20, m: int=3):
        self.window = window
        self.m = m

    def fit(self, hist_high: pd.Series, hist_low: pd.Series, hist_close: pd.Series):
        typical_price = (hist_high + hist_low + hist_close) / 3
        sma = typical_price.rolling(window=self.window).mean()
        std = typical_price.rolling(window=self.window).std()

        bollinger_upper = sma + std * self.m
        bollinger_lower = sma - std * self.m

        buy = np.where(hist_close <= bollinger_upper, 1, np.nan) * hist_close
        sell = np.where(hist_close >= bollinger_upper, 1, np.nan) * hist_close

        is_invested = np.where(sell >= 0, False, True)
        log_ret = np.log1p(hist_close.pct_change(1))
        log_ret = is_invested * log_ret

        self.buy = buy
        self.sell = sell
        self.log_ret = np.exp(log_ret) - 1

