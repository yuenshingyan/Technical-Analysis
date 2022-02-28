import pandas as pd
import numpy as np


class IchimokuCloud:
    def __init__(self, tenkan_window: int=9, kijun_window: int=26, senkouB_window: int=52):
        self.tenkan_window = tenkan_window
        self.kijun_window = kijun_window
        self.senkouB_window = senkouB_window

    def fit(self, hist_high: pd.Series, hist_low: pd.Series, hist: pd.Series):
        tenkan_sen = (hist_high.rolling(self.tenkan_window).max() + hist_low.rolling(self.tenkan_window).min()) / 2
        kijun_sen = (hist_high.rolling(self.kijun_window).max() + hist_low.rolling(self.kijun_window).min()) / 2
        chikou_span = hist.shift(-26)
        senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        senkou_b = ((hist_high.rolling(self.senkouB_window).max() + hist_low.rolling(self.senkouB_window).min()) / 2).shift(52)
        shade = np.where(senkou_a >= senkou_b, 1, 0)
        buy = np.where(senkou_a >= senkou_b, hist, 0)
        sell = np.where(senkou_a <= senkou_b, hist, 0)

        is_invested = np.where(sell > 0, False, True)
        log_ret = np.log1p(hist.pct_change(1))
        log_ret = is_invested * log_ret

        self.buy = buy
        self.sell = sell
        self.log_ret = np.exp(log_ret) - 1

