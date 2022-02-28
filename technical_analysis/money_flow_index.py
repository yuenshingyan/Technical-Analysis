import pandas as pd
import numpy as np


class MoneyFlowIndex:
    def __init__(self, threshold_upper: int=80, threshold_lower: int=20, window: int=14):
        self.threshold_upper = threshold_upper
        self.threshold_lower = threshold_lower
        self.window = window

    def fit(self, hist_high: pd.Series, hist_low: pd.Series, hist: pd.Series, vol: pd.Series):
        typical_price = (hist_high + hist_low + hist) / 3
        raw_money_flow = typical_price * vol
        money_flow_ratio = pd.Series(np.where(raw_money_flow.diff(1) > 0, raw_money_flow, 0)).rolling(14).sum() / pd.Series(np.where(raw_money_flow.diff(1) < 0, raw_money_flow, 0)).rolling(self.window).sum()
        mfi = 100 - 100 / (1 + money_flow_ratio)
        buy = np.where(mfi > self.threshold_upper, hist, np.nan)
        sell = np.where(mfi < self.threshold_lower, hist, np.nan)

        is_invested = np.where(sell > 0, False, True)
        log_ret = np.log1p(hist.pct_change(1))
        log_ret = is_invested * log_ret

        self.buy = buy
        self.sell = sell
        self.log_ret = np.exp(log_ret) - 1

