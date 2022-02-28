import pandas as pd
import numpy as np


class OnBalanceVolume:
    def __init__(self, window: int):
        self.window = window

    def fit(self, hist: pd.Series, vol: pd.Series):
        obv = []
        obv.append(0)

        for i in range(1, len(hist)):
            if hist.iloc[i] > hist.iloc[i - 1]:
                obv.append(obv[-1] + vol.iloc[i])

            elif hist.iloc[i] < hist.iloc[i - 1]:
                obv.append(obv[-1] - vol.iloc[i])

            else:
                obv.append(obv[-1])

        obv = pd.Series(obv)

        ema = obv.ewm(span=self.window).mean()
        buy = np.where(obv > ema, hist, np.nan)
        sell = np.where(obv < ema, hist, np.nan)

        is_invested = np.where(sell > 0, False, True)
        log_ret = np.log1p(hist.pct_change(1))
        log_ret = is_invested * log_ret

        self.buy = buy
        self.sell = sell
        self.log_ret = np.exp(log_ret) - 1

