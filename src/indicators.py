import pandas as pd
import ta


class FeatureEngineer:
    def add_all(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        close = df["close"]
        high = df["high"]
        low = df["low"]

        df["RSI_14"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()

        macd = ta.trend.MACD(close=close)
        df["MACD"] = macd.macd()
        df["MACD_signal"] = macd.macd_signal()

        df["EMA_20"] = ta.trend.EMAIndicator(close=close, window=20).ema_indicator()
        df["EMA_50"] = ta.trend.EMAIndicator(close=close, window=50).ema_indicator()
        df["EMA_200"] = ta.trend.EMAIndicator(close=close, window=200).ema_indicator()

        df["ATR_14"] = ta.volatility.AverageTrueRange(
            high=high, low=low, close=close, window=14
        ).average_true_range()

        df = df.dropna()
        return df
