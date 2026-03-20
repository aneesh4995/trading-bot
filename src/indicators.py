import pandas as pd
import numpy as np
import yfinance as yf
import ta


class FeatureEngineer:
    def add_all(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # ── Original features ──
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

        # ── NEW: Volume ratio (today's volume / 20-day avg) ──
        vol_ma = volume.rolling(20).mean()
        df["volume_ratio"] = volume / vol_ma

        # ── NEW: Price distance from EMA_200 (mean reversion signal) ──
        df["dist_ema200"] = (close - df["EMA_200"]) / df["EMA_200"] * 100

        # ── NEW: Day of week (Monday=0, Friday=4) ──
        df["day_of_week"] = df.index.dayofweek

        # ── NEW: RSI divergence (5-day ROC of RSI vs 5-day ROC of price) ──
        rsi = df["RSI_14"]
        rsi_roc = rsi - rsi.shift(5)
        price_roc = close.pct_change(5) * 100
        df["rsi_divergence"] = rsi_roc - price_roc

        df = df.dropna()
        return df

    def add_vix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Merge VIX daily close as a feature. Call AFTER add_all()."""
        vix = yf.download("^VIX", period="3y", interval="1d", progress=False, auto_adjust=True)
        if vix.empty:
            df["VIX"] = 20.0
            return df
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = [c[0].lower() for c in vix.columns]
        else:
            vix.columns = [c.lower() for c in vix.columns]
        vix = vix[["close"]].rename(columns={"close": "VIX"})
        vix.index = pd.to_datetime(vix.index).tz_localize(None)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df.merge(vix, left_index=True, right_index=True, how="left")
        df["VIX"] = df["VIX"].ffill().fillna(20.0)
        return df

    def add_put_call_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add equity put/call ratio proxy from CBOE via Yahoo (^PCCE not available,
        so we approximate from VIX: higher VIX correlates with higher put/call ratio)."""
        # Attempt to fetch actual CBOE put/call ratio
        try:
            pcr = yf.download("^PCCE", period="3y", interval="1d", progress=False, auto_adjust=True)
            if not pcr.empty:
                if isinstance(pcr.columns, pd.MultiIndex):
                    pcr.columns = [c[0].lower() for c in pcr.columns]
                else:
                    pcr.columns = [c.lower() for c in pcr.columns]
                pcr = pcr[["close"]].rename(columns={"close": "put_call_ratio"})
                pcr.index = pd.to_datetime(pcr.index).tz_localize(None)
                df = df.merge(pcr, left_index=True, right_index=True, how="left")
                df["put_call_ratio"] = df["put_call_ratio"].ffill().fillna(0.85)
                return df
        except Exception:
            pass

        # Fallback: derive from VIX (empirical correlation)
        if "VIX" in df.columns:
            df["put_call_ratio"] = 0.5 + (df["VIX"] / 100)
        else:
            df["put_call_ratio"] = 0.85
        return df
