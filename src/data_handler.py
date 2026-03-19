import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, time


class DataHandler:
    def fetch_ohlcv(self, ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if df.empty:
            raise ValueError(f"No data returned for {ticker}")
        df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
        df.index = pd.to_datetime(df.index)
        df = df[["open", "high", "low", "close", "volume"]].dropna()
        return df

    def fetch_vix(self) -> float:
        df = yf.download("^VIX", period="5d", interval="1d", progress=False, auto_adjust=True)
        if df.empty:
            return 20.0
        # Handle both MultiIndex columns (newer yfinance) and flat columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]
        val = df["close"].dropna().iloc[-1]
        return float(val) if np.isscalar(val) else float(val.iloc[0])

    def fetch_live_quote(self, ticker: str) -> dict:
        """Fetch near-real-time quote using yf.Ticker (not delayed download)."""
        t = yf.Ticker(ticker)
        info = t.fast_info
        try:
            last = float(info["lastPrice"])
        except (KeyError, TypeError):
            last = float(info.get("previousClose", 0))
        try:
            prev_close = float(info["previousClose"])
        except (KeyError, TypeError):
            prev_close = last

        change_pct = ((last - prev_close) / prev_close * 100) if prev_close else 0.0

        return {
            "ticker": ticker,
            "price": round(last, 2),
            "prev_close": round(prev_close, 2),
            "change_pct": round(change_pct, 2),
            "market_cap": info.get("marketCap", 0),
        }

    def fetch_premarket(self, ticker: str) -> float:
        """Fetch pre-market / after-hours price if available."""
        t = yf.Ticker(ticker)
        try:
            pre = t.info.get("preMarketPrice")
            if pre:
                return float(pre)
        except Exception:
            pass
        # Fallback: use 1m interval last bar
        try:
            df = yf.download(ticker, period="1d", interval="1m", progress=False,
                             auto_adjust=True, prepost=True)
            if not df.empty:
                col = "Close" if "Close" in df.columns else df.columns[3]
                return float(df[col].dropna().iloc[-1])
        except Exception:
            pass
        return self.fetch_live_quote(ticker)["price"]

    def fetch_options_iv(self, ticker: str) -> dict:
        """Fetch implied volatility from actual Yahoo options chain (free, no API key)."""
        t = yf.Ticker(ticker)
        try:
            expirations = t.options
        except Exception:
            return {"iv_atm_call": None, "iv_atm_put": None, "iv_rank": None}

        if not expirations:
            return {"iv_atm_call": None, "iv_atm_put": None, "iv_rank": None}

        # Use nearest expiration
        chain = t.option_chain(expirations[0])
        price = self.fetch_live_quote(ticker)["price"]

        # Find ATM options (closest strike to current price)
        calls = chain.calls
        puts = chain.puts

        if calls.empty or puts.empty:
            return {"iv_atm_call": None, "iv_atm_put": None, "iv_rank": None}

        atm_call_idx = (calls["strike"] - price).abs().idxmin()
        atm_put_idx = (puts["strike"] - price).abs().idxmin()

        iv_call = float(calls.loc[atm_call_idx, "impliedVolatility"])
        iv_put = float(puts.loc[atm_put_idx, "impliedVolatility"])

        # IV Rank: compare ATM IV to rolling historical vol
        try:
            df = self.fetch_ohlcv(ticker, period="1y", interval="1d")
            hist_vol = df["close"].pct_change().rolling(21).std() * np.sqrt(252)
            hist_vol = hist_vol.dropna()
            current_iv = (iv_call + iv_put) / 2
            iv_low = float(hist_vol.min())
            iv_high = float(hist_vol.max())
            iv_rank = ((current_iv - iv_low) / (iv_high - iv_low) * 100) if iv_high > iv_low else 50.0
            iv_rank = max(0, min(100, iv_rank))
        except Exception:
            iv_rank = None

        return {
            "iv_atm_call": round(iv_call * 100, 1),  # as percentage
            "iv_atm_put": round(iv_put * 100, 1),
            "iv_rank": round(iv_rank, 1) if iv_rank is not None else None,
            "nearest_expiry": expirations[0],
        }

    def fetch_options_chain(self, ticker: str, expiry_index: int = 0) -> dict:
        """Fetch full options chain for a given expiry. Returns calls + puts DataFrames."""
        t = yf.Ticker(ticker)
        expirations = t.options
        if not expirations:
            return {"calls": pd.DataFrame(), "puts": pd.DataFrame(), "expiry": None}

        idx = min(expiry_index, len(expirations) - 1)
        chain = t.option_chain(expirations[idx])
        return {
            "calls": chain.calls,
            "puts": chain.puts,
            "expiry": expirations[idx],
            "all_expiries": list(expirations),
        }

    def market_status(self) -> str:
        """Check if US market is open, pre-market, or closed."""
        now = datetime.utcnow()
        # US market hours in UTC: pre-market 8:00-13:30, regular 13:30-20:00, after 20:00-01:00
        hour = now.hour
        minute = now.minute
        t = hour * 60 + minute
        if 800 <= t < 810:  # ~pre-market open
            return "pre-market"
        if 810 <= t < 1200:
            return "pre-market"
        if 1330 <= t < 2000:
            return "open"
        if 2000 <= t < 2400:
            return "after-hours"
        return "closed"
