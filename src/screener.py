"""
Daily GO / WAIT screener for Bull Put Credit Spreads.
Checks all entry conditions from Ravish's methodology + ML signal.

Usage:
    python3 screener.py              # check SPY and QQQ
    python3 screener.py AAPL MSFT    # check specific tickers
"""

import sys
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
from dataclasses import dataclass

from data_handler import DataHandler
from indicators import FeatureEngineer
from predictor import SwingPredictor


# ── Known market events (FOMC 2026 dates, update as needed) ─────

FOMC_DATES_2026 = [
    "2026-01-28", "2026-01-29",
    "2026-03-17", "2026-03-18",
    "2026-05-05", "2026-05-06",
    "2026-06-16", "2026-06-17",
    "2026-07-28", "2026-07-29",
    "2026-09-15", "2026-09-16",
    "2026-11-03", "2026-11-04",
    "2026-12-15", "2026-12-16",
]

# CPI release dates are typically ~12th of each month
# NFP is first Friday of each month
# We flag if today or tomorrow is within 1 day of these


@dataclass
class Check:
    name: str
    passed: bool
    value: str
    threshold: str
    weight: float = 1.0


@dataclass
class ScreenResult:
    ticker: str
    checks: list
    timestamp: str

    @property
    def score(self) -> float:
        total_weight = sum(c.weight for c in self.checks)
        passed_weight = sum(c.weight for c in self.checks if c.passed)
        return passed_weight / total_weight if total_weight > 0 else 0.0

    @property
    def verdict(self) -> str:
        s = self.score
        if s >= 0.85:
            return "GO"
        elif s >= 0.60:
            return "CAUTION"
        else:
            return "WAIT"


class CreditSpreadScreener:
    VIX_LOW = 12.0
    VIX_HIGH = 28.0
    VIX_STOP = 30.0
    RSI_FLOOR = 30.0
    ATR_SPIKE_MULT = 1.5
    IV_RANK_MIN = 30.0
    ML_CONVICTION_MIN = 0.65

    def __init__(self):
        self.handler = DataHandler()
        self.engineer = FeatureEngineer()

    def _fetch_vix(self) -> float:
        return self.handler.fetch_vix()

    def _is_near_fomc(self) -> bool:
        today = datetime.date.today()
        for d in FOMC_DATES_2026:
            event = datetime.date.fromisoformat(d)
            if abs((event - today).days) <= 1:
                return True
        return False

    def _is_near_monthly_event(self) -> bool:
        """Approximate: CPI ~12th, NFP ~first Friday."""
        today = datetime.date.today()
        tomorrow = today + datetime.timedelta(days=1)
        # CPI window: 11th-13th
        if today.day in (11, 12, 13) or tomorrow.day in (11, 12, 13):
            return True
        # NFP window: first Friday (day 1-7 and weekday=4)
        for d in (today, tomorrow):
            if d.day <= 7 and d.weekday() == 4:
                return True
        return False

    def _compute_iv_rank(self, ticker: str) -> float:
        """IV Rank from real options chain if available, else historical vol estimate."""
        iv_data = self.handler.fetch_options_iv(ticker)
        if iv_data.get("iv_rank") is not None:
            return iv_data["iv_rank"]
        # Fallback to historical vol estimate
        df = self.handler.fetch_ohlcv(ticker, period="1y", interval="1d")
        returns = df["close"].pct_change().dropna()
        rolling_vol = returns.rolling(21).std() * np.sqrt(252)
        rolling_vol = rolling_vol.dropna()
        if len(rolling_vol) < 30:
            return 50.0
        current = float(rolling_vol.iloc[-1])
        low = float(rolling_vol.min())
        high = float(rolling_vol.max())
        if high == low:
            return 50.0
        return ((current - low) / (high - low)) * 100

    def screen(self, ticker: str) -> ScreenResult:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        checks = []

        # 1. Fetch data and compute indicators
        df = self.engineer.add_all(self.handler.fetch_ohlcv(ticker, period="2y"))
        latest = df.iloc[-1]

        # Use live quote if market is open, else daily close
        market = self.handler.market_status()
        if market in ("open", "pre-market", "after-hours"):
            try:
                live = self.handler.fetch_live_quote(ticker)
                close = live["price"]
            except Exception:
                close = float(latest["close"])
        else:
            close = float(latest["close"])
        rsi = float(latest["RSI_14"])
        ema20 = float(latest["EMA_20"])
        ema50 = float(latest["EMA_50"])
        ema200 = float(latest["EMA_200"])
        atr = float(latest["ATR_14"])
        atr_avg = float(df["ATR_14"].tail(20).mean())

        # 2. VIX check
        vix = self._fetch_vix()
        vix_ok = self.VIX_LOW <= vix <= self.VIX_HIGH
        checks.append(Check(
            name="VIX Level",
            passed=vix_ok,
            value=f"{vix:.1f}",
            threshold=f"{self.VIX_LOW}–{self.VIX_HIGH}",
            weight=2.0,
        ))

        # VIX hard stop
        if vix > self.VIX_STOP:
            checks.append(Check(
                name="VIX Hard Stop",
                passed=False,
                value=f"{vix:.1f}",
                threshold=f"< {self.VIX_STOP}",
                weight=3.0,
            ))

        # 3. Trend: price above EMA_20
        trend_ok = close > ema20
        checks.append(Check(
            name="Price > EMA_20",
            passed=trend_ok,
            value=f"${close:.2f} vs ${ema20:.2f}",
            threshold="Price above EMA_20",
            weight=1.5,
        ))

        # 4. Not in confirmed downtrend (below both EMA_50 and EMA_200)
        not_downtrend = not (close < ema50 and close < ema200)
        checks.append(Check(
            name="Not in downtrend",
            passed=not_downtrend,
            value=f"EMA50=${ema50:.0f} EMA200=${ema200:.0f}",
            threshold="Not below both EMAs",
            weight=2.0,
        ))

        # 5. RSI not deeply oversold
        rsi_ok = rsi > self.RSI_FLOOR
        checks.append(Check(
            name="RSI > 30",
            passed=rsi_ok,
            value=f"{rsi:.1f}",
            threshold=f"> {self.RSI_FLOOR}",
            weight=1.5,
        ))

        # 6. ATR not spiking
        atr_ratio = atr / atr_avg if atr_avg > 0 else 1.0
        atr_ok = atr_ratio < self.ATR_SPIKE_MULT
        checks.append(Check(
            name="ATR not spiking",
            passed=atr_ok,
            value=f"{atr:.2f} ({atr_ratio:.1f}x avg)",
            threshold=f"< {self.ATR_SPIKE_MULT}x 20d avg",
            weight=1.0,
        ))

        # 7. IV Rank
        iv_rank = self._compute_iv_rank(ticker)
        iv_ok = iv_rank >= self.IV_RANK_MIN
        checks.append(Check(
            name="IV Rank",
            passed=iv_ok,
            value=f"{iv_rank:.0f}%",
            threshold=f">= {self.IV_RANK_MIN}%",
            weight=1.5,
        ))

        # 8. No FOMC
        fomc_clear = not self._is_near_fomc()
        checks.append(Check(
            name="No FOMC",
            passed=fomc_clear,
            value="Clear" if fomc_clear else "FOMC nearby!",
            threshold="Not within 1 day of FOMC",
            weight=2.0,
        ))

        # 9. No CPI/NFP
        events_clear = not self._is_near_monthly_event()
        checks.append(Check(
            name="No CPI/NFP",
            passed=events_clear,
            value="Clear" if events_clear else "Event nearby!",
            threshold="Not near CPI or NFP",
            weight=1.0,
        ))

        # 10. ML signal
        predictor = SwingPredictor()
        predictor.train(df)
        label, prob = predictor.predict(df)
        ml_bullish = label == 1 and prob >= self.ML_CONVICTION_MIN
        checks.append(Check(
            name="ML Bullish Signal",
            passed=ml_bullish,
            value=f"{'Up' if label==1 else 'Down'} @ {prob:.1%}",
            threshold=f"Up + >{self.ML_CONVICTION_MIN:.0%}",
            weight=1.5,
        ))

        return ScreenResult(ticker=ticker, checks=checks, timestamp=now)


def print_screen(result: ScreenResult):
    width = 64
    sep = "─" * width
    verdict = result.verdict
    score = result.score

    if verdict == "GO":
        badge = ">>> GO — OPEN SPREAD <<<"
    elif verdict == "CAUTION":
        badge = "~~ CAUTION — CHECK CONDITIONS ~~"
    else:
        badge = "--- WAIT — DO NOT TRADE ---"

    print(f"\n┌{sep}┐")
    print(f"│{'  CREDIT SPREAD SCREENER: ' + result.ticker:^{width}}│")
    print(f"│{'  ' + result.timestamp:^{width}}│")
    print(f"├{sep}┤")

    for c in result.checks:
        icon = "PASS" if c.passed else "FAIL"
        line = f"  [{icon}] {c.name:<22} {c.value:<20} ({c.threshold})"
        # Truncate if too long
        print(f"│{line[:width]:<{width}}│")

    print(f"├{sep}┤")
    print(f"│{'':>{width}}│")
    print(f"│{'  Score: ' + f'{score:.0%}':^{width}}│")
    print(f"│{badge:^{width}}│")
    print(f"│{'':>{width}}│")

    if verdict == "GO":
        print(f"├{sep}┤")
        print(f"│{'  TRADE SETUP:':^{width}}│")
        print(f"│{'  Sell 50-delta put (ATM), buy put 10pts lower':^{width}}│")
        print(f"│{'  DTE: 14 days  |  Take profit: 50%':^{width}}│")
        print(f"│{'  No hard stop — roll if threatened':^{width}}│")

    print(f"└{sep}┘")


def main():
    tickers = sys.argv[1:] if len(sys.argv) > 1 else ["SPY", "QQQ"]
    screener = CreditSpreadScreener()

    for ticker in tickers:
        print(f"\nScreening {ticker}...")
        result = screener.screen(ticker)
        print_screen(result)


if __name__ == "__main__":
    main()
