#!/usr/bin/env python3
"""
ML-Driven Strategy Selection Backtest vs Bull Put Spread Only.

Compares two approaches over 4 years on SPY & QQQ:
  A) Bull Put Only   — enter bull put credit spread at every 14-day window
  B) ML Multi-Strat  — use EnsemblePredictor + RavishStrategyEngine to pick:
                        bull put, bear call, LEAPS, skip (no trade), etc.

Both use Black-Scholes option pricing from strategy_backtest.py.
Adaptive position sizing (VIX regime) applied to both.

Usage:
    python3 ml_strategy_backtest.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from data_handler import DataHandler
from indicators import FeatureEngineer
from predictor import EnsemblePredictor
from ravish_strategy import RavishStrategyEngine
from risk_manager import AdaptiveSizer
from strategy_backtest import (
    bs_price, find_strike_for_delta, _rolling_vol,
    Trade, StrategyResult,
)


# ── Config ───────────────────────────────────────────────────────────────

STARTING_CAPITAL = 1000.0
SPREAD_WIDTH = 10
BULL_PUT_DTE = 14
BEAR_CALL_DTE = 14
LEAPS_DTE_DAYS = 90    # hold period for LEAPS in backtest
LEAPS_EXPIRY = 1.0     # 12-month expiry
TP_PCT = 0.50           # take profit at 50%
MIN_TRAIN_ROWS = 252    # 1 year of data before first prediction
RETRAIN_EVERY = 63      # retrain ML model every ~quarter


# ── Individual Strategy Simulators ───────────────────────────────────────

def simulate_bull_put(df, i, vol, dte=BULL_PUT_DTE, width=SPREAD_WIDTH,
                      tp_pct=TP_PCT, vix=20.0):
    """Simulate one bull put credit spread trade starting at index i."""
    if i + dte >= len(df):
        return None
    S = df["close"].iloc[i]
    sigma = vol.iloc[i]
    T = dte / 252

    short_K = round(S)
    long_K = short_K - width

    credit = bs_price(S, short_K, T, sigma) - bs_price(S, long_K, T, sigma)
    max_loss = width - credit
    if credit <= 0 or max_loss <= 0:
        return None

    for j in range(i + 1, min(i + dte + 1, len(df))):
        S_j = df["close"].iloc[j]
        T_j = max((dte - (j - i)), 0) / 252
        sigma_j = vol.iloc[j]
        spread_val = bs_price(S_j, short_K, T_j, sigma_j) - bs_price(S_j, long_K, T_j, sigma_j)
        profit = credit - spread_val

        if profit >= credit * tp_pct:
            return Trade(
                entry_date=str(df.index[i].date()),
                exit_date=str(df.index[j].date()),
                entry_price=credit, exit_price=spread_val,
                pnl=profit, pnl_pct=profit / max_loss, won=True,
                vix_at_entry=vix,
            )

    # Hold to expiry
    S_exp = df["close"].iloc[min(i + dte, len(df) - 1)]
    intrinsic = max(short_K - S_exp, 0) - max(long_K - S_exp, 0)
    pnl = credit - intrinsic
    return Trade(
        entry_date=str(df.index[i].date()),
        exit_date=str(df.index[min(i + dte, len(df) - 1)].date()),
        entry_price=credit, exit_price=intrinsic,
        pnl=pnl, pnl_pct=pnl / max_loss, won=pnl > 0,
        vix_at_entry=vix,
    )


def simulate_bear_call(df, i, vol, dte=BEAR_CALL_DTE, width=SPREAD_WIDTH,
                       tp_pct=TP_PCT, vix=20.0):
    """Simulate one bear call credit spread trade."""
    if i + dte >= len(df):
        return None
    S = df["close"].iloc[i]
    sigma = vol.iloc[i]
    T = dte / 252

    short_K = round(S)
    long_K = short_K + width

    credit = bs_price(S, short_K, T, sigma, opt="call") - bs_price(S, long_K, T, sigma, opt="call")
    max_loss = width - credit
    if credit <= 0 or max_loss <= 0:
        return None

    for j in range(i + 1, min(i + dte + 1, len(df))):
        S_j = df["close"].iloc[j]
        T_j = max((dte - (j - i)), 0) / 252
        sigma_j = vol.iloc[j]
        spread_val = (bs_price(S_j, short_K, T_j, sigma_j, opt="call")
                      - bs_price(S_j, long_K, T_j, sigma_j, opt="call"))
        profit = credit - spread_val

        if profit >= credit * tp_pct:
            return Trade(
                entry_date=str(df.index[i].date()),
                exit_date=str(df.index[j].date()),
                entry_price=credit, exit_price=spread_val,
                pnl=profit, pnl_pct=profit / max_loss, won=True,
                vix_at_entry=vix,
            )

    S_exp = df["close"].iloc[min(i + dte, len(df) - 1)]
    intrinsic = max(S_exp - short_K, 0) - max(S_exp - long_K, 0)
    pnl = credit - intrinsic
    return Trade(
        entry_date=str(df.index[i].date()),
        exit_date=str(df.index[min(i + dte, len(df) - 1)].date()),
        entry_price=credit, exit_price=intrinsic,
        pnl=pnl, pnl_pct=pnl / max_loss, won=pnl > 0,
        vix_at_entry=vix,
    )


def simulate_leaps(df, i, vol, hold_days=LEAPS_DTE_DAYS, tp_pct=TP_PCT, vix=20.0):
    """Simulate one LEAPS swing trade (buy 70-delta call, 12m expiry, hold up to 90d)."""
    if i + hold_days >= len(df):
        return None
    S = df["close"].iloc[i]
    sigma = vol.iloc[i]
    T_entry = LEAPS_EXPIRY

    K = find_strike_for_delta(S, T_entry, sigma, 0.70, opt="call")
    entry_cost = bs_price(S, K, T_entry, sigma, opt="call")
    if entry_cost <= 0:
        return None

    for j in range(i + 1, min(i + hold_days + 1, len(df))):
        S_j = df["close"].iloc[j]
        T_j = T_entry - (j - i) / 252
        sigma_j = vol.iloc[j]
        current_val = bs_price(S_j, K, T_j, sigma_j, opt="call")
        profit_pct = (current_val - entry_cost) / entry_cost

        if profit_pct >= tp_pct:
            return Trade(
                entry_date=str(df.index[i].date()),
                exit_date=str(df.index[j].date()),
                entry_price=entry_cost, exit_price=current_val,
                pnl=current_val - entry_cost, pnl_pct=profit_pct, won=True,
                vix_at_entry=vix,
            )

    # Hold to end of period
    j = min(i + hold_days, len(df) - 1)
    S_exit = df["close"].iloc[j]
    T_exit = T_entry - hold_days / 252
    sigma_exit = vol.iloc[j]
    exit_val = bs_price(S_exit, K, T_exit, sigma_exit, opt="call")
    pnl_pct = (exit_val - entry_cost) / entry_cost

    return Trade(
        entry_date=str(df.index[i].date()),
        exit_date=str(df.index[j].date()),
        entry_price=entry_cost, exit_price=exit_val,
        pnl=exit_val - entry_cost, pnl_pct=pnl_pct, won=pnl_pct > 0,
        vix_at_entry=vix,
    )


# ── Strategy Dispatch ────────────────────────────────────────────────────

def execute_strategy(strategy_name, df, i, vol, vix):
    """Dispatch to the right simulator based on strategy name."""
    name_lower = strategy_name.lower()

    if "bull put" in name_lower:
        return simulate_bull_put(df, i, vol, vix=vix), BULL_PUT_DTE
    elif "bear call" in name_lower:
        return simulate_bear_call(df, i, vol, vix=vix), BEAR_CALL_DTE
    elif "leaps" in name_lower:
        return simulate_leaps(df, i, vol, vix=vix), LEAPS_DTE_DAYS
    elif "no trade" in name_lower or "wait" in name_lower:
        return None, BULL_PUT_DTE  # skip, advance by default interval
    else:
        # Fallback: bull put
        return simulate_bull_put(df, i, vol, vix=vix), BULL_PUT_DTE


# ── Equity Curve with Adaptive Sizing ────────────────────────────────────

def build_equity_curve(trades, starting_capital=STARTING_CAPITAL):
    """Build equity curve with adaptive VIX-based sizing."""
    sizer = AdaptiveSizer(starting_capital)
    eq = [starting_capital]

    for t in trades:
        sizer.account_size = eq[-1]
        risked = sizer.position_size(t.vix_at_entry)
        trade_pnl = risked * t.pnl_pct
        eq.append(max(eq[-1] + trade_pnl, 0.01))
        sizer.record_result(t.won)

    return eq


# ── Main Backtest ────────────────────────────────────────────────────────

def run_comparison(ticker="SPY"):
    """Run bull-put-only vs ML-multi-strategy backtest for one ticker."""
    print("Fetching data for {}...".format(ticker))
    handler = DataHandler()
    engineer = FeatureEngineer()

    df = engineer.add_all(handler.fetch_ohlcv(ticker, period="4y"))
    df = engineer.add_vix(df)
    df = engineer.add_put_call_ratio(df)

    vol = _rolling_vol(df)
    vix_series = df["VIX"] if "VIX" in df.columns else pd.Series(20.0, index=df.index)

    print("  {} rows of data loaded.".format(len(df)))

    # ── A) Bull Put Only ─────────────────────────────────────
    print("  Running Bull Put Only backtest...")
    bull_put_trades = []
    i = MIN_TRAIN_ROWS  # start at same point as ML for fair comparison
    while i < len(df) - BULL_PUT_DTE:
        vix_now = float(vix_series.iloc[i])
        trade = simulate_bull_put(df, i, vol, vix=vix_now)
        if trade:
            bull_put_trades.append(trade)
        i += BULL_PUT_DTE + 1

    # ── B) ML Multi-Strategy ────────────────────────────────
    print("  Running ML Multi-Strategy backtest...")
    ml_trades = []
    strategy_counts = {}
    engine = RavishStrategyEngine()

    i = MIN_TRAIN_ROWS
    last_train = 0
    ensemble = None

    while i < len(df) - BULL_PUT_DTE:
        vix_now = float(vix_series.iloc[i])

        # Retrain ML model periodically
        if ensemble is None or (i - last_train) >= RETRAIN_EVERY:
            ensemble = EnsemblePredictor()
            train_slice = df.iloc[:i]
            ensemble.train(train_slice)
            last_train = i

        # Get ML prediction
        pred_slice = df.iloc[:i + 1]
        ml_label, ml_prob = ensemble.predict(pred_slice)

        # Select strategy
        signal = engine.select(
            ml_label=ml_label,
            ml_prob=ml_prob,
            vix=vix_now,
            near_earnings=False,
            ticker=ticker,
            account_size=STARTING_CAPITAL,
        )

        strat_name = signal.name
        strategy_counts[strat_name] = strategy_counts.get(strat_name, 0) + 1

        # Execute
        trade, hold_period = execute_strategy(strat_name, df, i, vol, vix_now)
        if trade:
            ml_trades.append(trade)

        i += hold_period + 1

    return bull_put_trades, ml_trades, strategy_counts


def print_comparison(ticker, bull_trades, ml_trades, strategy_counts):
    """Print side-by-side comparison."""
    bull_eq = build_equity_curve(bull_trades)
    ml_eq = build_equity_curve(ml_trades)

    def stats(trades, eq):
        if not trades:
            return {"trades": 0, "win_rate": 0, "avg_win": 0, "avg_loss": 0,
                    "final": STARTING_CAPITAL, "ret": 0, "max_dd": 0, "pf": 0}
        wins = [t for t in trades if t.won]
        losses = [t for t in trades if not t.won]
        eq_arr = np.array(eq)
        peak = np.maximum.accumulate(eq_arr)
        dd = ((eq_arr - peak) / peak).min()
        gross_w = sum(t.pnl_pct for t in wins)
        gross_l = sum(abs(t.pnl_pct) for t in losses)
        return {
            "trades": len(trades),
            "win_rate": len(wins) / len(trades),
            "avg_win": np.mean([t.pnl_pct for t in wins]) if wins else 0,
            "avg_loss": np.mean([abs(t.pnl_pct) for t in losses]) if losses else 0,
            "final": eq[-1],
            "ret": (eq[-1] / eq[0] - 1) * 100,
            "max_dd": dd * 100,
            "pf": round(gross_w / gross_l, 2) if gross_l > 0 else float("inf"),
        }

    b = stats(bull_trades, bull_eq)
    m = stats(ml_trades, ml_eq)

    width = 72
    print("\n" + "=" * width)
    print("{:^{w}}".format("BACKTEST COMPARISON: " + ticker, w=width))
    print("{:^{w}}".format(
        "${:,.0f} starting capital | Adaptive VIX sizing".format(STARTING_CAPITAL), w=width))
    print("=" * width)

    print("\n  {:<30} {:>18} {:>18}".format("Metric", "Bull Put Only", "ML Multi-Strat"))
    print("  " + "-" * 68)
    print("  {:<30} {:>18} {:>18}".format("Total Trades",
          str(b["trades"]), str(m["trades"])))
    print("  {:<30} {:>17.1%} {:>17.1%}".format("Win Rate",
          b["win_rate"], m["win_rate"]))
    print("  {:<30} {:>17.2%} {:>17.2%}".format("Avg Win",
          b["avg_win"], m["avg_win"]))
    print("  {:<30} {:>17.2%} {:>17.2%}".format("Avg Loss",
          b["avg_loss"], m["avg_loss"]))
    print("  {:<30} {:>18.2f} {:>18.2f}".format("Profit Factor",
          b["pf"], m["pf"]))
    print("  {:<30} {:>17.1f}% {:>17.1f}%".format("Total Return",
          b["ret"], m["ret"]))
    print("  {:<30} {:>16s} {:>16s}".format("Final Value",
          "${:,.0f}".format(b["final"]), "${:,.0f}".format(m["final"])))
    print("  {:<30} {:>17.1f}% {:>17.1f}%".format("Max Drawdown",
          b["max_dd"], m["max_dd"]))

    # Strategy distribution
    print("\n  ML Strategy Selection Distribution:")
    total_decisions = sum(strategy_counts.values())
    for name, count in sorted(strategy_counts.items(), key=lambda x: -x[1]):
        pct = count / total_decisions * 100
        bar = "#" * int(pct / 2)
        print("    {:<30} {:>4} ({:>5.1f}%) {}".format(name, count, pct, bar))

    # Winner
    print()
    if m["final"] > b["final"]:
        diff = m["final"] - b["final"]
        print("  >>> ML Multi-Strategy WINS by ${:,.0f} ({:+.1f}% vs {:+.1f}%)".format(
            diff, m["ret"], b["ret"]))
    elif b["final"] > m["final"]:
        diff = b["final"] - m["final"]
        print("  >>> Bull Put Only WINS by ${:,.0f} ({:+.1f}% vs {:+.1f}%)".format(
            diff, b["ret"], m["ret"]))
    else:
        print("  >>> TIE")

    # Risk-adjusted
    bull_risk_adj = b["ret"] / abs(b["max_dd"]) if b["max_dd"] != 0 else 0
    ml_risk_adj = m["ret"] / abs(m["max_dd"]) if m["max_dd"] != 0 else 0
    print("  Risk-adjusted (Return/MaxDD): Bull Put {:.2f} | ML {:.2f}".format(
        bull_risk_adj, ml_risk_adj))

    print("=" * width)

    # Trade-by-trade for ML strategy (last 20)
    if ml_trades:
        print("\n  ML Strategy Recent Trades (last 20):")
        print("  {:<12} {:<12} {:>8} {:>8}".format("Entry", "Exit", "P&L%", "Result"))
        for t in ml_trades[-20:]:
            print("  {:<12} {:<12} {:>+7.2%} {:>8}".format(
                t.entry_date, t.exit_date, t.pnl_pct, "WIN" if t.won else "LOSS"))


def main():
    print("=" * 72)
    print("{:^72}".format("ML-DRIVEN STRATEGY vs BULL PUT ONLY"))
    print("{:^72}".format("4-Year Backtest with Adaptive VIX Sizing"))
    print("=" * 72)

    for ticker in ["SPY", "QQQ"]:
        bull_trades, ml_trades, strategy_counts = run_comparison(ticker)
        print_comparison(ticker, bull_trades, ml_trades, strategy_counts)

    print("\nDone.")


if __name__ == "__main__":
    main()
