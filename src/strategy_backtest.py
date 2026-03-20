"""
Backtest all Ravish strategies over 3 years using Black-Scholes option pricing.
Starting capital: $1,000. Tickers: SPY & QQQ.

Strategies tested:
  1. Bull Put Credit Spread  — 14 DTE, sell 50-delta put, buy 10pts lower, TP 50%
  2. LEAPS Swing             — QQQ, buy 70-delta call 12m out, hold up to 90d, TP 50%
  3. Diagonal Spread         — buy 70d 6m call + sell 30d monthly call, roll monthly
  4. Cash Secured Put        — sell 40-delta put, 30 DTE, TP 50%
  5. ZEBRA                   — 100-delta stock replacement, limited downside
  6. Earnings Short Put      — sell 20-delta put 2 DTE, close next day (simulated)
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from dataclasses import dataclass, field
from data_handler import DataHandler
from indicators import FeatureEngineer


# ── Black-Scholes ────────────────────────────────────────────

RISK_FREE = 0.045  # approximate average 3y T-bill rate


def bs_price(S, K, T, sigma, r=RISK_FREE, opt="put"):
    if T <= 0:
        if opt == "put":
            return max(K - S, 0.0)
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if opt == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_delta(S, K, T, sigma, r=RISK_FREE, opt="put"):
    if T <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if opt == "call":
        return norm.cdf(d1)
    return norm.cdf(d1) - 1.0


def find_strike_for_delta(S, T, sigma, target_delta, opt="put", r=RISK_FREE):
    """Binary search for strike that gives target delta.
    For calls: higher strike → lower delta. For puts: higher strike → more negative delta."""
    lo, hi = S * 0.3, S * 1.7
    for _ in range(100):
        mid = (lo + hi) / 2
        d = bs_delta(S, mid, T, sigma, r, opt)
        if opt == "call":
            # call delta decreases as strike increases
            if d > target_delta:
                lo = mid
            else:
                hi = mid
        else:
            # put delta (negative): becomes more negative as strike increases
            if d < target_delta:
                hi = mid
            else:
                lo = mid
    return round((lo + hi) / 2, 2)


# ── Trade record ─────────────────────────────────────────────

@dataclass
class Trade:
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float
    won: bool
    vix_at_entry: float = 20.0


@dataclass
class StrategyResult:
    name: str
    ticker: str
    trades: list = field(default_factory=list)
    starting_capital: float = 1000.0

    @property
    def total_trades(self):
        return len(self.trades)

    @property
    def wins(self):
        return sum(1 for t in self.trades if t.won)

    @property
    def win_rate(self):
        return self.wins / self.total_trades if self.total_trades else 0.0

    @property
    def avg_win_pct(self):
        w = [t.pnl_pct for t in self.trades if t.won]
        return np.mean(w) if w else 0.0

    @property
    def avg_loss_pct(self):
        l = [abs(t.pnl_pct) for t in self.trades if not t.won]
        return np.mean(l) if l else 0.0

    @property
    def equity_curve(self):
        """Fixed fractional: risk 20% of equity per trade."""
        alloc_frac = 0.20
        eq = [self.starting_capital]
        for t in self.trades:
            risked = eq[-1] * alloc_frac
            trade_pnl = risked * t.pnl_pct
            eq.append(max(eq[-1] + trade_pnl, 0.01))
        return eq

    @property
    def adaptive_equity_curve(self):
        """Adaptive sizing: VIX regime + loss streak adjustment."""
        from risk_manager import AdaptiveSizer
        sizer = AdaptiveSizer(self.starting_capital)
        eq = [self.starting_capital]
        for t in self.trades:
            sizer.account_size = eq[-1]
            risked = sizer.position_size(t.vix_at_entry)
            trade_pnl = risked * t.pnl_pct
            eq.append(max(eq[-1] + trade_pnl, 0.01))
            sizer.record_result(t.won)
        return eq

    @property
    def adaptive_final_value(self):
        return self.adaptive_equity_curve[-1]

    @property
    def adaptive_return_pct(self):
        return (self.adaptive_final_value / self.starting_capital - 1) * 100

    @property
    def adaptive_max_drawdown_pct(self):
        eq = np.array(self.adaptive_equity_curve)
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak
        return float(dd.min()) * 100

    @property
    def final_value(self):
        return self.equity_curve[-1]

    @property
    def total_return_pct(self):
        return (self.final_value / self.starting_capital - 1) * 100

    @property
    def total_pnl(self):
        return self.final_value - self.starting_capital

    @property
    def max_drawdown_pct(self):
        eq = np.array(self.equity_curve)
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak
        return float(dd.min()) * 100

    @property
    def profit_factor(self):
        gross_w = sum(t.pnl_pct for t in self.trades if t.won)
        gross_l = sum(abs(t.pnl_pct) for t in self.trades if not t.won)
        return round(gross_w / gross_l, 2) if gross_l > 0 else float("inf")


# ── Strategy Backtests ───────────────────────────────────────

def _rolling_vol(df, window=30):
    """Annualized historical vol from daily close returns."""
    ret = df["close"].pct_change()
    vol = ret.rolling(window).std() * np.sqrt(252)
    return vol.bfill().clip(lower=0.10)


def backtest_bull_put_spread(df, ticker="SPY", width=10, dte=14, tp_pct=0.50, vix_series=None):
    """Sell ATM put + buy put `width` pts lower, 14 DTE, take profit at 50%."""
    vol = _rolling_vol(df)
    trades = []
    i = 0
    while i < len(df) - dte:
        S = df["close"].iloc[i]
        sigma = vol.iloc[i]
        T = dte / 252
        vix_now = float(vix_series.iloc[i]) if vix_series is not None else 20.0

        short_K = round(S)
        long_K = short_K - width

        credit = bs_price(S, short_K, T, sigma) - bs_price(S, long_K, T, sigma)
        max_loss_per_dollar = width - credit
        if credit <= 0 or max_loss_per_dollar <= 0:
            i += dte
            continue

        # Check daily for 50% profit or hold to expiry
        exit_day = i + dte
        closed_early = False
        for j in range(i + 1, min(i + dte + 1, len(df))):
            S_j = df["close"].iloc[j]
            T_j = max((dte - (j - i)), 0) / 252
            sigma_j = vol.iloc[j]
            spread_val = bs_price(S_j, short_K, T_j, sigma_j) - bs_price(S_j, long_K, T_j, sigma_j)
            current_profit = credit - spread_val

            if current_profit >= credit * tp_pct:
                pnl_pct = current_profit / max_loss_per_dollar
                trades.append(Trade(
                    entry_date=str(df.index[i].date()),
                    exit_date=str(df.index[j].date()),
                    entry_price=credit,
                    exit_price=spread_val,
                    pnl=current_profit,
                    pnl_pct=pnl_pct,
                    won=True,
                    vix_at_entry=vix_now,
                ))
                exit_day = j
                closed_early = True
                break

        if not closed_early and exit_day < len(df):
            S_exp = df["close"].iloc[exit_day]
            intrinsic_short = max(short_K - S_exp, 0)
            intrinsic_long = max(long_K - S_exp, 0)
            spread_at_exp = intrinsic_short - intrinsic_long
            pnl = credit - spread_at_exp
            pnl_pct = pnl / max_loss_per_dollar
            trades.append(Trade(
                entry_date=str(df.index[i].date()),
                exit_date=str(df.index[min(exit_day, len(df)-1)].date()),
                entry_price=credit,
                exit_price=spread_at_exp,
                pnl=pnl,
                pnl_pct=pnl_pct,
                won=pnl > 0,
                vix_at_entry=vix_now,
            ))

        i = exit_day + 1

    return StrategyResult(name="Bull Put Credit Spread", ticker=ticker, trades=trades)


def backtest_leaps(df, ticker="QQQ", target_delta=0.70, hold_days=63, tp_pct=0.50):
    """Buy deep ITM call (70 delta, 12m expiry), hold up to 90d, TP at 50%."""
    vol = _rolling_vol(df)
    trades = []
    i = 0
    while i < len(df) - hold_days:
        S = df["close"].iloc[i]
        sigma = vol.iloc[i]
        T_entry = 1.0  # 12 months

        K = find_strike_for_delta(S, T_entry, sigma, target_delta, opt="call")
        entry_cost = bs_price(S, K, T_entry, sigma, opt="call")
        if entry_cost <= 0:
            i += hold_days
            continue

        exit_day = i + hold_days
        closed_early = False
        for j in range(i + 1, min(i + hold_days + 1, len(df))):
            S_j = df["close"].iloc[j]
            T_j = T_entry - (j - i) / 252
            sigma_j = vol.iloc[j]
            current_val = bs_price(S_j, K, T_j, sigma_j, opt="call")
            profit_pct = (current_val - entry_cost) / entry_cost

            if profit_pct >= tp_pct:
                trades.append(Trade(
                    entry_date=str(df.index[i].date()),
                    exit_date=str(df.index[j].date()),
                    entry_price=entry_cost,
                    exit_price=current_val,
                    pnl=current_val - entry_cost,
                    pnl_pct=profit_pct,
                    won=True,
                ))
                exit_day = j
                closed_early = True
                break

        if not closed_early and exit_day < len(df):
            S_exit = df["close"].iloc[exit_day]
            T_exit = T_entry - hold_days / 252
            sigma_exit = vol.iloc[exit_day]
            exit_val = bs_price(S_exit, K, T_exit, sigma_exit, opt="call")
            pnl_pct = (exit_val - entry_cost) / entry_cost
            trades.append(Trade(
                entry_date=str(df.index[i].date()),
                exit_date=str(df.index[min(exit_day, len(df)-1)].date()),
                entry_price=entry_cost,
                exit_price=exit_val,
                pnl=exit_val - entry_cost,
                pnl_pct=pnl_pct,
                won=pnl_pct > 0,
            ))

        i = exit_day + 1

    return StrategyResult(name="LEAPS Swing", ticker=ticker, trades=trades)


def backtest_diagonal(df, ticker="QQQ"):
    """Buy 70-delta 6m call, sell 30-delta 30d call. Roll monthly."""
    vol = _rolling_vol(df)
    trades = []
    i = 0
    while i < len(df) - 30:
        S = df["close"].iloc[i]
        sigma = vol.iloc[i]

        # Long leg: 6m, 70 delta
        T_long = 0.5
        K_long = find_strike_for_delta(S, T_long, sigma, 0.70, opt="call")
        long_cost = bs_price(S, K_long, T_long, sigma, opt="call")

        # Short leg: 30d, 30 delta
        T_short = 30 / 252
        K_short = find_strike_for_delta(S, T_short, sigma, 0.30, opt="call")
        short_credit = bs_price(S, K_short, T_short, sigma, opt="call")

        net_debit = long_cost - short_credit
        if net_debit <= 0:
            i += 30
            continue

        # After 30 days, close both legs
        j = min(i + 21, len(df) - 1)  # ~21 trading days = 1 month
        S_j = df["close"].iloc[j]
        sigma_j = vol.iloc[j]
        T_long_rem = T_long - 21 / 252
        long_val = bs_price(S_j, K_long, T_long_rem, sigma_j, opt="call")

        short_intrinsic = max(S_j - K_short, 0)
        short_val_at_exp = short_intrinsic

        net_value = long_val - short_val_at_exp
        pnl = net_value - net_debit
        pnl_pct = pnl / net_debit

        trades.append(Trade(
            entry_date=str(df.index[i].date()),
            exit_date=str(df.index[j].date()),
            entry_price=net_debit,
            exit_price=net_value,
            pnl=pnl,
            pnl_pct=pnl_pct,
            won=pnl > 0,
        ))
        i = j + 1

    return StrategyResult(name="Diagonal Spread", ticker=ticker, trades=trades)


def backtest_cash_secured_put(df, ticker="SPY", target_delta=-0.35, dte=21, tp_pct=0.50):
    """Sell ~35-delta put, 30 DTE, take profit at 50%."""
    vol = _rolling_vol(df)
    trades = []
    i = 0
    while i < len(df) - dte:
        S = df["close"].iloc[i]
        sigma = vol.iloc[i]
        T = dte / 252

        K = find_strike_for_delta(S, T, sigma, target_delta, opt="put")
        credit = bs_price(S, K, T, sigma, opt="put")
        if credit <= 0:
            i += dte
            continue

        # Risk = strike price (cash secured), return = credit / K
        risk_base = K  # cash needed to secure put

        exit_day = i + dte
        closed_early = False
        for j in range(i + 1, min(i + dte + 1, len(df))):
            S_j = df["close"].iloc[j]
            T_j = max((dte - (j - i)), 0) / 252
            sigma_j = vol.iloc[j]
            put_val = bs_price(S_j, K, T_j, sigma_j, opt="put")
            profit = credit - put_val

            if profit >= credit * tp_pct:
                pnl_pct = profit / risk_base
                trades.append(Trade(
                    entry_date=str(df.index[i].date()),
                    exit_date=str(df.index[j].date()),
                    entry_price=credit,
                    exit_price=put_val,
                    pnl=profit,
                    pnl_pct=pnl_pct,
                    won=True,
                ))
                exit_day = j
                closed_early = True
                break

        if not closed_early and exit_day < len(df):
            S_exp = df["close"].iloc[exit_day]
            intrinsic = max(K - S_exp, 0)
            pnl = credit - intrinsic
            pnl_pct = pnl / risk_base
            trades.append(Trade(
                entry_date=str(df.index[i].date()),
                exit_date=str(df.index[min(exit_day, len(df)-1)].date()),
                entry_price=credit,
                exit_price=intrinsic,
                pnl=pnl,
                pnl_pct=pnl_pct,
                won=pnl > 0,
            ))
        i = exit_day + 1

    return StrategyResult(name="Cash Secured Put", ticker=ticker, trades=trades)


def backtest_zebra(df, ticker="SPY"):
    """ZEBRA = 100 delta stock replacement. Simplified: track 1:1 with stock for 30d swings."""
    vol = _rolling_vol(df)
    trades = []
    hold = 21  # monthly
    i = 0
    while i < len(df) - hold:
        S_entry = df["close"].iloc[i]
        sigma = vol.iloc[i]
        T = 30 / 252

        # ZEBRA cost ≈ intrinsic value of deep ITM call (small extrinsic)
        K_deep = find_strike_for_delta(S_entry, T, sigma, 0.80, opt="call")
        # 2x ATM calls
        K_atm = round(S_entry)
        cost_2_atm = 2 * bs_price(S_entry, K_atm, T, sigma, opt="call")
        # sell 1x deep ITM call
        credit_deep = bs_price(S_entry, K_deep, T, sigma, opt="call")
        net_debit = cost_2_atm - credit_deep
        if net_debit <= 0:
            i += hold
            continue

        j = min(i + hold, len(df) - 1)
        S_exit = df["close"].iloc[j]

        # At expiry, ZEBRA payoff = 2*max(S-K_atm,0) - max(S-K_deep,0)
        payoff = 2 * max(S_exit - K_atm, 0) - max(S_exit - K_deep, 0)
        pnl = payoff - net_debit
        pnl_pct = pnl / net_debit if net_debit > 0 else 0

        trades.append(Trade(
            entry_date=str(df.index[i].date()),
            exit_date=str(df.index[j].date()),
            entry_price=net_debit,
            exit_price=payoff,
            pnl=pnl,
            pnl_pct=pnl_pct,
            won=pnl > 0,
        ))
        i = j + 1

    return StrategyResult(name="ZEBRA", ticker=ticker, trades=trades)


def backtest_earnings_put(df, ticker="SPY", dte=2, tp_pct=0.80):
    """Simulate quarterly earnings short put. Sell 20-delta put 2 DTE, close next day.
    Approximate: enter every ~63 trading days (quarterly)."""
    vol = _rolling_vol(df)
    trades = []
    i = 0
    while i < len(df) - dte:
        S = df["close"].iloc[i]
        sigma = vol.iloc[i]
        # Earnings IV bump: multiply vol by 1.5 at entry (IV inflated before earnings)
        sigma_entry = sigma * 1.5
        T = dte / 252

        K = find_strike_for_delta(S, T, sigma_entry, -0.20, opt="put")
        credit = bs_price(S, K, T, sigma_entry, opt="put")
        if credit <= 0:
            i += 63
            continue

        # Next day: IV crush (vol drops by ~40%), stock moves ±3%
        j = min(i + 1, len(df) - 1)
        S_j = df["close"].iloc[j]
        sigma_post = sigma * 0.9  # IV crush
        T_j = max(dte - 1, 0) / 252
        put_val = bs_price(S_j, K, T_j, sigma_post, opt="put")

        profit = credit - put_val
        # Risk for spread version = $1000 (10-wide spread)
        risk_base = max(credit, 0.01)
        pnl_pct = profit / risk_base

        # Cap pnl_pct to reasonable bounds
        pnl_pct = max(min(pnl_pct, 1.0), -1.0)

        trades.append(Trade(
            entry_date=str(df.index[i].date()),
            exit_date=str(df.index[j].date()),
            entry_price=credit,
            exit_price=put_val,
            pnl=profit,
            pnl_pct=pnl_pct,
            won=profit > 0,
        ))
        i += 63  # next quarter

    return StrategyResult(name="Earnings Short Put", ticker=ticker, trades=trades)


# ── Main ─────────────────────────────────────────────────────

def print_summary_table(results: list):
    print("\n" + "=" * 90)
    print(f"{'RAVISH STRATEGY BACKTEST — 3 YEARS — $1,000 STARTING CAPITAL':^90}")
    print("=" * 90)
    print(
        f"{'Strategy':<25} {'Ticker':<6} {'Trades':>6} {'Win%':>7} "
        f"{'AvgWin':>8} {'AvgLoss':>8} {'PF':>6} "
        f"{'Return':>9} {'Final$':>9} {'MaxDD':>8}"
    )
    print("-" * 90)
    for r in results:
        print(
            f"{r.name:<25} {r.ticker:<6} {r.total_trades:>6} "
            f"{r.win_rate:>6.1%} {r.avg_win_pct:>7.2%} {r.avg_loss_pct:>7.2%} "
            f"{r.profit_factor:>6.2f} {r.total_return_pct:>8.1f}% "
            f"${r.final_value:>8,.0f} {r.max_drawdown_pct:>7.1f}%"
        )
    print("-" * 90)

    best = max(results, key=lambda r: r.final_value)
    worst = min(results, key=lambda r: r.final_value)
    print(f"\n  Best performer  : {best.name} ({best.ticker}) → ${best.final_value:,.0f} ({best.total_return_pct:+.1f}%)")
    print(f"  Worst performer : {worst.name} ({worst.ticker}) → ${worst.final_value:,.0f} ({worst.total_return_pct:+.1f}%)")
    print()


def main():
    handler = DataHandler()
    engineer = FeatureEngineer()

    print("Fetching 3 years of data...")
    spy = engineer.add_all(handler.fetch_ohlcv("SPY", period="3y"))
    qqq = engineer.add_all(handler.fetch_ohlcv("QQQ", period="3y"))
    print(f"  SPY: {len(spy)} rows  |  QQQ: {len(qqq)} rows")

    print("\nRunning backtests...")
    results = [
        backtest_bull_put_spread(spy, "SPY"),
        backtest_bull_put_spread(qqq, "QQQ"),
        backtest_leaps(qqq, "QQQ"),
        backtest_diagonal(qqq, "QQQ"),
        backtest_cash_secured_put(spy, "SPY"),
        backtest_cash_secured_put(qqq, "QQQ"),
        backtest_zebra(spy, "SPY"),
        backtest_zebra(qqq, "QQQ"),
        backtest_earnings_put(spy, "SPY"),
        backtest_earnings_put(qqq, "QQQ"),
    ]

    print_summary_table(results)

    # Print individual trade details for top 3
    top3 = sorted(results, key=lambda r: r.total_return_pct, reverse=True)[:3]
    for r in top3:
        print(f"\n{'─'*70}")
        print(f"  TOP PERFORMER: {r.name} ({r.ticker})")
        print(f"  {r.total_trades} trades | Win rate {r.win_rate:.1%} | ${r.starting_capital:,.0f} → ${r.final_value:,.0f}")
        print(f"{'─'*70}")
        print(f"  {'Date':>12}  {'Exit':>12}  {'P&L%':>8}  {'Result':>6}")
        for t in r.trades[:20]:
            tag = "WIN" if t.won else "LOSS"
            print(f"  {t.entry_date:>12}  {t.exit_date:>12}  {t.pnl_pct:>+7.2%}  {tag:>6}")
        if r.total_trades > 20:
            print(f"  ... and {r.total_trades - 20} more trades")


if __name__ == "__main__":
    main()
