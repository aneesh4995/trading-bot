"""
Ravish Options Strategy Engine
Extracted from 13 videos on @OptionsWithRavish

Core Strategies:
  1. Bull Put Credit Spread  — 90% win rate, SPX/QQQ, 14 DTE, 50-delta sell
  2. LEAPS Swing             — 91-96% win rate, QQQ, 60-80 delta, 12m+ expiry, 3-4m hold
  3. Diagonal Spread         — buy 70d deep ITM + sell 30d monthly OTM
  4. Covered Call            — 30 delta monthly, roll up/out if challenged
  5. Cash Secured Put        — 30-50 delta, 4+ weeks, buy stocks at discount
  6. Earnings Short Put      — 20 delta put below expected move, open 3:30 close next open
  7. ZEBRA                   — 2x ATM call + short 1x deep ITM = 100 delta, ~zero theta

Rules (universal across all strategies):
  - Take profit at 50% for credit trades, 50% for LEAPS/debit
  - No hard stop-loss — roll/adjust instead
  - Never fight trend; only trade bullish setups on QQQ/SPX
  - Only trade when VIX is manageable (VIX > 30 = reduce size)
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class StrategySignal:
    name: str
    ticker: str
    bias: str                      # "bullish" | "bearish" | "neutral"
    action: str                    # full trade description
    sell_strike_delta: float
    sell_dte: int
    buy_strike_offset: Optional[float]  # points away for spread leg; None if single-leg
    profit_target_pct: float       # % of max credit/debit to close at
    stop_loss: str                 # "none — roll instead" or specific rule
    min_credit_or_debit: str       # ballpark cost/credit
    rationale: str
    small_account_friendly: bool
    set_and_forget: bool


class RavishStrategyEngine:
    """
    Maps ML prediction + market conditions → Ravish's specific options strategy.

    Decision tree (mirrors Ravish's stated preferences):
      VIX > 30  → reduce to LEAPS only or skip (high vol = expensive spreads)
      Earnings  → Earnings Short Put (if within 1-2 days of event)
      ML Up + high prob → Bull Put Credit Spread (primary income strategy)
      ML Down   → Bear Call Spread (mirror of bull put)
      Any       → LEAPS for longer-horizon conviction plays on QQQ
      Stock held→ Covered Call overlay
    """

    STRATEGIES = {
        "bull_put_credit_spread": StrategySignal(
            name="Bull Put Credit Spread",
            ticker="SPX / QQQ / Large Caps",
            bias="bullish",
            action=(
                "SELL 1x ATM put (50 delta), BUY 1x put 10 pts lower. "
                "Same expiry, 14 DTE."
            ),
            sell_strike_delta=0.50,
            sell_dte=14,
            buy_strike_offset=-10,
            profit_target_pct=50.0,
            stop_loss="none — roll down and out to next monthly if threatened",
            min_credit_or_debit="~$125–$200 credit per spread (SPX)",
            rationale=(
                "Ravish's bread-and-butter. 90% win rate, $84K profit over 5y backtest. "
                "Sell ATM put so you collect max credit, hedge 10pts below to cap risk. "
                "Works even in flat or slightly down markets."
            ),
            small_account_friendly=True,
            set_and_forget=True,
        ),
        "bear_call_credit_spread": StrategySignal(
            name="Bear Call Credit Spread",
            ticker="SPX / QQQ",
            bias="bearish",
            action=(
                "SELL 1x ATM call (50 delta), BUY 1x call 10 pts higher. "
                "Same expiry, 14 DTE."
            ),
            sell_strike_delta=0.50,
            sell_dte=14,
            buy_strike_offset=+10,
            profit_target_pct=50.0,
            stop_loss="none — roll up and out if challenged",
            min_credit_or_debit="~$125–$200 credit per spread",
            rationale="Mirror of bull put spread for bearish ML signal.",
            small_account_friendly=True,
            set_and_forget=True,
        ),
        "leaps": StrategySignal(
            name="LEAPS Swing (QQQ)",
            ticker="QQQ",
            bias="bullish",
            action=(
                "BUY 1x deep ITM call, 60-80 delta, 12+ months expiry. "
                "Hold 3-4 months. Take profit at 50%. "
                "Pick round-number strike for liquidity."
            ),
            sell_strike_delta=0.70,
            sell_dte=365,
            buy_strike_offset=None,
            profit_target_pct=50.0,
            stop_loss="none — QQQ historically up 5%+ in 85% of quarters",
            min_credit_or_debit="~$3,000–$6,000 debit per contract",
            rationale=(
                "91-96% win rate over 5y. QQQ moves up 5%+ in 85% of quarters. "
                "LEAPS gives 100-share exposure for fraction of cost vs buying stock. "
                "Slow theta decay allows time to be right."
            ),
            small_account_friendly=False,
            set_and_forget=True,
        ),
        "diagonal_spread": StrategySignal(
            name="Diagonal Spread (Poor Man's Covered Call)",
            ticker="Any liquid stock / ETF",
            bias="bullish",
            action=(
                "BUY 1x deep ITM call (70 delta, 6+ months expiry). "
                "SELL 1x OTM call (30 delta, current month). "
                "Roll short call monthly. Take profit 30-40% on short leg."
            ),
            sell_strike_delta=0.30,
            sell_dte=30,
            buy_strike_offset=None,
            profit_target_pct=40.0,
            stop_loss="exit if stock hits long call strike (deep ITM leg)",
            min_credit_or_debit="~$2,300 net debit (vs $10,000+ for covered call)",
            rationale=(
                "Ravish's preferred income strategy. Replaces covered call at 20% of cost. "
                "Short 30-delta monthly call decays fast. Roll monthly for ongoing income."
            ),
            small_account_friendly=True,
            set_and_forget=False,
        ),
        "cash_secured_put": StrategySignal(
            name="Cash Secured Put",
            ticker="Any stock you want to own",
            bias="bullish",
            action=(
                "SELL 1x put at 30-50 delta, 4+ weeks expiry. "
                "Take profit at 50-80%. If assigned, own stock at effective discount."
            ),
            sell_strike_delta=0.40,
            sell_dte=30,
            buy_strike_offset=None,
            profit_target_pct=65.0,
            stop_loss="none — happy to own stock at discount if assigned",
            min_credit_or_debit="varies — ~5-15% downside protection baked in",
            rationale=(
                "Buy stocks 5-30% below market by selling puts. "
                "If stock stays flat/up: keep premium. If assigned: own at a discount."
            ),
            small_account_friendly=False,
            set_and_forget=True,
        ),
        "earnings_short_put": StrategySignal(
            name="Earnings Short Put (< 2 DTE)",
            ticker="High-IV earnings stock (e.g. NVDA)",
            bias="neutral-bullish",
            action=(
                "SELL 1x put at 20 delta, just BELOW the expected move lower band. "
                "Open at 3:30 PM day before earnings. "
                "Close in first 30 min of next-day open to capture IV crush."
            ),
            sell_strike_delta=0.20,
            sell_dte=2,
            buy_strike_offset=-10,
            profit_target_pct=80.0,
            stop_loss="none — set and forget until next morning open",
            min_credit_or_debit="~$150–$300 credit; scale with put spread for small accounts",
            rationale=(
                "Ravish's earnings strategy: 80-100% win rate on NVDA over 3y. "
                "Uses expected move data (not charts). IV crush next morning = fast profit. "
                "Sell below lower expected move band for downside cushion."
            ),
            small_account_friendly=True,
            set_and_forget=True,
        ),
        "zebra": StrategySignal(
            name="ZEBRA (Zero Extrinsic Back Ratio)",
            ticker="SPY / QQQ / any liquid",
            bias="bullish",
            action=(
                "BUY 2x ATM calls (50 delta each = 100 delta total). "
                "SELL 1x deep ITM call (same expiry). "
                "Net extrinsic value ≈ 0 → near-zero theta decay."
            ),
            sell_strike_delta=0.80,
            sell_dte=30,
            buy_strike_offset=None,
            profit_target_pct=50.0,
            stop_loss="exit if stock falls below long call strike",
            min_credit_or_debit="small net debit or near zero cost",
            rationale=(
                "Better than LEAPS: 100 delta (full stock exposure) with ~zero theta. "
                "Eliminates IV crush risk. Useful for short-term directional swings. "
                "Works on 0DTE, weekly, or monthly timeframes."
            ),
            small_account_friendly=True,
            set_and_forget=False,
        ),
        "no_trade": StrategySignal(
            name="No Trade / Wait",
            ticker="—",
            bias="neutral",
            action="Stay cash. Wait for VIX to normalize or ML conviction to rise.",
            sell_strike_delta=0.0,
            sell_dte=0,
            buy_strike_offset=None,
            profit_target_pct=0.0,
            stop_loss="n/a",
            min_credit_or_debit="$0",
            rationale="VIX too high or ML conviction below threshold.",
            small_account_friendly=True,
            set_and_forget=True,
        ),
    }

    VIX_HIGH_THRESHOLD = 30.0
    CONVICTION_THRESHOLD = 0.65

    def select(
        self,
        ml_label: int,
        ml_prob: float,
        vix: float,
        near_earnings: bool = False,
        ticker: str = "SPY",
        account_size: float = 50_000,
    ) -> StrategySignal:
        # Low conviction → no trade
        if ml_prob < self.CONVICTION_THRESHOLD:
            return self.STRATEGIES["no_trade"]

        # Extremely high VIX → LEAPS only (or cash) — spreads get too expensive
        if vix > self.VIX_HIGH_THRESHOLD:
            if ml_label == 1 and ticker == "QQQ":
                return self.STRATEGIES["leaps"]
            return self.STRATEGIES["no_trade"]

        # Earnings window → earnings short put
        if near_earnings:
            return self.STRATEGIES["earnings_short_put"]

        # Normal market + bullish ML
        if ml_label == 1:
            # Small accounts → bull put spread (min $500 to trade)
            # Larger accounts → add diagonal for income
            if account_size < 25_000:
                return self.STRATEGIES["bull_put_credit_spread"]
            elif ticker == "QQQ":
                # Ravish heavily favours LEAPS on QQQ specifically
                return self.STRATEGIES["leaps"] if ml_prob > 0.80 else self.STRATEGIES["bull_put_credit_spread"]
            else:
                return self.STRATEGIES["bull_put_credit_spread"]

        # Bearish ML
        return self.STRATEGIES["bear_call_credit_spread"]

    def print_ravish_card(self, signal: StrategySignal, vix: float, ml_prob: float) -> None:
        width = 60
        sep = "─" * width
        print(f"\n┌{sep}┐")
        print(f"│{'  RAVISH STRATEGY SIGNAL':^{width}}│")
        print(f"├{sep}┤")
        print(f"│  Strategy  : {signal.name:<{width - 15}}│")
        print(f"│  Ticker    : {signal.ticker:<{width - 15}}│")
        print(f"│  Bias      : {signal.bias:<{width - 15}}│")
        print(f"├{sep}┤")
        print(f"│  ENTRY RULES{'':>{width - 13}}│")
        # wrap action across lines
        action_words = signal.action.split(". ")
        for line in action_words:
            if line.strip():
                print(f"│    • {line.strip()[:width - 7]:<{width - 7}}│")
        print(f"├{sep}┤")
        print(f"│  Sell Delta: {signal.sell_strike_delta:<{width - 15}.0%}│")
        print(f"│  DTE       : {signal.sell_dte:<{width - 15}}│")
        if signal.buy_strike_offset is not None:
            offset = f"{signal.buy_strike_offset:+.0f} pts"
            print(f"│  Hedge Leg : {offset:<{width - 15}}│")
        print(f"├{sep}┤")
        print(f"│  EXIT RULES{'':>{width - 13}}│")
        print(f"│  Profit at : {signal.profit_target_pct:.0f}% of max{'':>{width - 26}}│")
        print(f"│  Stop Loss : {signal.stop_loss[:width - 15]:<{width - 15}}│")
        print(f"├{sep}┤")
        print(f"│  Cost/Credit: {signal.min_credit_or_debit[:width - 16]:<{width - 16}}│")
        print(f"│  Small Acct : {'Yes' if signal.small_account_friendly else 'No':<{width - 16}}│")
        print(f"│  Set&Forget : {'Yes' if signal.set_and_forget else 'No (roll monthly)':<{width - 16}}│")
        print(f"├{sep}┤")
        # wrap rationale
        rationale = signal.rationale
        while rationale:
            chunk, rationale = rationale[:width - 4], rationale[width - 4:]
            print(f"│  {chunk:<{width - 2}}│")
        print(f"└{sep}┘")
