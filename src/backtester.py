import pandas as pd
import numpy as np
from predictor import SwingPredictor, FEATURES

CONVICTION_THRESHOLD = 0.65
HOLD_DAYS = 5
TRAIN_RATIO = 0.70


class BacktestResult:
    def __init__(self, trades: pd.DataFrame, equity: pd.Series):
        self.trades = trades
        self.equity = equity

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return float((self.trades["pnl_pct"] > 0).mean())

    @property
    def avg_win(self) -> float:
        wins = self.trades.loc[self.trades["pnl_pct"] > 0, "pnl_pct"]
        return float(wins.mean()) if len(wins) else 0.0

    @property
    def avg_loss(self) -> float:
        losses = self.trades.loc[self.trades["pnl_pct"] < 0, "pnl_pct"]
        return float(losses.abs().mean()) if len(losses) else 0.0

    @property
    def profit_factor(self) -> float:
        gross_win = self.trades.loc[self.trades["pnl_pct"] > 0, "pnl_pct"].sum()
        gross_loss = self.trades.loc[self.trades["pnl_pct"] < 0, "pnl_pct"].abs().sum()
        return round(gross_win / gross_loss, 3) if gross_loss > 0 else float("inf")

    @property
    def total_return(self) -> float:
        return float((self.equity.iloc[-1] / self.equity.iloc[0]) - 1) if len(self.equity) > 1 else 0.0

    @property
    def max_drawdown(self) -> float:
        roll_max = self.equity.cummax()
        drawdown = (self.equity - roll_max) / roll_max
        return float(drawdown.min())

    @property
    def sharpe_ratio(self) -> float:
        if self.total_trades < 2:
            return 0.0
        daily_returns = self.trades["pnl_pct"] / HOLD_DAYS
        if daily_returns.std() == 0:
            return 0.0
        annualized = daily_returns.mean() * 252 / (daily_returns.std() * np.sqrt(252))
        return round(float(annualized), 3)

    def summary(self, ticker: str) -> str:
        width = 52
        sep = "─" * width
        lines = [
            f"\n┌{sep}┐",
            f"│{'  BACKTEST RESULTS: ' + ticker:^{width}}│",
            f"│{'  (Train 70% | Test 30% | Hold ' + str(HOLD_DAYS) + 'd)':^{width}}│",
            f"├{sep}┤",
            f"│  Total Trades  : {self.total_trades:<{width - 19}}│",
            f"│  Win Rate      : {self.win_rate:.1%}{'':>{width - 24}}│",
            f"│  Avg Win       : {self.avg_win:.2%}{'':>{width - 25}}│",
            f"│  Avg Loss      : {self.avg_loss:.2%}{'':>{width - 25}}│",
            f"│  Profit Factor : {self.profit_factor:<{width - 19}.3f}│",
            f"├{sep}┤",
            f"│  Total Return  : {self.total_return:.2%}{'':>{width - 25}}│",
            f"│  Max Drawdown  : {self.max_drawdown:.2%}{'':>{width - 25}}│",
            f"│  Sharpe Ratio  : {self.sharpe_ratio:<{width - 19}.3f}│",
            f"└{sep}┘",
        ]
        return "\n".join(lines)


class BacktestEngine:
    def __init__(self, conviction_threshold: float = CONVICTION_THRESHOLD, hold_days: int = HOLD_DAYS):
        self.conviction_threshold = conviction_threshold
        self.hold_days = hold_days

    def run(self, df: pd.DataFrame) -> BacktestResult:
        df = df.copy().reset_index(drop=True)
        split = int(len(df) * TRAIN_RATIO)

        # Train on first 70%
        predictor = SwingPredictor()
        predictor.train(df.iloc[:split])

        test_df = df.iloc[split:].copy().reset_index(drop=True)

        records = []
        for i in range(len(test_df) - self.hold_days):
            row = test_df.iloc[[i]]
            if row[FEATURES].isna().any(axis=1).iloc[0]:
                continue

            label, prob = predictor.predict(row)
            if prob < self.conviction_threshold:
                continue

            entry = float(test_df.iloc[i]["close"])
            exit_price = float(test_df.iloc[i + self.hold_days]["close"])
            direction = "Up" if label == 1 else "Down"

            if direction == "Up":
                pnl_pct = (exit_price - entry) / entry
            else:
                pnl_pct = (entry - exit_price) / entry

            records.append({
                "entry_idx": i,
                "direction": direction,
                "conviction": prob,
                "entry": entry,
                "exit": exit_price,
                "pnl_pct": pnl_pct,
            })

        trades = pd.DataFrame(records)

        if trades.empty:
            return BacktestResult(trades, pd.Series([1.0]))

        # Build equity curve (compound each trade, position size = 100% for simplicity)
        equity = [1.0]
        for pnl in trades["pnl_pct"]:
            equity.append(equity[-1] * (1 + pnl))
        equity_series = pd.Series(equity)

        return BacktestResult(trades, equity_series)
