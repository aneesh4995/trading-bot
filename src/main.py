from data_handler import DataHandler
from indicators import FeatureEngineer
from predictor import SwingPredictor
from options_strategy import OptionsSignalMapper
from risk_manager import RiskManager
from backtester import BacktestEngine
from ravish_strategy import RavishStrategyEngine

TICKERS = ["SPY", "QQQ"]
ACCOUNT_SIZE = 50_000
RISK_TOLERANCE = "moderate"

RUN_BACKTEST = True


def print_signal_card(
    ticker: str,
    strategy: dict,
    probability: float,
    entry: float,
    stop: float,
    tp: float,
    max_pos: float,
    delta: float,
    vix: float,
) -> None:
    width = 52
    sep = "─" * width
    print(f"\n┌{sep}┐")
    print(f"│{'  SIGNAL CARD: ' + ticker:^{width}}│")
    print(f"├{sep}┤")
    print(f"│  Strategy  : {strategy['strategy']:<{width - 15}}│")
    print(f"│  Direction : {strategy['direction']:<{width - 15}}│")
    print(f"│  Conviction: {probability:.1%}{'':>{width - 22}}│")
    print(f"│  VIX       : {vix:.2f}{'':>{width - 21}}│")
    print(f"├{sep}┤")
    print(f"│  Entry     : ${entry:<{width - 16}.2f}│")
    print(f"│  Stop Loss : ${stop:<{width - 16}.2f}│")
    print(f"│  Take Profit: ${tp:<{width - 17}.2f}│")
    print(f"├{sep}┤")
    print(f"│  Max Position : ${max_pos:<,.0f}{'':>{width - 23}}│")
    print(f"│  Rec. Delta   : {delta:.2f}{'':>{width - 22}}│")
    print(f"├{sep}┤")
    print(f"│  Rationale: {strategy['rationale'][:width - 14]:<{width - 14}}│")
    print(f"└{sep}┘")


def main():
    handler = DataHandler()
    engineer = FeatureEngineer()
    mapper = OptionsSignalMapper()
    backtest_engine = BacktestEngine()
    ravish = RavishStrategyEngine()

    print("Fetching VIX...")
    vix = handler.fetch_vix()

    for ticker in TICKERS:
        print(f"\nProcessing {ticker}...")

        df_raw = handler.fetch_ohlcv(ticker, period="2y", interval="1d")
        df = engineer.add_all(df_raw)

        # --- Backtest first to derive realistic Kelly inputs ---
        bt_result = None
        if RUN_BACKTEST:
            print(f"  Running backtest for {ticker}...")
            bt_result = backtest_engine.run(df)

        # Use backtest stats for Kelly if available, else safe defaults
        if bt_result and bt_result.total_trades > 5:
            win_rate = bt_result.win_rate
            avg_win = bt_result.avg_win if bt_result.avg_win > 0 else 0.02
            avg_loss = bt_result.avg_loss if bt_result.avg_loss > 0 else 0.02
        else:
            win_rate, avg_win, avg_loss = 0.55, 0.03, 0.02

        risk = RiskManager(account_size=ACCOUNT_SIZE, risk_tolerance=RISK_TOLERANCE)

        # --- Live signal ---
        predictor = SwingPredictor()
        predictor.train(df)
        label, prob = predictor.predict(df)

        atr_series = df["ATR_14"]
        strategy = mapper.get_strategy(label, prob, atr_series)

        entry = float(df["close"].iloc[-1])
        atr = float(atr_series.iloc[-1])
        stop = risk.stop_loss(entry, atr)
        tp = risk.take_profit(entry, atr)
        max_pos = risk.max_position_size(win_rate, avg_win, avg_loss)
        delta = risk.recommended_delta()

        if bt_result:
            print(bt_result.summary(ticker))

        print_signal_card(ticker, strategy, prob, entry, stop, tp, max_pos, delta, vix)

        # --- Ravish strategy overlay ---
        ravish_signal = ravish.select(
            ml_label=label,
            ml_prob=prob,
            vix=vix,
            near_earnings=False,
            ticker=ticker,
            account_size=ACCOUNT_SIZE,
        )
        ravish.print_ravish_card(ravish_signal, vix, prob)


if __name__ == "__main__":
    main()
