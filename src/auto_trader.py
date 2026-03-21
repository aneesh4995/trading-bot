#!/usr/bin/env python3
"""
Automated Trading Scheduler — Local macOS runner for SPY/QQQ credit spreads.

Runs continuously, evaluates every 30 minutes during market hours.
Supports two modes:
  --semi-auto (default): Screen + alert via ntfy, no orders placed
  --full-auto:           Screen + auto-submit on IBKR paper account

Data failover: IBKR real-time → yfinance fallback.
State: uses existing trade_log.json to prevent duplicate trades.

Usage:
    caffeinate -i python3 src/auto_trader.py --semi-auto
    caffeinate -i python3 src/auto_trader.py --full-auto

macOS sleep prevention:
    caffeinate -i python3 auto_trader.py         # prevent idle sleep
    sudo pmset -a disablesleep 1                 # prevent all sleep
    sudo pmset -a disablesleep 0                 # re-enable sleep
"""

import argparse
import datetime
import json
import logging
import os
import signal
import sys
import time
import urllib.request
from logging.handlers import RotatingFileHandler
from typing import Optional

import schedule
import pandas_market_calendars as mcal

# ── Lazy imports from existing modules (same /src directory) ─────────────
from ibkr_trader import IBKRTrader, SpreadOrder
from ibkr_monitor import IBKRMonitor, TradeLog, send_monitor_alert, NTFY_TOPIC
from screener import CreditSpreadScreener
from data_handler import DataHandler
from risk_manager import RiskManager

# ── Constants ────────────────────────────────────────────────────────────

TICKERS = ["SPY", "QQQ"]
CYCLE_INTERVAL_MINUTES = 30
IBKR_MAX_RETRIES = 3
IBKR_BASE_BACKOFF_SEC = 2
PAPER_PORT = 7497
TRADER_CLIENT_ID = 10
MONITOR_CLIENT_ID = 11
DEFAULT_ACCOUNT_SIZE = 5000
SPREAD_WIDTH = 10

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
LOG_FILE = os.path.join(LOG_DIR, "auto_trader.log")

_shutdown = False


# ── Logging ──────────────────────────────────────────────────────────────

def setup_logging() -> logging.Logger:
    """Configure console + rotating file logging."""
    os.makedirs(LOG_DIR, exist_ok=True)

    logger = logging.getLogger("auto_trader")
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File (5 MB, 3 backups)
    fh = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ── Market Hours ─────────────────────────────────────────────────────────

def is_market_open() -> bool:
    """Check if NYSE is currently open (handles holidays, half-days, DST)."""
    try:
        nyse = mcal.get_calendar("NYSE")
        today = datetime.date.today()
        sched = nyse.schedule(start_date=today, end_date=today)

        if sched.empty:
            return False

        now_utc = datetime.datetime.now(datetime.timezone.utc)
        market_open = sched.iloc[0]["market_open"].to_pydatetime()
        market_close = sched.iloc[0]["market_close"].to_pydatetime()

        return market_open <= now_utc <= market_close
    except Exception:
        # Fallback: simple weekday + hour check (ET = UTC-5 / UTC-4)
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        if now_utc.weekday() >= 5:
            return False
        hour_utc = now_utc.hour + now_utc.minute / 60
        # Market roughly 13:30–20:00 UTC (9:30–4:00 ET)
        return 13.5 <= hour_utc <= 20.0


# ── IBKR Connection ─────────────────────────────────────────────────────

def connect_ibkr_with_retry(trader: IBKRTrader, logger: logging.Logger,
                            max_retries: int = IBKR_MAX_RETRIES) -> bool:
    """Connect to IBKR with exponential backoff. Returns True on success."""
    for attempt in range(max_retries):
        logger.info("IBKR connection attempt %d/%d...", attempt + 1, max_retries)
        try:
            if trader.connect(client_id=TRADER_CLIENT_ID):
                logger.info("IBKR connected successfully.")
                return True
        except Exception as e:
            logger.warning("IBKR attempt %d failed: %s", attempt + 1, e)

        if attempt < max_retries - 1:
            wait = IBKR_BASE_BACKOFF_SEC * (2 ** attempt)
            logger.info("Retrying in %ds...", wait)
            time.sleep(wait)

    logger.error("IBKR connection failed after %d attempts.", max_retries)
    return False


# ── Data Failover ────────────────────────────────────────────────────────

def get_live_price(ticker: str, trader: Optional[IBKRTrader],
                   logger: logging.Logger) -> dict:
    """Get live price: IBKR first, yfinance fallback."""
    if trader and trader.connected:
        try:
            quote = trader.get_quote(ticker)
            price = quote.get("mid") or quote.get("last", 0)
            if price and price > 0:
                logger.info("%s price from IBKR: $%.2f", ticker, price)
                return {"ticker": ticker, "price": price, "source": "ibkr"}
        except Exception as e:
            logger.warning("IBKR quote failed for %s: %s — falling back to yfinance", ticker, e)

    try:
        quote = DataHandler().fetch_live_quote(ticker)
        price = quote.get("price", 0)
        logger.info("%s price from yfinance: $%.2f", ticker, price)
        return {"ticker": ticker, "price": price, "source": "yfinance"}
    except Exception as e:
        logger.error("yfinance quote also failed for %s: %s", ticker, e)
        return {"ticker": ticker, "price": 0, "source": "none"}


# ── Duplicate Check ──────────────────────────────────────────────────────

def has_open_position(trade_log: TradeLog, ticker: str) -> bool:
    """Check if there's already an open position for this ticker."""
    for pos in trade_log.open_positions():
        if pos.get("ticker") == ticker:
            return True
    return False


# ── yfinance Spread Fallback ────────────────────────────────────────────

def find_spread_yfinance_fallback(ticker: str, logger: logging.Logger) -> Optional[dict]:
    """Estimate bull put spread from yfinance options chain (degraded path)."""
    try:
        handler = DataHandler()
        chain_data = handler.fetch_options_chain(ticker, expiry_index=0)
        puts = chain_data.get("puts")
        expiry = chain_data.get("expiry")

        if puts is None or puts.empty or not expiry:
            return None

        price = handler.fetch_live_quote(ticker).get("price", 0)
        if price <= 0:
            return None

        # ATM put (closest strike to price)
        short_strike = float(puts.iloc[(puts["strike"] - price).abs().argsort()].iloc[0]["strike"])
        long_strike = short_strike - SPREAD_WIDTH

        # Find long put
        long_row = puts[puts["strike"] == long_strike]
        if long_row.empty:
            long_row = puts.iloc[(puts["strike"] - long_strike).abs().argsort()].iloc[[0]]
            long_strike = float(long_row.iloc[0]["strike"])

        short_row = puts.iloc[(puts["strike"] - short_strike).abs().argsort()].iloc[0]
        long_row_data = long_row.iloc[0] if not long_row.empty else None

        short_mid = (float(short_row.get("bid", 0)) + float(short_row.get("ask", 0))) / 2
        long_mid = 0
        if long_row_data is not None:
            long_mid = (float(long_row_data.get("bid", 0)) + float(long_row_data.get("ask", 0))) / 2

        net_credit = round(short_mid - long_mid, 2)

        logger.info("yfinance spread estimate: %s SELL %s / BUY %s P, credit ~$%.2f",
                     ticker, short_strike, long_strike, net_credit)

        return {
            "ticker": ticker,
            "short_strike": short_strike,
            "long_strike": long_strike,
            "expiry": expiry,
            "net_credit": max(net_credit, 0),
            "max_loss": round((short_strike - long_strike) - max(net_credit, 0), 2),
            "source": "yfinance_estimate",
        }
    except Exception as e:
        logger.error("yfinance spread fallback failed for %s: %s", ticker, e)
        return None


# ── Notification ─────────────────────────────────────────────────────────

def send_ntfy(title: str, body: str, priority: str = "default",
              logger: Optional[logging.Logger] = None) -> None:
    """Send push notification via ntfy.sh."""
    try:
        data = body.encode("utf-8")
        req = urllib.request.Request(
            "https://ntfy.sh/{}".format(NTFY_TOPIC),
            data=data, method="POST")
        req.add_header("Title", title)
        req.add_header("Priority", priority)
        req.add_header("Tags", "chart_with_upwards_trend")
        urllib.request.urlopen(req, timeout=10)
        if logger:
            logger.info("Sent ntfy: %s", title)
    except Exception as e:
        if logger:
            logger.error("ntfy send failed: %s", e)


# ── Core Cycle ───────────────────────────────────────────────────────────

def run_cycle(mode: str, trade_log: TradeLog, logger: logging.Logger) -> dict:
    """Execute one screening + trading cycle."""
    cycle_start = time.time()
    summary = {"timestamp": datetime.datetime.now().isoformat(), "tickers": {}}

    # 1. Market hours check
    if not is_market_open():
        logger.info("Market closed — skipping cycle.")
        return {"market_open": False}

    logger.info("=== Cycle start (mode: %s) ===", mode)

    # 2. IBKR connection
    trader = IBKRTrader(paper=True, port=PAPER_PORT)
    ibkr_available = connect_ibkr_with_retry(trader, logger)

    if not ibkr_available:
        logger.warning("Running in yfinance-only mode.")
        trader = None

    # Get account size for position sizing
    account_size = DEFAULT_ACCOUNT_SIZE
    if trader and trader.connected:
        try:
            acct = trader.account_summary()
            account_size = acct.get("NetLiquidation", DEFAULT_ACCOUNT_SIZE)
            logger.info("Account size: $%,.0f", account_size)
        except Exception:
            pass

    try:
        # Get VIX once per cycle
        try:
            vix = DataHandler().fetch_vix()
        except Exception:
            vix = 20.0
        logger.info("VIX: %.1f", vix)

        screener = CreditSpreadScreener()

        for ticker in TICKERS:
            ticker_result = {"verdict": None, "action": None}

            try:
                # 3a. Screen
                logger.info("Screening %s...", ticker)
                result = screener.screen(ticker)
                verdict = result.verdict
                score = result.score
                ticker_result["verdict"] = verdict
                logger.info("%s: %s (%.0f%%)", ticker, verdict, score * 100)

                # 3b. Check for existing position
                if has_open_position(trade_log, ticker):
                    logger.info("%s: Open position exists — checking exits.", ticker)
                    ticker_result["action"] = "monitor_exits"

                    if trader and trader.connected:
                        _run_exit_evaluation(ticker, mode, trade_log, logger)
                    else:
                        logger.warning("Cannot evaluate exits without IBKR connection.")
                        ticker_result["action"] = "monitor_exits_skipped"

                # 3c. GO signal + no position → find and maybe place spread
                elif verdict == "GO":
                    logger.info("%s: GO signal — looking for spread.", ticker)
                    _handle_go_signal(ticker, mode, trader, trade_log,
                                      vix, account_size, logger)
                    ticker_result["action"] = "spread_search"

                else:
                    ticker_result["action"] = "wait"
                    failed = [c.name for c in result.checks if not c.passed]
                    if failed:
                        logger.info("%s: Blockers: %s", ticker, ", ".join(failed))

            except Exception as e:
                logger.error("Error processing %s: %s", ticker, e, exc_info=True)
                ticker_result["action"] = "error"
                ticker_result["error"] = str(e)

            summary["tickers"][ticker] = ticker_result

    finally:
        if trader and trader.connected:
            trader.disconnect()
            logger.info("IBKR disconnected.")

    elapsed = time.time() - cycle_start
    logger.info("=== Cycle complete in %.1fs ===", elapsed)
    summary["elapsed_sec"] = round(elapsed, 1)
    return summary


def _handle_go_signal(ticker: str, mode: str, trader: Optional[IBKRTrader],
                      trade_log: TradeLog, vix: float, account_size: float,
                      logger: logging.Logger) -> None:
    """Handle GO verdict: find spread, alert or execute."""
    # Position sizing
    risk = RiskManager(account_size=account_size, risk_tolerance="moderate")
    position_size = risk.adaptive_position_size(vix)
    num_contracts = max(1, min(10, int(position_size / (SPREAD_WIDTH * 100))))
    logger.info("%s: Adaptive size $%,.0f → %d contract(s)", ticker, position_size, num_contracts)

    # Find spread
    spread = None
    spread_info = None

    if trader and trader.connected:
        try:
            spread = trader.find_bull_put_spread(ticker, target_dte=14, width=SPREAD_WIDTH)
            if spread:
                spread.quantity = num_contracts
        except Exception as e:
            logger.warning("IBKR spread search failed for %s: %s", ticker, e)

    if not spread:
        # yfinance fallback (alert only, no execution)
        spread_info = find_spread_yfinance_fallback(ticker, logger)
        if not spread_info:
            logger.warning("%s: Could not find spread from any source.", ticker)
            return

    # Build alert body
    if spread:
        body = (
            "{ticker} Bull Put Credit Spread\n"
            "SELL {short} P / BUY {long} P\n"
            "Expiry: {expiry}\n"
            "Credit: ${credit:.2f} (${credit_100:.0f}/contract)\n"
            "Max Loss: ${loss:.2f} (${loss_100:.0f}/contract)\n"
            "Contracts: {qty}\n"
            "VIX: {vix:.1f}"
        ).format(
            ticker=ticker,
            short=spread.short_strike,
            long=spread.long_strike,
            expiry=spread.expiry,
            credit=spread.net_credit,
            credit_100=spread.net_credit * 100,
            loss=spread.max_loss,
            loss_100=spread.max_loss * 100,
            qty=spread.quantity,
            vix=vix,
        )
    else:
        body = (
            "{ticker} Bull Put Credit Spread (estimate)\n"
            "SELL {short} P / BUY {long} P\n"
            "Expiry: {expiry}\n"
            "Est. Credit: ${credit:.2f}\n"
            "Contracts: {qty}\n"
            "VIX: {vix:.1f}\n"
            "Source: yfinance (IBKR unavailable)"
        ).format(
            ticker=ticker,
            short=spread_info["short_strike"],
            long=spread_info["long_strike"],
            expiry=spread_info["expiry"],
            credit=spread_info["net_credit"],
            qty=num_contracts,
            vix=vix,
        )

    title = "GO: {} Spread Signal".format(ticker)

    if mode == "semi-auto":
        # Alert only
        send_ntfy(title + " [REVIEW]", body, priority="high", logger=logger)
        logger.info("%s: Semi-auto alert sent. No order placed.", ticker)

    elif mode == "full-auto":
        if not spread or not trader or not trader.connected:
            # Cannot auto-trade without IBKR — degrade to alert
            send_ntfy(title + " [NO IBKR - REVIEW]", body, priority="high", logger=logger)
            logger.warning("%s: Full-auto degraded to alert (no IBKR).", ticker)
            return

        # Submit order
        logger.info("%s: Submitting order (full-auto)...", ticker)
        try:
            fill = trader.place_spread_order(spread, dry_run=False)
            if fill and fill.status == "Filled":
                trade_log.log_entry({
                    "ticker": ticker,
                    "short_strike": spread.short_strike,
                    "long_strike": spread.long_strike,
                    "expiry": spread.expiry,
                    "entry_credit": spread.net_credit,
                    "quantity": spread.quantity,
                    "mode": "full-auto",
                })
                send_ntfy(
                    "FILLED: {} {}/{} P".format(ticker, spread.short_strike, spread.long_strike),
                    "Credit: ${:.2f} | {} contracts | Fill: ${:.2f}".format(
                        spread.net_credit, spread.quantity, fill.fill_price),
                    priority="high", logger=logger,
                )
                logger.info("%s: ORDER FILLED at $%.2f", ticker, fill.fill_price)
            else:
                status = fill.status if fill else "No fill"
                send_ntfy(
                    "ORDER STATUS: {} - {}".format(ticker, status),
                    body, priority="default", logger=logger,
                )
                logger.warning("%s: Order not filled. Status: %s", ticker, status)
        except Exception as e:
            logger.error("%s: Order execution failed: %s", ticker, e, exc_info=True)
            send_ntfy("ORDER ERROR: {}".format(ticker),
                      "Error: {}".format(e), priority="high", logger=logger)


def _run_exit_evaluation(ticker: str, mode: str, trade_log: TradeLog,
                         logger: logging.Logger) -> None:
    """Evaluate exits on open positions via IBKR."""
    monitor = IBKRMonitor(paper=True)
    try:
        monitor.ib.connect("127.0.0.1", PAPER_PORT, clientId=MONITOR_CLIENT_ID)
    except Exception as e:
        logger.error("Monitor connection failed: %s", e)
        return

    try:
        spreads = monitor.get_spread_positions()
        # Filter to this ticker
        ticker_spreads = [s for s in spreads if s.ticker == ticker]

        if not ticker_spreads:
            logger.info("%s: No IBKR spread positions found to evaluate.", ticker)
            return

        actions = monitor.evaluate_exits(ticker_spreads)
        dry_run = mode == "semi-auto"
        monitor.execute_actions(ticker_spreads, actions, dry_run=dry_run)

        # Log exits in full-auto mode
        if not dry_run:
            for sp, action in zip(ticker_spreads, actions):
                if action.action in ("close_tp", "close_dte"):
                    trade_log.log_exit({
                        "ticker": sp.ticker,
                        "short_strike": sp.short_strike,
                        "long_strike": sp.long_strike,
                        "expiry": sp.expiry,
                    }, reason=action.action, pnl_pct=action.pnl_pct)

        # Send monitor alert
        if actions:
            send_monitor_alert(actions)

        for action in actions:
            logger.info("%s exit eval: %s — %s", ticker, action.action, action.reason)

    except Exception as e:
        logger.error("Exit evaluation failed for %s: %s", ticker, e, exc_info=True)
    finally:
        monitor.disconnect()


# ── Signal Handler ───────────────────────────────────────────────────────

def handle_signal(signum, frame):
    """Graceful shutdown on SIGINT/SIGTERM."""
    global _shutdown
    _shutdown = True
    print("\nShutdown signal received — completing current cycle...")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Automated SPY/QQQ credit spread trader",
        epilog="Tip: wrap with 'caffeinate -i' to prevent macOS sleep",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--semi-auto", action="store_true", default=True,
                       help="Alert only, no orders placed (default)")
    group.add_argument("--full-auto", action="store_true",
                       help="Auto-submit orders on IBKR paper account")
    args = parser.parse_args()

    mode = "full-auto" if args.full_auto else "semi-auto"

    logger = setup_logging()

    # Startup banner
    logger.info("=" * 50)
    logger.info("  Auto Trader starting")
    logger.info("  Mode: %s", mode.upper())
    logger.info("  Tickers: %s", ", ".join(TICKERS))
    logger.info("  Cycle: every %d minutes", CYCLE_INTERVAL_MINUTES)
    logger.info("  IBKR port: %d (paper)", PAPER_PORT)
    logger.info("=" * 50)

    # Crash recovery: check for open positions
    trade_log = TradeLog()
    open_positions = trade_log.open_positions()
    if open_positions:
        logger.info("Found %d open position(s) from previous session:", len(open_positions))
        for pos in open_positions:
            logger.info("  %s %s/%s P exp %s",
                        pos.get("ticker"), pos.get("short_strike"),
                        pos.get("long_strike"), pos.get("expiry"))
    else:
        logger.info("No open positions found.")

    # Register signal handlers
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Run one immediate cycle
    logger.info("Running initial cycle...")
    run_cycle(mode, trade_log, logger)

    # Schedule recurring cycles
    schedule.every(CYCLE_INTERVAL_MINUTES).minutes.do(
        run_cycle, mode=mode, trade_log=trade_log, logger=logger)

    logger.info("Scheduler active — next cycle in %d minutes. Press Ctrl+C to stop.",
                CYCLE_INTERVAL_MINUTES)

    # Main loop
    while not _shutdown:
        schedule.run_pending()
        time.sleep(1)

    logger.info("Auto-trader shut down gracefully.")


if __name__ == "__main__":
    main()
