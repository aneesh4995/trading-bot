"""
Earnings Short Put Screener & Alert.

Checks upcoming earnings for major stocks and sends a notification
the day before earnings with the exact trade setup:
  - Sell 20-delta put, 2 DTE
  - Enter at 3:30 PM before close
  - Exit first 30 min of next-day open
  - Profits from IV crush overnight

Usage:
    python3 earnings_screener.py                # check all tickers
    python3 earnings_screener.py NVDA AAPL      # specific tickers
"""

import sys
import datetime
import urllib.request
import logging
import numpy as np
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

NTFY_TOPIC = "spy-qqq-screener-aneesh"

# Tickers to watch for earnings
EARNINGS_WATCHLIST = [
    "SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META",
    "TSLA", "AMD", "NFLX", "CRM", "COST", "JPM", "V", "MA",
]

# Minimum IV rank to make the trade worthwhile
MIN_IV_RANK = 25


def get_earnings_dates(ticker: str) -> list[datetime.date]:
    """Fetch upcoming earnings dates from Yahoo Finance."""
    try:
        t = yf.Ticker(ticker)

        # Method 1: calendar property
        try:
            cal = t.calendar
            if cal is not None:
                if hasattr(cal, "empty") and not cal.empty:
                    if hasattr(cal, "iloc") and "Earnings Date" in cal.index:
                        dates = cal.loc["Earnings Date"]
                        if hasattr(dates, "__iter__"):
                            return [d.date() if hasattr(d, "date") else d for d in dates]
                        return [dates.date() if hasattr(dates, "date") else dates]
                elif isinstance(cal, dict) and "Earnings Date" in cal:
                    dates = cal["Earnings Date"]
                    if isinstance(dates, list):
                        return [d.date() if hasattr(d, "date") else d for d in dates]
                    return [dates.date() if hasattr(dates, "date") else dates]
        except Exception:
            pass

        # Method 2: get_earnings_dates (shows past + future)
        try:
            ed = t.get_earnings_dates(limit=8)
            if ed is not None and not ed.empty:
                today = datetime.date.today()
                future = []
                for idx in ed.index:
                    d = idx.date() if hasattr(idx, "date") else idx
                    if isinstance(d, datetime.date) and d >= today:
                        future.append(d)
                if future:
                    return sorted(future)
        except Exception:
            pass

        return []
    except Exception as e:
        log.debug("Could not get earnings for %s: %s", ticker, e)
        return []


def get_earnings_from_options(ticker: str) -> dict:
    """Get IV and estimate if earnings are priced into options."""
    try:
        t = yf.Ticker(ticker)

        # Get current price
        info = t.fast_info
        price = info.get("lastPrice", 0) or info.get("previousClose", 0)
        if price == 0:
            return {}

        # Get nearest options expiry
        expirations = t.options
        if not expirations:
            return {}

        # Get options chain for nearest expiry
        nearest_exp = expirations[0]
        chain = t.option_chain(nearest_exp)
        puts = chain.puts

        if puts.empty:
            return {}

        # Find ~20-delta put (OTM put roughly 3-5% below price)
        target_strike = round(price * 0.95)
        puts_sorted = puts.iloc[(puts["strike"] - target_strike).abs().argsort()]
        target_put = puts_sorted.iloc[0]

        # IV rank estimate from historical vol
        hist = t.history(period="1y")
        if hist.empty:
            return {}
        returns = hist["Close"].pct_change().dropna()
        rolling_vol = returns.rolling(21).std() * np.sqrt(252)
        rolling_vol = rolling_vol.dropna()

        if len(rolling_vol) < 30:
            return {}

        current_iv = float(target_put.get("impliedVolatility", 0)) * 100
        hist_low = float(rolling_vol.min()) * 100
        hist_high = float(rolling_vol.max()) * 100

        iv_rank = ((current_iv - hist_low) / (hist_high - hist_low) * 100
                   if hist_high > hist_low else 50)

        # Expected move = IV * sqrt(DTE/365) * price
        exp_date = datetime.date.fromisoformat(nearest_exp)
        dte = (exp_date - datetime.date.today()).days
        expected_move = price * (current_iv / 100) * np.sqrt(max(dte, 1) / 365)

        return {
            "price": round(price, 2),
            "expiry": nearest_exp,
            "dte": dte,
            "target_strike": float(target_put["strike"]),
            "put_bid": float(target_put.get("bid", 0)),
            "put_ask": float(target_put.get("ask", 0)),
            "put_mid": round((float(target_put.get("bid", 0)) + float(target_put.get("ask", 0))) / 2, 2),
            "iv": round(current_iv, 1),
            "iv_rank": round(iv_rank, 0),
            "expected_move": round(expected_move, 2),
            "expected_move_pct": round(expected_move / price * 100, 1),
        }
    except Exception as e:
        log.debug("Options data error for %s: %s", ticker, e)
        return {}


def scan_earnings(tickers: list[str] = None) -> list[dict]:
    """Scan tickers for upcoming earnings within 2 days."""
    if tickers is None:
        tickers = EARNINGS_WATCHLIST

    today = datetime.date.today()
    tomorrow = today + datetime.timedelta(days=1)
    day_after = today + datetime.timedelta(days=2)

    results = []

    for ticker in tickers:
        log.info("Checking %s...", ticker)
        dates = get_earnings_dates(ticker)

        earnings_soon = False
        earnings_date = None
        for d in dates:
            if isinstance(d, datetime.date) and today <= d <= day_after:
                earnings_soon = True
                earnings_date = d
                break

        if not earnings_soon:
            continue

        # Get options data for the trade setup
        opts = get_earnings_from_options(ticker)
        if not opts:
            continue

        # Check if IV is high enough
        iv_rank = opts.get("iv_rank", 0)
        if iv_rank < MIN_IV_RANK:
            log.info("  %s: IV Rank %.0f%% too low (need >%d%%)", ticker, iv_rank, MIN_IV_RANK)
            continue

        days_until = (earnings_date - today).days

        result = {
            "ticker": ticker,
            "earnings_date": str(earnings_date),
            "days_until": days_until,
            "trade_today": days_until == 1,  # trade day before earnings
            **opts,
        }
        results.append(result)
        log.info("  %s: Earnings %s (%d days) | IV Rank: %.0f%% | Expected move: $%.2f (%.1f%%)",
                 ticker, earnings_date, days_until, iv_rank,
                 opts.get("expected_move", 0), opts.get("expected_move_pct", 0))

    return results


def print_earnings_card(result: dict):
    """Print trade setup card for an earnings short put."""
    width = 60
    sep = "-" * width

    ticker = result["ticker"]
    price = result.get("price", 0)
    strike = result.get("target_strike", 0)
    premium = result.get("put_mid", 0)
    iv = result.get("iv", 0)
    iv_rank = result.get("iv_rank", 0)
    exp_move = result.get("expected_move", 0)
    exp_move_pct = result.get("expected_move_pct", 0)
    dte = result.get("dte", 0)
    earnings_date = result.get("earnings_date", "")
    trade_today = result.get("trade_today", False)

    print()
    print("+" + sep + "+")
    print("|{:^{w}}|".format("EARNINGS SHORT PUT: " + ticker, w=width))
    print("|{:^{w}}|".format("Earnings: " + earnings_date, w=width))
    print("+" + sep + "+")

    if trade_today:
        print("|{:^{w}}|".format(">>> TRADE TODAY at 3:30 PM <<<", w=width))
    else:
        print("|{:^{w}}|".format("Earnings in {} day(s) - watch".format(result["days_until"]), w=width))
    print("+" + sep + "+")

    print("|  Stock Price : ${:<{w}}|".format("{:.2f}".format(price), w=width - 17))
    print("|  Sell Strike : ${:<{w}}|".format("{:.0f} (~20 delta, below expected move)".format(strike), w=width - 17))
    print("|  Premium     : ${:<{w}}|".format("{:.2f} (${:.0f} per contract)".format(premium, premium * 100), w=width - 17))
    print("|  IV          : {:<{w}}|".format("{:.1f}% (Rank: {:.0f}%)".format(iv, iv_rank), w=width - 17))
    print("|  Exp Move    : ${:<{w}}|".format("{:.2f} ({:.1f}%)".format(exp_move, exp_move_pct), w=width - 17))
    print("|  Expiry      : {:<{w}}|".format("{} ({} DTE)".format(result.get("expiry", ""), dte), w=width - 17))
    print("+" + sep + "+")
    print("|  ENTRY: Sell put at 3:30 PM (day before earnings){:>{w}}|".format("", w=width - 49))
    print("|  EXIT : Buy back in first 30 min of next open{:>{w}}|".format("", w=width - 47))
    print("|  PROFIT: IV crush overnight = keep ~80-100% credit{:>{w}}|".format("", w=width - 51))
    print("+" + sep + "+")

    # Risk check
    max_loss = (strike * 100) - (premium * 100)
    print("|  Max credit : ${:<{w}}|".format("{:,.0f}".format(premium * 100), w=width - 17))
    print("|  Max risk   : ${:<{w}}|".format("{:,.0f} (if stock drops to $0)".format(max_loss), w=width - 17))
    print("|  Realistic  : Stock rarely moves > expected move{:>{w}}|".format("", w=width - 50))
    print("+" + sep + "+")


def send_earnings_alert(results: list[dict]):
    """Send earnings trade alert via ntfy."""
    if not results:
        return

    trade_today = [r for r in results if r.get("trade_today")]
    watch = [r for r in results if not r.get("trade_today")]

    lines = []
    if trade_today:
        lines.append("TRADE TODAY at 3:30 PM:")
        for r in trade_today:
            lines.append("  {} | Sell ${:.0f} put @ ${:.2f} | IV:{:.0f}%".format(
                r["ticker"], r["target_strike"], r.get("put_mid", 0), r.get("iv", 0)))
        lines.append("")

    if watch:
        lines.append("UPCOMING:")
        for r in watch:
            lines.append("  {} earnings {} | IV Rank:{:.0f}%".format(
                r["ticker"], r["earnings_date"], r.get("iv_rank", 0)))

    title = "Earnings Alert: {} setup(s)".format(len(results))
    body = "\n".join(lines)
    priority = "high" if trade_today else "default"

    data = body.encode("utf-8")
    req = urllib.request.Request(
        "https://ntfy.sh/{}".format(NTFY_TOPIC),
        data=data, method="POST")
    req.add_header("Title", title)
    req.add_header("Priority", priority)
    req.add_header("Tags", "money_with_wings" if trade_today else "calendar")

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status == 200:
                log.info("Sent earnings alert to ntfy.sh/%s", NTFY_TOPIC)
    except Exception as e:
        log.error("Failed to send alert: %s", e)


def main():
    tickers = [a for a in sys.argv[1:] if not a.startswith("-")] or None

    print("=" * 50)
    print("  EARNINGS SHORT PUT SCREENER")
    print("  Scanning for earnings within 2 days...")
    print("=" * 50)

    results = scan_earnings(tickers)

    if not results:
        print("\nNo earnings trades found in the next 2 days.")
        print("Checked: {}".format(", ".join(tickers or EARNINGS_WATCHLIST)))
        return

    for r in results:
        print_earnings_card(r)

    # Send notification
    send_earnings_alert(results)

    trade_today = [r for r in results if r.get("trade_today")]
    if trade_today:
        print("\n{} trade(s) to execute TODAY at 3:30 PM!".format(len(trade_today)))
    else:
        print("\n{} upcoming earnings to watch.".format(len(results)))


if __name__ == "__main__":
    main()
