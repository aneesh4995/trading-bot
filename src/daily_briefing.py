#!/usr/bin/env python3
"""
Daily Trading Briefing — One combined notification with everything you need.

Combines:
  1. Credit spread screener (GO/CAUTION/WAIT) for SPY & QQQ
  2. ML ensemble confidence scores
  3. Next ideal entry window estimate
  4. Earnings short put setups for watchlist stocks

Usage:
    python3 daily_briefing.py                    # full briefing
    python3 daily_briefing.py --no-send          # print only, don't send ntfy
"""

import sys
import os
import datetime
import urllib.request
import logging
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "spy-qqq-screener-aneesh")

# Heavy-weight stocks whose earnings move SPY/QQQ
QQQ_HEAVYWEIGHTS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
    "AMD", "NFLX", "CRM", "COST", "AVGO", "ADBE",
]
SPY_HEAVYWEIGHTS = QQQ_HEAVYWEIGHTS + ["JPM", "V", "MA", "UNH", "JNJ", "PG"]


def run_credit_spread_screener():
    """Run screener for SPY and QQQ, return results."""
    from screener import CreditSpreadScreener
    screener = CreditSpreadScreener()
    results = {}
    for ticker in ["SPY", "QQQ"]:
        results[ticker] = screener.screen(ticker)
    return results


def get_ml_detail():
    """Get ensemble prediction detail for SPY and QQQ."""
    from data_handler import DataHandler
    from indicators import FeatureEngineer
    from predictor import EnsemblePredictor

    handler = DataHandler()
    engineer = FeatureEngineer()
    details = {}

    for ticker in ["SPY", "QQQ"]:
        df = engineer.add_all(handler.fetch_ohlcv(ticker, period="2y"))
        df = engineer.add_vix(df)
        df = engineer.add_put_call_ratio(df)

        ens = EnsemblePredictor()
        ens.train(df)
        label, prob = ens.predict(df)
        detail = ens.predict_detail(df)

        details[ticker] = {
            "direction": "UP" if label == 1 else "DOWN",
            "confidence": prob,
            "models": detail,
            "agreement": sum(1 for d in detail.values() if d["label"] == label),
        }
    return details


def estimate_next_go_window(screen_results: dict) -> dict:
    """Estimate when the next GO window might open."""
    from screener import FOMC_DATES_2026
    today = datetime.date.today()
    hints = {}

    for ticker, result in screen_results.items():
        blockers = []
        for check in result.checks:
            if not check.passed:
                blockers.append(check.name)

        # Estimate when each blocker clears
        days_to_clear = 0
        notes = []

        if "No FOMC" in blockers:
            # Find next FOMC date and when it clears
            for d in FOMC_DATES_2026:
                event = datetime.date.fromisoformat(d)
                if abs((event - today).days) <= 1:
                    clear_day = event + datetime.timedelta(days=2)
                    days_to_clear = max(days_to_clear, (clear_day - today).days)
                    notes.append("FOMC clears {}".format(clear_day.strftime("%a %m/%d")))
                    break

        if "No CPI/NFP" in blockers:
            days_to_clear = max(days_to_clear, 2)
            notes.append("CPI/NFP clears in ~2 days")

        if "Price > EMA_20" in blockers:
            notes.append("Price below EMA_20 (need rally)")

        if "VIX Level" in blockers or "VIX Hard Stop" in blockers:
            notes.append("VIX too high (need calm)")

        if "IV Rank" in blockers:
            notes.append("IV Rank low (premiums cheap)")

        if not blockers:
            hints[ticker] = {"estimate": "TODAY", "notes": ["All checks passed!"]}
        elif days_to_clear > 0 and len(blockers) <= 2:
            target = today + datetime.timedelta(days=days_to_clear)
            hints[ticker] = {
                "estimate": target.strftime("%a %m/%d"),
                "notes": notes,
            }
        else:
            hints[ticker] = {
                "estimate": "uncertain",
                "notes": notes or ["Multiple blockers active"],
            }

    return hints


def scan_heavyweight_earnings() -> list[dict]:
    """Check if any SPY/QQQ heavy-weight stocks have earnings within 5 days."""
    from earnings_screener import get_earnings_dates, get_earnings_from_options

    today = datetime.date.today()
    window = today + datetime.timedelta(days=5)
    upcoming = []

    for ticker in SPY_HEAVYWEIGHTS:
        dates = get_earnings_dates(ticker)
        for d in dates:
            if isinstance(d, datetime.date) and today <= d <= window:
                opts = get_earnings_from_options(ticker)
                upcoming.append({
                    "ticker": ticker,
                    "date": str(d),
                    "days": (d - today).days,
                    "trade_day": (d - today).days == 1,
                    "iv": opts.get("iv", 0),
                    "iv_rank": opts.get("iv_rank", 0),
                    "expected_move": opts.get("expected_move_pct", 0),
                    "put_strike": opts.get("target_strike", 0),
                    "put_premium": opts.get("put_mid", 0),
                })
                break

    return upcoming


def build_briefing() -> tuple[str, str, str]:
    """Build the full daily briefing. Returns (title, body, priority)."""
    print("Running credit spread screener...")
    screen_results = run_credit_spread_screener()

    print("Running ML ensemble predictions...")
    ml_details = get_ml_detail()

    print("Estimating next GO window...")
    go_windows = estimate_next_go_window(screen_results)

    print("Scanning heavyweight earnings...")
    earnings = scan_heavyweight_earnings()

    # Build the notification
    now = datetime.datetime.now().strftime("%m/%d %I:%M %p")
    from data_handler import DataHandler
    vix = DataHandler().fetch_vix()

    lines = []
    lines.append("Daily Briefing {}".format(now))
    lines.append("VIX: {:.1f}".format(vix))
    lines.append("")

    # Credit spread verdicts
    best_verdict = "WAIT"
    for ticker in ["SPY", "QQQ"]:
        r = screen_results[ticker]
        ml = ml_details[ticker]

        verdict = r.verdict
        score = r.score
        direction = ml["direction"]
        confidence = ml["confidence"]
        agreement = ml["agreement"]

        if verdict == "GO":
            best_verdict = "GO"
        elif verdict == "CAUTION" and best_verdict != "GO":
            best_verdict = "CAUTION"

        lines.append("{}: {} ({:.0%})".format(ticker, verdict, score))
        lines.append("  ML: {} @ {:.0%} ({}/3 models agree)".format(
            direction, confidence, agreement))

        # Per-model breakdown
        for name, d in ml["models"].items():
            lines.append("    {}: {} {:.0%}".format(name.upper(), d["direction"], d["prob"]))

        # Failed checks
        failed = [c.name for c in r.checks if not c.passed]
        if failed:
            lines.append("  Blockers: {}".format(", ".join(failed)))

        # Next GO window
        window = go_windows.get(ticker, {})
        est = window.get("estimate", "unknown")
        if est != "TODAY":
            notes = "; ".join(window.get("notes", []))
            lines.append("  Next GO: ~{} ({})".format(est, notes))
        else:
            lines.append("  >>> OPEN SPREAD TODAY <<<")

        lines.append("")

    # Earnings section
    if earnings:
        lines.append("--- EARNINGS ---")
        for e in earnings:
            day_label = "TODAY" if e["days"] == 0 else "TOMORROW" if e["days"] == 1 else "in {}d".format(e["days"])
            trade_flag = " [TRADE TODAY 3:30PM]" if e.get("trade_day") else ""
            lines.append("{} earnings {} | IV:{:.0f}% | Move:{:.1f}%{}".format(
                e["ticker"], day_label, e["iv_rank"], e["expected_move"], trade_flag))
            if e.get("trade_day") and e["put_premium"] > 0:
                lines.append("  Sell ${:.0f} put @ ${:.2f}".format(e["put_strike"], e["put_premium"]))
        lines.append("")

    # Trade recommendation
    lines.append("--- RECOMMENDATION ---")
    if best_verdict == "GO":
        lines.append("OPEN credit spread today!")
        lines.append("Sell 50d put, buy 10pts lower, 14 DTE, TP 50%")
    elif best_verdict == "CAUTION":
        lines.append("WAIT for blockers to clear.")
        for ticker in ["SPY", "QQQ"]:
            w = go_windows.get(ticker, {})
            if w.get("estimate") and w["estimate"] != "uncertain":
                lines.append("  {} likely GO: ~{}".format(ticker, w["estimate"]))
    else:
        lines.append("WAIT. Conditions unfavorable.")

    # Title
    spy_verdict = screen_results["SPY"].verdict
    qqq_verdict = screen_results["QQQ"].verdict
    spy_conf = ml_details["SPY"]["confidence"]
    qqq_conf = ml_details["QQQ"]["confidence"]
    title = "SPY:{} {:.0%} | QQQ:{} {:.0%} | VIX:{:.0f}".format(
        spy_verdict, spy_conf, qqq_verdict, qqq_conf, vix)

    # Priority
    if best_verdict == "GO":
        priority = "high"
    elif any(e.get("trade_day") for e in earnings):
        priority = "high"
    elif best_verdict == "CAUTION":
        priority = "default"
    else:
        priority = "low"

    body = "\n".join(lines)
    return title, body, priority


def send_briefing(title: str, body: str, priority: str):
    """Send briefing via ntfy."""
    data = body.encode("utf-8")
    req = urllib.request.Request(
        "https://ntfy.sh/{}".format(NTFY_TOPIC),
        data=data, method="POST")
    req.add_header("Title", title)
    req.add_header("Priority", priority)

    emoji = "green_circle" if "GO" in title else "yellow_circle" if "CAUTION" in title else "red_circle"
    req.add_header("Tags", emoji)

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception as e:
        print("Failed to send: {}".format(e))
        return False


def main():
    no_send = "--no-send" in sys.argv

    title, body, priority = build_briefing()

    print("\n" + "=" * 60)
    print("  " + title)
    print("=" * 60)
    print(body)
    print("=" * 60)

    if no_send:
        print("\n(--no-send flag: notification not sent)")
    else:
        print("\nSending notification...")
        ok = send_briefing(title, body, priority)
        print("Sent!" if ok else "FAILED to send")


if __name__ == "__main__":
    main()
