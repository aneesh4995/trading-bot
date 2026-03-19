#!/usr/bin/env python3
"""
Daily screener that sends GO/WAIT alerts via ntfy.sh push notifications.

Usage:
    python3 alert_screener.py                    # screen SPY QQQ, send alert
    python3 alert_screener.py AAPL MSFT          # custom tickers
    NTFY_TOPIC=my-topic python3 alert_screener.py  # custom ntfy topic

Setup:
    1. Install ntfy app on phone (iOS/Android)
    2. Subscribe to topic: spy-qqq-screener-aneesh
    3. Run this script or set up as cron job
"""

import sys
import json
import urllib.request
from screener import CreditSpreadScreener, ScreenResult


NTFY_TOPIC = "spy-qqq-screener-aneesh"


def format_alert(result: ScreenResult) -> tuple[str, str, str]:
    """Returns (title, body, priority) for ntfy."""
    verdict = result.verdict
    score = result.score

    if verdict == "GO":
        emoji = "green_circle"
        priority = "high"
    elif verdict == "CAUTION":
        emoji = "yellow_circle"
        priority = "default"
    else:
        emoji = "red_circle"
        priority = "low"

    title = f"{result.ticker}: {verdict} ({score:.0%})"

    lines = []
    for c in result.checks:
        icon = "+" if c.passed else "x"
        lines.append(f"[{icon}] {c.name}: {c.value}")

    if verdict == "GO":
        lines.append("")
        lines.append("TRADE: Sell 50d put, buy 10pts lower")
        lines.append("DTE: 14 | TP: 50% | No hard stop")

    body = "\n".join(lines)
    return title, body, priority, emoji


def send_ntfy(topic: str, title: str, body: str, priority: str = "default", emoji: str = "chart"):
    """Send push notification via ntfy.sh (no auth needed)."""
    import os
    topic = os.environ.get("NTFY_TOPIC", topic)

    data = body.encode("utf-8")
    req = urllib.request.Request(
        f"https://ntfy.sh/{topic}",
        data=data,
        method="POST",
    )
    req.add_header("Title", title)
    req.add_header("Priority", priority)
    req.add_header("Tags", emoji)

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception as e:
        print(f"  Failed to send ntfy: {e}")
        return False


def main():
    tickers = sys.argv[1:] if len(sys.argv) > 1 else ["SPY", "QQQ"]
    screener = CreditSpreadScreener()
    from data_handler import DataHandler
    handler = DataHandler()

    print("=" * 50)
    print("  DAILY CREDIT SPREAD SCREENER + ALERT")
    print("=" * 50)

    for ticker in tickers:
        print(f"\nScreening {ticker}...")

        # Fetch live price + IV
        try:
            quote = handler.fetch_live_quote(ticker)
            iv_data = handler.fetch_options_iv(ticker)
            market = handler.market_status()
            print(f"  Live: ${quote['price']} ({quote['change_pct']:+.2f}%) | Market: {market}")
            if iv_data.get("iv_atm_call"):
                print(f"  IV: Call={iv_data['iv_atm_call']}% Put={iv_data['iv_atm_put']}% Rank={iv_data.get('iv_rank', 'n/a')}")
        except Exception as e:
            print(f"  (live quote unavailable: {e})")

        result = screener.screen(ticker)
        title, body, priority, emoji = format_alert(result)

        print(f"  Verdict: {result.verdict} ({result.score:.0%})")
        print(f"  Sending notification...")

        ok = send_ntfy(NTFY_TOPIC, title, body, priority, emoji)
        if ok:
            print(f"  Sent to ntfy.sh/{NTFY_TOPIC}")
        else:
            print(f"  FAILED to send notification")

    print("\nDone.")


if __name__ == "__main__":
    main()
