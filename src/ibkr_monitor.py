"""
IBKR Position Monitor & Exit Manager for Bull Put Credit Spreads.

Checks open spread positions and manages exits:
  - Take profit at 50% of max credit
  - Close if DTE <= 1 (avoid expiration risk)
  - Alert to roll if spread is deeply threatened
  - Log all actions to trade_log.json

Usage:
    python3 ibkr_monitor.py              # check positions, auto-close winners
    python3 ibkr_monitor.py --dry-run    # show what it would do, don't close
    python3 ibkr_monitor.py --status     # just show positions, no action
"""

import sys
import json
import os
import datetime
import logging
import urllib.request
from dataclasses import dataclass, asdict
from typing import Optional

from ib_insync import IB, Option, Contract, ComboLeg, LimitOrder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

PAPER_PORT = 7497
CLIENT_ID = 2  # different from trader (1) to allow simultaneous connections
TRADE_LOG = os.path.join(os.path.dirname(__file__), "trade_log.json")
NTFY_TOPIC = "spy-qqq-screener-aneesh"

# Exit rules
TP_PCT = 0.50          # take profit at 50% of max credit
CLOSE_DTE = 1          # close if 1 day or less to expiry
ROLL_ALERT_BUFFER = 5  # alert if price within 5 pts of short strike


@dataclass
class SpreadPosition:
    ticker: str
    short_strike: float
    long_strike: float
    expiry: str
    quantity: int
    entry_credit: float
    entry_date: str
    current_value: float = 0.0
    current_pnl_pct: float = 0.0
    dte: int = 0
    status: str = "open"  # open, closed, rolled


@dataclass
class ExitAction:
    ticker: str
    action: str          # close_tp, close_dte, roll_alert, hold
    reason: str
    pnl_pct: float
    spread_value: float
    entry_credit: float


class TradeLog:
    """Persistent JSON trade log."""

    def __init__(self, path: str = TRADE_LOG):
        self.path = path
        self.trades = self._load()

    def _load(self) -> list:
        if os.path.exists(self.path):
            with open(self.path) as f:
                return json.load(f)
        return []

    def _save(self):
        with open(self.path, "w") as f:
            json.dump(self.trades, f, indent=2, default=str)

    def log_entry(self, spread: dict):
        """Log a new trade entry."""
        spread["event"] = "entry"
        spread["timestamp"] = str(datetime.datetime.now())
        self.trades.append(spread)
        self._save()

    def log_exit(self, spread: dict, reason: str, pnl_pct: float):
        """Log a trade exit."""
        spread["event"] = "exit"
        spread["reason"] = reason
        spread["pnl_pct"] = pnl_pct
        spread["timestamp"] = str(datetime.datetime.now())
        self.trades.append(spread)
        self._save()

    def open_positions(self) -> list[dict]:
        """Get positions that were entered but not yet exited."""
        entries = {}
        for t in self.trades:
            key = "{}-{}-{}-{}".format(t.get("ticker"), t.get("short_strike"),
                                        t.get("long_strike"), t.get("expiry"))
            if t.get("event") == "entry":
                entries[key] = t
            elif t.get("event") == "exit":
                entries.pop(key, None)
        return list(entries.values())

    def summary(self) -> dict:
        """Overall performance summary."""
        exits = [t for t in self.trades if t.get("event") == "exit"]
        if not exits:
            return {"total_trades": 0}
        wins = [t for t in exits if t.get("pnl_pct", 0) > 0]
        losses = [t for t in exits if t.get("pnl_pct", 0) <= 0]
        return {
            "total_trades": len(exits),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(exits) if exits else 0,
            "avg_win_pct": sum(t["pnl_pct"] for t in wins) / len(wins) if wins else 0,
            "avg_loss_pct": sum(t["pnl_pct"] for t in losses) / len(losses) if losses else 0,
        }


class IBKRMonitor:
    """Monitors open positions and manages exits."""

    def __init__(self, paper: bool = True):
        self.ib = IB()
        self.paper = paper
        self.port = PAPER_PORT if paper else 7496
        self.trade_log = TradeLog()

    def connect(self) -> bool:
        try:
            self.ib.connect("127.0.0.1", self.port, clientId=CLIENT_ID)
            log.info("Monitor connected to IBKR %s", "PAPER" if self.paper else "LIVE")
            return True
        except Exception as e:
            log.error("Failed to connect: %s", e)
            return False

    def disconnect(self):
        self.ib.disconnect()

    def get_spread_positions(self) -> list[SpreadPosition]:
        """Detect open bull put credit spreads from IBKR positions."""
        positions = self.ib.positions()
        today = datetime.date.today()

        # Group options by ticker+expiry
        option_positions = {}
        for pos in positions:
            c = pos.contract
            if c.secType != "OPT" or c.right != "P":
                continue
            key = (c.symbol, c.lastTradeDateOrContractMonth)
            if key not in option_positions:
                option_positions[key] = []
            option_positions[key].append({
                "strike": c.strike,
                "quantity": int(pos.position),
                "avg_cost": pos.avgCost,
                "conId": c.conId,
            })

        spreads = []
        for (ticker, expiry), legs in option_positions.items():
            # Find short leg (negative qty) and long leg (positive qty)
            shorts = [l for l in legs if l["quantity"] < 0]
            longs = [l for l in legs if l["quantity"] > 0]

            for short in shorts:
                # Find matching long leg (lower strike)
                matching_long = None
                for lng in longs:
                    if lng["strike"] < short["strike"]:
                        matching_long = lng
                        break

                if not matching_long:
                    continue

                # Calculate DTE
                if len(expiry) == 8:
                    exp_date = datetime.date(int(expiry[:4]), int(expiry[4:6]), int(expiry[6:8]))
                else:
                    exp_date = datetime.date.fromisoformat(expiry)
                dte = (exp_date - today).days

                # Get current spread value
                short_opt = Option(ticker, expiry, short["strike"], "P", "SMART")
                long_opt = Option(ticker, expiry, matching_long["strike"], "P", "SMART")
                self.ib.qualifyContracts(short_opt, long_opt)
                tickers_data = self.ib.reqTickers(short_opt, long_opt)

                short_mid = 0
                long_mid = 0
                if len(tickers_data) >= 2:
                    t1, t2 = tickers_data[0], tickers_data[1]
                    if t1.bid > 0 and t1.ask > 0:
                        short_mid = (t1.bid + t1.ask) / 2
                    elif t1.last > 0:
                        short_mid = t1.last
                    if t2.bid > 0 and t2.ask > 0:
                        long_mid = (t2.bid + t2.ask) / 2
                    elif t2.last > 0:
                        long_mid = t2.last

                current_spread_value = round(short_mid - long_mid, 2)

                # Estimate entry credit from avg costs
                entry_credit = round(
                    abs(short["avg_cost"]) / 100 - abs(matching_long["avg_cost"]) / 100, 2
                )
                if entry_credit <= 0:
                    entry_credit = current_spread_value * 1.5  # rough estimate

                max_loss = (short["strike"] - matching_long["strike"]) - entry_credit
                pnl_pct = (entry_credit - current_spread_value) / max_loss if max_loss > 0 else 0

                spreads.append(SpreadPosition(
                    ticker=ticker,
                    short_strike=short["strike"],
                    long_strike=matching_long["strike"],
                    expiry=expiry,
                    quantity=abs(short["quantity"]),
                    entry_credit=entry_credit,
                    entry_date="",
                    current_value=current_spread_value,
                    current_pnl_pct=round(pnl_pct, 4),
                    dte=dte,
                ))

        return spreads

    def evaluate_exits(self, spreads: list[SpreadPosition]) -> list[ExitAction]:
        """Decide what to do with each open spread."""
        actions = []

        for sp in spreads:
            profit_pct = sp.current_pnl_pct
            max_credit = sp.entry_credit

            # Rule 1: Take profit at 50%
            if sp.current_value <= max_credit * (1 - TP_PCT) and sp.current_value >= 0:
                actions.append(ExitAction(
                    ticker=sp.ticker,
                    action="close_tp",
                    reason="Take profit: spread worth ${:.2f} (was ${:.2f} credit, {:.0%} profit)".format(
                        sp.current_value, max_credit, profit_pct),
                    pnl_pct=profit_pct,
                    spread_value=sp.current_value,
                    entry_credit=max_credit,
                ))
                continue

            # Rule 2: Close near expiry
            if sp.dte <= CLOSE_DTE:
                actions.append(ExitAction(
                    ticker=sp.ticker,
                    action="close_dte",
                    reason="Expiry tomorrow (DTE={}). Current P&L: {:.0%}".format(sp.dte, profit_pct),
                    pnl_pct=profit_pct,
                    spread_value=sp.current_value,
                    entry_credit=max_credit,
                ))
                continue

            # Rule 3: Roll alert (spread losing badly)
            if profit_pct < -0.50:
                actions.append(ExitAction(
                    ticker=sp.ticker,
                    action="roll_alert",
                    reason="Spread losing {:.0%}. Consider rolling down and out.".format(profit_pct),
                    pnl_pct=profit_pct,
                    spread_value=sp.current_value,
                    entry_credit=max_credit,
                ))
                continue

            # Rule 4: Hold
            actions.append(ExitAction(
                ticker=sp.ticker,
                action="hold",
                reason="Holding. P&L: {:.0%} | DTE: {} | Spread: ${:.2f}".format(
                    profit_pct, sp.dte, sp.current_value),
                pnl_pct=profit_pct,
                spread_value=sp.current_value,
                entry_credit=max_credit,
            ))

        return actions

    def close_spread(self, spread: SpreadPosition, dry_run: bool = True) -> bool:
        """Buy back the spread to close the position."""
        short_opt = Option(spread.ticker, spread.expiry, spread.short_strike, "P", "SMART")
        long_opt = Option(spread.ticker, spread.expiry, spread.long_strike, "P", "SMART")
        self.ib.qualifyContracts(short_opt, long_opt)

        if dry_run:
            print("  [DRY RUN] Would close: BUY {} {} P, SELL {} {} P @ ${:.2f}".format(
                spread.ticker, spread.short_strike,
                spread.ticker, spread.long_strike,
                spread.current_value))
            return True

        # Build closing combo (reverse of opening)
        combo = Contract()
        combo.symbol = spread.ticker
        combo.secType = "BAG"
        combo.currency = "USD"
        combo.exchange = "SMART"

        leg1 = ComboLeg()
        leg1.conId = short_opt.conId
        leg1.ratio = 1
        leg1.action = "BUY"  # buy back short
        leg1.exchange = "SMART"

        leg2 = ComboLeg()
        leg2.conId = long_opt.conId
        leg2.ratio = 1
        leg2.action = "SELL"  # sell long
        leg2.exchange = "SMART"

        combo.comboLegs = [leg1, leg2]

        # Buy back at current spread value
        order = LimitOrder(
            action="BUY",
            totalQuantity=spread.quantity,
            lmtPrice=spread.current_value,
            tif="DAY",
        )

        log.info("Closing spread: BUY %s %s/%s P @ $%.2f",
                 spread.ticker, spread.short_strike, spread.long_strike,
                 spread.current_value)

        trade = self.ib.placeOrder(combo, order)
        self.ib.sleep(2)
        for _ in range(15):
            if trade.orderStatus.status in ("Filled", "Cancelled", "Inactive"):
                break
            self.ib.sleep(2)

        status = trade.orderStatus.status
        log.info("Close order status: %s", status)

        if status == "Filled":
            self.trade_log.log_exit({
                "ticker": spread.ticker,
                "short_strike": spread.short_strike,
                "long_strike": spread.long_strike,
                "expiry": spread.expiry,
            }, reason="take_profit", pnl_pct=spread.current_pnl_pct)

        return status == "Filled"

    def execute_actions(self, spreads: list[SpreadPosition],
                        actions: list[ExitAction], dry_run: bool = True):
        """Execute the exit decisions."""
        for sp, action in zip(spreads, actions):
            if action.action == "close_tp":
                print("  CLOSE (take profit): {} {}/{} P | {:.0%} profit".format(
                    sp.ticker, sp.short_strike, sp.long_strike, action.pnl_pct))
                self.close_spread(sp, dry_run=dry_run)

            elif action.action == "close_dte":
                print("  CLOSE (expiry): {} {}/{} P | DTE={} | {:.0%} P&L".format(
                    sp.ticker, sp.short_strike, sp.long_strike, sp.dte, action.pnl_pct))
                self.close_spread(sp, dry_run=dry_run)

            elif action.action == "roll_alert":
                print("  ROLL ALERT: {} {}/{} P | {:.0%} loss".format(
                    sp.ticker, sp.short_strike, sp.long_strike, action.pnl_pct))
                print("    -> Consider rolling down and out to next month")

            else:
                print("  HOLD: {} {}/{} P | {:.0%} P&L | {} DTE".format(
                    sp.ticker, sp.short_strike, sp.long_strike,
                    action.pnl_pct, sp.dte))


def send_monitor_alert(actions: list[ExitAction]):
    """Send position summary via ntfy."""
    if not actions:
        return

    lines = []
    for a in actions:
        icon = {"close_tp": "+", "close_dte": "!", "roll_alert": "X", "hold": "-"}
        lines.append("[{}] {} {} {:.0%}".format(
            icon.get(a.action, "?"), a.ticker, a.action.upper(), a.pnl_pct))

    title = "Position Monitor: {} spreads".format(len(actions))
    body = "\n".join(lines)

    has_close = any(a.action.startswith("close") for a in actions)
    has_roll = any(a.action == "roll_alert" for a in actions)
    priority = "high" if has_roll else ("default" if has_close else "low")

    data = body.encode("utf-8")
    req = urllib.request.Request(
        "https://ntfy.sh/{}".format(NTFY_TOPIC),
        data=data, method="POST")
    req.add_header("Title", title)
    req.add_header("Priority", priority)
    req.add_header("Tags", "chart_with_upwards_trend")

    try:
        urllib.request.urlopen(req, timeout=10)
        log.info("Sent monitor alert to ntfy.sh/%s", NTFY_TOPIC)
    except Exception as e:
        log.error("Failed to send alert: %s", e)


def main():
    dry_run = "--dry-run" in sys.argv
    status_only = "--status" in sys.argv

    monitor = IBKRMonitor(paper=True)
    if not monitor.connect():
        print("Cannot connect to IBKR. Is TWS/Gateway running?")
        return

    try:
        # Account summary
        acct = monitor.ib.accountSummary()
        net_liq = 0
        for item in acct:
            if item.tag == "NetLiquidation":
                net_liq = float(item.value)
                break
        print("Account: ${:,.0f} net liquidation".format(net_liq))

        # Get open spread positions
        print("\nScanning for open bull put spreads...")
        spreads = monitor.get_spread_positions()

        if not spreads:
            print("No open spread positions found.")

            # Check trade log for history
            summary = monitor.trade_log.summary()
            if summary["total_trades"] > 0:
                print("\nTrade history: {} trades | {:.0%} win rate".format(
                    summary["total_trades"], summary["win_rate"]))
            return

        print("Found {} open spread(s):\n".format(len(spreads)))

        # Show positions
        print("  {:<6} {:>8} {:>8} {:>8} {:>5} {:>8} {:>8}".format(
            "Ticker", "Short", "Long", "Credit", "DTE", "Value", "P&L"))
        print("  " + "-" * 58)
        for sp in spreads:
            print("  {:<6} {:>8.0f} {:>8.0f} ${:>6.2f} {:>5} ${:>6.2f} {:>+7.0%}".format(
                sp.ticker, sp.short_strike, sp.long_strike,
                sp.entry_credit, sp.dte, sp.current_value, sp.current_pnl_pct))

        if status_only:
            return

        # Evaluate exits
        print("\nEvaluating exit rules...")
        actions = monitor.evaluate_exits(spreads)

        print()
        monitor.execute_actions(spreads, actions, dry_run=dry_run)

        # Send notification
        send_monitor_alert(actions)

        # Show trade log summary
        summary = monitor.trade_log.summary()
        if summary["total_trades"] > 0:
            print("\nLifetime: {} trades | {:.0%} win rate | Avg win: {:.1%} | Avg loss: {:.1%}".format(
                summary["total_trades"], summary["win_rate"],
                summary["avg_win_pct"], summary["avg_loss_pct"]))

    finally:
        monitor.disconnect()


if __name__ == "__main__":
    main()
