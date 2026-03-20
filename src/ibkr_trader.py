"""
IBKR Paper Trading Integration for Bull Put Credit Spreads.

Connects to Interactive Brokers TWS or IB Gateway via ib_insync.
Supports: real-time quotes, options chains, multi-leg spread orders.

Setup:
    1. Download TWS or IB Gateway from ibkr.com
    2. Enable API access: TWS → File → Global Config → API → Settings
       - Check "Enable ActiveX and Socket Clients"
       - Socket port: 7497 (paper) or 7496 (live)
       - Check "Allow connections from localhost only"
    3. Log in to your paper trading account
    4. Run this module

Usage:
    python3 ibkr_trader.py              # dry run (show order, don't submit)
    python3 ibkr_trader.py --execute    # submit order to paper account
"""

import sys
import datetime
import logging
from dataclasses import dataclass
from typing import Optional

from ib_insync import (
    IB, Stock, Option, Contract, ComboLeg,
    LimitOrder, MarketOrder, TagValue,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

# Paper trading port (TWS default)
PAPER_PORT = 7497
GATEWAY_PAPER_PORT = 4002
LIVE_PORT = 7496
CLIENT_ID = 1


@dataclass
class SpreadOrder:
    ticker: str
    short_strike: float
    long_strike: float
    expiry: str          # YYYYMMDD
    net_credit: float
    max_loss: float
    quantity: int = 1


@dataclass
class SpreadFill:
    order_id: int
    status: str
    fill_price: float
    commission: float
    filled_time: str


class IBKRTrader:
    """Manages IBKR connection and bull put credit spread execution."""

    def __init__(self, paper: bool = True, port: int = None):
        self.ib = IB()
        self.paper = paper
        self.port = port or (PAPER_PORT if paper else LIVE_PORT)
        self.connected = False

    def connect(self, host: str = "127.0.0.1", client_id: int = CLIENT_ID) -> bool:
        """Connect to TWS/Gateway. Returns True if successful."""
        try:
            self.ib.connect(host, self.port, clientId=client_id)
            self.connected = True
            acct = self.ib.managedAccounts()
            log.info("Connected to IBKR %s (account: %s)",
                     "PAPER" if self.paper else "LIVE", acct)

            # Safety check: refuse to run on live if paper mode requested
            if self.paper and acct and not any("DU" in a or "paper" in a.lower() for a in acct):
                log.warning("WARNING: Account %s may be LIVE, not paper. Disconnecting.", acct)
                self.disconnect()
                return False
            return True
        except Exception as e:
            log.error("Failed to connect to IBKR on port %d: %s", self.port, e)
            log.error("Make sure TWS/Gateway is running and API is enabled.")
            return False

    def disconnect(self):
        if self.connected:
            self.ib.disconnect()
            self.connected = False
            log.info("Disconnected from IBKR.")

    # ── Market Data ──────────────────────────────────────────

    def get_quote(self, ticker: str) -> dict:
        """Get real-time quote for a stock."""
        contract = Stock(ticker, "SMART", "USD")
        self.ib.qualifyContracts(contract)
        [ticker_data] = self.ib.reqTickers(contract)
        return {
            "ticker": ticker,
            "bid": ticker_data.bid,
            "ask": ticker_data.ask,
            "last": ticker_data.last,
            "mid": round((ticker_data.bid + ticker_data.ask) / 2, 2) if ticker_data.bid > 0 else ticker_data.last,
            "volume": ticker_data.volume,
        }

    def get_options_chain(self, ticker: str, right: str = "P",
                          max_dte: int = 30) -> list[dict]:
        """Fetch put options chain for nearest expiry within max_dte days."""
        stock = Stock(ticker, "SMART", "USD")
        self.ib.qualifyContracts(stock)

        chains = self.ib.reqSecDefOptParams(stock.symbol, "", stock.secType, stock.conId)
        if not chains:
            log.error("No options chains found for %s", ticker)
            return []

        # Find the SMART exchange chain
        chain = next((c for c in chains if c.exchange == "SMART"), chains[0])

        today = datetime.date.today()
        target = today + datetime.timedelta(days=max_dte)

        # Find expiry closest to target DTE
        valid_expiries = []
        for exp_str in sorted(chain.expirations):
            # IBKR returns YYYYMMDD, convert to date
            if len(exp_str) == 8 and exp_str.isdigit():
                exp_date = datetime.date(int(exp_str[:4]), int(exp_str[4:6]), int(exp_str[6:8]))
            else:
                exp_date = datetime.date.fromisoformat(exp_str)
            dte = (exp_date - today).days
            if 7 <= dte <= max_dte + 7:
                valid_expiries.append((exp_str, dte))

        if not valid_expiries:
            log.error("No valid expiries found within %d DTE for %s", max_dte, ticker)
            return []

        # Pick expiry closest to target DTE (14 days)
        best_exp = min(valid_expiries, key=lambda x: abs(x[1] - 14))
        expiry = best_exp[0]
        dte = best_exp[1]
        log.info("Selected expiry: %s (%d DTE) for %s", expiry, dte, ticker)

        # Get strikes near current price
        quote = self.get_quote(ticker)
        price = quote["mid"] or quote["last"]

        # Filter strikes within reasonable range
        strikes = [s for s in sorted(chain.strikes)
                   if price - 30 <= s <= price + 10]

        # Build option contracts and get prices
        options = []
        for strike in strikes:
            opt = Option(ticker, expiry, strike, right, "SMART")
            options.append(opt)

        self.ib.qualifyContracts(*options)
        tickers = self.ib.reqTickers(*options)

        result = []
        for opt, tkr in zip(options, tickers):
            greeks = tkr.modelGreeks or tkr.lastGreeks
            result.append({
                "strike": opt.strike,
                "expiry": opt.lastTradeDateOrContractMonth,
                "bid": tkr.bid if tkr.bid > 0 else 0,
                "ask": tkr.ask if tkr.ask > 0 else 0,
                "mid": round((tkr.bid + tkr.ask) / 2, 2) if tkr.bid > 0 and tkr.ask > 0 else 0,
                "last": tkr.last if tkr.last > 0 else 0,
                "delta": round(greeks.delta, 3) if greeks and greeks.delta else None,
                "iv": round(greeks.impliedVol * 100, 1) if greeks and greeks.impliedVol else None,
                "theta": round(greeks.theta, 4) if greeks and greeks.theta else None,
                "gamma": round(greeks.gamma, 5) if greeks and greeks.gamma else None,
                "dte": dte,
                "conId": opt.conId,
            })

        return [r for r in result if r["mid"] > 0 or r["last"] > 0]

    # ── Spread Builder ───────────────────────────────────────

    def find_bull_put_spread(self, ticker: str, target_dte: int = 14,
                             width: int = 10, target_short_delta: float = -0.50
                             ) -> Optional[SpreadOrder]:
        """Find optimal bull put credit spread.

        Sells ~50-delta put, buys put `width` points lower.
        Returns SpreadOrder with strikes and estimated credit.
        """
        puts = self.get_options_chain(ticker, right="P", max_dte=target_dte + 7)
        if not puts:
            return None

        # Find put closest to target delta (e.g., -0.50)
        puts_with_delta = [p for p in puts if p["delta"] is not None]
        if not puts_with_delta:
            log.error("No delta data available. Market may be closed.")
            return None

        short_put = min(puts_with_delta, key=lambda p: abs(p["delta"] - target_short_delta))
        short_strike = short_put["strike"]
        long_strike = short_strike - width

        # Find the long put
        long_put = next((p for p in puts if p["strike"] == long_strike), None)
        if not long_put:
            # Find closest available strike
            long_put = min(puts, key=lambda p: abs(p["strike"] - long_strike))
            long_strike = long_put["strike"]

        # Calculate credit
        short_credit = short_put["mid"] or short_put["last"]
        long_cost = long_put["mid"] or long_put["last"]
        net_credit = round(short_credit - long_cost, 2)
        spread_width = short_strike - long_strike
        max_loss = round(spread_width - net_credit, 2)

        expiry = short_put["expiry"]

        log.info("Found spread: SELL %s %s P @ $%.2f | BUY %s %s P @ $%.2f",
                 ticker, short_strike, short_credit, ticker, long_strike, long_cost)
        log.info("Net credit: $%.2f | Max loss: $%.2f | Width: $%.0f",
                 net_credit, max_loss, spread_width)
        log.info("Short delta: %.3f | IV: %.1f%% | DTE: %d",
                 short_put["delta"], short_put["iv"] or 0, short_put["dte"])

        return SpreadOrder(
            ticker=ticker,
            short_strike=short_strike,
            long_strike=long_strike,
            expiry=expiry,
            net_credit=net_credit,
            max_loss=max_loss,
        )

    # ── Order Execution ──────────────────────────────────────

    def place_spread_order(self, spread: SpreadOrder,
                           dry_run: bool = True) -> Optional[SpreadFill]:
        """Place a bull put credit spread as a combo order.

        Args:
            spread: SpreadOrder with strikes and expiry
            dry_run: If True, only show order details without submitting
        """
        # Build the two option legs
        short_opt = Option(spread.ticker, spread.expiry, spread.short_strike, "P", "SMART")
        long_opt = Option(spread.ticker, spread.expiry, spread.long_strike, "P", "SMART")
        self.ib.qualifyContracts(short_opt, long_opt)

        if dry_run:
            print("\n" + "=" * 60)
            print("  DRY RUN — ORDER NOT SUBMITTED")
            print("=" * 60)
            print("  Bull Put Credit Spread on {}".format(spread.ticker))
            print("  SELL {} {} {} P".format(spread.quantity, spread.ticker, spread.short_strike))
            print("  BUY  {} {} {} P".format(spread.quantity, spread.ticker, spread.long_strike))
            print("  Expiry: {}".format(spread.expiry))
            print("  Credit: ${:.2f} (${:.0f} per contract)".format(
                spread.net_credit, spread.net_credit * 100))
            print("  Max Loss: ${:.2f} (${:.0f} per contract)".format(
                spread.max_loss, spread.max_loss * 100))
            print("  Max Profit: ${:.0f} | Max Risk: ${:.0f}".format(
                spread.net_credit * 100 * spread.quantity,
                spread.max_loss * 100 * spread.quantity))
            print("=" * 60)
            return None

        # Build combo contract
        combo = Contract()
        combo.symbol = spread.ticker
        combo.secType = "BAG"
        combo.currency = "USD"
        combo.exchange = "SMART"

        # Sell short put
        leg1 = ComboLeg()
        leg1.conId = short_opt.conId
        leg1.ratio = 1
        leg1.action = "SELL"
        leg1.exchange = "SMART"

        # Buy long put
        leg2 = ComboLeg()
        leg2.conId = long_opt.conId
        leg2.ratio = 1
        leg2.action = "BUY"
        leg2.exchange = "SMART"

        combo.comboLegs = [leg1, leg2]

        # Limit order at the net credit
        order = LimitOrder(
            action="SELL",
            totalQuantity=spread.quantity,
            lmtPrice=spread.net_credit,
            tif="DAY",
        )

        log.info("Submitting spread order to IBKR %s...",
                 "PAPER" if self.paper else "LIVE")
        trade = self.ib.placeOrder(combo, order)

        # Wait for fill (up to 30 seconds)
        self.ib.sleep(2)
        for _ in range(15):
            if trade.orderStatus.status in ("Filled", "Cancelled", "Inactive"):
                break
            self.ib.sleep(2)

        status = trade.orderStatus.status
        fill_price = trade.orderStatus.avgFillPrice or 0
        commission = sum(f.commissionReport.commission for f in trade.fills) if trade.fills else 0

        log.info("Order status: %s | Fill: $%.2f | Commission: $%.2f",
                 status, fill_price, commission)

        return SpreadFill(
            order_id=trade.order.orderId,
            status=status,
            fill_price=fill_price,
            commission=commission,
            filled_time=str(datetime.datetime.now()),
        )

    # ── Account Info ─────────────────────────────────────────

    def account_summary(self) -> dict:
        """Get paper account summary."""
        summary = self.ib.accountSummary()
        result = {}
        for item in summary:
            if item.tag in ("NetLiquidation", "TotalCashValue", "BuyingPower",
                            "AvailableFunds", "MaintMarginReq"):
                result[item.tag] = float(item.value)
        return result

    def open_positions(self) -> list[dict]:
        """Get current open positions."""
        positions = self.ib.positions()
        result = []
        for pos in positions:
            result.append({
                "symbol": pos.contract.symbol,
                "secType": pos.contract.secType,
                "strike": getattr(pos.contract, "strike", None),
                "right": getattr(pos.contract, "right", None),
                "expiry": getattr(pos.contract, "lastTradeDateOrContractMonth", None),
                "quantity": pos.position,
                "avg_cost": pos.avgCost,
                "value": pos.position * pos.avgCost,
            })
        return result


# ── Integrated Signal → Order Pipeline ───────────────────────

def run_signal_to_order(ticker: str = "SPY", dry_run: bool = True):
    """Full pipeline: screener → IBKR spread order."""
    from screener import CreditSpreadScreener, print_screen
    from risk_manager import RiskManager
    from data_handler import DataHandler

    # Step 1: Run screener
    print("Step 1: Running screener for {}...".format(ticker))
    screener = CreditSpreadScreener()
    result = screener.screen(ticker)
    print_screen(result)

    verdict = result.verdict
    score = result.score

    if verdict == "WAIT":
        print("\nVerdict is WAIT — no trade today.")
        return

    # Step 2: Connect to IBKR
    print("\nStep 2: Connecting to IBKR paper trading...")
    trader = IBKRTrader(paper=True)
    if not trader.connect():
        print("Cannot connect to IBKR. Is TWS/Gateway running?")
        return

    try:
        # Step 3: Get account info
        acct = trader.account_summary()
        print("\nAccount: ${:,.0f} net liq | ${:,.0f} buying power".format(
            acct.get("NetLiquidation", 0), acct.get("BuyingPower", 0)))

        # Step 4: Adaptive position sizing
        handler = DataHandler()
        vix = handler.fetch_vix()
        account_size = acct.get("NetLiquidation", 1000)
        risk = RiskManager(account_size=account_size, risk_tolerance="moderate")
        position_size = risk.adaptive_position_size(vix)

        # How many contracts? Each spread risks (width * 100) per contract
        # Cap at 10 contracts for safety (even on paper)
        width = 10
        risk_per_contract = width * 100  # $1,000 per contract
        num_contracts = max(1, min(10, int(position_size / risk_per_contract)))

        print("VIX: {:.1f} | Adaptive size: ${:,.0f} | Contracts: {}".format(
            vix, position_size, num_contracts))

        # Step 5: Find the spread
        print("\nStep 3: Finding optimal bull put credit spread...")
        spread = trader.find_bull_put_spread(ticker, target_dte=14, width=width)
        if not spread:
            print("Could not find valid spread. Market may be closed.")
            return

        spread.quantity = num_contracts

        # Step 6: Place order (or dry run)
        if verdict == "GO" or (verdict == "CAUTION" and not dry_run):
            print("\nStep 4: Placing order...")
            fill = trader.place_spread_order(spread, dry_run=dry_run)
            if fill and fill.status == "Filled":
                print("ORDER FILLED at ${:.2f}".format(fill.fill_price))
        else:
            print("\nStep 4: Showing order (CAUTION verdict, dry run only)...")
            trader.place_spread_order(spread, dry_run=True)

    finally:
        trader.disconnect()


def main():
    dry_run = "--execute" not in sys.argv
    tickers = [a for a in sys.argv[1:] if not a.startswith("--")] or ["SPY"]

    if not dry_run:
        print("!" * 50)
        print("  EXECUTE MODE — Orders will be submitted!")
        print("  (Paper trading account)")
        print("!" * 50)

    for ticker in tickers:
        run_signal_to_order(ticker, dry_run=dry_run)


if __name__ == "__main__":
    main()
