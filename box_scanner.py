#!/usr/bin/env python3
"""
box_scanner.py — T15-30: Same-venue box scanner (log-only).

Scans for box spreads within a single venue where YES + NO prices sum to ~100c.
In Kalshi 15m markets, YES + NO should = 100c (minus fees).
Any deviation is a riskless arbitrage opportunity (box spread).

Logs all findings for analysis — does NOT execute.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

import config
from kalshi_client import KalshiClient

log = logging.getLogger(__name__)


@dataclass
class BoxOpportunity:
    """A box spread opportunity within a single venue."""
    venue: str  # "kalshi"
    ticker: str
    series: str
    window_start: datetime
    strike: float
    yes_bid: int
    yes_ask: int
    no_bid: int
    no_ask: int
    # Sum of best bid/ask
    bid_sum: int  # yes_bid + no_bid
    ask_sum: int  # yes_ask + no_ask
    # Deviation from 100c
    bid_deviation: float  # 100 - bid_sum (positive = opportunity)
    ask_deviation: float  # ask_sum - 100 (positive = opportunity)
    # Estimated net after fees
    net_edge_cents: float
    timestamp: str


class BoxScanner:
    """
    Scans for box spreads (YES + NO ≈ 100c) within a venue.
    
    In a well-functioning market:
    - yes_bid + no_ask ≤ 100 (you can buy YES at bid and NO at ask for ≤$1)
    - yes_ask + no_bid ≥ 100 (you can sell YES at ask and NO at bid for ≥$1)
    
    Arbitrage exists when:
    - yes_bid + no_ask < 100 - fees (buy box for less than $1)
    - yes_ask + no_bid > 100 + fees (sell box for more than $1)
    """

    def __init__(
        self,
        check_interval_seconds: float = 30.0,
        min_edge_cents: float = 0.5,
        min_liquidity: int = 10,
    ):
        self.check_interval = check_interval_seconds
        self.min_edge_cents = min_edge_cents
        self.min_liquidity = min_liquidity
        self.kalshi = KalshiClient()
        self.running = False
        self.stats = {
            "cycles": 0,
            "boxes_found": 0,
            "boxes_logged": 0,
            "errors": 0,
        }
        self.opportunity_log_path = config.BACKTEST_CAPTURE_DIR / "box_opportunities.jsonl"

        log.info(
            "BoxScanner initialized: interval=%.1fs, min_edge=%.1fc, min_liq=%d",
            check_interval_seconds, min_edge_cents, min_liquidity
        )

    def get_kalshi_markets(self) -> List[Dict]:
        """Get active 15m markets from Kalshi for all series."""
        all_markets = []
        series_list = ["KXBTC15M", "KXETH15M", "KXSOL15M", "KXXRP15M", "KXDOGE15M", "KXBNB15M", "KXHYPE15M", "KXZEC15M", "KXNEAR15M"]
        for series in series_list:
            try:
                markets = self.kalshi.get_markets(series_ticker=series, status="open", limit=100)
                if markets and "markets" in markets:
                    all_markets.extend(markets["markets"])
            except Exception as e:
                log.warning(f"Failed to fetch Kalshi markets for {series}: {e}")
        return all_markets

    def get_orderbook(self, ticker: str) -> Optional[Dict]:
        """Get orderbook for a ticker."""
        try:
            ob = self.kalshi.get_orderbook(ticker)
            return ob
        except Exception as e:
            log.debug(f"Orderbook fetch failed for {ticker}: {e}")
            return None

    def scan_once(self) -> List[BoxOpportunity]:
        """Run one scan cycle."""
        opportunities = []
        markets = self.get_kalshi_markets()

        for market in markets:
            ticker = market.get("ticker", "")
            if not ticker:
                continue

            # Only scan 15m up/down markets
            if "15M" not in ticker:
                continue

            ob = self.get_orderbook(ticker)
            if not ob:
                continue

            # Extract prices (cents)
            yes_bid = int(ob.get("yes_bid", 0) * 100) if ob.get("yes_bid") else 0
            yes_ask = int(ob.get("yes_ask", 0) * 100) if ob.get("yes_ask") else 0
            no_bid = int(ob.get("no_bid", 0) * 100) if ob.get("no_bid") else 0
            no_ask = int(ob.get("no_ask", 0) * 100) if ob.get("no_ask") else 0

            # Check liquidity
            yes_bid_size = ob.get("yes_bid_size", 0)
            yes_ask_size = ob.get("yes_ask_size", 0)
            no_bid_size = ob.get("no_bid_size", 0)
            no_ask_size = ob.get("no_ask_size", 0)
            min_size = min(yes_bid_size, yes_ask_size, no_bid_size, no_ask_size)
            if min_size < self.min_liquidity:
                continue

            # Skip if any price is zero (illiquid)
            if not all([yes_bid, yes_ask, no_bid, no_ask]):
                continue

            # Box scan: check bid_sum and ask_sum
            bid_sum = yes_bid + no_bid      # Buy YES at bid, buy NO at bid
            ask_sum = yes_ask + no_ask      # Sell YES at ask, sell NO at ask

            # Deviation from 100c
            bid_deviation = 100 - bid_sum   # Positive = buy box cheap
            ask_deviation = ask_sum - 100   # Positive = sell box expensive

            # Net edge after Kalshi taker fees (0.07 * p * (1-p))
            # For box: buy at bid (yes_bid + no_bid) or sell at ask (yes_ask + no_ask)
            # Fees on both legs
            fee_yes_bid = math.ceil(0.07 * yes_bid * (100 - yes_bid) / 10000) / 100
            fee_no_bid = math.ceil(0.07 * no_bid * (100 - no_bid) / 10000) / 100
            fee_yes_ask = math.ceil(0.07 * yes_ask * (100 - yes_ask) / 10000) / 100
            fee_no_ask = math.ceil(0.07 * no_ask * (100 - no_ask) / 10000) / 100

            total_fee_bid = fee_yes_bid + fee_no_bid
            total_fee_ask = fee_yes_ask + fee_no_ask

            # Net edge for buying box (bid_sum) and selling box (ask_sum)
            edge_buy = bid_deviation - total_fee_bid  # cents
            edge_sell = ask_deviation - total_fee_ask  # cents

            net_edge = max(edge_buy, edge_sell)

            if net_edge > self.min_edge_cents:
                # Parse series and window
                series = ticker.split("-")[0] if "-" in ticker else "UNKNOWN"
                try:
                    # KXBTC15M-26NOV200445-45000 format
                    parts = ticker.split("-")
                    window_str = parts[1] if len(parts) > 1 else ""
                    strike_str = parts[2] if len(parts) > 2 else ""
                    strike = float(strike_str) / 100.0 if strike_str else 0.0
                    # Parse window: 26NOV200445 -> 2026-11-20 04:45
                    window_start = self._parse_window(window_str) if window_str else datetime.now(timezone.utc)
                except Exception:
                    strike = 0.0
                    window_start = datetime.now(timezone.utc)
                    series = ticker

                opp = BoxOpportunity(
                    venue="kalshi",
                    ticker=ticker,
                    series=series,
                    window_start=window_start,
                    strike=strike,
                    yes_bid=yes_bid,
                    yes_ask=yes_ask,
                    no_bid=no_bid,
                    no_ask=no_ask,
                    bid_sum=bid_sum,
                    ask_sum=ask_sum,
                    bid_deviation=bid_deviation,
                    ask_deviation=ask_deviation,
                    net_edge_cents=net_edge,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
                opportunities.append(opp)

        return opportunities

    def _parse_window(self, window_str: str) -> datetime:
        """Parse window string like '26NOV200445' to datetime."""
        try:
            # 26NOV200445 -> year=26, month=NOV, day=20, hour=04, minute=45
            year = 2000 + int(window_str[0:2])
            month_str = window_str[2:5]
            month_map = {
                "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
                "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
                "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12
            }
            month = month_map.get(month_str, 1)
            day = int(window_str[5:7])
            hour = int(window_str[7:9])
            minute = int(window_str[9:11])
            return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)
        except Exception:
            return datetime.now(timezone.utc)

    def _log_opportunity(self, opp: BoxOpportunity):
        """Log a box opportunity."""
        import json
        from pathlib import Path

        record = {
            "timestamp": opp.timestamp,
            "type": "box_spread",
            "venue": opp.venue,
            "ticker": opp.ticker,
            "series": opp.series,
            "window_start": opp.window_start.isoformat(),
            "strike": opp.strike,
            "yes_bid": opp.yes_bid,
            "yes_ask": opp.yes_ask,
            "no_bid": opp.no_bid,
            "no_ask": opp.no_ask,
            "bid_sum": opp.bid_sum,
            "ask_sum": opp.ask_sum,
            "bid_deviation": opp.bid_deviation,
            "ask_deviation": opp.ask_deviation,
            "net_edge_cents": opp.net_edge_cents,
        }

        self.opportunity_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.opportunity_log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        self.stats["boxes_logged"] += 1
        self.stats["boxes_found"] += 1

        log.info(
            "BOX: %s @ %s | bid_sum=%d (dev=%.1fc) ask_sum=%d (dev=%.1fc) | net_edge=%.2fc",
            opp.ticker, opp.window_start.strftime("%H:%M:%S"),
            opp.bid_sum, opp.bid_deviation,
            opp.ask_sum, opp.ask_deviation,
            opp.net_edge_cents
        )

    async def run(self):
        """Main scanning loop."""
        self.running = True
        log.info("Starting box scanner...")

        while self.running:
            cycle_start = time.time()
            self.stats["cycles"] += 1

            try:
                opportunities = self.scan_once()
                for opp in opportunities:
                    self._log_opportunity(opp)

            except Exception as e:
                log.error(f"Error in box scanner loop: {e}")
                self.stats["errors"] += 1

            elapsed = time.time() - cycle_start
            sleep_time = max(0, self.check_interval - elapsed)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

        log.info("Box scanner stopped. Stats: %s", self.stats)

    def stop(self):
        """Stop the scanner."""
        self.running = False


async def main():
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s — %(message)s",
    )

    scanner = BoxScanner(
        check_interval_seconds=30.0,
        min_edge_cents=0.5,
        min_liquidity=10,
    )

    try:
        await scanner.run()
    except KeyboardInterrupt:
        log.info("Shutdown requested")
    finally:
        scanner.stop()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())