#!/usr/bin/env python3
"""
polymarket_trader.main — Polymarket US trader entry point (T13-46).

Main entry point for the Polymarket trader. Can run in paper or live mode.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from typing import Any, Dict, List, Optional

from .state_db import PolymarketStateDB
from .ledger_writer import PolymarketLedgerWriter
from .fees import PolymarketFeeCalculator, Network
from .paper_fills import PaperFillSimulator, FillModel
from .mempalace_hooks import PolymarketMemPalaceHooks, get_hooks
from .guard_wiring import GuardWiring, RiskLimits
from .cycle import PolymarketCycle, CycleResult

log = logging.getLogger("polymarket_trader.main")


class PolymarketTrader:
    """Main Polymarket trader orchestrator."""

    def __init__(
        self,
        paper_mode: bool = True,
        network: Network = Network.POLYGON,
        starting_balance: float = 1000.0,
        risk_limits: RiskLimits = None,
        interval: float = 15.0,
    ):
        self.paper_mode = paper_mode
        self.network = network
        self.interval = interval
        self.running = False

        # Initialize components
        self.state_db = PolymarketStateDB()
        self.ledger = PolymarketLedgerWriter()
        self.fee_calculator = PolymarketFeeCalculator(network)
        self.fill_simulator = PaperFillSimulator(FillModel.REALISTIC)
        self.mempalace_hooks = get_hooks()
        self.guard_wiring = GuardWiring(risk_limits or RiskLimits(), starting_balance)

        # Initialize balance if not set
        if self.state_db.get_balance() == 0.0:
            self.state_db.set_balance(starting_balance)

        # Update guard wiring with starting balance
        self.guard_wiring.update_balance(starting_balance, 0.0)

        # Trading cycle
        self.cycle = PolymarketCycle(
            state_db=self.state_db,
            ledger=self.ledger,
            fee_calculator=self.fee_calculator,
            fill_simulator=self.fill_simulator,
            mempalace_hooks=self.mempalace_hooks,
            guard_wiring=self.guard_wiring,
            paper_mode=paper_mode,
            network=network,
        )

        # Signal handling
        self._shutdown = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        log.info("Received signal %d, shutting down...", signum)
        self._shutdown = True
        self.running = False

    def fetch_markets(self) -> List[Dict[str, Any]]:
        """
        Fetch active markets from Polymarket.
        In production, this would call the Polymarket API.
        """
        # Demo markets - replace with real API call
        return [
            {
                "market_id": "0x123",
                "asset": "BTC",
                "bid": 0.48,
                "ask": 0.52,
                "last_price": 0.50,
                "volume_24h": 100000,
                "open_interest": 50000,
            },
            {
                "market_id": "0x456",
                "asset": "ETH",
                "bid": 0.45,
                "ask": 0.55,
                "last_price": 0.50,
                "volume_24h": 80000,
                "open_interest": 40000,
            },
            {
                "market_id": "0x789",
                "asset": "SOL",
                "bid": 0.40,
                "ask": 0.60,
                "last_price": 0.50,
                "volume_24h": 50000,
                "open_interest": 25000,
            },
        ]

    def run(self, once: bool = False):
        """Run the trading loop."""
        self.running = True
        mode = "PAPER" if self.paper_mode else "LIVE"
        log.info("Starting Polymarket trader in %s mode (interval=%.1fs)", mode, self.interval)

        # Initial heartbeat
        self.mempalace_hooks.heartbeat("startup", {"mode": mode, "balance": self.state_db.get_balance()})

        try:
            while self.running and not self._shutdown:
                cycle_start = time.time()

                try:
                    # Fetch markets
                    markets = self.fetch_markets()

                    # Run cycle
                    result: CycleResult = self.cycle.run_cycle(markets)

                    # Log result
                    log.info(
                        "Cycle %d: markets=%d signals=%d orders=%d filled=%d pnl=%.2f time=%.1fms",
                        result.cycle, result.markets_scanned, result.signals_generated,
                        result.orders_placed, result.orders_filled, result.pnl, result.duration_ms
                    )

                    if result.errors:
                        for err in result.errors:
                            log.warning("Cycle error: %s", err)

                    # Print status
                    status = self.guard_wiring.get_status()
                    log.info(
                        "Status: balance=%.2f exposure=%.2f drawdown=%.2f%% halted=%s breaker=%s",
                        status["balance"], status["exposure"],
                        status["drawdown_pct"] * 100, status["halted"], status["api_breaker_state"]
                    )

                except Exception as e:
                    log.error("Cycle failed: %s", e)
                    self.guard_wiring.record_api_failure()
                    self.mempalace_hooks.file_incident(
                        f"cycle_error_{int(time.time())}",
                        f"Cycle failed: {e}",
                        "high"
                    )

                # Check if we should stop
                if once:
                    break

                # Sleep until next cycle
                elapsed = time.time() - cycle_start
                sleep_time = max(0.1, self.interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        finally:
            self.shutdown()

    def shutdown(self):
        """Graceful shutdown."""
        log.info("Shutting down Polymarket trader...")
        self.running = False

        # Final checkpoint
        balance = self.state_db.get_balance()
        positions = self.state_db.get_positions()
        open_orders = self.state_db.get_open_orders()
        daily_pnl = self.state_db.get_daily_pnl()
        session_pnl = self.state_db.get_session_pnl()

        self.mempalace_hooks.checkpoint(
            balance=balance,
            positions=len(positions),
            open_orders=len(open_orders),
            daily_pnl=daily_pnl,
            session_pnl=session_pnl,
        )

        # Final heartbeat
        self.mempalace_hooks.heartbeat("shutdown", {"balance": balance, "session_pnl": session_pnl})

        log.info("Final balance: %.2f, Session P&L: %.2f, Positions: %d", balance, session_pnl, len(positions))
        log.info("Polymarket trader stopped")


def main():
    parser = argparse.ArgumentParser(description="Polymarket US Trader")
    parser.add_argument("--live", action="store_true", help="Run in live mode (default: paper)")
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit")
    parser.add_argument("--interval", type=float, default=15.0, help="Cycle interval in seconds")
    parser.add_argument("--balance", type=float, default=1000.0, help="Starting balance (paper mode)")
    parser.add_argument("--network", choices=["polygon", "mainnet"], default="polygon", help="Network")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    network = Network.POLYGON if args.network == "polygon" else Network.MAINNET

    trader = PolymarketTrader(
        paper_mode=not args.live,
        network=network,
        starting_balance=args.balance,
        interval=args.interval,
    )

    trader.run(once=args.once)


if __name__ == "__main__":
    main()