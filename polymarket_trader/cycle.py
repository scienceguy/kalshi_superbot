"""
polymarket_trader.cycle — Main trading cycle (T13-46).

Orchestrates the Polymarket trading loop:
- Market discovery
- Signal generation
- Risk checks
- Order placement
- Fill simulation (paper) / execution (live)
- Settlement processing
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .state_db import PolymarketStateDB, Position, Order
from .ledger_writer import PolymarketLedgerWriter
from .fees import PolymarketFeeCalculator, Network, calculate_fee, effective_price
from .paper_fills import PaperFillSimulator, PaperOrder, OrderType, FillModel, MarketSnapshot
from .mempalace_hooks import PolymarketMemPalaceHooks, get_hooks
from .guard_wiring import GuardWiring, RiskLimits, GuardStatus

log = logging.getLogger("polymarket_trader.cycle")


@dataclass
class CycleResult:
    """Result of a trading cycle."""
    cycle: int
    markets_scanned: int
    signals_generated: int
    orders_placed: int
    orders_filled: int
    pnl: float
    duration_ms: float
    errors: List[str]


class PolymarketCycle:
    """Main trading cycle for Polymarket."""

    def __init__(
        self,
        state_db: PolymarketStateDB = None,
        ledger: PolymarketLedgerWriter = None,
        fee_calculator: PolymarketFeeCalculator = None,
        fill_simulator: PaperFillSimulator = None,
        mempalace_hooks: PolymarketMemPalaceHooks = None,
        guard_wiring: GuardWiring = None,
        paper_mode: bool = True,
        network: Network = Network.POLYGON,
    ):
        self.state_db = state_db or PolymarketStateDB()
        self.ledger = ledger or PolymarketLedgerWriter()
        self.fee_calculator = fee_calculator or PolymarketFeeCalculator(network)
        self.fill_simulator = fill_simulator or PaperFillSimulator()
        self.mempalace_hooks = mempalace_hooks or get_hooks()
        self.guard_wiring = guard_wiring or GuardWiring()
        self.paper_mode = paper_mode
        self.network = network

        self.cycle_count = 0
        self.running = False

    def run_cycle(self, markets: List[Dict[str, Any]]) -> CycleResult:
        """Run a single trading cycle."""
        start_time = time.time()
        self.cycle_count += 1
        self.mempalace_hooks.set_cycle(self.cycle_count)

        errors = []
        signals_generated = 0
        orders_placed = 0
        orders_filled = 0
        cycle_pnl = 0.0

        try:
            # Update balance and exposure from state
            balance = self.state_db.get_balance()
            positions = self.state_db.get_positions()
            exposure = sum(p.size * p.current_price for p in positions)
            self.guard_wiring.update_balance(balance, exposure)

            # Send heartbeat
            self.mempalace_hooks.heartbeat("trading", {"balance": balance, "exposure": exposure})

            # Update market snapshots for fill simulator
            market_snapshots = {}
            for m in markets:
                snapshot = MarketSnapshot(
                    market_id=m.get("market_id", ""),
                    bid=m.get("bid", 0.0),
                    ask=m.get("ask", 0.0),
                    last_price=m.get("last_price", 0.0),
                    volume_24h=m.get("volume_24h", 0.0),
                    open_interest=m.get("open_interest", 0.0),
                    timestamp=time.time(),
                )
                market_snapshots[m.get("market_id", "")] = snapshot
                self.fill_simulator.update_market(snapshot)

            # Check pre-trade guards
            guard_result = self.guard_wiring.check_pre_trade(0.0)  # General check
            if guard_result.blocks_trading:
                log.warning("Pre-trade guard blocked: %s", guard_result.reason)
                errors.append(guard_result.reason)
                return CycleResult(
                    cycle=self.cycle_count,
                    markets_scanned=len(markets),
                    signals_generated=0,
                    orders_placed=0,
                    orders_filled=0,
                    pnl=0.0,
                    duration_ms=(time.time() - start_time) * 1000,
                    errors=errors,
                )

            # Generate signals from markets
            signals = self._generate_signals(markets)
            signals_generated = len(signals)

            # Process each signal
            for signal in signals:
                try:
                    trade_size = signal.get("size_usd", 0.0)
                    market_id = signal.get("market_id", "")

                    # Check trade-specific guards
                    guard_result = self.guard_wiring.check_pre_trade(trade_size, market_id)
                    if guard_result.blocks_trading:
                        log.warning("Trade guard blocked for %s: %s", market_id, guard_result.reason)
                        errors.append(f"{market_id}: {guard_result.reason}")
                        continue

                    # Place order
                    order = self._place_order(signal)
                    if order:
                        orders_placed += 1

                        # Simulate fill (paper) or execute (live)
                        if self.paper_mode:
                            fill_result = self._simulate_fill(order, market_snapshots.get(market_id))
                            if fill_result.filled:
                                orders_filled += 1
                                cycle_pnl += self._process_fill(order, fill_result, signal)
                        else:
                            # Live execution would go here
                            pass

                except Exception as e:
                    log.error("Error processing signal: %s", e)
                    errors.append(str(e))

            # Process settlements
            cycle_pnl += self._process_settlements()

            # Check post-trade guards
            post_guard = self.guard_wiring.check_post_trade()
            if post_guard and post_guard.blocks_trading:
                log.warning("Post-trade guard triggered halt: %s", post_guard.reason)
                errors.append(post_guard.reason)

            # Update position values in guards
            for pos in self.state_db.get_positions():
                self.guard_wiring.update_position(pos.market_id, pos.size * pos.current_price)

            # Checkpoint
            new_balance = self.state_db.get_balance()
            daily_pnl = self.state_db.get_daily_pnl()
            session_pnl = self.state_db.get_session_pnl()
            open_orders = len(self.state_db.get_open_orders())
            self.mempalace_hooks.checkpoint(
                balance=new_balance,
                positions=len(positions),
                open_orders=open_orders,
                daily_pnl=daily_pnl,
                session_pnl=session_pnl,
            )

        except Exception as e:
            log.error("Cycle error: %s", e)
            errors.append(str(e))

        duration_ms = (time.time() - start_time) * 1000
        return CycleResult(
            cycle=self.cycle_count,
            markets_scanned=len(markets),
            signals_generated=signals_generated,
            orders_placed=orders_placed,
            orders_filled=orders_filled,
            pnl=cycle_pnl,
            duration_ms=duration_ms,
            errors=errors,
        )

    def _generate_signals(self, markets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate trading signals from market data."""
        signals = []

        for market in markets:
            # Simple signal generation logic
            # In production, this would use the forecaster, ML models, etc.
            spread = market.get("ask", 0) - market.get("bid", 0)
            mid = (market.get("bid", 0) + market.get("ask", 0)) / 2

            # Example: signal when spread is tight and price is in favorable range
            if spread < 0.05 and 0.3 < mid < 0.7:
                # Calculate position size based on Kelly/guard wiring
                balance = self.state_db.get_balance()
                max_trade = min(balance * 0.08, self.guard_wiring.limits.max_trade_usd)

                signals.append({
                    "market_id": market.get("market_id", ""),
                    "asset": market.get("asset", ""),
                    "side": "YES" if mid > 0.5 else "NO",
                    "price": market.get("ask" if mid > 0.5 else "bid", mid),
                    "size_usd": max_trade,
                    "signal_strength": 1.0 - spread * 10,
                })

        return signals

    def _place_order(self, signal: Dict[str, Any]) -> Optional[Order]:
        """Place an order based on signal."""
        market_id = signal["market_id"]
        side = signal["side"]
        price = signal["price"]
        size_usd = signal["size_usd"]
        size = size_usd / price if price > 0 else 0

        # Calculate fees
        fee_result = self.fee_calculator.calculate_trade_fee(size, price, self.network)
        eff_price = self.fee_calculator.calculate_effective_price(price, size, "buy", self.network)

        order = Order(
            order_id=f"poly_{int(time.time() * 1000)}",
            market_id=market_id,
            side=side,
            size=size,
            price=eff_price,
            status="open",
            created_at=time.time(),
        )

        self.state_db.add_order(order)
        self.guard_wiring.update_position(market_id, size * price)

        log.info("Placed order: %s %s %.4f @ %.4f (fee: %.4f)", side, market_id, size, eff_price, fee_result.total_fee)
        return order

    def _simulate_fill(self, order: Order, market: Optional[MarketSnapshot]) -> "FillResult":
        """Simulate fill for paper trading."""
        if not market:
            from .paper_fills import FillResult
            return FillResult(filled=False)

        paper_order = PaperOrder(
            order_id=order.order_id,
            market_id=order.market_id,
            side=order.side,
            order_type=OrderType.MAKER,
            size=order.size,
            limit_price=order.price,
            created_at=order.created_at,
        )

        return self.fill_simulator.try_fill(paper_order, market)

    def _process_fill(self, order: Order, fill_result: "FillResult", signal: Dict[str, Any]) -> float:
        """Process a fill and update state."""
        # Update order
        self.state_db.update_order(order.order_id,
            status="filled",
            filled_at=time.time(),
            filled_size=fill_result.fill_size,
            filled_price=fill_result.fill_price,
        )

        # Calculate P&L (for paper trading, we realize at settlement)
        # For now, track unrealized
        position = Position(
            market_id=order.market_id,
            asset=signal.get("asset", ""),
            side=order.side,
            size=fill_result.fill_size,
            entry_price=fill_result.fill_price,
            entry_time=fill_result.latency_ms,
            current_price=fill_result.fill_price,
        )
        self.state_db.add_position(position)

        # Write to ledger
        fee_result = self.fee_calculator.calculate_trade_fee(fill_result.fill_size, fill_result.fill_price, self.network)
        self.ledger.write_trade(
            market_id=order.market_id,
            asset=signal.get("asset", ""),
            side=order.side,
            size=fill_result.fill_size,
            price=fill_result.fill_price,
            fee=fee_result.total_fee,
            order_id=order.order_id,
        )

        # Record trade result in guards (assume win for paper trading demo)
        self.guard_wiring.record_trade_result(True, 0.0)

        log.info("Fill: %s %s %.4f @ %.4f", order.side, order.market_id, fill_result.fill_size, fill_result.fill_price)
        return 0.0  # P&L realized at settlement

    def _process_settlements(self) -> float:
        """Process settled positions and realize P&L."""
        pnl = 0.0
        positions = self.state_db.get_positions()

        for pos in positions:
            # In real implementation, check for settled markets
            # For demo, randomly settle some
            import random
            if random.random() < 0.01:  # 1% chance per cycle
                settled_price = 1.0 if random.random() > 0.5 else 0.0
                entry_price = pos.entry_price

                if pos.side == "YES":
                    realized_pnl = (settled_price - entry_price) * pos.size
                else:
                    realized_pnl = (entry_price - settled_price) * pos.size

                # Update balance
                new_balance = self.state_db.update_balance(realized_pnl)
                self.state_db.add_session_pnl(realized_pnl)
                self.state_db.add_daily_pnl(realized_pnl)

                # Write settlement to ledger
                self.ledger.write_settlement(
                    market_id=pos.market_id,
                    asset=pos.asset,
                    side=pos.side,
                    size=pos.size,
                    entry_price=entry_price,
                    settlement_price=settled_price,
                )

                # Remove position
                self.state_db.remove_position(pos.market_id, pos.side)
                self.guard_wiring.update_position(pos.market_id, 0.0)

                # Record in guards
                self.guard_wiring.record_trade_result(realized_pnl > 0, realized_pnl)

                pnl += realized_pnl
                log.info("Settled: %s %s @ %.4f -> %.4f, PnL=%.2f", pos.side, pos.market_id, entry_price, settled_price, realized_pnl)

        return pnl


# ─── CLI / demo ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    cycle = PolymarketCycle(paper_mode=True)

    # Demo markets
    markets = [
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
    ]

    for i in range(5):
        result = cycle.run_cycle(markets)
        print(f"Cycle {result.cycle}: scanned={result.markets_scanned} signals={result.signals_generated} orders={result.orders_placed} filled={result.orders_filled} pnl={result.pnl:.2f} time={result.duration_ms:.1f}ms")
        if result.errors:
            print(f"  Errors: {result.errors}")
        time.sleep(1)

    print("\nDone!")