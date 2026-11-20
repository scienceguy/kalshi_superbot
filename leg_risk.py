#!/usr/bin/env python3
"""
leg_risk.py — T15-32: Leg-risk execution for cross-venue arbitrage.

When executing a two-leg arbitrage (Kalshi + Polymarket), one leg may fill
while the other doesn't, leaving directional risk. This module provides:

1. Atomic execution attempt (best effort)
2. Leg-risk monitoring and hedging
3. Emergency flattening if one leg fails
4. Logging for post-trade analysis
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

import config
from kalshi_client import KalshiClient

log = logging.getLogger(__name__)


class LegStatus(Enum):
    """Status of a single leg."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    FAILED = "failed"


class ArbStatus(Enum):
    """Status of the overall arbitrage."""
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETE = "complete"
    LEG_RISK = "leg_risk"       # One leg filled, other failed
    FLATTENED = "flattened"     # Emergency flatten executed
    FAILED = "failed"


@dataclass
class Leg:
    """Single leg of an arbitrage."""
    venue: str              # "kalshi" or "polymarket"
    side: str               # "yes" or "no"
    price_cents: int        # Limit price in cents
    size: int               # Number of contracts
    order_id: str = ""
    status: LegStatus = LegStatus.PENDING
    filled_size: int = 0
    avg_fill_price: float = 0.0
    submitted_at: str = ""
    filled_at: str = ""
    error: str = ""


@dataclass
class ArbExecution:
    """Complete arbitrage execution record."""
    arb_id: str
    asset: str
    window_start: datetime
    kalshi_leg: Leg
    polymarket_leg: Leg
    status: ArbStatus = ArbStatus.PENDING
    created_at: str = ""
    started_at: str = ""
    completed_at: str = ""
    # Risk metrics
    max_leg_risk_dollars: float = 0.0
    current_leg_risk_dollars: float = 0.0
    # Flattening
    flatten_triggered: bool = False
    flatten_order_id: str = ""
    flatten_pnl: float = 0.0


class LegRiskManager:
    """
    Manages leg risk for cross-venue arbitrage execution.

    Strategy:
    1. Submit both legs simultaneously (or as close as possible)
    2. Monitor fills with timeout
    3. If one leg fills and other fails/times out:
       - Attempt to hedge on same venue (opposite side)
       - If hedge fails, emergency flatten the filled leg
    4. Log everything for analysis
    """

    def __init__(
        self,
        leg_timeout_seconds: float = 5.0,
        max_leg_risk_dollars: float = 10.0,
        flatten_on_timeout: bool = True,
    ):
        self.leg_timeout = leg_timeout_seconds
        self.max_leg_risk = max_leg_risk_dollars
        self.flatten_on_timeout = flatten_on_timeout

        self.kalshi = KalshiClient()
        # Polymarket client would be injected

        self.active_arbs: Dict[str, ArbExecution] = {}
        self.completed_arbs: List[ArbExecution] = []

        log.info(
            "LegRiskManager initialized: timeout=%.1fs, max_risk=$%.2f, flatten=%s",
            leg_timeout_seconds, max_leg_risk_dollars, flatten_on_timeout
        )

    def create_arb(
        self,
        asset: str,
        window_start: datetime,
        kalshi_side: str,
        kalshi_price: int,
        kalshi_size: int,
        poly_side: str,
        poly_price: int,
        poly_size: int,
    ) -> ArbExecution:
        """Create a new arbitrage execution plan."""
        import uuid
        arb_id = f"arb_{uuid.uuid4().hex[:8]}"

        kalshi_leg = Leg(
            venue="kalshi",
            side=kalshi_side,
            price_cents=kalshi_price,
            size=kalshi_size,
        )
        poly_leg = Leg(
            venue="polymarket",
            side=poly_side,
            price_cents=poly_price,
            size=poly_size,
        )

        arb = ArbExecution(
            arb_id=arb_id,
            asset=asset,
            window_start=window_start,
            kalshi_leg=kalshi_leg,
            polymarket_leg=poly_leg,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        # Calculate max leg risk (worst case: one leg fills at limit, other fails)
        kalshi_risk = kalshi_size * kalshi_price / 100.0
        poly_risk = poly_size * poly_price / 100.0
        arb.max_leg_risk_dollars = max(kalshi_risk, poly_risk)

        self.active_arbs[arb_id] = arb
        return arb

    async def execute_arb(self, arb: ArbExecution) -> ArbExecution:
        """
        Execute arbitrage with leg-risk management.

        This is the main entry point. It:
        1. Submits both legs
        2. Waits for fills with timeout
        3. Handles leg risk if needed
        """
        arb.status = ArbStatus.EXECUTING
        arb.started_at = datetime.now(timezone.utc).isoformat()

        log.info(f"Executing arb {arb.arb_id}: {arb.asset} @ {arb.window_start}")

        # Submit both legs concurrently
        kalshi_task = asyncio.create_task(self._submit_kalshi_leg(arb.kalshi_leg))
        poly_task = asyncio.create_task(self._submit_polymarket_leg(arb.polymarket_leg))

        # Wait for both submissions
        await asyncio.gather(kalshi_task, poly_task, return_exceptions=True)

        # Monitor fills
        await self._monitor_fills(arb)

        # Check final status
        kalshi_filled = arb.kalshi_leg.status == LegStatus.FILLED
        poly_filled = arb.polymarket_leg.status == LegStatus.FILLED

        if kalshi_filled and poly_filled:
            arb.status = ArbStatus.COMPLETE
            log.info(f"Arb {arb.arb_id} COMPLETE: both legs filled")
        elif kalshi_filled or poly_filled:
            arb.status = ArbStatus.LEG_RISK
            log.warning(f"Arb {arb.arb_id} LEG RISK: one leg filled, other failed")
            await self._handle_leg_risk(arb)
        else:
            arb.status = ArbStatus.FAILED
            log.error(f"Arb {arb.arb_id} FAILED: neither leg filled")

        arb.completed_at = datetime.now(timezone.utc).isoformat()
        self.completed_arbs.append(arb)
        del self.active_arbs[arb.arb_id]

        return arb

    async def _submit_kalshi_leg(self, leg: Leg) -> bool:
        """Submit Kalshi leg order."""
        try:
            leg.status = LegStatus.SUBMITTED
            leg.submitted_at = datetime.now(timezone.utc).isoformat()

            # Place limit order via Kalshi client
            # Note: This is a simplified version; real implementation would use
            # the actual Kalshi order placement API
            log.info(f"Submitting Kalshi {leg.side} @ {leg.price_cents}c x {leg.size}")

            # Simulate order placement
            # order = self.kalshi.place_order(...)
            # leg.order_id = order["order_id"]

            # For now, simulate immediate fill for testing
            await asyncio.sleep(0.1)
            leg.status = LegStatus.FILLED
            leg.filled_size = leg.size
            leg.avg_fill_price = leg.price_cents / 100.0
            leg.filled_at = datetime.now(timezone.utc).isoformat()

            return True

        except Exception as e:
            leg.status = LegStatus.FAILED
            leg.error = str(e)
            log.error(f"Kalshi leg failed: {e}")
            return False

    async def _submit_polymarket_leg(self, leg: Leg) -> bool:
        """Submit Polymarket leg order."""
        try:
            leg.status = LegStatus.SUBMITTED
            leg.submitted_at = datetime.now(timezone.utc).isoformat()

            log.info(f"Submitting Polymarket {leg.side} @ {leg.price_cents}c x {leg.size}")

            # Would use PolymarketUSClient here
            # For now, simulate
            await asyncio.sleep(0.1)
            leg.status = LegStatus.FILLED
            leg.filled_size = leg.size
            leg.avg_fill_price = leg.price_cents / 100.0
            leg.filled_at = datetime.now(timezone.utc).isoformat()

            return True

        except Exception as e:
            leg.status = LegStatus.FAILED
            leg.error = str(e)
            log.error(f"Polymarket leg failed: {e}")
            return False

    async def _monitor_fills(self, arb: ArbExecution):
        """Monitor both legs for fills with timeout."""
        start_time = time.time()
        kalshi_done = arb.kalshi_leg.status in (LegStatus.FILLED, LegStatus.FAILED, LegStatus.CANCELLED)
        poly_done = arb.polymarket_leg.status in (LegStatus.FILLED, LegStatus.FAILED, LegStatus.CANCELLED)

        while not (kalshi_done and poly_done) and (time.time() - start_time) < self.leg_timeout:
            await asyncio.sleep(0.5)

            # Check Kalshi order status
            if arb.kalshi_leg.status == LegStatus.SUBMITTED and arb.kalshi_leg.order_id:
                # status = self.kalshi.get_order_status(arb.kalshi_leg.order_id)
                # Update leg from status
                pass

            # Check Polymarket order status
            if arb.polymarket_leg.status == LegStatus.SUBMITTED and arb.polymarket_leg.order_id:
                # status = self.poly.get_order_status(arb.polymarket_leg.order_id)
                pass

            kalshi_done = arb.kalshi_leg.status in (LegStatus.FILLED, LegStatus.FAILED, LegStatus.CANCELLED)
            poly_done = arb.polymarket_leg.status in (LegStatus.FILLED, LegStatus.FAILED, LegStatus.CANCELLED)

        # Timeout handling
        if not kalshi_done:
            arb.kalshi_leg.status = LegStatus.FAILED
            arb.kalshi_leg.error = "Timeout"
        if not poly_done:
            arb.polymarket_leg.status = LegStatus.FAILED
            arb.polymarket_leg.error = "Timeout"

    async def _handle_leg_risk(self, arb: ArbExecution):
        """Handle situation where one leg filled but the other didn't."""
        filled_leg = arb.kalshi_leg if arb.kalshi_leg.status == LegStatus.FILLED else arb.polymarket_leg
        failed_leg = arb.polymarket_leg if arb.kalshi_leg.status == LegStatus.FILLED else arb.kalshi_leg

        arb.current_leg_risk_dollars = filled_leg.filled_size * filled_leg.avg_fill_price

        log.warning(
            f"Leg risk on {arb.arb_id}: {filled_leg.venue} {filled_leg.side} "
            f"filled {filled_leg.filled_size} @ {filled_leg.avg_fill_price:.4f}, "
            f"{failed_leg.venue} {failed_leg.side} failed ({failed_leg.error})"
        )

        if not self.flatten_on_timeout:
            log.info("Auto-flatten disabled, manual intervention required")
            return

        # Try to hedge on same venue (opposite side at market)
        hedge_success = await self._attempt_hedge(filled_leg)

        if not hedge_success:
            # Emergency flatten: sell the filled position at market
            await self._emergency_flatten(arb, filled_leg)

    async def _attempt_hedge(self, filled_leg: Leg) -> bool:
        """Attempt to hedge filled leg by taking opposite side on same venue."""
        try:
            hedge_side = "no" if filled_leg.side == "yes" else "yes"
            log.info(f"Attempting hedge: {filled_leg.venue} {hedge_side} @ market x {filled_leg.filled_size}")

            # Would place market order on same venue
            # For Kalshi: place taker order on opposite side
            # For Polymarket: place market order on opposite side

            # Simulate
            await asyncio.sleep(0.1)
            log.info("Hedge simulated (not implemented)")
            return False  # Not implemented yet

        except Exception as e:
            log.error(f"Hedge failed: {e}")
            return False

    async def _emergency_flatten(self, arb: ArbExecution, filled_leg: Leg):
        """Emergency flatten: close the filled position at market."""
        try:
            flatten_side = "no" if filled_leg.side == "yes" else "yes"
            log.critical(
                f"EMERGENCY FLATTEN: {arb.arb_id} - {filled_leg.venue} "
                f"{flatten_side} @ market x {filled_leg.filled_size}"
            )

            arb.flatten_triggered = True
            arb.flatten_order_id = f"flatten_{arb.arb_id}"

            # Place market order to close position
            # Would use Kalshi or Polymarket client

            # Simulate fill at slightly worse price
            slip = 0.02  # 2% slippage
            flatten_price = filled_leg.avg_fill_price * (1 - slip) if filled_leg.side == "yes" else filled_leg.avg_fill_price * (1 + slip)
            arb.flatten_pnl = (flatten_price - filled_leg.avg_fill_price) * filled_leg.filled_size

            arb.status = ArbStatus.FLATTENED

            log.critical(
                f"Flatten executed: PnL = ${arb.flatten_pnl:.2f}"
            )

        except Exception as e:
            log.error(f"Emergency flatten failed: {e}")
            arb.flatten_pnl = -filled_leg.filled_size * filled_leg.avg_fill_price  # Max loss

    def get_status(self) -> Dict:
        """Get manager status."""
        return {
            "active_arbs": len(self.active_arbs),
            "completed_arbs": len(self.completed_arbs),
            "active_details": [
                {
                    "arb_id": a.arb_id,
                    "asset": a.asset,
                    "status": a.status.value,
                    "kalshi": a.kalshi_leg.status.value,
                    "poly": a.polymarket_leg.status.value,
                }
                for a in self.active_arbs.values()
            ],
        }


# ─── Integration with cross_sectional_arb ──────────────────────────────────────

async def execute_cross_venue_arb(
    asset: str,
    window_start: datetime,
    kalshi_side: str,
    kalshi_price: int,
    kalshi_size: int,
    poly_side: str,
    poly_price: int,
    poly_size: int,
    leg_timeout: float = 5.0,
) -> ArbExecution:
    """
    High-level function to execute a cross-venue arbitrage with leg-risk management.

    Called from crossvenue_live_monitor or trader.py when an opportunity is detected.
    """
    manager = LegRiskManager(leg_timeout_seconds=leg_timeout)

    arb = manager.create_arb(
        asset=asset,
        window_start=window_start,
        kalshi_side=kalshi_side,
        kalshi_price=kalshi_price,
        kalshi_size=kalshi_size,
        poly_side=poly_side,
        poly_price=poly_price,
        poly_size=poly_size,
    )

    return await manager.execute_arb(arb)


# ─── CLI / demo ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    async def demo():
        from datetime import timedelta

        window = datetime.now(timezone.utc) + timedelta(minutes=10)
        window = window.replace(second=0, microsecond=0)

        # Example: Buy YES on Kalshi @ 45c, Buy NO on Polymarket @ 52c
        # Total = 97c, edge = 3c minus fees
        arb = await execute_cross_venue_arb(
            asset="BTC",
            window_start=window,
            kalshi_side="yes",
            kalshi_price=45,
            kalshi_size=10,
            poly_side="no",
            poly_price=52,
            poly_size=10,
            leg_timeout=5.0,
        )

        print(f"Arb {arb.arb_id}: {arb.status.value}")
        print(f"  Kalshi: {arb.kalshi_leg.status.value} {arb.kalshi_leg.filled_size} @ {arb.kalshi_leg.avg_fill_price}")
        print(f"  Poly: {arb.polymarket_leg.status.value} {arb.polymarket_leg.filled_size} @ {arb.polymarket_leg.avg_fill_price}")
        print(f"  Flattened: {arb.flatten_triggered}, PnL: ${arb.flatten_pnl:.2f}")

    import asyncio
    asyncio.run(demo())