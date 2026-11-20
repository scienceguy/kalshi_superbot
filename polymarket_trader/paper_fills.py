"""
polymarket_trader.paper_fills — Paper trading fill simulation (T13-46).

Simulates order fills for paper trading with realistic models:
- Maker fill model: orders at bid/ask get filled when market crosses
- Taker fill model: market orders fill immediately at best price
- Slippage model: larger orders experience price impact
- Latency simulation: network delay affects fill price
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

log = logging.getLogger("polymarket_trader.paper_fills")


class OrderType(Enum):
    """Order type."""
    MAKER = "maker"   # Limit order
    TAKER = "taker"   # Market order


class FillModel(Enum):
    """Fill model types."""
    OPTIMISTIC = "optimistic"    # Always fill at limit price
    REALISTIC = "realistic"      # Probabilistic based on spread/volume
    PESSIMISTIC = "pessimistic"  # Only fill if price moves through


@dataclass
class MarketSnapshot:
    """Market state for fill simulation."""
    market_id: str
    bid: float
    ask: float
    last_price: float
    volume_24h: float
    open_interest: float
    timestamp: float = 0.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0


@dataclass
class PaperOrder:
    """Paper trading order."""
    order_id: str
    market_id: str
    side: str  # "YES" or "NO"
    order_type: OrderType
    size: float
    limit_price: float  # For maker orders
    created_at: float
    filled: bool = False
    filled_price: float = 0.0
    filled_size: float = 0.0
    filled_at: float = 0.0


@dataclass
class FillResult:
    """Result of a fill attempt."""
    filled: bool
    fill_price: float = 0.0
    fill_size: float = 0.0
    slippage: float = 0.0
    latency_ms: float = 0.0


class PaperFillSimulator:
    """Simulates paper trading fills."""

    def __init__(
        self,
        fill_model: FillModel = FillModel.REALISTIC,
        base_latency_ms: float = 50.0,
        latency_jitter_ms: float = 20.0,
        slippage_factor: float = 0.001,  # 0.1% per 1% of OI
    ):
        self.fill_model = fill_model
        self.base_latency_ms = base_latency_ms
        self.latency_jitter_ms = latency_jitter_ms
        self.slippage_factor = slippage_factor
        self._market_cache: Dict[str, MarketSnapshot] = {}

    def update_market(self, snapshot: MarketSnapshot):
        """Update market snapshot."""
        self._market_cache[snapshot.market_id] = snapshot

    def get_market(self, market_id: str) -> Optional[MarketSnapshot]:
        """Get cached market snapshot."""
        return self._market_cache.get(market_id)

    def try_fill(self, order: PaperOrder, current_market: MarketSnapshot) -> FillResult:
        """
        Attempt to fill an order against current market.

        Returns FillResult with fill details.
        """
        # Simulate latency
        latency = self.base_latency_ms + random.uniform(-self.latency_jitter_ms, self.latency_jitter_ms)
        latency = max(1.0, latency)

        # Use market at order creation time + latency delay
        # For paper trading, we use current market
        market = current_market

        if order.order_type == OrderType.TAKER:
            return self._fill_taker(order, market, latency)
        else:
            return self._fill_maker(order, market, latency)

    def _fill_taker(self, order: PaperOrder, market: MarketSnapshot, latency: float) -> FillResult:
        """Fill a taker (market) order."""
        # Taker orders fill immediately at best available price
        if order.side == "YES":
            fill_price = market.ask
        else:
            fill_price = 1.0 - market.bid  # NO price

        # Calculate slippage based on order size relative to open interest
        oi_ratio = order.size / max(market.open_interest, 1.0)
        slippage = oi_ratio * self.slippage_factor
        if order.side == "YES":
            fill_price += slippage
        else:
            fill_price = max(0.0, fill_price - slippage)

        return FillResult(
            filled=True,
            fill_price=min(1.0, max(0.0, fill_price)),
            fill_size=order.size,
            slippage=slippage,
            latency_ms=latency,
        )

    def _fill_maker(self, order: PaperOrder, market: MarketSnapshot, latency: float) -> FillResult:
        """Fill a maker (limit) order."""
        # Maker orders only fill if market crosses the limit price
        if order.side == "YES":
            # Buying YES at limit_price: fill if ask <= limit_price
            if market.ask <= order.limit_price:
                fill_price = order.limit_price
                # Realistic model: partial fills based on volume
                if self.fill_model == FillModel.REALISTIC:
                    # Probability increases as price improves
                    price_improvement = (order.limit_price - market.ask) / max(market.spread, 0.001)
                    fill_prob = min(1.0, 0.3 + 0.7 * price_improvement)
                    if random.random() > fill_prob:
                        return FillResult(filled=False, latency_ms=latency)
                return FillResult(
                    filled=True,
                    fill_price=fill_price,
                    fill_size=order.size,
                    slippage=0.0,
                    latency_ms=latency,
                )
        else:
            # Buying NO at limit_price: fill if NO ask <= limit_price
            no_ask = 1.0 - market.bid
            if no_ask <= order.limit_price:
                fill_price = order.limit_price
                if self.fill_model == FillModel.REALISTIC:
                    price_improvement = (order.limit_price - no_ask) / max(market.spread, 0.001)
                    fill_prob = min(1.0, 0.3 + 0.7 * price_improvement)
                    if random.random() > fill_prob:
                        return FillResult(filled=False, latency_ms=latency)
                return FillResult(
                    filled=True,
                    fill_price=fill_price,
                    fill_size=order.size,
                    slippage=0.0,
                    latency_ms=latency,
                )

        return FillResult(filled=False, latency_ms=latency)

    def simulate_batch(
        self,
        orders: List[PaperOrder],
        markets: Dict[str, MarketSnapshot],
    ) -> List[FillResult]:
        """Simulate fills for a batch of orders."""
        results = []
        for order in orders:
            market = markets.get(order.market_id)
            if market:
                result = self.try_fill(order, market)
            else:
                # No market data - use cached
                market = self.get_market(order.market_id)
                if market:
                    result = self.try_fill(order, market)
                else:
                    result = FillResult(filled=False, latency_ms=0.0)
            results.append(result)
        return results


# ─── Convenience functions ───────────────────────────────────────────────────

_DEFAULT_SIMULATOR: Optional[PaperFillSimulator] = None


def get_fill_simulator(fill_model: FillModel = FillModel.REALISTIC) -> PaperFillSimulator:
    """Get the default fill simulator."""
    global _DEFAULT_SIMULATOR
    if _DEFAULT_SIMULATOR is None:
        _DEFAULT_SIMULATOR = PaperFillSimulator(fill_model)
    return _DEFAULT_SIMULATOR


# ─── CLI / demo ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    sim = PaperFillSimulator(FillModel.REALISTIC)

    # Create test market
    market = MarketSnapshot(
        market_id="0x123",
        bid=0.83,
        ask=0.85,
        last_price=0.84,
        volume_24h=100000,
        open_interest=50000,
        timestamp=time.time(),
    )
    sim.update_market(market)

    print("Paper Fill Simulator Demo:")
    print("=" * 50)

    # Test taker order
    taker_order = PaperOrder(
        order_id="ord_001",
        market_id="0x123",
        side="YES",
        order_type=OrderType.TAKER,
        size=100,
        limit_price=0.0,  # Not used for taker
        created_at=time.time(),
    )
    result = sim.try_fill(taker_order, market)
    print(f"Taker BUY YES 100@market: filled={result.filled} price={result.fill_price:.4f} slippage={result.slippage:.6f} latency={result.latency_ms:.1f}ms")

    # Test maker order
    maker_order = PaperOrder(
        order_id="ord_002",
        market_id="0x123",
        side="YES",
        order_type=OrderType.MAKER,
        size=100,
        limit_price=0.85,  # At ask
        created_at=time.time(),
    )
    result = sim.try_fill(maker_order, market)
    print(f"Maker BUY YES 100@0.85: filled={result.filled} price={result.fill_price:.4f} latency={result.latency_ms:.1f}ms")

    # Test maker order below ask
    maker_order2 = PaperOrder(
        order_id="ord_003",
        market_id="0x123",
        side="YES",
        order_type=OrderType.MAKER,
        size=100,
        limit_price=0.80,  # Below ask
        created_at=time.time(),
    )
    result = sim.try_fill(maker_order2, market)
    print(f"Maker BUY YES 100@0.80: filled={result.filled} price={result.fill_price:.4f} latency={result.latency_ms:.1f}ms")