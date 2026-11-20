"""
polymarket_trader.state_db — State management and database (T13-46).

Handles all persistent state for the Polymarket trader:
- Balance tracking
- Position management
- Order history
- Daily P&L
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from local_db import get_local_db

log = logging.getLogger("polymarket_trader.state_db")


@dataclass
class Position:
    """Polymarket position."""
    market_id: str
    asset: str
    side: str  # "YES" or "NO"
    size: float
    entry_price: float
    entry_time: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "market_id": self.market_id,
            "asset": self.asset,
            "side": self.side,
            "size": self.size,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time,
            "current_price": self.current_price,
            "unrealized_pnl": self.unrealized_pnl,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Position":
        return cls(**data)


@dataclass
class Order:
    """Polymarket order."""
    order_id: str
    market_id: str
    side: str
    size: float
    price: float
    status: str  # "open", "filled", "cancelled", "expired"
    created_at: float
    filled_at: float = 0.0
    filled_size: float = 0.0
    filled_price: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "order_id": self.order_id,
            "market_id": self.market_id,
            "side": self.side,
            "size": self.size,
            "price": self.price,
            "status": self.status,
            "created_at": self.created_at,
            "filled_at": self.filled_at,
            "filled_size": self.filled_size,
            "filled_price": self.filled_price,
        }


class PolymarketStateDB:
    """State database for Polymarket trader."""

    def __init__(self, prefix: str = "polymarket"):
        self.prefix = prefix
        self.db = get_local_db()

    # ─── Balance ─────────────────────────────────────────────────────────────

    def get_balance(self) -> float:
        """Get current balance from local DB."""
        return self.db.get_bot_state(f"{self.prefix}_balance") or 0.0

    def set_balance(self, balance: float):
        """Set current balance in local DB."""
        self.db.set_bot_state(f"{self.prefix}_balance", balance)

    def update_balance(self, delta: float) -> float:
        """Update balance by delta and return new balance."""
        current = self.get_balance()
        new_balance = current + delta
        self.set_balance(new_balance)
        return new_balance

    # ─── Positions ───────────────────────────────────────────────────────────

    def get_positions(self) -> List[Position]:
        """Get all open positions."""
        data = self.db.get_bot_state(f"{self.prefix}_positions") or []
        return [Position.from_dict(p) for p in data]

    def save_positions(self, positions: List[Position]):
        """Save positions to local DB."""
        self.db.set_bot_state(
            f"{self.prefix}_positions",
            [p.to_dict() for p in positions]
        )

    def add_position(self, position: Position):
        """Add a new position."""
        positions = self.get_positions()
        positions.append(position)
        self.save_positions(positions)

    def remove_position(self, market_id: str, side: str) -> Optional[Position]:
        """Remove and return a position."""
        positions = self.get_positions()
        for i, p in enumerate(positions):
            if p.market_id == market_id and p.side == side:
                positions.pop(i)
                self.save_positions(positions)
                return p
        return None

    def update_position_price(self, market_id: str, side: str, price: float):
        """Update position's current price and unrealized P&L."""
        positions = self.get_positions()
        for p in positions:
            if p.market_id == market_id and p.side == side:
                p.current_price = price
                if p.side == "YES":
                    p.unrealized_pnl = (price - p.entry_price) * p.size
                else:
                    p.unrealized_pnl = (p.entry_price - price) * p.size
        self.save_positions(positions)

    # ─── Orders ──────────────────────────────────────────────────────────────

    def get_orders(self) -> List[Order]:
        """Get all orders."""
        data = self.db.get_bot_state(f"{self.prefix}_orders") or []
        return [Order.from_dict(o) for o in data]

    def save_orders(self, orders: List[Order]):
        """Save orders to local DB."""
        self.db.set_bot_state(
            f"{self.prefix}_orders",
            [o.to_dict() for o in orders]
        )

    def add_order(self, order: Order):
        """Add a new order."""
        orders = self.get_orders()
        orders.append(order)
        self.save_orders(orders)

    def update_order(self, order_id: str, **kwargs) -> Optional[Order]:
        """Update an order's fields."""
        orders = self.get_orders()
        for o in orders:
            if o.order_id == order_id:
                for k, v in kwargs.items():
                    if hasattr(o, k):
                        setattr(o, k, v)
                self.save_orders(orders)
                return o
        return None

    def get_open_orders(self) -> List[Order]:
        """Get all open orders."""
        return [o for o in self.get_orders() if o.status == "open"]

    # ─── Daily P&L ───────────────────────────────────────────────────────────

    def get_daily_pnl(self, date_str: str = None) -> float:
        """Get daily P&L for a date (defaults to today)."""
        if date_str is None:
            date_str = time.strftime("%Y-%m-%d")
        return self.db.get_bot_state(f"{self.prefix}_daily_pnl_{date_str}") or 0.0

    def add_daily_pnl(self, pnl: float, date_str: str = None) -> float:
        """Add to daily P&L and return new total."""
        if date_str is None:
            date_str = time.strftime("%Y-%m-%d")
        current = self.get_daily_pnl(date_str)
        new_total = current + pnl
        self.db.set_bot_state(f"{self.prefix}_daily_pnl_{date_str}", new_total)
        return new_total

    # ─── Session P&L ─────────────────────────────────────────────────────────

    def get_session_pnl(self) -> float:
        """Get total session P&L."""
        return self.db.get_bot_state(f"{self.prefix}_session_pnl") or 0.0

    def add_session_pnl(self, pnl: float) -> float:
        """Add to session P&L and return new total."""
        current = self.get_session_pnl()
        new_total = current + pnl
        self.db.set_bot_state(f"{self.prefix}_session_pnl", new_total)
        return new_total

    # ─── Metadata ────────────────────────────────────────────────────────────

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value."""
        return self.db.get_bot_state(f"{self.prefix}_{key}", default)

    def set_metadata(self, key: str, value: Any):
        """Set metadata value."""
        self.db.set_bot_state(f"{self.prefix}_{key}", value)

    # ─── Health Check ────────────────────────────────────────────────────────

    def health_check(self) -> bool:
        """Check if DB is accessible."""
        try:
            self.db.health_check()
            return True
        except Exception as e:
            log.error("State DB health check failed: %s", e)
            return False