"""
polymarket_trader.ledger_writer — Ledger writing (T13-46).

Handles all ledger operations for the Polymarket trader:
- Trade logging
- Settlement recording
- P&L tracking
- Audit trail
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from local_db import get_local_db

log = logging.getLogger("polymarket_trader.ledger_writer")


@dataclass
class LedgerEntry:
    """A single ledger entry."""
    id: str
    timestamp: float
    entry_type: str  # "trade", "settlement", "fee", "adjustment"
    market_id: str
    asset: str
    side: str
    size: float
    price: float
    pnl: float
    fee: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "entry_type": self.entry_type,
            "market_id": self.market_id,
            "asset": self.asset,
            "side": self.side,
            "size": self.size,
            "price": self.price,
            "pnl": self.pnl,
            "fee": self.fee,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LedgerEntry":
        return cls(**data)


class PolymarketLedgerWriter:
    """Ledger writer for Polymarket trader."""

    def __init__(self, prefix: str = "polymarket"):
        self.prefix = prefix
        self.db = get_local_db()

    def write_trade(
        self,
        market_id: str,
        asset: str,
        side: str,
        size: float,
        price: float,
        fee: float = 0.0,
        order_id: str = None,
        metadata: Dict[str, Any] = None,
    ) -> LedgerEntry:
        """Record a trade execution."""
        entry = LedgerEntry(
            id=f"trade_{int(time.time() * 1000)}_{order_id or 'auto'}",
            timestamp=time.time(),
            entry_type="trade",
            market_id=market_id,
            asset=asset,
            side=side,
            size=size,
            price=price,
            pnl=0.0,  # P&L realized at settlement
            fee=fee,
            metadata=metadata or {"order_id": order_id},
        )
        self._append(entry)
        log.info("Ledger: TRADE %s %s %s@%.4f fee=%.4f", side, asset, size, price, fee)
        return entry

    def write_settlement(
        self,
        market_id: str,
        asset: str,
        side: str,
        size: float,
        entry_price: float,
        settlement_price: float,
        order_id: str = None,
        metadata: Dict[str, Any] = None,
    ) -> LedgerEntry:
        """Record a settlement with realized P&L."""
        if side == "YES":
            pnl = (settlement_price - entry_price) * size
        else:
            pnl = (entry_price - settlement_price) * size

        entry = LedgerEntry(
            id=f"settle_{int(time.time() * 1000)}_{order_id or 'auto'}",
            timestamp=time.time(),
            entry_type="settlement",
            market_id=market_id,
            asset=asset,
            side=side,
            size=size,
            price=settlement_price,
            pnl=pnl,
            fee=0.0,
            metadata=metadata or {"order_id": order_id, "entry_price": entry_price},
        )
        self._append(entry)
        log.info("Ledger: SETTLE %s %s %s@%.4f entry=%.4f pnl=%.2f", side, asset, size, settlement_price, entry_price, pnl)
        return entry

    def write_fee(
        self,
        market_id: str,
        asset: str,
        fee: float,
        order_id: str = None,
        metadata: Dict[str, Any] = None,
    ) -> LedgerEntry:
        """Record a fee payment."""
        entry = LedgerEntry(
            id=f"fee_{int(time.time() * 1000)}_{order_id or 'auto'}",
            timestamp=time.time(),
            entry_type="fee",
            market_id=market_id,
            asset=asset,
            side="",
            size=0.0,
            price=0.0,
            pnl=-fee,
            fee=fee,
            metadata=metadata or {"order_id": order_id},
        )
        self._append(entry)
        log.info("Ledger: FEE %s %.4f", asset, fee)
        return entry

    def write_adjustment(
        self,
        amount: float,
        reason: str,
        metadata: Dict[str, Any] = None,
    ) -> LedgerEntry:
        """Record a manual balance adjustment."""
        entry = LedgerEntry(
            id=f"adj_{int(time.time() * 1000)}",
            timestamp=time.time(),
            entry_type="adjustment",
            market_id="",
            asset="",
            side="",
            size=0.0,
            price=0.0,
            pnl=amount,
            fee=0.0,
            metadata=metadata or {"reason": reason},
        )
        self._append(entry)
        log.info("Ledger: ADJUST %.2f (%s)", amount, reason)
        return entry

    def _append(self, entry: LedgerEntry):
        """Append entry to ledger in local DB."""
        key = f"{self.prefix}_ledger"
        ledger = self.db.get_bot_state(key) or []
        ledger.append(entry.to_dict())
        # Keep only last 10000 entries
        if len(ledger) > 10000:
            ledger = ledger[-10000:]
        self.db.set_bot_state(key, ledger)

    def get_ledger(self, limit: int = 1000) -> List[LedgerEntry]:
        """Get recent ledger entries."""
        key = f"{self.prefix}_ledger"
        data = self.db.get_bot_state(key) or []
        return [LedgerEntry.from_dict(e) for e in data[-limit:]]

    def get_total_pnl(self) -> float:
        """Get total realized P&L from ledger."""
        ledger = self.get_ledger(limit=10000)
        return sum(e.pnl for e in ledger)

    def get_pnl_by_asset(self) -> Dict[str, float]:
        """Get P&L grouped by asset."""
        ledger = self.get_ledger(limit=10000)
        pnl_by_asset: Dict[str, float] = {}
        for e in ledger:
            if e.asset:
                pnl_by_asset[e.asset] = pnl_by_asset.get(e.asset, 0.0) + e.pnl
        return pnl_by_asset

    def get_fees_paid(self) -> float:
        """Get total fees paid."""
        ledger = self.get_ledger(limit=10000)
        return sum(e.fee for e in ledger)

    def export_ledger(self, filepath: str):
        """Export ledger to JSON file."""
        ledger = self.get_ledger(limit=10000)
        with open(filepath, "w") as f:
            json.dump([e.to_dict() for e in ledger], f, indent=2)
        log.info("Ledger exported to %s (%d entries)", filepath, len(ledger))