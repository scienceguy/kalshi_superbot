"""
polymarket_trader — Polymarket US trader split into modules (T13-46).

Modules:
- state_db: state management and database
- ledger_writer: ledger writing
- fees: fee calculations
- paper_fills: paper trading fill simulation
- mempalace_hooks: MemPalace integration
- guard_wiring: risk guards and circuit breakers
- cycle: main trading cycle
"""

from .main import PolymarketTrader, main

__all__ = ["PolymarketTrader", "main"]