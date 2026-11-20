"""
polymarket_trader.mempalace_hooks — MemPalace integration hooks (T13-46).

Provides MemPalace integration for the Polymarket trader:
- Checkpoints
- Drawers (durable narrative)
- KG facts
- Status heartbeats
- Incident filing
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from mempalace_bridge import get_bridge, MemPalaceBridge

log = logging.getLogger("polymarket_trader.mempalace_hooks")


@dataclass
class TraderCheckpoint:
    """Trader checkpoint data."""
    cycle: int
    balance: float
    positions: int
    open_orders: int
    daily_pnl: float
    session_pnl: float
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycle": self.cycle,
            "balance": self.balance,
            "positions": self.positions,
            "open_orders": self.open_orders,
            "daily_pnl": self.daily_pnl,
            "session_pnl": self.session_pnl,
            "timestamp": self.timestamp,
        }


class PolymarketMemPalaceHooks:
    """MemPalace hooks for Polymarket trader."""

    def __init__(self, bridge: Optional[MemPalaceBridge] = None, agent_id: str = "ubuntu:opencode:home"):
        self.bridge = bridge or get_bridge()
        self.agent_id = agent_id
        self._cycle_count = 0

    def set_cycle(self, cycle: int):
        """Update current cycle count."""
        self._cycle_count = cycle

    # ─── Checkpoints ──────────────────────────────────────────────────────────

    def checkpoint(
        self,
        balance: float,
        positions: int,
        open_orders: int,
        daily_pnl: float,
        session_pnl: float,
        forecast_id: str = None,
        model_version: str = None,
    ) -> bool:
        """Save a trading checkpoint."""
        cp = TraderCheckpoint(
            cycle=self._cycle_count,
            balance=balance,
            positions=positions,
            open_orders=open_orders,
            daily_pnl=daily_pnl,
            session_pnl=session_pnl,
        )

        result = self.bridge.checkpoint(
            checkpoint_id=f"poly_cycle_{self._cycle_count}",
            payload=cp.to_dict(),
            forecast_id=forecast_id,
            model_version=model_version,
        )
        return result.get("success", False)

    # ─── Drawers ──────────────────────────────────────────────────────────────

    def log_halt(self, reason: str, details: Dict[str, Any] = None):
        """Log a trading halt."""
        content = f"HALT: {reason}\n"
        content += f"Cycle: {self._cycle_count}\n"
        content += f"Timestamp: {datetime.now(timezone.utc).isoformat()}\n"
        if details:
            content += f"Details: {json.dumps(details, indent=2)}\n"

        self.bridge.drawer_put(
            wing="polymarket_trader",
            room="halts",
            content=content,
            source_file="polymarket_trader",
            added_by=self.agent_id,
        )

    def log_parameter_change(self, param: str, old_value: Any, new_value: Any, rationale: str):
        """Log a parameter change."""
        content = f"PARAMETER CHANGE: {param}\n"
        content += f"Old: {old_value}\n"
        content += f"New: {new_value}\n"
        content += f"Rationale: {rationale}\n"
        content += f"Cycle: {self._cycle_count}\n"
        content += f"Timestamp: {datetime.now(timezone.utc).isoformat()}\n"

        self.bridge.drawer_put(
            wing="polymarket_trader",
            room="parameter_changes",
            content=content,
            source_file="polymarket_trader",
            added_by=self.agent_id,
        )

    def log_incident_postmortem(self, incident_id: str, description: str, root_cause: str, resolution: str):
        """Log an incident postmortem."""
        content = f"INCIDENT POSTMORTEM: {incident_id}\n"
        content += f"Description: {description}\n"
        content += f"Root Cause: {root_cause}\n"
        content += f"Resolution: {resolution}\n"
        content += f"Cycle: {self._cycle_count}\n"
        content += f"Timestamp: {datetime.now(timezone.utc).isoformat()}\n"

        self.bridge.drawer_put(
            wing="polymarket_trader",
            room="incidents",
            content=content,
            source_file="polymarket_trader",
            added_by=self.agent_id,
        )

    def log_model_card(self, model_name: str, card_data: Dict[str, Any]):
        """Log a model card."""
        content = f"MODEL CARD: {model_name}\n"
        content += json.dumps(card_data, indent=2)
        content += f"\nCycle: {self._cycle_count}\n"
        content += f"Timestamp: {datetime.now(timezone.utc).isoformat()}\n"

        self.bridge.drawer_put(
            wing="polymarket_trader",
            room="model_cards",
            content=content,
            source_file="polymarket_trader",
            added_by=self.agent_id,
        )

    def log_nightly_forecast(self, forecast_data: Dict[str, Any]):
        """Log a nightly forecast report."""
        content = f"NIGHTLY FORECAST REPORT\n"
        content += f"Cycle: {self._cycle_count}\n"
        content += f"Timestamp: {datetime.now(timezone.utc).isoformat()}\n"
        content += json.dumps(forecast_data, indent=2)

        self.bridge.drawer_put(
            wing="polymarket_trader",
            room="forecasts",
            content=content,
            source_file="polymarket_trader",
            added_by=self.agent_id,
        )

    # ─── KG Facts ─────────────────────────────────────────────────────────────

    def record_win_rate(self, asset: str, entry_band: str, win_rate: float, sample_size: int):
        """Record measured win rate per band and asset."""
        key = f"win_rate_{asset}_{entry_band}"
        value = json.dumps({
            "asset": asset,
            "entry_band": entry_band,
            "win_rate": win_rate,
            "sample_size": sample_size,
            "cycle": self._cycle_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self.bridge.kg_add(key, value)

    def record_basis(self, asset: str, basis_bps: float, venue: str = "polymarket"):
        """Record composite-vs-settlement basis."""
        key = f"basis_{venue}_{asset}"
        value = json.dumps({
            "venue": venue,
            "asset": asset,
            "basis_bps": basis_bps,
            "cycle": self._cycle_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self.bridge.kg_add(key, value)

    def record_latency(self, stage: str, latency_p99_ms: float):
        """Record latency p99 by stage."""
        key = f"latency_p99_{stage}"
        value = json.dumps({
            "stage": stage,
            "latency_p99_ms": latency_p99_ms,
            "cycle": self._cycle_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self.bridge.kg_add(key, value)

    def record_model_weights(self, weights: Dict[str, float]):
        """Record ensemble model weights."""
        key = "model_weights"
        value = json.dumps({
            "weights": weights,
            "cycle": self._cycle_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self.bridge.kg_add(key, value)

    def record_sync_lag(self, lag_seconds: float):
        """Record sync lag."""
        key = "sync_lag"
        value = json.dumps({
            "lag_seconds": lag_seconds,
            "cycle": self._cycle_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self.bridge.kg_add(key, value)

    def record_forecast_calibration(self, forecast_id: str, predicted: float, actual: float):
        """Record forecast calibration score."""
        key = f"forecast_cal_{forecast_id}"
        value = json.dumps({
            "forecast_id": forecast_id,
            "predicted": predicted,
            "actual": actual,
            "error": abs(predicted - actual),
            "cycle": self._cycle_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self.bridge.kg_add(key, value)

    # ─── Heartbeats ───────────────────────────────────────────────────────────

    def heartbeat(self, task: str = "trading", metadata: Dict[str, Any] = None):
        """Send a status heartbeat."""
        meta = metadata or {}
        meta.update({"cycle": self._cycle_count, "agent_id": self.agent_id})
        self.bridge.heartbeat(
            role="poly_trader",
            task=task,
            cycles=self._cycle_count,
            metadata=meta,
        )

    # ─── Incidents ────────────────────────────────────────────────────────────

    def file_incident(self, incident_id: str, description: str, severity: str = "medium") -> bool:
        """File an incident."""
        result = self.bridge.file_incident(
            incident_id=incident_id,
            description=description,
            severity=severity,
            metadata={"agent": self.agent_id, "cycle": self._cycle_count},
        )
        return result.get("success", False)

    # ─── Search ───────────────────────────────────────────────────────────────

    def search(self, query: str) -> Dict[str, Any]:
        """Search MemPalace."""
        return self.bridge.search(query)


# ─── Global instance ──────────────────────────────────────────────────────────

_hooks: Optional[PolymarketMemPalaceHooks] = None


def get_hooks(bridge: Optional[MemPalaceBridge] = None, agent_id: str = "ubuntu:opencode:home") -> PolymarketMemPalaceHooks:
    """Get the global MemPalace hooks instance."""
    global _hooks
    if _hooks is None:
        _hooks = PolymarketMemPalaceHooks(bridge, agent_id)
    return _hooks


# ─── CLI / demo ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    hooks = get_hooks()

    # Demo
    hooks.set_cycle(42)
    hooks.checkpoint(1000.0, 3, 2, 15.5, 142.3)
    hooks.log_parameter_change("ENTRY_BAND_MIN", 15, 20, "Widen band after backtest")
    hooks.record_win_rate("BTC", "15-55c", 0.528, 246)
    hooks.heartbeat("trading", {"balance": 1000.0})
    hooks.file_incident("INC_001", "API timeout on order placement", "high")

    print("MemPalace hooks demo complete")