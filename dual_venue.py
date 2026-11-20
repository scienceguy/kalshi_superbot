"""
dual_venue.py — Dual-venue orchestration for Kalshi and Polymarket US (T13-34).

Runs Kalshi and Polymarket as independent traders sharing:
- Composite spot price (Binance + Coinbase WS)
- Model service (GBM, ML ensemble, forecaster)
- Risk officer (cash floor, exposure ceiling, drawdown halt)
- Combined dashboard data

Neither venue's failure halts the other unless the risk officer says so.
Accept: killing one venue's process leaves the other trading.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Callable, Any
from enum import Enum

from spot_composite import get_spot_composite
from forecaster import create_forecaster, EnsembleForecaster
from risk import RiskManager
from sizing import size_for_entry
from local_db import get_local_db

log = logging.getLogger("dual_venue")


class Venue(Enum):
    KALSHI = "kalshi"
    POLYMARKET = "polymarket"


@dataclass
class VenueState:
    """Independent state per venue."""
    venue: Venue
    balance: float = 0.0
    exposure: float = 0.0
    positions: Dict[str, Any] = field(default_factory=dict)
    orders: Dict[str, Any] = field(default_factory=dict)
    daily_pnl: float = 0.0
    total_pnl: float = 0.0
    halted: bool = False
    halt_reason: str = ""
    last_update: float = 0.0
    error_count: int = 0
    last_error: Optional[str] = None


@dataclass
class DualVenueConfig:
    """Configuration for dual-venue orchestration."""
    # Shared services
    spot_composite_enabled: bool = True
    model_service_enabled: bool = True
    risk_officer_enabled: bool = True

    # Per-venue settings
    kalshi_enabled: bool = True
    polymarket_enabled: bool = True

    # Risk limits (applied per venue)
    cash_floor_pct: float = 0.70
    exposure_ceiling_pct: float = 0.30
    per_trade_pct: float = 0.08
    max_drawdown_pct: float = 0.15
    max_consecutive_losses: int = 3

    # Health monitoring
    max_errors_before_halt: int = 10
    health_check_interval: float = 60.0  # seconds
    restart_on_halt: bool = False


class SharedServices:
    """Shared services across venues."""

    def __init__(self, config: DualVenueConfig):
        self.config = config
        self._spot_composite = None
        self._forecaster = None
        self._risk_manager = RiskManager()
        self._lock = threading.Lock()
        self._initialized = False

    def initialize(self, starting_balance: float):
        """Initialize shared services."""
        with self._lock:
            if self._initialized:
                return

            if self.config.spot_composite_enabled:
                try:
                    self._spot_composite = get_spot_composite()
                    log.info("Shared spot composite initialized")
                except Exception as e:
                    log.warning("Failed to initialize spot composite: %s", e)

            if self.config.model_service_enabled:
                try:
                    self._forecaster = create_forecaster()
                    log.info("Shared model service (forecaster) initialized")
                except Exception as e:
                    log.warning("Failed to initialize forecaster: %s", e)

            if self.config.risk_officer_enabled:
                self._risk_manager.initialize(starting_balance)
                log.info("Shared risk officer initialized with balance $%.2f", starting_balance)

            self._initialized = True

    def get_spot_price(self, asset: str) -> Optional[float]:
        """Get composite spot price for an asset."""
        if not self.config.spot_composite_enabled or self._spot_composite is None:
            return None
        try:
            result = self._spot_composite.get_spot_with_age(asset)
            if result and result.get("composite") is not None:
                return result["composite"]
        except Exception as e:
            log.warning("Spot composite error for %s: %s", asset, e)
        return None

    def get_forecaster(self) -> Optional[EnsembleForecaster]:
        """Get the shared forecaster."""
        return self._forecaster if self.config.model_service_enabled else None

    def get_risk_manager(self) -> RiskManager:
        """Get the shared risk manager."""
        return self._risk_manager

    def check_global_risk(self, venue_states: Dict[Venue, VenueState]) -> tuple[bool, str]:
        """Check global risk across all venues."""
        # Aggregate exposure across venues
        total_exposure = sum(vs.exposure for vs in venue_states.values())
        total_balance = sum(vs.balance for vs in venue_states.values())
        total_value = total_balance + total_exposure

        if total_value <= 0:
            return False, "Total portfolio value <= 0"

        # Check aggregate exposure ceiling
        exposure_pct = total_exposure / total_value
        if exposure_pct > self.config.exposure_ceiling_pct:
            return False, f"Aggregate exposure {exposure_pct:.1%} exceeds ceiling {self.config.exposure_ceiling_pct:.1%}"

        # Check aggregate cash floor
        cash_pct = total_balance / total_value
        if cash_pct < self.config.cash_floor_pct:
            return False, f"Aggregate cash {cash_pct:.1%} below floor {self.config.cash_floor_pct:.1%}"

        return True, "OK"


class DualVenueOrchestrator:
    """
    Orchestrates independent Kalshi and Polymarket traders.
    Each venue runs in its own thread with shared services.
    """

    def __init__(
        self,
        config: DualVenueConfig,
        kalshi_trader_fn: Optional[Callable] = None,
        polymarket_trader_fn: Optional[Callable] = None,
    ):
        self.config = config
        self.kalshi_trader_fn = kalshi_trader_fn
        self.polymarket_trader_fn = polymarket_trader_fn

        self.shared = SharedServices(config)
        self.venue_states: Dict[Venue, VenueState] = {
            Venue.KALSHI: VenueState(venue=Venue.KALSHI),
            Venue.POLYMARKET: VenueState(venue=Venue.POLYMARKET),
        }
        self._threads: Dict[Venue, threading.Thread] = {}
        self._stop_event = threading.Event()
        self._health_thread: Optional[threading.Thread] = None
        self._initialized = False

    def initialize(self, kalshi_balance: float = 0.0, polymarket_balance: float = 0.0):
        """Initialize the orchestrator with starting balances."""
        starting_balance = kalshi_balance + polymarket_balance
        self.shared.initialize(starting_balance)

        self.venue_states[Venue.KALSHI].balance = kalshi_balance
        self.venue_states[Venue.POLYMARKET].balance = polymarket_balance

        # Persist initial state to local DB
        self._persist_initial_state(kalshi_balance, polymarket_balance)

        self._initialized = True
        log.info(
            "DualVenueOrchestrator initialized: Kalshi=$%.2f, Polymarket=$%.2f",
            kalshi_balance, polymarket_balance
        )

    def _persist_initial_state(self, kalshi_balance: float, polymarket_balance: float):
        """Persist initial balances to local DB."""
        try:
            db = get_local_db()
            db.set_bot_state("dual_venue_kalshi_balance", kalshi_balance)
            db.set_bot_state("dual_venue_polymarket_balance", polymarket_balance)
            db.set_bot_state("dual_venue_total_balance", kalshi_balance + polymarket_balance)
            db.set_bot_state("dual_venue_start_ts", time.time())
            
            # T13-32: Record Day-0 balance snapshots for forecast reference
            import hashlib
            config_str = f"{kalshi_balance}{polymarket_balance}{time.time()}"
            config_hash = hashlib.md5(config_str.encode()).hexdigest()[:16]
            if kalshi_balance > 0:
                db.record_day0_balance("kalshi", kalshi_balance, config_hash)
            if polymarket_balance > 0:
                db.record_day0_balance("polymarket", polymarket_balance, config_hash)
        except Exception as e:
            log.warning("Failed to persist initial dual venue state: %s", e)

    def start(self):
        """Start both venue traders in separate threads."""
        if not self._initialized:
            raise RuntimeError("Orchestrator not initialized. Call initialize() first.")

        if self.config.kalshi_enabled and self.kalshi_trader_fn:
            self._start_venue(Venue.KALSHI, self.kalshi_trader_fn)

        if self.config.polymarket_enabled and self.polymarket_trader_fn:
            self._start_venue(Venue.POLYMARKET, self.polymarket_trader_fn)

        # Start health monitoring
        self._health_thread = threading.Thread(target=self._health_monitor, daemon=True)
        self._health_thread.start()

        log.info("DualVenueOrchestrator started with %d venue(s)", len(self._threads))

    def _start_venue(self, venue: Venue, trader_fn: Callable):
        """Start a venue trader in a separate thread."""
        def venue_runner():
            log.info("Starting %s trader thread", venue.value)
            try:
                # Pass venue state and shared services to trader
                trader_fn(
                    venue_state=self.venue_states[venue],
                    shared=self.shared,
                    stop_event=self._stop_event,
                )
            except Exception as e:
                log.error("%s trader thread crashed: %s", venue.value, e)
                self.venue_states[venue].halted = True
                self.venue_states[venue].halt_reason = f"Thread crash: {e}"
                self.venue_states[venue].error_count += 1
                self.venue_states[venue].last_error = str(e)

        thread = threading.Thread(target=venue_runner, name=f"trader-{venue.value}", daemon=True)
        thread.start()
        self._threads[venue] = thread

    def _health_monitor(self):
        """Monitor venue health and enforce global risk limits."""
        while not self._stop_event.is_set():
            time.sleep(self.config.health_check_interval)

            if self._stop_event.is_set():
                break

            # Check each venue's error count
            for venue, state in self.venue_states.items():
                if state.error_count >= self.config.max_errors_before_halt:
                    if not state.halted:
                        state.halted = True
                        state.halt_reason = f"Max errors ({state.error_count}) reached"
                        log.warning("%s halted: %s", venue.value, state.halt_reason)

            # Check global risk if risk officer enabled
            if self.config.risk_officer_enabled:
                ok, reason = self.shared.check_global_risk(self.venue_states)
                if not ok:
                    log.warning("Global risk check failed: %s", reason)
                    # Halt both venues on global risk breach
                    for vs in self.venue_states.values():
                        if not vs.halted:
                            vs.halted = True
                            vs.halt_reason = f"Global risk: {reason}"

            # Update dashboard data
            self._update_dashboard()

    def _update_dashboard(self):
        """Update combined dashboard data in local DB."""
        try:
            db = get_local_db()
            total_balance = sum(vs.balance for vs in self.venue_states.values())
            total_exposure = sum(vs.exposure for vs in self.venue_states.values())
            total_pnl = sum(vs.total_pnl for vs in self.venue_states.values())
            total_value = total_balance + total_exposure

            dashboard = {
                "timestamp": time.time(),
                "kalshi_balance": self.venue_states[Venue.KALSHI].balance,
                "kalshi_exposure": self.venue_states[Venue.KALSHI].exposure,
                "kalshi_pnl": self.venue_states[Venue.KALSHI].total_pnl,
                "kalshi_halted": self.venue_states[Venue.KALSHI].halted,
                "polymarket_balance": self.venue_states[Venue.POLYMARKET].balance,
                "polymarket_exposure": self.venue_states[Venue.POLYMARKET].exposure,
                "polymarket_pnl": self.venue_states[Venue.POLYMARKET].total_pnl,
                "polymarket_halted": self.venue_states[Venue.POLYMARKET].halted,
                "total_balance": total_balance,
                "total_exposure": total_exposure,
                "total_pnl": total_pnl,
                "total_value": total_value,
                "global_halted": any(vs.halted for vs in self.venue_states.values()),
            }
            db.set_bot_state("dual_venue_dashboard", dashboard)
        except Exception as e:
            log.warning("Failed to update dashboard: %s", e)

    def stop(self):
        """Stop all venue traders and health monitor."""
        log.info("Stopping DualVenueOrchestrator...")
        self._stop_event.set()

        for venue, thread in self._threads.items():
            thread.join(timeout=10.0)
            log.info("%s trader thread stopped", venue.value)

        if self._health_thread:
            self._health_thread.join(timeout=5.0)

        log.info("DualVenueOrchestrator stopped")

    def get_venue_state(self, venue: Venue) -> VenueState:
        """Get current state for a venue."""
        return self.venue_states[venue]

    def is_venue_running(self, venue: Venue) -> bool:
        """Check if a venue's thread is still running."""
        thread = self._threads.get(venue)
        return thread is not None and thread.is_alive()

    def halt_venue(self, venue: Venue, reason: str):
        """Manually halt a venue."""
        self.venue_states[venue].halted = True
        self.venue_states[venue].halt_reason = reason
        log.warning("%s manually halted: %s", venue.value, reason)

    def get_combined_stats(self) -> Dict[str, Any]:
        """Get combined statistics for dashboard."""
        return {
            "kalshi": {
                "balance": self.venue_states[Venue.KALSHI].balance,
                "exposure": self.venue_states[Venue.KALSHI].exposure,
                "pnl": self.venue_states[Venue.KALSHI].total_pnl,
                "halted": self.venue_states[Venue.KALSHI].halted,
                "running": self.is_venue_running(Venue.KALSHI),
            },
            "polymarket": {
                "balance": self.venue_states[Venue.POLYMARKET].balance,
                "exposure": self.venue_states[Venue.POLYMARKET].exposure,
                "pnl": self.venue_states[Venue.POLYMARKET].total_pnl,
                "halted": self.venue_states[Venue.POLYMARKET].halted,
                "running": self.is_venue_running(Venue.POLYMARKET),
            },
            "combined": {
                "total_balance": sum(vs.balance for vs in self.venue_states.values()),
                "total_exposure": sum(vs.exposure for vs in self.venue_states.values()),
                "total_pnl": sum(vs.total_pnl for vs in self.venue_states.values()),
                "all_running": all(self.is_venue_running(v) for v in self.venue_states),
            },
        }


# ─── Factory functions for easy integration ────────────────────────────────────

def create_orchestrator(
    kalshi_balance: float = 0.0,
    polymarket_balance: float = 0.0,
    kalshi_trader_fn: Optional[Callable] = None,
    polymarket_trader_fn: Optional[Callable] = None,
    config: Optional[DualVenueConfig] = None,
) -> DualVenueOrchestrator:
    """Create and initialize a dual-venue orchestrator."""
    if config is None:
        config = DualVenueConfig()

    orchestrator = DualVenueOrchestrator(
        config=config,
        kalshi_trader_fn=kalshi_trader_fn,
        polymarket_trader_fn=polymarket_trader_fn,
    )
    orchestrator.initialize(kalshi_balance, polymarket_balance)
    return orchestrator


# ─── CLI / demo ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Demo: show how to use the orchestrator
    def demo_kalshi_trader(venue_state, shared, stop_event):
        """Demo Kalshi trader function."""
        log.info("Demo Kalshi trader started")
        while not stop_event.is_set() and not venue_state.halted:
            time.sleep(5)
            # Simulate trading
            venue_state.balance += 0.1
            venue_state.total_pnl += 0.1

    def demo_polymarket_trader(venue_state, shared, stop_event):
        """Demo Polymarket trader function."""
        log.info("Demo Polymarket trader started")
        while not stop_event.is_set() and not venue_state.halted:
            time.sleep(5)
            # Simulate trading
            venue_state.balance += 0.15
            venue_state.total_pnl += 0.15

    logging.basicConfig(level=logging.INFO)
    config = DualVenueConfig(
        kalshi_enabled=True,
        polymarket_enabled=True,
        health_check_interval=10.0,
    )
    orch = create_orchestrator(
        kalshi_balance=1000.0,
        polymarket_balance=500.0,
        kalshi_trader_fn=demo_kalshi_trader,
        polymarket_trader_fn=demo_polymarket_trader,
        config=config,
    )
    orch.start()
    try:
        time.sleep(30)
    except KeyboardInterrupt:
        pass
    finally:
        orch.stop()
        stats = orch.get_combined_stats()
        print(f"Final stats: {stats}")