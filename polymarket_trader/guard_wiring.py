"""
polymarket_trader.guard_wiring — Risk guards and circuit breakers (T13-46).

Provides risk management guards for the Polymarket trader:
- Cash floor / exposure ceiling
- Per-trade limits
- Drawdown halt
- Consecutive loss cooldown
- Circuit breakers for API failures
- Position limits
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

log = logging.getLogger("polymarket_trader.guard_wiring")


class HaltReason(Enum):
    """Reasons for trading halt."""
    CASH_FLOOR = "cash_floor"
    EXPOSURE_CEILING = "exposure_ceiling"
    DRAWDOWN = "drawdown"
    CONSECUTIVE_LOSSES = "consecutive_losses"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    POSITION_LIMIT = "position_limit"
    API_CIRCUIT_BREAKER = "api_circuit_breaker"
    MANUAL = "manual"


class GuardStatus(Enum):
    """Guard check status."""
    PASS = "pass"
    WARN = "warn"
    BLOCK = "block"
    HALT = "halt"


@dataclass
class GuardResult:
    """Result of a guard check."""
    status: GuardStatus
    reason: str = ""
    details: Dict = field(default_factory=dict)

    @property
    def blocks_trading(self) -> bool:
        return self.status in (GuardStatus.BLOCK, GuardStatus.HALT)


@dataclass
class RiskLimits:
    """Risk limits configuration."""
    # Portfolio limits
    cash_floor_pct: float = 0.70       # Minimum 70% cash
    exposure_ceiling_pct: float = 0.30  # Maximum 30% exposure
    per_trade_pct: float = 0.08        # Max 8% per trade
    max_drawdown_pct: float = 0.15     # Halt at 15% drawdown

    # Trade limits
    max_trade_usd: float = 100.0       # Max $100 per trade
    max_position_usd: float = 500.0    # Max $500 per position
    max_positions: int = 10            # Max concurrent positions

    # Loss limits
    max_daily_loss_pct: float = 0.05   # Halt at 5% daily loss
    max_consecutive_losses: int = 3    # Cooldown after 3 losses

    # Circuit breakers
    max_api_failures: int = 5          # Open breaker after 5 failures
    api_recovery_seconds: float = 60.0 # Try again after 60s

    def __post_init__(self):
        # Validate
        if self.cash_floor_pct + self.exposure_ceiling_pct > 1.0:
            log.warning("Cash floor + exposure ceiling > 100%%, adjusting")
            self.exposure_ceiling_pct = 1.0 - self.cash_floor_pct


class CircuitBreaker:
    """Circuit breaker for API failures."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.last_failure = 0.0
        self.state = "closed"  # closed, open, half_open
        self._lock = threading.Lock()

    def record_success(self):
        with self._lock:
            self.failures = 0
            self.state = "closed"

    def record_failure(self):
        with self._lock:
            self.failures += 1
            self.last_failure = time.time()
            if self.failures >= self.failure_threshold:
                self.state = "open"
                log.warning("Circuit breaker OPENED after %d failures", self.failures)

    def can_attempt(self) -> bool:
        with self._lock:
            if self.state == "closed":
                return True
            if self.state == "open":
                if time.time() - self.last_failure > self.recovery_timeout:
                    self.state = "half_open"
                    log.info("Circuit breaker HALF_OPEN")
                    return True
                return False
            # half_open
            return True

    def get_state(self) -> str:
        with self._lock:
            return self.state


class GuardWiring:
    """Wires together all risk guards for the Polymarket trader."""

    def __init__(self, limits: RiskLimits = None, starting_balance: float = 0.0):
        self.limits = limits or RiskLimits()
        self.starting_balance = starting_balance
        self.session_high = starting_balance

        # State
        self.consecutive_losses = 0
        self.daily_loss = 0.0
        self.daily_loss_date = time.strftime("%Y-%m-%d")
        self.halted = False
        self.halt_reason: Optional[HaltReason] = None
        self.position_values: Dict[str, float] = {}  # market_id -> value

        # Circuit breakers
        self.api_breaker = CircuitBreaker(
            failure_threshold=self.limits.max_api_failures,
            recovery_timeout=self.limits.api_recovery_seconds,
        )

        self._lock = threading.Lock()

    def update_balance(self, balance: float, exposure: float = 0.0):
        """Update current balance and exposure."""
        with self._lock:
            self.current_balance = balance
            self.current_exposure = exposure
            self.total_value = balance + exposure

            # Update session high
            if self.total_value > self.session_high:
                self.session_high = self.total_value

            # Check daily loss date
            today = time.strftime("%Y-%m-%d")
            if today != self.daily_loss_date:
                self.daily_loss = 0.0
                self.daily_loss_date = today

    def record_trade_result(self, won: bool, pnl: float):
        """Record a trade result for loss tracking."""
        with self._lock:
            if won:
                self.consecutive_losses = 0
            else:
                self.consecutive_losses += 1
                self.daily_loss += abs(pnl)
                log.info("Loss recorded: consecutive=%d, daily_loss=%.2f", self.consecutive_losses, self.daily_loss)

    def record_api_failure(self):
        """Record an API failure."""
        self.api_breaker.record_failure()

    def record_api_success(self):
        """Record an API success."""
        self.api_breaker.record_success()

    def check_pre_trade(
        self,
        trade_size_usd: float,
        market_id: str = "",
    ) -> GuardResult:
        """
        Check all pre-trade guards.

        Returns GuardResult indicating if trade should proceed.
        """
        with self._lock:
            if self.halted:
                return GuardResult(GuardStatus.HALT, f"Trading halted: {self.halt_reason.value}")

            # Check API circuit breaker
            if not self.api_breaker.can_attempt():
                return GuardResult(
                    GuardStatus.BLOCK,
                    "API circuit breaker open",
                    {"breaker_state": self.api_breaker.get_state()},
                )

            # Check cash floor
            if self.total_value > 0:
                cash_pct = self.current_balance / self.total_value
                if cash_pct < self.limits.cash_floor_pct:
                    return GuardResult(
                        GuardStatus.BLOCK,
                        f"Cash floor breach: {cash_pct:.1%} < {self.limits.cash_floor_pct:.1%}",
                        {"cash_pct": cash_pct, "required": self.limits.cash_floor_pct},
                    )

            # Check exposure ceiling
            if self.total_value > 0:
                exposure_pct = self.current_exposure / self.total_value
                if exposure_pct > self.limits.exposure_ceiling_pct:
                    return GuardResult(
                        GuardStatus.BLOCK,
                        f"Exposure ceiling breach: {exposure_pct:.1%} > {self.limits.exposure_ceiling_pct:.1%}",
                        {"exposure_pct": exposure_pct, "limit": self.limits.exposure_ceiling_pct},
                    )

            # Check per-trade limit
            if trade_size_usd > self.limits.max_trade_usd:
                return GuardResult(
                    GuardStatus.BLOCK,
                    f"Trade size ${trade_size_usd:.2f} exceeds max ${self.limits.max_trade_usd:.2f}",
                    {"trade_size": trade_size_usd, "limit": self.limits.max_trade_usd},
                )

            # Check per-trade percentage
            if self.total_value > 0:
                trade_pct = trade_size_usd / self.total_value
                if trade_pct > self.limits.per_trade_pct:
                    return GuardResult(
                        GuardStatus.BLOCK,
                        f"Trade {trade_pct:.1%} exceeds per-trade limit {self.limits.per_trade_pct:.1%}",
                        {"trade_pct": trade_pct, "limit": self.limits.per_trade_pct},
                    )

            # Check position limit for this market
            current_position = self.position_values.get(market_id, 0.0)
            new_position = current_position + trade_size_usd
            if new_position > self.limits.max_position_usd:
                return GuardResult(
                    GuardStatus.BLOCK,
                    f"Position ${new_position:.2f} would exceed max ${self.limits.max_position_usd:.2f}",
                    {"current": current_position, "new": new_position, "limit": self.limits.max_position_usd},
                )

            # Check max concurrent positions
            open_positions = len([v for v in self.position_values.values() if v > 0])
            if open_positions >= self.limits.max_positions:
                return GuardResult(
                    GuardStatus.BLOCK,
                    f"Max positions ({self.limits.max_positions}) reached",
                    {"open_positions": open_positions},
                )

            # Check drawdown
            if self.session_high > 0:
                drawdown = (self.session_high - self.total_value) / self.session_high
                if drawdown > self.limits.max_drawdown_pct:
                    self.halted = True
                    self.halt_reason = HaltReason.DRAWDOWN
                    return GuardResult(
                        GuardStatus.HALT,
                        f"Drawdown halt: {drawdown:.1%} > {self.limits.max_drawdown_pct:.1%}",
                        {"drawdown": drawdown, "limit": self.limits.max_drawdown_pct},
                    )

            # Check daily loss
            if self.total_value > 0:
                daily_loss_pct = self.daily_loss / self.total_value
                if daily_loss_pct > self.limits.max_daily_loss_pct:
                    self.halted = True
                    self.halt_reason = HaltReason.DAILY_LOSS_LIMIT
                    return GuardResult(
                        GuardStatus.HALT,
                        f"Daily loss halt: {daily_loss_pct:.1%} > {self.limits.max_daily_loss_pct:.1%}",
                        {"daily_loss_pct": daily_loss_pct, "limit": self.limits.max_daily_loss_pct},
                    )

            # Check consecutive losses
            if self.consecutive_losses >= self.limits.max_consecutive_losses:
                return GuardResult(
                    GuardStatus.BLOCK,
                    f"Consecutive losses: {self.consecutive_losses} >= {self.limits.max_consecutive_losses}",
                    {"consecutive": self.consecutive_losses, "limit": self.limits.max_consecutive_losses},
                )

            return GuardResult(GuardStatus.PASS, "All guards passed")

    def check_post_trade(self) -> Optional[GuardResult]:
        """Check post-trade guards (drawdown, daily loss)."""
        with self._lock:
            # Check drawdown
            if self.session_high > 0:
                drawdown = (self.session_high - self.total_value) / self.session_high
                if drawdown > self.limits.max_drawdown_pct:
                    self.halted = True
                    self.halt_reason = HaltReason.DRAWDOWN
                    return GuardResult(
                        GuardStatus.HALT,
                        f"Drawdown halt: {drawdown:.1%} > {self.limits.max_drawdown_pct:.1%}",
                        {"drawdown": drawdown, "limit": self.limits.max_drawdown_pct},
                    )

            # Check daily loss
            if self.total_value > 0:
                daily_loss_pct = self.daily_loss / self.total_value
                if daily_loss_pct > self.limits.max_daily_loss_pct:
                    self.halted = True
                    self.halt_reason = HaltReason.DAILY_LOSS_LIMIT
                    return GuardResult(
                        GuardStatus.HALT,
                        f"Daily loss halt: {daily_loss_pct:.1%} > {self.limits.max_daily_loss_pct:.1%}",
                        {"daily_loss_pct": daily_loss_pct, "limit": self.limits.max_daily_loss_pct},
                    )

            return None

    def update_position(self, market_id: str, value: float):
        """Update position value for a market."""
        with self._lock:
            if value > 0:
                self.position_values[market_id] = value
            else:
                self.position_values.pop(market_id, None)

    def manual_halt(self, reason: HaltReason = HaltReason.MANUAL):
        """Manually halt trading."""
        with self._lock:
            self.halted = True
            self.halt_reason = reason
            log.warning("Manual halt: %s", reason.value)

    def resume(self):
        """Resume trading after halt."""
        with self._lock:
            self.halted = False
            self.halt_reason = None
            self.consecutive_losses = 0
            log.info("Trading resumed")

    def get_status(self) -> Dict:
        """Get current guard status."""
        with self._lock:
            return {
                "halted": self.halted,
                "halt_reason": self.halt_reason.value if self.halt_reason else None,
                "balance": self.current_balance,
                "exposure": self.current_exposure,
                "total_value": self.total_value,
                "session_high": self.session_high,
                "drawdown_pct": (self.session_high - self.total_value) / self.session_high if self.session_high > 0 else 0,
                "cash_pct": self.current_balance / self.total_value if self.total_value > 0 else 1.0,
                "exposure_pct": self.current_exposure / self.total_value if self.total_value > 0 else 0,
                "consecutive_losses": self.consecutive_losses,
                "daily_loss_pct": self.daily_loss / self.total_value if self.total_value > 0 else 0,
                "open_positions": len([v for v in self.position_values.values() if v > 0]),
                "api_breaker_state": self.api_breaker.get_state(),
            }


# ─── CLI / demo ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    limits = RiskLimits(
        cash_floor_pct=0.70,
        exposure_ceiling_pct=0.30,
        per_trade_pct=0.08,
        max_drawdown_pct=0.15,
        max_trade_usd=100.0,
        max_consecutive_losses=3,
    )

    guards = GuardWiring(limits, starting_balance=1000.0)
    guards.update_balance(1000.0, 0.0)

    print("Guard Wiring Demo:")
    print("=" * 50)

    # Test normal trade
    result = guards.check_pre_trade(50.0, "market_001")
    print(f"Trade $50: {result.status.value} - {result.reason}")

    # Test trade too large
    result = guards.check_pre_trade(150.0, "market_002")
    print(f"Trade $150: {result.status.value} - {result.reason}")

    # Simulate position
    guards.update_position("market_001", 50.0)
    guards.update_balance(950.0, 50.0)

    # Test drawdown
    guards.update_balance(800.0, 50.0)  # Total 850, high was 1000 -> 15% drawdown
    result = guards.check_pre_trade(10.0, "market_003")
    print(f"After drawdown: {result.status.value} - {result.reason}")

    print(f"\nStatus: {guards.get_status()}")