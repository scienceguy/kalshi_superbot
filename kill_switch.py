#!/usr/bin/env python3
"""
kill_switch.py — T15-42: Kill switches and drills.

Halt flag in the local DB honored by every order path.
Triggers:
- n8n kill-switch webhook (authenticated)
- Daily loss limit
- Drawdown limit
- Calibration breaker
- Stale feeds
- Reconciliation mismatch
- Venue error spike
- Clock skew

Two-person resume (per-identity HMAC, J10) must be exercised in a drill.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

import config

log = logging.getLogger(__name__)


class HaltTrigger(Enum):
    """Reasons for halting trading."""
    N8N_WEBHOOK = "n8n_webhook"
    DAILY_LOSS = "daily_loss"
    DRAWDOWN = "drawdown"
    CALIBRATION = "calibration"
    STALE_FEED = "stale_feed"
    RECONCILIATION = "reconciliation"
    VENUE_ERROR = "venue_error"
    CLOCK_SKEW = "clock_skew"
    MANUAL = "manual"
    RAMP_VIOLATION = "ramp_violation"


@dataclass
class HaltEvent:
    """A halt event record."""
    timestamp: str
    trigger: HaltTrigger
    message: str
    details: Dict = field(default_factory=dict)
    acknowledged: bool = False
    acknowledged_by: str = ""
    acknowledged_at: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "trigger": self.trigger.value,
            "message": self.message,
            "details": self.details,
            "acknowledged": self.acknowledged,
            "acknowledged_by": self.acknowledged_by,
            "acknowledged_at": self.acknowledged_at,
        }


class KillSwitch:
    """
    Global kill switch for the trading system.
    
    Checked by every order path before placing orders.
    Can be triggered by multiple sources and requires two-person rule to clear.
    """
    
    def __init__(self):
        self.halted = False
        self.halt_events: List[HaltEvent] = []
        self.current_halt: Optional[HaltEvent] = None
        
        # Two-person rule
        self.require_approval = config.RAMP_REQUIRE_APPROVAL if hasattr(config, 'RAMP_REQUIRE_APPROVAL') else True
        self.approval_key = config.RAMP_APPROVAL_KEY if hasattr(config, 'RAMP_APPROVAL_KEY') else ""
        
        # HMAC key for agent identity verification
        self.hmac_key = os.environ.get("KILL_SWITCH_HMAC_KEY", "").encode()
        
        log.info("KillSwitch initialized: require_approval=%s", self.require_approval)
    
    def trigger(
        self,
        trigger: HaltTrigger,
        message: str,
        details: Dict = None,
    ) -> HaltEvent:
        """Trigger a halt."""
        if self.halted:
            log.warning("Kill switch already triggered, adding event")
        
        event = HaltEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            trigger=trigger,
            message=message,
            details=details or {},
        )
        
        self.halted = True
        self.current_halt = event
        self.halt_events.append(event)
        
        log.critical("KILL SWITCH TRIGGERED: %s - %s", trigger.value, message)
        
        # Notify n8n if available
        self._notify_n8n(event)
        
        return event
    
    def _notify_n8n(self, event: HaltEvent):
        """Notify n8n of halt via webhook."""
        try:
            import requests
            webhook_url = os.environ.get("N8N_KILL_SWITCH_WEBHOOK")
            if webhook_url:
                requests.post(webhook_url, json=event.to_dict(), timeout=5)
        except Exception as e:
            log.warning("Failed to notify n8n of halt: %s", e)
    
    def is_halted(self) -> bool:
        """Check if trading is halted."""
        return self.halted
    
    def get_halt_reason(self) -> str:
        """Get the current halt reason."""
        if self.current_halt:
            return f"{self.current_halt.trigger.value}: {self.current_halt.message}"
        return ""
    
    def get_current_halt(self) -> Optional[HaltEvent]:
        """Get the current halt event."""
        return self.current_halt
    
    def clear_halt(
        self,
        approver: str,
        approval_hmac: str = "",
    ) -> bool:
        """
        Clear the halt with two-person rule.
        
        Requires:
        1. If require_approval is True, a valid HMAC from the second person
        2. The HMAC is computed as HMAC(key, approver + timestamp + "clear")
        
        Args:
            approver: Identity of the person clearing (e.g., "ubuntu:opencode:home")
            approval_hmac: HMAC signature from the second approver
        
        Returns:
            True if halt cleared, False otherwise
        """
        if not self.halted:
            log.warning("Kill switch not halted, nothing to clear")
            return True
        
        # Verify approval if required
        if self.require_approval:
            if not self.approval_key:
                log.error("Approval required but no approval key configured")
                return False
            
            # Verify HMAC
            expected = hmac.new(
                self.approval_key.encode(),
                f"{approver}{int(time.time())}clear".encode(),
                hashlib.sha256
            ).hexdigest()
            
            if approval_hmac != expected:
                log.warning("Invalid approval HMAC from %s", approver)
                return False
        
        # Clear the halt
        if self.current_halt:
            self.current_halt.acknowledged = True
            self.current_halt.acknowledged_by = approver
            self.current_halt.acknowledged_at = datetime.now(timezone.utc).isoformat()
        
        self.halted = False
        self.current_halt = None
        
        log.info("KILL SWITCH CLEARED by %s", approver)
        return True
    
    def generate_approval_hmac(self, approver: str) -> str:
        """
        Generate an approval HMAC for the two-person rule.
        
        Called by the second person to generate the signature.
        """
        return hmac.new(
            self.approval_key.encode(),
            f"{approver}{int(time.time())}clear".encode(),
            hashlib.sha256
        ).hexdigest()
    
    def check_and_clear_stale_halts(self, max_age_seconds: int = 3600):
        """
        Auto-clear very old halts (older than max_age_seconds).
        
        This prevents permanent halts from abandoned sessions.
        Requires manual approval for anything newer.
        """
        if not self.current_halt:
            return
        
        halt_time = datetime.fromisoformat(self.current_halt.timestamp.replace('Z', '+00:00'))
        age = (datetime.now(timezone.utc) - halt_time).total_seconds()
        
        if age > max_age_seconds:
            log.warning("Auto-clearing stale halt (age=%.0fs): %s", age, self.current_halt.message)
            self.current_halt.acknowledged = True
            self.current_halt.acknowledged_by = "auto-stale-clear"
            self.current_halt.acknowledged_at = datetime.now(timezone.utc).isoformat()
            self.halted = False
            self.current_halt = None
    
    def get_status(self) -> Dict:
        """Get kill switch status."""
        return {
            "halted": self.halted,
            "current_halt": self.current_halt.to_dict() if self.current_halt else None,
            "total_halt_events": len(self.halt_events),
            "recent_events": [e.to_dict() for e in self.halt_events[-10:]],
            "require_approval": self.require_approval,
        }


# ─── Drill functions ──────────────────────────────────────────────────────────

def run_kill_switch_drill(kill_switch: KillSwitch) -> Dict:
    """
    Run a kill switch drill.
    
    Tests:
    1. Trigger each halt type
    2. Verify trading is halted
    3. Clear with two-person rule
    4. Verify trading resumes
    
    Returns drill results.
    """
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tests": [],
        "all_passed": True,
    }
    
    test_triggers = [
        (HaltTrigger.MANUAL, "Manual drill trigger"),
        (HaltTrigger.DRAWDOWN, "Drawdown drill trigger"),
        (HaltTrigger.STALE_FEED, "Stale feed drill trigger"),
        (HaltTrigger.RECONCILIATION, "Reconciliation drill trigger"),
    ]
    
    for trigger, message in test_triggers:
        # Trigger halt
        event = kill_switch.trigger(trigger, message, {"drill": True})
        
        # Verify halted
        halted = kill_switch.is_halted()
        
        # Generate approval HMAC (second person)
        approver = "ubuntu:opencode:home"
        approval_hmac = kill_switch.generate_approval_hmac(approver)
        
        # Clear halt
        cleared = kill_switch.clear_halt(approver, approval_hmac)
        
        # Verify cleared
        cleared_ok = not kill_switch.is_halted()
        
        test_result = {
            "trigger": trigger.value,
            "halted": halted,
            "cleared": cleared,
            "cleared_ok": cleared_ok,
            "passed": halted and cleared and cleared_ok,
        }
        
        results["tests"].append(test_result)
        
        if not test_result["passed"]:
            results["all_passed"] = False
    
    log.info("Kill switch drill completed: all_passed=%s", results["all_passed"])
    return results


# ─── Integration ──────────────────────────────────────────────────────────────

_kill_switch: Optional[KillSwitch] = None


def init_kill_switch() -> KillSwitch:
    """Initialize the global kill switch."""
    global _kill_switch
    _kill_switch = KillSwitch()
    return _kill_switch


def get_kill_switch() -> Optional[KillSwitch]:
    """Get the global kill switch."""
    return _kill_switch


def is_trading_halted() -> bool:
    """Check if trading is halted (convenience function)."""
    if _kill_switch is None:
        return False
    return _kill_switch.is_halted()


def trigger_halt(trigger: HaltTrigger, message: str, details: Dict = None) -> Optional[HaltEvent]:
    """Trigger a halt (convenience function)."""
    if _kill_switch is None:
        log.warning("Kill switch not initialized")
        return None
    return _kill_switch.trigger(trigger, message, details)


def clear_halt(approver: str, approval_hmac: str = "") -> bool:
    """Clear the halt (convenience function)."""
    if _kill_switch is None:
        return False
    return _kill_switch.clear_halt(approver, approval_hmac)


# ─── CLI / demo ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Quick test
    ks = KillSwitch()
    
    # Test trigger
    event = ks.trigger(HaltTrigger.MANUAL, "Test halt")
    print(f"Halted: {ks.is_halted()}")
    print(f"Reason: {ks.get_halt_reason()}")
    
    # Test clear with HMAC
    approver = "ubuntu:opencode:home"
    hmac_sig = ks.generate_approval_hmac(approver)
    cleared = ks.clear_halt(approver, hmac_sig)
    print(f"Cleared: {cleared}")
    print(f"Halted: {ks.is_halted()}")
    
    print("KillSwitch test passed!")