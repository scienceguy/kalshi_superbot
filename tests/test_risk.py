"""Tests for risk.py — Risk management."""
import pytest
from risk import RiskManager, RiskState
import config


def test_risk_initialize():
    r = RiskManager()
    r.initialize(100.0)
    assert r.state.start_balance == 100.0
    assert r.state.session_high == 100.0
    assert not r.state.halted


def test_max_trade_dollars_halted():
    r = RiskManager()
    r.state.halted = True
    assert r.max_trade_dollars(100.0, 0.0) == 0.0


def test_max_trade_dollars_consecutive_losses():
    r = RiskManager()
    r.state.consecutive_losses = 3
    assert r.max_trade_dollars(100.0, 0.0) == 0.0


def test_max_trade_dollars_exposure_ceiling():
    config.EXPOSURE_CEILING_PCT = 0.30
    config.CASH_FLOOR_PCT = 0.70
    config.MIN_CASH_ABSOLUTE = 1.00
    config.PER_TRADE_PCT = 0.08
    config.MAX_TRADE_DOLLARS = 3.00
    r = RiskManager()
    r.state.start_balance = 100.0
    r.state.session_high = 100.0
    # portfolio = 100, exposure_room = 30 - 0 = 30, cash_room = 100 - 0 - 70 = 30
    # per_trade = 100 * 0.08 = 8, max = 3
    result = r.max_trade_dollars(100.0, 0.0)
    assert result == 3.00


def test_risk_check_blocks():
    r = RiskManager()
    r.state.start_balance = 100.0
    r.state.session_high = 100.0
    config.EXPOSURE_CEILING_PCT = 0.30
    config.CASH_FLOOR_PCT = 0.70
    config.MIN_CASH_ABSOLUTE = 1.00
    config.PER_TRADE_PCT = 0.08
    config.MAX_TRADE_DOLLARS = 3.00
    allowed, reason = r.check(100.0, 0.0, 50.0, "TEST")
    assert not allowed
    assert "risk cap" in reason


def test_reset_daily_keeps_the_peak_and_the_halt():
    """A day roll must not un-halt a bot that is still deep in drawdown."""
    r = RiskManager()
    r.state.start_balance = 100.0
    r.state.session_high = 150.0
    r.state.halted = True
    r.state.consecutive_losses = 3
    r.reset_daily()
    assert r.state.session_high == 150.0  # peak carried over
    assert r.state.halted is True         # halt lasts until the drawdown recovers
    assert r.state.consecutive_losses == 0


# ── T13-03 Equity-scaled risk limit tests ──────────────────────────────────────

def test_max_trade_pct_of_equity_at_35():
    """Equity-scaled max trade at $35 bankroll (micro-live entry)."""
    config.MAX_TRADE_PCT_OF_EQUITY = 0.05  # 5%
    config.MAX_TRADE_DOLLARS = 3.00
    config.EXPOSURE_CEILING_PCT = 0.30
    config.CASH_FLOOR_PCT = 0.70
    config.MIN_CASH_ABSOLUTE = 1.00
    config.PER_TRADE_PCT = 0.08
    
    r = RiskManager()
    r.initialize(35.0)
    r.state.start_balance = 35.0
    r.state.session_high = 35.0
    
    # equity_scaled_max = 35 * 0.05 = 1.75
    # max_trade_cap = min(1.75, 3.00) = 1.75
    # exposure_room = 35 * 0.30 = 10.5
    # cash_room = 35 - 0 - max(1.0, 35 * 0.70) = 35 - 24.5 = 10.5
    # per_trade = 35 * 0.08 = 2.8
    # Result = min(10.5, 10.5, 2.8, 1.75) = 1.75
    result = r.max_trade_dollars(35.0, 0.0)
    assert abs(result - 1.75) < 0.01, f"Expected 1.75, got {result}"


def test_max_trade_pct_of_equity_at_1k():
    """Equity-scaled max trade at $1,000 bankroll."""
    config.MAX_TRADE_PCT_OF_EQUITY = 0.05  # 5%
    config.MAX_TRADE_DOLLARS = 3.00
    config.EXPOSURE_CEILING_PCT = 0.30
    config.CASH_FLOOR_PCT = 0.70
    config.MIN_CASH_ABSOLUTE = 1.00
    config.PER_TRADE_PCT = 0.08
    
    r = RiskManager()
    r.initialize(1000.0)
    r.state.start_balance = 1000.0
    r.state.session_high = 1000.0
    
    # equity_scaled_max = 1000 * 0.05 = 50
    # max_trade_cap = min(50, 3.00) = 3.00 (legacy cap wins)
    # Result = 3.00
    result = r.max_trade_dollars(1000.0, 0.0)
    assert abs(result - 3.00) < 0.01, f"Expected 3.00, got {result}"


def test_max_trade_pct_of_equity_at_50k():
    """Equity-scaled max trade at $50,000 bankroll (scaled live)."""
    config.MAX_TRADE_PCT_OF_EQUITY = 0.05  # 5%
    config.MAX_TRADE_DOLLARS = 3.00
    config.EXPOSURE_CEILING_PCT = 0.30
    config.CASH_FLOOR_PCT = 0.70
    config.MIN_CASH_ABSOLUTE = 1.00
    config.PER_TRADE_PCT = 0.08
    
    r = RiskManager()
    r.initialize(50000.0)
    r.state.start_balance = 50000.0
    r.state.session_high = 50000.0
    
    # equity_scaled_max = 50000 * 0.05 = 2500
    # max_trade_cap = min(2500, 3.00) = 3.00 (legacy cap wins)
    # Result = 3.00
    result = r.max_trade_dollars(50000.0, 0.0)
    assert abs(result - 3.00) < 0.01, f"Expected 3.00, got {result}"


def test_min_cash_pct_of_equity_live():
    """Live mode equity-scaled minimum cash floor at $1,000."""
    config.MIN_CASH_PCT_OF_EQUITY = 0.005  # 0.5%
    config.MIN_CASH_ABSOLUTE = 1.00
    config.MAX_TRADE_PCT_OF_EQUITY = 0.50  # Higher so equity cap doesn't limit
    config.MAX_TRADE_DOLLARS = 100.00
    config.EXPOSURE_CEILING_PCT = 0.90
    config.CASH_FLOOR_PCT = 0.01  # Low so cash floor doesn't limit
    config.PER_TRADE_PCT = 0.50
    
    r = RiskManager()
    r.initialize(1000.0)
    r.state.start_balance = 1000.0
    r.state.session_high = 1000.0
    
    # equity_scaled_min = 1000 * 0.005 = 5.0
    # min_cash = max(1.00, 5.0) = 5.0
    # cash - trade_cost >= 5.0
    # With cash=1000, max trade = 1000 - 5.0 = 995, but capped by other limits
    # The check should pass for small trades, fail if it would drop below 5.0
    allowed, reason = r.check(1000.0, 0.0, 3.0, "TEST")
    assert allowed, f"Small trade should be allowed: {reason}"
    
    # Trade that would leave < 5.0 cash
    allowed, reason = r.check(6.0, 0.0, 2.0, "TEST")
    assert not allowed
    assert "cash floor" in reason


def test_min_cash_pct_of_equity_paper():
    """Paper mode equity-scaled minimum cash floor at $35."""
    config.PAPER_MIN_CASH_PCT_OF_EQUITY = 0.001  # 0.1%
    config.PAPER_MIN_CASH_ABSOLUTE = 0.50
    config.PAPER_CASH_FLOOR_PCT = 0.0  # No portfolio-based floor
    config.PAPER_EXPOSURE_CEILING_PCT = 1.0  # No exposure limit
    config.PAPER_PER_TRADE_PCT = 1.0  # No per-trade limit
    config.MAX_TRADE_PCT_OF_EQUITY = 1.0  # 100% so equity cap doesn't limit
    config.MAX_TRADE_DOLLARS = 1000.00  # High legacy cap
    
    r = RiskManager()
    r.initialize(35.0)
    r.state.start_balance = 35.0
    r.state.session_high = 35.0
    
    # equity_scaled_min = 35 * 0.001 = 0.035
    # min_cash = max(0.50, 0.035) = 0.50 (absolute wins at this size)
    # cash - trade_cost >= 0.50
    # The floor is enforced via max_trade_dollars' cash_room = cash - floor_needed
    # So trade_cost > 34.5 will fail with "risk cap" (which includes floor)
    # Trade that would leave 0.49 cash (< 0.50)
    allowed, reason = r.check(35.0, 0.0, 34.51, "TEST", paper=True)
    assert not allowed, f"Should block leaving 0.49 cash: {reason}"
    # The rejection reason is "risk cap" because max_trade already includes the floor
    assert "risk cap" in reason or "cash floor" in reason
    
    # Trade that would leave 0.60 cash (> 0.50)
    allowed, reason = r.check(35.0, 0.0, 34.4, "TEST", paper=True)
    assert allowed, f"Should allow leaving 0.60 cash: {reason}"


def test_equity_scaled_overrides_legacy_at_large_bankroll():
    """At very large bankroll, equity-scaled cap becomes more restrictive than legacy."""
    config.MAX_TRADE_PCT_OF_EQUITY = 0.001  # 0.1% (tight)
    config.MAX_TRADE_DOLLARS = 100.0  # Legacy allows $100
    config.EXPOSURE_CEILING_PCT = 0.30
    config.CASH_FLOOR_PCT = 0.70
    config.MIN_CASH_ABSOLUTE = 1.00
    config.PER_TRADE_PCT = 0.08
    
    r = RiskManager()
    r.initialize(200000.0)  # $200k
    r.state.start_balance = 200000.0
    r.state.session_high = 200000.0
    
    # equity_scaled_max = 200000 * 0.001 = 200
    # max_trade_cap = min(200, 100) = 100 (legacy wins)
    # Let's make equity-scaled more restrictive
    config.MAX_TRADE_PCT_OF_EQUITY = 0.0001  # 0.01%
    # equity_scaled_max = 200000 * 0.0001 = 20
    # max_trade_cap = min(20, 100) = 20 (equity-scaled wins)
    result = r.max_trade_dollars(200000.0, 0.0)
    assert abs(result - 20.0) < 0.01, f"Expected 20.0, got {result}"
