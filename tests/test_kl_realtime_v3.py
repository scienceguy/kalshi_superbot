from datetime import datetime, timezone

import pytest

import kl_realtime_v3 as kl


def _position(*, side="YES", direction="LONG", price=0.96, contracts=10, fee=0.01):
    return kl.Position(
        ticker="TEST",
        venue="KALSHI",
        side=side,
        entry_price=price,
        filled_price=price,
        contracts=contracts,
        filled_contracts=contracts,
        expiry_time=datetime.now(timezone.utc),
        signal_type="TEST",
        direction=direction,
        fee_paid=fee,
        margin_reserved=contracts if direction == "SHORT" else 0.0,
    )


def test_long_pnl_and_settlement_cash_are_consistent():
    pos = _position(side="YES", direction="LONG", price=0.96)

    assert pos.pnl_at_settle(1) == pytest.approx(0.39)
    assert pos.settlement_cash_delta(1) == 10.0
    assert pos.settlement_cash_delta(0) == 0.0


def test_short_pnl_and_margin_release_are_consistent():
    pos = _position(side="YES", direction="SHORT", price=0.96)

    # Entry consumes (1 - 0.96) * 10 buying power plus fee. A losing short
    # releases $10 of margin and earns the sale proceeds; a winning short pays
    # the $10 obligation from that margin.
    assert pos.pnl_at_settle(0) == pytest.approx(9.59)
    assert pos.pnl_at_settle(1) == pytest.approx(-0.41)
    assert pos.settlement_cash_delta(0) == 10.0
    assert pos.settlement_cash_delta(1) == 0.0


def test_venue_fee_models_use_the_published_shapes():
    assert kl.kalshi_taker_fee(100, 0.50) == 1.75
    assert kl.kalshi_taker_fee(1, 0.50) == 0.02
    assert kl.polymarket_us_fee(100, 0.50) == 2.93


def test_cross_venue_settlement_can_disagree(monkeypatch):
    bot = object.__new__(kl.RealTimeDryRun)
    expiry = datetime.now(timezone.utc)
    kalshi = _position(side="YES", price=0.50)
    polymarket = _position(side="YES", price=0.50)
    kalshi.expiry_time = expiry
    polymarket.expiry_time = expiry
    kalshi.venue = "KALSHI"
    polymarket.venue = "POLYMARKET"
    rolls = iter((0.10, 0.0))
    monkeypatch.setattr(kl.random, "random", lambda: next(rolls))

    assert bot._window_outcome(kalshi) == 1
    assert bot._window_outcome(polymarket) == 0


def test_two_leg_short_trade_is_atomic(monkeypatch):
    bot = object.__new__(kl.RealTimeDryRun)
    bot.kalshi_balance = 10.0
    bot.poly_balance = 10.0
    bot.kalshi_positions = []
    bot.poly_positions = []
    monkeypatch.setattr(kl.random, "random", lambda: 0.9)

    bot._place_trade(
        "KTEST", "YES", 0.96, "PTEST", "YES", 0.97,
        "TEST_LONG_SHORT", "LONG", "SHORT",
    )

    assert len(bot.kalshi_positions) == 1
    assert len(bot.poly_positions) == 1
    assert bot.kalshi_positions[0].direction == "LONG"
    assert bot.poly_positions[0].direction == "SHORT"
    assert bot.poly_positions[0].margin_reserved == bot.poly_positions[0].filled_contracts
    assert bot.kalshi_balance < 10.0
    assert bot.poly_balance < 10.0


def test_unfunded_leg_aborts_the_entire_pair(monkeypatch):
    bot = object.__new__(kl.RealTimeDryRun)
    bot.kalshi_balance = 0.10
    bot.poly_balance = 10.0
    bot.kalshi_positions = []
    bot.poly_positions = []
    monkeypatch.setattr(kl.random, "random", lambda: 0.9)

    bot._place_trade(
        "KTEST", "YES", 0.96, "PTEST", "YES", 0.97,
        "TEST_LONG_SHORT", "LONG", "SHORT",
    )

    assert bot.kalshi_positions == []
    assert bot.poly_positions == []
    assert bot.kalshi_balance == 0.10
    assert bot.poly_balance == 10.0
