#!/usr/bin/env python3
"""
backtest_data_capture.py — T15-13: Backtest data capture on the VPS.

Persists orderbook snapshots, spot prices, settlements, fees, and decisions
during live/dry-run trading for high-fidelity backtesting.

Unlike data_collector.py which scrapes historical closed markets, this module
captures data at ENTRY TIME (when the bot evaluates a trade) so backtests use
the exact same information the bot had — eliminating lookahead bias.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import config
from kalshi_client import KalshiClient

log = logging.getLogger("backtest_data_capture")


@dataclass
class EntrySnapshot:
    """Complete snapshot at the moment the bot evaluates a trade."""
    # Metadata
    timestamp: str  # ISO8601 UTC when snapshot was taken
    ticker: str
    series_ticker: str
    close_time: str
    minutes_left: float
    minutes_in: float
    window_start_iso: str
    
    # Kalshi orderbook (cents)
    yes_bid: int
    yes_ask: int
    no_bid: int
    no_ask: int
    mid_yes: int
    spread_yes_cents: int
    spread_no_cents: int
    yes_bid_size: int
    yes_ask_size: int
    no_bid_size: int
    no_ask_size: int
    
    # Polymarket US orderbook (dollars) — if available
    poly_yes_bid: Optional[float] = None
    poly_yes_ask: Optional[float] = None
    poly_no_bid: Optional[float] = None
    poly_no_ask: Optional[float] = None
    poly_spread_yes_cents: Optional[float] = None
    poly_spread_no_cents: Optional[float] = None
    
    # Spot / Reference prices
    spot_price: Optional[float] = None  # composite spot (Binance median) in dollars
    spot_source: Optional[str] = None   # "rest" | "ws" | "brti"
    spot_age_seconds: Optional[float] = None
    brti_60s_avg: Optional[float] = None  # BRTI tracker 60s running average
    brti_age_seconds: Optional[float] = None
    
    # Gate diagnostics
    gate_atr_pass: Optional[bool] = None
    gate_ev_pass: Optional[bool] = None
    gate_cushion_pass: Optional[bool] = None
    gate_all_pass: Optional[bool] = None
    gate_atr_value: Optional[float] = None
    gate_ev_value: Optional[float] = None
    gate_cushion_value: Optional[float] = None
    gate_shadow_mode: bool = True
    
    # Fee & sizing
    taker_fee_rate: float = config.KALSHI_FEE_RATE
    per_trade_pct: float = config.PER_TRADE_PCT
    dollar_risk: Optional[float] = None
    contracts: Optional[int] = None
    
    # Decision
    decision: str = "none"  # "enter_yes" | "enter_no" | "skip" | "blocked"
    decision_reason: str = ""
    side: str = ""
    limit_price_cents: Optional[int] = None
    
    # Settlement (filled later when market settles)
    settled_yes: Optional[int] = None
    won_yes: Optional[bool] = None
    settlement_price: Optional[float] = None  # BRTI settlement price
    settlement_ts: Optional[str] = None
    
    # P&L (filled after settlement)
    pnl_dollars: Optional[float] = None
    fee_paid_dollars: Optional[float] = None
    pnl_with_fees: Optional[float] = None
    
    # Cross-venue arb data
    arb_opportunity: Optional[bool] = None
    arb_edge_cents: Optional[float] = None
    arb_legs: Optional[List[Dict]] = None
    
    # Model probabilities
    gbm_prob_yes: Optional[float] = None
    momentum_prob_yes: Optional[float] = None
    locked_share_prob_yes: Optional[float] = None
    ensemble_prob_yes: Optional[float] = None


class BacktestDataCapture:
    """
    Captures entry-time snapshots during trading for high-fidelity backtesting.
    
    Usage:
        capture = BacktestDataCapture(Path("./backtest_data"))
        
        # In trader loop, when evaluating a market:
        snapshot = capture.capture_entry_snapshot(
            ticker="KXBTC15M-...",
            market_data=market,
            orderbook=orderbook,
            spot_price=spot_price,
            gate_results=gate_results,
            decision="enter_yes",
            ...
        )
        
        # Later, when market settles:
        capture.update_settlement(ticker, settled_yes, settlement_price, ...)
    """
    
    def __init__(self, output_dir: Path, session_id: str = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.session_id = session_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.pending_path = self.output_dir / f"pending_{self.session_id}.jsonl.gz"
        self.settled_path = self.output_dir / f"settled_{self.session_id}.jsonl.gz"
        
        # In-memory index for fast settlement updates
        self._pending: Dict[str, EntrySnapshot] = {}
        self._lock = None  # For thread safety if needed
        
        # Stats
        self.captured = 0
        self.settled = 0
        
        log.info("BacktestDataCapture initialized: session=%s, pending=%s", 
                 self.session_id, self.pending_path)
    
    def capture_entry_snapshot(self, **kwargs) -> EntrySnapshot:
        """
        Capture a full entry-time snapshot.
        
        Args passed as kwargs:
            ticker, series_ticker, close_time, minutes_left,
            orderbook (dict with yes/no bids/asks/sizes),
            poly_orderbook (dict, optional),
            spot_price, spot_source, spot_age_seconds,
            brti_60s_avg, brti_age_seconds,
            gate_results (dict), fee_rate, per_trade_pct,
            dollar_risk, contracts,
            decision, decision_reason, side, limit_price_cents,
            model_probs (dict), arb_data (dict)
        """
        now = datetime.now(timezone.utc)
        close_time = kwargs.get("close_time", "")
        
        # Parse minutes_left
        minutes_left = kwargs.get("minutes_left", 0.0)
        if isinstance(minutes_left, str):
            # Parse from close_time if needed
            from market_finder import _parse_api_time
            ct = _parse_api_time(close_time)
            if ct:
                minutes_left = (ct - now).total_seconds() / 60
        
        minutes_in = max(0.0, 15.0 - minutes_left)
        
        # Parse orderbook
        ob = kwargs.get("orderbook", {})
        yes_bid = ob.get("yes_bid", 0)
        yes_ask = ob.get("yes_ask", 0)
        no_bid = ob.get("no_bid", 0)
        no_ask = ob.get("no_ask", 0)
        mid_yes = (yes_bid + yes_ask) // 2 if yes_bid and yes_ask else 0
        spread_yes = yes_ask - yes_bid if yes_bid and yes_ask else 0
        spread_no = no_ask - no_bid if no_bid and no_ask else 0
        
        # Polymarket
        poly_ob = kwargs.get("poly_orderbook")
        poly_yes_bid = poly_ob.get("yes_bid") if poly_ob else None
        poly_yes_ask = poly_ob.get("yes_ask") if poly_ob else None
        poly_no_bid = poly_ob.get("no_bid") if poly_ob else None
        poly_no_ask = poly_ob.get("no_ask") if poly_ob else None
        poly_spread_yes = (poly_yes_ask - poly_yes_bid) * 100 if poly_yes_bid and poly_yes_ask else None
        poly_spread_no = (poly_no_ask - poly_no_bid) * 100 if poly_no_bid and poly_no_ask else None
        
        # Gate results
        gates = kwargs.get("gate_results", {})
        
        # Model probs
        models = kwargs.get("model_probs", {})
        
        # Arb data
        arb = kwargs.get("arb_data", {})
        
        snapshot = EntrySnapshot(
            timestamp=now.isoformat(),
            ticker=kwargs.get("ticker", ""),
            series_ticker=kwargs.get("series_ticker", ""),
            close_time=close_time,
            minutes_left=minutes_left,
            minutes_in=minutes_in,
            window_start_iso=kwargs.get("window_start_iso", ""),
            
            # Kalshi orderbook
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            no_bid=no_bid,
            no_ask=no_ask,
            mid_yes=mid_yes,
            spread_yes_cents=spread_yes,
            spread_no_cents=spread_no,
            yes_bid_size=ob.get("yes_bid_size", 0),
            yes_ask_size=ob.get("yes_ask_size", 0),
            no_bid_size=ob.get("no_bid_size", 0),
            no_ask_size=ob.get("no_ask_size", 0),
            
            # Polymarket
            poly_yes_bid=poly_yes_bid,
            poly_yes_ask=poly_yes_ask,
            poly_no_bid=poly_no_bid,
            poly_no_ask=poly_no_ask,
            poly_spread_yes_cents=poly_spread_yes,
            poly_spread_no_cents=poly_spread_no,
            
            # Spot
            spot_price=kwargs.get("spot_price"),
            spot_source=kwargs.get("spot_source"),
            spot_age_seconds=kwargs.get("spot_age_seconds"),
            brti_60s_avg=kwargs.get("brti_60s_avg"),
            brti_age_seconds=kwargs.get("brti_age_seconds"),
            
            # Gates
            gate_atr_pass=gates.get("atr_pass"),
            gate_ev_pass=gates.get("ev_pass"),
            gate_cushion_pass=gates.get("cushion_pass"),
            gate_all_pass=gates.get("all_pass"),
            gate_atr_value=gates.get("atr_value"),
            gate_ev_value=gates.get("ev_value"),
            gate_cushion_value=gates.get("cushion_value"),
            gate_shadow_mode=gates.get("shadow_mode", True),
            
            # Fees & sizing
            taker_fee_rate=kwargs.get("fee_rate", config.KALSHI_FEE_RATE),
            per_trade_pct=kwargs.get("per_trade_pct", config.PER_TRADE_PCT),
            dollar_risk=kwargs.get("dollar_risk"),
            contracts=kwargs.get("contracts"),
            
            # Decision
            decision=kwargs.get("decision", "none"),
            decision_reason=kwargs.get("decision_reason", ""),
            side=kwargs.get("side", ""),
            limit_price_cents=kwargs.get("limit_price_cents"),
            
            # Models
            gbm_prob_yes=models.get("gbm"),
            momentum_prob_yes=models.get("momentum"),
            locked_share_prob_yes=models.get("locked_share"),
            ensemble_prob_yes=models.get("ensemble"),
            
            # Arb
            arb_opportunity=arb.get("opportunity"),
            arb_edge_cents=arb.get("edge_cents"),
            arb_legs=arb.get("legs"),
        )
        
        # Store pending (indexed by ticker for settlement update)
        self._pending[snapshot.ticker] = snapshot
        self._write_pending(snapshot)
        self.captured += 1
        
        if self.captured % 100 == 0:
            log.info("Captured %d entry snapshots (session=%s)", self.captured, self.session_id)
        
        return snapshot
    
    def _write_pending(self, snapshot: EntrySnapshot):
        """Append snapshot to pending file."""
        with gzip.open(self.pending_path, "at", compresslevel=6) as f:
            f.write(json.dumps(asdict(snapshot)) + "\n")
    
    def update_settlement(
        self,
        ticker: str,
        settled_yes: int,
        settlement_price: float = None,
        settlement_ts: str = None,
        pnl_dollars: float = None,
        fee_paid_dollars: float = None,
    ) -> bool:
        """
        Update a pending snapshot with settlement outcome.
        Call this when a market settles (from trader settlement handler).
        """
        if ticker not in self._pending:
            log.debug("No pending snapshot for %s", ticker)
            return False
        
        snapshot = self._pending.pop(ticker)
        snapshot.settled_yes = settled_yes
        snapshot.won_yes = settled_yes >= 100 if settled_yes is not None else None
        snapshot.settlement_price = settlement_price
        snapshot.settlement_ts = settlement_ts or datetime.now(timezone.utc).isoformat()
        snapshot.pnl_dollars = pnl_dollars
        snapshot.fee_paid_dollars = fee_paid_dollars
        snapshot.pnl_with_fees = (pnl_dollars or 0) - (fee_paid_dollars or 0)
        
        # Write to settled file
        with gzip.open(self.settled_path, "at", compresslevel=6) as f:
            f.write(json.dumps(asdict(snapshot)) + "\n")
        
        self.settled += 1
        return True
    
    def get_pending_count(self) -> int:
        return len(self._pending)
    
    def get_stats(self) -> Dict:
        return {
            "session_id": self.session_id,
            "captured": self.captured,
            "settled": self.settled,
            "pending": len(self._pending),
            "pending_path": str(self.pending_path),
            "settled_path": str(self.settled_path),
        }
    
    def load_pending(self) -> List[EntrySnapshot]:
        """Reload pending snapshots from disk (e.g., after restart)."""
        snapshots = []
        if self.pending_path.exists():
            with gzip.open(self.pending_path, "rt", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        snapshots.append(EntrySnapshot(**data))
                        self._pending[data["ticker"]] = snapshots[-1]
        return snapshots
    
    def load_settled(self) -> List[EntrySnapshot]:
        """Load settled snapshots from disk for backtesting."""
        snapshots = []
        if self.settled_path.exists():
            with gzip.open(self.settled_path, "rt", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        snapshots.append(EntrySnapshot(**data))
        return snapshots


# ─── Integration helper for trader.py ─────────────────────────────────────────

_capture_instance: Optional[BacktestDataCapture] = None


def init_capture(output_dir: Path = Path("./backtest_data"), session_id: str = None) -> BacktestDataCapture:
    """Initialize the global capture instance."""
    global _capture_instance
    _capture_instance = BacktestDataCapture(output_dir, session_id)
    return _capture_instance


def get_capture() -> Optional[BacktestDataCapture]:
    """Get the global capture instance."""
    return _capture_instance


def capture_entry(
    ticker: str,
    market: dict,
    orderbook: dict,
    spot_price: float = None,
    spot_source: str = None,
    spot_age_seconds: float = None,
    brti_60s_avg: float = None,
    brti_age_seconds: float = None,
    gate_results: dict = None,
    poly_orderbook: dict = None,
    decision: str = "none",
    decision_reason: str = "",
    side: str = "",
    limit_price_cents: int = None,
    model_probs: dict = None,
    arb_data: dict = None,
) -> EntrySnapshot:
    """
    Convenience function to capture an entry snapshot from trader context.
    
    Call this in trader.py when evaluating a market for entry.
    """
    if _capture_instance is None:
        log.warning("Backtest capture not initialized, skipping")
        return None
    
    # Extract market data
    close_time = market.get("close_time", "")
    series_ticker = market.get("series_ticker", "")
    
    # Compute window_start from ticker or close_time
    window_start_iso = ""
    if close_time:
        try:
            ct = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
            window_start = ct.replace(minute=(ct.minute // 15) * 15, second=0, microsecond=0)
            window_start_iso = window_start.isoformat()
        except Exception:
            pass
    
    return _capture_instance.capture_entry_snapshot(
        ticker=ticker,
        series_ticker=series_ticker,
        close_time=close_time,
        minutes_left=market.get("minutes_left", 0),
        window_start_iso=window_start_iso,
        orderbook=orderbook,
        poly_orderbook=poly_orderbook,
        spot_price=spot_price,
        spot_source=spot_source,
        spot_age_seconds=spot_age_seconds,
        brti_60s_avg=brti_60s_avg,
        brti_age_seconds=brti_age_seconds,
        gate_results=gate_results,
        fee_rate=config.KALSHI_FEE_RATE,
        per_trade_pct=config.PER_TRADE_PCT,
        dollar_risk=market.get("dollar_risk"),
        contracts=market.get("contracts"),
        decision=decision,
        decision_reason=decision_reason,
        side=side,
        limit_price_cents=limit_price_cents,
        model_probs=model_probs,
        arb_data=arb_data,
    )


def capture_settlement(
    ticker: str,
    settled_yes: int,
    settlement_price: float = None,
    settlement_ts: str = None,
    pnl_dollars: float = None,
    fee_paid_dollars: float = None,
) -> bool:
    """Convenience function to update settlement."""
    if _capture_instance is None:
        return False
    return _capture_instance.update_settlement(
        ticker, settled_yes, settlement_price, settlement_ts, pnl_dollars, fee_paid_dollars
    )


if __name__ == "__main__":
    # Quick test
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        capture = BacktestDataCapture(Path(tmpdir), "test_session")
        
        # Capture an entry
        snap = capture.capture_entry_snapshot(
            ticker="KXBTC15M-26NOV200445-45",
            series_ticker="KXBTC15M",
            close_time="2026-11-20T04:45:00Z",
            minutes_left=10.5,
            window_start_iso="2026-11-20T04:30:00Z",
            orderbook={
                "yes_bid": 7500, "yes_ask": 7600,
                "no_bid": 2400, "no_ask": 2500,
                "yes_bid_size": 100, "yes_ask_size": 50,
                "no_bid_size": 50, "no_ask_size": 100
            },
            spot_price=95234.56,
            spot_source="ws",
            spot_age_seconds=0.5,
            brti_60s_avg=95230.00,
            brti_age_seconds=0.8,
            gate_results={"atr_pass": True, "ev_pass": True, "cushion_pass": True, "all_pass": True},
            fee_rate=0.07,
            per_trade_pct=0.08,
            dollar_risk=80.0,
            contracts=10,
            decision="enter_yes",
            decision_reason="all gates pass, EV=0.12",
            side="yes",
            limit_price_cents=7500,
            model_probs={"gbm": 0.62, "momentum": 0.58, "locked_share": 0.65, "ensemble": 0.61},
            arb_data={"opportunity": True, "edge_cents": 3.5, "legs": [{"venue": "kalshi", "side": "yes"}]}
        )
        
        print(f"Captured: {snap.ticker}, decision={snap.decision}")
        
        # Update settlement
        capture.update_settlement(
            "KXBTC15M-26NOV200445-45",
            settled_yes=100,
            settlement_price=95240.00,
            pnl_dollars=50.0,
            fee_paid_dollars=7.0
        )
        
        print(f"Stats: {capture.get_stats()}")
        
        # Load and verify
        settled = capture.load_settled()
        print(f"Loaded {len(settled)} settled snapshots")
        if settled:
            s = settled[0]
            print(f"  PnL: {s.pnl_dollars}, Fees: {s.fee_paid_dollars}, Won: {s.won_yes}")