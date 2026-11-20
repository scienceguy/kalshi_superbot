#!/usr/bin/env python3
"""
crossvenue_live_monitor.py — T15-31: Live cross-venue arbitrage monitor.

Continuously monitors Kalshi vs Polymarket US for the same asset/window,
logs all deviations (not "arbitrage" until basis risk is quantified).
Runs as a background process alongside the main trader.
"""

from __future__ import annotations

import asyncio
import logging
import math
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional

import config
from cross_sectional_arb import (
    VenueQuote,
    ArbOpportunity,
    get_kalshi_quote,
    get_polymarket_quote,
    find_cross_venue_arbs,
    kalshi_fee_taker,
    polymarket_fee_taker,
)

log = logging.getLogger(__name__)


class CrossVenueLiveMonitor:
    """
    Live monitor for cross-venue deviations between Kalshi and Polymarket US.
    
    Does NOT execute trades — only logs opportunities for the 14-day paper study.
    """
    
    def __init__(
        self,
        check_interval_seconds: float = 15.0,
        min_edge_cents: float = 0.5,
        assets: List[str] = None,
    ):
        self.check_interval = check_interval_seconds
        self.min_edge_cents = min_edge_cents
        self.assets = assets or ["BTC", "ETH", "SOL", "XRP"]
        
        # Clients
        from kalshi_client import KalshiClient
        sys.path.insert(0, '/home/ubuntu/kalshi_superbot/archive')
        from polymarket_client import PolymarketUSClient
        
        self.kalshi = KalshiClient()
        self.poly = PolymarketUSClient()
        
        # State
        self.running = False
        self.last_opportunities: Dict[str, ArbOpportunity] = {}
        self.opportunity_log_path = config.BACKTEST_CAPTURE_DIR / "crossvenue_opportunities.jsonl"
        self.stats = {
            "cycles": 0,
            "opportunities_found": 0,
            "opportunities_logged": 0,
            "errors": 0,
        }
        
        log.info(
            "CrossVenueLiveMonitor initialized: interval=%.1fs, assets=%s, min_edge=%.1fc",
            check_interval_seconds, self.assets, min_edge_cents
        )
    
    def get_current_window(self) -> datetime:
        """Get the current 15-minute window start time."""
        now = datetime.now(timezone.utc)
        minutes = (now.minute // 15) * 15
        return now.replace(minute=minutes, second=0, microsecond=0)
    
    async def check_once(self) -> List[ArbOpportunity]:
        """Run one check cycle for all assets."""
        window_start = self.get_current_window()
        all_opportunities = []
        
        for asset in self.assets:
            try:
                kalshi_q = get_kalshi_quote(self.kalshi, asset, window_start)
                poly_q = get_polymarket_quote(self.poly, asset, window_start)
                
                if not kalshi_q or not poly_q:
                    log.debug(f"No quotes for {asset} at {window_start}")
                    continue
                
                # Compute deviations (bidirectional)
                opportunities = self._compute_deviations(kalshi_q, poly_q, asset, window_start)
                all_opportunities.extend(opportunities)
                
            except Exception as e:
                log.warning(f"Error checking {asset}: {e}")
                self.stats["errors"] += 1
        
        return all_opportunities
    
    def _compute_deviations(
        self,
        kalshi_q: VenueQuote,
        poly_q: VenueQuote,
        asset: str,
        window_start: datetime,
    ) -> List[ArbOpportunity]:
        """Compute price deviations between venues."""
        opportunities = []
        
        # Direction 1: Buy YES on Kalshi, sell YES on Polymarket (buy NO on Polymarket)
        cost_kalshi_yes = kalshi_q.yes_ask + kalshi_fee_taker(kalshi_q.yes_ask)
        cost_poly_no = poly_q.no_ask + polymarket_fee_taker(poly_q.no_ask)
        total_cost_1 = cost_kalshi_yes + cost_poly_no
        edge_1 = 100 - total_cost_1
        
        # Direction 2: Buy YES on Polymarket, sell YES on Kalshi (buy NO on Kalshi)
        cost_poly_yes = poly_q.yes_ask + polymarket_fee_taker(poly_q.yes_ask)
        cost_kalshi_no = kalshi_q.no_ask + kalshi_fee_taker(kalshi_q.no_ask)
        total_cost_2 = cost_poly_yes + cost_kalshi_no
        edge_2 = 100 - total_cost_2
        
        # Check both directions
        if edge_1 > self.min_edge_cents:
            opp = ArbOpportunity(
                asset=asset,
                window_start=window_start,
                kalshi_quote=kalshi_q,
                poly_quote=poly_q,
                kalshi_side="yes",
                poly_side="no",
                gross_edge_cents=edge_1,
                net_edge_cents=edge_1,
                confidence=min(0.9, edge_1 / 2.0),
            )
            opportunities.append(opp)
        
        if edge_2 > self.min_edge_cents:
            opp = ArbOpportunity(
                asset=asset,
                window_start=window_start,
                kalshi_quote=kalshi_q,
                poly_quote=poly_q,
                kalshi_side="no",
                poly_side="yes",
                gross_edge_cents=edge_2,
                net_edge_cents=edge_2,
                confidence=min(0.9, edge_2 / 2.0),
            )
            opportunities.append(opp)
        
        # Also log DEVIATIONS even if below edge threshold (for paper study)
        # Track the raw price difference
        raw_diff_yes = kalshi_q.yes_ask - poly_q.yes_ask
        raw_diff_no = kalshi_q.no_ask - poly_q.no_ask
        
        if abs(raw_diff_yes) > 0.5 or abs(raw_diff_no) > 0.5:
            # Log deviation for study
            self._log_deviation(asset, window_start, kalshi_q, poly_q, raw_diff_yes, raw_diff_no)
        
        return opportunities
    
    def _log_deviation(
        self,
        asset: str,
        window_start: datetime,
        kalshi_q: VenueQuote,
        poly_q: VenueQuote,
        raw_diff_yes: float,
        raw_diff_no: float,
    ):
        """Log a deviation for the 14-day paper study."""
        import json
        from pathlib import Path
        
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "asset": asset,
            "window_start": window_start.isoformat(),
            "kalshi_yes_bid": kalshi_q.yes_bid,
            "kalshi_yes_ask": kalshi_q.yes_ask,
            "kalshi_no_bid": kalshi_q.no_bid,
            "kalshi_no_ask": kalshi_q.no_ask,
            "poly_yes_bid": poly_q.yes_bid,
            "poly_yes_ask": poly_q.yes_ask,
            "poly_no_bid": poly_q.no_bid,
            "poly_no_ask": poly_q.no_ask,
            "raw_diff_yes": raw_diff_yes,
            "raw_diff_no": raw_diff_no,
            "kalshi_fee_yes": kalshi_fee_taker(kalshi_q.yes_ask),
            "poly_fee_yes": polymarket_fee_taker(poly_q.yes_ask),
        }
        
        self.opportunity_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.opportunity_log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        
        self.stats["opportunities_logged"] += 1
    
    def _log_opportunity(self, opp: ArbOpportunity):
        """Log a tradable opportunity."""
        import json
        from pathlib import Path
        
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": "opportunity",
            "asset": opp.asset,
            "window_start": opp.window_start.isoformat(),
            "kalshi_side": opp.kalshi_side,
            "poly_side": opp.poly_side,
            "kalshi_yes_ask": opp.kalshi_quote.yes_ask,
            "kalshi_no_ask": opp.kalshi_quote.no_ask,
            "poly_yes_ask": opp.poly_quote.yes_ask,
            "poly_no_ask": opp.poly_quote.no_ask,
            "net_edge_cents": opp.net_edge_cents,
            "confidence": opp.confidence,
        }
        
        self.opportunity_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.opportunity_log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        
        self.stats["opportunities_found"] += 1
        
        # Also print to console
        log.info(
            "OPPORTUNITY: %s @ %s | Kalshi %s @ %.1fc, Poly %s @ %.1fc | Edge: %.2fc",
            opp.asset, opp.window_start.strftime("%H:%M:%S"),
            opp.kalshi_side.upper(), getattr(opp.kalshi_quote, f"{opp.kalshi_side}_ask"),
            opp.poly_side.upper(), getattr(opp.poly_quote, f"{opp.poly_side}_ask"),
            opp.net_edge_cents
        )
    
    async def run(self):
        """Main monitoring loop."""
        self.running = True
        log.info("Starting cross-venue live monitor...")
        
        while self.running:
            cycle_start = time.time()
            self.stats["cycles"] += 1
            
            try:
                opportunities = await self.check_once()
                
                for opp in opportunities:
                    # Deduplicate: only log new or significantly changed opportunities
                    key = f"{opp.asset}_{opp.window_start.isoformat()}_{opp.kalshi_side}_{opp.poly_side}"
                    if key not in self.last_opportunities:
                        self._log_opportunity(opp)
                        self.last_opportunities[key] = opp
                    elif abs(opp.net_edge_cents - self.last_opportunities[key].net_edge_cents) > 0.5:
                        # Significant change
                        self._log_opportunity(opp)
                        self.last_opportunities[key] = opp
                
                # Clean up old opportunities (older than 30 minutes)
                now = datetime.now(timezone.utc)
                cutoff = now.timestamp() - 1800
                self.last_opportunities = {
                    k: v for k, v in self.last_opportunities.items()
                    if v.window_start.timestamp() > cutoff
                }
                
            except Exception as e:
                log.error(f"Error in monitoring loop: {e}")
                self.stats["errors"] += 1
            
            # Wait for next cycle
            elapsed = time.time() - cycle_start
            sleep_time = max(0, self.check_interval - elapsed)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        log.info("Cross-venue monitor stopped. Stats: %s", self.stats)
    
    def stop(self):
        """Stop the monitor."""
        self.running = False


async def main():
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s — %(message)s",
    )
    
    monitor = CrossVenueLiveMonitor(
        check_interval_seconds=15.0,
        min_edge_cents=0.5,
        assets=["BTC", "ETH", "SOL", "XRP"],
    )
    
    try:
        await monitor.run()
    except KeyboardInterrupt:
        log.info("Shutdown requested")
    finally:
        monitor.stop()


if __name__ == "__main__":
    asyncio.run(main())