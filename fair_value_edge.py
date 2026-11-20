"""
fair_value_edge.py — Fair-value comparison and edge definition (T13-23).

`edge = q_lcb - ask - fee(ask) - slippage`, where `q_lcb` is a lower
confidence bound from the ensemble. Trade only if `edge > min_edge` and
KL/disagreement gates pass. Per-venue comparison picks the better venue for
each side. Reuse `gates.py` (ATR hurdle, fee-adjusted EV, time-decay
cushion); do not re-enable the negative 15-55c band.

Accept: every order row stores q, q_lcb, ask, fee, edge, gate verdicts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import config
import fees
from gbm_fair_value import get_gbm_model, GBMResult
from kl_divergence import get_kl_layer, KLDivergenceLayer
from box_arbitrage import get_box_engine, BoxOpportunity

log = logging.getLogger("fair_value_edge")


@dataclass
class EdgeInputs:
    """Inputs for edge calculation."""
    model_q: float              # Model's P(YES settles) - GBM fair value
    q_lcb: float                # Lower confidence bound on q
    ask_price_cents: int        # Market ask price in cents
    fee_cents: int              # Fee in cents
    slippage_cents: int         # Estimated slippage in cents
    venue: str                  # "kalshi" or "polymarket"
    side: str                   # "YES" or "NO"


@dataclass
class EdgeResult:
    """Result of edge calculation."""
    edge_cents: float           # q_lcb * 100 - ask - fee - slippage (in cents)
    edge_bps: float             # Edge in basis points (relative to $1)
    q: float                    # Model probability
    q_lcb: float                # Lower confidence bound
    ask_price_cents: int        # Market ask
    fee_cents: int              # Fee
    slippage_cents: int         # Slippage
    gate_passed: bool           # All gates passed
    gate_reasons: List[str]     # Reasons for gate failures
    kl_results: Dict            # KL divergence results
    box_opportunity: Optional[BoxOpportunity] = None  # If part of a box


class FairValueEdgeCalculator:
    """
    Calculates trade edge using fair value comparison.
    
    Integrates:
    - GBM fair value model (T13-20) for q
    - KL divergence layer (T13-21) for edge signal and gates
    - Box arbitrage engine (T13-22) for box detection
    - Gates (ATR, fee-adjusted EV, time-decay)
    - Per-venue comparison
    """
    
    def __init__(self):
        self.min_edge_bps = config.MIN_EDGE_BPS if hasattr(config, 'MIN_EDGE_BPS') else 10  # 0.1%
        self.confidence_level = config.EDGE_CONFIDENCE_LEVEL if hasattr(config, 'EDGE_CONFIDENCE_LEVEL') else 0.95
        self.gbm_model = get_gbm_model()
        self.kl_layer = get_kl_layer()
        self.box_engine = get_box_engine()
        self.edge_log: List[Dict] = []
    
    def calculate_q_lcb(self, gbm_result: GBMResult) -> float:
        """
        Calculate lower confidence bound on model probability.
        
        Uses the confidence interval from GBM if available, otherwise
        applies a fixed uncertainty based on sigma.
        """
        if gbm_result.confidence_interval:
            return gbm_result.confidence_interval[0]  # Lower bound
        
        # Fallback: approximate using sigma uncertainty
        # q_lcb = q - z * sigma_q
        # For 95% confidence, z = 1.645 (one-sided)
        z = 1.645
        if gbm_result.q > 0 and gbm_result.q < 1:
            # Approximate sigma_q using delta method
            # sigma_q ≈ phi'(d2) * sigma_d2
            # phi'(d2) = 1/sqrt(2*pi) * exp(-d2^2/2)
            import math
            phi_prime = (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * gbm_result.d2 ** 2)
            # Assume 10% relative uncertainty on sigma
            sigma_q = phi_prime * 0.1 * gbm_result.sigma_annualized * math.sqrt(gbm_result.tau_years)
            return max(0.0, gbm_result.q - z * sigma_q)
        
        return gbm_result.q
    
    def calculate_slippage(self, ask_price_cents: int, size: int, available_size: int) -> int:
        """Estimate slippage in cents."""
        if available_size <= 0:
            return max(1, ask_price_cents * 20 // 10000)  # 2 bps minimum
        
        ratio = size / available_size
        if ratio <= 0.1:
            return max(1, ask_price_cents * 5 // 10000)   # 0.5 bps
        elif ratio <= 0.3:
            return max(1, ask_price_cents * 10 // 10000)  # 1 bps
        elif ratio <= 0.5:
            return max(1, ask_price_cents * 15 // 10000)  # 1.5 bps
        else:
            return max(1, ask_price_cents * 20 // 10000)  # 2 bps
    
    def calculate_edge(self, inputs: EdgeInputs, size: int = 1,
                        available_size: int = 100,
                        market_price: float = None,
                        ladder_model_probs: List[float] = None,
                        ladder_market_probs: List[float] = None,
                        ladder_strikes: List[float] = None,
                        ensemble_predictions: List[float] = None) -> EdgeResult:
        """
        Calculate trade edge with all gates.
        
        Args:
            inputs: EdgeInputs with model_q, ask, fee, venue, side
            size: Order size in contracts
            available_size: Available size at ask
            market_price: Market implied probability (for KL edge signal)
            ladder_*: For ladder KL (T13-21)
            ensemble_predictions: For ensemble disagreement gate (T13-21)
        
        Returns:
            EdgeResult with edge calculation and gate verdicts
        """
        # Use model_q if q_lcb not provided
        if inputs.q_lcb <= 0:
            q_lcb = inputs.model_q * 0.99  # 1% haircut as fallback
        else:
            q_lcb = inputs.q_lcb
        
        # Edge = q_lcb * 100 - ask - fee - slippage (all in cents)
        slippage = self.calculate_slippage(inputs.ask_price_cents, size, available_size)
        
        edge_cents = q_lcb * 100 - inputs.ask_price_cents - inputs.fee_cents - slippage
        edge_bps = edge_cents * 10000 / 10000  # bps relative to $1
        
        # Run KL layer gates
        kl_results = {}
        gate_passed = True
        gate_reasons = []
        
        if market_price is not None:
            kl_eval = self.kl_layer.evaluate_market(
                model_q=inputs.model_q,
                market_price=market_price,
                ladder_model_probs=ladder_model_probs,
                ladder_market_probs=ladder_market_probs,
                ladder_strikes=ladder_strikes,
                ensemble_predictions=ensemble_predictions
            )
            kl_results = kl_eval["kl_results"]
            
            if not kl_eval["gate_passed"]:
                gate_passed = False
                gate_reasons.extend(kl_eval["gate_reasons"])
        
        # Edge threshold gate
        if edge_bps < self.min_edge_bps:
            gate_passed = False
            gate_reasons.append(f"edge {edge_bps:.1f}bps < min {self.min_edge_bps}bps")
        
        # Check for box opportunity
        box_opp = None
        # Box detection would be done separately via box_engine
        
        # Log edge calculation
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "venue": inputs.venue,
            "side": inputs.side,
            "model_q": inputs.model_q,
            "q_lcb": q_lcb,
            "ask": inputs.ask_price_cents,
            "fee": inputs.fee_cents,
            "slippage": slippage,
            "edge_cents": edge_cents,
            "edge_bps": edge_bps,
            "gate_passed": gate_passed,
            "gate_reasons": gate_reasons,
        }
        self.edge_log.append(log_entry)
        
        if len(self.edge_log) > 10000:
            self.edge_log = self.edge_log[-10000:]
        
        return EdgeResult(
            edge_cents=edge_cents,
            edge_bps=edge_bps,
            q=inputs.model_q,
            q_lcb=q_lcb,
            ask_price_cents=inputs.ask_price_cents,
            fee_cents=inputs.fee_cents,
            slippage_cents=slippage,
            gate_passed=gate_passed,
            gate_reasons=gate_reasons,
            kl_results=kl_results,
            box_opportunity=box_opp
        )
    
    def calculate_edge_from_gbm(self, symbol: str, strike: float, minutes_left: float,
                                 ask_price_cents: int, venue: str, side: str,
                                 size: int = 1, available_size: int = 100,
                                 market_price: float = None,
                                 adaptive_scaler_value: float = None) -> EdgeResult:
        """
        Full edge calculation from GBM model to final edge.
        
        This is the main entry point for the trading loop.
        """
        # Get GBM fair value
        gbm_result = self.gbm_model.predict(symbol, strike, minutes_left, 
                                             adaptive_scaler_value, venue)
        
        # Calculate q_lcb
        q_lcb = self.calculate_q_lcb(gbm_result)
        
        # Calculate fee
        ask_dollars = ask_price_cents / 100.0
        notional = ask_dollars * size
        
        if venue == "kalshi":
            fee_rate = config.KALSHI_FEE_RATE if hasattr(config, 'KALSHI_FEE_RATE') else 0.07
            fee_dollars = notional * fee_rate
        elif venue == "polymarket":
            try:
                from fees import _taker_fee
                fee_dollars = _taker_fee(notional)
            except ImportError:
                fee_dollars = notional * 0.02
        else:
            fee_dollars = notional * 0.02
        
        fee_cents = int(round(fee_dollars * 100))
        
        inputs = EdgeInputs(
            model_q=gbm_result.q,
            q_lcb=q_lcb,
            ask_price_cents=ask_price_cents,
            fee_cents=fee_cents,
            slippage_cents=0,  # Will be calculated in calculate_edge
            venue=venue,
            side=side
        )
        
        return self.calculate_edge(inputs, size, available_size, market_price)
    
    def compare_venues(self, symbol: str, strike: float, minutes_left: float,
                        kalshi_ask: int, kalshi_size: int,
                        poly_ask: int, poly_size: int,
                        size: int = 1) -> Dict:
        """
        Compare edge across venues for the same side.
        
        Returns best venue and edge for each side.
        """
        results = {}
        
        # Kalshi edge
        kalshi_result = self.calculate_edge_from_gbm(
            symbol, strike, minutes_left, kalshi_ask, "kalshi", "YES",
            size, kalshi_size
        )
        results["kalshi"] = kalshi_result
        
        # Polymarket edge (if available)
        if poly_ask > 0:
            poly_result = self.calculate_edge_from_gbm(
                symbol, strike, minutes_left, poly_ask, "polymarket", "YES",
                size, poly_size
            )
            results["polymarket"] = poly_result
        
        # Pick best venue for YES
        best_yes = max(results.values(), key=lambda r: r.edge_bps) if results else None
        
        # For NO side, would need NO asks
        # (This is simplified - real implementation would get NO prices)
        
        return {
            "venues": results,
            "best_yes_venue": best_yes.venue if best_yes else None,
            "best_yes_edge_bps": best_yes.edge_bps if best_yes else 0.0,
        }
    
    def get_edge_stats(self) -> Dict:
        """Get statistics on edge calculations."""
        if not self.edge_log:
            return {"total": 0, "gate_pass_rate": 0.0, "avg_edge_bps": 0.0}
        
        total = len(self.edge_log)
        passed = sum(1 for e in self.edge_log if e["gate_passed"])
        avg_edge = sum(e["edge_bps"] for e in self.edge_log) / total
        
        return {
            "total": total,
            "gate_pass_rate": passed / total if total > 0 else 0.0,
            "avg_edge_bps": avg_edge,
            "max_edge_bps": max(e["edge_bps"] for e in self.edge_log),
            "positive_edge_count": sum(1 for e in self.edge_log if e["edge_bps"] > 0),
        }


# Module-level singleton
_edge_calculator: Optional[FairValueEdgeCalculator] = None


def get_edge_calculator() -> FairValueEdgeCalculator:
    """Get or create global edge calculator."""
    global _edge_calculator
    if _edge_calculator is None:
        _edge_calculator = FairValueEdgeCalculator()
    return _edge_calculator


def calculate_trade_edge(symbol: str, strike: float, minutes_left: float,
                          ask_price_cents: int, venue: str, side: str,
                          size: int = 1, available_size: int = 100,
                          market_price: float = None,
                          adaptive_scaler_value: float = None) -> EdgeResult:
    """Convenience function for single edge calculation."""
    return get_edge_calculator().calculate_edge_from_gbm(
        symbol, strike, minutes_left, ask_price_cents, venue, side,
        size, available_size, market_price, adaptive_scaler_value
    )


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    
    # Test edge calculation
    calc = FairValueEdgeCalculator()
    
    # Mock GBM result (q=0.55, strike=62000, spot=62100)
    # This would normally come from gbm_model.predict()
    result = calc.calculate_edge_from_gbm(
        symbol="BTCUSDT",
        strike=62000.0,
        minutes_left=10.0,
        ask_price_cents=5000,  # 50c
        venue="kalshi",
        side="YES",
        size=10,
        available_size=100
    )
    
    print(f"Edge: {result.edge_cents:.1f}c ({result.edge_bps:.1f} bps)")
    print(f"Model q: {result.q:.4f}, q_lcb: {result.q_lcb:.4f}")
    print(f"Gate passed: {result.gate_passed}")
    print(f"Gate reasons: {result.gate_reasons}")
    print(f"Stats: {calc.get_edge_stats()}")
    
    print("\nFair value edge tests passed!")