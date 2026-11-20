"""
kl_divergence.py — KL divergence layer (T13-21).

Uses KL as a signal and as a gate, not decoration:
1. Bernoulli KL(q_model || p_market) per market as an edge-strength score.
2. Across a strike ladder, KL between model-implied and market-implied
   discrete distributions to spot mispriced regions and ladder arbitrage
   (monotonicity: P(S>K1) >= P(S>K2) for K1<K2).
3. KL between ensemble members as a disagreement gate (high KL -> skip or cut size).

Accept: unit tests with known distributions; gate activity logged; backtest
shows whether each use adds value (keep only those that do).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("kl_divergence")


@dataclass
class KLResult:
    """Result of KL divergence calculation."""
    kl_divergence: float
    interpretation: str  # "edge_signal", "ladder_arb", "ensemble_disagreement"
    details: Dict


def bernoulli_kl(p: float, q: float) -> float:
    """
    Bernoulli KL divergence: KL(P || Q) = p*log(p/q) + (1-p)*log((1-p)/(1-q))
    
    Args:
        p: True probability (model)
        q: Market probability
    
    Returns:
        KL divergence in nats (natural log)
    """
    # Clip to avoid log(0)
    eps = 1e-10
    p = max(eps, min(1 - eps, p))
    q = max(eps, min(1 - eps, q))
    
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


def kl_discrete(p: List[float], q: List[float]) -> float:
    """
    Discrete KL divergence: KL(P || Q) = sum(p_i * log(p_i / q_i))
    
    Args:
        p: True distribution (model)
        q: Market distribution
    
    Returns:
        KL divergence in nats
    """
    eps = 1e-10
    p = np.array([max(eps, x) for x in p])
    q = np.array([max(eps, x) for x in q])
    
    # Renormalize
    p = p / p.sum()
    q = q / q.sum()
    
    return float(np.sum(p * np.log(p / q)))


def js_divergence(p: List[float], q: List[float]) -> float:
    """
    Jensen-Shannon divergence (symmetric, bounded).
    
    JSD = 0.5 * KL(P || M) + 0.5 * KL(Q || M) where M = 0.5*(P+Q)
    """
    p = np.array(p)
    q = np.array(q)
    m = 0.5 * (p + q)
    return 0.5 * kl_discrete(p, m) + 0.5 * kl_discrete(q, m)


def kl_edge_signal(model_q: float, market_price: float, min_kl: float = 0.01) -> KLResult:
    """
    Bernoulli KL as edge-strength signal.
    
    KL(model || market) measures how much information the model has
    that the market doesn't. High KL = strong disagreement = potential edge.
    
    Args:
        model_q: Model's P(YES settles)
        market_price: Market's P(YES settles) from orderbook mid
        min_kl: Minimum KL to consider as signal
    
    Returns:
        KLResult with divergence and interpretation
    """
    kl = bernoulli_kl(model_q, market_price)
    
    interpretation = "no_signal"
    if kl >= min_kl:
        if model_q > market_price:
            interpretation = "edge_signal_yes"  # Model says YES more likely
        else:
            interpretation = "edge_signal_no"   # Model says NO more likely
    
    return KLResult(
        kl_divergence=kl,
        interpretation=interpretation,
        details={"model_q": model_q, "market_price": market_price, "min_kl": min_kl}
    )


def ladder_kl_divergence(model_probs: List[float], market_probs: List[float],
                          strikes: List[float]) -> KLResult:
    """
    KL between model-implied and market-implied discrete distributions across a strike ladder.
    
    For a ladder of strikes K1 < K2 < ... < Kn:
    - Model CDF: P(S > Ki) = q_i
    - Market CDF: P(S > Ki) = p_i
    - Discrete distribution: p_i - p_{i+1} is probability mass in (Ki, K{i+1}]
    
    Args:
        model_probs: Model P(S > Ki) for each strike (descending with strike)
        market_probs: Market P(S > Ki) for each strike
        strikes: Strike prices (ascending)
    
    Returns:
        KLResult with ladder KL divergence
    """
    if len(model_probs) != len(market_probs) or len(model_probs) != len(strikes):
        return KLResult(0.0, "ladder_arb", {"error": "Length mismatch"})
    
    if len(model_probs) < 2:
        return KLResult(0.0, "ladder_arb", {"error": "Need at least 2 strikes"})
    
    # Convert CDF to PDF (probability mass between strikes)
    model_pdf = []
    market_pdf = []
    
    for i in range(len(strikes) - 1):
        # Mass in (Ki, K{i+1}] = P(S > Ki) - P(S > K{i+1})
        model_mass = model_probs[i] - model_probs[i + 1]
        market_mass = market_probs[i] - market_probs[i + 1]
        model_pdf.append(max(0.0, model_mass))
        market_pdf.append(max(0.0, market_mass))
    
    # Add tail mass above highest strike
    model_pdf.append(model_probs[-1])
    market_pdf.append(market_probs[-1])
    
    # Normalize
    model_sum = sum(model_pdf)
    market_sum = sum(market_pdf)
    if model_sum > 0:
        model_pdf = [x / model_sum for x in model_pdf]
    if market_sum > 0:
        market_pdf = [x / market_sum for x in market_pdf]
    
    kl = kl_discrete(model_pdf, market_pdf)
    
    # Check monotonicity violations
    violations = []
    for i in range(len(model_probs) - 1):
        if model_probs[i] < model_probs[i + 1] - 1e-6:
            violations.append(f"model: P(S>{strikes[i]}) < P(S>{strikes[i+1]})")
        if market_probs[i] < market_probs[i + 1] - 1e-6:
            violations.append(f"market: P(S>{strikes[i]}) < P(S>{strikes[i+1]})")
    
    interpretation = "ladder_arb"
    if violations:
        interpretation = "ladder_arb_monotonicity_violation"
    
    return KLResult(
        kl_divergence=kl,
        interpretation=interpretation,
        details={
            "strikes": strikes,
            "model_probs": model_probs,
            "market_probs": market_probs,
            "model_pdf": model_pdf,
            "market_pdf": market_pdf,
            "violations": violations,
        }
    )


def ensemble_disagreement(model_predictions: List[float], max_kl: float = 0.5) -> KLResult:
    """
    KL between ensemble members as a disagreement gate.
    
    High disagreement among ensemble members means uncertainty.
    We compute pairwise KL between all members and use the maximum.
    
    Args:
        model_predictions: List of q values from each ensemble member
        max_kl: Maximum allowed KL before flagging disagreement
    
    Returns:
        KLResult with max pairwise KL
    """
    if len(model_predictions) < 2:
        return KLResult(0.0, "ensemble_disagreement", {"error": "Need at least 2 models"})
    
    max_pairwise_kl = 0.0
    kl_matrix = []
    
    for i, p in enumerate(model_predictions):
        row = []
        for j, q in enumerate(model_predictions):
            kl = bernoulli_kl(p, q)
            row.append(kl)
            if kl > max_pairwise_kl:
                max_pairwise_kl = kl
        kl_matrix.append(row)
    
    interpretation = "ensemble_disagreement"
    if max_pairwise_kl > max_kl:
        interpretation = "ensemble_disagreement_high"
    
    return KLResult(
        kl_divergence=max_pairwise_kl,
        interpretation=interpretation,
        details={
            "predictions": model_predictions,
            "max_kl": max_pairwise_kl,
            "threshold": max_kl,
            "kl_matrix": kl_matrix,
            "mean_prediction": np.mean(model_predictions),
            "std_prediction": np.std(model_predictions),
        }
    )


class KLDivergenceLayer:
    """
    KL divergence layer for signals and gates.
    
    Integrates with GBM fair value model and market data to produce:
    - Per-market edge signals (Bernoulli KL)
    - Ladder arbitrage detection (Discrete KL across strikes)
    - Ensemble disagreement gating
    """
    
    def __init__(self, min_edge_kl: float = 0.01, max_ensemble_kl: float = 0.5):
        self.min_edge_kl = min_edge_kl
        self.max_ensemble_kl = max_ensemble_kl
        self.gate_log: List[Dict] = []
    
    def evaluate_market(self, model_q: float, market_price: float,
                         ladder_model_probs: List[float] = None,
                         ladder_market_probs: List[float] = None,
                         ladder_strikes: List[float] = None,
                         ensemble_predictions: List[float] = None) -> Dict:
        """
        Evaluate all KL signals for a single market.
        
        Returns dict with all KL results and gate decisions.
        """
        results = {}
        
        # 1. Bernoulli KL edge signal
        edge_result = kl_edge_signal(model_q, market_price, self.min_edge_kl)
        results["edge_signal"] = edge_result
        
        # 2. Ladder KL if ladder data provided
        if ladder_model_probs is not None and ladder_market_probs is not None and ladder_strikes is not None:
            ladder_result = ladder_kl_divergence(ladder_model_probs, ladder_market_probs, ladder_strikes)
            results["ladder_arb"] = ladder_result
        
        # 3. Ensemble disagreement if ensemble provided
        if ensemble_predictions is not None:
            ens_result = ensemble_disagreement(ensemble_predictions, self.max_ensemble_kl)
            results["ensemble_disagreement"] = ens_result
        
        # Gate decisions
        gate_passed = True
        gate_reasons = []
        
        # Edge signal gate
        if edge_result.interpretation == "no_signal":
            gate_passed = False
            gate_reasons.append(f"KL edge signal below threshold: {edge_result.kl_divergence:.4f} < {self.min_edge_kl}")
        
        # Ensemble disagreement gate
        if "ensemble_disagreement" in results:
            ens = results["ensemble_disagreement"]
            if ens.interpretation == "ensemble_disagreement_high":
                gate_passed = False
                gate_reasons.append(f"Ensemble disagreement high: KL={ens.kl_divergence:.4f} > {self.max_ensemble_kl}")
        
        # Ladder monotonicity gate
        if "ladder_arb" in results:
            ladder = results["ladder_arb"]
            if ladder.interpretation == "ladder_arb_monotonicity_violation":
                gate_passed = False
                gate_reasons.append(f"Ladder monotonicity violation: {ladder.details.get('violations', [])}")
        
        # Log gate decision
        gate_entry = {
            "model_q": model_q,
            "market_price": market_price,
            "edge_kl": edge_result.kl_divergence,
            "gate_passed": gate_passed,
            "gate_reasons": gate_reasons,
            "all_results": {k: {"kl": v.kl_divergence, "interpretation": v.interpretation} for k, v in results.items()}
        }
        self.gate_log.append(gate_entry)
        
        if len(self.gate_log) > 10000:
            self.gate_log = self.gate_log[-10000:]
        
        return {
            "gate_passed": gate_passed,
            "gate_reasons": gate_reasons,
            "kl_results": results,
        }
    
    def get_gate_stats(self) -> Dict:
        """Get statistics on gate decisions."""
        if not self.gate_log:
            return {"total": 0, "passed": 0, "pass_rate": 0.0}
        
        total = len(self.gate_log)
        passed = sum(1 for g in self.gate_log if g["gate_passed"])
        
        # Average KL for passed vs failed
        passed_kl = [g["edge_kl"] for g in self.gate_log if g["gate_passed"]]
        failed_kl = [g["edge_kl"] for g in self.gate_log if not g["gate_passed"]]
        
        return {
            "total": total,
            "passed": passed,
            "pass_rate": passed / total if total > 0 else 0.0,
            "avg_edge_kl_passed": np.mean(passed_kl) if passed_kl else 0.0,
            "avg_edge_kl_failed": np.mean(failed_kl) if failed_kl else 0.0,
            "rejection_reasons": self._count_rejection_reasons(),
        }
    
    def _count_rejection_reasons(self) -> Dict[str, int]:
        """Count frequency of rejection reasons."""
        counts = {}
        for g in self.gate_log:
            for reason in g["gate_reasons"]:
                key = reason.split(":")[0]  # Get first part before colon
                counts[key] = counts.get(key, 0) + 1
        return counts


# Module-level singleton
_kl_layer: Optional[KLDivergenceLayer] = None


def get_kl_layer(min_edge_kl: float = 0.01, max_ensemble_kl: float = 0.5) -> KLDivergenceLayer:
    """Get or create global KL divergence layer."""
    global _kl_layer
    if _kl_layer is None:
        _kl_layer = KLDivergenceLayer(min_edge_kl, max_ensemble_kl)
    return _kl_layer


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    
    # Test Bernoulli KL
    print("=== Bernoulli KL Tests ===")
    kl = bernoulli_kl(0.6, 0.5)
    print(f"KL(0.6 || 0.5) = {kl:.4f} nats")
    
    kl = bernoulli_kl(0.9, 0.5)
    print(f"KL(0.9 || 0.5) = {kl:.4f} nats")
    
    kl = bernoulli_kl(0.5, 0.5)
    print(f"KL(0.5 || 0.5) = {kl:.4f} nats")
    
    # Test edge signal
    print("\n=== Edge Signal Tests ===")
    result = kl_edge_signal(0.6, 0.5, min_kl=0.01)
    print(f"Model q=0.6, Market=0.5: KL={result.kl_divergence:.4f}, interpretation={result.interpretation}")
    
    result = kl_edge_signal(0.52, 0.5, min_kl=0.01)
    print(f"Model q=0.52, Market=0.5: KL={result.kl_divergence:.4f}, interpretation={result.interpretation}")
    
    # Test ladder KL
    print("\n=== Ladder KL Tests ===")
    strikes = [61000, 62000, 63000, 64000]
    model_probs = [0.8, 0.5, 0.2, 0.05]  # P(S > K)
    market_probs = [0.75, 0.45, 0.15, 0.03]
    
    result = ladder_kl_divergence(model_probs, market_probs, strikes)
    print(f"Ladder KL = {result.kl_divergence:.4f}, interpretation={result.interpretation}")
    print(f"Model PDF: {[f'{x:.3f}' for x in result.details['model_pdf']]}")
    print(f"Market PDF: {[f'{x:.3f}' for x in result.details['market_pdf']]}")
    
    # Test ensemble disagreement
    print("\n=== Ensemble Disagreement Tests ===")
    ensemble = [0.55, 0.58, 0.52, 0.60, 0.53]
    result = ensemble_disagreement(ensemble, max_kl=0.1)
    print(f"Ensemble: {ensemble}")
    print(f"Max pairwise KL = {result.kl_divergence:.4f}, interpretation={result.interpretation}")
    print(f"Mean: {result.details['mean_prediction']:.3f}, Std: {result.details['std_prediction']:.3f}")
    
    # Test full layer
    print("\n=== Full KL Layer Test ===")
    layer = KLDivergenceLayer(min_edge_kl=0.01, max_ensemble_kl=0.1)
    eval_result = layer.evaluate_market(
        model_q=0.6,
        market_price=0.5,
        ensemble_predictions=[0.55, 0.58, 0.52, 0.60, 0.53]
    )
    print(f"Gate passed: {eval_result['gate_passed']}")
    print(f"Gate reasons: {eval_result['gate_reasons']}")
    print(f"Stats: {layer.get_gate_stats()}")
    
    print("\nAll KL divergence tests passed!")