"""
kl_divergence.py — KL divergence layer for edge scoring and gating (T13-21, T15-22).

Provides:
1. Bernoulli KL(q_model || p_market) per market as edge-strength score.
2. Ladder KL between model-implied and market-implied distributions.
3. Ensemble KL disagreement gate.
4. Symmetric KL divergence for cross-venue comparison.
5. PSI (Population Stability Index) for drift detection.
6. Log-loss-based ensemble weights.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("kl_divergence")

# Small epsilon for clipping probabilities to avoid log(0)
_KL_EPS = 1e-10


def _clip_prob(p: float) -> float:
    """Clip probability to [eps, 1-eps] for safe log operations."""
    return max(_KL_EPS, min(1.0 - _KL_EPS, p))


# ─── Bernoulli KL ──────────────────────────────────────────────────────────────


def bernoulli_kl(q: float, p: float) -> float:
    """
    Bernoulli KL divergence: KL(Ber(q) || Ber(p)) = q*log(q/p) + (1-q)*log((1-q)/(1-p))

    Returns 0 if q == p, infinity if p in {0,1} and q != p.
    Uses epsilon clipping for numerical stability for interior values.
    """
    # Handle edge cases that should return infinity
    if p <= 0.0 or p >= 1.0:
        if q <= 0.0 and p <= 0.0:
            return 0.0
        if q >= 1.0 and p >= 1.0:
            return 0.0
        return float("inf")
    if q <= 0.0:
        return math.log(1 / (1 - p))
    if q >= 1.0:
        return math.log(1 / p)
    if q == p:
        return 0.0
    return q * math.log(q / p) + (1 - q) * math.log((1 - q) / (1 - p))


def bernoulli_kl_safe(q: float, p: float) -> float:
    """
    Bernoulli KL with safe handling of edge cases (returns large finite value instead of inf).
    """
    if p <= 0.0 or p >= 1.0 or q <= 0.0 or q >= 1.0:
        return 10.0  # Large but finite penalty
    return bernoulli_kl(q, p)


def kl_edge_score(q_model: float, p_market: float) -> float:
    """
    Edge strength from KL divergence.
    Higher KL means model strongly disagrees with market price.
    """
    if q_model is None or p_market is None:
        return 0.0
    return bernoulli_kl(q_model, p_market)


# ─── Symmetric KL ───────────────────────────────────────────────────────────────


def symmetric_kl(p: float, q: float) -> float:
    """
    Symmetric KL divergence: 0.5 * (KL(p||q) + KL(q||p)).
    
    Also known as Jensen-Shannon divergence (up to a factor) or "symmetrized KL".
    Useful for cross-venue comparison where direction doesn't matter.
    
    Args:
        p: first probability (e.g., Kalshi price)
        q: second probability (e.g., Polymarket price)
    
    Returns symmetric KL in [0, ~1.386] (max at p=0, q=1 or vice versa).
    """
    p = _clip_prob(p)
    q = _clip_prob(q)
    if p == q:
        return 0.0
    kl_pq = p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))
    kl_qp = q * math.log(q / p) + (1 - q) * math.log((1 - q) / (1 - p))
    return 0.5 * (kl_pq + kl_qp)


def symmetric_kl_detailed(p: float, q: float) -> Tuple[float, float, float]:
    """
    Symmetric KL with detailed components.
    
    Returns (sym_kl, kl_pq, kl_qp) where:
        kl_pq = KL(p || q)
        kl_qp = KL(q || p)
        sym_kl = 0.5 * (kl_pq + kl_qp)
    """
    p = _clip_prob(p)
    q = _clip_prob(q)
    if p == q:
        return 0.0, 0.0, 0.0
    kl_pq = p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))
    kl_qp = q * math.log(q / p) + (1 - q) * math.log((1 - q) / (1 - p))
    return 0.5 * (kl_pq + kl_qp), kl_pq, kl_qp


# ─── Ladder KL (strike ladder) ─────────────────────────────────────────────────


def ladder_kl(
    model_probs: Dict[float, float],  # strike -> P(S > K)
    market_probs: Dict[float, float],  # strike -> market-implied P(S > K)
) -> float:
    """
    KL between model-implied and market-implied discrete distributions over strikes.
    Monotonicity constraint: P(S > K1) >= P(S > K2) for K1 < K2.

    Returns sum of Bernoulli KLs across strikes, +inf if monotonicity violated.
    """
    strikes = sorted(set(model_probs.keys()) | set(market_probs.keys()))
    if not strikes:
        return 0.0

    total_kl = 0.0
    prev_q = None
    prev_p = None
    for k in strikes:
        q = model_probs.get(k)
        p = market_probs.get(k)
        if q is None or p is None:
            continue
        if prev_q is not None and q > prev_q + 1e-12:
            return float("inf")  # monotonicity violated
        if prev_p is not None and p > prev_p + 1e-12:
            return float("inf")
        total_kl += bernoulli_kl(q, p)
        prev_q = q
        prev_p = p
    return total_kl


def ladder_mispricing(
    model_probs: Dict[float, float],
    market_probs: Dict[float, float],
) -> List[Tuple[float, float, float]]:
    """
    Identify strikes where model and market disagree most.
    Returns list of (strike, model_q, market_p) sorted by KL contribution descending.
    """
    strikes = sorted(set(model_probs.keys()) | set(market_probs.keys()))
    contributions = []
    for k in strikes:
        q = model_probs.get(k)
        p = market_probs.get(k)
        if q is None or p is None:
            continue
        kl = bernoulli_kl(q, p)
        if kl > 0:
            contributions.append((k, q, p, kl))
    contributions.sort(key=lambda x: x[3], reverse=True)
    return [(k, q, p) for k, q, p, _ in contributions]


# ─── Ensemble KL (disagreement gate) ───────────────────────────────────────────


def ensemble_kl(probs: List[float]) -> float:
    """
    Mean pairwise KL between ensemble members.
    High KL -> high disagreement -> skip or cut size.
    """
    n = len(probs)
    if n < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += bernoulli_kl(probs[i], probs[j])
            count += 1
    return total / count if count else 0.0


def ensemble_disagreement_gate(
    probs: List[float],
    threshold: float = 0.1,
) -> Tuple[bool, float]:
    """
    Returns (allow, avg_kl). allow=False if avg pairwise KL > threshold.
    """
    kl = ensemble_kl(probs)
    return (kl <= threshold, kl)


def ensemble_log_loss_weights(
    probs: List[float],
    outcomes: List[int],
    temperature: float = 1.0,
) -> List[float]:
    """
    Compute ensemble weights from log-loss on historical outcomes.
    
    Lower log-loss -> higher weight. Weights are softmax(-log_loss / temperature).
    
    Args:
        probs: list of predicted probabilities from each ensemble member
        outcomes: list of binary outcomes (0 or 1)
        temperature: softmax temperature (higher = more uniform)
    
    Returns normalized weights summing to 1.
    """
    import numpy as np
    
    if len(probs) != len(outcomes):
        raise ValueError("probs and outcomes must have same length")
    
    log_losses = []
    for p in probs:
        p_clipped = np.clip(p, _KL_EPS, 1 - _KL_EPS)
        ll = -np.mean(outcomes * np.log(p_clipped) + (1 - outcomes) * np.log(1 - p_clipped))
        log_losses.append(ll)
    
    log_losses = np.array(log_losses)
    weights = np.exp(-log_losses / temperature)
    return (weights / weights.sum()).tolist()


# ─── PSI (Population Stability Index) for drift detection ──────────────────────


def psi(expected: List[float], actual: List[float], bins: int = 10) -> float:
    """
    Population Stability Index (PSI) for drift detection.
    
    Compares distribution of expected (training) vs actual (live) feature values.
    PSI < 0.1: no significant drift
    PSI 0.1-0.25: moderate drift
    PSI > 0.25: significant drift (retrain/fallback)
    
    Args:
        expected: reference distribution (e.g., training feature values)
        actual: current distribution (e.g., live feature values)
        bins: number of bins for histogram
    
    Returns PSI value.
    """
    import numpy as np
    
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
    
    # Create bins from expected distribution
    try:
        _, bin_edges = np.histogram(expected, bins=bins)
        expected_hist, _ = np.histogram(expected, bins=bin_edges)
        actual_hist, _ = np.histogram(actual, bins=bin_edges)
    except Exception:
        return 0.0
    
    # Convert to probabilities with epsilon
    expected_pct = expected_hist / len(expected)
    actual_pct = actual_hist / len(actual)
    
    expected_pct = np.clip(expected_pct, _KL_EPS, 1.0)
    actual_pct = np.clip(actual_pct, _KL_EPS, 1.0)
    
    # PSI = sum((actual% - expected%) * ln(actual% / expected%))
    psi_val = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi_val)


# ─── Integration helpers ───────────────────────────────────────────────────────


def market_prob_from_price(price_cents: int, side: str = "yes") -> float:
    """Convert market price (cents) to probability p."""
    if side == "yes":
        return price_cents / 100.0
    return 1.0 - price_cents / 100.0


# Aliases for backward compatibility with forecaster.py
kl_edge_signal = kl_edge_score
ladder_kl_divergence = ladder_kl
ensemble_disagreement = ensemble_kl


def kl_edge_score(q_model: float, p_market: float) -> float:
    """
    Edge strength from KL divergence.
    Higher KL means model strongly disagrees with market price.
    """
    if q_model is None or p_market is None:
        return 0.0
    return bernoulli_kl(q_model, p_market)


# ─── Ladder KL (strike ladder) ─────────────────────────────────────────────────


def ladder_kl(
    model_probs: Dict[float, float],  # strike -> P(S > K)
    market_probs: Dict[float, float],  # strike -> market-implied P(S > K)
) -> float:
    """
    KL between model-implied and market-implied discrete distributions over strikes.
    Monotonicity constraint: P(S > K1) >= P(S > K2) for K1 < K2.

    Returns sum of Bernoulli KLs across strikes, +inf if monotonicity violated.
    """
    strikes = sorted(set(model_probs.keys()) | set(market_probs.keys()))
    if not strikes:
        return 0.0

    total_kl = 0.0
    prev_q = None
    prev_p = None
    for k in strikes:
        q = model_probs.get(k)
        p = market_probs.get(k)
        if q is None or p is None:
            continue
        if prev_q is not None and q > prev_q + 1e-12:
            return float("inf")  # monotonicity violated
        if prev_p is not None and p > prev_p + 1e-12:
            return float("inf")
        total_kl += bernoulli_kl(q, p)
        prev_q = q
        prev_p = p
    return total_kl


def ladder_mispricing(
    model_probs: Dict[float, float],
    market_probs: Dict[float, float],
) -> List[Tuple[float, float, float]]:
    """
    Identify strikes where model and market disagree most.
    Returns list of (strike, model_q, market_p) sorted by KL contribution descending.
    """
    strikes = sorted(set(model_probs.keys()) | set(market_probs.keys()))
    contributions = []
    for k in strikes:
        q = model_probs.get(k)
        p = market_probs.get(k)
        if q is None or p is None:
            continue
        kl = bernoulli_kl(q, p)
        if kl > 0:
            contributions.append((k, q, p, kl))
    contributions.sort(key=lambda x: x[3], reverse=True)
    return [(k, q, p) for k, q, p, _ in contributions]


# ─── Ensemble KL (disagreement gate) ───────────────────────────────────────────


def ensemble_kl(probs: List[float]) -> float:
    """
    Mean pairwise KL between ensemble members.
    High KL -> high disagreement -> skip or cut size.
    """
    n = len(probs)
    if n < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += bernoulli_kl(probs[i], probs[j])
            count += 1
    return total / count if count else 0.0


def ensemble_disagreement_gate(
    probs: List[float],
    threshold: float = 0.1,
) -> Tuple[bool, float]:
    """
    Returns (allow, avg_kl). allow=False if avg pairwise KL > threshold.
    """
    kl = ensemble_kl(probs)
    return (kl <= threshold, kl)


# ─── Integration helpers ───────────────────────────────────────────────────────


def market_prob_from_price(price_cents: int, side: str = "yes") -> float:
    """Convert market price (cents) to probability p."""
    if side == "yes":
        return price_cents / 100.0
    return 1.0 - price_cents / 100.0


# Aliases for backward compatibility with forecaster.py
kl_edge_signal = kl_edge_score
ladder_kl_divergence = ladder_kl
ensemble_disagreement = ensemble_kl


class KLDivergenceLayer:
    """Compatibility wrapper for forecaster.py."""

    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold

    def compute(self, q_model: float, price_cents: int, side: str = "yes",
                ensemble: Optional[List[float]] = None) -> Dict[str, float]:
        return compute_kl_gates(q_model, price_cents, side, ensemble)


def compute_kl_gates(
    q_model: float,
    entry_price_cents: int,
    side: str = "yes",
    ensemble_probs: Optional[List[float]] = None,
    ladder_probs: Optional[Dict[float, float]] = None,
    market_ladder: Optional[Dict[float, float]] = None,
) -> Dict[str, float]:
    """
    Compute all KL-based gate metrics for a candidate.

    Returns dict with keys:
    - kl_edge: Bernoulli KL(q_model || p_market)
    - ladder_kl: KL across strike ladder (inf if monotonicity violated)
    - ensemble_kl: mean pairwise KL among ensemble members
    - ensemble_allow: bool from ensemble_disagreement_gate
    """
    p_market = market_prob_from_price(entry_price_cents, side)
    out = {
        "kl_edge": kl_edge_score(q_model, p_market),
        "ladder_kl": 0.0,
        "ensemble_kl": 0.0,
        "ensemble_allow": True,
    }
    if ladder_probs and market_ladder:
        out["ladder_kl"] = ladder_kl(ladder_probs, market_ladder)
    if ensemble_probs:
        out["ensemble_kl"] = ensemble_kl(ensemble_probs)
        out["ensemble_allow"] = ensemble_disagreement_gate(ensemble_probs)[0]
    return out


# ─── CLI / demo ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Basic sanity checks
    print("Bernoulli KL tests:")
    print(f"  KL(0.5 || 0.5) = {bernoulli_kl(0.5, 0.5)} (expected 0)")
    print(f"  KL(0.6 || 0.5) = {bernoulli_kl(0.6, 0.5):.4f}")
    print(f"  KL(0.9 || 0.1) = {bernoulli_kl(0.9, 0.1):.4f}")

    print("\nEnsemble KL tests:")
    print(f"  [0.5, 0.5, 0.5] -> {ensemble_kl([0.5, 0.5, 0.5])} (expected 0)")
    print(f"  [0.3, 0.7] -> {ensemble_kl([0.3, 0.7]):.4f}")

    print("\nLadder KL tests:")
    model = {90000: 0.8, 95000: 0.5, 100000: 0.2}
    market = {90000: 0.75, 95000: 0.55, 100000: 0.3}
    print(f"  Ladder KL = {ladder_kl(model, market):.4f}")

    print("\nMispricing detection:")
    for k, q, p in ladder_mispricing(model, market):
        print(f"  Strike {k}: model={q:.2f} market={p:.2f}")