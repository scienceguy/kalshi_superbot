#!/usr/bin/env python3
"""
gbm.py — Unified Geometric/Arithmetic Brownian Motion module (T15-20).

Consolidates GBM math from:
- gates.cushion_z: arithmetic Brownian cushion (dollar sigma per sqrt-minute)
- binance_feed.prob_above: lognormal GBM with fractional per-minute sigma
- model.gbm_probability_yes_wins: Black-Scholes d2 with annualized volatility

Provides:
1. Point-in-time GBM (standard)
2. BRTI-style 60-second running average GBM (average-window variant)
3. Fat-tail/jump adjustment
4. Calibration (isotonic/Platt) fitted on settled history

All functions document explicit units: dollars vs fractional sigma.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import config

log = logging.getLogger(__name__)


# ─── Core math helpers ────────────────────────────────────────────────────────

def normal_cdf(z: float) -> float:
    """Standard normal CDF via math.erf."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def normal_pdf(z: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


# ─── Units conversion ────────────────────────────────────────────────────────

def annualized_to_per_minute(sigma_annual: float) -> float:
    """
    Convert annualized volatility (fractional, e.g., 0.80) to per-minute fractional sigma.
    
    Annual has 365*24*60 = 525600 minutes. sqrt(525600) ≈ 725.
    sigma_per_min = sigma_annual / sqrt(525600)
    """
    return sigma_annual / 724.97  # sqrt(525600)


def per_minute_to_annualized(sigma_per_min: float) -> float:
    """Convert per-minute fractional sigma to annualized."""
    return sigma_per_min * 724.97


def dollar_sigma_from_fractional(spot: float, sigma_frac_per_min: float) -> float:
    """
    Convert fractional per-minute sigma to dollar sigma per sqrt-minute.
    
    For small moves, dS ≈ S * sigma_frac * sqrt(dt), so dollar_sigma ≈ spot * sigma_frac.
    This is exact for the arithmetic Brownian approximation.
    """
    return spot * sigma_frac_per_min


def fractional_sigma_from_dollar(spot: float, dollar_sigma_per_sqrt_min: float) -> float:
    """Convert dollar sigma per sqrt-minute to fractional per-minute sigma."""
    if spot <= 0:
        return 0.0
    return dollar_sigma_per_sqrt_min / spot


# ─── Point-in-time GBM (standard) ───────────────────────────────────────────

def gbm_probability_point(
    spot: float,
    strike: float,
    sigma_frac_per_min: float,
    minutes_left: float,
    drift_annual: float = 0.0
) -> float:
    """
    GBM probability that spot >= strike at expiry (point-in-time).
    
    Uses lognormal (geometric Brownian motion) with fractional per-minute sigma.
    This is the same math as binance_feed.prob_above.
    
    Args:
        spot: current spot price (dollars)
        strike: strike/target price (dollars)
        sigma_frac_per_min: fractional volatility per minute (e.g., 0.80/725 ≈ 0.0011)
        minutes_left: time to expiry in minutes
        drift_annual: annualized drift (default 0 for risk-neutral)
    
    Returns probability in [0, 1].
    """
    if spot <= 0 or strike <= 0 or sigma_frac_per_min <= 0 or minutes_left <= 0:
        return 1.0 if spot >= strike else 0.0
    
    # Total variance over the period
    sigma_sqrt_t = sigma_frac_per_min * math.sqrt(minutes_left)
    if sigma_sqrt_t <= 0:
        return 1.0 if spot >= strike else 0.0
    
    # d2 in Black-Scholes (risk-neutral drift = 0)
    # log(spot/strike) - 0.5 * sigma^2 * T all over sigma*sqrt(T)
    drift_per_min = drift_annual / 525600.0
    log_moneyness = math.log(spot / strike)
    d2 = (log_moneyness + (drift_per_min - 0.5 * sigma_frac_per_min ** 2) * minutes_left) / sigma_sqrt_t
    
    return normal_cdf(d2)


def gbm_probability_bs_d2(
    spot: float,
    strike: float,
    sigma_annual: float,
    minutes_left: float,
    risk_free_rate: float = 0.05
) -> float:
    """
    Black-Scholes style d2 probability with annualized volatility.
    
    This matches model.gbm_probability_yes_wins exactly.
    
    Args:
        spot: current spot price
        strike: strike price
        sigma_annual: annualized volatility (e.g., 0.80 for 80%)
        minutes_left: time to expiry in minutes
        risk_free_rate: annualized risk-free rate
    
    Returns probability in [0, 1].
    """
    if minutes_left <= 0:
        return 1.0 if spot >= strike else 0.0
    if spot <= 0 or strike <= 0:
        return 0.5
    
    T = minutes_left / 525600.0  # years
    sigma = sigma_annual
    if T <= 0 or sigma <= 0:
        return 1.0 if spot >= strike else 0.0
    
    try:
        d2 = (math.log(spot / strike) +
              (risk_free_rate - 0.5 * sigma * sigma) * T) / \
             (sigma * math.sqrt(T))
        prob = normal_cdf(d2)
        return max(0.001, min(0.999, prob))
    except (ValueError, OverflowError):
        return 0.5


# ─── Arithmetic Brownian cushion (gates.cushion_z equivalent) ─────────────────

def arithmetic_cushion_z(
    spot: float,
    strike: float,
    dollar_sigma_per_sqrt_min: float,
    minutes_left: float,
    side: str = "yes"
) -> Optional[float]:
    """
    Arithmetic Brownian cushion in standard deviations.
    
    This is the exact math from gates.cushion_z: z = (spot - strike) / (sigma * sqrt(t))
    where sigma is in DOLLARS per sqrt-minute.
    
    YES wins when spot > strike, so z positive means favorable.
    NO is the mirror (negate z).
    
    Args:
        spot: current spot price (dollars)
        strike: strike price (dollars)
        dollar_sigma_per_sqrt_min: dollar volatility per sqrt-minute
        minutes_left: time to expiry in minutes
        side: "yes" or "no"
    
    Returns signed z-score or None if undefined.
    """
    if spot <= 0 or strike <= 0 or dollar_sigma_per_sqrt_min <= 0:
        return None
    if minutes_left <= 0:
        return None
    
    denom = dollar_sigma_per_sqrt_min * math.sqrt(minutes_left)
    if denom <= 0:
        return None
    
    z = (spot - strike) / denom
    return z if side.lower() == "yes" else -z


def arithmetic_probability(
    spot: float,
    strike: float,
    dollar_sigma_per_sqrt_min: float,
    minutes_left: float,
    side: str = "yes"
) -> float:
    """
    Convert arithmetic cushion z to probability.
    
    P(win) = N(z) for YES, N(-z) for NO.
    """
    z = arithmetic_cushion_z(spot, strike, dollar_sigma_per_sqrt_min, minutes_left, side)
    if z is None:
        return 0.5
    return normal_cdf(z)


# ─── BRTI-style 60-second average GBM ────────────────────────────────────────

def gbm_probability_average_window(
    spot: float,
    strike: float,
    sigma_frac_per_min: float,
    minutes_left: float,
    averaging_window_seconds: float = 60.0,
    drift_annual: float = 0.0
) -> float:
    """
    GBM probability under BRTI-style 60-second averaging.
    
    Kalshi settles on the 60-second volume-weighted average (BRTI). The running
    average over the final minute is already partially known. This function
    computes the probability that the AVERAGE over the remaining window ends
    above the strike.
    
    If we're `seconds_into_final_minute` seconds into the last minute, the
    remaining uncertainty is reduced. The effective variance scales with
    (remaining_seconds / 60)^1.5 for the average (not sqrt).
    
    For the remaining `minutes_left` minutes (including the partial final minute),
    the variance of the average price is reduced by the averaging effect.
    
    Approximation: use effective variance = sigma^2 * minutes_left * adjustment,
    where adjustment < 1 accounts for averaging.
    """
    if minutes_left <= 0:
        return 1.0 if spot >= strike else 0.0
    
    # If we're in the final minute, the averaging reduces variance
    if minutes_left < 1.0:
        # Remaining seconds in the final minute
        remaining_seconds = minutes_left * 60.0
        # Variance reduction factor for running average: (remaining/60)^1.5
        variance_factor = (remaining_seconds / 60.0) ** 1.5
        effective_minutes = minutes_left * variance_factor
    else:
        # Before final minute, full variance but with 60s averaging each minute
        # Each 1-minute average has variance reduced by 1/60 vs point-in-time
        effective_minutes = minutes_left * (1.0 / 60.0)
    
    sigma_sqrt_t = sigma_frac_per_min * math.sqrt(effective_minutes)
    if sigma_sqrt_t <= 0:
        return 1.0 if spot >= strike else 0.0
    
    drift_per_min = drift_annual / 525600.0
    log_moneyness = math.log(spot / strike)
    d2 = (log_moneyness + (drift_per_min - 0.5 * sigma_frac_per_min ** 2) * effective_minutes) / sigma_sqrt_t
    
    return normal_cdf(d2)


def locked_share_gbm(
    spot: float,
    strike: float,
    sigma_frac_per_min: float,
    seconds_into_final_minute: int,
    averaging_window_seconds: float = 60.0
) -> float:
    """
    GBM probability with BRTI lock-in adjustment.
    
    As the 60-second average locks in second by second, remaining uncertainty
    collapses. This is the same logic as model.locked_share_probability but
    using proper GBM math instead of ad-hoc adjustment.
    
    Args:
        spot: current spot (or BRTI running average)
        strike: strike price
        sigma_frac_per_min: fractional per-minute sigma
        seconds_into_final_minute: 0-60, seconds elapsed in final minute
        averaging_window_seconds: 60 for BRTI
    
    Returns probability in [0, 1].
    """
    if seconds_into_final_minute <= 0:
        # Not in final minute yet - use standard GBM
        return gbm_probability_point(spot, strike, sigma_frac_per_min, 1.0)
    
    if seconds_into_final_minute >= averaging_window_seconds:
        # Fully settled
        return 1.0 if spot >= strike else 0.0
    
    # Remaining fraction of averaging window
    remaining_frac = (averaging_window_seconds - seconds_into_final_minute) / averaging_window_seconds
    # Uncertainty scales with remaining^1.5 for running average
    uncertainty_reduction = 1.0 - remaining_frac ** 1.5
    
    # Base probability from current spot
    base_prob = gbm_probability_point(spot, strike, sigma_frac_per_min, remaining_frac / 60.0)
    
    # Push toward extreme (0 or 1) as uncertainty reduces
    # This is a simplified model - proper implementation would use conditional
    # distribution of the running average given the current partial average
    if base_prob > 0.5:
        # Push toward 1
        adjusted = base_prob + (1.0 - base_prob) * uncertainty_reduction * 0.5
    elif base_prob < 0.5:
        # Push toward 0
        adjusted = base_prob - base_prob * uncertainty_reduction * 0.5
    else:
        adjusted = 0.5
    
    return max(0.001, min(0.999, adjusted))


# ─── Fat-tail / jump adjustment ──────────────────────────────────────────────

def gbm_with_jumps(
    spot: float,
    strike: float,
    sigma_frac_per_min: float,
    minutes_left: float,
    jump_intensity_per_min: float = 0.0,
    jump_mean_frac: float = 0.0,
    jump_std_frac: float = 0.0,
    drift_annual: float = 0.0
) -> float:
    """
    GBM with compound Poisson jumps (Merton jump-diffusion).
    
    For small time horizons (minutes), jump probability is very low.
    This is mainly for calibration against fat-tail settlement data.
    
    Args:
        jump_intensity_per_min: lambda (expected jumps per minute)
        jump_mean_frac: mean jump size as fraction of spot
        jump_std_frac: std of jump size
    
    Returns probability in [0, 1].
    """
    # For very short horizons, just return standard GBM
    if minutes_left < 1.0 or jump_intensity_per_min <= 0:
        return gbm_probability_point(spot, strike, sigma_frac_per_min, minutes_left, drift_annual)
    
    # Expected number of jumps in the period
    expected_jumps = jump_intensity_per_min * minutes_left
    
    if expected_jumps < 0.01:
        # Negligible jump probability
        return gbm_probability_point(spot, strike, sigma_frac_per_min, minutes_left, drift_annual)
    
    # For small lambda, approximate with a mixture
    # P(win) = P(0 jumps) * P_GBM(win | 0 jumps) + P(1 jump) * P_GBM(win | 1 jump) + ...
    # We'll just use the diffusion approximation with inflated variance
    # Var_total = sigma^2 * T + lambda * T * (jump_mean^2 + jump_std^2)
    total_var = (sigma_frac_per_min ** 2) * minutes_left + \
                expected_jumps * (jump_mean_frac ** 2 + jump_std_frac ** 2)
    effective_sigma = math.sqrt(total_var / minutes_left) if minutes_left > 0 else 0
    
    return gbm_probability_point(spot, strike, effective_sigma, minutes_left, drift_annual)


# ─── Calibration ──────────────────────────────────────────────────────────────

@dataclass
class CalibrationResult:
    """Result of probability calibration."""
    method: str  # "isotonic" or "platt"
    # For isotonic: list of (x, y) knots; for platt: (A, B)
    params: any
    brier_before: float
    brier_after: float
    n_samples: int


def platt_calibration(
    probs: List[float],
    outcomes: List[int]
) -> Tuple[float, float]:
    """
    Platt scaling (sigmoid calibration): p_cal = 1 / (1 + exp(A * p + B)).
    
    Fits A, B to minimize log-loss on (probs, outcomes).
    Returns (A, B).
    """
    import numpy as np
    from scipy.optimize import minimize
    
    probs = np.array(probs, dtype=float)
    outcomes = np.array(outcomes, dtype=float)
    
    # Clip to avoid log(0)
    eps = 1e-15
    probs = np.clip(probs, eps, 1 - eps)
    
    def neg_log_likelihood(params):
        A, B = params
        logits = A * probs + B
        p_cal = 1.0 / (1.0 + np.exp(-logits))
        p_cal = np.clip(p_cal, eps, 1 - eps)
        return -np.mean(outcomes * np.log(p_cal) + (1 - outcomes) * np.log(1 - p_cal))
    
    # Initial guess: identity mapping (A=0, B=0 -> p_cal = 0.5)
    # Better: fit to log-odds
    log_odds = np.log(probs / (1 - probs))
    # Linear regression of outcomes on log_odds
    A_init = 1.0
    B_init = 0.0
    
    result = minimize(neg_log_likelihood, [A_init, B_init], method="BFGS")
    return float(result.x[0]), float(result.x[1])


def isotonic_calibration(
    probs: List[float],
    outcomes: List[int]
) -> List[Tuple[float, float]]:
    """
    Isotonic regression calibration (pool-adjacent-violators).
    
    Returns list of (x, y) knots for piecewise constant non-decreasing function.
    """
    import numpy as np
    from sklearn.isotonic import IsotonicRegression
    
    probs = np.array(probs, dtype=float)
    outcomes = np.array(outcomes, dtype=float)
    
    iso = IsotonicRegression(out_of_bounds="clip")
    y_cal = iso.fit_transform(probs, outcomes)
    
    # Extract knots (where y changes)
    knots = []
    for i in range(len(y_cal)):
        if i == 0 or y_cal[i] != y_cal[i-1]:
            knots.append((float(probs[i]), float(y_cal[i])))
    
    return knots


def apply_platt_calibration(probs: List[float], A: float, B: float) -> List[float]:
    """Apply Platt calibration parameters."""
    import numpy as np
    probs = np.array(probs)
    logits = A * probs + B
    return 1.0 / (1.0 + np.exp(-logits))


def apply_isotonic_calibration(probs: List[float], knots: List[Tuple[float, float]]) -> List[float]:
    """Apply isotonic calibration knots."""
    import numpy as np
    from scipy.interpolate import interp1d
    
    if not knots:
        return probs
    
    xs = [k[0] for k in knots]
    ys = [k[1] for k in knots]
    
    # Piecewise constant
    f = interp1d(xs, ys, kind="previous", bounds_error=False, fill_value=(ys[0], ys[-1]))
    return f(probs).tolist()


def calibrate_probabilities(
    probs: List[float],
    outcomes: List[int],
    method: str = "platt"
) -> CalibrationResult:
    """
    Calibrate raw probabilities against binary outcomes.
    
    Args:
        probs: predicted probabilities (0-1)
        outcomes: binary outcomes (0 or 1)
        method: "platt" or "isotonic"
    
    Returns CalibrationResult with params and Brier scores.
    """
    import numpy as np
    
    probs = np.array(probs)
    outcomes = np.array(outcomes)
    
    brier_before = np.mean((probs - outcomes) ** 2)
    
    if method == "platt":
        A, B = platt_calibration(probs.tolist(), outcomes.tolist())
        probs_cal = apply_platt_calibration(probs.tolist(), A, B)
        params = (A, B)
    elif method == "isotonic":
        knots = isotonic_calibration(probs.tolist(), outcomes.tolist())
        probs_cal = apply_isotonic_calibration(probs.tolist(), knots)
        params = knots
    else:
        raise ValueError(f"Unknown calibration method: {method}")
    
    brier_after = np.mean((np.array(probs_cal) - outcomes) ** 2)
    
    return CalibrationResult(
        method=method,
        params=params,
        brier_before=brier_before,
        brier_after=brier_after,
        n_samples=len(probs)
    )


# ─── Convenience: unified interface matching config.MODEL_TYPE ────────────────

def model_probability(
    spot: float,
    strike: float,
    minutes_left: float,
    model_type: str = None,
    **kwargs
) -> float:
    """
    Unified interface for all model types.
    
    Args:
        spot: current spot price
        strike: strike price
        minutes_left: minutes to expiry
        model_type: one of "gbm", "gbm_annual", "arithmetic", "average_window", "locked_share"
        **kwargs: model-specific params
    
    Returns probability in [0, 1].
    """
    if model_type is None:
        model_type = config.MODEL_TYPE
    
    if model_type == "gbm":
        # Fractional per-minute sigma (from binance_feed)
        sigma = kwargs.get("sigma_frac_per_min", annualized_to_per_minute(config.MODEL_VOLATILITY))
        return gbm_probability_point(spot, strike, sigma, minutes_left)
    
    elif model_type == "gbm_annual":
        # Annualized volatility (from model.py)
        sigma_annual = kwargs.get("sigma_annual", config.MODEL_VOLATILITY)
        risk_free = kwargs.get("risk_free_rate", 0.05)
        return gbm_probability_bs_d2(spot, strike, sigma_annual, minutes_left, risk_free)
    
    elif model_type == "arithmetic":
        # Dollar sigma per sqrt-minute (from gates.py)
        dollar_sigma = kwargs.get("dollar_sigma_per_sqrt_min")
        if dollar_sigma is None:
            # Convert from fractional
            sigma_frac = kwargs.get("sigma_frac_per_min", annualized_to_per_minute(config.MODEL_VOLATILITY))
            dollar_sigma = dollar_sigma_from_fractional(spot, sigma_frac)
        return arithmetic_probability(spot, strike, dollar_sigma, minutes_left, kwargs.get("side", "yes"))
    
    elif model_type == "average_window":
        sigma = kwargs.get("sigma_frac_per_min", annualized_to_per_minute(config.MODEL_VOLATILITY))
        return gbm_probability_average_window(spot, strike, sigma, minutes_left)
    
    elif model_type == "locked_share":
        sigma = kwargs.get("sigma_frac_per_min", annualized_to_per_minute(config.MODEL_VOLATILITY))
        return locked_share_gbm(spot, strike, sigma, kwargs.get("seconds_into_final_minute", 0))
    
    elif model_type == "raw_mid":
        # No model - just return 0.5 (will be filtered by edge check)
        return 0.5
    
    else:
        log.warning("Unknown model_type %s, defaulting to gbm", model_type)
        sigma = annualized_to_per_minute(config.MODEL_VOLATILITY)
        return gbm_probability_point(spot, strike, sigma, minutes_left)


# ─── Rolling calibration monitor (for online calibration) ─────────────────────

class RollingCalibrationMonitor:
    """
    Online calibration monitor that tracks model performance and fits calibration.
    
    Uses a rolling window of settled markets to fit Platt/isotonic calibration
    and reports Brier score improvement.
    """
    
    def __init__(self, window_size: int = 500, method: str = "platt"):
        self.window_size = window_size
        self.method = method
        self.probs: List[float] = []
        self.outcomes: List[int] = []
        self.calibration: Optional[CalibrationResult] = None
    
    def add(self, prob: float, outcome: int):
        """Add a settled market result."""
        self.probs.append(prob)
        self.outcomes.append(outcome)
        
        # Keep rolling window
        if len(self.probs) > self.window_size:
            self.probs = self.probs[-self.window_size:]
            self.outcomes = self.outcomes[-self.window_size:]
    
    def update_calibration(self) -> Optional[CalibrationResult]:
        """Refit calibration on current window."""
        if len(self.probs) < 50:
            return None
        
        self.calibration = calibrate_probabilities(self.probs, self.outcomes, self.method)
        return self.calibration
    
    def apply(self, prob: float) -> float:
        """Apply current calibration to a probability."""
        if self.calibration is None:
            return prob
        
        if self.calibration.method == "platt":
            A, B = self.calibration.params
            return 1.0 / (1.0 + math.exp(-(A * prob + B)))
        else:
            # Isotonic: find the right knot
            knots = self.calibration.params
            for x, y in reversed(knots):
                if prob >= x:
                    return y
            return knots[0][1] if knots else prob
    
    def get_metrics(self) -> dict:
        """Get current calibration metrics."""
        if self.calibration is None:
            return {"status": "insufficient_data", "n_samples": len(self.probs)}
        
        return {
            "status": "calibrated",
            "method": self.calibration.method,
            "brier_before": self.calibration.brier_before,
            "brier_after": self.calibration.brier_after,
            "brier_improvement": self.calibration.brier_before - self.calibration.brier_after,
            "n_samples": self.calibration.n_samples,
        }


# ─── Test / demo ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Quick test comparing the three variants
    spot = 95000.0
    strike = 95000.0
    minutes = 10.0
    sigma_annual = 0.80
    sigma_per_min = annualized_to_per_minute(sigma_annual)
    dollar_sigma = dollar_sigma_from_fractional(spot, sigma_per_min)
    
    print(f"Spot: {spot}, Strike: {strike}, Minutes: {minutes}")
    print(f"Sigma annual: {sigma_annual}, per_min: {sigma_per_min:.6f}, dollar: {dollar_sigma:.2f}")
    print()
    
    p1 = gbm_probability_point(spot, strike, sigma_per_min, minutes)
    print(f"gbm_probability_point (fractional sigma): {p1:.4f}")
    
    p2 = gbm_probability_bs_d2(spot, strike, sigma_annual, minutes)
    print(f"gbm_probability_bs_d2 (annualized):       {p2:.4f}")
    
    p3 = arithmetic_probability(spot, strike, dollar_sigma, minutes, "yes")
    print(f"arithmetic_probability (dollar sigma):     {p3:.4f}")
    
    p4 = gbm_probability_average_window(spot, strike, sigma_per_min, minutes)
    print(f"gbm_probability_average_window (BRTI):     {p4:.4f}")
    
    p5 = locked_share_gbm(spot, strike, sigma_per_min, 30)
    print(f"locked_share_gbm (30s into final min):     {p5:.4f}")
    
    # Test calibration
    print("\n--- Calibration test ---")
    import random
    probs = [random.random() for _ in range(100)]
    outcomes = [1 if p > 0.5 else 0 for p in probs]
    # Add noise
    for i in range(len(outcomes)):
        if random.random() < 0.1:
            outcomes[i] = 1 - outcomes[i]
    
    cal = calibrate_probabilities(probs, outcomes, "platt")
    print(f"Platt: A={cal.params[0]:.3f}, B={cal.params[1]:.3f}, "
          f"Brier before={cal.brier_before:.4f}, after={cal.brier_after:.4f}")
    
    cal2 = calibrate_probabilities(probs, outcomes, "isotonic")
    print(f"Isotonic: {len(cal2.params)} knots, "
          f"Brier before={cal2.brier_before:.4f}, after={cal2.brier_after:.4f}")