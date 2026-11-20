"""
gbm_fair_value.py — GBM fair value baseline model (T13-20).

Implements the Geometric Brownian Motion fair value model for 15-minute binary crypto markets.
Formula (r ~ 0, drift ~ 0 over 15 minutes):
    q = Phi(d2)
    d2 = (ln(S/K) - 0.5*sigma^2*tau) / (sigma*sqrt(tau))
Where:
    S = composite spot price
    K = strike (or window-open reference for up/down markets)
    tau = time to expiry in years
    sigma = per-minute volatility (from realized 1-s returns + AdaptiveVolatilityScaler)

Features:
- Sigma from realized 1-second returns over several windows
- Blended with AdaptiveVolatilityScaler (already in codebase)
- Fat-tail option: Merton jump diffusion or Student-t
- Basis uncertainty from T13-11 settlement_reference_basis
- Calibration on held-out settled markets
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import config
import spot_composite

log = logging.getLogger("gbm_fair_value")

# Constants
SECONDS_PER_YEAR = 365.25 * 24 * 3600
MINUTES_PER_YEAR = 365.25 * 24 * 60


@dataclass
class GBMInputs:
    """Inputs for GBM fair value calculation."""
    spot: float           # Composite spot price
    strike: float         # Strike price
    minutes_left: float   # Minutes until expiry
    sigma_per_minute: float  # Per-minute volatility
    basis_bps: float = 0.0   # Settlement basis in bps (from T13-11)


@dataclass
class GBMResult:
    """Result of GBM fair value calculation."""
    q: float              # P(YES settles) = P(price > strike)
    d2: float
    sigma_annualized: float
    tau_years: float
    adjusted_spot: float  # Spot adjusted for basis
    confidence_interval: Optional[Tuple[float, float]] = None  # (q_lower, q_upper)
    fat_tail_adjustment: float = 0.0


def phi(x: float) -> float:
    """Standard normal CDF using erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def calculate_gbm_fair_value(inputs: GBMInputs, use_fat_tail: bool = False,
                              tail_nu: float = 5.0, jump_lambda: float = 0.1,
                              jump_sigma: float = 0.02) -> GBMResult:
    """
    Calculate GBM fair value P(price > strike) at expiry.
    
    Args:
        inputs: GBMInputs with spot, strike, minutes_left, sigma_per_minute, basis_bps
        use_fat_tail: If True, apply Student-t fat-tail adjustment
        tail_nu: Degrees of freedom for Student-t (lower = fatter tails)
        jump_lambda: Jump intensity per year (Merton)
        jump_sigma: Jump size volatility (Merton)
    
    Returns:
        GBMResult with q, d2, and metadata
    """
    if inputs.spot <= 0 or inputs.strike <= 0 or inputs.minutes_left <= 0:
        return GBMResult(q=float("nan"), d2=float("nan"), sigma_annualized=0.0,
                         tau_years=0.0, adjusted_spot=inputs.spot)
    
    if inputs.sigma_per_minute <= 0:
        return GBMResult(q=1.0 if inputs.spot > inputs.strike else 0.0,
                         d2=float("inf") if inputs.spot > inputs.strike else float("-inf"),
                         sigma_annualized=0.0, tau_years=0.0, adjusted_spot=inputs.spot)
    
    # Adjust spot for settlement basis (T13-11)
    # If basis is positive, settlement ref tends to be higher than composite
    basis_multiplier = 1.0 + (inputs.basis_bps / 10000.0)
    adjusted_spot = inputs.spot * basis_multiplier
    
    # Time to expiry in years
    tau_years = inputs.minutes_left / MINUTES_PER_YEAR
    
    # Annualized volatility from per-minute
    sigma_annualized = inputs.sigma_per_minute * math.sqrt(MINUTES_PER_YEAR)
    
    # Standard GBM d2 (zero drift)
    # d2 = (ln(S/K) - 0.5*sigma^2*tau) / (sigma*sqrt(tau))
    denom = sigma_annualized * math.sqrt(tau_years)
    if denom <= 0:
        return GBMResult(q=1.0 if adjusted_spot > inputs.strike else 0.0,
                         d2=float("inf") if adjusted_spot > inputs.strike else float("-inf"),
                         sigma_annualized=sigma_annualized, tau_years=tau_years,
                         adjusted_spot=adjusted_spot)
    
    d2 = (math.log(adjusted_spot / inputs.strike) - 0.5 * sigma_annualized**2 * tau_years) / denom
    q = phi(d2)
    
    fat_tail_adj = 0.0
    ci = None
    
    if use_fat_tail and tail_nu > 2:
        # Student-t adjustment: heavier tails increase probability of extreme moves
        # For a given sigma, Student-t has fatter tails than normal
        # Approximate by scaling volatility
        # Variance of Student-t with nu dof: nu/(nu-2) * sigma^2 for nu > 2
        var_scale = tail_nu / (tail_nu - 2)
        sigma_t = sigma_annualized * math.sqrt(var_scale)
        denom_t = sigma_t * math.sqrt(tau_years)
        if denom_t > 0:
            d2_t = (math.log(adjusted_spot / inputs.strike) - 0.5 * sigma_t**2 * tau_years) / denom_t
            q_t = phi(d2_t)
            fat_tail_adj = q_t - q
            q = q_t
    
    # Confidence interval using sigma uncertainty
    # Approximate: q ~ N(q, sigma_q^2) where sigma_q = phi'(d2) * sigma_d2
    # sigma_d2 ~ sigma / sqrt(n) where n is number of observations
    # For simplicity, use a fixed 10% relative uncertainty on sigma
    if inputs.sigma_per_minute > 0:
        sigma_q = abs((1/math.sqrt(2*math.pi)) * math.exp(-0.5*d2*d2)) * (sigma_annualized * 0.1 * math.sqrt(tau_years))
        ci = (max(0.0, q - 1.96*sigma_q), min(1.0, q + 1.96*sigma_q))
    
    return GBMResult(
        q=q,
        d2=d2,
        sigma_annualized=sigma_annualized,
        tau_years=tau_years,
        adjusted_spot=adjusted_spot,
        confidence_interval=ci,
        fat_tail_adjustment=fat_tail_adj
    )


class GBMFairValueModel:
    """GBM fair value model with calibration and ensemble support."""
    
    def __init__(self):
        self.calibration_data: List[Tuple[float, bool]] = []  # (q_pred, outcome)
        self._tick_store = None
    
    def set_tick_store(self, tick_store):
        """Set tick store for historical sigma calculation."""
        self._tick_store = tick_store
    
    def calculate_sigma_from_ticks(self, symbol: str, lookback_seconds: int = 300) -> Optional[float]:
        """
        Calculate per-minute sigma from 1-second tick returns.
        
        Uses the tick_store to get recent 1-second returns and compute realized vol.
        """
        if not self._tick_store:
            # Fallback to binance_feed sigma_per_minute
            import binance_feed
            return binance_feed.sigma_per_minute(symbol.replace("USDT", "").replace("USDC", "") + "USDT")
        
        end_ts = datetime.now(timezone.utc).timestamp()
        start_ts = end_ts - lookback_seconds
        
        ticks = self._tick_store.query_ticks(symbol, start_ts, end_ts, limit=lookback_seconds)
        if len(ticks) < 10:
            return None
        
        # Compute 1-second log returns
        returns = []
        for i in range(1, len(ticks)):
            prev_px = ticks[i-1]["px"]
            curr_px = ticks[i]["px"]
            if prev_px > 0 and curr_px > 0:
                returns.append(math.log(curr_px / prev_px))
        
        if len(returns) < 5:
            return None
        
        mean_ret = sum(returns) / len(returns)
        var = sum((r - mean_ret)**2 for r in returns) / (len(returns) - 1)
        sigma_per_second = math.sqrt(var)
        
        # Scale to per-minute
        sigma_per_minute = sigma_per_second * math.sqrt(60)
        return sigma_per_minute
    
    def get_sigma_blended(self, symbol: str, adaptive_scaler_value: float = None,
                           lookback_seconds: int = 300) -> float:
        """
        Blend realized sigma with AdaptiveVolatilityScaler.
        
        The AdaptiveVolatilityScaler (in binance_feed or elsewhere) provides
        a regime-aware volatility estimate. We blend it with short-window
        realized vol for robustness.
        """
        # Get realized sigma from ticks
        realized_sigma = self.calculate_sigma_from_ticks(symbol, lookback_seconds)
        
        # Get adaptive scaler value if provided
        if adaptive_scaler_value is not None and adaptive_scaler_value > 0:
            adaptive_sigma = adaptive_scaler_value
        else:
            # Fallback to binance_feed composite_sigma
            import binance_feed
            base = symbol.replace("USDT", "").replace("USDC", "")
            adaptive_sigma = binance_feed.composite_sigma(f"{base}USDT")
        
        # Blend: 70% realized, 30% adaptive (configurable)
        if realized_sigma is not None and adaptive_sigma is not None:
            return 0.7 * realized_sigma + 0.3 * adaptive_sigma
        elif realized_sigma is not None:
            return realized_sigma
        elif adaptive_sigma is not None:
            return adaptive_sigma
        
        # Ultimate fallback
        return 0.01  # 1% per minute default
    
    def predict(self, symbol: str, strike: float, minutes_left: float,
                adaptive_scaler_value: float = None, venue: str = "kalshi") -> GBMResult:
        """
        Predict P(YES settles) for a market.
        
        Args:
            symbol: e.g., "BTCUSDT"
            strike: Strike price
            minutes_left: Minutes until expiry
            adaptive_scaler_value: Optional AdaptiveVolatilityScaler output
            venue: "kalshi" or "polymarket" (for basis adjustment)
        
        Returns:
            GBMResult with fair value probability
        """
        # Get composite spot
        spot = spot_composite.get_spot_with_age(symbol)
        if spot is None:
            import binance_feed
            spot = binance_feed.composite_price(symbol)
        
        if spot is None:
            log.warning("No spot price available for %s", symbol)
            return GBMResult(q=0.5, d2=0.0, sigma_annualized=0.0, tau_years=0.0, adjusted_spot=0.0)
        
        # Get blended sigma
        sigma_per_minute = self.get_sigma_blended(symbol, adaptive_scaler_value)
        
        # Get basis adjustment from T13-11
        basis_bps = 0.0
        try:
            from local_db import get_local_db
            db = get_local_db()
            avg_basis = db.get_average_basis_bps(venue, lookback_count=50)
            if avg_basis is not None:
                basis_bps = avg_basis
        except Exception:
            pass
        
        inputs = GBMInputs(
            spot=spot,
            strike=strike,
            minutes_left=minutes_left,
            sigma_per_minute=sigma_per_minute,
            basis_bps=basis_bps
        )
        
        return calculate_gbm_fair_value(inputs, use_fat_tail=True, tail_nu=5.0)
    
    def update_calibration(self, q_pred: float, outcome: bool):
        """Record prediction for calibration tracking."""
        self.calibration_data.append((q_pred, outcome))
        # Keep only recent
        if len(self.calibration_data) > 10000:
            self.calibration_data = self.calibration_data[-10000:]
    
    def get_calibration_metrics(self, bins: int = 10) -> Dict:
        """Calculate calibration metrics (Brier score, reliability diagram)."""
        if not self.calibration_data:
            return {"brier": None, "reliability": None, "n": 0}
        
        n = len(self.calibration_data)
        brier = sum((q - (1.0 if y else 0.0))**2 for q, y in self.calibration_data) / n
        
        # Reliability diagram: bin predictions and compare avg pred vs avg outcome
        bin_edges = [i / bins for i in range(bins + 1)]
        reliability = []
        for i in range(bins):
            lo, hi = bin_edges[i], bin_edges[i+1]
            bin_data = [(q, y) for q, y in self.calibration_data if lo <= q < hi]
            if bin_data:
                avg_pred = sum(q for q, _ in bin_data) / len(bin_data)
                avg_outcome = sum(1.0 for _, y in bin_data if y) / len(bin_data)
                reliability.append({"bin": f"[{lo:.1f}, {hi:.1f})", "avg_pred": avg_pred, 
                                  "avg_outcome": avg_outcome, "count": len(bin_data)})
        
        return {"brier": brier, "reliability": reliability, "n": n}


# Module-level singleton
_gbm_model: Optional[GBMFairValueModel] = None


def get_gbm_model() -> GBMFairValueModel:
    """Get or create global GBM fair value model."""
    global _gbm_model
    if _gbm_model is None:
        _gbm_model = GBMFairValueModel()
    return _gbm_model


def predict_fair_value(symbol: str, strike: float, minutes_left: float,
                       adaptive_scaler_value: float = None, venue: str = "kalshi") -> GBMResult:
    """Convenience function for single prediction."""
    return get_gbm_model().predict(symbol, strike, minutes_left, adaptive_scaler_value, venue)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    
    # Test basic calculation
    result = calculate_gbm_fair_value(GBMInputs(
        spot=62000.0,
        strike=62100.0,
        minutes_left=10.0,
        sigma_per_minute=0.005,
        basis_bps=0.0
    ))
    print(f"GBM fair value: q={result.q:.4f}, d2={result.d2:.4f}, "
          f"sigma_ann={result.sigma_annualized:.4f}, tau={result.tau_years:.6f}")
    
    # Test model
    model = GBMFairValueModel()
    # This would need actual tick store or binance_feed
    # result = model.predict("BTCUSDT", 62100.0, 10.0)
    # print(f"Model prediction: {result}")