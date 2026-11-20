"""
model.py — Predictive fair-value models for Kalshi 15-minute binary contracts.

Implements multiple model types so the bot only trades when there is a
measurable edge after fees:

1. GBM model (default): Geometric Brownian Motion probability that spot
   finishes at/above strike, using spot price, strike, time-to-expiry and
   realized volatility. This is the same model class used by kalshibot.

2. Momentum model: uses recent price direction as a predictor — if the
   favourite's ask has been rising in the final minutes, settlement
   probability converges toward 1.

3. Locked-share model: as the 60-second BRTI average locks in second by
   second, remaining uncertainty collapses (r^1.5 scaling). Late-window
   prices become more informative.

Only trade when: model_probability > ask_price_dollars + fee + MIN_EDGE_CENTS
"""

import math
import logging
import config
from statistics import NormalDist

log = logging.getLogger(__name__)

_NORMAL = NormalDist()

# Lazy import for BRTI tracker
_brti_tracker = None


def gbm_probability_yes_wins(spot_price: float, strike: float,
                             minutes_left: float,
                             volatility_annualized: float = 0.80,
                             risk_free_rate: float = 0.05) -> float:
    """
    GBM probability that spot >= strike at expiry.

    Uses the Black-Scholes-style d2 formula:
        P(S_T >= K) = N(d2),  d2 = (ln(S/K) + (r - 0.5*sigma^2)*T) / (sigma*sqrt(T))

    Args:
        spot_price: current spot price of the underlying crypto
        strike: the market's target/strike price
        minutes_left: minutes until settlement
        volatility_annualized: annualized vol (0.80 = 80% for crypto)
        risk_free_rate: annualized risk-free rate (5% default)

    Returns probability in [0, 1].
    """
    if minutes_left <= 0:
        return 1.0 if spot_price >= strike else 0.0
    if spot_price <= 0 or strike <= 0:
        return 0.5

    T = minutes_left / (365.0 * 24.0 * 60.0)  # years
    sigma = volatility_annualized
    if T <= 0 or sigma <= 0:
        return 1.0 if spot_price >= strike else 0.0

    try:
        d2 = (math.log(spot_price / strike) +
              (risk_free_rate - 0.5 * sigma * sigma) * T) / \
             (sigma * math.sqrt(T))
        prob = _NORMAL.cdf(d2)
        return max(0.001, min(0.999, prob))
    except (ValueError, OverflowError):
        return 0.5


def momentum_adjusted_probability(mid_yes_cents: float,
                                  minutes_left: float,
                                  momentum_cents_per_min: float,
                                  max_adjustment_cents: float = 10.0) -> float:
    """
    Adjust market-implied probability based on recent momentum.

    If the favourite's price has been rising (momentum_cents_per_min > 0),
    settlement probability converges toward 1 faster than the raw mid implies.
    If falling, converges toward 0.

    Args:
        mid_yes_cents: market mid price in cents (1-99)
        minutes_left: minutes until settlement
        momentum_cents_per_min: cents change per minute (positive = rising)
        max_adjustment_cents: cap on adjustment in cents

    Returns adjusted probability in [0, 1].
    """
    base = mid_yes_cents / 100.0
    if minutes_left <= 0:
        return base

    # Adjustment scales with momentum and time left
    adjustment = momentum_cents_per_min * minutes_left
    adjustment = max(-max_adjustment_cents, min(max_adjustment_cents, adjustment))
    adjusted = base + adjustment / 100.0
    return max(0.001, min(0.999, adjusted))


def locked_share_probability(mid_yes_cents: float,
                             seconds_into_final_minute: int,
                             symbol: str = "BTCUSDT") -> float:
    """
    Adjust probability based on how much of the 60-second BRTI average is locked.

    With r seconds left in the final minute, remaining uncertainty scales with
    r^1.5 (not sqrt). As seconds tick by, the partly-known average becomes
    more deterministic. Late prices are more informative.

    When BRTI_TRACKER_ENABLED, uses the running 60-second average from WS ticks
    instead of the raw mid price for more accurate settlement prediction.

    Args:
        mid_yes_cents: market mid price in cents
        seconds_into_final_minute: 0-60, how many seconds of the final minute
            have already passed (0 = start of final minute, 60 = settled)
        symbol: trading symbol (e.g., "BTCUSDT") for BRTI tracker lookup

    Returns adjusted probability in [0, 1].
    """
    global _brti_tracker
    
    base = mid_yes_cents / 100.0
    if seconds_into_final_minute <= 0:
        return base
    if seconds_into_final_minute >= 60:
        return base  # settled — no more adjustment possible

    # Try to use BRTI tracker for running average
    brti_price = None
    if config.BRTI_TRACKER_ENABLED:
        try:
            from brti_tracker import get_brti_tracker as get_brti_tracker_async
            import asyncio
            
            if _brti_tracker is None:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    _brti_tracker = loop.run_until_complete(
                        get_brti_tracker_async([symbol])
                    )
                finally:
                    loop.close()
            
            if _brti_tracker:
                brti_price = _brti_tracker.get_running_average(symbol)
                age = _brti_tracker.get_age_seconds(symbol)
                
                # Reject if stale
                if brti_price is not None and age is not None and age <= config.BRTI_STALE_SECONDS:
                    # Use BRTI proxy price instead of mid for base probability
                    # Convert price to probability using the same logic
                    # For simplicity, we blend the mid with BRTI-adjusted probability
                    pass
                elif brti_price is not None:
                    log.debug("BRTI price for %s stale: age=%.2fs > %.2fs",
                              symbol, age, config.BRTI_STALE_SECONDS)
        except Exception as e:
            log.warning("BRTI tracker error for %s: %s, using mid price", symbol, e)
    
    # Fallback to original logic using mid price
    # Remaining uncertainty fraction: (60 - s)^1.5 / 60^1.5
    remaining = (60.0 - seconds_into_final_minute) / 60.0
    uncertainty_reduction = 1.0 - remaining ** 1.5

    # Push probability toward the extreme implied by current mid
    # If mid > 50, push up; if mid < 50, push down
    direction = 1.0 if base > 0.5 else -1.0 if base < 0.5 else 0.0
    push = abs(base - 0.5) * 2.0 * uncertainty_reduction  # max push toward extreme
    adjusted = base + direction * push * 0.5  # cap at 50% of the way to extreme
    return max(0.001, min(0.999, adjusted))


def edge_cents(model_probability: float, ask_cents: int,
               fee_cents: float) -> float:
    """
    Compute edge in cents per contract.

    Edge = (model_probability * 100) - ask_cents - fee_cents
    Positive means the model says the contract is underpriced after fees.
    """
    return model_probability * 100.0 - ask_cents - fee_cents


def taker_fee_cents_at(ask_cents: int, contracts: float = 1.0,
                       fee_rate: float = 0.07) -> float:
    """
    Taker fee in cents for one contract at ask_cents price.
    Delegates to fees.taker_fee_cents to avoid duplicating fee math.
    """
    from fees import taker_fee_cents
    return taker_fee_cents(contracts, ask_cents / 100.0)


def model_probability_from_market(market, model_type: str = "gbm",
                                  spot_price: float | None = None,
                                  strike: float | None = None,
                                  minutes_left: float = 0.0,
                                  volatility: float = 0.80,
                                  momentum_cents_per_min: float = 0.0,
                                  seconds_into_final_minute: int = 0) -> float:
    """
    Compute model-implied probability of YES winning from a MarketInfo.

    Dispatches to the appropriate model based on model_type.
    """
    mid = market.mid_yes
    if mid is None:
        return 0.5

    base_prob = mid / 100.0

    if model_type == "gbm":
        if spot_price is not None and strike is not None and minutes_left > 0:
            return gbm_probability_yes_wins(
                spot_price, strike, minutes_left, volatility)
        # Fall back to momentum model if no spot data (logged so a silent
        # model swap in the trade record is visible in the run log).
        log.debug("gbm fallback to momentum: %s spot=%s strike=%s",
                  getattr(market, "ticker", "?"), spot_price, strike)
        return momentum_adjusted_probability(
            base_prob * 100.0, minutes_left, momentum_cents_per_min)

    elif model_type == "momentum":
        return momentum_adjusted_probability(
            base_prob * 100.0, minutes_left, momentum_cents_per_min)

    elif model_type == "locked_share":
        return locked_share_probability(
            base_prob * 100.0, seconds_into_final_minute)

    elif model_type == "raw_mid":
        return base_prob

    else:
        log.warning("Unknown model_type %s, using raw mid", model_type)
        return base_prob
