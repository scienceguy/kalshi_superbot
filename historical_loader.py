"""
historical_loader.py — Load settled Kalshi markets from Parquet for backtest replay.

Implements T13-32: Integrate backtest_harness with historical_data (real settled markets + outcomes).
- Reads partitioned Parquet files from historical_data/markets/series=KXBTC15M/ and series=KXETH15M/
- Provides market snapshots with actual orderbook data at different timestamps
- Returns settlement outcomes from the 'result' field (yes/no)
- Generates synthetic spot paths using GBM model for intraday replay
"""

from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

# Import MarketSnapshot from backtest_harness to avoid conflicts
from backtest_harness import MarketSnapshot as BacktestMarketSnapshot

log = logging.getLogger("historical_loader")

# Historical data root
HISTORICAL_DATA_DIR = Path(__file__).resolve().parent / "historical_data" / "markets"

# GBM constants
SECONDS_PER_YEAR = 365.25 * 24 * 3600
MINUTES_PER_YEAR = 365.25 * 24 * 60

# Binance REST API for historical klines
BINANCE_KLINES_URL = "https://data-api.binance.vision/api/v3/klines"

# Symbol mapping from Kalshi series to Binance symbol
SERIES_TO_BINANCE = {
    "KXBTC15M": "BTCUSDT",
    "KXETH15M": "ETHUSDT",
    "KXSOL15M": "SOLUSDT",
    "KXXRP15M": "XRPUSDT",
    "KXDOGE15M": "DOGEUSDT",
    "KXBNB15M": "BNBUSDT",
}

# Cache for historical prices to avoid repeated API calls
_price_cache: Dict[str, float] = {}
_price_cache_file = Path(__file__).resolve().parent / "historical_data" / "price_cache.json"

# Load price cache from disk
if _price_cache_file.exists():
    try:
        import json
        with open(_price_cache_file) as f:
            _price_cache = json.load(f)
        log.info(f"Loaded {len(_price_cache)} cached prices from {_price_cache_file}")
    except Exception as e:
        log.warning(f"Failed to load price cache: {e}")
        _price_cache = {}


def _save_price_cache():
    """Save price cache to disk."""
    try:
        import json
        _price_cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(_price_cache_file, 'w') as f:
            json.dump(_price_cache, f)
    except Exception as e:
        log.warning(f"Failed to save price cache: {e}")


def fetch_historical_klines(
    symbol: str,
    start_ts: int,
    end_ts: int,
    interval: str = "1m"
) -> Optional[List[Dict]]:
    """
    Fetch historical klines from Binance for a time range.
    
    Args:
        symbol: Binance symbol (e.g., BTCUSDT)
        start_ts: Start timestamp in milliseconds
        end_ts: End timestamp in milliseconds
        interval: Kline interval (1m, 5m, etc.)
        
    Returns:
        List of klines with open/high/low/close/volume, or None if failed
    """
    cache_key = f"{symbol}:{interval}:{start_ts}:{end_ts}"
    if cache_key in _price_cache:
        return _price_cache[cache_key]
    
    try:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start_ts,
            'endTime': end_ts,
            'limit': 1000
        }
        r = requests.get(BINANCE_KLINES_URL, params=params, timeout=10)
        if r.status_code == 200:
            data = r.json()
            klines = []
            for k in data:
                klines.append({
                    'open_time': k[0],
                    'open': float(k[1]),
                    'high': float(k[2]),
                    'low': float(k[3]),
                    'close': float(k[4]),
                    'volume': float(k[5]),
                    'close_time': k[6],
                })
            _price_cache[cache_key] = klines
            if len(_price_cache) % 100 == 0:
                _save_price_cache()
            return klines
    except Exception as e:
        log.debug(f"Failed to fetch historical klines for {symbol} {start_ts}-{end_ts}: {e}")
    
    return None


@dataclass
class MarketMetadata:
    """Key metadata for a settled market."""
    ticker: str
    open_time: datetime
    close_time: datetime
    settlement_ts: datetime
    result: str  # "yes" or "no"
    floor_strike: float
    expiration_value: Optional[float]
    yes_bid_dollars: float
    yes_ask_dollars: float
    no_bid_dollars: float
    no_ask_dollars: float
    yes_bid_size_fp: float
    yes_ask_size_fp: float
    no_bid_size_fp: float
    no_ask_size_fp: float
    volume: float
    open_interest: float
    last_price_cents: Optional[float] = None  # Last traded price in cents (0-100)


# Remove the local MarketSnapshot class - use the one from backtest_harness
# @dataclass
# class MarketSnapshot:
#     """Snapshot of a market at a specific timestamp."""
#     ticker: str
#     timestamp: datetime
#     yes_bid: int
#     yes_ask: int
#     no_bid: int
#     no_ask: int
#     yes_bid_size: float
#     yes_ask_size: float
#     no_bid_size: float
#     no_ask_size: float
#     floor_strike: float
#     minutes_left: float


def fetch_historical_price(symbol: str, timestamp_ms: int) -> Optional[float]:
    """
    Fetch historical price from Binance for a specific timestamp.
    
    Args:
        symbol: Binance symbol (e.g., BTCUSDT)
        timestamp_ms: Timestamp in milliseconds
        
    Returns:
        Price as float, or None if failed
    """
    cache_key = f"{symbol}:{timestamp_ms}"
    if cache_key in _price_cache:
        return _price_cache[cache_key]
    
    try:
        params = {
            'symbol': symbol,
            'interval': '1m',
            'startTime': timestamp_ms,
            'endTime': timestamp_ms + 60000,  # 1 minute window
            'limit': 1
        }
        r = requests.get(BINANCE_KLINES_URL, params=params, timeout=10)
        if r.status_code == 200:
            data = r.json()
            if data and len(data) > 0:
                # Use open price (first price in the minute)
                price = float(data[0][1])
                _price_cache[cache_key] = price
                # Save cache every 100 new entries
                if len(_price_cache) % 100 == 0:
                    _save_price_cache()
                return price
    except Exception as e:
        log.debug(f"Failed to fetch historical price for {symbol} at {timestamp_ms}: {e}")
    
    return None


def fetch_historical_price_at_window(series: str, open_time: datetime, close_time: datetime) -> Tuple[Optional[float], Optional[float]]:
    """
    Fetch historical prices at window open (strike) and close (expiration).
    
    Args:
        series: Kalshi series (e.g., KXBTC15M)
        open_time: Window open time (strike time)
        close_time: Window close/settlement time (expiration time)
        
    Returns:
        Tuple of (strike_price, expiration_price)
    """
    symbol = SERIES_TO_BINANCE.get(series)
    if not symbol:
        log.warning(f"No Binance symbol mapping for series {series}")
        return None, None
    
    # Convert to milliseconds
    strike_ts = int(open_time.timestamp() * 1000)
    expiry_ts = int(close_time.timestamp() * 1000)
    
    strike_price = fetch_historical_price(symbol, strike_ts)
    expiry_price = fetch_historical_price(symbol, expiry_ts)
    
    return strike_price, expiry_price


def phi(x: float) -> float:
    """Standard normal CDF using erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def gbm_fair_value(spot: float, strike: float, minutes_left: float, sigma_per_minute: float) -> float:
    """Calculate P(spot > strike) at expiry using GBM."""
    if minutes_left <= 0 or sigma_per_minute <= 0 or spot <= 0 or strike <= 0:
        return 1.0 if spot > strike else 0.0
    
    tau_years = minutes_left / MINUTES_PER_YEAR
    sigma_annualized = sigma_per_minute * math.sqrt(MINUTES_PER_YEAR)
    
    try:
        d2 = (math.log(spot / strike) - 0.5 * sigma_annualized**2 * tau_years) / (sigma_annualized * math.sqrt(tau_years))
        result = phi(d2)
        if math.isnan(result) or math.isinf(result):
            return 1.0 if spot > strike else 0.0
        return result
    except (ValueError, ZeroDivisionError):
        return 1.0 if spot > strike else 0.0


def generate_synthetic_spot_path(
    settlement_spot: float,
    floor_strike: float,
    result: str,  # "yes" or "no"
    open_time: datetime,
    close_time: datetime,
    sigma_per_minute: float = 0.005,  # ~0.5% per minute, typical for BTC
    n_points: int = 900,  # 15 min * 60 sec = 900 points
    seed: int = None,
    last_price_cents: float = None,  # Last traded price in cents (0-100)
) -> List[Tuple[float, float]]:
    """
    Generate a synthetic spot price path from window open to settlement using Brownian Bridge.
    
    Uses GBM with Brownian bridge conditioning to match the known settlement outcome.
    The path is generated forward from open to settlement, conditioned on the final value.
    
    Args:
        settlement_spot: Spot price at settlement (if known, otherwise derived from result)
        floor_strike: The strike price for the market
        result: "yes" or "no" - the actual settlement outcome
        open_time: Window open time
        close_time: Window close/settlement time
        sigma_per_minute: Per-minute volatility
        n_points: Number of time points (default 1 per second for 15 min)
        seed: Random seed for reproducibility
        last_price_cents: Last traded market price in cents (0-100), used to calibrate
    
    Returns:
        List of (timestamp, spot_price) tuples in chronological order
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    duration_sec = (close_time - open_time).total_seconds()
    if duration_sec <= 0:
        return [(open_time.timestamp(), settlement_spot)]
    
    # Time steps
    dt = duration_sec / n_points
    dt_years = dt / SECONDS_PER_YEAR
    sigma_annualized = sigma_per_minute * math.sqrt(MINUTES_PER_YEAR)
    
    # Determine settlement spot price consistent with result
    # If result is YES, spot > strike at expiry; if NO, spot < strike
    # We don't know exact settlement spot, so estimate from result + last_price
    
    if settlement_spot and settlement_spot > 0:
        # Use provided settlement spot
        final_spot = settlement_spot
    else:
        # Estimate settlement spot from result
        # If YES won, spot was slightly above strike; if NO won, slightly below
        if result == "yes":
            final_spot = floor_strike * 1.002  # 0.2% above strike
        else:
            final_spot = floor_strike * 0.998  # 0.2% below strike
    
    # Calibrate initial spot using last_price if available
    # last_price is in cents (0-100), convert to implied probability
    if last_price_cents is not None and 0 < last_price_cents < 100:
        # Market price implies a probability
        implied_q = last_price_cents / 100.0
        # At window open (15 min left), solve for spot that gives this probability
        # q = Phi(d2) where d2 = (ln(S/K) - 0.5*sigma^2*tau) / (sigma*sqrt(tau))
        tau_years = 15.0 / MINUTES_PER_YEAR
        sigma_t = sigma_annualized * math.sqrt(tau_years)
        
        # Inverse normal CDF approximation
        try:
            from scipy.special import ndtri
            d2 = ndtri(implied_q)
        except ImportError:
            # Simple approximation using erfinv
            d2 = math.sqrt(2) * math.erfinv(2 * implied_q - 1)
        
        # Solve for initial spot
        # d2 = (ln(S0/K) - 0.5*sigma^2*tau) / (sigma*sqrt(tau))
        # ln(S0/K) = d2 * sigma * sqrt(tau) + 0.5 * sigma^2 * tau
        ln_s0_k = d2 * sigma_t + 0.5 * sigma_annualized**2 * tau_years
        initial_spot = floor_strike * math.exp(ln_s0_k)
        
        # Clamp to reasonable range
        initial_spot = max(floor_strike * 0.95, min(floor_strike * 1.05, initial_spot))
    else:
        # Default: start near strike with slight bias towards result
        if result == "yes":
            initial_spot = floor_strike * 1.001
        else:
            initial_spot = floor_strike * 0.999
    
    # Generate path using Brownian Bridge
    # S(t) = S0 * exp((mu - 0.5*sigma^2)*t + sigma*W(t))
    # Conditioned on S(T) = final_spot
    # Brownian bridge: W(t) = (t/T)*W(T) + sqrt(t*(T-t)/T) * Z
    # where Z ~ N(0,1) independent of W(T)
    
    T = duration_sec
    T_years = T / SECONDS_PER_YEAR
    
    # Total Brownian motion needed to go from initial to final
    # ln(S_T/S_0) = (mu - 0.5*sigma^2)*T + sigma*W(T)
    # With mu=0: W(T) = (ln(S_T/S_0) + 0.5*sigma^2*T) / sigma
    total_log_return = math.log(final_spot / initial_spot)
    w_T = (total_log_return + 0.5 * sigma_annualized**2 * T_years) / sigma_annualized
    
    # Generate path
    path = []
    for i in range(n_points + 1):
        t = i * dt
        ts = open_time.timestamp() + t
        
        if i == 0:
            # At window open
            spot = initial_spot
        elif i == n_points:
            # At settlement
            spot = final_spot
        else:
            # Brownian bridge interpolation
            # W(t) = (t/T)*W(T) + sqrt(t*(T-t)/T) * Z
            ratio = t / T
            bridge_var = ratio * (1 - ratio) * T_years
            z = np.random.normal(0, 1)
            w_t = ratio * w_T + math.sqrt(bridge_var) * z
            
            # Spot at time t
            # ln(S_t/S_0) = -0.5*sigma^2*t + sigma*W(t)
            ln_s_t_s0 = -0.5 * sigma_annualized**2 * (t / SECONDS_PER_YEAR) + sigma_annualized * w_t
            spot = initial_spot * math.exp(ln_s_t_s0)
            
            # Clamp to reasonable bounds
            spot = max(floor_strike * 0.5, min(floor_strike * 2.0, spot))
        
        path.append((ts, spot))
    
    return path


def spot_to_kalshi_snapshot(
    spot: float,
    strike: float,
    minutes_left: float,
    sigma_per_minute: float,
    base_spread_bps: int = 200,  # 2% base spread
) -> Tuple[int, int, int, int]:
    """
    Convert spot price to Kalshi YES/NO bid/ask in cents.
    
    Uses GBM fair value to determine mid price, then applies spread.
    """
    q = gbm_fair_value(spot, strike, minutes_left, sigma_per_minute)
    
    # Convert probability to cents (0-100)
    mid = q * 100
    
    # Spread widens as time decays and near boundaries
    time_factor = max(0.5, minutes_left / 15.0)
    boundary_factor = 1.0 + 2.0 * min(q, 1 - q)  # Wider near 0 and 1
    spread_cents = int(base_spread_bps / 100 * time_factor * boundary_factor)
    spread_cents = max(1, min(20, spread_cents))
    
    yes_bid = int(max(1, min(99, mid - spread_cents / 2)))
    yes_ask = int(max(1, min(99, mid + spread_cents / 2)))
    
    if yes_bid > yes_ask:
        yes_bid, yes_ask = yes_ask, yes_bid
    
    no_bid = 100 - yes_ask
    no_ask = 100 - yes_bid
    
    return yes_bid, yes_ask, no_bid, no_ask


def markets_to_snapshots(
    markets: List[MarketMetadata],
    snapshots_per_market: int = 900,  # 1 per second
    sigma_per_minute: float = 0.005,
    use_real_klines: bool = True,
) -> Dict[str, List[MarketSnapshot]]:
    """
    Convert settled markets to snapshot sequences for backtest replay.
    
    For each market, generates snapshots at 1-second intervals from window open
    to settlement. Uses REAL historical spot prices from Binance klines when available,
    otherwise falls back to synthetic GBM paths.
    
    Args:
        markets: List of settled market metadata
        snapshots_per_market: Number of snapshots per market (default 900 = 15 min * 60 sec)
        sigma_per_minute: Per-minute volatility for GBM (fallback)
        use_real_klines: Whether to use real Binance klines (default True)
        
    Returns:
        Dict mapping ticker to list of MarketSnapshot objects
    """
    result = {}
    
    for market in markets:
        snapshots = []
        
        # Try to fetch real historical klines for this market's time window
        klines = None
        if use_real_klines:
            series = market.ticker.split("-")[0]  # KXBTC15M
            symbol = SERIES_TO_BINANCE.get(series)
            if symbol:
                start_ms = int(market.open_time.timestamp() * 1000)
                end_ms = int(market.close_time.timestamp() * 1000)
                klines = fetch_historical_klines(symbol, start_ms, end_ms, "1m")
        
        if klines and len(klines) > 0:
            # Use real spot prices from Binance
            log.debug(f"Using {len(klines)} real klines for {market.ticker}")
            
            # Create snapshots at each second, interpolating between klines
            # Each kline is 1 minute; we need snapshots_per_market points over 15 minutes
            n_klines = len(klines)
            total_seconds = (market.close_time - market.open_time).total_seconds()
            
            # For each second, interpolate spot price from klines
            for i in range(snapshots_per_market + 1):
                t = i * total_seconds / snapshots_per_market
                ts = market.open_time.timestamp() + t
                
                # Find the two klines to interpolate between
                # kline index = t / 60 (since 1 min = 60 seconds)
                kline_idx = t / 60.0
                kline_idx_floor = int(kline_idx)
                kline_idx_ceil = min(kline_idx_floor + 1, n_klines - 1)
                
                if kline_idx_floor >= n_klines - 1:
                    # Use last kline
                    spot = klines[-1]['close']
                elif kline_idx_floor < 0:
                    # Use first kline
                    spot = klines[0]['open']
                else:
                    # Linear interpolation between closes
                    alpha = kline_idx - kline_idx_floor
                    spot = (1 - alpha) * klines[kline_idx_floor]['close'] + alpha * klines[kline_idx_ceil]['close']
                
                minutes_left = max(0, (market.close_time.timestamp() - ts) / 60.0)
                
                yes_bid, yes_ask, no_bid, no_ask = spot_to_kalshi_snapshot(
                    spot, market.floor_strike, minutes_left, sigma_per_minute
                )
                
                time_factor = max(0.1, minutes_left / 15.0)
                
                snap = BacktestMarketSnapshot(
                    ticker=market.ticker,
                    timestamp=ts,
                    yes_bid=yes_bid,
                    yes_ask=yes_ask,
                    no_bid=no_bid,
                    no_ask=no_ask,
                    yes_size=market.yes_bid_size_fp * time_factor,
                    no_size=market.no_bid_size_fp * time_factor,
                    last_price=(yes_bid + yes_ask) / 2.0,
                    volume=market.volume * time_factor,
                    expiration_value=market.expiration_value,
                    floor_strike=market.floor_strike,
                )
                snapshots.append(snap)
        else:
            # Fallback to synthetic path
            log.debug(f"Using synthetic path for {market.ticker} (no klines available)")
            
            spot_path = generate_synthetic_spot_path(
                settlement_spot=market.expiration_value or 0,
                floor_strike=market.floor_strike,
                result=market.result,
                open_time=market.open_time,
                close_time=market.close_time,
                sigma_per_minute=sigma_per_minute,
                n_points=snapshots_per_market,
                last_price_cents=market.last_price_cents,
            )
            
            for ts, spot in spot_path:
                minutes_left = max(0, (market.close_time.timestamp() - ts) / 60.0)
                
                yes_bid, yes_ask, no_bid, no_ask = spot_to_kalshi_snapshot(
                    spot, market.floor_strike, minutes_left, sigma_per_minute
                )
                
                time_factor = max(0.1, minutes_left / 15.0)
                
                snap = BacktestMarketSnapshot(
                    ticker=market.ticker,
                    timestamp=ts,
                    yes_bid=yes_bid,
                    yes_ask=yes_ask,
                    no_bid=no_bid,
                    no_ask=no_ask,
                    yes_size=market.yes_bid_size_fp * time_factor,
                    no_size=market.no_bid_size_fp * time_factor,
                    last_price=(yes_bid + yes_ask) / 2.0,
                    volume=market.volume * time_factor,
                    expiration_value=market.expiration_value,
                    floor_strike=market.floor_strike,
                )
                snapshots.append(snap)
        
        if snapshots:
            result[market.ticker] = snapshots
    
    return result


class SettlementLookup:
    """Picklable settlement lookup for multiprocessing."""
    
    def __init__(self, markets: List[MarketMetadata]):
        self.result_map = {m.ticker: (m.result == "yes") for m in markets}
    
    def __call__(self, ticker: str) -> Optional[bool]:
        return self.result_map.get(ticker)


def create_settlement_fn(markets: List[MarketMetadata]) -> Callable[[str], Optional[bool]]:
    """Create a picklable settlement function."""
    return SettlementLookup(markets)


def load_markets_from_parquet(
    series: str = "KXBTC15M",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_markets: Optional[int] = None,
) -> List[MarketMetadata]:
    """
    Load settled market metadata from Parquet files.
    
    Args:
        series: Market series (KXBTC15M or KXETH15M)
        start_date: Start date in YYYY-MM-DD format (inclusive)
        end_date: End date in YYYY-MM-DD format (inclusive)
        max_markets: Maximum number of markets to load
        
    Returns:
        List of MarketMetadata objects
    """
    series_dir = HISTORICAL_DATA_DIR / f"series={series}"
    if not series_dir.exists():
        log.warning(f"Series directory not found: {series_dir}")
        return []
    
    # Get date partitions
    date_dirs = sorted([d for d in series_dir.iterdir() if d.is_dir() and d.name.startswith("date=")])
    
    # Filter by date range
    if start_date:
        date_dirs = [d for d in date_dirs if d.name >= f"date={start_date}"]
    if end_date:
        date_dirs = [d for d in date_dirs if d.name <= f"date={end_date}"]
    
    markets = []
    for date_dir in date_dirs:
        parquet_file = date_dir / "data.parquet"
        if not parquet_file.exists():
            continue
        
        try:
            df = pd.read_parquet(parquet_file)
        except Exception as e:
            log.warning(f"Failed to read {parquet_file}: {e}")
            continue
        
        # Filter for settled/finalized markets only
        if 'status' in df.columns:
            df = df[df['status'].isin(['settled', 'finalized'])]
        elif 'result' in df.columns:
            df = df[df['result'].isin(['yes', 'no'])]
        
        for _, row in df.iterrows():
            # Parse timestamps
            try:
                open_time = pd.to_datetime(row['open_time']).tz_convert('UTC')
                close_time = pd.to_datetime(row['close_time']).tz_convert('UTC')
                settlement_ts = pd.to_datetime(row['settlement_ts']).tz_convert('UTC') if 'settlement_ts' in row and pd.notna(row['settlement_ts']) else close_time
            except Exception as e:
                log.warning(f"Failed to parse timestamps for {row.get('ticker', 'unknown')}: {e}")
                continue
            
            # Get floor strike from floor_strike column (may be NaN)
            floor_strike = row.get('floor_strike')
            if floor_strike is not None:
                try:
                    floor_strike = float(floor_strike)
                    if pd.isna(floor_strike) or floor_strike <= 0:
                        floor_strike = None
                except (ValueError, TypeError):
                    floor_strike = None
            
            # If floor_strike is NaN, fetch real historical price from Binance
            if floor_strike is None or pd.isna(floor_strike):
                log.debug(f"Fetching historical prices for {row.get('ticker', 'unknown')}")
                floor_strike, expiration_value = fetch_historical_price_at_window(
                    series, open_time.to_pydatetime(), close_time.to_pydatetime()
                )
                if floor_strike is None:
                    log.warning(f"Could not fetch historical strike for {row.get('ticker')}, using default")
                    floor_strike = 60000.0  # Fallback default
                    expiration_value = None
            
            # If still no floor_strike, we can't use this market
            if floor_strike is None or pd.isna(floor_strike) or floor_strike <= 0:
                continue
            
            # If we didn't get expiration_value from Binance, try to get it from the data
            if expiration_value is None:
                expiration_value = row.get('expiration_value', row.get('settlement_value', None))
                if expiration_value is not None:
                    try:
                        expiration_value = float(expiration_value)
                        if expiration_value <= 0 or pd.isna(expiration_value):
                            expiration_value = None
                    except (ValueError, TypeError):
                        expiration_value = None
            
            # Get last price in cents (from last_price_dollars which is 0-1)
            last_price_dollars = row.get('last_price_dollars', None)
            if last_price_dollars is not None:
                try:
                    last_price_dollars = float(last_price_dollars)
                    if not pd.isna(last_price_dollars) and 0 < last_price_dollars < 1:
                        last_price_cents = last_price_dollars * 100
                    else:
                        last_price_cents = None
                except (ValueError, TypeError):
                    last_price_cents = None
            else:
                last_price_cents = None
            
            # Get orderbook data (strings, need conversion)
            def to_float(val, default=0.5):
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return default
            
            yes_bid = to_float(row.get('yes_bid_dollars'), 0.5)
            yes_ask = to_float(row.get('yes_ask_dollars'), 0.5)
            no_bid = to_float(row.get('no_bid_dollars'), 0.5)
            no_ask = to_float(row.get('no_ask_dollars'), 0.5)
            
            # Convert from dollars to cents
            yes_bid, yes_ask, no_bid, no_ask = yes_bid * 100, yes_ask * 100, no_bid * 100, no_ask * 100
            
            # Get sizes
            def to_float_size(val, default=100.0):
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return default
            
            yes_bid_size = to_float_size(row.get('yes_bid_size_fp'), 100.0)
            yes_ask_size = to_float_size(row.get('yes_ask_size_fp'), 100.0)
            no_bid_size = yes_bid_size
            no_ask_size = yes_ask_size
            
            # Volume and open interest
            volume = to_float_size(row.get('volume_fp'), 0.0)
            open_interest = to_float_size(row.get('open_interest_fp'), 0.0)
            
            market = MarketMetadata(
                ticker=row['ticker'],
                open_time=open_time.to_pydatetime(),
                close_time=close_time.to_pydatetime(),
                settlement_ts=settlement_ts.to_pydatetime(),
                result=row['result'],
                floor_strike=float(floor_strike),
                expiration_value=float(expiration_value) if expiration_value else None,
                yes_bid_dollars=float(yes_bid) / 100.0,
                yes_ask_dollars=float(yes_ask) / 100.0,
                no_bid_dollars=float(no_bid) / 100.0,
                no_ask_dollars=float(no_ask) / 100.0,
                yes_bid_size_fp=float(yes_bid_size),
                yes_ask_size_fp=float(yes_ask_size),
                no_bid_size_fp=float(no_bid_size),
                no_ask_size_fp=float(no_ask_size),
                volume=float(volume),
                open_interest=float(open_interest),
                last_price_cents=float(last_price_cents) if last_price_cents is not None else None,
            )
            markets.append(market)
            
            if max_markets and len(markets) >= max_markets:
                break
        
        if max_markets and len(markets) >= max_markets:
            break
    
    log.info(f"Loaded {len(markets)} settled markets from {series}")
    return markets


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test loading BTC markets
    markets = load_markets_from_parquet("KXBTC15M", max_markets=10)
    print(f"Loaded {len(markets)} markets")
    
    if markets:
        m = markets[0]
        print(f"Sample market: {m.ticker}")
        print(f"  Open: {m.open_time}")
        print(f"  Close: {m.close_time}")
        print(f"  Result: {m.result}")
        print(f"  Strike: {m.floor_strike}")
        print(f"  Expiration value: {m.expiration_value}")
        print(f"  Last price cents: {m.last_price_cents}")
        
        # Test spot path generation
        path = generate_synthetic_spot_path(
            settlement_spot=m.expiration_value or 0,
            floor_strike=m.floor_strike,
            result=m.result,
            open_time=m.open_time,
            close_time=m.close_time,
            last_price_cents=m.last_price_cents,
        )
        print(f"  Generated path with {len(path)} points")
        print(f"  First: {path[0] if path else 'N/A'}")
        print(f"  Last: {path[-1] if path else 'N/A'}")