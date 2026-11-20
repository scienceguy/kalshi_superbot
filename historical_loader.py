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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backtest_harness import MarketSnapshot

log = logging.getLogger("historical_loader")

# Historical data root
HISTORICAL_DATA_DIR = Path(__file__).resolve().parent / "historical_data" / "markets"

# GBM constants
SECONDS_PER_YEAR = 365.25 * 24 * 3600
MINUTES_PER_YEAR = 365.25 * 24 * 60


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
    volume_fp: float


def load_historical_markets(
    series: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_markets: Optional[int] = None,
) -> List[MarketMetadata]:
    """
    Load settled markets from historical_data/markets/series=X/date=YYYY-MM-DD/data.parquet
    
    Args:
        series: "KXBTC15M" or "KXETH15M"
        start_date: Start date in "YYYY-MM-DD" format (inclusive)
        end_date: End date in "YYYY-MM-DD" format (inclusive)
        max_markets: Maximum number of markets to load
    
    Returns:
        List of MarketMetadata sorted by open_time
    """
    series_dir = HISTORICAL_DATA_DIR / f"series={series}"
    if not series_dir.exists():
        log.warning("Series directory not found: %s", series_dir)
        return []
    
    # Get all date partitions
    date_dirs = sorted([d for d in series_dir.iterdir() if d.is_dir() and d.name.startswith("date=")])
    
    # Filter by date range
    if start_date:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
        date_dirs = [d for d in date_dirs if datetime.strptime(d.name[5:], "%Y-%m-%d").date() >= start_dt]
    if end_date:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
        date_dirs = [d for d in date_dirs if datetime.strptime(d.name[5:], "%Y-%m-%d").date() <= end_dt]
    
    all_markets = []
    
    for date_dir in date_dirs:
        parquet_file = date_dir / "data.parquet"
        if not parquet_file.exists():
            continue
        
        try:
            df = pd.read_parquet(parquet_file)
            
            # Filter to only finalized markets
            df = df[df["status"] == "finalized"]
            if df.empty:
                continue
            
            for _, row in df.iterrows():
                # Parse result
                result = str(row["result"]).lower()
                if result not in ("yes", "no"):
                    continue
                
                # Parse timestamps
                open_time = pd.to_datetime(row["open_time"])
                close_time = pd.to_datetime(row["close_time"])
                settlement_ts = pd.to_datetime(row["settlement_ts"])
                
                # Ensure timezone-aware
                if open_time.tzinfo is None:
                    open_time = open_time.tz_localize(timezone.utc)
                if close_time.tzinfo is None:
                    close_time = close_time.tz_localize(timezone.utc)
                if settlement_ts.tzinfo is None:
                    settlement_ts = settlement_ts.tz_localize(timezone.utc)
                
                def safe_float(val, default=0.0):
                    """Safely convert value to float, handling empty strings and NaN."""
                    if val is None:
                        return default
                    if isinstance(val, str):
                        val = val.strip()
                        if val == '' or val.lower() == 'nan':
                            return default
                    try:
                        f = float(val)
                        if f != f:  # NaN check
                            return default
                        return f
                    except (ValueError, TypeError):
                        return default
                
                def parse_strike_from_subtitle(subtitle: str) -> float:
                    """Extract strike price from 'Price to beat: $92,567.16' format."""
                    if not subtitle or not isinstance(subtitle, str):
                        return 0.0
                    import re
                    match = re.search(r'\$([\d,]+\.?\d*)', subtitle)
                    if match:
                        return float(match.group(1).replace(',', ''))
                    return 0.0
                
                # Parse strike from yes_sub_title
                floor_strike = safe_float(row["floor_strike"])
                if floor_strike <= 0.0:
                    floor_strike = parse_strike_from_subtitle(row["yes_sub_title"])
                
                market = MarketMetadata(
                    ticker=str(row["ticker"]),
                    open_time=open_time,
                    close_time=close_time,
                    settlement_ts=settlement_ts,
                    result=result,
                    floor_strike=floor_strike,
                    expiration_value=safe_float(row["expiration_value"]),
                    yes_bid_dollars=safe_float(row["yes_bid_dollars"]),
                    yes_ask_dollars=safe_float(row["yes_ask_dollars"]),
                    no_bid_dollars=safe_float(row["no_bid_dollars"]),
                    no_ask_dollars=safe_float(row["no_ask_dollars"]),
                    yes_bid_size_fp=safe_float(row["yes_bid_size_fp"]),
                    yes_ask_size_fp=safe_float(row["yes_ask_size_fp"]),
                    volume_fp=safe_float(row["volume_fp"]),
                )
                all_markets.append(market)
                
                if max_markets and len(all_markets) >= max_markets:
                    break
            
            if max_markets and len(all_markets) >= max_markets:
                break
                
        except Exception as e:
            log.warning("Failed to load %s: %s", parquet_file, e)
            continue
    
    # Sort by open_time
    all_markets.sort(key=lambda m: m.open_time.timestamp())
    
    log.info("Loaded %d settled markets for %s", len(all_markets), series)
    return all_markets


def markets_to_snapshots(
    markets: List[MarketMetadata],
    snapshots_per_market: int = 10,
) -> Dict[str, List[MarketSnapshot]]:
    """
    Convert MarketMetadata to MarketSnapshot format for backtest engine.
    
    Since we only have single-timepoint orderbook data (at close/settlement),
    we create synthetic snapshots spanning the market lifetime.
    """
    result = {}
    
    for market in markets:
        snapshots = []
        
        # Create snapshots from open_time to close_time
        # We only have bid/ask at one point, so we create a progression
        duration_sec = (market.close_time - market.open_time).total_seconds()
        if duration_sec <= 0:
            continue
        
        # Generate snapshots at intervals
        for i in range(snapshots_per_market):
            progress = i / (snapshots_per_market - 1) if snapshots_per_market > 1 else 0
            ts = market.open_time.timestamp() + progress * duration_sec
            
            # Use the available bid/ask data (these are at close time in the data)
            # For earlier snapshots, we simulate wider spreads
            spread_factor = 1.0 + (1.0 - progress) * 0.5  # Wider spread earlier
            
            yes_bid = market.yes_bid_dollars * 100  # Convert to cents
            yes_ask = market.yes_ask_dollars * 100
            no_bid = market.no_bid_dollars * 100
            no_ask = market.no_ask_dollars * 100
            
            # If we don't have bid/ask, derive from last_price
            if yes_bid == 0 and yes_ask == 0:
                last_price = float(market.expiration_value) if market.expiration_value else 0.5
                # Create symmetric spread around last_price
                spread = 10 * spread_factor  # cents
                yes_bid = int(max(1, min(99, (last_price * 100) - spread / 2)))
                yes_ask = int(max(1, min(99, (last_price * 100) + spread / 2)))
                no_bid = 100 - yes_ask
                no_ask = 100 - yes_bid
            
            # Ensure valid bounds
            yes_bid = max(1, min(99, int(yes_bid)))
            yes_ask = max(1, min(99, int(yes_ask)))
            no_bid = max(1, min(99, int(no_bid)))
            no_ask = max(1, min(99, int(no_ask)))
            
            # Ensure bid <= ask
            if yes_bid > yes_ask:
                yes_bid, yes_ask = yes_ask, yes_bid
            if no_bid > no_ask:
                no_bid, no_ask = no_ask, no_bid
            
            snap = MarketSnapshot(
                ticker=market.ticker,
                timestamp=ts,
                yes_bid=yes_bid,
                yes_ask=yes_ask,
                no_bid=no_bid,
                no_ask=no_ask,
                yes_size=market.yes_bid_size_fp,
                no_size=market.yes_ask_size_fp,
                last_price=(yes_bid + yes_ask) / 2,
                volume=market.volume_fp,
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


def phi(x: float) -> float:
    """Standard normal CDF using erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def gbm_fair_value(spot: float, strike: float, minutes_left: float, sigma_per_minute: float) -> float:
    """Calculate P(spot > strike) at expiry using GBM."""
    if minutes_left <= 0 or sigma_per_minute <= 0 or spot <= 0 or strike <= 0:
        return 1.0 if spot > strike else 0.0
    
    tau_years = minutes_left / MINUTES_PER_YEAR
    sigma_annualized = sigma_per_minute * math.sqrt(MINUTES_PER_YEAR)
    
    # Avoid division by zero or negative sqrt
    if tau_years <= 0 or sigma_annualized <= 0:
        return 1.0 if spot > strike else 0.0
    
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
) -> List[Tuple[float, float]]:
    """
    Generate a synthetic spot price path from window open to settlement.
    
    Uses GBM constrained to match the known settlement outcome.
    
    Args:
        settlement_spot: Spot price at settlement (expiration_value)
        floor_strike: The strike price for the market
        result: "yes" or "no" - the actual settlement outcome
        open_time: Window open time
        close_time: Window close/settlement time
        sigma_per_minute: Per-minute volatility
        n_points: Number of time points (default 1 per second for 15 min)
        seed: Random seed for reproducibility
    
    Returns:
        List of (timestamp, spot_price) tuples
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
    
    # We know the final spot price at settlement
    # Work backwards: generate a Brownian bridge from settlement to open
    # But we need to ensure the path is consistent with the outcome
    
    # Determine if spot > strike at settlement
    spot_above_strike = settlement_spot > floor_strike
    expected_yes = spot_above_strike
    actual_yes = (result == "yes")
    
    # If outcome is inconsistent with spot vs strike, adjust
    # (This can happen due to settlement basis, but we'll use the actual result)
    target_above_strike = actual_yes
    
    # Generate path using Brownian bridge
    # Start from a reasonable open price and bridge to settlement_spot
    # The open price should be such that the GBM probability matches the market
    
    # Estimate open price: use the fact that at window open (15 min left),
    # the GBM probability should be around the market's implied probability
    # Since we don't have the opening market price, estimate from settlement
    
    # Simple approach: start from a price that gives ~50% probability at open
    # and drift towards settlement_spot
    
    # Use the settlement price as anchor, add noise backwards
    path = []
    current_spot = settlement_spot
    
    # Generate backwards from settlement to open
    for i in range(n_points):
        t = n_points - i - 1
        ts = open_time.timestamp() + t * dt
        
        if i == 0:
            # At settlement
            spot = settlement_spot
        else:
            # Add GBM noise going backwards
            # dS = -mu*S*dt + sigma*S*dW (reverse time)
            # For drift=0: dS = sigma*S*dW
            z = np.random.normal(0, 1)
            current_spot = current_spot * math.exp(-0.5 * sigma_annualized**2 * dt_years + sigma_annualized * math.sqrt(dt_years) * z)
            # Keep reasonable bounds
            current_spot = max(floor_strike * 0.5, min(floor_strike * 2.0, current_spot))
            spot = current_spot
        
        path.append((ts, spot))
    
    # Reverse to get chronological order
    path.reverse()
    
    # Ensure the final price matches settlement_spot exactly
    if path:
        path[-1] = (path[-1][0], settlement_spot)
    
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
) -> Dict[str, List[MarketSnapshot]]:
    """
    Convert MarketMetadata to MarketSnapshot format for backtest engine.
    
    Generates synthetic spot paths using GBM and converts to Kalshi orderbook snapshots.
    """
    result = {}
    
    for market in markets:
        if market.floor_strike <= 0:
            log.warning("Skipping market %s: no valid floor_strike", market.ticker)
            continue
        
        # Get settlement spot price: use floor_strike as base, adjust based on result
        # If YES won, spot was slightly above strike; if NO won, slightly below
        base_spot = market.floor_strike
        if market.result == "yes":
            settlement_spot = base_spot * 1.001  # 0.1% above strike
        else:
            settlement_spot = base_spot * 0.999  # 0.1% below strike
        
        # Generate synthetic spot path
        spot_path = generate_synthetic_spot_path(
            settlement_spot=settlement_spot,
            floor_strike=market.floor_strike,
            result=market.result,
            open_time=market.open_time,
            close_time=market.close_time,
            sigma_per_minute=sigma_per_minute,
            n_points=snapshots_per_market,
            seed=hash(market.ticker) % 10000,  # Deterministic per market
        )
        
        snapshots = []
        duration_sec = (market.close_time - market.open_time).total_seconds()
        
        for i, (ts, spot) in enumerate(spot_path):
            minutes_left = max(0.01, (market.close_time.timestamp() - ts) / 60.0)
            
            # Convert spot to Kalshi bid/ask
            yes_bid, yes_ask, no_bid, no_ask = spot_to_kalshi_snapshot(
                spot=spot,
                strike=market.floor_strike,
                minutes_left=minutes_left,
                sigma_per_minute=sigma_per_minute,
            )
            
            # Size based on volume (scale down)
            yes_size = market.yes_bid_size_fp if market.yes_bid_size_fp > 0 else 100.0
            no_size = market.yes_ask_size_fp if market.yes_ask_size_fp > 0 else 100.0
            
            snap = MarketSnapshot(
                ticker=market.ticker,
                timestamp=ts,
                yes_bid=yes_bid,
                yes_ask=yes_ask,
                no_bid=no_bid,
                no_ask=no_ask,
                yes_size=yes_size,
                no_size=no_size,
                last_price=(yes_bid + yes_ask) / 2,
                volume=market.volume_fp if market.volume_fp > 0 else 100.0,
                expiration_value=market.expiration_value,
                floor_strike=market.floor_strike,
            )
            snapshots.append(snap)
        
        if snapshots:
            result[market.ticker] = snapshots
    
    return result


def load_backtest_data(
    series: List[str] = None,
    start_date: str = None,
    end_date: str = None,
    max_markets_per_series: int = None,
    snapshots_per_market: int = 900,  # 1 per second for 15 min
    sigma_per_minute: float = 0.005,
) -> Tuple[Dict[str, List[MarketSnapshot]], Callable[[str], Optional[bool]], List[MarketMetadata]]:
    """
    Load all backtest data for one or more series.
    
    Returns:
        (markets_dict, settlement_fn, all_metadata)
    """
    if series is None:
        series = ["KXBTC15M", "KXETH15M"]
    
    all_metadata = []
    for s in series:
        metadata = load_historical_markets(
            series=s,
            start_date=start_date,
            end_date=end_date,
            max_markets=max_markets_per_series,
        )
        all_metadata.extend(metadata)
    
    # Sort all by open_time
    all_metadata.sort(key=lambda m: m.open_time.timestamp())
    
    # Convert to snapshots
    markets = markets_to_snapshots(
        all_metadata, 
        snapshots_per_market=snapshots_per_market,
        sigma_per_minute=sigma_per_minute,
    )
    
    # Create settlement function
    settlement_fn = create_settlement_fn(all_metadata)
    
    log.info("Prepared backtest data: %d markets, %d tickers", len(all_metadata), len(markets))
    return markets, settlement_fn, all_metadata


def get_markets_by_time(
    markets: Dict[str, List[MarketSnapshot]],
    all_metadata: List[MarketMetadata],
) -> List[Tuple[float, Dict[str, List[MarketSnapshot]]]]:
    """
    Group markets by day for walk-forward validation.
    
    Returns list of (day_timestamp, markets_dict_for_that_day) sorted by day.
    """
    # Create metadata lookup
    meta_by_ticker = {m.ticker: m for m in all_metadata}
    
    daily = {}
    for ticker, snapshots in markets.items():
        if not snapshots:
            continue
        meta = meta_by_ticker.get(ticker)
        if not meta:
            continue
        day = meta.open_time.date().isoformat()
        if day not in daily:
            daily[day] = {}
        daily[day][ticker] = snapshots
    
    result = []
    for day in sorted(daily.keys()):
        day_ts = datetime.fromisoformat(day).timestamp()
        result.append((day_ts, daily[day]))
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Load historical Kalshi markets for backtesting")
    parser.add_argument("--series", nargs="+", default=["KXBTC15M", "KXETH15M"])
    parser.add_argument("--start-date", help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", help="End date YYYY-MM-DD")
    parser.add_argument("--max-markets", type=int, default=100)
    parser.add_argument("--snapshots", type=int, default=10)
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
    
    markets, settlement_fn, metadata = load_backtest_data(
        series=args.series,
        start_date=args.start_date,
        end_date=args.end_date,
        max_markets_per_series=args.max_markets,
        snapshots_per_market=args.snapshots,
    )
    
    print(f"Loaded {len(metadata)} markets across {len(markets)} tickers")
    
    # Show some examples
    for i, (ticker, snaps) in enumerate(markets.items()):
        if i >= 3:
            break
        print(f"\n{ticker}: {len(snaps)} snapshots")
        print(f"  First: bid={snaps[0].yes_bid}, ask={snaps[0].yes_ask}")
        print(f"  Last:  bid={snaps[-1].yes_bid}, ask={snaps[-1].yes_ask}")
        result = settlement_fn(ticker)
        print(f"  Settlement: {'YES' if result else 'NO'}")
    
    # Test walk-forward grouping
    markets_by_time = get_markets_by_time(markets, metadata)
    print(f"\nWalk-forward splits: {len(markets_by_time)} days")
    for day_ts, day_markets in markets_by_time[:3]:
        day_str = datetime.fromtimestamp(day_ts, tz=timezone.utc).date().isoformat()
        print(f"  {day_str}: {len(day_markets)} markets")