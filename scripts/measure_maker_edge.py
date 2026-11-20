#!/usr/bin/env python3
"""Measure the REAL net edge of resting (maker) bids vs taking the offer (taker).

Why this exists: the archived research says taker entries on 15-minute crypto
markets are roughly -0.4 to -0.9 cents per contract after fees, that only
resting bids on heavy favourites survive, and that the surviving edge is
short-volatility shaped (many small wins, rare large losses). None of that
can be checked without quotes observed BEFORE settlement.

The input is the live quote archive written by scripts/quote_recorder.py:

    candles(ticker, end_ts, yes_bid, yes_ask, no_bid, no_ask, ...)
    markets(ticker, series, open_ts, close_ts, result)

Fill model (deliberately pessimistic, no invented parameters):

* We rest a bid at the touch, at the price we saw when we would have placed it.
* We are filled only if a LATER quote shows the market traded THROUGH our bid
  (YES bid q fills when yes_ask <= q; NO bid q fills when no_ask <= q).
  Anything that only touches the quote is not counted, so the result is a
  lower bound: benign queue fills are missed, toxic through-fills are kept.
* Makers pay no fee on these series, so net per contract is
  (1 - q) on a win and -q on a loss. The taker comparison pays
  0.07 * p * (1 - p).

Every market is one cluster: a market settles once, so all quotes inside it
share one outcome and are not independent draws.

Usage:
    python scripts/measure_maker_edge.py
    python scripts/measure_maker_edge.py --db local_data/candles.sqlite --band 0.80 0.97
"""
import argparse
import math
import random
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

FEE_MULTIPLIER = 0.07  # Kalshi taker fee: 0.07 * p * (1 - p) per contract

BANDS = (
    (0.02, 0.10), (0.10, 0.20), (0.20, 0.30), (0.30, 0.40),
    (0.40, 0.50), (0.50, 0.60), (0.60, 0.70), (0.70, 0.80),
    (0.80, 0.90), (0.90, 0.95), (0.95, 0.99), (0.99, 1.00),
)


def taker_fee(p: float) -> float:
    """Fee per contract at price p, in dollars."""
    return FEE_MULTIPLIER * p * (1.0 - p)


@dataclass
class Observation:
    """One market, one side, one fill (or no fill)."""
    ticker: str
    side: str          # YES | NO
    price: float       # fill price in dollars (our resting bid)
    won: bool          # did our side settle the right way
    net: float         # net dollars per contract
    minutes_left: float
    filled: bool


def _parse_ts(value: str) -> Optional[datetime]:
    if not value:
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def load_market_paths(db_path: str, entry_min: float, entry_max: float
                      ) -> Dict[str, Dict]:
    """Return ticker -> {series, close_ts, result, quotes[]} for markets with quotes.

    quotes are (minutes_left, yes_bid, yes_ask, no_bid, no_ask) sorted by time.
    """
    conn = sqlite3.connect("file:%s?mode=ro" % db_path, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        markets = {}
        for row in conn.execute(
                "select ticker, series, close_ts, result from markets "
                "where result in ('yes','no')"):
            close_ts = _parse_ts(row["close_ts"])
            if close_ts is None:
                continue
            markets[row["ticker"]] = {
                "series": row["series"],
                "close_ts": close_ts,
                "result": row["result"],
                "quotes": [],
            }

        for row in conn.execute(
                "select ticker, end_ts, yes_bid, yes_ask, no_bid, no_ask "
                "from candles where yes_bid is not null and yes_ask is not null "
                "order by ticker, end_ts"):
            market = markets.get(row["ticker"])
            if market is None:
                continue
            end_ts = _parse_ts(row["end_ts"])
            if end_ts is None:
                continue
            minutes_left = (market["close_ts"] - end_ts).total_seconds() / 60.0
            yes_bid = row["yes_bid"]
            yes_ask = row["yes_ask"]
            no_bid = row["no_bid"]
            no_ask = row["no_ask"]
            if not (0.0 < yes_bid < 1.0 and 0.0 < yes_ask < 1.0):
                continue
            market["quotes"].append(
                (minutes_left, yes_bid, yes_ask, no_bid, no_ask))
    finally:
        conn.close()
    return markets


def simulate_maker(quotes: List[Tuple], side: str, entry_min: float,
                    entry_max: float, won: bool) -> Optional[Observation]:
    """Rest a bid at the touch inside the entry window; fill only on a through-move.

    Returns None when there is no quote inside the entry window (no chance to
    place the order). Returns an Observation with filled=False when the quote
    was never traded through.
    """
    entry_quotes = [q for q in quotes if entry_min <= q[0] < entry_max]
    if not entry_quotes:
        return None

    first = entry_quotes[0]
    minutes_left = first[0]
    if side == "YES":
        price = first[1]
        if price is None:
            return None
        # Fill when the YES offer later comes down to our bid or below.
        filled = any(q[2] is not None and q[2] <= price for q in entry_quotes[1:])
    else:
        price = first[4]
        if price is None:
            return None
        # NO ask = 1 - YES bid; a NO bid at q fills when NO ask <= q.
        filled = any((q[1] is not None and (1.0 - q[1]) <= price)
                     for q in entry_quotes[1:])

    if filled:
        net = (1.0 - price) if won else -price
    else:
        net = 0.0
    return Observation("", side, price, won, net, minutes_left, filled)


def simulate_taker(quotes: List[Tuple], side: str, entry_min: float,
                   entry_max: float, won: bool) -> Optional[Observation]:
    """Pay the offer at the first quote inside the entry window (baseline)."""
    entry_quotes = [q for q in quotes if entry_min <= q[0] < entry_max]
    if not entry_quotes:
        return None
    first = entry_quotes[0]
    price = first[2] if side == "YES" else first[4]
    if price is None or not (0.0 < price < 1.0):
        return None
    fee = taker_fee(price)
    net = (1.0 - price - fee) if won else (-price - fee)
    return Observation("", side, price, won, net, first[0], True)


def bootstrap_lower_bound(values: List[float], sims: int = 2000,
                          seed: int = 7, alpha: float = 0.05) -> float:
    """Clustered bootstrap: resample markets (one value each) and take the
    alpha-quantile of the mean. One value per market, so no clustering bug."""
    if not values:
        return float("nan")
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(sims):
        total = 0.0
        for _ in range(n):
            total += values[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    return means[max(0, int(math.floor(alpha * len(means))))]


def summarize(obs: List[Observation], label: str, min_markets: int) -> Dict:
    """Clustered summary: one observation per market per side."""
    by_market: Dict[str, Observation] = {}
    for o in obs:
        by_market.setdefault(o.ticker, o)
    units = list(by_market.values())
    filled = [o for o in units if o.filled]
    stats = {
        "label": label,
        "n_markets": len(units),
        "n_filled": len(filled),
        "fill_rate": (len(filled) / len(units)) if units else 0.0,
        "win_rate": (sum(1 for o in filled if o.won) / len(filled)) if filled else 0.0,
        "avg_price": (sum(o.price for o in filled) / len(filled)) if filled else 0.0,
        "net_ev": (sum(o.net for o in filled) / len(filled)) if filled else 0.0,
        "ev_lower95": bootstrap_lower_bound([o.net for o in filled]),
        "sufficient": len(units) >= min_markets and len(filled) >= 30,
    }
    print("%-22s markets=%-5d fills=%-5d fill_rate=%5.1f%% win=%6.2f%% "
          "price=%.4f netEV=%+.4f lo95=%+.4f %s"
          % (label, stats["n_markets"], stats["n_filled"], 100 * stats["fill_rate"],
             100 * stats["win_rate"], stats["avg_price"], stats["net_ev"],
             stats["ev_lower95"],
             "SUFFICIENT" if stats["sufficient"] else "insufficient evidence"))
    return stats


def run(db_path: str, entry_min: float, entry_max: float, band: Tuple[float, float],
        min_markets: int, sides=("YES", "NO")) -> Dict:
    markets = load_market_paths(db_path, entry_min, entry_max)
    maker_obs: List[Observation] = []
    taker_obs: List[Observation] = []
    for ticker, market in markets.items():
        won_yes = market["result"] == "yes"
        for side in sides:
            won = won_yes if side == "YES" else not won_yes
            m = simulate_maker(market["quotes"], side, entry_min, entry_max, won)
            if m is not None:
                m.ticker = ticker
                if band[0] <= m.price < band[1]:
                    maker_obs.append(m)
            t = simulate_taker(market["quotes"], side, entry_min, entry_max, won)
            if t is not None:
                t.ticker = ticker
                if band[0] <= t.price < band[1]:
                    taker_obs.append(t)

    print("markets with quotes: %d   entry window %.0f-%.0f min to close"
          % (len(markets), entry_min, entry_max))
    print("band %.2f-%.2f" % band)
    maker_stats = summarize(maker_obs, "MAKER (through-fill)", min_markets)
    taker_stats = summarize(taker_obs, "TAKER (at ask)", min_markets)
    return {"maker": maker_stats, "taker": taker_stats,
            "n_markets_with_quotes": len(markets)}


def main():
    parser = argparse.ArgumentParser(description="Measure maker vs taker edge")
    parser.add_argument("--db", default="local_data/candles.sqlite")
    parser.add_argument("--entry-min", type=float, default=6.0)
    parser.add_argument("--entry-max", type=float, default=11.0)
    parser.add_argument("--band", nargs=2, type=float, default=[0.80, 0.99],
                        metavar=("MIN", "MAX"))
    parser.add_argument("--min-markets", type=int, default=200,
                        help="clustered markets needed before a band can be judged")
    parser.add_argument("--all-bands", action="store_true",
                        help="print every standard band instead of --band")
    args = parser.parse_args()

    if args.all_bands:
        for band in BANDS:
            run(args.db, args.entry_min, args.entry_max, band, args.min_markets)
            print()
    else:
        run(args.db, args.entry_min, args.entry_max,
            (args.band[0], args.band[1]), args.min_markets)


if __name__ == "__main__":
    main()
