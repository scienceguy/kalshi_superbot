#!/usr/bin/env python3
"""
scripts/xsec_arb_scan.py - cross-sectional arbitrage scanner (read-only).

Finds Kalshi x Polymarket US Up/Down pairs on the same asset and the same
window whose two sides sum to less than $1.00, and reports what is left after
each venue's published fee model.

Why this is the same event on both sides (evidence in docs/POLY_US_MARKETS.md):
  Kalshi KX*15M rules: "If the simple average of the sixty seconds of CF
  Benchmarks' BRTI before <close> is at least the simple average of the sixty
  seconds of CF Benchmarks' BRTI before <open>, then the market resolves to Yes."
  Polymarket US cpc-*-updown-15m-*: settles Up when the 60-second BRTI average
  at the close is >= the 60-second BRTI average at the open.
  So Kalshi YES and Polymarket US NO are exact complements, and so are Kalshi
  NO and Polymarket US YES. Residual basis: Polymarket rounds each official
  value to 2 decimals.

Fee models (published, not assumed):
  Kalshi taker        round up(M x 0.07 x C x P x (1-P)), M=1 for KX*15M
  Polymarket US taker round up(Theta x C x P x (1-P)), Theta=0.095 at Tier 1
  Maker legs pay nothing on either venue (Kalshi maker multiplier 0; Poly US
  Tier 1 has no maker rebate), so the maker/maker case is the one with edge.

This script PLACES NO ORDERS. It reads public market data only and writes
opportunity records to a JSONL file (log-only, v15 T15-30/T15-31; live trading
stays human-gated per R5).

Usage:
  python3 scripts/xsec_arb_scan.py --once
  python3 scripts/xsec_arb_scan.py --once --assets btc eth --min-edge-bps 0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import fees
from box_arbitrage import get_box_engine
from kalshi_client import KalshiClient
from polymarket_us_client import PolymarketUSClient, taker_fee_dollars as poly_taker_fee

log = logging.getLogger("xsec_arb_scan")

# Kalshi series per asset (15-minute Up/Down).
KALSHI_SERIES = {
    "btc": "KXBTC15M",
    "eth": "KXETH15M",
    "sol": "KXSOL15M",
    "xrp": "KXXRP15M",
}

# Polymarket US slug prefixes (BTC omits the repeated asset tag).
POLY_SLUG_PREFIX = {
    "btc": "cpc-btc-updown",
    "eth": "cpceth-eth-updown",
    "sol": "cpcsol-sol-updown",
    "xrp": "cpcxrp-xrp-updown",
}

UNITS_PER_DOLLAR = 10000  # box_arbitrage.py unit: 10000 = $1.00


def cents_to_units(cents: float) -> int:
    return int(round(cents * 100))


def kalshi_asks(book: Dict) -> Dict[str, Optional[float]]:
    """Derive YES/NO ask prices and their sizes from Kalshi's bid arrays.

    Kalshi publishes bids on both sides. Buying YES at 1 - best NO bid hits the
    NO bid; buying NO at 1 - best YES bid hits the YES bid.
    """
    yes_bids = book.get("yes") or []
    no_bids = book.get("no") or []
    out: Dict[str, Optional[float]] = {"yes_ask": None, "no_ask": None,
                                       "yes_ask_size": 0.0, "no_ask_size": 0.0}
    if no_bids:
        price, size = no_bids[0]
        out["yes_ask"] = 100.0 - float(price)
        out["yes_ask_size"] = float(size)
    if yes_bids:
        price, size = yes_bids[0]
        out["no_ask"] = 100.0 - float(price)
        out["no_ask_size"] = float(size)
    return out


def poly_asks(bbo) -> Dict[str, Optional[float]]:
    """Polymarket US quotes in CENTS (the API quotes dollars).

    Polymarket US trades one instrument per market (the YES side); NO is a
    sell of YES. So the NO ask costs (1 - YES bid) and the NO bid is
    (1 - YES ask).
    """
    return {
        "yes_ask": None if bbo.yes_ask is None else bbo.yes_ask * 100.0,
        "no_ask": None if bbo.no_ask is None else bbo.no_ask * 100.0,
        "yes_ask_size": bbo.yes_ask_size,
        "no_ask_size": bbo.no_ask_size,
    }


def fee_cents(venue: str, price_cents: float, contracts: float, taker: bool) -> float:
    """Fee for one leg in cents."""
    if not taker:
        return 0.0
    price = price_cents / 100.0
    if venue == "kalshi":
        return fees.taker_fee_dollars(contracts, price) * 100.0
    return poly_taker_fee(contracts, price) * 100.0


def scan_once(args) -> List[Dict]:
    """One scan pass. Returns opportunity records (may be empty)."""
    kalshi = KalshiClient(require_key=False)
    poly = PolymarketUSClient()
    engine = get_box_engine()

    # Kalshi markets per asset, keyed by close instant (UTC).
    kalshi_markets: Dict[str, Dict[str, Dict]] = {}
    for asset in args.assets:
        series = KALSHI_SERIES.get(asset)
        if not series:
            continue
        markets = kalshi.get_markets(series_ticker=series, status="open") or []
        for m in markets:
            close_time = m.get("close_time")
            if not close_time:
                continue
            kalshi_markets.setdefault(asset, {})[close_time] = m

    # Polymarket US Up/Down windows.
    poly_markets = poly.list_updown_markets(
        horizons=(args.horizon,), assets=tuple(args.assets), within_seconds=args.within_seconds
    )

    records: List[Dict] = []
    for pm in poly_markets:
        asset = pm.asset_symbol.lower()
        km = (kalshi_markets.get(asset) or {}).get(pm.window_end)
        if not km:
            continue  # no Kalshi market with the identical close instant
        ticker = km.get("ticker")
        if not ticker:
            continue

        try:
            kbook = kalshi.get_orderbook(ticker, depth=args.depth)
        except Exception as e:  # a single bad market must not kill the scan
            log.warning("kalshi book fetch failed for %s: %s", ticker, e)
            continue

        # Polymarket quotes come from the order book (price AND size at the
        # touch). The market-list quote fields are stale/crossed (observed
        # bestBidQuote=0.99 with a null ask, and bid 0.68 > ask 0.99) and
        # must never be used for an executable price.
        try:
            p = poly_asks(poly.get_quotes(pm.slug))
        except Exception as e:
            log.warning("poly book unavailable for %s: %s", pm.slug, e)
            continue

        k = kalshi_asks(kbook)
        if None in (k["yes_ask"], k["no_ask"], p["yes_ask"], p["no_ask"]):
            continue

        # Four complementary pairs: two cross-venue, two same-venue.
        candidates = [
            ("kalshi_yes+poly_no", "kalshi", "YES", k["yes_ask"], k["yes_ask_size"],
             "polymarket", "NO", p["no_ask"], p["no_ask_size"]),
            ("poly_yes+kalshi_no", "polymarket", "YES", p["yes_ask"], p["yes_ask_size"],
             "kalshi", "NO", k["no_ask"], k["no_ask_size"]),
            ("kalshi_yes+kalshi_no", "kalshi", "YES", k["yes_ask"], k["yes_ask_size"],
             "kalshi", "NO", k["no_ask"], k["no_ask_size"]),
            ("poly_yes+poly_no", "polymarket", "YES", p["yes_ask"], p["yes_ask_size"],
             "polymarket", "NO", p["no_ask"], p["no_ask_size"]),
        ]

        for name, va, sa, pa, sa_size, vb, sb, pb, sb_size in candidates:
            sizes_known = sa_size > 0 and sb_size > 0
            contracts = min(sa_size, sb_size) if sizes_known else 0.0
            if sizes_known:
                if contracts < args.min_size:
                    continue
                contracts = min(contracts, args.max_size)

            gross_cents = pa + pb
            # Fees are charged per contract on both venues, so the economics
            # scale with size. With unknown depth, report the 1-contract case
            # and never claim a size we cannot see.
            size_for_fees = contracts if sizes_known else 1.0
            taker_fees = (fee_cents(va, pa, size_for_fees, True)
                          + fee_cents(vb, pb, size_for_fees, True))
            taker_net = gross_cents + taker_fees - 100.0
            maker_net = gross_cents - 100.0
            edge_bps = maker_net * 100.0  # 1 cent = 100 bps of $1.00

            record = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "pair": name,
                "asset": asset,
                "horizon": pm.horizon,
                "close_time": pm.window_end,
                "kalshi_ticker": ticker,
                "poly_slug": pm.slug,
                "index": {"kalshi": "CF Benchmarks BRTI", "poly": pm.index_symbol},
                "legs": [
                    {"venue": va, "side": sa, "price_cents": pa, "size": sa_size},
                    {"venue": vb, "side": sb, "price_cents": pb, "size": sb_size},
                ],
                "contracts": contracts,
                "depth_known": sizes_known,
                "gross_cents": gross_cents,
                "taker_fees_cents": taker_fees,
                "taker_net_cents": taker_net,
                "maker_net_cents": maker_net,
                "maker_edge_bps": edge_bps,
                "settlement_equivalent": asset in KALSHI_SERIES,
                "note": "log-only: no order placed",
            }

            # House engine verdict (same units, same min-edge gate). It needs a
            # real size, so an unknown-depth pair is not an opportunity.
            opp = None
            if sizes_known and contracts >= 1:
                if va != vb:
                    opp = engine.detect_cross_venue_box(
                        cents_to_units(pa), int(contracts), va, ticker,
                        cents_to_units(pb), int(contracts), vb, pm.slug,
                        min_size=1, equivalence_proven=True,
                    )
                else:
                    opp = engine.detect_same_venue_box(
                        cents_to_units(pa), int(contracts), cents_to_units(pb), int(contracts),
                        ticker, min_size=1, venue=va,
                    )
            record["engine_opportunity"] = opp is not None
            if opp is not None:
                record["engine_edge_bps"] = opp.edge_bps
                record["engine_tagged_as"] = opp.tagged_as

            if sizes_known and maker_net > 0 and edge_bps >= args.min_edge_bps:
                record["actionable"] = True
                records.append(record)
            else:
                record["actionable"] = False
                if args.log_all:
                    records.append(record)

    return records


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--assets", nargs="+", default=["btc", "eth", "sol", "xrp"])
    p.add_argument("--horizon", default="15m")
    p.add_argument("--within-seconds", type=int, default=3600)
    p.add_argument("--depth", type=int, default=5)
    p.add_argument("--with-depth", action="store_true",
                   help="also fetch Polymarket BBO for size at the touch")
    p.add_argument("--min-size", type=float, default=1.0)
    p.add_argument("--max-size", type=float, default=100.0)
    p.add_argument("--min-edge-bps", type=float, default=0.0,
                   help="minimum maker/maker edge in bps of $1.00 (0 = log every gap)")
    p.add_argument("--jsonl", default="logs/xsec_arb.jsonl")
    p.add_argument("--log-all", action="store_true",
                   help="log every pair, not only those with positive edge")
    p.add_argument("--once", action="store_true", help="one pass and exit")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)sZ %(levelname)s %(message)s",
                        datefmt="%Y-%m-%dT%H:%M:%S")
    print(f"{datetime.now(timezone.utc).isoformat()}Z config.BOX_MIN_EDGE_BPS={getattr(config, 'BOX_MIN_EDGE_BPS', 'unset (engine default 50)')}")

    records = scan_once(args)

    os.makedirs(os.path.dirname(args.jsonl) or ".", exist_ok=True)
    with open(args.jsonl, "a") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    scan_ts = datetime.now(timezone.utc).isoformat() + "Z"
    print(f"\n{scan_ts} scanned")
    print(f"{len(records)} record(s) written to {args.jsonl}")
    for r in records:
        if not r.get("actionable"):
            continue
        legs = r["legs"]
        print(f"  {scan_ts} {r['pair']} {r['asset']} {r['horizon']} close={r['close_time']} "
              f"legs={legs[0]['price_cents']:.1f}c + {legs[1]['price_cents']:.1f}c "
              f"gross={r['gross_cents']:.1f}c taker_net={r['taker_net_cents']:.2f}c "
              f"maker_net={r['maker_net_cents']:.2f}c size={r['contracts']:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
