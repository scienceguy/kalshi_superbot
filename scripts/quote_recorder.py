#!/usr/bin/env python3
"""Record REAL Kalshi 15M order books into the sqlite schema measure_edge.py reads.

Why this exists: measure_edge.py answers "do favourite price bands actually
clear breakeven after fees?" and it is only trustworthy when it is fed quotes
observed BEFORE settlement. data_collector.py cannot do that - it fetches
already-closed markets and back-fills `last_price_dollars`, which is a
lookahead proxy for the entry price. This recorder is the honest alternative:
it walks the live books and stores the BBO it saw at that moment, and it
stores the real settlement result once the market closes.

Schema (read by scripts/measure_edge.py):

    markets(ticker, series, open_ts, close_ts, result)   result: yes|no
    candles(ticker, end_ts, yes_bid, yes_ask)             prices in dollars

Prices are dollars. Kalshi returns cents (ints) for the legacy book and
decimal strings for the fixed-point book; kalshi_client.get_orderbook
normalises both to integer cents, best bid first.

Usage:
    python scripts/quote_recorder.py --once          # one sweep, then exit
    python scripts/quote_recorder.py --interval 20   # loop forever
"""
import argparse
import logging
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kalshi_client import KalshiClient  # noqa: E402

log = logging.getLogger("quote_recorder")

SERIES = ("KXBTC15M", "KXETH15M", "KXSOL15M", "KXXRP15M", "KXDOGE15M",
          "KXBNB15M", "KXHYPE15M", "KXZEC15M", "KXNEAR15M")

DEFAULT_DB = Path(__file__).resolve().parent.parent / "local_data" / "candles.sqlite"

SCHEMA = """
CREATE TABLE IF NOT EXISTS markets (
    ticker   TEXT PRIMARY KEY,
    series   TEXT,
    open_ts  TEXT,
    close_ts TEXT,
    result   TEXT
);
CREATE TABLE IF NOT EXISTS candles (
    ticker   TEXT,
    end_ts   TEXT,
    yes_bid  REAL,
    yes_ask  REAL,
    no_bid   REAL,
    no_ask   REAL,
    yes_bid_sz REAL,
    yes_ask_sz REAL,
    PRIMARY KEY (ticker, end_ts)
);
CREATE INDEX IF NOT EXISTS idx_candles_ticker ON candles(ticker);
"""


def connect(db_path):
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def _iso(value):
    """Normalise a Kalshi timestamp to ISO-8601 UTC, or None."""
    if not value:
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text).astimezone(timezone.utc).isoformat()
    except ValueError:
        return text


def best_bid(book, side):
    """Best (highest) bid in dollars, plus its size, from a get_orderbook dict."""
    levels = book.get(side) or []
    if not levels:
        return None, 0
    price_cents, qty = levels[0]
    return price_cents / 100.0, qty


def record_sweep(conn, client, series_list=SERIES, depth=5):
    """One sweep: store BBO for every live market and refresh settled results.

    Returns (n_candles, n_markets_closed).
    """
    now = datetime.now(timezone.utc).isoformat()
    live = {}
    for series in series_list:
        try:
            client.invalidate_markets(series_ticker=series, status="open")
            for m in client.get_markets(series_ticker=series, status="open", limit=20):
                if m.get("status") in ("open", "active") and m.get("ticker"):
                    live[m["ticker"]] = m
        except Exception as exc:  # a single series failing must not stop the sweep
            log.warning("market fetch failed for %s: %s", series, exc)

    rows = []
    for ticker, market in live.items():
        try:
            book = client.get_orderbook(ticker, depth=depth)
        except Exception as exc:
            log.warning("orderbook failed for %s: %s", ticker, exc)
            continue
        yes_bid, yes_bid_sz = best_bid(book, "yes")
        no_bid, _ = best_bid(book, "no")
        # YES ask = 1 - best NO bid; NO ask = 1 - best YES bid (Kalshi books both
        # sides as bid arrays). An empty NO side means the YES ask is 1.00.
        yes_ask = 1.0 - no_bid if no_bid is not None else 1.0
        no_ask = 1.0 - yes_bid if yes_bid is not None else 1.0
        if yes_bid is None:
            continue
        rows.append((ticker, now, yes_bid, yes_ask, no_bid, no_ask,
                     yes_bid_sz, 0))

    if rows:
        conn.executemany(
            "INSERT OR REPLACE INTO candles "
            "(ticker, end_ts, yes_bid, yes_ask, no_bid, no_ask, yes_bid_sz, yes_ask_sz) "
            "VALUES (?,?,?,?,?,?,?,?)", rows)
        conn.commit()

    closed = record_closed(conn, client)
    return len(rows), closed


def record_closed(conn, client, limit=50):
    """Store settlement results for recently settled markets (idempotent).

    Kalshi's `status=closed` returns nothing for these series; `status=settled`
    returns finalized markets with `result` set to yes/no.
    """
    closed = 0
    for series in SERIES:
        try:
            client.invalidate_markets(series_ticker=series, status="settled")
            markets = client.get_markets(series_ticker=series, status="settled", limit=limit)
        except Exception as exc:
            log.warning("settled fetch failed for %s: %s", series, exc)
            continue
        for m in markets:
            result = str(m.get("result") or "").strip().lower()
            ticker = m.get("ticker")
            if not ticker or result not in ("yes", "no"):
                continue
            conn.execute(
                "INSERT OR REPLACE INTO markets (ticker, series, open_ts, close_ts, result) "
                "VALUES (?,?,?,?,?)",
                (ticker, m.get("series_ticker") or series,
                 _iso(m.get("open_time")), _iso(m.get("close_time")), result))
            closed += 1
    conn.commit()
    return closed


def main():
    parser = argparse.ArgumentParser(description="Record real Kalshi 15M quotes")
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--interval", type=float, default=20.0,
                        help="seconds between sweeps (ignored with --once)")
    parser.add_argument("--once", action="store_true", help="one sweep and exit")
    parser.add_argument("--closed-every", type=int, default=6,
                        help="refresh settlement results every N sweeps")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s - %(message)s")
    conn = connect(args.db)
    client = KalshiClient()
    sweep = 0
    while True:
        try:
            n_c, n_m = record_sweep(conn, client)
            if sweep % args.closed_every == 0:
                log.info("sweep %d: %d candles, %d closed-market rows -> %s",
                         sweep, n_c, n_m, args.db)
        except Exception as exc:
            log.exception("sweep failed (continuing): %s", exc)
        sweep += 1
        if args.once:
            break
        time.sleep(max(1.0, args.interval))
    conn.close()


if __name__ == "__main__":
    main()
