"""
tick_store.py — append-only tick storage for backtest replay (T13-12).

Stores compact ticks (1s resolution) in SQLite WAL partitions by date.
Provides replay() to reconstruct any market window from stored ticks.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import orjson

log = logging.getLogger("tick_store")

# ─── configuration ─────────────────────────────────────────────────────────────
DB_DIR = Path(os.environ.get("TICK_STORE_DIR", "/home/ubuntu/kalshi_superbot/tick_store"))
DB_DIR.mkdir(parents=True, exist_ok=True)

# ─── schema ────────────────────────────────────────────────────────────────────

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS ticks (
    ts INTEGER NOT NULL,              -- unix epoch seconds (1s resolution)
    symbol TEXT NOT NULL,
    venue TEXT NOT NULL,
    px REAL NOT NULL,
    bid REAL,
    ask REAL,
    exch_ts REAL NOT NULL,
    recv_ts REAL NOT NULL,
    PRIMARY KEY (ts, symbol, venue)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS idx_ticks_symbol_ts ON ticks(symbol, ts);
CREATE INDEX IF NOT EXISTS idx_ticks_venue_ts ON ticks(venue, ts);
"""

# ─── TickStore ─────────────────────────────────────────────────────────────────


class TickStore:
    """Append-only tick storage with date-partitioned SQLite databases."""

    def __init__(self, db_dir: Path = DB_DIR):
        self.db_dir = db_dir
        self._conns: Dict[str, sqlite3.Connection] = {}
        self._locks: Dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()

    def _get_conn(self, date_str: str) -> sqlite3.Connection:
        """Get (or create) a connection for a given date partition."""
        with self._global_lock:
            if date_str not in self._conns:
                db_path = self.db_dir / f"ticks_{date_str}.db"
                conn = sqlite3.connect(str(db_path), check_same_thread=False)
                conn.execute("PRAGMA journal_mode = WAL")
                conn.execute("PRAGMA synchronous = NORMAL")
                conn.execute("PRAGMA page_size = 4096")
                conn.executescript(CREATE_SQL)
                self._conns[date_str] = conn
                self._locks[date_str] = threading.Lock()
            return self._conns[date_str]

    def _get_lock(self, date_str: str) -> threading.Lock:
        self._get_conn(date_str)  # ensure lock exists
        return self._locks[date_str]

    def append(self, ticks: List[Dict[str, Any]]) -> int:
        """Append a batch of ticks. Returns count written."""
        if not ticks:
            return 0
        # Group by date partition
        by_date: Dict[str, List[Dict[str, Any]]] = {}
        for t in ticks:
            dt = datetime.fromtimestamp(t["ts"], tz=timezone.utc)
            date_str = dt.strftime("%Y-%m-%d")
            by_date.setdefault(date_str, []).append(t)

        total = 0
        for date_str, batch in by_date.items():
            conn = self._get_conn(date_str)
            lock = self._get_lock(date_str)
            with lock:
                try:
                    conn.executemany(
                        """
                        INSERT OR IGNORE INTO ticks
                        (ts, symbol, venue, px, bid, ask, exch_ts, recv_ts)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        [
                            (
                                t["ts"],
                                t["symbol"],
                                t["venue"],
                                t["px"],
                                t.get("bid"),
                                t.get("ask"),
                                t["exch_ts"],
                                t["recv_ts"],
                            )
                            for t in batch
                        ],
                    )
                    conn.commit()
                    total += conn.total_changes
                except Exception as e:
                    log.warning("tick_store append failed for %s: %s", date_str, e)
        return total

    def replay(
        self,
        symbol: str,
        start_ts: float,
        end_ts: float,
        venues: Optional[List[str]] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """Replay ticks for a symbol in [start_ts, end_ts]."""
        start_dt = datetime.fromtimestamp(start_ts, tz=timezone.utc)
        end_dt = datetime.fromtimestamp(end_ts, tz=timezone.utc)
        date_str = start_dt.strftime("%Y-%m-%d")
        conn = self._get_conn(date_str)
        where = ["symbol = ?", "ts BETWEEN ? AND ?"]
        params: List[Any] = [symbol, int(start_ts), int(end_ts)]
        if venues:
            placeholders = ",".join("?" * len(venues))
            where.append(f"venue IN ({placeholders})")
            params.extend(venues)
        sql = f"SELECT ts, symbol, venue, px, bid, ask, exch_ts, recv_ts FROM ticks WHERE {' AND '.join(where)} ORDER BY ts, venue"
        cur = conn.execute(sql, params)
        for row in cur:
            yield {
                "ts": row[0],
                "symbol": row[1],
                "venue": row[2],
                "px": row[3],
                "bid": row[4],
                "ask": row[5],
                "exch_ts": row[6],
                "recv_ts": row[7],
            }

    def get_date_range(self, symbol: str) -> Optional[tuple]:
        """Return (min_ts, max_ts) available for a symbol across all partitions."""
        min_ts = None
        max_ts = None
        for db_file in self.db_dir.glob("ticks_*.db"):
            try:
                conn = sqlite3.connect(str(db_file))
                row = conn.execute(
                    "SELECT MIN(ts), MAX(ts) FROM ticks WHERE symbol = ?", (symbol,)
                ).fetchone()
                if row and row[0] is not None:
                    if min_ts is None or row[0] < min_ts:
                        min_ts = row[0]
                    if max_ts is None or row[1] > max_ts:
                        max_ts = row[1]
            except Exception:
                pass
        return (min_ts, max_ts) if min_ts is not None else None

    def close(self):
        """Close all connections."""
        with self._global_lock:
            for conn in self._conns.values():
                try:
                    conn.close()
                except Exception:
                    pass
            self._conns.clear()
            self._locks.clear()


# ─── global instance ───────────────────────────────────────────────────────────

_tick_store: Optional[TickStore] = None


def get_tick_store() -> TickStore:
    """Get the global TickStore singleton."""
    global _tick_store
    if _tick_store is None:
        _tick_store = TickStore()
    return _tick_store


# ─── integration helpers ────────────────────────────────────────────────────────

def tick_from_spot_composite_tick(tick: "Tick") -> Dict[str, Any]:
    """Convert a spot_composite.Tick to our storage format."""
    return {
        "ts": int(tick.recv_ts),  # 1s resolution
        "symbol": tick.symbol,
        "venue": tick.venue,
        "px": tick.px,
        "bid": tick.bid,
        "ask": tick.ask,
        "exch_ts": tick.exch_ts,
        "recv_ts": tick.recv_ts,
    }


# ─── demo / CLI ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    store = get_tick_store()
    if len(sys.argv) < 2:
        print("Usage: python tick_store.py <symbol> [start_ts] [end_ts]")
        sys.exit(1)
    symbol = sys.argv[1]
    start = float(sys.argv[2]) if len(sys.argv) > 2 else 0
    end = float(sys.argv[3]) if len(sys.argv) > 3 else time.time()
    print(f"Replaying {symbol} from {start} to {end}:")
    for t in store.replay(symbol, start, end):
        print(f"  {t['ts']} {t['venue']} px={t['px']:.2f} bid={t['bid']} ask={t['ask']} age={t['recv_ts']-t['exch_ts']:.3f}s")