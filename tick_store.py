"""
tick_store.py — 1-second resolution tick storage with Parquet/SQLite partitions.

Implements T13-12: Tick store for backtest replay and model training.
- Stores per-second ticks from spot_composite (venue, symbol, px, bid, ask, exch_ts, recv_ts)
- Partitioned by day (SQLite tables) and hour (Parquet files) for query performance
- Append-only, designed for high write throughput
- Provides range queries for backtest engine replay
"""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
import os
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import config

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False

log = logging.getLogger("tick_store")

# Tick store directory
TICK_STORE_DIR = Path(__file__).resolve().parent / "tick_data"
TICK_STORE_DIR.mkdir(exist_ok=True)

# SQLite database for metadata and recent ticks (last 24-48h)
TICK_DB_DIR = Path(__file__).resolve().parent / "tick_db"
TICK_DB_DIR.mkdir(exist_ok=True)
TICK_DB_PATH = TICK_DB_DIR / "ticks.db"

# Parquet partition directory
PARQUET_DIR = TICK_STORE_DIR / "parquet"
PARQUET_DIR.mkdir(exist_ok=True)

# Partition by day for SQLite, by hour for Parquet
PARTITION_DAY_FORMAT = "%Y-%m-%d"
PARTITION_HOUR_FORMAT = "%Y-%m-%d_%H"


class TickStore:
    """Manages tick storage with SQLite (recent) + Parquet (historical)."""

    def __init__(self, db_path: str = str(TICK_DB_PATH), parquet_dir: str = str(PARQUET_DIR),
                 max_sqlite_days: int = 2):
        self.db_path = db_path
        self.parquet_dir = Path(parquet_dir)
        self.parquet_dir.mkdir(exist_ok=True)
        self.max_sqlite_days = max_sqlite_days
        self._conn = None
        self._lock = threading.RLock()
        self._write_buffer: List[Dict] = []
        self._buffer_size = 1000
        self._flush_interval = 1.0  # seconds
        self._last_flush = time.time()
        self._flush_task = None
        self._running = False

    def _get_conn(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        return self._conn

    def _init_db(self):
        """Create tick table partitioned by day."""
        conn = self._get_conn()
        with self._lock:
            # We'll create tables dynamically per day
            # But maintain a meta table for tracking partitions
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tick_partitions (
                    partition_date TEXT PRIMARY KEY,
                    table_name TEXT NOT NULL,
                    row_count INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT (datetime('now'))
                )
            """)
            # Also create a consolidated view for cross-day queries
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ticks_meta (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,           -- ISO timestamp
                    venue TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    px REAL NOT NULL,
                    bid REAL,
                    ask REAL,
                    exch_ts REAL NOT NULL,
                    recv_ts REAL NOT NULL,
                    partition_date TEXT NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ticks_meta_ts ON ticks_meta(ts)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ticks_meta_symbol ON ticks_meta(symbol, ts)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ticks_meta_partition ON ticks_meta(partition_date)")
            conn.commit()
            log.info("TickStore initialized at %s", self.db_path)

    def _get_daily_table(self, date_str: str) -> str:
        """Get or create daily tick table name."""
        table_name = f"ticks_{date_str.replace('-', '_')}"
        conn = self._get_conn()
        with self._lock:
            # Check if table exists
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,)
            ).fetchone()
            if not row:
                # Create the daily table
                conn.execute(f"""
                    CREATE TABLE {table_name} (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        ts TEXT NOT NULL,
                        venue TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        px REAL NOT NULL,
                        bid REAL,
                        ask REAL,
                        exch_ts REAL NOT NULL,
                        recv_ts REAL NOT NULL
                    )
                """)
                conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_ts ON {table_name}(ts)")
                conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_symbol ON {table_name}(symbol, ts)")
                
                # Register partition
                conn.execute(
                    "INSERT OR IGNORE INTO tick_partitions (partition_date, table_name) VALUES (?, ?)",
                    (date_str, table_name)
                )
                conn.commit()
                log.info("Created tick partition table: %s", table_name)
        return table_name

    def _get_partition_date(self, ts: float) -> str:
        """Get partition date string from timestamp."""
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime(PARTITION_DAY_FORMAT)

    def _get_partition_hour(self, ts: float) -> str:
        """Get partition hour string from timestamp."""
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime(PARTITION_HOUR_FORMAT)

    def write_tick(self, venue: str, symbol: str, px: float, bid: Optional[float],
                   ask: Optional[float], exch_ts: float, recv_ts: float) -> bool:
        """Write a single tick to the buffer (async flush to DB)."""
        ts = datetime.fromtimestamp(recv_ts, tz=timezone.utc).isoformat()
        tick = {
            "ts": ts,
            "venue": venue,
            "symbol": symbol,
            "px": px,
            "bid": bid,
            "ask": ask,
            "exch_ts": exch_ts,
            "recv_ts": recv_ts,
            "partition_date": self._get_partition_date(recv_ts),
        }
        
        self._write_buffer.append(tick)
        
        # Flush if buffer full or interval elapsed
        now = time.time()
        if len(self._write_buffer) >= self._buffer_size or (now - self._last_flush) >= self._flush_interval:
            return self._flush_buffer()
        return True

    def _flush_buffer(self) -> bool:
        """Flush write buffer to SQLite."""
        if not self._write_buffer:
            return True
            
        buffer = self._write_buffer
        self._write_buffer = []
        self._last_flush = time.time()
        
        conn = self._get_conn()
        with self._lock:
            try:
                # Group by partition date for efficient inserts
                by_partition: Dict[str, List] = {}
                for tick in buffer:
                    pdate = tick["partition_date"]
                    if pdate not in by_partition:
                        by_partition[pdate] = []
                    by_partition[pdate].append(tick)
                
                for pdate, ticks in by_partition.items():
                    table_name = self._get_daily_table(pdate)
                    
                    # Insert into daily partition table
                    placeholders = ", ".join(["?" for _ in range(8)])
                    sql = f"INSERT INTO {table_name} (ts, venue, symbol, px, bid, ask, exch_ts, recv_ts) VALUES ({placeholders})"
                    
                    values = [
                        (t["ts"], t["venue"], t["symbol"], t["px"], t["bid"], t["ask"],
                         t["exch_ts"], t["recv_ts"])
                        for t in ticks
                    ]
                    conn.executemany(sql, values)
                    
                    # Also insert into meta table for cross-partition queries
                    meta_sql = "INSERT INTO ticks_meta (ts, venue, symbol, px, bid, ask, exch_ts, recv_ts, partition_date) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
                    meta_values = [
                        (t["ts"], t["venue"], t["symbol"], t["px"], t["bid"], t["ask"],
                         t["exch_ts"], t["recv_ts"], pdate)
                        for t in ticks
                    ]
                    conn.executemany(meta_sql, meta_values)
                    
                    # Update partition row count
                    conn.execute(
                        "UPDATE tick_partitions SET row_count = row_count + ? WHERE partition_date = ?",
                        (len(ticks), pdate)
                    )
                
                conn.commit()
                return True
            except Exception as e:
                log.warning("TickStore flush error: %s", e)
                # Put back failed ticks
                self._write_buffer = buffer + self._write_buffer
                return False

    def query_ticks(self, symbol: str, start_ts: float, end_ts: float,
                    venue: Optional[str] = None, limit: int = 10000) -> List[Dict]:
        """Query ticks for a symbol in time range."""
        start_iso = datetime.fromtimestamp(start_ts, tz=timezone.utc).isoformat()
        end_iso = datetime.fromtimestamp(end_ts, tz=timezone.utc).isoformat()
        
        conn = self._get_conn()
        with self._lock:
            try:
                if venue:
                    sql = """
                        SELECT * FROM ticks_meta
                        WHERE symbol = ? AND venue = ? AND ts >= ? AND ts <= ?
                        ORDER BY ts LIMIT ?
                    """
                    params = (symbol, venue, start_iso, end_iso, limit)
                else:
                    sql = """
                        SELECT * FROM ticks_meta
                        WHERE symbol = ? AND ts >= ? AND ts <= ?
                        ORDER BY ts LIMIT ?
                    """
                    params = (symbol, start_iso, end_iso, limit)
                
                rows = conn.execute(sql, params).fetchall()
                return [dict(r) for r in rows]
            except Exception as e:
                log.warning("TickStore query error: %s", e)
                return []

    def query_ticks_all_venues(self, symbol: str, start_ts: float, end_ts: float,
                                limit: int = 10000) -> Dict[str, List[Dict]]:
        """Query ticks for all venues for a symbol in time range."""
        start_iso = datetime.fromtimestamp(start_ts, tz=timezone.utc).isoformat()
        end_iso = datetime.fromtimestamp(end_ts, tz=timezone.utc).isoformat()
        
        conn = self._get_conn()
        with self._lock:
            try:
                rows = conn.execute(
                    """
                    SELECT * FROM ticks_meta
                    WHERE symbol = ? AND ts >= ? AND ts <= ?
                    ORDER BY venue, ts LIMIT ?
                    """,
                    (symbol, start_iso, end_iso, limit),
                ).fetchall()
                
                result = {}
                for row in rows:
                    venue = row["venue"]
                    if venue not in result:
                        result[venue] = []
                    result[venue].append(dict(row))
                return result
            except Exception as e:
                log.warning("TickStore query_all_venues error: %s", e)
                return {}

    def get_latest_tick(self, symbol: str, venue: Optional[str] = None) -> Optional[Dict]:
        """Get the most recent tick for a symbol."""
        conn = self._get_conn()
        with self._lock:
            try:
                if venue:
                    row = conn.execute(
                        "SELECT * FROM ticks_meta WHERE symbol = ? AND venue = ? ORDER BY ts DESC LIMIT 1",
                        (symbol, venue),
                    ).fetchone()
                else:
                    row = conn.execute(
                        "SELECT * FROM ticks_meta WHERE symbol = ? ORDER BY ts DESC LIMIT 1",
                        (symbol,),
                    ).fetchone()
                return dict(row) if row else None
            except Exception as e:
                log.warning("TickStore get_latest error: %s", e)
                return None

    def start_flush_loop(self):
        """Start background flush loop."""
        self._running = True
        self._flush_task = threading.Thread(target=self._flush_loop, daemon=True)
        self._flush_task.start()
        log.info("TickStore flush loop started")

    def _flush_loop(self):
        """Background flush loop."""
        while self._running:
            time.sleep(self._flush_interval)
            if self._running:
                self._flush_buffer()

    def stop(self):
        """Stop background flush and flush remaining buffer."""
        self._running = False
        if self._flush_task:
            self._flush_task.join(timeout=5)
        self._flush_buffer()
        log.info("TickStore stopped")

    def export_to_parquet(self, partition_date: str = None) -> bool:
        """Export a day's ticks to Parquet for long-term storage.
        
        If partition_date is None, exports all partitions older than max_sqlite_days.
        """
        if not PARQUET_AVAILABLE:
            log.warning("Parquet export skipped: pyarrow not available")
            return False
            
        conn = self._get_conn()
        with self._lock:
            try:
                if partition_date:
                    dates = [partition_date]
                else:
                    # Find partitions older than max_sqlite_days
                    cutoff = datetime.now(timezone.utc).date()
                    rows = conn.execute(
                        "SELECT partition_date FROM tick_partitions WHERE row_count > 0"
                    ).fetchall()
                    dates = []
                    for row in rows:
                        pdate = datetime.fromisoformat(row["partition_date"]).date()
                        if (cutoff - pdate).days >= self.max_sqlite_days:
                            dates.append(row["partition_date"])
                
                for pdate in dates:
                    table_name = f"ticks_{pdate.replace('-', '_')}"
                    rows = conn.execute(f"SELECT * FROM {table_name} ORDER BY ts").fetchall()
                    
                    if not rows:
                        continue
                    
                    # Convert to Arrow table
                    data = [dict(r) for r in rows]
                    schema = pa.schema([
                        ("id", pa.int64()),
                        ("ts", pa.string()),
                        ("venue", pa.string()),
                        ("symbol", pa.string()),
                        ("px", pa.float64()),
                        ("bid", pa.float64()),
                        ("ask", pa.float64()),
                        ("exch_ts", pa.float64()),
                        ("recv_ts", pa.float64()),
                    ])
                    
                    arrays = {
                        "id": pa.array([r["id"] for r in data]),
                        "ts": pa.array([r["ts"] for r in data]),
                        "venue": pa.array([r["venue"] for r in data]),
                        "symbol": pa.array([r["symbol"] for r in data]),
                        "px": pa.array([r["px"] for r in data]),
                        "bid": pa.array([r["bid"] if r["bid"] is not None else float("nan") for r in data]),
                        "ask": pa.array([r["ask"] if r["ask"] is not None else float("nan") for r in data]),
                        "exch_ts": pa.array([r["exch_ts"] for r in data]),
                        "recv_ts": pa.array([r["recv_ts"] for r in data]),
                    }
                    table = pa.Table.from_arrays(list(arrays.values()), schema=schema)
                    
                    # Write Parquet file (partitioned by hour)
                    pdate_obj = datetime.fromisoformat(pdate)
                    year_month = pdate_obj.strftime("%Y-%m")
                    day_dir = self.parquet_dir / year_month / pdate
                    day_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Partition by hour
                    for hour in range(24):
                        hour_str = f"{hour:02d}"
                        hour_data = [r for r in data if r["ts"].startswith(f"{pdate}T{hour_str}:")]
                        if not hour_data:
                            continue
                        
                        hour_arrays = {k: pa.array([r[k] for r in hour_data]) for k in arrays}
                        hour_table = pa.Table.from_arrays(list(hour_arrays.values()), schema=schema)
                        
                        parquet_file = day_dir / f"{pdate}_{hour_str}.parquet"
                        pq.write_table(hour_table, parquet_file, compression="snappy")
                    
                    log.info("Exported %d ticks for %s to Parquet", len(data), pdate)
                    
                    # Optionally drop SQLite partition after successful export
                    # conn.execute(f"DROP TABLE IF EXISTS {table_name}")
                    # conn.execute("DELETE FROM tick_partitions WHERE partition_date = ?", (pdate,))
                    # conn.commit()
                
                return True
            except Exception as e:
                log.warning("Parquet export error: %s", e)
                return False

    def get_partition_info(self) -> List[Dict]:
        """Get info about all tick partitions."""
        conn = self._get_conn()
        with self._lock:
            try:
                rows = conn.execute(
                    "SELECT partition_date, table_name, row_count FROM tick_partitions ORDER BY partition_date"
                ).fetchall()
                return [dict(r) for r in rows]
            except Exception as e:
                log.warning("get_partition_info error: %s", e)
                return []

    def close(self):
        """Close database connection."""
        self.stop()
        if self._conn:
            self._conn.close()
            self._conn = None


# Global instance
_tick_store: Optional[TickStore] = None


def get_tick_store() -> TickStore:
    """Get or create global TickStore instance."""
    global _tick_store
    if _tick_store is None:
        _tick_store = TickStore()
        _tick_store._init_db()
    return _tick_store


async def start_tick_store():
    """Initialize and start the tick store (async)."""
    store = get_tick_store()
    store.start_flush_loop()
    return store


async def stop_tick_store():
    """Stop the tick store."""
    global _tick_store
    if _tick_store:
        _tick_store.stop()
        _tick_store.close()
        _tick_store = None


if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    
    store = TickStore()
    store._init_db()
    store.start_flush_loop()
    
    # Simulate some ticks
    import random
    base_ts = time.time()
    for i in range(100):
        ts = base_ts + i
        store.write_tick(
            venue="binance" if i % 2 == 0 else "coinbase",
            symbol="BTCUSDT",
            px=62000 + random.uniform(-10, 10),
            bid=61999 + random.uniform(-5, 5),
            ask=62001 + random.uniform(-5, 5),
            exch_ts=ts - 0.01,
            recv_ts=ts,
        )
    
    # Wait for flush
    time.sleep(2)
    
    # Query
    ticks = store.query_ticks("BTCUSDT", base_ts, base_ts + 100)
    print(f"Queried {len(ticks)} ticks")
    
    # Get latest
    latest = store.get_latest_tick("BTCUSDT")
    print(f"Latest tick: {latest}")
    
    # Partition info
    print("Partitions:", store.get_partition_info())
    
    store.stop()
    print("Tick store test complete")