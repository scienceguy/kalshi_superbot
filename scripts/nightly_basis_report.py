#!/usr/bin/env python3
"""T13-11 Nightly basis report.

Publishes per-venue basis histogram (BRTI vs composite) to MemPalace KG and logs summary.
Run via cron or n8n nightly.
"""
import json
import logging
from datetime import datetime, timedelta, timezone

from local_db import get_local_db
from mempalace import mempalace_kg_add, mempalace_kg_query

log = logging.getLogger("nightly_basis")


def main() -> int:
    db = get_local_db()
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=1)
    cutoff_str = cutoff.isoformat()

    # Pull last 24h of basis records
    kalshi_basis = db.get_settlement_basis_history(venue="kalshi", limit=500)
    poly_basis = db.get_settlement_basis_history(venue="polymarket", limit=500)

    for venue, records in [("kalshi", kalshi_basis), ("polymarket", poly_basis)]:
        if not records:
            log.info("%s: no basis records in last 24h", venue)
            continue
        recent = [r for r in records if r["settled_at"] >= cutoff_str]
        if not recent:
            log.info("%s: no basis records after %s", venue, cutoff_str)
            continue
        bps_vals = [r["basis_bps"] for r in recent]
        avg = sum(bps_vals) / len(bps_vals)
        lo, hi = min(bps_vals), max(bps_vals)
        # simple histogram bins
        bins = {}
        for b in bps_vals:
            bin_key = int(b // 10) * 10  # 10-bps bins
            bins[bin_key] = bins.get(bin_key, 0) + 1
        hist = {str(k): v for k, v in sorted(bins.items())}

        fact = {
            "venue": venue,
            "count": len(recent),
            "avg_bps": round(avg, 2),
            "min_bps": round(lo, 2),
            "max_bps": round(hi, 2),
            "histogram": hist,
            "as_of": now.isoformat(),
        }
        # Upsert KG fact
        mempalace_kg_add(f"settlement_basis_{venue}", json.dumps(fact))
        log.info("%s basis: n=%d avg=%.2f bps min=%.2f max=%.2f", venue, len(recent), avg, lo, hi)

    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    raise SystemExit(main())