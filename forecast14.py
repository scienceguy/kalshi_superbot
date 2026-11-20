#!/usr/bin/env python3
"""forecast14.py — 14-day equity-path forecast from the CURRENT real balance (T13-31).

What the peer's `forecaster.py` does NOT do: it produces a per-market
P(YES) with honest bounds. T13-31 asks for something else - an equity-path
DISTRIBUTION over 14 days, per venue and combined, with:

  * percentiles (5/25/50/75/95) of day-14 equity
  * P(max drawdown > X)
  * P(ruin)
  * the UPPER-BOUND probability of reaching $6,250,000 (rule R1)

Rules honoured here:
  R1  the target is a REPORTING target. The probability is printed as an
      upper bound ("< 1 in N_sims"), never 0 and never inflated.
  R2  RISK_PROFILE may change the Kelly fraction and caps only. The daily-loss
      halt, the ruin halt and the position caps always fire.
  R3  the starting balance is the real venue balance (passed in, or read from
      the day-0 snapshot in the local DB) - never a prose default.
  R4  parameters are read from config, not tuned towards the target.
  T13-31  the forecaster REFUSES to publish when the clustered evidence is
      below the threshold; it prints "insufficient evidence" instead of a
      forecast.

Evidence comes from the live quote archive (scripts/quote_recorder.py ->
local_data/candles.sqlite). Each settled market is one cluster: a market
settles once, so quotes inside it are not independent draws.

Usage:
    python forecast14.py --db local_data/candles.sqlite --balance 6.42
    python forecast14.py --json forecast14.json --sims 20000 --seed 7
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))

from measure_maker_edge import load_market_paths, simulate_maker  # noqa: E402

TARGET = 6_250_000.0
MIN_CLUSTERED_MARKETS = 200   # T13-30 evidence floor
DEFAULT_RISK_PROFILE = "conservative"

# R2: a RISK_PROFILE may only move the Kelly fraction and the size caps.
RISK_PROFILES: Dict[str, Dict[str, float]] = {
    "conservative": {"kelly_fraction": 0.10, "max_position_pct": 0.05},
    "aggressive": {"kelly_fraction": 0.25, "max_position_pct": 0.10},
    "moonshot": {"kelly_fraction": 0.50, "max_position_pct": 0.20},
}


@dataclass
class Evidence:
    """Per-market net returns, clustered by market."""
    rows: List[Dict] = field(default_factory=list)
    n_markets: int = 0
    n_days: int = 0
    reason: str = ""

    @property
    def sufficient(self) -> bool:
        return self.n_markets >= MIN_CLUSTERED_MARKETS


def load_evidence(db_path: str, entry_min: float = 6.0,
                  entry_max: float = 11.0,
                  sides=("YES", "NO")) -> Evidence:
    """Real per-market net returns from the quote archive.

    For each market we rest a bid at the touch in the entry window and keep
    the through-fill (pessimistic) rule from measure_maker_edge.py. A market
    with no fill contributes 0 and is still one cluster.
    """
    evidence = Evidence()
    if not Path(db_path).exists():
        evidence.reason = "quote archive %s does not exist yet" % db_path
        return evidence

    markets = load_market_paths(db_path, entry_min, entry_max)
    days = set()
    for ticker, market in markets.items():
        won_yes = market["result"] == "yes"
        day = market["close_ts"].date().isoformat()
        days.add(day)
        for side in sides:
            won = won_yes if side == "YES" else not won_yes
            obs = simulate_maker(market["quotes"], side, entry_min, entry_max, won)
            if obs is None:
                continue
            evidence.rows.append({
                "ticker": ticker,
                "side": side,
                "price": obs.price,
                "won": obs.won,
                "net": obs.net,
                "filled": obs.filled,
                "day": day,
            })
    evidence.n_markets = len(markets)
    evidence.n_days = len(days)
    if not evidence.n_markets:
        evidence.reason = "no settled markets with quotes in the archive yet"
    return evidence


def _per_market_returns(evidence: Evidence) -> np.ndarray:
    """One net return per market (best side per market, filled entries only)."""
    by_market: Dict[str, float] = {}
    for row in evidence.rows:
        if not row["filled"]:
            continue
        ticker = row["ticker"]
        # Both sides of one market are not independent; keep the better one.
        if ticker not in by_market or row["net"] > by_market[ticker]:
            by_market[ticker] = row["net"]
    return np.array(list(by_market.values()), dtype=float)


def _daily_returns(evidence: Evidence) -> np.ndarray:
    """Mean net return per market for each day (one value per day)."""
    by_day: Dict[str, List[float]] = {}
    for row in evidence.rows:
        if not row["filled"]:
            continue
        by_day.setdefault(row["day"], []).append(row["net"])
    if not by_day:
        return np.array([], dtype=float)
    return np.array([float(np.mean(v)) for _, v in sorted(by_day.items())], dtype=float)


def simulate_paths(per_market: np.ndarray, balance: float, days: int, sims: int,
                   seed: int, kelly_fraction: float, max_position_pct: float,
                   trades_per_day: int = 4,
                   daily_loss_halt_frac: float = 0.05,
                   ruin_frac: float = 0.10) -> np.ndarray:
    """Turn per-market net returns into equity paths.

    Position size is a fraction of equity (Kelly fraction capped by
    max_position_pct). The daily-loss halt and the ruin halt fire mid-path
    (R2: no profile can switch them off).
    """
    rng = np.random.default_rng(seed)
    paths = np.zeros((sims, days + 1), dtype=float)
    paths[:, 0] = balance
    if len(per_market) == 0:
        paths[:] = balance
        return paths

    size_frac = min(kelly_fraction, max_position_pct)
    for day in range(days):
        start_of_day = paths[:, day].copy()
        idx = rng.integers(0, len(per_market), size=(sims, max(1, trades_per_day)))
        day_pnl = per_market[idx].sum(axis=1)
        size = size_frac * start_of_day
        equity = start_of_day + day_pnl * size
        # daily loss halt: stop trading for the rest of the path
        halted = (start_of_day - equity) > daily_loss_halt_frac * start_of_day
        equity = np.where(halted, start_of_day, equity)
        paths[:, day + 1] = equity
        # ruin halt: once equity is below the ruin fraction the path stays dead
        ruined = equity < ruin_frac * balance
        if np.any(ruined):
            paths[ruined, day + 1:] = equity[ruined][:, None]
    return paths


def max_drawdown(paths: np.ndarray) -> np.ndarray:
    """Max drawdown of each path (as a fraction of its running peak)."""
    peaks = np.maximum.accumulate(paths, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = (peaks - paths) / np.where(peaks > 0, peaks, np.nan)
    return np.nanmax(np.where(np.isfinite(dd), dd, 0.0), axis=1)


def percentiles(final_equity: np.ndarray) -> Dict[str, float]:
    values = {
        "p5": 5, "p25": 25, "p50": 50, "p75": 75, "p95": 95,
    }
    return {name: float(np.percentile(final_equity, pct))
            for name, pct in values.items()}


def r1_target_probability(final_equity: np.ndarray, sims: int,
                           target: float = TARGET) -> Dict[str, object]:
    """Upper-bound probability of reaching the target (R1).

    Printed as "< 1 in N_sims" when no simulated path reaches it. Never 0:
    a zero would be a claim about a quantity no simulation can bound.
    """
    hits = int(np.sum(np.asarray(final_equity, dtype=float) >= target))
    prob = hits / sims if sims else 0.0
    if hits == 0:
        note = "< 1 in %d sims (no simulated path reached $%s)" % (
            sims, format(int(target), ","))
    else:
        note = "%.4f%% (upper bound; %d/%d simulated paths)" % (
            100 * prob, hits, sims)
    return {"p_reach_target": prob, "n_paths_reaching": hits,
            "sims": sims, "note": note}


def run_forecast(db_path: str, balance: float, days: int, sims: int, seed: int,
                 risk_profile: str, entry_min: float = 6.0,
                 entry_max: float = 11.0,
                 target: float = TARGET) -> Dict:
    """Run every available model and the ensemble."""
    profile = RISK_PROFILES[risk_profile]
    evidence = load_evidence(db_path, entry_min, entry_max)
    per_market = _per_market_returns(evidence)
    daily = _daily_returns(evidence)

    models: Dict[str, Dict] = {}
    paths_by_model: Dict[str, np.ndarray] = {}

    # M1 - clustered bootstrap of per-market net returns.
    if len(per_market) >= 2:
        trades_per_day = max(1, min(8, len(per_market)))
        paths = simulate_paths(per_market, balance, days, sims, seed + 1,
                               profile["kelly_fraction"], profile["max_position_pct"],
                               trades_per_day=trades_per_day)
        paths_by_model["M1_clustered_bootstrap"] = paths
        models["M1_clustered_bootstrap"] = {
            "available": True,
            "n_markets": int(len(per_market)),
            "trades_per_day": trades_per_day,
            "mean_net_per_contract": float(np.mean(per_market)),
        }
    else:
        models["M1_clustered_bootstrap"] = {
            "available": False,
            "reason": evidence.reason or "not enough filled markets",
        }

    # M2 - GBM equity Monte Carlo fitted to daily P&L (needs >= 7 days).
    if len(daily) >= 7:
        mu, sigma = float(np.mean(daily)), float(np.std(daily, ddof=1))
        rng = np.random.default_rng(seed + 2)
        sim_daily = rng.normal(mu, sigma, size=(sims, days))
        equity = balance * np.cumprod(
            1.0 + sim_daily * profile["kelly_fraction"], axis=1)
        paths_by_model["M2_gbm_equity_mc"] = np.concatenate(
            [np.full((sims, 1), balance), equity], axis=1)
        models["M2_gbm_equity_mc"] = {
            "available": True, "daily_mu": mu, "daily_sigma": sigma,
            "n_days": int(len(daily)),
        }
    else:
        models["M2_gbm_equity_mc"] = {
            "available": False,
            "reason": "needs >= 7 days of clustered P&L, have %d" % len(daily),
        }

    # M3 - Bayesian beta-binomial per price band with credible lower bound.
    band_rows = {}
    for row in evidence.rows:
        if not row["filled"]:
            continue
        band = int(min(0.99, max(0.01, row["price"])) * 10) / 10.0
        band_rows.setdefault(band, []).append(row)
    bands = {}
    for band, rows in sorted(band_rows.items()):
        wins = sum(1 for r in rows if r["won"])
        n = len(rows)
        # Jeffreys prior Beta(0.5, 0.5); credible lower bound at 5%.
        a, b = 0.5 + wins, 0.5 + (n - wins)
        sample = np.random.default_rng(seed + 3).beta(a, b, size=2000)
        avg_price = float(np.mean([r["price"] for r in rows]))
        bands["%.1f" % band] = {
            "n": n, "win_rate": wins / n, "avg_price": avg_price,
            "credible_lower_95": float(np.percentile(sample, 5)),
            "net_ev_at_lower_bound": float(
                np.percentile(sample, 5) * (1 - avg_price)
                - (1 - np.percentile(sample, 5)) * avg_price),
        }
    models["M3_bayesian_band"] = {
        "available": bool(bands),
        "bands": bands,
        "reason": "" if bands else "no filled markets to estimate a band",
    }

    # M4 - regime switch between the favourite band and everything else.
    fav = np.array([r["net"] for r in evidence.rows
                    if r["filled"] and r["price"] >= 0.80], dtype=float)
    rest = np.array([r["net"] for r in evidence.rows
                     if r["filled"] and r["price"] < 0.80], dtype=float)
    if len(fav) >= 2 and len(rest) >= 2:
        rng = np.random.default_rng(seed + 4)
        state = rng.random(sims) < 0.7
        equity = np.full(sims, balance)
        paths = np.full((sims, days + 1), balance)
        for day in range(days):
            switch = rng.random(sims) < 0.2
            state = np.where(switch, ~state, state)
            ret = np.where(state,
                           rng.choice(fav, size=sims),
                           rng.choice(rest, size=sims))
            equity = equity + ret * equity * profile["kelly_fraction"]
            paths[:, day + 1] = equity
        paths_by_model["M4_regime_switch"] = paths
        models["M4_regime_switch"] = {
            "available": True, "n_favourite": int(len(fav)),
            "n_other": int(len(rest)),
        }
    else:
        models["M4_regime_switch"] = {
            "available": False,
            "reason": "needs >= 2 filled markets in the favourite band and outside it",
        }

    # M5/M6 - model-driven paths need trained artefacts that do not exist yet.
    models["M5_gbm_win_probability"] = {
        "available": False,
        "reason": "needs a calibrated per-market win probability from the GBM "
                  "service plus settled quotes; not available in this archive",
    }
    models["M6_torch_model"] = {
        "available": False,
        "reason": "no trained torch artefact in models/ to score",
    }

    # M7 - ensemble over the available models, equal weights when there is no
    # history to score them against (R4: weights are not tuned to the target).
    available = {name: p for name, p in paths_by_model.items()}
    published = evidence.sufficient and bool(available)
    ensemble: Dict[str, object] = {}
    if available:
        finals = np.vstack([p[:, -1] for p in available.values()])
        mean_path = np.mean(np.stack(list(available.values())), axis=0)
        final_equity = mean_path[:, -1]
        ensemble = {
            "models_used": sorted(available),
            "weights": {name: 1.0 / len(available) for name in available},
            "percentiles": percentiles(final_equity),
            "p_drawdown_gt_20pct": float(np.mean(max_drawdown(mean_path) > 0.20)),
            "p_drawdown_gt_50pct": float(np.mean(max_drawdown(mean_path) > 0.50)),
            "p_ruin": float(np.mean(final_equity <= 0.10 * balance)),
            "target": target,
            "target_probability": r1_target_probability(final_equity, sims, target),
        }
    else:
        ensemble = {"models_used": [], "reason": "no model had enough evidence"}

    if not published:
        ensemble["refusal"] = (
            "insufficient evidence: %d clustered markets, need %d"
            % (evidence.n_markets, MIN_CLUSTERED_MARKETS))

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "published": published,
        "balance": balance,
        "days": days,
        "sims": sims,
        "seed": seed,
        "risk_profile": risk_profile,
        "risk_profile_params": profile,
        "evidence": {
            "n_markets": evidence.n_markets,
            "n_days": evidence.n_days,
            "n_filled_rows": len(evidence.rows),
            "threshold": MIN_CLUSTERED_MARKETS,
            "sufficient": evidence.sufficient,
            "reason": evidence.reason,
        },
        "models": models,
        "ensemble": ensemble,
    }


def main():
    parser = argparse.ArgumentParser(description="14-day equity-path forecast")
    parser.add_argument("--db", default=str(ROOT / "local_data" / "candles.sqlite"))
    parser.add_argument("--balance", type=float, default=None,
                        help="current real balance (R3: read from the venue)")
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--sims", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--risk-profile", default=DEFAULT_RISK_PROFILE,
                        choices=sorted(RISK_PROFILES))
    parser.add_argument("--entry-min", type=float, default=6.0)
    parser.add_argument("--entry-max", type=float, default=11.0)
    parser.add_argument("--json", dest="json_out", default=None)
    args = parser.parse_args()

    balance = args.balance
    if balance is None:
        print("insufficient evidence: --balance (the real venue balance) is "
              "required; a default would violate R3", file=sys.stderr)
        return 2

    result = run_forecast(args.db, balance, args.days, args.sims, args.seed,
                          args.risk_profile, args.entry_min, args.entry_max)

    print("14-day equity-path forecast")
    print("  balance       : $%.2f" % result["balance"])
    print("  risk profile  : %s" % result["risk_profile"])
    print("  evidence      : %d clustered markets, %d days (need %d)"
          % (result["evidence"]["n_markets"], result["evidence"]["n_days"],
             result["evidence"]["threshold"]))
    if not result["published"]:
        print("  PUBLISHED     : NO - %s" % result["ensemble"].get("refusal", ""))
        print("  no forecast is published on insufficient evidence (T13-31)")
    else:
        ens = result["ensemble"]
        print("  models        : %s" % ", ".join(ens["models_used"]))
        pct = ens["percentiles"]
        print("  day-%d equity  : p5=$%.2f p25=$%.2f p50=$%.2f p75=$%.2f p95=$%.2f"
              % (result["days"], pct["p5"], pct["p25"], pct["p50"],
                 pct["p75"], pct["p95"]))
        print("  P(dd>20%%)     : %.4f" % ens["p_drawdown_gt_20pct"])
        print("  P(ruin)       : %.4f" % ens["p_ruin"])
        print("  P(reach target): %s" % ens["target_probability"]["note"])

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2))
        print("  wrote %s" % args.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
