"""
polymarket_client.py — Polymarket CLOB client for cross-market arbitrage.

Polymarket (https://polymarket.com) runs a Central Limit Order Book (CLOB)
at https://clob.polymarket.com. Each market has outcome tokens (e.g., "Yes",
"No") whose price in USDC represents the implied probability (0-1) of that
outcome occurring.

No authentication is required for public read endpoints (prices, order book).
Order placement requires a signed API key, which is optional — the bot can
read prices for arbitrage detection without placing Polymarket orders.

Usage:
    client = PolymarketClient()
    prices = client.get_midpoints(token_ids)  # dict of token_id -> mid price
    book = client.get_order_book(token_id)    # order book for a token
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter

import config

log = logging.getLogger("polymarket_client")

# Polymarket CLOB API base URL
BASE_URL = "https://clob.polymarket.com"
# Gamma API for market metadata (condition IDs, token IDs)
GAMMA_URL = "https://gamma-api.polymarket.com"

_MIN_REQUEST_GAP = 0.15  # seconds between requests to avoid rate limiting


class PolymarketClient:
    """Public CLOB client for Polymarket mainnet market data."""

    def __init__(self):
        self.base_url = BASE_URL
        self.gamma_url = GAMMA_URL
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "kalshi-superbot/1.0"})
        self.session.mount("https://", HTTPAdapter(pool_connections=10, pool_maxsize=10))
        self._last_req = 0.0

    def _get(self, url: str, params: Optional[Dict] = None) -> Any:
        elapsed = time.time() - self._last_req
        if elapsed < _MIN_REQUEST_GAP:
            time.sleep(_MIN_REQUEST_GAP - elapsed)
        resp = self.session.get(url, params=params, timeout=15)
        self._last_req = time.time()
        resp.raise_for_status()
        return resp.json()

    def get_gamma_markets(self, **params) -> List[Dict]:
        """Fetch markets from Gamma API with filtering."""
        return self._get(f"{self.gamma_url}/markets", params=params)

    def get_market(self, condition_id: str) -> Optional[Dict[str, Any]]:
        """Get market details by condition ID."""
        markets = self.get_gamma_markets(condition_ids=condition_id)
        return markets[0] if markets else None

    def get_token_map(self, condition_ids: List[str]) -> Dict[str, Dict[str, str]]:
        """Map condition_id -> {yes_token_id, no_token_id}."""
        token_map = {}
        for cid in condition_ids:
            m = self.get_market(cid)
            if m and m.get("clob_token_ids"):
                tokens = m["clob_token_ids"]
                if isinstance(tokens, list) and len(tokens) >= 2:
                    token_map[cid] = {"yes": tokens[0], "no": tokens[1]}
        return token_map

    def get_midpoints(self, token_ids: List[str]) -> Dict[str, float]:
        """Fetch mid prices for token IDs."""
        if not token_ids:
            return {}
        params = {"token_ids": token_ids}
        data = self._get(f"{self.base_url}/price", params=params)
        mids = {}
        for item in data.get("data", []):
            tid = item.get("token_id")
            mid = item.get("mid")
            if tid and mid is not None:
                mids[tid] = float(mid)
        return mids

    def get_order_book(self, token_id: str, depth: int = 10) -> Optional[Dict]:
        params = {"token_id": token_id, "depth": depth}
        return self._get(f"{self.base_url}/book", params=params)


def get_market(condition_id: str) -> Optional[Dict[str, Any]]:
    """Convenience wrapper: get market details for a condition ID."""
    return get_client().get_market(condition_id)


def get_client() -> PolymarketClient:
    if not hasattr(get_client, "_instance"):
        get_client._instance = PolymarketClient()
    return get_client._instance


# ── Polymarket US (gateway.polymarket.us) ──────────────────────────────────
# Polymarket US is the CFTC-regulated US venue. Markets are keyed by slug, not
# token id. 15-minute crypto up/down events use a deterministic slug pattern:
#   {asset}-updown-15m-{YYYY-MM-DD-HHMM}z   (timestamp = UTC window start)
# Reads need no key. BBO prices come as {"value": "0.51", "currency": "USD"}.
GATEWAY_US = "https://gateway.polymarket.us"


class PolymarketUSClient:
    """Public gateway client for Polymarket US market data."""

    def __init__(self):
        self.base_url = GATEWAY_US
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "kalshi-superbot/1.0"})
        self.session.mount("https://", HTTPAdapter(pool_connections=10, pool_maxsize=10))
        self._last_req = 0.0

    def _get(self, path: str, params: Optional[Dict] = None) -> Any:
        elapsed = time.time() - self._last_req
        if elapsed < _MIN_REQUEST_GAP:
            time.sleep(_MIN_REQUEST_GAP - elapsed)
        resp = self.session.get(f"{self.base_url}{path}", params=params, timeout=15)
        self._last_req = time.time()
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def window_event_slug(asset: str, window_start: datetime) -> str:
        return f"{asset.lower()}-updown-15m-{window_start:%Y-%m-%d-%H%M}z"

    def get_updown_15m_market(self, asset: str, window_start: datetime) -> Optional[Dict]:
        """Market dict for asset's 15m window starting at window_start (UTC), else None."""
        try:
            data = self._get(f"/v1/events/slug/{self.window_event_slug(asset, window_start)}")
        except Exception as e:
            log.debug("PolymarketUS event lookup %s: %s", asset, e)
            return None
        ev = data.get("event", data) if isinstance(data, dict) else {}
        want = window_start.strftime("%Y-%m-%dT%H:%M")
        mkts = ev.get("markets") or []
        for m in mkts:
            ws = (m.get("assetPriceTerms") or {}).get("windowStart") or ""
            if ws.startswith(want):
                return m
        return mkts[0] if mkts else None

    def get_bbo(self, slug: str) -> Optional[Dict]:
        try:
            data = self._get(f"/v1/markets/{slug}/bbo")
            return data.get("marketData", data) if isinstance(data, dict) else None
        except Exception as e:
            log.warning("PolymarketUS bbo(%s) failed: %s", slug, e)
            return None

    @staticmethod
    def _px(q: Any) -> Optional[float]:
        if q is None:
            return None
        v = q.get("value") if isinstance(q, dict) else q
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    def get_yes_no_prices(self, slug: str) -> Optional[Tuple[float, float]]:
        """Live (yes_mid, no_mid) from BBO; no_mid = 1 - yes_mid (binary complement)."""
        md = self.get_bbo(slug)
        if not md:
            return None
        bid = self._px(md.get("bestBid"))
        ask = self._px(md.get("bestAsk"))
        last = self._px(md.get("lastTradePx"))
        cur = self._px(md.get("currentPx"))
        if bid is not None and ask is not None:
            yes = (bid + ask) / 2.0
        elif last is not None:
            yes = last
        elif cur is not None:
            yes = cur
        else:
            return None
        yes = min(max(yes, 0.0), 1.0)
        return yes, 1.0 - yes