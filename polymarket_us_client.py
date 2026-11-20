"""
polymarket_us_client.py — Polymarket US public market data + fee model.

Read-only by design. Order placement is deliberately absent: live trading
stays human-gated (v15 R5), and the authenticated API (Ed25519 signing) needs
operator API keys that this host does not have.

Public API (no key):   https://gateway.polymarket.us
Authenticated API:     https://api.polymarket.us  (X-PM-Access-Key/Timestamp/Signature)

Market shape (verified live 2026-11-20):
  GET /v1/markets?categories=crypto&closed=false&endDateMin=..&endDateMax=..
      -> {"markets": [ {..., "assetPriceTerms": {...}, "marketSides": [...],
                         "orderPriceMinTickSize": 0.01, "minimumTradeQty": 0.01,
                         "feeCoefficient": 0.095} ]}
  GET /v1/markets/{slug}/bbo -> {"marketData": {bestBid, bestAsk, bidShares, askShares, ...}}
  GET /v1/markets/{slug}/book -> {"marketData": {bids: [{px:{value},qty}], offers: [...]}}

Fees (https://docs.polymarket.us/fees, effective 2026-11-14):
  taker: Fee = Theta x C x p x (1 - p), Theta by trailing 30-day taker volume:
         Tier 1 < $10,000 -> 0.095 | 2 < $50,000 -> 0.085
         3 < $250,000 -> 0.075 | 4 >= $250,000 -> 0.0695
  maker: no fee and no rebate below $10,000,000 maker volume (Tier 1).
  The live market object carries the account-independent default in
  `feeCoefficient` (0.095 observed on 2026-11-20).
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import requests

log = logging.getLogger("polymarket_us_client")

GATEWAY_BASE = "https://gateway.polymarket.us"
API_BASE = "https://api.polymarket.us"

# Trailing 30-day taker volume thresholds -> taker coefficient (Theta).
TAKER_TIERS = (
    (10_000.0, 0.095),
    (50_000.0, 0.085),
    (250_000.0, 0.075),
    (math.inf, 0.0695),
)

# Assets Polymarket US lists for Up/Down windows (live 2026-11-20).
UPDOWN_ASSETS = ("btc", "eth", "sol", "xrp")


class PolymarketUSError(Exception):
    """Raised when the Polymarket US API returns an error or unexpected body."""
    pass


def taker_theta(trailing_30d_taker_volume: float = 0.0) -> float:
    """Taker coefficient for a trailing 30-day taker volume (USD)."""
    for threshold, theta in TAKER_TIERS:
        if trailing_30d_taker_volume < threshold:
            return theta
    return TAKER_TIERS[-1][1]


def taker_fee_dollars(contracts: float, price: float, theta: float = 0.095) -> float:
    """Polymarket US taker fee: Theta x C x p x (1-p), rounded up to a cent."""
    if contracts <= 0 or price <= 0 or price >= 1:
        return 0.0
    raw = theta * contracts * price * (1.0 - price)
    return math.ceil(raw * 100 - 1e-9) / 100.0


def maker_fee_dollars(contracts: float, price: float) -> float:
    """Maker fills pay no fee and earn no rebate below $10M maker volume."""
    return 0.0


@dataclass
class UpDownMarket:
    """One Polymarket US Up/Down window."""
    slug: str
    asset: str
    horizon: str
    index_symbol: str
    window_start: str
    window_end: str
    price_to_beat: Optional[float] = None
    fee_coefficient: float = 0.095
    tick_size: float = 0.01
    minimum_trade_qty: float = 0.01
    best_bid: Optional[float] = None
    best_ask: Optional[float] = None
    raw: Dict = field(default_factory=dict)

    @property
    def asset_symbol(self) -> str:
        """Upper-case symbol (BTC/ETH/SOL/XRP) for matching against Kalshi."""
        return self.asset.upper()


@dataclass
class Book:
    """Top-of-book snapshot for one Polymarket US market, in dollars."""
    slug: str
    yes_bid: Optional[float] = None
    yes_ask: Optional[float] = None
    yes_bid_size: float = 0.0
    yes_ask_size: float = 0.0
    state: str = ""

    @property
    def no_ask(self) -> Optional[float]:
        """Buying NO at the Yes bid price: cost is 1 - yes_bid."""
        return None if self.yes_bid is None else 1.0 - self.yes_bid

    @property
    def no_bid(self) -> Optional[float]:
        """Selling NO at the Yes ask price: proceeds are 1 - yes_ask."""
        return None if self.yes_ask is None else 1.0 - self.yes_ask

    @property
    def no_ask_size(self) -> float:
        """The Yes bid queue is what a NO buy consumes."""
        return self.yes_bid_size

    @property
    def no_bid_size(self) -> float:
        return self.yes_ask_size


def _amount(value) -> Optional[float]:
    """Parse the API's {"value": "0.8300", "currency": "USD"} amounts."""
    if value is None:
        return None
    if isinstance(value, dict):
        raw = value.get("value")
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class PolymarketUSClient:
    """Read-only Polymarket US public API client (no credentials required).

    Two public hosts, both usable without an API key:
      gateway.polymarket.us - market list, market by slug
      api.polymarket.us     - per-market book and BBO
    The gateway's per-market book endpoints are Cloudflare-rate-limited for
    this host's IP (HTTP 429 / error 1015), so quote reads try the api host
    first and fall back to the gateway.
    """

    def __init__(self, base_url: str = GATEWAY_BASE, quote_url: str = API_BASE,
                 timeout: int = 10):
        self.base_url = base_url.rstrip("/")
        self.quote_url = quote_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    # ── HTTP ──────────────────────────────────────────────────────────────
    def _get(self, path: str, params: Optional[Dict] = None, retries: int = 3,
             base: Optional[str] = None) -> Dict:
        """GET with backoff on 429 (the gateway rate-limits bursts with
        Cloudflare error 1015) and on 5xx."""
        url = f"{(base or self.base_url)}{path}"
        last = ""
        for attempt in range(retries):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as e:
                last = str(e)
                time.sleep(1.0 * (attempt + 1))
                continue
            if resp.status_code == 404:
                raise PolymarketUSError(f"GET {url} -> 404 not found")
            if resp.status_code == 200:
                try:
                    return resp.json()
                except ValueError as e:
                    raise PolymarketUSError(f"GET {url} -> non-JSON body") from e
            last = f"HTTP {resp.status_code}: {resp.text[:160]}"
            if resp.status_code == 429 or resp.status_code >= 500:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise PolymarketUSError(f"GET {url} -> {last}")
        raise PolymarketUSError(f"GET {url} failed after {retries} attempts: {last}")

    # ── Markets ───────────────────────────────────────────────────────────
    def list_updown_markets(
        self,
        horizons: tuple = ("15m",),
        assets: tuple = UPDOWN_ASSETS,
        within_seconds: int = 3600,
        limit: int = 100,
    ) -> List[UpDownMarket]:
        """List active Up/Down windows ending within `within_seconds`.

        The default ``categories=crypto`` ordering buries the short-horizon
        windows behind hand-listed yearly futures, so the window is filtered by
        endDateMin/endDateMax instead of relying on the default sort.
        """
        from datetime import datetime, timedelta, timezone

        now = datetime.now(timezone.utc)
        end = now + timedelta(seconds=within_seconds)
        params = {
            "categories": "crypto",
            "closed": "false",
            "endDateMin": now.isoformat().replace("+00:00", "Z"),
            "endDateMax": end.isoformat().replace("+00:00", "Z"),
            "limit": limit,
        }
        body = self._get("/v1/markets", params)
        out: List[UpDownMarket] = []
        for m in body.get("markets", []):
            terms = m.get("assetPriceTerms") or {}
            if "UP_DOWN" not in (terms.get("marketType") or ""):
                continue
            asset = (terms.get("asset") or {}).get("symbol")
            horizon = terms.get("horizon")
            if assets and asset not in assets:
                continue
            if horizons and horizon not in horizons:
                continue
            out.append(UpDownMarket(
                slug=m.get("slug", ""),
                asset=asset or "",
                horizon=horizon or "",
                index_symbol=terms.get("indexSymbol") or "",
                window_start=terms.get("windowStart") or "",
                window_end=terms.get("windowEnd") or "",
                price_to_beat=_amount(terms.get("priceToBeat")),
                fee_coefficient=float(m.get("feeCoefficient") or 0.095),
                tick_size=float(m.get("orderPriceMinTickSize") or 0.01),
                minimum_trade_qty=float(m.get("minimumTradeQty") or 0.01),
                best_bid=_amount(m.get("bestBidQuote")),
                best_ask=_amount(m.get("bestAskQuote")),
                raw=m,
            ))
        return out

    def get_market(self, slug: str) -> Dict:
        return self._get("/v1/markets", {"slug": slug}).get("markets", [{}])[0]

    def _get_quote(self, path: str, retries: int = 3) -> Dict:
        """Read a book/BBO from the api host, falling back to the gateway."""
        last: Optional[Exception] = None
        for base in (self.quote_url, self.base_url):
            try:
                return self._get(path, None, retries, base=base)
            except PolymarketUSError as e:
                last = e
        raise last  # type: ignore[misc]

    def get_bbo(self, slug: str) -> Book:
        """Top-of-book snapshot (best bid/ask on the Up/Yes side)."""
        body = self._get_quote(f"/v1/markets/{slug}/bbo")
        md = body.get("marketData") or {}
        return Book(
            slug=slug,
            yes_bid=_amount(md.get("bestBid")),
            yes_ask=_amount(md.get("bestAsk")),
            yes_bid_size=float(md.get("bidShares") or 0.0),
            yes_ask_size=float(md.get("askShares") or 0.0),
            state=md.get("state") or "",
        )

    def get_book(self, slug: str) -> Dict:
        """Full book: {"bids": [{px, qty}], "offers": [{px, qty}]} on the Up side."""
        body = self._get_quote(f"/v1/markets/{slug}/book")
        md = body.get("marketData") or {}
        return {
            "slug": slug,
            "state": md.get("state") or "",
            "bids": md.get("bids") or [],
            "offers": md.get("offers") or [],
        }

    def get_quotes(self, slug: str) -> Book:
        """Executable YES-side quotes with size at the touch, from one book read.

        Polymarket US trades one instrument per market (the YES side); NO is
        a sell of YES. So from the YES book:
          buy YES -> best offer
          buy NO  -> sell YES into the best bid, cost = 1 - best bid
        """
        book = self.get_book(slug)
        bids, offers = book["bids"], book["offers"]
        yes_bid = yes_ask = None
        bid_size = ask_size = 0.0
        if bids:
            best = max(_amount(b.get("px")) or 0.0 for b in bids)
            yes_bid = best
            bid_size = sum(float(b.get("qty") or 0.0) for b in bids
                           if (_amount(b.get("px")) or 0.0) == best)
        if offers:
            best = min(_amount(o.get("px")) or 1.0 for o in offers)
            yes_ask = best
            ask_size = sum(float(o.get("qty") or 0.0) for o in offers
                           if (_amount(o.get("px")) or 0.0) == best)
        return Book(
            slug=slug,
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            yes_bid_size=bid_size,
            yes_ask_size=ask_size,
            state=book["state"],
        )
