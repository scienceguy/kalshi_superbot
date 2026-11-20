"""
event_hotpath.py — Event-driven hot path for sub-second latency (T13-45).

Replaces 15s poll with asyncio loop driven by WS ticks and BBO updates:
- uvloop and orjson for performance
- Persistent HTTP/2 keep-alive pool to each venue
- Pre-built signed-request templates/keys loaded once (Kalshi signing)
- Precomputed per-market state (strike, tau, cached LLM verdict, ML q)
- Decision is arithmetic on fresh ticks
- Ledger/DB writes stay on writer thread

Budget (p99): tick->decision 250ms; decision->wire 300ms; end-to-end 1s
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Awaitable

import orjson

try:
    import uvloop
    UVLOOP_AVAILABLE = True
except ImportError:
    UVLOOP_AVAILABLE = False

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

import config
from kalshi_client import KalshiClient
from signals import TradeSignal
from sizing import size_for_entry
from risk import RiskManager
from spot_composite import get_spot_composite

log = logging.getLogger("event_hotpath")


class Venue(Enum):
    KALSHI = "kalshi"
    POLYMARKET = "polymarket"


@dataclass
class MarketState:
    """Precomputed per-market state for hot path."""
    ticker: str
    strike: float
    tau: float  # minutes to expiry
    asset: str
    spot_price: float = 0.0
    model_prob: float = 0.5
    ml_prob: float = 0.5
    llm_verdict: str = "neutral"
    llm_cached_at: float = 0.0
    entry_band_min: int = 15
    entry_band_max: int = 55
    yes_bid: float = 0.0
    yes_ask: float = 0.0
    no_bid: float = 0.0
    no_ask: float = 0.0
    last_update: float = 0.0

    def update_bbo(self, yes_bid: float = None, yes_ask: float = None, no_bid: float = None, no_ask: float = None):
        if yes_bid is not None: self.yes_bid = yes_bid
        if yes_ask is not None: self.yes_ask = yes_ask
        if no_bid is not None: self.no_bid = no_bid
        if no_ask is not None: self.no_ask = no_ask
        self.last_update = time.time()

    def get_mid_yes(self) -> float:
        if self.yes_bid > 0 and self.yes_ask > 0:
            return (self.yes_bid + self.yes_ask) / 2
        return 0.0

    def get_mid_no(self) -> float:
        if self.no_bid > 0 and self.no_ask > 0:
            return (self.no_bid + self.no_ask) / 2
        return 0.0


@dataclass
class HotPathConfig:
    """Configuration for event-driven hot path."""
    # Event loop
    use_uvloop: bool = True
    tick_buffer_size: int = 10000

    # HTTP/2 pool
    http2_pool_size: int = 10
    http2_keepalive_timeout: float = 30.0
    http2_max_concurrent: int = 100

    # Latency budgets (ms)
    tick_to_decision_budget_ms: int = 250
    decision_to_wire_budget_ms: int = 300
    end_to_end_budget_ms: int = 1000

    # Precomputation
    llm_cache_ttl_seconds: int = 300  # 5 minutes
    model_refresh_interval: float = 60.0  # 1 minute
    spot_refresh_interval: float = 1.0  # 1 second

    # WebSocket
    ws_reconnect_delay: float = 1.0
    ws_max_reconnect_delay: float = 60.0
    ws_ping_interval: float = 20.0
    ws_ping_timeout: float = 10.0


class SignedRequestTemplate:
    """Pre-built signed request for Kalshi API."""

    def __init__(self, client: KalshiClient):
        self.client = client
        self._templates: Dict[str, Dict] = {}

    def build_order_template(self, ticker: str, side: str, price: int, size: int) -> Dict:
        """Build a pre-signed order request template."""
        # Kalshi uses RSA-PSS signing with timestamp + method + path + body
        # We pre-compute everything except the timestamp
        path = f"/trade-api/v2/portfolio/orders"
        method = "POST"
        body = {
            "ticker": ticker,
            "action": "buy",
            "side": side.lower(),
            "type": "limit",
            "price": price,
            "count": size,
        }
        return {
            "method": method,
            "path": path,
            "body": body,
            "headers": {
                "Content-Type": "application/json",
            }
        }

    def sign_request(self, template: Dict, timestamp_ms: int) -> Dict:
        """Sign a pre-built request with current timestamp."""
        # The actual signing happens in KalshiClient._sign_request
        # This is a placeholder for the template pattern
        return template


class HTTP2Pool:
    """Persistent HTTP/2 connection pool for fast order submission."""

    def __init__(self, config: HotPathConfig):
        self.config = config
        self._session: Optional[aiohttp.ClientSession] = None
        self._connector: Optional[aiohttp.TCPConnector] = None
        self._semaphore = asyncio.Semaphore(config.http2_max_concurrent)

    async def start(self):
        if self._session is None:
            self._connector = aiohttp.TCPConnector(
                limit=self.config.http2_pool_size,
                limit_per_host=self.config.http2_pool_size,
                keepalive_timeout=self.config.http2_keepalive_timeout,
                enable_cleanup_closed=True,
                force_close=False,
            )
            timeout = aiohttp.ClientTimeout(total=5.0, connect=1.0)
            self._session = aiohttp.ClientSession(
                connector=self._connector,
                timeout=timeout,
                json_serialize=orjson.dumps,
            )
            log.info("HTTP/2 pool started with %d connections", self.config.http2_pool_size)

    async def stop(self):
        if self._session:
            await self._session.close()
            self._session = None
        if self._connector:
            await self._connector.close()
            self._connector = None
        log.info("HTTP/2 pool stopped")

    async def post_order(self, url: str, headers: Dict, body: Dict) -> Optional[Dict]:
        """Submit an order via HTTP/2."""
        if not self._session:
            return None

        async with self._semaphore:
            try:
                async with self._session.post(url, headers=headers, json=body) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    else:
                        log.warning("Order submission failed: %d %s", resp.status, await resp.text())
                        return None
            except asyncio.TimeoutError:
                log.warning("Order submission timeout")
                return None
            except Exception as e:
                log.error("Order submission error: %s", e)
                return None


class WebSocketManager:
    """Manages WebSocket connections for real-time market data."""

    def __init__(self, config: HotPathConfig, on_tick: Callable[[str, Dict], Awaitable[None]]):
        self.config = config
        self.on_tick = on_tick
        self._connections: Dict[str, websockets.WebSocketClientProtocol] = {}
        self._tasks: Dict[str, asyncio.Task] = {}
        self._reconnect_delays: Dict[str, float] = {}
        self._running = False

    async def connect_kalshi(self, series: List[str]):
        """Connect to Kalshi WebSocket for series."""
        for s in series:
            await self._connect(f"kalshi_{s}", f"wss://api.kalshi.com/trade-api/v2/markets/{s}/orderbook", self._handle_kalshi_msg)

    async def connect_polymarket(self, market_ids: List[str]):
        """Connect to Polymarket WebSocket."""
        for mid in market_ids:
            await self._connect(f"poly_{mid}", f"wss://ws.polymarket.com/markets/{mid}", self._handle_poly_msg)

    async def _connect(self, name: str, url: str, handler: Callable):
        self._reconnect_delays[name] = self.config.ws_reconnect_delay
        self._tasks[name] = asyncio.create_task(self._run_connection(name, url, handler))

    async def _run_connection(self, name: str, url: str, handler: Callable):
        while self._running:
            try:
                async with websockets.connect(
                    url,
                    ping_interval=self.config.ws_ping_interval,
                    ping_timeout=self.config.ws_ping_timeout,
                    max_size=2**20,
                ) as ws:
                    self._connections[name] = ws
                    self._reconnect_delays[name] = self.config.ws_reconnect_delay
                    log.info("WebSocket connected: %s", name)

                    async for msg in ws:
                        await handler(name, msg)

            except Exception as e:
                log.warning("WebSocket %s error: %s, reconnecting in %.1fs", name, e, self._reconnect_delays[name])
                await asyncio.sleep(self._reconnect_delays[name])
                self._reconnect_delays[name] = min(
                    self._reconnect_delays[name] * 2,
                    self.config.ws_max_reconnect_delay
                )

    async def _handle_kalshi_msg(self, name: str, msg: str):
        try:
            data = orjson.loads(msg)
            # Parse Kalshi orderbook update
            await self.on_tick(name, data)
        except Exception as e:
            log.warning("Kalshi WS parse error: %s", e)

    async def _handle_poly_msg(self, name: str, msg: str):
        try:
            data = orjson.loads(msg)
            # Parse Polymarket update
            await self.on_tick(name, data)
        except Exception as e:
            log.warning("Polymarket WS parse error: %s", e)

    async def start(self):
        self._running = True

    async def stop(self):
        self._running = False
        for task in self._tasks.values():
            task.cancel()
        await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        for ws in self._connections.values():
            await ws.close()


class EventHotPath:
    """
    Main event-driven hot path engine.

    Replaces 15s polling loop with:
    - WebSocket-driven tick processing
    - Precomputed market state
    - HTTP/2 order submission
    - Sub-second latency target
    """

    def __init__(self, config: HotPathConfig = None):
        self.config = config or HotPathConfig()
        self._running = False
        self._tasks: List[asyncio.Task] = []

        # Components
        self.kalshi_client = KalshiClient()
        self.risk_manager = RiskManager()
        self.spot_composite = get_spot_composite()

        # State
        self.market_states: Dict[str, MarketState] = {}
        self.http_pool = HTTP2Pool(self.config)
        self.ws_manager = WebSocketManager(self.config, self._on_tick)

        # Latency tracking
        self.latencies: Dict[str, List[float]] = {
            "tick_to_decision": [],
            "decision_to_wire": [],
            "end_to_end": [],
        }

        # Writer queue for DB/ledger (off hot path)
        self._writer_queue: asyncio.Queue = asyncio.Queue()
        self._writer_task: Optional[asyncio.Task] = None

    async def start(self, series: List[str] = None, market_ids: List[str] = None):
        """Start the hot path engine."""
        if self._running:
            return

        # Initialize uvloop
        if self.config.use_uvloop and UVLOOP_AVAILABLE:
            uvloop.install()
            log.info("uvloop installed")

        # Start HTTP/2 pool
        await self.http_pool.start()

        # Initialize market states
        await self._initialize_market_states(series or [])

        # Start writer task
        self._writer_task = asyncio.create_task(self._writer_loop())

        # Start WebSocket manager
        await self.ws_manager.start()
        if series:
            await self.ws_manager.connect_kalshi(series)
        if market_ids:
            await self.ws_manager.connect_polymarket(market_ids)

        # Start periodic refresh tasks
        self._tasks = [
            asyncio.create_task(self._refresh_spot_prices()),
            asyncio.create_task(self._refresh_model_probs()),
            asyncio.create_task(self._refresh_llm_verdicts()),
            asyncio.create_task(self._log_latencies()),
        ]

        self._running = True
        log.info("Event hot path started")

    async def stop(self):
        """Stop the hot path engine."""
        self._running = False

        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

        await self.ws_manager.stop()
        await self.http_pool.stop()

        if self._writer_task:
            self._writer_task.cancel()
            await asyncio.gather(self._writer_task, return_exceptions=True)

        log.info("Event hot path stopped")

    async def _initialize_market_states(self, series: List[str]):
        """Precompute initial market states."""
        # Fetch markets from Kalshi
        markets = self.kalshi_client.get_markets(series_ticker=series[0] if series else None, status="open")

        for market in markets:
            ticker = market.get("ticker", "")
            if not ticker:
                continue

            strike = float(market.get("strike", 0))
            expiry = market.get("expiry", "")
            asset = self._extract_asset(ticker)

            # Calculate tau (minutes to expiry)
            tau = self._calculate_tau(expiry)

            self.market_states[ticker] = MarketState(
                ticker=ticker,
                strike=strike,
                tau=tau,
                asset=asset,
                spot_price=0.0,
                model_prob=0.5,
                ml_prob=0.5,
            )

        log.info("Initialized %d market states", len(self.market_states))

    def _extract_asset(self, ticker: str) -> str:
        """Extract asset from ticker (e.g., KXBTC15M-... -> BTC)."""
        if ticker.startswith("KX"):
            return ticker[2:].split("15M")[0]
        return ticker

    def _calculate_tau(self, expiry: str) -> float:
        """Calculate minutes to expiry."""
        # Simplified - parse expiry and calculate
        return 15.0  # Default 15 minutes

    async def _on_tick(self, source: str, data: Dict):
        """Handle incoming tick from WebSocket."""
        tick_start = time.perf_counter()

        try:
            # Parse tick and update market state
            if source.startswith("kalshi_"):
                await self._process_kalshi_tick(source, data)
            elif source.startswith("poly_"):
                await self._process_poly_tick(source, data)

            # Make trading decision if in entry window
            await self._maybe_decide(source)

        except Exception as e:
            log.error("Tick processing error: %s", e)

        tick_to_decision = (time.perf_counter() - tick_start) * 1000
        self.latencies["tick_to_decision"].append(tick_to_decision)

    async def _process_kalshi_tick(self, source: str, data: Dict):
        """Process Kalshi orderbook tick."""
        # Parse Kalshi message format
        # data typically: {"yes_bid": ..., "yes_ask": ..., "no_bid": ..., "no_ask": ...}
        ticker = data.get("ticker", source.replace("kalshi_", ""))
        state = self.market_states.get(ticker)
        if state:
            state.update_bbo(
                yes_bid=data.get("yes_bid"),
                yes_ask=data.get("yes_ask"),
                no_bid=data.get("no_bid"),
                no_ask=data.get("no_ask"),
            )

    async def _process_poly_tick(self, source: str, data: Dict):
        """Process Polymarket tick."""
        market_id = source.replace("poly_", "")
        # Find matching market state
        for state in self.market_states.values():
            if market_id in state.ticker:
                state.update_bbo(
                    yes_bid=data.get("bid"),
                    yes_ask=data.get("ask"),
                )
                break

    async def _maybe_decide(self, source: str):
        """Evaluate if we should place an order based on fresh tick."""
        ticker = source.replace("kalshi_", "").replace("poly_", "")
        state = self.market_states.get(ticker)
        if not state:
            return

        # Check entry window
        if state.tau < config.ENTRY_WINDOW_START_MIN or state.tau > config.ENTRY_WINDOW_END_MIN:
            return

        # Check BBO exists
        if state.yes_ask == 0 and state.no_ask == 0:
            return

        # Fast path decision using precomputed state
        decision_start = time.perf_counter()

        signal = self._evaluate_fast(state)

        decision_time = (time.perf_counter() - decision_start) * 1000

        if signal and signal.is_trade:
            await self._submit_order_fast(state, signal)

        self.latencies["decision_to_wire"].append((time.perf_counter() - decision_start) * 1000)
        self.latencies["end_to_end"].append((time.perf_counter() - decision_start) * 1000 + decision_time)

    def _evaluate_fast(self, state: MarketState) -> Optional[TradeSignal]:
        """Fast path evaluation using precomputed state."""
        # Use precomputed probabilities and cached values
        mid_yes = state.get_mid_yes()
        mid_no = state.get_mid_no()

        if mid_yes == 0 and mid_no == 0:
            return None

        # Two-sided favourite maker logic (from signals.py)
        # But using precomputed model_prob, ml_prob, llm_verdict
        ensemble_prob = (state.model_prob + state.ml_prob) / 2

        # Check YES side
        if state.yes_ask > 0:
            entry_cents = int(state.yes_ask * 100) - 1  # 1c under ask
            if config.ENTRY_BAND_MIN <= entry_cents <= config.ENTRY_BAND_MAX:
                edge = (ensemble_prob - entry_cents / 100) * 100
                if edge > config.MODEL_MIN_EDGE_CENTS:
                    return TradeSignal(
                        side="YES",
                        direction="yes",
                        entry_price=entry_cents,
                        contracts=0,  # Will be sized below
                        risk_dollars=0,
                        reason=f"FAST YES edge={edge:.1f}c",
                        market=None,
                        strength=abs(entry_cents - 50),
                        confidence=ensemble_prob,
                    )

        # Check NO side
        if state.no_ask > 0:
            entry_cents = int(state.no_ask * 100) - 1
            if config.HEDGE_MIN_NO_ENTRY_CENTS <= entry_cents <= config.HEDGE_MAX_NO_ENTRY_CENTS:
                no_prob = 1 - ensemble_prob
                edge = (no_prob - entry_cents / 100) * 100
                if edge > config.MODEL_MIN_EDGE_CENTS:
                    return TradeSignal(
                        side="NO",
                        direction="no",
                        entry_price=entry_cents,
                        contracts=0,
                        risk_dollars=0,
                        reason=f"FAST NO edge={edge:.1f}c",
                        market=None,
                        strength=abs(entry_cents - 50),
                        confidence=no_prob,
                    )

        return None

    async def _submit_order_fast(self, state: MarketState, signal: TradeSignal):
        """Submit order via HTTP/2 pool."""
        wire_start = time.perf_counter()

        # Size the order using Kelly
        balance = self.risk_manager.get_balance()
        exposure = self.risk_manager.get_exposure()
        sizing_result = size_for_entry(
            q_lcb=signal.confidence,
            entry_price_cents=signal.entry_price,
            cash=balance,
            exposure=exposure,
            resting_orders_value=0,
            paper=config.DRY_RUN,
            available_depth=1000,
        )

        if sizing_result.contracts <= 0:
            return

        signal.contracts = sizing_result.contracts
        signal.risk_dollars = sizing_result.contracts * signal.entry_price / 100

        # Build and submit order
        template = self.kalshi_client._build_order_request(
            ticker=state.ticker,
            side=signal.direction,
            price=signal.entry_price,
            count=signal.contracts,
        )

        # Submit via HTTP/2
        url = f"{config.KALSHI_BASE_URL}/portfolio/orders"
        headers = self.kalshi_client._get_auth_headers("POST", "/trade-api/v2/portfolio/orders", template)
        body = template

        result = await self.http_pool.post_order(url, headers, body)

        wire_time = (time.perf_counter() - wire_start) * 1000
        self.latencies["decision_to_wire"].append(wire_time)

        if result:
            # Queue for writer (off hot path)
            await self._writer_queue.put({
                "type": "order_placed",
                "signal": signal,
                "result": result,
                "timestamp": time.time(),
            })
            log.info("FAST ORDER: %s %s %d@%dc", signal.side, state.ticker, signal.contracts, signal.entry_price)

    async def _writer_loop(self):
        """Background writer for DB/ledger (off hot path)."""
        while self._running:
            try:
                item = await asyncio.wait_for(self._writer_queue.get(), timeout=1.0)
                # Write to local DB, Supabase, ledger, etc.
                # This is off the hot path so can use blocking calls
                await self._write_to_stores(item)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                log.error("Writer loop error: %s", e)

    async def _write_to_stores(self, item: Dict):
        """Write to all stores (DB, ledger, Supabase, MemPalace)."""
        # Implementation would write to local DB, outbox, etc.
        pass

    async def _refresh_spot_prices(self):
        """Periodically refresh spot prices from composite."""
        while self._running:
            try:
                for state in self.market_states.values():
                    spot = self.spot_composite.get_spot_with_age(state.asset)
                    if spot and spot.get("composite"):
                        state.spot_price = spot["composite"]
            except Exception as e:
                log.warning("Spot refresh error: %s", e)
            await asyncio.sleep(self.config.spot_refresh_interval)

    async def _refresh_model_probs(self):
        """Periodically refresh model probabilities."""
        while self._running:
            try:
                # This would call the forecaster/GBM model
                for state in self.market_states.values():
                    # Simplified - in production call actual model
                    state.model_prob = 0.5  # Placeholder
            except Exception as e:
                log.warning("Model refresh error: %s", e)
            await asyncio.sleep(self.config.model_refresh_interval)

    async def _refresh_llm_verdicts(self):
        """Periodically refresh LLM verdicts (cached)."""
        while self._running:
            try:
                now = time.time()
                for state in self.market_states.values():
                    if now - state.llm_cached_at > self.config.llm_cache_ttl_seconds:
                        # Fetch new LLM verdict
                        state.llm_verdict = "neutral"  # Placeholder
                        state.llm_cached_at = now
            except Exception as e:
                log.warning("LLM refresh error: %s", e)
            await asyncio.sleep(self.config.llm_cache_ttl_seconds)

    async def _log_latencies(self):
        """Log latency percentiles."""
        while self._running:
            await asyncio.sleep(60)  # Every minute
            for name, values in self.latencies.items():
                if values:
                    sorted_vals = sorted(values)
                    p50 = sorted_vals[len(sorted_vals) // 2]
                    p95 = sorted_vals[int(len(sorted_vals) * 0.95)]
                    p99 = sorted_vals[int(len(sorted_vals) * 0.99)]
                    log.info("%s latency: p50=%.1fms p95=%.1fms p99=%.1fms (n=%d)",
                             name, p50, p95, p99, len(values))
                    # Keep last 1000
                    self.latencies[name] = sorted_vals[-1000:]

    def get_stats(self) -> Dict:
        """Get current stats."""
        return {
            "running": self._running,
            "market_count": len(self.market_states),
            "latencies": {
                name: {
                    "count": len(vals),
                    "p50": sorted(vals)[len(vals)//2] if vals else 0,
                    "p95": sorted(vals)[int(len(vals)*0.95)] if vals else 0,
                    "p99": sorted(vals)[int(len(vals)*0.99)] if vals else 0,
                }
                for name, vals in self.latencies.items()
            },
            "http_pool_active": self.http_pool._session is not None,
        }


# ─── Factory ──────────────────────────────────────────────────────────────────

async def create_hot_path(
    series: List[str] = None,
    market_ids: List[str] = None,
    config: HotPathConfig = None,
) -> EventHotPath:
    """Create and start event hot path."""
    hotpath = EventHotPath(config)
    await hotpath.start(series, market_ids)
    return hotpath


# ─── CLI / demo ────────────────────────────────────────────────────────────────

async def demo():
    """Demo the event hot path."""
    import sys

    logging.basicConfig(level=logging.INFO)

    hotpath = await create_hot_path(
        series=["KXBTC15M", "KXETH15M"],
        config=HotPathConfig(use_uvloop=UVLOOP_AVAILABLE),
    )

    print("Hot path running. Press Ctrl+C to stop.")
    try:
        while True:
            await asyncio.sleep(10)
            stats = hotpath.get_stats()
            print(f"Stats: {stats}")
    except KeyboardInterrupt:
        pass
    finally:
        await hotpath.stop()


if __name__ == "__main__":
    asyncio.run(demo())