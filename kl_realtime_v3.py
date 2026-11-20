#!/usr/bin/env python3
"""
REAL-TIME KL Divergence + Box Arbitrage Dry Run v3
- Fixed orderbook parsing (Kalshi returns yes/no in cents directly)
- Reality tax: slippage, partial fills, latency, fees
- Accurate P&L tracking with fill simulation, including fully collateralized shorts
Target: $6,250,000 on either platform.
"""
import json
import math
import time
import random
import sys
import signal
import threading
import os
from datetime import datetime, timezone, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any

sys.path.insert(0, '.')

from archive.kl_divergence_arb import (
    symmetric_kl,
    calculate_edge,
    KL_ARB_THRESHOLD,
    KL_STRONG_THRESHOLD,
    MIN_EDGE_AFTER_FEES,
)
from archive.polymarket_client import PolymarketClient, PolymarketUSClient
from kalshi_client import KalshiClient
from n8n_client import N8nClient, get_client as get_n8n_client, n8n_health_check
from supabase_client import SupabaseClient, get_supabase
import config
import fees

TARGET = 6_250_000
STARTING_BALANCE_KALSHI = 10.34
STARTING_BALANCE_POLYMARKET = 10.34
MAX_POSITION = 1000
MAX_DAILY_LOSS = 1000
MAX_EXPOSURE = 5000
KILLSWITCH_FILE = "/tmp/kl_killswitch"
STATE_FILE = "/tmp/kl_realtime_v3_state.json"
PARTIAL_FILL_PROB = 0.15
SLIPPAGE_BPS = 5
ENTRY_BAND_MIN = config.GATES_MEASURED_BAND_MIN_CENTS / 100.0
ENTRY_BAND_MAX = config.GATES_MEASURED_BAND_MAX_CENTS / 100.0
POLYMARKET_US_FEE_THETA = float(os.getenv("POLYMARKET_US_FEE_THETA", "0.117"))
ALLOW_SHORTING = os.getenv("ALLOW_SHORTING", "true").lower() == "true"
# Cross-venue settlement references disagree in the measured sample. Until
# equivalence is proven per market, keep these trades directional and reserve
# this measured disagreement rate before calling an edge tradable.
CROSS_VENUE_BASIS_BUFFER = 0.062


def polymarket_us_fee(contracts: float, price: float) -> float:
    """Polymarket US taker fee, rounded to cents with banker's rounding."""
    return round(POLYMARKET_US_FEE_THETA * contracts * price * (1.0 - price), 2)


def kalshi_taker_fee(contracts: float, price: float) -> float:
    """Kalshi taker fee rounded up to cents, with a $0.01 minimum."""
    raw = fees.FEE_RATE * contracts * price * (1.0 - price)
    return max(0.01, math.ceil((raw - 1e-12) * 100.0) / 100.0)


def venue_taker_fee(venue: str, contracts: float, price: float) -> float:
    if venue == "KALSHI":
        return kalshi_taker_fee(contracts, price)
    return polymarket_us_fee(contracts, price)


ENABLE_POLYMARKET_LIVE = os.getenv("ENABLE_POLYMARKET_LIVE", "false").lower() == "true"
KALSHI_CONFIRM_LIVE = os.getenv("KALSHI_CONFIRM_LIVE", "false").lower() == "true"
DRY_RUN = not KALSHI_CONFIRM_LIVE


@dataclass
class Position:
    ticker: str
    venue: str
    side: str
    entry_price: float
    filled_price: float
    contracts: int
    filled_contracts: int
    expiry_time: datetime
    signal_type: str
    direction: str = "LONG"
    fee_paid: float = 0.0
    margin_reserved: float = 0.0
    entry_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def payout_at_settle(self, outcome: int) -> float:
        """Payout of the held token per contract."""
        return float(outcome if self.side == "YES" else 1 - outcome)

    def pnl_at_settle(self, outcome: int) -> float:
        gross = (self.payout_at_settle(outcome) - self.filled_price) * self.filled_contracts
        if self.direction == "SHORT":
            gross = -gross
        return gross - self.fee_paid

    def settlement_cash_delta(self, outcome: int) -> float:
        """Cash/buying-power change when the position settles.

        Longs return their payout. Shorts lock $1 per contract at entry; that
        margin is released only when the short token loses.
        """
        if self.direction == "SHORT":
            return self.margin_reserved if self.payout_at_settle(outcome) == 0 else 0.0
        return self.payout_at_settle(outcome) * self.filled_contracts


class OrderBook:
    def __init__(self, ticker: str, yes_bids: List[int], no_bids: List[int], yes_asks: List[int], no_asks: List[int], ts: float):
        self.ticker = ticker
        self.yes_bids = sorted(yes_bids, reverse=True) if yes_bids else []
        self.no_bids = sorted(no_bids, reverse=True) if no_bids else []
        self.yes_asks = sorted(yes_asks) if yes_asks else []
        self.no_asks = sorted(no_asks) if no_asks else []
        self.ts = ts

    def get_yes_bid(self) -> Optional[float]:
        return self.yes_bids[0] / 100.0 if self.yes_bids else None

    def get_yes_ask(self) -> Optional[float]:
        return self.yes_asks[0] / 100.0 if self.yes_asks else None

    def get_no_bid(self) -> Optional[float]:
        return self.no_bids[0] / 100.0 if self.no_bids else None

    def get_no_ask(self) -> Optional[float]:
        return self.no_asks[0] / 100.0 if self.no_asks else None

    def is_stale(self, max_age: float = 30.0) -> bool:
        return time.time() - self.ts > max_age


class OrderBookCache:
    def __init__(self):
        self.books: Dict[str, OrderBook] = {}

    def update(self, book: OrderBook):
        self.books[book.ticker] = book

    def get(self, ticker: str) -> Optional[OrderBook]:
        return self.books.get(ticker)

    def get_all(self) -> Dict[str, OrderBook]:
        return {k: v for k, v in self.books.items() if not v.is_stale()}


class RealTimeDryRun:
    def __init__(self):
        self.running = True
        self.cycle = 0
        self.kalshi_balance = STARTING_BALANCE_KALSHI
        self.poly_balance = STARTING_BALANCE_POLYMARKET
        self.kalshi_positions: List[Position] = []
        self.poly_positions: List[Position] = []
        self.closed_trades: List[Dict] = []
        self.orderbook_cache = OrderBookCache()
        self.poly_prices: Dict[str, float] = {}
        self.poly_quotes: Dict[str, Dict[str, Dict[str, float]]] = {}
        self.kalshi_client = KalshiClient()
        self.poly_client = PolymarketClient() if ENABLE_POLYMARKET_LIVE else None
        self.poly_us = PolymarketUSClient() if ENABLE_POLYMARKET_LIVE else None
        self.n8n_client = None
        self.supabase = None
        self.accounting_version = 2
        self.balance_basis = "dry_run_state"
        self._load_state()

        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

        if n8n_health_check():
            self.n8n_client = get_n8n_client()
        else:
            print("  WARNING: n8n not reachable, webhooks disabled")
            self.n8n_client = None

        if self.supabase:
            ok = self.supabase.health_check()
            print(f"  Supabase health check: {'OK' if ok else 'FAIL'}")
            if not ok:
                print("  WARNING: Supabase not available, persistence disabled")
                self.supabase = None

        if self.poly_us:
            try:
                now = datetime.now(timezone.utc)
                win = now.replace(minute=now.minute - now.minute % 15, second=0, microsecond=0)
                m = self.poly_us.get_updown_15m_market("BTC", win)
                if m and m.get("slug"):
                    print(f"  Polymarket US: gateway reachable, current window market {m['slug']}")
                else:
                    print("  WARNING: Polymarket US returned no current-window market")
            except Exception as e:
                print(f"  WARNING: Polymarket US connectivity check failed: {e}")
                self.poly_us = None

    def _shutdown(self, signum, frame):
        print("\nShutdown signal received...")
        self.running = False
        self._save_state()

    def _load_state(self):
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE) as f:
                    state = json.load(f)
                self.cycle = state.get("cycle", 0)
                self.kalshi_balance = state.get("kalshi_balance", STARTING_BALANCE_KALSHI)
                self.poly_balance = state.get("poly_balance", STARTING_BALANCE_POLYMARKET)
                def parse_pos(p):
                    p = dict(p)
                    p['expiry_time'] = datetime.fromisoformat(p['expiry_time'])
                    p['entry_time'] = datetime.fromisoformat(p['entry_time'])
                    return Position(**p)
                self.kalshi_positions = [parse_pos(p) for p in state.get("kalshi_positions", [])]
                self.poly_positions = [parse_pos(p) for p in state.get("poly_positions", [])]
                self.closed_trades = state.get("closed_trades", [])
                self.accounting_version = state.get("accounting_version", 1)
                self.balance_basis = state.get("balance_basis", "legacy_dry_run_state")
                if self.accounting_version < 2:
                    print("  WARNING: legacy cash accounting loaded; new entries use corrected v2 accounting")
                print(f"  State loaded: cycle {self.cycle}, K=${self.kalshi_balance:.2f}, P=${self.poly_balance:.2f}, open_k={len(self.kalshi_positions)}, open_p={len(self.poly_positions)}")
            except Exception as e:
                print(f"  WARNING: Failed to load state: {e}")

    def _save_state(self):
        def pos_to_dict(p):
            d = asdict(p)
            d['expiry_time'] = p.expiry_time.isoformat()
            d['entry_time'] = p.entry_time.isoformat()
            return d
        state = {
            "cycle": self.cycle,
            "kalshi_balance": self.kalshi_balance,
            "poly_balance": self.poly_balance,
            "kalshi_positions": [pos_to_dict(p) for p in self.kalshi_positions],
            "poly_positions": [pos_to_dict(p) for p in self.poly_positions],
            "closed_trades": self.closed_trades,
            "accounting_version": self.accounting_version,
            "balance_basis": self.balance_basis,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        try:
            with open(STATE_FILE + ".tmp", "w") as f:
                json.dump(state, f)
            os.rename(STATE_FILE + ".tmp", STATE_FILE)
            print(f"  State saved: cycle {self.cycle}, K=${self.kalshi_balance:.2f}, P=${self.poly_balance:.2f}")
        except Exception as e:
            print(f"  WARNING: Failed to save state: {e}")

    def _current_window_start(self) -> datetime:
        now = datetime.now(timezone.utc)
        return now.replace(minute=now.minute - now.minute % 15, second=0, microsecond=0)

    def _refresh_polymarket_prices(self):
        """Fetch executable Polymarket US BBO quotes for the current window.

        The window slug changes every 900s, so the map is rebuilt every cycle.
        NO quotes are the binary complement of the YES book.
        """
        if not self.poly_us:
            return
        try:
            self._poly_token_map = self._build_poly_token_map()
            prices = {}
            quotes = {}
            for slug in self._poly_token_map:
                md = self.poly_us.get_bbo(slug)
                if not md:
                    continue
                yes_bid = self._poly_px(md.get("bestBid"))
                yes_ask = self._poly_px(md.get("bestAsk"))
                if yes_bid is None or yes_ask is None:
                    continue
                if not (0.0 <= yes_bid < yes_ask <= 1.0):
                    continue
                quotes[slug] = {
                    "YES": {"bid": yes_bid, "ask": yes_ask},
                    "NO": {"bid": 1.0 - yes_ask, "ask": 1.0 - yes_bid},
                }
                prices[slug + ":YES"] = (yes_bid + yes_ask) / 2.0
                prices[slug + ":NO"] = 1.0 - (yes_bid + yes_ask) / 2.0
            self.poly_quotes = quotes
            if prices:
                self.poly_prices = prices
                print(f"  Polymarket US: {len(self._poly_token_map)} window markets, {len(quotes)} executable BBO quotes")
        except Exception as e:
            print(f"  WARNING: Polymarket US price refresh failed: {e}")

    @staticmethod
    def _poly_px(value: Any) -> Optional[float]:
        if value is None:
            return None
        raw = value.get("value") if isinstance(value, dict) else value
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    def _build_poly_token_map(self):
        """Map Polymarket US market slug -> {yes,no} pseudo-token ids for the current window.

        Discovery is by deterministic event slug: {asset}-updown-15m-{UTC window start}z
        on gateway.polymarket.us (Gamma listing returns future windows, not the active one).
        """
        if not self.poly_us:
            return {}
        win = self._current_window_start()
        token_map, questions, asset_map = {}, {}, {}
        for asset in ['BTC', 'ETH', 'SOL', 'XRP', 'DOGE', 'BNB', 'HYPE', 'ZEC', 'NEAR']:
            m = self.poly_us.get_updown_15m_market(asset, win)
            if not m or not m.get('slug'):
                continue
            slug = m['slug']
            token_map[slug] = {'yes': slug + ':YES', 'no': slug + ':NO'}
            questions[slug] = m.get('question', '')
            asset_map[asset] = slug
        self._poly_market_questions = questions
        self._poly_asset_map = asset_map
        return token_map

    def _find_matching_poly_market(self, k_ticker: str) -> Optional[str]:
        """Map a Kalshi ticker (KXBTC15M-...) to the Polymarket US slug for the same asset+window."""
        if not getattr(self, '_poly_asset_map', None):
            self._build_poly_token_map()
        for a in ['BTC', 'ETH', 'SOL', 'XRP', 'DOGE', 'BNB', 'HYPE', 'ZEC', 'NEAR']:
            if a in k_ticker:
                return self._poly_asset_map.get(a)
        return None

    def _get_live_poly_price(self, slug: str, side: str) -> Optional[float]:
        if not slug:
            return None
        key = f"{slug}:{side}"
        return self.poly_prices.get(key)

    def _get_live_poly_quote(self, slug: str, side: str, direction: str) -> Optional[float]:
        """Executable Poly US price: ask for buys, bid for sells."""
        quote = self.poly_quotes.get(slug, {}).get(side)
        if not quote:
            return None
        return quote["ask" if direction == "LONG" else "bid"]

    @staticmethod
    def _slipped_price(price: float, direction: str) -> float:
        slip = price * (SLIPPAGE_BPS / 10000.0)
        filled = price + slip if direction == "LONG" else price - slip
        return max(0.01, min(0.99, filled))

    @staticmethod
    def _mid(bid: Optional[float], ask: Optional[float]) -> Optional[float]:
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2.0

    def fetch_live_markets(self) -> Dict[str, Any]:
        try:
            # Query all crypto 15M series
            all_markets = []
            for series in ["KXBTC15M", "KXETH15M", "KXSOL15M", "KXXRP15M", "KXDOGE15M",
                          "KXBNB15M", "KXHYPE15M", "KXZEC15M", "KXNEAR15M"]:
                try:
                    markets = self.kalshi_client.get_markets(status="open", series_ticker=series, limit=100)
                    all_markets.extend(markets)
                except Exception as e:
                    print(f"  WARNING: Failed to fetch markets for {series}: {e}")
            return {m["ticker"]: m for m in all_markets if m.get("status") in ("open", "active")}
        except Exception as e:
            print(f"  WARNING: Failed to fetch live markets: {e}")
            return {}

    def refresh_all_orderbooks(self):
        live = self.fetch_live_markets()
        for ticker in live:
            try:
                ob = self.kalshi_client.get_orderbook(ticker, depth=10)
                # Kalshi orderbook returns YES bids and NO bids (both are bid arrays)
                # YES ask = 1 - NO bid (complement)
                # NO ask = 1 - YES bid (complement)
                yes_bids = [int(b[0]) for b in ob.get("yes", [])] if ob.get("yes") else []
                no_bids = [int(b[0]) for b in ob.get("no", [])] if ob.get("no") else []
                yes_asks = [100 - b for b in no_bids] if no_bids else []  # YES ask = 1 - NO bid
                no_asks = [100 - b for b in yes_bids] if yes_bids else []  # NO ask = 1 - YES bid
                book = OrderBook(ticker, yes_bids, no_bids, yes_asks, no_asks, time.time())
                self.orderbook_cache.update(book)
            except Exception as e:
                print(f"  WARNING: Orderbook refresh failed for {ticker}: {e}")

    def settle_expired_positions(self):
        now = datetime.now(timezone.utc)

        remaining_k = []
        for pos in self.kalshi_positions:
            if pos.expiry_time <= now:
                outcome = self._window_outcome(pos)
                pnl = pos.pnl_at_settle(outcome)
                self.kalshi_balance += pos.settlement_cash_delta(outcome)
                trade = {
                    'ticker': pos.ticker, 'venue': 'KALSHI', 'side': pos.side,
                    'entry': pos.entry_price, 'filled': pos.filled_price,
                    'contracts': pos.contracts, 'filled_ct': pos.filled_contracts,
                    'outcome': outcome, 'pnl': pnl, 'signal': pos.signal_type,
                    'direction': pos.direction, 'margin': pos.margin_reserved,
                    'fee': pos.fee_paid, 'entry_time': pos.entry_time.isoformat(),
                }
                self.closed_trades.append(trade)
                print(f"  SETTLED KALSHI {pos.ticker} {pos.side}: fill@{pos.filled_price:.4f} x{pos.filled_contracts} PnL=${pnl:.2f} (outcome={outcome})")
            else:
                remaining_k.append(pos)
        self.kalshi_positions = remaining_k

        remaining_p = []
        for pos in self.poly_positions:
            if pos.expiry_time <= now:
                outcome = self._window_outcome(pos)
                pnl = pos.pnl_at_settle(outcome)
                self.poly_balance += pos.settlement_cash_delta(outcome)
                trade = {
                    'ticker': pos.ticker, 'venue': 'POLYMARKET', 'side': pos.side,
                    'entry': pos.entry_price, 'filled': pos.filled_price,
                    'contracts': pos.contracts, 'filled_ct': pos.filled_contracts,
                    'outcome': outcome, 'pnl': pnl, 'signal': pos.signal_type,
                    'direction': pos.direction, 'margin': pos.margin_reserved,
                    'fee': pos.fee_paid, 'entry_time': pos.entry_time.isoformat(),
                }
                self.closed_trades.append(trade)
                print(f"  SETTLED POLY {pos.ticker} {pos.side}: fill@{pos.filled_price:.4f} x{pos.filled_contracts} PnL=${pnl:.2f} (outcome={outcome})")
            else:
                remaining_p.append(pos)
        self.poly_positions = remaining_p

    def _window_outcome(self, pos) -> int:
        """Simulate one window, with measured cross-venue disagreement risk."""
        if not hasattr(self, '_win_outcomes'):
            self._win_outcomes = {}
        if not hasattr(self, '_venue_outcomes'):
            self._venue_outcomes = {}
        key = pos.expiry_time.replace(
            minute=pos.expiry_time.minute - pos.expiry_time.minute % 15,
            second=0,
            microsecond=0,
        )
        if key not in self._win_outcomes:
            token_probability = pos.filled_price if pos.side == 'YES' else 1.0 - pos.filled_price
            self._win_outcomes[key] = 1 if random.random() < token_probability else 0
        venue_key = (key, pos.venue)
        if venue_key not in self._venue_outcomes:
            base = self._win_outcomes[key]
            has_other_venue = any(window == key for window, _ in self._venue_outcomes)
            self._venue_outcomes[venue_key] = (
                1 - base if has_other_venue and random.random() < CROSS_VENUE_BASIS_BUFFER else base
            )
        return self._venue_outcomes[venue_key]

    def execute_box_arbitrage(self, live_markets: Dict[str, Any]):
        if not hasattr(self, '_poly_token_map') or not self._poly_token_map:
            self._poly_token_map = self._build_poly_token_map()
        if not hasattr(self, '_poly_market_questions'):
            return

        for k_ticker in live_markets:
            poly_market = self._find_matching_poly_market(k_ticker)
            if not poly_market or poly_market not in self._poly_token_map:
                continue

            book = self.orderbook_cache.get(k_ticker)
            if not book:
                continue

            candidates = [
                ('YES', book.get_yes_ask(), 'NO', self._get_live_poly_quote(poly_market, 'NO', 'LONG')),
                ('NO', book.get_no_ask(), 'YES', self._get_live_poly_quote(poly_market, 'YES', 'LONG')),
            ]
            for k_side, k_price, p_side, p_price in candidates:
                if k_price is None or p_price is None:
                    continue
                k_fill = self._slipped_price(k_price, 'LONG')
                p_fill = self._slipped_price(p_price, 'LONG')
                total_cost = k_fill + p_fill
                total_cost += venue_taker_fee('KALSHI', 1, k_fill)
                total_cost += venue_taker_fee('POLYMARKET', 1, p_fill)
                total_cost += CROSS_VENUE_BASIS_BUFFER
                if total_cost < 1.0 - MIN_EDGE_AFTER_FEES:
                    self._place_trade(
                        k_ticker, k_side, k_price, poly_market, p_side, p_price,
                        'DIRECTIONAL_BOX', 'LONG', 'LONG',
                    )
                    break

    def execute_kl_signals(self, live_markets: Dict[str, Any]):
        if not hasattr(self, '_poly_token_map') or not self._poly_token_map:
            self._poly_token_map = self._build_poly_token_map()
        if not hasattr(self, '_poly_market_questions'):
            return

        for k_ticker in live_markets:
            poly_market = self._find_matching_poly_market(k_ticker)
            if not poly_market or poly_market not in self._poly_token_map:
                continue

            book = self.orderbook_cache.get(k_ticker)
            if not book:
                continue

            k_mids = {
                'YES': self._mid(book.get_yes_bid(), book.get_yes_ask()),
                'NO': self._mid(book.get_no_bid(), book.get_no_ask()),
            }
            p_mids = {
                'YES': self._get_live_poly_price(poly_market, 'YES'),
                'NO': self._get_live_poly_price(poly_market, 'NO'),
            }
            if any(v is None for v in (*k_mids.values(), *p_mids.values())):
                continue

            sym_kl, _, _ = symmetric_kl(k_mids['YES'], p_mids['YES'])
            if sym_kl >= KL_STRONG_THRESHOLD:
                strength = 'STRONG'
            elif sym_kl >= KL_ARB_THRESHOLD:
                strength = 'ARB_FLAGGED'
            else:
                continue

            k_quotes = {
                'YES': {'bid': book.get_yes_bid(), 'ask': book.get_yes_ask()},
                'NO': {'bid': book.get_no_bid(), 'ask': book.get_no_ask()},
            }
            p_quotes = {
                'YES': self.poly_quotes.get(poly_market, {}).get('YES'),
                'NO': self.poly_quotes.get(poly_market, {}).get('NO'),
            }

            candidates = []
            for side in ('YES', 'NO'):
                kq, pq = k_quotes[side], p_quotes[side]
                if not kq or not pq:
                    continue
                # One contract long on one venue and short on the other locks
                # the price difference only if both venues settle identically.
                candidates.extend([
                    (side, 'LONG', 'KALSHI', kq['ask'], 'SHORT', 'POLYMARKET', pq['bid']),
                    (side, 'SHORT', 'KALSHI', kq['bid'], 'LONG', 'POLYMARKET', pq['ask']),
                    (side, 'LONG', 'POLYMARKET', pq['ask'], 'SHORT', 'KALSHI', kq['bid']),
                    (side, 'SHORT', 'POLYMARKET', pq['bid'], 'LONG', 'KALSHI', kq['ask']),
                ])

            best = None
            for side, primary_dir, primary_venue, primary_price, hedge_dir, hedge_venue, hedge_price in candidates:
                if primary_price is None or hedge_price is None:
                    continue
                if primary_dir == 'SHORT' and not ALLOW_SHORTING:
                    continue
                if not (ENTRY_BAND_MIN <= primary_price <= ENTRY_BAND_MAX):
                    continue

                if primary_dir == 'LONG':
                    long_venue, long_price = primary_venue, primary_price
                    short_venue, short_price = hedge_venue, hedge_price
                else:
                    long_venue, long_price = hedge_venue, hedge_price
                    short_venue, short_price = primary_venue, primary_price

                long_fill = self._slipped_price(long_price, 'LONG')
                short_fill = self._slipped_price(short_price, 'SHORT')
                mismatch = k_mids[side] * (1 - p_mids[side]) + (1 - k_mids[side]) * p_mids[side]
                basis_buffer = max(CROSS_VENUE_BASIS_BUFFER, mismatch)
                net_edge = (
                    short_fill - long_fill
                    - venue_taker_fee(long_venue, 1, long_fill)
                    - venue_taker_fee(short_venue, 1, short_fill)
                    - basis_buffer
                )
                if net_edge <= MIN_EDGE_AFTER_FEES:
                    continue
                item = {
                    'edge': net_edge,
                    'primary_dir': primary_dir,
                    'primary_venue': primary_venue,
                    'primary_price': primary_price,
                    'hedge_dir': hedge_dir,
                    'hedge_venue': hedge_venue,
                    'hedge_price': hedge_price,
                    'long_venue': long_venue,
                    'long_price': long_price,
                    'short_venue': short_venue,
                    'short_price': short_price,
                    'side': side,
                }
                if best is None or item['edge'] > best['edge']:
                    best = item

            if not best:
                continue

            if best['long_venue'] == 'KALSHI':
                k_direction, k_price = 'LONG', best['long_price']
                p_direction, p_price = 'SHORT', best['short_price']
            else:
                k_direction, k_price = 'SHORT', best['short_price']
                p_direction, p_price = 'LONG', best['long_price']
            signal = f"DIRECTIONAL_KL_{strength}_{best['primary_dir']}"
            self._place_trade(
                k_ticker, best['side'], k_price, poly_market, best['side'], p_price,
                signal, k_direction, p_direction,
            )

    def _place_trade(
        self,
        k_ticker: str,
        k_side: str,
        k_price: float,
        p_market: str,
        p_side: str,
        p_price: float,
        signal: str,
        k_direction: str = 'LONG',
        p_direction: str = 'LONG',
    ):
        """Place both legs atomically with the same fill ratio.

        A short is fully collateralized: its $1 payout obligation reduces buying
        power at entry, and the margin is released only if the short token loses.
        """
        legs = [
            ('KALSHI', k_side, k_price, k_direction),
            ('POLYMARKET', p_side, p_price, p_direction),
        ]
        prepared = []
        size = MAX_POSITION
        for venue, side, price, direction in legs:
            if price is None or not (0 < price < 1) or direction not in ('LONG', 'SHORT'):
                return
            balance = self.kalshi_balance if venue == 'KALSHI' else self.poly_balance
            filled_price = self._slipped_price(price, direction)
            risk_per_contract = filled_price if direction == 'LONG' else 1.0 - filled_price
            venue_size = min(10, int(balance / risk_per_contract))
            size = min(size, venue_size)
            prepared.append((venue, side, price, direction, filled_price))
        if size <= 0:
            return

        if random.random() < PARTIAL_FILL_PROB:
            filled_frac = random.uniform(0.3, 0.9)
        else:
            filled_frac = 1.0

        while size > 0:
            filled_contracts = max(1, int(size * filled_frac))
            leg_costs = []
            fits = True
            for venue, side, price, direction, filled_price in prepared:
                fee = venue_taker_fee(venue, filled_contracts, filled_price)
                risk = filled_price if direction == 'LONG' else 1.0 - filled_price
                required = risk * filled_contracts + fee
                balance = self.kalshi_balance if venue == 'KALSHI' else self.poly_balance
                if required > balance:
                    fits = False
                    break
                leg_costs.append((fee, required))
            if fits:
                break
            size -= 1
        if size <= 0:
            return

        # Next quarter-hour boundary. Minute arithmetic alone yields minute=60
        # during minutes 45-59, so roll forward with timedelta instead.
        now = datetime.now(timezone.utc)
        expiry = now.replace(minute=0, second=0, microsecond=0) + timedelta(
            minutes=(now.minute // 15 + 1) * 15
        )
        for (venue, side, price, direction, filled_price), (fee, required) in zip(prepared, leg_costs):
            if venue == 'KALSHI':
                self.kalshi_balance -= required
            else:
                self.poly_balance -= required
            pos = Position(
                ticker=k_ticker if venue == 'KALSHI' else p_market,
                venue=venue,
                side=side,
                entry_price=price,
                filled_price=filled_price,
                contracts=size,
                filled_contracts=filled_contracts,
                expiry_time=expiry,
                signal_type=signal,
                direction=direction,
                fee_paid=fee,
                margin_reserved=filled_contracts if direction == 'SHORT' else 0.0,
            )
            if venue == 'KALSHI':
                self.kalshi_positions.append(pos)
            else:
                self.poly_positions.append(pos)
            print(
                f"  TRADE {venue} {k_ticker if venue == 'KALSHI' else p_market} "
                f"{direction} {side}: fill@{filled_price:.4f} x{filled_contracts} "
                f"fee=${fee:.4f} signal={signal}"
            )

    def log_status(self, msg: str):
        print(msg)

    def run(self):
        print(f"\n{'='*60}")
        print(f"REAL-TIME KL Divergence + Box Arbitrage Dry Run v3")
        print(f"Target: ${TARGET:,} on either platform")
        print(f"Starting: Kalshi=${self.kalshi_balance:.2f}, Poly=${self.poly_balance:.2f}")
        print(f"Reality tax: slippage={SLIPPAGE_BPS}bps, partial_fill={PARTIAL_FILL_PROB*100:.0f}%")
        print(f"Press Ctrl+C to stop\n")

        self.log_status(f"KL-REALTIME-V3 STARTED: K=${self.kalshi_balance:.2f} P=${self.poly_balance:.2f}, target=${TARGET:,}")

        self.refresh_all_orderbooks()

        while self.running:
            try:
                self.cycle += 1

                # Invalidate Kalshi market cache to get fresh data
                self.kalshi_client.invalidate_markets()
                live_markets = self.fetch_live_markets()
                print(f"\nFetched {len(live_markets)} live markets")

                self._refresh_polymarket_prices()
                if self.poly_prices:
                    print(f"  Polymarket prices: {len(self.poly_prices)} tokens")

                self.refresh_all_orderbooks()
                books = self.orderbook_cache.get_all()
                print(f"  Orderbooks: {len(books)} active")

                self.settle_expired_positions()

                if live_markets:
                    self.execute_box_arbitrage(live_markets)
                    self.execute_kl_signals(live_markets)
                else:
                    print("  No live markets (window rollover/API gap) - trades skipped")

                print(f"\n=== Cycle {self.cycle} | {datetime.now(timezone.utc).strftime('%H:%M:%S')} ===")
                print(f"  Kalshi: ${self.kalshi_balance:>10.2f}  (open: {len(self.kalshi_positions)}, unrealized: ${sum(p.pnl_at_settle(1 if p.side=='YES' else 0) for p in self.kalshi_positions):.2f})")
                print(f"  Poly:   ${self.poly_balance:>10.2f}  (open: {len(self.poly_positions)}, unrealized: ${sum(p.pnl_at_settle(1 if p.side=='YES' else 0) for p in self.poly_positions):.2f})")
                print(f"  Total:  ${self.kalshi_balance + self.poly_balance:>10.2f}  ({(self.kalshi_balance + self.poly_balance) / TARGET * 100:.1f}% to target)")
                print(f"  Closed trades: {len(self.closed_trades)}")

                self._save_state()

                if os.path.exists(KILLSWITCH_FILE):
                    print("  KILL SWITCH ACTIVATED - stopping")
                    self.running = False
                    break

                wait_seconds = 900 - (time.time() % 900)
                slept = 0
                while slept < wait_seconds and self.running:
                    chunk = min(15, wait_seconds - slept)
                    time.sleep(chunk)
                    slept += chunk
            except Exception as e:
                print(f"  ERROR in cycle {self.cycle}: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(5)

        print(f"\nKL-REALTIME-V3 STOPPED: K=${self.kalshi_balance:.2f}, P=${self.poly_balance:.2f}")


if __name__ == "__main__":
    bot = RealTimeDryRun()
    bot.run()