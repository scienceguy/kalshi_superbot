# Polymarket US production status and market rules (T15-01)

Date of evidence: 2026-11-20. All commands below were run this session from the
`v15-task-list` branch (worktree `/home/ubuntu/kalshi_superbot-v15`, HEAD
`f47919f`). Every number below is the command output, not a memory of one.

## 1. Resolution of the "PREPROD" conflict (v15 section 2)

v15 recorded a CONFLICT: a config comment dated 2026-11-13 said crypto Up/Down
markets were still in PREPROD on api.polymarket.us and that no crypto markets
existed on the US retail API.

Verified on HEAD:

```
$ grep -rn 'POLYMARKET_ENABLED|PREPROD|preprod|api.polymarket.us|polymarket.us' --include='*.py' .
(no output)
```

There is no `POLYMARKET_ENABLED` flag and no PREPROD comment in the repo at HEAD
`f47919f`. Nothing to update in config.py; the flag does not exist and was NOT
re-created (a dead config flag is worse than none - see T15-04's one-definition
rule). The discovery of the truth came from the live API instead.

## 2. Live production markets (public gateway, no API key)

Base URLs (from docs.polymarket.us API introduction):
- Public read API: `https://gateway.polymarket.us`
- Authenticated trading API: `https://api.polymarket.us` (401 without PM keys)

Command (windows ending in the next 3 hours):

```
GET https://gateway.polymarket.us/v1/markets?categories=crypto&closed=false
    &endDateMin=2026-11-20T08:40:00Z&endDateMax=2026-11-20T11:40:00Z&limit=100
```

Result: 12 active Up/Down windows, 3 per asset (5m, 15m, 1h) for btc, eth, sol,
xrp. Horizon/tick/min-size/fee/tie data from the live market object of
`cpc-btc-updown-15m-2026-11-20-0815z`:

| field | value |
|---|---|
| slug | cpc-btc-updown-15m-2026-11-20-0815z |
| assetPriceTerms.marketType | ASSET_PRICE_MARKET_TYPE_UP_DOWN |
| assetPriceTerms.asset | ASSET_CLASS_CRYPTO / btc |
| assetPriceTerms.indexSymbol | BRTI |
| assetPriceTerms.horizon | 15m |
| windowStart / windowEnd | 2026-11-20T08:15:00Z / 08:30:00Z |
| priceToBeat | 61813.52 USD (stamped at window open) |
| orderPriceMinTickSize | 0.01 |
| minimumTradeQty | 0.01 (decimal quantities supported) |
| feeCoefficient | 0.095 |
| status | MARKET_STATUS_OPEN, active=true, closed=false |
| BBO | bestBid 0.83 / bestAsk 0.84, bidShares 110000 / askShares 56000 |

Other assets' 15m windows in the same window set:

| asset | slug | indexSymbol | tick | minQty | feeCoefficient | priceToBeat |
|---|---|---|---|---|---|---|
| btc | cpc-btc-updown-15m-2026-11-20-0830z | BRTI | 0.01 | 0.01 | 0.095 | 61908.58 |
| eth | cpceth-eth-updown-15m-2026-11-20-0830z | ETHUSD_RTI | 0.01 | 0.01 | 0.095 | 1754.51 |
| sol | cpcsol-sol-updown-15m-2026-11-20-0830z | SOLUSD_RTI | 0.01 | 0.01 | 0.095 | 72.6388 |
| xrp | cpcxrp-xrp-updown-15m-2026-11-20-0830z | XRPUSD_RTI | 0.01 | 0.01 | 0.095 | 0.96564 |

Slug pattern: `cpc<asset>-<asset>-updown-<horizon>-<YYYY-MM-DD>-<HHMM>z`
(BTC omits the leading asset tag: `cpc-btc-...`). Windows are created well
ahead (15m listed ~12h ahead per the Crypto FAQ) and trade around the clock.

CONCLUSION: the automated crypto Up/Down families are live in PRODUCTION on the
US retail API as of 2026-11-20. The stale "preprod" reading is retired.

## 3. Settlement reference - the finding that matters most (F-01)

The live market description (verbatim from the API object):

> "This market will settle to Up if the Bitcoin price at 3:30AM ET on Nov 20,
> 2026 is greater than or equal to the price at 3:15AM ET on Nov 20, 2026.
> Otherwise, this market will settle to Down. Outcome sourced from CF Benchmarks'
> Bitcoin Real-Time Index (BRTI)... The price at the beginning and at the end of
> the time range is determined the same way: in the last minute before each of
> those two times, 60 BRTI prices are collected. The official value for that
> time is the simple average of these prices, rounded to the nearest 2 decimal
> places."

Kalshi's rules text for the same window, from the live public markets endpoint
(`GET /trade-api/v2/markets?series_ticker=KXBTC15M&status=open&limit=2`,
market `KXBTC15M-26NOV200330-30`):

> "If the simple average of the sixty seconds of CF Benchmarks' BRTI before
> 3:30 AM EST on Nov 20, 2026 is at least the simple average of the sixty
> seconds of CF Benchmarks' BRTI before 3:15 AM EST on November 20, 2026, then
> the market resolves to Yes."

So for BTC/ETH/SOL/XRP 15-minute windows, Kalshi and Polymarket US resolve the
SAME statistic: the simple average of 60 one-second CF Benchmarks index values
before the window open, compared with the same statistic before the window
close, with ">=" (i.e. at-or-above) resolving the long/Up/Yes side. Differences
recorded for the equivalence table (docs/EVAL_PROTOCOL.md, T15-31):

1. Poly US rounds each official value to 2 decimals; Kalshi's rules text states
   no rounding. At BRTI ~61813.52 a 0.005 rounding is immaterial to a 15-minute
   move, but it is a real, documented difference.
2. The averaging window wording is "the 60 seconds before X" on both; Poly's FAQ
   spells the interval as [T-60s, T). Kalshi's wording matches but is not
   spelled to the same precision. Same statistic under either reading.
3. Timezone conventions: Kalshi ticker times are ET; Poly `windowStart/End` are
   UTC. 3:15-3:30 AM ET == 08:15-08:30 UTC (November, EST). Verified equal for
   the sample window above.
4. Overlap is BTC, ETH, SOL, XRP. Kalshi additionally lists DOGE, BNB, HYPE,
   ZEC, NEAR, ADA, BCH 15M (from GET /series: CF Benchmarks settlement source).

Therefore a Kalshi-vs-Poly-US pair for the same asset and window is the same
binary event. It is NOT riskless (two order books, two fee models, two fills),
but it is settlement-equivalent, which is the precondition for the cross-venue
box/cross-sectional arbitrage in `box_arbitrage.py`. The INTERNATIONAL
Polymarket (gamma-api.polymarket.com, Binance 1m candle per crossvenue.py:11) is
a different venue and is NOT equivalent; nothing in this repo trades it.

## 4. Fee schedule (F-15 resolved by live evidence, not assumption)

From the live market object: `feeCoefficient: 0.095`.

From https://docs.polymarket.us/fees (fetched 2026-11-20): effective
exchange-wide from 12:00 AM ET, Saturday, November 14, 2026.

- Taker fee = Theta x C x p x (1-p), p in 0.01..0.99.
- Taker tiers by trailing 30-day taker volume: Tier 1 below $10,000 -> 0.095;
  Tier 2 $10,000-<$50,000 -> 0.085; Tier 3 $50,000-<$250,000 -> 0.075;
  Tier 4 $250,000+ -> 0.0695.
- Maker rebate tiers by qualifying maker volume: Tier 1 below $10,000,000 -> no
  maker rebate (0%). So at our size a resting maker fill pays NO fee and earns
  NO rebate: maker fills are fee-neutral.
- New-account window (first fill on/after launch): days 1-7 taker coefficient
  0.0695 regardless of volume; days 8-30 the higher of first-7-day volume x
  30/7 and trailing 30-day volume.

The live `feeCoefficient: 0.095` on the market object equals Tier 1, so the
account-independent default used by any model must be Theta = 0.095. Our real
account tier cannot be confirmed from public data: there are no Polymarket US
credentials on this host (`~/.secrets/` holds bw/, kalshi_private.pem,
n8n_api_key, supabase_service_key, superbot_webhook_secret only). Confirming the
account tier needs an operator-supplied API key. Recorded as an operator action,
not a finding.

NOTE (F-15 correction): v15 said "scripts/fee_probe.py exists". It does not:

```
$ ls scripts/
(no fee_probe.py)
```

and no tier model exists in code either:

```
$ grep -rn '0.095|Theta|fee_tier|TIER' --include='*.py' .
(no output)
$ grep -rn '_taker_fee|taker_fee' polymarket_trader/*.py
(no output)
```

`polymarket_trader/fees.py` is the legacy generic model (2% protocol fee on
mainnet, 0% on Polygon) and is not the US schedule. The tier model is a build
item for T15-21; the tier VALUE is now known from live evidence.

## 5. Order constraints (T15-01 acceptance)

From https://docs.polymarket.us/api-reference/orders/create-order.md (OpenAPI
for POST /v1/orders, authenticated API):

- Order types: `type` = limit or market. Limit requires `price`; market uses
  `cashOrderQty` instead of share quantity.
- Time in force: `tif` (TimeInForce) plus `goodTillTime` for
  TIME_IN_FORCE_GOOD_TILL_DATE. From the app docs, limit orders are GTC, until
  12am PT, or Immediate-or-cancel (IOC).
- Post-only: `participateDontInitiate: true` = "order must rest on the book
  prior to matching (maker only). Order will be rejected if it would immediately
  match." This is the flag the resting/maker strategy needs.
- Quantities are decimal on markets whose `minimumTradeQty` < 1 (live: 0.01).
- `synchronousExecution` + `maxBlockTime` for blocking until filled/rejected.
- `preview-order` endpoint exists (the repo's paper engine already simulates
  "real orders.preview validation" per v15 section 2).
- Auth: Ed25519 request signing, headers X-PM-Access-Key / X-PM-Timestamp /
  X-PM-Signature. Endpoints under the authenticated base also include
  `synchronousExecution` and batch order creation (up to 20).

## 6. Gap found in the repo (new finding F-20)

```
$ grep -rn 'requests.|aiohttp|httpx|urlopen' polymarket_trader/*.py
(no output)
```

There is no Polymarket US REST client anywhere in the repo. `polymarket_trader/`
is a paper engine fed by simulated books; `legacy/polymarket_client.py` (now
`archive/`) was the INTERNATIONAL client. R3's reviewer list names
`polymarket_us_client.py`; it has to be built (small public-gateway reader
first) before any Poly US work other than paper can run. Log-only cross-sectional
scanning needs no credentials, so it is not blocked by this.