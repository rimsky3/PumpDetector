# Data

## Source

All market data comes from the [Coinalyze API](https://api.coinalyze.net/v1/doc/).

- Rate limit: 40 calls per minute; **each symbol in a request counts as one call**.
  `fetch.py` tracks this with a sliding window and honours `Retry-After` on HTTP 429.
- Up to 20 symbols per request.
- Granularity used in v1: 1 hour (`1hour`). All timestamps are UTC and mark the **bar open**.

## Markets

For each base asset in `tickers.txt`, `fetch.py` finds perpetual contracts quoted in USDT, USD or USDC
on Binance, Bybit, OKX, Aster, Hyperliquid and MEXC. Each exchange contract is one row in `markets`,
with a Coinalyze symbol such as `AKEUSDT_PERP.A` (Binance) or `AKEUSDT.6` (Bybit).

Some exchanges list a coin under another base name (`1000PEPE`, `kPEPE`). If a ticker is not found,
check the naming on the exchange.

## Series

| Series | Endpoint | Table | Columns | Units |
|---|---|---|---|---|
| Open interest | `open-interest-history` | `open_interest` | o, h, l, c | USD (`convert_to_usd=true`) |
| OHLCV | `ohlcv-history` | `ohlcv` | o, h, l, c, v, bv, tx, btx | price in quote; volume see below |
| Funding rate | `funding-rate-history` | `funding_rate` | o, h, l, c | rate per funding period |
| Predicted funding | `predicted-funding-rate-history` | `predicted_funding_rate` | o, h, l, c | rate; **not collected since 2026-09-29** ([why](modeling.md#predicted-funding-removed)) |
| Liquidations | `liquidation-history` | `liquidations` | long_liq, short_liq | USD (`convert_to_usd=true`) |
| Long/short ratio | `long-short-ratio-history` | `long_short_ratio` | ratio, long_pct, short_pct | ratio, % of accounts |

`bv` is buy (taker) volume, `tx` is number of trades, `btx` is number of buy trades.

**Volume units differ by exchange.** `markets.volume_denominated` says whether OHLCV volume is in the
base asset (coins) or the quote asset (USD). It must be converted to one unit before aggregating across
exchanges, for example `v * close` for base-denominated markets.

## Storage

- Raw tables are keyed by `(symbol, interval, ts)` and written with upserts, so re-running a
  collection updates rows instead of duplicating them.
- `markets` is the reference table; every raw row references `markets.symbol`.
- Labels live in `pumps` (see [labeling.md](labeling.md)).
- `alt_index` (since 2026-10-08) — the ALT-EW 5m index (equal-weighted mean log return of ~250 liquid Binance alts),
  built by `alt_index.py` from the data.binance.vision archive and REST; input of the 15m model's own-move features.

## History depth

Coinalyze keeps a rolling window of intraday history. Observed in September 2026:

| Series | Hourly history available |
|---|---|
| Open interest | ~63 days |
| All other series | ~90 days |

The project uses **60 days** (the `fetch.py` default) so that every series covers the same period.
Older data disappears from the provider, so regular collection is the only way to build a longer history.

### Exchange APIs directly

Binance and Bybit publish the same series without a key and with far higher limits (checked September 2026,
see [modeling.md](modeling.md#experiment-exchange-apis-instead-of-coinalyze)):

| Series | Binance | Bybit |
|---|---|---|
| Candles (incl. buy volume on Binance) | years, 1m and up | years, 1m and up; no buy volume |
| Open interest, long/short | API: 30 days (5m, 1h); **archive** `data.binance.vision/data/futures/um/daily/metrics/` — OI every 5 min, long/short account ratio (all and top traders), taker buy/sell ratio, since at least Dec 2021 | at least a year (5m, 1h) |
| Funding (paid rates) | years | years |
| Liquidations, predicted funding | no history (websocket / current value only) | no history |

**OKX** (collected since 2026-09-30): candles and funding for months, taker buy/sell volume for 90 days, open interest
and long/short for only ~14 days at 15m/1h (3 days at 5m). All OKX series are stamped with the bar's *open* time (OI
is the value at the bar's close), while Binance and Bybit give OI as a snapshot at time T; `exchanges.py` aligns them.
Binance and Bybit kline endpoints also return the current, still-forming candle; `exchanges.closed_only` drops it, so
only closed bars are ever stored (a partial bar would otherwise sit in the database as if final until the next fetch).
OKX is stored but not yet used by the 15m model or the 15m/5m charts (`features.INTRADAY_EXCHANGES`): the model was
trained on Binance + Bybit sums, and with only 14 days of OKX OI an older chart would show a false OI step. It will be
switched on after enough history has accumulated and an experiment shows it helps.

`python fetch_exchange.py` stores 15m and 5m bars from these endpoints in the same raw tables (`interval` =
`15min` / `5min`, same market symbols as Coinalyze), for labeling and a future 15m model. Everything that reads
raw tables filters by `interval`; hourly data stays Coinalyze.

**Intraday liquidations** do exist in Coinalyze (`liquidation-history` with `15min` / `5min`), but it keeps a
limited number of non-empty points per symbol (~2,000–2,900): for small coins that is 1–2 months, for active ones
3–5 weeks. Bars without liquidations are simply absent, so "no row" means zero only after the provider's first
point. `fetch.collect_liquidations` stores that first point per market in `liquidation_coverage`; `features.aggregate`
treats earlier bars as unknown (NaN), not zero. Regular collection extends the history forward.

The values match Coinalyze exactly once aligned in time: an exchange OI snapshot at T+1h is the close of
bar T. Binance's `/futures/data/*` endpoints return the **latest** 500 points when the requested range is
longer, so page through them in windows of 500 hours.

Exchanges give open interest as snapshots, not candles. `fetch_exchange.py` builds a candle from them:
open = snapshot at the bar's start, close = at its end, high/low = the larger and smaller of the two (intrabar
extremes are unknown). As in Coinalyze, the whole candle is in USD at the bar's close, so dividing by the close
gives exact coins.

## Coverage and known gaps

Coverage differs by exchange and series. These are provider limitations, not collection errors:

- **Aster:** no open interest and no liquidations; OHLCV can have missing hours when there were no trades.
- **OKX:** no predicted funding and no long/short ratio.
- **Hyperliquid, MEXC:** not found for the current tickers.
- **Late listings:** a market listed during the window starts mid-way (for example AKE on OKX from
  2026-09-16). Summing OI across exchanges then produces a **step that is not a pump**; features must
  account for it.
- **Liquidations:** hours without liquidations are absent from the table. Treat them as 0, not as missing.

## Aggregation across exchanges

Coinalyze does not aggregate across exchanges; the project does it itself:

- Open interest and liquidations: sum in USD across exchanges.
- Volume: convert to USD per market, then sum.
- Price: a single reference exchange (Binance where available), since prices differ only marginally.
- Funding and long/short ratio: to be decided in `features.py` (for example OI-weighted mean).
