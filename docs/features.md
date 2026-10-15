# Features

`features.py` turns raw Coinalyze series into one row per ticker per hour, with 23 features and a label.
The same module is used for training, inference and the screener, so the feature logic exists in one place.

```bash
python features.py            # rebuild the features table for all tickers
python features.py AKE QNT    # recompute only these tickers
```

Run it after `fetch.py` (new data) or after changing labels (new targets). After changing the feature set
itself, run it without arguments so the table is recreated with the new columns.

## Pipeline

1. **Load** raw tables for the requested tickers.
2. **Aggregate** all contracts of a ticker into one hourly series on a regular hourly grid:
   - open interest, volume, buy volume and liquidations: sum in USD across contracts;
   - price, high, low: one reference contract (Binance USDT if available);
   - funding and long %: mean across contracts weighted by open interest
     (plain mean when no OI is available, as on Aster).
3. **Compute features** relative to the ticker's own history.
4. **Label** each hour: `1` if it falls inside a `pump` interval, otherwise `0`. `pattern_type` keeps the
   interval type, so false starts can be analysed separately.
5. **Filter**: keep only hours with a full 7-day history for the baseline windows, and only closed bars
   (the current, still-forming hour is dropped).

## Two rules

**No look-ahead.** A feature at hour `t` uses only data up to and including `t`. Rolling windows look
backwards, and z-scores compare the current value with the *previous* window. This is verified by
recomputing features on data truncated at time `T` and checking that every value up to `T` is unchanged.

**Listings are not pumps.** When a coin lists on a new exchange, summed OI jumps without any new
positions being opened. The OI features are therefore built on a *chain-linked OI index*: the hourly
change is computed only over contracts that had OI both in the previous hour and now, and these changes
are accumulated. The raw OI sum is kept for reference but not used for change features.

## Windows

| Constant | Value | Meaning |
|---|---|---|
| `BASE_WINDOW` | 168 h (7 days) | the "normal level" a move is measured against |
| `SHORT_WINDOW` | 24 h | the last day |

## Feature list

| Feature | Definition | Why |
|---|---|---|
| `oi_chg_1h`, `_4h`, `_24h` | log change of the OI index over 1 / 4 / 24 h | leverage building up |
| `oi_z_24h`, `oi_z_7d` | z-score of log OI index vs the previous 24 h / 7 days; a gap in OI of up to 4 hours is filled with the last value (`OI_GAP_HOURS`, since 2026-10-15) — before that a single missing bar made `oi_z_7d` undefined for a week, and the screener skips rows without it | OI above its normal level |
| `ret_1h`, `_4h`, `_24h` | log return of price | price move |
| `volatility_24h` | std of hourly log returns over 24 h | regime change |
| `range_1h` | (high − low) / close | intrabar violence |
| `oi_minus_ret_24h` | `oi_chg_24h − ret_24h` | OI growing faster than price: positions opened ahead of the move |
| `volume_z_7d` | z-score of log volume vs the previous 7 days | unusual activity |
| `volume_ratio_24h` | 24 h volume / average daily volume of the 7 days before | sustained activity, not one spike |
| `buy_ratio_1h`, `_4h` | taker buy volume / total volume | aggressive buying |
| `funding` | OI-weighted funding rate | who pays whom |
| `funding_z_7d` | z-score of funding vs the previous 7 days | funding unusual for this coin |
| `funding_chg_24h` | change of funding over 24 h | direction of crowding |
| `liq_long_4h_oi`, `liq_short_4h_oi` | long / short liquidations over 4 h, relative to OI | forced exits |
| `liq_imbalance_4h` | (short − long) / (short + long) liquidations over 4 h, 0 if none | squeeze direction |
| `ls_long_pct` | OI-weighted % of accounts long | crowd positioning |
| `ls_chg_24h` | change of long % over 24 h | positioning shift |

Missing values are left as `NaN` (for example, volume ratio before 8 days of history, or buy ratio
when volume is zero). Tree models handle them natively; linear models need imputation at training time.

## 15m model: own-move features (since 2026-10-08)

The 15m model adds 12 features that measure the coin's move **relative to the altcoin market** (`features.MOVE_FEATURES`,
`move_features()`; detailed explanations in Ukrainian — `notes/movers.md`). The market is **ALT-EW**: the equal-weighted
mean 5-minute log return of ~250 liquid Binance altcoins (average daily turnover $5M – $1B, no BTC / ETH / stablecoins),
table `alt_index`, built by `alt_index.py` (history from the data.binance.vision archive, live tail from REST every 5 minutes).
The index is stored at 5 minutes so that any bar grid can be rebuilt from it — regular 15m bars, the screener's rolling
15m bars shifted by :05 / :10, or hours (`alt_index_bars`: sum of the 5m returns inside the bar).

| Group | Feature | Definition |
|---|---|---|
| Own move | `beta`, `r2` | OLS beta and R² of the coin's 15m log returns on the index over 7 days, window ending one day before the bar |
| | `idio_1h`, `idio_4h`, `idio_24h` | return − β × index return over 1 / 4 / 24 h: the part of the move that is not the market |
| | `idio_z_4h` | `idio_4h` in standard deviations of the regression residual (σ × √bars) |
| Amihud | `illiq_x` | log of (mean \|r\| / dollar volume over 24 h) ÷ its median over the previous 7 days: thin vs normal book |
| Volatility | `park_24h`, `gk_24h` | Parkinson / Garman–Klass volatility per bar over 24 h (uses high / low / open) |
| | `park_x` | `park_24h` ÷ its median over the previous 7 days |
| | `ret_4h_vol`, `ret_24h_vol` | return over 4 / 24 h ÷ (Parkinson volatility × √bars): the move in units of the coin's own noise |

No look-ahead: every window ends at the bar; beta is also lagged one day so a spike does not teach the beta about itself.
`experiments/idio_features.py` (6 months, survivorship-free monthly coin lists, 405 coins): profit factor 0.89 → 0.93,
better than the baseline in 5 of 6 months. The production version differs in one window: `illiq_x` is normalised on the
7-day median instead of 30 days, so the screener needs only ~15 days of history for the latest bar.
Bars before the index starts get `NaN` in these features. `train.py --no-move` trains the 15m model without them.

## Output table

`features` is fully derived from the raw tables and `pumps`, so it is created by `features.py` rather than
`schema.sql`. Columns: `base_asset`, `ts`, `price` and `oi` (context, not features), the 23 features,
`label`, `pattern_type`. Primary key `(base_asset, ts)`.

## First look at the data (September 2026)

Median feature values by group, 8 tickers, 10 510 hours:

| Feature | normal | false start | pump |
|---|---|---|---|
| `oi_z_7d` | −0.30 | 3.80 | 3.43 |
| `oi_chg_24h` | −0.01 | 0.14 | 0.68 |
| `ret_24h` | 0.00 | 0.09 | 0.40 |
| `volume_z_7d` | −0.21 | 2.37 | 2.40 |
| `liq_imbalance_4h` | 0.00 | 0.05 | 0.42 |
| `ls_long_pct` | 61.8 | 64.5 | 44.8 |

Observations to test with the model, not conclusions:

- **False starts look like pumps on OI and volume z-scores.** OI deviation alone will not separate them,
  which is exactly why the simple OI rule is the baseline to beat.
- **During pumps most accounts are short** (`ls_long_pct` ≈ 45% vs ≈ 62% normally) and liquidations are
  dominated by shorts. This is consistent with a short squeeze feeding the pump.
- Medians are taken over whole intervals, including late stages. The hard part is recognising the
  *early* hours of a pump, and that is what validation must measure.

## Impossible open-interest jumps

Positions are opened and closed only by trades, so open interest cannot change by more than the volume traded in
the bar. `features.impossible_oi` marks contract bars where |ΔOI| (in coins) is above `OI_JUMP_X` = 3 × the bar's
volume and above 20% of OI; `aggregate` treats them as missing (since 2026-10-15). On AGT on 2026-10-15 Binance's own
`openInterestHist` reported OI ×3.4 for 1.5 hours with a flat price and ~2M coins of volume per bar; without the filter
the chained OI index keeps the jump in and `oi_z_7d` moves by up to 11 for days. On real data (4.7M 15m bars, 1.9M 5m,
0.3M 1h, all exchanges) the largest legitimate ratio is ~2.3 (Bybit snapshots OI at a slightly different moment than
the candle close); in a 10-ticker check the filter changed one hourly bar (NIGHT, Bybit) and nothing else.

## Open interest in coins

Coinalyze converts open interest to USD with the **close price of the hour**, applied to all four points of the
hourly OI candle. USD open interest therefore jumps between hours whenever the price moves, even if no positions
changed: on AIN at the start of the 2026-09-14 pump, OI in contracts moved 0.00% between 04:00 and 05:00, while
OI in USD jumped +45% because the price did.

`features.aggregate` divides each contract's OI by that contract's own hourly close, giving open interest in
coins: the pure number of open positions. Gaps between hours drop from 0.5–2.6% to about 0.01%, and coins can be
summed across exchanges. The same chained index as for USD protects against listing jumps.

Coin-based features (`oi_coin_chg_1h/4h/24h`, `oi_coin_z_24h`, `oi_coin_z_7d`) are stored in the `features` table
but **not used by the model**: they performed worse (see [modeling.md](modeling.md#experiment-open-interest-in-coins)).
The charts in the labeling tool and the dashboard show OI in coins by default.
