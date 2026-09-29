# Features

`features.py` turns raw Coinalyze series into one row per ticker per hour, with 24 features and a label.
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
   - funding, predicted funding and long %: mean across contracts weighted by open interest
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
| `oi_z_24h`, `oi_z_7d` | z-score of log OI index vs the previous 24 h / 7 days | OI above its normal level |
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
| `pred_funding_gap` | predicted − current funding | where funding is heading |
| `liq_long_4h_oi`, `liq_short_4h_oi` | long / short liquidations over 4 h, relative to OI | forced exits |
| `liq_imbalance_4h` | (short − long) / (short + long) liquidations over 4 h, 0 if none | squeeze direction |
| `ls_long_pct` | OI-weighted % of accounts long | crowd positioning |
| `ls_chg_24h` | change of long % over 24 h | positioning shift |

Missing values are left as `NaN` (for example, volume ratio before 8 days of history, or buy ratio
when volume is zero). Tree models handle them natively; linear models need imputation at training time.

## Output table

`features` is fully derived from the raw tables and `pumps`, so it is created by `features.py` rather than
`schema.sql`. Columns: `base_asset`, `ts`, `price` and `oi` (context, not features), the 24 features,
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
