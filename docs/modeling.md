# Modeling

`train.py` compares three models of increasing complexity on the `features` table.

```bash
python train.py                          # all models → reports/train_report.md, reports/oof_predictions.csv
python train.py --models logreg lightgbm
```

## Models

| Model | What it is | Question it answers |
|---|---|---|
| `rule` | signal when `oi_z_7d > a` and `volume_z_7d > b`; `a`, `b` picked by grid search for best F1 on training tickers | How far does a trader's rule get without ML? |
| `logreg` | median imputation → standardisation → logistic regression, `C=0.1`, balanced class weights | Do the other features add anything beyond OI? |
| `lightgbm` | 300 shallow trees (depth 3, 7 leaves, 40+ samples per leaf), subsampling, balanced class weights | Are there interactions between features that a linear model misses? |

Each step has to beat the previous one to justify its complexity.

## Validation

**Leave-ticker-out.** For each of the N tickers, the model is trained on the other N−1 and scored on the
held-out one. Every prediction therefore comes from a model that never saw that coin, which is how the
screener will meet new listings.

**Nested threshold selection.** A probability becomes a signal above a threshold. Choosing that threshold on
the test ticker would leak information, so inside each outer fold a second leave-ticker-out over the training
tickers produces out-of-fold probabilities, and the threshold with the best F1 on those is applied to the
held-out ticker. The rule's thresholds are likewise chosen on training tickers only.

**Metrics.**

- Hour level: precision, recall and F1 of the positive class, and PR-AUC (threshold-free). Accuracy is not
  used: with 3.6% positives, always answering "no pump" is 96.4% accurate.
- Event level, closer to how a trader experiences the model:
  - pumps caught: intervals with at least one signal inside;
  - median delay: hours from labeled start to the first signal;
  - false starts triggered: false-start intervals with a signal;
  - false alarms per 30 days: signal episodes (consecutive signal hours) outside any labeled interval,
    per ticker per 30 days.

## First results (September 2026)

9 tickers, 11 862 hours, 19 pump intervals, 8 false starts (labels as of 2026-09-29).

| Model | Precision | Recall | F1 | PR-AUC | Pumps caught | False starts triggered | False alarms / 30 d |
|---|---|---|---|---|---|---|---|
| rule | 0.47 | 0.71 | 0.57 | — | 18/19 | 6/8 | 3.8 |
| logreg | 0.59 | 0.70 | 0.64 | 0.64 | 18/19 | 5/8 | 2.6 |
| lightgbm | 0.65 | 0.78 | 0.71 | 0.72 | 18/19 | 3/8 | 1.6 |

**Scope decision: ZEC.** ZEC was labeled with three 3–8 day moves that made up almost half of all pump hours.
With it, F1 fell to 0.56 (LightGBM) and 0.61 (logistic regression), also on the other tickers: the model was
learning a multi-day trend of a large coin mixed with leveraged pumps of small ones. ZEC was removed from the
training set as out of scope (the screener targets small-cap perpetuals), not to improve the numbers.

**Noise level.** Re-running on the same 8 tickers a few hours later, with the 60-day window shifted, moved F1
by 0.02–0.03. Differences of that size are not meaningful at this sample size.

**Label revision.** Error review showed that AKE 2026-08-28, labeled a false start from 22:00, had OI rising
steadily from 16:00 with price following. Under the labeling rules it was relabeled as a pump from 16:00.
Metrics rose slightly (LightGBM F1 0.72 → 0.73), but that gain is mechanical: an interval the model already
flagged became positive. It is not counted as a model improvement; model comparisons are only made on the
same labels.

**What this shows**

- **Both models clearly beat the rule.** Precision rises from 0.45 to about 0.65 at similar or higher recall,
  false alarms drop by roughly 2.5×, and fewer false starts trigger a signal. The features beyond OI and
  volume carry information.
- **LightGBM vs logistic regression is not settled.** LightGBM is ahead on pooled metrics, but per-ticker F1
  swings widely (for example MYX: 0.27 for LightGBM vs 0.67 for logistic regression). With 9 tickers and 19
  pumps, the gap is within noise.
- **One pump is missed by every model:** AKE, 2026-09-05 00:00–03:00, four hours, +48% price. It is the
  smallest pump in the set; the models give it at most 0.11 probability.

**Caveats**

- **Detection delay is measured from the labeled start**, and labels start where OI visibly jumps. A median
  delay of 0 hours means the first signal comes on that bar, not before the move. Early detection is
  still the open question: 13–14 of 18 pumps are caught within their first 3 hours by every model.
- **The top LightGBM feature is `ret_24h` (47% of gain).** A large part of what the model recognises is
  "price is already up over the last day". This helps classify hours inside a pump but is weaker as an
  entry signal. The next iterations should measure performance on the first hours of each pump separately.
- **Feature importance points at a short squeeze.** In logistic regression, the share of accounts long
  (`ls_long_pct`) has the largest coefficient and it is negative: the fewer longs, the more likely a pump.
- **Small sample.** Conclusions will change as more tickers are labeled.

## Fresh unlabeled hours are not trained on

Tickers in `tickers.txt` are treated as fully labeled: any hour outside a labeled interval is a "0". That breaks
for the most recent hours, which nobody has reviewed yet. When the model was retrained on 2026-09-29 12:22, QNT's
live Alert at 09:00 (probability 0.88) went into training as "not a pump", and the new model scored the same
hour 0.79, below its threshold: the model had been taught to un-see a signal that might be real.

`train.py` therefore drops **unlabeled hours from the last 48 hours** (`--fresh-hours`, 0 to disable). Hours
inside labeled intervals are kept, so a freshly labeled pump still counts. The model's `data_end` is the cutoff;
recent labeled intervals are recorded separately so the screener still marks them in-sample. After the fix QNT's
09:00 hour scores 0.87 against a threshold of 0.81 again. Before retraining, label fresh signals on training
tickers if they were pumps.

## Experiment: open interest in coins

Hypothesis: OI in coins (pure position inflow, without the price effect Coinalyze bakes into USD OI) makes better
features. Same data and labels, `python experiments/oi_coins.py` → `reports/experiment_oi_coins.md`:

| Model | OI in USD (current) | OI in coins | Both |
|---|---|---|---|
| rule | 0.57 | 0.49 | — |
| logreg | 0.63 | 0.63 | 0.65 |
| lightgbm | **0.68** | 0.62 | 0.65 |

(F1, leave-ticker-out.) **Rejected.** USD OI mixes position inflow with price, and for pump detection that mix is
the stronger signal. In a short squeeze, price rises while shorts close positions: OI in coins falls, OI in USD
rises. On AIN's pump, USD OI grew about fourfold while coins grew about a third. The model keeps the USD
features; the coin features stay in the table for charts and later experiments. Re-run on 10 tickers
(with TRUMP): LightGBM 0.67 in USD vs 0.59 in coins — same conclusion. Differences of ±0.02–0.03 are
within noise, so "both" is not an improvement either.

## Liquidations removed from the models (2026-10-07)

On the old labels (long pump intervals) liquidations mattered: without them LightGBM F1 fell 0.71 → 0.60, and short
liquidations over 4 hours took 18% of the gain. After relabeling to impulses (from the start to the OI collapse) their
share fell to 3–5%: short squeezes describe a pump in full swing, not its start. `experiments/no_liq.py` retrained both
models with and without the five `liq_*` features on the same data (21 tickers, 101 impulses):

| | 15m caught / false starts / false per 30 d | hourly caught | backtest, 216 new coins, profit factor (partial exit) |
|---|---|---|---|
| with liquidations | 83/100 · 13/22 · 10.7 | 81/100 | 0.83 (≥ 20 M $: 1.00) |
| without | 84/100 · 15/22 · 11.2 | 78/100 | 0.89 (≥ 20 M $: 1.07) |

No loss — within noise, if anything slightly better on new coins. The models now train without them by default
(`train.py --with-liq` restores the old set), so the 15-minute scoring no longer waits for ~200 Coinalyze liquidation
calls per cycle. Liquidations are still collected (Coinalyze after scoring, plus the own WebSocket recorder) and the
features stay in the `features` table: they may matter for other targets later (e.g. a short-side model).

## Predicted funding removed

`pred_funding_gap` (predicted − current funding) had the lowest weight in both models (0.14% of LightGBM gain,
logreg coefficient −0.02). Leave-ticker-out with and without it, same data:

| Model | F1 with | F1 without | Pumps caught with / without |
|---|---|---|---|
| logreg | 0.661 | 0.665 | 21/21 · 21/21 |
| lightgbm | 0.678 | 0.672 | 20/21 · 18/21 |

The difference is within noise. Both missed pumps are on TRUMP, where the probability peaked at 0.910 against a
threshold of 0.922: a threshold shift, not lost information. Removed from the features (23 now) and from
collection: it was one of six Coinalyze series, so every fetch and screener cycle is ~1/6 cheaper, and exchanges
keep no history of it. The `predicted_funding_rate` table and its old rows stay in the database.

## Experiment: aggregated volume delta

Volume delta is taker buys minus taker sells. Its share of volume is already a feature (`buy_ratio_*`), so the
test adds the forms that were missing: perp cumulative delta over 24 h, delta relative to open interest, delta
against its weekly norm; and spot delta from Binance spot (spot buy share, spot CVD, spot-to-perp volume). Binance
spot exists for 6 of 10 tickers (Coinalyze lists spot only on large exchanges, for AKE, MYX and SOON only Kraken).
`python experiments/volume_delta.py` → `reports/experiment_volume_delta.md`:

| Model | base (23) | + perp delta | + spot delta | + both |
|---|---|---|---|---|
| logreg F1 | 0.67 | 0.66 | 0.63 | 0.62 |
| lightgbm F1 | 0.67 | 0.66 | 0.69 | 0.68 |
| lightgbm PR-AUC | 0.70 | 0.72 | 0.71 | 0.71 |
| lightgbm false alarms / 30 d | 1.7 | 1.7 | 1.3 | 1.7 |

**Not adopted yet.** The changes are within run-to-run noise, the new features take ~6% of LightGBM gain in
total, and logistic regression gets worse (spot features are empty for 4 tickers and imputed). The one hint is
fewer false alarms for LightGBM with spot delta. To be re-tested with confidence intervals and more labeled tickers.

## Experiment: exchange APIs instead of Coinalyze

Question: can the features come straight from Binance and Bybit public APIs (much higher rate limits, no key)
without changing the model? `python experiments/exchange_direct.py` → `reports/experiment_exchange_direct.md`.
Comparison window: the last ~22 days (Binance keeps open interest and long/short history for 30 days only,
features need 7 days of warm-up), 10 tickers, 10 labeled pumps.

**Raw series are identical.** Contract by contract, price, volume, buy volume, open interest, funding and
long/short from the exchange match Coinalyze exactly (median relative difference 0, correlation of hourly
changes 1.000). Coinalyze relays exchange data; it adds convenience, not information. Time alignment matters:
an exchange OI snapshot at T+1h is the close of Coinalyze's bar T, while long/short and funding paid at T
belong to bar T. The script checks this by correlating with ±1 h shifts.

**Features differ only through exchange coverage.** With Binance + Bybit alone (no OKX, Aster, Hyperliquid),
OI and price features keep a rank correlation of 0.98–1.00 with production values; funding and buy ratio drop to
0.89–0.96 because Bybit candles carry no buy volume (approximated from Binance's buy share).

F1 of models trained on Coinalyze (23 features, after predicted funding was dropped), scored on the held-out
ticker's features from each source:

| Model | coinalyze | Coinalyze, Binance+Bybit only | exchanges + Coinalyze liquidations | exchanges only |
|---|---|---|---|---|
| rule | 0.57 | 0.56 | 0.56 | 0.56 |
| logreg | 0.70 | 0.70 | 0.70 | 0.70 |
| lightgbm | 0.71 | 0.71 | 0.70 | **0.60** |

**Conclusion.** Switching the source is safe for price, volume, OI, funding and long/short. Exchanges keep no
history of liquidations, so they must be collected by the project itself (websocket) or dropped. Without them the
current LightGBM breaks (recall 0.75 → 0.46): liquidation features are among its strongest (~18% of gain), and
all-zero liquidations look like a calm market. A feature that is absent or different at inference is
training-serving skew. Retrained without liquidations, exchanges-only data is as good as Coinalyze in the same
window (LightGBM 0.67 vs 0.63, logreg 0.74 vs 0.76; few pumps, so these are ties).

A first run blamed predicted funding for the LightGBM drop. Re-running after predicted funding was removed
(below) showed the drop remains, so liquidations were the cause.

Side findings for the data layer:
- Bybit serves open interest and long/short at 5-minute and hourly resolution for at least a year back
  (Coinalyze: ~63 days hourly, ~1 week at 5 min). Binance: 30 days. Candles and funding: years on both.
- One run over 10 tickers is 198 public requests in ~2 minutes, against ~8 minutes of Coinalyze quota.

## Experiment: 15-minute and 5-minute bars

Question: does re-scoring more often catch a pump earlier? `python experiments/timeframe.py [--tf 5m]` →
`reports/experiment_timeframe_15m.md`, `reports/experiment_timeframe_5m.md`. Data: Binance + Bybit APIs
(Bybit keeps 5-minute open interest for at least a year), 10 tickers, 2026-08-07 – 09-29, 21 labeled pumps.
Both sides get the same 20 features (no liquidations) with the same wall-clock windows
(`features.compute_features(bars_per_hour=…)`), and the same labels: a 15m bar inside a labeled pump is 1.

| Model · bars | Pumps caught | Median delay, min | Move already gone at signal | Upside left | False alarms / 30 d |
|---|---|---|---|---|---|
| logreg · 1h | 20/21 | 60 | 22% | 50% | 3.1 |
| logreg · 15m | 20/21 | 45 | 13% | 57% | 5.0 |
| logreg · 5m | 21/21 | 35 | 9% | 57% | 6.2 |
| lightgbm · 1h | 20/21 | 120 | 27–30% | 42% | 2.1–2.9 |
| lightgbm · 15m | 19/21 | 45 | 12% | 69% | 2.3 |
| lightgbm · 5m | 19/21 | 45 | 15% | 70% | 2.1 |

Delay is measured from the labeled start to the close of the first signal bar, when a trader can act. A 1h model
cannot be faster than 60 minutes by construction. Bar-level F1 is about the same on all timeframes (0.60–0.67).

**Conclusion.** Smaller bars catch the same pumps 15–75 minutes earlier and roughly halve the part of the move
already gone at the signal (LightGBM: 27–30% → 12–15%). The price is more false alarms for logistic regression.
LightGBM keeps its false-alarm rate. 5m gives little over 15m: the pump start is labeled to the hour, so finer
bars cannot earn more credit, and 5m is noisier. **15m is the sweet spot.** Caveats: 21 pumps, so medians
move by a bar or two between runs (the two 1h reruns differ slightly), and there are no fees or slippage.
Hour-level labels are a ceiling: labeling starts to the 15 minutes would let finer models show more.

## Experiment: 15-minute model on impulse labels

After relabeling on 5m (2026-09-30) a pump is the **impulse phase**: median length 4 hours instead of 13. The
hourly model is too slow for that (F1 0.49, first signal ~95 minutes after the start). `python
experiments/m15_model.py` → `reports/experiment_m15_model.md`: 14 tickers, 49 pumps, 15m bars from the database
(exchange candles/OI + Coinalyze 15m liquidations, known on 99% of bars), leave-ticker-out, LightGBM:

| Variant | Caught | Delay, min | Move gone | False starts hit | False alarms / 30 d | PR-AUC |
|---|---|---|---|---|---|---|
| 1h, current features | 45/49 | 95 | 42% | 4/19 | 4.1 | 0.45 |
| 15m base (no liquidations) | 46/49 | 30 | 21% | 12/19 | 6.3 | 0.44 |
| 15m + liquidations | 44/49 | 27.5 | 22% | 12/19 | 5.8 | 0.46 |
| 15m + short features | 48/49 | 27.5 | 21% | 12/19 | 6.3 | 0.45 |
| 15m + liquidations + short | 47/49 | 25 | 22% | 11/19 | 5.3 | 0.47 |

On hourly bars 15m seems to cut the delay by ~70 minutes, but that overstates it (see the next section: measured on a
common 15m grid the gap is ~15 minutes). Liquidations and short
features help only a little (false alarms 6.3 → 5.3); the model still leans on day-scale features (`oi_chg_24h`
~49% of gain, `ret_24h` ~12%). The cost of speed is false starts: 11–12 of 19 trigger a 15m signal against 4 on 1h.
Detecting earlier means deciding with less information. Logistic regression is clearly worse on 15m.

## Experiment: two-tier signal (15m candidate, hourly confirmation)

`python experiments/two_tier.py` → `reports/experiment_two_tier.md`. Out-of-fold LightGBM scores of both models on
a **common 15m grid**: each 15m bar gets the hourly score of the last hourly bar closed by its close.

**Correction.** Earlier comparisons measured the hourly model's delay on hourly bars and ignored a signal from the
hour *before* a mid-hour start, although a trader sees it during the pump. On the common grid the hourly model's
median delay is 45 minutes, not 95, and 15m is ahead by ~15 minutes, not ~70.

| Rule | Caught | Delay, min | Move gone | False starts hit | False alarms / 30 d |
|---|---|---|---|---|---|
| 1h only | 44/49 | 45 | 29% | 8/19 | 4.3 |
| 15m only | 47/49 | 30 | 22% | 12/19 | 5.6 |
| 15m and 1h ≥ 0.5 × threshold | 47/49 | 45 | 26% | 8/19 | 4.5 |
| **15m and 1h ≥ 0.75 × threshold** | **47/49** | 50 | 26% | **8/19** | **3.8** |
| 15m and 1h ≥ threshold within 4 h | 44/49 | 55 | 32% | 7/19 | 2.8 |

The combination catches as many pumps as 15m, hits as few false starts as 1h and has the fewest false alarms among
rules that catch 47. **Adopted design:** 15m alone = *Candidate* (earliest, noisy: the trader filters it in the order
book), 15m confirmed by the hourly model at ≥ 0.75 of its threshold = *Alert*. With 49 pumps the exact thresholds are
provisional; the forward test will tune them.

## Experiment: stop measured from the model's entry

`python experiments/stop_from_entry.py` → `reports/experiment_stop.md`. The 15m stop multiplier was the 75th
percentile of drawdowns from the **labeled start** of pumps, in ATR at the start (k ≈ 1.16). But the 15m tier
enters on the first Alert, a median 45 minutes later, when the move is ~9% along and ATR is several times higher.
Measured from the model's actual entries (out-of-fold, 46 pumps), the drawdown before the peak is: median 0.55 ATR,
75th percentile 1.44, 90th 2.51. Trades on all out-of-fold Alerts (no fees):

| Stop, ATR | Trades | Win | Stopped out | Avg trade |
|---|---|---|---|---|
| 1.16 (old) | 207 | 32% | 62% | +3.7% |
| 1.44 | 182 | 40% | 52% | +4.7% |
| 2.0 | 166 | 44% | 38% | +5.2% |
| **2.5 (90th pct)** | 160 | **45%** | **30%** | **+5.3%** |
| 4.0 | 153 | 48% | 10% | +5.6% |

**Adopted:** k = 90th percentile of drawdown from the model's entry (`risk.entry_stop`, computed by `train.py` on
out-of-fold signals), ≈ 2.5 ATR(15m). The old start-based value is kept in the model file as `k_from_start`.
Typical trade profile: many small losses, few large wins (median trade still slightly negative, trades inside
labeled pumps +12–16% on average).

## Next steps

- More labeled tickers, which is the main lever.
- Metrics for the early phase of a pump (first N hours) and a model variant without trailing-return features.
- Per-ticker error review: `python label.py` shows out-of-fold predictions and errors (see [labeling.md](labeling.md#reviewing-model-errors)).
- Save the chosen model for the screener.
