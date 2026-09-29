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
features; the coin features stay in the table for charts and later experiments. Differences of ±0.02–0.03 are
within noise, so "both" is not an improvement either.

## Next steps

- More labeled tickers, which is the main lever.
- Metrics for the early phase of a pump (first N hours) and a model variant without trailing-return features.
- Per-ticker error review: `python label.py` shows out-of-fold predictions and errors (see [labeling.md](labeling.md#reviewing-model-errors)).
- Save the chosen model for the screener.
