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

8 tickers, 10 510 hours, 17 pump intervals, 9 false starts in the evaluated window.

| Model | Precision | Recall | F1 | PR-AUC | Pumps caught | False starts triggered | False alarms / 30 d |
|---|---|---|---|---|---|---|---|
| rule | 0.45 | 0.71 | 0.55 | — | 16/17 | 7/9 | 4.0 |
| logreg | 0.66 | 0.74 | 0.70 | 0.67 | 16/17 | 5/9 | 1.6 |
| lightgbm | 0.65 | 0.80 | 0.72 | 0.73 | 16/17 | 5/9 | 1.4 |

**What this shows**

- **Both models clearly beat the rule.** Precision rises from 0.45 to about 0.65 at similar or higher recall,
  false alarms drop by roughly 2.5×, and fewer false starts trigger a signal. The features beyond OI and
  volume carry information.
- **LightGBM vs logistic regression is not settled.** LightGBM is ahead on pooled metrics, but per-ticker F1
  swings widely (for example MYX: 0.27 for LightGBM vs 0.67 for logistic regression). With 8 tickers and 17
  pumps, the gap is within noise.
- **One pump is missed by every model:** AKE, 2026-09-05 00:00–03:00, four hours, +48% price. It is the
  smallest pump in the set; the models give it at most 0.11 probability.

**Caveats**

- **Detection delay is measured from the labeled start**, and labels start where OI visibly jumps. A median
  delay of 0 hours means the first signal comes on that bar, not before the move. Early detection is
  still the open question: 13–14 of 17 pumps are caught within their first 3 hours by every model.
- **The top LightGBM feature is `ret_24h` (47% of gain).** A large part of what the model recognises is
  "price is already up over the last day". This helps classify hours inside a pump but is weaker as an
  entry signal. The next iterations should measure performance on the first hours of each pump separately.
- **Feature importance points at a short squeeze.** In logistic regression, the share of accounts long
  (`ls_long_pct`) has the largest coefficient and it is negative: the fewer longs, the more likely a pump.
- **Small sample.** Conclusions will change as more tickers are labeled.

## Next steps

- More labeled tickers, which is the main lever.
- Metrics for the early phase of a pump (first N hours) and a model variant without trailing-return features.
- Per-ticker error review with `reports/oof_predictions.csv` in the labeling tool.
- Save the chosen model for the screener.
