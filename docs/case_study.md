# Case study: detecting impulses on crypto perpetuals

A one-page summary of what this project set out to do, how it was checked, what worked and what did not.
Numbers are as of 12 October 2026. The full experiment log has 135 entries; most of them are negative results.

## The problem

Small and mid-cap perpetual futures occasionally make a fast, leveraged move, called an *impulse* here. A discretionary
trader who wants to trade them has to watch several hundred coins. The goal was a model that flags an impulse in progress
on a coin it has never seen, early enough to act on, and explains which measurements produced the flag. The model is a
filter for a human who decides with the order book in front of them. It is not an automated strategy.

## Data and labels

- **Sources.** 15-minute and hourly bars of price, volume, taker buy volume, open interest, funding and long/short ratio
  from Binance and Bybit, hourly history from Coinalyze, and the Binance public archive for six months of history.
  Trades and liquidations are recorded live from exchange WebSockets.
- **Labels.** 154 impulses and 25 failed impulses on 33 coins, marked by hand to the minute in a purpose-built tool.
  The start is where open interest and price begin to accelerate together. The end follows a written rule (price gives
  back a third of the move, or open interest drops 10% from its peak), so the labels can be reproduced.
- **Features.** 41 features, all relative to the coin's own history (z-scores, ratios, log changes), so a $2M coin and a
  $200M coin are comparable. One module computes them for training, live scoring and every experiment.

## The model

Two LightGBM classifiers on 15-minute bars and one on hourly bars:

1. the main 15m model answers "is an impulse in progress?";
2. the hourly model confirms it, which removes a fifth to a third of the false alarms at the cost of some delay;
3. a second 15m model, trained only on the first 30 minutes of each impulse, answers "is this a fresh start?" and gates
   entries.

A simple rule on open interest and volume is kept as the baseline. Logistic regression is kept as a second reference.

## How results were checked

| Risk | What was done |
|---|---|
| The model memorises a coin | Leave-one-ticker-out: every coin is scored by a model trained without it |
| The model memorises a period | Purged K-fold by time blocks with a three-day embargo on both sides of the test block |
| A lucky run | Five random seeds and paired comparison; run-to-run spread is about 0.02 in profit factor |
| A lucky idea | The pass criterion is written into the script before the run, and results are split by coin halves and months |
| Look-ahead | Features audited bar by bar; labels that look forward get an embargo; delisted coins are kept in the six-month universe |
| Chance | A randomization test: the model's entries against 10,000 worlds with random entries on the same coins and days |
| Regressions | A retrained model replaces the live one only if it is not worse on a 30-day and a six-month backtest |

## Results

**Detection, leave-one-ticker-out, 153 labeled impulses.** Impulses are about 3% of bars.

| Model | Impulses caught | PR-AUC | False alarms per coin per 30 days |
|---|---|---|---|
| Rule: open interest + volume | 90 (59%) | — | 4.0 |
| Logistic regression, hourly | 116 (76%) | 0.26 | 7.2 |
| LightGBM, hourly | 125 (82%) | 0.28 | 8.8 |
| LightGBM, 15m | 135 (88%) | 0.31 | 10.4 |
| LightGBM 15m + hourly confirmation | 129 (84%) | — | 8.3 |

The model catches clearly more impulses than the rule and pays for it with more false alarms. The median delay from
the labeled start to the first signal is 40 to 75 minutes, depending on the tier.

**Does a signal make a trade?** Simulated trades on coins outside training, 0.1% fee, a fixed exit rule:

- Six months, 416 coins: profit factor about 0.9. The signal alone does not have an edge on new coins.
- The "fresh start" gate improves this (0.84 → 0.95 on liquid coins over six months, 0.79 → 0.82 in purged
  cross-validation, 1.26 → 1.43 in the last 30 days) and halves the number of trades.
- The randomization test shows where the value is: entering at the start of a signal is far better than a random later
  bar of the same signal (1.48 against 0.63–0.97), but not better than a random bar of the same coin on the same day.
  The model picks the coin and the day; it does not pick the moment.

So the honest summary is: **a good detector, a useful filter, not a trading edge by itself.**

## What did not work

Each of these was tested with a criterion fixed in advance and failed or stayed within noise: order-book depth, trade
flow and footprint as features; price levels; spot data; time of day; hidden Markov regimes; a target based on trade
outcome; an event-driven model; teacher–student labeling as a replacement for hand labels; seed ensembles on the full
data; six different exit rules; mechanical pullback entries; "in play" filters; shorting exhausted coins.

What did work: a two-tier signal, more labels, a partial exit, features that measure the coin's move against an
equal-weighted altcoin index, and the fresh-start gate.

## Lessons

- **Labels beat features.** Dropping 23 older impulses cost more (0.883 → 0.838) than any feature tested that week gained.
- **Incomplete labels hurt.** Coins marked as fully reviewed but only partly labeled turned real impulses into negatives;
  the retrained model lost to the live one and was rejected by the promotion check.
- **A slice that looks strong on one month usually is not.** "Pullback entries are better" was 1.82 against 1.38 on 30
  days and vanished on six months.
- **The null hypothesis can look ahead too.** A random-entry benchmark built on the days the model fired is biased upward.
- **A filter's cost is visible, its benefit is not.** The missed good trade is remembered; the avoided bad ones are not.
  Rules are judged on all trades.

## The system

PostgreSQL, a screener that scores a scanner-built watchlist every 15 minutes (hot coins every five), a dashboard with
charts and per-signal explanations (TreeSHAP), a Telegram bot, a labeling and review tool, drift and calibration
monitoring, all in Docker Compose on a single 2-vCPU server.

## Limitations

- One labeler. The rules are written down, but the judgement is one trader's.
- 154 impulses is a small set for a model with 41 features.
- Trade simulations assume fills at the bar close and ignore slippage, which flatters thin coins.
- Training and the six-month test overlap in time (the coins do not), so absolute profit factors are optimistic; comparisons
  between variants are fair.

## Next

Ratings of the model's own entries by the trader (about 200 are needed) to train a filter on human judgement rather
than on trade outcome, and a repeat of the out-of-time check on a full month of unseen data.
