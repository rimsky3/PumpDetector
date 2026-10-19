# Pump Detector

A machine-learning system that detects **impulses (pump episodes) in progress on crypto perpetual futures** and is built
to work on coins it never saw in training. It scans the market around the clock, flags coins, explains each flag, and
leaves the entry decision to a human trader.

The idea comes from trading practice: a pump on a small perpetual is driven by leverage, so it shows in **open interest**
before or together with the price. The model combines open interest with price, volume, funding and long/short ratio,
all measured against the coin's own history, and asks every 15 minutes: *is an impulse in progress right now?*

> **Status (October 2026).** The full pipeline runs in production on a small server: data collection, scoring every
> 15 minutes, dashboard, Telegram alerts, labeling and retraining with a promotion check.
> **Short version of the findings:** the model is a good detector and a useful filter; by itself it is not a trading
> edge on new coins. The evidence for both halves of that sentence is in the [case study](docs/case_study.md).

## Results

**Detection.** Leave-one-ticker-out: every coin is scored by a model trained without it. 153 hand-labeled impulses;
an impulse is about 3% of all bars.

| Model | Impulses caught | PR-AUC | False alarms per coin per 30 days |
|---|---|---|---|
| Rule: open interest + volume (baseline) | 90 (59%) | — | 4.0 |
| Logistic regression, hourly | 116 (76%) | 0.26 | 7.2 |
| LightGBM, hourly | 125 (82%) | 0.28 | 8.8 |
| LightGBM, 15-minute | 135 (88%) | 0.31 | 10.4 |
| LightGBM 15-minute + hourly confirmation | 129 (84%) | — | 8.3 |

The model catches clearly more impulses than the rule and pays for it with more false alarms.

**Trading on the signal.** Simulated trades on coins outside training (0.1% fee, fixed exit rule) come out near
break-even over six months: profit factor about 0.9 on 416 coins. A second model that recognises a *fresh start* of an
impulse and gates entries improves this in three separate checks and halves the number of trades. A randomization test
shows where the value is: the model picks the coin and the day, not the moment of entry.

Full numbers, the validation scheme and the list of what did not work: [docs/case_study.md](docs/case_study.md).

## How it is validated

| Risk | Check |
|---|---|
| Memorising a coin | Leave-one-ticker-out cross-validation with nested threshold selection |
| Memorising a period | Purged K-fold by time blocks with a three-day embargo |
| A lucky run | Five random seeds, paired comparison of variants |
| A lucky idea | Pass criterion written into the script before the run; results by coin halves and by month |
| Look-ahead | Feature audit, embargo for forward-looking labels, delisted coins kept in the long backtest |
| Chance | Randomization test against random entries on the same coins and days |
| Regressions | A retrained model goes live only if it is not worse than the current one on two backtests |
| Code | `python -m pytest tests` — unit tests of the core logic on synthetic data; each test was checked to fail when its safeguard is removed |

Every experiment, including the failed ones, is logged with its question, result and decision (135 entries so far).

## Approach

| Step | What happens |
|---|---|
| Data | 15-minute and 5-minute bars, open interest, funding and long/short ratio from Binance and Bybit; hourly history from [Coinalyze](https://coinalyze.net); six months of history from the Binance public archive; trades and liquidations recorded live from exchange WebSockets |
| Storage | PostgreSQL in Docker; idempotent upserts keyed by `(symbol, interval, ts)` |
| Labels | Impulses and failed impulses marked to the minute in a local tool, with a written rule for the start and the end |
| Features | 41 features relative to each coin's own history (z-scores, ratios, log changes) plus the coin's move against an equal-weighted altcoin index; one module shared by training, live scoring and experiments |
| Models | Rule baseline → logistic regression → LightGBM; a 15-minute model, an hourly confirmation model and a 15-minute "fresh start" model |
| Signal | Candidate = 15-minute model above its threshold; Alert = Candidate confirmed by the hourly model |
| Scanner | All Binance USDT perpetuals are filtered every 15 minutes into a watchlist (liquid coins plus dormant coins that woke up) |
| Screener | Scores the watchlist every 15 minutes, the hottest coins every 5 minutes on rolling bars |
| Dashboard | Charts with signal zones, simulated trades and stops, a watchlist with trade states, a journal that forward-tests every mode, backtests, per-signal explanations (TreeSHAP) |
| Alerts | Telegram bot driven by the same API as the dashboard |
| Cross-exchange spread | Live monitor of the same perpetual on Binance, Bybit and OKX: a chart pane, a per-exchange card and an alert when the executable spread is far from the coin's normal |
| Retraining | Label → retrain a candidate → compare with the live model on unseen coins → promote or reject |
| Monitoring | Signal frequency, calibration by score level, feature drift (PSI) |
| Deployment | Docker Compose on a 2-vCPU VPS: database, screener, dashboard behind HTTPS, bot, two recorders |

**Task framing.** Binary classification per bar: `1` = an impulse is in progress, `0` = normal market. This is not price
forecasting.

**The baseline matters.** A simple rule on open interest and volume is the first benchmark. A model that cannot beat it
is reported as a negative result.

## Dataset

- 154 impulses and 25 failed impulses on 33 coins, labeled by hand; 29 coins reviewed in full
- about 155,000 15-minute bars with a full feature history
- class imbalance (about 3% positive bars) is handled with class weights

Open-interest history is short at the source (about two months through the data provider, 30 days through the exchange
API), so coins are collected and labeled continuously and long backtests use the exchange's public archive.

## Quick start

```bash
docker compose up -d                 # PostgreSQL; schema.sql is applied on first start
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                 # add your Coinalyze API key and a database password

python fetch.py                      # hourly history for all tickers in tickers.txt
python fetch_exchange.py             # 15-minute and 5-minute series from the exchanges
python label.py                      # labeling tool, retraining and comparison pages
python features.py                   # compute features and labels
python train.py                      # leave-one-ticker-out comparison → reports/train_report.md, models/
python backtest.py --no-coinalyze    # trades on coins outside training
python screener.py --loop            # scan and score every 15 minutes → signals table
python dashboard.py                  # dashboard
```

Full instructions, including Windows specifics: [docs/setup.md](docs/setup.md). Every script and option:
[docs/commands.md](docs/commands.md).

## Project structure

```
fetch.py, fetch_exchange.py     collectors: Coinalyze (hourly) and exchanges (15m / 5m), rate-limited, idempotent
exchanges.py, binance_archive.py  exchange clients and the public-archive loader
liq_recorder.py, trade_recorder.py  live liquidations and trades from exchange WebSockets
features.py, alt_index.py       features (shared by training, scoring and experiments) and the altcoin index
label.py, tools/                labeling tool, retrain / compare pages, blind review and daily debrief of the model's entries
pumps.csv, load_pumps.py        hand labels (the source of truth) and their loader
train.py                        rule vs logistic regression vs LightGBM, three models, report
tiers.py, risk.py, strategies.py  signal tiers, simulated position (entry, partial exit, stop), trading modes
scanner.py, screener.py         market scanner and the scoring loop
wake.py, footprint.py, market.py, movers.py, drivers.py  early warnings, footprint marks, market breadth, movers, TreeSHAP
spread.py                       cross-exchange price spread monitor (in memory, inside the dashboard process)
dashboard.py, dashboard_app/, app/  dashboard: HTTP routes, screen logic by responsibility (list, ticker chart, trades,
                                journal, backtests, TWAP), page
watch_app/                      PD Watchlist: a small Windows window (pywebview, .exe) with only the ticker list
journal.py, backtest.py, monitor.py  forward test, backtest on unseen coins, drift and calibration
tg_bot.py                       Telegram alerts
experiments/                    one script per experiment, each with its criterion and report
tests/                          pytest: features (OI index, exchange glitches, gaps), trades, signal tiers, scanner, TWAP
schema.sql, docker-compose*.yml database schema, local and server stacks
docs/                           documentation
```

## Documentation

- [Case study](docs/case_study.md): the problem, the checks, the results and the negative results on one page
- [Setup](docs/setup.md): installation, configuration, troubleshooting
- [Commands](docs/commands.md): every script and option, typical workflows
- [Data](docs/data.md): sources, series, units, coverage and known gaps
- [Labeling](docs/labeling.md): what counts as an impulse, labeling rules, file format, tools
- [Features](docs/features.md): aggregation, feature definitions, look-ahead safeguards
- [Modeling](docs/modeling.md): models, validation scheme, metrics, experiments
- [Screener](docs/screener.md): live scoring, the signals table, dashboard, risk rules
- [Server](docs/server.md), [Telegram](docs/telegram.md), [Liquidations](docs/liquidations.md), [Trades](docs/trades.md)

## Limitations

- One labeler: the rules are written down, but the judgement is one trader's.
- 154 impulses is a small set for a model with 41 features.
- Simulated trades assume fills at the bar close and ignore slippage, which flatters thin coins.
- Training and the long backtest overlap in time (the coins do not), so absolute trade results are optimistic;
  comparisons between variants are fair.
- Exchange coverage differs by series, and some venues are not included.
