# Pump Detector

A machine-learning model that detects **ongoing pump-and-dump episodes on crypto perpetual futures**
from hourly derivatives data, and is designed to work on tickers it has never seen during training.

The core idea comes from trading practice: a pump on a low-cap perpetual is usually driven by leverage,
so it shows up in **open interest** before or together with the price. The model combines open interest
with price, volume, funding, liquidations and long/short ratio to answer one question every hour:
*is a pump in progress right now?*

> **Status:** work in progress. The full pipeline runs end to end; first results below are preliminary
> (8 tickers, 18 pumps).

## Approach

| Step | What happens | State |
|---|---|---|
| Data | Hourly OI, OHLCV, funding, predicted funding, liquidations and long/short ratio from [Coinalyze](https://coinalyze.net), aggregated across Binance, Bybit, OKX, Aster, Hyperliquid and MEXC | done |
| Storage | PostgreSQL in Docker; idempotent upserts keyed by `(symbol, interval, ts)` | done |
| Labels | Manual labeling of pump intervals in a purpose-built local tool | done (first batch) |
| Features | 24 features relative to each ticker's own history (z-scores, ratios, log changes), one module shared by training, inference and screener; verified free of look-ahead | done |
| Models | Simple OI + volume rule → Logistic Regression → LightGBM | first version |
| Validation | Leave-ticker-out cross-validation with nested threshold selection; hour- and event-level metrics | first version |
| Screener | 24/7 screener and public demo on a VPS | planned |

**Task framing.** Binary classification per hourly bar: `1` = a pump is in progress, `0` = normal market.
This is not price forecasting.

**Why leave-ticker-out.** The model must recognise pumps on new coins. Validating on hours from
tickers the model has already seen would leak each coin's specific behaviour and overstate performance.

**The baseline matters.** A simple rule on OI and volume is the first benchmark. If the model cannot
beat it, that is reported as an honest negative result.

## First results

Leave-ticker-out, every ticker scored by a model that never saw it:

| Model | Precision | Recall | F1 | PR-AUC | Pumps caught | False alarms / 30 days |
|---|---|---|---|---|---|---|
| OI + volume rule | 0.47 | 0.71 | 0.56 | — | 17/18 | 4.0 |
| Logistic regression | 0.68 | 0.75 | 0.72 | 0.70 | 17/18 | 1.9 |
| LightGBM | 0.67 | 0.81 | 0.73 | 0.74 | 17/18 | 1.2 |

Both models beat the rule on precision and false alarms. The difference between logistic regression and
LightGBM is within noise at this sample size. The most informative features point to a short squeeze:
pumps happen when most accounts are short. Details, caveats and next steps: [docs/modeling.md](docs/modeling.md).

## Current dataset

- 8 tickers, about 10 500 hourly bars with a full feature history
- 18 pump intervals and 10 false starts (moves that looked like a pump but did not develop)
- 392 pump hours, 3.7% of all bars, so class imbalance is handled explicitly

The dataset is intentionally small at this stage and will grow: Coinalyze keeps only ~2 months of
hourly open-interest history, so new tickers are collected and labeled continuously.

## Quick start

```bash
docker compose up -d                 # PostgreSQL; schema.sql is applied on first start
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                 # add your Coinalyze API key and a database password

python fetch.py                      # collect data for all tickers in tickers.txt
python fetch.py --new                # ...or only tickers not yet in the database
python label.py                      # open the labeling tool in the browser
python load_pumps.py                 # (optional) reload pumps.csv into the database
python features.py                   # compute features and labels into the features table
python train.py                      # leave-ticker-out comparison → reports/train_report.md
```

Full instructions, including Windows specifics: [docs/setup.md](docs/setup.md).

## Project structure

```
fetch.py            Coinalyze → PostgreSQL collector (rate-limited, idempotent)
tickers.txt         list of tickers to collect
label.py            local labeling server (127.0.0.1) for tools/label.html
tools/label.html    labeling UI: synced price / OI / volume charts, click to mark intervals
pumps.csv           manual labels, the source of truth for the target variable
load_pumps.py       validates pumps.csv and syncs it to the database
features.py         raw series → hourly features + labels (shared by training and screener)
train.py            rule vs logistic regression vs LightGBM, leave-ticker-out
reports/            generated training report
schema.sql          database schema
docker-compose.yml  local PostgreSQL
docs/               documentation
```

## Documentation

- [Setup](docs/setup.md): installation, configuration, troubleshooting
- [Data](docs/data.md): sources, series, units, coverage and known gaps
- [Labeling](docs/labeling.md): what counts as a pump, labeling rules, file format, tool usage
- [Features](docs/features.md): aggregation, feature definitions, look-ahead and listing safeguards
- [Modeling](docs/modeling.md): models, validation scheme, metrics, first results

## Limitations

- Small labeled set; conclusions from the first models will be preliminary.
- Hourly history is limited to ~60 days per ticker by the data provider.
- Exchange coverage differs by series (for example, Aster has no open interest in Coinalyze).
- Labels reflect one trader's judgement. The labeling rules are documented to keep them consistent.
