# Pump Detector

A machine-learning model that detects **ongoing pump-and-dump episodes on crypto perpetual futures**
from hourly derivatives data, and is designed to work on tickers it has never seen during training.

The core idea comes from trading practice: a pump on a low-cap perpetual is usually driven by leverage,
so it shows up in **open interest** before or together with the price. The model combines open interest
with price, volume, funding, liquidations and long/short ratio to answer one question every hour:
*is a pump in progress right now?*

> **Status:** work in progress. Data collection and labeling are done; feature engineering and modelling are next.
> No model results are reported yet.

## Approach

| Step | What happens | State |
|---|---|---|
| Data | Hourly OI, OHLCV, funding, predicted funding, liquidations and long/short ratio from [Coinalyze](https://coinalyze.net), aggregated across Binance, Bybit, OKX, Aster, Hyperliquid and MEXC | done |
| Storage | PostgreSQL in Docker; idempotent upserts keyed by `(symbol, interval, ts)` | done |
| Labels | Manual labeling of pump intervals in a purpose-built local tool | done (first batch) |
| Features | Features relative to each ticker's own history (z-scores, ratios, % changes), one module shared by training, inference and screener | next |
| Models | Simple OI + volume rule → Logistic Regression → LightGBM | planned |
| Validation | Leave-ticker-out cross-validation; precision / recall / F1, not accuracy | planned |
| Screener | 24/7 screener and public demo on a VPS | planned |

**Task framing.** Binary classification per hourly bar: `1` = a pump is in progress, `0` = normal market.
This is not price forecasting.

**Why leave-ticker-out.** The model must recognise pumps on new coins. Validating on hours from
tickers the model has already seen would leak each coin's specific behaviour and overstate performance.

**The baseline matters.** A simple rule on OI and volume is the first benchmark. If the model cannot
beat it, that is reported as an honest negative result.

## Current dataset

- 6 tickers (AIN, AKE, G, LSK, MYX, QNT), 60 days of hourly bars
- 18 labeled intervals: 10 pumps and 8 false starts (moves that looked like a pump but did not develop)
- About 270 pump hours, roughly 3% of all bars, so class imbalance is handled explicitly

The dataset is intentionally small at this stage and will grow: Coinalyze keeps only ~2 months of
hourly open-interest history, so new tickers are collected and labeled continuously.

## Quick start

```bash
docker compose up -d                 # PostgreSQL; schema.sql is applied on first start
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                 # add your Coinalyze API key and a database password

python fetch.py                      # collect data for all tickers in tickers.txt
python label.py                      # open the labeling tool in the browser
python load_pumps.py                 # (optional) reload pumps.csv into the database
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
schema.sql          database schema
docker-compose.yml  local PostgreSQL
docs/               documentation
```

## Documentation

- [Setup](docs/setup.md): installation, configuration, troubleshooting
- [Data](docs/data.md): sources, series, units, coverage and known gaps
- [Labeling](docs/labeling.md): what counts as a pump, labeling rules, file format, tool usage

## Limitations

- Small labeled set; conclusions from the first models will be preliminary.
- Hourly history is limited to ~60 days per ticker by the data provider.
- Exchange coverage differs by series (for example, Aster has no open interest in Coinalyze).
- Labels reflect one trader's judgement. The labeling rules are documented to keep them consistent.
