-- Pump Detector — схема БД для v1 (Midterm, 1h)
-- Запуск: psql -U postgres -d pump_detector -f schema.sql

-- Довідник ринків: один рядок на пару тікер × біржа
CREATE TABLE IF NOT EXISTS markets (
    symbol              TEXT PRIMARY KEY,          -- символ Coinalyze, напр. AKEUSDT_PERP.A
    exchange_code       TEXT NOT NULL,
    exchange_name       TEXT NOT NULL,
    base_asset          TEXT NOT NULL,             -- AKE
    quote_asset         TEXT NOT NULL,             -- USDT
    symbol_on_exchange  TEXT,
    volume_denominated  TEXT,                      -- BASE_ASSET або QUOTE_ASSET (в чому рахується обсяг OHLCV)
    has_ohlcv           BOOLEAN,
    has_buy_sell        BOOLEAN,
    has_long_short      BOOLEAN,
    updated_at          TIMESTAMPTZ DEFAULT now()
);

-- Сирі часові ряди. Ключ (symbol, interval, ts): повторний запуск збору робить upsert, а не дублікати.

CREATE TABLE IF NOT EXISTS open_interest (      -- в USD (convert_to_usd=true)
    symbol TEXT REFERENCES markets(symbol), interval TEXT NOT NULL, ts TIMESTAMPTZ NOT NULL,
    o DOUBLE PRECISION, h DOUBLE PRECISION, l DOUBLE PRECISION, c DOUBLE PRECISION,
    PRIMARY KEY (symbol, interval, ts)
);

CREATE TABLE IF NOT EXISTS ohlcv (
    symbol TEXT REFERENCES markets(symbol), interval TEXT NOT NULL, ts TIMESTAMPTZ NOT NULL,
    o DOUBLE PRECISION, h DOUBLE PRECISION, l DOUBLE PRECISION, c DOUBLE PRECISION,
    v   DOUBLE PRECISION,   -- загальний обсяг
    bv  DOUBLE PRECISION,   -- buy volume
    tx  DOUBLE PRECISION,   -- кількість угод
    btx DOUBLE PRECISION,   -- кількість buy-угод
    PRIMARY KEY (symbol, interval, ts)
);

CREATE TABLE IF NOT EXISTS funding_rate (
    symbol TEXT REFERENCES markets(symbol), interval TEXT NOT NULL, ts TIMESTAMPTZ NOT NULL,
    o DOUBLE PRECISION, h DOUBLE PRECISION, l DOUBLE PRECISION, c DOUBLE PRECISION,
    PRIMARY KEY (symbol, interval, ts)
);

CREATE TABLE IF NOT EXISTS predicted_funding_rate (
    symbol TEXT REFERENCES markets(symbol), interval TEXT NOT NULL, ts TIMESTAMPTZ NOT NULL,
    o DOUBLE PRECISION, h DOUBLE PRECISION, l DOUBLE PRECISION, c DOUBLE PRECISION,
    PRIMARY KEY (symbol, interval, ts)
);

CREATE TABLE IF NOT EXISTS liquidations (       -- в USD (convert_to_usd=true)
    symbol TEXT REFERENCES markets(symbol), interval TEXT NOT NULL, ts TIMESTAMPTZ NOT NULL,
    long_liq  DOUBLE PRECISION,
    short_liq DOUBLE PRECISION,
    PRIMARY KEY (symbol, interval, ts)
);

CREATE TABLE IF NOT EXISTS long_short_ratio (
    symbol TEXT REFERENCES markets(symbol), interval TEXT NOT NULL, ts TIMESTAMPTZ NOT NULL,
    ratio     DOUBLE PRECISION,
    long_pct  DOUBLE PRECISION,
    short_pct DOUBLE PRECISION,
    PRIMARY KEY (symbol, interval, ts)
);

-- Ручна розмітка пампів (завантажується з pumps.csv). Усі часи в UTC.
CREATE TABLE IF NOT EXISTS pumps (
    id           SERIAL PRIMARY KEY,
    base_asset   TEXT NOT NULL,
    start_ts     TIMESTAMPTZ NOT NULL,
    end_ts       TIMESTAMPTZ NOT NULL,
    pattern_type TEXT,
    notes        TEXT,
    CHECK (end_ts > start_ts)
);

-- Таблиці features і signals додамо разом із features.py та скринером.
