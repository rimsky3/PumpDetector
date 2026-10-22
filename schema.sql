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

-- З якого моменту внутрішньогодинні ліквідації (15min / 5min) повні: раніше — «невідомо», а не 0.
-- Coinalyze тримає обмежену кількість непорожніх точок, тож covered_from = перша віддана точка (fetch.collect_liquidations).
CREATE TABLE IF NOT EXISTS liquidation_coverage (
    symbol TEXT REFERENCES markets(symbol), interval TEXT NOT NULL, covered_from TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (symbol, interval)
);

-- Ручна розмітка пампів (завантажується з pumps.csv). Усі часи в UTC.
CREATE TABLE IF NOT EXISTS pumps (
    id           SERIAL PRIMARY KEY,
    base_asset   TEXT NOT NULL,
    start_ts     TIMESTAMPTZ NOT NULL,
    end_ts       TIMESTAMPTZ NOT NULL,
    pattern_type TEXT,
    notes        TEXT,
    entry_ts     TIMESTAMPTZ,          -- необов'язково: де власник зайшов би
    entry_price  DOUBLE PRECISION,
    stop_price   DOUBLE PRECISION,     -- і де поставив би стоп
    setup        TEXT,                 -- breakout / pullback / other (лише для аналізу, модель на ньому не вчиться)
    CHECK (end_ts > start_ts)
);

-- Таблицю features створює features.py (вона повністю похідна і перебудовується при зміні ознак).

-- Оцінки скринера: ймовірність пампу від кожної моделі на кожну закриту годину.
-- in_sample = true — ця година цього тікера була в навчальній вибірці моделі (оцінка не є передбаченням).
-- live = true — година була останньою закритою на момент оцінки і не in_sample: чесний forward-test.
-- live = false — дозаповнення історії при першому запуску, після пропуску або in-sample.
-- Перша оцінка години не перезаписується: так історія сигналів лишається чесною.
CREATE TABLE IF NOT EXISTS signals (
    base_asset    TEXT NOT NULL,
    ts            TIMESTAMPTZ NOT NULL,         -- час відкриття оціненого бару
    model         TEXT NOT NULL,                -- logreg / lightgbm
    model_version TEXT NOT NULL,                -- коли модель навчена (trained_at)
    probability   DOUBLE PRECISION,
    threshold     DOUBLE PRECISION,
    signal        BOOLEAN NOT NULL,
    live          BOOLEAN NOT NULL,
    in_sample     BOOLEAN NOT NULL DEFAULT false,
    scored_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (base_asset, ts, model, model_version)
);

-- Стакан: біди / аски ±2% від ціни (Binance + Bybit), знімки скринера для тікерів біля сигналу (orderbook.py).
-- ratio ≥ 1 — покупців у стакані не менше, ніж продавців; «OB» біля Alert у дашборді.
CREATE TABLE IF NOT EXISTS book_ratio (
    base_asset TEXT NOT NULL,
    ts         TIMESTAMPTZ NOT NULL,
    ratio      DOUBLE PRECISION,
    bid_usd    DOUBLE PRECISION,
    ask_usd    DOUBLE PRECISION,
    binance    DOUBLE PRECISION,
    bybit      DOUBLE PRECISION,
    PRIMARY KEY (base_asset, ts)
);

-- Ліквідації напряму з бірж через WebSocket (liq_recorder.py, з 2026-10-06): кожна подія + періоди живих з'єднань.
-- «Топ-трейдери» Binance (≈ 20% найбільших акаунтів за маржею), 15m, з 2026-10-22: частка лонгів за кількістю акаунтів
-- і за розміром позицій, %. Лише для панелі дашборда (натовп проти великих грошей), у модель не йде. Біржа віддає 30 днів.
CREATE TABLE IF NOT EXISTS top_long_short (
    symbol TEXT REFERENCES markets(symbol), interval TEXT NOT NULL, ts TIMESTAMPTZ NOT NULL,
    acc_long_pct DOUBLE PRECISION, pos_long_pct DOUBLE PRECISION,
    PRIMARY KEY (symbol, interval, ts)
);

-- Виконання угод власника на Binance USDⓈ-M Futures (my_positions.py, з 2026-10-20; ключ лише читання): для заміру edge
-- власника проти входів моделі на тих самих монетах. Позиції — з послідовності виконань (як my_trades.positions).
CREATE TABLE IF NOT EXISTS my_fills (
    symbol TEXT NOT NULL, trade_id BIGINT NOT NULL,
    ts TIMESTAMPTZ NOT NULL, side TEXT, position_side TEXT, order_type TEXT,
    price DOUBLE PRECISION, qty DOUBLE PRECISION, quote DOUBLE PRECISION,
    realized_pnl DOUBLE PRECISION, commission DOUBLE PRECISION, commission_asset TEXT,
    order_id BIGINT, maker BOOLEAN, reduce_only BOOLEAN,
    recv_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol, trade_id)
);

CREATE TABLE IF NOT EXISTS liq_events (
    exchange  TEXT NOT NULL,              -- Binance / Bybit / OKX (як markets.exchange_name)
    symbol    TEXT NOT NULL,              -- символ на біржі (markets.symbol_on_exchange): BTCUSDT, BTC-USDT-SWAP
    ts        TIMESTAMPTZ NOT NULL,       -- час ліквідації (біржовий, мс)
    side      TEXT NOT NULL,              -- long / short — чию позицію закрито
    qty       DOUBLE PRECISION,           -- у монетах
    price     DOUBLE PRECISION,
    usd       DOUBLE PRECISION,
    recv_at   TIMESTAMPTZ NOT NULL DEFAULT now()   -- коли повідомлення прийшло (затримка потоку = recv_at − ts)
);
CREATE INDEX IF NOT EXISTS liq_events_sym_ts ON liq_events (exchange, symbol, ts);
CREATE INDEX IF NOT EXISTS liq_events_ts ON liq_events (ts);
CREATE TABLE IF NOT EXISTS liq_ws_coverage (
    exchange  TEXT NOT NULL,
    conn      TEXT NOT NULL,              -- з'єднання (у Bybit їх кілька: bybit#0, bybit#1 …)
    started   TIMESTAMPTZ NOT NULL,       -- підключились і підписались
    last_seen TIMESTAMPTZ NOT NULL,       -- з'єднання ще живе станом на цей момент
    symbols   INTEGER,                    -- на скільки символів підписані (Binance/OKX — увесь ринок, NULL)
    PRIMARY KEY (exchange, conn, started)
);

-- Угоди Binance / Bybit через WebSocket (trade_recorder.py, з 2026-10-07): 5-хв зведення за стороною агресора
-- і кошиками розміру угоди + періоди живих з'єднань.
CREATE TABLE IF NOT EXISTS trade_bars (
    exchange  TEXT NOT NULL,              -- Binance / Bybit (як markets.exchange_name)
    symbol    TEXT NOT NULL,              -- символ на біржі (markets.symbol_on_exchange): BTCUSDT
    ts        TIMESTAMPTZ NOT NULL,       -- відкриття 5-хв бару
    n_buy     INTEGER NOT NULL,           -- угод, де агресор купував
    n_sell    INTEGER NOT NULL,
    v_buy     DOUBLE PRECISION NOT NULL,  -- $ агресивних купівель
    v_sell    DOUBLE PRECISION NOT NULL,
    max_buy   DOUBLE PRECISION,           -- найбільша угода бару, $
    max_sell  DOUBLE PRECISION,
    buy_n     INTEGER[] NOT NULL,         -- угод за кошиками розміру (межі — trade_recorder.SIZE_BOUNDS)
    sell_n    INTEGER[] NOT NULL,
    buy_usd   DOUBLE PRECISION[] NOT NULL,
    sell_usd  DOUBLE PRECISION[] NOT NULL,
    fp_lvl0   INTEGER,                    -- футпрінт: перший рівень ціни (footprint.level)
    fp_buy    DOUBLE PRECISION[],         -- $ агресивних купівель на рівнях fp_lvl0, fp_lvl0 + 1, …
    fp_sell   DOUBLE PRECISION[],
    PRIMARY KEY (exchange, symbol, ts)
);
ALTER TABLE trade_bars ADD COLUMN IF NOT EXISTS fp_lvl0 INTEGER;
ALTER TABLE trade_bars ADD COLUMN IF NOT EXISTS fp_buy DOUBLE PRECISION[];
ALTER TABLE trade_bars ADD COLUMN IF NOT EXISTS fp_sell DOUBLE PRECISION[];
CREATE INDEX IF NOT EXISTS trade_bars_ts ON trade_bars (ts);
CREATE TABLE IF NOT EXISTS trade_ws_coverage (
    exchange  TEXT NOT NULL,
    conn      TEXT NOT NULL,              -- binance#0, bybit#2 …
    started   TIMESTAMPTZ NOT NULL,       -- підключились і підписались
    last_seen TIMESTAMPTZ NOT NULL,       -- з'єднання ще живе станом на цей момент
    symbols   INTEGER,
    PRIMARY KEY (exchange, conn, started)
);

-- Мітки патернів футпрінту b / p («залочка», footprint.py, з 2026-10-07): скринер пише їх для тікерів з Кандидатом / Alert
-- за 6 год; дашборд — мітка під / над свічкою, бот — сповіщення; результат мітки рахується при читанні (footprint.outcome).
CREATE TABLE IF NOT EXISTS fp_marks (
    base_asset  TEXT NOT NULL,
    ts          TIMESTAMPTZ NOT NULL,        -- відкриття 15m-бару з патерном
    kind        TEXT NOT NULL,               -- b (залочка на лоу) / p (на хаях)
    close       DOUBLE PRECISION,            -- закриття бару = ціна входу підказки
    stop        DOUBLE PRECISION,            -- b: під лоу бару, p: над хаєм (STOP_PAD)
    level       DOUBLE PRECISION,            -- пробитий лоу (b) / хай (p) попередніх 2 год
    beyond      DOUBLE PRECISION,            -- частка обсягу бару за рівнем
    beyond_delta DOUBLE PRECISION,           -- дельта за рівнем (купівлі − продажі), частка обсягу бару
    edge        DOUBLE PRECISION,            -- частка обсягу на протилежному краї бару
    vol_x       DOUBLE PRECISION,            -- обсяг бару / медіана доби
    ctx         TEXT,                        -- alert / candidate — контекст моделі за 6 год
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (base_asset, ts, kind)
);

-- Стан ринку альтів щогодини (market.py; дашборд — кнопка «Market», журнал — режим «ринок не гарячий»)
CREATE TABLE IF NOT EXISTS market_breadth (
    ts TIMESTAMPTZ PRIMARY KEY,          -- відкриття години; значення відомі на її закритті (ts + 1 год)
    n INT,                               -- скільки альтів з історією за 7 днів
    breadth_24h DOUBLE PRECISION, breadth_7d DOUBLE PRECISION,       -- частка альтів у плюсі за 24 год / 7 днів
    alt_med_24h DOUBLE PRECISION, alt_med_7d DOUBLE PRECISION,       -- медіанна зміна альтів
    alt_vs_btc_24h DOUBLE PRECISION,     -- медіана альтів мінус BTC за 24 год
    btc_ret_24h DOUBLE PRECISION, btc_vs_ma7d DOUBLE PRECISION, btc_dd_7d DOUBLE PRECISION, btc_vol_x DOUBLE PRECISION,
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- ALT-EW: середня 5m log-доходність ~250 ліквідних альтів Binance (alt_index.py; ознаки власного руху 15m-моделі)
CREATE TABLE IF NOT EXISTS alt_index (
    ts TIMESTAMPTZ PRIMARY KEY,          -- відкриття 5m-бару; значення відоме на його закритті (ts + 5 хв)
    ret DOUBLE PRECISION NOT NULL,       -- середня log-доходність альтів за бар
    n INT NOT NULL,                      -- скільки монет на барі (менше 50 — бар не пишеться)
    updated_at TIMESTAMPTZ DEFAULT now()
);
