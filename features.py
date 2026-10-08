"""
features.py — ознаки для моделі з сирих рядів Coinalyze.

Один модуль для тренування, інференсу і скринера: логіка ознак живе лише тут.

Конвеєр:
    load_raw()          сирі таблиці з БД → pandas
    aggregate()         біржі тікера → один погодинний ряд (суми в USD, зважені середні)
    compute_features()  погодинний ряд → ознаки, відносні до власної історії тікера
    add_labels()        мітка 1 = година всередині розміченого пампу
    build()             усе разом для списку тікерів

Правило: ознака на годину t рахується лише з даних до t включно (без заглядання в майбутнє).

    python features.py              # перерахувати ознаки для тікерів з tickers.txt і записати в таблицю features
    python features.py AKE QNT      # лише для вибраних тікерів (решта рядків таблиці лишається)

Таблиця features — навчальна вибірка:
    тікери з tickers.txt — повністю переглянуті: усі години, нерозмічена година = «пампу не було» (reviewed = true);
    будь-який інший тікер з розміткою в pumps — лише години всередині розмічених інтервалів (reviewed = false).
Так розмітку окремого сигналу з журналу можна взяти в навчання, не оголошуючи решту 60 днів тікера «без пампів».
"""

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import psycopg
from dotenv import load_dotenv

from fetch import TICKERS_FILE, read_tickers_file

INTERVAL = "1hour"
PRICE_EXCHANGE = "Binance"   # еталонна біржа для ціни; якщо тікера там немає — перша за алфавітом
BASE_WINDOW = 168            # 7 днів: «звичайний рівень», від якого рахуємо відхилення
SHORT_WINDOW = 24            # коротке вікно: остання доба

# Ознаки, які бачить модель. Порядок фіксований — той самий для тренування й інференсу.
FEATURES = [
    # open interest: зміни по «зв'язаному» індексу, стійкі до стрибків при лістингу
    "oi_chg_1h", "oi_chg_4h", "oi_chg_24h", "oi_z_24h", "oi_z_7d",
    # ціна
    "ret_1h", "ret_4h", "ret_24h", "volatility_24h", "range_1h",
    # OI відносно ціни: OI росте швидше за ціну — набирають плечі
    "oi_minus_ret_24h",
    # обсяг
    "volume_z_7d", "volume_ratio_24h", "buy_ratio_1h", "buy_ratio_4h",
    # funding
    "funding", "funding_z_7d", "funding_chg_24h",
    # ліквідації
    "liq_long_4h_oi", "liq_short_4h_oi", "liq_imbalance_4h",
    # long/short ratio
    "ls_long_pct", "ls_chg_24h",
]

# OI у монетах: ті самі зміни й z-score, але без впливу ціни (див. aggregate).
# Ознаки для експерименту «OI у монетах проти OI у USD»; чи входять вони в модель — вирішує FEATURES.
OI_COIN_FEATURES = ["oi_coin_chg_1h", "oi_coin_chg_4h", "oi_coin_chg_24h", "oi_coin_z_24h", "oi_coin_z_7d"]
ALL_FEATURES = FEATURES + [f for f in OI_COIN_FEATURES if f not in FEATURES]

# 15m-модель: ті самі FEATURES (вікна за часом — «24h» = доба) + короткі ознаки бару (experiments/m15_model.py)
INTERVAL_15M = "15min"
# Біржі, з яких 15m/5m-ряди (свічки, OI, funding, L/S) ідуть у модель і графіки. OKX збирається з 2026-09-30, але
# 15m-модель навчена на сумі Binance + Bybit, а історії OKX-OI лише ~14 днів: вмикати — після накопичення,
# експерименту і перенавчання (інакше ознаки зсунуться, а на графіку з'явиться фальшивий «стрибок» OI).
INTRADAY_EXCHANGES = ("Binance", "Bybit")
SHORT_FEATURES = ["ret_bar", "ret_2bar", "oi_chg_bar", "oi_chg_2bar", "oi_accel_1h", "ret_accel_1h",
                  "vol_burst", "buy_ratio_bar", "range_bar", "liq_long_1h_oi", "liq_short_1h_oi"]
# Власний рух монети відносно ринку альтів (experiments/idio_features.py: 6 міс., чесний список монет — PF 0.89 → 0.93,
# кращий у 5 з 6 місяців; пояснення кожної — notes/movers.md). Ринок — ALT-EW 5m-індекс з таблиці alt_index (alt_index.py).
IDIO_FEATURES = ["beta", "r2", "idio_1h", "idio_4h", "idio_24h", "idio_z_4h"]      # бета / R² до ринку, рух понад ринок
AMIHUD_FEATURES = ["illiq_x"]                                                       # Amihud ILLIQ до свого звичайного
VOLX_FEATURES = ["park_24h", "gk_24h", "park_x", "ret_4h_vol", "ret_24h_vol"]       # волатильність з high/low і рух у ній
MOVE_FEATURES = IDIO_FEATURES + AMIHUD_FEATURES + VOLX_FEATURES
FEATURES_15M = FEATURES + SHORT_FEATURES
# Ознаки ліквідацій рахуються й лежать у таблиці features, але в моделі з 2026-10-07 не йдуть (train.py --with-liq — повернути):
# experiments/no_liq.py — на розмітці імпульсів без них не гірше (бектест PF 0.89 проти 0.83), а 15m-оцінка не чекає
# ~200 викликів Coinalyze. Самі ліквідації збираємо далі (Coinalyze, liq_recorder) — знадобляться (напр., шорт-модель).
LIQ_FEATURES = [f for f in FEATURES_15M if f.startswith("liq_")]
MODEL_FEATURES = [f for f in FEATURES if f not in LIQ_FEATURES]
MODEL_FEATURES_15M = [f for f in FEATURES_15M if f not in LIQ_FEATURES]
# Робоча 15m-модель з 2026-10-08 — MODEL_FEATURES_15M + MOVE_FEATURES (train.py; --no-move — без них). FEATURES_15M і
# MODEL_FEATURES_15M лишились базовими наборами: на них стоять експерименти до 10-08 (build_from_raw з short_features).


# ---------------------------------------------------------------- завантаження

def query_df(conn, sql: str, params=()) -> pd.DataFrame:
    cur = conn.execute(sql, params)
    df = pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description])
    if "ts" in df:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def load_raw(conn, bases: list[str] | None = None, interval: str = INTERVAL, since=None,
             exchanges=None) -> dict[str, pd.DataFrame]:
    """Сирі ряди з назвою тікера і біржі в кожному рядку. interval: '1hour' (Coinalyze) або '15min' / '5min'
    (біржі напряму + ліквідації Coinalyze; тоді ще й liq_coverage — з якого моменту ліквідації ринку повні)."""
    cond = "t.interval = %s" + (" AND m.base_asset = ANY(%s)" if bases else "")
    params = (interval, bases) if bases else (interval,)
    tcond, tparams = (cond + " AND t.ts >= %s", params + (since,)) if since is not None else (cond, params)
    # 15m/5m: лише біржі моделі (ліквідації — з Coinalyze по всіх біржах, як у навчанні, їх не фільтруємо)
    if exchanges is None and interval != INTERVAL:
        exchanges = INTRADAY_EXCHANGES
    xcond, xparams = (tcond + " AND m.exchange_name = ANY(%s)", tparams + (list(exchanges),)) if exchanges and exchanges != "all" \
        else (tcond, tparams)

    def q(table: str, cols: str, by_exchange: bool = True) -> pd.DataFrame:
        # market — конкретний контракт (на одній біржі їх буває кілька: USDT і USDC)
        c, p = (xcond, xparams) if by_exchange else (tcond, tparams)
        return query_df(conn, f"""
            SELECT m.base_asset, t.symbol AS market, m.exchange_name AS exchange, m.quote_asset, t.ts, {cols}
            FROM {table} t JOIN markets m USING (symbol)
            WHERE {c}""", p)

    raw = {
        "ohlcv": q("ohlcv", "t.o, t.h, t.l, t.c, t.v, t.bv, m.volume_denominated"),
        "oi": q("open_interest", "t.o, t.h, t.l, t.c"),
        "funding": q("funding_rate", "t.c"),
        "liq": q("liquidations", "t.long_liq, t.short_liq", by_exchange=False),
        "ls": q("long_short_ratio", "t.long_pct"),
    }
    if interval != INTERVAL:
        # внутрішньогодинні ліквідації Coinalyze тримає обмежено: до covered_from «рядка немає» ≠ «0 ліквідацій»
        raw["liq_coverage"] = query_df(conn, f"""
            SELECT m.base_asset, t.symbol AS market, t.covered_from FROM liquidation_coverage t JOIN markets m USING (symbol)
            WHERE {cond}""", params).assign(covered_from=lambda d: pd.to_datetime(d["covered_from"], utc=True))
    return raw


def load_pumps(conn) -> pd.DataFrame:
    return query_df(conn, "SELECT base_asset, start_ts, end_ts, pattern_type FROM pumps").assign(
        start_ts=lambda d: pd.to_datetime(d["start_ts"], utc=True),
        end_ts=lambda d: pd.to_datetime(d["end_ts"], utc=True),
    )


# ---------------------------------------------------------------- агрегація бірж

def weighted_mean(values: pd.DataFrame, weights: pd.DataFrame) -> pd.Series:
    """Середнє по біржах, зважене за OI. Якщо ваг немає (біржа без OI) — просте середнє."""
    w = weights.reindex(columns=values.columns).where(values.notna())
    num = (values * w).sum(axis=1, min_count=1)
    den = w.sum(axis=1, min_count=1)
    return (num / den).where(den > 0, values.mean(axis=1))


def chained_log_index(w: pd.DataFrame) -> pd.Series:
    """«Зв'язаний» лог-індекс суми по контрактах: погодинна зміна рахується лише по контрактах,
    які мали значення і годину тому, і зараз. Тоді поява нової біржі (лістинг) не виглядає як ріст OI.
    Нуль / від'ємне — пропуск (архів Binance зрідка має OI = 0: log(0) = −∞, і без цього весь ряд після нього ставав NaN)."""
    w = w.where(w > 0)
    prev = w.shift(1)
    both = w.notna() & prev.notna()
    step = np.log(w.where(both).sum(axis=1, min_count=1) / prev.where(both).sum(axis=1, min_count=1))
    return step.fillna(0).cumsum().where(w.notna().any(axis=1))


def split_raw(raw: dict[str, pd.DataFrame]) -> dict[str, dict[str, pd.DataFrame]]:
    """Сирі ряди багатьох тікерів → {тікер: сирі ряди лише цього тікера} одним проходом (groupby).
    aggregate(raw, base) сам відбирає рядки тікера, але по ВСІХ рядках raw — у циклі по десятках тікерів це
    квадратична робота (список дашборду, 2026-10-06). Результат aggregate той самий."""
    empty = {k: v.iloc[0:0] for k, v in raw.items()}
    out: dict[str, dict[str, pd.DataFrame]] = {}
    for k, v in raw.items():
        if "base_asset" not in v:
            continue
        for b, g in v.groupby("base_asset", sort=False):
            out.setdefault(b, dict(empty))[k] = g
    return out


def aggregate(raw: dict[str, pd.DataFrame], base: str, freq: str = "h", light: bool = False) -> pd.DataFrame:
    """Усі біржі тікера → один ряд на регулярній сітці (freq: "h" — години; "15min" — для експериментів).
    light=True — лише ціна (open/high/low/price), обсяг і OI (oi, oi_log_idx): для списку тікерів дашборду, де потрібні
    тільки зміни за 24 год і ATR для підказки ліміту. Ті самі значення, що й у повному ряді, але без свічок OI,
    funding, ліквідацій і L/S (повний ряд для ~64 тікерів — ~16 с на сервері, 2026-10-06)."""
    part = {k: v[v["base_asset"] == base] for k, v in raw.items() if k != "liq_coverage"}
    ohlcv = part["ohlcv"].copy()
    if ohlcv.empty:
        return pd.DataFrame()

    grid = pd.date_range(ohlcv["ts"].min(), ohlcv["ts"].max(), freq=freq)

    def wide(df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Рядки — години, колонки — контракти."""
        if df.empty:
            return pd.DataFrame(index=grid)
        # те саме, що pivot_table(aggfunc="last"), але в ~10× швидше (pivot_table — найповільніше місце списку
        # дашборду, 2026-10-06): дублікати (ts, контракт) — лишаємо останнє НЕпорожнє значення; контракти без
        # жодного значення прибираємо (pivot_table з dropna=True робить так само)
        d = df[["ts", "market", col]]
        if d.duplicated(["ts", "market"]).any():
            d = d.dropna(subset=[col]).drop_duplicates(["ts", "market"], keep="last")
        w = d.pivot(index="ts", columns="market", values=col).dropna(axis=1, how="all")
        return w.reindex(grid)

    # обсяг буває в монетах — переводимо в USD через ціну закриття
    in_coins = ohlcv["volume_denominated"] == "BASE_ASSET"
    ohlcv["v_usd"] = np.where(in_coins, ohlcv["v"] * ohlcv["c"], ohlcv["v"])
    ohlcv["bv_usd"] = np.where(in_coins, ohlcv["bv"] * ohlcv["c"], ohlcv["bv"])

    # контракт для ціни: еталонна біржа, серед її контрактів — USDT; далі найдовша історія
    counts = ohlcv.groupby(["market", "exchange", "quote_asset"]).size().rename("n").reset_index()
    counts["rank"] = list(zip(counts["exchange"] != PRICE_EXCHANGE, counts["quote_asset"] != "USDT", -counts["n"]))
    px_market = counts.sort_values("rank").iloc[0]["market"]
    px = ohlcv[ohlcv["market"] == px_market].set_index("ts").reindex(grid)

    oi_w = wide(part["oi"], "c")
    h = pd.DataFrame(index=grid)
    h.index.name = "ts"
    h["price"] = px["c"].ffill(limit=2)          # поодинокі пропуски заповнюємо попереднім значенням
    h["open"] = px["o"]                           # свічки в дашборді; волатильність Гарман–Класа (gk_24h)
    h["high"] = px["h"]
    h["low"] = px["l"]
    h["volume"] = wide(ohlcv, "v_usd").sum(axis=1, min_count=1)
    if light:
        h["oi"] = oi_w.sum(axis=1, min_count=1)
        h["oi_log_idx"] = chained_log_index(oi_w)
        return h
    # OI у монетах. Coinalyze переводить OI у USD за ціною закриття години — усі o/h/l/c свічки одразу.
    # Тому в USD OI «стрибає» між годинами разом із ціною, хоча позиції не змінювались.
    # Ділимо на close того самого контракту: отримуємо чисту кількість відкритих позицій у монетах
    # (розрив між годинами ≈ 0.01% замість 0.5–2.6%). Сума по біржах коректна: одиниця — монета.
    close_w = wide(ohlcv, "c").reindex(columns=oi_w.columns).ffill(limit=2)
    coins = {k: wide(part["oi"], k) / close_w for k in ("o", "h", "l", "c")}
    h["buy_volume"] = wide(ohlcv, "bv_usd").sum(axis=1, min_count=1)
    # частка покупок — лише по ринках, де buy volume є (у 15m/5m-свічок Bybit його немає; на 1h є скрізь)
    h["volume_bv"] = wide(ohlcv.assign(v_bv=ohlcv["v_usd"].where(ohlcv["bv"].notna())), "v_bv").sum(axis=1, min_count=1)
    h["oi"] = oi_w.sum(axis=1, min_count=1)
    h["oi_log_idx"] = chained_log_index(oi_w)
    for k in ("o", "h", "l", "c"):           # свічки OI (у монетах і в USD) для графіків
        h[f"oi_coins_{k}"] = coins[k].sum(axis=1, min_count=1)
        h[f"oi_usd_{k}"] = wide(part["oi"], k).sum(axis=1, min_count=1)
    h["oi_coins"] = h["oi_coins_c"]
    h["oi_coin_log_idx"] = chained_log_index(coins["c"])

    h["funding"] = weighted_mean(wide(part["funding"], "c"), oi_w)
    # години без ліквідацій відсутні в таблиці — це нуль, а не пропуск
    cov = raw.get("liq_coverage")
    if cov is None:                  # 1h: історія Coinalyze повна на всі 60 днів
        h["liq_long"] = wide(part["liq"], "long_liq").sum(axis=1).fillna(0)
        h["liq_short"] = wide(part["liq"], "short_liq").sum(axis=1).fillna(0)
    else:                            # 15m/5m: до covered_from ринку — невідомо (NaN), після — нуль, якщо рядка немає
        own = cov[cov["base_asset"] == base].set_index("market")["covered_from"]
        known = pd.DataFrame({m: grid >= t0 for m, t0 in own.items()}, index=grid)
        for k, col in (("liq_long", "long_liq"), ("liq_short", "short_liq")):
            w = wide(part["liq"], col).reindex(columns=own.index).fillna(0).where(known)
            h[k] = w.sum(axis=1, min_count=1)
    h["ls_long_pct"] = weighted_mean(wide(part["ls"], "long_pct"), oi_w)
    return h


# ---------------------------------------------------------------- ознаки

def zscore(s: pd.Series, window: int) -> pd.Series:
    """Наскільки поточне значення відхиляється від попередніх `window` годин (у стандартних відхиленнях).
    Вікно не включає поточну годину, тож сплеск не «розмиває» сам себе."""
    past = s.shift(1).rolling(window, min_periods=window)
    return (s - past.mean()) / past.std()


def compute_features(h: pd.DataFrame, bars_per_hour: int = 1) -> pd.DataFrame:
    """Ряд одного тікера → ознаки. Кожна ознака на бар t використовує лише дані до t.
    bars_per_hour > 1 — менший таймфрейм (4 = 15 хв): вікна множаться, тож «24h» і далі означає добу,
    а ознаки з «1h» у назві рахуються за останню годину, а не за один бар."""
    b = bars_per_hour
    short, base = SHORT_WINDOW * b, BASE_WINDOW * b
    f = pd.DataFrame(index=h.index)
    lp = np.log(h["price"])
    oi = h["oi_log_idx"]

    for k in (1, 4, 24):
        f[f"oi_chg_{k}h"] = oi.diff(k * b)           # лог-зміна OI за k годин
        f[f"ret_{k}h"] = lp.diff(k * b)              # лог-дохідність за k годин
    f["oi_z_24h"] = zscore(oi, short)
    f["oi_z_7d"] = zscore(oi, base)
    oc = h["oi_coin_log_idx"]
    for k in (1, 4, 24):
        f[f"oi_coin_chg_{k}h"] = oc.diff(k * b)
    f["oi_coin_z_24h"] = zscore(oc, short)
    f["oi_coin_z_7d"] = zscore(oc, base)

    f["volatility_24h"] = lp.diff().rolling(short).std()
    f["range_1h"] = (h["high"].rolling(b).max() - h["low"].rolling(b).min()) / h["price"]
    f["oi_minus_ret_24h"] = f["oi_chg_24h"] - f["ret_24h"]

    f["volume_z_7d"] = zscore(np.log1p(h["volume"]), base)
    day_vol = h["volume"].rolling(short).sum()
    f["volume_ratio_24h"] = day_vol / (h["volume"].shift(short).rolling(base).sum() / 7)
    f["buy_ratio_1h"] = h["buy_volume"].rolling(b).sum() / h["volume_bv"].rolling(b).sum()
    f["buy_ratio_4h"] = h["buy_volume"].rolling(4 * b).sum() / h["volume_bv"].rolling(4 * b).sum()

    f["funding"] = h["funding"]
    f["funding_z_7d"] = zscore(h["funding"], base)
    f["funding_chg_24h"] = h["funding"].diff(short)

    long4 = h["liq_long"].rolling(4 * b).sum()
    short4 = h["liq_short"].rolling(4 * b).sum()
    f["liq_long_4h_oi"] = long4 / h["oi"]
    f["liq_short_4h_oi"] = short4 / h["oi"]
    total4 = long4 + short4
    f["liq_imbalance_4h"] = ((short4 - long4) / total4).where(total4 != 0, 0.0)   # NaN (немає даних) лишається NaN

    f["ls_long_pct"] = h["ls_long_pct"]
    f["ls_chg_24h"] = h["ls_long_pct"].diff(short)

    # ділення на нуль (нульовий обсяг, стала серія) дає ±inf — для моделі це пропуск
    return f[ALL_FEATURES].replace([np.inf, -np.inf], np.nan)


# ---------------------------------------------------------------- мітки

BAR_1H = pd.Timedelta(hours=1)


def in_pump(ts, start, end, bar: pd.Timedelta = BAR_1H):
    """Чи належить бар (ts — час відкриття) інтервалу [start, end): так, якщо на момент ЗАКРИТТЯ бару памп уже
    триває — start < ts + bar ≤ end. Модель оцінює бар на закритті, тож мітка відповідає на те саме питання.
    Межі розмітки — моменти часу з будь-якою точністю (5 хв, 15 хв, година), а правило однакове для всіх таймфреймів.
    Для меж на рівних годинах і годинних барів це те саме, що start ≤ ts < end."""
    close = ts + bar
    return (close > start) & (close <= end)


def add_labels(df: pd.DataFrame, pumps: pd.DataFrame, base: str, bar: pd.Timedelta = BAR_1H) -> pd.DataFrame:
    """label = 1, якщо бар у пампі (див. in_pump); pattern_type — тип інтервалу, якщо він є."""
    df["label"] = 0
    df["pattern_type"] = None
    for p in pumps[pumps["base_asset"] == base].itertuples():
        inside = in_pump(df.index, p.start_ts, p.end_ts, bar)
        df.loc[inside, "pattern_type"] = p.pattern_type
        if p.pattern_type == "pump":
            df.loc[inside, "label"] = 1
    return df


# ---------------------------------------------------------------- збірка

def short_features(h: pd.DataFrame, b: int) -> pd.DataFrame:
    """Короткі ознаки 15m-бару (SHORT_FEATURES): лише дані до бару включно, відносні величини."""
    f = pd.DataFrame(index=h.index)
    lp, oi = np.log(h["price"]), h["oi_log_idx"]
    f["ret_bar"], f["ret_2bar"] = lp.diff(1), lp.diff(2)
    f["oi_chg_bar"], f["oi_chg_2bar"] = oi.diff(1), oi.diff(2)
    f["oi_accel_1h"] = oi.diff(b) - oi.diff(b).shift(b)          # остання година проти попередньої
    f["ret_accel_1h"] = lp.diff(b) - lp.diff(b).shift(b)
    f["vol_burst"] = h["volume"] / h["volume"].shift(1).rolling(24 * b).mean()   # бар проти середнього бару доби
    f["buy_ratio_bar"] = h["buy_volume"] / h["volume_bv"]
    f["range_bar"] = (h["high"] - h["low"]) / h["price"]
    f["liq_long_1h_oi"] = h["liq_long"].rolling(b).sum() / h["oi"]
    f["liq_short_1h_oi"] = h["liq_short"].rolling(b).sum() / h["oi"]
    return f.replace([np.inf, -np.inf], np.nan)


def load_alt_index(conn, since=None) -> pd.Series:
    """ALT-EW: середня log-доходність альтів на кожному 5m-барі (таблиця alt_index, індекс — відкриття бару) на повній
    5-хв сітці (пропуски — NaN). Таблиці немає / порожня — порожній ряд: ознаки власного руху будуть NaN."""
    try:
        rows = conn.execute("SELECT ts, ret FROM alt_index WHERE ts >= %s ORDER BY ts",
                            (since if since is not None else datetime(1970, 1, 1, tzinfo=timezone.utc),)).fetchall()
    except Exception:
        conn.rollback()
        rows = []
    if not rows:
        return pd.Series(dtype=float)
    s = pd.Series([r[1] for r in rows], index=pd.to_datetime([r[0] for r in rows], utc=True), dtype=float)
    return s.reindex(pd.date_range(s.index[0], s.index[-1], freq="5min"))


def alt_index_bars(idx5: pd.Series, opens: pd.DatetimeIndex, bar: pd.Timedelta) -> pd.Series:
    """Доходність індексу за бари довжини bar з відкриттями opens (будь-який зсув, кратний 5 хв: звичайні 15m-бари,
    ковзні :05 / :10, години) = сума 5m-доходностей усередині бару; бар з пропуском — NaN."""
    if idx5.empty:
        return pd.Series(np.nan, index=opens)
    k = int(bar / pd.Timedelta(minutes=5))
    s = idx5.rolling(k, min_periods=k).sum()
    s.index = s.index - (bar - pd.Timedelta(minutes=5))    # сума закінчується на останньому 5m-барі → відкриття бару
    return s.reindex(opens)


def move_features(h: pd.DataFrame, ix: pd.Series, b: int) -> pd.DataFrame:
    """Ознаки власного руху (MOVE_FEATURES) — код experiments/idio_features.py, вікна в барах через b (барів на годину).
    ix — доходність індексу за ті самі бари (alt_index_bars). Без погляду вперед: усі вікна закінчуються на барі, бета —
    ще й за добу до нього (сплеск не «вчить» бету сам себе). Відмінність від експерименту: illiq_x нормується на медіану
    7 днів, а не 30 — як інші «звичайні рівні» (BASE_WINDOW), і скринеру вистачає 15 днів історії на останній бар."""
    B, f = 24 * b, pd.DataFrame(index=h.index)
    W = 7 * B
    lr = np.log(h["price"].where(h["price"] > 0)).diff()
    ok = lr.notna() & ix.notna()
    ry, rx = lr.where(ok), ix.where(ok)
    mean = lambda s: s.rolling(W, min_periods=W // 2).mean()      # noqa: E731
    cov = (mean(ry * rx) - mean(ry) * mean(rx)).shift(B)
    vx = (mean(rx * rx) - mean(rx) ** 2).shift(B)
    vy = (mean(ry * ry) - mean(ry) ** 2).shift(B)
    beta = cov / vx.where(vx > 0)
    f["beta"] = beta
    f["r2"] = (cov ** 2 / (vx * vy)).where((vx > 0) & (vy > 0))
    resid = (vy - beta ** 2 * vx).clip(lower=1e-12)                # дисперсія залишку за бар
    for k, name in ((b, "idio_1h"), (4 * b, "idio_4h"), (B, "idio_24h")):
        f[name] = lr.rolling(k).sum() - beta * ix.rolling(k).sum()
    f["idio_z_4h"] = f["idio_4h"] / np.sqrt(resid * 4 * b)
    il = lr.abs() / h["volume"].where(h["volume"] > 0)            # Amihud: рух ціни на долар обороту
    roll = il.rolling(B, min_periods=int(0.75 * B)).mean()
    norm = roll.shift(B).rolling(W, min_periods=3 * B).median()
    f["illiq_x"] = np.log(roll / norm.where(norm > 0))
    hl = np.log(h["high"] / h["low"]) ** 2
    co = np.log(h["price"] / h["open"]) ** 2
    park = np.sqrt((hl / (4 * np.log(2))).rolling(B, min_periods=int(0.75 * B)).mean())
    gk = np.sqrt((0.5 * hl - (2 * np.log(2) - 1) * co).rolling(B, min_periods=int(0.75 * B)).mean().clip(lower=0))
    f["park_24h"], f["gk_24h"] = park, gk
    f["park_x"] = park / park.shift(B).rolling(W, min_periods=3 * B).median()
    f["ret_4h_vol"] = lr.rolling(4 * b).sum() / (park * np.sqrt(4 * b))
    f["ret_24h_vol"] = lr.rolling(B).sum() / (park * np.sqrt(B))
    return f.replace([np.inf, -np.inf], np.nan)


def intraday_extra(idx5: pd.Series):
    """extra для build_from_raw на 15m: короткі ознаки бару + власний рух відносно індексу idx5 (load_alt_index)."""
    def extra(h: pd.DataFrame, b: int) -> pd.DataFrame:
        return short_features(h, b).join(move_features(h, alt_index_bars(idx5, h.index, BAR_1H / b), b))
    return extra


def build_15m(conn, bases: list[str], pumps: pd.DataFrame | None = None, since=None) -> pd.DataFrame:
    """Ознаки 15m-моделі з бази (interval '15min': біржі напряму + ліквідації Coinalyze з покриттям) + індекс альтів.
    since — з якого моменту брати сирі ряди (для скринера: ~15 днів вистачає на тижневі вікна)."""
    return build_from_raw(load_raw(conn, bases, INTERVAL_15M, since), pumps, bars_per_hour=4,
                          extra=intraday_extra(load_alt_index(conn, since)))


def hourly_from_intraday(raw: dict[str, pd.DataFrame], offset_min: int = 0) -> dict[str, pd.DataFrame]:
    """15m-ряди контрактів (load_raw з INTERVAL_15M) → години, що починаються на :offset_min; лише повні години
    (4 бари). Так годинні ознаки рахуються з наших біржових даних без годинного збору Coinalyze:
    experiments/hourly_from_15m.py — рішення годинної моделі збігаються з Coinalyze на 98.6% годин."""
    return roll_intraday(raw, "h", offset_min, 4)


def roll_intraday(raw: dict[str, pd.DataFrame], freq: str, offset_min: int, need: int) -> dict[str, pd.DataFrame]:
    """Ряди контрактів дрібного ТФ → бари freq ("h", "15min"), що починаються зі зсувом offset_min хвилин; лише повні
    бари (need дрібних). Годинні з 15m (need=4) і ковзні 15m з 5m (need=3, зсув 5/10 хв — скринер кожні 5 хв)."""
    keys = ["base_asset", "market", "exchange", "quote_asset"]
    off = pd.Timedelta(minutes=offset_min)

    def roll(df, agg, extra=()):
        if df.empty:
            return df
        d = df.assign(ts=(df["ts"] - off).dt.floor(freq) + off)
        r = d.groupby(keys + list(extra) + ["ts"], dropna=False).agg(**agg, n=("ts", "size")).reset_index()
        return r[r["n"] == need].drop(columns="n")

    out = {
        "ohlcv": roll(raw["ohlcv"], dict(o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"), v=("v", "sum"),
                                         bv=("bv", lambda x: x.sum(min_count=1))), ["volume_denominated"]),
        "oi": roll(raw["oi"], dict(o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"))),
        "funding": roll(raw["funding"], dict(c=("c", "last"))),
        "ls": roll(raw["ls"], dict(long_pct=("long_pct", "last"))),
        "liq": roll(raw["liq"], dict(long_liq=("long_liq", "sum"), short_liq=("short_liq", "sum"))),
    }
    if "liq_coverage" in raw:
        out["liq_coverage"] = raw["liq_coverage"]
    return out


def build_hourly_own(conn, bases: list[str], since=None) -> pd.DataFrame:
    """Годинні ознаки (FEATURES) з наших 15m-даних (Binance+Bybit + 15m-ліквідації Coinalyze), без міток."""
    return build_from_raw(hourly_from_intraday(load_raw(conn, bases, INTERVAL_15M, since)))


def oi_seen_share(conn, bases: list[str], days: int = 30) -> dict[str, float]:
    """Частка OI тікера (середнє за days днів, годинні дані Coinalyze по всіх біржах), яку тримають біржі
    INTRADAY_EXCHANGES. Тікера без годинних даних Coinalyze у словнику немає."""
    rows = conn.execute("""
        SELECT base, sum(a) FILTER (WHERE ex = ANY(%s)) / NULLIF(sum(a), 0) FROM (
            SELECT m.base_asset AS base, m.exchange_name AS ex, o.symbol, avg(o.c) AS a
            FROM open_interest o JOIN markets m USING (symbol)
            WHERE m.base_asset = ANY(%s) AND o.interval = '1hour' AND o.ts > now() - make_interval(days => %s)
            GROUP BY 1, 2, 3) x GROUP BY 1""", (list(INTRADAY_EXCHANGES), bases, days)).fetchall()
    return {b: float(s) for b, s in rows if s is not None}


def build(conn, bases: list[str] | None = None, labels: bool = True) -> pd.DataFrame:
    """Таблиця ознак: рядок = тікер × година. Лише години з повною історією для вікон
    і лише закриті бари (поточна незакрита година відкидається)."""
    return build_from_raw(load_raw(conn, bases), load_pumps(conn) if labels else None)


def build_from_raw(raw: dict[str, pd.DataFrame], pumps: pd.DataFrame | None = None,
                   bars_per_hour: int = 1, extra=None) -> pd.DataFrame:
    """Те саме, що build(), але з уже завантажених сирих рядів у форматі load_raw()
    (так експерименти проганяють дані з інших джерел через той самий код ознак). pumps=None — без міток.
    bars_per_hour — таймфрейм сирих рядів (1 = 1h, 4 = 15 хв).
    extra(h, bars_per_hour) → DataFrame — додаткові ознаки з того самого зведеного ряду (для експериментів)."""
    labels = pumps is not None
    last_closed = pd.Timestamp(datetime.now(timezone.utc) - timedelta(hours=1) / bars_per_hour)
    frames = []
    for base in sorted(raw["ohlcv"]["base_asset"].unique()):
        h = aggregate(raw, base, pd.tseries.frequencies.to_offset(pd.Timedelta(hours=1) / bars_per_hour))
        if h.empty:
            continue
        f = compute_features(h, bars_per_hour)
        if extra is not None:
            f = f.join(extra(h, bars_per_hour))
        f = f[f["oi_z_7d"].notna() & (f.index <= last_closed)]
        f.insert(0, "price", h["price"])          # контекст для аналізу, не ознаки
        f.insert(1, "oi", h["oi"])
        if labels:
            f = add_labels(f, pumps, base, BAR_1H / bars_per_hour)
        f.insert(0, "base_asset", base)
        frames.append(f.reset_index())
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def save(conn, df: pd.DataFrame, bases: list[str] | None) -> None:
    """Таблиця features повністю похідна від сирих даних і pumps, тому створюється тут, а не в schema.sql.
    Зміна набору ознак = перерахунок: python features.py."""
    num_cols = ["price", "oi"] + ALL_FEATURES   # у таблиці — усі ознаки, модель бере FEATURES
    cols = ["base_asset", "ts"] + num_cols + ["label", "pattern_type", "reviewed"]
    if bases is None:
        conn.execute("DROP TABLE IF EXISTS features")
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS features (
            base_asset TEXT NOT NULL, ts TIMESTAMPTZ NOT NULL,
            {", ".join(f"{c} DOUBLE PRECISION" for c in num_cols)},
            label SMALLINT NOT NULL, pattern_type TEXT, reviewed BOOLEAN NOT NULL,
            PRIMARY KEY (base_asset, ts))""")
    if bases:
        conn.execute("DELETE FROM features WHERE base_asset = ANY(%s)", (bases,))
    out = df[cols].astype(object).where(df[cols].notna(), None)
    with conn.cursor().copy(f"COPY features ({', '.join(cols)}) FROM STDIN") as cp:
        for row in out.itertuples(index=False):
            cp.write_row(row)
    conn.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Розрахунок ознак і запис у таблицю features")
    parser.add_argument("tickers", nargs="*", help=f"базові активи (без аргументів — з {TICKERS_FILE})")
    args = parser.parse_args()

    load_dotenv()
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        sys.exit("Заповни DATABASE_URL у файлі .env")
    full = not args.tickers
    reviewed = {t.upper() for t in read_tickers_file(TICKERS_FILE)}

    with psycopg.connect(db_url) as conn:
        labeled = {r[0] for r in conn.execute("SELECT DISTINCT base_asset FROM pumps")}
        # без аргументів — переглянуті тікери і всі тікери з розміткою
        from fetch import excluded_tickers
        excluded = excluded_tickers()        # поза скоупом — навіть якщо розмічені
        reviewed, labeled = reviewed - excluded, labeled - excluded
        bases = [t.upper() for t in args.tickers] if args.tickers else sorted(reviewed | labeled)
        if not bases:
            sys.exit(f"Немає тікерів: передай їх аргументами або додай у {TICKERS_FILE}")
        df = build(conn, bases)
        if df.empty:
            sys.exit("Немає даних для ознак.")
        df["reviewed"] = df["base_asset"].isin(reviewed)
        # тікер не переглянуто повністю — лише розмічені інтервали: решта годин не «без пампу», а «невідомо»
        df = df[df["reviewed"] | df["pattern_type"].notna()].reset_index(drop=True)
        partial = sorted(set(df.loc[~df["reviewed"], "base_asset"]))
        if partial:
            print(f"Лише розмічені інтервали (тікер не в {TICKERS_FILE}): {', '.join(partial)}")
        lost = sorted(labeled - set(df["base_asset"]))
        if lost and not args.tickers:
            print(f"Розмітка без ознак (мало історії в базі — потрібно 7 днів до інтервалу): {', '.join(lost)}")
        save(conn, df, None if full else bases)

    print(f"Ознак: {len(FEATURES)}, рядків: {len(df)}\n")
    summary = df.groupby("base_asset").agg(
        hours=("ts", "size"), first=("ts", "min"), last=("ts", "max"), pump_hours=("label", "sum"))
    summary["first"] = summary["first"].dt.strftime("%m-%d %H:%M")
    summary["last"] = summary["last"].dt.strftime("%m-%d %H:%M")
    print(summary.to_string())
    print(f"\nЧастка годин пампу: {df['label'].mean():.1%}")
    missing = df[FEATURES].isna().mean().sort_values(ascending=False)
    missing = missing[missing > 0]
    if len(missing):
        print("\nПропуски в ознаках (частка рядків):")
        print(missing.map("{:.1%}".format).to_string())


if __name__ == "__main__":
    main()
