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

Таблиця features — навчальна вибірка, тому за замовчуванням лише tickers.txt (розмічені тікери).
Тікери зі скринера (watchlist.txt) теж лежать у базі, але не розмічені: якби вони потрапили в навчання,
модель вважала б, що пампів на них не було.
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


# ---------------------------------------------------------------- завантаження

def query_df(conn, sql: str, params=()) -> pd.DataFrame:
    cur = conn.execute(sql, params)
    df = pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description])
    if "ts" in df:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def load_raw(conn, bases: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Сирі ряди з назвою тікера і біржі в кожному рядку."""
    cond = "t.interval = %s" + (" AND m.base_asset = ANY(%s)" if bases else "")
    params = (INTERVAL, bases) if bases else (INTERVAL,)

    def q(table: str, cols: str) -> pd.DataFrame:
        # market — конкретний контракт (на одній біржі їх буває кілька: USDT і USDC)
        return query_df(conn, f"""
            SELECT m.base_asset, t.symbol AS market, m.exchange_name AS exchange, m.quote_asset, t.ts, {cols}
            FROM {table} t JOIN markets m USING (symbol)
            WHERE {cond}""", params)

    return {
        "ohlcv": q("ohlcv", "t.o, t.h, t.l, t.c, t.v, t.bv, m.volume_denominated"),
        "oi": q("open_interest", "t.o, t.h, t.l, t.c"),
        "funding": q("funding_rate", "t.c"),
        "liq": q("liquidations", "t.long_liq, t.short_liq"),
        "ls": q("long_short_ratio", "t.long_pct"),
    }


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
    які мали значення і годину тому, і зараз. Тоді поява нової біржі (лістинг) не виглядає як ріст OI."""
    prev = w.shift(1)
    both = w.notna() & prev.notna()
    step = np.log(w.where(both).sum(axis=1, min_count=1) / prev.where(both).sum(axis=1, min_count=1))
    return step.fillna(0).cumsum().where(w.notna().any(axis=1))


def aggregate(raw: dict[str, pd.DataFrame], base: str, freq: str = "h") -> pd.DataFrame:
    """Усі біржі тікера → один ряд на регулярній сітці (freq: "h" — години; "15min" — для експериментів)."""
    part = {k: v[v["base_asset"] == base] for k, v in raw.items()}
    ohlcv = part["ohlcv"].copy()
    if ohlcv.empty:
        return pd.DataFrame()

    grid = pd.date_range(ohlcv["ts"].min(), ohlcv["ts"].max(), freq=freq)

    def wide(df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Рядки — години, колонки — контракти."""
        if df.empty:
            return pd.DataFrame(index=grid)
        return df.pivot_table(index="ts", columns="market", values=col, aggfunc="last").reindex(grid)

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
    # OI у монетах. Coinalyze переводить OI у USD за ціною закриття години — усі o/h/l/c свічки одразу.
    # Тому в USD OI «стрибає» між годинами разом із ціною, хоча позиції не змінювались.
    # Ділимо на close того самого контракту: отримуємо чисту кількість відкритих позицій у монетах
    # (розрив між годинами ≈ 0.01% замість 0.5–2.6%). Сума по біржах коректна: одиниця — монета.
    close_w = wide(ohlcv, "c").reindex(columns=oi_w.columns).ffill(limit=2)
    coins = {k: wide(part["oi"], k) / close_w for k in ("o", "h", "l", "c")}
    h = pd.DataFrame(index=grid)
    h.index.name = "ts"
    h["price"] = px["c"].ffill(limit=2)          # поодинокі пропуски заповнюємо попереднім значенням
    h["open"] = px["o"]                           # для свічок у дашборді; в ознаках не використовується
    h["high"] = px["h"]
    h["low"] = px["l"]
    h["volume"] = wide(ohlcv, "v_usd").sum(axis=1, min_count=1)
    h["buy_volume"] = wide(ohlcv, "bv_usd").sum(axis=1, min_count=1)
    h["oi"] = oi_w.sum(axis=1, min_count=1)
    h["oi_log_idx"] = chained_log_index(oi_w)
    for k in ("o", "h", "l", "c"):           # свічки OI (у монетах і в USD) для графіків
        h[f"oi_coins_{k}"] = coins[k].sum(axis=1, min_count=1)
        h[f"oi_usd_{k}"] = wide(part["oi"], k).sum(axis=1, min_count=1)
    h["oi_coins"] = h["oi_coins_c"]
    h["oi_coin_log_idx"] = chained_log_index(coins["c"])

    h["funding"] = weighted_mean(wide(part["funding"], "c"), oi_w)
    # години без ліквідацій відсутні в таблиці — це нуль, а не пропуск
    h["liq_long"] = wide(part["liq"], "long_liq").sum(axis=1).fillna(0)
    h["liq_short"] = wide(part["liq"], "short_liq").sum(axis=1).fillna(0)
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
    f["buy_ratio_1h"] = h["buy_volume"].rolling(b).sum() / h["volume"].rolling(b).sum()
    f["buy_ratio_4h"] = h["buy_volume"].rolling(4 * b).sum() / h["volume"].rolling(4 * b).sum()

    f["funding"] = h["funding"]
    f["funding_z_7d"] = zscore(h["funding"], base)
    f["funding_chg_24h"] = h["funding"].diff(short)

    long4 = h["liq_long"].rolling(4 * b).sum()
    short4 = h["liq_short"].rolling(4 * b).sum()
    f["liq_long_4h_oi"] = long4 / h["oi"]
    f["liq_short_4h_oi"] = short4 / h["oi"]
    total4 = long4 + short4
    f["liq_imbalance_4h"] = ((short4 - long4) / total4).where(total4 > 0, 0.0)

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

def build(conn, bases: list[str] | None = None, labels: bool = True) -> pd.DataFrame:
    """Таблиця ознак: рядок = тікер × година. Лише години з повною історією для вікон
    і лише закриті бари (поточна незакрита година відкидається)."""
    return build_from_raw(load_raw(conn, bases), load_pumps(conn) if labels else None)


def build_from_raw(raw: dict[str, pd.DataFrame], pumps: pd.DataFrame | None = None,
                   bars_per_hour: int = 1) -> pd.DataFrame:
    """Те саме, що build(), але з уже завантажених сирих рядів у форматі load_raw()
    (так експерименти проганяють дані з інших джерел через той самий код ознак). pumps=None — без міток.
    bars_per_hour — таймфрейм сирих рядів (1 = 1h, 4 = 15 хв)."""
    labels = pumps is not None
    last_closed = pd.Timestamp(datetime.now(timezone.utc) - timedelta(hours=1) / bars_per_hour)
    frames = []
    for base in sorted(raw["ohlcv"]["base_asset"].unique()):
        h = aggregate(raw, base, pd.tseries.frequencies.to_offset(pd.Timedelta(hours=1) / bars_per_hour))
        if h.empty:
            continue
        f = compute_features(h, bars_per_hour)
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
    cols = ["base_asset", "ts"] + num_cols + ["label", "pattern_type"]
    if bases is None:
        conn.execute("DROP TABLE IF EXISTS features")
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS features (
            base_asset TEXT NOT NULL, ts TIMESTAMPTZ NOT NULL,
            {", ".join(f"{c} DOUBLE PRECISION" for c in num_cols)},
            label SMALLINT NOT NULL, pattern_type TEXT,
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
    bases = [t.upper() for t in (args.tickers or read_tickers_file(TICKERS_FILE))]
    if not bases:
        sys.exit(f"Немає тікерів: передай їх аргументами або додай у {TICKERS_FILE}")

    with psycopg.connect(db_url) as conn:
        df = build(conn, bases)
        if df.empty:
            sys.exit("Немає даних для ознак.")
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
