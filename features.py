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
    "funding", "funding_z_7d", "funding_chg_24h", "pred_funding_gap",
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
        "pred_funding": q("predicted_funding_rate", "t.c"),
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


def aggregate(raw: dict[str, pd.DataFrame], base: str) -> pd.DataFrame:
    """Усі біржі тікера → один погодинний ряд на регулярній сітці годин."""
    part = {k: v[v["base_asset"] == base] for k, v in raw.items()}
    ohlcv = part["ohlcv"].copy()
    if ohlcv.empty:
        return pd.DataFrame()

    grid = pd.date_range(ohlcv["ts"].min(), ohlcv["ts"].max(), freq="h")

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
    h["pred_funding"] = weighted_mean(wide(part["pred_funding"], "c"), oi_w)
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


def compute_features(h: pd.DataFrame) -> pd.DataFrame:
    """Погодинний ряд одного тікера → ознаки. Кожна ознака на годину t використовує лише дані до t."""
    f = pd.DataFrame(index=h.index)
    lp = np.log(h["price"])
    oi = h["oi_log_idx"]

    for k in (1, 4, 24):
        f[f"oi_chg_{k}h"] = oi.diff(k)           # лог-зміна OI за k годин
        f[f"ret_{k}h"] = lp.diff(k)              # лог-дохідність за k годин
    f["oi_z_24h"] = zscore(oi, SHORT_WINDOW)
    f["oi_z_7d"] = zscore(oi, BASE_WINDOW)
    oc = h["oi_coin_log_idx"]
    for k in (1, 4, 24):
        f[f"oi_coin_chg_{k}h"] = oc.diff(k)
    f["oi_coin_z_24h"] = zscore(oc, SHORT_WINDOW)
    f["oi_coin_z_7d"] = zscore(oc, BASE_WINDOW)

    f["volatility_24h"] = lp.diff().rolling(SHORT_WINDOW).std()
    f["range_1h"] = (h["high"] - h["low"]) / h["price"]
    f["oi_minus_ret_24h"] = f["oi_chg_24h"] - f["ret_24h"]

    f["volume_z_7d"] = zscore(np.log1p(h["volume"]), BASE_WINDOW)
    day_vol = h["volume"].rolling(SHORT_WINDOW).sum()
    f["volume_ratio_24h"] = day_vol / (h["volume"].shift(SHORT_WINDOW).rolling(BASE_WINDOW).sum() / 7)
    f["buy_ratio_1h"] = h["buy_volume"] / h["volume"]
    f["buy_ratio_4h"] = h["buy_volume"].rolling(4).sum() / h["volume"].rolling(4).sum()

    f["funding"] = h["funding"]
    f["funding_z_7d"] = zscore(h["funding"], BASE_WINDOW)
    f["funding_chg_24h"] = h["funding"].diff(SHORT_WINDOW)
    f["pred_funding_gap"] = h["pred_funding"] - h["funding"]

    long4 = h["liq_long"].rolling(4).sum()
    short4 = h["liq_short"].rolling(4).sum()
    f["liq_long_4h_oi"] = long4 / h["oi"]
    f["liq_short_4h_oi"] = short4 / h["oi"]
    total4 = long4 + short4
    f["liq_imbalance_4h"] = ((short4 - long4) / total4).where(total4 > 0, 0.0)

    f["ls_long_pct"] = h["ls_long_pct"]
    f["ls_chg_24h"] = h["ls_long_pct"].diff(SHORT_WINDOW)

    # ділення на нуль (нульовий обсяг, стала серія) дає ±inf — для моделі це пропуск
    return f[ALL_FEATURES].replace([np.inf, -np.inf], np.nan)


# ---------------------------------------------------------------- мітки

def add_labels(df: pd.DataFrame, pumps: pd.DataFrame, base: str) -> pd.DataFrame:
    """label = 1, якщо година всередині інтервалу pump; pattern_type — тип інтервалу, якщо він є."""
    df["label"] = 0
    df["pattern_type"] = None
    for p in pumps[pumps["base_asset"] == base].itertuples():
        inside = (df.index >= p.start_ts) & (df.index < p.end_ts)   # end_ts у БД — виключна межа
        df.loc[inside, "pattern_type"] = p.pattern_type
        if p.pattern_type == "pump":
            df.loc[inside, "label"] = 1
    return df


# ---------------------------------------------------------------- збірка

def build(conn, bases: list[str] | None = None, labels: bool = True) -> pd.DataFrame:
    """Таблиця ознак: рядок = тікер × година. Лише години з повною історією для вікон
    і лише закриті бари (поточна незакрита година відкидається)."""
    raw = load_raw(conn, bases)
    pumps = load_pumps(conn) if labels else None
    last_closed = pd.Timestamp(datetime.now(timezone.utc) - timedelta(hours=1))
    frames = []
    for base in sorted(raw["ohlcv"]["base_asset"].unique()):
        h = aggregate(raw, base)
        if h.empty:
            continue
        f = compute_features(h)
        f = f[f["oi_z_7d"].notna() & (f.index <= last_closed)]
        f.insert(0, "price", h["price"])          # контекст для аналізу, не ознаки
        f.insert(1, "oi", h["oi"])
        if labels:
            f = add_labels(f, pumps, base)
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
