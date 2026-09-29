"""
screener.py — щогодинний скринер пампів.

Цикл:
    1. збір свіжих даних Coinalyze для тікерів з watchlist.txt (останні 10 днів — вистачає на вікна ознак)
    2. ознаки через features.build() — той самий код, що й для навчання
    3. ймовірність пампу від кожної збереженої моделі (models/*.joblib, пише train.py)
    4. запис у таблицю signals і короткий звіт у консоль

    python screener.py              # один цикл
    python screener.py --loop       # щогодини о HH:05 UTC (бар щойно закрився і з'явився в Coinalyze)
    python screener.py --no-fetch   # без збору даних: оцінити те, що вже є в базі

Ліміт Coinalyze — 40 символів на хвилину, тож цикл по ~100 тікерах триває кілька хвилин.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import psycopg
from dotenv import load_dotenv

from features import build
from fetch import Coinalyze, collect, read_tickers_file
from train import PROBA_MODELS, load_model, model_path

WATCHLIST_FILE = "watchlist.txt"
STATUS_FILE = os.path.join("reports", "screener_status.json")   # читає дашборд, блок «Стан системи»
FETCH_DAYS = 10          # 7 днів базового вікна + доба + запас
BACKFILL_HOURS = 7 * 24  # скільки годин історії оцінювати, якщо їх ще немає в signals
RUN_MINUTE = 5           # хвилина кожної години, коли запускати цикл


def load_models() -> list[dict]:
    models = []
    for name in PROBA_MODELS:
        if os.path.exists(model_path(name)):
            models.append(load_model(name))
    if not models:
        sys.exit("Немає збережених моделей у models/ — спершу python train.py")
    return models


def score(df: pd.DataFrame, models: list[dict]) -> pd.DataFrame:
    """Ймовірності всіх моделей для кожного рядка ознак."""
    out = []
    for m in models:
        p = m["model"].predict_proba(df[m["features"]])[:, 1]
        # in_sample: ця година цього тікера була в навчанні моделі — оцінка не є чесним передбаченням
        seen = df["base_asset"].isin(m["tickers"]).to_numpy() & (df["ts"] <= m["data_end"]).to_numpy()
        for base, a, b in m.get("in_sample_extra", []):     # свіжі розмічені інтервали теж були в навчанні
            seen |= ((df["base_asset"] == base) & (df["ts"] >= a) & (df["ts"] <= b)).to_numpy()
        out.append(df[["base_asset", "ts"]].reset_index(drop=True).assign(
            model=m["name"], model_version=m["trained_at"],
            probability=p, threshold=m["threshold"], signal=p >= m["threshold"], in_sample=seen,
        ))
    return pd.concat(out, ignore_index=True)


def save_signals(conn, scored: pd.DataFrame, last_closed: pd.Timestamp) -> int:
    """Нові оцінки → signals. Уже записані години не чіпаємо (ON CONFLICT DO NOTHING)."""
    # live — чесна оцінка в реальному часі: остання закрита година, якої модель не бачила
    scored = scored.assign(live=(scored["ts"] == last_closed) & ~scored["in_sample"])
    rows = [(r.base_asset, r.ts.to_pydatetime(), r.model, r.model_version, float(r.probability),
             float(r.threshold), bool(r.signal), bool(r.live), bool(r.in_sample)) for r in scored.itertuples()]
    cur = conn.cursor()
    cur.executemany("""
        INSERT INTO signals (base_asset, ts, model, model_version, probability, threshold, signal, live, in_sample)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (base_asset, ts, model, model_version) DO NOTHING""", rows)
    conn.commit()
    return len(rows)


def score_bases(conn, bases: list[str], models: list[dict]) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Ознаки → ймовірності → signals для тікерів. Спільне для скринера і кнопки «Прогнозувати» в дашборді."""
    # остання закрита година — на момент оцінки, а не старту: збір може тривати кілька хвилин і перейти годину
    now = datetime.now(timezone.utc)
    last_closed = pd.Timestamp(now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1))
    df = build(conn, bases, labels=False)
    if df.empty:
        return pd.DataFrame(), last_closed
    df = df[df["ts"] > last_closed - pd.Timedelta(hours=BACKFILL_HOURS)]
    scored = score(df, models)
    save_signals(conn, scored, last_closed)
    return scored, last_closed


def write_status(**fields) -> None:
    os.makedirs(os.path.dirname(STATUS_FILE), exist_ok=True)
    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump({"updated": datetime.now(timezone.utc).isoformat(timespec="seconds"), **fields}, f, ensure_ascii=False)


def run_cycle(conn, bases: list[str], models: list[dict], fetch: bool, client: Coinalyze | None) -> None:
    started = datetime.now(timezone.utc)
    calls0, limited0 = (client.calls_made, client.rate_limited) if client else (0, 0)
    print(f"\n[{started:%Y-%m-%d %H:%M} UTC] цикл: {len(bases)} тікерів")

    if fetch:
        collect(client, conn, bases, FETCH_DAYS, verbose=False)
        print(f"  дані оновлено за {(datetime.now(timezone.utc) - started).seconds} с")
    scored, last_closed = score_bases(conn, bases, models)
    print(f"  остання закрита година: {last_closed:%m-%d %H:%M} UTC")
    if scored.empty:
        print("  немає тікерів з повною історією для ознак (потрібно ≥ 7 днів)")
        return

    # звіт: остання година по кожному тікеру
    latest = scored[scored["ts"] == scored.groupby("base_asset")["ts"].transform("max")]
    table = latest.pivot_table(index="base_asset", columns="model", values="probability")
    table["signal"] = latest.groupby("base_asset")["signal"].any()
    table["in_sample"] = latest.groupby("base_asset")["in_sample"].any()
    last_ts = latest.groupby("base_asset")["ts"].max()
    table["last_bar"] = last_ts.dt.strftime("%m-%d %H:%M")
    table = table.sort_values(table.columns[0], ascending=False)
    stale = table[last_ts.reindex(table.index) < last_closed]
    missing = sorted(set(bases) - set(table.index))

    active = table[table["signal"]]
    print(f"  активних сигналів: {len(active)}" + (f" — {', '.join(active.index)}" if len(active) else ""))
    print(table.head(10).round(3).to_string())
    if len(stale):
        print(f"  ! дані відстають (немає останньої години): {', '.join(stale.index)}")
    if missing:
        print(f"  ! без оцінки (мало історії або не знайдено): {', '.join(missing)}")
    write_status(ok=True, cycle_started=started.isoformat(timespec="seconds"),
                 seconds=int((datetime.now(timezone.utc) - started).total_seconds()),
                 tickers=len(bases), scored=len(table), stale=list(stale.index), missing=missing,
                 api_calls=(client.calls_made - calls0) if client else 0,
                 api_429=(client.rate_limited - limited0) if client else 0, fetched=fetch)


def main() -> None:
    parser = argparse.ArgumentParser(description="Щогодинний скринер пампів")
    parser.add_argument("--loop", action="store_true", help=f"працювати безперервно, щогодини о HH:{RUN_MINUTE:02d} UTC")
    parser.add_argument("--no-fetch", action="store_true", help="не збирати дані, оцінити наявні")
    args = parser.parse_args()

    load_dotenv()
    api_key, db_url = os.getenv("COINALYZE_API_KEY"), os.getenv("DATABASE_URL")
    if not db_url or (not api_key and not args.no_fetch):
        sys.exit("Заповни COINALYZE_API_KEY і DATABASE_URL у файлі .env")
    bases = [t.upper() for t in read_tickers_file(WATCHLIST_FILE)]
    if not bases:
        sys.exit(f"Додай тікери в {WATCHLIST_FILE} (один на рядок)")

    models = load_models()
    for m in models:
        print(f"Модель {m['name']}: навчена {m['trained_at']}, поріг {m['threshold']:.3f}, "
              f"{m['pump_intervals']} пампів на {len(m['tickers'])} тікерах")
    client = None if args.no_fetch else Coinalyze(api_key)

    while True:
        try:
            with psycopg.connect(db_url) as conn:
                run_cycle(conn, bases, models, not args.no_fetch, client)
        except Exception as e:  # у режимі --loop одна невдала година не має зупиняти скринер
            write_status(ok=False, error=f"{type(e).__name__}: {e}")
            if not args.loop:
                raise
            print(f"  ! помилка циклу: {e}")
        if not args.loop:
            break
        now = datetime.now(timezone.utc)
        nxt = now.replace(minute=RUN_MINUTE, second=0, microsecond=0)
        if nxt <= now:
            nxt += timedelta(hours=1)
        print(f"  наступний цикл о {nxt:%H:%M} UTC")
        time.sleep((nxt - now).total_seconds())


if __name__ == "__main__":
    main()
