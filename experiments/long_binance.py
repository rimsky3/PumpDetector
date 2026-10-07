"""
experiments/long_binance.py — 6-місячний бектест на архіві Binance з розбивкою по місяцях (2026-10-07).

Навіщо: усі попередні перевірки ознак — один місяць бектесту, ефекти ±0.05 PF не відрізнити від шуму. Архів Binance
(binance_archive.py: свічки з обсягом покупок, OI і частка лонгів; orderbook.archive_day: смуги стакану) дає ті самі
ознаки, що й наживо, за будь-який період. experiments/binance_only.py: модель «лише Binance» ≈ двом біржам, тож
порівняння варіантів на Binance-даних чесне. Критерій — «варіант кращий у більшості місяців», а не загальний PF.

Монети — ті самі, що в довгому бектесті (data/long_bt: USDT-перпи Binance, оборот ≥ 5 млн $, OI ≤ 500 млн, без навчальних
і виключених), період 01.04–30.09 (+8 днів на тижневі вікна). Етапи (кожен з кешем — повторний запуск продовжує):
    data      binance_archive.collect → сирі таблиці бази (лише Binance; позначка data/long_bin/_data_done.txt);
              паралельно — архів стакану (кеш data/book_cache)
    features  15m-ознаки (MODEL_FEATURES_15M) + ознаки стакану (book_features.py) + годинні з власних 15m + ціни
              → data/long_bin/<монета>.pkl
    test      15m-моделі base / book на навчальних тікерах (лише Binance, як binance_only.py); годинне підтвердження —
              робоча годинна модель на власних 15m Binance; угоди Alert, частковий вихід (½ на +3 ATR, б/з), стоп 2.48 ATR
              → reports/long_binance.md, reports/long_binance_trades.csv
Нова ідея ознак = нова функція в FEATURE_SETS (рахується на кешованих рядах, без повторного завантаження).

    python experiments/long_binance.py                   # усе
    python experiments/long_binance.py --stage test      # лише моделі й угоди (дані й ознаки — з кешу)
"""

import argparse
import glob
import os
import pickle
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd
import psycopg
from dotenv import load_dotenv

import binance_archive as BA
import book_features as BF
import exit_partial as P
import train
from features import (FEATURES, INTERVAL_15M, MODEL_FEATURES_15M, aggregate, build_from_raw, hourly_from_intraday, load_pumps,
                      load_raw, query_df, short_features)
from fetch import excluded_tickers, read_tickers_file
from tiers import tier_frame

START, END = "2026-04-01", "2026-09-30"
WARM = "2026-03-24"                       # тижневі вікна ознак
DIR = os.path.join("data", "long_bin")
DONE = os.path.join(DIR, "_data_done.txt")
K = 2.48
BIN = ["Binance"]
OUT = os.path.join("reports", "long_binance.md")
TRADES = os.path.join("reports", "long_binance_trades.csv")
VARIANTS = {"base": MODEL_FEATURES_15M, "book": MODEL_FEATURES_15M + BF.BOOK}


def coins_and_bands(training: set) -> tuple[list[str], dict]:
    coins = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join("data", "long_bt", "*.pkl")))
    band = {}
    lt = os.path.join("reports", "long_backtest_trades.csv")
    if os.path.exists(lt):
        band = pd.read_csv(lt, usecols=["ticker", "band"]).drop_duplicates("ticker").set_index("ticker")["band"].to_dict()
    ex = excluded_tickers(".")
    return [c for c in coins if c not in training and c not in ex], band


def stage_data(conn, coins: list[str], syms: dict) -> None:
    os.makedirs(DIR, exist_ok=True)
    done = set(open(DONE, encoding="utf-8").read().split()) if os.path.exists(DONE) else set()
    todo = [c for c in coins if c not in done and c in syms]
    # стакан — паралельно (інший тип файлів, той самий сайт)
    jobs = [(syms[c], d) for c in coins if c in syms for d in BA._days(WARM, END)]
    book = threading.Thread(target=BF.fetch_days, args=(jobs,), daemon=True)
    book.start()
    t0 = time.time()
    for n, c in enumerate(todo, 1):
        try:
            BA.collect(conn, [c], WARM, END, ("15m",), log=lambda m: None)
            with open(DONE, "a", encoding="utf-8") as f:
                f.write(c + "\n")
        except Exception as e:
            conn.rollback()
            print(f"  ! {c}: {type(e).__name__}: {e}", flush=True)
        if n % 10 == 0:
            print(f"  дані: {n}/{len(todo)} монет, {time.time() - t0:.0f} с", flush=True)
    print("  чекаю завантаження стакану…", flush=True)
    book.join()


def stage_features(conn, coins: list[str], syms: dict) -> None:
    since = pd.Timestamp(WARM, tz="UTC")
    days = BA._days(WARM, END)
    for n, c in enumerate(coins, 1):
        path = os.path.join(DIR, f"{c}.pkl")
        if os.path.exists(path) or c not in syms:
            continue
        raw = load_raw(conn, [c], INTERVAL_15M, since=since, exchanges=BIN)
        if raw["ohlcv"].empty:
            continue
        f15 = build_from_raw(raw, None, bars_per_hour=4, extra=short_features)
        h1 = build_from_raw(hourly_from_intraday(raw))
        h = aggregate(raw, c, "15min")
        if f15.empty or h1.empty:
            continue
        bk = BF.book_features(h, BF.book_wide(syms[c], days)).reindex(f15["ts"]).set_axis(f15.index)
        f15 = f15[["base_asset", "ts", *MODEL_FEATURES_15M]].join(bk)
        with open(path, "wb") as f:
            pickle.dump({"f15": f15, "h1": h1[["base_asset", "ts", *FEATURES]],
                         "h": h[["open", "high", "low", "price", "volume"]]}, f)
        if n % 20 == 0:
            print(f"  ознаки: {n}/{len(coins)}", flush=True)


def train_models(conn, excluded: set) -> tuple[dict, set]:
    """15m-моделі варіантів на навчальних тікерах — лише Binance, з ознаками стакану (як book_features.py)."""
    reviewed = {t.upper() for t in read_tickers_file("tickers.txt")} - excluded
    pumps = load_pumps(conn)
    pumps = pumps[~pumps["base_asset"].isin(excluded)]
    bases = sorted(reviewed | set(pumps["base_asset"]))
    raw = load_raw(conn, bases, INTERVAL_15M, exchanges=BIN)
    d = build_from_raw(raw, pumps, bars_per_hour=4, extra=short_features)
    first1 = query_df(conn, "SELECT base_asset, min(ts) AS t FROM features GROUP BY 1").set_index("base_asset")["t"]
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=48)
    d = d[d["base_asset"].isin(reviewed) | d["pattern_type"].notna()]
    d = d[(d["ts"] <= cutoff) | d["pattern_type"].notna()]
    d = d[d["ts"] >= d["base_asset"].map(first1)].reset_index(drop=True)
    syms = BF.binance_symbols(conn, bases)
    BF.fetch_days(sorted({(syms[b], x.strftime("%Y-%m-%d")) for b, g in d.groupby("base_asset") if b in syms
                         for x in pd.date_range((g["ts"].min() - pd.Timedelta(days=8)).normalize(), min(g["ts"].max().normalize(), BF.LAST_DAY))}))
    d = BF.add_book(None, d, raw, syms)
    y = d["label"].astype(int)
    models = {}
    for name, cols in VARIANTS.items():
        p = train.oof_proba(train.make_lightgbm, d[cols], y, d["base_asset"])
        models[name] = (train.make_lightgbm().fit(d[cols], y), train.best_f1_threshold(y.to_numpy(), p), cols)
        print(f"  модель {name}: {len(cols)} ознак, поріг {models[name][1]:.3f}", flush=True)
    return models, set(d["base_asset"])


def pf(v: pd.Series) -> float:
    gl = -v[v <= 0].sum()
    return v[v > 0].sum() / gl if gl else np.nan


def stage_test(conn, coins: list[str], band: dict, excluded: set) -> None:
    models, trained = train_models(conn, excluded)
    m1 = joblib.load(train.model_path("lightgbm"))
    m15 = joblib.load(train.model_path(train.M15_NAME))
    exit_bars, share = m15.get("exit_bars", 4), m15.get("confirm", {}).get("share", 0.75)
    start, end = pd.Timestamp(START, tz="UTC"), pd.Timestamp(END, tz="UTC") + pd.Timedelta(days=1)
    rows = []
    for c in coins:
        path = os.path.join(DIR, f"{c}.pkl")
        if c in trained or not os.path.exists(path):
            continue
        x = pickle.load(open(path, "rb"))
        f15, h1, h = x["f15"], x["h1"], x["h"]
        h = h[(h.index >= start - pd.Timedelta(days=1)) & (h.index < end)]
        s1 = h1[["base_asset", "ts"]].assign(probability=m1["model"].predict_proba(h1[m1["features"]])[:, 1], threshold=m1["threshold"])
        for name, (mo, thr, cols) in models.items():
            s15 = f15[["base_asset", "ts"]].assign(probability=mo.predict_proba(f15[cols])[:, 1], threshold=thr)
            t = tier_frame(s15, s1, share).set_index("ts").sort_index()
            sc = t.loc[(t.index >= start) & (t.index < end), "alert_score"]
            rows += [{"variant": name, "ticker": c, "band": band.get(c, ""), **r} for r in P.simulate(h, sc, K, exit_bars, 3.0, 0.5, True)]
    tr = pd.DataFrame(rows)
    tr.to_csv(TRADES, index=False)
    tr["month"] = pd.to_datetime(tr["ts"], utc=True).dt.strftime("%Y-%m")
    lines = ["# 6 місяців на архіві Binance: стакан у 15m-моделі", "",
             f"{tr['ticker'].nunique()} монет поза навчанням, {START} – {END}; Alert = 15m + годинна (власні 15m Binance) ≥ {share}; "
             f"частковий вихід, стоп {K} ATR, комісія 0.1%.", "", "## Загалом", "",
             "| варіант | угод | у плюсі | медіана | PF | PF без топ-3 | ≥ 20 млн | 5–20 млн |", "|---|---|---|---|---|---|---|---|"]
    for v, g in tr.groupby("variant"):
        r = g["ret"].sort_values()
        lines.append(f"| {v} | {len(r)} | {(r > 0).mean():.0%} | {r.median():+.2%} | {pf(r):.2f} | {pf(r.iloc[:-3]):.2f} | "
                     f"{pf(g[g['band'] == '≥ 20 млн']['ret']):.2f} | {pf(g[g['band'] == '5–20 млн']['ret']):.2f} |")
    piv = tr.groupby(["month", "variant"])["ret"].apply(pf).unstack()
    cnt = tr.groupby(["month", "variant"]).size().unstack()
    lines += ["", "## По місяцях (PF; у дужках — угод)", "", "| місяць | " + " | ".join(piv.columns) + " |", "|---|" + "---|" * len(piv.columns)]
    for m, r in piv.iterrows():
        lines.append(f"| {m} | " + " | ".join(f"{r[v]:.2f} ({cnt.loc[m, v]})" for v in piv.columns) + " |")
    if {"base", "book"} <= set(piv.columns):
        better = int((piv["book"] > piv["base"]).sum())
        lines += ["", f"**book краще за base у {better} з {len(piv)} місяців.**"]
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Звіт: {OUT}; угоди: {TRADES}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="6-місячний бектест на архіві Binance: дані → ознаки → моделі й угоди по місяцях")
    ap.add_argument("--stage", choices=["data", "features", "test", "all"], default="all", help="який етап (за замовчуванням усі)")
    ap.add_argument("--limit", type=int, help="лише перші N монет (перевірка конвеєра)")
    args = ap.parse_args()
    load_dotenv(".env")
    excluded = excluded_tickers(".")
    training = {t.upper() for t in read_tickers_file("tickers.txt")}
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        training |= set(load_pumps(conn)["base_asset"])
        coins, band = coins_and_bands(training)
        coins = coins[:args.limit] if args.limit else coins
        syms = BF.binance_symbols(conn, coins)
        print(f"{len(coins)} монет, {len(syms)} з ринком Binance", flush=True)
        if args.stage in ("data", "all"):
            stage_data(conn, coins, syms)
        if args.stage in ("features", "all"):
            stage_features(conn, coins, syms)
        if args.stage in ("test", "all"):
            stage_test(conn, coins, band, excluded)


if __name__ == "__main__":
    main()
