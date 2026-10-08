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
    python experiments/long_binance.py --universe monthly   # без упередження виживання (див. нижче)

--universe monthly (2026-10-07): список монет СТАНОМ НА КОЖЕН МІСЯЦЬ, а не на сьогодні. Раніше брали монети, що торгуються
зараз і мають оборот зараз (data/long_bt) — делістовані й згаслі за пів року (часто якраз після пампів) у бектест не
потрапляли, а розкачані пізніше — потрапляли з місяців, коли ними ніхто б не торгував (survivorship bias). Тепер: усі
USDT-перпи з архіву Binance (S3-лістинг, включно з делістованими; без акцій / товарів, BTC / ETH і стейблів), середній денний
оборот за 30 днів до початку місяця з денних свічок архіву; монета входить у бектест місяця, якщо цей оборот 5 млн – 1 млрд $
(як market.py / сканер) — угоди рахуються лише в її місяцях. Звіт — поруч зі старим списком «станом на сьогодні».
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
UNIVERSE = os.path.join(DIR, "_universe.json")
S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
MONTHS = ["2026-04", "2026-05", "2026-06", "2026-07", "2026-08", "2026-09"]
MIN_TURN, MAX_TURN = 5e6, 1e9
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


def archive_symbols() -> list[str]:
    """Усі символи USDT-M ф'ючерсів, що будь-коли мали місячні свічки в архіві (включно з делістованими)."""
    import re
    import requests
    out, marker = [], ""
    while True:
        r = requests.get(S3, params={"prefix": "data/futures/um/monthly/klines/", "delimiter": "/", "marker": marker}, timeout=60)
        r.raise_for_status()
        out += re.findall(r"<Prefix>data/futures/um/monthly/klines/([^/<]+)/</Prefix>", r.text)
        nxt = re.search(r"<NextMarker>([^<]*)</NextMarker>", r.text)
        if "<IsTruncated>true" not in r.text or not nxt:
            break
        marker = nxt.group(1)
    return sorted(set(out))


def universe_monthly(exclude: set) -> dict:
    """{монета: {місяць: смуга обороту}} — монети, що на початок місяця мали середній денний оборот 5 млн – 1 млрд $
    за попередні 30 днів (≥ 7 днів історії). Кеш data/long_bin/_universe.json."""
    import json
    if os.path.exists(UNIVERSE):
        return json.load(open(UNIVERSE, encoding="utf-8"))
    import exchanges as X
    import market
    from concurrent.futures import ThreadPoolExecutor
    info = X.get(X.BINANCE + "/fapi/v1/exchangeInfo", {})
    not_crypto = {s["symbol"] for s in info["symbols"] if s.get("contractType") not in ("PERPETUAL", None)}
    syms = [s for s in archive_symbols() if s.endswith("USDT") and s not in market.NOT_ALTS and s not in not_crypto
            and s[:-4] not in exclude]
    print(f"  символів в архіві: {len(syms)} USDT-перпів-кандидатів", flush=True)
    months_dl = ["2026-03"] + MONTHS

    def turnover(sym):
        parts = [BA._csv("klines", sym, m, "1d", monthly=True) for m in months_dl]
        parts = [x for x in parts if len(x)]
        if not parts:
            return sym, pd.Series(dtype=float)
        k = pd.concat(parts)
        s = pd.Series(k["quote_volume"].astype(float).to_numpy(), index=pd.to_datetime(k["open_time"].astype("int64"), unit="ms", utc=True))
        return sym, s[~s.index.duplicated()].sort_index()
    with ThreadPoolExecutor(8) as pool:
        tv = dict(pool.map(turnover, syms))
    uni = {}
    for sym, s in tv.items():
        if s.empty:
            continue
        for m in MONTHS:
            m0 = pd.Timestamp(f"{m}-01", tz="UTC")
            prev = s[(s.index >= m0 - pd.Timedelta(days=30)) & (s.index < m0)]
            listed = ((s.index >= m0) & (s.index < m0 + pd.offsets.MonthBegin(1))).any()
            if listed and len(prev) >= 7 and MIN_TURN <= prev.mean() <= MAX_TURN:
                uni.setdefault(sym[:-4], {})[m] = "≥ 20 млн" if prev.mean() >= 20e6 else "5–20 млн"
    json.dump(uni, open(UNIVERSE, "w", encoding="utf-8"), ensure_ascii=False)
    return uni


def ensure_markets(conn, coins: list[str]) -> None:
    """Рядки markets для монет без ринку Binance (делістовані: Coinalyze їх уже не знає) — того ж вигляду, що від fetch.py."""
    have = {r[0] for r in conn.execute("""SELECT base_asset FROM markets WHERE exchange_name = 'Binance' AND quote_asset = 'USDT'
                                          AND base_asset = ANY(%s)""", (coins,)).fetchall()}
    rows = [(f"{c}USDT_PERP.A", "A", "Binance", c, "USDT", f"{c}USDT", "BASE_ASSET", True, True, True) for c in coins if c not in have]
    if rows:
        conn.cursor().executemany("""INSERT INTO markets (symbol, exchange_code, exchange_name, base_asset, quote_asset,
                                     symbol_on_exchange, volume_denominated, has_ohlcv, has_buy_sell, has_long_short)
                                     VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (symbol) DO NOTHING""", rows)
        conn.commit()
        print(f"  додано ринків Binance (делістовані / нові): {len(rows)}", flush=True)


def stage_data(conn, coins: list[str], syms: dict, book: bool = True) -> None:
    os.makedirs(DIR, exist_ok=True)
    done = set(open(DONE, encoding="utf-8").read().split()) if os.path.exists(DONE) else set()
    todo = [c for c in coins if c not in done and c in syms]
    # стакан — паралельно (інший тип файлів, той самий сайт)
    jobs = [(syms[c], d) for c in coins if c in syms for d in BA._days(WARM, END)] if book else []
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


def stage_features(conn, coins: list[str], syms: dict, book: bool = True) -> None:
    """book=False — без ознак стакану (NaN): для нових монет їх архів не скачано, а качати по одному днях у циклі —
    ~5 хв на монету (07.10); стакан для моделі вже відкинуто (6 міс.: PF 0.97 проти 0.96)."""
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
        bk = (BF.book_features(h, BF.book_wide(syms[c], days)) if book else pd.DataFrame(index=h.index, columns=BF.ALL_BOOK, dtype=float)
              ).reindex(f15["ts"]).set_axis(f15.index)
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


def stage_test(conn, coins: list[str], band: dict, excluded: set, uni: dict | None = None, current: set | None = None) -> None:
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
    tr["month"] = pd.to_datetime(tr["ts"], utc=True).dt.strftime("%Y-%m")
    tr.to_csv(TRADES, index=False)
    if uni is not None:
        report_universe(tr, uni, current or set())
        return
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


def report_universe(tr: pd.DataFrame, uni: dict, current: set) -> None:
    """Список «станом на місяць» (чесний) проти «станом на сьогодні» (старий) на тих самих угодах."""
    inm = tr.apply(lambda r: r["month"] in uni.get(r["ticker"], {}), axis=1)
    tr = tr.assign(band=[uni.get(t, {}).get(m, b) for t, m, b in zip(tr["ticker"], tr["month"], tr["band"])])
    sets = {"станом на місяць (без упередження)": tr[inm], "станом на сьогодні (як раніше)": tr[tr["ticker"].isin(current)]}
    lines = ["# 6 місяців на архіві Binance: упередження виживання", "",
             f"Список «станом на місяць»: монети, що на початок місяця мали середній оборот 5 млн – 1 млрд $ за 30 днів "
             f"(включно з делістованими пізніше). Монет у місяцях: " + ", ".join(f"{m[5:]} {sum(m in v for v in uni.values())}" for m in MONTHS)
             + f"; усього різних {len(uni)}. Старий список — {len(current)} монет, що торгуються зараз.", ""]
    for name, d in sets.items():
        lines += [f"## {name}", "", "| варіант | угод | монет | у плюсі | медіана | PF | PF без топ-3 | ≥ 20 млн | 5–20 млн |", "|---|---|---|---|---|---|---|---|---|"]
        for v, g in d.groupby("variant"):
            r = g["ret"].sort_values()
            lines.append(f"| {v} | {len(r)} | {g['ticker'].nunique()} | {(r > 0).mean():.0%} | {r.median():+.2%} | {pf(r):.2f} | {pf(r.iloc[:-3]):.2f} | "
                         f"{pf(g[g['band'] == '≥ 20 млн']['ret']):.2f} | {pf(g[g['band'] == '5–20 млн']['ret']):.2f} |")
        piv = d.groupby(["month", "variant"])["ret"].apply(pf).unstack()
        cnt = d.groupby(["month", "variant"]).size().unstack()
        lines += ["", "| місяць | " + " | ".join(piv.columns) + " |", "|---|" + "---|" * len(piv.columns)]
        for m, r in piv.iterrows():
            lines.append(f"| {m} | " + " | ".join(f"{r[v]:.2f} ({cnt.loc[m, v]})" for v in piv.columns) + " |")
        lines.append("")
    gone = sorted({t for t, v in uni.items() if t not in current})
    lines += [f"Монет у чесному списку, яких немає в старому (делістовані, згаслі або тоді ще дрібні / вже великі): {len(gone)}.", ""]
    out = os.path.join("reports", "long_binance_universe.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Звіт: {out}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="6-місячний бектест на архіві Binance: дані → ознаки → моделі й угоди по місяцях")
    ap.add_argument("--stage", choices=["data", "features", "test", "all"], default="all", help="який етап (за замовчуванням усі)")
    ap.add_argument("--limit", type=int, help="лише перші N монет (перевірка конвеєра)")
    ap.add_argument("--universe", choices=["current", "monthly"], default="current",
                    help="current — монети, що торгуються зараз (як раніше); monthly — станом на кожен місяць, з делістованими")
    args = ap.parse_args()
    load_dotenv(".env")
    excluded = excluded_tickers(".")
    training = {t.upper() for t in read_tickers_file("tickers.txt")}
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        training |= set(load_pumps(conn)["base_asset"])
        coins, band = coins_and_bands(training)
        uni, current = None, set(coins)
        if args.universe == "monthly":
            VARIANTS.pop("book", None)            # стакан відкинуто; для нових монет його архіву немає — лише base
            uni = universe_monthly(training | excluded)
            coins = sorted(set(uni) | set(coins))
            ensure_markets(conn, coins)
        coins = coins[:args.limit] if args.limit else coins
        syms = BF.binance_symbols(conn, coins)
        print(f"{len(coins)} монет, {len(syms)} з ринком Binance", flush=True)
        if args.stage in ("data", "all"):
            stage_data(conn, coins, syms, book=args.universe == "current")
        if args.stage in ("features", "all"):
            stage_features(conn, coins, syms, book=args.universe == "current")
        if args.stage in ("test", "all"):
            stage_test(conn, coins, band, excluded, uni, current)


if __name__ == "__main__":
    main()
