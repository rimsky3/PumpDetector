"""
experiments/book_features.py — ознаки глибини стакану в 15m-моделі (2026-10-07).

Дешеве джерело: архів Binance bookDepth (orderbook.archive_day; кожні ~30–40 с сумарні біди / аски в межах ±0.2…5% від
ціни, будь-яка монета й день, ~350 КБ на монето-день, кеш data/book_cache). Мікроструктуру (спуфінг, айсберги) НЕ беремо.
Раніше перевірено лише «біди / аски ±2% ≥ 1 у момент сигналу» як фільтр — не тримається на 6 міс. Тут — ознаки моделі:

    book_imb_1 / _2 / _5      дисбаланс (біди − аски) / (біди + аски) у межах ±1 / 2 / 5%
    book_ask2_turn / ask5     log(аски до +2% / +5% ÷ оборот за 24 год) — «тонкий» стакан угорі: ціну легко проштовхнути
    book_depth2_turn          log((біди + аски) ±2% ÷ оборот 24 год) — наскільки ринок тонкий для своєї активності
    book_ask2_chg_1h / _4h    log(аски ±2% зараз / годину / 4 год тому) — аски знімають перед рухом?
    book_bid2_chg_1h / _4h    те саме для бідів — біди доливають?
    book_ask2_vs_7d           log(аски ±2% / медіана за 7 днів) — відносно власної норми монети

Знімок — останній ≤ закриття бару (не старший за 3 хв), тобто без погляду вперед. Немає архіву — NaN (LightGBM вміє).
Варіанти 15m-моделі: base = MODEL_FEATURES_15M (новий набір без ліквідацій), book = base + ознаки стакану. Годинне
підтвердження Alert — годинний кандидат без ліквідацій (models/candidate/lightgbm.joblib, або робоча, якщо кандидата немає).
Перевірка: leave-ticker-out (рівень «Кандидат») і бектест на монетах ≥ 20 млн $ поза навчанням (останні дні до
позавчора: архів за день з'являється наступного дня), поточний і частковий вихід.

    python experiments/book_features.py
"""

import argparse
import glob
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd
import psycopg
from dotenv import load_dotenv

import exit_partial as P
import orderbook
import train
from features import (INTERVAL_15M, MODEL_FEATURES_15M, aggregate, build_from_raw, build_hourly_own, load_pumps, load_raw,
                      query_df, short_features)
from fetch import excluded_tickers, read_tickers_file
from tiers import tier_frame

DAYS, K = 30, 2.48
OUT = os.path.join("reports", "book_features.md")
BOOK = ["book_imb_1", "book_imb_2", "book_imb_5", "book_ask2_turn", "book_ask5_turn", "book_depth2_turn",
        "book_ask2_chg_1h", "book_ask2_chg_4h", "book_bid2_chg_1h", "book_bid2_chg_4h", "book_ask2_vs_7d"]
# другий прохід (власник 07.10: «може, рахувати на 5 хв?»): коротка динаміка — зі ВСІХ знімків усередині бару
FAST = ["book_ask2_chg_5m", "book_bid2_chg_5m", "book_ask2_chg_15m", "book_bid2_chg_15m",
        "book_imb2_mean_bar", "book_imb2_min_bar", "book_imb2_max_bar", "book_ask2_sd_bar"]
VARIANTS = {"base": MODEL_FEATURES_15M, "book": MODEL_FEATURES_15M + BOOK, "book_fast": MODEL_FEATURES_15M + BOOK + FAST}
ALL_BOOK = BOOK + FAST
LAST_DAY = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=2)).normalize()   # новіші архіви ще можуть не вийти


def binance_symbols(conn, bases: list[str]) -> dict[str, str]:
    rows = conn.execute("""SELECT base_asset, symbol_on_exchange FROM markets WHERE base_asset = ANY(%s)
                           AND exchange_name = 'Binance' AND quote_asset = 'USDT'""", (bases,)).fetchall()
    return {b: s for b, s in rows}


def fetch_days(jobs: list[tuple[str, str]]) -> None:
    """Скачати (або взяти з кешу) архіви (символ, день) паралельно."""
    todo = [(s, d) for s, d in jobs if not os.path.exists(os.path.join(orderbook.ARCHIVE_CACHE, f"{s}-bookDepth-{d}.parquet"))]
    print(f"  архів стакану: {len(jobs)} монето-днів, докачати {len(todo)}…", flush=True)
    with ThreadPoolExecutor(8) as pool:
        for n, _ in enumerate(pool.map(lambda a: orderbook.archive_day(*a), todo), 1):
            if n % 300 == 0:
                print(f"    {n}/{len(todo)}", flush=True)


def book_wide(symbol: str, days: list[str]) -> pd.DataFrame:
    parts = [orderbook.archive_day(symbol, d) for d in days]
    parts = [p for p in parts if len(p)]
    if not parts:
        return pd.DataFrame()
    w = pd.concat(parts).pivot_table(index="timestamp", columns="percentage", values="notional", aggfunc="last").sort_index()
    return w[[c for c in (-5.0, -2.0, -1.0, 1.0, 2.0, 5.0) if c in w.columns]]


def book_features(h: pd.DataFrame, w: pd.DataFrame) -> pd.DataFrame:
    """15m-ряд монети (індекс — відкриття бару, volume у $) + смуги стакану → ознаки на закритті кожного бару."""
    out = pd.DataFrame(index=h.index, columns=ALL_BOOK, dtype=float)
    if w.empty or not {-5.0, -2.0, -1.0, 1.0, 2.0, 5.0} <= set(w.columns):
        return out
    close = pd.DataFrame({"t": h.index + pd.Timedelta(minutes=15)})
    s = pd.merge_asof(close, w.reset_index().rename(columns={"timestamp": "t"}), on="t", direction="backward",
                      tolerance=pd.Timedelta(minutes=3)).set_index(h.index)
    b1, b2, b5, a1, a2, a5 = (s[c] for c in (-1.0, -2.0, -5.0, 1.0, 2.0, 5.0))
    turn = h["volume"].rolling(96, min_periods=48).sum()
    lg = lambda x: np.log(x.where(x > 0))        # noqa: E731
    out["book_imb_1"] = (b1 - a1) / (b1 + a1)
    out["book_imb_2"] = (b2 - a2) / (b2 + a2)
    out["book_imb_5"] = (b5 - a5) / (b5 + a5)
    out["book_ask2_turn"] = lg(a2) - lg(turn)
    out["book_ask5_turn"] = lg(a5) - lg(turn)
    out["book_depth2_turn"] = lg(a2 + b2) - lg(turn)
    out["book_ask2_chg_1h"] = lg(a2) - lg(a2.shift(4))
    out["book_ask2_chg_4h"] = lg(a2) - lg(a2.shift(16))
    out["book_bid2_chg_1h"] = lg(b2) - lg(b2.shift(4))
    out["book_bid2_chg_4h"] = lg(b2) - lg(b2.shift(16))
    out["book_ask2_vs_7d"] = lg(a2) - lg(a2.rolling(672, min_periods=96).median())
    # коротка динаміка: знімок за 5 / 15 хв до закриття і всі знімки всередині бару (відкриття, закриття]
    for m in (5, 15):
        prev = pd.merge_asof(pd.DataFrame({"t": close["t"] - pd.Timedelta(minutes=m)}), w.reset_index().rename(columns={"timestamp": "t"}),
                             on="t", direction="backward", tolerance=pd.Timedelta(minutes=3)).set_index(h.index)
        out[f"book_ask2_chg_{m}m"] = lg(a2) - lg(prev[2.0])
        out[f"book_bid2_chg_{m}m"] = lg(b2) - lg(prev[-2.0])
    snap = pd.DataFrame({"imb": (w[-2.0] - w[2.0]) / (w[-2.0] + w[2.0]), "la": np.log(w[2.0].where(w[2.0] > 0))})
    bar = (snap.index - pd.Timedelta(microseconds=1)).floor("15min")         # знімок о 10:15:00 — ще бар 10:00
    g = snap.groupby(bar)
    agg = pd.DataFrame({"book_imb2_mean_bar": g["imb"].mean(), "book_imb2_min_bar": g["imb"].min(),
                        "book_imb2_max_bar": g["imb"].max(), "book_ask2_sd_bar": g["la"].std()})
    for c in agg:
        out[c] = agg[c].reindex(h.index)
    return out.replace([np.inf, -np.inf], np.nan)


def add_book(conn, d: pd.DataFrame, raw: pd.DataFrame, syms: dict[str, str]) -> pd.DataFrame:
    """Ознаки стакану для всіх тікерів кадру d (base_asset, ts)."""
    parts = []
    for b, g in d.groupby("base_asset"):
        sym = syms.get(b)
        h = aggregate(raw, b, "15min")
        if sym is None or h.empty:
            continue
        days = [x.strftime("%Y-%m-%d") for x in pd.date_range((g["ts"].min() - pd.Timedelta(days=8)).normalize(),
                                                                min(g["ts"].max().normalize(), LAST_DAY), freq="D")]
        f = book_features(h, book_wide(sym, days)).reindex(g["ts"]).set_axis(g.index)
        parts.append(f)
    bk = pd.concat(parts) if parts else pd.DataFrame(columns=ALL_BOOK)
    return d.drop(columns=[c for c in ALL_BOOK if c in d], errors="ignore").join(bk.reindex(d.index))


def pf(v: pd.Series) -> str:
    gl = -v[v <= 0].sum()
    return f"{v[v > 0].sum() / gl:.2f}" if gl else "∞"


def stats(df: pd.DataFrame) -> dict:
    v = df["ret"].sort_values()
    return {"угод": len(v), "у плюсі": f"{(v > 0).mean():.0%}", "медіана": f"{v.median():+.2%}", "PF": pf(v), "PF без топ-3": pf(v.iloc[:-3])}


def md(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    lines += ["| " + str(i) + " | " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |" for i, r in zip(df.index, df.to_numpy())]
    return "\n".join(lines)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    argparse.ArgumentParser(description="Ознаки глибини стакану (архів Binance bookDepth) у 15m-моделі: LTO і бектест").parse_args()
    load_dotenv(".env")
    excluded = excluded_tickers(".")
    reviewed = {t.upper() for t in read_tickers_file("tickers.txt")} - excluded
    p1 = train.model_path("lightgbm", train.CANDIDATE_DIR)
    m1 = joblib.load(p1 if os.path.exists(p1) else train.model_path("lightgbm"))
    m15 = joblib.load(train.model_path(train.M15_NAME))
    exit_bars, share = m15.get("exit_bars", 4), m15.get("confirm", {}).get("share", 0.75)
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=48)
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        pumps = load_pumps(conn)
        pumps = pumps[~pumps["base_asset"].isin(excluded)]
        bases = sorted(reviewed | set(pumps["base_asset"]))
        raw = load_raw(conn, bases, INTERVAL_15M)
        d = build_from_raw(raw, pumps, bars_per_hour=4, extra=short_features)
        first1 = query_df(conn, "SELECT base_asset, min(ts) AS t FROM features GROUP BY 1").set_index("base_asset")["t"]
        syms = binance_symbols(conn, bases)
    d = d[d["base_asset"].isin(reviewed) | d["pattern_type"].notna()]
    d = d[(d["ts"] <= cutoff) | d["pattern_type"].notna()]
    d = d[d["ts"] >= d["base_asset"].map(first1)].reset_index(drop=True)
    jobs = sorted({(syms[b], x.strftime("%Y-%m-%d")) for b, g in d.groupby("base_asset") if b in syms
                   for x in pd.date_range((g["ts"].min() - pd.Timedelta(days=8)).normalize(), min(g["ts"].max().normalize(), LAST_DAY))})
    fetch_days(jobs)
    d = add_book(None, d, raw, syms)
    cover = d["book_imb_2"].notna().mean()
    y, groups = d["label"].astype(int), d["base_asset"]
    trained = set(groups)
    print(f"15m: {len(d)} барів, {groups.nunique()} тікерів; ознаки стакану є в {cover:.0%} барів", flush=True)

    lto, models, imp = {}, {}, None
    for name, cols in VARIANTS.items():
        print(f"  {name}: {len(cols)} ознак, leave-ticker-out…", flush=True)
        p = train.oof_proba(train.make_lightgbm, d[cols], y, groups)
        thr = train.best_f1_threshold(y.to_numpy(), p)
        ev = train.event_metrics(d, p >= thr, pumps, train.BAR_15M)
        lto[name] = {"спіймано": ev["pumps_caught"], "затримка, хв": round(ev["median_delay_h"] * 60) if pd.notna(ev["median_delay_h"]) else None,
                     "фальстарти": ev["false_starts_triggered"], "хибних / 30 д": round(ev["false_alarms_per_30d"], 1)}
        model = train.make_lightgbm().fit(d[cols], y)
        models[name] = (model, thr, cols)
        if name == "book_fast":
            g = pd.Series(model.booster_.feature_importance("gain"), index=cols)
            imp = (g / g.sum()).sort_values(ascending=False)

    # бектест: монети ≥ 20 млн $ поза навчанням, останні DAYS днів до LAST_DAY
    coins = set()
    for path in glob.glob(os.path.join("reports", "backtests", "20261001_*_alert.json")):
        meta = json.load(open(path, encoding="utf-8"))["meta"]
        if meta["min_turnover"] >= 20e6:
            coins |= {u["base"] for u in meta["universe"]}
    coins = sorted(coins - trained - excluded)
    end = LAST_DAY + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=DAYS)
    since = start - pd.Timedelta(days=9)
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        bsyms = binance_symbols(conn, coins)
        fetch_days([(bsyms[b], x.strftime("%Y-%m-%d")) for b in coins if b in bsyms
                    for x in pd.date_range(start - pd.Timedelta(days=8), LAST_DAY)])
        res = {f"{n} · {e}": [] for n in VARIANTS for e in ("поточний вихід", "частковий вихід")}
        print(f"  бектест: {len(coins)} монет, {start:%m-%d} – {LAST_DAY:%m-%d}…", flush=True)
        for n, b in enumerate(coins, 1):
            rawb = load_raw(conn, [b], INTERVAL_15M, since=since)
            f15 = build_from_raw(rawb, None, bars_per_hour=4, extra=short_features)
            h1 = build_hourly_own(conn, [b], since=since)
            if f15.empty or h1.empty:
                continue
            f15 = add_book(None, f15, rawb, bsyms)
            h = aggregate(rawb, b, "15min")
            h = h[(h.index >= start - pd.Timedelta(days=1)) & (h.index < end)]
            s1 = h1[["base_asset", "ts"]].assign(probability=m1["model"].predict_proba(h1[m1["features"]])[:, 1], threshold=m1["threshold"])
            for name, (model, thr, cols) in models.items():
                s15 = f15[["base_asset", "ts"]].assign(probability=model.predict_proba(f15[cols])[:, 1], threshold=thr)
                t = tier_frame(s15, s1, share).set_index("ts").sort_index()
                sc = t.loc[(t.index >= start) & (t.index < end), "alert_score"]
                res[f"{name} · поточний вихід"] += P.simulate(h, sc, K, exit_bars, None, 0, False)
                res[f"{name} · частковий вихід"] += P.simulate(h, sc, K, exit_bars, 3.0, 0.5, True)
            if n % 25 == 0:
                print(f"    {n}/{len(coins)}", flush=True)

    t_lto = pd.DataFrame(lto).T
    t_bt = pd.DataFrame({k: stats(pd.DataFrame(v)) for k, v in res.items()}).T
    lines = ["# Ознаки глибини стакану (архів Binance bookDepth)", "",
             f"Навчання: {groups.nunique()} тікерів, {int((pumps['pattern_type'] == 'pump').sum())} імпульсів; ознаки стакану є в "
             f"{cover:.0%} 15m-барів (решта — немає архіву або Binance-ринку). base — {len(VARIANTS['base'])} ознак без ліквідацій.", "",
             "## 1. Leave-ticker-out (15m, рівень «Кандидат»)", "", md(t_lto), "",
             f"## 2. Бектест: {len(coins)} монет ≥ 20 млн $ поза навчанням, {start:%Y-%m-%d} – {LAST_DAY:%Y-%m-%d} "
             f"(Alert = 15m + годинна без ліквідацій ≥ {share}; стоп {K} ATR)", "", md(t_bt), "",
             "## 3. Важливість ознак (book_fast, gain): ознаки стакану і топ-5", ""]
    for f, v in imp.items():
        if f in ALL_BOOK or f in imp.index[:5]:
            lines.append(f"- {f}: {v:.1%}" + ("  ← стакан, коротка динаміка" if f in FAST else "  ← стакан" if f in BOOK else ""))
    lines.append(f"\nРазом стакан: {imp[ALL_BOOK].sum():.1%} (з них коротка динаміка {imp[FAST].sum():.1%})")
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Звіт: {OUT}")


if __name__ == "__main__":
    main()
