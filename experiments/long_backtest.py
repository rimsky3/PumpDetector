"""
experiments/long_backtest.py — довгий бектест (6 місяців) правил входу на даних Bybit (2026-10-02).

Широкий бектест (experiments/entry_rules_backtest.py) — лише вересень: фільтр стакану OB ≥ 1 + ліміт −0.5 ATR дали PF 1.08
проти 0.86, але майже все покращення — у другій половині місяця. Тут — чи тримається це в різні місяці.

Чому Bybit: модель потребує 15m OI, а Binance віддає його лише за ~30 днів; Bybit — щонайменше 200 днів.
Наближення (однакове для всіх варіантів, тож порівняння правил між собою чесне; абсолютні цифри — ні):
    • 15m і годинні ознаки — лише з Bybit (модель вчилась на сумі Binance + Bybit);
    • ліквідацій немає на всьому періоді (Coinalyze тримає 15m лише 1–2 міс.) → модель бачить «невідомо» (NaN) скрізь;
    • частки покупок у свічках Bybit немає → buy_ratio-ознаки порожні.
Стакан для фільтра — архів Binance bookDepth (є з 2025 р.), на момент кожного сигналу.

Етапи (кожен з кешем — повторний запуск продовжує):
    1) дані: 15m Bybit за DAYS + 9 днів (свічки, OI, L/S, funding) → база (fetch_exchange.collect, лише Bybit);
    2) оцінки: робоча 15m- і годинна модель → рівні (tiers) → data/long_bt/<монета>.pkl;
    3) OB на кожному початку Alert; 4) угоди 4 варіантів (L.simulate), звіт по місяцях, смугах обороту й загалом.

    python experiments/long_backtest.py              # усе
    python experiments/long_backtest.py --skip-fetch # без завантаження (лише те, що вже в базі)
"""

import argparse
import json
import os
import pickle
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd
import psycopg
from dotenv import load_dotenv

import book_spot as BS
import fetch_exchange
import limit_entry as L
import train
from features import FEATURES, FEATURES_15M, INTERVAL_15M, aggregate, build_from_raw, hourly_from_intraday, load_raw, short_features
from tiers import tier_frame

DAYS, WARM = 180, 9
CACHE_DIR = os.path.join("data", "long_bt")
OUT = os.path.join("reports", "long_backtest_trades.csv")
B15 = pd.Timedelta(minutes=15)
VARIANTS = {"як зараз (ринок)": ("market", 1, 0, 0, False), "+ фільтр OB ≥ 1": ("market", 1, 0, 0, True),
            "+ ліміт −0.5 ATR": ("atr", 1, 0.5, 4, False), "OB + ліміт": ("atr", 1, 0.5, 4, True)}


def no_liq(raw: dict) -> dict:
    """Ліквідації «невідомо» на всьому періоді: порожні ряди й порожнє покриття → NaN (а не нуль) у features.aggregate."""
    raw = dict(raw)
    raw["liq"] = raw["liq"].iloc[0:0]
    raw["liq_coverage"] = pd.DataFrame(columns=["base_asset", "market", "covered_from"])
    return raw


def pf(v: pd.Series) -> float:
    gl = -v[v <= 0].sum()
    return round(v[v > 0].sum() / gl, 2) if gl else np.nan


def main() -> None:
    ap = argparse.ArgumentParser(description="Довгий бектест (6 міс., Bybit) правил входу: ринок / OB / ліміт / обидва")
    ap.add_argument("--skip-fetch", action="store_true", help="не завантажувати дані з Bybit (лише наявне в базі)")
    ap.add_argument("--days", type=int, default=DAYS, help="скільки днів (за замовчуванням 180)")
    ap.add_argument("--limit", type=int, default=0, help="лише перші N монет (пробний прогін)")
    args = ap.parse_args()
    load_dotenv(".env")
    os.makedirs(CACHE_DIR, exist_ok=True)
    m15 = joblib.load(train.model_path(train.M15_NAME))
    m1 = joblib.load(train.model_path("lightgbm"))
    k, exit_bars, share = m15.get("risk", {}).get("k"), m15.get("exit_bars", 4), m15.get("confirm", {}).get("share", 0.75)
    band = {}
    for run in L.RUNS:
        d = json.load(open(os.path.join("reports", "backtests", run + ".json"), encoding="utf-8"))
        for u in d["meta"]["universe"]:
            band[u["base"]] = "≥ 20 млн" if d["meta"]["min_turnover"] >= 20e6 else "5–20 млн"
    coins = sorted(pickle.load(open(L.ALERTS, "rb")))
    now = pd.Timestamp(datetime.now(timezone.utc))
    start = now - pd.Timedelta(days=args.days)
    since = start - pd.Timedelta(days=WARM)
    feats = pd.read_csv(os.path.join("reports", "book_spot_features.csv"))
    binance_sym = dict(zip(feats["ticker"], feats["sym"]))

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        have = {b for (b,) in conn.execute("""SELECT DISTINCT base_asset FROM markets WHERE exchange_name = 'Bybit'
                                              AND quote_asset = 'USDT' AND base_asset = ANY(%s)""", (coins,))}
        coins = [c for c in coins if c in have][: args.limit or None]
        # 1) дані
        if not args.skip_fetch:
            t0 = time.time()
            first = dict(conn.execute("""SELECT m.base_asset, min(o.ts) FROM ohlcv o JOIN markets m USING (symbol)
                                         WHERE m.base_asset = ANY(%s) AND m.exchange_name = 'Bybit' AND o.interval = '15min'
                                         GROUP BY 1""", (coins,)).fetchall())
            todo = [b for b in coins if b not in first or pd.Timestamp(first[b]) > since + pd.Timedelta(days=1)]
            print(f"  дані: треба завантажити {len(todo)} з {len(coins)} монет", flush=True)
            for i in range(0, len(todo), 8):              # пачка з 8 монет — ринки качаються паралельно (fetch_exchange.WORKERS)
                part = todo[i:i + 8]
                try:
                    fetch_exchange.collect(conn, part, ["15m"], args.days + WARM + 1, log=lambda m: None, exchanges=["Bybit"])
                except Exception as e:
                    print(f"  ! {part}: {e}", flush=True)
                print(f"  дані: {min(i + 8, len(todo))}/{len(todo)} · {(time.time() - t0) / 60:.0f} хв", flush=True)
        # 2) оцінки
        for n, b in enumerate(coins, 1):
            path = os.path.join(CACHE_DIR, f"{b}.pkl")
            if os.path.exists(path):
                continue
            raw = no_liq(load_raw(conn, [b], INTERVAL_15M, since=since, exchanges=["Bybit"]))
            f15 = build_from_raw(raw, None, bars_per_hour=4, extra=short_features)
            f1 = build_from_raw(hourly_from_intraday(raw))
            if f15.empty or f1.empty:
                pickle.dump(None, open(path, "wb"))
                continue
            s15 = f15[["base_asset", "ts"]].assign(probability=m15["model"].predict_proba(f15[FEATURES_15M])[:, 1], threshold=m15["threshold"])
            s1 = f1[["base_asset", "ts"]].assign(probability=m1["model"].predict_proba(f1[FEATURES])[:, 1], threshold=m1["threshold"])
            t = tier_frame(s15, s1, share).set_index("ts").sort_index()
            h = aggregate(raw, b, "15min")[["open", "high", "low", "price", "volume"]]
            pickle.dump((t.loc[t.index >= start, ["alert_score", "tier"]], h[h.index >= start - pd.Timedelta(days=1)]), open(path, "wb"))
            if n % 20 == 0:
                print(f"  оцінки: {n}/{len(coins)} монет", flush=True)

    # 3–4) OB на сигналах і угоди
    rows = []
    for n, b in enumerate(coins, 1):
        got = pickle.load(open(os.path.join(CACHE_DIR, f"{b}.pkl"), "rb"))
        if got is None:
            continue
        t, h = got
        sc = t["alert_score"].reindex(h.index)
        al = (sc >= 1).to_numpy()
        starts = [i for i in range(1, len(h)) if al[i] and not al[i - 1]]
        sym = binance_sym.get(b, f"{b}USDT")
        days = sorted({d for i in starts for d in ((h.index[i] - pd.Timedelta(days=1)).strftime("%Y-%m-%d"), h.index[i].strftime("%Y-%m-%d"))})
        w = BS.book_wide(sym, days) if days else pd.DataFrame()
        imb = {i: BS.book_features(w, h.index[i] + B15).get("imb_2", np.nan) for i in starts}
        for name, (mode, ww, x, valid, filt) in VARIANTS.items():
            allow = (lambda i: imb.get(i, np.nan) >= 0) if filt else None      # перекіс ≥ 0 ⇔ біди / аски ≥ 1
            for r in L.simulate(h, sc, k, exit_bars, mode, ww, x, valid, allow):
                rows.append({"variant": name, "ticker": b, "ts": h.index[r["sig"]], "ret": r["ret"], "reason": r["reason"],
                             "band": band.get(b, "")})
        if n % 20 == 0:
            print(f"  угоди: {n}/{len(coins)} монет", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT, index=False)
    d["month"] = d["ts"].dt.strftime("%Y-%m")
    pd.set_option("display.width", 260)
    print(f"\n{d['ticker'].nunique()} монет, {start:%Y-%m-%d} … {now:%Y-%m-%d}; дані Bybit, без ліквідацій; вихід частковий\n")
    tot = {}
    for name in VARIANTS:
        g = d[d["variant"] == name]
        v = g["ret"].sort_values()
        tot[name] = {"угод": len(v), "у плюсі": f"{(v > 0).mean():.0%}", "медіана": f"{v.median():+.2%}", "EV": f"{v.mean():+.2%}",
                     "PF": pf(v), "PF без топ-3": pf(v.iloc[:-3]), "5–20 млн": pf(g[g["band"] == "5–20 млн"]["ret"]),
                     "≥ 20 млн": pf(g[g["band"] == "≥ 20 млн"]["ret"])}
    print("Загалом:")
    print(pd.DataFrame(tot).T.to_string())
    print("\nPF по місяцях (угод):")
    by = {name: {m: f"{pf(g['ret'])} ({len(g)})" for m, g in d[d["variant"] == name].groupby("month")} for name in VARIANTS}
    print(pd.DataFrame(by).to_string())
    base = d[d["variant"] == "як зараз (ринок)"].groupby("month")["ret"].apply(pf)
    best = d[d["variant"] == "OB + ліміт"].groupby("month")["ret"].apply(pf)
    both = pd.concat([base.rename("b"), best.rename("n")], axis=1).dropna()
    print(f"\n«OB + ліміт» кращий за «як зараз» у {(both['n'] > both['b']).sum()} з {len(both)} місяців")
    print(f"Угоди: {OUT}")


if __name__ == "__main__":
    main()
