"""
experiments/trade_count.py — кількість угод як ознака (2026-10-07).

Питання власника: чи дає щось КІЛЬКІСТЬ угод (а не лише обсяг у $)? Гіпотеза: імпульс із натовпом дрібних угод
(багато tx, малий середній розмір) і імпульс кількох великих гравців (мало tx, великий розмір) поводяться по-різному;
частка угод-купівель (btx/tx) — «агресія покупців», незалежна від розміру угод.

Джерела:
  * 1h — Coinalyze ohlcv.tx / btx (є в базі для всіх бірж, досі не використовувались);
  * 15m — Binance klines, поле «кількість угод» (r[8]); Bybit і OKX кількості не дають. exchanges.binance_klines його
    відкидає, тому тут — свій запит, кеш data/trades15m/<монета>.parquet. Кількості купівель Binance не дає.

Ознаки (усі відносні до історії тікера, лише дані до бару включно):
  tx_z_7d        — z-score log(угод за бар) проти 7 днів;
  tx_ratio_24h   — угод за добу / середня доба за попередні 7 днів;
  avg_trade_z_7d — z-score log(середній розмір угоди = обсяг / угоди) проти 7 днів (на тих самих ринках);
  tx_burst       — угод за останню годину / медіана годин за попередню добу;
  btx_ratio_1h / btx_ratio_4h — частка угод-купівель (лише 1h, Coinalyze).

Порівняння: base проти base + tx. 1h — leave-ticker-out годинної моделі (FEATURES з таблиці features).
15m — leave-ticker-out «Кандидат» + бектест на монетах поза навчанням (Alert = 15m + робоча 1h; поточний і частковий вихід).

    python experiments/trade_count.py [--tf 1h|15m|both] [--no-backtest]
"""

import argparse
import glob
import json
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import numpy as np
import pandas as pd
import psycopg
from dotenv import load_dotenv

import train
from exchanges import BINANCE, get, to_ts
from features import (FEATURES, FEATURES_15M, INTERVAL_15M, aggregate, build_from_raw, build_hourly_own, load_pumps,
                      load_raw, query_df, short_features, zscore)
from fetch import read_tickers_file
from tiers import tier_frame

# tabulate у .venv немає — таблиці у звіті простим текстом
pd.DataFrame.to_markdown = lambda self, *a, **k: "```\n" + self.to_string() + "\n```"

DAYS = 30
CACHE = os.path.join("data", "trades15m")
OUT = os.path.join("reports", "trade_count.md")
TX_15M = ["tx_z_7d", "tx_ratio_24h", "avg_trade_z_7d", "tx_burst"]
TX_1H = TX_15M + ["btx_ratio_1h", "btx_ratio_4h"]


def tx_features(tx: pd.Series, vol: pd.Series, b: int, btx: pd.Series | None = None) -> pd.DataFrame:
    """Ознаки кількості угод на ряд одного тікера. b — барів на годину. vol — обсяг $ тих самих ринків."""
    base, day = 7 * 24 * b, 24 * b
    tx = tx.where(tx > 0)
    f = pd.DataFrame(index=tx.index)
    f["tx_z_7d"] = zscore(np.log1p(tx), base)
    f["tx_ratio_24h"] = tx.rolling(day).sum() / (tx.shift(day).rolling(base).sum() / 7)
    f["avg_trade_z_7d"] = zscore(np.log(vol / tx), base)
    h1 = tx.rolling(b).sum()
    f["tx_burst"] = h1 / h1.shift(b).rolling(day).median()
    if btx is not None:
        f["btx_ratio_1h"] = btx.rolling(b).sum() / tx.rolling(b).sum()
        f["btx_ratio_4h"] = btx.rolling(4 * b).sum() / tx.rolling(4 * b).sum()
    return f.replace([np.inf, -np.inf], np.nan)


# ---------------------------------------------------------------- джерела

def coinalyze_tx(conn, bases: list[str]) -> pd.DataFrame:
    """1h: сума угод / купівель по всіх ринках тікера; обсяг $ — лише ринків, де tx є."""
    o = query_df(conn, """
        SELECT m.base_asset, t.ts, t.v, t.c, t.tx, t.btx, m.volume_denominated
        FROM ohlcv t JOIN markets m USING (symbol)
        WHERE t.interval = '1hour' AND m.base_asset = ANY(%s) AND t.tx IS NOT NULL""", (bases,))
    o["ts"] = pd.to_datetime(o["ts"], utc=True)
    o["v_usd"] = np.where(o["volume_denominated"] == "BASE_ASSET", o["v"] * o["c"], o["v"])
    g = o.groupby(["base_asset", "ts"])[["tx", "btx", "v_usd"]].sum(min_count=1).reset_index()
    out = []
    for b, s in g.groupby("base_asset"):
        s = s.set_index("ts").sort_index()
        s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="h", name="ts"))
        out.append(tx_features(s["tx"], s["v_usd"], 1, s["btx"]).assign(base_asset=b).reset_index())
    return pd.concat(out, ignore_index=True)


def binance_symbol(conn, base: str) -> str | None:
    r = conn.execute("""SELECT symbol_on_exchange FROM markets WHERE base_asset = %s AND exchange_name = 'Binance'
                        AND quote_asset = 'USDT' ORDER BY symbol LIMIT 1""", (base,)).fetchone()
    return r[0] if r else None


def binance_trades_15m(conn, base: str, since: pd.Timestamp, until: pd.Timestamp) -> pd.DataFrame:
    """15m-свічки Binance: кількість угод і обсяг $ (кеш; дотягує лише те, чого бракує)."""
    path = os.path.join(CACHE, f"{base}.parquet")
    have = pd.read_parquet(path) if os.path.exists(path) else pd.DataFrame(columns=["n", "q"])
    if len(have) and have.index.min() <= since + pd.Timedelta(hours=1) and have.index.max() >= until - pd.Timedelta(minutes=30):
        return have
    sym = binance_symbol(conn, base)
    if not sym:
        return have
    rows, t, end = [], int(since.timestamp() * 1000), int(until.timestamp() * 1000)
    while t < end:
        page = get(BINANCE + "/fapi/v1/klines", dict(symbol=sym, interval="15m", startTime=t, endTime=end, limit=1500))
        if not page:
            break
        rows += page
        t = page[-1][0] + 15 * 60_000
        time.sleep(0.15)
    if not rows:
        return have
    # [open_time, o, h, l, c, volume, close_time, quote_volume, trades, ...]
    new = pd.DataFrame({"n": [float(r[8]) for r in rows], "q": [float(r[7]) for r in rows]},
                       index=pd.DatetimeIndex([to_ts(r[0]) for r in rows], name="ts"))
    out = pd.concat([have, new]).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    os.makedirs(CACHE, exist_ok=True)
    out.to_parquet(path)
    return out


def tx15(conn, base: str, since, until) -> pd.DataFrame:
    k = binance_trades_15m(conn, base, since, until)
    if k.empty:
        return pd.DataFrame(columns=TX_15M + ["base_asset", "ts"])
    k = k.astype(float)            # з порожнього кешу стовпці приходять як object
    k = k.reindex(pd.date_range(k.index.min(), k.index.max(), freq="15min", name="ts"))
    return tx_features(k["n"], k["q"], 4).assign(base_asset=base).reset_index()


# ---------------------------------------------------------------- метрики

def lto(d: pd.DataFrame, variants: dict, pumps, bar) -> tuple[dict, dict]:
    y, groups = d["label"].astype(int), d["base_asset"]
    rows, models = {}, {}
    for name, cols in variants.items():
        print(f"  {name}: {len(cols)} ознак, leave-ticker-out…", flush=True)
        p = train.oof_proba(train.make_lightgbm, d[cols], y, groups)
        thr = train.best_f1_threshold(y.to_numpy(), p)
        ev = train.event_metrics(d, p >= thr, pumps, bar)
        sig = p >= thr
        tp = int((sig & (y == 1)).sum())
        prec, rec = tp / max(sig.sum(), 1), tp / max(y.sum(), 1)
        rows[name] = {"поріг": round(thr, 3), "F1 (бари)": round(2 * prec * rec / max(prec + rec, 1e-9), 3),
                      "спіймано": ev["pumps_caught"],
                      "затримка, хв": round(ev["median_delay_h"] * 60) if pd.notna(ev["median_delay_h"]) else None,
                      "фальстарти": ev["false_starts_triggered"], "хибних / 30 д": round(ev["false_alarms_per_30d"], 1)}
        models[name] = (train.make_lightgbm().fit(d[cols], y), thr, cols)
    return rows, models


def importance(model, cols, mark) -> list[str]:
    imp = pd.Series(model.booster_.feature_importance("gain"), index=cols)
    imp = (imp / imp.sum()).sort_values(ascending=False)
    return [f"- {f}: {v:.1%}" + (" ← угоди" if f in mark else "") for f, v in imp.head(15).items()] + \
           [f"- (усі ознаки угод разом: {imp[[c for c in mark if c in imp]].sum():.1%})"]


def _pf(v: pd.Series) -> str:
    gl = -v[v <= 0].sum()
    return f"{v[v > 0].sum() / gl:.2f}" if gl else "∞"


def stats(df: pd.DataFrame) -> dict:
    v = df["ret"].sort_values()
    if v.empty:
        return {"угод": 0}
    return {"угод": len(v), "у плюсі": f"{(v > 0).mean():.0%}", "медіана": f"{v.median():+.2%}", "без топ-3": f"{v.iloc[:-3].sum():+.0%}",
            "PF": _pf(v), "PF ≥ 20 млн": _pf(df[df["band"] == "≥ 20 млн"]["ret"]),
            "PF 5–20 млн": _pf(df[df["band"] == "5–20 млн"]["ret"]), "по стопу": f"{df['stop'].mean():.0%}"}


def keep_training(d: pd.DataFrame, reviewed: set, cutoff) -> pd.DataFrame:
    d = d[d["base_asset"].isin(reviewed) | d["pattern_type"].notna()]
    return d[(d["ts"] <= cutoff) | d["pattern_type"].notna()].reset_index(drop=True)


# ---------------------------------------------------------------- основне

def run_1h(conn, reviewed, pumps, cutoff, lines):
    print("1h: таблиця features + tx/btx Coinalyze…", flush=True)
    d = query_df(conn, "SELECT * FROM features ORDER BY base_asset, ts")
    d["ts"] = pd.to_datetime(d["ts"], utc=True)
    d = d[(d["ts"] <= cutoff) | d["pattern_type"].notna()].reset_index(drop=True)
    tx = coinalyze_tx(conn, sorted(d["base_asset"].unique()))
    d = d.merge(tx, on=["base_asset", "ts"], how="left")
    cover = d[TX_1H[0]].notna().mean()
    print(f"  {len(d)} годин, {d['base_asset'].nunique()} тікерів, ознаки угод є на {cover:.0%} годин", flush=True)
    # як змінюються ознаки угод на старті імпульсу (перша година) проти звичайних годин
    first = d[(d["label"] == 1) & (d.groupby("base_asset")["label"].shift(1) == 0)]
    prof = pd.DataFrame({"звичайні години": d[d["label"] == 0][TX_1H].median(), "перша година імпульсу": first[TX_1H].median(),
                         "усі години імпульсу": d[d["label"] == 1][TX_1H].median()}).round(3)
    rows, models = lto(d, {"base": FEATURES, "base + угоди": FEATURES + TX_1H, "base + угоди без btx": FEATURES + TX_15M},
                       pumps, train.H)
    lines += ["## 1h (Coinalyze tx / btx, усі біржі)", "", f"{len(d)} годин, ознаки угод є на {cover:.0%}.", "",
              "Медіани ознак угод:", "", prof.to_markdown(), "", pd.DataFrame(rows).T.to_markdown(), "",
              "Важливість (base + угоди, gain), топ-15:", ""] + importance(*models["base + угоди"][::2], TX_1H) + [""]
    print(prof.to_string(), "\n", pd.DataFrame(rows).T.to_string(), flush=True)


def run_15m(conn, reviewed, pumps, cutoff, lines, backtest: bool):
    trained = reviewed | set(pumps["base_asset"])
    raw = load_raw(conn, sorted(trained), INTERVAL_15M)
    d = build_from_raw(raw, pumps, bars_per_hour=4, extra=lambda h, b: short_features(h, b))
    d = keep_training(d, reviewed, cutoff)
    lo, hi = d["ts"].min() - pd.Timedelta(days=8), d["ts"].max() + pd.Timedelta(minutes=15)
    print(f"15m: угоди Binance для {d['base_asset'].nunique()} тікерів навчання…", flush=True)
    tx = pd.concat([tx15(conn, b, lo, hi) for b in sorted(d["base_asset"].unique())], ignore_index=True)
    d = d.merge(tx, on=["base_asset", "ts"], how="left")
    d[TX_15M] = d[TX_15M].astype(float)
    cover = d[TX_15M[0]].notna().mean()
    print(f"  {len(d)} барів, ознаки угод є на {cover:.0%}", flush=True)
    first = d[(d["label"] == 1) & (d.groupby("base_asset")["label"].shift(1) == 0)]
    prof = pd.DataFrame({"звичайні бари": d[d["label"] == 0][TX_15M].median(), "перший бар імпульсу": first[TX_15M].median(),
                         "усі бари імпульсу": d[d["label"] == 1][TX_15M].median()}).round(3)
    variants = {"base": FEATURES_15M, "base + угоди": FEATURES_15M + TX_15M}
    rows, models = lto(d, variants, pumps, train.BAR_15M)
    lines += ["## 15m (Binance, кількість угод)", "", f"{len(d)} барів, ознаки угод є на {cover:.0%}.", "",
              "Медіани ознак угод:", "", prof.to_markdown(), "", "Leave-ticker-out (рівень «Кандидат»):", "",
              pd.DataFrame(rows).T.to_markdown(), "", "Важливість (base + угоди, gain), топ-15:", ""] + \
             importance(*models["base + угоди"][::2], TX_15M) + [""]
    print(prof.to_string(), "\n", pd.DataFrame(rows).T.to_string(), flush=True)
    if not backtest:
        return

    import exit_partial as P
    m15 = joblib.load(train.model_path(train.M15_NAME))
    m1 = joblib.load(train.model_path("lightgbm"))
    k, exit_bars, share = m15.get("risk", {}).get("k"), m15.get("exit_bars", 4), m15.get("confirm", {}).get("share", 0.75)
    coins, band = set(), {}
    for path in glob.glob(os.path.join("reports", "backtests", "2026*_alert.json")):
        meta = json.load(open(path, encoding="utf-8"))["meta"]
        for u in meta["universe"]:
            coins.add(u["base"])
            band[u["base"]] = "≥ 20 млн" if meta["min_turnover"] >= 20e6 else "5–20 млн"
    coins = sorted(coins - trained)
    end = pd.Timestamp(conn.execute("SELECT max(ts) FROM ohlcv WHERE interval = '15min'").fetchone()[0])
    start = end - pd.Timedelta(days=DAYS)
    since = start - pd.Timedelta(days=9)
    res = {f"{name} · {e}": [] for name in variants for e in ("поточний вихід", "частковий вихід")}
    print(f"бектест: {len(coins)} монет поза навчанням, {start:%m-%d}…{end:%m-%d}", flush=True)
    for n, b in enumerate(coins, 1):
        raw = load_raw(conn, [b], INTERVAL_15M, since=since.to_pydatetime())
        f15 = build_from_raw(raw, None, bars_per_hour=4, extra=lambda h, bb: short_features(h, bb))
        d1 = build_hourly_own(conn, [b], since=since.to_pydatetime())
        if f15.empty or d1.empty:
            continue
        f15 = f15.merge(tx15(conn, b, since, end + pd.Timedelta(minutes=15)), on=["base_asset", "ts"], how="left")
        f15[TX_15M] = f15[TX_15M].astype(float)       # монета без Binance — порожні стовпці типу object
        s1 = d1[["base_asset", "ts"]].assign(probability=m1["model"].predict_proba(d1[FEATURES])[:, 1], threshold=m1["threshold"])
        h = aggregate(raw, b, "15min")
        h = h[h.index >= start - pd.Timedelta(days=1)]
        for name, (model, thr, cols) in models.items():
            s15 = f15[["base_asset", "ts"]].assign(probability=model.predict_proba(f15[cols])[:, 1], threshold=thr)
            t = tier_frame(s15, s1, share).set_index("ts").sort_index()
            sc = t[t.index >= start]["alert_score"]
            for x in P.simulate(h, sc, k, exit_bars, None, 0, False):
                res[f"{name} · поточний вихід"].append({"ticker": b, "ret": x["ret"], "stop": x["reason"] == "stop", "band": band.get(b, "")})
            for x in P.simulate(h, sc, k, exit_bars, 3.0, 0.5, True):
                res[f"{name} · частковий вихід"].append({"ticker": b, "ret": x["ret"], "stop": x["reason"] == "stop", "band": band.get(b, "")})
        if n % 25 == 0:
            print(f"  {n}/{len(coins)}", flush=True)
    bt = pd.DataFrame({name: stats(pd.DataFrame(rows)) if rows else {"угод": 0} for name, rows in res.items()}).T
    lines += [f"Бектест: {len(coins)} монет поза навчанням, {start:%m-%d}…{end:%m-%d}:", "", bt.to_markdown(), ""]
    print(bt.to_string(), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Кількість угод (Coinalyze 1h tx/btx, Binance 15m) як ознаки моделі")
    ap.add_argument("--tf", choices=["1h", "15m", "both"], default="both", help="який таймфрейм перевіряти")
    ap.add_argument("--no-backtest", action="store_true", help="15m: лише leave-ticker-out, без бектесту")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(".env")
    reviewed = {t.upper() for t in read_tickers_file("tickers.txt")}
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=48)
    pd.set_option("display.width", 220)
    lines = [f"# Кількість угод як ознака ({datetime.now(timezone.utc):%Y-%m-%d})", ""]
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        pumps = load_pumps(conn)
        if args.tf in ("1h", "both"):
            run_1h(conn, reviewed, pumps, cutoff, lines)
        if args.tf in ("15m", "both"):
            run_15m(conn, reviewed, pumps, cutoff, lines, not args.no_backtest)
    os.makedirs("reports", exist_ok=True)
    open(OUT, "w", encoding="utf-8").write("\n".join(lines))
    print(f"\nзвіт: {OUT}")


if __name__ == "__main__":
    main()
