"""
experiments/range_breakout.py — «справжні імпульси часто виходять з тривалих рейнджів / проторговок» (2026-10-09, власник,
приклади STRK 04.10 і MAGIC 09.10: тиждень-два вузького боковика з рівним OI → пробій).

Вікно рейнджу — 48 год (і 96 год), що закінчується за 4 год до входу (останні години — уже сам імпульс). Ознаки (до запуску):
    range_48 / range_96   — (хай − лоу) за вікно у % ціни;
    range_x48 / range_x96 — те саме ÷ медіана таких діапазонів за попередні 30 днів (стиснення відносно власної норми);
    er_48                 — коефіцієнт ефективності Кауфмана годинних закриттів (|зміна| ÷ сума |кроків|; мало = боковик);
    vol_quiet48           — обсяг за вікно ÷ медіана 48-год обсягу за 30 днів (тиша);
    oi_flat48             — |зміна OI у монетах за вікно| (архів metrics, 5 хв);
    breakout48            — закриття бару входу вище хаю вікна;
    setup                 — стиснення (range_x48 у нижніх 40%, межа з квіт–черв) + пробій.
Перевірка 1: угоди робочої моделі reports/long_binance_trades.csv (20 701, 6 міс. архіву, чесний список), квінтилі з межами
з квіт–черв, PF окремо квіт–черв (A) і лип–вер (B). Перевірка 2: старти імпульсів власника (pumps.csv, pump) проти
фальстартів (false_start) — ті самі ознаки на момент старту (лише монети з архівом).
→ reports/range_breakout.md, reports/range_breakout.csv

    python experiments/range_breakout.py [--workers 6]
"""

import argparse
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
ARCH = os.path.join(ROOT, "data", "binance_archive")
TRADES = os.path.join(ROOT, "reports", "long_binance_trades.csv")
OUT_MD = os.path.join(ROOT, "reports", "range_breakout.md")
OUT_CSV = os.path.join(ROOT, "reports", "range_breakout.csv")
B = pd.Timedelta(minutes=15)
GAP, W1, W2 = 4, 48, 96          # годин: відступ від входу, вікна
SPLIT = pd.Timestamp("2026-07-01", tz="UTC")
FEATS = {
    "range_48": "ширина рейнджу 48 год, % ціни",
    "range_96": "ширина рейнджу 96 год, % ціни",
    "range_x48": "ширина 48 год ÷ норма монети (30 дн)",
    "range_x96": "ширина 96 год ÷ норма монети (30 дн)",
    "er_48": "«боковість»: ефективність Кауфмана 48 год (мало = боковик)",
    "vol_quiet48": "обсяг за 48 год ÷ норма (30 дн)",
    "oi_flat48": "|зміна OI у монетах| за 48 год",
}


def pf(r: pd.Series) -> float:
    w, l = r[r > 0].sum(), -r[r <= 0].sum()
    return w / l if l > 0 else np.nan


def hourly(ticker: str) -> pd.DataFrame | None:
    sym = f"{ticker}USDT"
    kf = sorted(glob.glob(os.path.join(ARCH, "monthly", "klines", sym, "15m", "*.parquet")))
    if not kf:
        return None
    k = pd.concat([pd.read_parquet(f) for f in kf], ignore_index=True).drop_duplicates("open_time")
    k.index = pd.to_datetime(k["open_time"].astype("int64"), unit="ms", utc=True)
    k = k.sort_index()[["high", "low", "close", "quote_volume"]].astype(float)
    h = k.resample("1h").agg({"high": "max", "low": "min", "close": "last", "quote_volume": "sum"})
    h.index = h.index + pd.Timedelta(hours=1)                     # кінець години
    mf = sorted(glob.glob(os.path.join(ARCH, "metrics", sym, "*.parquet")))
    if mf:
        m = pd.concat([pd.read_parquet(f, columns=["create_time", "sum_open_interest"]) for f in mf], ignore_index=True)
        oi = pd.Series(m["sum_open_interest"].astype(float).to_numpy(),
                       index=pd.to_datetime(m["create_time"], utc=True) + pd.Timedelta(minutes=5))
        oi = oi[oi > 0]
        oi = oi[~oi.index.duplicated()].sort_index()
        h["oi"] = oi.reindex(h.index, method="ffill", tolerance=pd.Timedelta(minutes=30))
    else:
        h["oi"] = np.nan
    return h


def window_features(h: pd.DataFrame) -> pd.DataFrame:
    """Ознаки на кінець кожної години t: вікно [t − GAP − W, t − GAP)."""
    f = pd.DataFrame(index=h.index)
    hi_s, lo_s, c, v = h["high"].shift(GAP), h["low"].shift(GAP), h["close"].shift(GAP), h["quote_volume"].shift(GAP)
    for W in (W1, W2):
        hh, ll = hi_s.rolling(W).max(), lo_s.rolling(W).min()
        f[f"range_{W}"] = (hh / ll - 1) * 100
        f[f"hi_{W}"] = hh
        ref = f[f"range_{W}"].shift(W).rolling(30 * 24, min_periods=10 * 24).median()
        f[f"range_x{W}"] = f[f"range_{W}"] / ref
    steps = c.diff().abs().rolling(W1).sum()
    f["er_48"] = (c - c.shift(W1)).abs() / steps
    v48 = v.rolling(W1).sum()
    f["vol_quiet48"] = v48 / v48.shift(W1).rolling(30 * 24, min_periods=10 * 24).median()
    oi = h["oi"].shift(GAP)
    f["oi_flat48"] = (oi / oi.shift(W1) - 1).abs() * 100
    return f


def run_coin(args) -> list[dict]:
    ticker, moments = args                        # moments: [(key, момент, ціна закриття або None)]
    h = hourly(ticker)
    if h is None:
        return []
    f = window_features(h)
    out = []
    for key, t, px in moments:
        hr = pd.Timestamp(t).floor("h")           # остання закрита година на момент t
        if hr not in f.index:
            continue
        r = f.loc[hr]
        price = px if px is not None else h["close"].get(hr, np.nan)
        out.append({"key": key, **{k: r[k] for k in FEATS}, "breakout48": bool(price > r["hi_48"]) if np.isfinite(r["hi_48"]) else None})
    return out


def quint(x: pd.DataFrame, f: str, desc: str, lines: list, verdict: list) -> None:
    x = x.dropna(subset=[f])
    edges = np.unique(np.nanquantile(x[x.half == "A"][f], [0.2, 0.4, 0.6, 0.8]))
    x = x.assign(q=pd.cut(x[f], [-np.inf, *edges, np.inf], labels=[f"Q{i + 1}" for i in range(len(edges) + 1)]))
    lines += [f"### {desc} (`{f}`)", "", "| група | межі | угод A / B | PF A | PF B | медіана ret |", "|---|---|---|---|---|---|"]
    res = []
    for q, g in x.groupby("q", observed=True):
        ga, gb = g[g.half == "A"], g[g.half == "B"]
        lines.append(f"| {q} | {g[f].min():.3g} … {g[f].max():.3g} | {len(ga)} / {len(gb)} | {pf(ga['ret']):.2f} | {pf(gb['ret']):.2f} | "
                     f"{g['ret'].median():+.2%} |")
        res.append((str(q), pf(ga["ret"]), pf(gb["ret"]), len(g)))
    pa, pb = pf(x[x.half == "A"]["ret"]), pf(x[x.half == "B"]["ret"])
    for q, qa, qb, n in (res[0], res[-1]):
        if qa - pa >= 0.10 and qb - pb >= 0.10:
            verdict.append(f"✓ **{desc}: {q} КРАЩЕ** — PF A {qa:.2f} (усе {pa:.2f}), B {qb:.2f} (усе {pb:.2f}), {n} угод")
        elif pa - qa >= 0.10 and pb - qb >= 0.10:
            verdict.append(f"✓ **{desc}: {q} ГІРШЕ** — PF A {qa:.2f} (усе {pa:.2f}), B {qb:.2f} (усе {pb:.2f}), {n} угод")
    lines.append("")


def breakout_events(sym_dir: str) -> list[dict]:
    """Без моделі: перше закриття години вище хаю 96 год (одна подія на 2 доби) і що було далі за 24 год."""
    tk = os.path.basename(sym_dir)[:-4]
    h = hourly(tk)
    if h is None or len(h) < 24 * 40:
        return []
    c, hi, lo, v = h["close"], h["high"], h["low"], h["quote_volume"]
    hh96, ll96 = hi.shift(1).rolling(W2).max(), lo.shift(1).rolling(W2).min()
    rng = (hh96 / ll96 - 1) * 100
    rng_x = rng / rng.shift(W2).rolling(30 * 24, min_periods=10 * 24).median()
    er = (c.shift(1) - c.shift(W2 + 1)).abs() / c.shift(1).diff().abs().rolling(W2).sum()
    vol_x = v / v.shift(1).rolling(7 * 24, min_periods=72).median()
    oi4 = (h["oi"] / h["oi"].shift(4) - 1) * 100
    brk = (c > hh96) & (c.shift(1) <= hh96.shift(1))
    out, last = [], None
    for t in h.index[brk.fillna(False).to_numpy()]:
        if last is not None and t - last < pd.Timedelta(hours=48):
            continue
        last = t
        i = h.index.get_loc(t)
        if i + 24 >= len(h):
            continue
        p0, fw = c.iloc[i], h.iloc[i + 1:i + 25]
        out.append({"ticker": tk, "t": t, "rng96": rng.iloc[i], "rng_x96": rng_x.iloc[i], "er96": er.iloc[i], "vol_x": vol_x.iloc[i],
                    "oi4": oi4.iloc[i], "mfe24": fw["high"].max() / p0 - 1, "mae24": fw["low"].min() / p0 - 1,
                    "r4": fw["close"].iloc[3] / p0 - 1, "r24": fw["close"].iloc[-1] / p0 - 1})
    return out


def events_report(workers: int) -> None:
    syms = sorted(glob.glob(os.path.join(ARCH, "monthly", "klines", "*USDT")))
    rows = []
    with ProcessPoolExecutor(workers) as ex:
        for r in ex.map(breakout_events, syms, chunksize=8):
            rows += r
    e = pd.DataFrame(rows)
    e["half"] = np.where(e["t"] < SPLIT, "A", "B")
    e.to_csv(os.path.join(ROOT, "reports", "range_breakout_events.csv"), index=False)
    e = e.dropna(subset=["rng_x96", "er96"])
    L = ["# Пробої хаю 96 год: з вузького рейнджу проти решти (без моделі)", "",
         f"{len(e)} пробоїв на {e['ticker'].nunique()} монетах архіву Binance (квіт–вер; перше закриття години вище хаю 96 год, "
         "одна подія на 2 доби). Результат — від закриття години пробою: MFE і MAE за 24 год, ціна через 4 і 24 год. "
         "Межі квінтилів — з квіт–черв (A), B = лип–вер.", ""]
    sel = lambda m: [(h, e[m & (e["half"] == h)]) for h in "AB"]            # noqa: E731
    row = lambda lab, h, g: (f"| {lab} | {h} | {len(g)} | {g['mfe24'].median():+.1%} | {(g['mfe24'] >= 0.2).mean():.1%} | "     # noqa: E731
                             f"{g['r4'].median():+.2%} | {g['r24'].median():+.2%} | {g['mae24'].median():+.1%} |")
    head = ["| група | половина | пробоїв | MFE 24 год, медіана | MFE ≥ +20% | ціна +4 год | ціна +24 год | MAE 24 год |",
            "|---|---|---|---|---|---|---|---|"]
    for col, desc in (("rng_x96", "ширина рейнджу 96 год ÷ норма (Q1 — найвужчий)"), ("er96", "Кауфман 96 год (Q1 — найбільший боковик)")):
        edges = np.nanquantile(e[e["half"] == "A"][col], [0.2, 0.4, 0.6, 0.8])
        q = pd.cut(e[col], [-np.inf, *edges, np.inf], labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
        L += [f"## {desc}", ""] + head
        for qq in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
            L += [row(qq, h, g) for h, g in sel(q == qq)]
        L.append("")
    tight = (e["rng_x96"] <= np.nanquantile(e[e["half"] == "A"]["rng_x96"], 0.2)) & \
            (e["er96"] <= np.nanquantile(e[e["half"] == "A"]["er96"], 0.4))
    fuel = (e["vol_x"] >= 3) & (e["oi4"] > 0)
    L += ["## Сетап власника: довгий вузький боковик + пробій", ""] + head
    for lab, m in (("вузький боковик (Q1 ширини + нижні 40% Кауфмана)", tight), ("… + обсяг ≥ ×3 і OI ↑ на пробої", tight & fuel),
                   ("решта пробоїв + обсяг ≥ ×3 і OI ↑", ~tight & fuel), ("решта пробоїв", ~tight)):
        L += [row(lab, h, g) for h, g in sel(m)]
    L.append("")
    with open(os.path.join(ROOT, "reports", "range_breakout_events.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L[-7:]))


def main() -> None:
    ap = argparse.ArgumentParser(description="Імпульси з рейнджу: стиснення, боковик, тиша, рівний OI перед входом моделі і стартом імпульсу")
    ap.add_argument("--workers", type=int, default=6, help="процесів (за замовчуванням 6)")
    ap.add_argument("--events", action="store_true",
                    help="без моделі: усі пробої хаю 96 год на монетах архіву — з вузького рейнджу проти решти → reports/range_breakout_events.md")
    args = ap.parse_args()
    if args.events:
        events_report(args.workers)
        return
    from load_pumps import read_pumps
    t = pd.read_csv(TRADES)
    t["ts"] = pd.to_datetime(t["ts"], utc=True)
    t["entry"] = t["ts"] + B
    t["key"] = "t" + t.index.astype(str)
    p = pd.DataFrame([r[:4] for r in read_pumps()], columns=["ticker", "start", "end", "kind"])
    p = p[p["kind"].isin(["pump", "false_start"])].reset_index(drop=True)
    p["start"] = pd.to_datetime(p["start"], utc=True)
    p["key"] = "p" + p.index.astype(str)
    jobs = {}
    for r in t.itertuples():
        jobs.setdefault(r.ticker, []).append((r.key, r.entry, None))
    for r in p.itertuples():
        jobs.setdefault(r.ticker, []).append((r.key, r.start, None))
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for i, res in enumerate(ex.map(run_coin, list(jobs.items()), chunksize=4)):
            rows += res
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(jobs)} монет", flush=True)
    f = pd.DataFrame(rows).set_index("key")
    a = t.join(f, on="key", how="inner")
    a["half"] = np.where(a["entry"] < SPLIT, "A", "B")
    thr = np.nanquantile(a[a.half == "A"]["range_x48"], 0.4)
    a["setup"] = (a["range_x48"] <= thr) & (a["breakout48"] == True)                 # noqa: E712
    a.drop(columns=["key"]).to_csv(OUT_CSV, index=False)
    L = ["# Імпульси з рейнджу", "",
         f"Угоди моделі з ознаками: {len(a)} з {len(t)}. A = квітень–червень, B = липень–вересень. PF усіх: A {pf(a[a.half == 'A']['ret']):.2f} · "
         f"B {pf(a[a.half == 'B']['ret']):.2f}. Вікно — {W1} / {W2} год, що закінчується за {GAP} год до входу.", ""]
    verdict, body = [], []
    for k, d in FEATS.items():
        quint(a, k, d, body, verdict)
    body += ["### Пробій хаю рейнджу і сетап «стиснення + пробій»", "", "| умова | угод A / B | PF A | PF B | медіана ret |", "|---|---|---|---|---|"]
    for lab, m in ((f"вхід вище хаю 48 год", a["breakout48"] == True), ("вхід не вище хаю", a["breakout48"] == False),   # noqa: E712
                   (f"сетап: стиснення (range_x48 ≤ {thr:.2f}) + пробій", a["setup"]), ("решта", ~a["setup"])):
        g = a[m]
        body.append(f"| {lab} | {(g.half == 'A').sum()} / {(g.half == 'B').sum()} | {pf(g[g.half == 'A']['ret']):.2f} | "
                    f"{pf(g[g.half == 'B']['ret']):.2f} | {g['ret'].median():+.2%} |")
    body.append("")
    # перевірка 2: старти імпульсів власника проти фальстартів
    q = p.join(f, on="key", how="inner")
    L2 = ["## Розмітка власника: старти імпульсів проти фальстартів (медіани)", "",
          f"Імпульсів з ознаками {(q.kind == 'pump').sum()} з {(p.kind == 'pump').sum()}, фальстартів {(q.kind == 'false_start').sum()} з "
          f"{(p.kind == 'false_start').sum()} (лише монети з архівом Binance).", "",
          "| ознака | імпульси | фальстарти | угоди моделі (усі) |", "|---|---|---|---|"]
    for k, d in FEATS.items():
        L2.append(f"| {d} | {q[q.kind == 'pump'][k].median():.3g} | {q[q.kind == 'false_start'][k].median():.3g} | {a[k].median():.3g} |")
    L2.append(f"| старт вище хаю 48 год, частка | {q[q.kind == 'pump']['breakout48'].mean():.0%} | {q[q.kind == 'false_start']['breakout48'].mean():.0%} | "
              f"{a['breakout48'].mean():.0%} |")
    L2.append("")
    L += ["## Що повторилось в обох половинах (угоди моделі)", ""] + (verdict or ["Жоден крайній квінтиль не відрізняється в обох половинах."]) + [""]
    L += L2 + ["## Угоди моделі за квінтилями (межі з A)", ""] + body
    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(verdict) or "нічого стабільного")
    print(f"→ {OUT_MD}")


if __name__ == "__main__":
    main()
