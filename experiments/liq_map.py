"""
experiments/liq_map.py — «паливо» ліквідацій попереду ціни як фільтр угод моделі (2026-10-09, власник: «сильний рух можливий,
коли ціна пробиває рівні, за якими стоять стопи»; реалізовані ліквідації — уже спалене паливо, тут — ще не спалене).

Оцінка карти ліквідацій (як теплові карти Coinglass / Hyblock) з архіву Binance, 5m: OI у монетах (metrics, знімок +5 хв —
binance_archive) і частка покупок-агресорів (klines taker_buy). Параметри зафіксовано ДО запуску:
    • OI за 5 хв зріс → нові позиції; лонги = приріст × частка покупок-агресорів бару, шорти = приріст × (1 − частка);
    • плече порівну 10× / 25× / 50×: ліквідація шорта = ціна × (1 + 1/плече), лонга = ціна × (1 − 1/плече);
    • ціна дійшла до рівня (хай бару ≥ рівень шортів / лоу ≤ рівень лонгів) — паливо згоріло, рівень обнуляється;
    • OI впав — уся карта пропорційно менша; ще згасання з напівперіодом 3 доби (позиції закривають і без руху ціни);
    • сітка рівнів — крок 0.25% (лог-ціна); розігрів карти — 7 днів (угоди раніше не беруться).
Ознаки на вході (закриття 15m-бару ts): паливо шортів 0–3% / 3–8% над ціною і лонгів 0–3% / 3–8% під ціною (частки від OI),
співвідношення шорти / лонги в межах 8%, пробій хаю 24 год / 7 днів. Угоди — reports/trade_patterns.csv (ті самі 20 701 угода
робочої моделі, 6 міс., чесний список). Перевірка: квінтилі (межі з квіт–черв), PF окремо квіт–черв (A) і лип–вер (B);
варіанти: A — паливо шортів над ціною, Б — пробій хаю + паливо шортів, В — паливо лонгів під ціною (ризик зливу).
→ reports/liq_map.md, reports/liq_map.csv.

    python experiments/liq_map.py [--workers 6]
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
TRADES = os.path.join(ROOT, "reports", "trade_patterns.csv")
OUT_MD = os.path.join(ROOT, "reports", "liq_map.md")
OUT_CSV = os.path.join(ROOT, "reports", "liq_map.csv")
LEV = (10, 25, 50)
STEP = np.log(1.0025)                      # сітка рівнів — 0.25%
HALF_LIFE_BARS = 3 * 288                   # згасання карти: 3 доби 5m-барів
WARMUP = pd.Timedelta(days=7)
B15 = pd.Timedelta(minutes=15)
SPLIT = pd.Timestamp("2026-07-01", tz="UTC")
FEATS = {
    "short_0_3": "паливо шортів 0–3% над ціною, частка OI",
    "short_3_8": "паливо шортів 3–8% над ціною, частка OI",
    "long_0_3": "паливо лонгів 0–3% під ціною, частка OI",
    "long_3_8": "паливо лонгів 3–8% під ціною, частка OI",
    "fuel_ratio": "паливо шортів / лонгів у межах 8%",
}


def pf(r: pd.Series) -> float:
    w, l = r[r > 0].sum(), -r[r <= 0].sum()
    return w / l if l > 0 else np.nan


def load(ticker: str) -> pd.DataFrame | None:
    sym = f"{ticker}USDT"
    kf = sorted(glob.glob(os.path.join(ARCH, "monthly", "klines", sym, "5m", "*.parquet")))
    mf = sorted(glob.glob(os.path.join(ARCH, "metrics", sym, "*.parquet")))
    if not kf or not mf:
        return None
    k = pd.concat([pd.read_parquet(f) for f in kf], ignore_index=True).drop_duplicates("open_time")
    k.index = pd.to_datetime(k["open_time"].astype("int64"), unit="ms", utc=True)
    k = k.sort_index()[["high", "low", "close", "quote_volume", "taker_buy_quote_volume"]].astype(float)
    m = pd.concat([pd.read_parquet(f) for f in mf], ignore_index=True)
    # знімок архіву на 5 хв раніше за мітку API (binance_archive): мітка t = стан на t = кінець 5m-бару, що відкрився о t − 5 хв
    t = pd.to_datetime(m["create_time"], utc=True) + pd.Timedelta(minutes=5) - pd.Timedelta(minutes=5)
    oi = pd.Series(m["sum_open_interest"].astype(float).to_numpy(), index=t)
    oi = oi[oi > 0]
    oi = oi[~oi.index.duplicated()].sort_index()
    k["oi"] = oi.reindex(k.index)
    return k


def run_coin(args) -> list[dict]:
    ticker, trades = args
    k = load(ticker)
    if k is None or k["oi"].notna().sum() < 1000:
        return []
    hi, lo, c = k["high"].to_numpy(), k["low"].to_numpy(), k["close"].to_numpy()
    qv, tb = k["quote_volume"].to_numpy(), k["taker_buy_quote_volume"].to_numpy()
    oi = k["oi"].ffill().to_numpy()
    lp = np.log(np.concatenate([lo[lo > 0], hi[hi > 0]]))
    g0 = int(np.floor((lp.min() - np.log(1.6)) / STEP))         # запас: рівні лонгів під мінімумом, шортів над максимумом
    n = int(np.ceil((lp.max() + np.log(1.6)) / STEP)) - g0 + 2
    idx = lambda p: int(np.round(np.log(p) / STEP)) - g0        # noqa: E731
    sh, lg = np.zeros(n), np.zeros(n)          # паливо шортів (рівні над ціною входу) і лонгів (під), у монетах / scale
    scale, decay = 1.0, 0.5 ** (1 / HALF_LIFE_BARS)
    # моменти входу: закриття 15m-бару ts = кінець 5m-бару, що відкрився о ts + 10 хв
    want = {t + B15 - pd.Timedelta(minutes=5): t for t in trades["ts"]}
    pos = {k.index.get_loc(t5): t for t5, t in want.items() if t5 in k.index}
    start = k.index[0] + WARMUP
    # пробій хаю 24 год / 7 днів (лише попередні бари, без бару входу)
    h15 = k["high"].resample("15min", label="left").max()
    hi24, hi7 = h15.shift(1).rolling(96, min_periods=90).max(), h15.shift(1).rolling(96 * 7, min_periods=96 * 6).max()
    out = []
    for i in range(1, len(k)):
        if not (c[i] > 0 and hi[i] > 0 and lo[i] > 0):
            continue
        # 1) паливо, до якого дійшла ціна, згоріло
        sh[: idx(hi[i]) + 1] = 0.0
        lg[idx(lo[i]):] = 0.0
        # 2) згасання + пропорційне закриття, коли OI падає
        scale *= decay
        d_oi = oi[i] - oi[i - 1] if np.isfinite(oi[i]) and np.isfinite(oi[i - 1]) else 0.0
        if d_oi < 0 and oi[i - 1] > 0:
            scale *= max(0.0, 1 + d_oi / oi[i - 1])
        if scale < 1e-60:                    # перенормування, щоб не зникнути в нулі float
            sh *= scale; lg *= scale; scale = 1.0  # noqa: E702
        # 3) нові позиції, коли OI росте
        if d_oi > 0:
            b = tb[i] / qv[i] if qv[i] > 0 else 0.5
            for L in LEV:
                sh[idx(c[i] * (1 + 1 / L))] += d_oi * (1 - b) / len(LEV) / scale
                lg[idx(c[i] * (1 - 1 / L))] += d_oi * b / len(LEV) / scale
        if i in pos and k.index[i] >= start and oi[i] > 0:
            t, p, j = pos[i], c[i], idx(c[i])
            s = lambda a, x, y: a[max(0, x):max(0, y)].sum() * scale / oi[i]   # noqa: E731
            r = {"ticker": ticker, "ts": t,
                 "short_0_3": s(sh, j + 1, idx(p * 1.03) + 1), "short_3_8": s(sh, idx(p * 1.03) + 1, idx(p * 1.08) + 1),
                 "long_0_3": s(lg, idx(p * 0.97), j), "long_3_8": s(lg, idx(p * 0.92), idx(p * 0.97))}
            r["fuel_ratio"] = (r["short_0_3"] + r["short_3_8"]) / max(r["long_0_3"] + r["long_3_8"], 1e-9)
            r["brk24"] = bool(p >= hi24.get(t, np.inf))
            r["brk7d"] = bool(p >= hi7.get(t, np.inf))
            out.append(r)
    return out


def quint_table(a: pd.DataFrame, f: str, desc: str, lines: list, verdict: list) -> None:
    x = a.dropna(subset=[f])
    edges = np.unique(np.nanquantile(x[x.half == "A"][f], [0.2, 0.4, 0.6, 0.8]))
    x = x.assign(q=pd.cut(x[f], [-np.inf, *edges, np.inf], labels=[f"Q{i + 1}" for i in range(len(edges) + 1)]))
    lines += [f"### {desc} (`{f}`)", "", "| група | межі | угод A / B | PF A | PF B | медіана ret |", "|---|---|---|---|---|---|"]
    res = []
    for q, g in x.groupby("q", observed=True):
        ga, gb = g[g.half == "A"], g[g.half == "B"]
        lines.append(f"| {q} | {g[f].min():.3g} … {g[f].max():.3g} | {len(ga)} / {len(gb)} | {pf(ga['ret']):.2f} | "
                     f"{pf(gb['ret']):.2f} | {g['ret'].median():+.2%} |")
        res.append((str(q), pf(ga["ret"]), pf(gb["ret"]), len(g)))
    pa, pb = pf(x[x.half == "A"]["ret"]), pf(x[x.half == "B"]["ret"])
    for q, qa, qb, n in (res[0], res[-1]):
        if qa - pa >= 0.10 and qb - pb >= 0.10:
            verdict.append(f"✓ **{desc}: {q} КРАЩЕ** — PF A {qa:.2f} (усе {pa:.2f}), B {qb:.2f} (усе {pb:.2f}), {n} угод")
        elif pa - qa >= 0.10 and pb - qb >= 0.10:
            verdict.append(f"✓ **{desc}: {q} ГІРШЕ** — PF A {qa:.2f} (усе {pa:.2f}), B {qb:.2f} (усе {pb:.2f}), {n} угод")
    lines.append("")


def main() -> None:
    ap = argparse.ArgumentParser(description="Карта ліквідацій з OI архіву Binance: паливо над / під ціною на вході угод моделі")
    ap.add_argument("--workers", type=int, default=6, help="процесів (за замовчуванням 6)")
    args = ap.parse_args()
    t = pd.read_csv(TRADES)
    t["ts"] = pd.to_datetime(t["ts"], utc=True)
    jobs = [(tk, g[["ts"]]) for tk, g in t.groupby("ticker")]
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for i, r in enumerate(ex.map(run_coin, jobs, chunksize=4)):
            rows += r
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(jobs)} монет", flush=True)
    f = pd.DataFrame(rows)
    a = t.merge(f, on=["ticker", "ts"], how="inner")
    a.to_csv(OUT_CSV, index=False)
    lines = ["# Карта ліквідацій: паливо попереду ціни на вході угод моделі", "",
             f"{len(a)} з {len(t)} угод мають карту (розігрів 7 днів, OI в архіві). A = квітень–червень, B = липень–вересень. "
             f"PF усіх: A {pf(a[a.half == 'A']['ret']):.2f} · B {pf(a[a.half == 'B']['ret']):.2f}.", "",
             "Параметри зафіксовано до запуску: частка нових лонгів = частка покупок-агресорів 5m-бару, плече 10/25/50× порівну, "
             "паливо згорає, коли ціна доходить до рівня, згасання 3 доби.", ""]
    verdict, body = [], ["## А і В: паливо над / під ціною (квінтилі, межі з A)", ""]
    for k_, d in FEATS.items():
        quint_table(a, k_, d, body, verdict)
    # Б: пробій хаю + паливо шортів (верхні 2 квінтилі short_0_3 + short_3_8 за межами з A)
    a["short_0_8"] = a["short_0_3"] + a["short_3_8"]
    thr = np.nanquantile(a[a.half == "A"]["short_0_8"], 0.6)
    body += ["## Б: пробій хаю + паливо шортів над ним", "",
             f"Багато палива = паливо шортів 0–8% у верхніх 40% (поріг {thr:.3g} частки OI, з половини A).", "",
             "| пробій | паливо | угод A / B | PF A | PF B | медіана ret |", "|---|---|---|---|---|---|"]
    for brk in ("brk24", "brk7d"):
        for b_ in (True, False):
            for fuel in (True, False):
                g = a[(a[brk] == b_) & ((a["short_0_8"] >= thr) == fuel)]
                ga, gb = g[g.half == "A"], g[g.half == "B"]
                body.append(f"| {'хай 24 год' if brk == 'brk24' else 'хай 7 днів'} {'так' if b_ else 'ні'} | "
                            f"{'багато' if fuel else 'мало'} | {len(ga)} / {len(gb)} | {pf(ga['ret']):.2f} | {pf(gb['ret']):.2f} | "
                            f"{g['ret'].median():+.2%} |")
    body.append("")
    lines += ["## Що повторилось в обох половинах", ""] + (verdict or ["Жоден крайній квінтиль не відрізняється в обох половинах."]) + [""] + body
    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(verdict) or "нічого стабільного")
    print(f"→ {OUT_MD}")


if __name__ == "__main__":
    main()
