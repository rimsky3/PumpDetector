"""
experiments/label_noise.py — шум у негативах навчання (2026-10-10, з experiments/auto_label.py). Тікери з tickers.txt вважаються
переглянутими цілком: усі нерозмічені бари йдуть у навчання як «не імпульс». Авторозмітка (тригер T1 + клас заднім числом)
знайшла на них великі рухи БЕЗ розмітки власника — переважно до першої / після останньої його мітки на тікері (≥ 20%: 52 події,
~2 300 барів у навчанні як «0», при ~2 800 позитивних барах). Варіанти (зафіксовано ДО запуску):
    base      як зараз
    clean20   бари авто-імпульсів ≥ 20% без розмітки власника на переглянутих тікерах — поза навчанням (не «1» і не «0»)
    clean10   те саме для ≥ 10%
Оцінка LTO — на тих самих рядках для всіх варіантів (бари clean10 виключено з оцінки всіх); бектест 6 міс. — як ensemble.py.
→ reports/label_noise.md

    python experiments/label_noise.py
"""

import argparse
import hashlib
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import numpy as np
import pandas as pd

import early_target as ET
import exit_partial as P
import idio_features as I
import train
from fetch import read_tickers_file
from tiers import tier_frame

EVENTS = os.path.join("reports", "auto_label_events.csv")
VARIANTS = {"base": None, "clean20": 0.20, "clean10": 0.10}
OUT = os.path.join("reports", "label_noise.md")


def noisy_mask(d: pd.DataFrame, pumps: pd.DataFrame, min_move: float) -> np.ndarray:
    """Бари d (переглянуті тікери) усередині авто-імпульсів ≥ min_move, що не перетинають розмітку власника."""
    ev = pd.read_csv(EVENTS, parse_dates=["start", "end_ts"])
    ev = ev[(ev["trig"] == "T1") & (ev["cls"] == "pump") & (ev["move_pct"] >= min_move)]
    reviewed = {t.upper() for t in read_tickers_file("tickers.txt")}
    close = d["ts"] + ET.BAR
    m = np.zeros(len(d), bool)
    for r in ev[ev["base"].isin(reviewed)].itertuples():
        own = pumps[pumps["base_asset"] == r.base]
        if ((own["start_ts"] < r.end_ts) & (own["end_ts"] > r.start)).any():
            continue
        m |= ((d["base_asset"] == r.base) & (close > r.start) & (close <= r.end_ts)).to_numpy()
    return m & (d["label"].to_numpy() == 0)


def oof(X, y, keep, grp):
    p = np.zeros(len(y))
    for g in grp.unique():
        test = (grp == g).to_numpy()
        tr = ~test & keep
        p[test] = train.make_lightgbm().fit(X[tr], y[tr]).predict_proba(X[test])[:, 1]
    return p


def half(c: str) -> str:
    return "A" if int(hashlib.md5(c.encode()).hexdigest(), 16) % 2 == 0 else "B"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    argparse.ArgumentParser(description="Шум у негативах: виключити нерозмічені великі рухи переглянутих тікерів з навчання").parse_args()
    d, pumps, px, idx, uni = ET.load_data()
    X, y, grp = d[ET.COLS], d["label"].to_numpy().astype(int), d["base_asset"]
    masks = {v: (noisy_mask(d, pumps, mm) if mm else np.zeros(len(d), bool)) for v, mm in VARIANTS.items()}
    ev_rows = ~masks["clean10"]
    de = d[ev_rows].reset_index(drop=True)
    print(f"навчання: {len(d)} барів; поза навчанням clean20 {masks['clean20'].sum()}, clean10 {masks['clean10'].sum()}; "
          f"позитивних барів {y.sum()}", flush=True)
    thr, lto, models = {}, {}, {}
    for v in VARIANTS:
        keep = ~masks[v]
        p = oof(X, y, keep, grp)
        thr[v] = train.best_f1_threshold(y[keep], p[keep])
        sig = (p >= thr[v])[ev_rows]
        lto[v] = {**train.event_metrics(de, sig, pumps, ET.BAR), **ET.early(de, sig, pumps, px)}
        models[v] = train.make_lightgbm().fit(X[keep], y[keep])
        print(f"  {v} (поріг {thr[v]:.3f}): {lto[v]}", flush=True)

    m1 = joblib.load(train.model_path("lightgbm"))
    m15 = joblib.load(train.model_path(train.M15_NAME))
    exit_bars, share = m15.get("exit_bars", 4), m15.get("confirm", {}).get("share", 0.75)
    start, end = pd.Timestamp(I.START, tz="UTC"), pd.Timestamp(I.END, tz="UTC") + pd.Timedelta(days=1)
    trained = set(d["base_asset"])
    coins = sorted(c for c in uni if c not in trained and os.path.exists(os.path.join(I.DIR, f"{c}.pkl")))
    print(f"бектест: {len(coins)} монет…", flush=True)
    rows = []
    for n, c in enumerate(coins, 1):
        x = pickle.load(open(os.path.join(I.DIR, f"{c}.pkl"), "rb"))
        f15, h1, h = x["f15"], x["h1"], x["h"]
        f15 = f15.join(I.new_features(h, idx).reindex(f15["ts"]).set_axis(f15.index))
        hh = h[(h.index >= start - pd.Timedelta(days=1)) & (h.index < end)]
        s1 = h1[["base_asset", "ts"]].assign(probability=m1["model"].predict_proba(h1[m1["features"]])[:, 1], threshold=m1["threshold"])
        for v in VARIANTS:
            s15 = f15[["base_asset", "ts"]].assign(probability=models[v].predict_proba(f15[ET.COLS])[:, 1], threshold=thr[v])
            t = tier_frame(s15, s1, share).set_index("ts").sort_index()
            for r in P.simulate(hh, t.loc[(t.index >= start) & (t.index < end), "alert_score"], I.K, exit_bars, 3.0, 0.5, True):
                mth = pd.Timestamp(r["ts"]).strftime("%Y-%m")
                if mth in uni[c]:
                    rows.append({"variant": v, "ticker": c, "month": mth, "band": uni[c][mth], **r})
        if n % 50 == 0:
            print(f"  {n}/{len(coins)}", flush=True)
    tr = pd.DataFrame(rows)
    tr["coins"] = tr["ticker"].map(half)
    pf, V = I.pf, list(VARIANTS)
    L = ["# Шум у негативах: нерозмічені великі рухи переглянутих тікерів", "",
         f"Поза навчанням: clean20 — {masks['clean20'].sum()} барів, clean10 — {masks['clean10'].sum()} (позитивних барів {y.sum()}). "
         "Пороги — найкращий F1 на OOF: " + ", ".join(f"{k} {v:.3f}" for k, v in thr.items()) + ".", "",
         "## Leave-ticker-out (Кандидат 15m; оцінка на тих самих рядках — без барів clean10)", "",
         "| варіант | спіймано | затримка, хв | пройдено руху | фальстартів | хибних / 30 дн |", "|---|---|---|---|---|---|"]
    for v in V:
        m = lto[v]
        L.append(f"| {v} | {m['caught']} | {m['delay_min']:.0f} | {m['done']:.0%} | {m['false_starts_triggered']} | {m['false_alarms_per_30d']:.1f} |")
    L += ["", f"## Бектест 6 міс., {tr['ticker'].nunique()} монет чесного списку", "",
          "| варіант | угод | PF | без топ-3 | ≥ 20 млн | 5–20 млн | монети A | монети B | квіт–черв | лип–вер |", "|---|---|---|---|---|---|---|---|---|---|"]
    for v in V:
        g = tr[tr["variant"] == v]
        r = g["ret"].sort_values()
        a = g["month"] <= "2026-06"
        L.append(f"| {v} | {len(g)} | {pf(r):.2f} | {pf(r.iloc[:-3]):.2f} | {pf(g[g['band'] == '≥ 20 млн']['ret']):.2f} | "
                 f"{pf(g[g['band'] == '5–20 млн']['ret']):.2f} | {pf(g[g['coins'] == 'A']['ret']):.2f} | {pf(g[g['coins'] == 'B']['ret']):.2f} | "
                 f"{pf(g[a]['ret']):.2f} | {pf(g[~a]['ret']):.2f} |")
    piv20 = tr[tr["band"] == "≥ 20 млн"].groupby(["month", "variant"])["ret"].apply(pf).unstack()[V]
    piv = tr.groupby(["month", "variant"])["ret"].apply(pf).unstack()[V]
    L += ["", "Місяців кращих за base: " + ", ".join(
        f"{v} {int((piv[v] > piv['base'] + 1e-9).sum())} / ≥ 20 млн {int((piv20[v] > piv20['base'] + 1e-9).sum())}"
        for v in V if v != "base") + f" (з {len(piv)})", ""]
    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
