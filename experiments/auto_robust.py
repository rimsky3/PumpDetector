"""
experiments/auto_robust.py — стійкість навчання на авторозмітці (2026-10-10, після auto_train.py --train-until 2026-07-01:
auto8 PF 0.96 проти base 0.93, ≥ 20 млн 0.91 / 0.86). Усе у строгій перевірці за часом: авторозмітка лише до 28.06, бектест
лип–вер на монетах ІНШОЇ половини. Варіанти (зафіксовано ДО запуску):
    половини    навчання A → бектест B і навчання B → бектест A
    зерно       42 (поріг — найкращий F1, OOF 5 груп монет) і 43 (той самий поріг)
    поріг       свій і ±5% (зерно 42)
    ціль        auto8 (рух ≥ 8 ATR, ≥ 1.5 год) і auto15 (+ рух ≥ 15%)
base — робоча розмітка власника (одна модель для обох половин бектесту).
→ reports/auto_robust.md

    python experiments/auto_robust.py
"""

import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import numpy as np
import pandas as pd

import auto_train as AT
import early_target as ET
import exit_partial as P
import idio_features as I
import train
from tiers import tier_frame

COLS = AT.COLS
UNTIL = pd.Timestamp("2026-07-01", tz="UTC")
TARGETS = {"auto8": ("y8", "k8"), "auto15": ("y15", "k15")}
OUT = os.path.join("reports", "auto_robust.md")


def lgbm(seed: int):
    m = train.make_lightgbm()
    m.set_params(random_state=seed)
    return m


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    argparse.ArgumentParser(description="Стійкість навчання на авторозмітці: обмін половин монет, зерна, сусідні пороги (за часом)").parse_args()
    d, pumps, px, idx, uni = ET.load_data()
    trained = set(d["base_asset"])
    coins = sorted(c for c in uni if c not in trained and os.path.exists(os.path.join(I.DIR, f"{c}.pkl")))
    print(f"авторозмітка {len(coins)} монет до {UNTIL - pd.Timedelta(days=3):%Y-%m-%d}…", flush=True)
    parts = []
    for n, c in enumerate(coins, 1):
        x = pickle.load(open(os.path.join(I.DIR, f"{c}.pkl"), "rb"))
        f15, h = x["f15"], x["h"]
        f15 = f15.join(I.new_features(h, idx).reindex(f15["ts"]).set_axis(f15.index))
        f15 = f15[(f15["ts"] >= pd.Timestamp(I.START, tz="UTC")) & (f15["ts"] < UNTIL - pd.Timedelta(days=3))]
        s = AT.coin_series(c, h)
        y8, k8 = AT.labels_for(s, f15["ts"], 0.0)
        y15, k15 = AT.labels_for(s, f15["ts"], 0.15)
        parts.append(f15[["base_asset", "ts", *COLS]].assign(y8=y8, k8=k8, y15=y15, k15=k15, half=AT.half(c)))
        if n % 100 == 0:
            print(f"  {n}/{len(coins)}", flush=True)
    a = pd.concat(parts, ignore_index=True)
    del parts
    print(f"авто: {len(a)} барів", flush=True)

    Xo, yo = d[COLS], d["label"].to_numpy().astype(int)
    p = train.oof_proba(train.make_lightgbm, Xo, pd.Series(yo), d["base_asset"])
    base = (train.make_lightgbm().fit(Xo, yo), train.best_f1_threshold(yo, p))
    variants = {}                 # назва → (модель, поріг, половина навчання)
    for tgt, (yc, kc) in TARGETS.items():
        for hf in ("A", "B"):
            s = a[(a["half"] == hf) & a[kc]]
            thr = AT.gkf_threshold(s[COLS], s[yc].to_numpy(), s["base_asset"])
            m42 = lgbm(42).fit(s[COLS], s[yc].to_numpy())
            m43 = lgbm(43).fit(s[COLS], s[yc].to_numpy())
            variants[f"{tgt} {hf}→ s42"] = (m42, thr, hf)
            variants[f"{tgt} {hf}→ s43"] = (m43, thr, hf)
            variants[f"{tgt} {hf}→ поріг −5%"] = (m42, thr * 0.95, hf)
            variants[f"{tgt} {hf}→ поріг +5%"] = (m42, thr * 1.05, hf)
            print(f"  {tgt} навчання {hf}: поріг {thr:.3f}", flush=True)
    del a

    m1 = joblib.load(train.model_path("lightgbm"))
    m15 = joblib.load(train.model_path(train.M15_NAME))
    exit_bars, share = m15.get("exit_bars", 4), m15.get("confirm", {}).get("share", 0.75)
    start, end = UNTIL, pd.Timestamp(I.END, tz="UTC") + pd.Timedelta(days=1)
    print(f"бектест лип–вер: {len(coins)} монет…", flush=True)
    rows = []
    for n, c in enumerate(coins, 1):
        x = pickle.load(open(os.path.join(I.DIR, f"{c}.pkl"), "rb"))
        f15, h1, h = x["f15"], x["h1"], x["h"]
        f15 = f15.join(I.new_features(h, idx).reindex(f15["ts"]).set_axis(f15.index))
        hh = h[(h.index >= start - pd.Timedelta(days=1)) & (h.index < end)]
        s1 = h1[["base_asset", "ts"]].assign(probability=m1["model"].predict_proba(h1[m1["features"]])[:, 1], threshold=m1["threshold"])
        hf_c = AT.half(c)
        cache = {}
        todo = {"base": (*base, None)} | {k: v for k, v in variants.items() if v[2] != hf_c}
        for v, (model, thr, _) in todo.items():
            key = id(model)
            if key not in cache:
                cache[key] = model.predict_proba(f15[COLS])[:, 1]
            s15 = f15[["base_asset", "ts"]].assign(probability=cache[key], threshold=thr)
            t = tier_frame(s15, s1, share).set_index("ts").sort_index()
            for r in P.simulate(hh, t.loc[(t.index >= start) & (t.index < end), "alert_score"], I.K, exit_bars, 3.0, 0.5, True):
                mth = pd.Timestamp(r["ts"]).strftime("%Y-%m")
                if mth in uni[c] and mth >= UNTIL.strftime("%Y-%m"):
                    name = v.split(" ", 1)[0] + " " + v.split(" ", 2)[2] if v != "base" else "base"
                    rows.append({"variant": name, "test": hf_c, "ticker": c, "month": mth, "band": uni[c][mth], **r})
        if n % 100 == 0:
            print(f"  {n}/{len(coins)}", flush=True)
    tr = pd.DataFrame(rows)
    tr.to_csv(OUT.replace(".md", "_trades.csv"), index=False)
    pf = I.pf
    V = ["base"] + sorted({r for r in tr["variant"].unique() if r != "base"}, key=lambda s: (s.split()[0], s))
    L = ["# Стійкість навчання на авторозмітці (за часом: авторозмітка до 28.06, бектест лип–вер)", "",
         "Тест B — модель вчилась на монетах A, тест A — на монетах B. base — розмітка власника (серп–жовт, перекриття з бектестом "
         "на її користь). Поріг s43 = поріг s42.", "",
         "| варіант | тест B: угод | PF | ≥ 20 млн | тест A: угод | PF | ≥ 20 млн | обидва: PF | ≥ 20 млн | місяців кращих за base (≥ 20 млн) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for v in V:
        g = tr[tr["variant"] == v]
        cells = []
        for hf in ("B", "A"):
            x = g[g["test"] == hf]
            cells += [f"{len(x)}", f"{pf(x['ret']):.2f}", f"{pf(x[x['band'] == '≥ 20 млн']['ret']):.2f}"]
        big = g["band"] == "≥ 20 млн"
        b0 = tr[tr["variant"] == "base"]
        wins = sum(pf(g[(g["month"] == m) & big]["ret"]) > pf(b0[(b0["month"] == m) & (b0["band"] == "≥ 20 млн")]["ret"]) + 1e-9
                   for m in sorted(g["month"].unique()))
        wtxt = "—" if v == "base" else f"{wins} з {g['month'].nunique()}"
        L.append(f"| {v} | " + " | ".join(cells) + f" | {pf(g['ret']):.2f} | {pf(g[big]['ret']):.2f} | {wtxt} |")
    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
