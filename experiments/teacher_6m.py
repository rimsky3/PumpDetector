"""
experiments/teacher_6m.py — STUDENT (teacher + soft labels): повний 6-місячний бектест з поділом за монетами (2026-10-11; власник:
«чому не втілюємо student, якщо явне покращення?», «можемо прогнати бектест за 6 місяців цю нову модель?»).
walk-forward (teacher_combo.py) дав травень–вересень, жовтень — oct_holdout.py; тут — квітень–вересень цілком, як завжди міряли base:
student вчиться на розмітці teacher монет половини A (усі 6 міс.) і торгує половину B, потім навпаки (cross-fitting за монетами) —
кожна монета поза навчанням student. base — розмітка власника (серп–жовт; той самий формат, що в усіх 6-місячних бектестах).
Teacher — події власника (label_teacher.py), калібрована P (isotonic на OOF) → soft label бару, як teacher_combo.py (без змін).
Seeds 42 / 43 / 44 для обох. Бектест — чесний список, Alert = 15m + робоча годинна ≥ 0.75, частковий вихід, стоп I.K.
КРИТЕРІЙ (до запуску; на монетах ≥ 20 млн, які власник торгує): середнє за seeds — PF student ≥ base, і student ≥ base щонайменше
в 4 з 6 місяців. Звіт показує і 5–20 млн, і загальний, але рішення — за ≥ 20 млн.
→ reports/teacher_6m.md, reports/teacher_6m_trades.csv

    python experiments/teacher_6m.py
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
from sklearn.isotonic import IsotonicRegression

import auto_train as AT
import auto_walkforward as AW
import early_target as ET
import exit_partial as P
import idio_features as I
import label_teacher as LT
import teacher_combo as TC
import train
from tiers import tier_frame

COLS = ET.COLS
SEEDS = [42, 43, 44]
BIG, MID = "≥ 20 млн", "5–20 млн"
OUT = os.path.join("reports", "teacher_6m.md")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    argparse.ArgumentParser(description="Student (teacher + soft labels): 6 міс. бектест з поділом за монетами проти base, 3 seeds").parse_args()
    k, *_ = LT.owner_events()
    p_oof = train.oof_proba(train.make_lightgbm, k[LT.OPIS], k["y"], k["base"])
    cal = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p_oof, k["y"])
    teacher = LT.fit_teacher(k)
    d, pumps, px, idx, uni = ET.load_data()
    trained = set(d["base_asset"])
    coins = sorted(c for c in uni if c not in trained and os.path.exists(os.path.join(I.DIR, f"{c}.pkl")))
    print(f"teacher: {len(k)} подій; розмітка {len(coins)} монет…", flush=True)
    parts = []
    for n, c in enumerate(coins, 1):
        x, f15 = AW.load_coin(c, idx)
        s = AT.coin_series(c, x["h"])
        e = pd.DataFrame(LT.event_rows(s, idx.reindex(s.index)))
        if len(e):
            e["pc"] = cal.predict(teacher.predict_proba(e[LT.OPIS])[:, 1])
            pp, _, near = TC.soft_bars(f15["ts"], e)
        else:
            pp, near = np.zeros(len(f15)), np.zeros(len(f15), bool)
        keep = near | (np.arange(len(f15)) % TC.NEG_KEEP == 0)
        parts.append(f15[["base_asset", "ts", *COLS]].astype({q: "float32" for q in COLS})
                     .assign(p=pp, w=np.where(near, 1.0, float(TC.NEG_KEEP)), half=AT.half(c))[keep])
        if n % 100 == 0:
            print(f"  {n}/{len(coins)}", flush=True)
    a = pd.concat(parts, ignore_index=True)
    del parts
    models = {}
    Xo, yo = d[COLS], d["label"].to_numpy().astype(int)
    for sd in SEEDS:
        po = train.oof_proba(lambda s=sd: TC.lgbm(s), Xo, pd.Series(yo), d["base_asset"])
        models[("base", sd, None)] = (TC.lgbm(sd).fit(Xo, yo), train.best_f1_threshold(yo, po))
        for hf in ("A", "B"):
            s = a[a["half"] == hf]
            yh = (s["p"] >= 0.5).to_numpy().astype(int)
            thr = TC.gkf_threshold(s[COLS + ["p"]], yh, s["base_asset"], s["w"].to_numpy(), sd)
            Xe, ye, we = TC.expand(s[COLS + ["p"]], s["w"].to_numpy())
            models[("student", sd, hf)] = (TC.lgbm(sd).fit(Xe[COLS], ye, sample_weight=we), thr)
            print(f"  seed {sd}: student навчено на половині {hf} ({len(s)} рядків), поріг {thr:.3f}", flush=True)
    del a

    m1 = joblib.load(train.model_path("lightgbm"))
    m15 = joblib.load(train.model_path(train.M15_NAME))
    exit_bars, share = m15.get("exit_bars", 4), m15.get("confirm", {}).get("share", 0.75)
    start, end = pd.Timestamp(I.START, tz="UTC"), pd.Timestamp(I.END, tz="UTC") + pd.Timedelta(days=1)
    print(f"бектест: {len(coins)} монет…", flush=True)
    rows = []
    for n, c in enumerate(coins, 1):
        x, f15 = AW.load_coin(c, idx)
        h1, h = x["h1"], x["h"]
        hh = h[(h.index >= start - pd.Timedelta(days=1)) & (h.index < end)]
        s1 = h1[["base_asset", "ts"]].assign(probability=m1["model"].predict_proba(h1[m1["features"]])[:, 1], threshold=m1["threshold"])
        hc = AT.half(c)
        for (v, sd, hf), (mdl, thr) in models.items():
            if v == "student" and hf == hc:            # student торгує лише монети, на яких НЕ вчився
                continue
            s15 = f15[["base_asset", "ts"]].assign(probability=mdl.predict_proba(f15[COLS])[:, 1], threshold=thr)
            t = tier_frame(s15, s1, share).set_index("ts").sort_index()
            for r in P.simulate(hh, t.loc[(t.index >= start) & (t.index < end), "alert_score"], I.K, exit_bars, 3.0, 0.5, True):
                mth = pd.Timestamp(r["ts"]).strftime("%Y-%m")
                if mth in uni[c]:
                    rows.append({"variant": v, "seed": sd, "ticker": c, "half": hc, "month": mth, "band": uni[c][mth], **r})
        if n % 100 == 0:
            print(f"  {n}/{len(coins)}", flush=True)
    tr = pd.DataFrame(rows)
    tr.to_csv(OUT.replace(".md", "_trades.csv"), index=False)
    pf = I.pf
    months = sorted(tr["month"].unique())
    cell = {}
    for v in ("base", "student"):
        for sd in SEEDS:
            g = tr[(tr["variant"] == v) & (tr["seed"] == sd)]
            cell[(v, sd)] = {"all": pf(g["ret"]), "big": pf(g[g["band"] == BIG]["ret"]), "mid": pf(g[g["band"] == MID]["ret"]), "n": len(g),
                             "nbig": int((g["band"] == BIG).sum()), "big_wo3": pf(g[g["band"] == BIG]["ret"].sort_values().iloc[:-3]),
                             **{m: pf(g[(g["band"] == BIG) & (g["month"] == m)]["ret"]) for m in months},
                             **{f"h{hf}": pf(g[(g["band"] == BIG) & (g["half"] == hf)]["ret"]) for hf in ("A", "B")}}
    mean = lambda v, key: float(np.mean([cell[(v, sd)][key] for sd in SEEDS]))      # noqa: E731
    L = ["# Student (teacher + soft labels): 6 місяців, поділ за монетами", "",
         f"Квітень–вересень, {tr['ticker'].nunique()} монет чесного списку. Student вчиться на розмітці teacher однієї половини монет і торгує "
         "іншу (кожна монета — поза навчанням student). base — розмітка власника (серп–жовт), як у всіх 6-місячних бектестах. "
         f"Seeds {SEEDS}, значення — середнє.", "",
         "**Критерій (до запуску; монети ≥ 20 млн):** PF student ≥ base і student ≥ base щонайменше в 4 з 6 місяців.", "",
         "| модель | угод | з них ≥ 20 млн | PF загалом | **≥ 20 млн** | ≥ 20 млн без топ-3 | 5–20 млн | ≥ 20 млн, половина A / B |", "|---|---|---|---|---|---|---|---|"]
    for v in ("base", "student"):
        L.append(f"| {v} | {mean(v, 'n'):.0f} | {mean(v, 'nbig'):.0f} | {mean(v, 'all'):.3f} | **{mean(v, 'big'):.3f}** | {mean(v, 'big_wo3'):.3f} | "
                 f"{mean(v, 'mid'):.3f} | {mean(v, 'hA'):.3f} / {mean(v, 'hB'):.3f} |")
    L += ["", "## ≥ 20 млн по місяцях (середнє за seeds; у дужках — по seeds)", "", "| місяць | base | student | хто кращий |", "|---|---|---|---|"]
    better = 0
    for m in months:
        b, s_ = mean("base", m), mean("student", m)
        better += s_ >= b - 1e-9
        per = lambda v: " / ".join(f"{cell[(v, sd)][m]:.2f}" for sd in SEEDS)   # noqa: E731
        L.append(f"| {m} | {b:.2f} ({per('base')}) | {s_:.2f} ({per('student')}) | {'student' if s_ >= b - 1e-9 else 'base'} |")
    diffs = [cell[("student", sd)]["big"] - cell[("base", sd)]["big"] for sd in SEEDS]
    ok = mean("student", "big") >= mean("base", "big") and better >= 4
    L += ["", f"Різниця на ≥ 20 млн по seeds: {' / '.join(f'{x:+.3f}' for x in diffs)}; місяців, де student ≥ base: {better} з {len(months)}.", "",
          f"## Критерій\n\n→ **{'ВИКОНАНО' if ok else 'НЕ виконано'}**"]
    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
