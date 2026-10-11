"""
experiments/purged_student.py — STUDENT (teacher + soft labels): PURGED CROSS-VALIDATION ЗА ЧАСОВИМИ БЛОКАМИ (2026-10-11; власник:
«так запускай» — разом з purged_split.py). Блоки ті самі: серпень, вересень, 01–10.10; тестовий місяць вирізано з навчання з
запасом 3 дні з кожного боку.
Для кожного блоку: teacher — лише події власника поза блоком (калібрування isotonic на OOF); описи подій монет архіву
(label_teacher.event_rows) рахуються ОДИН раз, teacher кожного блоку лише переоцінює їх; student — soft labels, бари архіву поза
блоком, negative downsampling ×3 з вагою 3, поріг — найкращий F1 на OOF 3 груп монет (як teacher_combo.py); seeds 42 / 43 / 44.
base — з reports/purged_split_trades.csv (та сама модель і ті самі seeds, ті самі монети й правила угод).
КРИТЕРІЙ (до запуску): на ≥ 20 млн PF student ≥ base за всі блоки разом (середнє за seeds) і ≥ base щонайменше в 2 з 3 блоків.
→ reports/purged_student.md

    python experiments/purged_split.py       # спершу (base)
    python experiments/purged_student.py
"""

import argparse
import json
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
import idio_features as I
import label_teacher as LT
import oct_holdout as OH
import purged_split as PS
import teacher_combo as TC
import train
from tiers import tier_frame

COLS = ET.COLS
SEEDS = PS.SEEDS
BIG = PS.BIG
OUT = os.path.join("reports", "purged_student.md")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    argparse.ArgumentParser(description="Student (teacher + soft labels): purged cross-validation за часовими блоками проти base з purged_split.py").parse_args()
    base_tr = pd.read_csv(PS.OUT.replace(".md", "_trades.csv"))
    base_tr = base_tr[base_tr["variant"] == "base"]
    idx_full = OH.market_index_full()
    I.market_index = lambda: idx_full
    k_all, *_ = LT.owner_events()
    d, pumps, px, idx, uni = ET.load_data()
    trained = set(d["base_asset"])
    coins = sorted(c for c in uni if c not in trained and os.path.exists(os.path.join(I.DIR, f"{c}.pkl")))
    print(f"описи подій {len(coins)} монет архіву…", flush=True)
    bars, events = {}, {}
    for n, c in enumerate(coins, 1):
        x, f15 = AW.load_coin(c, idx_full)
        s = AT.coin_series(c, x["h"])
        events[c] = pd.DataFrame(LT.event_rows(s, idx_full.reindex(s.index)))
        bars[c] = f15[["base_asset", "ts", *COLS]].astype({q: "float32" for q in COLS})
        if n % 100 == 0:
            print(f"  {n}/{len(coins)}", flush=True)
    m1 = joblib.load(train.model_path("lightgbm"))
    m15 = joblib.load(train.model_path(train.M15_NAME))
    exit_bars, share = m15.get("exit_bars", 4), m15.get("confirm", {}).get("share", 0.75)
    ou = json.load(open(OH.UNIVERSE, encoding="utf-8"))
    rows = []
    for blk, (a, b, src) in PS.BLOCKS.items():
        start, end = pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC")
        lo, hi = start - PS.EMB, end + PS.EMB
        k = k_all[(k_all["end_ts"] <= lo) | (k_all["start"] >= hi)].reset_index(drop=True)
        p_oof = train.oof_proba(train.make_lightgbm, k[LT.OPIS], k["y"], k["base"])
        cal = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p_oof, k["y"])
        teacher = LT.fit_teacher(k)
        parts = []
        for c in coins:
            f = bars[c]
            out = ((f["ts"] < lo) | (f["ts"] >= hi)).to_numpy()
            e = events[c]
            e = e[(e["end_ts"] <= lo) | (e["start"] >= hi)] if len(e) else e
            if len(e):
                e = e.assign(pc=cal.predict(teacher.predict_proba(e[LT.OPIS])[:, 1]))
                pp, _, near = TC.soft_bars(f["ts"], e)
            else:
                pp, near = np.zeros(len(f)), np.zeros(len(f), bool)
            keep = out & (near | (np.arange(len(f)) % TC.NEG_KEEP == 0))
            parts.append(f.assign(p=pp, w=np.where(near, 1.0, float(TC.NEG_KEEP)))[keep])
        a_ = pd.concat(parts, ignore_index=True)
        yh = (a_["p"] >= 0.5).to_numpy().astype(int)
        models = {}
        for sd in SEEDS:
            thr = TC.gkf_threshold(a_[COLS + ["p"]], yh, a_["base_asset"], a_["w"].to_numpy(), sd)
            Xe, ye, we = TC.expand(a_[COLS + ["p"]], a_["w"].to_numpy())
            models[sd] = (TC.lgbm(sd).fit(Xe[COLS], ye, sample_weight=we), thr)
        print(f"{blk}: teacher {len(k)} подій; student {len(a_)} рядків; пороги " + ", ".join(f"{models[sd][1]:.3f}" for sd in SEEDS), flush=True)
        del a_, parts
        if src == "arch":
            tcoins = sorted(c for c in coins if blk in uni[c])
            band = {c: uni[c][blk] for c in tcoins}
            path = lambda c: os.path.join(I.DIR, f"{c}.pkl")          # noqa: E731
        else:
            tcoins = sorted(c for c in ou if c not in trained and os.path.exists(os.path.join(OH.DIR, f"{c}.pkl")))
            band = ou
            path = lambda c: os.path.join(OH.DIR, f"{c}.pkl")         # noqa: E731
        for c in tcoins:
            x = pickle.load(open(path(c), "rb"))
            f15, h1, h = x["f15"], x["h1"], x["h"]
            f15 = f15[(f15["ts"] >= start - pd.Timedelta(days=2)) & (f15["ts"] < end)]
            if f15.empty or h1.empty:
                continue
            f15 = f15.join(I.new_features(h, idx_full).reindex(f15["ts"]).set_axis(f15.index))
            hh = h[(h.index >= start - pd.Timedelta(days=1)) & (h.index < end)]
            s1 = h1[["base_asset", "ts"]].assign(probability=m1["model"].predict_proba(h1[m1["features"]])[:, 1], threshold=m1["threshold"])
            for sd, (mdl, thr) in models.items():
                s15 = f15[["base_asset", "ts"]].assign(probability=mdl.predict_proba(f15[COLS])[:, 1], threshold=thr)
                t = tier_frame(s15, s1, share).set_index("ts").sort_index()
                sc = t.loc[(t.index >= start - pd.Timedelta(days=1)) & (t.index < end), "alert_score"]
                for r in ET.simulate_split(hh, sc, sc, I.K, exit_bars, 3.0, 0.5, True):
                    if start <= pd.Timestamp(r["ts"]) < end:
                        rows.append({"block": blk, "variant": "student", "seed": sd, "ticker": c, "band": band[c], **r})
        print(f"  {blk}: бектест {len(tcoins)} монет", flush=True)
    tr = pd.concat([base_tr, pd.DataFrame(rows)], ignore_index=True)
    tr.to_csv(OUT.replace(".md", "_trades.csv"), index=False)
    pf = I.pf
    big = tr[tr["band"] == BIG]
    cell = lambda df, v: float(np.mean([pf(df[(df["variant"] == v) & (df["seed"] == sd)]["ret"]) for sd in SEEDS]))   # noqa: E731
    L = ["# Student (teacher + soft labels): purged cross-validation за часовими блоками", "",
         "Тестовий місяць вирізано з навчання (і teacher, і student) з запасом 3 дні; base — з purged_split.py (ті самі seeds і правила). "
         f"Seeds {SEEDS}, значення — середнє.", "",
         "**Критерій (до запуску):** на ≥ 20 млн PF student ≥ base за всі блоки разом і ≥ base щонайменше в 2 з 3 блоків.", "",
         "| блок | угод ≥ 20 млн base / student | **≥ 20 млн base** | **≥ 20 млн student** | загальний base | загальний student |", "|---|---|---|---|---|---|"]
    wins = 0
    for blk in PS.BLOCKS:
        g, gb = tr[tr["block"] == blk], big[big["block"] == blk]
        nb = lambda v: np.mean([len(gb[(gb["variant"] == v) & (gb["seed"] == sd)]) for sd in SEEDS])   # noqa: E731
        bb, bs = cell(gb, "base"), cell(gb, "student")
        wins += bs >= bb - 1e-9
        L.append(f"| {blk} | {nb('base'):.0f} / {nb('student'):.0f} | {bb:.3f} | {bs:.3f} | {cell(g, 'base'):.3f} | {cell(g, 'student'):.3f} |")
    tb, ts_ = cell(big, "base"), cell(big, "student")
    L.append(f"| **разом** | — | **{tb:.3f}** | **{ts_:.3f}** | {cell(tr, 'base'):.3f} | {cell(tr, 'student'):.3f} |")
    ok = ts_ >= tb and wins >= 2
    L += ["", f"## Критерій\n\n→ **{'ВИКОНАНО' if ok else 'НЕ виконано'}** (блоків, де student ≥ base: {wins} з {len(PS.BLOCKS)})"]
    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
