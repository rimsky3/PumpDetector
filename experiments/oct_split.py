"""
experiments/oct_split.py — SPLIT_START30 НА ЖОВТНІ (holdout) (2026-10-11; власник: «а ми його вже тестували на широкому бектесті?»,
«так»). На 6 міс. архіву split_start30 стабільно кращий (≥ 20 млн 0.84 → 0.95, early_target / early_target_check; 0.87 → 0.96,
potential_target), але модель «старт» вчилась на розмітці власника серп–жовт — та сама перевага перекриття за часом, що в base.
Жовтень (01–10.10) не бачив жоден варіант; дані — data/oct_bin (oct_holdout.py --stage data).
Як early_target.py: модель «старт» — ціль «закриття бару в перші 30 хв імпульсу», решта імпульсу поза навчанням, поза
імпульсами — 0; поріг — та сама кількість хибних на 30 дн, що в base на її порозі (найкращий F1 на OOF). Обидві моделі — лише
бари до 27.09; seeds 42 / 43 / 44. Схеми: base — вхід і утримання за Alert base; split — вхід за Alert «старту», утримання —
поки горить base або «старт», вихід — 4 бари без обох (early_target.simulate_split); частковий вихід, стоп I.K, годинна робоча.
КРИТЕРІЙ (до запуску): на монетах ≥ 20 млн середній за seeds PF split ≥ base. Загальний PF і кількість угод — довідково.
→ reports/oct_split.md

    python experiments/oct_split.py
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

import early_target as ET
import idio_features as I
import oct_holdout as OH
import train
from tiers import tier_frame

COLS = ET.COLS
SEEDS = [42, 43, 44]
BIG = "≥ 20 млн"
OUT = os.path.join("reports", "oct_split.md")


def lgbm(seed: int):
    m = train.make_lightgbm()
    m.set_params(random_state=seed)
    return m


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    argparse.ArgumentParser(description="split_start30 на жовтні (holdout): вхід за моделлю «старт», утримання — base або «старт»").parse_args()
    idx_full = OH.market_index_full()
    I.market_index = lambda: idx_full
    d, pumps, px, idx, uni = ET.load_data()
    trained = set(d["base_asset"])
    do = d[d["ts"] < OH.CUT].reset_index(drop=True)
    pu = pumps[pumps["start_ts"] < OH.CUT]
    X, grp = do[COLS], do["base_asset"]
    y = do["label"].to_numpy().astype(int)
    ys, keep = ET.target(do, pu, pd.Timedelta(0), pd.Timedelta(minutes=30))
    ys = ys.astype(int)
    models = {}
    for sd in SEEDS:
        p0 = train.oof_proba(lambda s=sd: lgbm(s), X, pd.Series(y), grp)
        t0 = train.best_f1_threshold(y, p0)
        fa0 = train.event_metrics(do, p0 >= t0, pu, ET.BAR)["false_alarms_per_30d"]
        ps = np.zeros(len(do))
        for g in grp.unique():
            te = (grp == g).to_numpy()
            tr = ~te & keep
            ps[te] = lgbm(sd).fit(X[tr], ys[tr]).predict_proba(X[te])[:, 1]
        ts_ = ET.matched_threshold(do, ps, pu, fa0)
        models[("base", sd)] = (lgbm(sd).fit(X, y), t0)
        models[("start", sd)] = (lgbm(sd).fit(X[keep], ys[keep]), ts_)
        print(f"  seed {sd}: поріг base {t0:.3f} (хибних {fa0:.1f}), «старт» {ts_:.3f}", flush=True)
    m1 = joblib.load(train.model_path("lightgbm"))
    m15 = joblib.load(train.model_path(train.M15_NAME))
    exit_bars, share = m15.get("exit_bars", 4), m15.get("confirm", {}).get("share", 0.75)
    ou = json.load(open(OH.UNIVERSE, encoding="utf-8"))
    oc = sorted(c for c in ou if c not in trained and os.path.exists(os.path.join(OH.DIR, f"{c}.pkl")))
    start, end = pd.Timestamp(OH.HOLD_FROM, tz="UTC"), pd.Timestamp(OH.END, tz="UTC") + pd.Timedelta(days=1)
    rows = []
    for c in oc:
        x = pickle.load(open(os.path.join(OH.DIR, f"{c}.pkl"), "rb"))
        f15, h1, h = x["f15"], x["h1"], x["h"]
        f15 = f15.join(I.new_features(h, idx_full).reindex(f15["ts"]).set_axis(f15.index))
        hh = h[(h.index >= start - pd.Timedelta(days=1)) & (h.index < end)]
        s1 = h1[["base_asset", "ts"]].assign(probability=m1["model"].predict_proba(h1[m1["features"]])[:, 1], threshold=m1["threshold"])

        def alert(mdl, thr):
            s15 = f15[["base_asset", "ts"]].assign(probability=mdl.predict_proba(f15[COLS])[:, 1], threshold=thr)
            t = tier_frame(s15, s1, share).set_index("ts").sort_index()
            return t.loc[(t.index >= start - pd.Timedelta(days=1)) & (t.index < end), "alert_score"]
        for sd in SEEDS:
            a0 = alert(*models[("base", sd)])
            ast = alert(*models[("start", sd)]).reindex(a0.index)
            for v, (en, ho) in {"base": (a0, a0), "split": (ast, np.maximum(a0, ast))}.items():
                for r in ET.simulate_split(hh, en, ho, I.K, exit_bars, 3.0, 0.5, True):
                    if pd.Timestamp(r["ts"]) >= start:
                        rows.append({"variant": v, "seed": sd, "ticker": c, "band": ou[c], **r})
    tr = pd.DataFrame(rows)
    tr.to_csv(OUT.replace(".md", "_trades.csv"), index=False)
    pf = I.pf
    res = {}
    for v in ("base", "split"):
        for sd in SEEDS:
            g = tr[(tr["variant"] == v) & (tr["seed"] == sd)]
            b = g[g["band"] == BIG]
            res[(v, sd)] = {"n": len(g), "nb": len(b), "pf": pf(g["ret"]), "pf20": pf(b["ret"]), "wo3": pf(b["ret"].sort_values().iloc[:-3]),
                            "med": b["ret"].median()}
    mean = lambda v, k: float(np.mean([res[(v, sd)][k] for sd in SEEDS]))      # noqa: E731
    ok = mean("split", "pf20") >= mean("base", "pf20")
    L = ["# split_start30 на жовтні (holdout)", "",
         f"01–10.10, {len(oc)} монет жовтня поза тікерами власника (≥ 20 млн — {sum(ou[c] == BIG for c in oc)}). Обидві моделі — лише бари до "
         f"{OH.CUT:%d.%m}; seeds {SEEDS}.", "",
         "**Критерій (до запуску):** на ≥ 20 млн середній PF split ≥ base.", "",
         "| схема | seed | угод | з них ≥ 20 млн | PF | **≥ 20 млн** | ≥ 20 млн без топ-3 | медіана ≥ 20 млн |", "|---|---|---|---|---|---|---|---|"]
    for (v, sd), r in res.items():
        L.append(f"| {v} | {sd} | {r['n']} | {r['nb']} | {r['pf']:.2f} | **{r['pf20']:.2f}** | {r['wo3']:.2f} | {r['med']:+.2%} |")
    L += ["", "| схема (середнє) | угод | PF | ≥ 20 млн |", "|---|---|---|---|"]
    for v in ("base", "split"):
        L.append(f"| {v} | {mean(v, 'n'):.0f} | {mean(v, 'pf'):.3f} | {mean(v, 'pf20'):.3f} |")
    L += ["", f"## Критерій\n\n→ **{'ВИКОНАНО' if ok else 'НЕ виконано'}**"]
    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
