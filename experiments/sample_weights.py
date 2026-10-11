"""
experiments/sample_weights.py — ВАГА КОЖНОГО ІМПУЛЬСУ = 1 і РЕГУЛЯРИЗАЦІЯ LightGBM (2026-10-11, план дня п. 1; ідеї з розбору
notes/noise_reduction.md; власник: «давай з 1 по 3»).
Гіпотеза: імпульс на 20 год дає моделі 80 позитивних 15m-барів, імпульс на 1 год — 4. Довгі розвинуті тренди важать у навчанні
в ~20 разів більше → модель найвпевненіша «посеред тренду», звідси «найвищі P — найгірші угоди» (signal_ranking).
Sample weights за унікальністю (López de Prado, «Advances in Financial ML», гл. 4): кожен імпульс отримує однакову сумарну вагу.
Варіанти (зафіксовано ДО запуску; кожен на random seeds 42 / 43 / 44, поріг — найкращий F1 на OOF свого seed):
    base        як зараз (class_weight="balanced", без ваг)
    w_event     ваги позитивних барів = L̄ / L (L — бари імпульсу, L̄ — середнє), сума ваг позитивів та сама; негативи — 1
    reg         регуляризація: min_child_samples 100, extra_trees, path_smooth 10 (class_weight="balanced" лишається)
    spw8        scale_pos_weight 8 замість class_weight="balanced" (~35×)
    w_event_reg w_event + reg
LTO (Кандидат 15m: спіймано, затримка, пройдено руху, фальстарти, хибні) і бектест 6 міс. на чесному списку (Alert = 15m + робоча
годинна ≥ 0.75, частковий вихід, стоп I.K), половини монет; PF за квінтилями оцінки на вході (чи зникає «найвищі P — найгірші»).
КРИТЕРІЙ (до запуску), для кожного варіанта проти base, у парах за seed: на ≥ 20 млн середня різниця PF ≥ +0.02 і краще на ≥ 2 з 3
seeds; середній загальний PF ≥ base; на обох половинах монет середній PF ≥ base.
→ reports/sample_weights.md, reports/sample_weights_trades.csv

    python experiments/sample_weights.py
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

COLS = ET.COLS
SEEDS = [42, 43, 44]
REG = {"min_child_samples": 100, "extra_trees": True, "path_smooth": 10.0}
VARIANTS = {"base": ({}, False), "w_event": ({}, True), "reg": (REG, False),
            "spw8": ({"class_weight": None, "scale_pos_weight": 8.0}, False), "w_event_reg": (REG, True)}
BIG = "≥ 20 млн"
OUT = os.path.join("reports", "sample_weights.md")


def make(params: dict, seed: int):
    m = train.make_lightgbm()
    m.set_params(random_state=seed, **params)
    return m


def event_weights(d: pd.DataFrame, pumps: pd.DataFrame) -> np.ndarray:
    """Вага позитивного бару = L̄ / L його імпульсу; негативи — 1 (сума ваг позитивів не змінюється)."""
    w = np.ones(len(d))
    close = d["ts"] + ET.BAR
    ids = np.full(len(d), -1)
    for k, r in enumerate(pumps[pumps["pattern_type"] == "pump"].itertuples()):
        m = ((d["base_asset"] == r.base_asset) & (close > r.start_ts) & (close <= r.end_ts)).to_numpy() & (d["label"].to_numpy() == 1)
        ids[m & (ids < 0)] = k
    pos = ids >= 0
    n = pd.Series(ids[pos]).value_counts()
    L = pd.Series(ids[pos]).map(n).to_numpy()
    w[pos] = n.mean() / L
    other = (d["label"].to_numpy() == 1) & ~pos          # позитиви поза знайденими інтервалами (не повинно бути) — вага 1
    w[other] = 1.0
    return w


def oof(params: dict, seed: int, X, y, grp, w) -> np.ndarray:
    p = np.zeros(len(y))
    for g in grp.unique():
        te = (grp == g).to_numpy()
        p[te] = make(params, seed).fit(X[~te], y[~te], sample_weight=w[~te]).predict_proba(X[te])[:, 1]
    return p


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    argparse.ArgumentParser(description="Вага кожного імпульсу = 1 і регуляризація LightGBM: LTO і бектест 6 міс. на 3 random seeds").parse_args()
    d, pumps, px, idx, uni = ET.load_data()
    trained = set(d["base_asset"])
    X, y, grp = d[COLS], d["label"].to_numpy().astype(int), d["base_asset"]
    we = event_weights(d, pumps)
    pos = y == 1
    print(f"навчання: {len(d)} барів, позитивів {pos.sum()}; ваги позитивів w_event: мін {we[pos].min():.2f}, медіана "
          f"{np.median(we[pos]):.2f}, макс {we[pos].max():.2f}", flush=True)
    ones = np.ones(len(d))
    models, thr, lto = {}, {}, {}
    for v, (params, use_w) in VARIANTS.items():
        w = we if use_w else ones
        for sd in SEEDS:
            key = f"{v}_s{sd}"
            p = oof(params, sd, X, y, grp, w)
            thr[key] = train.best_f1_threshold(y, p)
            sig = p >= thr[key]
            lto[key] = {**train.event_metrics(d, sig, pumps, ET.BAR), **ET.early(d, sig, pumps, px)}
            models[key] = make(params, sd).fit(X, y, sample_weight=w)
            print(f"  {key} (поріг {thr[key]:.3f}): {lto[key]}", flush=True)

    m1 = joblib.load(train.model_path("lightgbm"))
    m15 = joblib.load(train.model_path(train.M15_NAME))
    exit_bars, share = m15.get("exit_bars", 4), m15.get("confirm", {}).get("share", 0.75)
    start, end = pd.Timestamp(I.START, tz="UTC"), pd.Timestamp(I.END, tz="UTC") + pd.Timedelta(days=1)
    coins = sorted(c for c in uni if c not in trained and os.path.exists(os.path.join(I.DIR, f"{c}.pkl")))
    print(f"бектест: {len(coins)} монет чесного списку…", flush=True)
    rows = []
    for n, c in enumerate(coins, 1):
        x = pickle.load(open(os.path.join(I.DIR, f"{c}.pkl"), "rb"))
        f15, h1, h = x["f15"], x["h1"], x["h"]
        f15 = f15.join(I.new_features(h, idx).reindex(f15["ts"]).set_axis(f15.index))
        hh = h[(h.index >= start - pd.Timedelta(days=1)) & (h.index < end)]
        s1 = h1[["base_asset", "ts"]].assign(probability=m1["model"].predict_proba(h1[m1["features"]])[:, 1], threshold=m1["threshold"])
        for key, model in models.items():
            s15 = f15[["base_asset", "ts"]].assign(probability=model.predict_proba(f15[COLS])[:, 1], threshold=thr[key])
            t = tier_frame(s15, s1, share).set_index("ts").sort_index()
            sc = t.loc[(t.index >= start) & (t.index < end), "alert_score"]
            for r in P.simulate(hh, sc, I.K, exit_bars, 3.0, 0.5, True):
                mth = pd.Timestamp(r["ts"]).strftime("%Y-%m")
                if mth in uni[c]:
                    v, sd = key.rsplit("_s", 1)
                    rows.append({"variant": v, "seed": int(sd), "ticker": c, "half": AT.half(c), "month": mth, "band": uni[c][mth],
                                 "score": float(sc.get(r["ts"], np.nan)), **r})
        if n % 50 == 0:
            print(f"  {n}/{len(coins)}", flush=True)
    tr = pd.DataFrame(rows)
    tr.to_csv(OUT.replace(".md", "_trades.csv"), index=False)
    write_report(tr, lto, thr, we[pos])


def write_report(tr, lto, thr, wpos) -> None:
    pf = I.pf
    V = list(VARIANTS)
    L = ["# Вага кожного імпульсу = 1 і регуляризація LightGBM", "",
         f"Random seeds {SEEDS}; кожен варіант × seed — окреме навчання, свій поріг і бектест 6 міс. Ваги позитивних барів w_event: "
         f"від {wpos.min():.2f} (найдовші імпульси) до {wpos.max():.2f} (найкоротші), медіана {np.median(wpos):.2f}.", "",
         "**Критерій (до запуску):** проти base у парах за seed — на ≥ 20 млн середня різниця ≥ +0.02 і краще на ≥ 2 з 3; середній "
         "загальний PF ≥ base; на обох половинах середній PF ≥ base.", "",
         "## Leave-ticker-out (Кандидат 15m), середнє за seeds", "",
         "| варіант | спіймано | затримка, хв | пройдено руху | фальстартів | хибних / 30 дн |", "|---|---|---|---|---|---|"]
    for v in V:
        ks = [f"{v}_s{sd}" for sd in SEEDS]
        caught = np.mean([int(lto[k]["caught"].split("/")[0]) for k in ks])
        fs = np.mean([int(lto[k]["false_starts_triggered"].split("/")[0]) for k in ks])
        L.append(f"| {v} | {caught:.0f}/{lto[ks[0]]['caught'].split('/')[1]} | {np.mean([lto[k]['delay_min'] for k in ks]):.0f} | "
                 f"{np.mean([lto[k]['done'] for k in ks]):.0%} | {fs:.0f}/{lto[ks[0]]['false_starts_triggered'].split('/')[1]} | "
                 f"{np.mean([lto[k]['false_alarms_per_30d'] for k in ks]):.1f} |")
    L += ["", "## Бектест 6 міс. по seeds (PF; ≥ 20 млн у дужках)", "", "| варіант | " + " | ".join(f"seed {s}" for s in SEEDS) +
          " | середнє | ≥ 20 млн середнє | різниця з base (≥ 20 млн) по seeds | A ≥ 20 млн | B ≥ 20 млн | угод |", "|---|" + "---|" * (len(SEEDS) + 6)]
    big = tr["band"] == BIG
    cell = {}
    for v in V:
        for sd in SEEDS:
            g = tr[(tr["variant"] == v) & (tr["seed"] == sd)]
            b = g[g["band"] == BIG]
            cell[(v, sd)] = {"pf": pf(g["ret"]), "pf20": pf(b["ret"]), "A": pf(b[b["half"] == "A"]["ret"]), "B": pf(b[b["half"] == "B"]["ret"]), "n": len(g)}
    for v in V:
        c = [cell[(v, sd)] for sd in SEEDS]
        diffs = [cell[(v, sd)]["pf20"] - cell[("base", sd)]["pf20"] for sd in SEEDS]
        L.append(f"| {v} | " + " | ".join(f"{x['pf']:.2f} ({x['pf20']:.2f})" for x in c) + f" | {np.mean([x['pf'] for x in c]):.3f} | "
                 f"{np.mean([x['pf20'] for x in c]):.3f} | " + ("—" if v == "base" else " / ".join(f"{q:+.3f}" for q in diffs)) +
                 f" | {np.mean([x['A'] for x in c]):.3f} | {np.mean([x['B'] for x in c]):.3f} | {np.mean([x['n'] for x in c]):.0f} |")
    L += ["", "## PF за квінтилями оцінки на вході (усі seeds разом; Q5 — найвпевненіші сигнали)", "",
          "| варіант | Q1 | Q2 | Q3 | Q4 | Q5 | Q5 ≥ 20 млн |", "|---|---|---|---|---|---|---|"]
    for v in V:
        g = tr[tr["variant"] == v].dropna(subset=["score"]).copy()
        g["q"] = g.groupby("seed")["score"].transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False))
        L.append(f"| {v} | " + " | ".join(f"{pf(g[g['q'] == q]['ret']):.2f}" for q in range(5)) +
                 f" | {pf(g[(g['q'] == 4) & (g['band'] == BIG)]['ret']):.2f} |")
    L += ["", "## Критерій", ""]
    for v in V[1:]:
        diffs = np.array([cell[(v, sd)]["pf20"] - cell[("base", sd)]["pf20"] for sd in SEEDS])
        tot = np.mean([cell[(v, sd)]["pf"] for sd in SEEDS]) >= np.mean([cell[("base", sd)]["pf"] for sd in SEEDS])
        halves = all(np.mean([cell[(v, sd)][hf] for sd in SEEDS]) >= np.mean([cell[("base", sd)][hf] for sd in SEEDS]) for hf in ("A", "B"))
        ok = diffs.mean() >= 0.02 and (diffs > 0).sum() >= 2 and tot and halves
        L.append(f"- {v}: ≥ 20 млн {diffs.mean():+.3f} (краще на {(diffs > 0).sum()} з {len(SEEDS)}); загальний ≥ base {'так' if tot else 'ні'}; "
                 f"половини {'так' if halves else 'ні'} → **{'ВИКОНАНО' if ok else 'НЕ виконано'}**")
    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
