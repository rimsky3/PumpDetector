"""
experiments/time_seeds.py — ОЗНАКИ ЧАСУ і УСЕРЕДНЕННЯ SEEDS (2026-10-12; ідеї з розбору рішень Kaggle —
notes/books/ideas/kaggle_competitions.md: G-Research 7-ме місце і Jane Street 2-ге місце — час доби як ознака; Jane Street
2-ге місце — ансамбль 17 seeds; власник: «запусти»).

Варіанти (зафіксовано ДО запуску; random seeds 42–46, поріг — найкращий F1 на OOF):
    base        як зараз (ознаки ET.COLS = MODEL_FEATURES_15M + «власний рух»)
    time        base + 3 ознаки часу на момент ЗАКРИТТЯ бару (UTC): tf_hour — година доби (0…23.75), tf_fund8 — хвилин до
                найближчих 00 / 08 / 16 UTC (типовий час funding), tf_weekend — субота / неділя
    base_ens    середнє P п'яти моделей base (seed ensemble / bagging); поріг — найкращий F1 на усередненому OOF
    time_ens    те саме для time
LTO (Кандидат 15m) і бектест 6 міс. на чесному списку (Alert = 15m + робоча годинна ≥ 0.75, частковий вихід, стоп I.K) —
той самий рушій, що experiments/sample_weights.py.

КРИТЕРІЙ (до запуску):
  time проти base, у парах за seed: на ≥ 20 млн середня різниця PF ≥ +0.02 і краще на ≥ 4 з 5 seeds; середній загальний
      PF ≥ base; на обох половинах монет середній PF ≥ base. Додатково дивимось частку важливості (gain) ознак часу:
      плюс від ознак, якими модель майже не користується, — шум (урок fe_relative).
  base_ens проти середнього по окремих seeds base: PF на ≥ 20 млн ≥ середнього і на обох половинах ≥ середнього → усереднення
      приймаємо (воно нічого не коштує, головна користь — стабільність: одна модель замість лотереї seed).
Третя ідея — перенавчання за розкладом — зараз чесно не перевіряється: до 30.08 у розмітці лише 27 імпульсів, до 13.09 — 47;
повернутись наприкінці жовтня разом з повтором жовтневого holdout.
→ reports/time_seeds.md, reports/time_seeds_trades.csv

    python experiments/time_seeds.py            # потрібні локальна база і кеш data/long_bin/
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

SEEDS = [42, 43, 44, 45, 46]
TIME = ["tf_hour", "tf_fund8", "tf_weekend"]
FEATS = {"base": ET.COLS, "time": ET.COLS + TIME}
BIG = "≥ 20 млн"
OUT = os.path.join("reports", "time_seeds.md")


def time_features(ts: pd.Series) -> pd.DataFrame:
    """Ознаки часу на момент закриття 15m-бару (ts — відкриття, UTC); індекс — як у ts."""
    c = pd.DatetimeIndex(ts) + ET.BAR
    mins = c.hour * 60 + c.minute
    return pd.DataFrame({"tf_hour": mins / 60.0, "tf_fund8": (480 - mins % 480) % 480,
                         "tf_weekend": (c.dayofweek >= 5).astype(int)}, index=ts.index)


def make(seed: int):
    m = train.make_lightgbm()
    m.set_params(random_state=seed)
    return m


def oof(seed: int, X, y, grp) -> np.ndarray:
    p = np.zeros(len(y))
    for g in grp.unique():
        te = (grp == g).to_numpy()
        p[te] = make(seed).fit(X[~te], y[~te]).predict_proba(X[te])[:, 1]
    return p


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    argparse.ArgumentParser(description="Ознаки часу і усереднення seeds: LTO і бектест 6 міс. на 5 random seeds").parse_args()
    d, pumps, px, idx, uni = ET.load_data()
    d = d.join(time_features(d["ts"]))
    trained = set(d["base_asset"])
    y, grp = d["label"].to_numpy().astype(int), d["base_asset"]
    print(f"навчання: {len(d)} барів, позитивів {(y == 1).sum()}, тікерів {grp.nunique()}", flush=True)
    models, thr, lto, gain = {}, {}, {}, []
    for v, cols in FEATS.items():
        X = d[cols]
        oofs = []
        for sd in SEEDS:
            key = f"{v}_s{sd}"
            p = oof(sd, X, y, grp)
            oofs.append(p)
            thr[key] = train.best_f1_threshold(y, p)
            sig = p >= thr[key]
            lto[key] = {**train.event_metrics(d, sig, pumps, ET.BAR), **ET.early(d, sig, pumps, px)}
            models[key] = make(sd).fit(X, y)
            if v == "time":
                g = models[key].booster_.feature_importance(importance_type="gain")
                gain.append(dict(zip(cols, g / g.sum())))
            print(f"  {key} (поріг {thr[key]:.3f}): {lto[key]}", flush=True)
        pe = np.mean(oofs, axis=0)                                       # ансамбль: середнє OOF п'яти seeds
        key = f"{v}_ens_s0"
        thr[key] = train.best_f1_threshold(y, pe)
        sig = pe >= thr[key]
        lto[key] = {**train.event_metrics(d, sig, pumps, ET.BAR), **ET.early(d, sig, pumps, px)}
        print(f"  {key} (поріг {thr[key]:.3f}): {lto[key]}", flush=True)
    tgain = {f: float(np.mean([g[f] for g in gain])) for f in TIME}

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
        f15 = f15.join(time_features(f15["ts"]))
        hh = h[(h.index >= start - pd.Timedelta(days=1)) & (h.index < end)]
        s1 = h1[["base_asset", "ts"]].assign(probability=m1["model"].predict_proba(h1[m1["features"]])[:, 1], threshold=m1["threshold"])
        probs = {}
        for v, cols in FEATS.items():
            for sd in SEEDS:
                probs[f"{v}_s{sd}"] = models[f"{v}_s{sd}"].predict_proba(f15[cols])[:, 1]
            probs[f"{v}_ens_s0"] = np.mean([probs[f"{v}_s{sd}"] for sd in SEEDS], axis=0)
        for key, p in probs.items():
            s15 = f15[["base_asset", "ts"]].assign(probability=p, threshold=thr[key])
            t = tier_frame(s15, s1, share).set_index("ts").sort_index()
            sc = t.loc[(t.index >= start) & (t.index < end), "alert_score"]
            for r in P.simulate(hh, sc, I.K, exit_bars, 3.0, 0.5, True):
                mth = pd.Timestamp(r["ts"]).strftime("%Y-%m")
                if mth in uni[c]:
                    v, sd = key.rsplit("_s", 1)
                    rows.append({"variant": v, "seed": int(sd), "ticker": c, "half": AT.half(c), "month": mth, "band": uni[c][mth], **r})
        if n % 50 == 0:
            print(f"  {n}/{len(coins)}", flush=True)
    tr = pd.DataFrame(rows)
    tr.to_csv(OUT.replace(".md", "_trades.csv"), index=False)
    write_report(tr, lto, thr, tgain)


def write_report(tr, lto, thr, tgain) -> None:
    pf = I.pf

    def cell(v, sd):
        g = tr[(tr["variant"] == v) & (tr["seed"] == sd)]
        b = g[g["band"] == BIG]
        return {"pf": pf(g["ret"]), "pf20": pf(b["ret"]), "A": pf(b[b["half"] == "A"]["ret"]), "B": pf(b[b["half"] == "B"]["ret"]),
                "n": len(g), "n20": len(b), "months": {m: pf(x["ret"]) for m, x in b.groupby("month")}}

    C = {(v, sd): cell(v, sd) for v in FEATS for sd in SEEDS}
    E = {v: cell(f"{v}_ens", 0) for v in FEATS}
    mean = lambda v, k: float(np.mean([C[(v, sd)][k] for sd in SEEDS]))      # noqa: E731
    L = ["# Ознаки часу і усереднення seeds (12.10)", "",
         f"Random seeds {SEEDS}; бектест 6 міс. на чесному списку, Alert = 15m + робоча годинна, частковий вихід. "
         "Ознаки часу: година доби, хвилин до 00 / 08 / 16 UTC, вихідний.", "",
         "## Leave-ticker-out (Кандидат 15m)", "",
         "| варіант | спіймано | затримка, хв | пройдено руху | фальстартів | хибних / 30 дн |", "|---|---|---|---|---|---|"]
    for v in FEATS:
        ks = [f"{v}_s{sd}" for sd in SEEDS]
        tot, fst = lto[ks[0]]["caught"].split("/")[1], lto[ks[0]]["false_starts_triggered"].split("/")[1]
        L.append(f"| {v} (середнє 5 seeds) | {np.mean([int(lto[k]['caught'].split('/')[0]) for k in ks]):.0f}/{tot} | "
                 f"{np.mean([lto[k]['delay_min'] for k in ks]):.0f} | {np.mean([lto[k]['done'] for k in ks]):.0%} | "
                 f"{np.mean([int(lto[k]['false_starts_triggered'].split('/')[0]) for k in ks]):.0f}/{fst} | "
                 f"{np.mean([lto[k]['false_alarms_per_30d'] for k in ks]):.1f} |")
        k = f"{v}_ens_s0"
        L.append(f"| {v}_ens | {lto[k]['caught']} | {lto[k]['delay_min']:.0f} | {lto[k]['done']:.0%} | "
                 f"{lto[k]['false_starts_triggered']} | {lto[k]['false_alarms_per_30d']:.1f} |")
    L += ["", "## Бектест 6 міс.: PF по seeds (≥ 20 млн у дужках)", "",
          "| варіант | " + " | ".join(f"seed {s}" for s in SEEDS) + " | середнє | ≥ 20 млн середнє | розкид ≥ 20 млн | A | B | угод |",
          "|---|" + "---|" * (len(SEEDS) + 6)]
    for v in FEATS:
        p20 = [C[(v, sd)]["pf20"] for sd in SEEDS]
        L.append(f"| {v} | " + " | ".join(f"{C[(v, sd)]['pf']:.2f} ({C[(v, sd)]['pf20']:.2f})" for sd in SEEDS) +
                 f" | {mean(v, 'pf'):.3f} | {np.mean(p20):.3f} | {max(p20) - min(p20):.3f} | {mean(v, 'A'):.3f} | {mean(v, 'B'):.3f} | {mean(v, 'n'):.0f} |")
    for v in FEATS:
        e = E[v]
        L.append(f"| {v}_ens | " + " | ".join("—" for _ in SEEDS) + f" | {e['pf']:.3f} | {e['pf20']:.3f} | — | {e['A']:.3f} | {e['B']:.3f} | {e['n']} |")
    L += ["", "## Ознаки часу проти base (у парах за seed, ≥ 20 млн)", ""]
    diffs = np.array([C[("time", sd)]["pf20"] - C[("base", sd)]["pf20"] for sd in SEEDS])
    tot = mean("time", "pf") >= mean("base", "pf")
    halves = all(mean("time", hf) >= mean("base", hf) for hf in ("A", "B"))
    ok = diffs.mean() >= 0.02 and (diffs > 0).sum() >= 4 and tot and halves
    L += [f"- різниця по seeds: {' / '.join(f'{q:+.3f}' for q in diffs)}; середня {diffs.mean():+.3f}, краще на {(diffs > 0).sum()} з {len(SEEDS)}",
          f"- загальний PF ≥ base: {'так' if tot else 'ні'} ({mean('time', 'pf'):.3f} проти {mean('base', 'pf'):.3f}); "
          f"половини A / B: {mean('time', 'A'):.3f} / {mean('time', 'B'):.3f} проти {mean('base', 'A'):.3f} / {mean('base', 'B'):.3f}",
          "- частка важливості (gain) ознак часу: " + ", ".join(f"{f} {g:.1%}" for f, g in tgain.items()) + f" — разом {sum(tgain.values()):.1%}",
          f"- **Критерій time: {'ВИКОНАНО' if ok else 'НЕ ВИКОНАНО'}**", "",
          "## Усереднення seeds проти окремих моделей (≥ 20 млн)", ""]
    for v in FEATS:
        e, m20 = E[v], mean(v, "pf20")
        ok_e = e["pf20"] >= m20 and e["A"] >= mean(v, "A") and e["B"] >= mean(v, "B")
        months = sorted(e["months"])
        better = sum(e["months"][m] >= np.mean([C[(v, sd)]["months"].get(m, np.nan) for sd in SEEDS]) for m in months)
        L.append(f"- {v}_ens: PF {e['pf20']:.3f} проти середнього {m20:.3f} по окремих seeds (діапазон "
                 f"{min(C[(v, sd)]['pf20'] for sd in SEEDS):.3f}…{max(C[(v, sd)]['pf20'] for sd in SEEDS):.3f}); A / B {e['A']:.3f} / {e['B']:.3f} "
                 f"проти {mean(v, 'A'):.3f} / {mean(v, 'B'):.3f}; кращий за середнє в {better} з {len(months)} місяців; угод {e['n20']} "
                 f"проти {mean(v, 'n20'):.0f} → **{'ВИКОНАНО' if ok_e else 'НЕ ВИКОНАНО'}**")
    L += ["", "Пороги: " + ", ".join(f"{k.replace('_s0', '')} {t:.3f}" for k, t in thr.items())]
    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
