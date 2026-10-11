"""
experiments/teacher_combo.py — ОБ'ЄДНАНИЙ TEACHER–STUDENT: soft labels + ознаки топ-трейдерів, кілька random seeds
(2026-10-11, план дня п. 2; власник: «давай з 1 по 3»). Після teacher_walkforward.py (жорсткі мітки t100 / t90: PF 0.91 / 0.90
проти base 0.89, ≥ 20 млн 0.88 проти 0.86, один seed) і fe_relative.py --seeds top (+0.018 на ≥ 20 млн, 4 з 5 seeds).
Зміни проти teacher_walkforward (зафіксовано ДО запуску):
  * soft labels (fidelity weighting, Dehghani et al. 2017): замість порогів 1 / сіра зона / 0 — КАЛІБРОВАНА ймовірність teacher.
    Калібрування: isotonic regression на out-of-fold P teacher проти міток власника (teacher вчився з class_weight="balanced",
    його P завищені). Бар у вікні події (від max(старт, тригер − 1 год) до кінця) отримує p = калібрована P (максимум, якщо подій
    кілька); поза подіями — 0. Навчання з soft label — дублювання рядка: (y=1, вага p) і (y=0, вага 1 − p); це рівно cross-entropy
    з м'якою ціллю, class_weight="balanced" працює як звичайно.
  * negative downsampling: бари без жодної події ±12 год — кожен 3-й з вагою 3 (незміщено, лише швидше навчання).
  * поріг student — найкращий F1 на OOF 3 груп монет проти «жорсткої» мітки p ≥ 0.5.
Варіанти: base (розмітка власника, робочі ознаки), soft (soft labels, робочі ознаки), soft_top (+ 5 ознак топ-трейдерів,
fe_relative.GROUPS["top"]); кожен на seeds 42 / 43 / 44. Walk-forward як auto_walkforward: тест-місяць травень…вересень, навчання до
(m − 3 дні). --event-weights — ще й сумарна вага кожної події teacher однакова (якщо sample_weights.py підтвердить ідею).
КРИТЕРІЙ (до запуску), проти base у парах за seed: ≥ 20 млн середня різниця ≥ +0.02 і краще на ≥ 2 з 3; середній загальний PF ≥ base;
на обох половинах монет середній PF ≥ base; хибних на розмітці власника (student вересневого кроку) ≤ 1.5 × base.
→ reports/teacher_combo.md, reports/teacher_combo_trades.csv

    python experiments/teacher_combo.py [--event-weights]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold

import auto_train as AT
import auto_walkforward as AW
import early_target as ET
import exit_partial as P
import fe_relative as FR
import idio_features as I
import label_teacher as LT
import train
from tiers import tier_frame

COLS = AT.COLS
TOP = FR.GROUPS["top"]
SEEDS = [42, 43, 44]
VARIANTS = {"soft": COLS, "soft_top": COLS + TOP}
MONTHS = AW.MONTHS
NEG_KEEP, NEAR = 3, 48                  # бари без подій ±12 год (48 барів): кожен 3-й з вагою 3
BIG = "≥ 20 млн"
OUT = os.path.join("reports", "teacher_combo.md")
H1 = np.timedelta64(1, "h")


def lgbm(seed: int):
    m = train.make_lightgbm()
    m.set_params(random_state=seed)
    return m


def soft_bars(ts: pd.Series, e: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Для барів ts: p (калібрована P події, 0 поза подіями), id події (−1 — поза), «біля події» (±12 год)."""
    close = ts.values + np.timedelta64(15, "m")
    p, eid, near = np.zeros(len(ts)), np.full(len(ts), -1), np.zeros(len(ts), bool)
    for k, r in enumerate(e.itertuples()):
        st = max(np.datetime64(r.start.tz_convert(None)), np.datetime64(r.trigger.tz_convert(None)) - H1)
        en = np.datetime64(r.end_ts.tz_convert(None))
        m = (close > st) & (close <= en)
        upd = m & (r.pc > p)
        p[upd], eid[upd] = r.pc, k
        near |= (close > st - 12 * H1) & (close <= en + 12 * H1)
    return p, eid, near


def expand(df: pd.DataFrame, w: np.ndarray) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Soft label → два рядки: (y=1, w·p) і (y=0, w·(1−p)); рядки з p = 0 — один (y=0)."""
    p = df["p"].to_numpy()
    soft = p > 0
    X = pd.concat([df, df[soft]], ignore_index=True)
    y = np.concatenate([np.where(soft, 1, 0), np.zeros(soft.sum(), int)])
    ww = np.concatenate([np.where(soft, w * p, w), (w * (1 - p))[soft]])
    keep = ww > 0
    return X[keep], y[keep], ww[keep]


def gkf_threshold(X, y_hard, g, w, seed, n=3) -> float:
    p = np.zeros(len(y_hard))
    for tr, te in GroupKFold(n).split(X, y_hard, g):
        Xe, ye, we = expand(X.iloc[tr], w[tr])
        p[te] = lgbm(seed).fit(Xe[X.columns.drop(["p", "eid"], errors="ignore")], ye, sample_weight=we).predict_proba(
            X.iloc[te][X.columns.drop(["p", "eid"], errors="ignore")])[:, 1]
    return train.best_f1_threshold(y_hard, p)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Об'єднаний teacher–student: soft labels (калібрована P teacher) + ознаки топ-трейдерів, 3 random seeds")
    ap.add_argument("--event-weights", action="store_true", help="однакова сумарна вага кожної події teacher (ідея sample_weights.py)")
    args = ap.parse_args()
    out = OUT.replace(".md", "_ew.md") if args.event_weights else OUT

    k, own_p, own_f, reviewed, days = LT.owner_events()
    p_oof = train.oof_proba(train.make_lightgbm, k[LT.OPIS], k["y"], k["base"])
    cal = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p_oof, k["y"])
    teacher = LT.fit_teacher(k)
    print(f"teacher: {len(k)} подій; калібрування: сира P 0.2 / 0.4 / 0.6 / 0.8 → " +
          " / ".join(f"{cal.predict([q])[0]:.2f}" for q in (0.2, 0.4, 0.6, 0.8)), flush=True)

    d, pumps, px, idx, uni = ET.load_data()
    trained = set(d["base_asset"])
    mk = FR.market()
    parts = [FR.features(px[b], FR.metrics(b), FR.premium(b), mk)[TOP].reindex(g["ts"]).set_axis(g.index) for b, g in d.groupby("base_asset")]
    d = d.join(pd.concat(parts))
    coins = sorted(c for c in uni if c not in trained and os.path.exists(os.path.join(I.DIR, f"{c}.pkl")))
    print(f"розмітка teacher {len(coins)} монет…", flush=True)
    parts, n_ev = [], 0
    for n, c in enumerate(coins, 1):
        x, f15 = AW.load_coin(c, idx)
        s = AT.coin_series(c, x["h"])
        e = pd.DataFrame(LT.event_rows(s, idx.reindex(s.index)))
        if len(e):
            e["pc"] = cal.predict(teacher.predict_proba(e[LT.OPIS])[:, 1])
            p, eid, near = soft_bars(f15["ts"], e)
            n_ev += len(e)
        else:
            p, eid, near = np.zeros(len(f15)), np.full(len(f15), -1), np.zeros(len(f15), bool)
        top = FR.features(x["h"], FR.metrics(c), FR.premium(c), mk)[TOP].reindex(f15["ts"]).set_axis(f15.index)
        f = f15[["base_asset", "ts", *COLS]].join(top).astype({q: "float32" for q in COLS + TOP})
        far = ~near
        keep = ~far | (np.arange(len(f15)) % NEG_KEEP == 0)
        parts.append(f.assign(p=p, eid=np.where(eid >= 0, n_ev - len(e) + eid if len(e) else -1, -1), w=np.where(far, float(NEG_KEEP), 1.0))[keep])
        if n % 100 == 0:
            print(f"  {n}/{len(coins)}", flush=True)
    a = pd.concat(parts, ignore_index=True)
    del parts
    if args.event_weights:                  # сумарна вага бару події = L̄ / L
        m = a["eid"] >= 0
        cnt = a.loc[m, "eid"].map(a.loc[m, "eid"].value_counts())
        a.loc[m, "w"] = a.loc[m, "w"] * cnt.mean() / cnt
    print(f"розмічено: {len(a)} рядків після downsampling; подій {n_ev}; середня p {a['p'].mean():.3%}, p ≥ 0.5 — {(a['p'] >= 0.5).mean():.2%} "
          f"барів; власник {d['label'].mean():.2%}", flush=True)

    Xo, yo = d[COLS], d["label"].to_numpy().astype(int)
    base, folds, info, gold = {}, {}, [], {}
    for sd in SEEDS:
        po = train.oof_proba(lambda s=sd: lgbm(s), Xo, pd.Series(yo), d["base_asset"])
        base[sd] = (lgbm(sd).fit(Xo, yo), train.best_f1_threshold(yo, po))
        sig = po >= base[sd][1]
        gold[("base", sd)] = {**train.event_metrics(d, sig, pumps, ET.BAR), **ET.early(d, sig, pumps, px)}
    for m in MONTHS:
        cut = pd.Timestamp(m + "-01", tz="UTC") - AW.GAP
        s = a[a["ts"] < cut]
        yh = (s["p"] >= 0.5).to_numpy().astype(int)
        for v, cols in VARIANTS.items():
            for sd in SEEDS:
                thr = gkf_threshold(s[cols + ["p"]], yh, s["base_asset"], s["w"].to_numpy(), sd)
                Xe, ye, we = expand(s[cols + ["p"]], s["w"].to_numpy())
                folds[(v, sd, m)] = (lgbm(sd).fit(Xe[cols], ye, sample_weight=we), thr)
                info.append({"month": m, "variant": v, "seed": sd, "rows": len(s), "thr": thr})
                print(f"  {m} {v} s{sd}: {len(s)} рядків до {cut:%Y-%m-%d}, поріг {thr:.3f}", flush=True)
    del a
    for v, cols in VARIANTS.items():
        for sd in SEEDS:
            model, thr = folds[(v, sd, MONTHS[-1])]
            sig = model.predict_proba(d[cols])[:, 1] >= thr
            gold[(v, sd)] = {**train.event_metrics(d, sig, pumps, ET.BAR), **ET.early(d, sig, pumps, px)}
            print(f"  {v} s{sd} на розмітці власника: {gold[(v, sd)]}", flush=True)

    m1 = joblib.load(train.model_path("lightgbm"))
    m15 = joblib.load(train.model_path(train.M15_NAME))
    exit_bars, share = m15.get("exit_bars", 4), m15.get("confirm", {}).get("share", 0.75)
    start, end = pd.Timestamp(MONTHS[0] + "-01", tz="UTC"), pd.Timestamp(I.END, tz="UTC") + pd.Timedelta(days=1)
    print(f"бектест {MONTHS[0]}…{MONTHS[-1]}: {len(coins)} монет…", flush=True)
    rows = []
    for n, c in enumerate(coins, 1):
        x, f15 = AW.load_coin(c, idx)
        f15 = f15.join(FR.features(x["h"], FR.metrics(c), FR.premium(c), mk)[TOP].reindex(f15["ts"]).set_axis(f15.index))
        h1, h = x["h1"], x["h"]
        hh = h[(h.index >= start - pd.Timedelta(days=1)) & (h.index < end)]
        s1 = h1[["base_asset", "ts"]].assign(probability=m1["model"].predict_proba(h1[m1["features"]])[:, 1], threshold=m1["threshold"])
        mon = f15["ts"].dt.strftime("%Y-%m").to_numpy()
        for v in ["base", *VARIANTS]:
            for sd in SEEDS:
                prob, thr = np.full(len(f15), np.nan), np.full(len(f15), np.nan)
                if v == "base":
                    prob[:], thr[:] = base[sd][0].predict_proba(f15[COLS])[:, 1], base[sd][1]
                else:
                    for m in MONTHS:
                        q = mon == m
                        if q.any():
                            model, t = folds[(v, sd, m)]
                            prob[q], thr[q] = model.predict_proba(f15.loc[q, VARIANTS[v]])[:, 1], t
                s15 = f15[["base_asset", "ts"]].assign(probability=prob, threshold=thr).dropna()
                t = tier_frame(s15, s1, share).set_index("ts").sort_index()
                for r in P.simulate(hh, t.loc[(t.index >= start) & (t.index < end), "alert_score"], I.K, exit_bars, 3.0, 0.5, True):
                    mth = pd.Timestamp(r["ts"]).strftime("%Y-%m")
                    if mth in uni[c] and mth in MONTHS:
                        rows.append({"variant": v, "seed": sd, "ticker": c, "half": AT.half(c), "month": mth, "band": uni[c][mth], **r})
        if n % 100 == 0:
            print(f"  {n}/{len(coins)}", flush=True)
    tr = pd.DataFrame(rows)
    tr.to_csv(out.replace(".md", "_trades.csv"), index=False)
    write_report(out, tr, gold, n_ev, args.event_weights)


def write_report(out, tr, gold, n_ev, ew) -> None:
    pf = I.pf
    V = ["base", *VARIANTS]
    L = ["# Об'єднаний teacher–student: soft labels + топ-трейдери" + (" + ваги подій" if ew else ""), "",
         f"Teacher (label_teacher.py) оцінює {n_ev} подій на монетах архіву; soft label бару — калібрована (isotonic) P події. "
         f"Walk-forward травень–вересень, random seeds {SEEDS}. base — розмітка власника (серп–жовт; перекриття за часом на її користь).", "",
         "**Критерій (до запуску):** проти base у парах за seed — ≥ 20 млн середня різниця ≥ +0.02 і краще на ≥ 2 з 3; середній загальний "
         "PF ≥ base; обидві половини ≥ base; хибних на розмітці власника ≤ 1.5 × base.", "",
         "## На розмітці власника (Кандидат 15m; base — OOF, student — модель вересневого кроку), середнє за seeds", "",
         "| варіант | спіймано | затримка, хв | пройдено руху | фальстартів | хибних / 30 дн |", "|---|---|---|---|---|---|"]
    for v in V:
        g = [gold[(v, sd)] for sd in SEEDS]
        L.append(f"| {v} | {np.mean([int(x['caught'].split('/')[0]) for x in g]):.0f}/{g[0]['caught'].split('/')[1]} | "
                 f"{np.mean([x['delay_min'] for x in g]):.0f} | {np.mean([x['done'] for x in g]):.0%} | "
                 f"{np.mean([int(x['false_starts_triggered'].split('/')[0]) for x in g]):.0f}/{g[0]['false_starts_triggered'].split('/')[1]} | "
                 f"{np.mean([x['false_alarms_per_30d'] for x in g]):.1f} |")
    cell = {}
    for v in V:
        for sd in SEEDS:
            g = tr[(tr["variant"] == v) & (tr["seed"] == sd)]
            b = g[g["band"] == BIG]
            cell[(v, sd)] = {"pf": pf(g["ret"]), "pf20": pf(b["ret"]), "A": pf(b[b["half"] == "A"]["ret"]), "B": pf(b[b["half"] == "B"]["ret"]), "n": len(g)}
    L += ["", "## Бектест травень–вересень по seeds (PF; ≥ 20 млн у дужках)", "",
          "| варіант | " + " | ".join(f"seed {s}" for s in SEEDS) + " | середнє | ≥ 20 млн | різниця з base (≥ 20 млн) | A ≥ 20 млн | B ≥ 20 млн | угод |",
          "|---|" + "---|" * (len(SEEDS) + 6)]
    for v in V:
        c = [cell[(v, sd)] for sd in SEEDS]
        diffs = [cell[(v, sd)]["pf20"] - cell[("base", sd)]["pf20"] for sd in SEEDS]
        L.append(f"| {v} | " + " | ".join(f"{x['pf']:.2f} ({x['pf20']:.2f})" for x in c) + f" | {np.mean([x['pf'] for x in c]):.3f} | "
                 f"{np.mean([x['pf20'] for x in c]):.3f} | " + ("—" if v == "base" else " / ".join(f"{q:+.3f}" for q in diffs)) +
                 f" | {np.mean([x['A'] for x in c]):.3f} | {np.mean([x['B'] for x in c]):.3f} | {np.mean([x['n'] for x in c]):.0f} |")
    piv = tr.groupby(["month", "variant"])["ret"].apply(pf).unstack().reindex(columns=V)
    piv20 = tr[tr["band"] == BIG].groupby(["month", "variant"])["ret"].apply(pf).unstack().reindex(columns=V)
    L += ["", "## По місяцях (усі seeds разом; PF, у дужках ≥ 20 млн)", "", "| місяць | " + " | ".join(V) + " |", "|---|" + "---|" * len(V)]
    for mth in piv.index:
        L.append(f"| {mth} | " + " | ".join(f"{piv.at[mth, v]:.2f} ({piv20.at[mth, v]:.2f})" for v in V) + " |")
    L += ["", "## Критерій", ""]
    fa0 = np.mean([gold[("base", sd)]["false_alarms_per_30d"] for sd in SEEDS])
    for v in VARIANTS:
        diffs = np.array([cell[(v, sd)]["pf20"] - cell[("base", sd)]["pf20"] for sd in SEEDS])
        tot = np.mean([cell[(v, sd)]["pf"] for sd in SEEDS]) >= np.mean([cell[("base", sd)]["pf"] for sd in SEEDS])
        halves = all(np.mean([cell[(v, sd)][hf] for sd in SEEDS]) >= np.mean([cell[("base", sd)][hf] for sd in SEEDS]) for hf in ("A", "B"))
        fa = np.mean([gold[(v, sd)]["false_alarms_per_30d"] for sd in SEEDS])
        ok = diffs.mean() >= 0.02 and (diffs > 0).sum() >= 2 and tot and halves and fa <= 1.5 * fa0
        L.append(f"- {v}: ≥ 20 млн {diffs.mean():+.3f} (краще на {(diffs > 0).sum()} з {len(SEEDS)}); загальний ≥ base {'так' if tot else 'ні'}; "
                 f"половини {'так' if halves else 'ні'}; хибних {fa:.1f} проти 1.5 × {fa0:.1f} — {'так' if fa <= 1.5 * fa0 else 'ні'} → "
                 f"**{'ВИКОНАНО' if ok else 'НЕ виконано'}**")
    open(out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
