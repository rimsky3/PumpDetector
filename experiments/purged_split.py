"""
experiments/purged_split.py — SPLIT_START30: PURGED CROSS-VALIDATION ЗА ЧАСОВИМИ БЛОКАМИ (2026-10-11; власник: «чому не бектестимо
вересень?», «так запускай»). Жовтень (10 днів, ~75 угод на ≥ 20 млн) суперечить 6 місяцям — замало. Вчити лише «до вересня» —
27 імпульсів власника, модель недоросла. Тому (López de Prado, purged K-fold): тестовий місяць вирізано з навчання з запасом
3 дні з кожного боку; вчимося на решті розмітки власника (і раніших, і пізніших місяцях); торгуємо тестовий місяць на монетах
поза тікерами власника. Блоки: серпень, вересень (кеш data/long_bin), 01–10.10 (data/oct_bin).
Моделі (як early_target.py / oct_split.py): base — «імпульс триває», поріг — найкращий F1 на OOF; «старт» — перші 30 хв імпульсу,
поріг — та сама кількість хибних, що в base. Схеми: base; split — вхід за Alert «старту», утримання — base або «старт».
Seeds 42 / 43 / 44. Частковий вихід, стоп I.K, годинна — робоча.
КРИТЕРІЙ (до запуску): на ≥ 20 млн PF split ≥ base за всі три блоки разом (середнє за seeds) і ≥ base щонайменше в 2 з 3 блоків.
→ reports/purged_split.md

    python experiments/purged_split.py
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
EMB = pd.Timedelta(days=3)
BLOCKS = {"2026-08": ("2026-08-01", "2026-09-01", "arch"), "2026-09": ("2026-09-01", "2026-10-01", "arch"),
          "2026-10": ("2026-10-01", "2026-10-11", "oct")}
OUT = os.path.join("reports", "purged_split.md")


def lgbm(seed: int):
    m = train.make_lightgbm()
    m.set_params(random_state=seed)
    return m


def fit_pair(do: pd.DataFrame, pu: pd.DataFrame, sd: int):
    """base і «старт» на рядках do (розмітка pu): моделі й пороги, як early_target / oct_split."""
    X, grp, y = do[COLS], do["base_asset"], do["label"].to_numpy().astype(int)
    ys, keep = ET.target(do, pu, pd.Timedelta(0), pd.Timedelta(minutes=30))
    ys = ys.astype(int)
    p0 = train.oof_proba(lambda s=sd: lgbm(s), X, pd.Series(y), grp)
    t0 = train.best_f1_threshold(y, p0)
    fa0 = train.event_metrics(do, p0 >= t0, pu, ET.BAR)["false_alarms_per_30d"]
    ps = np.zeros(len(do))
    for g in grp.unique():
        te = (grp == g).to_numpy()
        tr = ~te & keep
        ps[te] = lgbm(sd).fit(X[tr], ys[tr]).predict_proba(X[te])[:, 1]
    ts_ = ET.matched_threshold(do, ps, pu, fa0)
    return (lgbm(sd).fit(X, y), t0), (lgbm(sd).fit(X[keep], ys[keep]), ts_)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    argparse.ArgumentParser(description="split_start30: purged cross-validation за часовими блоками (серпень, вересень, 01–10.10)").parse_args()
    idx_full = OH.market_index_full()
    I.market_index = lambda: idx_full
    d, pumps, px, idx, uni = ET.load_data()
    trained = set(d["base_asset"])
    ou = json.load(open(OH.UNIVERSE, encoding="utf-8"))
    m1 = joblib.load(train.model_path("lightgbm"))
    m15 = joblib.load(train.model_path(train.M15_NAME))
    exit_bars, share = m15.get("exit_bars", 4), m15.get("confirm", {}).get("share", 0.75)
    rows = []
    for blk, (a, b, src) in BLOCKS.items():
        start, end = pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC")
        cut_lo, cut_hi = start - EMB, end + EMB
        do = d[(d["ts"] < cut_lo) | (d["ts"] >= cut_hi)].reset_index(drop=True)
        pu = pumps[(pumps["end_ts"] <= cut_lo) | (pumps["start_ts"] >= cut_hi)]
        n_imp = int((pu["pattern_type"] == "pump").sum())
        models = {sd: fit_pair(do, pu, sd) for sd in SEEDS}
        print(f"{blk}: навчання {len(do)} барів, {n_imp} імпульсів; пороги «старту» " +
              ", ".join(f"{models[sd][1][1]:.3f}" for sd in SEEDS), flush=True)
        if src == "arch":
            coins = sorted(c for c in uni if c not in trained and blk in uni[c] and os.path.exists(os.path.join(I.DIR, f"{c}.pkl")))
            band = {c: uni[c][blk] for c in coins}
            path = lambda c: os.path.join(I.DIR, f"{c}.pkl")          # noqa: E731
        else:
            coins = sorted(c for c in ou if c not in trained and os.path.exists(os.path.join(OH.DIR, f"{c}.pkl")))
            band = ou
            path = lambda c: os.path.join(OH.DIR, f"{c}.pkl")         # noqa: E731
        for c in coins:
            x = pickle.load(open(path(c), "rb"))
            f15, h1, h = x["f15"], x["h1"], x["h"]
            f15 = f15[(f15["ts"] >= start - pd.Timedelta(days=2)) & (f15["ts"] < end)]
            if f15.empty or h1.empty:
                continue
            f15 = f15.join(I.new_features(h, idx_full).reindex(f15["ts"]).set_axis(f15.index))
            hh = h[(h.index >= start - pd.Timedelta(days=1)) & (h.index < end)]
            s1 = h1[["base_asset", "ts"]].assign(probability=m1["model"].predict_proba(h1[m1["features"]])[:, 1], threshold=m1["threshold"])

            def alert(mdl, thr):
                s15 = f15[["base_asset", "ts"]].assign(probability=mdl.predict_proba(f15[COLS])[:, 1], threshold=thr)
                t = tier_frame(s15, s1, share).set_index("ts").sort_index()
                return t.loc[(t.index >= start - pd.Timedelta(days=1)) & (t.index < end), "alert_score"]
            for sd in SEEDS:
                (mb, tb), (ms, tst) = models[sd]
                a0 = alert(mb, tb)
                ast = alert(ms, tst).reindex(a0.index)
                for v, (en, ho) in {"base": (a0, a0), "split": (ast, np.maximum(a0, ast))}.items():
                    for r in ET.simulate_split(hh, en, ho, I.K, exit_bars, 3.0, 0.5, True):
                        if start <= pd.Timestamp(r["ts"]) < end:
                            rows.append({"block": blk, "variant": v, "seed": sd, "ticker": c, "band": band[c], **r})
        print(f"  {blk}: бектест {len(coins)} монет", flush=True)
    tr = pd.DataFrame(rows)
    tr.to_csv(OUT.replace(".md", "_trades.csv"), index=False)
    pf = I.pf
    big = tr[tr["band"] == BIG]
    cell = lambda df, v: float(np.mean([pf(df[(df["variant"] == v) & (df["seed"] == sd)]["ret"]) for sd in SEEDS]))   # noqa: E731
    L = ["# split_start30: purged cross-validation за часовими блоками", "",
         "Тестовий місяць вирізано з навчання з запасом 3 дні з кожного боку; моделі вчаться на решті розмітки власника; торгують монети "
         f"поза тікерами власника. Seeds {SEEDS}, значення — середнє.", "",
         "**Критерій (до запуску):** на ≥ 20 млн PF split ≥ base за всі блоки разом і ≥ base щонайменше в 2 з 3 блоків.", "",
         "| блок | угод base / split (≥ 20 млн) | **≥ 20 млн base** | **≥ 20 млн split** | загальний base | загальний split | ≥ 20 млн без топ-3 base / split |",
         "|---|---|---|---|---|---|---|"]
    wins = 0
    for blk in BLOCKS:
        g, gb = tr[tr["block"] == blk], big[big["block"] == blk]
        nb = lambda v: np.mean([len(gb[(gb["variant"] == v) & (gb["seed"] == sd)]) for sd in SEEDS])       # noqa: E731
        wo3 = lambda v: np.mean([pf(gb[(gb["variant"] == v) & (gb["seed"] == sd)]["ret"].sort_values().iloc[:-3]) for sd in SEEDS])   # noqa: E731
        bb, bs = cell(gb, "base"), cell(gb, "split")
        wins += bs >= bb - 1e-9
        L.append(f"| {blk} | {nb('base'):.0f} / {nb('split'):.0f} | {bb:.3f} | {bs:.3f} | {cell(g, 'base'):.3f} | {cell(g, 'split'):.3f} | "
                 f"{wo3('base'):.2f} / {wo3('split'):.2f} |")
    tb, ts_ = cell(big, "base"), cell(big, "split")
    L.append(f"| **разом** | — | **{tb:.3f}** | **{ts_:.3f}** | {cell(tr, 'base'):.3f} | {cell(tr, 'split'):.3f} | — |")
    ok = ts_ >= tb and wins >= 2
    L += ["", f"## Критерій\n\n→ **{'ВИКОНАНО' if ok else 'НЕ виконано'}** (блоків, де split ≥ base: {wins} з {len(BLOCKS)})"]
    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
