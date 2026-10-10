"""
experiments/early_target_check.py — перевірка стійкості split_start30 з experiments/early_target.py (2026-10-09). Там: вхід ЛИШЕ за
моделлю «старт імпульсу» (ціль — перші 30 хв), утримання — поки горить base або рання, PF 0.90 → 0.96, ≥ 20 млн 0.84 → 0.95,
кращий 6 з 6 місяців, але угод 11 632 → 4 714. Питання (зафіксовано ДО запуску):
    1) сусідні пороги ранньої моделі: 0.88 / 0.89 / 0.905 (як у звіті) / 0.92 / 0.93;
    2) інше зерно ранньої моделі: 43, 44 (поріг 0.905);
    3) дві половини монет (md5 назви) і два півріччя;
    4) ЧИ ВХІД РАНІШИЙ: для кожної угоди split — перший Alert base на тій самій монеті в [вхід − 4 год, вихід]:
       частка угод без Alert base взагалі, медіана «Alert base мінус вхід» (хв) і ціни (Alert base / вхід − 1).
Пороги base і start30 — зі звіту early_target.md (0.828 / 0.905), моделі перенавчаються на всій розмітці.
→ reports/early_target_check.md

    python experiments/early_target_check.py
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
import idio_features as I
import train
from tiers import tier_frame

THR_BASE, THR = 0.828, 0.905
THRS = (0.88, 0.89, 0.905, 0.92, 0.93)
SEEDS = (43, 44)
OUT = os.path.join("reports", "early_target_check.md")


def lgbm(seed: int):
    m = train.make_lightgbm()
    m.set_params(random_state=seed)
    return m


def half(c: str) -> str:
    return "A" if int(hashlib.md5(c.encode()).hexdigest(), 16) % 2 == 0 else "B"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    argparse.ArgumentParser(description="Стійкість split_start30: пороги, зерна, половини монет, чи вхід раніший").parse_args()
    d, pumps, px, idx, uni = ET.load_data()
    X = d[ET.COLS]
    pos, keep = ET.target(d, pumps, *ET.TARGETS["start30"])
    base = train.make_lightgbm().fit(X, d["label"].astype(int))
    early = {s: lgbm(s).fit(X[keep], pos[keep].astype(int)) for s in (42, *SEEDS)}
    variants = {f"split thr {t}": (42, t) for t in THRS} | {f"split seed {s}": (s, THR) for s in SEEDS}
    trained = set(d["base_asset"])

    m1 = joblib.load(train.model_path("lightgbm"))
    m15 = joblib.load(train.model_path(train.M15_NAME))
    exit_bars, share = m15.get("exit_bars", 4), m15.get("confirm", {}).get("share", 0.75)
    start, end = pd.Timestamp(I.START, tz="UTC"), pd.Timestamp(I.END, tz="UTC") + pd.Timedelta(days=1)
    coins = sorted(c for c in uni if c not in trained and os.path.exists(os.path.join(I.DIR, f"{c}.pkl")))
    print(f"бектест: {len(coins)} монет…", flush=True)
    rows, lead = [], []
    for n, c in enumerate(coins, 1):
        x = pickle.load(open(os.path.join(I.DIR, f"{c}.pkl"), "rb"))
        f15, h1, h = x["f15"], x["h1"], x["h"]
        f15 = f15.join(I.new_features(h, idx).reindex(f15["ts"]).set_axis(f15.index))
        hh = h[(h.index >= start - pd.Timedelta(days=1)) & (h.index < end)]
        s1 = h1[["base_asset", "ts"]].assign(probability=m1["model"].predict_proba(h1[m1["features"]])[:, 1], threshold=m1["threshold"])

        def alert(model, thr):
            s15 = f15[["base_asset", "ts"]].assign(probability=model.predict_proba(f15[ET.COLS])[:, 1], threshold=thr)
            t = tier_frame(s15, s1, share).set_index("ts").sort_index()
            return t.loc[(t.index >= start) & (t.index < end), "alert_score"]
        a0 = alert(base, THR_BASE)
        sc = {"base": (a0, a0)}
        cache = {}
        for v, (seed, thr) in variants.items():
            if (seed, thr) not in cache:
                cache[(seed, thr)] = alert(early[seed], thr).reindex(a0.index)
            ea = cache[(seed, thr)]
            sc[v] = (ea, np.maximum(a0, ea))
        months = uni[c]
        base_on = a0.index[a0.to_numpy() >= 1]
        for v in sc:
            for r in ET.simulate_split(hh, *sc[v], I.K, exit_bars, 3.0, 0.5, True):
                ts = pd.Timestamp(r["ts"])
                mth = ts.strftime("%Y-%m")
                if mth not in months:
                    continue
                rows.append({"variant": v, "ticker": c, "month": mth, "band": months[mth], **r})
                if v == f"split thr {THR}":
                    ex = ts + r["bars"] * ET.BAR
                    b = base_on[(base_on >= ts - pd.Timedelta(hours=4)) & (base_on <= ex)]
                    lead.append({"band": months[mth], "ret": r["ret"], "has_base": bool(len(b)),
                                 "lead_min": (b[0] - ts) / pd.Timedelta(minutes=1) if len(b) else np.nan,
                                 "px": hh.at[b[0], "price"] / hh.at[ts, "price"] - 1 if len(b) and b[0] in hh.index else np.nan})
        if n % 50 == 0:
            print(f"  {n}/{len(coins)}", flush=True)
    tr = pd.DataFrame(rows)
    tr["coins"] = tr["ticker"].map(half)
    tr["period"] = np.where(tr["month"] <= "2026-06", "квіт–черв", "лип–вер")
    pf = I.pf
    L = ["# Стійкість split_start30 (вхід — рання модель, утримання — обидві)", "",
         f"Пороги: base {THR_BASE}, рання {THR} (зі звіту early_target.md). {tr['ticker'].nunique()} монет чесного списку, 6 міс.", "",
         "## Пороги, зерна, половини", "",
         "| варіант | угод | PF | ≥ 20 млн | 5–20 млн | монети A | монети B | квіт–черв | лип–вер | медіана |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for v in ["base", *variants]:
        g = tr[tr["variant"] == v]
        r = g["ret"]
        L.append(f"| {v} | {len(g)} | {pf(r):.2f} | {pf(r[g['band'] == '≥ 20 млн']):.2f} | {pf(r[g['band'] == '5–20 млн']):.2f} | "
                 f"{pf(r[g['coins'] == 'A']):.2f} | {pf(r[g['coins'] == 'B']):.2f} | {pf(r[g['period'] == 'квіт–черв']):.2f} | "
                 f"{pf(r[g['period'] == 'лип–вер']):.2f} | {r.median():+.2%} |")
    ld = pd.DataFrame(lead)
    w = ld[ld["has_base"]]
    L += ["", f"## Чи вхід раніший (split thr {THR}, {len(ld)} угод)", "",
          f"- Alert base був у вікні [вхід − 4 год, вихід]: {ld['has_base'].mean():.0%} угод; без нього — {(~ld['has_base']).mean():.0%}.",
          f"- Де був: Alert base РАНІШЕ за вхід split — {(w['lead_min'] < 0).mean():.0%}, у той самий бар — {(w['lead_min'] == 0).mean():.0%}, "
          f"пізніше — {(w['lead_min'] > 0).mean():.0%}; медіана «Alert base − вхід» серед пізніших {w.loc[w['lead_min'] > 0, 'lead_min'].median():.0f} хв, "
          f"ціна на Alert base вища за вхід на {w.loc[w['lead_min'] > 0, 'px'].median():+.2%} (медіана).",
          f"- PF угод з Alert base у вікні {pf(w['ret']):.2f} (≥ 20 млн {pf(w.loc[w['band'] == '≥ 20 млн', 'ret']):.2f}); "
          f"без нього {pf(ld.loc[~ld['has_base'], 'ret']):.2f} (≥ 20 млн {pf(ld.loc[~ld['has_base'] & (ld['band'] == '≥ 20 млн'), 'ret']):.2f}).", ""]
    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
