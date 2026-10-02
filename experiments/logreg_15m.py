"""
experiments/logreg_15m.py — логістична регресія проти LightGBM у 15m-контурі на нових монетах (2026-10-02).

Усі перевірки на нових монетах досі робила LightGBM. Логрегресія простіша й менше «запам'ятовує» навчальні тікери (їх 17) —
можливо, на нових монетах вона узагальнює краще. Порівняння, однакові ознаки (FEATURES_15M, як у роботі) і дані:

  1) ціль «імпульс триває» (розмітка власника): leave-ticker-out на навчальних тікерах (рівень «Кандидат»), потім бектест
     на монетах бектесту поза навчанням: Alert = 15m-модель ≥ порогу (найкращий F1 на OOF) і робоча 1h ≥ 0.75 порогу;
     вихід поточний (4 бари без Alert / стоп k·ATR) і частковий (½ на +3 ATR, далі беззбиток);
  2) ціль «результат угоди» (triple barrier з experiments/outcome_target.py, той самий кеш і розбиття за монетами й часом):
     AUC і угоди на топ 10% / 2% прогнозів.

    python experiments/logreg_15m.py
"""

import argparse
import glob
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd
import psycopg
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score

import exit_partial as P
import outcome_target as O
import train
from features import FEATURES, FEATURES_15M, INTERVAL_15M, aggregate, build_from_raw, build_hourly_own, load_pumps, load_raw, short_features
from fetch import read_tickers_file
from tiers import tier_frame

MAKERS = {"lightgbm": train.make_lightgbm, "logreg": train.make_logreg}
DAYS = 30


def pf(v: pd.Series) -> float:
    gl = -v[v <= 0].sum()
    return round(v[v > 0].sum() / gl, 2) if gl else np.nan


def summ(rows: list[dict]) -> dict:
    r = pd.DataFrame(rows)
    if r.empty:
        return {"угод": 0}
    v = r["ret"].sort_values()
    return {"угод": len(v), "у плюсі": f"{(v > 0).mean():.0%}", "медіана": f"{v.median():+.2%}", "PF": pf(v),
            "PF без топ-3": pf(v.iloc[:-3]), "PF ≥ 20 млн": pf(r[r["band"] == "≥ 20 млн"]["ret"]),
            "PF 5–20 млн": pf(r[r["band"] == "5–20 млн"]["ret"])}


def main() -> None:
    argparse.ArgumentParser(description="Логрегресія проти LightGBM у 15m-контурі на нових монетах (експеримент)").parse_args()
    load_dotenv(".env")
    reviewed = {t.upper() for t in read_tickers_file("tickers.txt")}
    m15 = joblib.load(train.model_path(train.M15_NAME))
    m1 = joblib.load(train.model_path("lightgbm"))
    k, exit_bars, share = m15.get("risk", {}).get("k"), m15.get("exit_bars", 4), m15.get("confirm", {}).get("share", 0.75)

    # ---------- 1) ціль «імпульс триває»
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        pumps = load_pumps(conn)
        bases = sorted(reviewed | set(pumps["base_asset"]))
        d = build_from_raw(load_raw(conn, bases, INTERVAL_15M), pumps, bars_per_hour=4, extra=short_features)
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=48)
    d = d[d["base_asset"].isin(reviewed) | d["pattern_type"].notna()]
    d = d[(d["ts"] <= cutoff) | d["pattern_type"].notna()].reset_index(drop=True)
    y, groups = d["label"].astype(int), d["base_asset"]
    trained = set(groups)
    lto, models = {}, {}
    for name, make in MAKERS.items():
        print(f"  {name}: leave-ticker-out…", flush=True)
        p = train.oof_proba(make, d[FEATURES_15M], y, groups)
        thr = train.best_f1_threshold(y.to_numpy(), p)
        ev = train.event_metrics(d, p >= thr, pumps, train.BAR_15M)
        lto[name] = {"спіймано": ev["pumps_caught"], "затримка, хв": round(ev["median_delay_h"] * 60) if pd.notna(ev["median_delay_h"]) else None,
                     "фальстарти": ev["false_starts_triggered"], "хибних / 30 д": round(ev["false_alarms_per_30d"], 1)}
        models[name] = (make().fit(d[FEATURES_15M], y), thr)

    coins, band = set(), {}
    for path in glob.glob(os.path.join("reports", "backtests", "2026*_alert.json")):
        dd = json.load(open(path, encoding="utf-8"))
        for u in dd["meta"]["universe"]:
            coins.add(u["base"])
            band[u["base"]] = "≥ 20 млн" if dd["meta"]["min_turnover"] >= 20e6 else "5–20 млн"
    coins = sorted(coins - trained)
    start = pd.Timestamp(datetime.now(timezone.utc)) - pd.Timedelta(days=DAYS)
    since = start - pd.Timedelta(days=9)
    bt = {f"{n} · {e}": [] for n in MAKERS for e in ("поточний вихід", "частковий вихід")}
    print(f"  бектест: {len(coins)} монет поза навчанням…", flush=True)
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        for b in coins:
            raw = load_raw(conn, [b], INTERVAL_15M, since=since)
            f15 = build_from_raw(raw, None, bars_per_hour=4, extra=short_features)
            d1 = build_hourly_own(conn, [b], since=since)
            if f15.empty or d1.empty:
                continue
            s1 = d1[["base_asset", "ts"]].assign(probability=m1["model"].predict_proba(d1[FEATURES])[:, 1], threshold=m1["threshold"])
            h = aggregate(raw, b, "15min")
            h = h[h.index >= start - pd.Timedelta(days=1)]
            for name, (model, thr) in models.items():
                s15 = f15[["base_asset", "ts"]].assign(probability=model.predict_proba(f15[FEATURES_15M])[:, 1], threshold=thr)
                t = tier_frame(s15, s1, share).set_index("ts").sort_index()
                sc = t.loc[t.index >= start, "alert_score"]
                bt[f"{name} · поточний вихід"] += [{**x, "band": band.get(b, "")} for x in P.simulate(h, sc, k, exit_bars, None, 0, False)]
                bt[f"{name} · частковий вихід"] += [{**x, "band": band.get(b, "")} for x in P.simulate(h, sc, k, exit_bars, 3.0, 0.5, True)]

    # ---------- 2) ціль «результат угоди»
    o = O.load_all(False)
    o = o[o["y"].notna()]
    o = o[(o["vol_x"] >= 2) | (o["r1h"].abs() >= 0.02)].reset_index(drop=True)
    cs = np.array(sorted(o["base_asset"].unique()))
    grp = dict(zip(cs, np.random.default_rng(0).permutation(len(cs)) % 2))
    o["grp"] = o["base_asset"].map(grp)
    split = o["ts"].min() + (o["ts"].max() - o["ts"].min()) * 0.6
    out2 = {}
    for name, make in MAKERS.items():
        tests, aucs = [], []
        for tr_g, te_g in ((0, 1), (1, 0)):
            tr = o[(o["grp"] == tr_g) & (o["ts"] < split - pd.Timedelta(hours=4))]
            te = o[(o["grp"] == te_g) & (o["ts"] >= split)].copy()
            model = make().fit(tr[O.COLS], tr["y"].astype(int))
            p_tr = model.predict_proba(tr[O.COLS])[:, 1]
            te["p"] = model.predict_proba(te[O.COLS])[:, 1]
            te["q"] = np.searchsorted(np.sort(p_tr), te["p"]) / len(p_tr)
            aucs.append(roc_auc_score(te["y"], te["p"]))
            tests.append(te)
        te = pd.concat(tests, ignore_index=True)
        for qname, q in (("топ 10%", 0.9), ("топ 2%", 0.98)):
            t = O.simulate(te, (te["q"] >= q).to_numpy())
            v = t["ret"] if len(t) else pd.Series(dtype=float)
            out2[f"{name} · {qname}"] = {"AUC": f"{aucs[0]:.3f} / {aucs[1]:.3f}", "угод": len(v),
                                         "до цілі": f"{t['y'].mean():.0%}" if len(t) else "—", "медіана": f"{v.median():+.2%}" if len(v) else "—",
                                         "PF": pf(v) if len(v) else None}

    pd.set_option("display.width", 220)
    print("\n1а) Ціль «імпульс триває», leave-ticker-out на навчальних тікерах (рівень «Кандидат»):")
    print(pd.DataFrame(lto).T.to_string())
    print(f"\n1б) Бектест на {len(coins)} монетах поза навчанням (Alert = 15m-модель + робоча 1h):")
    print(pd.DataFrame({k_: summ(v) for k_, v in bt.items()}).T.to_string())
    print("\n2) Ціль «результат угоди» (тест — інші монети й пізніший період):")
    print(pd.DataFrame(out2).T.to_string())


if __name__ == "__main__":
    main()
