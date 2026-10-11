"""
experiments/oct_holdout.py — ЖОВТЕНЬ ЯК HOLDOUT: student (teacher + soft labels) проти base і робочої моделі (2026-10-11; власник:
«окей давай», «якщо teacher + soft labels дасть кращий результат — приймаємо?»). Жовтня не було в жодному відборі варіантів.
Етап data: монети жовтня — ті самі правила, що й чесний список (середній денний оборот за 30 днів до 01.10 — 5 млн … 1 млрд $, з
денних свічок архіву), з наявних у кеші data/long_bin; дані 01.09–10.10 з архіву Binance → локальна база (binance_archive.collect,
лише Binance, як long_binance.py) → ознаки (15m робочі + годинні з власних 15m + ціни) → data/oct_bin/<монета>.pkl.
Етап test (КРИТЕРІЙ зафіксовано ДО запуску й до перегляду жовтня; власник погодив у чаті 11.10):
  * чистий період: base (розмітка власника) і student (розмітка teacher на монетах архіву) вчаться лише на барах до 27.09;
    teacher — лише на подіях власника зі стартом до 27.09; random seeds 42 / 43 / 44 для обох;
  * робоча модель (models/lightgbm_15m.joblib, 08.10) — як є (вона бачила жовтень лише тікерів власника, не монет holdout);
  * бектест 01.10–10.10 на монетах жовтня поза тікерами власника; Alert = 15m + робоча годинна ≥ 0.75; частковий вихід; стоп I.K;
  * student ПРИЙМАЄТЬСЯ ЯК КАНДИДАТ, якщо: середній PF student ≥ PF base і ≥ PF робочої — і загалом, і на ≥ 20 млн; і хибних на
    жовтневій розмітці власника (Кандидат 15m) ≤ base. Нічия або гірше — лишаємо поточну. Кандидат ≠ прийняття: далі train.py,
    /compare, журнал, рішення власника.
→ reports/oct_holdout.md

    python experiments/oct_holdout.py --stage data
    python experiments/oct_holdout.py --stage test
"""

import argparse
import json
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import psycopg
from dotenv import load_dotenv

import binance_archive as BA
import long_binance as LB
from features import FEATURES, INTERVAL_15M, MODEL_FEATURES_15M, aggregate, build_from_raw, hourly_from_intraday, load_raw, short_features

DIR = os.path.join("data", "oct_bin")
WARM, END = "2026-09-01", "2026-10-10"
HOLD_FROM = "2026-10-01"
UNIVERSE = os.path.join(DIR, "_universe.json")
DONE = os.path.join(DIR, "_data_done.txt")
OUT = os.path.join("reports", "oct_holdout.md")


def universe_oct(coins: list[str]) -> dict:
    """{монета: смуга} на жовтень: середній денний оборот за 30 днів до 01.10 (денні свічки архіву), 5 млн – 1 млрд $."""
    if os.path.exists(UNIVERSE):
        return json.load(open(UNIVERSE, encoding="utf-8"))
    from concurrent.futures import ThreadPoolExecutor
    m0 = pd.Timestamp(HOLD_FROM, tz="UTC")

    def one(c):
        k = BA._csv("klines", f"{c}USDT", "2026-09", "1d", monthly=True)
        if not len(k):
            return c, None
        s = pd.Series(k["quote_volume"].astype(float).to_numpy(), index=pd.to_datetime(k["open_time"].astype("int64"), unit="ms", utc=True))
        prev = s[(s.index >= m0 - pd.Timedelta(days=30)) & (s.index < m0)]
        if len(prev) < 7 or not (LB.MIN_TURN <= prev.mean() <= LB.MAX_TURN):
            return c, None
        return c, "≥ 20 млн" if prev.mean() >= 20e6 else "5–20 млн"
    with ThreadPoolExecutor(8) as ex:
        uni = {c: b for c, b in ex.map(one, coins) if b}
    os.makedirs(DIR, exist_ok=True)
    json.dump(uni, open(UNIVERSE, "w", encoding="utf-8"), ensure_ascii=False)
    return uni


def stage_data() -> None:
    load_dotenv(".env")
    have = sorted(os.path.basename(p)[:-4] for p in os.listdir(LB.DIR) if p.endswith(".pkl"))
    uni = universe_oct(have)
    coins = sorted(uni)
    print(f"монет жовтня: {len(coins)} (з {len(have)} у кеші архіву); ≥ 20 млн — {sum(b == '≥ 20 млн' for b in uni.values())}", flush=True)
    os.makedirs(DIR, exist_ok=True)
    done = set(open(DONE, encoding="utf-8").read().split()) if os.path.exists(DONE) else set()
    t0 = time.time()
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        LB.ensure_markets(conn, coins)
        for n, c in enumerate(coins, 1):
            if c not in done:
                try:
                    BA.collect(conn, [c], WARM, END, ("15m",), log=lambda m: None)
                    with open(DONE, "a", encoding="utf-8") as f:
                        f.write(c + "\n")
                except Exception as e:                  # noqa: BLE001
                    conn.rollback()
                    print(f"  ! {c}: {type(e).__name__}: {e}", flush=True)
                    continue
            path = os.path.join(DIR, f"{c}.pkl")
            if not os.path.exists(path):
                raw = load_raw(conn, [c], INTERVAL_15M, since=pd.Timestamp(WARM, tz="UTC"), exchanges=LB.BIN)
                if not raw["ohlcv"].empty:
                    f15 = build_from_raw(raw, None, bars_per_hour=4, extra=short_features)
                    h1 = build_from_raw(hourly_from_intraday(raw))
                    h = aggregate(raw, c, "15min")
                    if not f15.empty and not h1.empty:
                        with open(path + ".tmp", "wb") as f:
                            pickle.dump({"f15": f15[["base_asset", "ts", *MODEL_FEATURES_15M]], "h1": h1[["base_asset", "ts", *FEATURES]],
                                         "h": h[["open", "high", "low", "price", "volume"]]}, f)
                        os.replace(path + ".tmp", path)
            if n % 20 == 0:
                print(f"  {n}/{len(coins)}, {time.time() - t0:.0f} с", flush=True)
    print("готово", flush=True)


CUT = pd.Timestamp("2026-09-27", tz="UTC")
SEEDS = [42, 43, 44]
BIG = "≥ 20 млн"


def market_index_full() -> pd.Series:
    """ALT-EW: 6 місяців з кешу архіву + вересень–жовтень з data/oct_bin (той самий розрахунок, що idio_features.market_index)."""
    import glob
    import idio_features as I
    rets = {}
    for p in glob.glob(os.path.join(DIR, "*.pkl")):
        h = pickle.load(open(p, "rb"))["h"]
        rets[os.path.basename(p)[:-4]] = np.log(h["price"].where(h["price"] > 0)).diff()
    w = pd.DataFrame(rets)
    oct_idx = w.mean(axis=1).where(w.notna().sum(axis=1) >= 50)
    base = I.market_index()
    return base.combine_first(oct_idx).sort_index()


def stage_test() -> None:
    import joblib
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
    idx_full = market_index_full()
    I.market_index = lambda: idx_full                       # ET.load_data і ознаки власного руху — з індексом, що покриває жовтень
    k, own_p, own_f, reviewed, days = LT.owner_events()
    k = k[k["start"] < CUT].reset_index(drop=True)
    p_oof = train.oof_proba(train.make_lightgbm, k[LT.OPIS], k["y"], k["base"])
    cal = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p_oof, k["y"])
    teacher = LT.fit_teacher(k)
    print(f"teacher: {len(k)} подій власника до {CUT:%d.%m}", flush=True)

    d, pumps, px, idx, uni = ET.load_data()
    trained = set(d["base_asset"])
    coins = sorted(c for c in uni if c not in trained and os.path.exists(os.path.join(I.DIR, f"{c}.pkl")))
    parts = []
    for n, c in enumerate(coins, 1):
        x, f15 = AW.load_coin(c, idx_full)
        f15 = f15[f15["ts"] < CUT]
        s = AT.coin_series(c, x["h"])
        e = pd.DataFrame(LT.event_rows(s, idx_full.reindex(s.index)))
        e = e[e["end_ts"] < CUT] if len(e) else e
        if len(e):
            e["pc"] = cal.predict(teacher.predict_proba(e[LT.OPIS])[:, 1])
            pp, _, near = TC.soft_bars(f15["ts"], e)
        else:
            pp, near = np.zeros(len(f15)), np.zeros(len(f15), bool)
        keep = near | (np.arange(len(f15)) % TC.NEG_KEEP == 0)
        parts.append(f15[["base_asset", "ts", *COLS]].astype({q: "float32" for q in COLS}).assign(p=pp, w=np.where(near, 1.0, float(TC.NEG_KEEP)))[keep])
        if n % 100 == 0:
            print(f"  розмітка teacher: {n}/{len(coins)}", flush=True)
    a = pd.concat(parts, ignore_index=True)
    yh = (a["p"] >= 0.5).to_numpy().astype(int)
    do = d[d["ts"] < CUT]
    Xo, yo = do[COLS], do["label"].to_numpy().astype(int)
    models = {}
    for sd in SEEDS:
        po = train.oof_proba(lambda s=sd: TC.lgbm(s), Xo, pd.Series(yo), do["base_asset"])
        models[("base", sd)] = (TC.lgbm(sd).fit(Xo, yo), train.best_f1_threshold(yo, po))
        thr = TC.gkf_threshold(a[COLS + ["p"]], yh, a["base_asset"], a["w"].to_numpy(), sd)
        Xe, ye, we = TC.expand(a[COLS + ["p"]], a["w"].to_numpy())
        models[("student", sd)] = (TC.lgbm(sd).fit(Xe[COLS], ye, sample_weight=we), thr)
        print(f"  seed {sd}: пороги base {models[('base', sd)][1]:.3f}, student {thr:.3f}", flush=True)
    del a
    m15 = joblib.load(train.model_path(train.M15_NAME))
    m1 = joblib.load(train.model_path("lightgbm"))
    exit_bars, share = m15.get("exit_bars", 4), m15.get("confirm", {}).get("share", 0.75)

    # жовтнева розмітка власника (Кандидат 15m)
    dq = d[d["ts"] >= pd.Timestamp(HOLD_FROM, tz="UTC")].reset_index(drop=True)
    pq = pumps[pumps["start_ts"] >= pd.Timestamp(HOLD_FROM, tz="UTC")]
    gold = {}
    for (v, sd), (mdl, thr) in models.items():
        gold[(v, sd)] = train.event_metrics(dq, mdl.predict_proba(dq[COLS])[:, 1] >= thr, pq, ET.BAR)
    gold[("working", 0)] = train.event_metrics(dq, m15["model"].predict_proba(dq[m15["features"]])[:, 1] >= m15["threshold"], pq, ET.BAR)

    # бектест 01.10–10.10
    ou = json.load(open(UNIVERSE, encoding="utf-8"))
    oc = sorted(c for c in ou if c not in trained and os.path.exists(os.path.join(DIR, f"{c}.pkl")))
    start, end = pd.Timestamp(HOLD_FROM, tz="UTC"), pd.Timestamp(END, tz="UTC") + pd.Timedelta(days=1)
    rows = []
    for c in oc:
        x = pickle.load(open(os.path.join(DIR, f"{c}.pkl"), "rb"))
        f15, h1, h = x["f15"], x["h1"], x["h"]
        f15 = f15.join(I.new_features(h, idx_full).reindex(f15["ts"]).set_axis(f15.index))
        hh = h[(h.index >= start - pd.Timedelta(days=1)) & (h.index < end)]
        s1 = h1[["base_asset", "ts"]].assign(probability=m1["model"].predict_proba(h1[m1["features"]])[:, 1], threshold=m1["threshold"])
        sets = {**{key: (mdl, thr, COLS) for key, (mdl, thr) in models.items()},
                ("working", 0): (m15["model"], m15["threshold"], m15["features"])}
        for (v, sd), (mdl, thr, cols) in sets.items():
            s15 = f15[["base_asset", "ts"]].assign(probability=mdl.predict_proba(f15[cols])[:, 1], threshold=thr)
            t = tier_frame(s15, s1, share).set_index("ts").sort_index()
            for r in P.simulate(hh, t.loc[(t.index >= start) & (t.index < end), "alert_score"], I.K, exit_bars, 3.0, 0.5, True):
                if pd.Timestamp(r["ts"]) >= start:
                    rows.append({"variant": v, "seed": sd, "ticker": c, "band": ou[c], **r})
    tr = pd.DataFrame(rows)
    tr.to_csv(OUT.replace(".md", "_trades.csv"), index=False)
    pf = I.pf
    res = {}
    for (v, sd) in sets:
        g = tr[(tr["variant"] == v) & (tr["seed"] == sd)]
        res[(v, sd)] = {"pf": pf(g["ret"]), "pf20": pf(g[g["band"] == BIG]["ret"]), "n": len(g), "wo3": pf(g["ret"].sort_values().iloc[:-3]),
                        "med": g["ret"].median(), "fa": gold[(v, sd)]["false_alarms_per_30d"], "caught": gold[(v, sd)]["pumps_caught"]}
    mean = lambda v, key: float(np.mean([res[(v, sd)][key] for sd in (SEEDS if v != "working" else [0])]))   # noqa: E731
    L = ["# Жовтень як holdout: student (teacher + soft labels) проти base і робочої моделі", "",
         f"Бектест {HOLD_FROM} – {END}, {len(oc)} монет жовтня поза тікерами власника (≥ 20 млн — {sum(ou[c] == BIG for c in oc)}). base і "
         f"student вчились лише до {CUT:%d.%m} (teacher — на подіях власника до {CUT:%d.%m}), seeds {SEEDS}; робоча модель — як є (08.10). "
         f"Розмітка власника за жовтень: {len(pq[pq['pattern_type'] == 'pump'])} імпульсів.", "",
         "**Критерій (до запуску, погоджено з власником 11.10):** student — кандидат, якщо середній PF ≥ base і ≥ робочої і загалом, і "
         "на ≥ 20 млн, а хибних на жовтневій розмітці власника ≤ base. Нічия або гірше — лишаємо поточну.", "",
         "| модель | seed | угод | PF | PF без топ-3 | ≥ 20 млн | медіана | розмітка жовтня: спіймано | хибних / 30 дн |", "|---|---|---|---|---|---|---|---|---|"]
    for (v, sd), r in res.items():
        L.append(f"| {v} | {sd or '—'} | {r['n']} | {r['pf']:.2f} | {r['wo3']:.2f} | {r['pf20']:.2f} | {r['med']:+.2%} | {r['caught']} | {r['fa']:.1f} |")
    L += ["", "| модель (середнє) | PF | ≥ 20 млн | хибних / 30 дн |", "|---|---|---|---|"]
    for v in ("base", "student", "working"):
        L.append(f"| {v} | {mean(v, 'pf'):.3f} | {mean(v, 'pf20'):.3f} | {mean(v, 'fa'):.1f} |")
    ok = (mean("student", "pf") >= max(mean("base", "pf"), mean("working", "pf")) and
          mean("student", "pf20") >= max(mean("base", "pf20"), mean("working", "pf20")) and mean("student", "fa") <= mean("base", "fa"))
    L += ["", f"## Критерій\n\n→ **{'ВИКОНАНО — student кандидат' if ok else 'НЕ виконано — лишаємо поточну'}**"]
    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Жовтень як holdout: student (teacher + soft labels) проти base і робочої моделі")
    ap.add_argument("--stage", choices=["data", "test"], required=True, help="data — архів жовтня і ознаки; test — моделі й бектест")
    args = ap.parse_args()
    stage_data() if args.stage == "data" else stage_test()


if __name__ == "__main__":
    main()
