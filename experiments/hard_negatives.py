"""
experiments/hard_negatives.py — як зменшити хибні тривоги 15m-моделі (2026-10-09, власник: «практики, щоб зменшити кількість
хибних»). Причина хибних відома з 01.10: навчальні монети обрано, бо на них були пампи, — на звичайних монетах сплеск частіше
нічого не означає. Ознаки — робочої моделі (MODEL_FEATURES_15M + власний рух / Amihud / волатильність).

Варіанти (зафіксовано ДО запуску):
    base      як зараз: class_weight balanced, поріг — найкращий F1 на OOF
    f05       той самий, поріг — найкращий F0.5 (точність важить удвічі більше за повноту)
    softw     ваги класів м'якші: scale_pos_weight = √(негативи / позитиви) замість повного вирівнювання; поріг F1
    neg       + «звичайні» монети як негативи: половина A монет кешу бектесту (за хешем назви), 100 монет, квітень–червень,
              кожен 2-й 15m-бар, мітка 0 (нерозмічені імпульси там стають шумом — свідомо); поріг F1
    all       neg + softw + поріг F0.5
Перевірка 1 — leave-ticker-out на розмітці (негативи завжди в навчальній частині; поріг — з OOF розмічених тікерів).
Перевірка 2 — бектест лише на половині B монет і лише ЛИПЕНЬ–ВЕРЕСЕНЬ (монети й місяці, яких модель не бачила); Alert =
варіант + робоча годинна, частковий вихід, стоп 2.48 ATR.
→ reports/hard_negatives.md, reports/hard_negatives_trades.csv

    python experiments/hard_negatives.py                 # негативи — половина A, перевірка — B
    python experiments/hard_negatives.py --neg-half B    # навпаки (перехресна перевірка) → reports/hard_negatives_swap.md
"""

import argparse
import hashlib
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import numpy as np
import pandas as pd
import psycopg
from dotenv import load_dotenv
from sklearn.metrics import precision_recall_curve

import exit_partial as P
import idio_features as I
import range_features as R
import train
from features import INTERVAL_15M, MODEL_FEATURES_15M, build_from_raw, aggregate, load_pumps, load_raw, query_df, short_features
from fetch import excluded_tickers, read_tickers_file
from tiers import tier_frame

COLS = MODEL_FEATURES_15M + R.MOVE
VARIANTS = ["base", "f05", "softw", "neg", "all"]
N_NEG, NEG_FROM, NEG_TO, NEG_STEP = 100, "2026-04-01", "2026-07-01", 2
TEST_FROM, TEST_TO = "2026-07-01", "2026-09-30"
OUT = os.path.join("reports", "hard_negatives.md")


def half(coin: str) -> str:
    return "A" if int(hashlib.md5(coin.encode()).hexdigest(), 16) % 2 == 0 else "B"


def fbeta_threshold(y: np.ndarray, p: np.ndarray, beta: float) -> float:
    prec, rec, thr = precision_recall_curve(y, p)
    b2 = beta * beta
    f = (1 + b2) * prec[:-1] * rec[:-1] / np.maximum(b2 * prec[:-1] + rec[:-1], 1e-12)
    return float(thr[np.argmax(f)])


def make(soft: bool, ratio: float):
    m = train.make_lightgbm()
    if soft:
        m.set_params(class_weight=None, scale_pos_weight=float(np.sqrt(ratio)))
    return m


def oof(X, y, grp, Xn, soft: bool) -> np.ndarray:
    """OOF на розмічених тікерах; негативи Xn (якщо є) — завжди в навчальній частині."""
    p = np.zeros(len(y))
    for g in grp.unique():
        te = (grp == g).to_numpy()
        Xt, yt = X[~te], y[~te]
        if Xn is not None:
            Xt, yt = pd.concat([Xt, Xn]), pd.concat([yt, pd.Series(0, index=Xn.index)])
        ratio = (yt == 0).sum() / max((yt == 1).sum(), 1)
        p[te] = make(soft, ratio).fit(Xt, yt).predict_proba(X[te])[:, 1]
    return p


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Менше хибних тривог: поріг F0.5, м'якші ваги класів, звичайні монети як негативи")
    ap.add_argument("--neg-half", choices=["A", "B"], default="A",
                    help="з якої половини кешу брати негативи (інша — перевірка); B — перехресна перевірка → reports/hard_negatives_swap.md")
    args = ap.parse_args()
    nh, th = args.neg_half, "B" if args.neg_half == "A" else "A"
    out_md = OUT if nh == "A" else OUT.replace(".md", "_swap.md")
    out_csv = os.path.join("reports", "hard_negatives_trades.csv" if nh == "A" else "hard_negatives_swap_trades.csv")
    load_dotenv(".env")
    uni = json.load(open(os.path.join(I.DIR, "_universe.json"), encoding="utf-8"))
    print("індекс ALT-EW з кешу…", flush=True)
    idx = I.market_index()
    excluded = excluded_tickers(".")
    reviewed = {t.upper() for t in read_tickers_file("tickers.txt")} - excluded
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        pumps = load_pumps(conn)
        pumps = pumps[~pumps["base_asset"].isin(excluded)]
        bases = sorted(reviewed | set(pumps["base_asset"]))
        raw = load_raw(conn, bases, INTERVAL_15M, exchanges=["Binance"])
        d = build_from_raw(raw, pumps, bars_per_hour=4, extra=short_features)
        first1 = query_df(conn, "SELECT base_asset, min(ts) AS t FROM features GROUP BY 1").set_index("base_asset")["t"]
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=48)
    d = d[d["base_asset"].isin(reviewed) | d["pattern_type"].notna()]
    d = d[(d["ts"] <= cutoff) | d["pattern_type"].notna()]
    d = d[d["ts"] >= d["base_asset"].map(first1)].reset_index(drop=True)
    parts = []
    for b, g in d.groupby("base_asset"):
        a = aggregate(raw, b, "15min")
        parts.append(I.new_features(a, idx).reindex(g["ts"]).set_axis(g.index))
    d = d.join(pd.concat(parts)).sort_values(["base_asset", "ts"]).reset_index(drop=True)
    X, y, grp = d[COLS], d["label"].astype(int), d["base_asset"]
    trained = set(d["base_asset"])
    coins = sorted(c for c in uni if c not in trained and os.path.exists(os.path.join(I.DIR, f"{c}.pkl")))
    neg_coins = [c for c in coins if half(c) == nh][:N_NEG]
    test_coins = [c for c in coins if half(c) == th]
    # негативи: звичайні монети половини A, квіт–черв, кожен 2-й бар
    nf, n0, n1 = [], pd.Timestamp(NEG_FROM, tz="UTC"), pd.Timestamp(NEG_TO, tz="UTC")
    for c in neg_coins:
        x = pickle.load(open(os.path.join(I.DIR, f"{c}.pkl"), "rb"))
        f15, h = x["f15"], x["h"]
        f15 = f15.join(I.new_features(h, idx).reindex(f15["ts"]).set_axis(f15.index))
        f15 = f15[(f15["ts"] >= n0) & (f15["ts"] < n1)].iloc[::NEG_STEP]
        nf.append(f15[COLS])
    Xn = pd.concat(nf, ignore_index=True)
    Xn.index = Xn.index + 10 ** 9                                       # не перетинатись з індексами X
    print(f"навчання: {len(d)} барів розмітки ({y.mean():.1%} імпульсних) + {len(Xn)} барів звичайних монет "
          f"({len(neg_coins)} монет, половина {nh}); перевірка — {len(test_coins)} монет половини {th}, {TEST_FROM} – {TEST_TO}", flush=True)
    yy = y.to_numpy()
    spec = {"base": (False, False, 1.0), "f05": (False, False, 0.5), "softw": (True, False, 1.0),
            "neg": (False, True, 1.0), "all": (True, True, 0.5)}     # (м'які ваги, негативи, β порогу)
    cache, thr, lto, models = {}, {}, {}, {}
    for v in VARIANTS:
        soft, use_neg, beta = spec[v]
        key = (soft, use_neg)
        if key not in cache:
            cache[key] = oof(X, y, grp, Xn if use_neg else None, soft)
            Xa, ya = (pd.concat([X, Xn]), pd.concat([y, pd.Series(0, index=Xn.index)])) if use_neg else (X, y)
            ratio = (ya == 0).sum() / max((ya == 1).sum(), 1)
            models[key] = make(soft, ratio).fit(Xa, ya)
            print(f"  модель {key} готова", flush=True)
        p = cache[key]
        thr[v] = fbeta_threshold(yy, p, beta)
        lto[v] = train.event_metrics(d, p >= thr[v], pumps, train.BAR_15M)
        print(f"  {v}: поріг {thr[v]:.3f}, {lto[v]}", flush=True)

    m1 = joblib.load(train.model_path("lightgbm"))
    m15 = joblib.load(train.model_path(train.M15_NAME))
    exit_bars, share = m15.get("exit_bars", 4), m15.get("confirm", {}).get("share", 0.75)
    start, end = pd.Timestamp(TEST_FROM, tz="UTC"), pd.Timestamp(TEST_TO, tz="UTC") + pd.Timedelta(days=1)
    rows, alerts = [], {v: 0 for v in VARIANTS}
    for n, c in enumerate(test_coins, 1):
        x = pickle.load(open(os.path.join(I.DIR, f"{c}.pkl"), "rb"))
        f15, h1, h = x["f15"], x["h1"], x["h"]
        f15 = f15.join(I.new_features(h, idx).reindex(f15["ts"]).set_axis(f15.index))
        hh = h[(h.index >= start - pd.Timedelta(days=1)) & (h.index < end)]
        s1 = h1[["base_asset", "ts"]].assign(probability=m1["model"].predict_proba(h1[m1["features"]])[:, 1], threshold=m1["threshold"])
        months = uni[c]
        for v in VARIANTS:
            soft, use_neg, _ = spec[v]
            s15 = f15[["base_asset", "ts"]].assign(probability=models[(soft, use_neg)].predict_proba(f15[COLS])[:, 1], threshold=thr[v])
            t = tier_frame(s15, s1, share).set_index("ts").sort_index()
            t = t[(t.index >= start) & (t.index < end)]
            alerts[v] += int((t["tier"] == "alert").sum())
            for r in P.simulate(hh, t["alert_score"], I.K, exit_bars, 3.0, 0.5, True):
                mth = pd.Timestamp(r["ts"]).strftime("%Y-%m")
                if mth in months:
                    rows.append({"variant": v, "ticker": c, "month": mth, "band": months[mth], **r})
        if n % 50 == 0:
            print(f"  {n}/{len(test_coins)}", flush=True)
    tr = pd.DataFrame(rows)
    tr.to_csv(out_csv, index=False)
    pf, days = I.pf, (end - start).days
    L = ["# Менше хибних тривог: поріг F0.5, м'якші ваги, звичайні монети як негативи", "",
         f"Ознаки робочої моделі. Негативи — {len(neg_coins)} звичайних монет (половина {nh} кешу), {NEG_FROM} – {NEG_TO}, кожен "
         f"{NEG_STEP}-й бар, {len(Xn)} барів. Бектест — лише половина {th} ({tr['ticker'].nunique()} монет з угодами) і лише "
         f"{TEST_FROM} – {TEST_TO}.", "",
         "## Leave-ticker-out на розмітці (Кандидат 15m)", "",
         "| варіант | поріг | спіймано | затримка, год | фальстартів | хибних / 30 дн |", "|---|---|---|---|---|---|"]
    for v in VARIANTS:
        m = lto[v]
        L.append(f"| {v} | {thr[v]:.3f} | {m['pumps_caught']} | {m['median_delay_h']:.2f} | {m['false_starts_triggered']} | "
                 f"{m['false_alarms_per_30d']:.1f} |")
    L += ["", "## Бектест на невидимих монетах і місяцях", "",
          "| варіант | угод | Alert-барів / день | у плюсі | медіана | PF | PF без топ-3 | ≥ 20 млн | 5–20 млн |", "|---|---|---|---|---|---|---|---|---|"]
    for v in VARIANTS:
        g = tr[tr["variant"] == v]
        r = g["ret"].sort_values()
        L.append(f"| {v} | {len(r)} | {alerts[v] / days:.0f} | {(r > 0).mean():.0%} | {r.median():+.2%} | {pf(r):.2f} | "
                 f"{pf(r.iloc[:-3]):.2f} | {pf(g[g['band'] == '≥ 20 млн']['ret']):.2f} | {pf(g[g['band'] == '5–20 млн']['ret']):.2f} |")
    piv = tr.groupby(["month", "variant"])["ret"].apply(pf).unstack()[VARIANTS]
    piv20 = tr[tr["band"] == "≥ 20 млн"].groupby(["month", "variant"])["ret"].apply(pf).unstack()[VARIANTS]
    L += ["", "## По місяцях (PF; у дужках ≥ 20 млн)", "", "| місяць | " + " | ".join(VARIANTS) + " |", "|---|" + "---|" * len(VARIANTS)]
    for mth in piv.index:
        L.append(f"| {mth} | " + " | ".join(f"{piv.at[mth, v]:.2f} ({piv20.at[mth, v]:.2f})" for v in VARIANTS) + " |")
    L.append("")
    open(out_md, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
