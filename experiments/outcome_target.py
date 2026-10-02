"""
experiments/outcome_target.py — 15m-модель з ціллю «результат угоди з цього бару» замість «імпульс триває» (2026-10-02).

experiments/level_features.py: ознаки рівнів модель використала, щоб упізнавати СТАН імпульсу (біля хаїв), а не добрий
ВХІД — бо ціль «імпульс триває». Тут ціль інша — triple barrier (López de Prado): вхід за закриттям бару t, і що настане
першим за HORIZON барів:
    +TP_ATR × ATR(15m)    → 1 (угода в плюс)
    −SL_ATR × ATR(15m)    → 0 (стоп)
    час вийшов            → 0
Мітка дивиться в майбутнє лише як ціль; ознаки — лише дані до бару включно (FEATURES_15M + рівні з level_features.py).

Вибірка — лише «живі» моменти (EVENT: обсяг 4 год ≥ 2× норми 7 днів або |зміна ціни за 1 год| ≥ 2%) на всіх монетах з
15m-даними (тікери навчання + монети бектесту). Перевірка без підглядання ні в монети, ні в час:
    монети — дві групи A / B (випадково, seed 0); час — ранній період (до SPLIT) / пізній
    fold 1: навчання — A, ранній → тест — B, пізній;   fold 2: навчання — B, ранній → тест — A, пізній
Поріг входу — квантиль прогнозів на НАВЧАЛЬНІЙ частині (топ 20 / 10 / 5 / 2%), на тесті не підбирається.
Угоди на тесті — за тими самими бар'єрами, без перекриття (наступний вхід — після виходу), комісія 0.1%.
Порівняння: усі живі моменти (база), робоча модель (Alert 15m + 1h на тих самих тестових монетах і періоді).

    python experiments/outcome_target.py
    python experiments/outcome_target.py --refresh      # перебудувати ознаки (кеш reports/backtests/_outcome_feats.pkl)
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd
import psycopg
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score

import train
from features import FEATURES_15M, INTERVAL_15M, aggregate, build_from_raw, load_raw
from fetch import read_tickers_file
from level_features import LEVELS, extra
from risk import atr

TP_ATR, SL_ATR, HORIZON, FEE = 3.0, 1.5, 16, 0.001
CACHE = os.path.join("reports", "backtests", "_outcome_feats.pkl")
COLS = FEATURES_15M + LEVELS


def barrier(h: pd.DataFrame) -> pd.DataFrame:
    """Мітка й результат угоди з кожного бару: вхід за закриттям, бар'єри ±ATR, час HORIZON барів."""
    c, hi, lo = h["price"].to_numpy(), h["high"].to_numpy(), h["low"].to_numpy()
    a = atr(h).to_numpy()
    n = len(h)
    y, ret, bars = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    for i in range(n - HORIZON):
        if np.isnan(a[i]) or np.isnan(c[i]) or a[i] <= 0:
            continue
        up, dn = c[i] + TP_ATR * a[i], c[i] - SL_ATR * a[i]
        res, k_out = None, HORIZON
        for k in range(1, HORIZON + 1):
            if lo[i + k] <= dn:                        # консервативно: у барі спершу стоп
                res, k_out = (0, dn / c[i] - 1), k
                break
            if hi[i + k] >= up:
                res, k_out = (1, up / c[i] - 1), k
                break
        if res is None:
            res = (0, c[i + HORIZON] / c[i] - 1)
        y[i], ret[i], bars[i] = res[0], res[1] - FEE, k_out
    return pd.DataFrame({"y": y, "ret": ret, "hold": bars}, index=h.index)


def load_all(refresh: bool) -> pd.DataFrame:
    if os.path.exists(CACHE) and not refresh:
        return pd.read_pickle(CACHE)
    coins = {t.upper() for t in read_tickers_file("tickers.txt")}
    for path in glob.glob(os.path.join("reports", "backtests", "2026*_alert.json")):
        coins |= {u["base"] for u in json.load(open(path, encoding="utf-8"))["meta"]["universe"]}
    parts = []
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        for n, b in enumerate(sorted(coins), 1):
            raw = load_raw(conn, [b], INTERVAL_15M)
            f = build_from_raw(raw, None, bars_per_hour=4, extra=extra)
            h = aggregate(raw, b, "15min")
            if f.empty or len(h) < 10 * 96:
                continue
            v4 = h["volume"].rolling(16).sum()
            ev = pd.DataFrame({"vol_x": v4 / v4.shift(16).rolling(7 * 96, min_periods=5 * 96).median(),
                               "r1h": h["price"] / h["price"].shift(4) - 1}, index=h.index)
            lab = barrier(h).join(ev)
            f = f.set_index("ts").join(lab, how="inner")
            f["base_asset"] = b
            parts.append(f.reset_index()[["base_asset", "ts"] + COLS + ["y", "ret", "hold", "vol_x", "r1h"]])
            if n % 25 == 0:
                print(f"  ознаки: {n}/{len(coins)}", flush=True)
    d = pd.concat(parts, ignore_index=True)
    d.to_pickle(CACHE)
    return d


def simulate(g: pd.DataFrame, take: np.ndarray) -> pd.DataFrame:
    """Угоди без перекриття по кожній монеті: вхід на сигнальному барі, наступний — після виходу."""
    out = []
    for b, gg in g[take].sort_values("ts").groupby("base_asset"):
        free = pd.Timestamp.min.tz_localize("UTC")
        for r in gg.itertuples():
            if r.ts < free:
                continue
            out.append({"base_asset": b, "ret": r.ret, "y": r.y})
            free = r.ts + pd.Timedelta(minutes=15) * (r.hold + 1)
    return pd.DataFrame(out)


def stats(t: pd.DataFrame) -> dict:
    if t.empty:
        return {"угод": 0}
    v = t["ret"].sort_values()
    gw, gl = v[v > 0].sum(), -v[v <= 0].sum()
    return {"угод": len(v), "монет": t["base_asset"].nunique(), "у плюсі": f"{(v > 0).mean():.0%}", "до цілі": f"{t['y'].mean():.0%}",
            "медіана": f"{v.median():+.2%}", "сер.": f"{v.mean():+.2%}", "без топ-3": f"{v.iloc[:-3].sum():+.0%}",
            "PF": f"{gw / gl:.2f}" if gl else "∞"}


def main() -> None:
    ap = argparse.ArgumentParser(description="15m-модель з ціллю «результат угоди» (triple barrier), перевірка поза монетами й часом")
    ap.add_argument("--refresh", action="store_true", help="перебудувати ознаки й мітки (інакше кеш)")
    args = ap.parse_args()
    load_dotenv(".env")
    d = load_all(args.refresh)
    d = d[d["y"].notna()]
    d = d[(d["vol_x"] >= 2) | (d["r1h"].abs() >= 0.02)].reset_index(drop=True)           # лише живі моменти
    coins = np.array(sorted(d["base_asset"].unique()))
    rng = np.random.default_rng(0)
    grp = dict(zip(coins, rng.permutation(len(coins)) % 2))
    d["grp"] = d["base_asset"].map(grp)
    split = d["ts"].min() + (d["ts"].max() - d["ts"].min()) * 0.6
    print(f"живих моментів: {len(d)} на {len(coins)} монетах; частка «до цілі» {d['y'].mean():.0%}; "
          f"ранній період до {split:%m-%d}, пізній — після (тест)\n", flush=True)

    m15 = joblib.load(train.model_path(train.M15_NAME))
    rows, tests, aucs, imps = {}, [], [], []
    qs = {"топ 20%": 0.80, "топ 10%": 0.90, "топ 5%": 0.95, "топ 2%": 0.98}
    for fold, (tr_g, te_g) in enumerate(((0, 1), (1, 0)), 1):
        tr = d[(d["grp"] == tr_g) & (d["ts"] < split - pd.Timedelta(hours=4))]       # без перекриття міток з тестом
        te = d[(d["grp"] == te_g) & (d["ts"] >= split)].copy()
        model = train.make_lightgbm().fit(tr[COLS], tr["y"].astype(int))
        p_tr = model.predict_proba(tr[COLS])[:, 1]
        te["p"] = model.predict_proba(te[COLS])[:, 1]
        te["p_rank"] = np.searchsorted(np.sort(p_tr), te["p"]) / len(p_tr)            # квантиль за навчальною частиною
        te["p15"] = m15["model"].predict_proba(te[FEATURES_15M])[:, 1] / m15["threshold"]
        aucs.append(roc_auc_score(te["y"], te["p"]))
        imps.append(pd.Series(model.booster_.feature_importance("gain"), index=COLS))
        tests.append(te)
    te = pd.concat(tests, ignore_index=True)
    rows["усі живі моменти (без моделі)"] = stats(simulate(te, np.ones(len(te), dtype=bool)))
    rows["робоча модель: 15m ≥ порогу"] = stats(simulate(te, (te["p15"] >= 1).to_numpy()))
    for name, q in qs.items():
        rows[f"нова ціль: {name}"] = stats(simulate(te, (te["p_rank"] >= q).to_numpy()))
    rows["нова ціль: топ 10% і 15m ≥ порогу"] = stats(simulate(te, ((te["p_rank"] >= 0.9) & (te["p15"] >= 1)).to_numpy()))
    pd.set_option("display.width", 220)
    print(f"Тест: пізній період, інші монети ({len(te)} живих моментів). AUC нової моделі: "
          f"{aucs[0]:.3f} / {aucs[1]:.3f} (0.5 — вгадування)\n")
    print(pd.DataFrame(rows).T.to_string())
    imp = sum(imps)
    imp = (imp / imp.sum()).sort_values(ascending=False)
    print("\nВажливість ознак (gain), топ-12:")
    for f, v in imp.head(12).items():
        print(f"  {f:<18} {v:.1%}" + ("   ← рівні" if f in LEVELS else ""))


if __name__ == "__main__":
    main()
