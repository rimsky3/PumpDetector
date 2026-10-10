"""
experiments/flow_memory.py — «справжній» потік покупок і «пам'ять» монети про пампи (2026-10-09, власник: «так» на ідеї з джерел).
Ознаки (зафіксовано ДО запуску; усе — лише минуле, коефіцієнти — з попередніх 7 днів монети без поточного бару):
  ПОТІК (Order flow and cryptocurrency returns, J. Financial Markets 2026: частина потоку, пояснена ПОПЕРЕДНІМИ доходностями,
  дає відкат, непояснена — продовження). y = частка покупок − 0.5; ковзна OLS (7 днів, мін. 2 дні) на попередні доходності:
      flow_trans_1h / flow_perm_1h   y = buy_ratio_1h на ret_1h, ret_4h, ret_24h годину тому → пояснена / залишок
      flow_perm_1h_z                 залишок ÷ його ковзне σ за 7 днів
      flow_perm_4h                   y = buy_ratio_4h на ret_4h, ret_24h 4 год тому → залишок
      flow_impact_1h                 y = buy_ratio_1h на ret_1h ТІЄЇ Ж години → залишок (купують більше, ніж «дає» ціна = поглинання)
  ПАМ'ЯТЬ (Xu & Livshits, USENIX 2019: кількість попередніх пампів — серед найсильніших ознак цілі P&D). r24 = ціна / ціна 24 год тому − 1;
  епізод — r24 ≥ X без іншого такого бару за попередні 24 год; вікно (t − 30 дн, t − 24 год], потрібно ≥ 14 днів історії (рахунки → на 30 дн):
      mem_n30_10, mem_n30_20         епізодів ≥ 10% / ≥ 20%
      mem_max30                      найбільший r24 у вікні
      mem_days20                     днів від останнього бару з r24 ≥ 20% (стеля 30)
Варіанти: base (робочі ознаки), flow, mem, both. Перевірка — як ensemble.py: leave-ticker-out (Кандидат 15m) і бектест 6 міс.
на чесному списку (Alert = варіант + робоча годинна, частковий вихід, стоп I.K); половини монет (md5); травень–вересень окремо
(у квітні пам'ять неповна — ціни в кеші з 24.03).
→ reports/flow_memory.md, reports/flow_memory_trades.csv

    python experiments/flow_memory.py
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

import early_target as ET
import exit_partial as P
import idio_features as I
import train
from features import INTERVAL_15M, aggregate, build_from_raw, load_pumps, load_raw, query_df, short_features
from fetch import excluded_tickers, read_tickers_file
from tiers import tier_frame

BASE = ET.COLS
FLOW = ["flow_trans_1h", "flow_perm_1h", "flow_perm_1h_z", "flow_perm_4h", "flow_impact_1h"]
MEM = ["mem_n30_10", "mem_n30_20", "mem_max30", "mem_days20"]
SETS = {"base": BASE, "flow": BASE + FLOW, "mem": BASE + MEM, "both": BASE + FLOW + MEM}
DAY = 96                       # 15m-барів на добу
OUT = os.path.join("reports", "flow_memory.md")


def rolling_resid(y: pd.Series, X: pd.DataFrame, win: int = 7 * DAY, minp: int = 2 * DAY) -> tuple[pd.Series, pd.Series]:
    """Ковзна OLS y ~ 1 + X за попередні win барів (без поточного) → (пояснена частина, залишок)."""
    Z = pd.concat([pd.Series(1.0, index=y.index, name="c"), X], axis=1)
    ok = Z.notna().all(axis=1) & y.notna()
    Zv, yv = Z.where(ok, 0.0).to_numpy(), y.where(ok, 0.0).to_numpy()
    k = Zv.shape[1]
    prods = {}
    for i in range(k):
        for j in range(i, k):
            prods[f"z{i}{j}"] = Zv[:, i] * Zv[:, j]
        prods[f"y{i}"] = Zv[:, i] * yv
    R = pd.DataFrame(prods, index=y.index).rolling(win, min_periods=1).sum().shift(1)
    n = R["z00"].to_numpy()
    A = np.zeros((len(y), k, k))
    b = np.zeros((len(y), k))
    for i in range(k):
        b[:, i] = R[f"y{i}"].to_numpy()
        for j in range(i, k):
            A[:, i, j] = A[:, j, i] = R[f"z{i}{j}"].to_numpy()
    good = np.nan_to_num(n) >= minp
    beta = np.full((len(y), k), np.nan)
    if good.any():
        Ag = A[good] + np.eye(k)[None] * 1e-9 * np.maximum(n[good], 1)[:, None, None]
        beta[good] = np.linalg.solve(Ag, b[good][..., None])[..., 0]
    fit = pd.Series((Z.to_numpy() * beta).sum(axis=1), index=y.index)
    fit[~ok] = np.nan
    return fit, y - fit


def flow_features(f: pd.DataFrame) -> pd.DataFrame:
    """f — рядки однієї монети на регулярній 15m-сітці (index ts) з buy_ratio_1h/4h, ret_1h/4h/24h."""
    y1, y4 = f["buy_ratio_1h"] - 0.5, f["buy_ratio_4h"] - 0.5
    lag1 = f[["ret_1h", "ret_4h", "ret_24h"]].shift(4)
    lag4 = f[["ret_4h", "ret_24h"]].shift(16)
    out = pd.DataFrame(index=f.index)
    out["flow_trans_1h"], out["flow_perm_1h"] = rolling_resid(y1, lag1)
    sd = out["flow_perm_1h"].rolling(7 * DAY, min_periods=2 * DAY).std().shift(1)
    out["flow_perm_1h_z"] = out["flow_perm_1h"] / sd
    out["flow_perm_4h"] = rolling_resid(y4, lag4)[1]
    out["flow_impact_1h"] = rolling_resid(y1, f[["ret_1h"]])[1]
    return out


def memory_features(price: pd.Series) -> pd.DataFrame:
    """price — 15m-закриття монети на регулярній сітці."""
    r24 = price / price.shift(DAY) - 1
    have = price.notna().astype(float).rolling(30 * DAY, min_periods=1).sum().shift(DAY)    # барів історії у вікні
    enough = have >= 14 * DAY
    scale = 30 * DAY / have
    out = pd.DataFrame(index=price.index)
    for x, name in ((0.10, "mem_n30_10"), (0.20, "mem_n30_20")):
        hot = (r24 >= x).astype(float)
        start = (hot == 1) & (hot.shift(1).rolling(DAY, min_periods=1).max().fillna(0) == 0)
        cnt = start.astype(float).rolling(29 * DAY, min_periods=1).sum().shift(DAY)
        out[name] = (cnt * scale).where(enough)
    out["mem_max30"] = r24.rolling(29 * DAY, min_periods=DAY).max().shift(DAY).where(enough)
    hot20 = pd.Series(np.where(r24 >= 0.20, np.arange(len(r24)), np.nan), index=price.index).ffill().shift(DAY)
    days = (np.arange(len(r24)) - hot20) / DAY
    out["mem_days20"] = days.clip(upper=30).fillna(30).where(enough)
    return out


def coin_extra(f: pd.DataFrame, price: pd.Series) -> pd.DataFrame:
    """Нові ознаки для рядків f (base_asset, ts, buy_ratio_*, ret_*) за ціною price (index ts)."""
    grid = pd.date_range(min(f["ts"].min(), price.index.min()), max(f["ts"].max(), price.index.max()), freq="15min")
    g = f.set_index("ts")[["buy_ratio_1h", "buy_ratio_4h", "ret_1h", "ret_4h", "ret_24h"]]
    g = g[~g.index.duplicated()].reindex(grid)
    nf = flow_features(g).join(memory_features(price[~price.index.duplicated()].reindex(grid)))
    return nf.reindex(f["ts"]).set_axis(f.index)


def load():
    """Як early_target.load_data, але нові ознаки рахуються на ПОВНИХ рядах до фільтра навчальних рядків."""
    load_dotenv(".env")
    uni = json.load(open(os.path.join(I.DIR, "_universe.json"), encoding="utf-8"))
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
    parts, px = [], {}
    for b, g in d.groupby("base_asset"):
        a = aggregate(raw, b, "15min")
        px[b] = a[["price", "high"]]
        parts.append(I.new_features(a, idx).reindex(g["ts"]).set_axis(g.index).join(coin_extra(g, a["price"])))
    d = d.join(pd.concat(parts))
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=48)
    d = d[d["base_asset"].isin(reviewed) | d["pattern_type"].notna()]
    d = d[(d["ts"] <= cutoff) | d["pattern_type"].notna()]
    d = d[d["ts"] >= d["base_asset"].map(first1)]
    return d.sort_values(["base_asset", "ts"]).reset_index(drop=True), pumps, px, idx, uni


def half(c: str) -> str:
    return "A" if int(hashlib.md5(c.encode()).hexdigest(), 16) % 2 == 0 else "B"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    argparse.ArgumentParser(description="«Справжній» потік покупок і пам'ять монети про пампи: LTO і бектест 6 міс.").parse_args()
    d, pumps, px, idx, uni = load()
    y, grp = d["label"].astype(int), d["base_asset"]
    trained = set(d["base_asset"])
    print(f"навчання: {len(d)} барів, {d['base_asset'].nunique()} тікерів; покриття нових ознак: "
          + ", ".join(f"{c} {d[c].notna().mean():.0%}" for c in FLOW + MEM), flush=True)
    thr, lto, models, gain = {}, {}, {}, {}
    for v, cols in SETS.items():
        p = train.oof_proba(train.make_lightgbm, d[cols], y, grp)
        thr[v] = train.best_f1_threshold(y.to_numpy(), p)
        sig = p >= thr[v]
        lto[v] = {**train.event_metrics(d, sig, pumps, ET.BAR), **ET.early(d, sig, pumps, px)}
        models[v] = train.make_lightgbm().fit(d[cols], y)
        g = pd.Series(models[v].booster_.feature_importance("gain"), index=cols)
        gain[v] = g / g.sum()
        print(f"  {v} (поріг {thr[v]:.3f}): {lto[v]}", flush=True)

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
        f15 = f15.join(I.new_features(h, idx).reindex(f15["ts"]).set_axis(f15.index)).join(coin_extra(f15, h["price"]))
        hh = h[(h.index >= start - pd.Timedelta(days=1)) & (h.index < end)]
        s1 = h1[["base_asset", "ts"]].assign(probability=m1["model"].predict_proba(h1[m1["features"]])[:, 1], threshold=m1["threshold"])
        months = uni[c]
        for v, cols in SETS.items():
            s15 = f15[["base_asset", "ts"]].assign(probability=models[v].predict_proba(f15[cols])[:, 1], threshold=thr[v])
            t = tier_frame(s15, s1, share).set_index("ts").sort_index()
            a_sc = t.loc[(t.index >= start) & (t.index < end), "alert_score"]
            for r in P.simulate(hh, a_sc, I.K, exit_bars, 3.0, 0.5, True):
                mth = pd.Timestamp(r["ts"]).strftime("%Y-%m")
                if mth in months:
                    rows.append({"variant": v, "ticker": c, "month": mth, "band": months[mth], **r})
        if n % 50 == 0:
            print(f"  {n}/{len(coins)}", flush=True)
    tr = pd.DataFrame(rows)
    tr["coins"] = tr["ticker"].map(half)
    tr.to_csv(os.path.join("reports", "flow_memory_trades.csv"), index=False)
    pf, V = I.pf, list(SETS)
    L = ["# «Справжній» потік покупок і пам'ять монети про пампи", "",
         "Пороги — найкращий F1 на OOF: " + ", ".join(f"{k} {v:.3f}" for k, v in thr.items()) + ".", "",
         "## Leave-ticker-out (Кандидат 15m)", "",
         "Спіймано / затримка / пройдено — перший сигнал від «старт − 1 год» до кінця імпульсу.", "",
         "| варіант | спіймано | затримка, хв | пройдено руху | фальстартів | хибних / 30 дн | частка gain нових ознак |",
         "|---|---|---|---|---|---|---|"]
    for v in V:
        m = lto[v]
        new = [c for c in SETS[v] if c not in BASE]
        L.append(f"| {v} | {m['caught']} | {m['delay_min']:.0f} | {m['done']:.0%} | {m['false_starts_triggered']} | "
                 f"{m['false_alarms_per_30d']:.1f} | {gain[v][new].sum():.0%} |")
    top = gain["both"][FLOW + MEM].sort_values(ascending=False)
    L += ["", "Gain нових ознак у «both»: " + ", ".join(f"{k} {v:.1%}" for k, v in top.items()) + ".", "",
          f"## Бектест 6 міс., {tr['ticker'].nunique()} монет чесного списку", "",
          "| варіант | угод | у плюсі | медіана | PF | без топ-3 | ≥ 20 млн | 5–20 млн | монети A | монети B | трав–вер | трав–вер ≥ 20 млн |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for v in V:
        g = tr[tr["variant"] == v]
        r = g["ret"].sort_values()
        mv = g["month"] >= "2026-05"
        L.append(f"| {v} | {len(r)} | {(r > 0).mean():.0%} | {r.median():+.2%} | {pf(r):.2f} | {pf(r.iloc[:-3]):.2f} | "
                 f"{pf(g[g['band'] == '≥ 20 млн']['ret']):.2f} | {pf(g[g['band'] == '5–20 млн']['ret']):.2f} | "
                 f"{pf(g[g['coins'] == 'A']['ret']):.2f} | {pf(g[g['coins'] == 'B']['ret']):.2f} | {pf(g[mv]['ret']):.2f} | "
                 f"{pf(g[mv & (g['band'] == '≥ 20 млн')]['ret']):.2f} |")
    piv = tr.groupby(["month", "variant"])["ret"].apply(pf).unstack()[V]
    piv20 = tr[tr["band"] == "≥ 20 млн"].groupby(["month", "variant"])["ret"].apply(pf).unstack()[V]
    L += ["", "## По місяцях (PF; у дужках ≥ 20 млн)", "", "| місяць | " + " | ".join(V) + " |", "|---|" + "---|" * len(V)]
    for mth in piv.index:
        L.append(f"| {mth} | " + " | ".join(f"{piv.at[mth, v]:.2f} ({piv20.at[mth, v]:.2f})" for v in V) + " |")
    L += ["", "Місяців кращих за base: " + ", ".join(
        f"{v} {int((piv[v] > piv['base'] + 1e-9).sum())} / ≥ 20 млн {int((piv20[v] > piv20['base'] + 1e-9).sum())}"
        for v in V if v != "base") + f" (з {len(piv)})", ""]
    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
