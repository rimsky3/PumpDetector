"""
experiments/wakeup_5m.py — сигнал «прокидається» на 5m-барах: чи передує він сигналам моделі і що з ціною далі (2026-10-03).

Питання власника (AIN 03.10: рух з 14:00, Кандидат лише о 14:36): чи може просте правило на 5m-барах попереджати раніше
за модель — «монета прокидається, подивись»? Правило — як у сканері, але на коротких вікнах:
    обсяг за останні 15 хв ≥ K × норми (медіана 15-хв обсягу за попередні 7 днів, без останніх 15 хв)
    і ціна за 15 хв ≥ +P %;   варіант «+ OI»: ще й OI (Binance + Bybit, ланцюговий індекс) за останню годину ≥ +2 %
    (OI — з 15m-барів бази, останній закритий ≤ моменту спрацювання).
Після спрацювання — 4 год тиші для цієї монети (одне попередження на рух). Пороги K × P — сітка, зафіксована до прогону.

Дані: монети бектесту 03.10 з середнім оборотом ≥ 20 млн $ (поза навчанням), вересень; 5m — з 1m-свічок Binance
(experiments/klines1m.py); Кандидат / Alert — робоча модель 03.10 на 15m-даних бази (backtest.tiers_and_bars), момент
сигналу — закриття 15m-бару (без затримки скринера). Контроль — випадкові 5m-бари тих самих монет.

    python experiments/wakeup_5m.py     # → reports/wakeup_5m.md, reports/wakeup_5m.csv
"""

import argparse
import itertools
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

import klines1m as K

RUN = "20261003_1216_candidate_market_k2.48"           # універсум ≥ 20 млн (середній оборот за 30 днів)
START, END = pd.Timestamp("2026-09-04", tz="UTC"), pd.Timestamp("2026-10-01", tz="UTC")
KS, PS = (3, 5, 10), (0.01, 0.02, 0.03)
OI_MIN = 0.02
COOLDOWN = pd.Timedelta(hours=4)
B5, B15 = pd.Timedelta(minutes=5), pd.Timedelta(minutes=15)
RANDOM_X, SEED = 3, 42
CACHE = os.path.join("reports", "backtests", "_wakeup_tiers.pkl")
OUT_MD, OUT_CSV = os.path.join("reports", "wakeup_5m.md"), os.path.join("reports", "wakeup_5m.csv")


def model_tiers(bases: list[str]) -> dict:
    """{монета: (рівні 15m робочої моделі, 15m-бари бази)} — кеш, бо оцінка всіх монет ~хвилини."""
    if os.path.exists(CACHE):
        return pickle.load(open(CACHE, "rb"))
    import backtest
    import train
    m15, m1 = joblib.load(train.model_path(train.M15_NAME)), joblib.load(train.model_path("lightgbm"))
    out = {}
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        for n, b in enumerate(bases, 1):
            th = backtest.tiers_and_bars(conn, b, m15, m1, 31)
            if th is not None:
                t, h = th
                out[b] = (t[["s15", "alert_score", "tier"]], h[["price", "oi_log_idx"]])
            if n % 20 == 0:
                print(f"  оцінки моделі: {n}/{len(bases)}", flush=True)
    pickle.dump(out, open(CACHE, "wb"))
    return out


def starts(score: pd.Series) -> pd.DatetimeIndex:
    """Моменти старту епізодів сигналу (score ≥ 1 після бару без сигналу) — час ЗАКРИТТЯ 15m-бару."""
    on = (score.fillna(0) >= 1)
    st = on & ~on.shift(1, fill_value=False)
    return score.index[st.to_numpy()] + B15


def triggers(b5: pd.DataFrame, k: float, p: float, oi_chg: pd.Series | None, local: float = 0) -> pd.DatetimeIndex:
    v15 = b5["v"].rolling(3).sum()
    norm = v15.shift(3).rolling(7 * 288, min_periods=3 * 288).median()
    ret15 = b5["c"] / b5["c"].shift(3) - 1
    ok = (v15 >= k * norm) & (ret15 >= p)
    if local:            # ще й проти останніх 4 год: монета, що вже в русі, не «прокидається» вдруге (AIN 03.10 18:05)
        ok &= v15 >= local * v15.shift(3).rolling(48, min_periods=24).median()
    if oi_chg is not None:
        ok &= oi_chg.reindex(b5.index).to_numpy() >= OI_MIN
    idx = b5.index[ok.fillna(False).to_numpy()] + B5                  # момент — закриття 5m-бару
    out, last = [], None
    for t in idx:
        if START <= t < END and (last is None or t - last >= COOLDOWN):
            out.append(t)
            last = t
    return pd.DatetimeIndex(out)


def forward(m1c: pd.Series, t: pd.Timestamp) -> dict:
    """Ціна після моменту t (закриття 1m): через 1 / 4 год, максимум і мінімум за 4 год; рух до t — від мінімуму за 4 год."""
    i = m1c.index.searchsorted(t - pd.Timedelta(minutes=1))
    if i >= len(m1c):
        return {}
    p0 = m1c.iloc[i]
    w = m1c.iloc[i + 1:i + 241]
    pre = m1c.iloc[max(0, i - 240):i + 1]
    return {"r1h": m1c.iloc[min(i + 60, len(m1c) - 1)] / p0 - 1, "r4h": w.iloc[-1] / p0 - 1 if len(w) else np.nan,
            "mfe4h": w.max() / p0 - 1 if len(w) else np.nan, "mae4h": w.min() / p0 - 1 if len(w) else np.nan,
            "run_in": p0 / pre.min() - 1 if len(pre) else np.nan}


def first_after(t: pd.Timestamp, ev: pd.DatetimeIndex, within: pd.Timedelta) -> float:
    j = ev.searchsorted(t)
    return (ev[j] - t).total_seconds() / 60 if j < len(ev) and ev[j] - t <= within else np.nan


def main() -> None:
    argparse.ArgumentParser(description="Сигнал «прокидається» на 5m: чи передує він Кандидату / Alert моделі (вересень, ≥ 20 млн).").parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(".env")
    meta = json.load(open(os.path.join("reports", "backtests", RUN + ".json"), encoding="utf-8"))["meta"]
    bases = [u["base"] for u in meta["universe"] if u["base"] in set(K.available())]
    tiers = model_tiers(bases)
    rng = np.random.default_rng(SEED)
    rows, sig_rows, coins = [], [], 0
    for b in bases:
        if b not in tiers:
            continue
        t, h = tiers[b]
        m1 = K.load(b, START - pd.Timedelta(days=8), END + pd.Timedelta(hours=5))
        if m1.empty:
            continue
        coins += 1
        b5 = K.resample(m1, "5min")
        cand, alert = starts(t["s15"]), starts(t["alert_score"])
        cand, alert = cand[(cand >= START) & (cand < END)], alert[(alert >= START) & (alert < END)]
        # OI за годину до моменту: останній закритий 15m-бар (закриття = відкриття + 15 хв) ≤ закриття 5m-бару
        oi = h["oi_log_idx"].dropna()
        oi_1h = (oi - oi.shift(4)).apply(np.expm1)
        oi_1h.index = oi_1h.index + B15
        oi_at = oi_1h.reindex(b5.index + B5, method="ffill")
        oi_at.index = b5.index
        variants = {f"K{k} P{p:.0%}": (k, p, None) for k, p in itertools.product(KS, PS)}
        variants.update({f"K{k} P{p:.0%} + OI": (k, p, oi_at) for k, p in ((3, 0.01), (5, 0.02))})
        variants.update({f"K5 P2% + OI + 4г ×{lx}": (5, 0.02, oi_at, lx) for lx in (2, 3)})
        m1c = m1["c"]
        for name, spec in variants.items():
            k, p, o = spec[:3]
            for ts in triggers(b5, k, p, o, spec[3] if len(spec) > 3 else 0):
                rows.append({"variant": name, "base": b, "ts": ts, "cand_min": first_after(ts, cand, pd.Timedelta(hours=2)),
                             "alert_min": first_after(ts, alert, pd.Timedelta(hours=2)), **forward(m1c, ts)})
        # випадкові моменти (закриття 5m-бару) — база порівняння
        cands = b5.index[(b5.index + B5 >= START) & (b5.index + B5 < END)] + B5
        for j in rng.choice(len(cands), size=min(len(cands), 30 * RANDOM_X), replace=False):
            ts = cands[j]
            rows.append({"variant": "випадковий момент", "base": b, "ts": ts, "cand_min": first_after(ts, cand, pd.Timedelta(hours=2)),
                         "alert_min": first_after(ts, alert, pd.Timedelta(hours=2)), **forward(m1c, ts)})
        for kind, ev in (("Кандидат", cand), ("Alert", alert)):
            for ts in ev:
                sig_rows.append({"kind": kind, "base": b, "ts": ts, **forward(m1c, ts)})
    R, S = pd.DataFrame(rows), pd.DataFrame(sig_rows)
    days = (END - START).days
    R.to_csv(OUT_CSV, index=False)

    # чи передував сигналу моделі «прокидається» (той самий варіант) за ≤ 2 год і наскільки раніше
    def lead(sig: pd.DataFrame, trig: pd.DataFrame) -> tuple[float, float, float]:
        got, leads, gain = 0, [], []
        for r in sig.itertuples():
            tt = trig.loc[trig["base"] == r.base, "ts"]
            prev = tt[(tt <= r.ts) & (tt >= r.ts - pd.Timedelta(hours=2))]
            if len(prev):
                got += 1
                leads.append((r.ts - prev.max()).total_seconds() / 60)
        return got / max(len(sig), 1), (np.median(leads) if leads else np.nan), len(sig)

    md = ["# Сигнал «прокидається» на 5m-барах — вересень, монети ≥ 20 млн поза навчанням", "",
          f"{coins} монет, {days} днів (04.09–30.09). Правило: обсяг за 15 хв ≥ K × норми (медіана за 7 днів) і ціна за 15 хв ≥ +P%; "
          f"«+ OI» — ще й OI за годину ≥ +{OI_MIN:.0%}. Після спрацювання 4 год тиші для монети. Кандидат / Alert — робоча модель 03.10, "
          "момент — закриття 15m-бару (без затримки скринера). MFE / MAE — найвища / найнижча ціна за 4 год від спрацювання.", ""]
    cols = ["варіант", "спрацювань", "на день (усі монети)", "далі Кандидат ≤ 2 год", "далі Alert ≤ 2 год", "раніше за Кандидат, хв (медіана)",
            "MFE 4 год ≥ +5%", "≥ +10%", "MAE 4 год ≤ −5%", "ціна через 4 год (медіана)", "ріст до спрацювання (медіана)",
            "Alert, перед яким було ≤ 2 год", "наскільки раніше за Alert, хв"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    al = S[S["kind"] == "Alert"]
    order = [v for v in R["variant"].unique() if v != "випадковий момент"] + ["випадковий момент"]
    for v in order:
        g = R[R["variant"] == v]
        share_a, lead_a, _ = lead(al, g) if v != "випадковий момент" else (np.nan, np.nan, 0)
        lines.append("| " + " | ".join(str(x) for x in [
            v, len(g), f"{len(g) / days:.1f}" if v != "випадковий момент" else "—",
            f"{g['cand_min'].notna().mean():.0%}", f"{g['alert_min'].notna().mean():.0%}",
            f"{g['cand_min'].median():.0f}" if g['cand_min'].notna().any() else "—",
            f"{(g['mfe4h'] >= 0.05).mean():.0%}", f"{(g['mfe4h'] >= 0.10).mean():.0%}", f"{(g['mae4h'] <= -0.05).mean():.0%}",
            f"{g['r4h'].median():+.2%}", f"{g['run_in'].median():+.1%}",
            f"{share_a:.0%}" if share_a == share_a else "—", f"{lead_a:.0f}" if lead_a == lead_a else "—"]) + " |")
    md += ["## Спрацювання і що було далі", "", *lines, ""]
    md += ["## Для порівняння: самі сигнали моделі", "", "| сигнал | епізодів | на день | MFE 4 год ≥ +5% | ≥ +10% | MAE ≤ −5% | ціна через 4 год | ріст до сигналу |",
           "|---|---|---|---|---|---|---|---|"]
    for kind, g in S.groupby("kind"):
        md.append(f"| {kind} | {len(g)} | {len(g) / days:.1f} | {(g['mfe4h'] >= 0.05).mean():.0%} | {(g['mfe4h'] >= 0.10).mean():.0%} | "
                  f"{(g['mae4h'] <= -0.05).mean():.0%} | {g['r4h'].median():+.2%} | {g['run_in'].median():+.1%} |")
    open(OUT_MD, "w", encoding="utf-8").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
