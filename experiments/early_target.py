"""
experiments/early_target.py — ЦІЛЬ «СТАРТ ІМПУЛЬСУ» замість «імпульс триває» (2026-10-09, власник: пізній Alert — головна
проблема). Робоча ціль вчить модель упізнавати розвинутий імпульс, тому впевненою вона стає, коли пройдено ~¼ руху.
Тут позитив — лише ПОЧАТОК імпульсу, а решта його барів у навчання не йде (ні «1», ні «0»: це не «звичайний ринок»).
Фальстарти лишаються негативами (саме їх рання модель має відрізняти). Ознаки — робочі (MODEL_FEATURES_15M + власний рух).
Варіанти цілі (зафіксовано ДО запуску; бар — 15m, «закриття» = відкриття + 15 хв):
    base      як зараз: закриття бару всередині імпульсу
    start30   закриття в (старт, старт + 30 хв] — перші 2 бари
    start60   закриття в (старт, старт + 60 хв] — перші 4 бари
    pre30     закриття в (старт − 30 хв, старт + 30 хв] — ще й 2 бари ДО старту
Перевірка 1 — leave-ticker-out на розмітці. Поріг ранніх моделей підбирається так, щоб хибних епізодів на 30 дн було
стільки ж, скільки в base на її порозі (найкращий F1 на OOF) — порівняння «хто раніше за тієї самої кількості хибних».
Метрики: спіймано (сигнал від старт − 1 год до кінця), затримка першого сигналу (може бути < 0), пройдено руху на ньому,
фальстарти, хибні; і стандартні train.event_metrics.
Перевірка 2 — бектест 6 міс. на чесному списку (частковий вихід, стоп I.K): base = Alert як зараз; or_<v> — вхід, коли
Alert base АБО рання модель ≥ порогу і годинна ≥ 0.75 (утримання — поки горить будь-яка з двох; вихід — 4 бари без обох);
split_<v> (власник 09.10: «одна модель на вхід, друга — на утримання») — вхід ЛИШЕ за ранньою моделлю (+ годинна ≥ 0.75),
утримання — поки горить base або рання, вихід — 4 бари без обох; split_<v>_c — вхід за ранньою без годинного підтвердження.
→ reports/early_target.md, reports/early_target_trades.csv

    python experiments/early_target.py
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
import psycopg
from dotenv import load_dotenv

import ensemble as E
import exit_partial as P
import idio_features as I
import range_features as R
import train
from features import INTERVAL_15M, MODEL_FEATURES_15M, aggregate, build_from_raw, load_pumps, load_raw, query_df, short_features
from fetch import excluded_tickers, read_tickers_file
from tiers import tier_frame

COLS = MODEL_FEATURES_15M + R.MOVE
BAR = train.BAR_15M
TARGETS = {"start30": (pd.Timedelta(0), pd.Timedelta(minutes=30)),
           "start60": (pd.Timedelta(0), pd.Timedelta(minutes=60)),
           "pre30": (pd.Timedelta(minutes=-30), pd.Timedelta(minutes=30))}
VARIANTS = ["base", *TARGETS]
PRE = pd.Timedelta(hours=1)          # сигнал за ≤ 1 год до старту — теж «спіймав рано»
OUT = os.path.join("reports", "early_target.md")


def target(d: pd.DataFrame, pumps: pd.DataFrame, lo: pd.Timedelta, hi: pd.Timedelta) -> tuple[np.ndarray, np.ndarray]:
    """Позитив — бари із закриттям у (старт + lo, старт + hi]; train — False для решти барів імпульсу."""
    close = d["ts"] + BAR
    pos = np.zeros(len(d), dtype=bool)
    for r in pumps[pumps["pattern_type"] == "pump"].itertuples():
        m = (d["base_asset"] == r.base_asset) & (close > r.start_ts + lo) & (close <= min(r.start_ts + hi, r.end_ts))
        pos |= m.to_numpy()
    keep = pos | (d["label"].to_numpy() == 0)
    return pos, keep


def oof(X: pd.DataFrame, y: np.ndarray, keep: np.ndarray, grp: pd.Series) -> np.ndarray:
    """OOF за тікерами; вчиться лише на рядках keep, передбачає всі рядки свого тікера."""
    p = np.zeros(len(y))
    for g in grp.unique():
        test = (grp == g).to_numpy()
        tr = ~test & keep
        p[test] = train.make_lightgbm().fit(X[tr], y[tr]).predict_proba(X[test])[:, 1]
    return p


def early(d: pd.DataFrame, sig: np.ndarray, pumps: pd.DataFrame, px: dict) -> dict:
    """Перший сигнал у вікні [старт − 1 год, кінець): спіймано, затримка (хв, медіана), пройдено руху (медіана)."""
    s = d.loc[sig, ["base_asset", "ts"]]
    delay, done, n = [], [], 0
    for r in pumps[pumps["pattern_type"] == "pump"].itertuples():
        if r.base_asset not in px:
            continue
        n += 1
        close = s.loc[s["base_asset"] == r.base_asset, "ts"] + BAR
        c = close[(close > r.start_ts - PRE) & (close <= r.end_ts)]
        if not len(c):
            continue
        t = c.min()
        delay.append((t - r.start_ts) / pd.Timedelta(minutes=1))
        h = px[r.base_asset]
        w = h[(h.index >= r.start_ts - BAR) & (h.index < r.end_ts)]
        before = h["price"][h.index < r.start_ts]
        p0 = before.iloc[-1] if len(before) else w["price"].iloc[0]
        peak, bar_open = w["high"].max(), t - BAR
        if peak > p0 and bar_open in h.index:
            done.append((h.at[bar_open, "price"] - p0) / (peak - p0))
    return {"caught": f"{len(delay)}/{n}", "delay_min": float(np.median(delay)) if delay else np.nan,
            "pre_start": int(sum(x <= 0 for x in delay)), "done": float(np.median(done)) if done else np.nan}


def matched_threshold(d, p, pumps, target_fa: float) -> float:
    """Поріг, за якого хибних епізодів на 30 дн найближче до target_fa (сітка квантилів)."""
    best, gap = None, np.inf
    for q in np.linspace(0.90, 0.999, 45):
        thr = float(np.quantile(p, q))
        fa = train.event_metrics(d, p >= thr, pumps, BAR)["false_alarms_per_30d"]
        if abs(fa - target_fa) < gap:
            best, gap = thr, abs(fa - target_fa)
    return best


def simulate_split(h, entry, hold, sl, exit_bars, tp, frac, be) -> list[dict]:
    """Як exit_partial.simulate, але вхід — за entry ≥ 1, а «бари без сигналу» для виходу — за hold."""
    a = P.atr(h).reindex(h.index).to_numpy()
    en, ho = entry.reindex(h.index).to_numpy(), hold.reindex(h.index).to_numpy()
    hi, lo, c = h["high"].to_numpy(), h["low"].to_numpy(), h["price"].to_numpy()
    out, tr, below = [], None, 0
    for i in range(len(h)):
        if np.isnan(ho[i]):
            continue
        if tr is None:
            if en[i] >= 1 and not np.isnan(c[i]) and not np.isnan(a[i]):
                tr = {"e": c[i], "stop": c[i] - sl * a[i], "tp": c[i] + tp * a[i] if tp else None, "part": 0.0, "i0": i}
                below = 0
            continue
        e = tr["e"]
        if lo[i] <= tr["stop"]:
            rest = tr["stop"] / e - 1
            out.append({"ret": tr["part"] + (1 - (frac if tr["part"] else 0)) * rest - P.FEE, "reason": "stop",
                        "took": bool(tr["part"]), "bars": i - tr["i0"], "ts": h.index[tr["i0"]]})
            tr = None
            continue
        if tr["tp"] is not None and not tr["part"] and hi[i] >= tr["tp"]:
            tr["part"] = frac * (tr["tp"] / e - 1)
            if be:
                tr["stop"] = max(tr["stop"], e)
        below = below + 1 if ho[i] < 1 else 0
        if below >= exit_bars:
            rest = c[i] / e - 1
            out.append({"ret": tr["part"] + (1 - (frac if tr["part"] else 0)) * rest - P.FEE, "reason": "model",
                        "took": bool(tr["part"]), "bars": i - tr["i0"], "ts": h.index[tr["i0"]]})
            tr = None
    return out


def load_data():
    """Навчальні 15m-бари з ознаками (як ensemble.py) → d, pumps, px (ціна / high), idx (ALT-EW), uni (чесний список)."""
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
    parts, px = [], {}
    for b, g in d.groupby("base_asset"):
        a = aggregate(raw, b, "15min")
        px[b] = a[["price", "high", "low", "volume", "oi_coin_log_idx"]]
        parts.append(I.new_features(a, idx).reindex(g["ts"]).set_axis(g.index))
    d = d.join(pd.concat(parts)).sort_values(["base_asset", "ts"]).reset_index(drop=True)
    return d, pumps, px, idx, uni


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    argparse.ArgumentParser(description="Ціль «старт імпульсу» замість «імпульс триває»: LTO і бектест 6 міс.").parse_args()
    d, pumps, px, idx, uni = load_data()
    X, grp = d[COLS], d["base_asset"]
    trained = set(d["base_asset"])
    print(f"навчання: {len(d)} барів, {d['base_asset'].nunique()} тікерів", flush=True)

    ys, keeps = {"base": d["label"].to_numpy().astype(int)}, {"base": np.ones(len(d), dtype=bool)}
    for v, (lo, hi) in TARGETS.items():
        pos, keep = target(d, pumps, lo, hi)
        ys[v], keeps[v] = pos.astype(int), keep
        print(f"  {v}: позитивів {pos.sum()}, у навчанні {keep.sum()} з {len(d)}", flush=True)
    p, thr, lto = {}, {}, {}
    for v in VARIANTS:
        p[v] = oof(X, ys[v], keeps[v], grp)
        if v == "base":
            thr[v] = train.best_f1_threshold(ys[v], p[v])
            fa0 = train.event_metrics(d, p[v] >= thr[v], pumps, BAR)["false_alarms_per_30d"]
        else:
            thr[v] = matched_threshold(d, p[v], pumps, fa0)
        sig = p[v] >= thr[v]
        lto[v] = {**train.event_metrics(d, sig, pumps, BAR), **early(d, sig, pumps, px), "move_done": E.move_done(d, sig, pumps, px)}
        print(f"  {v} (поріг {thr[v]:.3f}): {lto[v]}", flush=True)
    models = {v: train.make_lightgbm().fit(X[keeps[v]], ys[v][keeps[v]]) for v in VARIANTS}

    m1 = joblib.load(train.model_path("lightgbm"))
    m15 = joblib.load(train.model_path(train.M15_NAME))
    exit_bars, share = m15.get("exit_bars", 4), m15.get("confirm", {}).get("share", 0.75)
    start, end = pd.Timestamp(I.START, tz="UTC"), pd.Timestamp(I.END, tz="UTC") + pd.Timedelta(days=1)
    bt = ["base"] + [f"{k}_{v}{s}" for v in TARGETS for k, s in (("or", ""), ("split", ""), ("split", "_c"))]
    rows = []
    coins = sorted(c for c in uni if c not in trained and os.path.exists(os.path.join(I.DIR, f"{c}.pkl")))
    print(f"бектест: {len(coins)} монет чесного списку…", flush=True)
    for n, c in enumerate(coins, 1):
        x = pickle.load(open(os.path.join(I.DIR, f"{c}.pkl"), "rb"))
        f15, h1, h = x["f15"], x["h1"], x["h"]
        f15 = f15.join(I.new_features(h, idx).reindex(f15["ts"]).set_axis(f15.index))
        hh = h[(h.index >= start - pd.Timedelta(days=1)) & (h.index < end)]
        s1 = h1[["base_asset", "ts"]].assign(probability=m1["model"].predict_proba(h1[m1["features"]])[:, 1], threshold=m1["threshold"])

        def frame(v):
            s15 = f15[["base_asset", "ts"]].assign(probability=models[v].predict_proba(f15[COLS])[:, 1], threshold=thr[v])
            t = tier_frame(s15, s1, share).set_index("ts").sort_index()
            return t.loc[(t.index >= start) & (t.index < end)]
        fr = {v: frame(v) for v in VARIANTS}
        a0 = fr["base"]["alert_score"]
        sc = {"base": (a0, a0)}                                  # (оцінка входу, оцінка утримання)
        for v in TARGETS:
            ea, ec = fr[v]["alert_score"].reindex(a0.index), fr[v]["s15"].reindex(a0.index)
            sc[f"or_{v}"] = (np.maximum(a0, ea),) * 2
            sc[f"split_{v}"] = (ea, np.maximum(a0, ea))
            sc[f"split_{v}_c"] = (ec, np.maximum(a0, ec))
        months = uni[c]
        for v in bt:
            for r in simulate_split(hh, *sc[v], I.K, exit_bars, 3.0, 0.5, True):
                mth = pd.Timestamp(r["ts"]).strftime("%Y-%m")
                if mth in months:
                    rows.append({"variant": v, "ticker": c, "month": mth, "band": months[mth], **r})
        if n % 50 == 0:
            print(f"  {n}/{len(coins)}", flush=True)
    tr = pd.DataFrame(rows)
    tr.to_csv(os.path.join("reports", "early_target_trades.csv"), index=False)
    pf = I.pf
    L = ["# Ціль «старт імпульсу» замість «імпульс триває»", "",
         "Ознаки — робочі (без рейнджу). Поріг base — найкращий F1 на OOF; ранніх — та сама кількість хибних на 30 дн: " +
         ", ".join(f"{k} {v:.3f}" for k, v in thr.items()) + ".", "",
         "## Leave-ticker-out", "",
         "Спіймано / затримка / пройдено — перший сигнал від «старт − 1 год» до кінця імпульсу (затримка < 0 — сигнал до старту).", "",
         "| ціль | спіймано | з них до старту | затримка, хв | пройдено руху | спіймано (як train) | фальстартів | хибних / 30 дн |",
         "|---|---|---|---|---|---|---|---|"]
    for v in VARIANTS:
        m = lto[v]
        L.append(f"| {v} | {m['caught']} | {m['pre_start']} | {m['delay_min']:.0f} | {m['done']:.0%} | {m['pumps_caught']} | "
                 f"{m['false_starts_triggered']} | {m['false_alarms_per_30d']:.1f} |")
    L += ["", f"## Бектест 6 міс., {tr['ticker'].nunique()} монет чесного списку", "",
          "base — Alert як зараз; or_X — вхід і утримання за Alert base АБО ранньої X (з годинним підтвердженням); "
          "split_X — вхід ЛИШЕ за ранньою X (+ годинна ≥ 0.75), утримання — поки горить base або X, вихід — 4 бари без обох; "
          "split_X_c — те саме, вхід за ранньою без годинного підтвердження.", "",
          "| варіант | угод | у плюсі | медіана | PF | PF без топ-3 | ≥ 20 млн | 5–20 млн |", "|---|---|---|---|---|---|---|---|"]
    for v in bt:
        g = tr[tr["variant"] == v]
        r = g["ret"].sort_values()
        L.append(f"| {v} | {len(r)} | {(r > 0).mean():.0%} | {r.median():+.2%} | {pf(r):.2f} | {pf(r.iloc[:-3]):.2f} | "
                 f"{pf(g[g['band'] == '≥ 20 млн']['ret']):.2f} | {pf(g[g['band'] == '5–20 млн']['ret']):.2f} |")
    piv = tr.groupby(["month", "variant"])["ret"].apply(pf).unstack()[bt]
    piv20 = tr[tr["band"] == "≥ 20 млн"].groupby(["month", "variant"])["ret"].apply(pf).unstack()[bt]
    L += ["", "## По місяцях (PF; у дужках ≥ 20 млн)", "", "| місяць | " + " | ".join(bt) + " |", "|---|" + "---|" * len(bt)]
    for mth in piv.index:
        L.append(f"| {mth} | " + " | ".join(f"{piv.at[mth, v]:.2f} ({piv20.at[mth, v]:.2f})" for v in bt) + " |")
    L += ["", "Місяців кращих за base: " + ", ".join(
        f"{v} {int((piv[v] > piv['base'] + 1e-9).sum())} / ≥ 20 млн {int((piv20[v] > piv20['base'] + 1e-9).sum())}"
        for v in bt if v != "base") + f" (з {len(piv)})", ""]
    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
