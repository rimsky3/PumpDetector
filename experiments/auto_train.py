"""
experiments/auto_train.py — навчання на АВТОРОЗМІТЦІ монет архіву (2026-10-10, кроки 4–5 плану авторозмітки; experiments/auto_label.py).
Авторозмітка на кеші 6 міс. (data/long_bin, Binance 15m + OI з архіву): тригер T1 (за 1 год ціна ≥ 2 ATR, OI ≥ +2%, обсяг ≥ 2×
медіани) → старт (мінімум low за 3 год) → кінець за правилом розмітки → клас заднім числом. Мітка бару: 1 — у авто-імпульсі,
сіра зона — поза навчанням, решта — 0. Монети ділимо навпіл за md5 назви: A — навчання, B — бектест.
Варіанти (зафіксовано ДО запуску):
    base     робоча розмітка власника (як зараз)
    auto8    лише авторозмітка монет A, імпульс = рух ≥ 8 ATR і ≥ 1.5 год (auto_label.classify)
    auto15   те саме + рух ≥ 15% (коротші / дрібніші рухи → сіра зона)
    mix15    розмітка власника + auto15
Пороги — найкращий F1: base — OOF за тікерами; auto / mix — OOF 5 груп монет (GroupKFold).
Перевірка 1 — на РУЧНІЙ розмітці власника: auto-моделі її не бачили (чистий перенос), base / mix — OOF за тікерами власника.
Перевірка 2 — бектест 6 міс. на монетах B (Alert = варіант + робоча годинна, частковий вихід, стоп I.K), квіт–черв і лип–вер.
→ reports/auto_train.md

    python experiments/auto_train.py
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
from sklearn.model_selection import GroupKFold

import auto_label as AL
import early_target as ET
import exit_partial as P
import idio_features as I
import range_features as R
import train
from risk import atr
from tiers import tier_frame

COLS = ET.COLS
OUT = os.path.join("reports", "auto_train.md")


def half(c: str) -> str:
    return "A" if int(hashlib.md5(c.encode()).hexdigest(), 16) % 2 == 0 else "B"


def coin_series(c: str, h: pd.DataFrame) -> pd.DataFrame:
    """Ряд для авторозмітки з кешу: ціна / high / low / обсяг, OI (USD-масштаб з архіву монетного OI), ATR, медіана обсягу."""
    s = h[["high", "low", "price", "volume"]].copy()
    s["oi"] = np.exp(R.archive_oi_log(c, s.index)) * s["price"]
    s["atr"] = atr(s)
    s["vol_med"] = s["volume"].rolling(7 * AL.DAY, min_periods=AL.DAY).median().shift(1)
    return s.dropna(subset=["price", "high", "low"])


def labels_for(s: pd.DataFrame, ts: pd.Series, min_pct: float) -> tuple[np.ndarray, np.ndarray]:
    """Мітки (1 / 0) і маска «в навчанні» для рядків ts за авто-подіями ряду s."""
    e = AL.auto_events(s, AL.TRIGGERS["T1"])
    if e.empty:
        return np.zeros(len(ts), int), np.ones(len(ts), bool)
    e = e.copy()
    small = (e["cls"] == "pump") & (e["move_pct"] < min_pct)
    e.loc[small, "cls"] = "grey"
    close = ts.values + np.timedelta64(15, "m")
    y, grey = np.zeros(len(ts), bool), np.zeros(len(ts), bool)
    for r in e.itertuples():
        m = (close > np.datetime64(r.start.tz_convert(None))) & (close <= np.datetime64(r.end_ts.tz_convert(None)))
        if r.cls == "pump":
            y |= m
        elif r.cls == "grey":
            grey |= m
    return y.astype(int), ~(grey & ~y)


def gkf_threshold(X, y, g) -> float:
    p = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, g):
        p[te] = train.make_lightgbm().fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
    return train.best_f1_threshold(y, p)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Навчання на авторозмітці монет архіву: перенос на розмітку власника і бектест на інших монетах")
    ap.add_argument("--train-until", help="перевірка за часом: авторозмітка лише до цієї дати (мінус 3 дні), бектест — з цього місяця, "
                                          "напр. 2026-07-01 → reports/auto_train_until.md")
    args = ap.parse_args()
    until = pd.Timestamp(args.train_until, tz="UTC") if args.train_until else None
    out = OUT.replace(".md", "_until.md") if until is not None else OUT
    d, pumps, px, idx, uni = ET.load_data()
    trained = set(d["base_asset"])
    coins = sorted(c for c in uni if c not in trained and os.path.exists(os.path.join(I.DIR, f"{c}.pkl")))
    A = [c for c in coins if half(c) == "A"]
    B = [c for c in coins if half(c) == "B"]
    print(f"монет A {len(A)}, B {len(B)}; авторозмітка A…", flush=True)
    parts = []
    for n, c in enumerate(A, 1):
        x = pickle.load(open(os.path.join(I.DIR, f"{c}.pkl"), "rb"))
        f15, h = x["f15"], x["h"]
        f15 = f15.join(I.new_features(h, idx).reindex(f15["ts"]).set_axis(f15.index))
        f15 = f15[f15["ts"] >= pd.Timestamp(I.START, tz="UTC")]
        if until is not None:                    # мітка бачить до 72 год уперед — запас 3 дні до межі
            f15 = f15[f15["ts"] < until - pd.Timedelta(days=3)]
        s = coin_series(c, h)
        y8, k8 = labels_for(s, f15["ts"], 0.0)
        y15, k15 = labels_for(s, f15["ts"], 0.15)
        parts.append(f15[["base_asset", "ts", *COLS]].assign(y8=y8, k8=k8, y15=y15, k15=k15))
        if n % 50 == 0:
            print(f"  {n}/{len(A)}", flush=True)
    a = pd.concat(parts, ignore_index=True)
    print(f"авто: {len(a)} барів; імпульсних auto8 {a['y8'].sum()} ({a['y8'].mean():.2%}), auto15 {a['y15'].sum()} ({a['y15'].mean():.2%}); "
          f"власник {d['label'].sum()} ({d['label'].mean():.2%})", flush=True)

    Xo, yo, go = d[COLS], d["label"].to_numpy().astype(int), d["base_asset"]
    thr, models, lto = {}, {}, {}
    # base — як зараз
    p = train.oof_proba(train.make_lightgbm, Xo, pd.Series(yo), go)
    thr["base"] = train.best_f1_threshold(yo, p)
    models["base"] = train.make_lightgbm().fit(Xo, yo)
    lto["base"] = p
    # auto8 / auto15 — лише авторозмітка A; на розмітці власника — пряме передбачення
    for v, yc, kc in (("auto8", "y8", "k8"), ("auto15", "y15", "k15")):
        s = a[a[kc]]
        thr[v] = gkf_threshold(s[COLS], s[yc].to_numpy(), s["base_asset"])
        models[v] = train.make_lightgbm().fit(s[COLS], s[yc].to_numpy())
        lto[v] = models[v].predict_proba(Xo)[:, 1]
        print(f"  {v}: поріг {thr[v]:.3f}", flush=True)
    # mix15 — власник + auto15; на розмітці власника — OOF за тікерами власника (авто-частина завжди в навчанні)
    s = a[a["k15"]]
    Xa, ya = s[COLS], s["y15"].to_numpy()
    pm = np.zeros(len(yo))
    for g in go.unique():
        te = (go == g).to_numpy()
        m = train.make_lightgbm().fit(pd.concat([Xo[~te], Xa]), np.concatenate([yo[~te], ya]))
        pm[te] = m.predict_proba(Xo[te])[:, 1]
    thr["mix15"] = train.best_f1_threshold(yo, pm)          # поріг — найкращий F1 на OOF розмітки власника
    models["mix15"] = train.make_lightgbm().fit(pd.concat([Xo, Xa]), np.concatenate([yo, ya]))
    lto["mix15"] = pm
    V = list(models)
    res = {}
    for v in V:
        sig = lto[v] >= thr[v]
        res[v] = {**train.event_metrics(d, sig, pumps, ET.BAR), **ET.early(d, sig, pumps, px)}
        print(f"  {v} на розмітці власника: {res[v]}", flush=True)

    m1 = joblib.load(train.model_path("lightgbm"))
    m15 = joblib.load(train.model_path(train.M15_NAME))
    exit_bars, share = m15.get("exit_bars", 4), m15.get("confirm", {}).get("share", 0.75)
    start, end = pd.Timestamp(I.START, tz="UTC"), pd.Timestamp(I.END, tz="UTC") + pd.Timedelta(days=1)
    print(f"бектест: {len(B)} монет B…", flush=True)
    rows = []
    for n, c in enumerate(B, 1):
        x = pickle.load(open(os.path.join(I.DIR, f"{c}.pkl"), "rb"))
        f15, h1, h = x["f15"], x["h1"], x["h"]
        f15 = f15.join(I.new_features(h, idx).reindex(f15["ts"]).set_axis(f15.index))
        hh = h[(h.index >= start - pd.Timedelta(days=1)) & (h.index < end)]
        s1 = h1[["base_asset", "ts"]].assign(probability=m1["model"].predict_proba(h1[m1["features"]])[:, 1], threshold=m1["threshold"])
        for v in V:
            s15 = f15[["base_asset", "ts"]].assign(probability=models[v].predict_proba(f15[COLS])[:, 1], threshold=thr[v])
            t = tier_frame(s15, s1, share).set_index("ts").sort_index()
            for r in P.simulate(hh, t.loc[(t.index >= start) & (t.index < end), "alert_score"], I.K, exit_bars, 3.0, 0.5, True):
                mth = pd.Timestamp(r["ts"]).strftime("%Y-%m")
                if mth in uni[c] and (until is None or mth >= until.strftime("%Y-%m")):
                    rows.append({"variant": v, "ticker": c, "month": mth, "band": uni[c][mth], **r})
        if n % 50 == 0:
            print(f"  {n}/{len(B)}", flush=True)
    tr = pd.DataFrame(rows)
    tr.to_csv(out.replace(".md", "_trades.csv"), index=False)
    pf = I.pf
    L = ["# Навчання на авторозмітці монет архіву", "",
         f"Авторозмітка {len(A)} монет (половина A), {len(a)} барів: імпульсних auto8 {a['y8'].mean():.2%}, auto15 {a['y15'].mean():.2%} "
         f"(у розмітці власника {d['label'].mean():.2%}). Пороги: " + ", ".join(f"{k} {v:.3f}" for k, v in thr.items()) + ".", "",
         "## Перевірка на розмітці власника (Кандидат 15m)", "",
         "auto8 / auto15 розмітки власника не бачили; base / mix15 — OOF за тікерами власника.", "",
         "| варіант | спіймано | затримка, хв | пройдено руху | фальстартів | хибних / 30 дн |", "|---|---|---|---|---|---|"]
    for v in V:
        m = res[v]
        L.append(f"| {v} | {m['caught']} | {m['delay_min']:.0f} | {m['done']:.0%} | {m['false_starts_triggered']} | {m['false_alarms_per_30d']:.1f} |")
    L += ["", f"## Бектест 6 міс., {tr['ticker'].nunique()} монет B (не в навчанні жодного варіанта)", "",
          "| варіант | угод | PF | без топ-3 | медіана | ≥ 20 млн | 5–20 млн | квіт–черв | лип–вер | лип–вер ≥ 20 млн |", "|---|---|---|---|---|---|---|---|---|---|"]
    for v in V:
        g = tr[tr["variant"] == v]
        r = g["ret"].sort_values()
        e = g["month"] <= "2026-06"
        big = g["band"] == "≥ 20 млн"
        L.append(f"| {v} | {len(g)} | {pf(r):.2f} | {pf(r.iloc[:-3]):.2f} | {r.median():+.2%} | {pf(g[big]['ret']):.2f} | "
                 f"{pf(g[g['band'] == '5–20 млн']['ret']):.2f} | {pf(g[e]['ret']):.2f} | {pf(g[~e]['ret']):.2f} | {pf(g[~e & big]['ret']):.2f} |")
    piv = tr.groupby(["month", "variant"])["ret"].apply(pf).unstack()[V]
    piv20 = tr[tr["band"] == "≥ 20 млн"].groupby(["month", "variant"])["ret"].apply(pf).unstack()[V]
    L += ["", "## По місяцях (PF; у дужках ≥ 20 млн)", "", "| місяць | " + " | ".join(V) + " |", "|---|" + "---|" * len(V)]
    for mth in piv.index:
        L.append(f"| {mth} | " + " | ".join(f"{piv.at[mth, v]:.2f} ({piv20.at[mth, v]:.2f})" for v in V) + " |")
    L += ["", "Місяців кращих за base: " + ", ".join(
        f"{v} {int((piv[v] > piv['base'] + 1e-9).sum())} / ≥ 20 млн {int((piv20[v] > piv20['base'] + 1e-9).sum())}"
        for v in V if v != "base") + f" (з {len(piv)})", "",
        "Застереження: монети A і B торгуються в ті самі місяці — загальний рух ринку спільний; чистіше — колонка «лип–вер» лише як "
        "орієнтир (навчання auto теж на квіт–вер)."]
    if until is not None:
        L.insert(1, f"\n**Перевірка за часом:** авторозмітка лише до {until - pd.Timedelta(days=3):%Y-%m-%d}, бектест — з {until:%Y-%m}. "
                    "base і mix15 вчаться на розмітці власника (серпень–жовтень) — для них перекриття з бектестом лишається.\n")
    open(out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
