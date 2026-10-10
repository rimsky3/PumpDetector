"""
experiments/auto_walkforward.py — авторозмітка ВСІХ монет архіву + перевірка за часом (walk-forward) (2026-10-10, крок 1 плану
«розмітка без рук, надійніша модель»; власник: «роби»). Після auto_robust.py: модель на авторозмітці половини монет залежала від
того, ЯКА половина (A→B краще за base, B→A гірше) — вчила особливості набору монет. Тут навчання на всіх монетах (крім тікерів
власника), а перевірка — на МАЙБУТНЬОМУ місяці:
    тест-місяць m (травень…вересень): навчання — авто-мітки барів усіх монет з квітня до (початок m − 3 дні; мітка бачить
    ≤ 72 год уперед), поріг — найкращий F1 на OOF 3 груп монет, бектест — місяць m на всіх монетах (свій поріг і модель на місяць).
Варіанти (зафіксовано ДО запуску):
    base     робоча розмітка власника (одна модель; навчена на серп–жовт — для серп–вер перекриття за часом на її користь)
    auto8    авто: імпульс = рух ≥ 8 ATR і ≥ 1.5 год (auto_label.classify; auto_train.labels_for)
    auto15   те саме + рух ≥ 15%
Бектест — як в auto_train (Alert = 15m-варіант + робоча годинна ≥ 0.75, частковий вихід, стоп I.K, чесний список по місяцях).
Половини A / B (md5 назви) — лише для звіту: модель вчилась на обох, тож це перевірка «чи плюс не тримається на одній половині».
Перевірка на розмітці власника — модель вересневого кроку (навчання до 28.08; монети власника не в навчанні).
КРИТЕРІЙ УСПІХУ (зафіксовано ДО запуску): PF не нижчий за base щонайменше в 4 з 5 тест-місяців (≥ 20 млн і всі) на ОБОХ
половинах, і на розмітці власника хибних / 30 дн ≤ 1.5 × base.
→ reports/auto_walkforward.md, reports/auto_walkforward_trades.csv

    python experiments/auto_walkforward.py
"""

import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import auto_train as AT
import early_target as ET
import exit_partial as P
import idio_features as I
import train
from tiers import tier_frame

COLS = AT.COLS
TARGETS = {"auto8": ("y8", "k8"), "auto15": ("y15", "k15")}
MONTHS = ["2026-05", "2026-06", "2026-07", "2026-08", "2026-09"]
GAP = pd.Timedelta(days=3)
BIG = "≥ 20 млн"
OUT = os.path.join("reports", "auto_walkforward.md")


def gkf_threshold(X, y, g, n: int = 3) -> float:
    p = np.zeros(len(y))
    for tr, te in GroupKFold(n).split(X, y, g):
        p[te] = train.make_lightgbm().fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
    return train.best_f1_threshold(y, p)


def load_coin(c: str, idx: pd.Series):
    x = pickle.load(open(os.path.join(I.DIR, f"{c}.pkl"), "rb"))
    f15 = x["f15"]
    f15 = f15.join(I.new_features(x["h"], idx).reindex(f15["ts"]).set_axis(f15.index))
    return x, f15[f15["ts"] >= pd.Timestamp(I.START, tz="UTC")]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    argparse.ArgumentParser(description="Авторозмітка всіх монет архіву + перевірка за часом (walk-forward по місяцях) проти розмітки власника").parse_args()
    d, pumps, px, idx, uni = ET.load_data()
    trained = set(d["base_asset"])
    coins = sorted(c for c in uni if c not in trained and os.path.exists(os.path.join(I.DIR, f"{c}.pkl")))
    print(f"авторозмітка {len(coins)} монет…", flush=True)
    parts = []
    for n, c in enumerate(coins, 1):
        x, f15 = load_coin(c, idx)
        s = AT.coin_series(c, x["h"])
        y8, k8 = AT.labels_for(s, f15["ts"], 0.0)
        y15, k15 = AT.labels_for(s, f15["ts"], 0.15)
        parts.append(f15[["base_asset", "ts", *COLS]].astype({k: "float32" for k in COLS})
                     .assign(y8=y8.astype("int8"), k8=k8, y15=y15.astype("int8"), k15=k15))
        if n % 100 == 0:
            print(f"  {n}/{len(coins)}", flush=True)
    a = pd.concat(parts, ignore_index=True)
    del parts
    print(f"авто: {len(a)} барів; імпульсних auto8 {a['y8'].mean():.2%}, auto15 {a['y15'].mean():.2%}; власник {d['label'].mean():.2%}", flush=True)

    Xo, yo = d[COLS], d["label"].to_numpy().astype(int)
    p = train.oof_proba(train.make_lightgbm, Xo, pd.Series(yo), d["base_asset"])
    base = (train.make_lightgbm().fit(Xo, yo), train.best_f1_threshold(yo, p))
    lto = {"base": p}
    thr_gold = {"base": base[1]}
    folds = {}                                         # (варіант, місяць) → (модель, поріг)
    info = []
    for m in MONTHS:
        cut = pd.Timestamp(m + "-01", tz="UTC") - GAP
        tr_rows = a["ts"] < cut
        for v, (yc, kc) in TARGETS.items():
            s = a[tr_rows & a[kc]]
            X, y = s[COLS], s[yc].to_numpy()
            thr = gkf_threshold(X, y, s["base_asset"])
            folds[(v, m)] = (train.make_lightgbm().fit(X, y), thr)
            info.append({"month": m, "variant": v, "rows": len(s), "pos": float(y.mean()), "thr": thr})
            print(f"  {m} {v}: навчання {len(s)} барів до {cut:%Y-%m-%d}, імпульсних {y.mean():.2%}, поріг {thr:.3f}", flush=True)
    del a
    for v in TARGETS:                                  # перенос на розмітку власника — модель вересневого кроку
        model, thr = folds[(v, MONTHS[-1])]
        lto[v], thr_gold[v] = model.predict_proba(Xo)[:, 1], thr
    V = ["base", *TARGETS]
    gold = {}
    for v in V:
        sig = lto[v] >= thr_gold[v]
        gold[v] = {**train.event_metrics(d, sig, pumps, ET.BAR), **ET.early(d, sig, pumps, px)}
        print(f"  {v} на розмітці власника: {gold[v]}", flush=True)

    m1 = joblib.load(train.model_path("lightgbm"))
    m15 = joblib.load(train.model_path(train.M15_NAME))
    exit_bars, share = m15.get("exit_bars", 4), m15.get("confirm", {}).get("share", 0.75)
    start, end = pd.Timestamp(MONTHS[0] + "-01", tz="UTC"), pd.Timestamp(I.END, tz="UTC") + pd.Timedelta(days=1)
    print(f"бектест {MONTHS[0]}…{MONTHS[-1]}: {len(coins)} монет…", flush=True)
    rows = []
    for n, c in enumerate(coins, 1):
        x, f15 = load_coin(c, idx)
        h1, h = x["h1"], x["h"]
        hh = h[(h.index >= start - pd.Timedelta(days=1)) & (h.index < end)]
        s1 = h1[["base_asset", "ts"]].assign(probability=m1["model"].predict_proba(h1[m1["features"]])[:, 1], threshold=m1["threshold"])
        mon = f15["ts"].dt.strftime("%Y-%m").to_numpy()
        X = f15[COLS]
        for v in V:
            prob, thr = np.full(len(f15), np.nan), np.full(len(f15), np.nan)
            if v == "base":
                prob[:], thr[:] = base[0].predict_proba(X)[:, 1], base[1]
            else:
                for m in MONTHS:
                    k = mon == m
                    if k.any():
                        model, t = folds[(v, m)]
                        prob[k], thr[k] = model.predict_proba(X[k])[:, 1], t
            s15 = f15[["base_asset", "ts"]].assign(probability=prob, threshold=thr).dropna()
            t = tier_frame(s15, s1, share).set_index("ts").sort_index()
            for r in P.simulate(hh, t.loc[(t.index >= start) & (t.index < end), "alert_score"], I.K, exit_bars, 3.0, 0.5, True):
                mth = pd.Timestamp(r["ts"]).strftime("%Y-%m")
                if mth in uni[c] and mth in MONTHS:
                    rows.append({"variant": v, "ticker": c, "half": AT.half(c), "month": mth, "band": uni[c][mth], **r})
        if n % 100 == 0:
            print(f"  {n}/{len(coins)}", flush=True)
    tr = pd.DataFrame(rows)
    tr.to_csv(OUT.replace(".md", "_trades.csv"), index=False)
    write_report(tr, gold, info, len(coins), V)


def write_report(tr: pd.DataFrame, gold: dict, info: list, n_coins: int, V: list) -> None:
    pf = I.pf
    L = ["# Авторозмітка всіх монет + перевірка за часом (walk-forward)", "",
         f"{n_coins} монет архіву (без тікерів власника). Тест-місяць m: навчання на авто-мітках усіх монет з квітня до (m − 3 дні), "
         "бектест місяця m на всіх монетах. base — розмітка власника (серп–жовт; для серп–вер перекриття за часом на її користь).", "",
         "**Критерій (до запуску):** PF ≥ base у ≥ 4 з 5 місяців (усі і ≥ 20 млн) на обох половинах монет; хибних на розмітці "
         "власника ≤ 1.5 × base.", "",
         "## Навчання по кроках", "", "| місяць | варіант | барів | імпульсних | поріг |", "|---|---|---|---|---|"]
    L += [f"| {r['month']} | {r['variant']} | {r['rows']} | {r['pos']:.2%} | {r['thr']:.3f} |" for r in info]
    L += ["", "## Перенос на розмітку власника (Кандидат 15m)", "",
          "auto — модель вересневого кроку (навчання до 28.08), розмітки власника не бачила; base — OOF за тікерами власника.", "",
          "| варіант | спіймано | затримка, хв | пройдено руху | фальстартів | хибних / 30 дн |", "|---|---|---|---|---|---|"]
    for v in V:
        m = gold[v]
        L.append(f"| {v} | {m['caught']} | {m['delay_min']:.0f} | {m['done']:.0%} | {m['false_starts_triggered']} | {m['false_alarms_per_30d']:.1f} |")
    L += ["", f"## Бектест травень–вересень ({tr['ticker'].nunique()} монет з угодами)", "",
          "| варіант | угод | PF | без топ-3 | медіана | ≥ 20 млн | 5–20 млн | половина A | A ≥ 20 млн | половина B | B ≥ 20 млн |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for v in V:
        g = tr[tr["variant"] == v]
        r = g["ret"].sort_values()
        big = g["band"] == BIG
        cells = []
        for hf in ("A", "B"):
            x = g[g["half"] == hf]
            cells += [f"{pf(x['ret']):.2f}", f"{pf(x[x['band'] == BIG]['ret']):.2f}"]
        L.append(f"| {v} | {len(g)} | {pf(r):.2f} | {pf(r.iloc[:-3]):.2f} | {r.median():+.2%} | {pf(g[big]['ret']):.2f} | "
                 f"{pf(g[g['band'] == '5–20 млн']['ret']):.2f} | " + " | ".join(cells) + " |")
    L += ["", "## По місяцях (PF; у дужках ≥ 20 млн)", ""]
    for hf in ("усі", "A", "B"):
        g = tr if hf == "усі" else tr[tr["half"] == hf]
        piv = g.groupby(["month", "variant"])["ret"].apply(pf).unstack().reindex(columns=V)
        piv20 = g[g["band"] == BIG].groupby(["month", "variant"])["ret"].apply(pf).unstack().reindex(columns=V)
        L += [f"**Монети: {hf}**", "", "| місяць | " + " | ".join(V) + " |", "|---|" + "---|" * len(V)]
        for mth in piv.index:
            L.append(f"| {mth} | " + " | ".join(f"{piv.at[mth, v]:.2f} ({piv20.at[mth, v]:.2f})" for v in V) + " |")
        L += ["", "Місяців ≥ base: " + ", ".join(
            f"{v} {int((piv[v] >= piv['base'] - 1e-9).sum())} / ≥ 20 млн {int((piv20[v] >= piv20['base'] - 1e-9).sum())}"
            for v in V if v != "base") + f" (з {len(piv)})", ""]
    L += ["## Критерій", ""]
    gb = gold["base"]["false_alarms_per_30d"]
    for v in V:
        if v == "base":
            continue
        ok = []
        for hf in ("A", "B"):
            g = tr[tr["half"] == hf]
            for big in (False, True):
                x = g[g["band"] == BIG] if big else g
                piv = x.groupby(["month", "variant"])["ret"].apply(pf).unstack().reindex(columns=V)
                ok.append(int((piv[v] >= piv["base"] - 1e-9).sum()) >= 4)
        fa = gold[v]["false_alarms_per_30d"] <= 1.5 * gb
        L.append(f"- {v}: місяці (A, A ≥ 20, B, B ≥ 20) {'/'.join('так' if o else 'ні' for o in ok)}; хибних "
                 f"{gold[v]['false_alarms_per_30d']:.1f} проти 1.5 × {gb:.1f} — {'так' if fa else 'ні'} → "
                 f"**{'ВИКОНАНО' if all(ok) and fa else 'НЕ виконано'}**")
    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
