"""
experiments/teacher_walkforward.py — РОЗМІТКА ВСІХ МОНЕТ «ВЧИТЕЛЕМ» + перевірка за часом (2026-10-10, етап 3 після
experiments/label_teacher.py: вчитель на описах руху заднім числом відтворює розмітку власника — AUC 0.88 на відкладених тікерах,
при тих самих 106/129 знайдених імпульсах зайвих 202 проти 467 у правила).
Вчитель (LightGBM на всіх подіях тікерів власника) оцінює кожну подію авторозмітки (тригер T1) на монетах архіву: P «власник
назвав би це імпульсом». Мітки барів для причинної 15m-моделі («учня»): бари від max(старт, тригер − 1 год) до кінця події —
    P ≥ hi   → 1 (імпульс)
    lo…hi    → сіра зона, поза навчанням
    P < lo   → 0 (як звичайний ринок)
Пороги — з кривої вчителя на відкладених тікерах власника (не з бектесту): lo — де вчитель знаходить стільки ж імпульсів
власника, скільки правило; hi — де знаходить ≥ 100 (варіант t100) або ≥ 90 (t90) зі 129.
Перевірка за часом — як experiments/auto_walkforward.py: тест-місяць m (травень…вересень), навчання на мітках усіх монет з
квітня до (m − 3 дні), поріг учня — найкращий F1 на OOF 3 груп монет, бектест місяця m (Alert = 15m + робоча годинна ≥ 0.75,
частковий вихід, стоп I.K, чесний список). base — модель на розмітці власника (одна; серп–жовт — перекриття на її користь).
Перенос на розмітку власника — учень вересневого кроку (тікери власника в навчанні учня НЕ беруть участі; вчитель їх бачив —
тому це перевірка учня, а не вчителя).
КРИТЕРІЙ (до запуску, той самий, що в auto_walkforward): PF ≥ base у ≥ 4 з 5 місяців (усі і ≥ 20 млн) на обох половинах монет;
хибних на розмітці власника ≤ 1.5 × base.
→ reports/teacher_walkforward.md, reports/teacher_walkforward_trades.csv

    python experiments/teacher_walkforward.py
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import numpy as np
import pandas as pd

import auto_train as AT
import auto_walkforward as AW
import early_target as ET
import exit_partial as P
import idio_features as I
import label_teacher as LT
import train
from tiers import tier_frame

COLS = AT.COLS
TARGETS = {"t100": ("y100", "k100"), "t90": ("y90", "k90")}
FOUND = {"t100": 100, "t90": 90}
MONTHS = AW.MONTHS
OUT = os.path.join("reports", "teacher_walkforward.md")
H1 = np.timedelta64(1, "h")


def bar_labels(ts: pd.Series, e: pd.DataFrame, hi: float, lo: float) -> tuple[np.ndarray, np.ndarray]:
    """Мітки (1 / 0) і маска «в навчанні» для барів ts за подіями e з оцінкою вчителя p."""
    close = ts.values + np.timedelta64(15, "m")
    y, grey = np.zeros(len(ts), bool), np.zeros(len(ts), bool)
    for r in e.itertuples():
        st = max(np.datetime64(r.start.tz_convert(None)), np.datetime64(r.trigger.tz_convert(None)) - H1)
        m = (close > st) & (close <= np.datetime64(r.end_ts.tz_convert(None)))
        if r.p >= hi:
            y |= m
        elif r.p >= lo:
            grey |= m
    return y, ~(grey & ~y)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    argparse.ArgumentParser(description="Розмітка всіх монет архіву «вчителем» + перевірка за часом проти розмітки власника").parse_args()
    k, own_p, own_f, reviewed, days = LT.owner_events()
    p = train.oof_proba(train.make_lightgbm, k[LT.OPIS], k["y"], k["base"])
    rule = LT.score(k, (k["cls"] == "pump").to_numpy(), own_p, reviewed)
    grid = np.unique(np.round(p, 3))[::-1]
    curve = [(t, LT.score(k, p >= t, own_p, reviewed)) for t in grid]
    at = lambda n: next((t for t, s in curve if s["found"] >= n), grid[-1])          # noqa: E731
    lo = at(rule["found"])
    his = {v: at(n) for v, n in FOUND.items()}
    teacher = LT.fit_teacher(k)
    print(f"вчитель: {len(k)} подій власника; lo {lo:.3f}, hi " + ", ".join(f"{v} {t:.3f}" for v, t in his.items()), flush=True)

    d, pumps, px, idx, uni = ET.load_data()
    trained = set(d["base_asset"])
    coins = sorted(c for c in uni if c not in trained and os.path.exists(os.path.join(I.DIR, f"{c}.pkl")))
    print(f"розмітка вчителем {len(coins)} монет…", flush=True)
    parts, n_ev = [], {"всі": 0, **{v: 0 for v in TARGETS}}
    for n, c in enumerate(coins, 1):
        x, f15 = AW.load_coin(c, idx)
        s = AT.coin_series(c, x["h"])
        rows = LT.event_rows(s, idx.reindex(s.index))
        e = pd.DataFrame(rows)
        if len(e):
            e["p"] = teacher.predict_proba(e[LT.OPIS])[:, 1]
            n_ev["всі"] += len(e)
        lab = {}
        for v, (yc, kc) in TARGETS.items():
            if len(e):
                yv, kv = bar_labels(f15["ts"], e, his[v], lo)
                n_ev[v] += int((e["p"] >= his[v]).sum())
            else:
                yv, kv = np.zeros(len(f15), bool), np.ones(len(f15), bool)
            lab[yc], lab[kc] = yv.astype("int8"), kv
        parts.append(f15[["base_asset", "ts", *COLS]].astype({q: "float32" for q in COLS}).assign(**lab))
        if n % 100 == 0:
            print(f"  {n}/{len(coins)}", flush=True)
    a = pd.concat(parts, ignore_index=True)
    del parts
    print(f"розмічено: {len(a)} барів; подій {n_ev}; імпульсних барів " + ", ".join(f"{v} {a[yc].mean():.2%}" for v, (yc, _) in TARGETS.items())
          + f"; власник {d['label'].mean():.2%}", flush=True)

    Xo, yo = d[COLS], d["label"].to_numpy().astype(int)
    po = train.oof_proba(train.make_lightgbm, Xo, pd.Series(yo), d["base_asset"])
    base = (train.make_lightgbm().fit(Xo, yo), train.best_f1_threshold(yo, po))
    lto, thr_gold = {"base": po}, {"base": base[1]}
    folds, info = {}, []
    for m in MONTHS:
        cut = pd.Timestamp(m + "-01", tz="UTC") - AW.GAP
        tr_rows = a["ts"] < cut
        for v, (yc, kc) in TARGETS.items():
            s = a[tr_rows & a[kc]]
            X, y = s[COLS], s[yc].to_numpy()
            thr = AW.gkf_threshold(X, y, s["base_asset"])
            folds[(v, m)] = (train.make_lightgbm().fit(X, y), thr)
            info.append({"month": m, "variant": v, "rows": len(s), "pos": float(y.mean()), "thr": thr})
            print(f"  {m} {v}: навчання {len(s)} барів до {cut:%Y-%m-%d}, імпульсних {y.mean():.2%}, поріг {thr:.3f}", flush=True)
    del a
    for v in TARGETS:
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
        x, f15 = AW.load_coin(c, idx)
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
                    q = mon == m
                    if q.any():
                        model, t = folds[(v, m)]
                        prob[q], thr[q] = model.predict_proba(X[q])[:, 1], t
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
    AW.OUT = OUT
    AW.write_report(tr, gold, info, len(coins), V)
    txt = open(OUT, encoding="utf-8").read().split("\n", 2)
    head = ["# Розмітка всіх монет «вчителем» + перевірка за часом", "",
            f"Вчитель — LightGBM на описах руху заднім числом, навчений на {len(k)} подіях тікерів власника (label_teacher.py). Пороги з його "
            f"кривої на відкладених тікерах: lo {lo:.3f} (нижче — «не імпульс»), hi " + ", ".join(f"{v} {t:.3f}" for v, t in his.items()) +
            f" (вище — імпульс; між — сіра зона). Подій на монетах архіву: {n_ev['всі']}, з них імпульсів " +
            ", ".join(f"{v} {n_ev[v]}" for v in TARGETS) + ". Бари імпульсу — від max(старт, тригер − 1 год) до кінця події.", ""]
    open(OUT, "w", encoding="utf-8").write("\n".join(head) + txt[2])
    print("\n".join(head))


if __name__ == "__main__":
    main()
