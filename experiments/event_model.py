"""
experiments/event_model.py — ПОДІЄВА ЛОГІКА «СПЛЕСК → ІМПУЛЬС ЧИ ФАЛЬСТАРТ» (2026-10-10, власник: «так, давай протестуємо»; план
«інша логіка», крок 2). Зараз модель оцінює кожен 15m-бар, 97% барів — нудні нулі, а in_play показав: модель бачить сплеск, але
не розрізняє «з продовженням» і «згасне». Тут модель відповідає ЛИШЕ в момент сплеску — і вчиться лише на таких моментах.
Сплеск (лише минуле, на закритті бару) — тригер T1 авторозмітки: за 1 год ціна ≥ 2 ATR(15m), OI ≥ +2%, обсяг ≥ 2× медіани 7 днів
(auto_label.trigger_mask). Епізод — бари з тригером, розрив ≤ 1 год; рішення — на перших 4 барах епізоду (ev_k = 0…3 — ознака).
Розмітка — власника (як навчання робочої моделі: переглянуті тікери цілком, інші — лише розмічені інтервали).
Варіанти цілі (зафіксовано ДО запуску):
    ev_in      1 = закриття бару тригера в (старт − 1 год, кінець] імпульсу власника; решта (фальстарти, нерозмічені сплески) — 0
    ev_start   1 = закриття в (старт − 1 год, старт + 1 год]; пізніші тригери всередині імпульсу — ПОЗА навчанням (урок
               potential_target: схожі бари з протилежною міткою — шум); решта — 0
Ознаки — робочі (MODEL_FEATURES_15M + власний рух) + ev_k. LightGBM (train.make_lightgbm), OOF за тікерами, поріг — найкращий F1
на OOF рядків-тригерів. LTO: сигнал моделі лише на барах-тригерах → train.event_metrics / early_target.early на всіх барах.
Бектест 6 міс. на чесному списку (частковий вихід, стоп I.K, як potential_target):
    base            Alert як зараз (вхід і утримання)
    trig            вхід на БУДЬ-ЯКОМУ тригері (+ годинна ≥ 0.75), утримання — base або сигнал входу (що дає сам тригер)
    X               вхід — тригер і модель X ≥ порогу (+ годинна ≥ 0.75), утримання — base або сигнал входу, вихід — 4 бари без
    X_c             те саме без годинного підтвердження
Орієнтир — split_start30 з reports/potential_target.md (≥ 20 млн 0.96, загальний 0.95, ті самі монети й налаштування).
КРИТЕРІЙ (до запуску): схема X проходить, якщо на ≥ 20 млн PF > base у ≥ 4 з 6 місяців, > base на обох половинах монет, і
загальний PF ≥ base; і краща за trig (модель додає щось до самого тригера).
→ reports/event_model.md, reports/event_model_trades.csv

    python experiments/event_model.py
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

import auto_label as AL
import auto_train as AT
import early_target as ET
import ensemble as E
import idio_features as I
import train
from risk import atr
from tiers import tier_frame

COLS = ET.COLS + ["ev_k"]
BAR = ET.BAR
TRIG = AL.TRIGGERS["T1"]
GAP_BARS, FIRST_K = 4, 4             # епізод — тригери з розривом ≤ 1 год; рішення — на перших 4 барах епізоду
PRE, POST = pd.Timedelta(hours=1), pd.Timedelta(hours=1)
TARGETS = ["ev_in", "ev_start"]
BIG = "≥ 20 млн"
OUT = os.path.join("reports", "event_model.md")


def trigger_k(s: pd.DataFrame) -> pd.Series:
    """Номер бару в епізоді сплеску (0…FIRST_K−1) для барів з тригером T1; NaN — не тригер або пізніший бар епізоду."""
    m = AL.trigger_mask(s, *TRIG)
    pos = np.flatnonzero(m)
    k = np.full(len(s), np.nan)
    prev, n = -10 ** 9, 0
    for i in pos:
        n = n + 1 if i - prev <= GAP_BARS else 0
        prev = i
        if n < FIRST_K:
            k[i] = n
    return pd.Series(k, index=s.index)


def own_series(p: pd.DataFrame) -> pd.DataFrame:
    """Ряд для тригера з px розмітки (як auto_label.series): OI — монетний індекс × ціна, ATR, медіана обсягу 7 днів."""
    s = p[["high", "low", "price", "volume"]].copy()
    s["oi"] = np.exp(p["oi_coin_log_idx"]) * p["price"]
    s["atr"] = atr(s)
    s["vol_med"] = s["volume"].rolling(7 * AL.DAY, min_periods=AL.DAY).median().shift(1)
    return s.dropna(subset=["price", "high", "low"])


def targets(d: pd.DataFrame, pumps: pd.DataFrame) -> dict:
    """Мітки рядків d для обох цілей: (y, keep)."""
    close = d["ts"] + BAR
    inn, st = np.zeros(len(d), bool), np.zeros(len(d), bool)
    for r in pumps[pumps["pattern_type"] == "pump"].itertuples():
        b = (d["base_asset"] == r.base_asset).to_numpy()
        inn |= b & ((close > r.start_ts - PRE) & (close <= r.end_ts)).to_numpy()
        st |= b & ((close > r.start_ts - PRE) & (close <= min(r.start_ts + POST, r.end_ts))).to_numpy()
    return {"ev_in": (inn.astype(int), np.ones(len(d), bool)), "ev_start": (st.astype(int), ~(inn & ~st))}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    argparse.ArgumentParser(description="Подієва логіка: модель відповідає лише в момент сплеску (тригер T1) — імпульс чи фальстарт").parse_args()
    d, pumps, px, idx, uni = ET.load_data()
    trained = set(d["base_asset"])
    d["ev_k"] = np.nan
    for b, g in d.groupby("base_asset"):
        d.loc[g.index, "ev_k"] = trigger_k(own_series(px[b])).reindex(g["ts"]).to_numpy()
    ev = d["ev_k"].notna().to_numpy()
    e = d[ev].reset_index()                      # рядки-тригери; колонка index — позиція в d
    print(f"навчання: {len(d)} барів, тригерних {len(e)} ({ev.mean():.2%}), тікерів {d['base_asset'].nunique()}", flush=True)

    tg = targets(d, pumps)
    thr, lto, models, info = {}, {}, {}, []
    p0 = train.oof_proba(train.make_lightgbm, d[ET.COLS], d["label"].astype(int), d["base_asset"])
    thr["base"] = train.best_f1_threshold(d["label"].to_numpy().astype(int), p0)
    sig0 = p0 >= thr["base"]
    lto["base"] = {**train.event_metrics(d, sig0, pumps, BAR), **ET.early(d, sig0, pumps, px)}
    trig_sig = ev.copy()
    lto["trig"] = {**train.event_metrics(d, trig_sig, pumps, BAR), **ET.early(d, trig_sig, pumps, px)}
    for v in TARGETS:
        y, keep = tg[v][0][e["index"]], tg[v][1][e["index"]]
        info.append((v, int(y.sum()), int(keep.sum()), len(e)))
        p = ET.oof(e[COLS], y, keep, e["base_asset"])
        thr[v] = train.best_f1_threshold(y[keep], p[keep])
        sig = np.zeros(len(d), bool)
        sig[e["index"].to_numpy()] = p >= thr[v]
        lto[v] = {**train.event_metrics(d, sig, pumps, BAR), **ET.early(d, sig, pumps, px), "move_done": E.move_done(d, sig, pumps, px)}
        models[v] = train.make_lightgbm().fit(e.loc[keep, COLS], y[keep])
        print(f"  {v}: позитивів {y.sum()}, у навчанні {keep.sum()} з {len(e)}, поріг {thr[v]:.3f}: {lto[v]}", flush=True)
    for k in ("base", "trig"):
        print(f"  {k}: {lto[k]}", flush=True)
    base_model = train.make_lightgbm().fit(d[ET.COLS], d["label"].to_numpy().astype(int))

    m1 = joblib.load(train.model_path("lightgbm"))
    m15 = joblib.load(train.model_path(train.M15_NAME))
    exit_bars, share = m15.get("exit_bars", 4), m15.get("confirm", {}).get("share", 0.75)
    start, end = pd.Timestamp(I.START, tz="UTC"), pd.Timestamp(I.END, tz="UTC") + pd.Timedelta(days=1)
    bt = ["base", "trig", "trig_c"] + [f"{v}{s}" for v in TARGETS for s in ("", "_c")]
    rows, n_trig = [], 0
    coins = sorted(c for c in uni if c not in trained and os.path.exists(os.path.join(I.DIR, f"{c}.pkl")))
    print(f"бектест: {len(coins)} монет чесного списку…", flush=True)
    for n, c in enumerate(coins, 1):
        x = pickle.load(open(os.path.join(I.DIR, f"{c}.pkl"), "rb"))
        f15, h1, h = x["f15"], x["h1"], x["h"]
        f15 = f15.join(I.new_features(h, idx).reindex(f15["ts"]).set_axis(f15.index))
        f15["ev_k"] = trigger_k(AT.coin_series(c, h)).reindex(f15["ts"]).to_numpy()
        hh = h[(h.index >= start - pd.Timedelta(days=1)) & (h.index < end)]
        s1 = h1[["base_asset", "ts"]].assign(probability=m1["model"].predict_proba(h1[m1["features"]])[:, 1], threshold=m1["threshold"])
        tm = f15["ev_k"].notna().to_numpy()
        n_trig += int(tm[(f15["ts"] >= start).to_numpy()].sum())

        def frame(prob, t):
            s15 = f15[["base_asset", "ts"]].assign(probability=prob, threshold=t)
            fr = tier_frame(s15, s1, share).set_index("ts").sort_index()
            return fr.loc[(fr.index >= start) & (fr.index < end)]
        b = frame(base_model.predict_proba(f15[ET.COLS])[:, 1], thr["base"])
        a0 = b["alert_score"]
        sc = {"base": (a0, a0)}
        entries = {"trig": tm.astype(float)}               # будь-який тригер = «сигнал» 1.0 (поріг 1)
        for v in TARGETS:
            pv = np.zeros(len(f15))
            if tm.any():
                pv[tm] = models[v].predict_proba(f15.loc[tm, COLS])[:, 1]
            entries[v] = pv / thr[v]
        for v, s in entries.items():
            fr = frame(s, 1.0)
            for suf, ent in (("", fr["alert_score"]), ("_c", fr["s15"])):
                ent = ent.reindex(a0.index)
                sc[f"{v}{suf}"] = (ent, np.maximum(a0, ent))
        months = uni[c]
        for v in bt:
            for r in ET.simulate_split(hh, *sc[v], I.K, exit_bars, 3.0, 0.5, True):
                mth = pd.Timestamp(r["ts"]).strftime("%Y-%m")
                if mth in months:
                    rows.append({"variant": v, "ticker": c, "half": AT.half(c), "month": mth, "band": months[mth], **r})
        if n % 50 == 0:
            print(f"  {n}/{len(coins)}", flush=True)
    tr = pd.DataFrame(rows)
    tr.to_csv(OUT.replace(".md", "_trades.csv"), index=False)
    write_report(tr, lto, thr, info, bt, n_trig, len(coins))


def write_report(tr, lto, thr, info, bt, n_trig, n_coins) -> None:
    pf = I.pf
    L = ["# Подієва логіка: сплеск → імпульс чи фальстарт", "",
         f"Тригер T1 (за 1 год ціна ≥ 2 ATR, OI ≥ +2%, обсяг ≥ 2× медіани 7 днів), рішення на перших {FIRST_K} барах епізоду. "
         f"У бектесті тригерних барів {n_trig} на {n_coins} монетах (≈ {n_trig / max(n_coins, 1) / 6:.1f} на монету в місяць). "
         "Пороги: " + ", ".join(f"{k} {v:.3f}" for k, v in thr.items()) + ".", "",
         "**Критерій (до запуску):** ≥ 20 млн PF > base у ≥ 4 з 6 місяців і на обох половинах монет, загальний PF ≥ base, і краще за trig. "
         "Орієнтир: split_start30 (reports/potential_target.md) — ≥ 20 млн 0.96, загальний 0.95.", "",
         "## Мітки (рядки-тригери на розмітці власника)", "", "| ціль | позитивів | у навчанні | тригерних рядків |", "|---|---|---|---|"]
    L += [f"| {v} | {p} | {k} | {n} |" for v, p, k, n in info]
    L += ["", "## Leave-ticker-out (сигнал лише на барах-тригерах; base — Кандидат як зараз; trig — будь-який тригер)", "",
          "| варіант | спіймано | до старту | затримка, хв | пройдено руху | фальстартів | хибних / 30 дн |", "|---|---|---|---|---|---|---|"]
    for v, m in lto.items():
        L.append(f"| {v} | {m['caught']} | {m['pre_start']} | {m['delay_min']:.0f} | {m['done']:.0%} | "
                 f"{m['false_starts_triggered']} | {m['false_alarms_per_30d']:.1f} |")
    L += ["", f"## Бектест 6 міс., {tr['ticker'].nunique()} монет з угодами", "",
          "X — вхід на тригері, якщо модель X ≥ порогу (+ годинна ≥ 0.75); X_c — без годинної; утримання — base або сигнал входу.", "",
          "| схема | угод | у плюсі | медіана | PF | без топ-3 | ≥ 20 млн | 5–20 млн | A ≥ 20 млн | B ≥ 20 млн |", "|---|---|---|---|---|---|---|---|---|---|"]
    for v in bt:
        g = tr[tr["variant"] == v]
        r = g["ret"].sort_values()
        big = g["band"] == BIG
        L.append(f"| {v} | {len(r)} | {(r > 0).mean():.0%} | {r.median():+.2%} | {pf(r):.2f} | {pf(r.iloc[:-3]):.2f} | "
                 f"{pf(g[big]['ret']):.2f} | {pf(g[g['band'] == '5–20 млн']['ret']):.2f} | "
                 f"{pf(g[big & (g['half'] == 'A')]['ret']):.2f} | {pf(g[big & (g['half'] == 'B')]['ret']):.2f} |")
    piv = tr.groupby(["month", "variant"])["ret"].apply(pf).unstack().reindex(columns=bt)
    piv20 = tr[tr["band"] == BIG].groupby(["month", "variant"])["ret"].apply(pf).unstack().reindex(columns=bt)
    L += ["", "## По місяцях (PF; у дужках ≥ 20 млн)", "", "| місяць | " + " | ".join(bt) + " |", "|---|" + "---|" * len(bt)]
    for mth in piv.index:
        L.append(f"| {mth} | " + " | ".join(f"{piv.at[mth, v]:.2f} ({piv20.at[mth, v]:.2f})" for v in bt) + " |")
    L += ["", "## Критерій", ""]
    b = tr[tr["variant"] == "base"]
    for v in bt:
        if v in ("base", "trig", "trig_c"):
            continue
        g = tr[tr["variant"] == v]
        t = tr[tr["variant"] == ("trig_c" if v.endswith("_c") else "trig")]
        months = int((piv20[v] > piv20["base"] + 1e-9).sum())
        halves = all(pf(g[(g["band"] == BIG) & (g["half"] == hf)]["ret"]) > pf(b[(b["band"] == BIG) & (b["half"] == hf)]["ret"])
                     for hf in ("A", "B"))
        total = pf(g["ret"]) >= pf(b["ret"])
        vs_trig = pf(g[g["band"] == BIG]["ret"]) > pf(t[t["band"] == BIG]["ret"])
        ok = months >= 4 and halves and total and vs_trig
        L.append(f"- {v}: місяців ≥ 20 млн краще за base {months} з {len(piv20)}; обидві половини {'так' if halves else 'ні'}; "
                 f"загальний ≥ base {'так' if total else 'ні'}; краще за тригер {'так' if vs_trig else 'ні'} → "
                 f"**{'ВИКОНАНО' if ok else 'НЕ виконано'}**")
    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
