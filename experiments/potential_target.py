"""
experiments/potential_target.py — ЦІЛЬ «ЗВІДСИ ЩЕ Є ПОТЕНЦІАЛ» (2026-10-10, власник: «так роби»; після auto_walkforward — авторозмітка
не замінює ручну, тож ціль будуємо на розмітці власника). Чому: signal_ranking — найвпевненіші сигнали (Q5 за P) найгірші
(≥ 20 млн PF 0.74 проти 0.94–0.98): ціль «імпульс триває» робить модель найвпевненішою посеред розвинутого руху, де входити
вже пізно; split_start30 (лише перші 30 хв) — грубий фільтр «свіжий старт», і він стабільно піднімав PF.
Тут позитив — бар ВСЕРЕДИНІ імпульсу власника, з закриття якого ціна ще пройде +TP·ATR(15m) раніше, ніж −K·ATR (стоп бектесту
I.K), за ≤ 12 год (той самий бар з обома — стоп, обережно). Решта барів імпульсу — «пізно» (0 або поза навчанням).
Поза імпульсами — як зараз (0), фальстарти — 0. Ознаки — робочі (MODEL_FEATURES_15M + власний рух).
Варіанти (зафіксовано ДО запуску):
    base       як зараз: бар у імпульсі = 1
    pot3       позитив — у імпульсі і +3 ATR до стопу (ціль часткового виходу); «пізні» бари імпульсу = 0
    pot3_x     те саме, але «пізні» бари імпульсу поза навчанням (відділяє ефект «пізно = 0» від звуження позитиву)
    pot5       позитив — у імпульсі і +5 ATR до стопу; «пізні» = 0
    start30    референс: перші 30 хв імпульсу (early_target), решта імпульсу поза навчанням
LTO — як early_target (поріг base — найкращий F1; інших — та сама кількість хибних на 30 дн, що в base).
Бектест 6 міс. на чесному списку (частковий вихід, стоп I.K, Alert = 15m + робоча годинна ≥ 0.75):
    base        вхід і утримання — Alert base
    split_X     вхід — Alert X, утримання — поки горить base або X, вихід — 4 бари без обох (схема власника 09.10)
    only_X      вхід і утримання — лише Alert X (для pot-варіантів)
КРИТЕРІЙ (до запуску): схема pot-варіанта проходить, якщо на ≥ 20 млн PF > base у ≥ 4 з 6 місяців, > base на обох половинах
монет (A / B за md5), і загальний PF ≥ base. split_start30 — лише орієнтир (найкраще з попереднього).
→ reports/potential_target.md, reports/potential_target_trades.csv

    python experiments/potential_target.py
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

import auto_train as AT
import early_target as ET
import ensemble as E
import idio_features as I
import train
from risk import atr
from tiers import tier_frame

COLS = ET.COLS
BAR = ET.BAR
HORIZON = 48                          # 12 год 15m-барів
POT = {"pot3": (3.0, True), "pot3_x": (3.0, False), "pot5": (5.0, True)}     # ціль у ATR, «пізні» = 0 (True) / поза навчанням
VARIANTS = ["base", *POT, "start30"]
BIG = "≥ 20 млн"
OUT = os.path.join("reports", "potential_target.md")


def reach(h: pd.DataFrame, ts: pd.Series, tp: float, sl: float) -> np.ndarray:
    """Для барів ts (відкриття) монети: чи з закриття ціна дійде до +tp·ATR раніше за −sl·ATR за HORIZON барів."""
    a = atr(h).to_numpy()
    c, hi, lo = h["price"].to_numpy(), h["high"].to_numpy(), h["low"].to_numpy()
    pos = h.index.get_indexer(ts)
    out = np.zeros(len(ts), dtype=bool)
    for k, i in enumerate(pos):
        if i < 0 or np.isnan(a[i]) or np.isnan(c[i]):
            continue
        up, dn = c[i] + tp * a[i], c[i] - sl * a[i]
        for j in range(i + 1, min(i + 1 + HORIZON, len(c))):
            if lo[j] <= dn:
                break
            if hi[j] >= up:
                out[k] = True
                break
    return out


def impulse_phase(d: pd.DataFrame, pumps: pd.DataFrame) -> np.ndarray:
    """Частка пройденого часу імпульсу на закритті бару (NaN поза імпульсами власника)."""
    close = d["ts"] + BAR
    ph = np.full(len(d), np.nan)
    for r in pumps[pumps["pattern_type"] == "pump"].itertuples():
        m = ((d["base_asset"] == r.base_asset) & (close > r.start_ts) & (close <= r.end_ts)).to_numpy()
        ph[m] = ((close[m] - r.start_ts) / (r.end_ts - r.start_ts)).to_numpy()
    return ph


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    argparse.ArgumentParser(description="Ціль «звідси ще є потенціал» (+TP·ATR до стопу всередині імпульсу): LTO і бектест 6 міс.").parse_args()
    d, pumps, px, idx, uni = ET.load_data()
    X, grp = d[COLS], d["base_asset"]
    trained = set(d["base_asset"])
    lab = d["label"].to_numpy().astype(bool)
    print(f"навчання: {len(d)} барів, {d['base_asset'].nunique()} тікерів, у імпульсах {lab.sum()}", flush=True)

    hit = {tp: np.zeros(len(d), dtype=bool) for tp in {v[0] for v in POT.values()}}
    for b, g in d[lab].groupby("base_asset"):
        for tp in hit:
            hit[tp][g.index.to_numpy()] = reach(px[b], g["ts"], tp, I.K)
    phase = impulse_phase(d, pumps)
    ys, keeps = {"base": lab.astype(int)}, {"base": np.ones(len(d), dtype=bool)}
    for v, (tp, late0) in POT.items():
        pos = lab & hit[tp]
        ys[v] = pos.astype(int)
        keeps[v] = np.ones(len(d), dtype=bool) if late0 else ~(lab & ~pos)
    pos, keep = ET.target(d, pumps, pd.Timedelta(0), pd.Timedelta(minutes=30))
    ys["start30"], keeps["start30"] = pos.astype(int), keep
    stats = []
    for v in VARIANTS:
        y = ys[v].astype(bool)
        terc = [y[lab & (phase > lo) & (phase <= hi)].mean() for lo, hi in ((0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.01))]
        stats.append((v, int(y.sum()), int(keeps[v].sum()), terc))
        print(f"  {v}: позитивів {y.sum()}, у навчанні {keeps[v].sum()}; частка позитиву по третинах імпульсу "
              + " / ".join(f"{t:.0%}" for t in terc), flush=True)

    p, thr, lto = {}, {}, {}
    for v in VARIANTS:
        p[v] = ET.oof(X, ys[v], keeps[v], grp)
        if v == "base":
            thr[v] = train.best_f1_threshold(ys[v], p[v])
            fa0 = train.event_metrics(d, p[v] >= thr[v], pumps, BAR)["false_alarms_per_30d"]
        else:
            thr[v] = ET.matched_threshold(d, p[v], pumps, fa0)
        sig = p[v] >= thr[v]
        lto[v] = {**train.event_metrics(d, sig, pumps, BAR), **ET.early(d, sig, pumps, px), "move_done": E.move_done(d, sig, pumps, px)}
        print(f"  {v} (поріг {thr[v]:.3f}): {lto[v]}", flush=True)
    models = {v: train.make_lightgbm().fit(X[keeps[v]], ys[v][keeps[v]]) for v in VARIANTS}

    m1 = joblib.load(train.model_path("lightgbm"))
    m15 = joblib.load(train.model_path(train.M15_NAME))
    exit_bars, share = m15.get("exit_bars", 4), m15.get("confirm", {}).get("share", 0.75)
    start, end = pd.Timestamp(I.START, tz="UTC"), pd.Timestamp(I.END, tz="UTC") + pd.Timedelta(days=1)
    others = [v for v in VARIANTS if v != "base"]
    bt = ["base"] + [f"split_{v}" for v in others] + [f"only_{v}" for v in POT]
    rows = []
    coins = sorted(c for c in uni if c not in trained and os.path.exists(os.path.join(I.DIR, f"{c}.pkl")))
    print(f"бектест: {len(coins)} монет чесного списку…", flush=True)
    for n, c in enumerate(coins, 1):
        x = pickle.load(open(os.path.join(I.DIR, f"{c}.pkl"), "rb"))
        f15, h1, h = x["f15"], x["h1"], x["h"]
        f15 = f15.join(I.new_features(h, idx).reindex(f15["ts"]).set_axis(f15.index))
        hh = h[(h.index >= start - pd.Timedelta(days=1)) & (h.index < end)]
        s1 = h1[["base_asset", "ts"]].assign(probability=m1["model"].predict_proba(h1[m1["features"]])[:, 1], threshold=m1["threshold"])
        fr = {}
        for v in VARIANTS:
            s15 = f15[["base_asset", "ts"]].assign(probability=models[v].predict_proba(f15[COLS])[:, 1], threshold=thr[v])
            t = tier_frame(s15, s1, share).set_index("ts").sort_index()
            fr[v] = t.loc[(t.index >= start) & (t.index < end), "alert_score"]
        a0 = fr["base"]
        sc = {"base": (a0, a0)}
        for v in others:
            ea = fr[v].reindex(a0.index)
            sc[f"split_{v}"] = (ea, np.maximum(a0, ea))
            if v in POT:
                sc[f"only_{v}"] = (ea, ea)
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
    write_report(tr, lto, thr, stats, bt)


def write_report(tr: pd.DataFrame, lto: dict, thr: dict, stats: list, bt: list) -> None:
    pf = I.pf
    L = ["# Ціль «звідси ще є потенціал»", "",
         f"Позитив pot — бар у імпульсі власника, з якого ціна дійде до +3 / +5 ATR(15m) раніше за стоп −{I.K} ATR за ≤ 12 год. "
         "Поріг base — найкращий F1 на OOF; інших — та сама кількість хибних на 30 дн: " + ", ".join(f"{k} {v:.3f}" for k, v in thr.items()) + ".", "",
         "**Критерій (до запуску):** ≥ 20 млн PF > base у ≥ 4 з 6 місяців і на обох половинах монет, загальний PF ≥ base.", "",
         "## Мітки", "", "| ціль | позитивів | у навчанні | частка позитиву: 1-а / 2-а / 3-я третина імпульсу |", "|---|---|---|---|"]
    L += [f"| {v} | {n} | {k} | " + " / ".join(f"{t:.0%}" for t in terc) + " |" for v, n, k, terc in stats]
    L += ["", "## Leave-ticker-out (Кандидат 15m)", "",
          "| ціль | спіймано | до старту | затримка, хв | пройдено руху | фальстартів | хибних / 30 дн |", "|---|---|---|---|---|---|---|"]
    for v, m in lto.items():
        L.append(f"| {v} | {m['caught']} | {m['pre_start']} | {m['delay_min']:.0f} | {m['done']:.0%} | "
                 f"{m['false_starts_triggered']} | {m['false_alarms_per_30d']:.1f} |")
    L += ["", f"## Бектест 6 міс., {tr['ticker'].nunique()} монет чесного списку", "",
          "split_X — вхід за Alert X, утримання — base або X; only_X — вхід і утримання лише за X.", "",
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
        if v == "base" or v.endswith("start30"):
            continue
        g = tr[tr["variant"] == v]
        months = int((piv20[v] > piv20["base"] + 1e-9).sum())
        halves = all(pf(g[(g["band"] == BIG) & (g["half"] == hf)]["ret"]) > pf(b[(b["band"] == BIG) & (b["half"] == hf)]["ret"])
                     for hf in ("A", "B"))
        total = pf(g["ret"]) >= pf(b["ret"])
        ok = months >= 4 and halves and total
        L.append(f"- {v}: місяців ≥ 20 млн краще за base {months} з {len(piv20)}; обидві половини {'так' if halves else 'ні'}; "
                 f"загальний PF ≥ base {'так' if total else 'ні'} → **{'ВИКОНАНО' if ok else 'НЕ виконано'}**")
    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
