"""
experiments/age_threshold.py — ПОРІГ ВХОДУ ЗАЛЕЖНО ВІД «ВІКУ» СПЛЕСКУ (2026-10-10, огляд джерел, теги «точність» і «швидший вхід»;
власник: «давай по черзі»). Джерело ідеї — «early classification of time series» (arXiv 2406.18332, 2005.02595): рішення
«спрацювати зараз чи чекати» з явною ціною затримки, а не один поріг на всі моменти. Наші факти: найвпевненіші сигнали (Q5 за P)
найгірші; вхід лише на свіжому старті (split_start30) — найкращий; основна затримка — модель довго під порогом.
Вік сплеску (лише минуле): бари від ПЕРШОГО тригера T1 поточного епізоду (auto_label.trigger_mask: за 1 год ціна ≥ 2 ATR,
OI ≥ +2%, обсяг ≥ 2× медіани 7 днів); епізод триває, поки між тригерами ≤ 4 год; без жодного тригера за 4 год — «сплеску немає».
Модель одна — base (ознаки робочої, розмітка власника). Міняється лише правило ВХОДУ; утримання й вихід — Alert base, як зараз.
Варіанти входу (зафіксовано ДО запуску; множник — на поріг Alert):
    base       як зараз (вік не важить)
    fresh2     вхід лише при віці ≤ 2 год (після — заборонено; без сплеску — заборонено)
    fresh4     вхід лише при віці ≤ 4 год
    early      ≤ 1 год — поріг × 0.85 (раніше), 1–2 год — × 1.0, далі заборонено
    ramp       ≤ 1 год × 0.85, 1–2 год × 1.0, 2–4 год × 1.15, старше або без сплеску × 1.3 (без заборони)
Орієнтир — split_start30 (reports/potential_target.md, той самий бектест): ≥ 20 млн 0.96.
LTO на розмітці власника (рівень Кандидат, той самий множник): спіймано, затримка, пройдено руху, фальстарти, хибні.
Бектест 6 міс., чесний список, частковий вихід, стоп I.K, половини монет.
КРИТЕРІЙ (до запуску). Точність: ≥ 20 млн PF − base ≥ 0.03, кращий у ≥ 4 з 6 місяців і на обох половинах, загальний PF ≥ base.
Швидкість (для early / ramp): на розмітці затримка або пройдено руху МЕНШІ за base, хибних ≤ 1.5 × base, і PF ≥ 20 млн не нижчий за base.
→ reports/age_threshold.md, reports/age_threshold_trades.csv

    python experiments/age_threshold.py
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
import event_model as EV
import idio_features as I
import train
from tiers import tier_frame

H = 4                                   # 15m-барів на годину
EPISODE_GAP = 4 * H                     # епізод триває, поки між тригерами ≤ 4 год
INF = 10 ** 6


def mult_fresh(hours: float):
    return lambda age: np.where(age <= hours * H, 1.0, np.inf)


def mult_early(age):
    return np.select([age <= 1 * H, age <= 2 * H], [0.85, 1.0], np.inf)


def mult_ramp(age):
    return np.select([age <= 1 * H, age <= 2 * H, age <= 4 * H], [0.85, 1.0, 1.15], 1.3)


RULES = {"base": lambda age: np.ones(len(age)), "fresh2": mult_fresh(2), "fresh4": mult_fresh(4), "early": mult_early, "ramp": mult_ramp}
BIG = "≥ 20 млн"
OUT = os.path.join("reports", "age_threshold.md")


def spike_age(s: pd.DataFrame) -> pd.Series:
    """Бари від першого тригера T1 поточного епізоду (INF — сплеску немає: жодного тригера за останні 4 год)."""
    m = AL.trigger_mask(s, *EV.TRIG)
    age = np.full(len(s), INF, dtype=float)
    first, last = None, None
    for i in range(len(s)):
        if m[i]:
            if last is None or i - last > EPISODE_GAP:
                first = i
            last = i
        if last is not None and i - last <= EPISODE_GAP:
            age[i] = i - first
    return pd.Series(age, index=s.index)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    argparse.ArgumentParser(description="Поріг входу залежно від віку сплеску (бари від першого тригера T1): LTO і бектест 6 міс.").parse_args()
    d, pumps, px, idx, uni = ET.load_data()
    trained = set(d["base_asset"])
    d["age"] = float(INF)
    for b, g in d.groupby("base_asset"):
        d.loc[g.index, "age"] = spike_age(EV.own_series(px[b])).reindex(g["ts"]).fillna(INF).to_numpy()
    age = d["age"].to_numpy()
    y = d["label"].to_numpy().astype(int)
    p = train.oof_proba(train.make_lightgbm, d[ET.COLS], pd.Series(y), d["base_asset"])
    thr = train.best_f1_threshold(y, p)
    print(f"навчання: {len(d)} барів; поріг base {thr:.3f}; барів зі сплеском (вік ≤ 4 год) {(age <= 4 * H).mean():.1%}", flush=True)
    lto = {}
    for v, f in RULES.items():
        sig = p / thr / f(age) >= 1
        lto[v] = {**train.event_metrics(d, sig, pumps, ET.BAR), **ET.early(d, sig, pumps, px)}
        print(f"  {v}: {lto[v]}", flush=True)
    model = train.make_lightgbm().fit(d[ET.COLS], y)

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
        f15 = f15.join(I.new_features(h, idx).reindex(f15["ts"]).set_axis(f15.index))
        hh = h[(h.index >= start - pd.Timedelta(days=1)) & (h.index < end)]
        s15 = f15[["base_asset", "ts"]].assign(probability=model.predict_proba(f15[ET.COLS])[:, 1], threshold=thr)
        s1 = h1[["base_asset", "ts"]].assign(probability=m1["model"].predict_proba(h1[m1["features"]])[:, 1], threshold=m1["threshold"])
        t = tier_frame(s15, s1, share).set_index("ts").sort_index()
        a0 = t.loc[(t.index >= start) & (t.index < end), "alert_score"]
        ag = spike_age(AT.coin_series(c, h)).reindex(a0.index).fillna(INF).to_numpy()
        months = uni[c]
        for v, f in RULES.items():
            ent = a0 / f(ag)
            for r in ET.simulate_split(hh, ent, a0, I.K, exit_bars, 3.0, 0.5, True):
                mth = pd.Timestamp(r["ts"]).strftime("%Y-%m")
                if mth in months:
                    rows.append({"variant": v, "ticker": c, "half": AT.half(c), "month": mth, "band": months[mth], **r})
        if n % 50 == 0:
            print(f"  {n}/{len(coins)}", flush=True)
    tr = pd.DataFrame(rows)
    tr.to_csv(OUT.replace(".md", "_trades.csv"), index=False)
    write_report(tr, lto, thr)


def write_report(tr, lto, thr) -> None:
    pf = I.pf
    V = list(RULES)
    L = ["# Поріг входу залежно від віку сплеску", "",
         "Вік — бари від першого тригера T1 поточного епізоду сплеску (ціна ≥ 2 ATR, OI ≥ +2%, обсяг ≥ 2× за годину). Модель одна "
         f"(base, поріг {thr:.3f}); міняється лише правило входу, утримання й вихід — Alert base.", "",
         "fresh2 / fresh4 — вхід лише в перші 2 / 4 год сплеску; early — перша година поріг × 0.85, друга × 1.0, далі заборонено; "
         "ramp — × 0.85 / 1.0 / 1.15 / 1.3 (до 1 / 2 / 4 год / старше або без сплеску).", "",
         "**Критерій (до запуску).** Точність: ≥ 20 млн PF − base ≥ 0.03, ≥ 4 з 6 місяців, обидві половини, загальний ≥ base. "
         "Швидкість (early / ramp): затримка або пройдено руху менші за base, хибних ≤ 1.5 × base, PF ≥ 20 млн не нижчий. "
         "Орієнтир: split_start30 — ≥ 20 млн 0.96.", "",
         "## Leave-ticker-out на розмітці власника (рівень Кандидат)", "",
         "| вхід | спіймано | до старту | затримка, хв | пройдено руху | фальстартів | хибних / 30 дн |", "|---|---|---|---|---|---|---|"]
    for v in V:
        m = lto[v]
        L.append(f"| {v} | {m['caught']} | {m['pre_start']} | {m['delay_min']:.0f} | {m['done']:.0%} | {m['false_starts_triggered']} | {m['false_alarms_per_30d']:.1f} |")
    L += ["", f"## Бектест 6 міс., {tr['ticker'].nunique()} монет", "",
          "| вхід | угод | у плюсі | медіана | PF | без топ-3 | ≥ 20 млн | 5–20 млн | A ≥ 20 млн | B ≥ 20 млн |", "|---|---|---|---|---|---|---|---|---|---|"]
    for v in V:
        g = tr[tr["variant"] == v]
        r = g["ret"].sort_values()
        big = g["band"] == BIG
        L.append(f"| {v} | {len(r)} | {(r > 0).mean():.0%} | {r.median():+.2%} | {pf(r):.2f} | {pf(r.iloc[:-3]):.2f} | {pf(g[big]['ret']):.2f} | "
                 f"{pf(g[g['band'] == '5–20 млн']['ret']):.2f} | {pf(g[big & (g['half'] == 'A')]['ret']):.2f} | {pf(g[big & (g['half'] == 'B')]['ret']):.2f} |")
    piv = tr.groupby(["month", "variant"])["ret"].apply(pf).unstack().reindex(columns=V)
    piv20 = tr[tr["band"] == BIG].groupby(["month", "variant"])["ret"].apply(pf).unstack().reindex(columns=V)
    L += ["", "## По місяцях (PF; у дужках ≥ 20 млн)", "", "| місяць | " + " | ".join(V) + " |", "|---|" + "---|" * len(V)]
    for mth in piv.index:
        L.append(f"| {mth} | " + " | ".join(f"{piv.at[mth, v]:.2f} ({piv20.at[mth, v]:.2f})" for v in V) + " |")
    L += ["", "## Критерій", ""]
    big = tr[tr["band"] == BIG]
    b = big[big["variant"] == "base"]
    for v in V[1:]:
        g = big[big["variant"] == v]
        diff = pf(g["ret"]) - pf(b["ret"])
        months = int((piv20[v] > piv20["base"] + 1e-9).sum())
        halves = all(pf(g[g["half"] == hf]["ret"]) > pf(b[b["half"] == hf]["ret"]) for hf in ("A", "B"))
        total = pf(tr[tr["variant"] == v]["ret"]) >= pf(tr[tr["variant"] == "base"]["ret"])
        acc = diff >= 0.03 and months >= 4 and halves and total
        m, m0 = lto[v], lto["base"]
        faster = m["delay_min"] < m0["delay_min"] or m["done"] < m0["done"] - 1e-9
        speed = faster and m["false_alarms_per_30d"] <= 1.5 * m0["false_alarms_per_30d"] and diff >= 0
        L.append(f"- {v}: ≥ 20 млн {diff:+.3f}, місяців {months} з {len(piv20)}, половини {'так' if halves else 'ні'}, загальний ≥ base "
                 f"{'так' if total else 'ні'} → точність **{'ВИКОНАНО' if acc else 'НЕ виконано'}**; раніше {'так' if faster else 'ні'} "
                 f"(затримка {m['delay_min']:.0f} проти {m0['delay_min']:.0f} хв, пройдено {m['done']:.0%} проти {m0['done']:.0%}, хибних "
                 f"{m['false_alarms_per_30d']:.1f} проти {m0['false_alarms_per_30d']:.1f}) → швидкість **{'ВИКОНАНО' if speed else 'НЕ виконано'}**")
    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
