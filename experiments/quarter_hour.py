"""
experiments/quarter_hour.py — ОЗНАКИ «ВІДКРИТТЯ 15-ХВИЛИНКИ» (2026-10-10, огляд джерел за тегом «точність», власник: «давай по черзі»).
Джерело: Kim & Hansen, «The Quarter-Hour Effect» (arXiv 2607.09426): на Binance-перпах алгоритми дають сплески обсягу на межах
1 / 5 / 15 хв; дисбаланс ордерів НА ВІДКРИТТІ 15-хвилинки прогнозує дохідність на 4–12 год (у статті — 6 контрактів, імовірно
великі монети). У нас buy_ratio_bar / 1h / 4h — частка покупок за ВЕСЬ бар; окремо «перша хвилина» не дивились.
Дані — 1m-свічки Binance з архіву (experiments/klines1m.py: місячні + денні файли жовтня), v — оборот, bv — покупки-агресори.
Ознаки (зафіксовано ДО запуску; усі відомі на закритті свого 15m-бару; «перша хвилина» — хвилина :00 / :15 / :30 / :45):
    qh_buy_open      частка покупок у першій хвилині бару
    qh_buy_open_4    те саме за 4 останні бари (сума покупок ÷ сума обороту перших хвилин, 1 год)
    qh_buy_open_16   те саме за 16 барів (4 год)
    qh_open_gap      qh_buy_open − частка покупок за весь бар (чим відкриття відрізняється від решти бару)
    qh_open_vol      частка обороту 4 останніх барів, що припала на їхні перші хвилини (рівномірно було б 1/15)
    qh_open_ret_4    сума дохідностей перших хвилин 4 останніх барів ÷ (ст. відх. хвилинної дохідності за 24 год × 2)
Варіанти (обидва свіжонавчені однаково, розмітка власника, ознаки робочої моделі):
    base       MODEL_FEATURES_15M + власний рух
    base_s43   те саме з іншим зерном — МІРА ШУМУ між запусками (поріг свій)
    qh         base + 6 ознак
LTO (Кандидат 15m, поріг — найкращий F1 на OOF) і бектест 6 міс. на чесному списку (Alert = 15m + робоча годинна, частковий
вихід, стоп I.K), половини монет A / B.
КРИТЕРІЙ (до запуску): на ≥ 20 млн PF qh − base ≥ 0.03 і більше, ніж |base_s43 − base|; кращий за base у ≥ 4 з 6 місяців;
кращий на обох половинах монет; загальний PF ≥ base.
→ reports/quarter_hour.md, reports/quarter_hour_trades.csv

    python experiments/klines1m.py                       # 1m-свічки (докачує відсутнє)
    python experiments/quarter_hour.py
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
import exit_partial as P
import idio_features as I
import klines1m as K
import train
from tiers import tier_frame

QH = ["qh_buy_open", "qh_buy_open_4", "qh_buy_open_16", "qh_open_gap", "qh_open_vol", "qh_open_ret_4"]
VARIANTS = {"base": (ET.COLS, 42), "base_s43": (ET.COLS, 43), "qh": (ET.COLS + QH, 42)}
BIG = "≥ 20 млн"
OUT = os.path.join("reports", "quarter_hour.md")


def qh_features(m: pd.DataFrame) -> pd.DataFrame:
    """1m-свічки монети → ознаки на 15m-барах (індекс — відкриття бару)."""
    if m.empty:
        return pd.DataFrame(columns=QH)
    first = m[m.index.minute % 15 == 0]
    bar = K.resample(m, "15min").reindex(first.index)
    f = pd.DataFrame(index=first.index)
    f["qh_buy_open"] = (first["bv"] / first["v"]).where(first["v"] > 0)
    for n in (4, 16):
        f[f"qh_buy_open_{n}"] = first["bv"].rolling(n, min_periods=n).sum() / first["v"].rolling(n, min_periods=n).sum()
    f["qh_open_gap"] = f["qh_buy_open"] - (bar["bv"] / bar["v"]).where(bar["v"] > 0)
    f["qh_open_vol"] = first["v"].rolling(4, min_periods=4).sum() / bar["v"].rolling(4, min_periods=4).sum()
    r1 = np.log(m["c"].where(m["c"] > 0)).diff()
    sd = r1.rolling(1440, min_periods=720).std().reindex(first.index)
    ro = np.log(first["c"] / first["o"]).where((first["o"] > 0) & (first["c"] > 0))
    f["qh_open_ret_4"] = ro.rolling(4, min_periods=4).sum() / (2 * sd)
    return f.replace([np.inf, -np.inf], np.nan)


def lgbm(seed: int):
    m = train.make_lightgbm()
    m.set_params(random_state=seed)
    return m


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    argparse.ArgumentParser(description="Ознаки «відкриття 15-хвилинки» (частка покупок у першій хвилині бару) у 15m-моделі: LTO і бектест 6 міс.").parse_args()
    d, pumps, px, idx, uni = ET.load_data()
    trained = set(d["base_asset"])
    parts = []
    for b, g in d.groupby("base_asset"):
        parts.append(qh_features(K.load(b)).reindex(g["ts"]).set_axis(g.index))
    d = d.join(pd.concat(parts))
    cov = d[QH[0]].notna().mean()
    print(f"навчання: {len(d)} барів, {d['base_asset'].nunique()} тікерів; покриття 1m-ознаками {cov:.1%}", flush=True)

    y, grp = d["label"].to_numpy().astype(int), d["base_asset"]
    thr, lto, models = {}, {}, {}
    for v, (cols, seed) in VARIANTS.items():
        p = train.oof_proba(lambda s=seed: lgbm(s), d[cols], pd.Series(y), grp)
        thr[v] = train.best_f1_threshold(y, p)
        sig = p >= thr[v]
        lto[v] = {**train.event_metrics(d, sig, pumps, ET.BAR), **ET.early(d, sig, pumps, px)}
        models[v] = lgbm(seed).fit(d[cols], y)
        print(f"  {v} (поріг {thr[v]:.3f}): {lto[v]}", flush=True)
    g = pd.Series(models["qh"].booster_.feature_importance("gain"), index=VARIANTS["qh"][0])
    imp = (g / g.sum()).sort_values(ascending=False)

    m1 = joblib.load(train.model_path("lightgbm"))
    m15 = joblib.load(train.model_path(train.M15_NAME))
    exit_bars, share = m15.get("exit_bars", 4), m15.get("confirm", {}).get("share", 0.75)
    start, end = pd.Timestamp(I.START, tz="UTC"), pd.Timestamp(I.END, tz="UTC") + pd.Timedelta(days=1)
    coins = sorted(c for c in uni if c not in trained and os.path.exists(os.path.join(I.DIR, f"{c}.pkl")) and os.path.exists(K.path(c)))
    print(f"бектест: {len(coins)} монет чесного списку з 1m-свічками…", flush=True)
    rows = []
    for n, c in enumerate(coins, 1):
        x = pickle.load(open(os.path.join(I.DIR, f"{c}.pkl"), "rb"))
        f15, h1, h = x["f15"], x["h1"], x["h"]
        f15 = f15.join(I.new_features(h, idx).reindex(f15["ts"]).set_axis(f15.index))
        f15 = f15.join(qh_features(K.load(c)).reindex(f15["ts"]).set_axis(f15.index))
        hh = h[(h.index >= start - pd.Timedelta(days=1)) & (h.index < end)]
        s1 = h1[["base_asset", "ts"]].assign(probability=m1["model"].predict_proba(h1[m1["features"]])[:, 1], threshold=m1["threshold"])
        for v, (cols, _) in VARIANTS.items():
            s15 = f15[["base_asset", "ts"]].assign(probability=models[v].predict_proba(f15[cols])[:, 1], threshold=thr[v])
            t = tier_frame(s15, s1, share).set_index("ts").sort_index()
            for r in P.simulate(hh, t.loc[(t.index >= start) & (t.index < end), "alert_score"], I.K, exit_bars, 3.0, 0.5, True):
                mth = pd.Timestamp(r["ts"]).strftime("%Y-%m")
                if mth in uni[c]:
                    rows.append({"variant": v, "ticker": c, "half": AT.half(c), "month": mth, "band": uni[c][mth], **r})
        if n % 50 == 0:
            print(f"  {n}/{len(coins)}", flush=True)
    tr = pd.DataFrame(rows)
    tr.to_csv(OUT.replace(".md", "_trades.csv"), index=False)
    write_report(tr, lto, thr, imp, cov)


def write_report(tr, lto, thr, imp, cov) -> None:
    pf = I.pf
    V = list(VARIANTS)
    L = ["# Ознаки «відкриття 15-хвилинки»", "",
         "Частка покупок-агресорів у першій хвилині 15m-бару та похідні (Kim & Hansen, «The Quarter-Hour Effect»). "
         f"Покриття навчальних барів 1m-даними {cov:.1%}. Пороги: " + ", ".join(f"{k} {v:.3f}" for k, v in thr.items()) + ". "
         "base_s43 — ті самі ознаки, інше зерно: міра шуму.", "",
         "**Критерій (до запуску):** ≥ 20 млн PF qh − base ≥ 0.03 і більше за |base_s43 − base|; кращий у ≥ 4 з 6 місяців; "
         "кращий на обох половинах; загальний PF ≥ base.", "",
         "## Leave-ticker-out (Кандидат 15m)", "",
         "| варіант | спіймано | затримка, хв | пройдено руху | фальстартів | хибних / 30 дн |", "|---|---|---|---|---|---|"]
    for v in V:
        m = lto[v]
        L.append(f"| {v} | {m['caught']} | {m['delay_min']:.0f} | {m['done']:.0%} | {m['false_starts_triggered']} | {m['false_alarms_per_30d']:.1f} |")
    L += ["", "## Важливість нових ознак (частка gain у моделі qh)", "", "| ознака | частка | місце |", "|---|---|---|"]
    rank = {k: i + 1 for i, k in enumerate(imp.index)}
    L += [f"| {k} | {imp[k]:.1%} | {rank[k]} з {len(imp)} |" for k in QH]
    L += [f"| **разом** | **{imp[QH].sum():.1%}** | |", "",
          f"## Бектест 6 міс., {tr['ticker'].nunique()} монет", "",
          "| варіант | угод | у плюсі | медіана | PF | без топ-3 | ≥ 20 млн | 5–20 млн | A ≥ 20 млн | B ≥ 20 млн |", "|---|---|---|---|---|---|---|---|---|---|"]
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
    big = tr[tr["band"] == BIG]
    p20 = {v: pf(big[big["variant"] == v]["ret"]) for v in V}
    diff, noise = p20["qh"] - p20["base"], abs(p20["base_s43"] - p20["base"])
    months = int((piv20["qh"] > piv20["base"] + 1e-9).sum())
    halves = all(pf(big[(big["variant"] == "qh") & (big["half"] == hf)]["ret"]) > pf(big[(big["variant"] == "base") & (big["half"] == hf)]["ret"])
                 for hf in ("A", "B"))
    total = pf(tr[tr["variant"] == "qh"]["ret"]) >= pf(tr[tr["variant"] == "base"]["ret"])
    ok = diff >= 0.03 and diff > noise and months >= 4 and halves and total
    L += ["", "## Критерій", "",
          f"- ≥ 20 млн: qh − base = {diff:+.3f} (шум між зернами {noise:.3f}); місяців краще {months} з {len(piv20)}; обидві половини "
          f"{'так' if halves else 'ні'}; загальний ≥ base {'так' if total else 'ні'} → **{'ВИКОНАНО' if ok else 'НЕ виконано'}**"]
    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
