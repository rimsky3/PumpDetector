"""
experiments/fe_anchor.py — ОЗНАКИ, ПРИВ'ЯЗАНІ ДО СТАРТУ РУХУ, і ще три дешеві групи (2026-10-11; власник: «пошукай і подумай про
feature engineering з тих ознак, що маємо», «окей давай»).
Ідея (anchored features / причинні наближення privileged information): teacher (label_teacher.py) показав, за чим власник
відрізняє імпульс — приріст OI за рух 31%, кількість сплесків у русі 8%, рух за добу ДО старту 8%, розмір руху 7%, пік обсягу 7%,
діапазон до старту 4%, відкат усередині руху 3%. Teacher бачить рух цілком; тут ті самі величини «на цей момент» від ймовірного
старту поточного руху. Наші ознаки рахуються у фіксованих вікнах (1 / 4 / 24 год), які змішують те, що було ДО руху, з рухом.
Точка відліку (anchor) — найнижчий low за останні 12 год (48 барів) включно з поточним; лише минуле.
Групи (зафіксовано ДО запуску):
  anchor  bars_since_anchor, move_anchor_pct, move_anchor_atr, oi_since_anchor (log OI USD), oi_coin_since_anchor (log OI монети),
          oi_per_move (монетний OI на одиницю log-руху ціни; рух < 0.5% — NaN), n_trig_since_anchor (бари тригера T1),
          pre_ret_anchor (рух за 24 год до anchor), pre_range_anchor (log діапазону 96 год до anchor), vol_peak_anchor,
          vol_mean_anchor (обсяг бару від anchor ÷ медіана 7 днів), dd_anchor (найглибший відкат від anchor ÷ рух), eff_anchor
          (Кауфман від anchor)
  voln    нормування на власну волатильність (ReVol): oi_chg_1h_v / 4h_v / 24h_v (зміна log OI USD ÷ ст. відх. такої зміни за 7 днів),
          ret_1h_v, range_1h_v (діапазон 1 год ÷ медіана 7 днів), funding_chg_v (зміна funding за добу ÷ ст. відх. за 7 днів)
  oiq     режим «OI + ціна» (власник: ціна ↑ + OI ↑ — нові лонги, ціна ↑ + OI ↓ — закриття шортів): oi_ret_ratio_1h / 4h
          (зміна log OI монети ÷ |зміна log ціни|, рух < 0.2% — NaN, обрізано ±10), oi_quad_1h / 4h (знак ціни + 2 × знак OI)
  liq     log_turnover_30d — log середнього денного обороту за 30 днів (до вчора); ліквідні монети частіше продовжують рух,
          неліквідні — розвертаються (Zaremba et al. 2021, Fičura 2023)
Варіанти: base, +anchor, +voln, +oiq, +liq, +all; random seeds 42 / 43 / 44 (кожен варіант × seed — своє навчання і поріг).
КРИТЕРІЙ (до запуску), проти base у парах за seed: ≥ 20 млн середня різниця ≥ +0.02 і краще на ≥ 2 з 3; середній загальний PF ≥ base;
обидві половини ≥ base; і частка feature importance групи ≥ 3% (плюс від ознак, якими модель не користується, — шум).
→ reports/fe_anchor.md, reports/fe_anchor_trades.csv

    python experiments/fe_anchor.py
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
from numpy.lib.stride_tricks import sliding_window_view

import auto_label as AL
import auto_train as AT
import early_target as ET
import exit_partial as P
import idio_features as I
import range_features as R
import train
from risk import atr
from tiers import tier_frame

W = 48                     # 12 год 15m-барів — вікно пошуку anchor
D1, W7, D30 = 96, 7 * 96, 30
GROUPS = {
    "anchor": ["bars_since_anchor", "move_anchor_pct", "move_anchor_atr", "oi_since_anchor", "oi_coin_since_anchor", "oi_per_move",
               "n_trig_since_anchor", "pre_ret_anchor", "pre_range_anchor", "vol_peak_anchor", "vol_mean_anchor", "dd_anchor", "eff_anchor"],
    "voln": ["oi_chg_1h_v", "oi_chg_4h_v", "oi_chg_24h_v", "ret_1h_v", "range_1h_v", "funding_chg_v"],
    "oiq": ["oi_ret_ratio_1h", "oi_ret_ratio_4h", "oi_quad_1h", "oi_quad_4h"],
    "liq": ["log_turnover_30d"],
}
ALL = [f for fs in GROUPS.values() for f in fs]
SEEDS = [42, 43, 44]
VARIANTS = {"base": ET.COLS, **{f"+{g}": ET.COLS + fs for g, fs in GROUPS.items()}, "+all": ET.COLS + ALL}
BIG = "≥ 20 млн"
OUT = os.path.join("reports", "fe_anchor.md")


def features(h: pd.DataFrame, oi_coin_log: pd.Series, funding: pd.Series) -> pd.DataFrame:
    """15m-ряд (high, low, price, volume у $) + log OI монети + funding → нові ознаки на кожному барі (лише минуле)."""
    n = len(h)
    hi, lo, c, v = (h[k].to_numpy(dtype=float) for k in ("high", "low", "price", "volume"))
    lc = np.log(np.where(c > 0, c, np.nan))
    oic = oi_coin_log.reindex(h.index).to_numpy(dtype=float)
    oiu = oic + lc
    a_ = atr(h.rename(columns={})).to_numpy()
    vmed = pd.Series(v, index=h.index).rolling(W7, min_periods=D1).median().shift(1).to_numpy()
    s = pd.DataFrame({"high": hi, "low": lo, "price": c, "volume": v, "oi": np.exp(oiu), "atr": a_, "vol_med": vmed}, index=h.index)
    trig = AL.trigger_mask(s, *AL.TRIGGERS["T1"]).astype(float)
    f = pd.DataFrame(index=h.index)
    if n > W:
        lw = sliding_window_view(np.where(np.isnan(lo), np.inf, lo), W)          # вікно, що закінчується на барі t = i + W − 1
        rel = lw.argmin(axis=1)                                                  # позиція anchor у вікні
        t = np.arange(W - 1, n)
        a = t - (W - 1) + rel
        bars = t - a
        cs = lambda x: np.concatenate([[0.0], np.nancumsum(x)])                  # noqa: E731
        ct, cv = cs(trig), cs(v)
        dc = np.abs(np.diff(c, prepend=np.nan))
        cdc = cs(np.nan_to_num(dc))
        out = {k: np.full(n, np.nan) for k in GROUPS["anchor"]}
        low_a = lo[a]
        move = c[t] - low_a
        out["bars_since_anchor"][t] = bars
        out["move_anchor_pct"][t] = c[t] / low_a - 1
        atr_a = a_[np.maximum(a - 1, 0)]
        out["move_anchor_atr"][t] = move / np.where(atr_a > 0, atr_a, np.nan)
        out["oi_since_anchor"][t] = oiu[t] - oiu[a]
        out["oi_coin_since_anchor"][t] = oic[t] - oic[a]
        lm = np.log(c[t] / low_a)
        out["oi_per_move"][t] = np.where(lm >= 0.005, (oic[t] - oic[a]) / np.where(lm > 0, lm, np.nan), np.nan)
        out["n_trig_since_anchor"][t] = ct[t + 1] - ct[a + 1]
        pa = np.where(a >= D1, a - D1, 0)
        out["pre_ret_anchor"][t] = np.where(a >= D1, c[a] / c[pa] - 1, np.nan)
        rmax = pd.Series(hi).rolling(384, min_periods=192).max().shift(1).to_numpy()
        rmin = pd.Series(lo).rolling(384, min_periods=192).min().shift(1).to_numpy()
        out["pre_range_anchor"][t] = np.log(rmax[a] / rmin[a])
        vm_a = vmed[a]
        cnt = np.maximum(bars, 1)
        out["vol_mean_anchor"][t] = np.where(bars > 0, (cv[t + 1] - cv[a + 1]) / cnt, np.nan) / np.where(vm_a > 0, vm_a, np.nan)
        hw, vw, low_w = sliding_window_view(hi, W), sliding_window_view(v, W), sliding_window_view(lo, W)
        after = np.arange(W)[None, :] > rel[:, None]                             # позиції після anchor
        out["vol_peak_anchor"][t] = np.where(after.any(1), np.where(after, vw, -np.inf).max(1), np.nan) / np.where(vm_a > 0, vm_a, np.nan)
        from_a = np.arange(W)[None, :] >= rel[:, None]
        runmax = np.maximum.accumulate(np.where(from_a, hw, -np.inf), axis=1)
        pull = np.where(after, runmax - low_w, -np.inf).max(1)
        rng = runmax[:, -1] - low_a
        out["dd_anchor"][t] = np.where((rng > 0) & after.any(1), pull / np.where(rng > 0, rng, np.nan), np.nan)
        path = cdc[t + 1] - cdc[a + 1]
        out["eff_anchor"][t] = np.where(path > 0, (c[t] - c[a]) / np.where(path > 0, path, np.nan), np.nan)
        for k, arr in out.items():
            f[k] = arr
    S = lambda x: pd.Series(x, index=h.index)                                    # noqa: E731
    sd = lambda x: x.rolling(W7, min_periods=D1).std().shift(1)                   # noqa: E731
    for name, k in (("1h", 4), ("4h", 16), ("24h", 96)):
        d = S(oiu).diff(k)
        f[f"oi_chg_{name}_v"] = d / sd(d)
    r1 = S(lc).diff(4)
    f["ret_1h_v"] = r1 / sd(r1)
    rg = np.log(S(hi).rolling(4).max() / S(lo).rolling(4).min())
    f["range_1h_v"] = rg / rg.rolling(W7, min_periods=D1).median().shift(1)
    fc = funding.reindex(h.index).astype(float).diff(D1)
    f["funding_chg_v"] = fc / sd(fc)
    for name, k in (("1h", 4), ("4h", 16)):
        dp, do = S(lc).diff(k), S(oic).diff(k)
        f[f"oi_ret_ratio_{name}"] = (do / dp.abs().where(dp.abs() >= 0.002)).clip(-10, 10)
        f[f"oi_quad_{name}"] = np.sign(dp) + 2 * np.sign(do)
    daily = S(v).resample("1D").sum()
    t30 = np.log(daily.rolling(D30, min_periods=7).mean().shift(1).where(lambda x: x > 0))
    f["log_turnover_30d"] = t30.reindex(h.index, method="ffill")
    return f[ALL].replace([np.inf, -np.inf], np.nan)


def lgbm(seed: int):
    m = train.make_lightgbm()
    m.set_params(random_state=seed)
    return m


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    argparse.ArgumentParser(description="Ознаки, прив'язані до старту руху (anchored), нормування на волатильність, режим OI + ціна, ліквідність: LTO і бектест 6 міс., 3 seeds").parse_args()
    d, pumps, px, idx, uni = ET.load_data()
    trained = set(d["base_asset"])
    parts = []
    for b, g in d.groupby("base_asset"):
        p = px[b]
        fund = g.set_index("ts")["funding"]
        parts.append(features(p, p["oi_coin_log_idx"], fund).reindex(g["ts"]).set_axis(g.index))
    d = d.join(pd.concat(parts))
    cov = {g: float(d[fs].notna().any(axis=1).mean()) for g, fs in GROUPS.items()}
    print(f"навчання: {len(d)} барів; покриття: " + ", ".join(f"{g} {c:.0%}" for g, c in cov.items()), flush=True)
    y, grp = d["label"].to_numpy().astype(int), d["base_asset"]
    models, thr, lto, imp = {}, {}, {}, {}
    for v, cols in VARIANTS.items():
        for sd in SEEDS:
            key = f"{v}_s{sd}"
            p = train.oof_proba(lambda s=sd: lgbm(s), d[cols], pd.Series(y), grp)
            thr[key] = train.best_f1_threshold(y, p)
            sig = p >= thr[key]
            lto[key] = {**train.event_metrics(d, sig, pumps, ET.BAR), **ET.early(d, sig, pumps, px)}
            models[key] = lgbm(sd).fit(d[cols], y)
            g = pd.Series(models[key].booster_.feature_importance("gain"), index=cols)
            imp[key] = g / g.sum()
            print(f"  {key} (поріг {thr[key]:.3f}): {lto[key]}", flush=True)

    m1 = joblib.load(train.model_path("lightgbm"))
    m15 = joblib.load(train.model_path(train.M15_NAME))
    exit_bars, share = m15.get("exit_bars", 4), m15.get("confirm", {}).get("share", 0.75)
    start, end = pd.Timestamp(I.START, tz="UTC"), pd.Timestamp(I.END, tz="UTC") + pd.Timedelta(days=1)
    coins = sorted(c for c in uni if c not in trained and os.path.exists(os.path.join(I.DIR, f"{c}.pkl")))
    print(f"бектест: {len(coins)} монет…", flush=True)
    rows = []
    for n, c in enumerate(coins, 1):
        x = pickle.load(open(os.path.join(I.DIR, f"{c}.pkl"), "rb"))
        f15, h1, h = x["f15"], x["h1"], x["h"]
        f15 = f15.join(I.new_features(h, idx).reindex(f15["ts"]).set_axis(f15.index))
        fe = features(h, R.archive_oi_log(c, h.index), f15.set_index("ts")["funding"])
        f15 = f15.join(fe.reindex(f15["ts"]).set_axis(f15.index))
        hh = h[(h.index >= start - pd.Timedelta(days=1)) & (h.index < end)]
        s1 = h1[["base_asset", "ts"]].assign(probability=m1["model"].predict_proba(h1[m1["features"]])[:, 1], threshold=m1["threshold"])
        for key, model in models.items():
            v = key.rsplit("_s", 1)[0]
            s15 = f15[["base_asset", "ts"]].assign(probability=model.predict_proba(f15[VARIANTS[v]])[:, 1], threshold=thr[key])
            t = tier_frame(s15, s1, share).set_index("ts").sort_index()
            for r in P.simulate(hh, t.loc[(t.index >= start) & (t.index < end), "alert_score"], I.K, exit_bars, 3.0, 0.5, True):
                mth = pd.Timestamp(r["ts"]).strftime("%Y-%m")
                if mth in uni[c]:
                    rows.append({"variant": v, "seed": int(key.rsplit("_s", 1)[1]), "ticker": c, "half": AT.half(c), "month": mth,
                                 "band": uni[c][mth], **r})
        if n % 50 == 0:
            print(f"  {n}/{len(coins)}", flush=True)
    tr = pd.DataFrame(rows)
    tr.to_csv(OUT.replace(".md", "_trades.csv"), index=False)
    write_report(tr, lto, imp, cov)


def write_report(tr, lto, imp, cov) -> None:
    pf = I.pf
    V = list(VARIANTS)
    L = ["# Ознаки, прив'язані до старту руху, і ще три групи", "",
         "Групи: " + "; ".join(f"**{g}** ({len(fs)})" for g, fs in GROUPS.items()) + ". Покриття навчальних барів: " +
         ", ".join(f"{g} {c:.0%}" for g, c in cov.items()) + f". Random seeds {SEEDS}.", "",
         "**Критерій (до запуску):** проти base у парах за seed — ≥ 20 млн середня різниця ≥ +0.02 і краще на ≥ 2 з 3; середній загальний "
         "PF ≥ base; обидві половини ≥ base; feature importance групи ≥ 3%.", "",
         "## Leave-ticker-out (Кандидат 15m), середнє за seeds", "",
         "| варіант | спіймано | затримка, хв | пройдено руху | фальстартів | хибних / 30 дн | частка gain нових ознак |", "|---|---|---|---|---|---|---|"]
    for v in V:
        ks = [f"{v}_s{sd}" for sd in SEEDS]
        new = [f for f in VARIANTS[v] if f in ALL]
        share = np.mean([imp[k][new].sum() for k in ks]) if new else 0.0
        L.append(f"| {v} | {np.mean([int(lto[k]['caught'].split('/')[0]) for k in ks]):.0f}/{lto[ks[0]]['caught'].split('/')[1]} | "
                 f"{np.mean([lto[k]['delay_min'] for k in ks]):.0f} | {np.mean([lto[k]['done'] for k in ks]):.0%} | "
                 f"{np.mean([int(lto[k]['false_starts_triggered'].split('/')[0]) for k in ks]):.0f}/{lto[ks[0]]['false_starts_triggered'].split('/')[1]} | "
                 f"{np.mean([lto[k]['false_alarms_per_30d'] for k in ks]):.1f} | {share:.1%} |")
    allimp = pd.concat([imp[f"+all_s{sd}"] for sd in SEEDS], axis=1).mean(axis=1).sort_values(ascending=False)
    L += ["", "Топ-12 ознак моделі +all (середнє за seeds): " + ", ".join(f"{k} {allimp[k]:.1%}" for k in allimp.index[:12]) + ".", ""]
    cell = {}
    for v in V:
        for sd in SEEDS:
            g = tr[(tr["variant"] == v) & (tr["seed"] == sd)]
            b = g[g["band"] == BIG]
            cell[(v, sd)] = {"pf": pf(g["ret"]), "pf20": pf(b["ret"]), "A": pf(b[b["half"] == "A"]["ret"]), "B": pf(b[b["half"] == "B"]["ret"]), "n": len(g)}
    L += ["## Бектест 6 міс. по seeds (PF; ≥ 20 млн у дужках)", "",
          "| варіант | " + " | ".join(f"seed {s}" for s in SEEDS) + " | середнє | ≥ 20 млн | різниця з base (≥ 20 млн) | A ≥ 20 млн | B ≥ 20 млн | угод |",
          "|---|" + "---|" * (len(SEEDS) + 6)]
    for v in V:
        c = [cell[(v, sd)] for sd in SEEDS]
        diffs = [cell[(v, sd)]["pf20"] - cell[("base", sd)]["pf20"] for sd in SEEDS]
        L.append(f"| {v} | " + " | ".join(f"{x['pf']:.2f} ({x['pf20']:.2f})" for x in c) + f" | {np.mean([x['pf'] for x in c]):.3f} | "
                 f"{np.mean([x['pf20'] for x in c]):.3f} | " + ("—" if v == "base" else " / ".join(f"{q:+.3f}" for q in diffs)) +
                 f" | {np.mean([x['A'] for x in c]):.3f} | {np.mean([x['B'] for x in c]):.3f} | {np.mean([x['n'] for x in c]):.0f} |")
    L += ["", "## Критерій", ""]
    for v in V[1:]:
        diffs = np.array([cell[(v, sd)]["pf20"] - cell[("base", sd)]["pf20"] for sd in SEEDS])
        tot = np.mean([cell[(v, sd)]["pf"] for sd in SEEDS]) >= np.mean([cell[("base", sd)]["pf"] for sd in SEEDS])
        halves = all(np.mean([cell[(v, sd)][hf] for sd in SEEDS]) >= np.mean([cell[("base", sd)][hf] for sd in SEEDS]) for hf in ("A", "B"))
        new = [f for f in VARIANTS[v] if f in ALL]
        share = np.mean([imp[f"{v}_s{sd}"][new].sum() for sd in SEEDS])
        ok = diffs.mean() >= 0.02 and (diffs > 0).sum() >= 2 and tot and halves and share >= 0.03
        L.append(f"- {v}: ≥ 20 млн {diffs.mean():+.3f} (краще на {(diffs > 0).sum()} з {len(SEEDS)}); загальний ≥ base {'так' if tot else 'ні'}; "
                 f"половини {'так' if halves else 'ні'}; gain нових {share:.1%} → **{'ВИКОНАНО' if ok else 'НЕ виконано'}**")
    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
