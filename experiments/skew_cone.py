"""
experiments/skew_cone.py — асиметричний конус: чи прогнозуються межі руху вгору і вниз окремо краще, ніж симетричним
конусом HAR, 2026-10-20.

Питання власника: «у пампах рух однобокий (виніс угору, потім злив) — як урахувати skew у моделі?» Конус volfc.py
симетричний: ±kσ від ціни. Тут прогнозуємо квантилі 5% / 25% / 75% / 95% лог-доходності за наступні 4 год і порівнюємо:
    S0  симетричний конус (як зараз): σ̂ з HAR-S, межі ±c·σ̂, c — з навчання (середнє |квантилів| обох боків);
    S0a стала асиметрія: межі = квантилі z = r / σ̂ з навчання (одна форма для всіх моментів) — опис;
    M1  semivariance HAR (Patton & Sheppard 2015): окремо σ̂⁺ і σ̂⁻ з дисперсій плюсових і мінусових 5m-доходностей — порівняння;
    M2  quantile LightGBM — ГОЛОВНА: квантилі z = r / σ̂ за ознаками фази руху (рух за 1 / 4 / 24 год у σ, відстань від хаю і лоу
        доби, реалізована асиметрія доби, частка руху вгору / вниз, волатильність відносно тижня, година доби);
    M3  емпіричні групи: квантилі z у 20 групах (рух за 4 год у σ × волатильність години відносно тижня) — порівняння.
Дані — 1m Binance з архіву (experiments/klines1m.py), 5m-доходності; точка — кожна закрита година; walk-forward по місяцях
(травень – жовтень) з ембарго 4 год, як experiments/har_vol.py. Метрики: pinball loss (сума 4 квантилів; менше — краще) і
покриття (частка r нижче кожного квантиля; має дорівнювати 5 / 25 / 75 / 95%).

КРИТЕРІЙ (записано до запуску, 20.10), для M2 проти S0:
    1) pinball loss менший в обох половинах монет (md5) і у ≥ 5 з 6 тестових місяців;
    2) покриття кожного з 4 квантилів — у межах ±2 п.п. від номіналу (усі точки);
    3) у «гарячі» години (остання година ≥ 3× звичайної тижневої) pinball loss менший щонайменше на 5%.
Пройде — асиметричний конус у дашборд (окремим рішенням). Ні — лишаємо симетричний. M1 / M3 — порівняння; якщо пройде лише
котрась із них — це гіпотеза, побачена після запуску.

    python experiments/skew_cone.py              # → reports/skew_cone.md
    python experiments/skew_cone.py --limit 30   # перевірка
"""

import argparse
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import klines1m as K

QS = (0.05, 0.25, 0.75, 0.95)
TEST_MONTHS = ["2026-05", "2026-06", "2026-07", "2026-08", "2026-09", "2026-10"]
EMBARGO = pd.Timedelta(hours=4)
MIN_FILL = 0.8
EPS = 1e-12
WIN = {"1": 12, "4": 48, "24": 288, "168": 2016}
HOT = np.log(9)                     # остання година ≥ 3× звичайної тижневої (у дисперсії ×9)
TRAIN_MAX = 400_000                 # підвибірка навчання LightGBM (швидкість)
SEED = 42
OUT_MD = os.path.join("reports", "skew_cone.md")
FEATS = ["z1", "z4", "z24", "dhi", "dlo", "rskew", "rsr24", "rsr4", "v1", "v4", "v24", "hour"]


def coin_rows(base: str) -> pd.DataFrame:
    m1 = K.load(base)
    if len(m1) < 20 * 1440:
        return pd.DataFrame()
    b5 = K.resample(m1, "5min")
    grid = pd.date_range(b5.index[0], b5.index[-1], freq="5min", tz="UTC")
    c, h, lo = (b5[k].reindex(grid) for k in ("c", "h", "l"))
    lc = np.log(c)
    r = lc.diff()
    sq, have = r ** 2, r.notna().astype(float)
    pos, neg, cub = sq.where(r > 0, 0.0).where(r.notna()), sq.where(r < 0, 0.0).where(r.notna()), r ** 3
    d = {}

    def past(s, n):                                   # середнє барів [T − n, T) → рядок T
        sm, k = s.rolling(n, min_periods=1).sum(), have.rolling(n, min_periods=1).sum()
        return (sm / k).where(k >= MIN_FILL * n).shift(1)

    def fut(s, n):                                    # середнє барів [T, T + n)
        sm = s[::-1].rolling(n, min_periods=1).sum()[::-1]
        k = have[::-1].rolling(n, min_periods=1).sum()[::-1]
        out = (sm / k).where(k >= MIN_FILL * n)
        out.iloc[len(out) - n + 1:] = np.nan
        return out

    for k, n in WIN.items():
        d["x" + k] = np.log(past(sq, n) + EPS)
        d["p" + k] = np.log(past(pos, n) + EPS)
        d["n" + k] = np.log(past(neg, n) + EPS)
    d["yv"], d["yp"], d["yn"] = np.log(fut(sq, 48) + EPS), np.log(fut(pos, 48) + EPS), np.log(fut(neg, 48) + EPS)
    px = lc.shift(1)                                  # ціна в момент T (закриття бару, що закінчився о T)
    d["r4"] = lc.shift(-47) - px                      # ціль: лог-доходність за наступні 4 год
    v24 = past(sq, 288)
    for k in (1, 4, 24):
        d[f"z{k}"] = (px - lc.shift(1 + 12 * k)) / np.sqrt(v24 * 12 * k)
    hi24, lo24 = h.rolling(288, min_periods=200).max().shift(1), lo.rolling(288, min_periods=200).min().shift(1)
    d["dhi"] = (px - np.log(hi24)) / np.sqrt(v24 * 12)
    d["dlo"] = (px - np.log(lo24)) / np.sqrt(v24 * 12)
    s3, s2, k3 = cub.rolling(288, min_periods=200).sum(), sq.rolling(288, min_periods=200).sum(), have.rolling(288, min_periods=200).sum()
    d["rskew"] = (np.sqrt(k3) * s3 / s2 ** 1.5).shift(1)
    turn = m1["v"].resample("1h").sum()
    d["turn30"] = (turn.rolling(720, min_periods=240).sum() / 30).shift(1).reindex(grid, method="ffill")
    df = pd.DataFrame(d, index=grid)
    df = df[df.index.minute == 0]
    df["rsr24"], df["rsr4"] = df["p24"] - df["n24"], df["p4"] - df["n4"]
    df["v1"], df["v4"], df["v24"] = df["x1"] - df["x168"], df["x4"] - df["x168"], df["x24"] - df["x168"]
    df["hour"] = df.index.hour
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["x1", "x4", "x24", "x168", "r4", "z24", "dhi", "rskew"])
    df["base"] = base
    return df.reset_index(names="ts").astype({c: "float32" for c in df.columns if c not in ("base", "hour")} | {})


def ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta, float(np.var(y - X @ beta))


def har_design(d, extra=()):
    hrs = d["hour"].to_numpy()
    cols = [np.ones(len(d))] + [d[c].to_numpy(dtype=float) for c in ("x1", "x4", "x24", "x168", *extra)]
    return np.column_stack(cols + [(hrs == k).astype(float) for k in range(1, 24)])


def pinball(y, q, p):
    e = y - p
    return float(np.mean(np.maximum(q * e, (q - 1) * e)))


def fold(tr: pd.DataFrame, te: pd.DataFrame) -> dict:
    """Прогнози квантилів r4 усіх моделей для тестового місяця."""
    import lightgbm as lgb
    out = {}
    # σ̂ — HAR-S на дисперсію наступних 4 год (як volfc.py)
    b, s2 = ols(har_design(tr), tr["yv"].to_numpy(dtype=float))
    sig_tr = np.sqrt(np.exp(har_design(tr) @ b + s2 / 2) * 48)
    sig_te = np.sqrt(np.exp(har_design(te) @ b + s2 / 2) * 48)
    z_tr = tr["r4"].to_numpy(dtype=float) / sig_tr
    qz = {q: np.quantile(z_tr, q) for q in QS}
    a = (abs(qz[0.05]) + qz[0.95]) / 2
    bb = (abs(qz[0.25]) + qz[0.75]) / 2
    out["S0"] = {0.05: -a * sig_te, 0.25: -bb * sig_te, 0.75: bb * sig_te, 0.95: a * sig_te}
    out["S0a"] = {q: qz[q] * sig_te for q in QS}
    # M1 — semivariance HAR: окремо σ̂⁺ і σ̂⁻
    ex = ("p1", "p4", "p24", "p168", "n1", "n4", "n24", "n168")
    m1 = {}
    for side, col in (("+", "yp"), ("-", "yn")):
        bs, ss = ols(har_design(tr, ex), tr[col].to_numpy(dtype=float))
        m1[side] = (np.sqrt(np.exp(har_design(tr, ex) @ bs + ss / 2) * 48), np.sqrt(np.exp(har_design(te, ex) @ bs + ss / 2) * 48))
    r_tr = tr["r4"].to_numpy(dtype=float)
    out["M1"] = {q: np.quantile(r_tr / m1["+" if q > 0.5 else "-"][0], q) * m1["+" if q > 0.5 else "-"][1] for q in QS}
    # M2 — quantile LightGBM на z за ознаками фази
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(tr), size=min(TRAIN_MAX, len(tr)), replace=False)
    Xtr, Xte = tr[FEATS].to_numpy(dtype=float)[idx], te[FEATS].to_numpy(dtype=float)
    preds = []
    for q in QS:
        m = lgb.LGBMRegressor(objective="quantile", alpha=q, n_estimators=200, learning_rate=0.05, num_leaves=31,
                              min_child_samples=500, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=SEED)
        m.fit(Xtr, z_tr[idx])
        preds.append(m.predict(Xte))
    P = np.sort(np.column_stack(preds), axis=1)               # без перетину квантилів
    out["M2"] = {q: P[:, i] * sig_te for i, q in enumerate(QS)}
    # M3 — емпіричні групи: рух за 4 год у σ × волатильність години відносно тижня
    def grp(d):
        zb = np.digitize(d["z4"].to_numpy(dtype=float), [-2, -0.5, 0.5, 2])
        vb = np.digitize(d["v1"].to_numpy(dtype=float), [0, np.log(4), HOT])
        return zb * 4 + vb
    g_tr, g_te = grp(tr), grp(te)
    out["M3"] = {}
    for q in QS:
        qq = pd.Series(z_tr).groupby(g_tr).quantile(q)
        cnt = pd.Series(z_tr).groupby(g_tr).size()
        qq = qq[cnt >= 500]
        out["M3"][q] = pd.Series(g_te).map(qq).fillna(qz[q]).to_numpy() * sig_te
    return out


def evaluate(y, pr: dict) -> dict:
    return {"pin": sum(pinball(y, q, pr[q]) for q in QS), **{f"cov{int(q * 100)}": float(np.mean(y < pr[q])) for q in QS}}


def half(base: str) -> str:
    return "A" if int(hashlib.md5(base.encode()).hexdigest(), 16) % 2 == 0 else "B"


def table(rows):
    cols = list(rows[0])
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
                     + ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows])


def main() -> None:
    ap = argparse.ArgumentParser(description="Асиметричний конус: квантилі руху за 4 год (quantile LightGBM, semivariance HAR, групи) "
                                             "проти симетричного HAR; walk-forward, pinball loss і покриття.")
    ap.add_argument("--limit", type=int, default=0, help="лише перші N монет (перевірка)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    bases = K.available()[: args.limit or None]
    parts = []
    for n, base in enumerate(bases, 1):
        parts.append(coin_rows(base))
        if n % 50 == 0:
            print(f"  {n}/{len(bases)} монет", flush=True)
    df = pd.concat([p for p in parts if len(p)], ignore_index=True)
    df["hour"] = df["hour"].astype(int)
    print(f"точок: {len(df)}, монет: {df['base'].nunique()}", flush=True)

    res = []
    for m in TEST_MONTHS:
        start = pd.Timestamp(m + "-01", tz="UTC")
        end = start + pd.offsets.MonthBegin(1)
        tr = df[df["ts"] < start - EMBARGO]
        te = df[(df["ts"] >= start) & (df["ts"] < end)].copy()
        if te.empty or len(tr) < 50_000:
            continue
        pr = fold(tr, te)
        for name, qd in pr.items():
            for q in QS:
                te[f"{name}_{q}"] = qd[q]
        te["month"] = m
        res.append(te)
        print(f"  {m}: навчання {len(tr)}, тест {len(te)}", flush=True)
    t = pd.concat(res, ignore_index=True)
    t["coins"] = t["base"].map(half)
    t["hot"] = t["v1"] >= HOT
    models = ["S0", "S0a", "M1", "M2", "M3"]

    def block(d):
        y = d["r4"].to_numpy(dtype=float)
        ev = {mm: evaluate(y, {q: d[f"{mm}_{q}"].to_numpy(dtype=float) for q in QS}) for mm in models}
        return ev

    def rows_for(slices):
        rows = []
        for name, d in slices:
            ev = block(d)
            r = {"зріз": name, "точок": len(d)}
            for mm in models:
                r[mm] = f"{ev[mm]['pin'] * 100:.4f}"
                if mm != "S0":
                    r[mm + " vs S0"] = f"{1 - ev[mm]['pin'] / ev['S0']['pin']:+.1%}"
            rows.append(r)
        return rows

    slices = [("усі", t)] + [(f"монети {h}", t[t["coins"] == h]) for h in "AB"] + \
             [("гарячі години", t[t["hot"]]), ("решта", t[~t["hot"]])] + \
             [(m, t[t["month"] == m]) for m in TEST_MONTHS if (t["month"] == m).any()]
    cov_rows = []
    for name, d in (("усі", t), ("гарячі години", t[t["hot"]])):
        ev = block(d)
        for mm in models:
            cov_rows.append({"зріз": name, "модель": mm, **{f"нижче q{int(q * 100)}": f"{ev[mm][f'cov{int(q * 100)}']:.1%}" for q in QS}})

    ev_all, ev_hot = block(t), block(t[t["hot"]])
    halves_ok = all(block(t[t["coins"] == h])["M2"]["pin"] < block(t[t["coins"] == h])["S0"]["pin"] for h in "AB")
    months = [m for m in TEST_MONTHS if (t["month"] == m).any()]
    month_win = sum(block(t[t["month"] == m])["M2"]["pin"] < block(t[t["month"] == m])["S0"]["pin"] for m in months)
    cov_ok = all(abs(ev_all["M2"][f"cov{int(q * 100)}"] - q) <= 0.02 for q in QS)
    hot_gain = 1 - ev_hot["M2"]["pin"] / ev_hot["S0"]["pin"]
    ok = halves_ok and month_win >= 5 and cov_ok and hot_gain >= 0.05

    md = ["# Асиметричний конус: квантилі руху за 4 год", "",
          f"{t['base'].nunique()} монет Binance, 1m-архів; точка — кожна закрита година; ціль — лог-доходність за наступні 4 год; "
          "walk-forward травень – жовтень. Pinball loss — сума 4 квантилів (5 / 25 / 75 / 95%), ×100; менше — краще. "
          "S0 — симетричний конус (як у дашборді), S0a — стала асиметрія, M1 — semivariance HAR, M2 — quantile LightGBM (головна), "
          "M3 — емпіричні групи.", "",
          "## Pinball loss", "", table(rows_for(slices)), "",
          "## Покриття (частка фактичних r нижче межі; має бути 5 / 25 / 75 / 95%)", "", table(cov_rows), "",
          "## Критерій (записано до запуску, M2 проти S0)", "",
          f"- обидві половини монет краще: {'так' if halves_ok else 'НІ'}",
          f"- місяців краще: {month_win} з {len(months)} (потрібно ≥ 5)",
          f"- покриття M2 в межах ±2 п.п.: {'так' if cov_ok else 'НІ'} ("
          + ", ".join(f"q{int(q * 100)} {ev_all['M2'][f'cov{int(q * 100)}']:.1%}" for q in QS) + ")",
          f"- гарячі години: {hot_gain:+.1%} (потрібно ≥ +5%)", "",
          f"**Висновок за критерієм: {'проходить' if ok else 'НЕ проходить'}.**", ""]
    os.makedirs("reports", exist_ok=True)
    open(OUT_MD, "w", encoding="utf-8").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
