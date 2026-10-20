"""
experiments/alert_decomp.py — ЩО НЕСЕ СИГНАЛ МОДЕЛІ: РОЗМІР РУХУ ЧИ НАПРЯМОК (2026-10-20; ідея з Anatolyev & Gospodinov,
«Modeling Financial Return Dynamics via Decomposition», notes/books/ideas/anatolyev_gospodinov_decomposition.md).

Дохідність = розмір × знак. Розмір (волатильність) прогнозується добре (HAR, volfc.py), знак — погано; навіть «вгадування
знаку» часто приходить від волатильності. Питання: Alert / вхід моделі каже «тут буде рух» (розмір) чи «ціна піде вгору»
(напрямок)? HAR — контроль: дивимось, що сигнал додає ПОНАД прогноз розмаху.

Дані: готові models/ (15m, «старт», годинна), монети — чесний список 6 міс. (data/long_bin) без тікерів навчання, розмітки й
виключених, як у p_deciles_delay.py; 5m-свічки з 1m-архіву (data/klines1m). Точка T — закриття кожного 15m-бару.
    групи: Calm (s15 < 1; 25% випадкових — пам'ять), Candidate (s15 ≥ 1, Alert ні), Alert (alert_score ≥ 1),
           Entry — вхід split: Alert моделі «старт», якого не було 4 попередні бари (початок епізоду);
    σ̂ — прогноз розмаху на 4 год: HAR з коефіцієнтами experiments/har_vol.py (усі дані; для порівняння груп між собою
        спільний рівень σ̂ не важливий — однаковий для всіх груп);
    розмір: move_x = фактичний рух за 4 год (√ суми квадратів 5m-доходностей) ÷ σ̂;
    напрямок: частка r4 > 0; бар'єр ±kσ̂ (k = 0.5, 1): яка межа торкнулась першою за 4 год (5m high / low; обидві в одному
              барі — нічия, не рахується).

КРИТЕРІЇ (записано до запуску, 20.10), монети ≥ 20 млн окремо — опис:
    РОЗМІР: медіана move_x у Alert ÷ медіана в Calm ≥ 1.10 в обох половинах монет (md5) і обох половинах часу
            (квіт–черв / лип–вер) → «Alert несе інформацію про розмах понад HAR».
    НАПРЯМОК: частка «+1σ першим» у Entry мінус у Calm ≥ +3 п.п., 95% bootstrap-інтервал по монетах вище нуля,
            і той самий знак (≥ 0) в обох половинах монет і часу → «вхід моделі несе інформацію про напрямок».
Висновок — матриця: лише розмір → Alert = «тут зараз буде рух», напрям — з читання стакану власником (конус ширший);
лише напрямок або обидва — сигнал справді напрямковий.

    python experiments/alert_decomp.py              # → reports/alert_decomp.md (~20–40 хв)
    python experiments/alert_decomp.py --limit 20   # перевірка
"""

import argparse
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import auto_train as AT
import idio_features as I
import klines1m as K
from candidate_6m import load_set, proba
from fetch import excluded_tickers, read_tickers_file
from tiers import tier_frame

OUT = os.path.join("reports", "alert_decomp.md")
HAR = (-1.10, 0.23, 0.19, 0.32, 0.17)          # const, 1 год, 4 год, доба, тиждень — experiments/har_vol.py (усі дані, 4 год)
WIN = (12, 48, 288, 2016)
MIN_FILL = 0.8
CALM_KEEP, SEED = 0.25, 42
KS = (0.5, 1.0)
BIG = "≥ 20 млн"


def coin_points(c5: pd.DataFrame) -> pd.DataFrame:
    """Точки T = закриття кожного 15m-бару: σ̂ (HAR, 4 год), фактичний рух, r4, бар'єри ±kσ̂."""
    grid = pd.date_range(c5.index[0], c5.index[-1], freq="5min", tz="UTC")
    c, h, lo = (c5[k].reindex(grid) for k in ("c", "h", "l"))
    lc = np.log(c)
    r = lc.diff()
    sq, have = r ** 2, r.notna().astype(float)
    xs = []
    for n in WIN:
        s, k = sq.rolling(n, min_periods=1).sum(), have.rolling(n, min_periods=1).sum()
        xs.append(np.log((s / k).where(k >= MIN_FILL * n)).shift(1))
    lv = HAR[0] + sum(b * x for b, x in zip(HAR[1:], xs))
    sig = np.sqrt(np.exp(lv) * 48)
    fs = sq[::-1].rolling(48, min_periods=1).sum()[::-1]
    fk = have[::-1].rolling(48, min_periods=1).sum()[::-1]
    move = np.sqrt(fs / fk * 48).where(fk >= MIN_FILL * 48)
    p0 = lc.shift(1)
    r4 = lc.shift(-47) - p0
    d = pd.DataFrame({"sig": sig, "move": move, "r4": r4, "p0": p0}, index=grid)
    keep = (d.index.minute % 15 == 0) & d["sig"].notna() & d["move"].notna() & d["r4"].notna()
    d = d[keep]
    # бар'єри: перші 48 барів після T (рядки T … T + 47)
    pos = grid.get_indexer(d.index)
    H, L = np.log(h.to_numpy()), np.log(lo.to_numpy())
    idx = pos[:, None] + np.arange(48)[None, :]
    idx = np.clip(idx, 0, len(grid) - 1)
    hh, ll = H[idx] - d["p0"].to_numpy()[:, None], L[idx] - d["p0"].to_numpy()[:, None]
    for k in KS:
        b = k * d["sig"].to_numpy()[:, None]
        up, dn = np.nan_to_num(hh, nan=-9) >= b, np.nan_to_num(ll, nan=9) <= -b
        fu = np.where(up.any(1), up.argmax(1), 99)
        fd = np.where(dn.any(1), dn.argmax(1), 99)
        res = np.where((fu == 99) & (fd == 99), np.nan, np.where(fu < fd, 1.0, np.where(fd < fu, 0.0, np.nan)))
        d[f"up{k}"] = res
    d["move_x"] = d["move"] / d["sig"]
    d["rz"] = d["r4"] / d["sig"]
    return d.drop(columns=["p0"])


def main() -> None:
    ap = argparse.ArgumentParser(description="Що несе сигнал моделі — розмір руху чи напрямок (HAR як контроль, 6 міс. архіву)")
    ap.add_argument("--limit", type=int, default=0, help="лише перші N монет (перевірка)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    s = load_set("models")
    uni = json.load(open(os.path.join(I.DIR, "_universe.json"), encoding="utf-8"))
    idx = I.market_index()
    trained = ({t.upper() for t in read_tickers_file("tickers.txt")} | set(pd.read_csv("pumps.csv")["base_asset"].str.upper())
               | excluded_tickers(".") | set(s["m15"].get("tickers", [])) | set(s["start"].get("tickers", [])))
    have1m = set(K.available())
    coins = sorted(c for c in uni if c not in trained and c in have1m and os.path.exists(os.path.join(I.DIR, f"{c}.pkl")))
    coins = coins[: args.limit or None]
    start, end = pd.Timestamp(I.START, tz="UTC"), pd.Timestamp(I.END, tz="UTC") + pd.Timedelta(days=1)
    share = s["m15"].get("confirm", {}).get("share", 0.75)
    rng = np.random.default_rng(SEED)
    print(f"монет {len(coins)}", flush=True)
    parts = []
    for n, c in enumerate(coins, 1):
        x = pickle.load(open(os.path.join(I.DIR, f"{c}.pkl"), "rb"))
        f15, h1 = x["f15"], x["h1"]
        f15 = f15.join(I.new_features(x["h"], idx).reindex(f15["ts"]).set_axis(f15.index))
        s1 = proba(s["m1"], h1)
        tm = tier_frame(proba(s["m15"], f15), s1, share).set_index("ts").sort_index()
        ts_ = tier_frame(proba(s["start"], f15), s1, share).set_index("ts").sort_index()
        tm, ts_ = tm[~tm.index.duplicated(keep="last")], ts_[~ts_.index.duplicated(keep="last")]
        g = pd.DataFrame({"s15": tm["s15"], "alert": tm["alert_score"], "start": ts_["alert_score"].reindex(tm.index)})
        g.index = g.index + pd.Timedelta(minutes=15)                 # момент рішення — закриття бару
        g = g[(g.index >= start) & (g.index < end)]
        m1 = K.load(c)
        if m1.empty or g.empty:
            continue
        d = coin_points(K.resample(m1, "5min")).join(g, how="inner")
        if d.empty:
            continue
        st = (d["start"] >= 1).astype(int)
        prev4 = st.shift(1).rolling(4, min_periods=1).max().fillna(0)
        d["group"] = np.where(d["alert"] >= 1, "Alert", np.where(d["s15"] >= 1, "Candidate", "Calm"))
        d["entry"] = (st == 1) & (prev4 == 0)
        d = d[(d["group"] != "Calm") | d["entry"] | (rng.random(len(d)) < CALM_KEEP)]
        mth = d.index.strftime("%Y-%m")
        d = d.assign(ticker=c, half=AT.half(c), month=mth, band=[uni[c].get(m) for m in mth])
        parts.append(d.reset_index(names="T"))
        if n % 50 == 0:
            print(f"  {n}/{len(coins)}", flush=True)
    t = pd.concat(parts, ignore_index=True)
    t = t[t["band"].notna()]
    t["period"] = np.where(t["month"] <= "2026-06", "квіт–черв", "лип–вер")
    t.to_csv(OUT.replace(".md", "_points.csv"), index=False)

    def sets(d):
        return {"Calm": d[(d["group"] == "Calm") & ~d["entry"]], "Candidate": d[d["group"] == "Candidate"],
                "Alert": d[d["group"] == "Alert"], "Entry (split)": d[d["entry"]]}

    def stat(x):
        if not len(x):
            return {"n": 0}
        return {"n": len(x), "монет": x["ticker"].nunique(), "рух ÷ σ̂ (медіана)": round(x["move_x"].median(), 2),
                "r4 > 0": f"{(x['r4'] > 0).mean():.1%}", "сер. r4 у σ̂": round(x["rz"].mean(), 3),
                "+0.5σ першим": f"{x['up0.5'].mean():.1%}", "+1σ першим": f"{x['up1.0'].mean():.1%}",
                "торкнулось ±1σ": f"{x['up1.0'].notna().mean():.0%}"}

    def table(rows):
        cols = list(dict.fromkeys(k for r in rows for k in r))
        return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
                         + ["| " + " | ".join(str(r.get(k, "")) for k in cols) + " |" for r in rows])

    def boot_diff(a, b, col, n=1000, seed=7):
        """95% інтервал різниці середніх (a − b) з бутстрепом по монетах."""
        r = np.random.default_rng(seed)
        ta, tb = a.groupby("ticker")[col], b.groupby("ticker")[col]
        sa, ca, sb, cb = ta.sum(), ta.count(), tb.sum(), tb.count()
        coins_ = np.array(sorted(set(sa.index) | set(sb.index)))
        sa, ca, sb, cb = (v.reindex(coins_).fillna(0).to_numpy() for v in (sa, ca, sb, cb))
        out = []
        for _ in range(n):
            k = r.integers(0, len(coins_), len(coins_))
            if ca[k].sum() and cb[k].sum():
                out.append(sa[k].sum() / ca[k].sum() - sb[k].sum() / cb[k].sum())
        return np.percentile(out, [2.5, 97.5])

    L = ["# Що несе сигнал моделі: розмір руху чи напрямок (6 міс., 20.10)", "",
         f"Готові models/; {t['ticker'].nunique()} монет чесного списку поза навчанням; точка — закриття 15m-бару; σ̂ — HAR на 4 год; "
         f"Calm — {CALM_KEEP:.0%} випадкових. Рух ÷ σ̂: > 1 — рухалось сильніше за прогноз розмаху. «+1σ першим» — частка випадків, коли "
         "ціна за 4 год спершу торкнулась +1σ̂, а не −1σ̂ (нічиї й «не торкнулась» не рахуються).", ""]
    for name, d in (("Усі монети", t), (f"Монети {BIG}", t[t["band"] == BIG])):
        L += [f"## {name}", "", table([{"група": g, **stat(x)} for g, x in sets(d).items()]), ""]
    sl = []
    for col, vals in (("half", ("A", "B")), ("period", ("квіт–черв", "лип–вер"))):
        for v in vals:
            for g, x in sets(t[t[col] == v]).items():
                if g in ("Calm", "Alert", "Entry (split)"):
                    sl.append({"зріз": f"{'монети ' + v if col == 'half' else v}", "група": g, **stat(x)})
    L += ["## Половини монет і часу", "", table(sl), ""]

    S = sets(t)
    size_ok, dir_ok, notes = True, True, []
    for col, vals in (("half", ("A", "B")), ("period", ("квіт–черв", "лип–вер"))):
        for v in vals:
            ss = sets(t[t[col] == v])
            ratio = ss["Alert"]["move_x"].median() / ss["Calm"]["move_x"].median()
            dd = ss["Entry (split)"]["up1.0"].mean() - ss["Calm"]["up1.0"].mean()
            size_ok &= ratio >= 1.10
            dir_ok &= dd >= 0
            notes.append(f"{v}: розмір ×{ratio:.2f}, напрямок {dd * 100:+.1f} п.п.")
    ratio_all = S["Alert"]["move_x"].median() / S["Calm"]["move_x"].median()
    e, cm = S["Entry (split)"].dropna(subset=["up1.0"]), S["Calm"].dropna(subset=["up1.0"])
    dd_all = e["up1.0"].mean() - cm["up1.0"].mean()
    lo, hi = boot_diff(e, cm, "up1.0")
    dir_ok = dir_ok and dd_all >= 0.03 and lo > 0
    L += ["## Критерії (записано до запуску)", "",
          f"- РОЗМІР: медіана рух ÷ σ̂ у Alert ÷ у Calm = ×{ratio_all:.2f} (потрібно ≥ 1.10 в усіх половинах) → **{'так' if size_ok else 'ні'}**",
          f"- НАПРЯМОК: «+1σ першим» Entry − Calm = {dd_all * 100:+.1f} п.п., 95% інтервал [{lo * 100:+.1f}; {hi * 100:+.1f}] "
          f"(потрібно ≥ +3 п.п., інтервал > 0, усі половини ≥ 0) → **{'так' if dir_ok else 'ні'}**",
          "- половини: " + "; ".join(notes), "",
          "**Висновок:** " + ("сигнал несе і розмір, і напрямок." if size_ok and dir_ok else
                              "сигнал — це передусім «тут буде рух» (розмір), напрямку понад випадковість не видно." if size_ok else
                              "сигнал несе напрямок, але не розмір понад HAR." if dir_ok else
                              "ні розміру понад HAR, ні напрямку за цими критеріями."), ""]
    open(OUT, "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
