"""
experiments/pullback_grid.py — вхід з відкату після Alert: глибина відкату × час дії заявки × стоп (2026-10-03).

Питання власника: (1) з фіксованим стопом 2% вхід з відкату часто вибиває, а з 3% — удвічі рідше? (2) який рівень
відкату на історії дає найкращу точку входу? Руками наперед цього не визначити — тому сітка на історії.

Правило (як limit_entry.py, розширено): на першому барі Alert (позиції й заявки немає) —
    ринок        вхід за закриттям сигнального бару (як зараз);
    −x ATR       ліміт = закриття − x·ATR(15m);   −y%  ліміт = закриття × (1 − y);
    заявка діє VALID барів; виконання — лоу бару ≤ ліміту (за лімітом або відкриттям, якщо бар відкрився нижче);
    не виконалась — угоди немає (рахуємо, які рухи пропущено).
Стоп від ціни входу: фіксований % або k·ATR (k робочої моделі). На барі виконання стоп перевіряється (консервативно:
спершу виконання, потім можливий стоп); ціль ½ і вихід за моделлю — з наступного бару.
Вихід — поточний (risk.PARTIAL_EXIT: ½ на +3 ATR, далі беззбиток; решта — 4 бари без Alert або стоп). Комісія 0.1%.

Дані (без бази, кеші попередніх експериментів, монети поза навчанням):
    A/B — 6 міс. Bybit 15m (experiments/long_backtest.py, data/long_bt): вибір параметрів на A = квітень–червень,
          перевірка на B = липень–вересень, плюс скільки місяців з PF > 1;
    вер — вересень, Binance + Bybit + ліквідації (exit_structure / style_backtest) — інше джерело даних.

    python experiments/pullback_grid.py      # → reports/pullback_grid.md, reports/pullback_grid.csv
"""

import argparse
import itertools
import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import numpy as np
import pandas as pd

FEE = 0.001
FRAC, TP_ATR = 0.5, 3.0
EXIT_BARS = 4
SPLIT = pd.Timestamp("2026-07-01", tz="UTC")
ENTRIES = [("ринок", "market", 0.0)] + [(f"−{x:g} ATR", "atr", x) for x in (0.25, 0.5, 1.0, 1.5)] \
    + [(f"−{y:.0%}", "pct", y) for y in (0.01, 0.02, 0.03, 0.05)]
VALIDS = (4, 8, 16)                     # 1 / 2 / 4 год
STOPS_PCT = (0.015, 0.02, 0.03, 0.04, 0.05, 0.07)
CACHE = os.path.join("reports", "backtests", "_pullback_arrays.pkl")
OUT_MD, OUT_CSV = os.path.join("reports", "pullback_grid.md"), os.path.join("reports", "pullback_grid.csv")
RUNS = ["20261001_1441_current_alert", "20261001_1517_current_alert"]


def atr_np(h: pd.DataFrame, n: int = 14) -> np.ndarray:
    pc = h["price"].shift(1)
    tr = pd.concat([h["high"] - h["low"], (h["high"] - pc).abs(), (h["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean().to_numpy()


def arrays(h: pd.DataFrame, score: pd.Series) -> dict:
    sc = score.reindex(h.index).to_numpy(dtype=float)
    return {"t": h.index.as_unit("ns").asi8, "o": h["open"].to_numpy(float), "h": h["high"].to_numpy(float),
            "l": h["low"].to_numpy(float), "c": h["price"].to_numpy(float), "a": atr_np(h), "al": np.nan_to_num(sc) >= 1}


def prepare(k: float) -> dict:
    """Масиви по монетах для обох наборів даних (один раз, кеш)."""
    if os.path.exists(CACHE):
        return pickle.load(open(CACHE, "rb"))
    band = {}
    for run in RUNS:
        d = json.load(open(os.path.join("reports", "backtests", run + ".json"), encoding="utf-8"))
        for u in d["meta"]["universe"]:
            band[u["base"]] = "≥ 20 млн" if d["meta"]["min_turnover"] >= 20e6 else "5–20 млн"
    data = {"6міс": {}, "вер": {}, "band": band}
    lb = os.path.join("data", "long_bt")
    for f in sorted(os.listdir(lb)):
        got = pickle.load(open(os.path.join(lb, f), "rb"))
        if got is None:
            continue
        t, h = got
        data["6міс"][f[:-4]] = arrays(h.dropna(subset=["price"]), t["alert_score"])
    alerts = pickle.load(open(os.path.join("reports", "backtests", "_exit_structure_cache.pkl"), "rb"))
    bars = pickle.load(open(os.path.join("reports", "backtests", "_style_bars.pkl"), "rb"))
    for b in sorted(set(alerts) & set(bars)):
        t, _ = alerts[b]
        h = bars[b]
        h = h[h.index >= t.index.min() - pd.Timedelta(days=1)].dropna(subset=["price"])
        data["вер"][b] = arrays(h, t["alert_score"])
    pickle.dump(data, open(CACHE, "wb"))
    return data


def simulate(A: dict, mode: str, x: float, valid: int, stop_kind: str, stop_v: float) -> list[tuple]:
    """Угоди однієї монети: (індекс сигналу, час сигналу, результат | None якщо не виконано, причина, очікування в барах)."""
    o, hi, lo, c, a, al, t = A["o"], A["h"], A["l"], A["c"], A["a"], A["al"], A["t"]
    n, out, busy = len(c), [], -1
    starts = np.flatnonzero(al[1:] & ~al[:-1]) + 1
    for i in starts:
        if i <= busy or np.isnan(c[i]) or np.isnan(a[i]):
            continue
        if mode == "market":
            f, e, first = i, c[i], i + 1
        else:
            px = c[i] - x * a[i] if mode == "atr" else c[i] * (1 - x)
            f = None
            for j in range(i + 1, min(i + valid, n - 1) + 1):
                if lo[j] <= px:
                    f, e = j, (min(o[j], px) if not np.isnan(o[j]) else px)
                    break
            if f is None:
                out.append((i, t[i], None, "не виконано", valid))
                busy = i + valid
                continue
            first = f                          # стоп — уже на барі виконання
        stop = e * (1 - stop_v) if stop_kind == "pct" else e - stop_v * a[i]
        tp, part, below, res, why, end = e + TP_ATR * a[i], 0.0, 0, None, None, n - 1
        for j in range(first, n):
            if lo[j] <= stop:
                res, why, end = part + (1 - (FRAC if part else 0)) * (stop / e - 1) - FEE, "стоп" if not part else "б/з", j
                break
            if j == f:
                continue
            if not part and hi[j] >= tp:
                part = FRAC * (tp / e - 1)
                stop = max(stop, e)
            below = below + 1 if not al[j] else 0
            if below >= EXIT_BARS:
                res, why, end = part + (1 - (FRAC if part else 0)) * (c[j] / e - 1) - FEE, "модель", j
                break
        if res is None:
            res, why = part + (1 - (FRAC if part else 0)) * (c[-1] / e - 1) - FEE, "кінець"
        out.append((i, t[i], res, why, f - i))
        busy = end
    return out


_D = None


def _init(data):
    global _D
    _D = data


def run_combo(args) -> list[dict]:
    ds, (ename, mode, x), valid, (sname, skind, sval) = args
    rows = []
    for b, A in _D[ds].items():
        for i, ts, res, why, wait in simulate(A, mode, x, valid, skind, sval):
            rows.append({"ds": ds, "entry": ename, "valid": valid if mode != "market" else 0, "stop": sname, "base": b,
                         "band": _D["band"].get(b, ""), "sig": int(i), "ts": int(ts), "ret": res, "why": why, "wait": wait})
    return rows


def pf(v: pd.Series) -> float:
    v = v.dropna()
    gl = -v[v <= 0].sum()
    return round(v[v > 0].sum() / gl, 2) if gl else np.nan


def stats(g: pd.DataFrame, mkt: pd.Series) -> dict:
    v = g["ret"].dropna()
    filled = g["ret"].notna()
    key = list(zip(g["base"], g["sig"]))
    mk = mkt.reindex(pd.MultiIndex.from_tuples(key)) if len(key) else pd.Series(dtype=float)
    missed = mk[~filled.to_numpy()] if len(mk) else mk
    s = v.sort_values()
    return {"сигналів": len(g), "виконано": f"{filled.mean():.0%}", "угод": len(v), "по стопу": f"{g['why'].eq('стоп').sum() / max(len(v), 1):.0%}",
            "у плюсі": f"{(v > 0).mean():.0%}", "PF": pf(v), "PF без топ-3": pf(s.iloc[:-3]), "медіана %": round(v.median() * 100, 2),
            "сума %": round(v.sum() * 100), "пропущено ≥ +10%": int((missed >= 0.10).sum()), "пропущено, сума %": round(missed.sum() * 100)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Вхід з відкату після Alert: сітка глибина × час заявки × стоп на 6 міс. + вересні.")
    ap.add_argument("--workers", type=int, default=8, help="процесів")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    import train
    k = joblib.load(train.model_path(train.M15_NAME)).get("risk", {}).get("k", 2.5)
    data = prepare(k)
    stops = [(f"{s:.1%}".replace(".0%", "%"), "pct", s) for s in STOPS_PCT] + [(f"{k:.2f} ATR", "atr", k)]
    combos = [(ds, e, v, s) for ds in ("6міс", "вер") for e in ENTRIES for v in (VALIDS if e[1] != "market" else (0,)) for s in stops]
    print(f"{len(combos)} наборів, монет: 6 міс. {len(data['6міс'])}, вересень {len(data['вер'])}", flush=True)
    rows = []
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(data,)) as ex:
        for n, r in enumerate(ex.map(run_combo, combos), 1):
            rows += r
            if n % 40 == 0:
                print(f"  {n}/{len(combos)}", flush=True)
    df = pd.DataFrame(rows)
    df["dt"] = pd.to_datetime(df["ts"], utc=True)
    df["period"] = np.where(df["dt"] < SPLIT, "A", "B")
    df.loc[df["ds"] == "вер", "period"] = "вер"
    df["month"] = df["dt"].dt.strftime("%m")
    # ринок з k·ATR (як зараз) — для «пропущено»
    kname = stops[-1][0]
    mkt = {ds: df[(df["ds"] == ds) & (df["entry"] == "ринок") & (df["stop"] == kname)].set_index(["base", "sig"])["ret"] for ds in ("6міс", "вер")}

    res = []
    for (entry, valid, stop), g in df.groupby(["entry", "valid", "stop"], sort=False):
        r = {"вхід": entry, "заявка, год": valid / 4 if valid else "—", "стоп": stop}
        a, b = g[g["period"] == "A"], g[g["period"] == "B"]
        s6 = stats(g[g["ds"] == "6міс"], mkt["6міс"])
        r.update({f"6міс {k_}": v for k_, v in s6.items()})
        r.update({"PF A": pf(a["ret"]), "PF B": pf(b["ret"]), "PF вер": pf(g[g["period"] == "вер"]["ret"])})
        months = g[g["ds"] == "6міс"].groupby("month")["ret"].apply(pf)
        r["міс. PF>1"] = f"{int((months > 1).sum())}/{len(months)}"
        sv = stats(g[g["ds"] == "вер"], mkt["вер"])
        r.update({"вер по стопу": sv["по стопу"], "вер виконано": sv["виконано"]})
        r["PF ≥20 млн (6міс)"] = pf(g[(g["ds"] == "6міс") & (g["band"] == "≥ 20 млн")]["ret"])
        res.append(r)
    R = pd.DataFrame(res)
    R.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    def md_table(d: pd.DataFrame) -> str:
        cols = list(d.columns)
        return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] +
                         ["| " + " | ".join(str(x) for x in row) + " |" for row in d.itertuples(index=False)])

    md = ["# Вхід з відкату після Alert — сітка параметрів", "",
          f"Робоча модель (Alert), монети поза навчанням. 6 міс. Bybit 15m: {len(data['6міс'])} монет (A = квіт–черв, B = лип–вер); "
          f"вересень Binance + Bybit + ліквідації: {len(data['вер'])} монет. Вихід — поточний (½ на +3 ATR, б/з; 4 бари без Alert / стоп), "
          "комісія 0.1%. **Параметри обирати за A, перевіряти на B і вересні.**", ""]
    # 1) розподіл відкатів після сигналу (питання 2)
    pb = []
    for ds in ("6міс", "вер"):
        for b, A in data[ds].items():
            st = np.flatnonzero(A["al"][1:] & ~A["al"][:-1]) + 1
            for i in st:
                if np.isnan(A["a"][i]) or i + 16 >= len(A["c"]):
                    continue
                for hrs, w in ((1, 4), (2, 8), (4, 16)):
                    m = np.nanmin(A["l"][i + 1:i + 1 + w])
                    pb.append({"ds": ds, "год": hrs, "%": (A["c"][i] - m) / A["c"][i] * 100, "ATR": (A["c"][i] - m) / A["a"][i]})
    P = pd.DataFrame(pb)
    rows_ = []
    for (ds, hrs), g in P.groupby(["ds", "год"]):
        rows_.append({"дані": ds, "за": f"{hrs} год", "сигналів": len(g),
                      "медіана відкату": f"{g['%'].median():.1f}% · {g['ATR'].median():.2f} ATR",
                      "25% сигналів ≤": f"{g['%'].quantile(.25):.1f}%", "75% ≤": f"{g['%'].quantile(.75):.1f}%",
                      "відкат ≥ 0.5 ATR": f"{(g['ATR'] >= 0.5).mean():.0%}", "≥ 1 ATR": f"{(g['ATR'] >= 1).mean():.0%}",
                      "≥ 2%": f"{(g['%'] >= 2).mean():.0%}", "≥ 3%": f"{(g['%'] >= 3).mean():.0%}", "≥ 5%": f"{(g['%'] >= 5).mean():.0%}"})
    md += ["## 1. Наскільки ціна відкочується після сигналу (найнижчий лоу після закриття сигнального бару)", "",
           md_table(pd.DataFrame(rows_)), ""]
    # 2) теплові таблиці: заявка 2 год
    for metric, title in (("PF A", "PF на A (квіт–черв) — вибір"), ("PF B", "PF на B (лип–вер) — перевірка"),
                          ("PF вер", "PF на вересні (Binance, інше джерело) — перевірка"), ("6міс по стопу", "частка угод, закритих стопом (6 міс.)"),
                          ("6міс виконано", "частка виконаних заявок (6 міс.)")):
        sub = R[(R["заявка, год"].isin([2.0, "—"]))]
        piv = sub.pivot_table(index="вхід", columns="стоп", values=metric, aggfunc="first", sort=False)
        piv = piv.reindex([e[0] for e in ENTRIES])[[s[0] for s in stops]]
        md += [f"## 2. {title} — заявка діє 2 год", "", md_table(piv.reset_index()), ""]
    # 3) найкращі на A і як вони на B / вересні
    top = R.sort_values("PF A", ascending=False).head(15)
    cols = ["вхід", "заявка, год", "стоп", "6міс сигналів", "6міс виконано", "6міс угод", "6міс по стопу", "PF A", "PF B", "PF вер",
            "міс. PF>1", "6міс медіана %", "6міс PF без топ-3", "6міс пропущено ≥ +10%", "PF ≥20 млн (6міс)"]
    md += ["## 3. Топ-15 за A (обрано на A) — як вони на B і вересні", "", md_table(top[cols]), ""]
    base = R[(R["вхід"] == "ринок") & (R["стоп"] == kname)]
    md += ["## Для порівняння: як зараз (ринок, стоп k·ATR)", "", md_table(base[cols]), ""]
    open(OUT_MD, "w", encoding="utf-8").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
