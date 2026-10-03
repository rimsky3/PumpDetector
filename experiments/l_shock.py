"""
experiments/l_shock.py — «L-shock / liquidation fade» з Medium (Insilico «Volume moves price»), 2026-10-03.

Гіпотеза автора: велика червона свічка з великим обсягом (каскад ліквідацій), за нею зелена свічка помітного розміру
на ВІДНОСНО МАЛОМУ обсязі — каскад закінчився, стакан порожній → «almost always mechanical scalp up».
Вхід на закритті зеленої, частина на 1:1 (як у PDF Big Picture Trading), решта — на початку зливу («point of origin»).

Формалізація (пороги зафіксовано ДО прогону, не підбираються):
    злив F:     червона свічка, тіло ≥ BODY_ATR × ATR(14) (до F), обсяг ≥ VOL_X × медіана обсягу за 24 год (до F);
                якщо за RESP_BARS барів іде ще один злив — каскад триває, рахуємо від останнього;
    відповідь G: перша з наступних RESP_BARS свічок — зелена, тіло ≥ RESP_BODY × тіла F, обсяг ≤ RESP_VOL × обсягу F;
    угода:      вхід — закриття G; стоп — лоу зливу (мінімум F…G); ½ на +1R; решта — на відкритті F (не нижче +1R),
                стоп або закриття через HORIZON; комісія FEE на всю позицію. Якщо стоп і ціль в одній 1m-свічці — стоп.
    варіант OI: OI у монетах (Bybit, 15m) від останнього закриття ≤ F до останнього закриття ≤ входу впав ≥ OI_DROP —
                ознака ліквідацій (Binance-ліквідацій в архіві немає, Coinalyze тримає 1–2 міс.).
Контроль: (а) злив без відповіді — вхід на закритті F+1, стоп і цілі за тими самими правилами;
          (б) усі зливи, вхід на закритті F+1; (в) випадкові бари тих самих монет, стоп і цілі — медіанні для подій у ATR.
ТФ: 5m (основний; 1m → 5m) і 1m (як у джерелі). Угоди моделюються на 1m-барах. Дані — experiments/klines1m.py.
Періоди: A = квітень–червень, B = липень–вересень 2026.

    python experiments/l_shock.py            # → reports/l_shock.md, reports/l_shock_trades.csv
    python experiments/l_shock.py --no-oi    # без бази (лише ціна й обсяг)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import klines1m as K

VOL_X, BODY_ATR, RESP_BARS, RESP_BODY, RESP_VOL = 5.0, 2.0, 3, 0.5, 0.5
OI_DROP = 0.01
HORIZON = pd.Timedelta(hours=4)
FEE = 0.001
TFS = {"5m": ("5min", 288), "1m": ("1min", 1440)}
SPLIT = pd.Timestamp("2026-07-01", tz="UTC")
RANDOM_PER_EVENT, SEED = 3, 42
OUT_MD = os.path.join("reports", "l_shock.md")
OUT_CSV = os.path.join("reports", "l_shock_trades.csv")


def atr(b: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = b["c"].shift(1)
    tr = pd.concat([b["h"] - b["l"], (b["h"] - pc).abs(), (b["l"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def simulate(m1: pd.DataFrame, t_entry: pd.Timestamp, entry: float, stop: float, t2: float) -> dict | None:
    """Угода на 1m-барах після входу: ½ на +1R, решта — t2 / стоп / час. Результат — % від входу, з комісією."""
    r = entry - stop
    if r <= 0 or not np.isfinite(r):
        return None
    t1 = entry + r
    t2 = max(t2, t1)
    w = m1.loc[t_entry:t_entry + HORIZON - pd.Timedelta(minutes=1)]
    if w.empty:
        return None
    lo, hi, cl, ts = w["l"].to_numpy(), w["h"].to_numpy(), w["c"].to_numpy(), w.index
    half = None                      # ціна виходу першої половини
    for k in range(len(w)):
        if lo[k] <= stop:            # стоп (раніше за ціль у тій самій свічці)
            p1 = half if half is not None else stop
            res = 0.5 * (p1 / entry - 1) + 0.5 * (stop / entry - 1)
            return {"res": res - FEE, "exit_ts": ts[k], "reason": "stop" if half is None else "partial+stop"}
        if half is None and hi[k] >= t1:
            half = t1
        if half is not None and hi[k] >= t2:
            res = 0.5 * (half / entry - 1) + 0.5 * (t2 / entry - 1)
            return {"res": res - FEE, "exit_ts": ts[k], "reason": "target"}
    p1 = half if half is not None else cl[-1]
    res = 0.5 * (p1 / entry - 1) + 0.5 * (cl[-1] / entry - 1)
    return {"res": res - FEE, "exit_ts": ts[-1], "reason": "time" if half is None else "partial+time"}


def fwd(m1c: pd.Series, t: pd.Timestamp, entry: float) -> dict:
    """Ціна через 15 хв / 1 год / 4 год після входу (закриття 1m), % від входу."""
    out = {}
    for name, d in (("f15m", 15), ("f1h", 60), ("f4h", 240)):
        j = m1c.index.searchsorted(t + pd.Timedelta(minutes=d - 1))
        out[name] = m1c.iloc[j] / entry - 1 if j < len(m1c) else np.nan
    return out


def oi_series(conn, base: str) -> pd.Series | None:
    from features import aggregate, load_raw
    raw = load_raw(conn, [base], "15min", exchanges=["Bybit"])
    h = aggregate(raw, base, "15min")
    if h.empty or "oi_coins_c" not in h:
        return None
    s = h["oi_coins_c"].dropna()
    return s if len(s) else None


def oi_drop(oi: pd.Series | None, t_from: pd.Timestamp, t_to: pd.Timestamp) -> float:
    """Зміна OI від останнього закриття 15m-бару ≤ t_from до останнього ≤ t_to (закриття = відкриття + 15 хв)."""
    if oi is None:
        return np.nan
    close = oi.index + pd.Timedelta(minutes=15)
    a, b = close.searchsorted(t_from, "right") - 1, close.searchsorted(t_to, "right") - 1
    if a < 0 or b < 0 or b <= a:
        return np.nan
    return oi.iloc[b] / oi.iloc[a] - 1


def events(m1: pd.DataFrame, tf: str, oi, base: str, rng) -> list[dict]:
    rule, med_n = TFS[tf]
    b = K.resample(m1, rule) if rule != "1min" else m1
    if len(b) < med_n + 50:
        return []
    bar = pd.Timedelta(rule)
    a = atr(b).shift(1)
    med = b["v"].rolling(med_n, min_periods=med_n // 2).median().shift(1)
    o, h, l, c, v = (b[k].to_numpy() for k in "ohlcv")
    body = o - c
    fl = (c < o) & (body >= BODY_ATR * a.to_numpy()) & (v >= VOL_X * med.to_numpy())
    idx = np.flatnonzero(fl)
    idx = [i for n_, i in enumerate(idx) if not (n_ + 1 < len(idx) and idx[n_ + 1] <= i + RESP_BARS)]   # кінець каскаду
    m1c = m1["c"]
    out = []
    for i in idx:
        if i + RESP_BARS >= len(b):
            continue
        g = None
        for k in range(1, RESP_BARS + 1):
            j = i + k
            if c[j] > o[j] and (c[j] - o[j]) >= RESP_BODY * body[i] and v[j] <= RESP_VOL * v[i]:
                g = j
                break
        aF = a.iloc[i]
        for kind, j in (("lshock", g), ("all_flush", i + 1)) + ((("no_response", i + 1),) if g is None else ()):
            if j is None:
                continue
            t_e = b.index[j] + bar                                    # закриття бару входу
            entry = c[j]
            stop = min(l[i:j + 1])
            sim = simulate(m1, t_e, entry, stop, o[i])
            if sim is None:
                continue
            out.append({"base": base, "tf": tf, "kind": kind, "flush_ts": b.index[i], "entry_ts": t_e, "entry": entry,
                        "stop": stop, "origin": o[i], "atr": aF, "r_atr": (entry - stop) / aF, "orig_atr": (o[i] - entry) / aF,
                        "delay": j - i, "flush_vol_x": v[i] / med.iloc[i], "flush_body_atr": body[i] / aF,
                        "oi_chg": oi_drop(oi, b.index[i], t_e), **sim, **fwd(m1c, t_e, entry)})
    return out


def random_trades(m1: pd.DataFrame, tf: str, base: str, n: int, r_atr: float, orig_atr: float, rng) -> list[dict]:
    rule, _ = TFS[tf]
    b = K.resample(m1, rule) if rule != "1min" else m1
    a = atr(b)
    ok = np.flatnonzero(a.notna().to_numpy())
    ok = ok[ok < len(b) - 1]
    if not len(ok) or n <= 0:
        return []
    bar = pd.Timedelta(rule)
    out = []
    for j in rng.choice(ok, size=min(n, len(ok)), replace=False):
        t_e, entry, aj = b.index[j] + bar, b["c"].iloc[j], a.iloc[j]
        sim = simulate(m1, t_e, entry, entry - r_atr * aj, entry + orig_atr * aj)
        if sim:
            out.append({"base": base, "tf": tf, "kind": "random", "entry_ts": t_e, "entry": entry, "atr": aj,
                        "r_atr": r_atr, "orig_atr": orig_atr, "oi_chg": np.nan, **sim, **fwd(m1["c"], t_e, entry)})
    return out


def stats(d: pd.DataFrame) -> dict:
    r = d["res"].dropna() * 100
    if r.empty:
        return {"n": 0}
    pos, neg = r[r > 0].sum(), -r[r < 0].sum()
    g = r + FEE * 100                                  # без комісії: чи є перевага до витрат
    gpos, gneg = g[g > 0].sum(), -g[g < 0].sum()
    top3 = r.sort_values(ascending=False).iloc[:3].sum()
    hold = (pd.to_datetime(d["exit_ts"], utc=True) - pd.to_datetime(d["entry_ts"], utc=True)).dt.total_seconds() / 60
    return {"n": len(r), "монет": d["base"].nunique(), "R %": round((d["entry"] - d["stop"]).div(d["entry"]).median() * 100, 2) if "stop" in d and d["stop"].notna().any() else "", "у плюсі": f"{(r > 0).mean():.0%}", "PF": round(pos / neg, 2) if neg else np.inf,
            "PF без ком.": round(gpos / gneg, 2) if gneg else np.inf,
            "сер. %": round(r.mean(), 2), "медіана %": round(r.median(), 2), "сума %": round(r.sum()),
            "без топ-3 %": round(r.sum() - top3), "15хв %": round(d["f15m"].median() * 100, 2),
            "1год %": round(d["f1h"].median() * 100, 2), "4год %": round(d["f4h"].median() * 100, 2),
            "утрим. хв": round(hold.median()), "≥10 хв": f"{(hold >= 10).mean():.0%}"}


def table(rows: list[dict], cols: list[str]) -> str:
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="L-shock (liquidation fade): подієвий тест на 1m-свічках Binance з контролем.")
    ap.add_argument("--no-oi", action="store_true", help="без OI з бази (варіант «+ падіння OI» не рахується)")
    ap.add_argument("--limit", type=int, default=0, help="лише перші N монет (перевірка)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    conn = None
    if not args.no_oi:
        import psycopg
        from dotenv import load_dotenv
        load_dotenv(os.path.join(K.ROOT, ".env"))
        conn = psycopg.connect(os.environ["DATABASE_URL"])
    bases = K.available()[: args.limit or None]
    rng = np.random.default_rng(SEED)
    rows = []
    for n, base in enumerate(bases, 1):
        m1 = K.load(base)
        if m1.empty:
            continue
        oi = oi_series(conn, base) if conn else None
        for tf in TFS:
            rows += events(m1, tf, oi, base, rng)
        if n % 20 == 0:
            print(f"  {n}/{len(bases)} монет, подій {len(rows)}", flush=True)
    ev = pd.DataFrame(rows)
    # випадкові входи: стоп і цілі — медіанні для L-shock у ATR цього ТФ
    rnd = []
    for tf in TFS:
        e = ev[(ev["tf"] == tf) & (ev["kind"] == "lshock")]
        if e.empty:
            continue
        r_atr, o_atr = e["r_atr"].median(), e["orig_atr"].clip(lower=0).median()
        per = e.groupby("base").size()
        for base, k in per.items():
            rnd += random_trades(K.load(base), tf, base, int(k * RANDOM_PER_EVENT), r_atr, o_atr, rng)
    allr = pd.concat([ev, pd.DataFrame(rnd)], ignore_index=True)
    allr["period"] = np.where(pd.to_datetime(allr["entry_ts"], utc=True) < SPLIT, "A", "B")
    os.makedirs("reports", exist_ok=True)
    allr.to_csv(OUT_CSV, index=False)

    cols = ["група", "n", "монет", "R %", "у плюсі", "PF", "PF без ком.", "сер. %", "медіана %", "сума %", "без топ-3 %", "15хв %", "1год %", "4год %",
            "утрим. хв", "≥10 хв"]
    md = ["# L-shock (liquidation fade) — подієвий тест", "",
          f"Монет: {allr['base'].nunique()}, 1m Binance, квітень–вересень 2026. Пороги: обсяг ≥ {VOL_X:g}× медіани 24 год, тіло ≥ "
          f"{BODY_ATR:g} ATR; відповідь за {RESP_BARS} бари: зелена, тіло ≥ {RESP_BODY:g} зливу, обсяг ≤ {RESP_VOL:g} зливу. "
          f"Угода: стоп — лоу зливу, ½ на 1R, решта — початок зливу / стоп / {HORIZON.total_seconds() / 3600:g} год; комісія {FEE:.1%}.", ""]
    for tf in TFS:
        d = allr[allr["tf"] == tf]
        groups = [("L-shock", d[d["kind"] == "lshock"]),
                  (f"L-shock + OI −{OI_DROP:.0%}", d[(d["kind"] == "lshock") & (d["oi_chg"] <= -OI_DROP)]),
                  ("L-shock, OI відомий, без падіння", d[(d["kind"] == "lshock") & (d["oi_chg"] > -OI_DROP)]),
                  ("злив без відповіді (вхід F+1)", d[d["kind"] == "no_response"]),
                  ("усі зливи (вхід F+1)", d[d["kind"] == "all_flush"]),
                  ("випадковий вхід", d[d["kind"] == "random"])]
        md += [f"## {tf}", "", table([{"група": g, **stats(x)} for g, x in groups], cols), ""]
        per = []
        for g, x in groups[:1] + groups[3:]:
            for p in "AB":
                per.append({"група": f"{g} · {'квіт–черв' if p == 'A' else 'лип–вер'}", **stats(x[x["period"] == p])})
        md += ["### За періодами", "", table(per, cols), ""]
        ls = d[d["kind"] == "lshock"].copy()
        if len(ls):
            ls["month"] = pd.to_datetime(ls["entry_ts"], utc=True).dt.strftime("%Y-%m")
            md += ["### L-shock за місяцями", "", table([{"група": m, **stats(x)} for m, x in ls.groupby("month")], cols), ""]
            hold = (pd.to_datetime(ls["exit_ts"], utc=True) - pd.to_datetime(ls["entry_ts"], utc=True)).dt.total_seconds() / 60
            md += ["### L-shock за утриманням (опис, не правило: утримання відоме лише після виходу)", "",
                   table([{"група": "< 10 хв", **stats(ls[hold < 10])}, {"група": "≥ 10 хв", **stats(ls[hold >= 10])}], cols), ""]
            md += [f"Виходи L-shock: {ls['reason'].value_counts().to_dict()}", ""]
    open(OUT_MD, "w", encoding="utf-8").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
