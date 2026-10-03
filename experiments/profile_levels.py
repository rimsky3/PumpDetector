"""
experiments/profile_levels.py — чи тримають ціну рівні профілю обсягу (статті ATAS про Volume Profile / FRVP), 2026-10-03.

Гіпотеза статей: POC і межі value area (VAH / VAL) — підтримка / опір: при поверненні до рівня ціна частіше
відскакує, ніж проходить крізь нього. Рівні є всюди, тому порівнюємо з ВИПАДКОВИМИ рівнями з того самого діапазону.

Формалізація (зафіксовано до прогону):
    профіль дня D (UTC) — volume_profile.build з 5m-свічок (як на графіку), крок = «круглий» (медіанний денний діапазон
    монети / 40); рівні — POC, VAH, VAL; контроль — RANDOM_PER_DAY випадкових рівнів, рівномірно в [лоу D, хай D];
    тест — доба D+1 (1m-свічки): рівень «заряджено зверху», коли закриття ≥ L + d, і дотик — перший лоу ≤ L
    (зустрічний — закриття ≤ L − d, дотик — хай ≥ L); d = ATR(14) годинних барів на кінець D. Лише перший дотик рівня за добу;
    результат — що раніше за 24 год після дотику: відскок на d у бік, звідки прийшли, чи прохід на d крізь рівень.
    Обидва в одній 1m-свічці — нечітко (не рахується); жодного за 24 год — «час».
    Угода-еквівалент: ліміт на рівні, стоп d за рівнем, ціль d → очікування в R = 2p − 1 (без комісії; комісія 0.1% у R —
    0.1% / d).
Імпульси з розмітки (сетап pullback): профіль імпульсу start→end (pumps.csv, pattern pump), POC і контроль — 5 випадкових
рівнів у [лоу, хай] імпульсу; тест — 72 год після кінця, ті самі правила (d — ATR(14) 1h на кінець імпульсу).

    python experiments/profile_levels.py      # → reports/profile_levels.md, reports/profile_levels_events.csv
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import klines1m as K
import volume_profile as VP

RANDOM_PER_DAY, RANDOM_PER_IMPULSE, SEED = 3, 5, 42
OUTCOME = pd.Timedelta(hours=24)
IMPULSE_WINDOW = pd.Timedelta(hours=72)
FEE = 0.001
SPLIT = pd.Timestamp("2026-07-01", tz="UTC")
OUT_MD = os.path.join("reports", "profile_levels.md")
OUT_CSV = os.path.join("reports", "profile_levels_events.csv")


def atr1h(m1: pd.DataFrame) -> pd.Series:
    """ATR(14) Уайлдера на годинних барах; індекс — ЗАКРИТТЯ години (значення відоме з цього моменту)."""
    h = K.resample(m1, "h")
    pc = h["c"].shift(1)
    tr = pd.concat([h["h"] - h["l"], (h["h"] - pc).abs(), (h["l"] - pc).abs()], axis=1).max(axis=1)
    a = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    a.index = a.index + pd.Timedelta(hours=1)
    return a


def prof_bars(b5: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"high": b5["h"], "low": b5["l"], "volume": b5["v"],
                         "buy_share": (b5["bv"] / b5["v"]).where(b5["v"] > 0)}, index=b5.index)


def touch(t: np.ndarray, lo: np.ndarray, hi: np.ndarray, cl: np.ndarray, L: float, d: float,
          i0: int, i1: int, i_end: int) -> dict | None:
    """Перший дотик рівня L у [i0, i1) після «заряджання» на d; результат — до i_end (24 год після дотику)."""
    seg = slice(i0, i1)
    up = np.flatnonzero(cl[seg] >= L + d)
    dn = np.flatnonzero(cl[seg] <= L - d)
    cands = []
    if len(up):                                   # зверху: після заряджання — перший лоу ≤ L
        k = np.flatnonzero(lo[i0 + up[0] + 1:i1] <= L)
        if len(k):
            cands.append((i0 + up[0] + 1 + k[0], "зверху"))
    if len(dn):
        k = np.flatnonzero(hi[i0 + dn[0] + 1:i1] >= L)
        if len(k):
            cands.append((i0 + dn[0] + 1 + k[0], "знизу"))
    if not cands:
        return None
    j, side = min(cands)
    end = min(len(t), np.searchsorted(t, t[j] + OUTCOME.value, "left"))
    if side == "зверху":
        good, bad = hi[j:end] >= L + d, lo[j:end] <= L - d
    else:
        good, bad = lo[j:end] <= L - d, hi[j:end] >= L + d
    g = np.argmax(good) if good.any() else None
    b = np.argmax(bad) if bad.any() else None
    if g is None and b is None:
        res = "час"
    elif b is None or (g is not None and g < b):
        res = "відскок"
    elif g is None or b < g:
        res = "прохід"
    else:
        res = "нечітко"
    return {"touch_ts": pd.Timestamp(t[j], tz="UTC"), "side": side, "result": res}


def day_events(base: str, m1: pd.DataFrame, rng) -> list[dict]:
    b5 = K.resample(m1, "5min")
    if len(b5) < 288 * 3:
        return []
    days = b5.groupby(b5.index.floor("D"))
    rng_d = [g["h"].max() - g["l"].min() for _, g in days]
    step = VP.nice_step(float(np.median(rng_d)) / VP.ROWS_PERIOD)
    a1h = atr1h(m1)
    t = m1.index.as_unit("ns").asi8
    lo, hi, cl = m1["l"].to_numpy(), m1["h"].to_numpy(), m1["c"].to_numpy()
    out = []
    keys = list(days.groups.keys())
    for D, nxt in zip(keys[:-1], keys[1:]):
        if nxt - D != pd.Timedelta(days=1):
            continue
        g = days.get_group(D)
        p = VP.build(prof_bars(g), step)
        j = a1h.index.searchsorted(nxt, "right") - 1
        if p is None or j < 0 or not np.isfinite(a1h.iloc[j]):
            continue
        d = float(a1h.iloc[j])
        i0, i1 = np.searchsorted(t, nxt.value), np.searchsorted(t, (nxt + pd.Timedelta(days=1)).value)
        if i1 - i0 < 600:
            continue
        lv = [("POC", p["poc"]), ("VAH", p["vah"]), ("VAL", p["val"])]
        lv += [("випадковий", x) for x in rng.uniform(g["l"].min(), g["h"].max(), RANDOM_PER_DAY)]
        for kind, L in lv:
            r = touch(t, lo, hi, cl, L, d, i0, i1, len(t))
            if r:
                out.append({"base": base, "test": "день", "day": nxt, "level": kind, "L": L, "d": d, "d_pct": d / L, **r})
    return out


def impulse_events(base: str, m1: pd.DataFrame, pumps: list, rng) -> list[dict]:
    if m1.empty:
        return []
    a1h = atr1h(m1)
    t = m1.index.as_unit("ns").asi8
    lo, hi, cl = m1["l"].to_numpy(), m1["h"].to_numpy(), m1["c"].to_numpy()
    out = []
    for start, end in pumps:
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        w = m1.loc[s:e - pd.Timedelta(minutes=1)]
        if len(w) < 30:
            continue
        b5 = K.resample(w, "5min")
        rng_p = b5["h"].max() - b5["l"].min()
        p = VP.build(prof_bars(b5), VP.nice_step(rng_p / VP.ROWS_RANGE))
        j = a1h.index.searchsorted(e, "right") - 1
        if p is None or j < 0 or not np.isfinite(a1h.iloc[j]):
            continue
        d = float(a1h.iloc[j])
        i0, i1 = np.searchsorted(t, e.value), np.searchsorted(t, (e + IMPULSE_WINDOW).value)
        lv = [("POC імпульсу", p["poc"])] + [("випадковий (імпульс)", x) for x in rng.uniform(b5["l"].min(), b5["h"].max(), RANDOM_PER_IMPULSE)]
        for kind, L in lv:
            r = touch(t, lo, hi, cl, L, d, i0, i1, len(t))
            if r:
                out.append({"base": base, "test": "імпульс", "day": e.floor("D"), "level": kind, "L": L, "d": d, "d_pct": d / L, **r})
    return out


def summary(d: pd.DataFrame) -> dict:
    n = len(d)
    vc = d["result"].value_counts()
    b, p_ = vc.get("відскок", 0), vc.get("прохід", 0)
    if b + p_ == 0:
        return {"подій": n}
    p = b / (b + p_)
    se = np.sqrt(p * (1 - p) / (b + p_))
    fee_r = (FEE / d["d_pct"]).median()
    return {"подій": n, "монет": d["base"].nunique(), "відскок": b, "прохід": p_, "час": vc.get("час", 0), "нечітко": vc.get("нечітко", 0),
            "p відскоку": f"{p:.1%} ± {1.96 * se:.1%}", "EV, R": round(2 * p - 1, 3), "комісія, R": round(fee_r, 3),
            "d, %": round(d["d_pct"].median() * 100, 2)}


def table(rows: list[dict]) -> str:
    cols = ["група", "подій", "монет", "відскок", "прохід", "час", "нечітко", "p відскоку", "EV, R", "комісія, R", "d, %"]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Рівні профілю обсягу (POC / VAH / VAL дня, POC імпульсу) проти випадкових рівнів.")
    ap.add_argument("--limit", type=int, default=0, help="лише перші N монет (перевірка)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    from load_pumps import read_pumps
    pumps = {}
    for r in read_pumps():
        if r[3] == "pump":
            pumps.setdefault(r[0], []).append((r[1], r[2]))
    rng = np.random.default_rng(SEED)
    rows = []
    bases = K.available()[: args.limit or None]
    for n, base in enumerate(bases, 1):
        m1 = K.load(base)
        if m1.empty:
            continue
        rows += day_events(base, m1, rng)
        rows += impulse_events(base, m1, pumps.get(base, []), rng)
        if n % 20 == 0:
            print(f"  {n}/{len(bases)} монет, подій {len(rows)}", flush=True)
    ev = pd.DataFrame(rows)
    ev["period"] = np.where(pd.to_datetime(ev["day"], utc=True) < SPLIT, "A", "B")
    os.makedirs("reports", exist_ok=True)
    ev.to_csv(OUT_CSV, index=False)
    md = ["# Рівні профілю обсягу проти випадкових рівнів", "",
          f"Монет: {ev['base'].nunique()}, 1m Binance, квітень–вересень 2026. Профіль — з 5m (як на графіку); рівень «заряджено» відходом "
          "на d = ATR(14) 1h, результат — відскок на d чи прохід на d крізь рівень за 24 год після першого дотику.", "",
          "**EV, R** — очікування угоди «ліміт на рівні, стоп і ціль по d» без комісії (2p − 1); **комісія, R** — 0.1% у частках d.", ""]
    day = ev[ev["test"] == "день"]
    for side in ("зверху", "знизу"):
        x = day[day["side"] == side]
        rows_ = [{"група": f"{lv} · {side}", **summary(x[x["level"] == lv])} for lv in ("POC", "VAH", "VAL", "випадковий")]
        md += [f"## Рівні попереднього дня — дотик {side} ({'підтримка' if side == 'зверху' else 'опір'})", "", table(rows_), ""]
    per = []
    for p in "AB":
        x = day[day["period"] == p]
        for lv in ("POC", "VAH", "VAL", "випадковий"):
            per.append({"група": f"{lv} · {'квіт–черв' if p == 'A' else 'лип–вер'}", **summary(x[x["level"] == lv])})
    md += ["## За періодами (обидва боки разом)", "", table(per), ""]
    imp = ev[ev["test"] == "імпульс"]
    if len(imp):
        md += ["## Імпульси з розмітки: POC імпульсу після його кінця (72 год)", "",
               table([{"група": f"{lv} · {sd}", **summary(imp[(imp["level"] == lv) & (imp["side"] == sd)])}
                      for lv in ("POC імпульсу", "випадковий (імпульс)") for sd in ("зверху", "знизу")]), ""]
    open(OUT_MD, "w", encoding="utf-8").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
