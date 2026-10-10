"""
experiments/auto_label.py — АВТОРОЗМІТКА імпульсів і фальстартів (2026-10-10, власник: «руками важко визначити, де хибний, а де
істинний — подумай, як розмітити автоматично, ітераціями»). Розмітка робиться ЗАДНІМ ЧИСЛОМ, тож клас — вимір, а не думка:
    1) тригер «старт» — лише минуле (як побачила б модель): за останні W барів ціна ↑ ≥ a·ATR, OI ↑ ≥ b%, обсяг ≥ c × медіани;
    2) від спрацювання — рух до кінця за правилом розмітки (label.impulse_end: віддав ⅓ / OI −10%): рух ≥ M → Impulse,
       < m → Failed impulse, між — «сіра зона» (у навчання не йде);
    3) звірка з ручною розміткою власника на його тікерах (знайдено, межі, клас).
Етапи:
    --stage stats   ознаки на старті / через 1 год і рух заднім числом: імпульси власника проти його фальстартів
    --stage match   тригер + клас на тікерах власника → збіг з ручною розміткою (події, межі, зайві, бари)
→ reports/auto_label_stats.md, reports/auto_label_match.md, reports/auto_label_events.csv

    python experiments/auto_label.py --stage stats
    python experiments/auto_label.py --stage match
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import psycopg
from dotenv import load_dotenv

import label as LB
import train
from features import INTERVAL_15M, aggregate, load_pumps, load_raw
from fetch import excluded_tickers
from risk import atr

B = train.BAR_15M
DAY = 96


def series(raw: pd.DataFrame, base: str) -> pd.DataFrame:
    """15m-ряд монети: open/high/low/price/volume, oi (USD, масштаб довільний — важать лише відношення), atr."""
    a = aggregate(raw, base, "15min")
    h = a[["high", "low", "price", "volume"]].copy()
    h["oi"] = np.exp(a["oi_coin_log_idx"]) * a["price"]
    h["atr"] = atr(h)
    h["vol_med"] = h["volume"].rolling(7 * DAY, min_periods=DAY).median().shift(1)
    return h.dropna(subset=["price", "high", "low"])


def outcome(h: pd.DataFrame, start: pd.Timestamp, after: pd.Timestamp | None = None) -> dict:
    """Рух заднім числом від бару старту: кінець за правилом розмітки, пік, рух у % і ATR."""
    fw = h[(h.index >= start) & (h.index < start + pd.Timedelta(hours=72))]
    if len(fw) < 4:
        return {}
    end = LB.impulse_end(fw, after or start)
    stop = (end - B) if end is not None else fw.index[-1]
    w = fw.loc[:stop]
    p0, a0 = fw["low"].iloc[0], h["atr"].shift(1).get(start, np.nan)
    pk = w["high"].max()
    return {"end": end, "peak_ts": w["high"].idxmax(), "move_pct": pk / p0 - 1, "move_atr": (pk - p0) / a0 if a0 else np.nan,
            "dur_h": ((end or fw.index[-1]) - start) / pd.Timedelta(hours=1)}


def early(h: pd.DataFrame, t: pd.Timestamp, bars: int = 4) -> dict:
    """Стан на закритті бару t за останні bars барів (лише минуле)."""
    i = h.index.get_indexer([t])[0]
    if i < bars + DAY:
        return {}
    w, prev = h.iloc[i - bars + 1:i + 1], h.iloc[i - bars]
    a0 = prev["atr"]
    return {"ret_atr": (w["price"].iloc[-1] - prev["price"]) / a0, "oi_pct": w["oi"].iloc[-1] / prev["oi"] - 1,
            "vol_x": w["volume"].sum() / (bars * prev["vol_med"]) if prev["vol_med"] else np.nan}


def stats(conn) -> None:
    excl = excluded_tickers(".")
    pumps = load_pumps(conn)
    pumps = pumps[~pumps["base_asset"].isin(excl) & pumps["pattern_type"].isin(["pump", "false_start"])]
    raw = load_raw(conn, sorted(pumps["base_asset"].unique()), INTERVAL_15M, exchanges=["Binance"])
    rows = []
    for b, g in pumps.groupby("base_asset"):
        h = series(raw, b)
        if h.empty:
            continue
        for r in g.itertuples():
            s = h.index[h.index >= r.start_ts.floor("15min")]
            if not len(s):
                continue
            s0 = s[0]
            o = outcome(h, s0)
            own_end = r.end_ts
            wo = h[(h.index >= s0) & (h.index < own_end)]
            own_move = wo["high"].max() / h.at[s0, "low"] - 1 if len(wo) else np.nan
            row = {"base": b, "type": r.pattern_type, "start": s0, "own_move_pct": own_move,
                   "own_dur_h": (own_end - r.start_ts) / pd.Timedelta(hours=1), **o}
            for k in (1, 2, 4):           # стан через k барів після старту
                t = s0 + (k - 1) * B
                if t in h.index:
                    row |= {f"{n}_{k}": v for n, v in early(h, t, k).items()}
            rows.append(row)
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join("reports", "auto_label_stats.csv"), index=False)
    q = lambda s: f"{s.quantile(0.1):.2f} / {s.median():.2f} / {s.quantile(0.9):.2f}"  # noqa: E731
    cols = ["own_move_pct", "own_dur_h", "move_pct", "move_atr", "dur_h", "ret_atr_1", "oi_pct_1", "vol_x_1",
            "ret_atr_2", "oi_pct_2", "vol_x_2", "ret_atr_4", "oi_pct_4", "vol_x_4"]
    L = ["# Авторозмітка, етап 1: старт і рух заднім числом (розмітка власника)", "",
         f"Імпульсів {int((d['type'] == 'pump').sum())}, фальстартів {int((d['type'] == 'false_start').sum())}. "
         "Значення — 10-й перцентиль / медіана / 90-й. *_k — стан на закритті k-го бару від старту (за останні k барів): "
         "ret_atr — ціна в ATR(15m), oi_pct — OI, vol_x — обсяг ÷ медіана 7 днів. move_* — рух від старту до кінця за правилом "
         "розмітки (impulse_end), own_* — за межами власника.", "",
         "| величина | імпульси | фальстарти |", "|---|---|---|"]
    for c in cols:
        L.append(f"| {c} | {q(d.loc[d['type'] == 'pump', c])} | {q(d.loc[d['type'] == 'false_start', c])} |")
    # наскільки рух заднім числом розділяє класи
    L += ["", "## Чи розділяє рух заднім числом імпульси і фальстарти", "", "| поріг | імпульсів ≥ | фальстартів ≥ |", "|---|---|---|"]
    for col, ths in (("move_pct", (0.03, 0.05, 0.08, 0.12)), ("move_atr", (2, 3, 4, 6, 8))):
        for t in ths:
            L.append(f"| {col} ≥ {t} | {(d.loc[d['type'] == 'pump', col] >= t).mean():.0%} | "
                     f"{(d.loc[d['type'] == 'false_start', col] >= t).mean():.0%} |")
    out = os.path.join("reports", "auto_label_stats.md")
    open(out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


IMP_ATR, IMP_DUR = 8.0, 1.5          # імпульс заднім числом: рух ≥ 8 ATR і триває ≥ 1.5 год (етап 1: 84% імпульсів, 21% фальстартів)
GREY_ATR, GREY_DUR = 6.0, 1.0         # сіра зона: 6–8 ATR при тривалості ≥ 1 год або ≥ 8 ATR при 1–1.5 год — у навчання не йде
LOOKBACK = pd.Timedelta(hours=3)      # старт — мінімум low за 3 год до тригера
TRIGGERS = {"T1": (2.0, 0.02, 2.0), "T2": (1.5, 0.02, 2.0), "T3": (2.0, 0.0, 3.0), "T4": (3.0, 0.03, 3.0)}   # ціна ATR, OI, обсяг × за 1 год


def trigger_mask(h: pd.DataFrame, a: float, b: float, c: float, bars: int = 4) -> np.ndarray:
    """Тригер на закритті бару (лише минуле): за останні bars барів ціна ≥ a·ATR, OI ≥ +b, обсяг ≥ c × медіани 7 днів."""
    a0 = h["atr"].shift(bars)
    ret = (h["price"] - h["price"].shift(bars)) / a0
    oi = h["oi"] / h["oi"].shift(bars) - 1
    vx = h["volume"].rolling(bars).sum() / (bars * h["vol_med"].shift(bars - 1))
    return ((ret >= a) & (oi >= b) & (vx >= c)).to_numpy()


def classify(o: dict) -> str:
    m, dur = o.get("move_atr", np.nan), o.get("dur_h", np.nan)
    if not np.isfinite(m):
        return "none"
    if m >= IMP_ATR and dur >= IMP_DUR:
        return "pump"
    if (GREY_ATR <= m < IMP_ATR and dur >= GREY_DUR) or (m >= IMP_ATR and GREY_DUR <= dur < IMP_DUR):
        return "grey"
    return "failed"


def auto_events(h: pd.DataFrame, trig: tuple) -> pd.DataFrame:
    """Події на всьому ряду монети: тригер → старт (мінімум low за 3 год) → кінець за правилом розмітки → клас.
    Наступний тригер — лише після кінця попередньої події."""
    m = trigger_mask(h, *trig)
    idx = h.index
    out, busy_until = [], None
    for i in np.flatnonzero(m):
        t = idx[i]
        if busy_until is not None and t < busy_until:
            continue
        w = h[(idx >= t - LOOKBACK) & (idx <= t)]
        start = w["low"].idxmin()
        o = outcome(h, start, after=t)
        if not o:
            continue
        cls = classify(o)
        if o["end"] is None:                 # кінця за 72 год немає (рух < 5% без обвалу OI або довгий тренд) — межа невідома
            end = start + pd.Timedelta(hours=o["dur_h"]) + B
            cls = "grey" if cls == "pump" else cls
        else:
            end = o["end"]
        ev = {"trigger": t, "start": start, "end_ts": end, "cls": cls, **{k: o[k] for k in ("move_pct", "move_atr", "dur_h")}}
        if out and start <= out[-1]["start"]:    # той самий рух після «фальстарту» — переоцінка пізнішим тригером
            out[-1] = ev
        else:
            out.append(ev)
        busy_until = end if cls in ("pump", "grey") else t + pd.Timedelta(hours=1)
    return pd.DataFrame(out)


def bar_labels(h: pd.DataFrame, e: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Мітки барів з подій: (імпульс, сіра зона) — правило бару як features.in_pump (закриття бару всередині)."""
    close = h.index + B
    a_lab, grey = np.zeros(len(h), bool), np.zeros(len(h), bool)
    for r in e.itertuples():
        m = (close > r.start) & (close <= r.end_ts)
        if r.cls == "pump":
            a_lab |= m
        elif r.cls == "grey":
            grey |= m
    return a_lab, grey & ~a_lab


def match(conn) -> None:
    """Тригер + клас на тікерах власника → збіг з ручною розміткою (події і бари)."""
    from fetch import read_tickers_file
    excl = excluded_tickers(".")
    pumps = load_pumps(conn)
    pumps = pumps[~pumps["base_asset"].isin(excl)]
    reviewed = {t.upper() for t in read_tickers_file("tickers.txt")} - excl
    bases = sorted(reviewed | set(pumps["base_asset"]))
    raw = load_raw(conn, bases, INTERVAL_15M, exchanges=["Binance"])
    H = {b: series(raw, b) for b in bases}
    H = {b: h for b, h in H.items() if len(h) > 2 * DAY}
    own_p = pumps[pumps["pattern_type"] == "pump"]
    own_f = pumps[pumps["pattern_type"] == "false_start"]
    L = ["# Авторозмітка, етап 2: тригер + клас на тікерах власника проти ручної розмітки", "",
         f"Клас заднім числом: імпульс — рух ≥ {IMP_ATR:g} ATR і ≥ {IMP_DUR:g} год; сіра зона — {GREY_ATR:g}–{IMP_ATR:g} ATR "
         f"(≥ {GREY_DUR:g} год) або ≥ {IMP_ATR:g} ATR при {GREY_DUR:g}–{IMP_DUR:g} год; решта — фальстарт / шум. "
         "Старт — мінімум low за 3 год до тригера.", "",
         "«знайдено» — авто-імпульс перетинає імпульс власника; межі — медіана |різниці| старту / кінця (хв) серед знайдених; "
         "«фс → імпульс» — фальстарти власника, які авто назвала імпульсом; «зайві / 30 дн» — авто-імпульси на ПЕРЕГЛЯНУТИХ тікерах "
         "(tickers.txt), що не перетинають жодної розмітки власника. Бари — точність / повнота міток «1» на переглянутих тікерах "
         "(сіра зона не рахується).", "",
         "| тригер (ціна ATR, OI, обсяг ×) | подій | імпульсів | знайдено | старт, хв | кінець, хв | фс → імпульс | зайві / 30 дн | "
         "бари: точність | повнота |", "|---|---|---|---|---|---|---|---|---|---|"]
    allev = []
    for name, trig in TRIGGERS.items():
        ev, pn, pd_, rn, rd, days = [], 0, 0, 0, 0, 0.0
        for b, h in H.items():
            e = auto_events(h, trig)
            if e.empty:
                continue
            ev.append(e.assign(base=b))
            if b in reviewed:
                close = h.index + B
                o_lab = np.zeros(len(h), bool)
                for r in own_p[own_p["base_asset"] == b].itertuples():
                    o_lab |= (close > r.start_ts) & (close <= r.end_ts)
                a_lab, grey = bar_labels(h, e)
                pn += (a_lab & o_lab).sum()
                pd_ += a_lab.sum()
                rn += (a_lab & o_lab).sum()
                rd += (o_lab & ~grey).sum()
                days += len(h) / DAY
        ev = pd.concat(ev, ignore_index=True).assign(trig=name)
        allev.append(ev)
        imp = ev[ev["cls"] == "pump"]

        def hits(r):
            return imp[(imp["base"] == r.base_asset) & (imp["start"] < r.end_ts) & (imp["end_ts"] > r.start_ts)]
        found, ds, de = 0, [], []
        for r in own_p.itertuples():
            x = hits(r)
            if len(x):
                found += 1
                ds.append(abs((x["start"].iloc[0] - r.start_ts) / pd.Timedelta(minutes=1)))
                de.append(abs((x["end_ts"].iloc[-1] - r.end_ts) / pd.Timedelta(minutes=1)))
        fs_imp = sum(bool(len(hits(r))) for r in own_f.itertuples())
        extra = 0
        for r in imp[imp["base"].isin(reviewed)].itertuples():
            own = pumps[pumps["base_asset"] == r.base]
            extra += not ((own["start_ts"] < r.end_ts) & (own["end_ts"] > r.start)).any()
        L.append(f"| {name} {trig} | {len(ev)} | {len(imp)} | {found}/{len(own_p)} | {np.median(ds):.0f} | {np.median(de):.0f} | "
                 f"{fs_imp}/{len(own_f)} | {extra / days * 30:.1f} | {pn / max(pd_, 1):.0%} | {rn / max(rd, 1):.0%} |")
        print(L[-1], flush=True)
    pd.concat(allev, ignore_index=True).to_csv(os.path.join("reports", "auto_label_events.csv"), index=False)
    open(os.path.join("reports", "auto_label_match.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Авторозмітка імпульсів і фальстартів заднім числом: статистика і звірка з власником")
    ap.add_argument("--stage", choices=["stats", "match"], default="stats",
                    help="stats — ознаки на старті і рух заднім числом; match — тригер + клас проти ручної розмітки")
    args = ap.parse_args()
    load_dotenv(".env")
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        if args.stage == "stats":
            stats(conn)
        else:
            match(conn)


if __name__ == "__main__":
    main()
