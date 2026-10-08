"""
experiments/wake_channel.py — канал сканера «Прокинулась» (2026-10-08).

Питання власника: OGN 08.10 (+90%) пройшов повз сканер — середній оборот за 30 днів 2.7 млн $ (поріг 20 млн), хоча
о 08:30 UTC обсяг ×46, OI +68%, а за добу оборот 352 млн. Серед його системних лонгів 28 угод (+107 USDT) — саме на таких
«тихих» монетах. Ідея: другий канал сканера — монети з середнім оборотом < 20 млн, які ЗАРАЗ ліквідні й різко ожили.

Правило каналу (на закритті 15m-бару, вікно 4 год, як у scanner.metrics_from_bars):
  середній оборот за 30 завершених днів < 20 млн (як scanner.liquidity) І темп обороту (обсяг 4 год × 6) ≥ pace І
  обсяг 4 год ≥ vol_x × медіани 4-год обсягу за 7 днів до цього І ціна за 4 год ≥ price% І OI (USD) за 4 год ≥ oi%.
Подія — перше спрацювання; повторне — лише після 48 год без спрацювань (TTL авто-списку, scanner.entries_on_history).

Результат події (вхід — закриття бару спрацювання): максимум ціни за 24 год (≥ +10 / 20 / 30%), просідання до +10%,
ціна через 4 / 24 год, «уже пройшло» від мінімуму 24 год до входу, частка руху до входу. База — випадкові бари тих самих
монет у тиші (середній < 20 млн).

Дані: монети, що торгуються зараз (делістовані за місяць не враховано), 15m-свічки REST, OI 15m REST — лише ~30 днів;
угоди власника (травень–вересень) — OI з архіву data.binance.vision (metrics).

    python experiments/wake_channel.py           # 30 днів усього ринку → reports/wake_channel.md (кеш data/wake_channel/)
    python experiments/wake_channel.py --owner   # + чи ловить правило угоди власника на тихих монетах
"""

import argparse
import os
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

import binance_archive as BA
import exchanges as X

DIR = os.path.join("data", "wake_channel")
OUT = os.path.join("reports", "wake_channel.md")
LIQ = 20e6                  # межа ліквідних (канал 1, «Баланс»)
BPH = 4                     # 15m-барів на годину
W = 4 * BPH                 # вікно 4 год
DAY = 24 * BPH
TTL = 48 * BPH              # повторна подія — після 48 год без спрацювань
NOW = pd.Timestamp.now(tz="UTC").floor("15min")
OI_DAYS = 29.5              # /futures/data/openInterestHist — лише останні ~30 днів
START = (NOW - pd.Timedelta(days=OI_DAYS)).ceil("1D")
LOAD_FROM = START - pd.Timedelta(days=8)       # 7 днів норми обсягу + запас
VARIANTS = {                                   # назва: (pace, vol_x, price%, oi% або None)
    "A · ×10, +5%, OI +20%": (20e6, 10, 5, 20),
    "B · м'якше ×5, +3%, OI +10%": (20e6, 5, 3, 10),
    "C · ×10, +5%, без OI": (20e6, 10, 5, None),
    "D · жорсткіше ×20, +10%, OI +30%": (20e6, 20, 10, 30),
    "E · як A, темп ≥ 10 млн": (10e6, 10, 5, 20),
}


def _cache(name: str, fn):
    path = os.path.join(DIR, name + ".pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    try:
        v = fn()
    except RuntimeError:                # ліміт / збій біржі на одній монеті — пропустити її, не кешуючи
        return None
    if v is None:
        return None
    os.makedirs(DIR, exist_ok=True)
    pickle.dump(v, open(path + ".tmp", "wb"))
    os.replace(path + ".tmp", path)
    return v


def symbols() -> list[str]:
    info = X.get(X.BINANCE + "/fapi/v1/exchangeInfo", {})
    return [s["symbol"] for s in info["symbols"] if s.get("contractType") == "PERPETUAL" and s.get("quoteAsset") == "USDT"
            and s.get("status") == "TRADING" and s.get("underlyingType") != "TRADIFI_PERPETUAL"]


def daily(sym: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Денний оборот у $ (quote volume) за [start, end)."""
    rows, t, e = [], int(start.timestamp() * 1000), int(end.timestamp() * 1000)
    while t < e:
        page = X.get(X.BINANCE + "/fapi/v1/klines", dict(symbol=sym, interval="1d", startTime=t, endTime=e, limit=1000))
        if not page:
            break
        rows += page
        t = page[-1][0] + 86_400_000
    return pd.Series([float(r[7]) for r in rows], index=pd.to_datetime([r[0] for r in rows], unit="ms", utc=True), dtype=float)


def avg30(d: pd.Series) -> pd.Series:
    """Середній денний оборот за 30 ЗАВЕРШЕНИХ днів до дня (мінімум 7) — як scanner.liquidity у живому скані."""
    return d.shift(1).rolling(30, min_periods=7).mean()


def klines15(sym: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    rows, t, e = [], int(start.timestamp() * 1000), int(end.timestamp() * 1000)
    while t < e:
        page = X.get(X.BINANCE + "/fapi/v1/klines", dict(symbol=sym, interval="15m", startTime=t, endTime=e, limit=1500))
        if not page:
            break
        rows += page
        t = page[-1][0] + 900_000
    rows = [r for r in rows if int(r[6]) < int(NOW.timestamp() * 1000)]      # лише закриті
    return pd.DataFrame({"h": [float(r[2]) for r in rows], "l": [float(r[3]) for r in rows], "c": [float(r[4]) for r in rows],
                         "q": [float(r[7]) for r in rows]},
                        index=pd.to_datetime([r[0] for r in rows], unit="ms", utc=True)).sort_index().pipe(lambda x: x[~x.index.duplicated()])


def oi15_rest(sym: str) -> pd.Series:
    """OI у $ на 15m-сітці; знімок на T = закриття бару, відкритого о T−15 хв (як scanner._symbol_metrics)."""
    df = X.binance_hist("/futures/data/openInterestHist", sym, int(START.timestamp() * 1000) - 3_600_000,
                        int(NOW.timestamp() * 1000), "sumOpenInterestValue", "15m")
    return pd.Series(df["value"].to_numpy(), index=pd.DatetimeIndex(df["ts"]) - pd.Timedelta(minutes=15), dtype=float)


def oi15_archive(sym: str, days: list[str]) -> pd.Series:
    """Те саме з архіву (metrics: create_time на 5 хв раніше за мітку API, лише межі 15m)."""
    parts = []
    for d in days:
        m = BA._csv("metrics", sym, d)
        if len(m):
            t = pd.to_datetime(m["create_time"], utc=True) + pd.Timedelta(minutes=5)
            s = pd.Series(m["sum_open_interest_value"].astype(float).to_numpy(), index=t)
            parts.append(s[(s.index.minute % 15 == 0) & (s > 0)])
    if not parts:
        return pd.Series(dtype=float)
    s = pd.concat(parts).sort_index()
    s = s[~s.index.duplicated()]
    s.index = s.index - pd.Timedelta(minutes=15)
    return s


def frame(k: pd.DataFrame, oi: pd.Series, d: pd.Series) -> pd.DataFrame:
    h = k.copy()
    h["oi"] = oi.reindex(h.index)
    h["avg30"] = avg30(d).reindex(h.index.normalize()).to_numpy()
    v4 = h["q"].rolling(W).sum()
    norm = v4.shift(W).rolling(7 * DAY, min_periods=3 * DAY - W).median()
    h["vol_x"] = v4 / norm
    h["pace"] = v4 * 6
    h["px4"] = (h["c"] / h["c"].shift(W) - 1) * 100
    h["oi4"] = (h["oi"] / h["oi"].shift(W) - 1) * 100
    return h


def trig(h: pd.DataFrame, v: tuple) -> pd.Series:
    pace, vx, px, oix = v
    ok = (h["avg30"] < LIQ) & (h["pace"] >= pace) & (h["vol_x"] >= vx) & (h["px4"] >= px)
    if oix is not None:
        ok &= h["oi4"] >= oix
    return ok.fillna(False)


def events(ok: pd.Series) -> list[int]:
    """Позиції барів-подій: перше спрацювання після ≥ 48 год без спрацювань."""
    out, last = [], -10 ** 9
    for i in np.flatnonzero(ok.to_numpy()):
        if i - last > TTL:
            out.append(int(i))
        last = i
    return out


def outcome(h: pd.DataFrame, i: int) -> dict:
    p0 = h["c"].iat[i]
    fut = h.iloc[i + 1:i + 1 + DAY]
    full = len(fut) == DAY
    hi = fut["h"].to_numpy() / p0 - 1
    lo = fut["l"].to_numpy() / p0 - 1
    k10 = np.flatnonzero(hi >= 0.10)
    past = h.iloc[max(0, i - DAY + 1):i + 1]
    low24 = past["l"].min()
    peak = h["h"].iloc[max(0, i - DAY + 1):i + 1 + 3 * DAY].max()
    return {"ts": h.index[i] + pd.Timedelta(minutes=15), "full": full, "p0": p0,
            "mfe": hi.max() if len(hi) else np.nan, "hit10": bool(len(k10)), "hit20": bool((hi >= 0.2).any()),
            "hit30": bool((hi >= 0.3).any()),
            "t10_h": (k10[0] + 1) / BPH if len(k10) else np.nan,
            "mae10": lo[:k10[0] + 1].min() if len(k10) else (lo.min() if len(lo) else np.nan),
            "r4": h["c"].iat[i + W] / p0 - 1 if i + W < len(h) else np.nan,
            "r24": h["c"].iat[i + DAY] / p0 - 1 if i + DAY < len(h) else np.nan,
            "already": p0 / low24 - 1, "share": (p0 - low24) / (peak - low24) if peak > low24 else np.nan,
            "vol_x": h["vol_x"].iat[i], "px4": h["px4"].iat[i], "oi4": h["oi4"].iat[i], "pace": h["pace"].iat[i],
            "avg30": h["avg30"].iat[i]}


def load_coin(sym: str, d: pd.Series) -> pd.DataFrame | None:
    def get():
        k = klines15(sym, LOAD_FROM, NOW)
        return (k, oi15_rest(sym)) if len(k) else None
    x = _cache(f"{sym}_{NOW:%Y%m%d}", get)
    return frame(x[0], x[1], d) if x else None


def stats(df: pd.DataFrame, days: float) -> dict:
    f = df[df["full"]]
    return {"подій": len(df), "на день": len(df) / days, "монет": df["sym"].nunique(), "повних": len(f),
            "+10%": f["hit10"].mean(), "+20%": f["hit20"].mean(), "+30%": f["hit30"].mean(), "MFE мед.": f["mfe"].median(),
            "до +10 год": f["t10_h"].median(), "просід. до +10%": f.loc[f["hit10"], "mae10"].median(),
            "4 год": f["r4"].median(), "24 год": f["r24"].median(), "уже пройшло": f["already"].median(),
            "частка руху": f["share"].median()}


def fmt(v, kind):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return {"n": f"{v:.0f}", "d": f"{v:.1f}", "p": f"{v:.0%}", "s": f"{v:+.1%}", "h": f"{v:.1f}"}[kind]


COLS = [("подій", "n"), ("на день", "d"), ("монет", "n"), ("+10%", "p"), ("+20%", "p"), ("+30%", "p"), ("MFE мед.", "s"),
        ("до +10 год", "h"), ("просід. до +10%", "s"), ("4 год", "s"), ("24 год", "s"), ("уже пройшло", "s"), ("частка руху", "p")]


def table(rows: dict) -> list[str]:
    out = ["| варіант | " + " | ".join(c for c, _ in COLS) + " |", "|---|" + "---|" * len(COLS)]
    for name, s in rows.items():
        out.append(f"| {name} | " + " | ".join(fmt(s[c], k) for c, k in COLS) + " |")
    return out


def market(log=print) -> list[str]:
    syms = symbols()
    log(f"{len(syms)} USDT-перпів; денні обороти…")
    dstart = LOAD_FROM - pd.Timedelta(days=40)
    with ThreadPoolExecutor(2) as pool:
        dd = dict(zip(syms, pool.map(lambda s: _cache(f"d_{s}_{NOW:%Y%m%d}", lambda: daily(s, dstart, NOW + pd.Timedelta(days=1))), syms)))
    dd = {s: d for s, d in dd.items() if d is not None}
    syms = list(dd)
    # кандидати: у вікні хоч один день із середнім < 20 млн
    cand = [s for s in syms if len(dd[s]) and (avg30(dd[s])[dd[s].index >= START] < LIQ).any()]
    log(f"тихих (сер. < 20 млн хоч день у вікні): {len(cand)}; 15m-свічки й OI…")
    with ThreadPoolExecutor(2) as pool:
        frames = dict(zip(cand, pool.map(lambda s: load_coin(s, dd[s]), cand)))
    frames = {s: h for s, h in frames.items() if h is not None}
    days = (NOW - START) / pd.Timedelta(days=1)
    rows, ev_all = {}, {}
    for name, v in VARIANTS.items():
        evs = []
        for s, h in frames.items():
            hh_mask = h.index >= START
            ok = trig(h, v) & hh_mask
            evs += [{"sym": s, **outcome(h, i)} for i in events(ok)]
        df = pd.DataFrame(evs)
        ev_all[name] = df
        rows[name] = stats(df, days) if len(df) else {c: np.nan for c, _ in COLS}
    # база: випадкові бари в тиші (середній < 20 млн), ті самі монети й вікно
    rng = np.random.default_rng(42)
    base = []
    for s, h in frames.items():
        idx = np.flatnonzero(((h.index >= START) & (h["avg30"] < LIQ)).to_numpy())
        idx = idx[idx < len(h) - DAY]
        for i in rng.choice(idx, size=min(20, len(idx)), replace=False) if len(idx) else []:
            base.append({"sym": s, **outcome(h, int(i))})
    rows["база · випадкові бари в тиші"] = stats(pd.DataFrame(base), days)
    rows["база · випадкові бари в тиші"].update({"подій": np.nan, "на день": np.nan})
    pd.concat([d.assign(variant=n) for n, d in ev_all.items() if len(d)]).to_csv(os.path.join("reports", "wake_channel_events.csv"), index=False)
    a = ev_all[next(iter(VARIANTS))]
    lines = [f"## Увесь ринок, {START:%Y-%m-%d} – {NOW:%Y-%m-%d %H:%M} UTC ({days:.0f} днів)", "",
             f"{len(syms)} USDT-перпів Binance, що торгуються зараз; тихих (середній оборот за 30 днів < 20 млн хоч день у вікні) — "
             f"{len(frames)}. Рух — максимум ціни за 24 год після закриття бару спрацювання; «+10%» — частка подій, де він ≥ 10% "
             "(лише події з повними 24 год). «Уже пройшло» — від мінімуму 24 год до ціни входу; «частка руху» — яку частину руху "
             "(мінімум 24 год до → максимум 48 год після) пройдено на момент входу.", ""] + table(rows)
    if len(a):
        top = a.sort_values("mfe", ascending=False).head(15)
        lines += ["", "### Варіант A — найбільші рухи", "", "| монета | спрацювання UTC | сер. оборот | темп | обсяг × | ціна 4 год | OI 4 год | уже пройшло | макс. 24 год |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for _, r in top.iterrows():
            lines.append(f"| {r['sym'][:-4]} | {r['ts']:%m-%d %H:%M} | {r['avg30'] / 1e6:.1f} млн | {r['pace'] / 1e6:.0f} млн | ×{r['vol_x']:.0f} | "
                         f"{r['px4']:+.0f}% | {r['oi4']:+.0f}% | {r['already']:+.0%} | {r['mfe']:+.0%} |")
        perday = a.groupby(a["ts"].dt.date).size()
        lines += ["", f"Варіант A по днях: медіана {perday.median():.0f}, максимум {perday.max()} ({perday.idxmax()}), днів без подій — "
                      f"{len(set(pd.date_range(START, NOW, freq='D').date) - set(perday.index))}.", ""]
        o = a[a["sym"] == "OGNUSDT"]
        if len(o):
            r = o.iloc[-1]
            lines.append(f"OGN: спрацювання {r['ts']:%m-%d %H:%M} UTC, ціна {r['p0']:.5f}, уже пройшло {r['already']:+.0%}.")
    return lines


def owner(log=print) -> list[str]:
    """Чи спрацювало б правило до входу власника: його системні лонги на тихих монетах (травень–вересень)."""
    p = pd.read_parquet(os.path.join("reports", "my_trades", "positions.parquet"))
    p = p[(p["minutes"] >= 10) & ~p["revenge"] & ~p["burst"] & (p["side"] == "long")]
    rows = []
    for sym, g in p.groupby("symbol"):
        d = _cache(f"d_{sym}_owner", lambda: daily(sym, pd.Timestamp("2026-03-01", tz="UTC"), pd.Timestamp("2026-10-01", tz="UTC")))
        if d is None:
            continue
        a = avg30(d)
        for _, t in g.iterrows():
            day = t["entry_time"].normalize()
            if not (a.get(day, np.nan) < LIQ):
                continue
            s, e = day - pd.Timedelta(days=9), day + pd.Timedelta(days=1)
            def get():
                k = klines15(sym, s, e)
                return k, oi15_archive(sym, [x.strftime("%Y-%m-%d") for x in pd.date_range(s, e - pd.Timedelta(days=1), freq="D")])
            x = _cache(f"o_{sym}_{day:%Y%m%d}", get)
            if x is None or not len(x[0]):
                continue
            k, oi = x
            if not len(k):
                continue
            h = frame(k, oi, d)
            row = {"sym": sym, "entry": t["entry_time"], "net": t["net"], "avg30": a.get(day), "day_turn": d.get(day)}
            bar = h.index[h.index + pd.Timedelta(minutes=15) <= t["entry_time"]]
            for name, v in VARIANTS.items():
                ok = trig(h, v)
                ok = ok[(ok.index >= t["entry_time"] - pd.Timedelta(hours=24)) & (ok.index.isin(bar))]
                hit = ok[ok]
                row[name] = (t["entry_time"] - (hit.index[0] + pd.Timedelta(minutes=15))) / pd.Timedelta(hours=1) if len(hit) else np.nan
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join("reports", "wake_channel_owner.csv"), index=False)
    woke = df[df["day_turn"] >= LIQ]
    lines = ["## Угоди власника на тихих монетах (травень–вересень, OI з архіву)", "",
             f"Системні лонги (утримання ≥ 10 хв, без тільту) на монетах із середнім оборотом < 20 млн: {len(df)}; із них у день "
             f"входу оборот ≥ 20 млн — {len(woke)} (net {woke['net'].sum():+.0f} USDT). «Спіймано» — правило спрацювало за 24 год до "
             "входу власника; «за год» — медіана, на скільки раніше.", "",
             "| варіант | спіймано (усі тихі) | з них net | спіймано (оборот дня ≥ 20 млн) | за год до входу (мед.) |", "|---|---|---|---|---|"]
    for name in VARIANTS:
        c, cw = df[name].notna(), woke[name].notna()
        lines.append(f"| {name} | {c.sum()} з {len(df)} | {df.loc[c, 'net'].sum():+.0f} | {cw.sum()} з {len(woke)} | "
                     f"{fmt(df.loc[c, name].median(), 'h')} |")
    return lines + [""]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Канал сканера «Прокинулась»: тихі монети (сер. оборот < 20 млн), що різко ожили — "
                                             "30 днів усього ринку Binance + угоди власника")
    ap.add_argument("--owner", action="store_true", help="ще й перевірити угоди власника на тихих монетах (архів OI)")
    args = ap.parse_args()
    lines = ["# Канал «Прокинулась»: тихі монети, що різко ожили", ""]
    lines += market()
    if args.owner:
        lines += [""] + owner()
    open(OUT, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Звіт: {OUT}")


if __name__ == "__main__":
    main()
