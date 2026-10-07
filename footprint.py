"""
footprint.py — футпрінт 15m-барів і патерни b / p («залочка», 2026-10-07). Один модуль для експериментів, скринера,
дашборду і бота — правило й результат мітки визначені лише тут.

Футпрінт: обсяг агресивних купівель / продажів на рівнях ціни; рівень = floor(ln ціни / ln 1.001) — одна сітка через 0.1%
для всіх барів і монет, тож 15m-бар = сума трьох 5m (trade_recorder) або день з архіву (experiments/flow_archive.py).

Патерни (правило зафіксоване в experiments/spring_pattern.py ДО прогону, тут — без змін):
  b  «залочка» на лоу: лоу бару нижче за лоу попередніх 2 год (L); під L — ≥ 30% обсягу бару; закриття вище L; у верхній
     третині діапазону бару ≤ 20% обсягу (обсяг унизу — форма b); обсяг бару ≥ 2× медіани за попередню добу.
     Ціна проколює лоу, там стопи / ліквідації / шорти на пробій, але великий покупець стоїть лімітками — ціна вертається.
  p  дзеркально на хаях (над H ≥ 30%, закриття нижче H, у нижній третині ≤ 20%) — покупців на хаях поглинули.
Сам по собі патерн — шум (≈ випадковий бар, spring_pattern). У контексті моделі — Кандидат / Alert за останні 6 год —
b дав +2R раніше за стоп у 38% проти 29% у входу моделі (spring_context, 218 монет поза навчанням, 07.10). Тому скринер
шукає b / p лише в контексті; мітка — підказка для ручного входу (стоп під лоу бару b), модель не змінюється.

Живі дані: OHLC і обсяг — 15m-свічки Binance у базі (fetch_exchange), футпрінт — trade_bars (trade_recorder.py, Binance).
"""

import math
import os

import numpy as np
import pandas as pd

LVL = math.log(1.001)
LOOK = 8                 # попередній діапазон — 2 год (8 барів 15m)
BEYOND_MIN = 0.3         # частка обсягу бару за межею (під лоу / над хаєм)
EDGE_MAX = 0.2           # частка обсягу на протилежному краї бару (верхня третина для b, нижня для p)
VOL_X = 2.0              # обсяг бару проти медіани попередньої доби
CTX_BARS = 24            # контекст: Кандидат / Alert за останні 6 год (включно з поточним баром)
HORIZON = 48             # результат мітки — 12 год
FEE = 0.001
STOP_PAD = 0.001         # стоп на 0.1% за лоу / хаєм бару
BAR = pd.Timedelta(minutes=15)
ROOT = os.path.dirname(os.path.abspath(__file__))
SIGNALS_STAMP = os.path.join(ROOT, "reports", "signals_stamp")     # як screener.SIGNALS_STAMP: дашборд перечитує список

DDL = """
CREATE TABLE IF NOT EXISTS fp_marks (
    base_asset  TEXT NOT NULL,
    ts          TIMESTAMPTZ NOT NULL,        -- відкриття 15m-бару з патерном
    kind        TEXT NOT NULL,               -- b (залочка на лоу) / p (на хаях)
    close       DOUBLE PRECISION,            -- закриття бару = ціна входу підказки
    stop        DOUBLE PRECISION,            -- b: під лоу бару, p: над хаєм (STOP_PAD)
    level       DOUBLE PRECISION,            -- пробитий лоу (b) / хай (p) попередніх 2 год
    beyond      DOUBLE PRECISION,            -- частка обсягу бару за рівнем
    beyond_delta DOUBLE PRECISION,           -- дельта за рівнем (купівлі − продажі), частка обсягу бару
    edge        DOUBLE PRECISION,            -- частка обсягу на протилежному краї бару
    vol_x       DOUBLE PRECISION,            -- обсяг бару / медіана доби
    ctx         TEXT,                        -- alert / candidate — контекст моделі за 6 год
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    live        BOOLEAN NOT NULL DEFAULT true,  -- false — знайдено пізніше на історії (дозаповнення з архіву, trade_backfill.py)
    PRIMARY KEY (base_asset, ts, kind)
);
ALTER TABLE fp_marks ADD COLUMN IF NOT EXISTS live BOOLEAN NOT NULL DEFAULT true;
"""


def level(price: float) -> int:
    return int(math.floor(math.log(price) / LVL))


# ---------------------------------------------------------------- правило

def bar_metrics(b: pd.DataFrame, fp: pd.DataFrame) -> pd.DataFrame:
    """b: 15m-бари (open, high, low, price; vol, delta — або v_buy, v_sell), індекс ts. fp: рядки (ts, lvl, buy, sell).
    Додає: L / H (лоу / хай попередніх 2 год), vol_ratio, atr, частки обсягу під L / над H, дельту там, краї бару."""
    from risk import atr
    b = b.copy()
    if "vol" not in b:
        b["vol"] = b["v_buy"] + b["v_sell"]
    if "delta" not in b and "v_buy" in b:
        b["delta"] = b["v_buy"] - b["v_sell"]
    b["L"] = b["low"].shift(1).rolling(LOOK, min_periods=LOOK).min()
    b["H"] = b["high"].shift(1).rolling(LOOK, min_periods=LOOK).max()
    b["vol_ratio"] = b["vol"] / b["vol"].shift(1).rolling(96, min_periods=48).median()
    b["atr"] = atr(b)
    shares = ["below_share", "below_delta", "above_share", "above_delta", "top_share", "bot_share"]
    if fp.empty:                                   # футпрінту ще немає (записувач щойно запущено) — патернів не буде
        return b.assign(**{c: np.nan for c in shares})
    f = fp.merge(b[["L", "H", "low", "high"]], left_on="ts", right_index=True).reset_index(drop=True)
    f["p_lo"], f["p_hi"] = np.exp(f["lvl"] * LVL), np.exp((f["lvl"] + 1) * LVL)
    f["vol"], f["d"] = f["buy"] + f["sell"], f["buy"] - f["sell"]
    pos = ((f["p_lo"] + f["p_hi"]) / 2 - f["low"]) / (f["high"] - f["low"])
    below, above = f["p_hi"] <= f["L"] * 1.0005, f["p_lo"] >= f["H"] * 0.9995     # рівень (0.1%) майже цілком за межею
    g = lambda m, col: f[col].where(m, 0).groupby(f["ts"]).sum()
    tot = f.groupby("ts")["vol"].sum()
    b["below_share"], b["below_delta"] = g(below, "vol") / tot, g(below, "d") / tot
    b["above_share"], b["above_delta"] = g(above, "vol") / tot, g(above, "d") / tot
    b["top_share"], b["bot_share"] = g(pos >= 2 / 3, "vol") / tot, g(pos <= 1 / 3, "vol") / tot
    return b


def patterns(b: pd.DataFrame) -> dict[str, pd.Series]:
    """Маски патернів b / p на барах з bar_metrics()."""
    ok = b["L"].notna() & b["H"].notna() & b["vol_ratio"].notna()
    vol = b["vol_ratio"] >= VOL_X
    pb = (b["low"] < b["L"]) & (b["below_share"] >= BEYOND_MIN) & (b["price"] > b["L"]) & (b["top_share"] <= EDGE_MAX) & vol
    pp = (b["high"] > b["H"]) & (b["above_share"] >= BEYOND_MIN) & (b["price"] < b["H"]) & (b["bot_share"] <= EDGE_MAX) & vol
    return {"b": pb & ok, "p": pp & ok}


def stop_of(b: pd.DataFrame, i: int, kind: str) -> float:
    return b["low"].iloc[i] * (1 - STOP_PAD) if kind == "b" else b["high"].iloc[i] * (1 + STOP_PAD)


def outcome(b: pd.DataFrame, i: int, stop: float, short: bool = False, horizon: int = HORIZON) -> dict | None:
    """Угода від закриття бару i зі стопом stop: ціна через 1 / 4 / 12 год, +1R / +2R раніше за стоп, R при виході
    «+2R / стоп / horizon барів» з комісією. Неповний горизонт (живі мітки) — done=False, R поки за останнім баром."""
    c, hi, lo = b["price"].to_numpy(), b["high"].to_numpy(), b["low"].to_numpy()
    sgn = -1 if short else 1
    R = sgn * (c[i] - stop)
    if not R > 0:
        return None
    end = min(i + horizon, len(b) - 1)
    r = {f"ret_{h}h": (sgn * (c[i + 4 * h] / c[i] - 1) if i + 4 * h < len(b) else None) for h in (1, 4, 12)}
    if "atr" in b and i + 16 < len(b):
        r["ret_4h_atr"] = sgn * (c[i + 16] - c[i]) / b["atr"].iloc[i]
    res, hit1, hit2, j_end = None, False, False, end
    for j in range(i + 1, end + 1):
        adverse = hi[j] >= stop if short else lo[j] <= stop
        fav = (c[i] - lo[j]) if short else (hi[j] - c[i])
        if adverse:                          # стоп і ціль в одному барі — рахуємо стоп (обережно)
            res, j_end = -1.0, j
            break
        hit1 |= fav >= R
        if fav >= 2 * R:
            hit2, res, j_end = True, 2.0, j
            break
    done = res is not None or i + horizon < len(b)
    if res is None:
        res = sgn * (c[end] - c[i]) / R
    r.update({"hit1R": hit1 or hit2, "hit2R": hit2, "R": res - FEE * c[i] / R, "R_pct": R / c[i], "done": done,
              "exit_ts": b.index[j_end], "reason": "target" if hit2 else "stop" if res == -1.0 else "time" if done else "open"})
    return r


# ---------------------------------------------------------------- живі дані (скринер, дашборд)

def ensure_table(conn) -> None:
    conn.execute(DDL)
    conn.commit()


def _binance_symbol(conn, base: str) -> tuple[str, str] | None:
    """(символ у базі, символ на біржі) USDT-перпа Binance."""
    r = conn.execute("""SELECT symbol, symbol_on_exchange FROM markets WHERE base_asset = %s AND exchange_name = 'Binance'
                        AND quote_asset = 'USDT' ORDER BY symbol LIMIT 1""", (base,)).fetchone()
    return (r[0], r[1]) if r else None


def load_live(conn, base: str, since: pd.Timestamp, until: pd.Timestamp | None = None) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """15m-бари Binance (OHLC, обсяг $ зі свічок) з since і футпрінт 15m з trade_bars (сума трьох 5m-барів).
    Бар, для якого записано не всі три 5m-бари, отримує порожній футпрінт (патерну на ньому не буде)."""
    from features import query_df
    sym = _binance_symbol(conn, base)
    if sym is None:
        return None
    until = until if until is not None else pd.Timestamp.now(tz="UTC")
    o = query_df(conn, """SELECT t.ts, t.o, t.h, t.l, t.c, t.v, m.volume_denominated FROM ohlcv t JOIN markets m USING (symbol)
                          WHERE t.symbol = %s AND t.interval = '15min' AND t.ts >= %s AND t.ts < %s ORDER BY t.ts""",
                 (sym[0], since.to_pydatetime(), until.to_pydatetime()))
    if o.empty:
        return None
    o["ts"] = pd.to_datetime(o["ts"], utc=True)
    b = o.set_index("ts").rename(columns={"o": "open", "h": "high", "l": "low", "c": "price"})
    b["vol"] = np.where(b["volume_denominated"] == "BASE_ASSET", b["v"] * b["price"], b["v"])
    t = query_df(conn, """SELECT ts, v_buy, v_sell, fp_lvl0, fp_buy, fp_sell FROM trade_bars
                          WHERE exchange = 'Binance' AND symbol = %s AND ts >= %s AND ts < %s ORDER BY ts""",
                 (sym[1], since.to_pydatetime(), until.to_pydatetime()))
    fp = pd.DataFrame(columns=["ts", "lvl", "buy", "sell"])
    if len(t):
        t["ts"] = pd.to_datetime(t["ts"], utc=True)
        t["bar"] = t["ts"].dt.floor("15min")
        full = t.groupby("bar")["ts"].transform("count") == 3
        t = t[full & t["fp_lvl0"].notna()]
        d = t.groupby("bar")[["v_buy", "v_sell"]].sum()
        b["delta"] = (d["v_buy"] - d["v_sell"]).reindex(b.index)
        rows = [(r.bar, r.fp_lvl0 + k, bu, se) for r in t.itertuples() for k, (bu, se) in enumerate(zip(r.fp_buy, r.fp_sell))]
        if rows:
            fp = pd.DataFrame(rows, columns=["ts", "lvl", "buy", "sell"]).groupby(["ts", "lvl"], as_index=False).sum()
    return b.drop(columns=["v", "volume_denominated"]), fp


def context_frame(conn, bases: list[str], since: pd.Timestamp, until: pd.Timestamp, m15: dict) -> pd.DataFrame:
    """Контекст моделі на кожному барі [since, until]: (base_asset, ts, ctx) лише для барів, де за останні CTX_BARS барів
    (включно з поточним) був Кандидат або Alert; ctx = 'alert', якщо серед них був Alert, інакше 'candidate'."""
    from features import query_df
    from tiers import tier_frame
    if not bases:
        return pd.DataFrame(columns=["base_asset", "ts", "ctx"])
    confirm = m15.get("confirm", {}).get("model", "lightgbm")
    # остання версія моделі — серед рядків цих тікерів і цього проміжку (індекс signals: base_asset, ts), а НЕ по всій
    # таблиці: max(model_version) по всіх signals — мільйони рядків, хвилини на сервері (07.10 гальмувало скринер)
    sig = query_df(conn, """
        WITH w AS (SELECT base_asset, ts, model, model_version, probability, threshold FROM signals
                   WHERE base_asset = ANY(%s) AND model = ANY(%s) AND ts > %s AND ts <= %s)
        SELECT w.base_asset, w.ts, w.model, w.probability, w.threshold FROM w
        JOIN (SELECT model, max(model_version) AS v FROM w GROUP BY model) m ON w.model = m.model AND w.model_version = m.v""",
                   (bases, [m15["name"], confirm], (since - (CTX_BARS + 8) * BAR).to_pydatetime(), until.to_pydatetime()))
    if sig.empty:
        return pd.DataFrame(columns=["base_asset", "ts", "ctx"])
    sig["ts"] = pd.to_datetime(sig["ts"], utc=True)
    t = tier_frame(sig[sig["model"] == m15["name"]], sig[sig["model"] == confirm], m15.get("confirm", {}).get("share", 0.75))
    out = []
    for base, g in t.groupby("base_asset"):
        g = g.set_index("ts").sort_index()
        grid = pd.date_range(g.index.min(), until, freq="15min")
        tier = g["tier"].reindex(grid)
        sig_ = tier.isin(["candidate", "alert"]).astype(int).rolling(CTX_BARS, min_periods=1).max()
        al = (tier == "alert").astype(int).rolling(CTX_BARS, min_periods=1).max()
        f = pd.DataFrame({"base_asset": base, "ts": grid, "ctx": np.where(al > 0, "alert", "candidate")})[sig_.to_numpy() > 0]
        out.append(f[(f["ts"] >= since) & (f["ts"] <= until)])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["base_asset", "ts", "ctx"])


def find(conn, base: str, since: pd.Timestamp, until: pd.Timestamp, ctx: pd.Series) -> list[dict]:
    """Патерни b / p на барах [since, until] тікера base, лише там, де є контекст (ctx: ts → 'alert' / 'candidate')."""
    x = load_live(conn, base, since - (96 + LOOK + 4) * BAR, until + BAR)
    if x is None or x[1].empty:
        return []
    b = bar_metrics(*x)
    pat = patterns(b)
    found = []
    for kind, m in pat.items():
        for ts in m.index[m & (m.index >= since) & (m.index <= until)]:
            if ts not in ctx.index:
                continue
            i = b.index.get_loc(ts)
            row = b.iloc[i]
            found.append({"base_asset": base, "ts": ts, "kind": kind, "close": float(row["price"]),
                          "stop": float(stop_of(b, i, kind)), "level": float(row["L"] if kind == "b" else row["H"]),
                          "beyond": float(row["below_share"] if kind == "b" else row["above_share"]),
                          "beyond_delta": float(row["below_delta"] if kind == "b" else row["above_delta"]),
                          "edge": float(row["top_share"] if kind == "b" else row["bot_share"]),
                          "vol_x": float(row["vol_ratio"]), "ctx": ctx.loc[ts]})
    return found


def save_marks(conn, found: list[dict], live: bool) -> None:
    """Мітки → fp_marks. Уже записані не змінюються (живі лишаються живими, коли історію дозаповнюють з архіву)."""
    if not found:
        return
    with conn.cursor() as cur:
        cur.executemany("""INSERT INTO fp_marks (base_asset, ts, kind, close, stop, level, beyond, beyond_delta, edge, vol_x, ctx, live)
                           VALUES (%(base_asset)s, %(ts)s, %(kind)s, %(close)s, %(stop)s, %(level)s, %(beyond)s,
                                   %(beyond_delta)s, %(edge)s, %(vol_x)s, %(ctx)s, %(live)s) ON CONFLICT DO NOTHING""",
                        [f | {"ts": pd.Timestamp(f["ts"]).to_pydatetime(), "live": live} for f in found])
    conn.commit()
    if not live:
        return                                 # історичні (дозаповнення): не смикаємо дашборд — йому вистачить наступної оцінки
    try:                                       # дашборд перечитує список (і сповіщення) за міткою оцінок
        os.makedirs(os.path.dirname(SIGNALS_STAMP), exist_ok=True)
        with open(SIGNALS_STAMP, "w") as f:
            f.write(pd.Timestamp.now(tz="UTC").isoformat())
    except OSError:
        pass


def detect(conn, bases: list[str], last: pd.Timestamp, m15: dict, log=print) -> list[dict]:
    """Скринер після 15m-оцінки: патерни b / p на щойно закритому барі last у тікерів з контекстом → fp_marks (живі)."""
    if conn.execute("SELECT to_regclass('trade_bars')").fetchone()[0] is None:
        return []                                  # записувач угод ще не запускався — футпрінту немає
    ctx = context_frame(conn, bases, last, last, m15)
    found = []
    for base, g in ctx.groupby("base_asset"):
        found += find(conn, base, last, last, g.set_index("ts")["ctx"])
    save_marks(conn, found, live=True)
    if found:
        log("  b/p: " + ", ".join(f"{f['base_asset']} {f['kind']}" for f in found))
    return found


def detect_range(conn, base: str, since: pd.Timestamp, until: pd.Timestamp, m15: dict) -> list[dict]:
    """Історичні мітки (live=false) на барах [since, until] — після дозаповнення футпрінту з архіву (trade_backfill.py)."""
    ctx = context_frame(conn, [base], since, until, m15)
    if ctx.empty:
        return []
    found = find(conn, base, since, until, ctx.set_index("ts")["ctx"])
    save_marks(conn, found, live=False)
    return found


def levels(conn, base: str, since: pd.Timestamp, until: pd.Timestamp, tf_sec: int) -> dict:
    """Футпрінт для графіка: бари таймфрейму tf_sec (5m / 15m / 1h — сума 5m-барів trade_bars, Binance) у [since, until):
    {"step": крок рівня, "bars": [{"t", "lvl0", "buy": [...], "sell": [...]}]}; рівень k ціни — [e^(k·LVL), e^((k+1)·LVL))."""
    from features import query_df
    sym = _binance_symbol(conn, base)
    out = {"step": LVL, "bars": []}
    if sym is None or conn.execute("SELECT to_regclass('trade_bars')").fetchone()[0] is None:
        return out
    t = query_df(conn, """SELECT ts, fp_lvl0, fp_buy, fp_sell FROM trade_bars
                          WHERE exchange = 'Binance' AND symbol = %s AND ts >= %s AND ts < %s AND fp_lvl0 IS NOT NULL ORDER BY ts""",
                 (sym[1], since.to_pydatetime(), until.to_pydatetime()))
    if t.empty:
        return out
    t["ts"] = pd.to_datetime(t["ts"], utc=True)
    secs = (t["ts"] - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1)       # не залежить від одиниць (мкс / нс)
    t["bar"] = (secs // tf_sec) * tf_sec
    for bar, g in t.groupby("bar"):
        lo = int(g["fp_lvl0"].min())
        hi = int(max(r.fp_lvl0 + len(r.fp_buy) - 1 for r in g.itertuples()))
        buy, sell = np.zeros(hi - lo + 1), np.zeros(hi - lo + 1)
        for r in g.itertuples():
            k = int(r.fp_lvl0) - lo
            buy[k:k + len(r.fp_buy)] += r.fp_buy
            sell[k:k + len(r.fp_sell)] += r.fp_sell
        out["bars"].append({"t": int(bar), "lvl0": lo, "buy": np.round(buy).tolist(), "sell": np.round(sell).tolist()})
    return out


def marks(conn, bases: list[str] | None = None, since: pd.Timestamp | None = None, until: pd.Timestamp | None = None) -> pd.DataFrame:
    from features import query_df
    cond, params = ["true"], []
    if bases:
        cond.append("base_asset = ANY(%s)"); params.append(bases)
    if since is not None:
        cond.append("ts >= %s"); params.append(pd.Timestamp(since).to_pydatetime())
    if until is not None:
        cond.append("ts < %s"); params.append(pd.Timestamp(until).to_pydatetime())
    df = query_df(conn, f"SELECT * FROM fp_marks WHERE {' AND '.join(cond)} ORDER BY ts", tuple(params))
    for c in ("ts", "created_at"):
        if len(df):
            df[c] = pd.to_datetime(df[c], utc=True)
    return df


def with_outcomes(conn, m: pd.DataFrame) -> pd.DataFrame:
    """Мітки + результат (outcome) за 15m-барами Binance після мітки: +1R / +2R / стоп / 12 год, R."""
    from features import query_df
    rows = []
    for base, g in m.groupby("base_asset"):
        sym = _binance_symbol(conn, base)
        if sym is None:
            continue
        o = query_df(conn, """SELECT ts, h, l, c FROM ohlcv WHERE symbol = %s AND interval = '15min' AND ts >= %s
                              AND ts <= %s ORDER BY ts""",
                     (sym[0], g["ts"].min().to_pydatetime(), (g["ts"].max() + (HORIZON + 1) * BAR).to_pydatetime()))
        if o.empty:
            continue
        o["ts"] = pd.to_datetime(o["ts"], utc=True)
        b = o.set_index("ts").rename(columns={"h": "high", "l": "low", "c": "price"})
        for r in g.itertuples():
            if r.ts not in b.index:
                continue
            i = b.index.get_loc(r.ts)
            res = outcome(b, i, r.stop, short=r.kind == "p")
            if res:
                rows.append({k: v for k, v in r._asdict().items() if k != "Index"} | res)
    return pd.DataFrame(rows)


def stats(conn, since: pd.Timestamp, until: pd.Timestamp, bases: list[str] | None = None) -> dict:
    """Підсумок міток за період (за ts бару): кількість, завершені, +2R / стоп / час, середній R — окремо b і p, живі ("b", "p")
    і знайдені пізніше на історії ("b_hist", "p_hist"): живі — чесний тест на нових даних, історичні — повтор бектесту."""
    m = marks(conn, bases, since, until)
    o = with_outcomes(conn, m) if len(m) else pd.DataFrame()
    out = {}
    for kind in ("b", "p"):
        for live, sfx in ((True, ""), (False, "_hist")):
            mm = m[(m["kind"] == kind) & (m["live"] == live)] if len(m) else m
            g = o[(o["kind"] == kind) & (o["live"] == live)] if len(o) else o
            d = g[g["done"]] if len(g) else g
            out[kind + sfx] = {"n": len(mm), "done": len(d),
                               "target": int((d["reason"] == "target").sum()) if len(d) else 0,
                               "stop": int((d["reason"] == "stop").sum()) if len(d) else 0,
                               "time": int((d["reason"] == "time").sum()) if len(d) else 0,
                               "hit1R": int(d["hit1R"].sum()) if len(d) else 0,
                               "avg_R": float(d["R"].mean()) if len(d) else None}
    return out
