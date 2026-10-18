"""
dashboard.py — дашборд Pump Detector: екрани «Прогноз» і «Скринер» (макет Pump Detector App).

Локальний сервер на 127.0.0.1; сторінка — app/dashboard.html. Дані — таблиця signals
(пише screener.py), сирі ряди з бази, збережені моделі (models/*.joblib) і розрахунок ризику (risk.py).

Основна «модель» — tier15: 15m-LightGBM з підтвердженням годинним LightGBM (tiers.py). Статуси:
«Alert» — 15m-сигнал і годинна оцінка ≥ 0.75 порогу; «Кандидат» — лише 15m-сигнал. Годинні моделі
(lightgbm, logreg) лишаються для порівняння, на годинних барах, зі старими статусами.

    python dashboard.py
    python dashboard.py --port 8766 --no-browser
"""

import argparse
import json
import math
import os
import sys
import threading
import time
import webbrowser
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import psycopg
from dotenv import load_dotenv

from features import INTERVAL_15M, INTRADAY_EXCHANGES, aggregate, compute_features, load_raw, query_df, split_raw
from fetch import Coinalyze, collect, read_tickers_file
from risk import EXIT_BARS, PARTIAL_EXIT, atr, intrabar_stop, position_state, trade_stats


def last_atr(h: pd.DataFrame) -> float | None:
    """Поточний ATR(14) на барах h — для стопу «якщо вхід зараз»."""
    a = atr(h).dropna()
    return float(a.iloc[-1]) if len(a) else None
from screener import (FETCH_DAYS, INTRADAY_DAYS, SIGNALS_STAMP, STATUS_FILE, WATCHLIST_FILE, last_closed_15m, load_model_15m,
                      load_models, next_hourly, next_run, score_15m, score_bases, score_bases_own, update_15m)
import drivers as DRV
import fetch_exchange
import footprint
import market
import movers
import orderbook
import volume_profile
import wake
import strategies
import scanner
import spread
import journal as J
import tiers as T

ROOT = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.join(ROOT, "app", "dashboard.html")
VENDOR = os.path.join(ROOT, "app", "vendor")
JS_DIR = os.path.join(ROOT, "app", "js")                              # спільні скрипти сторінок
MODEL_NAMES = {"lightgbm": "LightGBM", "logreg": "Логістична регресія"}
CANDIDATE_SHARE = 0.5     # «Кандидат»: ймовірність ≥ половини порогу сигналу
SPARK_HOURS = 24
QUIET_HOURS = 24          # рядок списку блідий («тихо N год»), якщо стільки годин без Кандидата / Alert і «прокидається»
BAR_15M = pd.Timedelta(minutes=15)
refresh_lock = threading.Lock()


# ---------------------------------------------------------------- утиліти

def clean(o):
    """pandas/numpy → JSON: час у секундах epoch, NaN → null."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if o is pd.NaT:
        return None
    if isinstance(o, (pd.Timestamp, datetime)):
        return int(o.timestamp())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if math.isnan(o) or math.isinf(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def status_of(p: float | None, thr: float) -> str:
    if p is None or pd.isna(p):
        return "none"
    if p >= thr:
        return "alert"
    return "candidate" if p >= CANDIDATE_SHARE * thr else "calm"


def latest_signals(conn, base: str | None = None, bases: list[str] | None = None, days: float | None = None) -> pd.DataFrame:
    """Оцінки лише найсвіжішої версії кожної моделі. Для одного тікера — версія серед ЙОГО рядків (індекс signals:
    base_asset, ts — мілісекунди): max(model_version) по всій таблиці на кожне відкриття тікера під навантаженням тривав
    хвилину, запити накладались один на одного (07.10: 15 таких одночасно, база 117% CPU, графіки не відкривались)."""
    if base:
        return query_df(conn, """
            WITH w AS (SELECT * FROM signals WHERE base_asset = %s)
            SELECT w.base_asset, w.ts, w.model, w.probability, w.threshold, w.signal, w.live, w.in_sample, w.scored_at
            FROM w JOIN (SELECT model, max(model_version) AS v FROM w GROUP BY model) m
              ON w.model = m.model AND w.model_version = m.v
            ORDER BY w.ts""", (base,))
    if bases is not None:
        # список дашборда (13.10): лише його тікери і лише останні days днів КОЖНОГО тікера (від його останньої оцінки —
        # так тікер, який давно не оцінювали, не зникає зі списку, і копія бази на ноутбуці показує те саме).
        # Читається за індексом (base_asset, ts), а не вся таблиця.
        return query_df(conn, """
            WITH last AS (SELECT b AS base_asset, (SELECT max(ts) FROM signals WHERE base_asset = b) AS mx
                          FROM unnest(%s::text[]) AS b),
                 w AS (SELECT s.* FROM last l JOIN signals s ON s.base_asset = l.base_asset AND s.ts >= l.mx - %s::interval)
            SELECT w.base_asset, w.ts, w.model, w.probability, w.threshold, w.signal, w.live, w.in_sample, w.scored_at
            FROM w JOIN (SELECT model, max(model_version) AS v FROM w GROUP BY model) m
              ON w.model = m.model AND w.model_version = m.v
            ORDER BY w.base_asset, w.ts""", (list(bases), timedelta(days=days)))
    return query_df(conn, """
        SELECT s.base_asset, s.ts, s.model, s.probability, s.threshold, s.signal, s.live, s.in_sample, s.scored_at
        FROM signals s
        JOIN (SELECT model, max(model_version) AS v FROM signals GROUP BY model) m
          ON s.model = m.model AND s.model_version = m.v
        ORDER BY s.base_asset, s.ts""")


def screener_status() -> dict | None:
    try:
        with open(os.path.join(ROOT, STATUS_FILE), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def next_scan(now: datetime) -> datetime:
    return next_run(now) if State.m15 is not None else next_hourly(now)


# ---------------------------------------------------------------- дані для екранів

class State:
    db_url = ""
    api_key = ""
    models: list[dict] = []
    m15: dict | None = None          # 15m-модель (models/lightgbm_15m.joblib)
    client: Coinalyze | None = None
    spread: "spread.Monitor | None" = None   # монітор розбіжності цін між біржами (spread.py), потік у цьому процесі


def share() -> float:
    return (State.m15 or {}).get("confirm", {}).get("share", 0.75)


def tier_rows(conn, base: str | None = None, rolling: bool = False, bases: list[str] | None = None,
              days: float | None = None) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Рівні сигналу (tiers.py) з оцінок найсвіжіших версій 15m- і годинного LightGBM.
    rolling=True — ще й рівні на ковзних 15m-барах (5-хв цикл скринера): (звичайні, ковзні).
    bases + days — лише ці тікери і лише останні days днів кожного (список дашборда)."""
    sig = latest_signals(conn, base, bases, days)
    s1h = sig[sig["model"] == J.CONFIRM_MODEL]
    t = T.tier_frame(sig[sig["model"] == J.M15], s1h, share())
    return (t, T.tier_frame(sig[sig["model"] == T.ROLL_MODEL], s1h, share())) if rolling else t


def start_scores(conn, base: str) -> pd.Series | None:
    """alert_score моделі «старт» тікера (режим split; індекс — ts) з найсвіжішої версії; None — оцінок ще немає."""
    sig = latest_signals(conn, base)
    st = sig[sig["model"] == T.START_MODEL]
    if st.empty:
        return None
    return T.tier_frame(st, sig[sig["model"] == J.CONFIRM_MODEL], share()).set_index("ts").sort_index()["alert_score"]


def virtual_models() -> list[dict]:
    """Моделі 15m-контуру для перемикача і журналу: tier15 — «Alert», cand15 — «Кандидат» (лише журнал)."""
    m = State.m15
    if m is None:
        return []
    common = {"tf": "15m", "bar": 900, "trained_at": m["trained_at"], "threshold": m["threshold"], "tickers": m["tickers"],
              "pump_intervals": m["pump_intervals"], "risk": m.get("risk", {}), "settings": m.get("settings", {}),
              "confirm": m.get("confirm", {}), "exit_bars": m.get("exit_bars")}
    return [{"name": T.TIER_MODEL, "title": "15m + 1h", "validation": m.get("tier_validation", {}), **common},
            {"name": T.CAND_MODEL, "title": "15m кандидат", "validation": m.get("validation", {}), **common},
            {"name": T.LIMIT_MODEL, "title": "15m + 1h · лімітом", "validation": m.get("tier_validation", {}), **common}]


def meta(conn) -> dict:
    now = datetime.now(timezone.utc)
    sig = query_df(conn, "SELECT max(scored_at) AS last FROM signals")
    exchanges = conn.execute("SELECT count(DISTINCT exchange_name) FROM markets").fetchone()[0]
    manual, auto = list_sources()
    watch = manual + sorted(auto)
    scored = [r[0] for r in conn.execute("SELECT DISTINCT base_asset FROM signals ORDER BY 1").fetchall()]
    hourly = [{
        "name": m["name"], "title": MODEL_NAMES.get(m["name"], m["name"]) + " 1h", "tf": "1h", "bar": 3600,
        "trained_at": m["trained_at"], "threshold": m["threshold"],
        "tickers": m["tickers"], "pump_intervals": m["pump_intervals"],
        "validation": m["validation"], "risk": m.get("risk", {}), "settings": m.get("settings", {}),
    } for m in State.models]
    virt = virtual_models()
    return {
        "models": virt[:1] + hourly,              # перемикач: 15m + 1h (основна), далі годинні
        "journal_models": [{"name": s["id"], "title": s["title"], "note": s.get("note")} for s in strategies.load()],
        "strategies": [strategies.public(s) for s in strategies.load()],     # режими: один список для трьох вкладок
        "tickers": sorted(set(watch) | set(scored)),
        "auto": sorted(auto),
        "watchlist": watch,
        "exchanges": exchanges,
        "last_scored_at": sig["last"].iloc[0] if len(sig) else None,
        "screener": screener_status(),
        "next_scan": next_scan(now),
        "now": now,
    }


def episodes(sig: pd.DataFrame, bar: pd.Timedelta = pd.Timedelta(hours=1)) -> list[dict]:
    """Початок кожного епізоду сигналу (сусідні бари з сигналом = один епізод)."""
    out = []
    for base, g in sig.sort_values("ts").groupby("base_asset"):
        last_sig = None
        for r in g.itertuples():
            # той самий епізод — якщо від ПОПЕРЕДНЬОГО СИГНАЛЬНОГО рядка не більше bar. Не «попередній рядок»: між двома
            # звичайними 15m-барами з Alert стоять ковзні 5-хв оцінки без Alert, і кожна рвала епізод (ORCA 10-06
            # 10:00 / 10:15 / 10:30 / 10:45 — чотири сповіщення в одному безперервному Alert)
            if r.signal:
                if last_sig is None or r.ts - last_sig > bar:
                    kind = "live" if r.live else ("in-sample" if r.in_sample else "history")
                    out.append({"ticker": base, "ts": r.ts, "probability": r.probability, "kind": kind})
                last_sig = r.ts
    return sorted(out, key=lambda e: e["ts"], reverse=True)


def watch_list_15m(conn) -> dict:
    """Права панель для 15m + 1h: P 15m-моделі, рівень сигналу, міні-графік P за 24 год (96 барів)."""
    # РОЗВАНТАЖЕННЯ СЕРВЕРА (13.10): раніше список рахувався для ВСІХ тікерів, що колись мали оцінки (на сервері — утричі
    # більше, ніж у списку), на всій їх історії, і щохвилини: ~50% одного ядра. Тепер — лише тікери списку (решту
    # watch_list однаково відкидав) і оцінки за LIST_SIGNAL_DAYS днів: угоди й епізоди рахуються на 2 добах барів.
    now = datetime.now(timezone.utc)
    manual, auto = list_sources()
    keep = sorted(set(manual) | set(auto))
    if not keep:
        return {"rows": [], "recent": [], "alerts": []}
    t, troll = tier_rows(conn, rolling=True, bases=keep, days=LIST_SIGNAL_DAYS)
    if t.empty:
        return {"rows": [], "recent": [], "alerts": []}
    raw = load_raw(conn, sorted(t["base_asset"].unique()), INTERVAL_15M, since=now - timedelta(days=2))
    parts = split_raw(raw)                      # сирі ряди по тікерах — один раз, а не фільтр усього raw на кожен тікер
    newest = troll.sort_values("ts").groupby("base_asset").tail(1).set_index("base_asset") if len(troll) else None
    rows, hmap = [], {}
    woke = wake.last_wake()                    # «прокинулись» за останні 30 хв — фіолетова крапка в списку
    woke_day = wake.last_wake(minutes=QUIET_HOURS * 60)     # для «тихо N год»
    for base, g in t.groupby("base_asset"):
        g = g.sort_values("ts")
        last = g.iloc[-1]
        # статус і P — з найсвіжішої оцінки: ковзний бар 5-хв циклу новіший за останній звичайний 15m-бар
        # статус і P — офіційного 15m-бару, як на графіку, в заголовку вікна і в позиції (SYN 06.10: список показував
        # ковзну 0.46, а графік — Alert 0.91 на барі). Ковзна 5-хв оцінка, якщо новіша, — окремим полем roll (підпис «5m»)
        roll = None
        if newest is not None and base in newest.index and newest.at[base, "ts"] > last["ts"]:
            n = newest.loc[base]
            roll = {"p15": n["p15"], "tier": n["tier"], "close": n["ts"] + BAR_15M}
        streak = 0
        for x in reversed(g["tier"].tolist()):
            if x != last["tier"] or x == "calm":
                break
            streak += 1
        # «тихо N год» (бліді рядки — кандидати на «×»): скільки годин без Кандидата / Alert і без «прокидається»
        act = g.loc[g["tier"] != "calm", "ts"]
        last_act = act.max() if len(act) else g["ts"].min()
        # секція NOW у списку: Alert / Candidate тримається ще годину після закриття останнього сигнального бару
        last_sig = {"close": act.max() + BAR_15M, "tier": g.loc[g["ts"] == act.max(), "tier"].iloc[-1]} if len(act) else None
        if base in woke_day:
            last_act = max(last_act, pd.Timestamp(woke_day[base]["at"]))
        quiet_h = (g["ts"].iloc[-1] + BAR_15M - last_act) / pd.Timedelta(hours=1)
        h = aggregate(parts.get(base, raw), base, "15min", light=True)   # лише ціна й OI: зміни за 24 год, ATR для ліміту
        hmap[base] = h
        lt = g["ts"].iloc[-1]                     # спарклайн і зміни за 24 год — на звичайних 15m-барах
        h = h[h.index <= lt] if not h.empty else h
        spark = g.set_index("ts")["p15"].reindex(pd.date_range(lt - 95 * BAR_15M, lt, freq="15min"))
        chg = oi24 = None
        if len(h) > 96:
            p0, p1 = h["price"].iloc[-97], h["price"].iloc[-1]
            chg = p1 / p0 - 1 if p0 else None
            oi24 = math.expm1(h["oi_log_idx"].iloc[-1] - h["oi_log_idx"].iloc[-97])
        rows.append({
            "ticker": base, "probability": last["p15"], "threshold": last["thr15"], "status": last["tier"],
            "streak": streak, "streak_min": streak * 15, "in_sample": bool(last["in_sample"]), "last_bar": last["ts"],
            "spark": spark.tolist(), "chg_24h": chg, "oi_24h": oi24,
            "p1h": last["p1h"], "thr1h": last["thr1h"], "need1h": share() * last["thr1h"] if pd.notna(last["thr1h"]) else None,
            "wake": woke.get(base), "roll": roll, "quiet_h": round(float(quiet_h), 1), "last_sig": last_sig,
        })
    order = {"alert": 0, "candidate": 1, "calm": 2}
    rows.sort(key=lambda r: (order.get(r["status"], 3), -(r["probability"] or 0)))
    t15 = t                                   # лише звичайні 15m-бари — для угод моделі (як позиція на графіку)
    t = pd.concat([t, troll], ignore_index=True) if len(troll) else t     # епізоди — з ковзними: старт на ~5 хв раніше
    al = t.assign(signal=t["tier"] == "alert", probability=t["p15"])
    ep = episodes(al, BAR_15M)
    recent_cut = pd.Timestamp(now - timedelta(hours=6))
    alerts = [e for e in ep if e["kind"] == "live" and e["ts"] >= recent_cut]
    # старт кожного епізоду «Кандидат або Alert» (15m-модель перетнула поріг) — для сповіщень про Кандидатів
    cand = t.assign(signal=t["tier"] != "calm", probability=t["p15"])
    cands = [e for e in episodes(cand, BAR_15M) if e["kind"] == "live" and e["ts"] >= recent_cut]
    # Сповіщення за угодами моделі (2026-10-08, власник: «заходити з моделлю, а не щоразу, коли P вище порогу»):
    # Alert — лише ВХІД моделі (та сама угода, що мітка «Вхід» на графіку); новий спалах Alert, поки угода відкрита, —
    # Re-alert (окремий перемикач, за замовчуванням вимкнений); Кандидат — лише коли модель без позиції і годину тихо.
    active = {r["ticker"] for r in rows if r["status"] != "calm" or (r["last_sig"] and pd.Timestamp(now) - r["last_sig"]["close"] <= pd.Timedelta(hours=1))}
    trades = {b: model_trades(conn, t15[t15["base_asset"] == b], hmap.get(b)) for b in active}
    for r in rows:
        r["trade"] = (trade_state(trades.get(r["ticker"], []), r, now, alert_since(t15[t15["base_asset"] == r["ticker"]]))
                      if r["ticker"] in active else None)
    alerts, realerts, cands, lates = split_by_trades(conn, t15, t, hmap, alerts, cands, recent_cut, trades)
    # для сповіщення: OB у момент сигналу і ліміт −0.5 ATR (ковзний бар 5-хв циклу → останній звичайний 15m-бар до нього)
    for e in alerts:
        hb = hmap.get(e["ticker"])
        try:
            rows_b = orderbook.history(conn, e["ticker"], (recent_cut - timedelta(hours=1)).to_pydatetime())
        except Exception:
            conn.rollback()
            rows_b = []
        e["ob"] = ob_at_signal(rows_b, e["ts"])
        if hb is not None and len(hb):
            base_ts = hb.index[hb.index <= pd.Timestamp(e["ts"])]
            lim = limit_hint(hb, base_ts[-1]) if len(base_ts) else None
            e["limit"] = lim["price"] if lim else None
    # «прокидається» за останні 30 хв — для голосового сповіщення («Volume STRK»)
    wakes = [{"ticker": b, "ts": w["at"], "vol_x": w["vol_x"], "ret15": w["ret15"], "oi1h": w["oi1h"]} for b, w in woke.items()]
    # живі мітки футпрінту b / p за 6 год (footprint.py) — для сповіщень бота; історичні (з архіву) не надсилаються
    fps = [{"ticker": r["base_asset"], **{k: r[k] for k in ("ts", "kind", "close", "stop", "level", "beyond", "beyond_delta", "vol_x", "ctx")}}
           for r in fp_rows(conn, since=pd.Timestamp(now - timedelta(hours=6))) if r.get("live", True)]
    last_fp = {f["ticker"]: f for f in sorted(fps, key=lambda f: f["ts"])}       # мітка рядка: b · Reclaim / p — остання
    for r in rows:
        f = last_fp.get(r["ticker"])
        r["fp"] = {"kind": f["kind"], "at": pd.Timestamp(f["ts"]) + BAR_15M} if f else None
    return {"rows": rows, "recent": ep[:30], "alerts": alerts, "realerts": realerts, "cands": cands, "lates": lates, "wakes": wakes, "fps": fps,
            "minis": mini_charts(conn, t15, hmap, trades)}


MINI_BARS = 192                                # у кеші — дві доби 15m-барів (з них же збирається 1h-вигляд)
MINI_VIEW = {"5m": (96, 300), "15m": (96, 900), "1h": (48, 3600)}     # таймфрейм вкладки «Charts» → (барів, секунд у барі)
MINI_KEYS = ("t", "o", "h", "l", "c", "tier")


def mini_charts(conn, t15: pd.DataFrame, hmap: dict, trades: dict) -> dict:
    """Дані міні-графіків вкладки «Charts» (13.10): останні MINI_BARS закритих 15m-барів кожного тікера списку, рівень сигналу
    на бар, входи моделі й стоп відкритої угоди (основний режим — ті самі угоди, що в колонці State). Рахуються разом зі
    списком з уже завантажених барів, тому додаткових запитів до бази майже немає; у відповідь /api/list не йдуть
    (watch_list їх прибирає) — їх віддає /api/minicharts лише для запитаних тікерів."""
    code = {"alert": 2, "candidate": 1, "calm": 0}
    try:                                        # символ Binance — для живої свічки (один WebSocket на всю сітку)
        syms = dict(conn.execute("""SELECT base_asset, symbol_on_exchange FROM markets
                                    WHERE exchange_name = 'Binance' AND quote_asset = 'USDT' AND base_asset = ANY(%s)""",
                                 (list(hmap),)).fetchall())
    except Exception:
        conn.rollback()
        syms = {}
    closed, out = last_closed_15m(), {}
    for base, g in t15.groupby("base_asset"):
        h = hmap.get(base)
        if h is None or h.empty:
            continue
        v = h.loc[h.index <= closed, ["open", "high", "low", "price"]].dropna().tail(MINI_BARS)
        if v.empty:
            continue
        tier = g.drop_duplicates("ts", keep="last").set_index("ts")["tier"].map(code).reindex(v.index)
        tr = trades.get(base) or []
        last = tr[-1] if tr else None
        out[base] = {"t": list(v.index), "o": v["open"].tolist(), "h": v["high"].tolist(), "l": v["low"].tolist(),
                     "c": v["price"].tolist(), "tier": tier.tolist(), "binance": syms.get(base),
                     "entries": [x["entry_ts"] for x in tr if x["entry_ts"] >= v.index[0]],
                     "stop": last.get("stop") if last and last.get("open") else None}
    return out


def _mini_15m(m: dict, n: int) -> dict:
    t0 = m["t"][-n:][0]
    return {**m, **{k: m[k][-n:] for k in MINI_KEYS}, "entries": [e for e in m["entries"] if e >= t0]}


def _mini_1h(m: dict, n: int) -> dict:
    """Години з 15m-барів кешу: рівень сигналу — найвищий за годину (Alert усередині години не губиться), вхід — на свою годину.
    Остання година може бути неповною — її доповнює жива свічка на сторінці."""
    d = pd.DataFrame({k: m[k] for k in MINI_KEYS[1:]}, index=pd.DatetimeIndex(m["t"]))
    g = d.groupby(d.index.floor("h"))
    a = pd.DataFrame({"o": g["o"].first(), "h": g["h"].max(), "l": g["l"].min(), "c": g["c"].last(), "tier": g["tier"].max()}).tail(n)
    ent = sorted({pd.Timestamp(e).floor("h") for e in m["entries"]})
    return {**m, "t": list(a.index), **{k: a[k].tolist() for k in MINI_KEYS[1:]}, "entries": [e for e in ent if e >= a.index[0]]}


_mini5_cache: dict = {}                        # (тікери) → (час, відповідь): 5m-бари читаються з бази, не частіше разу на 20 с


def _mini_5m(conn, symbols: list[str], minis: dict, n: int) -> dict:
    """5m-вигляд: бари біржі графіка (Binance, інакше Bybit) одним запитом на всі запитані тікери; рівень сигналу і вхід —
    з 15m-кешу (5m-бар бере рівень свого 15m-бару; вхід — на останній 5m-бар сигнального 15m-бару, тобто на його закриття)."""
    key, now = tuple(sorted(symbols)), time.monotonic()
    hit = _mini5_cache.get(key)
    if hit and now - hit[0] < 20:
        return hit[1]
    five = pd.Timedelta(minutes=5)
    closed = pd.Timestamp(datetime.now(timezone.utc)).floor("5min") - five
    rows = conn.execute("""SELECT m.base_asset, m.exchange_name, o.ts, o.o, o.h, o.l, o.c FROM ohlcv o JOIN markets m USING (symbol)
                           WHERE m.base_asset = ANY(%s) AND m.quote_asset = 'USDT' AND m.exchange_name = ANY(%s)
                             AND o.interval = '5min' AND o.ts >= %s""",
                        (symbols, list(INTRADAY_EXCHANGES), (closed - n * five).to_pydatetime())).fetchall()
    out = {}
    if rows:
        df = pd.DataFrame(rows, columns=["base", "ex", "ts", "o", "h", "l", "c"])
        for base, g in df.groupby("base"):
            ex = "Binance" if (g["ex"] == "Binance").any() else g["ex"].iloc[0]
            v = g[g["ex"] == ex].drop_duplicates("ts", keep="last").set_index("ts").sort_index()
            v.index = pd.to_datetime(v.index, utc=True)
            v = v.loc[v.index <= closed, ["o", "h", "l", "c"]].astype(float).dropna().tail(n)
            if v.empty:
                continue
            m = minis[base]
            tier = pd.Series(m["tier"], index=pd.DatetimeIndex(m["t"])).reindex(v.index.floor("15min"))
            ent = [pd.Timestamp(e) + 2 * five for e in m["entries"]]
            out[base] = {**m, "t": list(v.index), **{k: v[k].tolist() for k in MINI_KEYS[1:5]}, "tier": tier.tolist(),
                         "entries": [e for e in ent if e >= v.index[0]]}
    if len(_mini5_cache) > 20:
        _mini5_cache.clear()
    _mini5_cache[key] = (now, out)
    return out


def minicharts(conn, model: str, symbols: list[str], tf: str = "15m") -> dict:
    """/api/minicharts: міні-графіки запитаних тікерів. 15m і 1h — з кешу списку (той самий стан, що й рядки списку),
    5m — ще один запит свічок до бази на всі тікери разом."""
    tf = tf if tf in MINI_VIEW else "15m"
    n, sec = MINI_VIEW[tf]
    minis = _watch_list_cached(conn, model).get("minis") or {}
    have = [s for s in symbols[:80] if s in minis]
    if tf == "5m":
        charts = _mini_5m(conn, have, minis, n) if have else {}
    else:
        charts = {s: (_mini_1h if tf == "1h" else _mini_15m)(minis[s], n) for s in have}
    return {"tf": tf, "bar": sec, "charts": charts}


def model_trades(conn, g: pd.DataFrame, h: pd.DataFrame | None) -> list[dict]:
    """Угоди моделі на тікері за основним режимом — та сама risk.position_state, що позиція й мітки «Вхід» на графіку
    (ticker_15m); g — рядки tier_frame тікера (звичайні 15m-бари), h — 15m-бари. Закриті + відкрита (з "open")."""
    if h is None or h.empty or g.empty:
        return []
    m15, strat = State.m15, strategies.get(None)
    score, entry_p = g.set_index("ts").sort_index()["alert_score"], None
    if strategies.is_split(strat):                  # основний режим split: вхід — модель «старт», утримання — обидві
        entry_p, score = T.split_scores(score, start_scores(conn, g["base_asset"].iloc[0]))
    pos = position_state(h, score, 1.0, m15.get("risk", {}).get("k"),
                         m15.get("exit_bars", 4), PARTIAL_EXIT, strategies.entry_rule(strat),
                         market.strategy_allow(conn, strat), entry_prob=entry_p)
    pos = live_stop(conn, g["base_asset"].iloc[0], pos, h)
    return list(pos["trades"]) + ([pos] if pos.get("open") else [])


def live_stop(conn, base: str, pos: dict, h: pd.DataFrame) -> dict:
    """Відкрита угода + 5m-бари, що закрились після останнього 15m-бару в h: стоп спрацьовує одразу, як його торкнувся
    5m-бар (risk.intrabar_stop), — список, заголовок і графік показують вихід на ~5 хв після події, а не після закриття
    15m-бару. 5m-хвіст усього списку скринер тягне кожні 5 хв. Помилка читання — стан без змін."""
    if not pos.get("open") or pos.get("stop") is None or h is None or h.empty:
        return pos
    try:
        after = h["price"].dropna().index.max() + BAR_15M
        h5 = aggregate(load_raw(conn, [base], "5min", since=after), base, "5min", light=True)
        return pos if h5.empty else intrabar_stop(pos, h5.loc[h5.index >= after, "low"].dropna(), BAR_15M, PARTIAL_EXIT)
    except Exception as e:
        conn.rollback()
        print(f"! стоп наживо {base}: {type(e).__name__}: {e}")
        return pos


def _in_trade(tr: list[dict], ts) -> dict | None:
    """Угода, відкрита на момент ts (вхід раніше, вихід — не раніше за ts)."""
    ts = pd.Timestamp(ts)
    return next((x for x in tr if x["entry_ts"] < ts and (x.get("exit_ts") is None or ts <= x["exit_ts"])), None)


def split_by_trades(conn, t15: pd.DataFrame, t: pd.DataFrame, hmap: dict, alerts: list[dict], cands: list[dict],
                    recent_cut: pd.Timestamp, trades: dict | None = None) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Епізоди Alert → (входи моделі, Re-alert, Кандидати, late); Кандидати — без тих, що всередині угоди або після
    активності за годину. Вхід, що не почав нового епізоду (повторний вхід після стопу посеред безперервного Alert),
    додається окремо. Режим split (основний): новий епізод Alert — НЕ вхід сам по собі; вхід — лише угода моделі
    (вхід за моделлю «старт»), а епізод Alert без угоди — у late (без звуку; список — чіп Late, Telegram — ⚪ Late, 12.10).
    trades — {тікер: угоди основного режиму} (model_trades), щоб не рахувати двічі."""
    split = strategies.is_split(strategies.get(None))
    bases = {e["ticker"] for e in alerts + cands}
    trades = dict(trades or {})
    for b in bases - set(trades):
        trades[b] = model_trades(conn, t15[t15["base_asset"] == b], hmap.get(b))
    entries, re, late = [], [], []
    for e in alerts:
        cur = _in_trade(trades.get(e["ticker"], []), e["ts"])
        if cur is None:
            (late if split else entries).append(e)
        else:
            re.append({**e, "trade": {"entry_ts": cur["entry_ts"] + BAR_15M, "entry": cur["entry"],
                                      "result_pct": cur.get("result_pct"), "open": bool(cur.get("open"))}})
    for b, tr in trades.items():
        g = t15[t15["base_asset"] == b].set_index("ts")
        for x in tr:
            ets = x["entry_ts"]
            if ets < recent_cut or ets not in g.index or not bool(g.at[ets, "live"]):
                continue
            # ковзне вікно (зсув 5–10 хв) могло почати епізод раніше за бар входу — той самий вхід, той самий ключ
            if any(e["ticker"] == b and ets - 2 * BAR_15M <= e["ts"] <= ets for e in entries):
                continue
            entries.append({"ticker": b, "ts": ets, "probability": float(g.at[ets, "p15"]), "kind": "live", "reentry": not split})
    if split:                                   # епізод, у якому модель таки увійшла (бар входу поруч), — не late
        late = [e for e in late if not any(x["ticker"] == e["ticker"] and abs(x["ts"] - e["ts"]) <= 2 * BAR_15M for x in entries)]
    act = t[t["tier"] != "calm"]
    def quiet_before(e) -> bool:
        a = act[(act["base_asset"] == e["ticker"]) & (act["ts"] >= e["ts"] - pd.Timedelta(hours=1)) & (act["ts"] < e["ts"] - BAR_15M)]
        return a.empty
    cands = [e for e in cands if _in_trade(trades.get(e["ticker"], []), e["ts"]) is None and quiet_before(e)]
    key = lambda e: e["ts"]                                                       # noqa: E731
    return sorted(entries, key=key, reverse=True), sorted(re, key=key, reverse=True), cands, sorted(late, key=key, reverse=True)


def alert_since(g: pd.DataFrame) -> pd.Timestamp | None:
    """Початок поточного (останнього) епізоду Alert тікера: бари Alert з проміжками ≤ 1 год — один епізод
    (як train.signal_episodes). g — рядки tier_frame тікера (звичайні 15m-бари)."""
    a = g.loc[g["tier"] == "alert", "ts"].sort_values().to_list()
    if not a:
        return None
    start = a[-1]
    for ts in reversed(a[:-1]):
        if start - ts > pd.Timedelta(hours=1):
            break
        start = ts
    return start


def trade_state(tr: list[dict], row: dict, now: datetime, since: pd.Timestamp | None = None) -> dict | None:
    """Стан рядка списку за угодами основного режиму (власник 12.10, варіант A): entry — модель увійшла за останню годину і
    угода відкрита; open — угода відкрита давніше; closed — Alert ще горить, але модель у ЦЬОМУ епізоді вже відторгувала
    і вийшла (стоп або вихід за моделлю; since — початок епізоду, alert_since; KGEN 12.10: після стопу рядок показував «Late»);
    late — Alert зараз (або згас < 1 год), а угоди в цьому епізоді не було (режим не увійшов: не свіжий старт — split,
    фільтр режиму). None — нічого з цього (рядок показує звичайний статус)."""
    op = next((x for x in tr if x.get("open")), None)
    if op is not None:
        entered = pd.Timestamp(op["entry_ts"]) + BAR_15M
        recent = pd.Timestamp(now) - entered <= pd.Timedelta(hours=1)
        return {"state": "entry" if recent else "open", "entry_at": entered, "entry": op.get("entry"),
                "result_pct": op.get("result_pct")}
    sig = row.get("last_sig")
    alert_now = row.get("status") == "alert" or (sig and sig["tier"] == "alert" and pd.Timestamp(now) - sig["close"] <= pd.Timedelta(hours=1))
    if alert_now:
        last = max(tr, key=lambda x: x["entry_ts"]) if tr else None
        if last is not None and since is not None and last.get("exit_ts") is not None and pd.Timestamp(last["exit_ts"]) >= since:
            return {"state": "closed", "reason": last.get("reason"), "result_pct": last.get("result_pct"),
                    "entry": last.get("entry"), "exit": last.get("exit"), "exit_at": pd.Timestamp(last["exit_ts"]) + BAR_15M}
        return {"state": "late", "last_exit": last.get("exit_ts") + BAR_15M if last and last.get("exit_ts") is not None else None,
                "last_reason": last.get("reason") if last else None}
    return None


def fp_rows(conn, bases: list[str] | None = None, since: pd.Timestamp | None = None, outcomes: bool = False) -> list[dict]:
    """Мітки b / p з fp_marks (порожньо, якщо таблиці ще немає); outcomes — ще й результат «+2R / стоп / 12 год»."""
    try:
        m = footprint.marks(conn, bases, since)
        if outcomes and len(m):
            o = footprint.with_outcomes(conn, m)
            m = o if len(o) else m
        return m.to_dict("records")
    except Exception as e:
        conn.rollback()
        if "fp_marks" not in str(e):
            print(f"! мітки b/p: {e}")
        return []


# ---- список тікерів: важка частина рахується у фоні в окремому процесі одразу після нових оцінок (2026-10-06) ----
# Раніше кожен запит /api/list (сторінка, кожна нова оцінка, щохвилини з кожної вкладки) рахував ~7 с на сервері, а
# сповіщення чекали цього розрахунку. Тепер list_warmer помічає нові оцінки (mtime signals_stamp / wake_events.json)
# і одразу перераховує; запит віддає готове. Нові оцінки → запит чекає свіжий розрахунок (сповіщення мають бути
# актуальні); лише «застаріло за часом» (LIST_MAX_AGE, бо «N хв тому» і 6-год вікна зсуваються) → віддаємо наявне
# і оновлюємо у фоні.
LIST_MAX_AGE = 300        # с: без нових даних список перераховується раз на 5 хв (було 60 — половина ядра сервера на це)
LIST_SIGNAL_DAYS = 3      # скільки днів оцінок кожного тікера читає список (бари для угод — 2 доби); «тихо N год» — не більше цього
LIST_WARM_MODELS = (T.TIER_MODEL,)          # основна модель списку — гріємо завжди; решта — за запитом, з кешем
_list_pool = None
_list_lock = threading.Lock()
_list_cache: dict = {}                      # модель → {"key", "at", "data"}
_list_futures: dict = {}                    # модель → (key, future)


def _list_key() -> tuple:
    mt = lambda p: os.path.getmtime(p) if os.path.exists(p) else 0      # noqa: E731
    # склад списку — теж у ключі: список рахується лише для своїх тікерів, тож доданий тікер має викликати перерахунок
    return mt(SIGNALS_STAMP), mt(wake.EVENTS_FILE), tuple(sorted(list_tickers()))


def _watch_list_in_child(db_url: str, model: str) -> dict:
    """_watch_list в окремому процесі (моделі читає сам, один раз): не тримає процес дашборду."""
    if not State.models and State.m15 is None:
        State.models, State.m15 = load_models(), load_model_15m()
    with psycopg.connect(db_url) as conn:
        return _watch_list(conn, model)


def _list_store(model: str, key: tuple, fut) -> None:
    try:
        data = fut.result()
    except Exception as e:
        print(f"  ! список ({model}) у фоні: {type(e).__name__}: {e}")
        return
    with _list_lock:
        _list_cache[model] = {"key": key, "at": time.time(), "data": data}


def _list_future(model: str, key: tuple):
    """Запустити розрахунок списку для стану key (або взяти вже запущений для того самого стану)."""
    global _list_pool
    with _list_lock:
        f = _list_futures.get(model)
        if f is not None and f[0] == key and not f[1].done():
            return f[1]
        if _list_pool is None:
            from concurrent.futures import ProcessPoolExecutor
            _list_pool = ProcessPoolExecutor(max_workers=1)
        fut = _list_pool.submit(_watch_list_in_child, State.db_url, model)
        _list_futures[model] = (key, fut)
    fut.add_done_callback(lambda ff, m=model, k=key: _list_store(m, k, ff))
    return fut


def _watch_list_cached(conn, model: str) -> dict:
    key = _list_key()
    c = _list_cache.get(model)
    if c is not None and c["key"] == key:
        if time.time() - c["at"] >= LIST_MAX_AGE:
            _list_future(model, key)              # застаріло лише за часом — віддаємо наявне, оновлюємо у фоні
        return c["data"]
    try:
        fut = _list_future(model, key)
        data = fut.result(timeout=180)
        _list_store(model, key, fut)              # колбек міг ще не спрацювати — кладемо в кеш одразу
        return data
    except Exception as e:                        # процес не стартував / впав — рахуємо як раніше, тут
        print(f"  ! список в окремому процесі: {type(e).__name__}: {e} — рахую тут")
        global _list_pool
        _list_pool = None
        return _watch_list(conn, model)


def list_warmer() -> None:
    """Щойно скринер записав нові оцінки або «прокидається» — перерахувати список основної моделі у фоні."""
    while True:
        try:
            key = _list_key()
            for model in LIST_WARM_MODELS:
                c = _list_cache.get(model)
                if c is None or c["key"] != key or time.time() - c["at"] >= LIST_MAX_AGE:
                    _list_future(model, key)
        except Exception as e:
            print(f"  ! список (фон): {e}")
        time.sleep(2)


def watch_list(conn, model: str) -> dict:
    """Права панель: лише тікери з watchlist.txt і живого авто-списку сканера (прострочені авто — геть),
    з позначкою джерела: source manual / auto, причина і до коли тікер у списку.
    Важка частина (_watch_list) — з кешу, що гріється у фоні; джерела й «додається…» — свіжі на кожен запит."""
    res = dict(_watch_list_cached(conn, model))
    res.pop("minis", None)                      # міні-графіки вкладки «Charts» — окремим запитом /api/minicharts
    manual, auto = list_sources()
    keep = set(manual) | set(auto)
    woke = scanner.load_woke()      # події каналу WOKE (і ручних тікерів): first — «Woke · 3h» і година в секції NOW

    def src(b: str) -> dict:
        a = auto.get(b, {})
        w = woke.get(b)
        return {"source": "auto" if b in auto else "manual", "channel": scanner.channel(a) if a else None,
                "reason": a.get("reason"), "expires": a.get("expires"), "held": a.get("held"),
                "wokeup": {"first": w["first"], "last": w["last"], "reason": w.get("first_reason") or w["reason"]} if w else None}
    # Movers (movers.info): рамка в списку — ріст понад ринок, z ≥ 2.5 за 1H / 4H; з файлу щогодини
    res["rows"] = [r | src(r["ticker"]) | {"mover": movers.info(r["ticker"])} for r in res["rows"] if r["ticker"] in keep]
    have = {r["ticker"] for r in res["rows"]}
    res["wakes"] = [w for w in res.get("wakes", []) if w["ticker"] in keep]
    res["fps"] = [f for f in res.get("fps", []) if f["ticker"] in keep]
    res["pending"] = [{"ticker": b, **src(b), "job": add_status(b)} for b in manual + sorted(auto) if b not in have]
    # для бота: перші спрацювання WOKE за 6 год (повторні — не новина)
    cut = pd.Timestamp(datetime.now(timezone.utc) - timedelta(hours=6))
    res["wokeups"] = [{"ticker": b, "ts": w["first"], "reason": w.get("first_reason") or w["reason"],
                       **{k: w.get(k) for k in ("turnover_avg", "pace", "vol_x", "price_pct", "oi_pct")},
                       "source": "auto" if b in auto else "manual"}
                      for b, w in woke.items() if b in keep and pd.Timestamp(w["first"]) >= cut]
    return res


def _watch_list(conn, model: str) -> dict:
    if model == T.TIER_MODEL:
        return watch_list_15m(conn)
    sig = latest_signals(conn)
    sig = sig[sig["model"] == model]
    if sig.empty:
        return {"rows": [], "recent": [], "alerts": []}
    raw = load_raw(conn, sorted(sig["base_asset"].unique()))
    rows = []
    for base, g in sig.groupby("base_asset"):
        g = g.sort_values("ts")
        last = g.iloc[-1]
        streak = 0
        for x in reversed(g["signal"].tolist()):
            if not x:
                break
            streak += 1
        h = aggregate(raw, base, light=True)
        h = h[h.index <= last["ts"]] if not h.empty else h
        spark = g.set_index("ts")["probability"].reindex(pd.date_range(last["ts"] - pd.Timedelta(hours=SPARK_HOURS - 1), last["ts"], freq="h"))
        chg = oi24 = None
        if len(h) > 24:
            p0, p1 = h["price"].iloc[-25], h["price"].iloc[-1]
            chg = p1 / p0 - 1 if p0 else None
            oi24 = math.expm1(h["oi_log_idx"].iloc[-1] - h["oi_log_idx"].iloc[-25])   # без стрибків від лістингів
        rows.append({
            "ticker": base, "probability": last["probability"], "threshold": last["threshold"],
            "status": status_of(last["probability"], last["threshold"]),
            "streak": streak, "in_sample": bool(last["in_sample"]), "last_bar": last["ts"],
            "spark": spark.tolist(), "chg_24h": chg, "oi_24h": oi24,
        })
    rows.sort(key=lambda r: -(r["probability"] or 0))
    ep = episodes(sig)
    # для сповіщень: live-епізоди, що почались за останні 6 годин
    recent_cut = pd.Timestamp(datetime.now(timezone.utc) - timedelta(hours=6))
    alerts = [e for e in ep if e["kind"] == "live" and e["ts"] >= recent_cut]
    return {"rows": rows, "recent": ep[:30], "alerts": alerts}


LIMIT_ATR, LIMIT_VALID = 0.5, pd.Timedelta(hours=1)      # experiments/limit_entry.py: стабільний ліміт −0.5 ATR, 1 год


def alert_starts(t: pd.DataFrame) -> list[pd.Timestamp]:
    """Бари, на яких почався епізод Alert (рівень alert, попередній бар — ні)."""
    al = (t["tier"] == "alert").to_numpy() if len(t) else []
    return [t.index[i] for i in range(len(al)) if al[i] and (i == 0 or not al[i - 1])]


def ob_at_signal(rows, ts: pd.Timestamp) -> float | None:
    """OB у момент сигналу: перший знімок стакану за ≤ 12 хв після закриття сигнального бару (скринер пише після оцінки)."""
    return orderbook.at_signal(rows, ts)


def limit_hint(h: pd.DataFrame, ts: pd.Timestamp) -> dict | None:
    """Ліміт −0.5 ATR від закриття сигнального бару, дійсний 1 год після закриття; None — бару немає."""
    if ts not in h.index:
        return None
    a = atr(h).get(ts)
    if a is None or pd.isna(a):
        return None
    return {"price": float(h.at[ts, "price"] - LIMIT_ATR * a), "until": pd.Timestamp(ts) + BAR_15M + LIMIT_VALID}


_bars_cache: dict = {}                     # (тікер, дні) → (час, raw, h): легкий і повний запити перемикання тікера — одні дані
BARS_TTL = 20.0


def _bars_15m(conn, base: str, days: int, now: datetime):
    """Сирі 15m-ряди й агрегат тікера; перемикання тікера — два запити (легкий, потім повний), тож другий бере дані першого
    (до BARS_TTL с), а не перечитує базу й не агрегує заново."""
    key, t = (base, days), time.monotonic()
    hit = _bars_cache.get(key)
    if hit and t - hit[0] < BARS_TTL:
        return hit[1], hit[2]
    raw = load_raw(conn, [base], INTERVAL_15M, since=now - timedelta(days=days + INTRADAY_DAYS))
    h = aggregate(raw, base, "15min")
    if len(_bars_cache) > 200:
        _bars_cache.clear()
    _bars_cache[key] = (t, raw, h)
    return raw, h


def ticker_15m(conn, base: str, days: int, strat: dict | None = None, light: bool = False) -> dict:
    """Екран тікера для 15m + 1h: 15m-свічки, P 15m-моделі, годинний контекст, рівні сигналу, позиція.
    light=True («Повна інфо» вимкнена, 2026-10-06): без позиції / угод, стакану, «прокидається» — лише свічки, OI,
    обсяг, P, зони сигналу і картки ймовірності; помітно швидше."""
    m15 = State.m15
    now = datetime.now(timezone.utc)
    raw, h = _bars_15m(conn, base, days, now)
    if h.empty:
        raise ValueError(f"немає 15m-даних для {base}: натисни «Refresh» або дочекайся циклу скринера")
    h = h[h.index <= last_closed_15m()]
    # ознаки потрібні лише для останнього бару (картки): найдовше вікно — 24 год + 7 днів, тож 10 днів дають той самий
    # результат, що й уся історія (+10 днів запасу), а рахуються в рази швидше
    f = compute_features(h.iloc[-10 * 96:], 4)
    cur = f.dropna(subset=["oi_chg_4h"]).iloc[-1] if f["oi_chg_4h"].notna().any() else None
    t, troll = tier_rows(conn, base, rolling=True)
    t = t.set_index("ts").sort_index()
    # остання ковзна оцінка (5-хв цикл для «гарячих»), якщо вона новіша за останній оцінений 15m-бар: з неї приходять
    # сповіщення, тож графік показує її окремою точкою — інакше «Кандидат 0.87» у сповіщенні і 0.73 на графіку
    rolling = None
    if len(troll):
        rl = troll.sort_values("ts").iloc[-1]
        if not len(t) or rl["ts"] > t.index.max():
            rolling = {"ts": rl["ts"], "close": rl["ts"] + pd.Timedelta(minutes=15), "p15": rl["p15"], "thr15": rl["thr15"],
                       "s15": rl["s15"], "p1h": rl["p1h"], "tier": rl["tier"]}
    k, exit_bars = m15.get("risk", {}).get("k"), m15.get("exit_bars", 4)
    strat = strat or strategies.get(None)
    # стакан — для міток OB і фільтра режиму «OB ≥ 1»
    book_rows = []
    if not light:
        try:
            book_rows = orderbook.history(conn, base, (h.index.min() if len(h) else pd.Timestamp(now)).to_pydatetime())
        except Exception:
            conn.rollback()
    # позиція за режимом: сигнал (Alert / Кандидат), вхід (ринок / ліміт), фільтр OB; вихід спільний (частковий)
    score = t["s15"] if strat.get("signal") == "candidate" and len(t) else (t["alert_score"] if len(t) else None)
    if light or not len(t):
        pos = {"open": False, "trades": []}
    else:
        p_marks = footprint.p_times(conn, [base], h.index.min()).get(base, set()) if strat.get("p_be") else None
        entry_p = None
        if strategies.is_split(strat):             # режим split: вхід — модель «старт», утримання — основна або «старт»
            entry_p, score = T.split_scores(score, start_scores(conn, base))
        pos = position_state(h, score, 1.0, k, exit_bars, PARTIAL_EXIT, strategies.entry_rule(strat),
                             market.both(orderbook.allow_fn(book_rows) if strat.get("ob") else None,
                                         market.strategy_allow(conn, strat)), p_marks, entry_prob=entry_p)
        pos = live_stop(conn, base, pos, h)             # стоп на 5m-барах поточного, ще не закритого 15m-бару
    last = t.iloc[-1] if len(t) else None
    # Drivers: внески ознак у P останнього оціненого бару (drivers.py; ознаки — як у скринері, бар кешується)
    drv = None
    if last is not None:
        try:
            drv = DRV.for_bar(conn, m15, base, t.index.max())
        except Exception as e:
            print(f"! drivers {base}: {type(e).__name__}: {e}")
            conn.rollback()
    streak = 0
    if last is not None:
        for x in reversed(t["tier"].tolist()):
            if x != last["tier"] or x == "calm":
                break
            streak += 1
    view = h[h.index > h.index.max() - pd.Timedelta(days=days)]
    code = {"alert": 2, "candidate": 1, "calm": 0}
    chg24 = (h["price"].iloc[-1] / h["price"].iloc[-97] - 1) if len(h) > 96 and h["price"].iloc[-97] else None
    thr = float(last["thr15"]) if last is not None else m15["threshold"]
    trained = base in m15.get("tickers", []) or any(b == base for b, *_ in m15.get("in_sample_extra", []))
    # угода на історії (оцінки дораховані заднім числом, напр. коли тікер щойно потрапив у список) чи live —
    # на графіку історичні мітки бліді: такий сигнал ти б не отримав
    live = t["live"].astype(bool) if "live" in t else pd.Series(False, index=t.index)
    for tr in list(pos.get("trades", [])) + ([pos] if pos.get("open") else []):
        tr["live"] = bool(live.get(tr["entry_ts"], False))
    history = None if light else {**trade_stats(pos.get("trades", [])), "trained": trained,
                                  "days": round((t.index.max() - t.index.min()) / pd.Timedelta(days=1), 1) if len(t) else 0}
    # стакан (біди / аски ±2%) на початках Alert — зі знімків скринера; ліміт −0.5 ATR, поки дійсний
    starts = [s for s in alert_starts(t) if s >= view.index.min()]
    ob_marks = [{"ts": s, "v": v} for s in starts if (v := ob_at_signal(book_rows, s)) is not None]
    lim = limit_hint(h, starts[-1]) if starts else None          # підказка: ліміт −0.5 ATR від останнього Alert, 1 год
    if lim and lim["until"] <= pd.Timestamp(now):
        lim = None
    return {
        "ticker": base, "model": T.TIER_MODEL, "tf": "15m", "bar": 900, "threshold": thr, "k": k, "history": history,
        "streak": streak, "streak_min": streak * 15, "chg_24h": chg24,
        "price": h["price"].iloc[-1] if len(h) else None,
        "ls_long_pct": h["ls_long_pct"].dropna().iloc[-1] if h["ls_long_pct"].notna().any() else None,
        "exit_bars": exit_bars, "atr": last_atr(h),
        "last_bar": view.index.max(),
        "in_sample": bool(last["in_sample"]) if last is not None else None,
        "ctx": {"p1h": last["p1h"] if last is not None else None, "thr1h": last["thr1h"] if last is not None else None,
                "share": share(), "s1h": last["s1h"] if last is not None else None},
        "cards": {
            "probability": last["p15"] if last is not None else None,
            "status": last["tier"] if last is not None else "none",
            "oi_chg_4h": math.expm1(cur["oi_chg_4h"]) if cur is not None else None,
            "volume_z_7d": cur["volume_z_7d"] if cur is not None else None,
        },
        "position": pos, "ob_marks": ob_marks, "limit": lim, "rolling": rolling, "drivers": drv,
        # Кандидат / Alert, що були ЛИШЕ на ковзних вікнах 5-хв циклу (з них приходили сповіщення): на графіку — бліда зона і
        # кружечок «5m» на панелі P, щоб сигнал не зникав, коли звичайний 15m-бар оцінено нижче порогу (VIRTUAL 04.10 16:55)
        "rolling_sig": [{"ts": r.ts, "close": r.ts + pd.Timedelta(minutes=15), "p15": r.p15, "tier": r.tier}
                        for r in troll.itertuples() if r.tier in ("candidate", "alert") and len(view) and r.ts >= view.index.min()]
        if len(troll) else [],
        # «прокидається» (wake.py) на 5m-барах за видимий період: свічка й обсяг на графіку — фіолетові
        "wake": [] if light else wake_marks(conn, base, view.index.min() if len(view) else pd.Timestamp(now)),
        # мітки футпрінту b («залочка» на лоу) / p (на хаях) за видимий період — під / над свічкою, з результатом
        "fp": fp_rows(conn, [base], view.index.min() if len(view) else pd.Timestamp(now), outcomes=not light),
        "light": light,
        "series": {
            "t": list(view.index),
            "open": view["open"].tolist(), "price": view["price"].tolist(),
            "high": view["high"].tolist(), "low": view["low"].tolist(),
            "oi": view["oi"].tolist(), "volume": view["volume"].tolist(),
            "coinO": view["oi_coins_o"].tolist(), "coinH": view["oi_coins_h"].tolist(),
            "coinL": view["oi_coins_l"].tolist(), "coin": view["oi_coins_c"].tolist(),
            "usdO": view["oi_usd_o"].tolist(), "usdH": view["oi_usd_h"].tolist(), "usdL": view["oi_usd_l"].tolist(),
            "prob": {T.TIER_MODEL: t["p15"].reindex(view.index).tolist() if len(t) else [],
                     "ctx": t["p1h"].reindex(view.index).tolist() if len(t) else []},
            "tier": t["tier"].map(code).reindex(view.index).tolist() if len(t) else [],
        },
        "thresholds": {T.TIER_MODEL: thr},
    }


_oi_share_cache: dict = {}


_wake_cache: dict = {}                     # (тікер, день початку вікна) → (5-хв відрізок, мітки)


def wake_marks(conn, base: str, since: pd.Timestamp) -> list[dict]:
    """Мітки «прокидається» за видимий період. wake.history перечитує 5m-дані за ~38 днів (ZEC — до 2 с на перемикання,
    заміри 2026-10-11), тож результат кешується до наступного 5-хв бару: повторні відкриття тікера — миттєві."""
    since = pd.Timestamp(since)
    key, slot = (base, since.floor("1D")), pd.Timestamp.now(tz="UTC").floor("5min")
    hit = _wake_cache.get(key)
    if hit and hit[0] == slot:
        return [w for w in hit[1] if w["ts"] >= since]
    try:
        res = wake.history(conn, base, since.floor("1D"))
    except Exception as e:                     # 5m-даних може не бути — графік без міток, а не помилка
        print(f"! прокидається {base}: {e}")
        conn.rollback()
        return []
    if len(_wake_cache) > 500:
        _wake_cache.clear()
    _wake_cache[key] = (slot, res)
    return [w for w in res if w["ts"] >= since]


def oi_share(conn, base: str) -> list[dict] | None:
    """Частка OI тікера за біржами (середнє за 30 днів, годинні дані Coinalyze по всіх біржах) і чи бачить її модель:
    15m-модель бере лише features.INTRADAY_EXCHANGES. Кеш на годину."""
    hit = _oi_share_cache.get(base)
    if hit and time.time() - hit[0] < 3600:
        return hit[1]
    # середнє по кожному контракту, потім сума контрактів біржі (USDT і USDC на одній біржі — разом)
    rows = conn.execute("""SELECT e, sum(a) FROM (
                               SELECT m.exchange_name AS e, o.symbol, avg(o.c) AS a FROM open_interest o JOIN markets m USING (symbol)
                               WHERE m.base_asset = %s AND o.interval = '1hour' AND o.ts > now() - interval '30 days'
                               GROUP BY 1, 2) x GROUP BY 1""", (base,)).fetchall()
    total = sum(float(v or 0) for _, v in rows)
    res = None
    if total > 0:
        res = sorted(({"exchange": e, "share": float(v or 0) / total, "seen": e in INTRADAY_EXCHANGES} for e, v in rows),
                     key=lambda x: (not x["seen"], -x["share"]))
        res = [x for x in res if x["share"] >= 0.005]
    _oi_share_cache[base] = (time.time(), res)
    return res


def scanner_entries(conn, base: str, days: int) -> list:
    """Для авто-тікерів: коли сканер з поточними порогами додав би тікер у список, якби працював увесь час
    (15m-дані Binance з бази — як у живому скані). Для твоїх тікерів з watchlist.txt — нічого: вони в списку завжди."""
    manual, _ = list_sources()
    if base in manual:
        return []
    cfg = scanner.load_config()
    since = datetime.now(timezone.utc) - timedelta(days=days + scanner.AVG_DAYS + 1)   # + середній оборот за 30 днів
    h = aggregate(load_raw(conn, [base], INTERVAL_15M, since=since, exchanges=["Binance"]), base, "15min")
    if h.empty:
        return []
    h = h[h.index <= last_closed_15m()]
    cut = pd.Timestamp(datetime.now(timezone.utc) - timedelta(days=days))
    return [t for t in scanner.entries_on_history(h, scanner.params(cfg), cfg["ttl_hours"]) if t >= cut]


CHART_TF = {"5m": ("5min", "5min", 300), "15m": ("15min", "15min", 900), "1h": ("1hour", "h", 3600)}


def rebar(conn, res: dict, base: str, tf: str, days: int) -> dict:
    """Графік на іншому таймфреймі, ніж рідний для моделі. Модель і позиція лишаються своїми; змінюються бари:
    крупніший ТФ — максимум оцінок і рівня за бар (Alert усередині години не губиться);
    дрібніший — остання оцінка, відома на момент закриття бару (нічого з майбутнього).
    Мітки входу/виходу — на бар, що закривається разом з баром моделі."""
    if tf not in CHART_TF or tf == res["tf"]:
        return res
    interval, freq, sec = CHART_TF[tf]
    nb, cb = pd.Timedelta(seconds=res["bar"]), pd.Timedelta(seconds=sec)
    now = datetime.now(timezone.utc)
    h = aggregate(load_raw(conn, [base], interval, since=now - timedelta(days=days + 1)), base, freq)
    if h.empty:
        raise ValueError(f"немає {tf}-даних для {base}" + (" — вони з'являться після циклу скринера" if tf == "5m" else ""))
    h = h[h.index <= pd.Timestamp(now).floor(freq) - cb]
    view = h[h.index > h.index.max() - pd.Timedelta(days=days)]
    s = res["series"]
    native = pd.DataFrame({f"p_{k}": v for k, v in s["prob"].items() if len(v)}, index=pd.DatetimeIndex(s["t"]))
    if s.get("tier"):
        native["tier"] = s["tier"]
    native = native.apply(pd.to_numeric, errors="coerce")
    if cb >= nb:                       # крупніший: максимум за бар
        mapped = native.groupby(native.index.floor(freq)).max().reindex(view.index)
    else:                              # дрібніший: остання оцінка з закриттям ≤ закриття бару графіка
        left = pd.DataFrame({"close": view.index + cb})
        right = native.assign(close=native.index + nb).sort_values("close")
        mapped = pd.merge_asof(left, right, on="close", direction="backward").drop(columns="close")
        mapped.index = view.index
    snap = lambda ts: (pd.Timestamp(ts) + nb - pd.Timedelta(microseconds=1)).floor(freq) if ts is not None else None  # noqa: E731
    pos = dict(res["position"])
    for tr in [pos] + [pos.get("trades", [])][0]:
        for key in ("entry_ts", "exit_ts", "partial_ts"):
            if tr.get(key) is not None:
                tr[key] = snap(tr[key])
    res["ob_marks"] = [{**m, "ts": snap(m["ts"])} for m in res.get("ob_marks", [])]
    res.update({
        "tf": tf, "bar": sec, "native_tf": res["tf"], "position": pos, "last_bar": view.index.max(),
        "series": {
            "t": list(view.index),
            "open": view["open"].tolist(), "price": view["price"].tolist(),
            "high": view["high"].tolist(), "low": view["low"].tolist(),
            "oi": view["oi"].tolist(), "volume": view["volume"].tolist(),
            "coinO": view["oi_coins_o"].tolist(), "coinH": view["oi_coins_h"].tolist(),
            "coinL": view["oi_coins_l"].tolist(), "coin": view["oi_coins_c"].tolist(),
            "usdO": view["oi_usd_o"].tolist(), "usdH": view["oi_usd_h"].tolist(), "usdL": view["oi_usd_l"].tolist(),
            "prob": {k: mapped[f"p_{k}"].tolist() if f"p_{k}" in mapped else [] for k in s["prob"]},
            "tier": mapped["tier"].tolist() if "tier" in mapped else [],
        },
    })
    return res


_fresh_at: dict = {}
_fresh_lock = threading.Lock()


def ensure_fresh(conn, base: str, tf: str) -> bool:
    """Графік відкритого тікера не чекає циклу скринера: якщо в базі бракує останніх закритих 15m/5m-барів,
    дозавантажуємо короткий хвіст з бірж у ФОНІ (~10 с, без Coinalyze) — відповідь не чекає. Повертає True, якщо
    дозавантаження запущено: сторінка тоді перезапитає дані за ~15 с. Не частіше разу на 45 с для тікера й ТФ.
    Модель це не переоцінює — вона рахується скринером на закритих 15m-барах."""
    if tf not in ("5m", "15m"):
        return False
    interval, freq, sec = CHART_TF[tf]
    key = (base, tf)
    with _fresh_lock:
        if time.monotonic() - _fresh_at.get(key, -1e9) < 45:
            return False
        _fresh_at[key] = time.monotonic()
    last = conn.execute("""SELECT max(o.ts) FROM ohlcv o JOIN markets m USING (symbol)
                           WHERE m.base_asset = %s AND o.interval = %s""", (base, interval)).fetchone()[0]
    closed = pd.Timestamp(datetime.now(timezone.utc)).floor(freq) - pd.Timedelta(seconds=sec)
    if last is None or pd.Timestamp(last) >= closed:
        return False
    days = min(2.0, max(0.05, (closed - pd.Timestamp(last)).total_seconds() / 86400 + 0.02))

    def work():
        try:
            with psycopg.connect(State.db_url) as c2:
                fetch_exchange.collect(c2, [base], [tf], days, log=lambda m: None)
        except Exception as e:
            print(f"! дозавантаження {base} {tf}: {e}")
    threading.Thread(target=work, daemon=True).start()
    return True


def live_symbols(conn, base: str) -> dict:
    """Символи USDT-перпів на Binance і Bybit — для живої свічки на сторінці (WebSocket бірж напряму);
    OKX — лише для панелі розбіжності бірж (ціна, без обсягу й OI)."""
    rows = dict(conn.execute("""SELECT exchange_name, symbol_on_exchange FROM markets
                                WHERE base_asset = %s AND quote_asset = 'USDT' AND exchange_name IN ('Binance', 'Bybit', 'OKX')""",
                             (base,)).fetchall())
    return {"binance": rows.get("Binance"), "bybit": rows.get("Bybit"), "okx": rows.get("OKX"),
            "price_exchange": "Binance" if "Binance" in rows else ("Bybit" if "Bybit" in rows else None)}


def exchange_diffs(conn, base: str, res: dict) -> dict | None:
    """Панель «Δ бірж» на графіку (spread.py, 13.10): закриття бару на кожній біржі відносно еталонної біржі графіка
    (Binance, а якщо її немає — Bybit), частка: 0.01 = на 1% дорожче. Бари — ті самі, що на графіку (5m / 15m з бірж,
    1h з Coinalyze). None — тікер торгується лише на одній з трьох бірж."""
    t = res.get("series", {}).get("t") or []
    interval = {"5m": "5min", "15m": "15min", "1h": "1hour"}.get(res.get("tf"))
    if not t or interval is None:
        return None
    rows = conn.execute("""SELECT m.exchange_name, o.ts, o.c FROM ohlcv o JOIN markets m USING (symbol)
                           WHERE m.base_asset = %s AND m.quote_asset = 'USDT' AND m.exchange_name = ANY(%s)
                             AND o.interval = %s AND o.ts >= %s AND o.ts <= %s""",
                        (base, list(spread.EXCHANGES), interval, t[0], t[-1])).fetchall()
    if not rows:
        return None
    w = pd.DataFrame(rows, columns=["ex", "ts", "c"]).pivot_table(index="ts", columns="ex", values="c", aggfunc="last")
    w.index = pd.to_datetime(w.index, utc=True)
    w = w.reindex(pd.DatetimeIndex(t)).astype(float)
    w = w.where(w > 0)
    ref = next((e for e in spread.EXCHANGES if e in w), None)
    others = [e for e in spread.EXCHANGES if e in w and e != ref]
    if ref is None or not others:
        return None
    return {"ref": ref, "diff": {e: (w[e] / w[ref] - 1).tolist() for e in others}}


_ticker_cache: dict = {}                    # (тікер, модель, дні, ТФ, режим, light) → (стан, час, відповідь)
_inflight: dict = {}                        # той самий ключ → threading.Event, поки відповідь рахується
_inflight_lock = threading.Lock()
TICKER_CACHE_SEC = 30


def ticker(conn, base: str, model: str, days: int, tf: str | None = None, strategy: str | None = None,
           light: bool = False) -> dict:
    """Дані графіка тікера. Кеш: та сама відповідь до нових оцінок скринера (signals_stamp) і не довше 30 с —
    перемикання між двома вікнами і назад відкриває тікер миттєво (2026-10-06)."""
    key = (base, model, days, tf, strategy, light)
    state = _list_key()
    c = _ticker_cache.get(key)
    if c is not None and c[0] == state and time.time() - c[1] < TICKER_CACHE_SEC:
        return c[2]
    # однаковий запит, що вже рахується (два вікна, пульс сторінки, бот), — чекаємо його, а не рахуємо вдруге:
    # інакше під навантаженням повільні запити накладались і гальмували один одного
    with _inflight_lock:
        ev = _inflight.get(key)
        owner = ev is None
        if owner:
            ev = _inflight[key] = threading.Event()
    if not owner:
        ev.wait(timeout=90)
        c = _ticker_cache.get(key)
        if c is not None and c[0] == state:
            return c[2]
    try:
        res = _ticker(conn, base, model, days, tf, strategy, light)
        if not res.get("tail_refresh"):             # дані ще дозавантажуються — не кешуємо
            if len(_ticker_cache) > 64:
                _ticker_cache.clear()
            _ticker_cache[key] = (state, time.time(), res)
        return res
    finally:
        if owner:
            with _inflight_lock:
                _inflight.pop(key, None)
            ev.set()


def _ticker(conn, base: str, model: str, days: int, tf: str | None = None, strategy: str | None = None,
            light: bool = False) -> dict:
    strat = strategies.get(strategy) if strategy else None
    if strat is not None:                       # режим задає модель: 15m-режими — tier15, решта — годинні
        model = T.TIER_MODEL if strategies.is15(strat) else strat["model"]
    native = "15m" if model == T.TIER_MODEL else "1h"
    refreshing = ensure_fresh(conn, base, tf or native)
    if model == T.TIER_MODEL:
        if State.m15 is None:
            raise ValueError("15m-моделі немає: python train.py")
        res = ticker_15m(conn, base, days, strat, light)
    else:
        res = ticker_1h(conn, base, model, days)
    res = rebar(conn, res, base, tf, days) if tf else res
    res["live"] = live_symbols(conn, base)
    try:
        res["xs"] = exchange_diffs(conn, base, res)
    except Exception as e:                      # панель розбіжності — доповнення: без неї графік має відкритись
        print(f"! розбіжність бірж {base}: {type(e).__name__}: {e}")
        conn.rollback()
        res["xs"] = None
    res["tail_refresh"] = refreshing
    if refreshing:                              # свіжі бари ще дописуються у фоні — наступний запит має перечитати базу
        _bars_cache.pop((base, days), None)
    res["oi_share"] = None if light else oi_share(conn, base)
    res["scanner_entries"] = [] if light else scanner_entries(conn, base, days)
    return res


def ticker_1h(conn, base: str, model: str, days: int) -> dict:
    raw = load_raw(conn, [base])
    h = aggregate(raw, base)
    if h.empty:
        raise ValueError(f"немає даних для {base}: додай у watchlist.txt і запусти скринер")
    sig = latest_signals(conn, base)
    probs = {m: g.set_index("ts")["probability"] for m, g in sig.groupby("model")}
    thresholds = {m: float(g["threshold"].iloc[-1]) for m, g in sig.groupby("model")}
    bundle = next((m for m in State.models if m["name"] == model), None)
    thr = thresholds.get(model, bundle["threshold"] if bundle else 0.5)
    k = (bundle or {}).get("risk", {}).get("k")

    last_closed = pd.Timestamp(datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0) - timedelta(hours=1))
    h = h[h.index <= last_closed]
    f = compute_features(h)
    cur = f.dropna(subset=["oi_chg_4h"]).iloc[-1] if f["oi_chg_4h"].notna().any() else None

    p = probs.get(model, pd.Series(dtype=float))
    pos = position_state(h, p, thr, k) if len(p) else {"open": False}
    last = sig[sig["model"] == model].iloc[-1] if (sig["model"] == model).any() else None

    view = h[h.index > h.index.max() - pd.Timedelta(days=days)]
    streak = 0
    for x in reversed(sig[sig["model"] == model]["signal"].tolist()):
        if not x:
            break
        streak += 1
    chg24 = (h["price"].iloc[-1] / h["price"].iloc[-25] - 1) if len(h) > 24 and h["price"].iloc[-25] else None
    return {
        "ticker": base, "model": model, "tf": "1h", "bar": 3600, "threshold": thr, "k": k, "streak": streak,
        "streak_min": streak * 60, "chg_24h": chg24,
        "price": h["price"].iloc[-1] if len(h) else None,
        "ls_long_pct": h["ls_long_pct"].dropna().iloc[-1] if h["ls_long_pct"].notna().any() else None,
        "exit_bars": EXIT_BARS, "atr": last_atr(h),
        "last_bar": view.index.max(),
        "in_sample": bool(last["in_sample"]) if last is not None else None,
        "cards": {
            "probability": last["probability"] if last is not None else None,
            "status": status_of(last["probability"] if last is not None else None, thr),
            "oi_chg_4h": math.expm1(cur["oi_chg_4h"]) if cur is not None else None,
            "volume_z_7d": cur["volume_z_7d"] if cur is not None else None,
        },
        "position": pos,
        "series": {
            "t": list(view.index),
            "open": view["open"].tolist(), "price": view["price"].tolist(),
            "high": view["high"].tolist(), "low": view["low"].tolist(),
            "oi": view["oi"].tolist(), "volume": view["volume"].tolist(),
            # свічки OI у монетах (без впливу ціни) — див. features.aggregate
            "coinO": view["oi_coins_o"].tolist(), "coinH": view["oi_coins_h"].tolist(),
            "coinL": view["oi_coins_l"].tolist(), "coin": view["oi_coins_c"].tolist(),
            "usdO": view["oi_usd_o"].tolist(), "usdH": view["oi_usd_h"].tolist(), "usdL": view["oi_usd_l"].tolist(),
            "prob": {m: s.reindex(view.index).tolist() for m, s in probs.items()},
        },
        "thresholds": thresholds,
    }


def refresh(conn, base: str) -> dict:
    """Кнопка «Refresh»: свіжі дані тікера з Coinalyze і нова оцінка обома моделями."""
    if not State.api_key:
        raise ValueError("немає COINALYZE_API_KEY у .env")
    started = time.monotonic()
    last_closed = pd.Timestamp(datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0) - timedelta(hours=1))
    with refresh_lock:
        # Скринер щогодини оновлює watchlist. Якщо остання закрита година вже в базі, в API не йдемо:
        # скринер і дашборд ділять один ліміт Coinalyze, і зайві запити упираються в 429 з очікуванням до хвилини.
        have = conn.execute("""SELECT max(o.ts) FROM ohlcv o JOIN markets m USING (symbol)
                               WHERE m.base_asset = %s AND o.interval = '1hour'""", (base,)).fetchone()[0]
        fetched = have is None or pd.Timestamp(have) < last_closed
        if fetched:
            if State.client is None:
                State.client = Coinalyze(State.api_key)
            if not collect(State.client, conn, [base], FETCH_DAYS, verbose=False):
                raise ValueError(f"{base} не знайдено на біржах")
        scored, last_closed = score_bases(conn, [base], State.models)
        n15 = 0
        if State.m15 is not None:          # 15m: хвіст з бірж і ліквідації (дешево), потім оцінка
            if State.client is None:
                State.client = Coinalyze(State.api_key)
            update_15m(conn, State.client, [base])
            s15, _ = score_15m(conn, [base], State.m15)
            n15 = len(s15)
    if scored.empty and not n15:
        raise ValueError(f"для {base} ще замало історії: потрібно ≥ 7 днів")
    return {"ok": True, "fetched": fetched, "seconds": round(time.monotonic() - started, 1), "last_closed": last_closed}


# ---------------------------------------------------------------- сканер ринку і додавання тікерів

ADD_JOBS: dict = {}          # тікер → "loading" / "done" / текст помилки (дашборд показує прогрес)
_universe_cache = {"at": 0.0, "bases": set()}


def binance_bases() -> set[str]:
    """Базові активи USDT-перпів Binance (кеш 10 хв) — щоб «＋ у watchlist» не приймав друкарських помилок."""
    if time.time() - _universe_cache["at"] > 600 or not _universe_cache["bases"]:
        _universe_cache.update(at=time.time(), bases={u["base"] for u in scanner.universe()})
    return _universe_cache["bases"]


def list_sources() -> tuple[list[str], dict]:
    """Твій watchlist.txt і живий авто-список сканера."""
    from fetch import excluded_tickers
    excluded = excluded_tickers(ROOT)
    manual = [t.upper() for t in read_tickers_file(os.path.join(ROOT, WATCHLIST_FILE)) if t.upper() not in excluded]
    auto = {b: r for b, r in scanner.active_auto().items() if b not in set(manual)}
    return manual, auto


_list_tickers: dict = {"at": 0.0, "set": set()}


def list_tickers() -> set[str]:
    """Тікери списку (твої + авто) для монітора розбіжності; файли перечитуємо не частіше разу на 30 с."""
    if time.time() - _list_tickers["at"] > 30:
        try:
            manual, auto = list_sources()
            _list_tickers.update(at=time.time(), set=set(manual) | set(auto))
        except Exception as e:
            print(f"  ! список для spread: {e}")
            _list_tickers["at"] = time.time()
    return _list_tickers["set"]


# ---------- TWAP Hyperliquid (hl_twap.py, контейнер hltwap, лише в пам'яті; 18.10) ----------
HLTWAP_URL = os.getenv("HLTWAP_URL", "http://hltwap:8790/twaps")
TWAP_CACHE_SEC = 3
TWAP_EVENT_HOURS = 2            # скільки тримати події «новий TWAP» для сповіщень (браузер, Telegram)
TWAP_CONFIG_FILE = os.path.join(ROOT, "twap_config.json")       # фільтри — одні для секції, звуку й Telegram (19.10, власник)
TWAP_DEFAULTS = {"min_vol_pct": 3.0, "min_usd_k": 0.0, "binance_only": True, "alert_open_only": True}
TWAP_LIMITS = {"min_vol_pct": (0.0, 100.0), "min_usd_k": (0.0, 100000.0)}
_twaps: dict = {"at": 0.0, "data": None, "primed": False}
_twap_alerted: dict[int, float] = {}         # id → коли сповістили (або побачили при старті — без сповіщення)
_twap_events: list[dict] = []


def twap_config() -> dict:
    try:
        with open(TWAP_CONFIG_FILE, encoding="utf-8") as f:
            saved = json.load(f)
    except (OSError, ValueError):
        saved = {}
    return {**TWAP_DEFAULTS, **{k: v for k, v in saved.items() if k in TWAP_DEFAULTS}}


def save_twap_config(body: dict) -> dict:
    cfg = twap_config()
    for k, v in body.items():
        if k not in TWAP_DEFAULTS:
            continue
        if isinstance(TWAP_DEFAULTS[k], bool):
            cfg[k] = bool(v)
        else:
            lo, hi = TWAP_LIMITS[k]
            x = float(v)
            if not lo <= x <= hi:
                raise ValueError(f"{k}: {lo}…{hi}")
            cfg[k] = x
    tmp = TWAP_CONFIG_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=1)
    os.replace(tmp, TWAP_CONFIG_FILE)
    # нові фільтри — з наступного запиту; TWAP, що вже активні й тепер проходять м'якший фільтр, — без сповіщень
    # (інакше кожна зміна порогу засипала б звуком і Telegram старими TWAP)
    _twaps.update(at=0.0, primed=False)
    return cfg


def twaps_view(conn) -> dict:
    """Активні TWAP Hyperliquid з контейнера hltwap + % Vol: швидкість TWAP ($ за хвилину) ÷ оборот монети за хвилину за
    останню годину на Binance (закриті 5-хв бари trade_bars від traderec) і Hyperliquid разом. in_list — монета у списку.
    shown — проходить фільтри (twap_config.json); events — «новий TWAP пройшов фільтри» за TWAP_EVENT_HOURS (кожен — раз;
    після старту дашборда вже активні TWAP подій не дають), для звуку в дашборді й Telegram."""
    if _twaps["data"] is not None and time.time() - _twaps["at"] < TWAP_CACHE_SEC:
        return _twaps["data"]
    import requests
    cfg = twap_config()
    try:
        d = requests.get(HLTWAP_URL, timeout=3).json()
    except Exception as e:
        return {"ok": False, "error": f"TWAP feed unavailable ({type(e).__name__})", "twaps": [], "config": cfg, "events": _twap_events}
    syms = sorted({t["binance"] for t in d.get("twaps", []) if t.get("binance")})
    bmin = {}
    if syms:
        rows = conn.execute("""SELECT symbol, sum(v_buy + v_sell), count(*) FROM trade_bars
                               WHERE exchange = 'Binance' AND symbol = ANY(%s)
                                 AND ts >= now() - interval '65 minutes' AND ts < now() - interval '5 minutes'
                               GROUP BY 1""", (syms,)).fetchall()
        bmin = {s: float(v) / (n * 5) for s, v, n in rows if n}        # $ за хвилину по наявних закритих барах
    lst = list_tickers()
    now = time.time()
    for t in d.get("twaps", []):
        b = bmin.get(t["binance"]) if t.get("binance") else None
        flow = (t.get("hl_min") or 0) + (b or 0)
        t["binance_min"] = b
        t["pct_vol"] = (t["usd"] / t["minutes"]) / flow if flow > 0 and t.get("minutes") else None
        t["base"] = t["binance"][:-4] if t.get("binance") else t["coin"]
        t["in_list"] = t["base"] in lst
        t["depth_pct"] = t["slice_usd"] / t["depth_usd"] if t.get("depth_usd") else None
        t["dpx"] = t["px"] / t["px_start"] - 1 if t.get("px") and t.get("px_start") else None
        t["shown"] = bool((not cfg["binance_only"] or t.get("binance")) and t["pct_vol"] is not None
                          and t["pct_vol"] * 100 >= cfg["min_vol_pct"] and t["usd"] >= cfg["min_usd_k"] * 1000)
        if t["shown"] and (t["open"] or not cfg["alert_open_only"]) and t["id"] not in _twap_alerted:
            _twap_alerted[t["id"]] = now
            if _twaps["primed"]:                                   # після старту дашборда — лише нові
                _twap_events.append({"ts": now, **{k: t.get(k) for k in ("id", "coin", "base", "binance", "side", "open", "pct_vol", "usd",
                                                                         "minutes", "started", "slice_usd", "depth_pct", "dpx", "in_list")}})
    _twaps["primed"] = True
    cut = now - TWAP_EVENT_HOURS * 3600
    _twap_events[:] = [e for e in _twap_events if e["ts"] >= cut]
    for k in [k for k, v in _twap_alerted.items() if v < now - 3 * 86400]:     # пам'ять: старі TWAP
        del _twap_alerted[k]
    d.update(ok=True, config=cfg, events=_twap_events)
    _twaps.update(at=now, data=d)
    return d


def scan_view(q: dict) -> dict:
    """Картка «Сканер»: налаштування, скільки пройшло б зараз, результат набору на розмічених пампах, авто-список.
    Параметри запиту (preset або пороги «Свій») — попередній перегляд ДО збереження."""
    cfg = scanner.load_config()
    manual, auto = list_sources()
    preset = q.get("preset", cfg["preset"])
    if preset not in scanner.PRESET_NAMES:
        preset = cfg["preset"]
    p = dict(cfg["custom"] if preset == "custom" else scanner.PRESETS[preset])
    for k in scanner.PRESETS["balance"]:
        if k in q:
            p[k] = scanner._clip(k, q[k])
    p["max_oi"] = cfg["max_oi"]
    return {"config": cfg, "preset": preset, "params": p,
            "presets": {k: {"name": v, "params": scanner.PRESETS.get(k, cfg["custom"])} for k, v in scanner.PRESET_NAMES.items()},
            "preview": scanner.preview(p, set(manual)), "recall": scanner.recall_for(p, cfg["ttl_hours"]),
            "auto": [{"ticker": b, **r} for b, r in sorted(auto.items(), key=lambda x: x[1]["last_hit"], reverse=True)],
            "manual": manual, "jobs": {b: add_status(b) for b in set(ADD_JOBS) | set(ADD_PROCS)}}


def add_to_watchlist(base: str) -> dict:
    """«＋ у watchlist»: дописати у watchlist.txt і одразу (у фоні) — лише швидку частину: контракти, 15m/5m з бірж,
    15m- і годинна оцінка з власних 15m (quick_add). Годинну історію Coinalyze скринер дозбирає сам у своїй черзі —
    раніше тут був повний Refresh (60 днів Coinalyze), і дашборд зі скринером ловили 429 на спільному ліміті (07.10, GRIFFAIN)."""
    from fetch import set_ticker_in_file
    base = base.upper().removesuffix("USDT").strip()
    if not base.isalnum():
        raise ValueError("введи базовий актив, напр. BANK")
    if base not in binance_bases():
        raise ValueError(f"{base}: немає серед USDT-перпів Binance")
    set_ticker_in_file(os.path.join(ROOT, WATCHLIST_FILE), base, True, "додано з дашборду")
    # окремим процесом з нижчим пріоритетом: потік у процесі дашборда (ознаки, модель) ділив з запитами графіків
    # один GIL — поки додавався тікер, графіки гальмували (07.10, BEAT)
    import subprocess
    import tempfile
    log = open(os.path.join(tempfile.gettempdir(), f"pd_add_{base}.log"), "w", encoding="utf-8")
    kw = {"preexec_fn": lambda: os.nice(10)} if hasattr(os, "nice") else {}
    ADD_PROCS[base] = (subprocess.Popen([sys.executable, os.path.abspath(__file__), "--quick-add", base],
                                        stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, **kw), log.name)
    ADD_JOBS[base] = "loading"
    return {"ok": True, "ticker": base}


ADD_PROCS: dict = {}         # тікер → (процес quick_add, файл журналу)


def add_status(base: str) -> str | None:
    """Стан «＋ у watchlist»: loading / done / остання строка помилки процесу quick_add."""
    item = ADD_PROCS.get(base)
    if item is not None:
        p, log = item
        rc = p.poll()
        if rc is None:
            return "loading"
        ADD_PROCS.pop(base, None)
        if rc == 0:
            ADD_JOBS[base] = "done"
        else:
            try:
                lines = [x.strip() for x in open(log, encoding="utf-8", errors="replace") if x.strip()]
            except OSError:
                lines = []
            ADD_JOBS[base] = lines[-1][:200] if lines else f"помилка (код {rc})"
    return ADD_JOBS.get(base)


def quick_add(conn, base: str) -> None:
    """Новий тікер без годинної історії Coinalyze: контракти бірж (2 виклики Coinalyze, якщо їх ще немає в markets),
    10 днів 15m/5m з бірж (без ліміту Coinalyze), 15m-оцінка і годинна з власних 15m — графік і сигнали за хвилину."""
    from features import INTRADAY_EXCHANGES
    from fetch import find_markets, upsert_markets
    have = conn.execute("SELECT 1 FROM markets WHERE base_asset = %s AND exchange_name = ANY(%s) LIMIT 1",
                        (base, list(INTRADAY_EXCHANGES))).fetchone()
    if not have:
        if State.client is None:
            State.client = Coinalyze(State.api_key)
        ms = find_markets(State.client, [base])
        if not ms:
            raise ValueError(f"{base}: контрактів не знайдено")
        upsert_markets(conn, ms)
        conn.commit()
    fetch_exchange.collect(conn, [base], ["15m", "5m"], 10, log=lambda m: None)
    if State.m15 is not None:
        score_15m(conn, [base], State.m15)
        score_bases_own(conn, [base], State.models)


REMOVED: dict = {}      # «×» у списку: base → ("manual" | "auto", запис авто-списку) — для «Повернути»


def remove_from_watchlist(base: str) -> dict:
    """«×» у рядку списку: ручний тікер — геть з watchlist.txt, авто — з авто-списку; обидва блокуються «до нового
    спрацювання» (scanner.block): той самий сплеск, що й досі проходить фільтр, їх не поверне, а новий — поверне як авто.
    Скринер не оцінює тікер з наступного циклу."""
    from fetch import set_ticker_in_file
    base = base.upper()
    manual, _ = list_sources()
    out = {"ok": True, "ticker": base, "source": None}
    if base in manual:
        set_ticker_in_file(os.path.join(ROOT, WATCHLIST_FILE), base, False)
        REMOVED[base] = ("manual", None)
        out["source"] = "manual"
    rec = scanner.remove_auto(base)
    if rec is not None:
        REMOVED[base] = ("auto", rec)
        out["source"] = "auto"
    scanner.block(base)
    ADD_JOBS.pop(base, None)
    return out


def undo_remove(base: str) -> dict:
    """«Повернути» після «×»: ручний — назад у watchlist.txt (без перезавантаження даних), авто — запис і без паузи."""
    from fetch import set_ticker_in_file
    base = base.upper()
    item = REMOVED.pop(base, None)
    if item is None:
        raise ValueError(f"{base}: нічого повертати")
    kind, rec = item
    if kind == "manual":
        set_ticker_in_file(os.path.join(ROOT, WATCHLIST_FILE), base, True, "повернено з дашборду")
        scanner.unblock(base)
    else:
        scanner.restore_auto(base, rec)
    return {"ok": True, "ticker": base}


# ---------------------------------------------------------------- бектест (backtest.py) — вкладка «Бектест»

BT_DIR = os.path.join(ROOT, "reports", "backtests")
BT_JOB: dict = {"proc": None, "log": [], "args": None}


def backtest_runs() -> list[dict]:
    """Список прогонів: мета й підсумок (без угод) з reports/backtests/*.json, нові першими."""
    out = []
    if not os.path.isdir(BT_DIR):
        return out
    for f in sorted(os.listdir(BT_DIR), reverse=True):
        if not f.endswith(".json") or f.startswith("_"):
            continue
        try:
            with open(os.path.join(BT_DIR, f), encoding="utf-8") as fh:
                d = json.load(fh)
        except (OSError, ValueError):
            continue
        out.append({**d["meta"], **{k: d.get(k) for k in ("coins", "trades", "sum", "sum_wo_top3", "median", "win_rate")}})
    return out


def backtest_run(run_id: str) -> dict:
    if not run_id or "/" in run_id or "\\" in run_id or ".." in run_id:
        raise ValueError("невірний id прогону")
    path = os.path.join(BT_DIR, run_id + ".json")
    if not os.path.exists(path):
        raise ValueError(f"прогону {run_id} немає")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def backtest_status() -> dict:
    p = BT_JOB["proc"]
    running = p is not None and p.poll() is None
    st = {}
    try:
        with open(os.path.join(BT_DIR, "_status.json"), encoding="utf-8") as fh:
            st = json.load(fh)
    except (OSError, ValueError):
        pass
    return {"running": running, "code": None if p is None or running else p.returncode, **st,
            "log": BT_JOB["log"][-4:], "args": BT_JOB["args"]}


def backtest_start(body: dict) -> dict:
    """Запуск backtest.py у фоні (окремий процес — довгий: дані для ~100 монет ~30–60 хв першого разу)."""
    import subprocess
    if backtest_status()["running"]:
        raise ValueError("бектест уже йде")
    args = ["--min-turnover", str(float(body.get("min_turnover", 20e6))), "--days", str(int(body.get("days", 30))),
            *(["--max-turnover", str(float(body["max_turnover"]))] if body.get("max_turnover") else []),
            "--model", "candidate" if body.get("model") == "candidate" else "current",
            "--strategy", strategies.get(body.get("strategy"))["id"]]
    BT_JOB.update(log=[], args=args)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    p = subprocess.Popen([sys.executable, os.path.join(ROOT, "backtest.py"), *args], cwd=ROOT, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", env=env)
    BT_JOB["proc"] = p

    def pump():
        for line in p.stdout:
            if line.strip():
                BT_JOB["log"] = (BT_JOB["log"] + [line.rstrip()])[-60:]
    threading.Thread(target=pump, daemon=True).start()
    return {"ok": True}


def rescore_on_start() -> None:
    """Після перенавчання в signals ще немає оцінок нової версії моделі, а дашборд показує лише її.
    Оцінюємо наявні в базі дані одразу, не чекаючи наступного циклу скринера (без запитів до API)."""
    try:
        with psycopg.connect(State.db_url) as conn:
            known = {r[0] for r in conn.execute("SELECT DISTINCT base_asset FROM markets")}
            bases = [t for t in read_tickers_file(os.path.join(ROOT, WATCHLIST_FILE)) if t.upper() in known]
            if not bases:
                return
            versions = [m["trained_at"] for m in State.models]
            have = {r[0] for r in conn.execute(
                "SELECT DISTINCT base_asset FROM signals WHERE model_version = ANY(%s)", (versions,))}
            missing = [b.upper() for b in bases if b.upper() not in have]
            if missing:
                with refresh_lock:
                    score_bases(conn, missing, State.models)
                print(f"Оцінено новою моделлю: {', '.join(missing)}")
            if State.m15 is not None:
                have15 = {r[0] for r in conn.execute(
                    "SELECT DISTINCT base_asset FROM signals WHERE model = %s AND model_version = %s",
                    (J.M15, State.m15["trained_at"]))}
                missing15 = [b.upper() for b in bases if b.upper() not in have15]
                if missing15:
                    with refresh_lock:
                        score_15m(conn, missing15, State.m15)
                    print(f"Оцінено 15m-моделлю: {', '.join(missing15)}")
    except Exception as e:
        print(f"! не вдалося оцінити тікери при старті: {e}")


_journal_cache: dict = {}
# угоди всіх моделей (journal.prepare) — ключ: правило «Impulse», версія оцінок, розмітка; період лише фільтрує їх
_prepared: dict = {"key": None, "data": None, "rule": dict(J.DEFAULT_RULE)}
_prepared_lock = threading.Lock()


def journal_bundles() -> dict:
    """Режими для журналу: {id режиму: {scores, bundle, strat}} (strategies.json); режим без своєї моделі — пропускається."""
    hourly = {m["name"]: m for m in State.models}
    out = {}
    for s in strategies.load():
        bundle = State.m15 if strategies.is15(s) else hourly.get(s["model"])
        if bundle is not None:
            out[s["id"]] = {"scores": strategies.scores_name(s), "bundle": bundle, "strat": s}
    return out


def journal_version(conn) -> tuple:
    """Коли журнал треба перерахувати: оцінено НОВИЙ бар (без ковзних 5-хв — вони в журнал не йдуть) або змінена розмітка.
    Ключ — останній оцінений бар і скільки тікерів на ньому, а не max(scored_at): скринер постійно дописує дооцінки
    (нові тікери, історія, годинна частина), і журнал на сервері (~80 с на 2 vCPU) перераховувався майже без перерви —
    дашборд їв ~60% процесора і гальмував скринер (2026-10-06)."""
    last = conn.execute("""SELECT ts, count(*) FROM signals
                           WHERE model <> %s AND ts = (SELECT max(ts) FROM signals WHERE model <> %s) GROUP BY ts""",
                        (T.ROLL_MODEL, T.ROLL_MODEL)).fetchone()
    p = os.path.join(ROOT, "pumps.csv")
    return last, os.path.getmtime(p) if os.path.exists(p) else 0


_journal_pool = None


def _prepare_in_child(db_url: str, rule: dict) -> dict:
    """J.prepare в окремому процесі: важкий pandas-розрахунок журналу (до хвилини на сервері) інакше тримає GIL
    процесу дашборду, і графік тікера чи список, запитані в цей момент, чекали в черзі (2026-10-06).
    Моделі дочірній процес читає сам — один раз (дашборд теж читає їх лише при старті)."""
    if not State.models and State.m15 is None:
        State.models, State.m15 = load_models(), load_model_15m()
    with psycopg.connect(db_url) as conn:
        return J.prepare(conn, journal_bundles(), rule)


def _prepare(conn, rule: dict) -> dict:
    global _journal_pool
    try:
        if _journal_pool is None:
            from concurrent.futures import ProcessPoolExecutor
            _journal_pool = ProcessPoolExecutor(max_workers=1)
        return _journal_pool.submit(_prepare_in_child, State.db_url, dict(rule)).result()
    except Exception as e:                    # процес не стартував / впав — рахуємо як раніше, у цьому ж процесі
        print(f"  ! журнал в окремому процесі: {type(e).__name__}: {e} — рахую тут")
        _journal_pool = None
        return J.prepare(conn, journal_bundles(), rule)


def prepared_for(conn, rule: dict) -> tuple[dict, tuple]:
    key = (tuple(rule.values()),) + journal_version(conn)
    with _prepared_lock:                      # фоновий прогрів і запит не рахують одне й те саме двічі
        if _prepared["key"] != key:
            _prepared["data"] = _prepare(conn, rule)
            _prepared["key"] = key
        _prepared["rule"] = dict(rule)
        return _prepared["data"], key


JOURNAL_KEEP_WARM = 2 * 3600                   # с: стільки після останнього звернення до журналу фон тримає його свіжим
_journal_used = {"at": 0.0}                    # коли журнал востаннє запитували (вкладка «Журнал», «Результати» бота)


def journal_warmer() -> None:
    """Фоновий перерахунок угод — ЛИШЕ поки журналом користуються (13.10, власник: «завантажувати тільки коли потрібно»):
    перше відкриття після перерви рахує журнал на запит (на сервері ~1–2 хв), далі JOURNAL_KEEP_WARM секунд після
    останнього звернення фон оновлює його з кожним новим баром, і вкладка відкривається одразу. Ніхто не дивиться —
    нічого не рахується (раніше журнал рахувався цілодобово, двічі на кожен 15m-бар)."""
    seen = None
    while True:
        try:
            if time.time() - _journal_used["at"] < JOURNAL_KEEP_WARM:
                with psycopg.connect(State.db_url) as conn:
                    # Скринер оцінює бар двома порціями (гарячі тікери першими, решта — за 1–4 хв): версія журналу
                    # змінюється двічі на бар. Рахуємо, коли вона однакова на двох перевірках поспіль.
                    v = journal_version(conn)
                    if v == seen:
                        prepared_for(conn, _prepared["rule"])
                    seen = v
            else:
                seen = None
        except Exception as e:  # журнал не має валити дашборд
            print(f"  ! журнал (фон): {e}")
        time.sleep(60)


def journal_view(conn, q: dict) -> dict:
    _journal_used["at"] = time.time()           # журналом користуються — фон тримає його свіжим ще JOURNAL_KEEP_WARM
    rule = {
        "horizon": max(1, min(168, int(q.get("horizon", J.DEFAULT_RULE["horizon"])))),
        "price": float(q.get("price", J.DEFAULT_RULE["price"])),
        "oi": float(q.get("oi", J.DEFAULT_RULE["oi"])),
    }
    days = None if q.get("days", "30") == "all" else max(1, int(q.get("days", 30)))
    model, kind = strategies.get(q.get("model"))["id"], q.get("kind", "hist")       # model = id режиму
    # вибір дат (YYYY-MM-DD, дні UTC включно) — замість «останніх N днів»
    day = lambda s: pd.Timestamp(s, tz="UTC").normalize() if s else None  # noqa: E731
    d_from, d_to = day(q.get("from")), day(q.get("to"))
    if d_from is not None and d_to is not None and d_to < d_from:
        d_from, d_to = d_to, d_from
    prepared, pkey = prepared_for(conn, rule)
    key = (model, days, kind, d_from, d_to) + pkey
    if key not in _journal_cache:
        if len(_journal_cache) > 32:
            _journal_cache.clear()
        res = J.journal(conn, journal_bundles(), model, days, kind, rule, d_from, d_to, prepared)
        val = {m["name"]: m["validation"] for m in State.models}
        for v in virtual_models():
            val[v["name"]] = v["validation"]
        res["validation"] = {sid: val.get(spec["scores"]) for sid, spec in journal_bundles().items()}
        _journal_cache[key] = res
    return _journal_cache[key]


# ---------------------------------------------------------------- сервер

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send(self, code: int, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else json.dumps(clean(body), ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _route(self, fn):
        try:
            with psycopg.connect(State.db_url) as conn:
                return self._send(200, fn(conn))
        except ValueError as e:
            return self._send(400, {"error": str(e)})
        except Exception as e:  # помилка бази чи API не має валити сервер
            return self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        model = q.get("model", T.TIER_MODEL if State.m15 else "lightgbm")
        if url.path == "/":
            with open(PAGE, "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")
        if url.path.startswith(("/vendor/", "/js/")):   # локальна копія бібліотеки графіків, інтернет не потрібен
            base_dir = VENDOR if url.path.startswith("/vendor/") else JS_DIR
            path = os.path.normpath(os.path.join(base_dir, os.path.basename(url.path)))
            if not path.startswith(base_dir) or not os.path.isfile(path):
                return self._send(404, {"error": "не знайдено"})
            with open(path, "rb") as f:
                return self._send(200, f.read(), "text/javascript; charset=utf-8")
        if url.path == "/api/pulse":
            # дешева перевірка «чи є нове» (без бази): mtime мітки нових оцінок скринера і подій «прокидається»
            mt = lambda p: os.path.getmtime(p) if os.path.exists(p) else 0      # noqa: E731
            return self._send(200, {"sig": mt(SIGNALS_STAMP), "wake": mt(wake.EVENTS_FILE), "woke": mt(scanner.WOKE_FILE)})
        if url.path == "/api/meta":
            return self._route(meta)
        if url.path == "/api/list":
            return self._route(lambda c: watch_list(c, model))
        if url.path == "/api/minicharts":       # вкладка «Charts»: міні-графіки тікерів із секцій NOW / WOKE UP
            syms = [x for x in q.get("symbols", "").upper().split(",") if x]
            return self._route(lambda c: minicharts(c, model, syms, q.get("tf", "15m")))
        if url.path == "/api/journal":
            return self._route(lambda c: journal_view(c, q))
        if url.path == "/api/footprint":
            # футпрінт для графіка (кнопка FP): рівні купівель / продажів по барах ТФ у видимому діапазоні (≤ 3 доби)
            tf = {"5m": 300, "15m": 900, "1h": 3600}.get(q.get("tf", "15m"), 900)
            a, b = int(q.get("from", 0)), int(q.get("to", 0))
            if b <= a:
                return self._send(400, {"error": "from / to"})
            a = max(a, b - 3 * 86400)
            return self._route(lambda c: footprint.levels(c, q.get("symbol", "").upper(), pd.Timestamp(a, unit="s", tz="UTC"),
                                                          pd.Timestamp(b, unit="s", tz="UTC"), tf))
        if url.path == "/api/fp_stats":
            # підсумок міток b / p за дні UTC from…to включно (бот: «📊 Результати»)
            day = lambda s: pd.Timestamp(s, tz="UTC").normalize()      # noqa: E731
            a = day(q.get("from")) if q.get("from") else pd.Timestamp.now(tz="UTC").normalize()
            b = day(q.get("to")) if q.get("to") else a
            return self._route(lambda c: footprint.stats(c, a, b + pd.Timedelta(days=1)))
        if url.path in ("/api/backtests", "/api/backtest", "/api/backtest/status"):
            try:
                body = {"/api/backtests": lambda: {"runs": backtest_runs()},
                        "/api/backtest": lambda: backtest_run(q.get("id", "")),
                        "/api/backtest/status": backtest_status}[url.path]()
                return self._send(200, body)
            except ValueError as e:
                return self._send(400, {"error": str(e)})
        if url.path == "/api/scan":
            try:
                return self._send(200, scan_view(q))
            except ValueError as e:
                return self._send(400, {"error": str(e)})
        if url.path == "/api/movers":          # монети з власним рухом (кнопка «Movers»): reports/movers.json від market.py
            return self._send(200, {k: v for k, v in movers.load().items() if k != "all"})   # «all» — для чіпа / рамки, не таблиці
        if url.path == "/api/market":          # стан ринку альтів (кнопка «Market» над графіком): market_breadth
            return self._route(lambda c: market.api(c, int(q.get("days", 7))))
        if url.path == "/api/book":            # біди / аски ±2% наживо (Binance + Bybit), кеш кілька секунд
            return self._route(lambda c: orderbook.live(c, q.get("symbol", "").upper()) or {"ratio": None})
        if url.path == "/api/twaps":            # активні TWAP Hyperliquid (hl_twap.py) + % Vol з обороту Binance
            return self._route(twaps_view)
        if url.path == "/api/spreads":          # розбіжність цін між біржами (spread.py): події, тікери списку, налаштування
            return self._send(200, spread.api(State.spread))
        if url.path == "/api/spread":           # картка тікера: бід / аск, розмір на кращій ціні, funding, 3 години
            if State.spread is None:
                return self._send(400, {"error": "монітор розбіжності не запущено"})
            return self._send(200, State.spread.detail(q.get("symbol", "").upper()))
        if url.path == "/api/profile":          # профіль обсягу: дні / тижні / фрагмент (app/js/profile.js)
            return self._route(lambda c: volume_profile.api(c, q))
        if url.path == "/api/ticker":
            days = max(1, min(60, int(q.get("days", 30))))
            sym = q.get("symbol", "").upper()
            # чіп «MOVER» у заголовку вікна — поза кешем ticker(): movers оновлюються щогодини, незалежно від оцінок
            return self._route(lambda c: {**ticker(c, sym, model, days, q.get("tf"), q.get("strategy"), q.get("light") == "1"),
                                          "mover": movers.info(sym)})
        return self._send(404, {"error": "не знайдено"})

    def do_POST(self):
        url = urlparse(self.path)
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        if url.path == "/api/refresh":
            return self._route(lambda c: refresh(c, str(body.get("symbol", "")).upper()))
        try:
            if url.path == "/api/backtest/run":
                return self._send(200, backtest_start(body))
            if url.path == "/api/spread/config":
                cfg = spread.save_config(body)
                if State.spread is not None:
                    State.spread.cfg = cfg
                return self._send(200, {"ok": True, "config": cfg})
            if url.path == "/api/scan/config":
                return self._send(200, {"ok": True, "config": scanner.save_config(body)})
            if url.path == "/api/twaps/config":       # фільтри TWAP · HL — одні для секції, звуку й Telegram
                return self._send(200, {"ok": True, "config": save_twap_config(body)})
            if url.path == "/api/watchlist/add":
                return self._send(200, add_to_watchlist(str(body.get("symbol", ""))))
            if url.path == "/api/watchlist/remove":
                return self._send(200, remove_from_watchlist(str(body.get("symbol", ""))))
            if url.path == "/api/watchlist/undo":
                return self._send(200, undo_remove(str(body.get("symbol", ""))))
        except ValueError as e:
            return self._send(400, {"error": str(e)})
        except Exception as e:
            return self._send(500, {"error": f"{type(e).__name__}: {e}"})
        return self._send(404, {"error": "не знайдено"})


def main() -> None:
    parser = argparse.ArgumentParser(description="Дашборд Pump Detector")
    parser.add_argument("--port", type=int, default=8766, help="порт")
    parser.add_argument("--host", default="127.0.0.1",
                        help="адреса, на якій слухати (у Docker-контейнері на сервері — 0.0.0.0; назовні порт однаково "
                             "прив'язаний лише до 127.0.0.1 сервера, див. docker-compose.server.yml)")
    parser.add_argument("--no-browser", action="store_true", help="не відкривати браузер автоматично")
    parser.add_argument("--quick-add", metavar="TICKER",
                        help="лише швидке додавання тікера (контракти, 15m/5m з бірж, оцінка) і вихід — так його запускає сам "
                             "дашборд окремим процесом після «＋ у watchlist», щоб не гальмувати графіки")
    args = parser.parse_args()

    load_dotenv(os.path.join(ROOT, ".env"))
    State.db_url = os.getenv("DATABASE_URL") or ""
    State.api_key = os.getenv("COINALYZE_API_KEY") or ""
    if not State.db_url:
        sys.exit("Заповни DATABASE_URL у файлі .env")
    State.models = load_models()
    State.m15 = load_model_15m()
    if args.quick_add:                          # окремий процес «＋ у watchlist» (add_to_watchlist)
        with psycopg.connect(State.db_url) as conn:
            quick_add(conn, args.quick_add.upper())
        return
    if State.m15 is None:
        print("15m-моделі немає (python train.py) — показую лише годинні моделі")
    threading.Thread(target=rescore_on_start, daemon=True).start()
    # фоновий прогрів журналу — лише якщо не вимкнено: на сервері з 20% базової потужності CPU (Lightsail) він
    # з'їдав кредити процесора; там журнал рахується при відкритті вкладки (JOURNAL_WARMER=0 у docker-compose.server.yml)
    if os.getenv("JOURNAL_WARMER", "1") != "0":
        threading.Thread(target=journal_warmer, daemon=True).start()
    if os.getenv("LIST_WARMER", "1") != "0":
        threading.Thread(target=list_warmer, daemon=True).start()
    # розбіжність цін між біржами (spread.py): три запити на весь ринок кожні 15 с, у пам'яті; SPREAD_MONITOR=0 — вимкнути
    if os.getenv("SPREAD_MONITOR", "1") != "0":
        State.spread = spread.Monitor(State.db_url, list_tickers).start()

    # На Windows SO_REUSEADDR дозволяє двом процесам зайняти один порт: другий «стартує», але запитів не отримує.
    # Вимикаємо його, щоб повторний запуск одразу казав, що порт зайнятий.
    ThreadingHTTPServer.allow_reuse_address = False
    try:
        server = ThreadingHTTPServer((args.host, args.port), Handler)
    except OSError:
        sys.exit(f"Порт {args.port} зайнятий: мабуть, уже запущено інший екземпляр. "
                 f"Зупини його (Ctrl+C у тому терміналі) або запусти з --port 8767.")
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Pump Detector: {url}  (Ctrl+C — зупинити)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nЗупинено.")


if __name__ == "__main__":
    main()
