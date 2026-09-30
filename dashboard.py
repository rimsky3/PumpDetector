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

from features import INTERVAL_15M, INTRADAY_EXCHANGES, aggregate, compute_features, load_raw, query_df
from fetch import Coinalyze, collect, read_tickers_file
from risk import EXIT_BARS, TRAIL_NATR, atr, position_state, trade_stats


def last_atr(h: pd.DataFrame) -> float | None:
    """Поточний ATR(14) на барах h — для стопу й trailing «якщо вхід зараз»."""
    a = atr(h).dropna()
    return float(a.iloc[-1]) if len(a) else None
from screener import (FETCH_DAYS, INTRADAY_DAYS, STATUS_FILE, WATCHLIST_FILE, last_closed_15m, load_model_15m,
                      load_models, next_hourly, next_run, score_15m, score_bases, update_15m)
import fetch_exchange
import scanner
import journal as J
import tiers as T

ROOT = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.join(ROOT, "app", "dashboard.html")
VENDOR = os.path.join(ROOT, "app", "vendor")
JS_DIR = os.path.join(ROOT, "app", "js")                              # спільні скрипти сторінок
MODEL_NAMES = {"lightgbm": "LightGBM", "logreg": "Логістична регресія"}
CANDIDATE_SHARE = 0.5     # «Кандидат»: ймовірність ≥ половини порогу сигналу
SPARK_HOURS = 24
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


def latest_signals(conn, base: str | None = None) -> pd.DataFrame:
    """Оцінки лише найсвіжішої версії кожної моделі."""
    where = "WHERE s.base_asset = %s" if base else ""
    return query_df(conn, f"""
        SELECT s.base_asset, s.ts, s.model, s.probability, s.threshold, s.signal, s.live, s.in_sample, s.scored_at
        FROM signals s
        JOIN (SELECT model, max(model_version) AS v FROM signals GROUP BY model) m
          ON s.model = m.model AND s.model_version = m.v
        {where}
        ORDER BY s.base_asset, s.ts""", (base,) if base else ())


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


def share() -> float:
    return (State.m15 or {}).get("confirm", {}).get("share", 0.75)


def tier_rows(conn, base: str | None = None) -> pd.DataFrame:
    """Рівні сигналу (tiers.py) з оцінок найсвіжіших версій 15m- і годинного LightGBM."""
    sig = latest_signals(conn, base)
    return T.tier_frame(sig[sig["model"] == J.M15], sig[sig["model"] == J.CONFIRM_MODEL], share())


def virtual_models() -> list[dict]:
    """Моделі 15m-контуру для перемикача і журналу: tier15 — «Alert», cand15 — «Кандидат» (лише журнал)."""
    m = State.m15
    if m is None:
        return []
    common = {"tf": "15m", "bar": 900, "trained_at": m["trained_at"], "threshold": m["threshold"], "tickers": m["tickers"],
              "pump_intervals": m["pump_intervals"], "risk": m.get("risk", {}), "settings": m.get("settings", {}),
              "confirm": m.get("confirm", {}), "exit_bars": m.get("exit_bars")}
    return [{"name": T.TIER_MODEL, "title": "15m + 1h", "validation": m.get("tier_validation", {}), **common},
            {"name": T.CAND_MODEL, "title": "15m кандидат", "validation": m.get("validation", {}), **common}]


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
        "journal_models": virt + hourly,          # журнал: ще й «Кандидат» окремим рядком
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
        prev_ts, prev_sig = None, False
        for r in g.itertuples():
            if r.signal and not (prev_sig and prev_ts is not None and r.ts - prev_ts == bar):
                kind = "live" if r.live else ("in-sample" if r.in_sample else "history")
                out.append({"ticker": base, "ts": r.ts, "probability": r.probability, "kind": kind})
            prev_ts, prev_sig = r.ts, r.signal
    return sorted(out, key=lambda e: e["ts"], reverse=True)


def watch_list_15m(conn) -> dict:
    """Права панель для 15m + 1h: P 15m-моделі, рівень сигналу, міні-графік P за 24 год (96 барів)."""
    t = tier_rows(conn)
    if t.empty:
        return {"rows": [], "recent": [], "alerts": []}
    now = datetime.now(timezone.utc)
    raw = load_raw(conn, sorted(t["base_asset"].unique()), INTERVAL_15M, since=now - timedelta(days=2))
    rows = []
    for base, g in t.groupby("base_asset"):
        g = g.sort_values("ts")
        last = g.iloc[-1]
        streak = 0
        for x in reversed(g["tier"].tolist()):
            if x != last["tier"] or x == "calm":
                break
            streak += 1
        h = aggregate(raw, base, "15min")
        h = h[h.index <= last["ts"]] if not h.empty else h
        spark = g.set_index("ts")["p15"].reindex(pd.date_range(last["ts"] - 95 * BAR_15M, last["ts"], freq="15min"))
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
        })
    order = {"alert": 0, "candidate": 1, "calm": 2}
    rows.sort(key=lambda r: (order.get(r["status"], 3), -(r["probability"] or 0)))
    al = t.assign(signal=t["tier"] == "alert", probability=t["p15"])
    ep = episodes(al, BAR_15M)
    recent_cut = pd.Timestamp(now - timedelta(hours=6))
    alerts = [e for e in ep if e["kind"] == "live" and e["ts"] >= recent_cut]
    return {"rows": rows, "recent": ep[:6], "alerts": alerts}


def watch_list(conn, model: str) -> dict:
    """Права панель: лише тікери з watchlist.txt і живого авто-списку сканера (прострочені авто — геть),
    з позначкою джерела: source manual / auto, причина і до коли тікер у списку."""
    res = _watch_list(conn, model)
    manual, auto = list_sources()
    keep = set(manual) | set(auto)
    res["rows"] = [r | {"source": "auto" if r["ticker"] in auto else "manual",
                        "reason": auto.get(r["ticker"], {}).get("reason"), "expires": auto.get(r["ticker"], {}).get("expires"),
                        "held": auto.get(r["ticker"], {}).get("held")}
                   for r in res["rows"] if r["ticker"] in keep]
    have = {r["ticker"] for r in res["rows"]}
    res["pending"] = [{"ticker": b, "source": "auto" if b in auto else "manual", "reason": auto.get(b, {}).get("reason"),
                       "job": ADD_JOBS.get(b)} for b in manual + sorted(auto) if b not in have]
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
        h = aggregate(raw, base)
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
    return {"rows": rows, "recent": ep[:6], "alerts": alerts}


def ticker_15m(conn, base: str, days: int) -> dict:
    """Екран тікера для 15m + 1h: 15m-свічки, P 15m-моделі, годинний контекст, рівні сигналу, позиція."""
    m15 = State.m15
    now = datetime.now(timezone.utc)
    raw = load_raw(conn, [base], INTERVAL_15M, since=now - timedelta(days=days + INTRADAY_DAYS))
    h = aggregate(raw, base, "15min")
    if h.empty:
        raise ValueError(f"немає 15m-даних для {base}: натисни «Refresh» або дочекайся циклу скринера")
    h = h[h.index <= last_closed_15m()]
    f = compute_features(h, 4)
    cur = f.dropna(subset=["oi_chg_4h"]).iloc[-1] if f["oi_chg_4h"].notna().any() else None
    t = tier_rows(conn, base).set_index("ts").sort_index()
    k, exit_bars = m15.get("risk", {}).get("k"), m15.get("exit_bars", 4)
    pos = position_state(h, t["alert_score"], 1.0, k, exit_bars) if len(t) else {"open": False}
    last = t.iloc[-1] if len(t) else None
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
    history = {**trade_stats(pos.get("trades", [])), "trained": trained,
               "days": round((t.index.max() - t.index.min()) / pd.Timedelta(days=1), 1) if len(t) else 0}
    return {
        "ticker": base, "model": T.TIER_MODEL, "tf": "15m", "bar": 900, "threshold": thr, "k": k, "history": history,
        "streak": streak, "streak_min": streak * 15, "chg_24h": chg24,
        "price": h["price"].iloc[-1] if len(h) else None,
        "ls_long_pct": h["ls_long_pct"].dropna().iloc[-1] if h["ls_long_pct"].notna().any() else None,
        "trail_natr": TRAIL_NATR, "exit_bars": exit_bars, "atr": last_atr(h),
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
        "position": pos,
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
    since = datetime.now(timezone.utc) - timedelta(days=days + 8)        # +8 днів на 7-денну норму обсягу
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
        for key in ("entry_ts", "exit_ts"):
            if tr.get(key) is not None:
                tr[key] = snap(tr[key])
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
    """Символи USDT-перпів на Binance і Bybit — для живої свічки на сторінці (WebSocket бірж напряму)."""
    rows = dict(conn.execute("""SELECT exchange_name, symbol_on_exchange FROM markets
                                WHERE base_asset = %s AND quote_asset = 'USDT' AND exchange_name IN ('Binance', 'Bybit')""",
                             (base,)).fetchall())
    return {"binance": rows.get("Binance"), "bybit": rows.get("Bybit"),
            "price_exchange": "Binance" if "Binance" in rows else ("Bybit" if "Bybit" in rows else None)}


def ticker(conn, base: str, model: str, days: int, tf: str | None = None) -> dict:
    native = "15m" if model == T.TIER_MODEL else "1h"
    refreshing = ensure_fresh(conn, base, tf or native)
    if model == T.TIER_MODEL:
        if State.m15 is None:
            raise ValueError("15m-моделі немає: python train.py")
        res = ticker_15m(conn, base, days)
    else:
        res = ticker_1h(conn, base, model, days)
    res = rebar(conn, res, base, tf, days) if tf else res
    res["live"] = live_symbols(conn, base)
    res["tail_refresh"] = refreshing
    res["oi_share"] = oi_share(conn, base)
    res["scanner_entries"] = scanner_entries(conn, base, days)
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
        "trail_natr": TRAIL_NATR, "exit_bars": EXIT_BARS, "atr": last_atr(h),
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
    manual = [t.upper() for t in read_tickers_file(os.path.join(ROOT, WATCHLIST_FILE))]
    auto = {b: r for b, r in scanner.active_auto().items() if b not in set(manual)}
    return manual, auto


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
            "manual": manual, "jobs": ADD_JOBS}


def add_to_watchlist(base: str) -> dict:
    """«＋ у watchlist»: дописати у watchlist.txt і одразу підтягнути дані й оцінку (у фоні, як Refresh)."""
    from fetch import set_ticker_in_file
    base = base.upper().removesuffix("USDT").strip()
    if not base.isalnum():
        raise ValueError("введи базовий актив, напр. BANK")
    if base not in binance_bases():
        raise ValueError(f"{base}: немає серед USDT-перпів Binance")
    set_ticker_in_file(os.path.join(ROOT, WATCHLIST_FILE), base, True, "додано з дашборду")
    ADD_JOBS[base] = "loading"

    def work():
        try:
            with psycopg.connect(State.db_url) as conn:
                refresh(conn, base)
            ADD_JOBS[base] = "done"
        except Exception as e:
            ADD_JOBS[base] = f"{type(e).__name__}: {e}"
    threading.Thread(target=work, daemon=True).start()
    return {"ok": True, "ticker": base}


def remove_from_watchlist(base: str) -> dict:
    from fetch import set_ticker_in_file
    set_ticker_in_file(os.path.join(ROOT, WATCHLIST_FILE), base.upper(), False)
    ADD_JOBS.pop(base.upper(), None)
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


def journal_view(conn, q: dict) -> dict:
    rule = {
        "horizon": max(1, min(168, int(q.get("horizon", J.DEFAULT_RULE["horizon"])))),
        "price": float(q.get("price", J.DEFAULT_RULE["price"])),
        "oi": float(q.get("oi", J.DEFAULT_RULE["oi"])),
    }
    days = None if q.get("days", "30") == "all" else max(1, int(q.get("days", 30)))
    model, kind = q.get("model", T.TIER_MODEL if State.m15 else "lightgbm"), q.get("kind", "live")
    last = conn.execute("SELECT max(scored_at) FROM signals").fetchone()[0]
    labels_mtime = os.path.getmtime(os.path.join(ROOT, "pumps.csv")) if os.path.exists(os.path.join(ROOT, "pumps.csv")) else 0
    key = (model, days, kind, tuple(rule.values()), last, labels_mtime)
    if key not in _journal_cache:
        _journal_cache.clear()
        bundles = {m["name"]: m for m in State.models}
        if State.m15 is not None:
            bundles = {T.TIER_MODEL: State.m15, T.CAND_MODEL: State.m15, **bundles}
        res = J.journal(conn, bundles, model, days, kind, rule)
        res["validation"] = {m["name"]: m["validation"] for m in State.models}
        for v in virtual_models():
            res["validation"][v["name"]] = v["validation"]
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
        if url.path == "/api/meta":
            return self._route(meta)
        if url.path == "/api/list":
            return self._route(lambda c: watch_list(c, model))
        if url.path == "/api/journal":
            return self._route(lambda c: journal_view(c, q))
        if url.path == "/api/scan":
            try:
                return self._send(200, scan_view(q))
            except ValueError as e:
                return self._send(400, {"error": str(e)})
        if url.path == "/api/ticker":
            days = max(1, min(60, int(q.get("days", 30))))
            return self._route(lambda c: ticker(c, q.get("symbol", "").upper(), model, days, q.get("tf")))
        return self._send(404, {"error": "не знайдено"})

    def do_POST(self):
        url = urlparse(self.path)
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        if url.path == "/api/refresh":
            return self._route(lambda c: refresh(c, str(body.get("symbol", "")).upper()))
        try:
            if url.path == "/api/scan/config":
                return self._send(200, {"ok": True, "config": scanner.save_config(body)})
            if url.path == "/api/watchlist/add":
                return self._send(200, add_to_watchlist(str(body.get("symbol", ""))))
            if url.path == "/api/watchlist/remove":
                return self._send(200, remove_from_watchlist(str(body.get("symbol", ""))))
        except ValueError as e:
            return self._send(400, {"error": str(e)})
        except Exception as e:
            return self._send(500, {"error": f"{type(e).__name__}: {e}"})
        return self._send(404, {"error": "не знайдено"})


def main() -> None:
    parser = argparse.ArgumentParser(description="Дашборд Pump Detector")
    parser.add_argument("--port", type=int, default=8766, help="порт на 127.0.0.1")
    parser.add_argument("--no-browser", action="store_true", help="не відкривати браузер автоматично")
    args = parser.parse_args()

    load_dotenv(os.path.join(ROOT, ".env"))
    State.db_url = os.getenv("DATABASE_URL") or ""
    State.api_key = os.getenv("COINALYZE_API_KEY") or ""
    if not State.db_url:
        sys.exit("Заповни DATABASE_URL у файлі .env")
    State.models = load_models()
    State.m15 = load_model_15m()
    if State.m15 is None:
        print("15m-моделі немає (python train.py) — показую лише годинні моделі")
    threading.Thread(target=rescore_on_start, daemon=True).start()

    # На Windows SO_REUSEADDR дозволяє двом процесам зайняти один порт: другий «стартує», але запитів не отримує.
    # Вимикаємо його, щоб повторний запуск одразу казав, що порт зайнятий.
    ThreadingHTTPServer.allow_reuse_address = False
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
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
