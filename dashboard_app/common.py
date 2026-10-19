"""
dashboard_app/common.py — спільне для екранів дашборда: шляхи сторінки, стан сервера (State), оцінки моделей з таблиці signals, рівні
Candidate / Alert, склад списку тікерів (watchlist.txt + авто-список сканера), /api/meta.
"""

import json
import math
import os
import threading
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import journal as J
import scanner
import spread
import strategies
import tiers as T
from features import query_df
from fetch import Coinalyze, read_tickers_file
from risk import atr
from screener import STATUS_FILE, WATCHLIST_FILE, next_hourly, next_run


def last_atr(h: pd.DataFrame) -> float | None:
    """Поточний ATR(14) на барах h — для стопу «якщо вхід зараз»."""
    a = atr(h).dropna()
    return float(a.iloc[-1]) if len(a) else None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))    # корінь проєкту (пакет — на рівень нижче)
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
