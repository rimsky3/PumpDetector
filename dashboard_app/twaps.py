"""
dashboard_app/twaps.py — активні TWAP Hyperliquid (контейнер hltwap, hl_twap.py) для секції «TWAP · HL»: % Vol з обороту Binance (trade_bars),
фільтри twap_config.json (одні для секції, звуку й Telegram), події «новий TWAP пройшов фільтри».
"""

import json
import os
import time

from dashboard_app.common import ROOT, list_tickers


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
