"""
dashboard.py — дашборд Pump Detector: екрани «Прогноз» і «Скринер» (макет Pump Detector App).

Локальний сервер на 127.0.0.1; сторінка — app/dashboard.html. Дані — таблиця signals
(пише screener.py), сирі ряди з бази, збережені моделі (models/*.joblib) і розрахунок ризику (risk.py).

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

from features import aggregate, compute_features, load_raw, query_df
from fetch import Coinalyze, collect, read_tickers_file
from risk import EXIT_BARS, TRAIL_NATR, position_state
from screener import FETCH_DAYS, RUN_MINUTE, WATCHLIST_FILE, load_models, score_bases

ROOT = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.join(ROOT, "app", "dashboard.html")
MODEL_NAMES = {"lightgbm": "LightGBM", "logreg": "Логістична регресія"}
CANDIDATE_SHARE = 0.5     # «Кандидат»: ймовірність ≥ половини порогу сигналу
SPARK_HOURS = 48
refresh_lock = threading.Lock()


# ---------------------------------------------------------------- утиліти

def clean(o):
    """pandas/numpy → JSON: час у секундах epoch, NaN → null."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
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


def next_scan(now: datetime) -> datetime:
    nxt = now.replace(minute=RUN_MINUTE, second=0, microsecond=0)
    return nxt if nxt > now else nxt + timedelta(hours=1)


# ---------------------------------------------------------------- дані для екранів

class State:
    db_url = ""
    api_key = ""
    models: list[dict] = []
    client: Coinalyze | None = None


def meta(conn) -> dict:
    now = datetime.now(timezone.utc)
    sig = query_df(conn, "SELECT max(scored_at) AS last FROM signals")
    exchanges = conn.execute("SELECT count(DISTINCT exchange_name) FROM markets").fetchone()[0]
    watch = [t.upper() for t in read_tickers_file(os.path.join(ROOT, WATCHLIST_FILE))]
    scored = [r[0] for r in conn.execute("SELECT DISTINCT base_asset FROM signals ORDER BY 1").fetchall()]
    return {
        "models": [{
            "name": m["name"], "title": MODEL_NAMES.get(m["name"], m["name"]),
            "trained_at": m["trained_at"], "threshold": m["threshold"],
            "tickers": m["tickers"], "pump_intervals": m["pump_intervals"],
            "validation": m["validation"], "risk": m.get("risk", {}),
        } for m in State.models],
        "tickers": sorted(set(watch) | set(scored)),
        "watchlist": watch,
        "exchanges": exchanges,
        "last_scored_at": sig["last"].iloc[0] if len(sig) else None,
        "next_scan": next_scan(now),
        "now": now,
    }


def scanner(conn, model: str) -> dict:
    sig = latest_signals(conn)
    sig = sig[sig["model"] == model]
    if sig.empty:
        return {"rows": [], "log": []}
    raw = load_raw(conn, sorted(sig["base_asset"].unique()))
    rows = []
    for base, g in sig.groupby("base_asset"):
        g = g.sort_values("ts")
        last = g.iloc[-1]
        streak = 0
        for s in reversed(g["signal"].tolist()):
            if not s:
                break
            streak += 1
        h = aggregate(raw, base)
        spark = h["price"].dropna().iloc[-SPARK_HOURS:] if not h.empty else pd.Series(dtype=float)
        rows.append({
            "ticker": base, "probability": last["probability"], "threshold": last["threshold"],
            "status": status_of(last["probability"], last["threshold"]),
            "streak": streak, "in_sample": bool(last["in_sample"]), "last_bar": last["ts"],
            "spark": spark.tolist(),
            "change_48h": (spark.iloc[-1] / spark.iloc[0] - 1) if len(spark) > 1 else None,
        })
    rows.sort(key=lambda r: -(r["probability"] or 0))

    # лог: початок кожного епізоду сигналу (сусідні години з сигналом = один епізод)
    log = []
    for base, g in sig.sort_values("ts").groupby("base_asset"):
        prev_ts, prev_sig = None, False
        for r in g.itertuples():
            starts = r.signal and not (prev_sig and prev_ts is not None and r.ts - prev_ts == pd.Timedelta(hours=1))
            if starts:
                kind = "live" if r.live else ("in-sample" if r.in_sample else "history")
                log.append({"ticker": base, "ts": r.ts, "probability": r.probability, "kind": kind})
            prev_ts, prev_sig = r.ts, r.signal
    log.sort(key=lambda e: e["ts"], reverse=True)
    return {"rows": rows, "log": log[:30]}


def ticker(conn, base: str, model: str, days: int) -> dict:
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
    return {
        "ticker": base, "model": model, "threshold": thr, "k": k,
        "trail_natr": TRAIL_NATR, "exit_bars": EXIT_BARS,
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
            "price": view["price"].tolist(), "high": view["high"].tolist(), "low": view["low"].tolist(),
            "oi": view["oi"].tolist(),
            "prob": {m: s.reindex(view.index).tolist() for m, s in probs.items()},
        },
        "thresholds": thresholds,
    }


def refresh(conn, base: str) -> dict:
    """Кнопка «Прогнозувати»: свіжі дані тікера з Coinalyze і нова оцінка обома моделями."""
    if not State.api_key:
        raise ValueError("немає COINALYZE_API_KEY у .env")
    started = time.monotonic()
    with refresh_lock:
        if State.client is None:
            State.client = Coinalyze(State.api_key)
        if not collect(State.client, conn, [base], FETCH_DAYS, verbose=False):
            raise ValueError(f"{base} не знайдено на біржах")
        scored, last_closed = score_bases(conn, [base], State.models)
    if scored.empty:
        raise ValueError(f"для {base} ще замало історії: потрібно ≥ 7 днів")
    return {"ok": True, "seconds": round(time.monotonic() - started, 1), "last_closed": last_closed}


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
        model = q.get("model", "lightgbm")
        if url.path == "/":
            with open(PAGE, "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")
        if url.path == "/api/meta":
            return self._route(meta)
        if url.path == "/api/scanner":
            return self._route(lambda c: scanner(c, model))
        if url.path == "/api/ticker":
            days = max(1, min(7, int(q.get("days", 3))))
            return self._route(lambda c: ticker(c, q.get("symbol", "").upper(), model, days))
        return self._send(404, {"error": "не знайдено"})

    def do_POST(self):
        url = urlparse(self.path)
        if url.path == "/api/refresh":
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            return self._route(lambda c: refresh(c, str(body.get("symbol", "")).upper()))
        return self._send(404, {"error": "не знайдено"})


def main() -> None:
    parser = argparse.ArgumentParser(description="Дашборд Pump Detector")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    load_dotenv(os.path.join(ROOT, ".env"))
    State.db_url = os.getenv("DATABASE_URL") or ""
    State.api_key = os.getenv("COINALYZE_API_KEY") or ""
    if not State.db_url:
        sys.exit("Заповни DATABASE_URL у файлі .env")
    State.models = load_models()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
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
