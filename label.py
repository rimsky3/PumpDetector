"""
label.py — локальний інструмент розмітки пампів.

Піднімає вебсервер на 127.0.0.1 (доступний лише з цього комп'ютера) і відкриває
сторінку з графіками ціни, open interest і обсягу по кожному тікеру з бази.
Виділені інтервали одразу записуються в pumps.csv і синхронізуються з таблицею pumps.
Якщо є reports/oof_predictions.csv (python train.py), показує сигнали моделей і їхні помилки.

    python label.py
    python label.py --days 30 --port 8765
"""

import argparse
import csv
import json
import os
import sys
import threading
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import psycopg
from dotenv import load_dotenv

from load_pumps import (check_overlaps, parse_row, parse_ts, read_pumps,
                        sync_db, write_pumps, TS_FORMAT)

ROOT = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.join(ROOT, "tools", "label.html")
PREDICTIONS = os.path.join(ROOT, "reports", "oof_predictions.csv")   # пише train.py
lock = threading.Lock()  # запис у pumps.csv — по одному


def predictions(base: str) -> dict:
    """Out-of-fold передбачення моделей для тікера (з останнього запуску train.py)."""
    if not os.path.exists(PREDICTIONS):
        return {"available": False}
    with open(PREDICTIONS, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        models = [c[:-len("_score")] for c in reader.fieldnames if c.endswith("_score")]
        rows = [r for r in reader if r["base_asset"] == base]
    num = lambda s: float(s) if s not in ("", None) else None
    out = {
        "available": True,
        "updated": datetime.fromtimestamp(os.path.getmtime(PREDICTIONS), timezone.utc).strftime(TS_FORMAT),
        "t": [int(datetime.fromisoformat(r["ts"]).timestamp()) for r in rows],
        "models": {},
    }
    for m in models:
        thr = next((num(r[f"{m}_threshold"]) for r in rows if r.get(f"{m}_threshold")), None)
        out["models"][m] = {
            "score": [num(r[f"{m}_score"]) for r in rows],
            "signal": [int(r[f"{m}_signal"]) for r in rows],
            "threshold": thr,
        }
    return out


def ticker_data(conn, base: str, days: int) -> dict:
    """Погодинні ряди тікера: ціна Binance (або перша біржа з OHLCV), OI і обсяг — сума по біржах у USD."""
    since = f"now() - interval '{int(days)} days'"
    q = lambda sql: conn.execute(sql, (base,)).fetchall()

    price_ex = q("""SELECT m.exchange_name FROM markets m WHERE m.base_asset = %s AND m.has_ohlcv
                    ORDER BY m.exchange_name <> 'Binance', m.exchange_name LIMIT 1""")
    price_ex = price_ex[0][0] if price_ex else None
    price = dict(conn.execute(f"""
        SELECT extract(epoch FROM o.ts)::bigint, o.c FROM ohlcv o JOIN markets m USING (symbol)
        WHERE m.base_asset = %s AND m.exchange_name = %s AND o.ts >= {since}""", (base, price_ex)).fetchall())

    oi_rows = q(f"""
        SELECT extract(epoch FROM o.ts)::bigint, m.exchange_name, o.c FROM open_interest o JOIN markets m USING (symbol)
        WHERE m.base_asset = %s AND o.ts >= {since}""")
    # обсяг OHLCV буває в монетах — переводимо в USD через ціну закриття
    vol = dict(q(f"""
        SELECT extract(epoch FROM o.ts)::bigint,
               sum(CASE WHEN m.volume_denominated = 'BASE_ASSET' THEN o.v * o.c ELSE o.v END)
        FROM ohlcv o JOIN markets m USING (symbol)
        WHERE m.base_asset = %s AND o.ts >= {since} GROUP BY 1"""))

    oi, first = {}, {}
    for t, ex, v in oi_rows:
        oi.setdefault(t, {})[ex] = v
        first[ex] = min(first.get(ex, t), t)
    exs = sorted(first)
    ts = sorted(set(price) | set(oi))
    get = lambda t, ex: oi.get(t, {}).get(ex)
    return {
        "priceExchange": price_ex,
        "t": ts,
        "price": [price.get(t) for t in ts],
        "oi": [round(sum(oi[t].values())) if t in oi else None for t in ts],
        "oiBy": {ex: [round(get(t, ex)) if get(t, ex) is not None else None for t in ts] for ex in exs},
        "vol": [round(vol[t]) if vol.get(t) is not None else None for t in ts],
        "oiStart": first,
    }


def rows_to_json(rows: list[tuple]) -> list[dict]:
    return [{"base_asset": b, "start_ts": s.strftime(TS_FORMAT), "end_ts": e.strftime(TS_FORMAT),
             "pattern_type": p, "notes": n or ""} for b, s, e, p, n in rows]


class Handler(BaseHTTPRequestHandler):
    db_url: str = ""
    days: int = 60

    def log_message(self, fmt, *args):  # без логу кожного запиту в консоль
        pass

    def _send(self, code: int, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _error(self, code: int, msg: str):
        self._send(code, {"error": msg})

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        url = urlparse(self.path)
        try:
            if url.path == "/":
                with open(PAGE, "rb") as f:
                    return self._send(200, f.read(), "text/html; charset=utf-8")
            if url.path == "/api/tickers":
                with psycopg.connect(self.db_url) as conn:
                    tickers = [r[0] for r in conn.execute(
                        "SELECT DISTINCT base_asset FROM markets ORDER BY 1").fetchall()]
                return self._send(200, {"tickers": tickers, "days": self.days})
            if url.path == "/api/data":
                base = (parse_qs(url.query).get("ticker") or [""])[0].upper()
                with psycopg.connect(self.db_url) as conn:
                    return self._send(200, ticker_data(conn, base, self.days))
            if url.path == "/api/predictions":
                base = (parse_qs(url.query).get("ticker") or [""])[0].upper()
                return self._send(200, predictions(base))
            if url.path == "/api/pumps":
                with lock:
                    return self._send(200, {"rows": rows_to_json(read_pumps())})
            return self._error(404, "не знайдено")
        except ValueError as e:
            return self._error(400, str(e))

    def _modify(self, change):
        """Прочитати pumps.csv, змінити, перевірити, записати і синхронізувати з БД."""
        with lock:
            rows = read_pumps()
            rows = change(rows)
            errors = check_overlaps(rows)
            if errors:
                raise ValueError("; ".join(errors))
            write_pumps(rows)
            with psycopg.connect(self.db_url) as conn:
                sync_db(conn, rows)
            return rows

    def do_POST(self):
        if urlparse(self.path).path != "/api/pumps":
            return self._error(404, "не знайдено")
        try:
            new = parse_row(self._body())
            rows = self._modify(lambda rows: rows + [new])
            return self._send(200, {"rows": rows_to_json(rows)})
        except (ValueError, json.JSONDecodeError) as e:
            return self._error(400, str(e))

    def do_DELETE(self):
        if urlparse(self.path).path != "/api/pumps":
            return self._error(404, "не знайдено")
        try:
            body = self._body()
            base, start = body["base_asset"].upper(), parse_ts(body["start_ts"])
            rows = self._modify(lambda rows: [r for r in rows if not (r[0] == base and r[1] == start)])
            return self._send(200, {"rows": rows_to_json(rows)})
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            return self._error(400, str(e))


def main() -> None:
    parser = argparse.ArgumentParser(description="Локальний інструмент розмітки пампів")
    parser.add_argument("--days", type=int, default=60, help="скільки днів історії показувати")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="не відкривати браузер автоматично")
    args = parser.parse_args()

    load_dotenv()
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        sys.exit("Заповни DATABASE_URL у файлі .env")
    try:
        read_pumps()
    except ValueError as e:
        sys.exit(f"{e}\nВиправ pumps.csv і запусти знову.")

    Handler.db_url, Handler.days = db_url, args.days
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Розмітка пампів: {url}  (Ctrl+C — зупинити)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nЗупинено.")


if __name__ == "__main__":
    main()
