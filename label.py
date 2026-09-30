"""
label.py — локальний інструмент розмітки пампів.

Піднімає вебсервер на 127.0.0.1 (доступний лише з цього комп'ютера) і відкриває
сторінку з графіками ціни, open interest і обсягу по кожному тікеру з бази.
Таймфрейми: 1h (Coinalyze, усі біржі) і 15m / 5m (Binance + Bybit, python fetch_exchange.py) — межі пампу
можна ставити на будь-якому, це моменти часу.

Права панель — усі USDT-перпи Binance. «Завантажити» ставить тікер у чергу: 1h з Coinalyze (fetch.collect),
потім 15m/5m з бірж (fetch_exchange.collect).
Розмічені інтервали будь-якого тікера йдуть у навчання (features.py). Решта годин тікера — лише коли він у
tickers.txt: кнопка «У навчання» = «переглянув увесь тікер, де пампу не розмічено — його не було».
Виділені інтервали одразу записуються в pumps.csv і синхронізуються з таблицею pumps.
Якщо є reports/oof_predictions.csv (python train.py), показує сигнали моделей і їхні помилки.

    python label.py
    python label.py --days 30 --port 8765
"""

import argparse
import csv
import json
import os
import queue
import re
import sys
import threading
import time
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import psycopg
import requests
from dotenv import load_dotenv

import fetch_exchange
from exchanges import INTERVAL
from fetch import TICKERS_FILE, Coinalyze, collect, collect_liquidations, read_tickers_file, set_ticker_in_file
from load_pumps import (check_overlaps, parse_row, parse_ts, read_pumps,
                        sync_db, write_pumps, TS_FORMAT)

ROOT = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.join(ROOT, "tools", "label.html")
VENDOR = os.path.join(ROOT, "app", "vendor")                          # бібліотека графіків, спільна з дашбордом
JS_DIR = os.path.join(ROOT, "app", "js")                              # спільні скрипти сторінок
PREDICTIONS = os.path.join(ROOT, "reports", "oof_predictions.csv")   # пише train.py
TICKERS_PATH = os.path.join(ROOT, TICKERS_FILE)
lock = threading.Lock()  # запис у pumps.csv — по одному


# ---------------------------------------------------------------- усі перпи Binance

_universe = {"at": 0.0, "rows": []}


def binance_universe() -> list[dict]:
    """USDT-перпи Binance, що торгуються, зі зміною ціни і обсягом за 24 год (кеш 10 хв; два публічні запити)."""
    if time.time() - _universe["at"] < 600 and _universe["rows"]:
        return _universe["rows"]
    info = requests.get("https://fapi.binance.com/fapi/v1/exchangeInfo", timeout=20).json()
    day = {t["symbol"]: t for t in requests.get("https://fapi.binance.com/fapi/v1/ticker/24hr", timeout=20).json()}
    rows = []
    for s in info["symbols"]:
        if s.get("contractType") != "PERPETUAL" or s.get("quoteAsset") != "USDT" or s.get("status") != "TRADING":
            continue
        t = day.get(s["symbol"], {})
        rows.append({"base": s["baseAsset"], "chg": float(t.get("priceChangePercent") or 0) / 100,
                     "vol": float(t.get("quoteVolume") or 0)})
    _universe.update(at=time.time(), rows=rows)
    return rows


# ---------------------------------------------------------------- черга завантажень

JOBS: dict[str, dict] = {}          # тікер → {state: queued/running/done/error, msg}
_jobs = queue.Queue()


def load_worker(db_url: str, api_key: str, days: int) -> None:
    """По одному тікеру: 1h з Coinalyze, потім 15m/5m з Binance і Bybit. Ліміт Coinalyze спільний зі скринером,
    тож черга, а не паралельно."""
    client = Coinalyze(api_key) if api_key else None
    while True:
        base = _jobs.get()
        job = JOBS[base]
        try:
            if client is None:
                raise ValueError("немає COINALYZE_API_KEY у .env")
            job.update(state="running", msg="1h з Coinalyze…")
            with psycopg.connect(db_url) as conn:
                if not collect(client, conn, [base], days, verbose=False):
                    raise ValueError("не знайдено в Coinalyze (буває інша назва: 1000… чи k…)")
                job.update(msg="15m/5m з Binance і Bybit…")
                n = fetch_exchange.collect(conn, [base], ["15m", "5m"], days, log=lambda m: job.update(msg=m.strip()))
                job.update(msg="ліквідації 15m/5m з Coinalyze…")
                collect_liquidations(client, conn, [base], ["15m", "5m"], days, log=lambda m: job.update(msg=m.strip()))
            job.update(state="done", msg="готово" if n else "готово (лише 1h: на Binance/Bybit немає)")
        except Exception as e:
            job.update(state="error", msg=str(e))


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


def ticker_data(conn, base: str, days: int, tf: str = "1h") -> dict:
    """Ряди тікера на таймфреймі tf: ціна Binance (або перша біржа з OHLCV), OI і обсяг — сума по біржах у USD.
    1h — з Coinalyze (усі біржі); 15m/5m — з Binance і Bybit (fetch_exchange.py)."""
    interval = INTERVAL[tf]
    since = f"now() - interval '{int(days)} days'"
    q = lambda sql: conn.execute(sql, (base, interval)).fetchall()

    price_ex = q("""SELECT m.exchange_name FROM markets m WHERE m.base_asset = %s AND m.has_ohlcv
                    AND EXISTS (SELECT 1 FROM ohlcv o WHERE o.symbol = m.symbol AND o.interval = %s)
                    ORDER BY m.exchange_name <> 'Binance', m.exchange_name LIMIT 1""")
    price_ex = price_ex[0][0] if price_ex else None
    # свічки: open, high, low, close з еталонної біржі
    candles = {r[0]: r[1:] for r in conn.execute(f"""
        SELECT extract(epoch FROM o.ts)::bigint, o.o, o.h, o.l, o.c FROM ohlcv o JOIN markets m USING (symbol)
        WHERE m.base_asset = %s AND m.exchange_name = %s AND o.interval = %s AND o.ts >= {since}""",
        (base, price_ex, interval)).fetchall()}
    price = {t: c[3] for t, c in candles.items()}

    oi_rows = q(f"""
        SELECT extract(epoch FROM o.ts)::bigint, m.exchange_name, o.o, o.h, o.l, o.c, p.c
        FROM open_interest o JOIN markets m USING (symbol)
        LEFT JOIN ohlcv p ON p.symbol = o.symbol AND p.interval = o.interval AND p.ts = o.ts
        WHERE m.base_asset = %s AND o.interval = %s AND o.ts >= {since}""")
    # обсяг OHLCV буває в монетах — переводимо в USD через ціну закриття
    vol = dict(q(f"""
        SELECT extract(epoch FROM o.ts)::bigint,
               sum(CASE WHEN m.volume_denominated = 'BASE_ASSET' THEN o.v * o.c ELSE o.v END)
        FROM ohlcv o JOIN markets m USING (symbol)
        WHERE m.base_asset = %s AND o.interval = %s AND o.ts >= {since} GROUP BY 1"""))

    # свічки OI — сума по біржах: open і close точні, high/low — сума екстремумів бірж
    # (верхня/нижня межа: піки на різних біржах могли бути в різні хвилини години)
    # OI у монетах: Coinalyze множить усю годинну свічку OI на close години, тож ділимо назад
    # на close того самого контракту — зникають стрибки між годинами, які дає рух ціни
    oi, oi_ohl, coin, first = {}, {}, {}, {}
    for t, ex, o, h, l, c, px in oi_rows:
        oi.setdefault(t, {})[ex] = c
        agg = oi_ohl.setdefault(t, [0.0, 0.0, 0.0])
        agg[0] += o if o is not None else c
        agg[1] += h if h is not None else c
        agg[2] += l if l is not None else c
        if px:
            ca = coin.setdefault(t, [0.0, 0.0, 0.0, 0.0])
            for k, v in enumerate((o, h, l, c)):
                ca[k] += (v if v is not None else c) / px
        first[ex] = min(first.get(ex, t), t)
    exs = sorted(first)
    ts = sorted(set(price) | set(oi))
    get = lambda t, ex: oi.get(t, {}).get(ex)
    return {
        "tf": tf,
        "exchanges": exs,
        "priceExchange": price_ex,
        "t": ts,
        "price": [price.get(t) for t in ts],
        "open": [candles[t][0] if t in candles else None for t in ts],
        "high": [candles[t][1] if t in candles else None for t in ts],
        "low": [candles[t][2] if t in candles else None for t in ts],
        "oi": [round(sum(oi[t].values())) if t in oi else None for t in ts],
        "oiO": [round(oi_ohl[t][0]) if t in oi_ohl else None for t in ts],
        "oiH": [round(oi_ohl[t][1]) if t in oi_ohl else None for t in ts],
        "oiL": [round(oi_ohl[t][2]) if t in oi_ohl else None for t in ts],
        "coinO": [coin[t][0] if t in coin else None for t in ts],
        "coinH": [coin[t][1] if t in coin else None for t in ts],
        "coinL": [coin[t][2] if t in coin else None for t in ts],
        "coin": [coin[t][3] if t in coin else None for t in ts],
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
            if url.path.startswith(("/vendor/", "/js/")):
                base_dir = VENDOR if url.path.startswith("/vendor/") else JS_DIR
                path = os.path.normpath(os.path.join(base_dir, os.path.basename(url.path)))
                if not path.startswith(base_dir) or not os.path.isfile(path):
                    return self._error(404, "не знайдено")
                with open(path, "rb") as f:
                    return self._send(200, f.read(), "text/javascript; charset=utf-8")
            if url.path == "/api/tickers":
                with psycopg.connect(self.db_url) as conn:
                    tickers = [r[0] for r in conn.execute(
                        "SELECT DISTINCT base_asset FROM markets ORDER BY 1").fetchall()]
                return self._send(200, {"tickers": tickers, "days": self.days})
            if url.path == "/api/data":
                qs = parse_qs(url.query)
                base = (qs.get("ticker") or [""])[0].upper()
                tf = (qs.get("tf") or ["1h"])[0]
                if tf not in INTERVAL:
                    raise ValueError(f"невідомий таймфрейм {tf}")
                with psycopg.connect(self.db_url) as conn:
                    return self._send(200, ticker_data(conn, base, self.days, tf))
            if url.path == "/api/predictions":
                base = (parse_qs(url.query).get("ticker") or [""])[0].upper()
                return self._send(200, predictions(base))
            if url.path == "/api/pumps":
                with lock:
                    return self._send(200, {"rows": rows_to_json(read_pumps())})
            if url.path == "/api/universe":
                with psycopg.connect(self.db_url) as conn:
                    db = [r[0] for r in conn.execute("SELECT DISTINCT base_asset FROM markets ORDER BY 1")]
                try:
                    binance, err = binance_universe(), None
                except Exception as e:           # без інтернету — лише тікери з бази
                    binance, err = [], str(e)
                return self._send(200, {"binance": binance, "db": db, "error": err,
                                        "training": [t.upper() for t in read_tickers_file(TICKERS_PATH)]})
            if url.path == "/api/jobs":
                return self._send(200, JOBS)
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
        path = urlparse(self.path).path
        try:
            body = self._body()
            if path == "/api/pumps":
                new = parse_row(body)
                rows = self._modify(lambda rows: rows + [new])
                return self._send(200, {"rows": rows_to_json(rows)})
            if path == "/api/load":
                base = str(body.get("ticker", "")).upper()
                if not re.fullmatch(r"[A-Z0-9]{1,20}", base):
                    raise ValueError("невірна назва тікера")
                if JOBS.get(base, {}).get("state") not in ("queued", "running"):
                    JOBS[base] = {"state": "queued", "msg": "у черзі"}
                    _jobs.put(base)
                return self._send(200, JOBS[base])
            if path == "/api/training":
                base = str(body.get("ticker", "")).upper()
                if not re.fullmatch(r"[A-Z0-9]{1,20}", base):
                    raise ValueError("невірна назва тікера")
                set_ticker_in_file(TICKERS_PATH, base, bool(body.get("on")), "переглянуто повністю в label.py")
                return self._send(200, {"training": [t.upper() for t in read_tickers_file(TICKERS_PATH)]})
            return self._error(404, "не знайдено")
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
    parser.add_argument("--port", type=int, default=8765, help="порт на 127.0.0.1")
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
    threading.Thread(target=load_worker, args=(db_url, os.getenv("COINALYZE_API_KEY"), args.days), daemon=True).start()
    # На Windows SO_REUSEADDR дозволяє двом процесам зайняти один порт: другий «стартує», але запитів не отримує.
    # Вимикаємо його, щоб повторний запуск одразу казав, що порт зайнятий.
    ThreadingHTTPServer.allow_reuse_address = False
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError:
        sys.exit(f"Порт {args.port} зайнятий: мабуть, уже запущено інший екземпляр. "
                 f"Зупини його (Ctrl+C у тому терміналі) або запусти з --port 8766.")
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
