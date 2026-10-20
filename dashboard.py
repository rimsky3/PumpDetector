"""
dashboard.py — дашборд Pump Detector: екрани «Прогноз» і «Скринер» (макет Pump Detector App).

Локальний сервер на 127.0.0.1; сторінка — app/dashboard.html. Дані — таблиця signals
(пише screener.py), сирі ряди з бази, збережені моделі (models/*.joblib) і розрахунок ризику (risk.py).

Основна «модель» — tier15: 15m-LightGBM з підтвердженням годинним LightGBM (tiers.py). Статуси:
«Alert» — 15m-сигнал і годинна оцінка ≥ 0.75 порогу; «Кандидат» — лише 15m-сигнал. Годинні моделі
(lightgbm, logreg) лишаються для порівняння, на годинних барах, зі старими статусами.

    python dashboard.py
    python dashboard.py --port 8766 --no-browser

Логіка екранів — у пакеті dashboard_app/ (19.10.2026): тут лише HTTP-маршрути (Handler) і запуск (main).
"""

import argparse
import json
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pandas as pd
import psycopg
from dotenv import load_dotenv

import footprint
import market
import movers
import my_positions
import orderbook
import scanner
import spread
import tiers as T
import volfc
import volume_profile
import wake
from screener import SIGNALS_STAMP, load_model_15m, load_models

from dashboard_app.common import JS_DIR, PAGE, ROOT, State, VENDOR, clean, list_tickers, meta
from dashboard_app.watchlist_ops import add_to_watchlist, quick_add, remove_from_watchlist, scan_view, undo_remove
from dashboard_app.listview import list_warmer, minicharts, watch_list
from dashboard_app.tickerview import refresh, ticker
from dashboard_app.twaps import save_twap_config, twaps_view
from dashboard_app.backtests import backtest_run, backtest_runs, backtest_start, backtest_status, rescore_on_start
from dashboard_app.journal_view import journal_view, journal_warmer


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
            # page — версія сторінки (найновіший час зміни dashboard.html і app/js/*): змінилась після деплою → вікна
            # (і PD Watchlist, де F5 не працює) перезавантажуються самі
            page = max([mt(PAGE)] + [mt(os.path.join(JS_DIR, f)) for f in os.listdir(JS_DIR) if f.endswith(".js")])
            return self._send(200, {"sig": mt(SIGNALS_STAMP), "wake": mt(wake.EVENTS_FILE), "woke": mt(scanner.WOKE_FILE), "page": page})
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
        if url.path == "/api/positions":       # позиції власника Binance Futures (my_positions.py, контейнер myacct): пін конуса
            return self._send(200, my_positions.load())
        if url.path == "/api/vol":             # очікувана волатильність (кнопка «Volatility»): reports/volfc.json від volfc.py
            return self._send(200, volfc.view(volfc.load(), volfc.load_config(), list_tickers()))
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
                                          "mover": movers.info(sym), "vol": volfc.info(sym)})
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
            if url.path == "/api/vol/config":         # межі обороту таблиці «Volatility ▾» (vol_config.json)
                return self._send(200, {"ok": True, "config": volfc.save_config(body)})
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
