"""
trade_recorder.py — угоди Binance і Bybit через WebSocket → 5-хвилинні зведення в таблиці trade_bars (2026-10-07).

Навіщо: зі свічок видно лише обсяг і (на 1h Coinalyze) кількість угод — experiments/trade_count.py показав, що це той самий
обсяг іншими словами. Склад угод — скільки великих, хто агресор, найбільша угода — є лише в потоці угод, і біржі не
віддають його в історії (лише кілька днів). Тому пишемо самі, з моменту запуску; експеримент — через кілька тижнів.

Що зберігаємо (по біржі, символу, 5-хв бару — за часом угоди на біржі):
    n_buy / n_sell, v_buy / v_sell ($)  — угоди, де агресор (taker) купував / продавав
    max_buy / max_sell ($)              — найбільша угода бару
    buy_n[], sell_n[], buy_usd[], sell_usd[] — розподіл за розміром угоди в $, кошики SIZE_BOUNDS:
        [0] < 500, [1] 500–2.5k, [2] 2.5k–10k, [3] 10k–50k, [4] 50k–250k, [5] ≥ 250k
Сирі угоди не зберігаємо: увесь ринок — ~1000 угод/с у спокійний час, у піки в рази більше (десятки ГБ на місяць).

Потоки (усі USDT-перпи, нові лістинги — раз на 6 год):
    Binance  wss://fstream.binance.com/market/ws, <symbol>@aggTrade — угоди одного taker-ордера за однією ціною злиті
             в одну («розмір ордера»); m = true → покупець був maker, тобто агресор продавав
    Bybit    wss://stream.bybit.com/v5/public/linear, publicTrade.<SYMBOL> — кожне виконання окремо (S — сторона taker);
             тому кількості угод Binance і Bybit напряму не порівнюються — порівнювати кожну біржу з її ж історією
Бар пишеться один раз, коли закрився (+ GRACE_SEC на запізнілі повідомлення), і лише якщо з'єднання було живе весь бар:
неповний бар (підключились посередині) не пишемо. «Немає рядка» ≠ «0 угод» — дивись trade_ws_coverage.

    python trade_recorder.py            # працювати безперервно (на сервері — контейнер traderec)
    python trade_recorder.py --stats    # що записано за останні години, покриття з'єднань
"""

import argparse
import asyncio
import json
import os
import sys
import time
from bisect import bisect_right
from datetime import datetime, timedelta, timezone

import psycopg
import requests
from dotenv import load_dotenv

from liq_recorder import Recorder, now_utc

ROOT = os.path.dirname(os.path.abspath(__file__))
BINANCE_WS = "wss://fstream.binance.com/market/ws"       # стара адреса /ws/ мовчить
BYBIT_WS = "wss://stream.bybit.com/v5/public/linear"
BAR_SEC = 300
GRACE_SEC = 5                 # скільки чекати запізнілих угод після закриття бару
SIZE_BOUNDS = [500, 2_500, 10_000, 50_000, 250_000]     # межі кошиків розміру угоди, $
NB = len(SIZE_BOUNDS) + 1
BINANCE_PER_CONN = 300        # потоків на з'єднання Binance (ліміт біржі 1024)
BINANCE_PARAMS_PER_MSG = 50   # потоків в одному SUBSCRIBE (біржа приймає ≤ 10 повідомлень/с)
BYBIT_PER_CONN = 200
BYBIT_ARGS_PER_MSG = 10
FLUSH_SEC = 5
HEARTBEAT_SEC = 60
REFRESH_SYMBOLS_H = 6
STATS_EVERY_SEC = 600

DDL = """
CREATE TABLE IF NOT EXISTS trade_bars (
    exchange  TEXT NOT NULL,              -- Binance / Bybit (як markets.exchange_name)
    symbol    TEXT NOT NULL,              -- символ на біржі (markets.symbol_on_exchange): BTCUSDT
    ts        TIMESTAMPTZ NOT NULL,       -- відкриття 5-хв бару
    n_buy     INTEGER NOT NULL,           -- угод, де агресор купував
    n_sell    INTEGER NOT NULL,
    v_buy     DOUBLE PRECISION NOT NULL,  -- $ агресивних купівель
    v_sell    DOUBLE PRECISION NOT NULL,
    max_buy   DOUBLE PRECISION,           -- найбільша угода бару, $
    max_sell  DOUBLE PRECISION,
    buy_n     INTEGER[] NOT NULL,         -- угод за кошиками розміру (межі — trade_recorder.SIZE_BOUNDS)
    sell_n    INTEGER[] NOT NULL,
    buy_usd   DOUBLE PRECISION[] NOT NULL,
    sell_usd  DOUBLE PRECISION[] NOT NULL,
    PRIMARY KEY (exchange, symbol, ts)
);
CREATE INDEX IF NOT EXISTS trade_bars_ts ON trade_bars (ts);
CREATE TABLE IF NOT EXISTS trade_ws_coverage (
    exchange  TEXT NOT NULL,
    conn      TEXT NOT NULL,              -- binance#0, bybit#2 …
    started   TIMESTAMPTZ NOT NULL,       -- підключились і підписались
    last_seen TIMESTAMPTZ NOT NULL,       -- з'єднання ще живе станом на цей момент
    symbols   INTEGER,
    PRIMARY KEY (exchange, conn, started)
);
"""


def bar_of(t_ms: int) -> int:
    """Початок 5-хв бару (секунди епохи) для часу угоди в мс."""
    return int(t_ms) // 1000 // BAR_SEC * BAR_SEC


class TradeRecorder(Recorder):
    """З'єднання, перепідключення і список символів Bybit — з liq_recorder.Recorder; тут — накопичення барів і запис."""

    def __init__(self, db_url: str):
        super().__init__(db_url)
        self.bars: dict[tuple, list] = {}           # (exchange, symbol, bar) → [n_b, n_s, v_b, v_s, max_b, max_s, bn, sn, bu, su]
        self.sym_conn: dict[tuple, str] = {}        # (exchange, symbol) → з'єднання, яке його веде
        self.trades: dict[str, int] = {}
        self.written = 0

    # ---------- накопичення ----------
    def trade(self, exchange: str, symbol: str, t_ms: int, buy: bool, usd: float):
        key = (exchange, symbol, bar_of(t_ms))
        b = self.bars.get(key)
        if b is None:
            b = self.bars[key] = [0, 0, 0.0, 0.0, 0.0, 0.0, [0] * NB, [0] * NB, [0.0] * NB, [0.0] * NB]
        k = bisect_right(SIZE_BOUNDS, usd)
        if buy:
            b[0] += 1; b[2] += usd; b[4] = max(b[4], usd); b[6][k] += 1; b[8][k] += usd
        else:
            b[1] += 1; b[3] += usd; b[5] = max(b[5], usd); b[7][k] += 1; b[9][k] += usd
        self.trades[exchange] = self.trades.get(exchange, 0) + 1

    def closed_rows(self) -> list[tuple]:
        """Закриті бари → рядки; бар без повного покриття з'єднанням (підключились посередині) відкидаємо."""
        cut = time.time() - GRACE_SEC
        rows = []
        for key in [k for k in self.bars if k[2] + BAR_SEC <= cut]:
            b = self.bars.pop(key)
            exchange, symbol, bar = key
            live = self.live.get((exchange, self.sym_conn.get((exchange, symbol), "")))
            if live is None or live["started"].timestamp() > bar:
                continue
            rows.append((exchange, symbol, datetime.fromtimestamp(bar, tz=timezone.utc), b[0], b[1], b[2], b[3],
                         b[4] or None, b[5] or None, b[6], b[7], b[8], b[9]))
        return rows

    # ---------- запис ----------
    def _write(self, rows: list[tuple], live: list[tuple]):
        with psycopg.connect(self.db_url) as conn:
            if rows:
                conn.cursor().executemany("""
                    INSERT INTO trade_bars (exchange, symbol, ts, n_buy, n_sell, v_buy, v_sell, max_buy, max_sell,
                                            buy_n, sell_n, buy_usd, sell_usd)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""", rows)
            for exchange, cn, started, symbols in live:
                conn.execute("""INSERT INTO trade_ws_coverage (exchange, conn, started, last_seen, symbols) VALUES (%s,%s,%s,now(),%s)
                                ON CONFLICT (exchange, conn, started) DO UPDATE SET last_seen = now(), symbols = EXCLUDED.symbols""",
                             (exchange, cn, started, symbols))
            conn.commit()

    async def flusher(self):
        last_hb, last_stats, pending = 0.0, time.time(), []
        while True:
            await asyncio.sleep(FLUSH_SEC)
            rows = pending + self.closed_rows()
            hb = time.time() - last_hb >= HEARTBEAT_SEC
            live = [(k[0], k[1], v["started"], v["symbols"]) for k, v in self.live.items()] if hb else []
            pending = []
            if rows or live:
                try:
                    await asyncio.get_running_loop().run_in_executor(None, self._write, rows, live)
                    self.written += len(rows)
                    if hb:
                        last_hb = time.time()
                except Exception as e:           # база недоступна — спробуємо з наступним записом
                    print(f"! запис у базу: {e}", flush=True)
                    pending = rows
            if time.time() - last_stats >= STATS_EVERY_SEC:
                sec = time.time() - last_stats
                parts = [f"{x} {self.trades.get(x, 0) / sec:.0f} угод/с" for x in ("Binance", "Bybit")]
                print(f"[{now_utc():%H:%M} UTC] " + "; ".join(parts) + f"; записано барів {self.written}; "
                      f"з'єднань {len(self.live)}", flush=True)
                self.trades, self.written, last_stats = {}, 0, time.time()

    # ---------- підписки ----------
    async def subscribe_all(self, exchange: str, fetch, per_conn: int, make_sub, url: str, on_msg, ping: str | None):
        """Підписки частинами по per_conn символів; при нових лістингах — перепідписка всіх з'єднань біржі."""
        syms, stop, tasks = None, None, []
        while True:
            try:
                fresh = fetch()
            except Exception as e:
                print(f"! {exchange}: список символів ({e})", flush=True)
                fresh = syms
            if fresh and fresh != syms:
                if stop is not None:
                    stop.set()
                    await asyncio.gather(*tasks, return_exceptions=True)
                syms, stop, tasks = fresh, asyncio.Event(), []
                for i in range(0, len(syms), per_conn):
                    chunk, cn = syms[i:i + per_conn], f"{exchange.lower()}#{i // per_conn}"
                    for s in chunk:
                        self.sym_conn[(exchange, s)] = cn
                    tasks.append(asyncio.create_task(self.run_conn(
                        exchange, cn, url, make_sub(chunk), on_msg, len(chunk), ping, stop)))
            await asyncio.sleep(REFRESH_SYMBOLS_H * 3600 if syms else 60)

    # ---------- Binance ----------
    @staticmethod
    def binance_symbols() -> list[str]:
        r = requests.get("https://fapi.binance.com/fapi/v1/exchangeInfo", timeout=20).json()
        return sorted(s["symbol"] for s in r["symbols"]
                      if s.get("quoteAsset") == "USDT" and s.get("contractType") == "PERPETUAL" and s.get("status") == "TRADING")

    def on_binance(self, m: dict):
        if m.get("e") != "aggTrade":
            return                                # відповіді на SUBSCRIBE ({"result": null, "id": …})
        self.trade("Binance", m["s"], m["T"], not m["m"], float(m["p"]) * float(m["q"]))

    async def binance(self):
        def make_sub(chunk):
            async def sub(ws):
                for j in range(0, len(chunk), BINANCE_PARAMS_PER_MSG):
                    await ws.send(json.dumps({"method": "SUBSCRIBE", "id": j + 1,
                                              "params": [f"{s.lower()}@aggTrade" for s in chunk[j:j + BINANCE_PARAMS_PER_MSG]]}))
                    await asyncio.sleep(0.25)
            return sub
        await self.subscribe_all("Binance", self.binance_symbols, BINANCE_PER_CONN, make_sub, BINANCE_WS, self.on_binance, None)

    # ---------- Bybit ----------
    def on_bybit(self, m: dict):
        if not str(m.get("topic", "")).startswith("publicTrade."):
            return
        for d in m.get("data", []):
            self.trade("Bybit", d["s"], d["T"], d["S"] == "Buy", float(d["p"]) * float(d["v"]))

    async def bybit(self):
        def make_sub(chunk):
            async def sub(ws):
                for j in range(0, len(chunk), BYBIT_ARGS_PER_MSG):
                    await ws.send(json.dumps({"op": "subscribe",
                                              "args": [f"publicTrade.{s}" for s in chunk[j:j + BYBIT_ARGS_PER_MSG]]}))
            return sub
        await self.subscribe_all("Bybit", self.bybit_symbols, BYBIT_PER_CONN, make_sub, BYBIT_WS, self.on_bybit, '{"op":"ping"}')


def ensure_tables(db_url: str):
    with psycopg.connect(db_url) as conn:
        conn.execute(DDL)
        conn.commit()


async def run(db_url: str):
    rec = TradeRecorder(db_url)
    await asyncio.gather(rec.flusher(), rec.binance(), rec.bybit())


def stats(db_url: str, hours: float):
    since = now_utc() - timedelta(hours=hours)
    with psycopg.connect(db_url) as conn:
        print(f"Угоди з {since:%m-%d %H:%M} UTC ({hours:g} год), 5-хв бари:")
        for ex, bars, syms, n, v, big in conn.execute("""
                SELECT exchange, count(*), count(DISTINCT symbol), sum(n_buy + n_sell), sum(v_buy + v_sell),
                       sum(buy_n[6] + sell_n[6])           -- масиви в PostgreSQL нумеруються з 1: [6] — угоди ≥ $250k
                FROM trade_bars WHERE ts >= %s GROUP BY 1 ORDER BY 1""", (since,)):
            print(f"  {ex:8s} {bars:7d} барів  {syms:4d} символів  {n:11,d} угод  ${v:>16,.0f}  угод ≥ $250k: {big:,d}")
        print("З'єднання (останні):")
        for cn, st, ls, sy in conn.execute("""
                SELECT conn, started, last_seen, symbols FROM trade_ws_coverage
                WHERE last_seen >= %s ORDER BY exchange, conn, started""", (since,)):
            print(f"  {cn:10s} {st:%m-%d %H:%M} … {ls:%m-%d %H:%M}  ({sy} символів)")


def main():
    ap = argparse.ArgumentParser(description="Угоди Binance / Bybit через WebSocket → 5-хв зведення в таблиці trade_bars")
    ap.add_argument("--stats", action="store_true", help="не записувати, а показати, що записано")
    ap.add_argument("--hours", type=float, default=6, help="за скільки годин показувати --stats")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(os.path.join(ROOT, ".env"))
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        sys.exit("Заповни DATABASE_URL у .env")
    ensure_tables(db_url)
    if args.stats:
        stats(db_url, args.hours)
        return
    print("Записувач угод: Binance (aggTrade), Bybit (publicTrade), 5-хв бари", flush=True)
    try:
        asyncio.run(run(db_url))
    except KeyboardInterrupt:
        print("Зупинено.")


if __name__ == "__main__":
    main()
