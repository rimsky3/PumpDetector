"""
liq_recorder.py — ліквідації напряму з бірж через WebSocket (Binance, Bybit, OKX) → таблиця liq_events (2026-10-06).

Навіщо: 15m-ліквідації Coinalyze — найповільніша частина циклу скринера (~50 викликів на гарячі тікери при ліміті 40/хв).
Біржі самі надсилають кожну ліквідацію в момент, коли вона сталася: без запитів і ліміту. Спершу 1–2 тижні пишемо
паралельно з Coinalyze і звіряємо (модель навчена на Coinalyze — входи мають збігатися), потім 15m-оцінка бере
ліквідації звідси («швидкий контур», сигнал за ~30–60 с після закриття бару).

Потоки (увесь ринок USDT-перпів, не лише список — нова монета в списку вже матиме історію):
    Binance  wss://fstream.binance.com/market/ws/!forceOrder@arr   (стара адреса /ws/ мовчить)
             ⚠️ біржа надсилає не більше 1 ліквідації на секунду по символу — у каскадах недорахунок
    Bybit    wss://stream.bybit.com/v5/public/linear  allLiquidation.<SYMBOL> — підписка на кожен символ,
             усі ліквідації; кілька з'єднань по BYBIT_PER_CONN символів
    OKX      wss://ws.okx.com:8443/ws/v5/public  liquidation-orders, instType SWAP — розмір у контрактах →
             монети через ctVal (public/instruments); теж до 1 ліквідації на секунду по контракту
Сторона: side = 'long' — примусово закрита ДОВГА позиція (примусовий продаж), 'short' — коротка (примусова купівля).
usd = кількість у монетах × ціна виконання (Binance — середня ap, Bybit/OKX — ціна банкрутства).

«Немає даних» ≠ «0»: таблиця liq_ws_coverage — періоди, коли з'єднання було живе (started … last_seen).
Звірка з Coinalyze і перемикання моделі — окремим кроком (див. docs/liquidations.md).

    python liq_recorder.py            # працювати безперервно (на сервері — контейнер liqrec)
    python liq_recorder.py --stats    # що записано: події й суми за останні години, покриття з'єднань
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import psycopg
import requests
import websockets
from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.abspath(__file__))
BINANCE_WS = "wss://fstream.binance.com/market/ws/!forceOrder@arr"
BYBIT_WS = "wss://stream.bybit.com/v5/public/linear"
OKX_WS = "wss://ws.okx.com:8443/ws/v5/public"
BYBIT_PER_CONN = 200          # символів на одне з'єднання Bybit
BYBIT_ARGS_PER_MSG = 10       # аргументів в одному повідомленні subscribe
FLUSH_SEC = 5                 # як часто писати накопичене в базу
HEARTBEAT_SEC = 60            # як часто оновлювати last_seen живих з'єднань
REFRESH_SYMBOLS_H = 6         # як часто перечитувати списки символів (нові лістинги)
STATS_EVERY_SEC = 600         # підсумок у лог

DDL = """
CREATE TABLE IF NOT EXISTS liq_events (
    exchange  TEXT NOT NULL,              -- Binance / Bybit / OKX (як markets.exchange_name)
    symbol    TEXT NOT NULL,              -- символ на біржі (markets.symbol_on_exchange): BTCUSDT, BTC-USDT-SWAP
    ts        TIMESTAMPTZ NOT NULL,       -- час ліквідації (біржовий, мс)
    side      TEXT NOT NULL,              -- long / short — чию позицію закрито
    qty       DOUBLE PRECISION,           -- у монетах
    price     DOUBLE PRECISION,
    usd       DOUBLE PRECISION,
    recv_at   TIMESTAMPTZ NOT NULL DEFAULT now()   -- коли повідомлення прийшло (затримка потоку = recv_at − ts)
);
CREATE INDEX IF NOT EXISTS liq_events_sym_ts ON liq_events (exchange, symbol, ts);
CREATE INDEX IF NOT EXISTS liq_events_ts ON liq_events (ts);
CREATE TABLE IF NOT EXISTS liq_ws_coverage (
    exchange  TEXT NOT NULL,
    conn      TEXT NOT NULL,              -- з'єднання (у Bybit їх кілька: bybit#0, bybit#1 …)
    started   TIMESTAMPTZ NOT NULL,       -- підключились і підписались
    last_seen TIMESTAMPTZ NOT NULL,       -- з'єднання ще живе станом на цей момент
    symbols   INTEGER,                    -- на скільки символів підписані (Binance/OKX — увесь ринок, NULL)
    PRIMARY KEY (exchange, conn, started)
);
"""


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def ms(t) -> datetime:
    return datetime.fromtimestamp(int(t) / 1000, tz=timezone.utc)


class Recorder:
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.buf: list[tuple] = []
        self.live: dict[tuple, dict] = {}        # (exchange, conn) → {started, symbols}
        self.count: dict[str, int] = {}
        self.usd: dict[str, float] = {}
        self.okx_ct: dict[str, tuple[float, bool]] = {}     # instId → (ctVal, інверсний: ctVal у USD, а не в монетах)
        self.binance_cm: dict[str, float] = {}              # COIN-M Binance: символ → розмір контракту, USD

    # ---------- запис ----------
    def add(self, exchange: str, symbol: str, ts: datetime, side: str, qty: float, price: float):
        self.buf.append((exchange, symbol, ts, side, qty, price, qty * price, now_utc()))
        self.count[exchange] = self.count.get(exchange, 0) + 1
        self.usd[exchange] = self.usd.get(exchange, 0.0) + qty * price

    def _write(self, rows: list[tuple], live: list[tuple]):
        with psycopg.connect(self.db_url) as conn:
            if rows:
                conn.cursor().executemany(
                    "INSERT INTO liq_events (exchange, symbol, ts, side, qty, price, usd, recv_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)", rows)
            for exchange, cn, started, symbols in live:
                conn.execute("""INSERT INTO liq_ws_coverage (exchange, conn, started, last_seen, symbols) VALUES (%s,%s,%s,now(),%s)
                                ON CONFLICT (exchange, conn, started) DO UPDATE SET last_seen = now(), symbols = EXCLUDED.symbols""",
                             (exchange, cn, started, symbols))
            conn.commit()

    async def flusher(self):
        last_hb, last_stats = 0.0, time.time()
        while True:
            await asyncio.sleep(FLUSH_SEC)
            rows, self.buf = self.buf, []
            hb = time.time() - last_hb >= HEARTBEAT_SEC
            live = [(k[0], k[1], v["started"], v["symbols"]) for k, v in self.live.items()] if hb else []
            if rows or live:
                try:
                    await asyncio.get_running_loop().run_in_executor(None, self._write, rows, live)
                    if hb:
                        last_hb = time.time()
                except Exception as e:           # база недоступна — повертаємо рядки в буфер, спробуємо пізніше
                    print(f"! запис у базу: {e}", flush=True)
                    self.buf = rows + self.buf
            if time.time() - last_stats >= STATS_EVERY_SEC:
                parts = [f"{x} {self.count.get(x, 0)} / ${self.usd.get(x, 0):,.0f}" for x in ("Binance", "Bybit", "OKX")]
                conns = ", ".join(sorted(f"{k[1]}" for k in self.live))
                print(f"[{now_utc():%H:%M} UTC] за {STATS_EVERY_SEC // 60} хв: " + "; ".join(parts) + f" · з'єднання: {conns}", flush=True)
                self.count, self.usd, last_stats = {}, {}, time.time()

    # ---------- з'єднання ----------
    async def run_conn(self, exchange: str, cn: str, url: str, subscribe, on_msg, symbols: int | None = None,
                       ping: str | None = None, stop: asyncio.Event | None = None):
        """Одне з'єднання з перепідключенням. subscribe(ws) — надіслати підписку; on_msg(dict) — розібрати повідомлення."""
        delay = 1
        while stop is None or not stop.is_set():
            key = (exchange, cn)
            try:
                async with websockets.connect(url, open_timeout=20, ping_interval=20, ping_timeout=20,
                                              max_size=2 ** 22) as ws:
                    await subscribe(ws)
                    self.live[key] = {"started": now_utc(), "symbols": symbols}
                    print(f"[{now_utc():%H:%M:%S}] {cn}: підключено" + (f" ({symbols} символів)" if symbols else ""), flush=True)
                    delay = 1
                    last_ping = time.time()
                    while stop is None or not stop.is_set():
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=15)
                        except asyncio.TimeoutError:
                            raw = None
                        if ping and time.time() - last_ping >= 20:      # Bybit/OKX чекають власний ping (інакше рвуть)
                            await ws.send(ping)
                            last_ping = time.time()
                        if raw is None or raw == "pong":
                            continue
                        try:
                            on_msg(json.loads(raw))
                        except (ValueError, KeyError, TypeError) as e:
                            print(f"! {cn}: не розібрав повідомлення ({e}): {str(raw)[:200]}", flush=True)
            except Exception as e:
                print(f"[{now_utc():%H:%M:%S}] {cn}: обрив ({type(e).__name__}: {e}) — перепідключення через {delay} с", flush=True)
            finally:
                self.live.pop(key, None)
            if stop is not None and stop.is_set():
                break
            await asyncio.sleep(delay)
            delay = min(60, delay * 2)

    # ---------- Binance ----------
    @staticmethod
    def binance_coinm_sizes() -> dict[str, float]:
        """COIN-M контракти Binance (BTCUSD_PERP, квартальні): розмір контракту в USD (BTC — 100, решта — 10)."""
        r = requests.get("https://dapi.binance.com/dapi/v1/exchangeInfo", timeout=20).json()
        return {s["symbol"]: float(s["contractSize"]) for s in r.get("symbols", []) if s.get("contractSize")}

    def on_binance(self, m: dict):
        o = m.get("o")
        if not o:
            return
        qty = float(o.get("z") or 0) or float(o["q"])
        price = float(o.get("ap") or 0) or float(o["p"])
        sym = o["s"]
        if not sym.endswith(("USDT", "USDC")):     # COIN-M (інверсний): кількість — контракти по contractSize USD, не монети
            size = self.binance_cm.get(sym) or (100.0 if sym.startswith("BTC") else 10.0)
            qty = qty * size / price               # → монети; usd = qty × price = контракти × розмір (07.10: було × ціна зайвий раз)
        self.add("Binance", sym, ms(o["T"]), "long" if o["S"] == "SELL" else "short", qty, price)

    async def binance(self):
        try:
            self.binance_cm = self.binance_coinm_sizes()
        except Exception as e:
            print(f"! Binance COIN-M: довідник контрактів ({e}) — розмір за замовчуванням (BTC 100, решта 10 USD)", flush=True)

        async def sub(ws):
            pass                                  # потік уже в адресі
        await self.run_conn("Binance", "binance", BINANCE_WS, sub, self.on_binance)

    # ---------- Bybit ----------
    def on_bybit(self, m: dict):
        if not str(m.get("topic", "")).startswith("allLiquidation."):
            return
        for d in m.get("data", []):
            # S — сторона ПОЗИЦІЇ: Buy = ліквідовано довгу позицію
            self.add("Bybit", d["s"], ms(d["T"]), "long" if d["S"] == "Buy" else "short", float(d["v"]), float(d["p"]))

    @staticmethod
    def bybit_symbols() -> list[str]:
        r = requests.get("https://api.bybit.com/v5/market/instruments-info",
                         params={"category": "linear", "limit": 1000}, timeout=20).json()
        return sorted(i["symbol"] for i in r["result"]["list"]
                      if i.get("status") == "Trading" and i.get("quoteCoin") == "USDT" and i.get("contractType") == "LinearPerpetual")

    async def bybit(self):
        syms, stop, tasks = None, None, []
        while True:
            try:
                fresh = self.bybit_symbols()
            except Exception as e:
                print(f"! Bybit: список символів ({e})", flush=True)
                fresh = syms
            if fresh and fresh != syms:           # перший запуск або нові лістинги — (пере)підписуємось
                if stop is not None:
                    stop.set()
                    await asyncio.gather(*tasks, return_exceptions=True)
                syms, stop, tasks = fresh, asyncio.Event(), []
                for i in range(0, len(syms), BYBIT_PER_CONN):
                    chunk = syms[i:i + BYBIT_PER_CONN]

                    async def sub(ws, chunk=chunk):
                        for j in range(0, len(chunk), BYBIT_ARGS_PER_MSG):
                            await ws.send(json.dumps({"op": "subscribe",
                                                      "args": [f"allLiquidation.{s}" for s in chunk[j:j + BYBIT_ARGS_PER_MSG]]}))
                    tasks.append(asyncio.create_task(self.run_conn(
                        "Bybit", f"bybit#{i // BYBIT_PER_CONN}", BYBIT_WS, sub, self.on_bybit, len(chunk), '{"op":"ping"}', stop)))
            await asyncio.sleep(REFRESH_SYMBOLS_H * 3600 if syms else 60)

    # ---------- OKX ----------
    def okx_ct_val(self):
        r = requests.get("https://www.okx.com/api/v5/public/instruments", params={"instType": "SWAP"}, timeout=20).json()
        # інверсні (BTC-USD-SWAP: ctType inverse, ctVal 100, ctValCcy USD) — розмір контракту в доларах, а не в монетах
        self.okx_ct = {i["instId"]: (float(i["ctVal"]), i.get("ctType") == "inverse") for i in r["data"] if i.get("ctVal")}

    def on_okx(self, m: dict):
        for d in m.get("data", []):
            inst = d["instId"]
            spec = self.okx_ct.get(inst)
            if spec is None:                      # новий контракт — перечитаємо довідник при наступному оновленні
                continue
            ct, inverse = spec
            for x in d.get("details", []):
                # posSide — сторона позиції; у режимі net буває порожня → за side ордера (sell = закрили лонг)
                side = x.get("posSide") if x.get("posSide") in ("long", "short") else ("long" if x.get("side") == "sell" else "short")
                px = float(x["bkPx"])
                # лінійний: контракти × ctVal = монети; інверсний: контракти × ctVal = USD → монети = USD / ціна
                qty = float(x["sz"]) * ct / px if inverse else float(x["sz"]) * ct
                self.add("OKX", inst, ms(x["ts"]), side, qty, px)

    async def okx(self):
        while not self.okx_ct:
            try:
                self.okx_ct_val()
            except Exception as e:
                print(f"! OKX: довідник контрактів ({e}) — повтор через 60 с", flush=True)
                await asyncio.sleep(60)

        async def refresh():
            while True:
                await asyncio.sleep(REFRESH_SYMBOLS_H * 3600)
                try:
                    self.okx_ct_val()
                except Exception as e:
                    print(f"! OKX: довідник контрактів ({e})", flush=True)

        async def sub(ws):
            await ws.send(json.dumps({"op": "subscribe", "args": [{"channel": "liquidation-orders", "instType": "SWAP"}]}))
        await asyncio.gather(refresh(), self.run_conn("OKX", "okx", OKX_WS, sub, self.on_okx, None, "ping"))


def ensure_tables(db_url: str):
    with psycopg.connect(db_url) as conn:
        conn.execute(DDL)
        conn.commit()


async def run(db_url: str):
    rec = Recorder(db_url)
    await asyncio.gather(rec.flusher(), rec.binance(), rec.bybit(), rec.okx())


def stats(db_url: str, hours: float):
    since = now_utc() - timedelta(hours=hours)
    with psycopg.connect(db_url) as conn:
        print(f"Ліквідації з {since:%m-%d %H:%M} UTC ({hours:g} год):")
        for ex, n, usd, syms, lag in conn.execute("""
                SELECT exchange, count(*), sum(usd), count(DISTINCT symbol),
                       percentile_cont(0.5) WITHIN GROUP (ORDER BY extract(epoch FROM recv_at - ts))
                FROM liq_events WHERE ts >= %s GROUP BY 1 ORDER BY 1""", (since,)):
            print(f"  {ex:8s} {n:7d} подій  ${usd:>14,.0f}  {syms:4d} символів  затримка потоку (медіана) {lag:.1f} с")
        print("З'єднання (останні):")
        for ex, cn, st, ls, sy in conn.execute("""
                SELECT exchange, conn, started, last_seen, symbols FROM liq_ws_coverage
                WHERE last_seen >= %s ORDER BY exchange, conn, started""", (since,)):
            print(f"  {cn:10s} {st:%m-%d %H:%M} … {ls:%m-%d %H:%M}" + (f"  ({sy} символів)" if sy else ""))


def main():
    ap = argparse.ArgumentParser(description="Ліквідації з Binance / Bybit / OKX через WebSocket → таблиця liq_events")
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
    print("Записувач ліквідацій: Binance, Bybit, OKX", flush=True)
    try:
        asyncio.run(run(db_url))
    except KeyboardInterrupt:
        print("Зупинено.")


if __name__ == "__main__":
    main()
