"""
hl_twap.py — активні TWAP-ордери Hyperliquid по всьому ринку, лише в пам'яті → внутрішній API для дашборда (2026-10-18;
власник: «мене цікавлять роботи і TWAP», «просто хочу бачити, коли з'являється TWAP» — без запису в базу).

Офіційний API Hyperliquid віддає TWAP лише для конкретної адреси (twapHistory / twapStates), а списку «всі активні TWAP»
немає. Зате в публічному потоці угод (trades) є адреси покупця й продавця, а TWAP виконується порціями кожні 30 с. Тому:
    1. слухаємо угоди всіх перпів основного dex (одне WebSocket-з'єднання, ~100–150 угод/с);
    2. порції адреси на монеті в один бік (угоди в межах SLICE_GAP с — одна порція); ≥ MIN_SLICES порцій, з них
       ≥ MIN_30S проміжків 26–34 с → кандидат (перевірка 17–18.10: 12 з 12 перевірених кандидатів — справжні TWAP;
       ~50 активних TWAP за 2–3 хв, усі відомі з перевірки по адресах знайдені);
    3. підтвердження — запит info twapHistory по адресі (усі її TWAP з twapId і статусом): активні = activated без
       завершального запису; звідти розмір, хвилини, reduceOnly, час старту;
    4. виконання далі рахуємо самі з потоку угод (порції цієї адреси), до виявлення — за рівномірним графіком TWAP
       (sz × частка часу, що минула); кінець — виконано ≥ 99.9%, минув час або порцій немає STALE_MIN хв;
    5. для монет з активним TWAP — стакан Hyperliquid (l2Book): $ у межах ±DEPTH_PCT від ціни з того боку, куди б'є TWAP.
Поруч — оборот кожної монети на Hyperliquid за останню годину ($ за хвилину; у першу годину після запуску — за той час, що є).

Дашборд питає GET http://hltwap:8790/twaps (docker-compose.server.yml, змінна HLTWAP_URL) і сам додає оборот Binance з
trade_bars — так рахується % Vol (частка TWAP у всіх угодах монети). Після перезапуску контейнер за 2–3 хв знову знаходить
усі активні TWAP; історії немає (рішення власника 18.10).

    python hl_twap.py               # працювати безперервно, API на :8790 (на сервері — контейнер hltwap)
    python hl_twap.py --port 8791   # інший порт (локальна перевірка дашборда)
    python hl_twap.py --probe 240   # слухати N секунд і надрукувати знайдені TWAP, без API
"""

import argparse
import asyncio
import json
import sys
import time
from collections import defaultdict, deque

import requests

from liq_recorder import Recorder, now_utc

HL_WS = "wss://api.hyperliquid.xyz/ws"
HL_INFO = "https://api.hyperliquid.xyz/info"
PORT = 8790
SLICE_GAP = 3            # угоди адреси на монеті в один бік ближче за 3 с — одна порція TWAP
MIN_SLICES, MIN_30S = 4, 3
GAP_LO, GAP_HI = 26, 34  # TWAP Hyperliquid — порція кожні 30 с
KEEP_SEC = 600           # скільки тримати порції адреси для пошуку шаблону
CHECK_COOLDOWN = 300     # не питати ту саму адресу частіше (поки вона не з'явилась на новій монеті)
STALE_MIN = 3            # порцій немає стільки хвилин — TWAP зупинено / завершено
DEPTH_PCT = 0.003        # стакан ±0.3% від ціни — «% of depth» порції
HOUSEKEEP_SEC = 10
REFRESH_H = 6
INFO_PAUSE = 1.0         # с між запитами info (ліміт Hyperliquid — вага запитів за хвилину з IP)


def binance_name(coin: str) -> str:
    """Назва Hyperliquid → база Binance: kPEPE → 1000PEPE (k — тисяча монет)."""
    return "1000" + coin[1:] if coin.startswith("k") and coin[1:].isupper() else coin


class TwapWatcher(Recorder):
    def __init__(self):
        super().__init__("")
        self.coins: list[str] = []
        self.binance: set[str] = set()
        self.slices: dict[tuple, deque] = defaultdict(deque)        # (coin, user, side) → deque[(t, px, sz)] порцій
        self.flow: dict[str, deque] = defaultdict(deque)            # coin → deque[(t, ntl, px)] угод за годину
        self.flow_sum: dict[str, float] = defaultdict(float)
        self.last_px: dict[str, float] = {}
        self.flow_since: float | None = None                        # з якого моменту рахуємо оборот (підключення)
        self.pending: set[str] = set()                              # адреси в черзі (без дублів)
        self.checked: dict[str, tuple[float, set]] = {}             # user → (коли питали, монети, які вже перевірено)
        self.queue: asyncio.Queue = asyncio.Queue()
        self.active: dict[int, dict] = {}                           # twap_id → стан
        self.by_key: dict[tuple, list[int]] = defaultdict(list)     # (coin, user, side) → [twap_id]
        self.depth: dict[str, dict] = {}                            # coin → {"B": $ асків до +0.3%, "A": $ бідів до −0.3%}
        self.book_coins: set[str] = set()                           # на чий стакан підписані
        self.ws = None
        self.n_trades = 0

    # ---------- довідники ----------
    def refresh_universe(self):
        meta, _ctx = requests.post(HL_INFO, json={"type": "metaAndAssetCtxs"}, timeout=30).json()
        self.coins = [u["name"] for u in meta["universe"] if not u.get("isDelisted")]
        try:
            info = requests.get("https://fapi.binance.com/fapi/v1/exchangeInfo", timeout=30).json()
            self.binance = {s["symbol"] for s in info["symbols"] if s.get("contractType") == "PERPETUAL" and s.get("status") == "TRADING"}
        except Exception as e:
            print(f"! Binance exchangeInfo: {e}", flush=True)

    def binance_symbol(self, coin: str) -> str | None:
        s = binance_name(coin) + "USDT"
        return s if self.binance and s in self.binance else None

    # ---------- потік ----------
    def on_msg(self, m: dict):
        ch = m.get("channel")
        if ch == "l2Book":
            self.on_book(m["data"])
        if ch != "trades":
            return
        for tr in m["data"]:
            self.n_trades += 1
            coin, t, px, sz = tr["coin"], tr["time"] / 1000, float(tr["px"]), float(tr["sz"])
            self.last_px[coin] = px
            if self.flow_since is None:
                self.flow_since = time.time()
            f = self.flow[coin]
            f.append((t, px * sz, px))
            self.flow_sum[coin] += px * sz
            while f and f[0][0] < t - 3600:
                self.flow_sum[coin] -= f.popleft()[1]
            for user, side in zip(tr["users"], ("B", "A")):
                self.add_fill(coin, user, side, t, px, sz)

    def on_book(self, d: dict):
        bids, asks = d["levels"]
        if not bids or not asks:
            return
        mid = (float(bids[0]["px"]) + float(asks[0]["px"])) / 2
        self.depth[d["coin"]] = {
            "B": sum(float(x["px"]) * float(x["sz"]) for x in asks if float(x["px"]) <= mid * (1 + DEPTH_PCT)),   # купівля їсть аски
            "A": sum(float(x["px"]) * float(x["sz"]) for x in bids if float(x["px"]) >= mid * (1 - DEPTH_PCT))}

    def add_fill(self, coin: str, user: str, side: str, t: float, px: float, sz: float):
        key = (coin, user, side)
        dq = self.slices[key]
        if dq and t - dq[-1][0] <= SLICE_GAP:
            t0, p0, s0 = dq[-1]
            dq[-1] = (t0, (p0 * s0 + px * sz) / (s0 + sz), s0 + sz)       # та сама порція — середня ціна і сума
        else:
            dq.append((t, px, sz))
            while dq and dq[0][0] < t - KEEP_SEC:
                dq.popleft()
            if key not in self.by_key and len(dq) >= MIN_SLICES:
                gaps = [b[0] - a[0] for a, b in zip(dq, list(dq)[1:])]
                if sum(GAP_LO <= g <= GAP_HI for g in gaps[-(MIN_SLICES - 1):]) >= MIN_30S:
                    last, seen = self.checked.get(user, (0.0, set()))
                    if (coin not in seen or time.time() - last > CHECK_COOLDOWN) and user not in self.pending:
                        self.checked[user] = (time.time(), seen | {coin})
                        self.pending.add(user)
                        self.queue.put_nowait(user)
        for tid in self.by_key.get(key, []):                           # уже відомий TWAP — виконання з угод
            a = self.active.get(tid)
            if a and t * 1000 >= a["started_ms"]:
                a["executed_sz"] += sz / len(self.by_key[key])
                a["last_fill"] = t

    def price_at(self, coin: str, t: float) -> float | None:
        """Ціна першої угоди монети не раніше t (з годинного буфера), якщо буфер покриває t."""
        f = self.flow.get(coin)
        if not f or f[0][0] > t:
            return None
        return next((px for tt, _, px in f if tt >= t), None)

    # ---------- стакан ----------
    def _send(self, msg: dict):
        if self.ws is not None:
            asyncio.get_running_loop().create_task(self._safe_send(json.dumps(msg)))

    async def _safe_send(self, raw: str):
        try:
            await self.ws.send(raw)
        except Exception:
            pass                                    # з'єднання перепідключиться і підпишеться заново (sub)

    def sync_books(self):
        want = {a["coin"] for a in self.active.values()}
        for c in want - self.book_coins:
            self._send({"method": "subscribe", "subscription": {"type": "l2Book", "coin": c}})
        for c in self.book_coins - want:
            self._send({"method": "unsubscribe", "subscription": {"type": "l2Book", "coin": c}})
            self.depth.pop(c, None)
        self.book_coins = want

    # ---------- підтвердження ----------
    async def checker(self):
        loop = asyncio.get_running_loop()
        while True:
            user = await self.queue.get()
            # будь-яка помилка однієї адреси — у лог і далі (раніше виняток тихо зупиняв перевірку назавжди)
            try:
                hist = await loop.run_in_executor(None, lambda: requests.post(
                    HL_INFO, json={"type": "twapHistory", "user": user}, timeout=30).json())
                self.adopt(user, hist if isinstance(hist, list) else [])
                self.sync_books()
            except Exception as e:
                print(f"! twapHistory {user[:10]}: {type(e).__name__}: {e}", flush=True)
                await asyncio.sleep(5)
            finally:
                self.pending.discard(user)
            await asyncio.sleep(INFO_PAUSE)

    def adopt(self, user: str, hist: list[dict]):
        """З історії TWAP адреси — активні (activated без завершального запису) на перпах основного dex."""
        # старі записи історії бувають без twapId — такі пропускаємо (ними не відстежити виконання)
        final = {h.get("twapId") for h in hist if (h.get("status") or {}).get("status") != "activated"}
        now = time.time()
        coins = set(self.coins)
        for h in hist:
            st, tid = h.get("state") or {}, h.get("twapId")
            if tid is None or (h.get("status") or {}).get("status") != "activated" or tid in final or tid in self.active:
                continue
            coin = st.get("coin", "")
            if coin not in coins:                                      # спот (@151) чи інший dex — не наш ринок
                continue
            started = st["timestamp"] / 1000
            minutes = int(st.get("minutes") or 0)
            if minutes <= 0 or now > started + minutes * 60 + 120:
                continue
            sz = float(st["sz"])
            key = (coin, user, st["side"])
            dq = self.slices.get(key)
            px0 = self.price_at(coin, started) or (dq[0][1] if dq else self.last_px.get(coin))
            self.active[tid] = {"id": tid, "coin": coin, "binance": self.binance_symbol(coin), "user": user,
                                "side": st["side"], "sz": sz, "minutes": minutes, "open": not st.get("reduceOnly"),
                                "randomize": bool(st.get("randomize")), "started_ms": st["timestamp"], "detected": now,
                                "px_start": px0,
                                # до виявлення — за рівномірним графіком TWAP (порція кожні 30 с), далі — з угод
                                "executed_sz": sz * min(1.0, (now - started) / (minutes * 60)), "last_fill": now}
            self.by_key[key].append(tid)
            print(f"[{now_utc():%H:%M:%S}] TWAP {'BUY ' if st['side'] == 'B' else 'SELL'} {coin:<8} ${sz * (px0 or 0):,.0f} за {minutes} хв"
                  f"{' · close' if st.get('reduceOnly') else ''} · {user[:10]}… (id {tid})", flush=True)

    def expire(self):
        now = time.time()
        for tid, a in list(self.active.items()):
            end_t = a["started_ms"] / 1000 + a["minutes"] * 60
            if a["executed_sz"] >= a["sz"] * 0.999 or now > end_t + 60 or now - a["last_fill"] > STALE_MIN * 60:
                del self.active[tid]
                key = (a["coin"], a["user"], a["side"])
                self.by_key[key] = [x for x in self.by_key[key] if x != tid]
                if not self.by_key[key]:
                    del self.by_key[key]
        for key in [k for k, dq in self.slices.items() if not dq or dq[-1][0] < now - KEEP_SEC]:
            del self.slices[key]                                      # пам'ять: адреси, що давно не торгували
        self.sync_books()

    # ---------- видача ----------
    def ntl_min(self, coin: str) -> float | None:
        """Оборот монети на Hyperliquid, $ за хвилину: за останню годину або за час з підключення, якщо менше години."""
        if self.flow_since is None:
            return None
        span = min(3600.0, max(60.0, time.time() - self.flow_since))
        return self.flow_sum.get(coin, 0.0) / (span / 60)

    def snapshot(self) -> dict:
        out = []
        for a in self.active.values():
            px = self.last_px.get(a["coin"]) or a["px_start"]
            usd = a["sz"] * (a["px_start"] or px or 0)
            out.append({"id": a["id"], "coin": a["coin"], "binance": a["binance"], "side": a["side"], "open": a["open"],
                        "user": a["user"], "randomize": a["randomize"], "minutes": a["minutes"], "started": a["started_ms"] / 1000,
                        "usd": usd, "px_start": a["px_start"], "px": px, "filled": min(1.0, a["executed_sz"] / a["sz"]) if a["sz"] else None,
                        "slice_usd": usd / (a["minutes"] * 2), "depth_usd": (self.depth.get(a["coin"]) or {}).get(a["side"]),
                        "hl_min": self.ntl_min(a["coin"])})
        live = any(k[1] == "hl-trades" for k in self.live)
        return {"connected": live, "since": self.flow_since, "flow_span_min": (min(3600.0, time.time() - self.flow_since) / 60) if self.flow_since else 0,
                "now": time.time(), "twaps": sorted(out, key=lambda x: -x["usd"])}

    async def serve(self, port: int):
        async def handle(reader, writer):
            try:
                line = (await asyncio.wait_for(reader.readline(), 5)).decode("latin-1")
                while (await asyncio.wait_for(reader.readline(), 5)) not in (b"\r\n", b"\n", b""):
                    pass                                           # заголовки не потрібні
                path = line.split(" ")[1] if " " in line else "/"
                code, body = (200, json.dumps(self.snapshot())) if path.startswith("/twaps") else \
                             (200, json.dumps({"ok": True})) if path.startswith("/health") else (404, '{"error": "not found"}')
                raw = body.encode()
                writer.write(f"HTTP/1.1 {code} OK\r\nContent-Type: application/json\r\nContent-Length: {len(raw)}\r\n"
                             f"Connection: close\r\n\r\n".encode() + raw)
                await writer.drain()
            except Exception:
                pass
            finally:
                writer.close()
        server = await asyncio.start_server(handle, "0.0.0.0", port)
        print(f"API: http://0.0.0.0:{port}/twaps", flush=True)
        async with server:
            await server.serve_forever()

    async def housekeeper(self):
        last_stats, last_ref = time.time(), time.time()
        loop = asyncio.get_running_loop()
        while True:
            await asyncio.sleep(HOUSEKEEP_SEC)
            self.expire()
            if time.time() - last_stats >= 600:
                print(f"[{now_utc():%H:%M} UTC] угод за 10 хв: {self.n_trades}; активних TWAP: {len(self.active)}; "
                      f"адрес у пам'яті: {len(self.slices)}; стаканів: {len(self.book_coins)}", flush=True)
                self.n_trades, last_stats = 0, time.time()
            if time.time() - last_ref >= REFRESH_H * 3600:
                try:
                    await loop.run_in_executor(None, self.refresh_universe)
                    last_ref = time.time()
                except Exception as e:
                    print(f"! довідник монет: {e}", flush=True)

    async def stream(self, stop: asyncio.Event | None = None):
        async def sub(ws):
            self.ws = ws
            for c in list(self.coins):
                await ws.send(json.dumps({"method": "subscribe", "subscription": {"type": "trades", "coin": c}}))
            for c in self.book_coins:                       # після перепідключення — стакани активних TWAP знову
                await ws.send(json.dumps({"method": "subscribe", "subscription": {"type": "l2Book", "coin": c}}))
        await self.run_conn("Hyperliquid", "hl-trades", HL_WS, sub, self.on_msg, symbols=len(self.coins),
                            ping=json.dumps({"method": "ping"}), stop=stop)

    async def main(self, seconds: int | None = None, port: int | None = PORT):
        self.refresh_universe()
        print(f"Hyperliquid: перпів {len(self.coins)}, з них на Binance {sum(self.binance_symbol(c) is not None for c in self.coins)}", flush=True)
        stop = asyncio.Event()
        tasks = [asyncio.create_task(self.stream(stop)), asyncio.create_task(self.checker()), asyncio.create_task(self.housekeeper())]
        if port:
            tasks.append(asyncio.create_task(self.serve(port)))
        if seconds:
            await asyncio.sleep(seconds)
            stop.set()
            for t in tasks:
                t.cancel()
            return
        await asyncio.gather(*tasks)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Активні TWAP Hyperliquid з публічного потоку угод — у пам'яті, API для дашборда")
    ap.add_argument("--port", type=int, default=PORT, help=f"порт внутрішнього API (за замовчуванням {PORT})")
    ap.add_argument("--probe", type=int, metavar="SEC", help="слухати SEC секунд і надрукувати знайдені TWAP, без API")
    args = ap.parse_args()
    w = TwapWatcher()
    if not args.probe:
        asyncio.run(w.main(port=args.port))
        return
    asyncio.run(w.main(args.probe, port=None))
    w.expire()
    snap = w.snapshot()
    print(f"\nактивних TWAP наприкінці: {len(snap['twaps'])}; оборот HL зібрано за {snap['flow_span_min']:.0f} хв")
    for a in snap["twaps"][:25]:
        hl = a["hl_min"]
        share = (a["usd"] / a["minutes"]) / hl if hl else float("nan")
        depth = f"{a['slice_usd'] / a['depth_usd']:.0%} of depth" if a["depth_usd"] else "depth —"
        print(f"  {a['coin']:<8} {'BUY ' if a['side'] == 'B' else 'SELL'}{' open ' if a['open'] else ' close'} ${a['usd']:>11,.0f} / {a['minutes']:>4}m · "
              f"filled {a['filled']:.0%} · slice ${a['slice_usd']:,.0f} · {depth} · HL-only vol {share:.0%} · {a['binance'] or '—'}")


if __name__ == "__main__":
    main()
