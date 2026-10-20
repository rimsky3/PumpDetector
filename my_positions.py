"""
my_positions.py — позиції власника на Binance USDⓈ-M Futures наживо (лише читання): закріплений конус від входу і
запис виконань у базу (2026-10-20, контейнер myacct).

Власник: «зробити, щоб конус автоматично пінився, коли я заходжу в угоду, і розпінювався, коли виходжу» + «записувати
угоди, щоб виміряти мій edge проти моделі». Ключ — той самий, що в my_trades.py: BINANCE_API_KEY / BINANCE_API_SECRET у
.env, ЛИШЕ ЧИТАННЯ (без торгівлі й виведення; перевірка — python my_trades.py --check). Ключ нікуди не пишеться.

Як працює:
    старт — знімок відкритих позицій (GET /fapi/v2/positionRisk, підписаний) і звірка зі збереженим станом;
    далі — потік подій акаунта (user data stream: POST listenKey → wss://fstream.binance.com/ws/<listenKey>,
    продовження PUT раз на 30 хв, нове з'єднання при обриві / listenKeyExpired / за 23 год — Binance рве через 24 год);
    ACCOUNT_UPDATE — позиція з'явилась (0 → не 0): «пін» = момент і ціна входу + прогноз Volatility на цю мить
        (reports/volfc.json: exp_1, exp_4); добір / частковий вихід — пін не рухається; позиція 0 — пін знято;
        розворот (лонг → шорт) — закриття й нове відкриття;
    ORDER_TRADE_UPDATE з x = TRADE — виконання → таблиця my_fills (без дублів: ключ symbol + trade_id).
Стан → reports/my_positions.json (при кожній зміні і щохвилини — дашборд бачить свіжість). Позиція, відкрита, поки сервіс
не працював, — пін від часу останньої зміни позиції Binance і поточного прогнозу (approx: true).

    python my_positions.py --loop     # сервіс (контейнер myacct)
    python my_positions.py --once     # знімок позицій → reports/my_positions.json, без потоку (перевірка)
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "reports", "my_positions.json")
VOLFC = os.path.join(ROOT, "reports", "volfc.json")
FAPI, WS = "https://fapi.binance.com", "wss://fstream.binance.com/ws/"
KEEPALIVE_SEC = 30 * 60
RECONNECT_SEC = 23 * 3600
HEARTBEAT_SEC = 60
DDL = """CREATE TABLE IF NOT EXISTS my_fills (
    symbol TEXT NOT NULL, trade_id BIGINT NOT NULL,
    ts TIMESTAMPTZ NOT NULL, side TEXT, position_side TEXT, order_type TEXT,
    price DOUBLE PRECISION, qty DOUBLE PRECISION, quote DOUBLE PRECISION,
    realized_pnl DOUBLE PRECISION, commission DOUBLE PRECISION, commission_asset TEXT,
    order_id BIGINT, maker BOOLEAN, reduce_only BOOLEAN,
    recv_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol, trade_id)
)"""


def base_of(symbol: str) -> str:
    for q in ("USDT", "USDC"):
        if symbol.endswith(q):
            return symbol[: -len(q)]
    return symbol


# ---------------------------------------------------------------- стан позицій (без мережі — tests/test_my_positions.py)

def apply_positions(state: dict, rows: list[dict], at: float, forecast, approx: bool = False) -> list[str]:
    """rows — позиції з події ACCOUNT_UPDATE (s, pa, ep, ps) або знімка; state — {ключ: позиція}. Повертає зміни
    («open BTC», «close BTC»). Пін — при відкритті (і при розвороті); добір і частковий вихід його не рухають."""
    changes = []
    for r in rows:
        sym, ps = r["s"], r.get("ps", "BOTH")
        key, qty, ep = f"{sym}|{ps}", float(r["pa"]), float(r.get("ep") or 0)
        old = state.get(key)
        if qty == 0:
            if old:
                del state[key]
                changes.append(f"close {base_of(sym)}")
            continue
        side = "long" if qty > 0 else "short"
        if old and old["side"] == side:
            old.update(qty=abs(qty), entry_px=ep)
            continue
        if old:
            changes.append(f"close {base_of(sym)}")
        e1, e4 = forecast(base_of(sym))
        state[key] = {"symbol": sym, "ps": ps, "base": base_of(sym), "side": side, "qty": abs(qty), "entry_px": ep, "opened": at,
                      "approx": approx, "cone": {"at": at, "p": ep, "e1": e1, "e4": e4}}
        changes.append(f"open {base_of(sym)} {side}")
    return changes


def fill_row(o: dict) -> tuple | None:
    """Виконання з ORDER_TRADE_UPDATE.o → рядок my_fills (лише x = TRADE)."""
    if o.get("x") != "TRADE":
        return None
    px, q = float(o.get("L") or 0), float(o.get("l") or 0)
    return (o["s"], int(o["t"]), datetime.fromtimestamp(int(o["T"]) / 1000, timezone.utc), o.get("S"), o.get("ps"), o.get("o"),
            px, q, px * q, float(o.get("rp") or 0), float(o.get("n") or 0), o.get("N"), int(o.get("i") or 0),
            bool(o.get("m")), bool(o.get("R")))


class Forecast:
    """Прогноз Volatility (exp_1, exp_4) монети з reports/volfc.json; файл перечитується, лише коли змінився."""

    def __init__(self, path: str = VOLFC):
        self.path, self.mt, self.rows = path, None, {}

    def __call__(self, base: str) -> tuple[float | None, float | None]:
        try:
            mt = os.path.getmtime(self.path)
            if mt != self.mt:
                with open(self.path, encoding="utf-8") as f:
                    self.rows = {r["base"]: r for r in json.load(f).get("rows") or []}
                self.mt = mt
        except (OSError, ValueError):
            return None, None
        r = self.rows.get(base) or {}
        return r.get("exp_1"), r.get("exp_4")


def save(state: dict, connected: bool, path: str = OUT) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump({"ts": int(time.time()), "connected": connected, "positions": list(state.values())}, f, ensure_ascii=False)
    os.replace(path + ".tmp", path)


def load_state(path: str = OUT) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return {f"{p['symbol']}|{p.get('ps', 'BOTH')}": p for p in json.load(f).get("positions") or []}
    except (OSError, ValueError, KeyError):
        return {}


# ---------------------------------------------------------------- читання: дашборд

_cache: dict = {}


def load() -> dict:
    """Вміст reports/my_positions.json для /api/positions (позиції, свіжість, чи живий потік)."""
    try:
        with open(OUT, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {"positions": [], "error": "немає даних (myacct не запущено)"}


def info(base: str) -> dict | None:
    """Відкрита позиція власника на тікері (для конуса від входу) або None. Файл перечитується лише після зміни."""
    try:
        mt = os.path.getmtime(OUT)
    except OSError:
        return None
    if _cache.get("mt") != mt:
        try:
            with open(OUT, encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, ValueError):
            return None
        _cache.update(mt=mt, by={p["base"]: p for p in d.get("positions") or []})
    return _cache["by"].get(base)


# ---------------------------------------------------------------- сервіс

def snapshot(b) -> list[dict]:
    """Відкриті позиції з REST у форматі ACCOUNT_UPDATE (s, pa, ep, ps) + час останньої зміни."""
    rows = b.get(FAPI, "/fapi/v2/positionRisk")
    return [{"s": r["symbol"], "pa": r["positionAmt"], "ep": r["entryPrice"], "ps": r.get("positionSide", "BOTH"),
             "t": int(r.get("updateTime") or 0) / 1000} for r in rows if float(r["positionAmt"]) != 0]


def reconcile(state: dict, snap: list[dict], forecast) -> list[str]:
    """Після старту / обриву потоку: закрити зниклі позиції, додати нові (пін від updateTime, approx)."""
    open_keys = {f"{r['s']}|{r['ps']}" for r in snap}
    changes = apply_positions(state, [{"s": k.split("|")[0], "pa": 0, "ps": k.split("|")[1]} for k in list(state) if k not in open_keys],
                              time.time(), forecast)
    for r in snap:
        old = state.get(f"{r['s']}|{r['ps']}")
        if old and old["side"] == ("long" if float(r["pa"]) > 0 else "short"):
            continue
        changes += apply_positions(state, [r], r["t"] or time.time(), forecast, approx=True)
    return changes


async def run(log=print) -> None:
    import psycopg
    import websockets
    from my_trades import client
    b = client()
    db_url = os.environ.get("DATABASE_URL")
    if db_url:
        with psycopg.connect(db_url) as conn:
            conn.execute(DDL)
            conn.commit()
    forecast, state = Forecast(), load_state()
    fills: list[tuple] = []

    def write_fills():
        if not fills or not db_url:
            return
        rows = fills[:]
        try:
            with psycopg.connect(db_url) as conn:
                conn.cursor().executemany("""INSERT INTO my_fills (symbol, trade_id, ts, side, position_side, order_type, price, qty,
                    quote, realized_pnl, commission, commission_asset, order_id, maker, reduce_only)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (symbol, trade_id) DO NOTHING""", rows)
                conn.commit()
            del fills[:len(rows)]
        except Exception as e:                       # база недоступна — спробуємо пізніше
            log(f"! my_fills: {type(e).__name__}: {e}")

    delay = 1
    while True:
        try:
            ch = reconcile(state, await asyncio.to_thread(snapshot, b), forecast)
            save(state, False)
            if ch:
                log(f"[{datetime.now(timezone.utc):%H:%M:%S}] знімок: " + ", ".join(ch))
            lk = (await asyncio.to_thread(lambda: b.s.post(f"{FAPI}/fapi/v1/listenKey", timeout=10))).json()["listenKey"]
            started = last_alive = last_hb = time.time()
            async with websockets.connect(WS + lk, open_timeout=20, ping_interval=20, ping_timeout=20) as ws:
                log(f"[{datetime.now(timezone.utc):%H:%M:%S}] потік акаунта підключено, відкритих позицій: {len(state)}")
                save(state, True)
                delay = 1
                while time.time() - started < RECONNECT_SEC:
                    try:
                        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=HEARTBEAT_SEC))
                    except asyncio.TimeoutError:
                        msg = None
                    now = time.time()
                    if now - last_alive >= KEEPALIVE_SEC:
                        await asyncio.to_thread(lambda: b.s.put(f"{FAPI}/fapi/v1/listenKey", timeout=10))
                        last_alive = now
                    if msg:
                        e = msg.get("e")
                        if e == "listenKeyExpired":
                            break
                        if e == "ACCOUNT_UPDATE":
                            ch = apply_positions(state, (msg.get("a") or {}).get("P") or [], msg.get("E", now * 1000) / 1000, forecast)
                            if ch:
                                save(state, True)
                                log(f"[{datetime.now(timezone.utc):%H:%M:%S}] " + ", ".join(ch))
                        elif e == "ORDER_TRADE_UPDATE":
                            row = fill_row(msg.get("o") or {})
                            if row:
                                fills.append(row)
                                await asyncio.to_thread(write_fills)
                    if now - last_hb >= HEARTBEAT_SEC:
                        save(state, True)
                        await asyncio.to_thread(write_fills)
                        last_hb = now
        except Exception as e:
            log(f"! потік акаунта: {type(e).__name__}: {e}; повтор через {delay} с")
            try:
                save(state, False)
            except OSError:
                pass
            await asyncio.sleep(delay)
            delay = min(delay * 2, 120)


def main() -> None:
    ap = argparse.ArgumentParser(description="Позиції власника Binance Futures наживо (лише читання): пін конуса від входу, запис виконань")
    ap.add_argument("--loop", action="store_true", help="сервіс: потік подій акаунта (контейнер myacct)")
    ap.add_argument("--once", action="store_true", help="лише знімок відкритих позицій → reports/my_positions.json")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if args.loop:
        asyncio.run(run(lambda m: print(m, flush=True)))
    elif args.once:
        from my_trades import client
        state = load_state()
        ch = reconcile(state, snapshot(client()), Forecast())
        save(state, False)
        print(f"відкритих позицій: {len(state)}" + (f"; зміни: {', '.join(ch)}" if ch else ""))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
