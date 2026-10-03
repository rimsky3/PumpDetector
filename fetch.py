"""
fetch.py — збір годинних (1h) даних з Coinalyze у PostgreSQL.

Для кожного тікера знаходить його перпетуальні ф'ючерси на Binance, Bybit, OKX,
Aster, Hyperliquid і MEXC та тягне п'ять рядів: open interest, OHLCV, funding,
ліквідації, long/short ratio.
Predicted funding не збираємо (з 2026-09-29): ознака не додавала якості, а коштувала 1/6 ліміту.

Приклади:
    python fetch.py                  # усі тікери з tickers.txt
    python fetch.py --new            # лише тікери з tickers.txt, яких ще немає в базі
    python fetch.py AKE
    python fetch.py AKE LSK MYX G --days 30

Налаштування — у файлі .env поруч зі скриптом (див. .env.example).
"""

import argparse
import collections
import os
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

import psycopg
import requests
from dotenv import load_dotenv

BASE_URL = "https://api.coinalyze.net/v1"
INTERVAL = "1hour"
WANTED_EXCHANGES = ["binance", "bybit", "okx", "aster", "hyperliquid", "mexc"]
QUOTE_ASSETS = {"USDT", "USD", "USDC"}
RATE_LIMIT_PER_MIN = 40      # кожен символ у запиті = один виклик
BACKGROUND_HEADROOM = 10     # фонові виклики (годинна частина скринера) лишають стільки з 40/хв терміновим (15m / 5m)
MAX_SYMBOLS_PER_REQUEST = 20
TICKERS_FILE = "tickers.txt"

# (ендпоінт, таблиця, [(колонка в БД, поле в JSON)], додаткові параметри, прапорець з future-markets)
ENDPOINTS = [
    ("open-interest-history", "open_interest",
     [("o", "o"), ("h", "h"), ("l", "l"), ("c", "c")], {"convert_to_usd": "true"}, None),
    ("ohlcv-history", "ohlcv",
     [("o", "o"), ("h", "h"), ("l", "l"), ("c", "c"),
      ("v", "v"), ("bv", "bv"), ("tx", "tx"), ("btx", "btx")], {}, "has_ohlcv_data"),
    ("funding-rate-history", "funding_rate",
     [("o", "o"), ("h", "h"), ("l", "l"), ("c", "c")], {}, None),
    ("liquidation-history", "liquidations",
     [("long_liq", "l"), ("short_liq", "s")], {"convert_to_usd": "true"}, None),
    ("long-short-ratio-history", "long_short_ratio",
     [("ratio", "r"), ("long_pct", "l"), ("short_pct", "s")], {}, "has_long_short_ratio_data"),
]


class Coinalyze:
    """Клієнт з дотриманням ліміту 40 викликів/хв і обробкою 429 (Retry-After).
    Потокобезпечний: скринер тягне годинні дані у фоновому потоці (with client.background(): …). Фонові виклики —
    нижчий пріоритет: беруть лише RATE_LIMIT_PER_MIN − BACKGROUND_HEADROOM і чекають, поки термінові (15m / 5m-цикл)
    стоять у черзі на ліміт, — тож годинне завантаження не затримує ліквідації для свіжого 15m-бару."""

    def __init__(self, api_key: str):
        self.session = requests.Session()
        self.session.headers["api_key"] = api_key
        self._calls = collections.deque()  # моменти спожитих викликів
        self._lock = threading.Lock()
        self._local = threading.local()     # .low — цей потік фоновий
        self._urgent_waiting = 0
        self.calls_made = 0                # лічильники для «Стану системи» в дашборді
        self.rate_limited = 0

    @contextmanager
    def background(self):
        """Виклики з цього потоку — фонові (нижчий пріоритет)."""
        self._local.low = True
        try:
            yield self
        finally:
            self._local.low = False

    def _reserve(self, cost: int) -> None:
        low = getattr(self._local, "low", False)
        if not low:
            with self._lock:
                self._urgent_waiting += 1
        told = False
        try:
            while True:
                with self._lock:
                    now = time.monotonic()
                    while self._calls and now - self._calls[0] >= 60:
                        self._calls.popleft()
                    limit = RATE_LIMIT_PER_MIN - (BACKGROUND_HEADROOM if low else 0)
                    room = len(self._calls) + cost <= limit
                    if room and not (low and self._urgent_waiting):
                        self._calls.extend([now] * cost)
                        return
                    wait = (60 - (now - self._calls[0]) + 0.5) if not room and self._calls else 1.0
                if not low and not told and wait > 2:
                    print(f"    ліміт API: чекаю {wait:.0f} с…")
                    told = True
                time.sleep(min(wait, 2.0))       # перевіряємо часто: термінові могли звільнити чергу
        finally:
            if not low:
                with self._lock:
                    self._urgent_waiting -= 1

    def get(self, path: str, params: dict | None = None, cost: int = 1):
        for _ in range(5):
            self._reserve(cost)
            resp = self.session.get(f"{BASE_URL}/{path}", params=params, timeout=30)
            self.calls_made += cost
            if resp.status_code == 429:
                self.rate_limited += 1
                wait = float(resp.headers.get("Retry-After", 10))
                print(f"    429 від Coinalyze: чекаю {wait:.0f} с…")
                time.sleep(wait + 0.5)
                continue
            if resp.status_code == 401:
                sys.exit("Помилка 401: невірний або відсутній COINALYZE_API_KEY у .env")
            resp.raise_for_status()
            return resp.json()
        raise RuntimeError(f"{path}: забагато відповідей 429 поспіль")


def find_markets(client: Coinalyze, bases: list[str]) -> list[dict]:
    """Перпетуальні ринки потрібних тікерів на потрібних біржах."""
    names = {e["code"]: e["name"] for e in client.get("exchanges")}
    found = []
    for m in client.get("future-markets"):
        name = names.get(m["exchange"], m["exchange"])
        if (m.get("is_perpetual")
                and m.get("base_asset", "").upper() in bases
                and m.get("quote_asset", "").upper() in QUOTE_ASSETS
                and any(w in name.lower() for w in WANTED_EXCHANGES)):
            m["exchange_name"] = name
            found.append(m)
    return found


def upsert_markets(conn, markets: list[dict]) -> None:
    conn.cursor().executemany(
        """
        INSERT INTO markets (symbol, exchange_code, exchange_name, base_asset, quote_asset,
                             symbol_on_exchange, volume_denominated, has_ohlcv, has_buy_sell,
                             has_long_short, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
        ON CONFLICT (symbol) DO UPDATE SET
            exchange_name = EXCLUDED.exchange_name,
            volume_denominated = EXCLUDED.volume_denominated,
            has_ohlcv = EXCLUDED.has_ohlcv,
            has_buy_sell = EXCLUDED.has_buy_sell,
            has_long_short = EXCLUDED.has_long_short,
            updated_at = now()
        """,
        [(m["symbol"], m["exchange"], m["exchange_name"], m["base_asset"].upper(),
          m["quote_asset"].upper(), m.get("symbol_on_exchange"),
          m.get("oi_lq_vol_denominated_in"), m.get("has_ohlcv_data"),
          m.get("has_buy_sell_data"), m.get("has_long_short_ratio_data"))
         for m in markets],
    )
    conn.commit()


def fetch_endpoint(client, conn, markets, endpoint, table, columns, extra, flag, ts_from, ts_to,
                   interval: str = INTERVAL) -> list:
    symbols = [m["symbol"] for m in markets if flag is None or m.get(flag)]
    if not symbols:
        return 0

    db_cols = [c for c, _ in columns]
    sql = (
        f"INSERT INTO {table} (symbol, interval, ts, {', '.join(db_cols)}) "
        f"VALUES (%s, %s, %s, {', '.join(['%s'] * len(db_cols))}) "
        f"ON CONFLICT (symbol, interval, ts) DO UPDATE SET "
        + ", ".join(f"{c} = EXCLUDED.{c}" for c in db_cols)
    )

    total = []
    for i in range(0, len(symbols), MAX_SYMBOLS_PER_REQUEST):
        chunk = symbols[i:i + MAX_SYMBOLS_PER_REQUEST]
        params = {"symbols": ",".join(chunk), "interval": interval,
                  "from": ts_from, "to": ts_to, **extra}
        rows = []
        for item in client.get(endpoint, params, cost=len(chunk)):
            for p in item.get("history", []):
                ts = datetime.fromtimestamp(p["t"], tz=timezone.utc)
                rows.append((item["symbol"], interval, ts, *[p.get(k) for _, k in columns]))
        if rows:
            conn.cursor().executemany(sql, rows)
            conn.commit()
        total += rows
    return total


LIQ_INTERVALS = {"15m": "15min", "5m": "5min"}


def collect_liquidations(client: Coinalyze, conn, bases: list[str], tfs: list[str], days: float, log=print) -> int:
    """Ліквідації 15m/5m з Coinalyze для ринків тікерів, що вже є в markets.

    Coinalyze тримає внутрішньогодинні ліквідації за кількістю НЕпорожніх точок (~2000–2900 на символ): у дрібних
    монет це 1–2 місяці, в активних — 3–5 тижнів. Бари без ліквідацій у відповіді відсутні, тож «рядка немає» означає
    «0 ліквідацій» лише там, де дані повні. Тому для кожного ринку запам'ятовуємо covered_from — першу точку, яку
    віддав провайдер (раніше неї — «невідомо», а не нуль). Регулярний збір продовжує історію вперед."""
    conn.execute("""CREATE TABLE IF NOT EXISTS liquidation_coverage (
        symbol TEXT REFERENCES markets(symbol), interval TEXT NOT NULL, covered_from TIMESTAMPTZ NOT NULL,
        PRIMARY KEY (symbol, interval))""")
    markets = [{"symbol": r[0]} for r in conn.execute(
        "SELECT symbol FROM markets WHERE base_asset = ANY(%s) AND exchange_name <> 'Aster'", (bases,))]  # Aster без ліквідацій
    if not markets:
        return 0
    now = datetime.now(timezone.utc)
    ts_from, ts_to = int((now - timedelta(days=days)).timestamp()), int(now.timestamp())
    endpoint, table, columns, extra, _ = next(e for e in ENDPOINTS if e[1] == "liquidations")
    total = 0
    for tf in tfs:
        rows = fetch_endpoint(client, conn, markets, endpoint, table, columns, extra, None, ts_from, ts_to,
                              interval=LIQ_INTERVALS[tf])
        first = {}
        for sym, _, ts, *_ in rows:
            first[sym] = min(first.get(sym, ts), ts)
        conn.cursor().executemany("""
            INSERT INTO liquidation_coverage (symbol, interval, covered_from) VALUES (%s, %s, %s)
            ON CONFLICT (symbol, interval) DO UPDATE SET covered_from = least(liquidation_coverage.covered_from,
                                                                               EXCLUDED.covered_from)""",
            [(sym, LIQ_INTERVALS[tf], ts) for sym, ts in first.items()])
        conn.commit()
        total += len(rows)
        log(f"  ліквідації {tf}: {len(rows)} точок, {len(first)} ринків з даними")
    return total


def read_tickers_file(path: str) -> list[str]:
    """Тікери з файлу: один на рядок, після # — коментар, порожні рядки пропускаються."""
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [t for line in f if (t := line.split("#")[0].strip())]


EXCLUDED_FILE = "excluded.txt"   # тікери поза скоупом (смикана ціна, проливи, тонкий стакан): не навчання, не скан, не скринер


def excluded_tickers(root: str | None = None) -> set[str]:
    """Тікери з excluded.txt (рядок: тікер # причина)."""
    path = os.path.join(root, EXCLUDED_FILE) if root else EXCLUDED_FILE
    return {t.upper() for t in read_tickers_file(path)}


def set_ticker_in_file(path: str, base: str, on: bool, comment: str = "") -> bool:
    """Додати тікер у файл (on=True) або прибрати його рядок (on=False); решта рядків і коментарі лишаються.
    Повертає True, якщо файл змінився."""
    base = base.upper()
    lines = open(path, encoding="utf-8").read().splitlines() if os.path.exists(path) else []
    has = any(line.split("#")[0].strip().upper() == base for line in lines)
    if on == has:
        return False
    if on:
        lines.append(f"{base}    # {comment}" if comment else base)
    else:
        lines = [line for line in lines if line.split("#")[0].strip().upper() != base]
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp, path)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Збір 1h-даних Coinalyze у PostgreSQL")
    parser.add_argument("tickers", nargs="*",
                        help=f"базові активи, напр. AKE LSK MYX (без аргументів — з {TICKERS_FILE})")
    parser.add_argument("--days", type=int, default=60,
                        help="скільки днів історії тягнути (1h зберігається ~60-80 днів)")
    parser.add_argument("--new", action="store_true",
                        help="лише тікери, яких ще немає в базі (таблиця markets)")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("COINALYZE_API_KEY")
    db_url = os.getenv("DATABASE_URL")
    if not api_key or not db_url:
        sys.exit("Заповни COINALYZE_API_KEY і DATABASE_URL у файлі .env")

    bases = [t.upper() for t in (args.tickers or read_tickers_file(TICKERS_FILE))]
    if not bases:
        sys.exit(f"Немає тікерів: передай їх аргументами або додай у {TICKERS_FILE}")
    if args.new:
        with psycopg.connect(db_url) as conn:
            known = {r[0] for r in conn.execute("SELECT DISTINCT base_asset FROM markets")}
        skipped = [b for b in bases if b in known]
        bases = [b for b in bases if b not in known]
        if skipped:
            print(f"Вже в базі, пропускаю: {', '.join(skipped)}")
        if not bases:
            sys.exit("Нових тікерів немає — нічого збирати.")
    with psycopg.connect(db_url) as conn:
        if not collect(Coinalyze(api_key), conn, bases, args.days):
            sys.exit("Жодного ринку не знайдено — нічого збирати.")
    print("Готово.")


def collect(client: Coinalyze, conn, bases: list[str], days: int, verbose: bool = True) -> list[dict]:
    """Знайти ринки тікерів і записати `days` днів усіх рядів. Спільне для fetch.py і screener.py."""
    now = datetime.now(timezone.utc)
    ts_to = int(now.timestamp())
    ts_from = int((now - timedelta(days=days)).timestamp())

    if verbose:
        print("Шукаю ринки…")
    markets = find_markets(client, bases)
    for base in bases:
        own = [m for m in markets if m["base_asset"].upper() == base]
        if verbose:
            exchanges = ", ".join(sorted({m["exchange_name"] for m in own})) or "—"
            print(f"  {base}: ринків — {len(own)} ({exchanges})")
        if not own:
            print(f"  ! {base} не знайдено. Перевір назву базового активу "
                  f"(на деяких біржах буває 1000{base} чи k{base}).")
    if not markets:
        return []

    upsert_markets(conn, markets)
    for endpoint, table, columns, extra, flag in ENDPOINTS:
        n = len(fetch_endpoint(client, conn, markets, endpoint, table, columns,
                               extra, flag, ts_from, ts_to))
        if verbose:
            print(f"  {table:<24} рядків: {n}")
    return markets


if __name__ == "__main__":
    main()
