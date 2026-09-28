"""
fetch.py — збір годинних (1h) даних з Coinalyze у PostgreSQL.

Для кожного тікера знаходить його перпетуальні ф'ючерси на Binance, Bybit, OKX,
Aster, Hyperliquid і MEXC та тягне шість рядів: open interest, OHLCV, funding,
predicted funding, ліквідації, long/short ratio.

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
import time
from datetime import datetime, timedelta, timezone

import psycopg
import requests
from dotenv import load_dotenv

BASE_URL = "https://api.coinalyze.net/v1"
INTERVAL = "1hour"
WANTED_EXCHANGES = ["binance", "bybit", "okx", "aster", "hyperliquid", "mexc"]
QUOTE_ASSETS = {"USDT", "USD", "USDC"}
RATE_LIMIT_PER_MIN = 40      # кожен символ у запиті = один виклик
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
    ("predicted-funding-rate-history", "predicted_funding_rate",
     [("o", "o"), ("h", "h"), ("l", "l"), ("c", "c")], {}, None),
    ("liquidation-history", "liquidations",
     [("long_liq", "l"), ("short_liq", "s")], {"convert_to_usd": "true"}, None),
    ("long-short-ratio-history", "long_short_ratio",
     [("ratio", "r"), ("long_pct", "l"), ("short_pct", "s")], {}, "has_long_short_ratio_data"),
]


class Coinalyze:
    """Клієнт з дотриманням ліміту 40 викликів/хв і обробкою 429 (Retry-After)."""

    def __init__(self, api_key: str):
        self.session = requests.Session()
        self.session.headers["api_key"] = api_key
        self._calls = collections.deque()  # моменти спожитих викликів

    def _reserve(self, cost: int) -> None:
        while True:
            now = time.monotonic()
            while self._calls and now - self._calls[0] >= 60:
                self._calls.popleft()
            if len(self._calls) + cost <= RATE_LIMIT_PER_MIN:
                self._calls.extend([now] * cost)
                return
            wait = 60 - (now - self._calls[0]) + 0.5
            print(f"    ліміт API: чекаю {wait:.0f} с…")
            time.sleep(wait)

    def get(self, path: str, params: dict | None = None, cost: int = 1):
        for _ in range(5):
            self._reserve(cost)
            resp = self.session.get(f"{BASE_URL}/{path}", params=params, timeout=30)
            if resp.status_code == 429:
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


def fetch_endpoint(client, conn, markets, endpoint, table, columns, extra, flag, ts_from, ts_to) -> int:
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

    total = 0
    for i in range(0, len(symbols), MAX_SYMBOLS_PER_REQUEST):
        chunk = symbols[i:i + MAX_SYMBOLS_PER_REQUEST]
        params = {"symbols": ",".join(chunk), "interval": INTERVAL,
                  "from": ts_from, "to": ts_to, **extra}
        rows = []
        for item in client.get(endpoint, params, cost=len(chunk)):
            for p in item.get("history", []):
                ts = datetime.fromtimestamp(p["t"], tz=timezone.utc)
                rows.append((item["symbol"], INTERVAL, ts, *[p.get(k) for _, k in columns]))
        if rows:
            conn.cursor().executemany(sql, rows)
            conn.commit()
        total += len(rows)
    return total


def read_tickers_file(path: str) -> list[str]:
    """Тікери з файлу: один на рядок, після # — коментар, порожні рядки пропускаються."""
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [t for line in f if (t := line.split("#")[0].strip())]


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
    now = datetime.now(timezone.utc)
    ts_to = int(now.timestamp())
    ts_from = int((now - timedelta(days=args.days)).timestamp())

    client = Coinalyze(api_key)
    print("Шукаю ринки…")
    markets = find_markets(client, bases)

    for base in bases:
        own = [m for m in markets if m["base_asset"].upper() == base]
        exchanges = ", ".join(sorted({m["exchange_name"] for m in own})) or "—"
        print(f"  {base}: ринків — {len(own)} ({exchanges})")
        if not own:
            print(f"  ! {base} не знайдено. Перевір назву базового активу "
                  f"(на деяких біржах буває 1000{base} чи k{base}).")

    if not markets:
        sys.exit("Жодного ринку не знайдено — нічого збирати.")

    with psycopg.connect(db_url) as conn:
        upsert_markets(conn, markets)
        for endpoint, table, columns, extra, flag in ENDPOINTS:
            n = fetch_endpoint(client, conn, markets, endpoint, table, columns,
                               extra, flag, ts_from, ts_to)
            print(f"  {table:<24} рядків: {n}")

    print("Готово.")


if __name__ == "__main__":
    main()
