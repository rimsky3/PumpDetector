"""
my_trades.py — угоди власника з Binance USDⓈ-M Futures (лише читання) → позиції → ознаки тільту.

Навіщо: реальні угоди власника — його фільтр у чистому вигляді. Найкращі угоди — приклади патерну (що він ловить, як
заходить і виходить, де стоп), а позначки тільту відсіюють несистемні угоди без опори на пам'ять.

Ключ — у .env: BINANCE_API_KEY / BINANCE_API_SECRET, окремий, ЛИШЕ ЧИТАННЯ (без торгівлі й виведення).
Ключ і підпис нікуди не пишуться (ні в логи, ні у звіти). Сирі угоди — локально в reports/my_trades/ (не в git).

    python my_trades.py --check          # дозволи ключа (без значень)
    python my_trades.py --fetch          # завантажити угоди (до ~6 міс., вікнами по 7 днів)
"""

import argparse
import hashlib
import hmac
import json
import os
import sys
import time
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "reports", "my_trades")
FAPI, SAPI = "https://fapi.binance.com", "https://api.binance.com"
DAY_MS = 86_400_000


class Binance:
    """Підписані GET-запити (HMAC SHA256). Лише читання: інших методів тут немає."""

    def __init__(self, key: str, secret: str):
        self.key, self.secret = key, secret.encode()
        self.s = requests.Session()
        self.s.headers["X-MBX-APIKEY"] = key
        self.offset = 0

    def sync(self) -> None:
        t = self.s.get(f"{FAPI}/fapi/v1/time", timeout=10).json()["serverTime"]
        self.offset = t - int(time.time() * 1000)

    def get(self, base: str, path: str, **params) -> dict | list:
        params = {k: v for k, v in params.items() if v is not None}
        for attempt in range(8):
            # підпис — на кожну спробу: мітка часу старіша за recvWindow відхиляється
            q = urlencode({**params, "timestamp": int(time.time() * 1000) + self.offset, "recvWindow": 10_000})
            sig = hmac.new(self.secret, q.encode(), hashlib.sha256).hexdigest()
            try:
                r = self.s.get(f"{base}{path}?{q}&signature={sig}", timeout=20)
            except requests.RequestException:          # обрив з'єднання (Binance скидає довгі серії) — пауза і ще раз
                time.sleep(3 + attempt * 3)
                continue
            if r.status_code in (418, 429):
                time.sleep(int(r.headers.get("Retry-After", 5)) + attempt * 2)
                continue
            if r.status_code != 200:
                raise RuntimeError(f"{path}: HTTP {r.status_code} {r.text[:200]}")
            return r.json()
        raise RuntimeError(f"{path}: ліміт запитів")


def client() -> Binance:
    load_dotenv(os.path.join(ROOT, ".env"))
    key, secret = os.getenv("BINANCE_API_KEY"), os.getenv("BINANCE_API_SECRET")
    if not key or not secret:
        sys.exit("Немає BINANCE_API_KEY / BINANCE_API_SECRET у .env")
    b = Binance(key, secret)
    b.sync()
    return b


def check(b: Binance) -> dict:
    """Дозволи ключа: має бути лише читання."""
    r = b.get(SAPI, "/sapi/v1/account/apiRestrictions")
    keys = ["enableReading", "enableFutures", "enableSpotAndMarginTrading", "enableWithdrawals", "enableInternalTransfer",
            "enableMargin", "permitsUniversalTransfer", "enableVanillaOptions", "ipRestrict"]
    return {k: r.get(k) for k in keys if k in r}


def symbols_traded(b: Binance, days: int) -> list[str]:
    """Символи з рухом коштів (реалізований PnL / комісія) за days днів — userTrades без символу не працює."""
    end = int(time.time() * 1000)
    syms = set()
    start = end - days * DAY_MS
    while start < end:
        stop = min(start + 7 * DAY_MS, end)
        cur = start
        while True:
            rows = b.get(FAPI, "/fapi/v1/income", startTime=cur, endTime=stop, limit=1000)
            syms |= {r["symbol"] for r in rows if r.get("symbol") and r.get("incomeType") in ("REALIZED_PNL", "COMMISSION")}
            if len(rows) < 1000:
                break
            cur = int(rows[-1]["time"]) + 1
        start = stop
    return sorted(syms)


def fetch(b: Binance, days: int, log=print) -> pd.DataFrame:
    """Усі виконання (userTrades) за days днів по символах з руху коштів; вікна по 7 днів (обмеження Binance)."""
    syms = symbols_traded(b, days)
    log(f"символів з угодами: {len(syms)}")
    end = int(time.time() * 1000)
    # прогрес по символах — у файлі: після обриву продовжуємо з наступного символу
    part_path = os.path.join(OUT_DIR, "_partial.json")
    saved = json.load(open(part_path, encoding="utf-8")) if os.path.exists(part_path) else {}
    rows = [r for v in saved.values() for r in v]
    for n, sym in enumerate(syms, 1):
        if sym in saved:
            continue
        got = []
        start = end - days * DAY_MS
        while start < end:
            stop = min(start + 7 * DAY_MS, end)
            cur = start
            while True:
                part = b.get(FAPI, "/fapi/v1/userTrades", symbol=sym, startTime=cur, endTime=stop, limit=1000)
                got += part
                if len(part) < 1000:
                    break
                cur = int(part[-1]["time"]) + 1
            start = stop
        rows += got
        saved[sym] = got
        json.dump(saved, open(part_path, "w", encoding="utf-8"))
        if n % 10 == 0:
            log(f"  {n}/{len(syms)}")
    df = pd.DataFrame(rows)
    if os.path.exists(part_path):
        os.remove(part_path)
    if df.empty:
        return df
    for c in ("price", "qty", "quoteQty", "realizedPnl", "commission"):
        df[c] = df[c].astype(float)
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    return df.drop_duplicates("id").sort_values(["time", "id"]).reset_index(drop=True)


def positions(tr: pd.DataFrame) -> pd.DataFrame:
    """Виконання → позиції: від нуля до нуля нетто-позиції по символу (і по positionSide у hedge-режимі)."""
    out = []
    for (sym, ps), g in tr.groupby(["symbol", "positionSide"]):
        net, cur = 0.0, None
        for r in g.itertuples():
            q = r.qty if r.side == "BUY" else -r.qty
            if cur is None:
                cur = {"symbol": sym, "side": "long" if q > 0 else "short", "entry_time": r.time, "fills": 0, "buy_qty": 0.0,
                       "buy_usd": 0.0, "sell_qty": 0.0, "sell_usd": 0.0, "pnl": 0.0, "fee": 0.0, "max_usd": 0.0}
            cur["fills"] += 1
            if r.side == "BUY":
                cur["buy_qty"] += r.qty; cur["buy_usd"] += r.quoteQty
            else:
                cur["sell_qty"] += r.qty; cur["sell_usd"] += r.quoteQty
            cur["pnl"] += r.realizedPnl
            cur["fee"] += r.commission if r.commissionAsset in ("USDT", "USDC", "BUSD") else 0.0
            net += q
            cur["max_usd"] = max(cur["max_usd"], abs(net) * r.price)
            if abs(net) < 1e-12 * max(1.0, abs(q)) or abs(net) * r.price < 0.5:      # позиція закрита (пил < $0.5)
                cur["exit_time"] = r.time
                out.append(cur)
                net, cur = 0.0, None
        if cur is not None:
            cur["exit_time"] = pd.NaT
            out.append(cur)
    p = pd.DataFrame(out)
    if p.empty:
        return p
    long = p["side"] == "long"
    p["entry_px"] = (p["buy_usd"] / p["buy_qty"]).where(long, p["sell_usd"] / p["sell_qty"])
    p["exit_px"] = (p["sell_usd"] / p["sell_qty"]).where(long, p["buy_usd"] / p["buy_qty"])
    p["net"] = p["pnl"] - p["fee"]
    p["ret"] = p["net"] / p["max_usd"]
    p["minutes"] = (p["exit_time"] - p["entry_time"]).dt.total_seconds() / 60
    p["base"] = p["symbol"].str.replace(r"(USDT|USDC)$", "", regex=True).str.replace(r"^1000+", "", regex=True)
    return p.sort_values("entry_time").reset_index(drop=True)


REVENGE_MIN, BURST_N, BURST_MIN = 5, 5, 20      # пороги власника (2026-10-02)


def tilt_flags(p: pd.DataFrame) -> pd.DataFrame:
    """Ознаки тільту з самих угод: «відігратися» — вхід ≤ 5 хв після закриття збиткової позиції (будь-яка монета);
    «серія» — 5+ входів за 20 хв (вікно навколо входу). «Тиха монета» — окремо (потрібні ринкові дані)."""
    p = p.copy()
    closed_loss = p[p["exit_time"].notna() & (p["net"] < 0)]["exit_time"].sort_values().to_numpy()
    et = p["entry_time"].to_numpy()

    j = np.searchsorted(closed_loss, et, side="right") - 1
    gap = np.where(j >= 0, (et - closed_loss[np.clip(j, 0, None)]) / np.timedelta64(1, "m"), np.inf) if len(closed_loss) else np.full(len(p), np.inf)
    p["revenge"] = gap <= REVENGE_MIN
    win = np.timedelta64(BURST_MIN, "m")
    # серія = у будь-якому 20-хвилинному вікні, що містить цей вхід, ≥ 5 входів
    burst = np.zeros(len(p), dtype=bool)
    for i, t in enumerate(et):
        lo = np.searchsorted(et, t - win, side="left")
        for k in range(lo, i + 1):
            hi = np.searchsorted(et, et[k] + win, side="right")
            if hi - k >= BURST_N:
                burst[i] = True
                break
    p["burst"] = burst
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description="Угоди власника з Binance Futures (лише читання)")
    ap.add_argument("--check", action="store_true", help="перевірити дозволи ключа")
    ap.add_argument("--fetch", action="store_true", help="завантажити угоди")
    ap.add_argument("--days", type=int, default=180, help="скільки днів назад (Binance віддає ~6 міс.)")
    ap.add_argument("--update", action="store_true",
                    help="лише нові угоди: від останньої в fills.parquet (мінус доба) і злити зі збереженими; хвилини замість годин")
    args = ap.parse_args()
    b = client()
    if args.check or not (args.fetch or args.update):
        print(json.dumps(check(b), ensure_ascii=False, indent=2))
    if args.fetch or args.update:
        os.makedirs(OUT_DIR, exist_ok=True)
        fills_path, old, days = os.path.join(OUT_DIR, "fills.parquet"), None, args.days
        if args.update and os.path.exists(fills_path):
            old = pd.read_parquet(fills_path)
            days = max(1, int(np.ceil((pd.Timestamp.now(tz="UTC") - old["time"].max()) / pd.Timedelta(days=1))) + 1)
            print(f"збережено по {old['time'].max():%Y-%m-%d %H:%M} UTC — дотягую {days} дн.")
        tr = fetch(b, days)
        if old is not None:
            tr = (pd.concat([old, tr], ignore_index=True) if len(tr) else old)
            tr = tr.drop_duplicates("id").sort_values(["time", "id"]).reset_index(drop=True)
        if tr.empty:
            sys.exit("угод не знайдено")
        tr.to_parquet(os.path.join(OUT_DIR, "fills.parquet"))
        p = tilt_flags(positions(tr))
        p.to_parquet(os.path.join(OUT_DIR, "positions.parquet"))
        print(f"виконань: {len(tr)}, позицій: {len(p)} ({(p['side'] == 'long').sum()} лонг, {(p['side'] == 'short').sum()} шорт), "
              f"з {p['entry_time'].min():%Y-%m-%d} по {p['entry_time'].max():%Y-%m-%d}")


if __name__ == "__main__":
    main()
