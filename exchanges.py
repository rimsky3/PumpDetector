"""
exchanges.py — публічні API Binance і Bybit (USDT-перпи): свічки, open interest, long/short, funding.

Ключ не потрібен, ліміти на порядки вищі за Coinalyze. Значення збігаються з Coinalyze один в один
(experiments/exchange_direct.py), якщо правильно вирівняти час — див. normalize().

Глибина історії (вересень 2026): свічки і funding — роки; OI і long/short на Bybit — щонайменше рік,
на Binance — лише 30 днів. Ліквідацій і predicted funding історії немає.

Використовують: fetch_exchange.py (15m/5m у базу) і експерименти.
"""

import threading
import time

import numpy as np
import pandas as pd
import requests

BINANCE_HIST_DAYS = 29.5          # /futures/data/* віддає лише останні 30 днів
H = pd.Timedelta(hours=1)
FUNDING_LOOKBACK = 2 * 24 * 3_600_000   # funding платиться раз на 1–8 год: беремо з запасом, щоб у вікні була остання ставка
MS_H = 3_600_000
# таймфрейм → мілісекунди бару і назви інтервалу в API бірж (використовує і experiments/timeframe.py)
TF = {
    "1h": dict(ms=MS_H, binance="1h", bybit_kline="60", bybit="1h"),
    "15m": dict(ms=MS_H // 4, binance="15m", bybit_kline="15", bybit="15min"),
    "5m": dict(ms=MS_H // 12, binance="5m", bybit_kline="5", bybit="5min"),
}

BINANCE = "https://fapi.binance.com"
BYBIT = "https://api.bybit.com"
stats = {"requests": 0, "seconds": 0.0}
_stats_lock = threading.Lock()   # get() викликають з кількох потоків


# ---------------------------------------------------------------- завантаження з бірж

def get(url: str, params: dict):
    for attempt in range(5):
        t0 = time.time()
        r = requests.get(url, params=params, timeout=30)
        with _stats_lock:
            stats["requests"] += 1
            stats["seconds"] += time.time() - t0
        if r.status_code in (418, 429) or r.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict) and data.get("retCode", 0) == 10006:
            time.sleep(2 ** attempt)          # Bybit повідомляє про ліміт кодом у тілі, а не HTTP 429
            continue
        if isinstance(data, dict) and data.get("retCode", 0) != 0:
            raise RuntimeError(f"{url}: {data.get('retMsg')}")
        return data
    raise RuntimeError(f"{url}: забагато повторів")


def to_ts(ms) -> pd.Timestamp:
    return pd.to_datetime(int(ms), unit="ms", utc=True)


def binance_klines(sym: str, start: int, end: int, tf: str = "1h") -> pd.DataFrame:
    rows, t = [], start
    while t < end:
        page = get(BINANCE + "/fapi/v1/klines", dict(symbol=sym, interval=TF[tf]["binance"], startTime=t, endTime=end, limit=1500))
        if not page:
            break
        rows += page
        t = page[-1][0] + TF[tf]["ms"]
    # [open_time, o, h, l, c, volume (монети), close_time, quote_volume, trades, taker_buy_base, ...]
    return pd.DataFrame({"ts": [to_ts(r[0]) for r in rows], "o": [float(r[1]) for r in rows],
                         "h": [float(r[2]) for r in rows], "l": [float(r[3]) for r in rows],
                         "c": [float(r[4]) for r in rows], "v": [float(r[5]) for r in rows],
                         "bv": [float(r[9]) for r in rows]})


def binance_hist(path: str, sym: str, start: int, end: int, field: str, tf: str = "1h") -> pd.DataFrame:
    """/futures/data/*: OI і long/short. Лише 30 днів, до 500 точок за запит.
    Якщо діапазон довший за 500 барів, Binance віддає ОСТАННІ 500 — тому йдемо вікнами по 500 барів."""
    rows, t = [], max(start, int(end - BINANCE_HIST_DAYS * 24 * MS_H))
    while t < end:
        e = min(t + 499 * TF[tf]["ms"], end)
        rows += get(BINANCE + path, dict(symbol=sym, period=TF[tf]["binance"], startTime=t, endTime=e, limit=500))
        t = e + 1
    df = pd.DataFrame({"ts": [to_ts(r["timestamp"]) for r in rows], "value": [float(r[field]) for r in rows]})
    return df.drop_duplicates("ts").sort_values("ts")


def binance_funding(sym: str, start: int, end: int) -> pd.DataFrame:
    rows, t = [], start
    while t < end:
        page = get(BINANCE + "/fapi/v1/fundingRate", dict(symbol=sym, startTime=t, endTime=end, limit=1000))
        if not page:
            break
        rows += page
        if len(page) < 1000:
            break
        t = int(page[-1]["fundingTime"]) + 1
    return pd.DataFrame({"ts": [to_ts(r["fundingTime"]) for r in rows], "value": [float(r["fundingRate"]) for r in rows]})


def bybit_klines(sym: str, start: int, end: int, tf: str = "1h") -> pd.DataFrame:
    rows, e = [], end
    while e > start:
        page = get(BYBIT + "/v5/market/kline", dict(category="linear", symbol=sym, interval=TF[tf]["bybit_kline"],
                                                     start=start, end=e, limit=1000))["result"]["list"]
        if not page:
            break
        rows += page                      # новіші спершу
        e = int(page[-1][0]) - 1
    # [start, o, h, l, c, volume (монети), turnover (USD)]; buy volume Bybit у свічках не віддає
    df = pd.DataFrame({"ts": [to_ts(r[0]) for r in rows], "o": [float(r[1]) for r in rows],
                       "h": [float(r[2]) for r in rows], "l": [float(r[3]) for r in rows],
                       "c": [float(r[4]) for r in rows], "v": [float(r[5]) for r in rows]})
    return df.drop_duplicates("ts").sort_values("ts")


def bybit_cursor(path: str, sym: str, start: int, end: int, extra: dict, field: str, limit: int) -> pd.DataFrame:
    rows, cursor = [], None
    while True:
        p = dict(category="linear", symbol=sym, startTime=start, endTime=end, limit=limit, **extra)
        if cursor:
            p["cursor"] = cursor
        res = get(BYBIT + path, p)["result"]
        rows += res["list"]
        cursor = res.get("nextPageCursor")
        if not cursor or not res["list"]:
            break
    df = pd.DataFrame({"ts": [to_ts(r["timestamp"]) for r in rows], "value": [float(r[field]) for r in rows]})
    return df.drop_duplicates("ts").sort_values("ts")


def bybit_funding(sym: str, start: int, end: int) -> pd.DataFrame:
    rows, e = [], end
    while e > start:
        page = get(BYBIT + "/v5/market/funding/history",
                   dict(category="linear", symbol=sym, startTime=start, endTime=e, limit=200))["result"]["list"]
        if not page:
            break
        rows += page
        e = int(page[-1]["fundingRateTimestamp"]) - 1
    df = pd.DataFrame({"ts": [to_ts(r["fundingRateTimestamp"]) for r in rows],
                       "value": [float(r["fundingRate"]) for r in rows]})
    return df.drop_duplicates("ts").sort_values("ts")


def download(exchange: str, sym: str, start: int, end: int, tf: str = "1h") -> dict[str, pd.DataFrame]:
    if exchange == "Binance":
        return {
            "klines": binance_klines(sym, start, end, tf),
            "oi": binance_hist("/futures/data/openInterestHist", sym, start, end, "sumOpenInterest", tf),   # у монетах
            "ls": binance_hist("/futures/data/globalLongShortAccountRatio", sym, start, end, "longAccount", tf),
            "funding": binance_funding(sym, start - FUNDING_LOOKBACK, end),
        }
    return {
        "klines": bybit_klines(sym, start, end, tf),
        "oi": bybit_cursor("/v5/market/open-interest", sym, start, end, {"intervalTime": TF[tf]["bybit"]}, "openInterest", 200),
        "ls": bybit_cursor("/v5/market/account-ratio", sym, start, end, {"period": TF[tf]["bybit"]}, "buyRatio", 500),
        "funding": bybit_funding(sym, start - FUNDING_LOOKBACK, end),
    }


# назви інтервалів у колонці interval сирих таблиць (як у Coinalyze)
INTERVAL = {"1h": "1hour", "15m": "15min", "5m": "5min"}


def normalize(data: dict[str, pd.DataFrame], tf: str) -> dict[str, pd.DataFrame]:
    """Ряди біржі → одиниці й вирівнювання сирих таблиць Coinalyze (ключ — час ВІДКРИТТЯ бару):
    - ohlcv: обсяг у монетах; buy volume є лише в Binance (у Bybit — NaN);
    - OI: знімок біржі в момент T+бар — це закриття бару T; у USD за close бару (o = h = l = c);
    - funding: остання виплачена ставка на початок бару (виплата о T входить у бар T), у відсотках;
    - long/short: частка лонг-акаунтів у відсотках, знімок на момент T."""
    bar = pd.Timedelta(milliseconds=TF[tf]["ms"])
    # порожня відповідь біржі (напр. без виплат funding у вікні) дає колонку ts без типу — приводимо до часу UTC
    data = {k: v.assign(ts=pd.to_datetime(v["ts"], utc=True).astype("datetime64[us, UTC]")) if "ts" in v else v
            for k, v in data.items()}
    k = data["klines"].copy()
    if "bv" not in k:
        k["bv"] = np.nan
    oi = data["oi"].assign(ts=data["oi"]["ts"] - bar).merge(k[["ts", "c"]], on="ts").sort_values("ts")
    # свічка зі знімків: open = знімок на початку бару (закриття попереднього), close — на кінці;
    # high/low — більше й менше з них (внутрішньобарових екстремумів біржа не дає). Після пропуску open = close.
    # У USD уся свічка — за close бару, як у Coinalyze: тоді ділення на close повертає точні монети.
    coins_open = oi["value"].shift(1).where(oi["ts"].diff() == bar).fillna(oi["value"])
    oi["o"] = coins_open * oi["c"]
    oi["c"] = oi["value"] * oi["c"]
    oi = oi.assign(h=oi[["o", "c"]].max(axis=1), l=oi[["o", "c"]].min(axis=1))[["ts", "o", "h", "l", "c"]]
    # Binance ставить час виплати на 1 мс пізніше (16:00:00.001) — округлюємо до хвилини
    paid = data["funding"].assign(ts=data["funding"]["ts"].dt.floor("min"))
    f = pd.merge_asof(k[["ts"]].sort_values("ts"), paid.sort_values("ts"), on="ts", direction="backward")
    f = f.assign(c=f["value"] * 100).dropna(subset=["c"])
    f = f.assign(o=f["c"], h=f["c"], l=f["c"])[["ts", "o", "h", "l", "c"]]
    ls = data["ls"].assign(long_pct=data["ls"]["value"] * 100)
    ls = ls.assign(short_pct=100 - ls["long_pct"], ratio=ls["long_pct"] / (100 - ls["long_pct"]))
    return {"ohlcv": k[["ts", "o", "h", "l", "c", "v", "bv"]], "oi": oi, "funding": f,
            "ls": ls[["ts", "ratio", "long_pct", "short_pct"]]}
