"""
experiments/klines1m.py — 1m-свічки Binance USDT-перпів з публічного архіву data.binance.vision (2026-10-03).

Для експериментів з мікроструктурою (L-shock, рівні профілю обсягу): 1m — найдрібніші свічки, які є в архіві
за будь-який період без ключа й лімітів. Місячні zip (1–3 МБ на монету) → data/klines1m/<монета>.parquet.
Колонки: o h l c, v — оборот у USDT (quote volume), bv — оборот маркет-покупок (taker buy quote), n — угод.
Час — відкриття хвилини, UTC. Повторний запуск докачує лише відсутні місяці.

Монети за замовчуванням: універсум довгого бектесту (data/long_bt) + tickers.txt + тікери з розміткою.

    python experiments/klines1m.py                         # квітень–вересень 2026
    python experiments/klines1m.py --months 2026-09 SOON   # один місяць, одна монета
"""

import argparse
import io
import os
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(ROOT, "data", "klines1m")
URL = "https://data.binance.vision/data/futures/um/monthly/klines/{s}/1m/{s}-1m-{m}.zip"
MONTHS = ["2026-04", "2026-05", "2026-06", "2026-07", "2026-08", "2026-09"]
COLS = ["open_time", "o", "h", "l", "c", "vol", "close_time", "v", "n", "bvol", "bv", "ignore"]


def default_bases() -> list[str]:
    from fetch import read_tickers_file
    from load_pumps import read_pumps
    lb = os.path.join(ROOT, "data", "long_bt")
    bases = {f[:-4] for f in os.listdir(lb) if f.endswith(".pkl")} if os.path.isdir(lb) else set()
    bases |= set(read_tickers_file(os.path.join(ROOT, "tickers.txt")))
    bases |= {r[0] for r in read_pumps()}
    return sorted(bases)


def path(base: str) -> str:
    return os.path.join(DIR, f"{base}.parquet")


def _month(base: str, month: str) -> pd.DataFrame | None:
    s = quote(f"{base}USDT")
    try:
        r = requests.get(URL.format(s=s, m=month), timeout=60)
    except requests.RequestException:
        return None
    if r.status_code != 200:
        return None                                   # монети ще / вже не було на біржі
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        raw = z.read(z.namelist()[0])
    head = raw[:20].decode("ascii", "ignore")
    df = pd.read_csv(io.BytesIO(raw), header=0 if head.startswith("open_time") else None, names=COLS)
    df.index = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df[["o", "h", "l", "c", "v", "bv", "n"]].astype("float64")


def fetch(base: str, months: list[str]) -> str:
    os.makedirs(DIR, exist_ok=True)
    have = pd.read_parquet(path(base)) if os.path.exists(path(base)) else pd.DataFrame()
    got = set(have.index.strftime("%Y-%m").unique()) if len(have) else set()
    parts = [have] if len(have) else []
    new = 0
    for m in months:
        if m in got:
            continue
        df = _month(base, m)
        if df is not None and len(df):
            parts.append(df)
            new += 1
    if new:
        out = pd.concat(parts).sort_index()
        out = out[~out.index.duplicated(keep="last")]
        out.to_parquet(path(base))
    return f"{base}: +{new} міс."


def load(base: str, start=None, end=None) -> pd.DataFrame:
    """1m-свічки з кешу (порожньо, якщо монети немає)."""
    if not os.path.exists(path(base)):
        return pd.DataFrame()
    df = pd.read_parquet(path(base))
    if start is not None:
        df = df[df.index >= pd.Timestamp(start, tz="UTC") if not isinstance(start, pd.Timestamp) else df.index >= start]
    if end is not None:
        df = df[df.index < (pd.Timestamp(end, tz="UTC") if not isinstance(end, pd.Timestamp) else end)]
    return df


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """1m → крупніші бари (5min, 15min, h): OHLC, суми обороту."""
    if df.empty:
        return df
    g = df.resample(rule, label="left", closed="left")
    out = pd.DataFrame({"o": g["o"].first(), "h": g["h"].max(), "l": g["l"].min(), "c": g["c"].last(),
                        "v": g["v"].sum(), "bv": g["bv"].sum(), "n": g["n"].sum()})
    return out.dropna(subset=["c"])


def available() -> list[str]:
    return sorted(f[:-8] for f in os.listdir(DIR) if f.endswith(".parquet")) if os.path.isdir(DIR) else []


def main() -> None:
    ap = argparse.ArgumentParser(description="1m-свічки Binance USDT-перпів з архіву data.binance.vision → data/klines1m/.")
    ap.add_argument("bases", nargs="*", help="монети (за замовчуванням — універсум довгого бектесту + навчальні + розмічені)")
    ap.add_argument("--months", nargs="+", default=MONTHS, help="місяці YYYY-MM (за замовчуванням квітень–вересень 2026)")
    ap.add_argument("--workers", type=int, default=8, help="паралельних завантажень")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    bases = [b.upper() for b in args.bases] or default_bases()
    print(f"{len(bases)} монет × {len(args.months)} міс. → {DIR}")
    with ThreadPoolExecutor(args.workers) as ex:
        for i, msg in enumerate(ex.map(lambda b: fetch(b, args.months), bases), 1):
            if i % 20 == 0 or i == len(bases):
                print(f"  {i}/{len(bases)}  {msg}", flush=True)


if __name__ == "__main__":
    main()
