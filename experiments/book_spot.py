"""
experiments/book_spot.py — стакан і спот на момент входу моделі: чи відділяють прибуткові входи від збиткових (2026-10-02).

Усі попередні перевірки: з ціни, обсягу, OI, funding, L/S, ліквідацій модель упізнає сплеск, але не відрізняє той, що
продовжиться, від того, що згасне; перевага власника — у стакані («тонко», «відкуповують лоу») і в розумінні, хто
рухає ціну. Тут — дві нові групи даних на момент входу (лише минуле):

  СТАКАН — архів Binance data.binance.vision, futures/um/daily/bookDepth: кожні 30 с сумарний обсяг заявок у смугах
           ±0.2 / 1 / 2 / 5% від ціни (біди — мінус, аски — плюс). Ознаки:
               imb_02 / imb_1 / imb_2 / imb_5   (біди − аски) / (біди + аски) у смузі; > 0 — покупців більше
               d_imb_1_15m                      зміна imb_1 за 15 хв до входу
               depth_1_rel                      (біди + аски) ±1% / медіана за попередню добу — тонкість стакану
               ask_1_rel, bid_1_rel             окремо аски / біди ±1% відносно своєї медіани за добу
  СПОТ   — Binance spot API і премія перпа (fapi premiumIndexKlines):
               has_spot                         чи є спот-ринок на Binance
               spot_share_4h                    частка спотового обороту в (спот + перп) за 4 год
               spot_buy_1h                      частка агресивних покупок на споті за 1 год
               premium, premium_chg_4h          премія перпа до індексу (спот) і її зміна за 4 год

Угоди — останні прогони backtest.py робочої моделі (Alert, частковий вихід) на монетах поза навчанням. Для кожної ознаки:
квінтилі (5 рівних груп за значенням) → угод, у плюсі, середня, PF; кореляція Спірмена з результатом. Чесна перевірка
правила: поріг обирається на першій половині угод (за часом), перевіряється на другій.

Сирі файли — кеш data/book_cache/ (не в git, не в базі); ознаки — reports/book_spot_features.csv.

    python experiments/book_spot.py
"""

import argparse
import io
import json
import os
import sys
import time
import zipfile
from urllib.parse import quote

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import psycopg
import requests
from dotenv import load_dotenv

RUNS = ["20261002_0001_current_alert", "20261002_0003_current_alert"]
CACHE = os.path.join("data", "book_cache")
OUT = os.path.join("reports", "book_spot_features.csv")
B15 = pd.Timedelta(minutes=15)
S = requests.Session()


def get_json(url: str, params: dict):
    for attempt in range(6):
        try:
            r = S.get(url, params=params, timeout=20)
            if r.status_code in (418, 429):
                time.sleep(5 + attempt * 5)
                continue
            if r.status_code == 400:
                return None                       # немає такого символу (наприклад, споту)
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            time.sleep(2 + attempt * 2)
    return None


def book_day(sym: str, day: str) -> pd.DataFrame:
    """Денний файл bookDepth (кеш на диску). Порожній — якщо файлу немає."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{sym}-bookDepth-{day}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    url = f"https://data.binance.vision/data/futures/um/daily/bookDepth/{quote(sym)}/{quote(sym)}-bookDepth-{day}.zip"
    df = pd.DataFrame(columns=["timestamp", "percentage", "notional"])
    for attempt in range(4):
        try:
            r = S.get(url, timeout=60)
            if r.status_code == 404:
                break
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                df = pd.read_csv(z.open(z.namelist()[0]), usecols=["timestamp", "percentage", "notional"])
            break
        except (requests.RequestException, zipfile.BadZipFile):
            time.sleep(3 + attempt * 3)
    if len(df):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.to_parquet(path)
    return df


def book_wide(sym: str, days: list[str]) -> pd.DataFrame:
    """Зрізи стакану: рядок = момент, колонки = смуги (−5 … 5), значення — notional, $."""
    parts = [book_day(sym, d) for d in days]
    parts = [p for p in parts if len(p)]
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts)
    return df.pivot_table(index="timestamp", columns="percentage", values="notional", aggfunc="last").sort_index()


def book_features(w: pd.DataFrame, T: pd.Timestamp) -> dict:
    if w.empty:
        return {}
    pre = w[w.index <= T]
    if pre.empty or T - pre.index[-1] > pd.Timedelta(minutes=5):
        return {}

    def imb(row, p):
        b, a = row.get(-p, np.nan), row.get(p, np.nan)
        return (b - a) / (b + a) if b + a else np.nan
    now = pre.iloc[-1]
    ago = pre[pre.index <= T - B15]
    day = pre[pre.index > T - pd.Timedelta(hours=24)]
    tot1 = day[-1.0] + day[1.0]
    return {"imb_02": imb(now, 0.2), "imb_1": imb(now, 1.0), "imb_2": imb(now, 2.0), "imb_5": imb(now, 5.0),
            "d_imb_1_15m": imb(now, 1.0) - imb(ago.iloc[-1], 1.0) if len(ago) else np.nan,
            "depth_1_rel": (now[-1.0] + now[1.0]) / tot1.median() if len(day) > 100 else np.nan,
            "ask_1_rel": now[1.0] / day[1.0].median() if len(day) > 100 else np.nan,
            "bid_1_rel": now[-1.0] / day[-1.0].median() if len(day) > 100 else np.nan}


def klines(base: str, path: str, sym: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    rows = get_json(base + path, {"symbol": sym, "interval": "15m", "startTime": int(start.timestamp() * 1000),
                                  "endTime": int(end.timestamp() * 1000), "limit": 1000})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).iloc[:, :11]
    df.columns = ["t", "o", "h", "l", "c", "v", "ct", "qv", "n", "tbv", "tbqv"]
    df["t"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    return df.set_index("t").astype(float)


def spot_features(sym: str, T: pd.Timestamp) -> dict:
    """T — момент закриття сигнального бару; беремо лише бари, що закрились до T."""
    start = T - pd.Timedelta(hours=6)
    sp = klines("https://api.binance.com", "/api/v3/klines", sym, start, T - pd.Timedelta(seconds=1))
    fu = klines("https://fapi.binance.com", "/fapi/v1/klines", sym, start, T - pd.Timedelta(seconds=1))
    pr = klines("https://fapi.binance.com", "/fapi/v1/premiumIndexKlines", sym, start, T - pd.Timedelta(seconds=1))
    out = {"has_spot": float(len(sp) > 0)}
    if len(sp) and len(fu):
        s4, f4 = sp["qv"].tail(16).sum(), fu["qv"].tail(16).sum()
        out["spot_share_4h"] = s4 / (s4 + f4) if s4 + f4 else np.nan
        out["spot_buy_1h"] = sp["tbqv"].tail(4).sum() / sp["qv"].tail(4).sum() if sp["qv"].tail(4).sum() else np.nan
    if len(pr) >= 17:
        out["premium"] = pr["c"].iloc[-1]
        out["premium_chg_4h"] = pr["c"].iloc[-1] - pr["c"].iloc[-17]
    return out


def pf(v: pd.Series) -> float:
    gl = -v[v <= 0].sum()
    return v[v > 0].sum() / gl if gl else np.nan


def main() -> None:
    argparse.ArgumentParser(description="Стакан (архів Binance bookDepth) і спот на момент входу моделі (експеримент)").parse_args()
    load_dotenv(".env")
    trades = []
    for run in RUNS:
        d = json.load(open(os.path.join("reports", "backtests", run + ".json"), encoding="utf-8"))
        t = pd.DataFrame(d["trades_list"])
        t["band"] = "≥ 20 млн" if not d["meta"].get("max_turnover") else "5–20 млн"
        trades.append(t[t["reason"] != "open"])
    t = pd.concat(trades, ignore_index=True)
    t["T"] = pd.to_datetime(t["entry_ts"], unit="s", utc=True) + B15          # закриття сигнального бару = момент входу
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        sym = dict(conn.execute("""SELECT base_asset, symbol_on_exchange FROM markets
                                   WHERE exchange_name = 'Binance' AND quote_asset = 'USDT'""").fetchall())
    t["sym"] = t["ticker"].map(sym)
    t = t[t["sym"].notna()].reset_index(drop=True)
    print(f"угод: {len(t)} на {t['ticker'].nunique()} монетах", flush=True)

    feats = []
    for n, (s, g) in enumerate(t.groupby("sym"), 1):
        days = sorted({d for T in g["T"] for d in ((T - pd.Timedelta(days=1)).strftime("%Y-%m-%d"), T.strftime("%Y-%m-%d"))})
        w = book_wide(s, days)
        for i, r in g.iterrows():
            feats.append({"i": i, **book_features(w, r["T"]), **spot_features(s, r["T"])})
        if n % 20 == 0:
            print(f"  {n}/{t['sym'].nunique()} монет", flush=True)
    f = pd.DataFrame(feats).set_index("i")
    d = t.join(f)
    d.drop(columns=["T"]).to_csv(OUT, index=False)
    cols = [c for c in ["imb_02", "imb_1", "imb_2", "imb_5", "d_imb_1_15m", "depth_1_rel", "ask_1_rel", "bid_1_rel",
                        "has_spot", "spot_share_4h", "spot_buy_1h", "premium", "premium_chg_4h"] if c in d]
    print(f"\nз даними стакану: {d['imb_1'].notna().sum()} з {len(d)}; зі спотом: {int(d.get('has_spot', pd.Series()).sum())}\n")

    print("Кореляція Спірмена ознаки з результатом угоди (|ρ| > 0.05 — вже щось):")
    for c in cols:
        x = d[[c, "ret"]].dropna()
        if len(x) > 50 and x[c].nunique() > 1:
            print(f"  {c:<16} ρ = {x[c].corr(x['ret'], method='spearman'):+.3f}   (n = {len(x)})")

    print("\nКвінтилі (Q1 — найменші значення, Q5 — найбільші): угод · у плюсі · середня угода · PF")
    for c in cols:
        x = d[[c, "ret"]].dropna()
        if len(x) < 100 or x[c].nunique() < 5:
            continue
        x["q"] = pd.qcut(x[c].rank(method="first"), 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
        row = "  ".join(f"{q}: {len(g)} · {(g['ret'] > 0).mean():.0%} · {g['ret'].mean():+.2%} · {pf(g['ret']):.2f}"
                        for q, g in x.groupby("q", observed=True))
        print(f"  {c:<16} {row}")
    if "has_spot" in d:
        for v, g in d.groupby("has_spot"):
            print(f"\n  спот {'є' if v else 'немає'}: {len(g)} угод, у плюсі {(g['ret'] > 0).mean():.0%}, середня {g['ret'].mean():+.2%}, PF {pf(g['ret']):.2f}")

    # чесна перевірка: поріг за першою половиною угод (за часом) → друга половина
    d = d.sort_values("T")
    half = d["T"].iloc[len(d) // 2]
    first, second = d[d["T"] < half], d[d["T"] >= half]
    print(f"\nЧесна перевірка (поріг за першою половиною до {half:%m-%d}, перевірка — після):")
    print(f"  усі угоди другої половини: {len(second)}, PF {pf(second['ret']):.2f}, середня {second['ret'].mean():+.2%}")
    for c in cols:
        x1 = first[[c, "ret"]].dropna()
        if len(x1) < 100 or x1[c].nunique() < 5:
            continue
        best = None
        for q in (0.2, 0.4, 0.6, 0.8):
            thr = x1[c].quantile(q)
            for side in ("≥", "≤"):
                keep = x1[x1[c] >= thr] if side == "≥" else x1[x1[c] <= thr]
                if len(keep) >= 50 and (best is None or pf(keep["ret"]) > best[0]):
                    best = (pf(keep["ret"]), side, thr)
        if best is None:
            continue
        _, side, thr = best
        x2 = second[[c, "ret"]].dropna()
        keep = x2[x2[c] >= thr] if side == "≥" else x2[x2[c] <= thr]
        print(f"  {c:<16} правило {side} {thr:+.4f}: на першій PF {best[0]:.2f} → на другій {len(keep)} угод, PF {pf(keep['ret']):.2f}, "
              f"середня {keep['ret'].mean():+.2%}")


if __name__ == "__main__":
    main()
