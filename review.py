"""
review.py — сліпий перегляд сигналів моделі (meta-labeling): власник вирішує «зайшов би / пропустив» лише за тим,
що видно ДО входу, а результат угоди ховається до підсумку.

Навіщо: бектест показав, що вхід моделі на нових монетах не кращий за випадковий (experiments/in_play.py), а правила
виходу цього не виправляють (experiments/exit_structure.py). Власник же каже про найбільші стопи дня «туди б і не
зайшов» — перевага, схоже, у його виборі входу. Щоб перевірити це чесно (без знання результату) і згодом навчити
другу модель-фільтр, рішення збираються наосліп.

    пул      — угоди 15m-контуру (Alert) поза навчанням: прогони backtest.py + журнал скринера; лише закриті;
               перемішані з фіксованим seed (reports/review_pool.json; перебудовується кнопкою або --rebuild)
    рішення  — reviews.csv (джерело істини, як pumps.csv): id, ticker, entry_ts, decision (take / skip / unsure),
               setup (breakout / pullback / other), reasons (теги через «;»), stop_price, note, ticker_seen, reviewed_at
    графік   — бари лише до закриття сигнального бару (сервер обрізає, майбутнього в браузері немає)
    підсумок — результат угод «зайшов би» проти всіх сигналів (вихід як зараз; і зі стопом власника, якщо поставлено)

    python review.py --rebuild      # перебудувати пул (нові прогони бектесту, нові угоди журналу)
    сторінка — http://127.0.0.1:8765/review (сервер label.py)
"""

import argparse
import csv
import glob
import json
import os
import random
import threading
from datetime import datetime, timezone

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
REVIEWS_PATH = os.path.join(ROOT, "reviews.csv")
POOL_PATH = os.path.join(ROOT, "reports", "review_pool.json")
COLUMNS = ["id", "ticker", "entry_ts", "decision", "setup", "reasons", "stop_price", "note", "ticker_seen", "reviewed_at"]
DECISIONS = ("take", "skip", "unsure")
SETUPS = ("breakout", "pullback", "other")
# теги причин — зі слів власника (розбір MOVR / US 2026-10-01) + типові; вільний текст — у нотатці
REASONS = {
    "skip": ["на хаях без пробою", "немає продовження", "червоні свічки, OI виходить", "вже злетів", "тінь / злив у свічці",
             "смикана ціна / тонко", "OI не підтверджує", "мертва монета"],
    "take": ["пробій рейнджу", "підбір після зливу", "OI і ціна разом", "продовження після паузи", "сильний обсяг"],
}
BAR = 900                      # 15m-бар моделі, с
FEE = 0.001
_lock = threading.Lock()


# ---------------------------------------------------------------- пул сигналів

def build_pool(conn) -> list[dict]:
    """Угоди Alert поза навчанням: усі прогони бектесту + журнал скринера (дублікати — за тікером і входом)."""
    import journal as J
    import train
    from tiers import TIER_MODEL
    items: dict[str, dict] = {}
    for path in sorted(glob.glob(os.path.join(ROOT, "reports", "backtests", "*.json"))):
        if os.path.basename(path).startswith("_"):
            continue
        d = json.load(open(path, encoding="utf-8"))
        if d.get("meta", {}).get("entry", "alert") != "alert":
            continue
        for t in d.get("trades_list", []):
            if t.get("reason") == "open" or t.get("exit_ts") is None:
                continue
            key = f"{t['ticker']}|{int(t['entry_ts'])}"
            items.setdefault(key, {"id": key, "ticker": t["ticker"], "entry_ts": int(t["entry_ts"]), "entry": t["entry"],
                                   "exit_ts": int(t["exit_ts"]), "ret": t["ret"], "reason": t["reason"],
                                   "source": "backtest"})
    m15 = train.load_model(train.M15_NAME)
    trained = set(m15.get("tickers", []))
    tr = J.build_trades(conn, TIER_MODEL, m15, J.DEFAULT_RULE)
    for r in tr.itertuples():
        if r.open or r.ticker in trained or r.kind == "in-sample" or r.result_pct is None or pd.isna(r.result_pct):
            continue
        ts = int(pd.Timestamp(r.entry_ts).timestamp())
        key = f"{r.ticker}|{ts}"
        items.setdefault(key, {"id": key, "ticker": r.ticker, "entry_ts": ts, "entry": float(r.entry),
                               "exit_ts": int(pd.Timestamp(r.exit_ts).timestamp()), "ret": float(r.result_pct) - FEE,
                               "reason": r.reason, "source": "journal"})
    pool = list(items.values())
    random.Random(42).shuffle(pool)       # випадковий порядок: без «серій» одного тікера і без хронологічного якоря
    os.makedirs(os.path.dirname(POOL_PATH), exist_ok=True)
    json.dump({"built": datetime.now(timezone.utc).isoformat(timespec="seconds"), "items": pool},
              open(POOL_PATH, "w", encoding="utf-8"), ensure_ascii=False)
    return pool


def load_pool(conn=None) -> list[dict]:
    if os.path.exists(POOL_PATH):
        return json.load(open(POOL_PATH, encoding="utf-8"))["items"]
    if conn is None:
        return []
    return build_pool(conn)


# ---------------------------------------------------------------- рішення

def read_reviews() -> dict[str, dict]:
    if not os.path.exists(REVIEWS_PATH):
        return {}
    with open(REVIEWS_PATH, encoding="utf-8", newline="") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def save_review(body: dict) -> dict:
    rid = str(body.get("id", ""))
    decision = body.get("decision")
    if decision not in DECISIONS:
        raise ValueError("рішення: take / skip / unsure")
    setup = body.get("setup") or ""
    if setup and setup not in SETUPS:
        raise ValueError("сетап: breakout / pullback / other")
    stop = body.get("stop_price")
    row = {"id": rid, "ticker": rid.split("|")[0], "entry_ts": rid.split("|")[1] if "|" in rid else "",
           "decision": decision, "setup": setup,
           "reasons": ";".join(str(x).replace(";", ",") for x in body.get("reasons") or []),
           "stop_price": f"{float(stop):.10g}" if stop not in (None, "") else "",
           "note": str(body.get("note") or "").replace("\n", " ")[:300],
           "ticker_seen": "1" if body.get("ticker_seen") else "",
           "reviewed_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")}
    with _lock:
        rows = read_reviews()
        rows[rid] = row
        with open(REVIEWS_PATH, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(rows.values())
    return row


def public(item: dict) -> dict:
    """Що бачить сторінка до рішення: без результату, виходу і дати (дата — лише якщо власник відкрив тікер)."""
    return {k: item[k] for k in ("id", "ticker", "entry_ts", "entry")}


def next_item(conn, after: str | None = None) -> dict:
    pool, done = load_pool(conn), read_reviews()
    left = [x for x in pool if x["id"] not in done]
    if after:                                   # «пропустити поки» — наступний після вказаного
        ids = [x["id"] for x in left]
        if after in ids:
            left = left[ids.index(after) + 1:] + left[:ids.index(after)]
    item = left[0] if left else None
    return {"item": public(item) if item else None, "done": len(done), "total": len(pool),
            "review": None, "reasons": REASONS}


def upcoming(conn, n: int = 5) -> list[dict]:
    """Наступні n непереглянутих — для фонового дозавантаження 5m, поки власник дивиться поточний."""
    done = read_reviews()
    return [x for x in load_pool(conn) if x["id"] not in done][1:n + 1]


def get_item(conn, rid: str) -> dict:
    pool = {x["id"]: x for x in load_pool(conn)}
    if rid not in pool:
        raise ValueError("сигналу немає в пулі")
    done = read_reviews()
    return {"item": public(pool[rid]), "done": len(done), "total": len(pool), "review": done.get(rid), "reasons": REASONS}


# ---------------------------------------------------------------- графік до моменту входу

def bars(conn, ticker: str, entry_ts: int, tf: str) -> dict:
    """Бари тікера до закриття сигнального 15m-бару (entry_ts + 15 хв) — і нічого після. 5m, яких немає в базі
    (монети з бектесту), дозавантажуються з бірж (Binance/Bybit тримають 5m ~30 днів)."""
    import fetch_exchange
    from features import INTRADAY_EXCHANGES, aggregate, load_raw
    cutoff = pd.Timestamp(entry_ts + BAR, unit="s", tz="UTC")
    back = {"5m": pd.Timedelta(days=2), "15m": pd.Timedelta(days=5), "1h": pd.Timedelta(days=14)}[tf]
    interval, freq, step = {"5m": ("5min", "5min", 300), "15m": ("15min", "15min", 900), "1h": ("15min", "15min", 900)}[tf]

    def load():
        raw = load_raw(conn, [ticker], interval, since=cutoff - back - pd.Timedelta(hours=2))
        return aggregate(raw, ticker, freq)
    h = load()
    if tf == "5m" and (h.empty or h.index.max() + pd.Timedelta(seconds=step) < cutoff):
        age = (pd.Timestamp(datetime.now(timezone.utc)) - cutoff) / pd.Timedelta(days=1)
        if age < 29:
            # лише вікно графіка (~2.5 дні до сигналу), а не все від сьогодні — кілька секунд замість ~30
            fetch_exchange.collect(conn, [ticker], ["5m"], 2.6, log=lambda m: None,
                                   exchanges=list(INTRADAY_EXCHANGES), until=cutoff + pd.Timedelta(minutes=30))
            h = load()
    if h.empty:
        raise ValueError(f"немає {tf}-даних для цього сигналу")
    h = h[h.index + pd.Timedelta(seconds=step) <= cutoff]            # лише закриті до моменту входу
    if tf == "1h":
        # година, що ще не закрилась на момент входу, — частковою свічкою (як її бачив би трейдер)
        h = h.resample("1h").agg({"open": "first", "high": "max", "low": "min", "price": "last", "volume": "sum",
                                  "oi": "last"}).dropna(subset=["price"])
    h = h[h.index >= cutoff - back]
    t = [int(x.timestamp()) for x in h.index]
    col = lambda c: [None if pd.isna(v) else float(v) for v in h[c]]  # noqa: E731
    return {"t": t, "o": col("open"), "h": col("high"), "l": col("low"), "c": col("price"), "v": col("volume"),
            "oi": col("oi") if "oi" in h else [None] * len(t), "cutoff": int(cutoff.timestamp()), "tf": tf}


# ---------------------------------------------------------------- підсумок

def _stats(v: pd.Series) -> dict:
    v = v.dropna().sort_values()
    if v.empty:
        return {"n": 0}
    gw, gl = float(v[v > 0].sum()), float(-v[v <= 0].sum())
    return {"n": int(len(v)), "win": float((v > 0).mean()), "mean": float(v.mean()), "median": float(v.median()),
            "sum": float(v.sum()), "sum_wo_top3": float(v.iloc[:-3].sum()) if len(v) > 3 else float(v.sum()),
            "pf": gw / gl if gl else None}


def owner_stop_ret(conn, item: dict, stop: float) -> float:
    """Результат угоди зі стопом власника: якщо до виходу моделі ціна торкнулась стопу — вихід по стопу."""
    from features import INTERVAL_15M, aggregate, load_raw
    start = pd.Timestamp(item["entry_ts"] + BAR, unit="s", tz="UTC")
    end = pd.Timestamp(item["exit_ts"], unit="s", tz="UTC")
    h = aggregate(load_raw(conn, [item["ticker"]], INTERVAL_15M, since=start - pd.Timedelta(hours=1)), item["ticker"], "15min")
    w = h[(h.index >= start) & (h.index <= end)]
    if stop < item["entry"] and len(w) and (w["low"] <= stop).any():
        return stop / item["entry"] - 1 - FEE
    return item["ret"]


def summary(conn) -> dict:
    pool = {x["id"]: x for x in load_pool(conn)}
    rv = [r | {"item": pool[r["id"]]} for r in read_reviews().values() if r["id"] in pool]
    if not rv:
        return {"n": 0}
    df = pd.DataFrame([{"decision": r["decision"], "setup": r["setup"], "reasons": r["reasons"],
                        "ret": r["item"]["ret"], "stop": float(r["stop_price"]) if r["stop_price"] else None,
                        "item": r["item"], "ticker_seen": bool(r["ticker_seen"])} for r in rv])
    take = df[df["decision"] == "take"]
    own = take.apply(lambda r: owner_stop_ret(conn, r["item"], r["stop"]) if r["stop"] else r["ret"], axis=1) \
        if len(take) else pd.Series(dtype=float)
    by_reason = []
    for dec in ("take", "skip"):
        g = df[df["decision"] == dec]
        tags = g.assign(tag=g["reasons"].str.split(";")).explode("tag")
        tags = tags[tags["tag"].fillna("") != ""]
        for tag, gg in tags.groupby("tag"):
            by_reason.append({"decision": dec, "tag": tag, **_stats(gg["ret"])})
    by_setup = [{"setup": s or "—", **_stats(g["ret"])} for s, g in take.groupby(take["setup"].fillna(""))] if len(take) else []
    return {
        "n": int(len(df)), "counts": df["decision"].value_counts().to_dict(),
        "all": _stats(df[df["decision"] != "unsure"]["ret"]),
        "take": _stats(take["ret"]), "take_own_stop": _stats(own), "skip": _stats(df[df["decision"] == "skip"]["ret"]),
        "with_stop": int(take["stop"].notna().sum()) if len(take) else 0,
        "by_setup": by_setup, "by_reason": sorted(by_reason, key=lambda x: (x["decision"], -x["n"])),
        "ticker_seen": int(df["ticker_seen"].sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Сліпий перегляд сигналів: пул угод і підсумок (сторінка — label.py /review)")
    ap.add_argument("--rebuild", action="store_true", help="перебудувати пул сигналів (бектест + журнал)")
    ap.add_argument("--summary", action="store_true", help="надрукувати підсумок рішень")
    args = ap.parse_args()
    import psycopg
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        if args.rebuild or not os.path.exists(POOL_PATH):
            pool = build_pool(conn)
            print(f"пул: {len(pool)} сигналів ({sum(x['source'] == 'backtest' for x in pool)} з бектесту, "
                  f"{sum(x['source'] == 'journal' for x in pool)} з журналу) → {POOL_PATH}")
        if args.summary:
            print(json.dumps(summary(conn), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
