"""
volfc.py — очікувана волатильність (HAR): кнопка «Volatility ▾» і чіп на графіку в дашборді (2026-10-20).

Експеримент experiments/har_vol.py (476 монет, 6 міс., walk-forward): HAR прогнозує рух наступних 4 год стабільно краще
за ATR(15m) — на ~4% (QLIKE), з годиною доби (HAR-S) — на ~7%; у 6 з 6 місяців і в обох половинах монет. Тут те саме
наживо: опис ринку («де чекати руху і якого розміру»), не сигнал моделі. Пояснення для власника — notes/learning.md,
«Лінійна регресія з нуля» і «Прогноз волатильності».

Дані: 5m-свічки Binance усіх крипто-перпів із середнім оборотом ≥ 5 млн $ (без верхньої межі, з BTC / ETH) + тікери
списку скринера (universe), кеш reports/volfc_5m.pkl, останні HIST_DAYS днів (перший запуск ~25 хв, далі — один
короткий запит на монету щогодини, вага 1).
Модель (HAR-S, у логах дисперсії 5m-доходностей):
    log v_H = a + b1·log v_1h + b2·log v_4h + b3·log v_24h + b4·log v_7d + година доби      (H = 1 год і 4 год)
    коефіцієнти — pooled OLS на всіх монетах кешу (лише точки, де майбутні H год уже відомі), щогодини наново;
    прогноз дисперсії = exp(лог + σ²/2) (поправка логнормального зсуву).
Для кожної монети на останній закритій годині:
    exp_H     очікуваний типовий рух (±1σ) за наступні H год, частка ціни: √(v̂ · 12H)
    usual_H   звичний для монети рух за H год — медіана фактичного за останні 30 днів
    vs_H      exp_H ÷ usual_H (у скільки разів більше звичного) → режим Calm / Normal / Active / Hot (REGIMES)
    last_1h   фактичний рух останньої години (у гарячі години HAR занижує — для порівняння)
    atr15     ATR(14) 15m ÷ ціна (як risk.atr; ним рахується стоп моделі)
    turnover_24h  $-оборот Binance за 24 год
Скринер (watcher, щогодини о HH:08) пише reports/volfc.json; дашборд — /api/vol і поле «vol» у /api/ticker.

    python volfc.py               # оновити кеш і порахувати (перший раз ~20 хв)
    python volfc.py --limit 20    # перевірка на 20 монетах
    python volfc.py --show        # лише показати останній результат
"""

import argparse
import json
import os
import pickle
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "reports", "volfc.json")
CACHE = os.path.join(ROOT, "reports", "volfc_5m.pkl")
HIST_DAYS = 31                 # 7 днів вікна HAR + 30 днів для «звичного» руху (перекриваються) + запас
USUAL_DAYS = 30
BAR = pd.Timedelta(minutes=5)
WIN = {"1h": 12, "4h": 48, "24h": 288, "7d": 2016}          # вікна HAR у 5m-барах
HORIZONS = (1, 4)               # години; 24 год не перевіряли в har_vol — не показуємо
MIN_FILL = 0.8                  # частка наявних 5m-доходностей у вікні
ATR_PERIOD = 14
REGIMES = ((0.8, "Calm"), (1.5, "Normal"), (2.5, "Active"), (np.inf, "Hot"))   # межі vs (не перевірені — пропозиція 20.10)
MIN_TURNOVER = 5e6             # нижня межа РОЗРАХУНКУ (оборот за 30 днів або 24 год); показ — vol_config.json (view)
CONFIG_FILE = os.path.join(ROOT, "vol_config.json")    # фільтр таблиці «Volatility ▾»: змінює дашборд, deploy не чіпає, у git немає
CONFIG_DEFAULTS = {"min_turnover_m": 5.0, "max_turnover_m": None}   # $-оборот за 24 год, млн; None — без стелі
CONFIG_MAX = 100_000.0
NOT_CRYPTO = {"USDCUSDT", "FDUSDUSDT", "BTCDOMUSDT", "PAXGUSDT", "XAUTUSDT", "FRAXUSDT", "USTCUSDT"}
WORKERS = 4
WEIGHT_PER_SEC = 8             # market.py бере 15 / с; разом зі скринером і сканером — під лімітом Binance 2400 / хв


# ---------------------------------------------------------------- модель (без мережі — тести tests/test_volfc.py)

def regime(vs: float | None) -> str | None:
    if vs is None or not np.isfinite(vs):
        return None
    return next(name for edge, name in REGIMES if vs < edge)


def hourly_points(c: pd.Series) -> pd.DataFrame:
    """Точки прогнозу (кожна закрита година) з 5m-закриттів однієї монети: логи минулих дисперсій (x1, x4, x24, x168),
    майбутніх (y1, y4 — NaN, де майбутнє ще не настало), година доби. Індекс — момент T (закриття години)."""
    if c.empty:
        return pd.DataFrame()
    grid = pd.date_range(c.index[0], c.index[-1] + BAR, freq=BAR)    # + 1 бар: точка T = закриття останнього бару
    c = c.reindex(grid)
    sq = np.log(c.where(c > 0)).diff() ** 2
    have = sq.notna().astype(float)
    out = {}
    for name, n in WIN.items():                     # бари з відкриттям у [T − w, T): рядок T − 5 хв → shift(1) на T
        s, k = sq.rolling(n, min_periods=1).sum(), have.rolling(n, min_periods=1).sum()
        out["x" + {"1h": "1", "4h": "4", "24h": "24", "7d": "168"}[name]] = (s / k).where(k >= MIN_FILL * n).shift(1)
    for h in HORIZONS:                              # бари з відкриттям у [T, T + h)
        n = 12 * h
        s = sq[::-1].rolling(n, min_periods=1).sum()[::-1]
        k = have[::-1].rolling(n, min_periods=1).sum()[::-1]
        y = (s / k).where(k >= MIN_FILL * n)
        y[y.index > grid[-1] - (n - 1) * BAR] = np.nan               # майбутнє ще не повне
        out[f"y{h}"] = y
    d = pd.DataFrame(out, index=grid)
    d = d[d.index.minute == 0]
    d = np.log(d.where(d > 0))
    d["hour"] = d.index.hour
    return d


def design(d: pd.DataFrame) -> np.ndarray:
    hours = d["hour"].to_numpy()
    return np.column_stack([np.ones(len(d)), d["x1"], d["x4"], d["x24"], d["x168"],
                            *[(hours == h).astype(float) for h in range(1, 24)]])


def fit(points: pd.DataFrame, h: int) -> dict | None:
    """Pooled OLS HAR-S для горизонту h год: коефіцієнти і σ² залишків."""
    d = points.dropna(subset=["x1", "x4", "x24", "x168", f"y{h}"])
    if len(d) < 500:
        return None
    X, y = design(d), d[f"y{h}"].to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return {"beta": beta, "s2": float(np.var(y - X @ beta)), "rows": len(d)}


def forecast(row: pd.DataFrame, m: dict, h: int) -> float | None:
    """Очікуваний типовий рух (частка ціни) за h год з останньої точки."""
    if m is None or row[["x1", "x4", "x24", "x168"]].isna().any(axis=None):
        return None
    lv = float((design(row) @ m["beta"])[0]) + m["s2"] / 2
    return float(np.sqrt(np.exp(lv) * 12 * h))


def usual_move(c: pd.Series, h: int, days: int = USUAL_DAYS) -> float | None:
    """Звичний рух монети за h год: медіана фактичного руху (√ суми квадратів 5m-доходностей) по закритих годинах
    за останні days днів."""
    grid = pd.date_range(c.index[0], c.index[-1], freq=BAR)
    sq = np.log(c.reindex(grid).where(lambda x: x > 0)).diff() ** 2
    n = 12 * h
    s, k = sq.rolling(n, min_periods=1).sum(), sq.notna().rolling(n, min_periods=1).sum()
    mv = np.sqrt(s / k * n).where(k >= MIN_FILL * n)
    mv = mv[(mv.index.minute == 55) & (mv.index > grid[-1] - pd.Timedelta(days=days))]   # вікна, що закінчились на :00
    mv = mv.dropna()
    return float(mv.median()) if len(mv) >= 24 * 7 else None


def atr15(b5: pd.DataFrame) -> float | None:
    b = b5.resample("15min", label="left", closed="left").agg({"h": "max", "l": "min", "c": "last"}).dropna()
    if len(b) < ATR_PERIOD * 2:
        return None
    prev = b["c"].shift(1)
    tr = pd.concat([b["h"] - b["l"], (b["h"] - prev).abs(), (b["l"] - prev).abs()], axis=1).max(axis=1)
    a = tr.ewm(alpha=1 / ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean()
    return float(a.iloc[-1] / b["c"].iloc[-1])


def compute(bars: dict) -> dict:
    """bars: {символ: 5m DataFrame (c, h, l, q), лише закриті бари} → вміст reports/volfc.json."""
    pts = {s: hourly_points(b["c"]) for s, b in bars.items() if len(b) > WIN["7d"]}
    pts = {s: p for s, p in pts.items() if len(p)}
    if not pts:
        return {"error": "немає даних"}
    allp = pd.concat(pts.values(), ignore_index=True)
    models = {h: fit(allp, h) for h in HORIZONS}
    rows = []
    for s, p in pts.items():
        b = bars[s]
        last = p.iloc[[-1]]
        if last.index[0] < b.index[-1] + BAR - pd.Timedelta(hours=1):          # остання повна година не пізніше
            continue
        r = {"symbol": s, "base": s[:-4] if s.endswith("USDT") else s}
        sq = np.log(b["c"]).diff().iloc[-12:] ** 2
        r["last_1h"] = float(np.sqrt(sq.sum())) if sq.notna().sum() >= 10 else None
        r["atr15"] = atr15(b)
        r["turnover_24h"] = float(b["q"].iloc[-288:].sum())
        for h in HORIZONS:
            e, u = forecast(last, models[h], h), usual_move(b["c"], h)
            vs = e / u if e and u else None
            r.update({f"exp_{h}": e, f"usual_{h}": u, f"vs_{h}": vs, f"regime_{h}": regime(vs)})
        if r["exp_4"] is not None:
            rows.append(r)
    clean = lambda v: None if v is None or (isinstance(v, float) and not np.isfinite(v)) else (round(v, 6) if isinstance(v, float) else v)  # noqa: E731
    ts = max(p.index[-1] for p in pts.values())
    out = {"ts": int(ts.timestamp()), "n": len(rows), "regimes": [[e if np.isfinite(e) else None, n] for e, n in REGIMES],
           "model": {str(h): ({"coef": [round(float(x), 4) for x in m["beta"][:5]], "rows": m["rows"]} if m else None)
                     for h, m in models.items()},
           "rows": [{k: clean(v) for k, v in r.items()} for r in sorted(rows, key=lambda r: -(r["vs_4"] or 0))]}
    for h in HORIZONS:
        out[f"hot_{h}"] = sum(1 for r in rows if r[f"regime_{h}"] == "Hot")
    return out


# ---------------------------------------------------------------- дані з Binance

def _klines(sym: str, start: pd.Timestamp, now: pd.Timestamp, lim) -> pd.DataFrame:
    """5m-свічки з start до останнього закритого бару: c, h, l, q ($-оборот)."""
    import exchanges as X
    out, t, end = [], int(start.timestamp() * 1000), int(now.timestamp() * 1000)
    while t < end:
        limit = min(1000, int((end - t) / 300_000) + 1)
        lim.take(1 if limit < 100 else 2 if limit < 500 else 5)
        try:
            page = X.get(X.BINANCE + "/fapi/v1/klines", dict(symbol=sym, interval="5m", startTime=t, limit=limit))
        except Exception:
            break
        if not page:
            break
        out += [(r[0], float(r[4]), float(r[2]), float(r[3]), float(r[7])) for r in page]
        if len(page) < limit:
            break
        t = page[-1][0] + 300_000
    if not out:
        return pd.DataFrame(columns=["c", "h", "l", "q"], dtype="float32")
    df = pd.DataFrame(out, columns=["ms", "c", "h", "l", "q"])
    df.index = pd.to_datetime(df.pop("ms"), unit="ms", utc=True)
    df = df[~df.index.duplicated(keep="last")]
    return df[df.index + BAR <= now].astype("float64")


def load_config() -> dict:
    try:
        with open(CONFIG_FILE, encoding="utf-8") as f:
            return {**CONFIG_DEFAULTS, **{k: v for k, v in json.load(f).items() if k in CONFIG_DEFAULTS}}
    except (OSError, ValueError):
        return dict(CONFIG_DEFAULTS)


def save_config(body: dict) -> dict:
    """Межі обороту таблиці «Volatility ▾» (млн $ за 24 год); max порожній / 0 — без стелі. ValueError — некоректні межі."""
    cfg = load_config()
    if "min_turnover_m" in body:
        x = float(body["min_turnover_m"] or 0)
        if not 0 <= x <= CONFIG_MAX:
            raise ValueError(f"min_turnover_m: 0…{CONFIG_MAX:g}")
        cfg["min_turnover_m"] = x
    if "max_turnover_m" in body:
        v = body["max_turnover_m"]
        x = None if v in (None, "") or float(v) <= 0 else float(v)
        if x is not None and x > CONFIG_MAX:
            raise ValueError(f"max_turnover_m: 0…{CONFIG_MAX:g}")
        cfg["max_turnover_m"] = x
    if cfg["max_turnover_m"] is not None and cfg["max_turnover_m"] <= cfg["min_turnover_m"]:
        raise ValueError("стеля обороту має бути більшою за мінімум")
    with open(CONFIG_FILE + ".tmp", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=1)
    os.replace(CONFIG_FILE + ".tmp", CONFIG_FILE)
    return cfg


def view(data: dict, cfg: dict, keep: set | None = None) -> dict:
    """Таблиця «Volatility ▾» за фільтром обороту за 24 год [min, max] (млн $); тікери списку (keep) — завжди. Чіп на графіку
    фільтр не чіпає (volfc.info). Лічильники Hot — по показаних рядках."""
    if data.get("error"):
        return {**data, "config": cfg}
    lo = (cfg.get("min_turnover_m") or 0) * 1e6
    hi = cfg["max_turnover_m"] * 1e6 if cfg.get("max_turnover_m") else None
    keep = keep or set()
    rows = [r for r in data.get("rows") or []
            if r["base"] in keep or ((r.get("turnover_24h") or 0) >= lo and (hi is None or (r.get("turnover_24h") or 0) <= hi))]
    out = {**data, "rows": rows, "n_all": data.get("n"), "n": len(rows), "config": cfg}
    for h in HORIZONS:
        out[f"hot_{h}"] = sum(1 for r in rows if r.get(f"regime_{h}") == "Hot")
    return out


def universe() -> list[str]:
    """Монети прогнозу: крипто-перпи USDT Binance, у яких середній оборот за 30 днів АБО оборот за 24 год ≥ нижньої межі
    (менша з MIN_TURNOVER і мінімуму в vol_config.json), — БЕЗ верхньої межі (власник 20.10: Q уночі мав оборот > 1 млрд
    і був активний; стеля — лише фільтр показу, view), разом з BTC / ETH; без стейблів і золота. Плюс усі тікери списку
    скринера (watchlist + авто). Акції / товари (TRADIFI) не беремо: лише contractType PERPETUAL."""
    import exchanges as X
    import scanner
    floor = min(MIN_TURNOVER, (load_config().get("min_turnover_m") or 0) * 1e6)
    info = X.get(X.BINANCE + "/fapi/v1/exchangeInfo", {})
    perps = {s["symbol"] for s in info["symbols"] if s.get("contractType") == "PERPETUAL" and s.get("quoteAsset") == "USDT"
             and s.get("status") == "TRADING" and s["symbol"] not in NOT_CRYPTO}
    day = {t["symbol"]: float(t.get("quoteVolume") or 0) for t in X.get(X.BINANCE + "/fapi/v1/ticker/24hr", {})}
    avg = (scanner._read_json(scanner.AVG_FILE, {}) or {}).get("values") or {}
    syms = {s for s in perps if max(avg.get(s) or 0, day.get(s) or 0) >= floor}
    try:
        from screener import cycle_bases
        syms |= {b + "USDT" for b in cycle_bases() if b + "USDT" in perps}
    except Exception as e:
        print(f"  ! vol: список скринера: {type(e).__name__}: {e}")
    return sorted(syms)


def load_cache() -> dict:
    try:
        with open(CACHE, "rb") as f:
            return pickle.load(f)
    except (OSError, EOFError, pickle.UnpicklingError):
        return {}


def update(limit: int = 0, log=print) -> dict:
    """Дотягнути 5m-свічки ліквідних альтів, порахувати прогноз → reports/volfc.json."""
    import market
    now = pd.Timestamp(datetime.now(timezone.utc))
    start = (now - pd.Timedelta(days=HIST_DAYS)).floor("h")
    syms = universe()[: limit or None]
    cache = load_cache()
    lim = market._Limiter(WEIGHT_PER_SEC)

    def job(sym):
        have = cache.get(sym)
        frm = have.index.max() + BAR if have is not None and len(have) else start
        return sym, _klines(sym, max(frm, start), now, lim)

    t0 = time.time()
    with ThreadPoolExecutor(WORKERS) as pool:
        got = dict(pool.map(job, syms))
    new = {}
    for s in syms:
        parts = [x for x in (cache.get(s), got.get(s)) if x is not None and len(x)]
        if parts:
            df = pd.concat(parts)
            df = df[~df.index.duplicated(keep="last")].sort_index()
            new[s] = df[df.index >= start]
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE + ".tmp", "wb") as f:
        pickle.dump(new, f)
    os.replace(CACHE + ".tmp", CACHE)
    out = compute(new)
    with open(OUT + ".tmp", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    os.replace(OUT + ".tmp", OUT)
    log(f"  vol: {out.get('n', 0)} монет, Hot 4H {out.get('hot_4', '—')}, {time.time() - t0:.0f} с")
    return out


def watcher(minute: int = 8) -> threading.Thread:
    """Для screener.py --loop: щогодини о HH:08 (і одразу при старті) у власному потоці, з nice (2 vCPU на сервері);
    :08 — між 15m-циклами скринера (:02 / :17) і після market.py (:01)."""
    def run():
        try:
            os.nice(5)
        except (AttributeError, OSError):
            pass
        while True:
            try:
                update()
            except Exception as e:
                print(f"  ! vol: {type(e).__name__}: {e}")
            now = datetime.now(timezone.utc)
            nxt = now.replace(minute=minute, second=30, microsecond=0)
            if nxt <= now:
                nxt += timedelta(hours=1)
            time.sleep((nxt - now).total_seconds())
    th = threading.Thread(target=run, daemon=True, name="volfc")
    th.start()
    return th


# ---------------------------------------------------------------- читання: дашборд

def load() -> dict:
    try:
        with open(OUT, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {"error": "немає даних (volfc.py ще не порахував прогноз)"}


_cache: dict = {}


def info(base: str) -> dict | None:
    """Чіп «Exp 4H» на графіку тікера: прогноз 1H / 4H, режим, останній рух. Файл перечитується лише після оновлення."""
    try:
        mt = os.path.getmtime(OUT)
    except OSError:
        return None
    if _cache.get("mt") != mt:
        d = load()
        _cache.update(mt=mt, ts=d.get("ts"), rows={r["base"]: r for r in d.get("rows") or []})
    r = _cache["rows"].get(base)
    return {**r, "ts": _cache["ts"]} if r else None


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Очікувана волатильність (HAR-S з 5m Binance) ліквідних альтів → reports/volfc.json")
    ap.add_argument("--limit", type=int, default=0, help="лише перші N монет (перевірка)")
    ap.add_argument("--show", action="store_true", help="лише показати останній результат, без оновлення")
    args = ap.parse_args()
    out = load() if args.show else update(args.limit)
    if out.get("error"):
        print(out["error"])
        return
    print(f"{datetime.fromtimestamp(out['ts'], timezone.utc):%Y-%m-%d %H:%M} UTC · {out['n']} монет · Hot 4H {out['hot_4']}"
          f" · коефіцієнти 4H (const, 1h, 4h, 24h, 7d): {out['model']['4']['coef'] if out['model']['4'] else '—'}")
    for r in out["rows"][:15]:
        print(f"  {r['base']:<10} exp 4H ±{r['exp_4']:.2%}  ×{r['vs_4']:.1f} {r['regime_4']:<6}  last 1H ±{(r['last_1h'] or 0):.2%}"
              f"  ATR15 {(r['atr15'] or 0):.2%}  24H ${r['turnover_24h'] / 1e6:.1f}M")


if __name__ == "__main__":
    main()
