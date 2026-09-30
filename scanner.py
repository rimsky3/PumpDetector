"""
scanner.py — сканер ринку: які USDT-перпи Binance «прокидаються» і мають потрапити в скринер автоматично.

Воронка: весь ринок (≈500 перпів) → фільтр → авто-список → скринер (дані + модель) → Alert.
Фільтр навмисно м'якший за модель: його задача — не пропустити памп; відсіювати зайве — робота моделі.

Правило (пороги — у scan_config.json, набори Суворо / Баланс / Широко / Свій):
    ліквідність:  оборот за 24 год ≥ min_turnover  і  OI ≤ max_oi (великі монети — не наш скоуп, урок ZEC)
    прокидання:   обсяг за 4 год ≥ vol_x × норми (медіана 4-годинного обсягу за попередні 7 днів) і ціна за 4 год ≥ 0
                  АБО  OI за 4 год ≥ +oi_pct %  і  ціна за 4 год ≥ +price_pct %
Скан — на закритих 15m-барах (кожні 15 хв); --recall — на годинних даних з бази, вікна ті самі.
Тікер, що пройшов, живе в авто-списку ttl_hours після останнього спрацювання (або сигналу моделі), максимум max_auto.

Дані — публічний API Binance (ключ не потрібен, Coinalyze не витрачаємо): 1 запит на весь ринок
+ 2 на кожен перп, що проходить найм'якшу межу обороту (15m-свічки за 7.5 дня і OI за 4 год). Кожні 15 хв (screener.py).

    python scanner.py              # один скан: хто проходить зараз (авто-список НЕ змінює)
    python scanner.py --apply      # те саме + оновити авто-список (так робить screener.py щогодини)
    python scanner.py --recall     # як кожен набір порогів спрацював би на розмічених пампах (з бази)
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import exchanges as X

ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(ROOT, "scan_config.json")
AUTO_FILE = os.path.join(ROOT, "reports", "auto_watchlist.json")
LAST_FILE = os.path.join(ROOT, "reports", "scan_last.json")
RECALL_FILE = os.path.join(ROOT, "reports", "scan_recall.json")   # --recall: результат наборів на пампах
WORKERS = 6
SCAN_TF = "15m"                    # скан кожні 15 хв — на закритих 15m-барах (на годинних 3 скани з 4 бачили б те саме)
BARS_PER_HOUR = 4
KLINE_BARS = 7 * 24 * BARS_PER_HOUR + 24   # 7 днів норми + 6 год
RECALL_BARS = 8 * 24                  # --recall: годинні бари з бази, 8 днів
PRESETS = {
    "strict":  {"min_turnover": 30e6, "vol_x": 4.0, "oi_pct": 15.0, "price_pct": 3.0},
    "balance": {"min_turnover": 20e6, "vol_x": 3.0, "oi_pct": 10.0, "price_pct": 2.0},
    "wide":    {"min_turnover": 5e6,  "vol_x": 2.0, "oi_pct": 6.0,  "price_pct": 1.0},
}
PRESET_NAMES = {"strict": "Суворо", "balance": "Баланс", "wide": "Широко", "custom": "Свій"}
DEFAULT_CONFIG = {"enabled": True, "preset": "balance", "custom": dict(PRESETS["balance"]),
                  "max_oi": 500e6, "max_auto": 30, "ttl_hours": 48}
PARAM_LIMITS = {"min_turnover": (1e6, 1e9), "vol_x": (1.0, 20.0), "oi_pct": (1.0, 100.0), "price_pct": (0.0, 50.0),
                "max_oi": (10e6, 1e11), "max_auto": (0, 100), "ttl_hours": (1, 24 * 7)}


# ---------------------------------------------------------------- конфіг і авто-список

def _read_json(path: str, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def _write_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1, default=str)
    os.replace(tmp, path)            # атомарно: скринер і дашборд не прочитають напівзаписаний файл


def load_config() -> dict:
    return {**DEFAULT_CONFIG, **_read_json(CONFIG_FILE, {})}


def save_config(new: dict) -> dict:
    """Перевірити і зберегти налаштування з дашборду. Діють з наступного скану."""
    cfg = load_config()
    if "preset" in new:
        if new["preset"] not in PRESET_NAMES:
            raise ValueError(f"невідомий набір: {new['preset']}")
        cfg["preset"] = new["preset"]
    if "enabled" in new:
        cfg["enabled"] = bool(new["enabled"])
    for k, v in (new.get("custom") or {}).items():
        if k in PRESETS["balance"]:
            cfg["custom"][k] = _clip(k, v)
    for k in ("max_oi", "max_auto", "ttl_hours"):
        if k in new:
            cfg[k] = _clip(k, new[k])
    _write_json(CONFIG_FILE, cfg)
    return cfg


def _clip(k: str, v) -> float:
    lo, hi = PARAM_LIMITS[k]
    try:
        v = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{k}: потрібне число")
    return min(hi, max(lo, v))


def params(cfg: dict) -> dict:
    """Пороги активного набору + спільні межі."""
    p = cfg["custom"] if cfg["preset"] == "custom" else PRESETS[cfg["preset"]]
    return {**p, "max_oi": cfg["max_oi"]}


def load_auto() -> dict:
    return _read_json(AUTO_FILE, {})


def active_auto(now: datetime | None = None) -> dict:
    """Авто-тікери, чий термін ще не минув: {base: {added, last_hit, expires, reason, ...}}."""
    now = now or datetime.now(timezone.utc)
    return {b: r for b, r in load_auto().items() if pd.Timestamp(r["expires"]) > pd.Timestamp(now)}


def touch(bases: list[str], reason: str = "сигнал моделі") -> None:
    """Скринер: авто-тікер із сигналом моделі не випадає, доки сигнал живий — продовжуємо термін."""
    auto = load_auto()
    ttl = timedelta(hours=load_config()["ttl_hours"])
    now = datetime.now(timezone.utc)
    changed = False
    for b in bases:
        if b in auto:
            auto[b].update(expires=(now + ttl).isoformat(timespec="minutes"), held=reason)
            changed = True
    if changed:
        _write_json(AUTO_FILE, auto)


# ---------------------------------------------------------------- метрики з Binance

def universe() -> list[dict]:
    """Усі USDT-перпи Binance, що торгуються, з оборотом за 24 год (два запити на весь ринок)."""
    info = X.get(X.BINANCE + "/fapi/v1/exchangeInfo", {})
    day = {t["symbol"]: t for t in X.get(X.BINANCE + "/fapi/v1/ticker/24hr", {})}
    out = []
    for s in info["symbols"]:
        if s.get("contractType") != "PERPETUAL" or s.get("quoteAsset") != "USDT" or s.get("status") != "TRADING":
            continue
        t = day.get(s["symbol"], {})
        out.append({"symbol": s["symbol"], "base": s["baseAsset"], "turnover": float(t.get("quoteVolume") or 0)})
    return out


def metrics_from_bars(h: pd.DataFrame, bph: int = 1) -> dict | None:
    """Метрики фільтра з ряду барів (колонки volume у USD, price, oi у USD; індекс — час відкриття бару;
    bph — барів на годину: 4 для живого 15m-скану, 1 для --recall на годинних даних з бази).
    Спільна функція для живого скану і перевірки на історії, щоб логіка не розходилась."""
    h = h.dropna(subset=["volume", "price"])
    w = 4 * bph                                  # вікно «прокидання» — 4 години
    if len(h) < 24 * 3 * bph:
        return None
    vol, px = h["volume"], h["price"]
    v4 = vol.rolling(w).sum()
    # норма — медіана 4-годинного обсягу за 7 днів ДО останніх 4 год (щоб сплеск не рахував сам себе)
    norm = v4.iloc[:-w].tail(7 * 24 * bph).median()
    oi = h["oi"].dropna() if "oi" in h else pd.Series(dtype=float)
    return {
        "turnover": float(vol.tail(24 * bph).sum()),
        "vol_x": float(v4.iloc[-1] / norm) if norm and norm > 0 else None,
        "price_pct": float((px.iloc[-1] / px.iloc[-1 - w] - 1) * 100),
        "oi_pct": float((oi.iloc[-1] / oi.iloc[-1 - w] - 1) * 100) if len(oi) > w and oi.iloc[-1 - w] > 0 else None,
        "oi": float(oi.iloc[-1]) if len(oi) else None,
    }


def entries_on_history(h: pd.DataFrame, p: dict, ttl_hours: float, bph: int = 4) -> list[pd.Timestamp]:
    """Коли сканер з порогами p додав би тікер у список, якби працював увесь час (для міток на графіку дашборду).
    h — ряд барів (volume у USD, price, oi у USD) з тими ж біржами, що й скан (Binance). Та сама логіка, що
    metrics_from_bars() + passes(), але векторно по всьому ряду. Повертає моменти входу (закриття бару);
    повторний вхід — лише після того, як тікер випав би зі списку (ttl_hours без спрацювань)."""
    if h.empty:
        return []
    w = 4 * bph
    vol, px, oi = h["volume"], h["price"], h["oi"]
    v4 = vol.rolling(w).sum()
    norm = v4.shift(w).rolling(7 * 24 * bph, min_periods=3 * 24 * bph - w).median()
    turnover = vol.rolling(24 * bph).sum()
    price_pct = (px / px.shift(w) - 1) * 100
    oi_pct = (oi / oi.shift(w) - 1) * 100
    ok = (turnover >= p["min_turnover"]) & ~(oi > p["max_oi"]) & (
        ((v4 / norm >= p["vol_x"]) & (price_pct >= 0)) | ((oi_pct >= p["oi_pct"]) & (price_pct >= p["price_pct"])))
    bar = pd.Timedelta(hours=1) / bph
    out, expires = [], None
    for ts in ok[ok.fillna(False)].index:
        t = ts + bar
        if expires is None or t > expires:
            out.append(t)
        expires = t + pd.Timedelta(hours=ttl_hours)
    return out


def passes(m: dict, p: dict) -> tuple[bool, str]:
    """Чи проходить тікер фільтр і чому (текст причини — для дашборду)."""
    if m.get("turnover") is None or m["turnover"] < p["min_turnover"]:
        return False, ""
    if m.get("oi") is not None and m["oi"] > p["max_oi"]:
        return False, ""
    why = []
    if m.get("vol_x") is not None and m["vol_x"] >= p["vol_x"] and m["price_pct"] >= 0:
        why.append(f"обсяг ×{m['vol_x']:.1f}")
    if m.get("oi_pct") is not None and m["oi_pct"] >= p["oi_pct"] and m["price_pct"] >= p["price_pct"]:
        why.append(f"OI {m['oi_pct']:+.0f}%")
    if not why:
        return False, ""
    why.append(f"ціна {m['price_pct']:+.1f}% за 4 год · оборот {m['turnover'] / 1e6:.0f} млн $")
    return True, " · ".join(why)


def _symbol_metrics(u: dict) -> dict | None:
    end = int(datetime.now(timezone.utc).timestamp() * 1000)
    try:
        k = X.get(X.BINANCE + "/fapi/v1/klines", dict(symbol=u["symbol"], interval=SCAN_TF, limit=KLINE_BARS + 1))
        oi = X.get(X.BINANCE + "/futures/data/openInterestHist",
                   dict(symbol=u["symbol"], period=SCAN_TF, limit=4 * BARS_PER_HOUR + 2))
    except Exception:
        return None
    # останню свічку (ще не закриту) відкидаємо; quote volume (USDT) — r[7]
    k = [r for r in k if int(r[6]) < end]
    h = pd.DataFrame({"volume": [float(r[7]) for r in k], "price": [float(r[4]) for r in k]},
                     index=pd.to_datetime([int(r[0]) for r in k], unit="ms", utc=True))
    # OI: знімок на T = закриття попереднього бару (exchanges.py) → зсуваємо на бар назад
    s = pd.Series([float(r["sumOpenInterestValue"]) for r in oi],
                  index=pd.to_datetime([int(r["timestamp"]) for r in oi], unit="ms", utc=True) - X.H / BARS_PER_HOUR)
    h["oi"] = s.reindex(h.index)
    m = metrics_from_bars(h, BARS_PER_HOUR)
    if m:
        m.update(symbol=u["symbol"], base=u["base"], last_bar=h.index[-1].isoformat())
    return m


def scan(log=print) -> dict:
    """Метрики всіх перпів, що проходять найм'якшу межу обороту (щоб дашборд міг показати результат будь-якого набору)."""
    t0 = datetime.now(timezone.utc)
    cfg = load_config()
    floor = min([p["min_turnover"] for p in PRESETS.values()] + [cfg["custom"]["min_turnover"]])
    uni = universe()
    cand = [u for u in uni if u["turnover"] >= floor * 0.8]      # 24h ticker — ковзне вікно, беремо із запасом
    with ThreadPoolExecutor(WORKERS) as ex:
        rows = [m for m in ex.map(_symbol_metrics, cand) if m]
    res = {"at": t0.isoformat(timespec="seconds"), "universe": len(uni), "checked": len(rows),
           "seconds": round((datetime.now(timezone.utc) - t0).total_seconds(), 1), "rows": rows}
    _write_json(LAST_FILE, res)
    log(f"скан: {len(uni)} перпів, перевірено {len(rows)} (оборот від {floor / 1e6:.0f} млн $) за {res['seconds']} с")
    return res


def hits(last: dict, p: dict, exclude: set[str] = frozenset()) -> list[dict]:
    """Хто проходить набір порогів p — від найсильнішого прокидання."""
    out = []
    for m in last.get("rows", []):
        ok, why = passes(m, p)
        if ok and m["base"] not in exclude:
            out.append({**m, "reason": why})
    return sorted(out, key=lambda m: -(m.get("vol_x") or 0))


def apply(last: dict, manual: set[str], log=print) -> dict:
    """Оновити авто-список: нові спрацювання, продовження, прострочені — геть, не більше max_auto."""
    cfg = load_config()
    now = datetime.now(timezone.utc)
    auto = {b: r for b, r in load_auto().items() if pd.Timestamp(r["expires"]) > pd.Timestamp(now)}
    if not cfg["enabled"]:
        _write_json(AUTO_FILE, auto)
        return auto
    ttl = timedelta(hours=cfg["ttl_hours"])
    new = []
    for m in hits(last, params(cfg), exclude=manual):
        b = m["base"]
        if b not in auto:
            new.append(b)
        auto[b] = {"added": auto.get(b, {}).get("added", now.isoformat(timespec="minutes")),
                   "last_hit": now.isoformat(timespec="minutes"), "expires": (now + ttl).isoformat(timespec="minutes"),
                   "reason": m["reason"], "vol_x": m.get("vol_x"), "oi_pct": m.get("oi_pct"), "turnover": m["turnover"]}
    if len(auto) > cfg["max_auto"]:        # стеля: лишаємо найсвіжіші спрацювання
        keep = sorted(auto, key=lambda b: (auto[b]["last_hit"], auto[b].get("vol_x") or 0), reverse=True)[:int(cfg["max_auto"])]
        auto = {b: auto[b] for b in keep}
        new = [b for b in new if b in auto]
    _write_json(AUTO_FILE, auto)
    log(f"авто-список: {len(auto)}" + (f", нові: {', '.join(new)}" if new else ""))
    return auto


def preview(p: dict, manual: set[str]) -> dict:
    """Для дашборду: скільки пройшло б за останнім сканом з такими порогами."""
    last = _read_json(LAST_FILE, {})
    h = hits(last, p, exclude=manual)
    return {"at": last.get("at"), "checked": last.get("checked", 0), "universe": last.get("universe", 0),
            "count": len(h), "top": [{"base": m["base"], "reason": m["reason"]} for m in h[:8]]}


# ---------------------------------------------------------------- перевірка на розмічених пампах

def recall(conn, p: dict, lookback_h: float = 48) -> dict:
    """Як фільтр спрацював би на розмічених пампах: скільки разів він додав би тікер до старту імпульсу
    (тікер живе в списку ttl_hours, тож рахується спрацювання за lookback_h до старту), скільки — лише під час, скільки — ніколи.
    Дані — годинні з бази (сума бірж Coinalyze), тож оборот трохи більший, ніж лише Binance."""
    from features import aggregate, load_pumps, load_raw
    pumps = load_pumps(conn)
    pumps = pumps[pumps["pattern_type"] == "pump"]
    early = during = never = 0
    for base in sorted(pumps["base_asset"].unique()):
        h = aggregate(load_raw(conn, [base]), base)
        if "ts" in h.columns:
            h = h.set_index("ts")
        h = h.sort_index()
        for r in pumps[pumps["base_asset"] == base].itertuples():
            fired = None
            for t in h.index[(h.index >= r.start_ts - pd.Timedelta(hours=lookback_h)) & (h.index < r.end_ts)]:
                m = metrics_from_bars(h.loc[:t].tail(RECALL_BARS), 1)
                if m and passes(m, p)[0]:
                    fired = t + X.H              # бар закрився — тоді скан його бачить
                    break
            if fired is None:
                never += 1
            elif fired <= r.start_ts:
                early += 1
            else:
                during += 1
    return {"pumps": early + during + never, "early": early, "during": during, "never": never}


def recall_key(p: dict, ttl) -> str:
    """Підпис порогів: дашборд показує «вчасно N/56» лише якщо --recall рахували саме з такими порогами."""
    return json.dumps({k: round(float(p[k]), 4) for k in sorted(p)} | {"ttl": float(ttl)}, sort_keys=True)


def recall_for(p: dict, ttl) -> dict | None:
    key = recall_key(p, ttl)
    for r in _read_json(RECALL_FILE, {}).get("sets", {}).values():
        if r.get("key") == key:
            return r
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Сканер ринку: хто з перпів Binance «прокидається» (авто-список скринера)")
    parser.add_argument("--apply", action="store_true", help="оновити авто-список reports/auto_watchlist.json (як скринер)")
    parser.add_argument("--recall", action="store_true",
                        help="перевірити набори порогів на розмічених пампах з бази (без запитів до бірж)")
    args = parser.parse_args()
    from fetch import read_tickers_file
    manual = {t.upper() for t in read_tickers_file(os.path.join(ROOT, "watchlist.txt"))}

    if args.recall:
        import psycopg
        from dotenv import load_dotenv
        load_dotenv(os.path.join(ROOT, ".env"))
        cfg = load_config()
        with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
            out = {"at": datetime.now(timezone.utc).isoformat(timespec="minutes"), "sets": {}}
            sets = dict(PRESETS, custom=cfg["custom"])
            for name, p in sets.items():
                r = recall(conn, {**p, "max_oi": cfg["max_oi"]}, cfg["ttl_hours"])
                out["sets"][name] = {**r, "key": recall_key({**p, "max_oi": cfg["max_oi"]}, cfg["ttl_hours"])}
                print(f"{PRESET_NAMES[name]:<7} вчасно {r['early']}/{r['pumps']} · лише під час {r['during']} · не спрацював {r['never']}")
        _write_json(RECALL_FILE, out)
        return

    last = scan()
    cfg = load_config()
    for name in PRESETS:
        h = hits(last, {**PRESETS[name], "max_oi": cfg["max_oi"]}, exclude=manual)
        print(f"{PRESET_NAMES[name]:<7} {len(h):>3}: " + ", ".join(m["base"] for m in h[:12]) + (" …" if len(h) > 12 else ""))
    if args.apply:
        apply(last, manual)


if __name__ == "__main__":
    main()
