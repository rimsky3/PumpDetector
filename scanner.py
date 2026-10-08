"""
scanner.py — сканер ринку: які USDT-перпи Binance «прокидаються» і мають потрапити в скринер автоматично.

Воронка: весь ринок (≈500 перпів) → фільтр → авто-список → скринер (дані + модель) → Alert.
Фільтр навмисно м'якший за модель: його задача — не пропустити памп; відсіювати зайве — робота моделі.

Правило (пороги — у scan_config.json, набори Суворо / Баланс / Широко / Свій):
    ліквідність:  СЕРЕДНІЙ денний оборот за 30 днів ≥ min_turnover  і  OI ≤ max_oi (великі монети — не наш скоуп, урок ZEC)
                  (з 2026-10-03: за 24 год сканер пропускав ліквідні монети на тихому старті — SYN +51% при 7 млн за добу
                  і 36 млн у середньому — і брав дрібні, що щойно «розігнались»; experiments/turnover_at_entry.py)
    прокидання:   обсяг за 4 год ≥ vol_x × норми (медіана 4-годинного обсягу за попередні 7 днів) і ціна за 4 год ≥ 0
                  АБО  OI за 4 год ≥ +oi_pct %  і  ціна за 4 год ≥ +price_pct %
Скан — на закритих 15m-барах (кожні 15 хв); --recall — на годинних даних з бази, вікна ті самі.
Тікер, що пройшов, живе в авто-списку ttl_hours після останнього спрацювання (або сигналу моделі), максимум max_auto.

Канал «Прокинулась» (WOKE, 2026-10-08, experiments/wake_channel.py): ТИХА монета (середній оборот < min_turnover), що ожила:
    темп обороту (обсяг 4 год × 6) ≥ 20 млн $/добу  і  обсяг 4 год ≥ ×10 норми  і  ціна за 4 год ≥ +5%  і  OI ≥ +20%.
На 29 днях: ~10 подій на день, 42% дають +10% за добу (база 12%), але медіана за добу −5% — це «де дивитись», не вхід.
Окремий канал авто-списку (channel "woke", стеля WOKE_CAP), події — reports/woke_events.json (і для ручних тікерів).

Дані — публічний API Binance (ключ не потрібен, Coinalyze не витрачаємо): 1 запит на весь ринок
+ 2 на кожен перп, що проходить найм'якшу межу обороту (15m-свічки за 7.5 дня і OI за 4 год). Кожні 15 хв (screener.py).
Середній оборот — денні свічки за 30 завершених днів, раз на добу для кожного перпа (кеш reports/scan_turnover_avg.json).

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
SNOOZE_FILE = os.path.join(ROOT, "reports", "auto_snooze.json")   # прибрані «×» тікери: {base: коли прибрано}
BLOCK_MIN_HOURS = 1               # «×» блокує до нового спрацювання сканера (власник, 2026-10-07), але не менше години
RECALL_FILE = os.path.join(ROOT, "reports", "scan_recall.json")   # --recall: результат наборів на пампах
AVG_FILE = os.path.join(ROOT, "reports", "scan_turnover_avg.json")  # середній денний оборот перпів, раз на добу
AVG_DAYS = 30                      # ліквідність — середній денний оборот за стільки завершених днів
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
# канал «Прокинулась» (WOKE): пороги — варіант A з experiments/wake_channel.py, у налаштування не винесено
WOKE = {"pace": 20e6, "vol_x": 10.0, "price_pct": 5.0, "oi_pct": 20.0}
WOKE_CAP = 15                      # стеля каналу: найстаріший за останнім спрацюванням витісняється
WOKE_TTL_HOURS = 48                # подія живе 48 год після останнього спрацювання (сигнал моделі продовжує — touch)
WOKE_FILE = os.path.join(ROOT, "reports", "woke_events.json")   # {base: {first, last, reason, ...}} — і ручні тікери
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
    excluded = _excluded()
    return {b: r for b, r in load_auto().items() if pd.Timestamp(r["expires"]) > pd.Timestamp(now) and b not in excluded}


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

def _daily_avg(symbol: str) -> float | None:
    """Середній денний оборот (USDT) за AVG_DAYS завершених днів — денні свічки Binance; поточний день не рахується."""
    try:
        k = X.get(X.BINANCE + "/fapi/v1/klines", dict(symbol=symbol, interval="1d", limit=AVG_DAYS + 1))
    except Exception:
        return None
    now = int(datetime.now(timezone.utc).timestamp() * 1000)
    vols = [float(r[7]) for r in k if int(r[6]) < now][-AVG_DAYS:]
    return float(np.mean(vols)) if vols else None


def turnover_avg(symbols: list[str]) -> dict[str, float]:
    """Середній денний оборот перпів; рахується раз на добу (UTC) — кеш у файлі, докачуються лише нові символи."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    cache = _read_json(AVG_FILE, {})
    vals = cache.get("values", {}) if cache.get("date") == today else {}
    todo = [s for s in symbols if s not in vals]
    if todo:
        with ThreadPoolExecutor(WORKERS) as ex:
            for sym, v in zip(todo, ex.map(_daily_avg, todo)):
                if v is not None:
                    vals[sym] = v
        _write_json(AVG_FILE, {"date": today, "values": vals})
    return vals


def liquidity(m: dict) -> float | None:
    """Оборот для фільтра ліквідності: середній за AVG_DAYS днів, а якщо його немає — за 24 год."""
    v = m.get("turnover_avg")
    return v if v is not None else m.get("turnover")


def universe() -> list[dict]:
    """Усі USDT-перпи Binance, що торгуються: оборот за 24 год (два запити на весь ринок) і середній за 30 днів (кеш на добу)."""
    info = X.get(X.BINANCE + "/fapi/v1/exchangeInfo", {})
    day = {t["symbol"]: t for t in X.get(X.BINANCE + "/fapi/v1/ticker/24hr", {})}
    out = []
    for s in info["symbols"]:
        if s.get("contractType") != "PERPETUAL" or s.get("quoteAsset") != "USDT" or s.get("status") != "TRADING":
            continue
        t = day.get(s["symbol"], {})
        out.append({"symbol": s["symbol"], "base": s["baseAsset"], "turnover": float(t.get("quoteVolume") or 0)})
    avg = turnover_avg([u["symbol"] for u in out])
    for u in out:
        u["turnover_avg"] = avg.get(u["symbol"])
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
    tail = vol.tail(AVG_DAYS * 24 * bph)
    return {
        "turnover": float(vol.tail(24 * bph).sum()),
        "turnover_4h": float(v4.iloc[-1]),           # для каналу WOKE: темп обороту = обсяг 4 год × 6
        # середній денний за наявну історію (до AVG_DAYS днів); живий скан замінює його денними свічками за 30 днів
        "turnover_avg": float(tail.sum() / (len(tail) / (24 * bph))),
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
    ok = passes_series(h, p, bph)
    bar = pd.Timedelta(hours=1) / bph
    out, expires = [], None
    for ts in ok[ok.fillna(False)].index:
        t = ts + bar
        if expires is None or t > expires:
            out.append(t)
        expires = t + pd.Timedelta(hours=ttl_hours)
    return out


def passes_series(h: pd.DataFrame, p: dict, bph: int = 4) -> pd.Series:
    """Правило фільтра (metrics_from_bars + passes) векторно по всьому ряду: True на барах, де сканер з порогами p
    пропустив би тікер (на закритті бару). Для міток на графіку і бектесту з фільтром сканера."""
    w = 4 * bph
    vol, px, oi = h["volume"], h["price"], h["oi"]
    v4 = vol.rolling(w).sum()
    norm = v4.shift(w).rolling(7 * 24 * bph, min_periods=3 * 24 * bph - w).median()
    # середній денний оборот за AVG_DAYS днів (не менше 7 днів історії) — як ліквідність у живому скані
    turnover = vol.rolling(AVG_DAYS * 24 * bph, min_periods=7 * 24 * bph).mean() * 24 * bph
    price_pct = (px / px.shift(w) - 1) * 100
    oi_pct = (oi / oi.shift(w) - 1) * 100
    return (turnover >= p["min_turnover"]) & ~(oi > p["max_oi"]) & (
        ((v4 / norm >= p["vol_x"]) & (price_pct >= 0)) | ((oi_pct >= p["oi_pct"]) & (price_pct >= p["price_pct"])))


def passes(m: dict, p: dict) -> tuple[bool, str]:
    """Чи проходить тікер фільтр і чому (текст причини — для дашборду)."""
    liq = liquidity(m)
    if liq is None or liq < p["min_turnover"]:
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
    why.append(f"ціна {m['price_pct']:+.1f}% за 4 год · оборот {m['turnover'] / 1e6:.0f} млн $ за добу, "
               f"сер. {liq / 1e6:.0f} млн")
    return True, " · ".join(why)


def woke_passes(m: dict, liq_floor: float) -> tuple[bool, str]:
    """Канал WOKE: тиха монета (середній оборот за 30 днів < liq_floor) ожила. Причина — англійською (рядок у списку)."""
    avg = m.get("turnover_avg")
    pace = (m.get("turnover_4h") or 0) * 6
    if avg is None or avg >= liq_floor or pace < WOKE["pace"]:
        return False, ""
    if (m.get("vol_x") or 0) < WOKE["vol_x"] or m["price_pct"] < WOKE["price_pct"]:
        return False, ""
    if m.get("oi_pct") is None or m["oi_pct"] < WOKE["oi_pct"]:
        return False, ""
    return True, (f"Avg ${avg / 1e6:.1f}M → ${pace / 1e6:.0f}M/day · Vol ×{m['vol_x']:.0f} · "
                  f"Px {m['price_pct']:+.0f}% · OI {m['oi_pct']:+.0f}%")


def woke_hits(last: dict, liq_floor: float, exclude: set[str] = frozenset()) -> list[dict]:
    out = []
    for m in last.get("rows", []):
        ok, why = woke_passes(m, liq_floor)
        if ok and m["base"] not in exclude:
            out.append({**m, "reason": why})
    return sorted(out, key=lambda m: -(m.get("vol_x") or 0))


def load_woke() -> dict:
    return _read_json(WOKE_FILE, {})


def record_woke(found: list[dict], now: datetime, keep: set[str] = frozenset()) -> list[str]:
    """Події WOKE (усі тікери, і ручні): перше спрацювання — first (у дашборді «Woke · just now» і година в NOW,
    Telegram), повторне — лише last. Подія живе WOKE_TTL_HOURS після last; keep — тікери, яких тримає канал (сигнал
    моделі продовжив), їхня подія не зникає, і повторне спрацювання не вважається новим. Повертає нові."""
    ev = load_woke()
    ts = now.isoformat(timespec="minutes")
    cut = pd.Timestamp(now) - pd.Timedelta(hours=WOKE_TTL_HOURS)
    ev = {b: e for b, e in ev.items() if pd.Timestamp(e["last"]) > cut or b in keep}
    new = []
    for m in found:
        b = m["base"]
        rec = {"last": ts, "reason": m["reason"], "turnover_avg": m.get("turnover_avg"),
               "pace": (m.get("turnover_4h") or 0) * 6, "vol_x": m.get("vol_x"), "price_pct": m.get("price_pct"),
               "oi_pct": m.get("oi_pct")}
        if b in ev:
            ev[b].update(rec)
        else:
            ev[b] = {"first": ts, "first_reason": m["reason"], **rec}
            new.append(b)
    _write_json(WOKE_FILE, ev)
    return new


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
        if u.get("turnover_avg") is not None:
            m["turnover_avg"] = u["turnover_avg"]
    return m


def scan(log=print) -> dict:
    """Метрики всіх перпів, що проходять найм'якшу межу обороту (щоб дашборд міг показати результат будь-якого набору)."""
    t0 = datetime.now(timezone.utc)
    cfg = load_config()
    floor = min([p["min_turnover"] for p in PRESETS.values()] + [cfg["custom"]["min_turnover"]])
    uni = universe()
    # із запасом; + канал WOKE: у тихої монети, що ожила, оборот за 24 год уже ≥ темпу за 4 год
    cand = [u for u in uni if (liquidity(u) or 0) >= floor * 0.8 or u["turnover"] >= WOKE["pace"] / 6]
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


def _excluded() -> set[str]:
    from fetch import excluded_tickers
    return excluded_tickers(ROOT)


def snoozed() -> dict:
    """Тікери, прибрані власником «×» у списку дашборду, — блок «до нового спрацювання»: {base: коли прибрано (ISO)}.
    Старий формат файлу ({base: «до коли»}) читається так само — як момент блоку."""
    return dict(_read_json(SNOOZE_FILE, {}))


def block(base: str) -> None:
    """«×»: сканер не додає тікер, поки триває той самий сплеск (тікер безперервно проходить фільтр). Щойно на скані він
    фільтр не пройде (сплеск вичерпався, і минуло ≥ BLOCK_MIN_HOURS) — блок знято, наступне спрацювання поверне його як авто."""
    sn = snoozed()
    sn[base.upper()] = datetime.now(timezone.utc).isoformat(timespec="minutes")
    _write_json(SNOOZE_FILE, sn)


def unblock(base: str) -> None:
    sn = snoozed()
    if sn.pop(base.upper(), None) is not None:
        _write_json(SNOOZE_FILE, sn)


def release_blocks(passing: set[str], now: datetime) -> set[str]:
    """Після скану: тікери, що вже не проходять фільтр і заблоковані ≥ BLOCK_MIN_HOURS, — розблоковано; повертає ті, що
    лишаються заблокованими. Мінімум — щоб фільтр, який «мерехтить» (то проходить, то ні), не повернув той самий сплеск."""
    sn = snoozed()
    now = pd.Timestamp(now)
    keep = {b: t for b, t in sn.items()           # min(…, now): старі записи «до коли» (у майбутньому) — як «щойно»
            if b in passing or now - min(pd.Timestamp(t), now) < pd.Timedelta(hours=BLOCK_MIN_HOURS)}
    if len(keep) != len(sn):
        _write_json(SNOOZE_FILE, keep)
    return set(keep)


def remove_auto(base: str) -> dict | None:
    """«×» у списку дашборду: авто-тікер — геть зі списку і блок до нового спрацювання. Повертає прибраний запис
    (для «Повернути»), None — тікера в авто-списку не було (блок однаково ставить dashboard.remove_from_watchlist)."""
    base = base.upper()
    auto = load_auto()
    rec = auto.pop(base, None)
    if rec is None:
        return None
    _write_json(AUTO_FILE, auto)
    block(base)
    return rec


def restore_auto(base: str, rec: dict) -> None:
    """«Повернути»: запис — назад в авто-список, блок знято."""
    base = base.upper()
    auto = load_auto()
    auto[base] = rec
    _write_json(AUTO_FILE, auto)
    unblock(base)


def channel(rec: dict) -> str:
    """Канал авто-запису: "liquid" (звичайний сканер; старі записи без поля) або "woke" (тиха монета ожила)."""
    return rec.get("channel", "liquid")


def apply(last: dict, manual: set[str], log=print) -> dict:
    """Оновити авто-список: нові спрацювання, продовження, прострочені — геть; стеля окремо для каналів
    liquid (max_auto) і woke (WOKE_CAP). WOKE-монета, що пройшла звичайний фільтр, переходить у liquid."""
    cfg = load_config()
    now = datetime.now(timezone.utc)
    excluded = _excluded()
    auto = {b: r for b, r in load_auto().items() if pd.Timestamp(r["expires"]) > pd.Timestamp(now) and b not in excluded}
    if not cfg["enabled"]:
        _write_json(AUTO_FILE, auto)
        return auto
    ttl = timedelta(hours=cfg["ttl_hours"])
    ts = now.isoformat(timespec="minutes")
    p = params(cfg)
    found = hits(last, p, exclude=set(manual) | excluded)
    woke = woke_hits(last, p["min_turnover"], exclude=excluded)
    # події WOKE — для всіх, і ручних (у дашборді мітка «Woke» і година в NOW); тримаються, поки канал тримає тікер
    new_woke = record_woke(woke, now, keep={b for b, r in auto.items() if channel(r) == "woke"})
    first = {b: e["first"] for b, e in load_woke().items()}
    # прибрані «×» — лише після нового спрацювання (будь-якого з каналів)
    blocked = release_blocks({m["base"] for m in found} | {m["base"] for m in woke}, now)
    new = []
    for m in [m for m in found if m["base"] not in blocked]:
        b = m["base"]
        if b not in auto:
            new.append(b)
        auto[b] = {"added": auto.get(b, {}).get("added", ts), "channel": "liquid",
                   "last_hit": ts, "expires": (now + ttl).isoformat(timespec="minutes"),
                   "reason": m["reason"], "vol_x": m.get("vol_x"), "oi_pct": m.get("oi_pct"), "turnover": m["turnover"],
                   "turnover_avg": m.get("turnover_avg")}
    for m in [m for m in woke if m["base"] not in blocked and m["base"] not in manual]:
        b = m["base"]
        if b in auto and channel(auto[b]) == "liquid":
            continue                      # уже в звичайному каналі — там і лишається
        if b not in auto:
            new.append(b)
        auto[b] = {"added": auto.get(b, {}).get("added", ts), "channel": "woke",
                   "last_hit": ts, "expires": (now + timedelta(hours=WOKE_TTL_HOURS)).isoformat(timespec="minutes"),
                   "reason": m["reason"], "vol_x": m.get("vol_x"), "oi_pct": m.get("oi_pct"), "turnover": m["turnover"],
                   "turnover_avg": m.get("turnover_avg"), "woke_first": first.get(b, ts)}
    for ch, cap in (("liquid", int(cfg["max_auto"])), ("woke", WOKE_CAP)):    # стеля: лишаємо найсвіжіші спрацювання
        mine = [b for b in auto if channel(auto[b]) == ch]
        if len(mine) > cap:
            for b in sorted(mine, key=lambda b: (auto[b]["last_hit"], auto[b].get("vol_x") or 0), reverse=True)[cap:]:
                auto.pop(b)
    new = [b for b in new if b in auto]
    _write_json(AUTO_FILE, auto)
    n_woke = sum(channel(r) == "woke" for r in auto.values())
    log(f"авто-список: {len(auto)} (woke {n_woke})" + (f", нові: {', '.join(new)}" if new else "")
        + (f"; WOKE-події: {', '.join(new_woke)}" if new_woke else ""))
    return auto


def preview(p: dict, manual: set[str]) -> dict:
    """Для дашборду: скільки пройшло б за останнім сканом з такими порогами."""
    last = _read_json(LAST_FILE, {})
    h = hits(last, p, exclude=set(manual) | _excluded() | set(snoozed()))     # заблоковані «×» — не рахуємо
    return {"at": last.get("at"), "checked": last.get("checked", 0), "universe": last.get("universe", 0),
            "count": len(h), "top": [{"base": m["base"], "reason": m["reason"]} for m in h[:8]]}


# ---------------------------------------------------------------- перевірка на розмічених пампах

def recall(conn, p: dict, lookback_h: float = 48) -> dict:
    """Як фільтр спрацював би на розмічених пампах: скільки разів він додав би тікер до старту імпульсу
    (тікер живе в списку ttl_hours, тож рахується спрацювання за lookback_h до старту), скільки — лише під час, скільки — ніколи.
    Дані — годинні з бази (сума бірж Coinalyze), тож оборот трохи більший, ніж лише Binance; середній оборот — за 8 днів вікна."""
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
    w = woke_hits(last, params(cfg)["min_turnover"])
    print(f"WOKE    {len(w):>3}: " + "; ".join(f"{m['base']} ({m['reason']})" for m in w[:8]))
    if args.apply:
        apply(last, manual)


if __name__ == "__main__":
    main()
