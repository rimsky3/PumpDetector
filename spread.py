"""
spread.py — розбіжність цін одного тікера між біржами (Binance / Bybit / OKX), наживо (2026-10-13).

Навіщо: під час імпульсу біржі розходяться (DOS 12.10: на піку OKX на 5.5% нижче Binance, після обвалу Bybit 18 хв на 5%
вище) — це видно на графіку (панель Δ, чіп у заголовку з карткою), у списку (значок Δ) і приходить сповіщенням. У базу
НІЧОГО не пишеться (рішення власника): монітор живе в процесі дашборда і тримає в пам'яті останні 3 години.

Як рахується:
    кожні POLL_SEC секунд — три запити на весь ринок (по одному на біржу): найкращі бід і аск усіх USDT-перпів;
    розбіжність пари бірж «по біду й аску» = бід дорожчої / аск дешевшої − 1 (те, що реально можна взяти ринком);
    «звичайна» для монети й пари — медіана |розбіжності середин спреду| за останні norm_hours годин;
    АНОМАЛІЯ: розбіжність ≥ min_pct  І  ≥ norm_x × звичайна  І  тримається ≥ hold_sec. Кінець — два зрізи поспіль нижче
    0.7 порогу. Повторна подія по тій самій монеті раніше ніж за cooldown_min — без сповіщення (quiet).

Запобіжники: |розбіжність| > max_pct — під одним тікером різні активи, пара ігнорується; стара котирування (біржа не
оновлювала > STALE_SEC) — біржа пропускається; поки для пари немає MIN_SAMPLES зрізів (≈ 5 хв після старту) — подій немає,
тому постійний зсув між біржами стає «звичайним» і не сповіщає.

Налаштування — spread_config.json (кнопка «Spread ▾» у дашборді; на сервері файл змінює сам дашборд — deploy його не чіпає).

    python spread.py                 # два зрізи з перервою: найбільші розбіжності зараз (перевірка, без дашборда)
    python spread.py DOS             # картка однієї монети: бід / аск, розмір на кращій ціні, funding
"""

import argparse
import json
import os
import sys
import threading
import time
import warnings
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np
import requests

ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(ROOT, "spread_config.json")
AVG_FILE = os.path.join(ROOT, "reports", "scan_turnover_avg.json")     # середній оборот перпів Binance (scanner.py)
EXCHANGES = ("Binance", "Bybit", "OKX")
PAIRS = (("Bybit", "Binance"), ("OKX", "Binance"), ("OKX", "Bybit"))
POLL_SEC = 15
RING_HOURS = 3                    # скільки історії тримати в пам'яті (міні-графік картки, «звичайна» розбіжність)
CAP = 1600                        # максимум монет у кільці
MIN_SAMPLES = 20                  # зрізів пари, щоб знати її звичайну розбіжність (≈ 5 хв)
NORM_FLOOR = 0.0005               # звичайна не менша за 0.05% — інакше «у 5 разів більша» спрацьовує на шумі
NORM_EVERY = 60                   # с: перерахунок медіан
MAP_EVERY = 600                   # с: перечитати символи з бази і оборот
STALE_SEC = 180                   # котирування старіше — біржу для монети пропускаємо
EXIT_SHARE = 0.7                  # подія триває, поки розбіжність ≥ 0.7 порогу (без миготіння біля межі)
RECENT_HOURS = 6

DEFAULT_CONFIG = {
    "enabled": True,
    "min_pct": 1.0,               # розбіжність по біду й аску, не менше, %
    "norm_x": 5.0,                # і в стільки разів більша за звичайну для монети
    "hold_sec": 30,               # тримається стільки секунд
    "exchanges": list(EXCHANGES),
    "scope": "list",              # list — лише тікери списку дашборда; all — увесь ринок
    "min_turnover": 20.0,         # для scope = all: оборот монети, млн $ за добу
    "cooldown_min": 30,           # повторне сповіщення по монеті не раніше, хв
    "max_pct": 30.0,              # більша розбіжність = інший актив під тим самим тікером
    "norm_hours": 2.0,            # вікно «звичайної» розбіжності, год
    "ignore": [],                 # тікери, які не перевіряти
}
LIMITS = {"min_pct": (0.1, 20.0), "norm_x": (1.0, 50.0), "hold_sec": (0, 600), "min_turnover": (0.0, 1000.0),
          "cooldown_min": (0, 720), "max_pct": (2.0, 500.0), "norm_hours": (0.25, float(RING_HOURS))}


# ---------------------------------------------------------------- налаштування

def load_config() -> dict:
    try:
        with open(CONFIG_FILE, encoding="utf-8") as f:
            saved = json.load(f)
    except (OSError, ValueError):
        saved = {}
    return {**DEFAULT_CONFIG, **{k: v for k, v in saved.items() if k in DEFAULT_CONFIG}}


def save_config(new: dict) -> dict:
    """Перевірити і зберегти налаштування з дашборда; діють з наступного зрізу."""
    cfg = load_config()
    for k, (lo, hi) in LIMITS.items():
        if k in new:
            try:
                cfg[k] = min(hi, max(lo, float(new[k])))
            except (TypeError, ValueError):
                raise ValueError(f"{k}: потрібне число")
    if "enabled" in new:
        cfg["enabled"] = bool(new["enabled"])
    if "scope" in new:
        if new["scope"] not in ("list", "all"):
            raise ValueError("scope: list або all")
        cfg["scope"] = new["scope"]
    if "exchanges" in new:
        ex = [e for e in EXCHANGES if e in (new["exchanges"] or [])]
        if len(ex) < 2:
            raise ValueError("потрібні щонайменше дві біржі")
        cfg["exchanges"] = ex
    if "ignore" in new:
        raw = new["ignore"] if isinstance(new["ignore"], list) else str(new["ignore"]).replace(",", " ").split()
        cfg["ignore"] = sorted({str(t).strip().upper() for t in raw if str(t).strip()})[:200]
    tmp = CONFIG_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=1)
    os.replace(tmp, CONFIG_FILE)
    return cfg


# ---------------------------------------------------------------- біржі: один запит на весь ринок

def _f(v) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if x > 0 else None


def _get(s: requests.Session, url: str, **params):
    r = s.get(url, params=params, timeout=8)
    r.raise_for_status()
    return r.json()


def binance_quotes(s: requests.Session) -> dict:
    out = {}
    for x in _get(s, "https://fapi.binance.com/fapi/v1/ticker/bookTicker"):
        b, a = _f(x.get("bidPrice")), _f(x.get("askPrice"))
        if b and a and a >= b:
            out[x["symbol"]] = {"bid": b, "ask": a, "bid_usd": b * float(x.get("bidQty") or 0), "ask_usd": a * float(x.get("askQty") or 0),
                                "ts": (x.get("time") or 0) / 1000 or None}
    return out


def bybit_quotes(s: requests.Session) -> dict:
    out = {}
    for x in _get(s, "https://api.bybit.com/v5/market/tickers", category="linear")["result"]["list"]:
        b, a = _f(x.get("bid1Price")), _f(x.get("ask1Price"))
        if b and a and a >= b:
            nxt = _f(x.get("nextFundingTime"))
            out[x["symbol"]] = {"bid": b, "ask": a, "bid_usd": b * float(x.get("bid1Size") or 0), "ask_usd": a * float(x.get("ask1Size") or 0),
                                "ts": None, "turnover": _f(x.get("turnover24h")), "index": _f(x.get("indexPrice")),
                                "funding": float(x["fundingRate"]) if x.get("fundingRate") not in (None, "") else None,
                                "next": nxt / 1000 if nxt else None}
    return out


def okx_quotes(s: requests.Session, ct: dict) -> dict:
    out = {}
    for x in _get(s, "https://www.okx.com/api/v5/market/tickers", instType="SWAP")["data"]:
        b, a = _f(x.get("bidPx")), _f(x.get("askPx"))
        if b and a and a >= b:
            k = ct.get(x["instId"])             # розмір контракту в монетах: bidSz / askSz приходять у контрактах
            last, vol = _f(x.get("last")), _f(x.get("volCcy24h"))
            out[x["instId"]] = {"bid": b, "ask": a, "bid_usd": b * float(x.get("bidSz") or 0) * k if k else None,
                                "ask_usd": a * float(x.get("askSz") or 0) * k if k else None,
                                "ts": (_f(x.get("ts")) or 0) / 1000 or None, "turnover": vol * last if vol and last else None}
    return out


def okx_contracts(s: requests.Session) -> dict:
    """instId → монет в одному контракті (ctVal × ctMult), для розміру на кращій ціні в $."""
    out = {}
    for x in _get(s, "https://www.okx.com/api/v5/public/instruments", instType="SWAP")["data"]:
        v, m = _f(x.get("ctVal")), _f(x.get("ctMult")) or 1.0
        if v and x.get("ctValCcy") == x.get("instId", "").split("-")[0]:
            out[x["instId"]] = v * m
    return out


def coin_of(exchange: str, sym: str) -> str | None:
    """Назва монети з символу USDT-перпа; None — не безстроковий USDT-контракт."""
    if exchange == "OKX":
        return sym[:-10] if sym.endswith("-USDT-SWAP") else None
    if not sym.endswith("USDT") or "_" in sym or "-" in sym:
        return None
    return sym[:-4] or None


# ---------------------------------------------------------------- монітор

class Monitor:
    """Фоновий потік: зрізи котирувань → розбіжності → події. Читачі беруть готові self.snap / detail()."""

    def __init__(self, db_url: str | None = None, list_fn=None):
        self.db_url, self.list_fn = db_url, list_fn or (lambda: set())
        self.cfg = load_config()
        self.W = int(RING_HOURS * 3600 / POLL_SEC)
        self.mid = np.full((len(EXCHANGES), CAP, self.W), np.nan, dtype=np.float32)
        self.ts = np.zeros(self.W)
        self.n = 0                                   # скільки зрізів записано (позиція в кільці = n % W)
        self.idx: dict[str, int] = {}                # монета → рядок кільця
        self.syms: dict[str, dict] = {}              # монета → {біржа: символ}
        self.db_map: dict[str, dict] = {}
        self.avg: dict[str, float] = {}              # символ Binance → середній оборот (scanner.py)
        self.quotes: dict[str, dict] = {}            # біржа → {символ: котирування} — останній вдалий зріз
        self.ok_at: dict[str, float] = {}            # біржа → час останнього вдалого зрізу
        self.err: dict[str, str] = {}
        self.norm = np.full((len(PAIRS), CAP), np.nan)
        self.norm_at = self.map_at = 0.0
        self.pending: dict[str, dict] = {}
        self.active: dict[str, dict] = {}
        self.recent: deque = deque(maxlen=60)
        self.ended: dict[str, float] = {}            # монета → коли скінчилась остання подія (cooldown)
        self.ct: dict[str, float] = {}
        self.ct_at = 0.0
        self.snap: dict = {"at": None, "events": [], "recent": [], "rows": {}, "status": "старт"}
        self.fund: dict = {}                         # монета → (час, funding по біржах)
        self.http = requests.Session()
        self.pool = ThreadPoolExecutor(3)
        self.lock = threading.Lock()

    # ---------- довідники
    def _refresh_maps(self) -> None:
        if self.db_url:
            try:
                import psycopg
                with psycopg.connect(self.db_url, connect_timeout=5) as c:
                    rows = c.execute("""SELECT base_asset, exchange_name, symbol_on_exchange FROM markets
                                        WHERE quote_asset = 'USDT' AND symbol_on_exchange IS NOT NULL
                                          AND exchange_name = ANY(%s)""", (list(EXCHANGES),)).fetchall()
                m: dict[str, dict] = {}
                for base, ex, sym in rows:
                    m.setdefault(base, {}).setdefault(ex, sym)
                self.db_map = m
            except Exception as e:                   # база недоступна — працюємо зі старим довідником / назвами
                self.err["db"] = f"{type(e).__name__}: {e}"
        try:
            with open(AVG_FILE, encoding="utf-8") as f:
                self.avg = json.load(f).get("values", {})
        except (OSError, ValueError):
            pass
        if time.time() - self.ct_at > 6 * 3600:
            try:
                self.ct, self.ct_at = okx_contracts(self.http), time.time()
            except Exception as e:
                self.err["okx_ct"] = f"{type(e).__name__}: {e}"

    def _build_syms(self) -> None:
        """Монета → символи на біржах. База (markets) — точна прив'язка (PUMP → PUMPFUNUSDT на Bybit); решта ринку — за назвою."""
        syms = {b: dict(m) for b, m in self.db_map.items()}
        used = {(ex, s) for m in syms.values() for ex, s in m.items()}
        for ex, book in self.quotes.items():
            for sym in book:
                if (ex, sym) not in used:
                    c = coin_of(ex, sym)
                    if c:
                        syms.setdefault(c, {}).setdefault(ex, sym)
        for c in syms:
            if c not in self.idx and len(self.idx) < CAP:
                self.idx[c] = len(self.idx)
        self.syms = syms

    def turnover(self, coin: str) -> float | None:
        m = self.syms.get(coin, {})
        vals = [self.avg.get(m.get("Binance"))] + [(self.quotes.get(ex, {}).get(m.get(ex)) or {}).get("turnover") for ex in ("Bybit", "OKX")]
        vals = [v for v in vals if v]
        return max(vals) if vals else None

    # ---------- зріз
    def _fetch(self, enabled: list[str]) -> None:
        jobs = {"Binance": lambda: binance_quotes(self.http), "Bybit": lambda: bybit_quotes(self.http),
                "OKX": lambda: okx_quotes(self.http, self.ct)}
        futs = {ex: self.pool.submit(jobs[ex]) for ex in enabled}
        for ex, f in futs.items():
            try:
                self.quotes[ex] = f.result(timeout=15)
                self.ok_at[ex] = time.time()
                self.err.pop(ex, None)
            except Exception as e:                   # біржа не відповіла — цього зрізу в неї немає (не «нуль»)
                self.err[ex] = f"{type(e).__name__}: {str(e)[:120]}"

    def _quote(self, coin: str, ex: str, now: float) -> dict | None:
        if now - self.ok_at.get(ex, 0) > POLL_SEC * 2.5:
            return None
        q = self.quotes.get(ex, {}).get(self.syms.get(coin, {}).get(ex))
        if not q or (q["ts"] is not None and now - q["ts"] > STALE_SEC):
            return None
        return q

    def step(self) -> None:
        cfg = self.cfg = load_config()
        now = time.time()
        enabled = [e for e in EXCHANGES if e in cfg["exchanges"]]
        if now - self.map_at > MAP_EVERY:
            self._refresh_maps()
        self._fetch(enabled)
        if now - self.map_at > MAP_EVERY or not self.syms:
            self._build_syms()
            self.map_at = now
        pos = self.n % self.W
        self.mid[:, :, pos] = np.nan
        self.ts[pos] = now
        qs: dict[str, dict] = {}
        for coin, row in self.idx.items():
            got = {}
            for ei, ex in enumerate(EXCHANGES):
                q = self._quote(coin, ex, now) if ex in enabled else None
                if q:
                    got[ex] = q
                    self.mid[ei, row, pos] = (q["bid"] + q["ask"]) / 2
            if len(got) >= 2:
                qs[coin] = got
        self.n += 1
        if now - self.norm_at > NORM_EVERY:
            self._norms(cfg)
            self.norm_at = now
        self._events(cfg, qs, now)
        self.snap = self._snapshot(cfg, qs, now)

    def _window(self, hours: float) -> np.ndarray:
        """Індекси кільця за останні hours годин у порядку часу."""
        k = min(self.n, self.W, max(1, int(hours * 3600 / POLL_SEC)))
        return (np.arange(self.n - k, self.n)) % self.W

    def _norms(self, cfg: dict) -> None:
        w, n = self._window(cfg["norm_hours"]), len(self.idx)
        if not n:
            return
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)          # рядки без жодного значення
            for pi, (a, b) in enumerate(PAIRS):
                d = np.abs(self.mid[EXCHANGES.index(a), :n][:, w] / self.mid[EXCHANGES.index(b), :n][:, w] - 1)
                cnt = np.isfinite(d).sum(axis=1)
                self.norm[pi, :n] = np.where(cnt >= MIN_SAMPLES, np.maximum(np.nanmedian(d, axis=1), NORM_FLOOR), np.nan)

    def _in_scope(self, coin: str, cfg: dict, listed: set) -> bool:
        if coin in cfg["ignore"]:
            return False
        if coin in listed:
            return True
        return cfg["scope"] == "all" and (self.turnover(coin) or 0) >= cfg["min_turnover"] * 1e6

    def _events(self, cfg: dict, qs: dict, now: float) -> None:
        listed = set(self.list_fn() or ())
        lo_pct, hi_pct = cfg["min_pct"] / 100, cfg["max_pct"] / 100
        for coin in set(qs) | set(self.pending):
            got, row, best = qs.get(coin, {}), self.idx.get(coin), None
            live = coin in self.active
            if cfg["enabled"] and row is not None and self._in_scope(coin, cfg, listed):
                for pi, (a, b) in enumerate(PAIRS):
                    qa, qb = got.get(a), got.get(b)
                    nrm = self.norm[pi, row]
                    if not qa or not qb or not np.isfinite(nrm):
                        continue
                    if abs((qa["bid"] + qa["ask"]) / (qb["bid"] + qb["ask"]) - 1) > hi_pct:
                        continue                                     # різні активи під одним тікером
                    e1, e2 = qa["bid"] / qb["ask"] - 1, qb["bid"] / qa["ask"] - 1
                    e, hi, lo = (e1, a, b) if e1 >= e2 else (e2, b, a)
                    need = max(lo_pct, cfg["norm_x"] * nrm) * (EXIT_SHARE if live else 1.0)
                    if e >= need and (best is None or e > best["pct"]):
                        best = {"pct": e, "hi": hi, "lo": lo, "norm": float(nrm)}
            p = self.pending.get(coin)
            if best:
                if p is None:
                    p = self.pending[coin] = {"since": now, "peak": 0.0}
                p.update(best, miss=0, peak=max(p["peak"], best["pct"]))
                if not live and now - p["since"] >= cfg["hold_sec"]:
                    quiet = now - self.ended.get(coin, 0) < cfg["cooldown_min"] * 60
                    self.active[coin] = {"ticker": coin, "ts": datetime.fromtimestamp(p["since"], tz=timezone.utc).isoformat(),
                                         "since": p["since"], "quiet": quiet}
                if coin in self.active:
                    self.active[coin].update(pct=p["pct"], hi=p["hi"], lo=p["lo"], norm=p["norm"], peak=p["peak"],
                                             in_list=coin in listed)
            elif p is not None:
                p["miss"] = p.get("miss", 0) + 1
                if p["miss"] >= 2 or not cfg["enabled"]:
                    del self.pending[coin]
                    ev = self.active.pop(coin, None)
                    if ev:
                        ev["end"] = now
                        self.ended[coin] = now
                        self.recent.appendleft(ev)

    def _row(self, coin: str, got: dict, hi_pct: float) -> dict | None:
        """Найбільша розбіжність середин спреду відносно еталонної біржі графіка (Binance, інакше Bybit)."""
        ref = next((e for e in EXCHANGES if e in got), None)
        best = None
        for ex, q in got.items():
            if ex == ref:
                continue
            d = (q["bid"] + q["ask"]) / (got[ref]["bid"] + got[ref]["ask"]) - 1
            if abs(d) <= hi_pct and (best is None or abs(d) > abs(best["d"])):
                best = {"ex": ex, "d": d, "ref": ref}
        return best

    def _snapshot(self, cfg: dict, qs: dict, now: float) -> dict:
        listed = set(self.list_fn() or ())
        hi_pct = cfg["max_pct"] / 100
        rows = {c: r for c in listed if c in qs and (r := self._row(c, qs[c], hi_pct))}
        view = lambda e: {**e, "held": now - e["since"]}               # noqa: E731
        cut = now - RECENT_HOURS * 3600
        down = [e for e in cfg["exchanges"] if now - self.ok_at.get(e, 0) > POLL_SEC * 2.5]
        warm = self.n < MIN_SAMPLES
        return {"at": now, "enabled": cfg["enabled"], "coins": len(qs), "down": down, "warmup": warm,
                "status": ("вимкнено" if not cfg["enabled"] else f"немає відповіді: {', '.join(down)}" if down
                           else "збираю звичайну розбіжність (перші 5 хв)" if warm else "працює"),
                "events": sorted((view(e) for e in self.active.values()), key=lambda e: -e["pct"]),
                "recent": [e for e in self.recent if e["end"] >= cut][:20], "rows": rows}

    # ---------- картка тікера
    def _funding(self, coin: str) -> dict:
        hit = self.fund.get(coin)
        if hit and time.time() - hit[0] < 30:
            return hit[1]
        m, out = self.syms.get(coin, {}), {}
        if m.get("Binance"):
            try:
                j = _get(self.http, "https://fapi.binance.com/fapi/v1/premiumIndex", symbol=m["Binance"])
                out["Binance"] = {"funding": float(j["lastFundingRate"]), "next": j["nextFundingTime"] / 1000, "index": _f(j.get("indexPrice"))}
            except Exception:
                pass
        if m.get("OKX"):
            try:
                j = _get(self.http, "https://www.okx.com/api/v5/public/funding-rate", instId=m["OKX"])["data"][0]
                out["OKX"] = {"funding": float(j["fundingRate"]), "next": float(j["fundingTime"]) / 1000}
            except Exception:
                pass
        q = self.quotes.get("Bybit", {}).get(m.get("Bybit"))
        if q:
            out["Bybit"] = {"funding": q.get("funding"), "next": q.get("next"), "index": q.get("index")}
        if len(self.fund) > 200:
            self.fund.clear()
        self.fund[coin] = (time.time(), out)
        return out

    def detail(self, coin: str) -> dict:
        now, cfg = time.time(), self.cfg
        got = {ex: q for ex in EXCHANGES if ex in cfg["exchanges"] and (q := self._quote(coin, ex, now))}
        if not got:
            return {"ticker": coin, "ex": [], "note": self.snap.get("status")}
        ref = next(e for e in EXCHANGES if e in got)
        fund, row = self._funding(coin), self.idx.get(coin)
        mid = lambda q: (q["bid"] + q["ask"]) / 2                       # noqa: E731
        ex_rows, norms = [], {}
        for ex in EXCHANGES:
            q = got.get(ex)
            if not q:
                continue
            f = fund.get(ex, {})
            ex_rows.append({"name": ex, "bid": q["bid"], "ask": q["ask"], "d": None if ex == ref else mid(q) / mid(got[ref]) - 1,
                            "bid_usd": q.get("bid_usd"), "ask_usd": q.get("ask_usd"), "funding": f.get("funding"), "next": f.get("next")})
            if ex != ref and row is not None:
                pi = next((i for i, p in enumerate(PAIRS) if set(p) == {ex, ref}), None)
                if pi is not None and np.isfinite(self.norm[pi, row]):
                    norms[ex] = float(self.norm[pi, row])
        hist = {"t": []}
        if row is not None and self.n:
            w = self._window(RING_HOURS)
            step = max(1, len(w) // 180)
            w = w[::-1][::step][::-1]                                   # проріджуємо, останній зріз лишається
            hist["t"] = self.ts[w].tolist()
            for ex in got:
                if ex != ref:
                    d = self.mid[EXCHANGES.index(ex), row, w] / self.mid[EXCHANGES.index(ref), row, w] - 1
                    hist[ex] = [None if not np.isfinite(v) else float(v) for v in d]
        index = next((fund[e]["index"] for e in ("Binance", "Bybit") if fund.get(e, {}).get("index")), None)
        ev = self.active.get(coin)
        return {"ticker": coin, "at": now, "ref": ref, "ex": ex_rows, "norm": norms, "index": index, "hist": hist,
                "event": {**ev, "held": now - ev["since"]} if ev else None,
                "rule": {"min_pct": cfg["min_pct"], "norm_x": cfg["norm_x"], "hold_sec": cfg["hold_sec"]}}

    # ---------- потік
    def run(self) -> None:
        while True:
            t0 = time.time()
            try:
                with self.lock:
                    self.step()
            except Exception as e:                   # помилка зрізу не зупиняє монітор
                self.err["step"] = f"{type(e).__name__}: {e}"
                print(f"  ! spread: {type(e).__name__}: {e}", flush=True)
            time.sleep(max(1.0, POLL_SEC - (time.time() - t0)))

    def start(self) -> "Monitor":
        threading.Thread(target=self.run, daemon=True, name="spread").start()
        return self


def api(mon: Monitor | None) -> dict:
    """Відповідь /api/spreads: події, розбіжність тікерів списку, налаштування."""
    if mon is None:
        return {"at": None, "events": [], "recent": [], "rows": {}, "status": "монітор не запущено", "config": load_config()}
    return {**mon.snap, "config": mon.cfg, "errors": dict(mon.err)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Розбіжність цін між біржами: перевірка без дашборда (нічого не записує)")
    ap.add_argument("ticker", nargs="?", help="монета — показати картку (бід / аск, розмір, funding); без неї — найбільші розбіжності ринку")
    ap.add_argument("--top", type=int, default=15, help="скільки монет показати")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    mon = Monitor()
    mon.step()
    time.sleep(3)
    mon.step()
    now = time.time()
    if args.ticker:
        d = mon.detail(args.ticker.upper())
        for r in d["ex"]:
            print(f"{r['name']:<8} бід {r['bid']:.6g}  аск {r['ask']:.6g}  до {d['ref']}: {'—' if r['d'] is None else f'{r['d']:+.2%}'}  "
                  f"на кращій ціні ${(r['bid_usd'] or 0):,.0f} / ${(r['ask_usd'] or 0):,.0f}  funding "
                  f"{'—' if r['funding'] is None else f'{r['funding']:+.3%}'}")
        return
    rows = []
    for coin in mon.idx:
        got = {ex: q for ex in EXCHANGES if (q := mon._quote(coin, ex, now))}
        r = mon._row(coin, got, mon.cfg["max_pct"] / 100) if len(got) >= 2 else None
        if r:
            rows.append((abs(r["d"]), coin, r, mon.turnover(coin)))
    print(f"монет на ≥ 2 біржах: {len(rows)}; помилки: {mon.err or 'немає'}")
    for _, coin, r, turn in sorted(rows, reverse=True)[: args.top]:
        print(f"  {coin:<12} {r['ex']} проти {r['ref']}: {r['d']:+.2%}   оборот {'—' if not turn else f'{turn / 1e6:,.0f} млн'}")


if __name__ == "__main__":
    main()
