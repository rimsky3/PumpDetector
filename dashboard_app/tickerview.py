"""
dashboard_app/tickerview.py — графік тікера (/api/ticker): 15m / 5m / 1h бари, зони сигналу, угоди і стоп, мітки «прокидається» і сканера,
дозавантаження свіжих барів з бірж, розбіжність цін між біржами; «Оновити» (/api/refresh).
"""

import math
import threading
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import psycopg

import drivers as DRV
import fetch_exchange
import footprint
import market
import orderbook
import scanner
import spread
import strategies
import tiers as T
import wake
from features import INTERVAL_15M, INTRADAY_EXCHANGES, aggregate, compute_features, load_raw
from fetch import Coinalyze, collect
from risk import EXIT_BARS, PARTIAL_EXIT, position_state, trade_stats
from screener import FETCH_DAYS, INTRADAY_DAYS, last_closed_15m, score_15m, score_bases, update_15m

from dashboard_app.common import State, last_atr, latest_signals, list_sources, refresh_lock, share, start_scores, status_of, tier_rows
from dashboard_app.trades import alert_starts, fp_rows, limit_hint, live_stop, ob_at_signal
from dashboard_app.listview import _list_key


_bars_cache: dict = {}                     # (тікер, дні) → (час, raw, h): легкий і повний запити перемикання тікера — одні дані
BARS_TTL = 20.0


def _bars_15m(conn, base: str, days: int, now: datetime):
    """Сирі 15m-ряди й агрегат тікера; перемикання тікера — два запити (легкий, потім повний), тож другий бере дані першого
    (до BARS_TTL с), а не перечитує базу й не агрегує заново."""
    key, t = (base, days), time.monotonic()
    hit = _bars_cache.get(key)
    if hit and t - hit[0] < BARS_TTL:
        return hit[1], hit[2]
    raw = load_raw(conn, [base], INTERVAL_15M, since=now - timedelta(days=days + INTRADAY_DAYS))
    h = aggregate(raw, base, "15min")
    if len(_bars_cache) > 200:
        _bars_cache.clear()
    _bars_cache[key] = (t, raw, h)
    return raw, h


def ticker_15m(conn, base: str, days: int, strat: dict | None = None, light: bool = False) -> dict:
    """Екран тікера для 15m + 1h: 15m-свічки, P 15m-моделі, годинний контекст, рівні сигналу, позиція.
    light=True («Повна інфо» вимкнена, 2026-10-06): без позиції / угод, стакану, «прокидається» — лише свічки, OI,
    обсяг, P, зони сигналу і картки ймовірності; помітно швидше."""
    m15 = State.m15
    now = datetime.now(timezone.utc)
    raw, h = _bars_15m(conn, base, days, now)
    if h.empty:
        raise ValueError(f"немає 15m-даних для {base}: натисни «Refresh» або дочекайся циклу скринера")
    h = h[h.index <= last_closed_15m()]
    # ознаки потрібні лише для останнього бару (картки): найдовше вікно — 24 год + 7 днів, тож 10 днів дають той самий
    # результат, що й уся історія (+10 днів запасу), а рахуються в рази швидше
    f = compute_features(h.iloc[-10 * 96:], 4)
    cur = f.dropna(subset=["oi_chg_4h"]).iloc[-1] if f["oi_chg_4h"].notna().any() else None
    t, troll = tier_rows(conn, base, rolling=True)
    t = t.set_index("ts").sort_index()
    # остання ковзна оцінка (5-хв цикл для «гарячих»), якщо вона новіша за останній оцінений 15m-бар: з неї приходять
    # сповіщення, тож графік показує її окремою точкою — інакше «Кандидат 0.87» у сповіщенні і 0.73 на графіку
    rolling = None
    if len(troll):
        rl = troll.sort_values("ts").iloc[-1]
        if not len(t) or rl["ts"] > t.index.max():
            rolling = {"ts": rl["ts"], "close": rl["ts"] + pd.Timedelta(minutes=15), "p15": rl["p15"], "thr15": rl["thr15"],
                       "s15": rl["s15"], "p1h": rl["p1h"], "tier": rl["tier"]}
    k, exit_bars = m15.get("risk", {}).get("k"), m15.get("exit_bars", 4)
    strat = strat or strategies.get(None)
    # стакан — для міток OB і фільтра режиму «OB ≥ 1»
    book_rows = []
    if not light:
        try:
            book_rows = orderbook.history(conn, base, (h.index.min() if len(h) else pd.Timestamp(now)).to_pydatetime())
        except Exception:
            conn.rollback()
    # позиція за режимом: сигнал (Alert / Кандидат), вхід (ринок / ліміт), фільтр OB; вихід спільний (частковий)
    score = t["s15"] if strat.get("signal") == "candidate" and len(t) else (t["alert_score"] if len(t) else None)
    if light or not len(t):
        pos = {"open": False, "trades": []}
    else:
        p_marks = footprint.p_times(conn, [base], h.index.min()).get(base, set()) if strat.get("p_be") else None
        entry_p = None
        if strategies.is_split(strat):             # режим split: вхід — модель «старт», утримання — основна або «старт»
            entry_p, score = T.split_scores(score, start_scores(conn, base))
        pos = position_state(h, score, 1.0, k, exit_bars, PARTIAL_EXIT, strategies.entry_rule(strat),
                             market.both(orderbook.allow_fn(book_rows) if strat.get("ob") else None,
                                         market.strategy_allow(conn, strat)), p_marks, entry_prob=entry_p)
        pos = live_stop(conn, base, pos, h)             # стоп на 5m-барах поточного, ще не закритого 15m-бару
    last = t.iloc[-1] if len(t) else None
    # Drivers: внески ознак у P останнього оціненого бару (drivers.py; ознаки — як у скринері, бар кешується)
    drv = None
    if last is not None:
        try:
            drv = DRV.for_bar(conn, m15, base, t.index.max())
        except Exception as e:
            print(f"! drivers {base}: {type(e).__name__}: {e}")
            conn.rollback()
    streak = 0
    if last is not None:
        for x in reversed(t["tier"].tolist()):
            if x != last["tier"] or x == "calm":
                break
            streak += 1
    view = h[h.index > h.index.max() - pd.Timedelta(days=days)]
    code = {"alert": 2, "candidate": 1, "calm": 0}
    chg24 = (h["price"].iloc[-1] / h["price"].iloc[-97] - 1) if len(h) > 96 and h["price"].iloc[-97] else None
    thr = float(last["thr15"]) if last is not None else m15["threshold"]
    trained = base in m15.get("tickers", []) or any(b == base for b, *_ in m15.get("in_sample_extra", []))
    # угода на історії (оцінки дораховані заднім числом, напр. коли тікер щойно потрапив у список) чи live —
    # на графіку історичні мітки бліді: такий сигнал ти б не отримав
    live = t["live"].astype(bool) if "live" in t else pd.Series(False, index=t.index)
    for tr in list(pos.get("trades", [])) + ([pos] if pos.get("open") else []):
        tr["live"] = bool(live.get(tr["entry_ts"], False))
    history = None if light else {**trade_stats(pos.get("trades", [])), "trained": trained,
                                  "days": round((t.index.max() - t.index.min()) / pd.Timedelta(days=1), 1) if len(t) else 0}
    # стакан (біди / аски ±2%) на початках Alert — зі знімків скринера; ліміт −0.5 ATR, поки дійсний
    starts = [s for s in alert_starts(t) if s >= view.index.min()]
    ob_marks = [{"ts": s, "v": v} for s in starts if (v := ob_at_signal(book_rows, s)) is not None]
    lim = limit_hint(h, starts[-1]) if starts else None          # підказка: ліміт −0.5 ATR від останнього Alert, 1 год
    if lim and lim["until"] <= pd.Timestamp(now):
        lim = None
    return {
        "ticker": base, "model": T.TIER_MODEL, "tf": "15m", "bar": 900, "threshold": thr, "k": k, "history": history,
        "streak": streak, "streak_min": streak * 15, "chg_24h": chg24,
        "price": h["price"].iloc[-1] if len(h) else None,
        "ls_long_pct": h["ls_long_pct"].dropna().iloc[-1] if h["ls_long_pct"].notna().any() else None,
        "ls_flow": ls_flow(h),
        "exit_bars": exit_bars, "atr": last_atr(h),
        "last_bar": view.index.max(),
        "in_sample": bool(last["in_sample"]) if last is not None else None,
        "ctx": {"p1h": last["p1h"] if last is not None else None, "thr1h": last["thr1h"] if last is not None else None,
                "share": share(), "s1h": last["s1h"] if last is not None else None},
        "cards": {
            "probability": last["p15"] if last is not None else None,
            "status": last["tier"] if last is not None else "none",
            "oi_chg_4h": math.expm1(cur["oi_chg_4h"]) if cur is not None else None,
            "volume_z_7d": cur["volume_z_7d"] if cur is not None else None,
        },
        "position": pos, "ob_marks": ob_marks, "limit": lim, "rolling": rolling, "drivers": drv,
        # Кандидат / Alert, що були ЛИШЕ на ковзних вікнах 5-хв циклу (з них приходили сповіщення): на графіку — бліда зона і
        # кружечок «5m» на панелі P, щоб сигнал не зникав, коли звичайний 15m-бар оцінено нижче порогу (VIRTUAL 04.10 16:55)
        "rolling_sig": [{"ts": r.ts, "close": r.ts + pd.Timedelta(minutes=15), "p15": r.p15, "tier": r.tier}
                        for r in troll.itertuples() if r.tier in ("candidate", "alert") and len(view) and r.ts >= view.index.min()]
        if len(troll) else [],
        # «прокидається» (wake.py) на 5m-барах за видимий період: свічка й обсяг на графіку — фіолетові
        "wake": [] if light else wake_marks(conn, base, view.index.min() if len(view) else pd.Timestamp(now)),
        # мітки футпрінту b («залочка» на лоу) / p (на хаях) за видимий період — під / над свічкою, з результатом
        "fp": fp_rows(conn, [base], view.index.min() if len(view) else pd.Timestamp(now), outcomes=not light),
        "light": light,
        "series": {
            "t": list(view.index),
            "open": view["open"].tolist(), "price": view["price"].tolist(),
            "high": view["high"].tolist(), "low": view["low"].tolist(),
            "oi": view["oi"].tolist(), "volume": view["volume"].tolist(),
            "coinO": view["oi_coins_o"].tolist(), "coinH": view["oi_coins_h"].tolist(),
            "coinL": view["oi_coins_l"].tolist(), "coin": view["oi_coins_c"].tolist(),
            "usdO": view["oi_usd_o"].tolist(), "usdH": view["oi_usd_h"].tolist(), "usdL": view["oi_usd_l"].tolist(),
            "prob": {T.TIER_MODEL: t["p15"].reindex(view.index).tolist() if len(t) else [],
                     "ctx": t["p1h"].reindex(view.index).tolist() if len(t) else []},
            "tier": t["tier"].map(code).reindex(view.index).tolist() if len(t) else [],
        },
        "thresholds": {T.TIER_MODEL: thr},
    }


_oi_share_cache: dict = {}


_wake_cache: dict = {}                     # (тікер, день початку вікна) → (5-хв відрізок, мітки)


def wake_marks(conn, base: str, since: pd.Timestamp) -> list[dict]:
    """Мітки «прокидається» за видимий період. wake.history перечитує 5m-дані за ~38 днів (ZEC — до 2 с на перемикання,
    заміри 2026-10-11), тож результат кешується до наступного 5-хв бару: повторні відкриття тікера — миттєві."""
    since = pd.Timestamp(since)
    key, slot = (base, since.floor("1D")), pd.Timestamp.now(tz="UTC").floor("5min")
    hit = _wake_cache.get(key)
    if hit and hit[0] == slot:
        return [w for w in hit[1] if w["ts"] >= since]
    try:
        res = wake.history(conn, base, since.floor("1D"))
    except Exception as e:                     # 5m-даних може не бути — графік без міток, а не помилка
        print(f"! прокидається {base}: {e}")
        conn.rollback()
        return []
    if len(_wake_cache) > 500:
        _wake_cache.clear()
    _wake_cache[key] = (slot, res)
    return [w for w in res if w["ts"] >= since]


def ls_flow(h: pd.DataFrame) -> dict:
    """Для рядка заголовка графіка (варіант C макета, 22.10): частка лонгів натовпу зараз і її зміна за 4 год (п.п.),
    частка ринкових покупок в обсязі за останню годину (Binance і OKX дають buy volume; Bybit — ні). h — бари тікера
    будь-якого ТФ з індексом-часом."""
    out = {"ls": None, "ls_chg_4h": None, "buy_1h": None}
    if h.empty:
        return out
    ls = h["ls_long_pct"].dropna() if "ls_long_pct" in h else pd.Series(dtype=float)
    if len(ls):
        out["ls"] = float(ls.iloc[-1])
        then = ls[ls.index <= ls.index[-1] - pd.Timedelta(hours=4)]
        out["ls_chg_4h"] = float(ls.iloc[-1] - then.iloc[-1]) if len(then) else None
    if "buy_volume" in h and "volume_bv" in h:
        last = h[h.index > h.index[-1] - pd.Timedelta(minutes=59)]
        v = last["volume_bv"].sum()
        out["buy_1h"] = float(last["buy_volume"].sum() / v) if v > 0 else None
    return out


def top_traders(conn, base: str) -> dict | None:
    """«Топ-трейдери» Binance на останньому 15m-барі (top_long_short): частка лонгів за акаунтами і за розміром позицій.
    Поруч із «Часткою лонгів» (усі акаунти) — видно, коли натовп і великі гроші стоять у різні боки."""
    try:
        rows = conn.execute("""SELECT t.ts, t.acc_long_pct, t.pos_long_pct FROM top_long_short t JOIN markets m USING (symbol)
                              WHERE m.base_asset = %s AND m.exchange_name = 'Binance' AND m.quote_asset = 'USDT'
                                AND t.interval = '15min' AND t.ts > now() - interval '1 day'
                              ORDER BY t.ts DESC LIMIT 20""", (base,)).fetchall()
    except Exception:                       # таблиці ще немає (скринер не створив) — без рядка
        conn.rollback()
        return None
    if not rows:
        return None
    r = rows[0]
    then = next((x for x in rows if x[0] <= r[0] - pd.Timedelta(hours=4)), None)     # 4 год тому — для стрілки
    return {"ts": r[0], "acc": r[1], "pos": r[2],
            "pos_chg_4h": r[2] - then[2] if then is not None and r[2] is not None and then[2] is not None else None}


def oi_share(conn, base: str) -> list[dict] | None:
    """Частка OI тікера за біржами (середнє за 30 днів, годинні дані Coinalyze по всіх біржах) і чи бачить її модель:
    15m-модель бере лише features.INTRADAY_EXCHANGES. Кеш на годину."""
    hit = _oi_share_cache.get(base)
    if hit and time.time() - hit[0] < 3600:
        return hit[1]
    # середнє по кожному контракту, потім сума контрактів біржі (USDT і USDC на одній біржі — разом)
    rows = conn.execute("""SELECT e, sum(a) FROM (
                               SELECT m.exchange_name AS e, o.symbol, avg(o.c) AS a FROM open_interest o JOIN markets m USING (symbol)
                               WHERE m.base_asset = %s AND o.interval = '1hour' AND o.ts > now() - interval '30 days'
                               GROUP BY 1, 2) x GROUP BY 1""", (base,)).fetchall()
    total = sum(float(v or 0) for _, v in rows)
    res = None
    if total > 0:
        res = sorted(({"exchange": e, "share": float(v or 0) / total, "seen": e in INTRADAY_EXCHANGES} for e, v in rows),
                     key=lambda x: (not x["seen"], -x["share"]))
        res = [x for x in res if x["share"] >= 0.005]
    _oi_share_cache[base] = (time.time(), res)
    return res


def scanner_entries(conn, base: str, days: int) -> list:
    """Для авто-тікерів: коли сканер з поточними порогами додав би тікер у список, якби працював увесь час
    (15m-дані Binance з бази — як у живому скані). Для твоїх тікерів з watchlist.txt — нічого: вони в списку завжди."""
    manual, _ = list_sources()
    if base in manual:
        return []
    cfg = scanner.load_config()
    since = datetime.now(timezone.utc) - timedelta(days=days + scanner.AVG_DAYS + 1)   # + середній оборот за 30 днів
    h = aggregate(load_raw(conn, [base], INTERVAL_15M, since=since, exchanges=["Binance"]), base, "15min")
    if h.empty:
        return []
    h = h[h.index <= last_closed_15m()]
    cut = pd.Timestamp(datetime.now(timezone.utc) - timedelta(days=days))
    return [t for t in scanner.entries_on_history(h, scanner.params(cfg), cfg["ttl_hours"]) if t >= cut]


CHART_TF = {"5m": ("5min", "5min", 300), "15m": ("15min", "15min", 900), "1h": ("1hour", "h", 3600)}


def rebar(conn, res: dict, base: str, tf: str, days: int) -> dict:
    """Графік на іншому таймфреймі, ніж рідний для моделі. Модель і позиція лишаються своїми; змінюються бари:
    крупніший ТФ — максимум оцінок і рівня за бар (Alert усередині години не губиться);
    дрібніший — остання оцінка, відома на момент закриття бару (нічого з майбутнього).
    Мітки входу/виходу — на бар, що закривається разом з баром моделі."""
    if tf not in CHART_TF or tf == res["tf"]:
        return res
    interval, freq, sec = CHART_TF[tf]
    nb, cb = pd.Timedelta(seconds=res["bar"]), pd.Timedelta(seconds=sec)
    now = datetime.now(timezone.utc)
    h = aggregate(load_raw(conn, [base], interval, since=now - timedelta(days=days + 1)), base, freq)
    if h.empty:
        raise ValueError(f"немає {tf}-даних для {base}" + (" — вони з'являться після циклу скринера" if tf == "5m" else ""))
    h = h[h.index <= pd.Timestamp(now).floor(freq) - cb]
    view = h[h.index > h.index.max() - pd.Timedelta(days=days)]
    s = res["series"]
    native = pd.DataFrame({f"p_{k}": v for k, v in s["prob"].items() if len(v)}, index=pd.DatetimeIndex(s["t"]))
    if s.get("tier"):
        native["tier"] = s["tier"]
    native = native.apply(pd.to_numeric, errors="coerce")
    if cb >= nb:                       # крупніший: максимум за бар
        mapped = native.groupby(native.index.floor(freq)).max().reindex(view.index)
    else:                              # дрібніший: остання оцінка з закриттям ≤ закриття бару графіка
        left = pd.DataFrame({"close": view.index + cb})
        right = native.assign(close=native.index + nb).sort_values("close")
        mapped = pd.merge_asof(left, right, on="close", direction="backward").drop(columns="close")
        mapped.index = view.index
    snap = lambda ts: (pd.Timestamp(ts) + nb - pd.Timedelta(microseconds=1)).floor(freq) if ts is not None else None  # noqa: E731
    pos = dict(res["position"])
    for tr in [pos] + [pos.get("trades", [])][0]:
        for key in ("entry_ts", "exit_ts", "partial_ts"):
            if tr.get(key) is not None:
                tr[key] = snap(tr[key])
    res["ob_marks"] = [{**m, "ts": snap(m["ts"])} for m in res.get("ob_marks", [])]
    res.update({
        "tf": tf, "bar": sec, "native_tf": res["tf"], "position": pos, "last_bar": view.index.max(),
        "series": {
            "t": list(view.index),
            "open": view["open"].tolist(), "price": view["price"].tolist(),
            "high": view["high"].tolist(), "low": view["low"].tolist(),
            "oi": view["oi"].tolist(), "volume": view["volume"].tolist(),
            "coinO": view["oi_coins_o"].tolist(), "coinH": view["oi_coins_h"].tolist(),
            "coinL": view["oi_coins_l"].tolist(), "coin": view["oi_coins_c"].tolist(),
            "usdO": view["oi_usd_o"].tolist(), "usdH": view["oi_usd_h"].tolist(), "usdL": view["oi_usd_l"].tolist(),
            "prob": {k: mapped[f"p_{k}"].tolist() if f"p_{k}" in mapped else [] for k in s["prob"]},
            "tier": mapped["tier"].tolist() if "tier" in mapped else [],
        },
    })
    return res


_fresh_at: dict = {}
_fresh_lock = threading.Lock()


def ensure_fresh(conn, base: str, tf: str) -> bool:
    """Графік відкритого тікера не чекає циклу скринера: якщо в базі бракує останніх закритих 15m/5m-барів,
    дозавантажуємо короткий хвіст з бірж у ФОНІ (~10 с, без Coinalyze) — відповідь не чекає. Повертає True, якщо
    дозавантаження запущено: сторінка тоді перезапитає дані за ~15 с. Не частіше разу на 45 с для тікера й ТФ.
    Модель це не переоцінює — вона рахується скринером на закритих 15m-барах."""
    if tf not in ("5m", "15m"):
        return False
    interval, freq, sec = CHART_TF[tf]
    key = (base, tf)
    with _fresh_lock:
        if time.monotonic() - _fresh_at.get(key, -1e9) < 45:
            return False
        _fresh_at[key] = time.monotonic()
    last = conn.execute("""SELECT max(o.ts) FROM ohlcv o JOIN markets m USING (symbol)
                           WHERE m.base_asset = %s AND o.interval = %s""", (base, interval)).fetchone()[0]
    closed = pd.Timestamp(datetime.now(timezone.utc)).floor(freq) - pd.Timedelta(seconds=sec)
    if last is None or pd.Timestamp(last) >= closed:
        return False
    days = min(2.0, max(0.05, (closed - pd.Timestamp(last)).total_seconds() / 86400 + 0.02))

    def work():
        try:
            with psycopg.connect(State.db_url) as c2:
                fetch_exchange.collect(c2, [base], [tf], days, log=lambda m: None)
        except Exception as e:
            print(f"! дозавантаження {base} {tf}: {e}")
    threading.Thread(target=work, daemon=True).start()
    return True


def live_symbols(conn, base: str) -> dict:
    """Символи USDT-перпів на Binance і Bybit — для живої свічки на сторінці (WebSocket бірж напряму);
    OKX — лише для панелі розбіжності бірж (ціна, без обсягу й OI)."""
    rows = dict(conn.execute("""SELECT exchange_name, symbol_on_exchange FROM markets
                                WHERE base_asset = %s AND quote_asset = 'USDT' AND exchange_name IN ('Binance', 'Bybit', 'OKX')""",
                             (base,)).fetchall())
    return {"binance": rows.get("Binance"), "bybit": rows.get("Bybit"), "okx": rows.get("OKX"),
            "price_exchange": "Binance" if "Binance" in rows else ("Bybit" if "Bybit" in rows else None)}


def exchange_diffs(conn, base: str, res: dict) -> dict | None:
    """Панель «Δ бірж» на графіку (spread.py, 13.10): закриття бару на кожній біржі відносно еталонної біржі графіка
    (Binance, а якщо її немає — Bybit), частка: 0.01 = на 1% дорожче. Бари — ті самі, що на графіку (5m / 15m з бірж,
    1h з Coinalyze). None — тікер торгується лише на одній з трьох бірж."""
    t = res.get("series", {}).get("t") or []
    interval = {"5m": "5min", "15m": "15min", "1h": "1hour"}.get(res.get("tf"))
    if not t or interval is None:
        return None
    rows = conn.execute("""SELECT m.exchange_name, o.ts, o.c FROM ohlcv o JOIN markets m USING (symbol)
                           WHERE m.base_asset = %s AND m.quote_asset = 'USDT' AND m.exchange_name = ANY(%s)
                             AND o.interval = %s AND o.ts >= %s AND o.ts <= %s""",
                        (base, list(spread.EXCHANGES), interval, t[0], t[-1])).fetchall()
    if not rows:
        return None
    w = pd.DataFrame(rows, columns=["ex", "ts", "c"]).pivot_table(index="ts", columns="ex", values="c", aggfunc="last")
    w.index = pd.to_datetime(w.index, utc=True)
    w = w.reindex(pd.DatetimeIndex(t)).astype(float)
    w = w.where(w > 0)
    ref = next((e for e in spread.EXCHANGES if e in w), None)
    others = [e for e in spread.EXCHANGES if e in w and e != ref]
    if ref is None or not others:
        return None
    return {"ref": ref, "diff": {e: (w[e] / w[ref] - 1).tolist() for e in others}}


_ticker_cache: dict = {}                    # (тікер, модель, дні, ТФ, режим, light) → (стан, час, відповідь)
_inflight: dict = {}                        # той самий ключ → threading.Event, поки відповідь рахується
_inflight_lock = threading.Lock()
TICKER_CACHE_SEC = 30


def ticker(conn, base: str, model: str, days: int, tf: str | None = None, strategy: str | None = None,
           light: bool = False) -> dict:
    """Дані графіка тікера. Кеш: та сама відповідь до нових оцінок скринера (signals_stamp) і не довше 30 с —
    перемикання між двома вікнами і назад відкриває тікер миттєво (2026-10-06)."""
    key = (base, model, days, tf, strategy, light)
    state = _list_key()
    c = _ticker_cache.get(key)
    if c is not None and c[0] == state and time.time() - c[1] < TICKER_CACHE_SEC:
        return c[2]
    # однаковий запит, що вже рахується (два вікна, пульс сторінки, бот), — чекаємо його, а не рахуємо вдруге:
    # інакше під навантаженням повільні запити накладались і гальмували один одного
    with _inflight_lock:
        ev = _inflight.get(key)
        owner = ev is None
        if owner:
            ev = _inflight[key] = threading.Event()
    if not owner:
        ev.wait(timeout=90)
        c = _ticker_cache.get(key)
        if c is not None and c[0] == state:
            return c[2]
    try:
        res = _ticker(conn, base, model, days, tf, strategy, light)
        if not res.get("tail_refresh"):             # дані ще дозавантажуються — не кешуємо
            if len(_ticker_cache) > 64:
                _ticker_cache.clear()
            _ticker_cache[key] = (state, time.time(), res)
        return res
    finally:
        if owner:
            with _inflight_lock:
                _inflight.pop(key, None)
            ev.set()


def _ticker(conn, base: str, model: str, days: int, tf: str | None = None, strategy: str | None = None,
            light: bool = False) -> dict:
    strat = strategies.get(strategy) if strategy else None
    if strat is not None:                       # режим задає модель: 15m-режими — tier15, решта — годинні
        model = T.TIER_MODEL if strategies.is15(strat) else strat["model"]
    native = "15m" if model == T.TIER_MODEL else "1h"
    refreshing = ensure_fresh(conn, base, tf or native)
    if model == T.TIER_MODEL:
        if State.m15 is None:
            raise ValueError("15m-моделі немає: python train.py")
        res = ticker_15m(conn, base, days, strat, light)
    else:
        res = ticker_1h(conn, base, model, days)
    res = rebar(conn, res, base, tf, days) if tf else res
    res["live"] = live_symbols(conn, base)
    try:
        res["xs"] = exchange_diffs(conn, base, res)
    except Exception as e:                      # панель розбіжності — доповнення: без неї графік має відкритись
        print(f"! розбіжність бірж {base}: {type(e).__name__}: {e}")
        conn.rollback()
        res["xs"] = None
    res["tail_refresh"] = refreshing
    if refreshing:                              # свіжі бари ще дописуються у фоні — наступний запит має перечитати базу
        _bars_cache.pop((base, days), None)
    res["oi_share"] = None if light else oi_share(conn, base)
    res["top_ls"] = None if light else top_traders(conn, base)
    res["scanner_entries"] = [] if light else scanner_entries(conn, base, days)
    return res


def ticker_1h(conn, base: str, model: str, days: int) -> dict:
    raw = load_raw(conn, [base])
    h = aggregate(raw, base)
    if h.empty:
        raise ValueError(f"немає даних для {base}: додай у watchlist.txt і запусти скринер")
    sig = latest_signals(conn, base)
    probs = {m: g.set_index("ts")["probability"] for m, g in sig.groupby("model")}
    thresholds = {m: float(g["threshold"].iloc[-1]) for m, g in sig.groupby("model")}
    bundle = next((m for m in State.models if m["name"] == model), None)
    thr = thresholds.get(model, bundle["threshold"] if bundle else 0.5)
    k = (bundle or {}).get("risk", {}).get("k")

    last_closed = pd.Timestamp(datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0) - timedelta(hours=1))
    h = h[h.index <= last_closed]
    f = compute_features(h)
    cur = f.dropna(subset=["oi_chg_4h"]).iloc[-1] if f["oi_chg_4h"].notna().any() else None

    p = probs.get(model, pd.Series(dtype=float))
    pos = position_state(h, p, thr, k) if len(p) else {"open": False}
    last = sig[sig["model"] == model].iloc[-1] if (sig["model"] == model).any() else None

    view = h[h.index > h.index.max() - pd.Timedelta(days=days)]
    streak = 0
    for x in reversed(sig[sig["model"] == model]["signal"].tolist()):
        if not x:
            break
        streak += 1
    chg24 = (h["price"].iloc[-1] / h["price"].iloc[-25] - 1) if len(h) > 24 and h["price"].iloc[-25] else None
    return {
        "ticker": base, "model": model, "tf": "1h", "bar": 3600, "threshold": thr, "k": k, "streak": streak,
        "streak_min": streak * 60, "chg_24h": chg24,
        "price": h["price"].iloc[-1] if len(h) else None,
        "ls_long_pct": h["ls_long_pct"].dropna().iloc[-1] if h["ls_long_pct"].notna().any() else None,
        "ls_flow": ls_flow(h),
        "exit_bars": EXIT_BARS, "atr": last_atr(h),
        "last_bar": view.index.max(),
        "in_sample": bool(last["in_sample"]) if last is not None else None,
        "cards": {
            "probability": last["probability"] if last is not None else None,
            "status": status_of(last["probability"] if last is not None else None, thr),
            "oi_chg_4h": math.expm1(cur["oi_chg_4h"]) if cur is not None else None,
            "volume_z_7d": cur["volume_z_7d"] if cur is not None else None,
        },
        "position": pos,
        "series": {
            "t": list(view.index),
            "open": view["open"].tolist(), "price": view["price"].tolist(),
            "high": view["high"].tolist(), "low": view["low"].tolist(),
            "oi": view["oi"].tolist(), "volume": view["volume"].tolist(),
            # свічки OI у монетах (без впливу ціни) — див. features.aggregate
            "coinO": view["oi_coins_o"].tolist(), "coinH": view["oi_coins_h"].tolist(),
            "coinL": view["oi_coins_l"].tolist(), "coin": view["oi_coins_c"].tolist(),
            "usdO": view["oi_usd_o"].tolist(), "usdH": view["oi_usd_h"].tolist(), "usdL": view["oi_usd_l"].tolist(),
            "prob": {m: s.reindex(view.index).tolist() for m, s in probs.items()},
        },
        "thresholds": thresholds,
    }


def refresh(conn, base: str) -> dict:
    """Кнопка «Refresh»: свіжі дані тікера з Coinalyze і нова оцінка обома моделями."""
    if not State.api_key:
        raise ValueError("немає COINALYZE_API_KEY у .env")
    started = time.monotonic()
    last_closed = pd.Timestamp(datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0) - timedelta(hours=1))
    with refresh_lock:
        # Скринер щогодини оновлює watchlist. Якщо остання закрита година вже в базі, в API не йдемо:
        # скринер і дашборд ділять один ліміт Coinalyze, і зайві запити упираються в 429 з очікуванням до хвилини.
        have = conn.execute("""SELECT max(o.ts) FROM ohlcv o JOIN markets m USING (symbol)
                               WHERE m.base_asset = %s AND o.interval = '1hour'""", (base,)).fetchone()[0]
        fetched = have is None or pd.Timestamp(have) < last_closed
        if fetched:
            if State.client is None:
                State.client = Coinalyze(State.api_key)
            if not collect(State.client, conn, [base], FETCH_DAYS, verbose=False):
                raise ValueError(f"{base} не знайдено на біржах")
        scored, last_closed = score_bases(conn, [base], State.models)
        n15 = 0
        if State.m15 is not None:          # 15m: хвіст з бірж і ліквідації (дешево), потім оцінка
            if State.client is None:
                State.client = Coinalyze(State.api_key)
            update_15m(conn, State.client, [base])
            s15, _ = score_15m(conn, [base], State.m15)
            n15 = len(s15)
    if scored.empty and not n15:
        raise ValueError(f"для {base} ще замало історії: потрібно ≥ 7 днів")
    return {"ok": True, "fetched": fetched, "seconds": round(time.monotonic() - started, 1), "last_closed": last_closed}
