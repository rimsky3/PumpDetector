"""
dashboard_app/listview.py — список дашборда (/api/list): рядки, секції, сповіщення; рахується в окремому процесі й кешується за станом
даних (див. CLAUDE.md, «Пастки»: лише тікери списку і 3 дні оцінок); міні-графіки вкладки Charts (/api/minicharts).
"""

import math
import os
import threading
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import psycopg

import movers
import orderbook
import scanner
import tiers as T
import wake
from features import INTERVAL_15M, INTRADAY_EXCHANGES, aggregate, load_raw, split_raw
from screener import SIGNALS_STAMP, last_closed_15m, load_model_15m, load_models

from dashboard_app.common import BAR_15M, QUIET_HOURS, SPARK_HOURS, State, episodes, latest_signals, list_sources, list_tickers, share, status_of, tier_rows
from dashboard_app.trades import alert_since, fp_rows, limit_hint, model_trades, ob_at_signal, split_by_trades, trade_state
from dashboard_app.watchlist_ops import add_status


def watch_list_15m(conn) -> dict:
    """Права панель для 15m + 1h: P 15m-моделі, рівень сигналу, міні-графік P за 24 год (96 барів)."""
    # РОЗВАНТАЖЕННЯ СЕРВЕРА (13.10): раніше список рахувався для ВСІХ тікерів, що колись мали оцінки (на сервері — утричі
    # більше, ніж у списку), на всій їх історії, і щохвилини: ~50% одного ядра. Тепер — лише тікери списку (решту
    # watch_list однаково відкидав) і оцінки за LIST_SIGNAL_DAYS днів: угоди й епізоди рахуються на 2 добах барів.
    now = datetime.now(timezone.utc)
    manual, auto = list_sources()
    keep = sorted(set(manual) | set(auto))
    if not keep:
        return {"rows": [], "recent": [], "alerts": []}
    t, troll = tier_rows(conn, rolling=True, bases=keep, days=LIST_SIGNAL_DAYS)
    if t.empty:
        return {"rows": [], "recent": [], "alerts": []}
    raw = load_raw(conn, sorted(t["base_asset"].unique()), INTERVAL_15M, since=now - timedelta(days=2))
    parts = split_raw(raw)                      # сирі ряди по тікерах — один раз, а не фільтр усього raw на кожен тікер
    newest = troll.sort_values("ts").groupby("base_asset").tail(1).set_index("base_asset") if len(troll) else None
    rows, hmap = [], {}
    woke = wake.last_wake()                    # «прокинулись» за останні 30 хв — фіолетова крапка в списку
    woke_day = wake.last_wake(minutes=QUIET_HOURS * 60)     # для «тихо N год»
    for base, g in t.groupby("base_asset"):
        g = g.sort_values("ts")
        last = g.iloc[-1]
        # статус і P — з найсвіжішої оцінки: ковзний бар 5-хв циклу новіший за останній звичайний 15m-бар
        # статус і P — офіційного 15m-бару, як на графіку, в заголовку вікна і в позиції (SYN 06.10: список показував
        # ковзну 0.46, а графік — Alert 0.91 на барі). Ковзна 5-хв оцінка, якщо новіша, — окремим полем roll (підпис «5m»)
        roll = None
        if newest is not None and base in newest.index and newest.at[base, "ts"] > last["ts"]:
            n = newest.loc[base]
            roll = {"p15": n["p15"], "tier": n["tier"], "close": n["ts"] + BAR_15M}
        streak = 0
        for x in reversed(g["tier"].tolist()):
            if x != last["tier"] or x == "calm":
                break
            streak += 1
        # «тихо N год» (бліді рядки — кандидати на «×»): скільки годин без Кандидата / Alert і без «прокидається»
        act = g.loc[g["tier"] != "calm", "ts"]
        last_act = act.max() if len(act) else g["ts"].min()
        # секція NOW у списку: Alert / Candidate тримається ще годину після закриття останнього сигнального бару
        last_sig = {"close": act.max() + BAR_15M, "tier": g.loc[g["ts"] == act.max(), "tier"].iloc[-1]} if len(act) else None
        if base in woke_day:
            last_act = max(last_act, pd.Timestamp(woke_day[base]["at"]))
        quiet_h = (g["ts"].iloc[-1] + BAR_15M - last_act) / pd.Timedelta(hours=1)
        h = aggregate(parts.get(base, raw), base, "15min", light=True)   # лише ціна й OI: зміни за 24 год, ATR для ліміту
        hmap[base] = h
        lt = g["ts"].iloc[-1]                     # спарклайн і зміни за 24 год — на звичайних 15m-барах
        h = h[h.index <= lt] if not h.empty else h
        spark = g.set_index("ts")["p15"].reindex(pd.date_range(lt - 95 * BAR_15M, lt, freq="15min"))
        chg = oi24 = None
        if len(h) > 96:
            p0, p1 = h["price"].iloc[-97], h["price"].iloc[-1]
            chg = p1 / p0 - 1 if p0 else None
            oi24 = math.expm1(h["oi_log_idx"].iloc[-1] - h["oi_log_idx"].iloc[-97])
        rows.append({
            "ticker": base, "probability": last["p15"], "threshold": last["thr15"], "status": last["tier"],
            "streak": streak, "streak_min": streak * 15, "in_sample": bool(last["in_sample"]), "last_bar": last["ts"],
            "spark": spark.tolist(), "chg_24h": chg, "oi_24h": oi24,
            "p1h": last["p1h"], "thr1h": last["thr1h"], "need1h": share() * last["thr1h"] if pd.notna(last["thr1h"]) else None,
            "wake": woke.get(base), "roll": roll, "quiet_h": round(float(quiet_h), 1), "last_sig": last_sig,
        })
    order = {"alert": 0, "candidate": 1, "calm": 2}
    rows.sort(key=lambda r: (order.get(r["status"], 3), -(r["probability"] or 0)))
    t15 = t                                   # лише звичайні 15m-бари — для угод моделі (як позиція на графіку)
    t = pd.concat([t, troll], ignore_index=True) if len(troll) else t     # епізоди — з ковзними: старт на ~5 хв раніше
    al = t.assign(signal=t["tier"] == "alert", probability=t["p15"])
    ep = episodes(al, BAR_15M)
    recent_cut = pd.Timestamp(now - timedelta(hours=6))
    alerts = [e for e in ep if e["kind"] == "live" and e["ts"] >= recent_cut]
    # старт кожного епізоду «Кандидат або Alert» (15m-модель перетнула поріг) — для сповіщень про Кандидатів
    cand = t.assign(signal=t["tier"] != "calm", probability=t["p15"])
    cands = [e for e in episodes(cand, BAR_15M) if e["kind"] == "live" and e["ts"] >= recent_cut]
    # Сповіщення за угодами моделі (2026-10-08, власник: «заходити з моделлю, а не щоразу, коли P вище порогу»):
    # Alert — лише ВХІД моделі (та сама угода, що мітка «Вхід» на графіку); новий спалах Alert, поки угода відкрита, —
    # Re-alert (окремий перемикач, за замовчуванням вимкнений); Кандидат — лише коли модель без позиції і годину тихо.
    active = {r["ticker"] for r in rows if r["status"] != "calm" or (r["last_sig"] and pd.Timestamp(now) - r["last_sig"]["close"] <= pd.Timedelta(hours=1))}
    trades = {b: model_trades(conn, t15[t15["base_asset"] == b], hmap.get(b)) for b in active}
    for r in rows:
        r["trade"] = (trade_state(trades.get(r["ticker"], []), r, now, alert_since(t15[t15["base_asset"] == r["ticker"]]))
                      if r["ticker"] in active else None)
    alerts, realerts, cands, lates = split_by_trades(conn, t15, t, hmap, alerts, cands, recent_cut, trades)
    # для сповіщення: OB у момент сигналу і ліміт −0.5 ATR (ковзний бар 5-хв циклу → останній звичайний 15m-бар до нього)
    for e in alerts:
        hb = hmap.get(e["ticker"])
        try:
            rows_b = orderbook.history(conn, e["ticker"], (recent_cut - timedelta(hours=1)).to_pydatetime())
        except Exception:
            conn.rollback()
            rows_b = []
        e["ob"] = ob_at_signal(rows_b, e["ts"])
        if hb is not None and len(hb):
            base_ts = hb.index[hb.index <= pd.Timestamp(e["ts"])]
            lim = limit_hint(hb, base_ts[-1]) if len(base_ts) else None
            e["limit"] = lim["price"] if lim else None
    # «прокидається» за останні 30 хв — для голосового сповіщення («Volume STRK»)
    wakes = [{"ticker": b, "ts": w["at"], "vol_x": w["vol_x"], "ret15": w["ret15"], "oi1h": w["oi1h"]} for b, w in woke.items()]
    # живі мітки футпрінту b / p за 6 год (footprint.py) — для сповіщень бота; історичні (з архіву) не надсилаються
    fps = [{"ticker": r["base_asset"], **{k: r[k] for k in ("ts", "kind", "close", "stop", "level", "beyond", "beyond_delta", "vol_x", "ctx")}}
           for r in fp_rows(conn, since=pd.Timestamp(now - timedelta(hours=6))) if r.get("live", True)]
    last_fp = {f["ticker"]: f for f in sorted(fps, key=lambda f: f["ts"])}       # мітка рядка: b · Reclaim / p — остання
    for r in rows:
        f = last_fp.get(r["ticker"])
        r["fp"] = {"kind": f["kind"], "at": pd.Timestamp(f["ts"]) + BAR_15M} if f else None
    return {"rows": rows, "recent": ep[:30], "alerts": alerts, "realerts": realerts, "cands": cands, "lates": lates, "wakes": wakes, "fps": fps,
            "minis": mini_charts(conn, t15, hmap, trades)}


MINI_BARS = 192                                # у кеші — дві доби 15m-барів (з них же збирається 1h-вигляд)
MINI_VIEW = {"5m": (96, 300), "15m": (96, 900), "1h": (48, 3600)}     # таймфрейм вкладки «Charts» → (барів, секунд у барі)
MINI_KEYS = ("t", "o", "h", "l", "c", "tier")


def mini_charts(conn, t15: pd.DataFrame, hmap: dict, trades: dict) -> dict:
    """Дані міні-графіків вкладки «Charts» (13.10): останні MINI_BARS закритих 15m-барів кожного тікера списку, рівень сигналу
    на бар, входи моделі й стоп відкритої угоди (основний режим — ті самі угоди, що в колонці State). Рахуються разом зі
    списком з уже завантажених барів, тому додаткових запитів до бази майже немає; у відповідь /api/list не йдуть
    (watch_list їх прибирає) — їх віддає /api/minicharts лише для запитаних тікерів."""
    code = {"alert": 2, "candidate": 1, "calm": 0}
    try:                                        # символ Binance — для живої свічки (один WebSocket на всю сітку)
        syms = dict(conn.execute("""SELECT base_asset, symbol_on_exchange FROM markets
                                    WHERE exchange_name = 'Binance' AND quote_asset = 'USDT' AND base_asset = ANY(%s)""",
                                 (list(hmap),)).fetchall())
    except Exception:
        conn.rollback()
        syms = {}
    closed, out = last_closed_15m(), {}
    for base, g in t15.groupby("base_asset"):
        h = hmap.get(base)
        if h is None or h.empty:
            continue
        v = h.loc[h.index <= closed, ["open", "high", "low", "price"]].dropna().tail(MINI_BARS)
        if v.empty:
            continue
        tier = g.drop_duplicates("ts", keep="last").set_index("ts")["tier"].map(code).reindex(v.index)
        tr = trades.get(base) or []
        last = tr[-1] if tr else None
        out[base] = {"t": list(v.index), "o": v["open"].tolist(), "h": v["high"].tolist(), "l": v["low"].tolist(),
                     "c": v["price"].tolist(), "tier": tier.tolist(), "binance": syms.get(base),
                     "entries": [x["entry_ts"] for x in tr if x["entry_ts"] >= v.index[0]],
                     "stop": last.get("stop") if last and last.get("open") else None}
    return out


def _mini_15m(m: dict, n: int) -> dict:
    t0 = m["t"][-n:][0]
    return {**m, **{k: m[k][-n:] for k in MINI_KEYS}, "entries": [e for e in m["entries"] if e >= t0]}


def _mini_1h(m: dict, n: int) -> dict:
    """Години з 15m-барів кешу: рівень сигналу — найвищий за годину (Alert усередині години не губиться), вхід — на свою годину.
    Остання година може бути неповною — її доповнює жива свічка на сторінці."""
    d = pd.DataFrame({k: m[k] for k in MINI_KEYS[1:]}, index=pd.DatetimeIndex(m["t"]))
    g = d.groupby(d.index.floor("h"))
    a = pd.DataFrame({"o": g["o"].first(), "h": g["h"].max(), "l": g["l"].min(), "c": g["c"].last(), "tier": g["tier"].max()}).tail(n)
    ent = sorted({pd.Timestamp(e).floor("h") for e in m["entries"]})
    return {**m, "t": list(a.index), **{k: a[k].tolist() for k in MINI_KEYS[1:]}, "entries": [e for e in ent if e >= a.index[0]]}


_mini5_cache: dict = {}                        # (тікери) → (час, відповідь): 5m-бари читаються з бази, не частіше разу на 20 с


def _mini_5m(conn, symbols: list[str], minis: dict, n: int) -> dict:
    """5m-вигляд: бари біржі графіка (Binance, інакше Bybit) одним запитом на всі запитані тікери; рівень сигналу і вхід —
    з 15m-кешу (5m-бар бере рівень свого 15m-бару; вхід — на останній 5m-бар сигнального 15m-бару, тобто на його закриття)."""
    key, now = tuple(sorted(symbols)), time.monotonic()
    hit = _mini5_cache.get(key)
    if hit and now - hit[0] < 20:
        return hit[1]
    five = pd.Timedelta(minutes=5)
    closed = pd.Timestamp(datetime.now(timezone.utc)).floor("5min") - five
    rows = conn.execute("""SELECT m.base_asset, m.exchange_name, o.ts, o.o, o.h, o.l, o.c FROM ohlcv o JOIN markets m USING (symbol)
                           WHERE m.base_asset = ANY(%s) AND m.quote_asset = 'USDT' AND m.exchange_name = ANY(%s)
                             AND o.interval = '5min' AND o.ts >= %s""",
                        (symbols, list(INTRADAY_EXCHANGES), (closed - n * five).to_pydatetime())).fetchall()
    out = {}
    if rows:
        df = pd.DataFrame(rows, columns=["base", "ex", "ts", "o", "h", "l", "c"])
        for base, g in df.groupby("base"):
            ex = "Binance" if (g["ex"] == "Binance").any() else g["ex"].iloc[0]
            v = g[g["ex"] == ex].drop_duplicates("ts", keep="last").set_index("ts").sort_index()
            v.index = pd.to_datetime(v.index, utc=True)
            v = v.loc[v.index <= closed, ["o", "h", "l", "c"]].astype(float).dropna().tail(n)
            if v.empty:
                continue
            m = minis[base]
            tier = pd.Series(m["tier"], index=pd.DatetimeIndex(m["t"])).reindex(v.index.floor("15min"))
            ent = [pd.Timestamp(e) + 2 * five for e in m["entries"]]
            out[base] = {**m, "t": list(v.index), **{k: v[k].tolist() for k in MINI_KEYS[1:5]}, "tier": tier.tolist(),
                         "entries": [e for e in ent if e >= v.index[0]]}
    if len(_mini5_cache) > 20:
        _mini5_cache.clear()
    _mini5_cache[key] = (now, out)
    return out


def minicharts(conn, model: str, symbols: list[str], tf: str = "15m") -> dict:
    """/api/minicharts: міні-графіки запитаних тікерів. 15m і 1h — з кешу списку (той самий стан, що й рядки списку),
    5m — ще один запит свічок до бази на всі тікери разом."""
    tf = tf if tf in MINI_VIEW else "15m"
    n, sec = MINI_VIEW[tf]
    minis = _watch_list_cached(conn, model).get("minis") or {}
    have = [s for s in symbols[:80] if s in minis]
    if tf == "5m":
        charts = _mini_5m(conn, have, minis, n) if have else {}
    else:
        charts = {s: (_mini_1h if tf == "1h" else _mini_15m)(minis[s], n) for s in have}
    return {"tf": tf, "bar": sec, "charts": charts}


# ---- список тікерів: важка частина рахується у фоні в окремому процесі одразу після нових оцінок (2026-10-06) ----
# Раніше кожен запит /api/list (сторінка, кожна нова оцінка, щохвилини з кожної вкладки) рахував ~7 с на сервері, а
# сповіщення чекали цього розрахунку. Тепер list_warmer помічає нові оцінки (mtime signals_stamp / wake_events.json)
# і одразу перераховує; запит віддає готове. Нові оцінки → запит чекає свіжий розрахунок (сповіщення мають бути
# актуальні); лише «застаріло за часом» (LIST_MAX_AGE, бо «N хв тому» і 6-год вікна зсуваються) → віддаємо наявне
# і оновлюємо у фоні.
LIST_MAX_AGE = 300        # с: без нових даних список перераховується раз на 5 хв (було 60 — половина ядра сервера на це)
LIST_SIGNAL_DAYS = 3      # скільки днів оцінок кожного тікера читає список (бари для угод — 2 доби); «тихо N год» — не більше цього
LIST_WARM_MODELS = (T.TIER_MODEL,)          # основна модель списку — гріємо завжди; решта — за запитом, з кешем
_list_pool = None
_list_lock = threading.Lock()
_list_cache: dict = {}                      # модель → {"key", "at", "data"}
_list_futures: dict = {}                    # модель → (key, future)


def _list_key() -> tuple:
    mt = lambda p: os.path.getmtime(p) if os.path.exists(p) else 0      # noqa: E731
    # склад списку — теж у ключі: список рахується лише для своїх тікерів, тож доданий тікер має викликати перерахунок
    return mt(SIGNALS_STAMP), mt(wake.EVENTS_FILE), tuple(sorted(list_tickers()))


def _watch_list_in_child(db_url: str, model: str) -> dict:
    """_watch_list в окремому процесі (моделі читає сам, один раз): не тримає процес дашборду."""
    if not State.models and State.m15 is None:
        State.models, State.m15 = load_models(), load_model_15m()
    with psycopg.connect(db_url) as conn:
        return _watch_list(conn, model)


def _list_store(model: str, key: tuple, fut) -> None:
    try:
        data = fut.result()
    except Exception as e:
        print(f"  ! список ({model}) у фоні: {type(e).__name__}: {e}")
        return
    with _list_lock:
        _list_cache[model] = {"key": key, "at": time.time(), "data": data}


def _list_future(model: str, key: tuple):
    """Запустити розрахунок списку для стану key (або взяти вже запущений для того самого стану)."""
    global _list_pool
    with _list_lock:
        f = _list_futures.get(model)
        if f is not None and f[0] == key and not f[1].done():
            return f[1]
        if _list_pool is None:
            from concurrent.futures import ProcessPoolExecutor
            _list_pool = ProcessPoolExecutor(max_workers=1)
        fut = _list_pool.submit(_watch_list_in_child, State.db_url, model)
        _list_futures[model] = (key, fut)
    fut.add_done_callback(lambda ff, m=model, k=key: _list_store(m, k, ff))
    return fut


def _watch_list_cached(conn, model: str) -> dict:
    key = _list_key()
    c = _list_cache.get(model)
    if c is not None and c["key"] == key:
        if time.time() - c["at"] >= LIST_MAX_AGE:
            _list_future(model, key)              # застаріло лише за часом — віддаємо наявне, оновлюємо у фоні
        return c["data"]
    try:
        fut = _list_future(model, key)
        data = fut.result(timeout=180)
        _list_store(model, key, fut)              # колбек міг ще не спрацювати — кладемо в кеш одразу
        return data
    except Exception as e:                        # процес не стартував / впав — рахуємо як раніше, тут
        print(f"  ! список в окремому процесі: {type(e).__name__}: {e} — рахую тут")
        global _list_pool
        _list_pool = None
        return _watch_list(conn, model)


def list_warmer() -> None:
    """Щойно скринер записав нові оцінки або «прокидається» — перерахувати список основної моделі у фоні."""
    while True:
        try:
            key = _list_key()
            for model in LIST_WARM_MODELS:
                c = _list_cache.get(model)
                if c is None or c["key"] != key or time.time() - c["at"] >= LIST_MAX_AGE:
                    _list_future(model, key)
        except Exception as e:
            print(f"  ! список (фон): {e}")
        time.sleep(2)


def watch_list(conn, model: str) -> dict:
    """Права панель: лише тікери з watchlist.txt і живого авто-списку сканера (прострочені авто — геть),
    з позначкою джерела: source manual / auto, причина і до коли тікер у списку.
    Важка частина (_watch_list) — з кешу, що гріється у фоні; джерела й «додається…» — свіжі на кожен запит."""
    res = dict(_watch_list_cached(conn, model))
    res.pop("minis", None)                      # міні-графіки вкладки «Charts» — окремим запитом /api/minicharts
    manual, auto = list_sources()
    keep = set(manual) | set(auto)
    woke = scanner.load_woke()      # події каналу WOKE (і ручних тікерів): first — «Woke · 3h» і година в секції NOW

    def src(b: str) -> dict:
        a = auto.get(b, {})
        w = woke.get(b)
        return {"source": "auto" if b in auto else "manual", "channel": scanner.channel(a) if a else None,
                "reason": a.get("reason"), "expires": a.get("expires"), "held": a.get("held"),
                "wokeup": {"first": w["first"], "last": w["last"], "reason": w.get("first_reason") or w["reason"]} if w else None}
    # Movers (movers.info): рамка в списку — ріст понад ринок, z ≥ 2.5 за 1H / 4H; з файлу щогодини
    res["rows"] = [r | src(r["ticker"]) | {"mover": movers.info(r["ticker"])} for r in res["rows"] if r["ticker"] in keep]
    have = {r["ticker"] for r in res["rows"]}
    res["wakes"] = [w for w in res.get("wakes", []) if w["ticker"] in keep]
    res["fps"] = [f for f in res.get("fps", []) if f["ticker"] in keep]
    res["pending"] = [{"ticker": b, **src(b), "job": add_status(b)} for b in manual + sorted(auto) if b not in have]
    # для бота: перші спрацювання WOKE за 6 год (повторні — не новина)
    cut = pd.Timestamp(datetime.now(timezone.utc) - timedelta(hours=6))
    res["wokeups"] = [{"ticker": b, "ts": w["first"], "reason": w.get("first_reason") or w["reason"],
                       **{k: w.get(k) for k in ("turnover_avg", "pace", "vol_x", "price_pct", "oi_pct")},
                       "source": "auto" if b in auto else "manual"}
                      for b, w in woke.items() if b in keep and pd.Timestamp(w["first"]) >= cut]
    return res


def _watch_list(conn, model: str) -> dict:
    if model == T.TIER_MODEL:
        return watch_list_15m(conn)
    sig = latest_signals(conn)
    sig = sig[sig["model"] == model]
    if sig.empty:
        return {"rows": [], "recent": [], "alerts": []}
    raw = load_raw(conn, sorted(sig["base_asset"].unique()))
    rows = []
    for base, g in sig.groupby("base_asset"):
        g = g.sort_values("ts")
        last = g.iloc[-1]
        streak = 0
        for x in reversed(g["signal"].tolist()):
            if not x:
                break
            streak += 1
        h = aggregate(raw, base, light=True)
        h = h[h.index <= last["ts"]] if not h.empty else h
        spark = g.set_index("ts")["probability"].reindex(pd.date_range(last["ts"] - pd.Timedelta(hours=SPARK_HOURS - 1), last["ts"], freq="h"))
        chg = oi24 = None
        if len(h) > 24:
            p0, p1 = h["price"].iloc[-25], h["price"].iloc[-1]
            chg = p1 / p0 - 1 if p0 else None
            oi24 = math.expm1(h["oi_log_idx"].iloc[-1] - h["oi_log_idx"].iloc[-25])   # без стрибків від лістингів
        rows.append({
            "ticker": base, "probability": last["probability"], "threshold": last["threshold"],
            "status": status_of(last["probability"], last["threshold"]),
            "streak": streak, "in_sample": bool(last["in_sample"]), "last_bar": last["ts"],
            "spark": spark.tolist(), "chg_24h": chg, "oi_24h": oi24,
        })
    rows.sort(key=lambda r: -(r["probability"] or 0))
    ep = episodes(sig)
    # для сповіщень: live-епізоди, що почались за останні 6 годин
    recent_cut = pd.Timestamp(datetime.now(timezone.utc) - timedelta(hours=6))
    alerts = [e for e in ep if e["kind"] == "live" and e["ts"] >= recent_cut]
    return {"rows": rows, "recent": ep[:30], "alerts": alerts}
