"""
dashboard_app/trades.py — угоди моделі для списку і сповіщень: угоди основного режиму (model_trades), стан угоди в рядку списку
(trade_state), епізод Alert (alert_since), сповіщення за входами (split_by_trades), стоп наживо на 5m (live_stop),
підказки для сповіщень (стакан у момент сигналу, ліміт −0.5 ATR), мітки футпрінту.
"""

from datetime import datetime

import pandas as pd

import footprint
import market
import orderbook
import strategies
import tiers as T
from features import aggregate, load_raw
from risk import PARTIAL_EXIT, atr, intrabar_stop, position_state

from dashboard_app.common import BAR_15M, State, start_scores


def model_trades(conn, g: pd.DataFrame, h: pd.DataFrame | None) -> list[dict]:
    """Угоди моделі на тікері за основним режимом — та сама risk.position_state, що позиція й мітки «Вхід» на графіку
    (ticker_15m); g — рядки tier_frame тікера (звичайні 15m-бари), h — 15m-бари. Закриті + відкрита (з "open")."""
    if h is None or h.empty or g.empty:
        return []
    m15, strat = State.m15, strategies.get(None)
    score, entry_p = g.set_index("ts").sort_index()["alert_score"], None
    if strategies.is_split(strat):                  # основний режим split: вхід — модель «старт», утримання — обидві
        entry_p, score = T.split_scores(score, start_scores(conn, g["base_asset"].iloc[0]))
    pos = position_state(h, score, 1.0, m15.get("risk", {}).get("k"),
                         m15.get("exit_bars", 4), PARTIAL_EXIT, strategies.entry_rule(strat),
                         market.strategy_allow(conn, strat), entry_prob=entry_p)
    pos = live_stop(conn, g["base_asset"].iloc[0], pos, h)
    return list(pos["trades"]) + ([pos] if pos.get("open") else [])


def live_stop(conn, base: str, pos: dict, h: pd.DataFrame) -> dict:
    """Відкрита угода + 5m-бари, що закрились після останнього 15m-бару в h: стоп спрацьовує одразу, як його торкнувся
    5m-бар (risk.intrabar_stop), — список, заголовок і графік показують вихід на ~5 хв після події, а не після закриття
    15m-бару. 5m-хвіст усього списку скринер тягне кожні 5 хв. Помилка читання — стан без змін."""
    if not pos.get("open") or pos.get("stop") is None or h is None or h.empty:
        return pos
    try:
        after = h["price"].dropna().index.max() + BAR_15M
        h5 = aggregate(load_raw(conn, [base], "5min", since=after), base, "5min", light=True)
        return pos if h5.empty else intrabar_stop(pos, h5.loc[h5.index >= after, "low"].dropna(), BAR_15M, PARTIAL_EXIT)
    except Exception as e:
        conn.rollback()
        print(f"! стоп наживо {base}: {type(e).__name__}: {e}")
        return pos


def _in_trade(tr: list[dict], ts) -> dict | None:
    """Угода, відкрита на момент ts (вхід раніше, вихід — не раніше за ts)."""
    ts = pd.Timestamp(ts)
    return next((x for x in tr if x["entry_ts"] < ts and (x.get("exit_ts") is None or ts <= x["exit_ts"])), None)


def split_by_trades(conn, t15: pd.DataFrame, t: pd.DataFrame, hmap: dict, alerts: list[dict], cands: list[dict],
                    recent_cut: pd.Timestamp, trades: dict | None = None) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Епізоди Alert → (входи моделі, Re-alert, Кандидати, late); Кандидати — без тих, що всередині угоди або після
    активності за годину. Вхід, що не почав нового епізоду (повторний вхід після стопу посеред безперервного Alert),
    додається окремо. Режим split (основний): новий епізод Alert — НЕ вхід сам по собі; вхід — лише угода моделі
    (вхід за моделлю «старт»), а епізод Alert без угоди — у late (без звуку; список — чіп Late, Telegram — ⚪ Late, 12.10).
    trades — {тікер: угоди основного режиму} (model_trades), щоб не рахувати двічі."""
    split = strategies.is_split(strategies.get(None))
    bases = {e["ticker"] for e in alerts + cands}
    trades = dict(trades or {})
    for b in bases - set(trades):
        trades[b] = model_trades(conn, t15[t15["base_asset"] == b], hmap.get(b))
    entries, re, late = [], [], []
    for e in alerts:
        cur = _in_trade(trades.get(e["ticker"], []), e["ts"])
        if cur is None:
            (late if split else entries).append(e)
        else:
            re.append({**e, "trade": {"entry_ts": cur["entry_ts"] + BAR_15M, "entry": cur["entry"],
                                      "result_pct": cur.get("result_pct"), "open": bool(cur.get("open"))}})
    for b, tr in trades.items():
        g = t15[t15["base_asset"] == b].set_index("ts")
        for x in tr:
            ets = x["entry_ts"]
            if ets < recent_cut or ets not in g.index or not bool(g.at[ets, "live"]):
                continue
            # ковзне вікно (зсув 5–10 хв) могло почати епізод раніше за бар входу — той самий вхід, той самий ключ
            if any(e["ticker"] == b and ets - 2 * BAR_15M <= e["ts"] <= ets for e in entries):
                continue
            entries.append({"ticker": b, "ts": ets, "probability": float(g.at[ets, "p15"]), "kind": "live", "reentry": not split})
    if split:                                   # епізод, у якому модель таки увійшла (бар входу поруч), — не late
        late = [e for e in late if not any(x["ticker"] == e["ticker"] and abs(x["ts"] - e["ts"]) <= 2 * BAR_15M for x in entries)]
    act = t[t["tier"] != "calm"]
    def quiet_before(e) -> bool:
        a = act[(act["base_asset"] == e["ticker"]) & (act["ts"] >= e["ts"] - pd.Timedelta(hours=1)) & (act["ts"] < e["ts"] - BAR_15M)]
        return a.empty
    cands = [e for e in cands if _in_trade(trades.get(e["ticker"], []), e["ts"]) is None and quiet_before(e)]
    key = lambda e: e["ts"]                                                       # noqa: E731
    return sorted(entries, key=key, reverse=True), sorted(re, key=key, reverse=True), cands, sorted(late, key=key, reverse=True)


def alert_since(g: pd.DataFrame) -> pd.Timestamp | None:
    """Початок поточного (останнього) епізоду Alert тікера: бари Alert з проміжками ≤ 1 год — один епізод
    (як train.signal_episodes). g — рядки tier_frame тікера (звичайні 15m-бари)."""
    a = g.loc[g["tier"] == "alert", "ts"].sort_values().to_list()
    if not a:
        return None
    start = a[-1]
    for ts in reversed(a[:-1]):
        if start - ts > pd.Timedelta(hours=1):
            break
        start = ts
    return start


def trade_state(tr: list[dict], row: dict, now: datetime, since: pd.Timestamp | None = None) -> dict | None:
    """Стан рядка списку за угодами основного режиму (власник 12.10, варіант A): entry — модель увійшла за останню годину і
    угода відкрита; open — угода відкрита давніше; closed — Alert ще горить, але модель у ЦЬОМУ епізоді вже відторгувала
    і вийшла (стоп або вихід за моделлю; since — початок епізоду, alert_since; KGEN 12.10: після стопу рядок показував «Late»);
    late — Alert зараз (або згас < 1 год), а угоди в цьому епізоді не було (режим не увійшов: не свіжий старт — split,
    фільтр режиму). None — нічого з цього (рядок показує звичайний статус)."""
    op = next((x for x in tr if x.get("open")), None)
    if op is not None:
        entered = pd.Timestamp(op["entry_ts"]) + BAR_15M
        recent = pd.Timestamp(now) - entered <= pd.Timedelta(hours=1)
        return {"state": "entry" if recent else "open", "entry_at": entered, "entry": op.get("entry"),
                "result_pct": op.get("result_pct")}
    sig = row.get("last_sig")
    alert_now = row.get("status") == "alert" or (sig and sig["tier"] == "alert" and pd.Timestamp(now) - sig["close"] <= pd.Timedelta(hours=1))
    if alert_now:
        last = max(tr, key=lambda x: x["entry_ts"]) if tr else None
        if last is not None and since is not None and last.get("exit_ts") is not None and pd.Timestamp(last["exit_ts"]) >= since:
            return {"state": "closed", "reason": last.get("reason"), "result_pct": last.get("result_pct"),
                    "entry": last.get("entry"), "exit": last.get("exit"), "exit_at": pd.Timestamp(last["exit_ts"]) + BAR_15M}
        return {"state": "late", "last_exit": last.get("exit_ts") + BAR_15M if last and last.get("exit_ts") is not None else None,
                "last_reason": last.get("reason") if last else None}
    return None


def fp_rows(conn, bases: list[str] | None = None, since: pd.Timestamp | None = None, outcomes: bool = False) -> list[dict]:
    """Мітки b / p з fp_marks (порожньо, якщо таблиці ще немає); outcomes — ще й результат «+2R / стоп / 12 год»."""
    try:
        m = footprint.marks(conn, bases, since)
        if outcomes and len(m):
            o = footprint.with_outcomes(conn, m)
            m = o if len(o) else m
        return m.to_dict("records")
    except Exception as e:
        conn.rollback()
        if "fp_marks" not in str(e):
            print(f"! мітки b/p: {e}")
        return []


LIMIT_ATR, LIMIT_VALID = 0.5, pd.Timedelta(hours=1)      # experiments/limit_entry.py: стабільний ліміт −0.5 ATR, 1 год


def alert_starts(t: pd.DataFrame) -> list[pd.Timestamp]:
    """Бари, на яких почався епізод Alert (рівень alert, попередній бар — ні)."""
    al = (t["tier"] == "alert").to_numpy() if len(t) else []
    return [t.index[i] for i in range(len(al)) if al[i] and (i == 0 or not al[i - 1])]


def ob_at_signal(rows, ts: pd.Timestamp) -> float | None:
    """OB у момент сигналу: перший знімок стакану за ≤ 12 хв після закриття сигнального бару (скринер пише після оцінки)."""
    return orderbook.at_signal(rows, ts)


def limit_hint(h: pd.DataFrame, ts: pd.Timestamp) -> dict | None:
    """Ліміт −0.5 ATR від закриття сигнального бару, дійсний 1 год після закриття; None — бару немає."""
    if ts not in h.index:
        return None
    a = atr(h).get(ts)
    if a is None or pd.isna(a):
        return None
    return {"price": float(h.at[ts, "price"] - LIMIT_ATR * a), "until": pd.Timestamp(ts) + BAR_15M + LIMIT_VALID}
