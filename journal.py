"""
journal.py — журнал сигналів і forward-статистика для дашборду (екран «Журнал»).

Джерело — таблиця signals. Для кожної години тікера береться ПЕРША оцінка моделі (найраніший scored_at
серед усіх версій): саме те, що модель сказала в реальному часі. Пізніші перенавчання, які дооцінюють
минулі години, історію не переписують.

Угода — за правилами risk.py: вхід на першій годині з сигналом, вихід — ймовірність нижче порогу
EXIT_BARS години поспіль або стоп. Результат угоди:
    розмітка — інтервал угоди перетинає розмічений памп (pump) чи фальстарт (false_start / other);
               для тікерів з навчання години до кінця навчальних даних теж вважаються розміченими
               (там, де памп не розмічено, його не було);
    авто     — розмітки немає: за HORIZON годин після входу ціна і OI виросли не менше порогів → памп;
    очікує   — розмітки немає і горизонт автоправила ще не минув.
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from features import aggregate, load_pumps, load_raw, query_df
from risk import position_state

H = pd.Timedelta(hours=1)
DEFAULT_RULE = {"horizon": 24, "price": 0.15, "oi": 0.20}


def first_scores(conn, model: str) -> pd.DataFrame:
    """Перша оцінка кожної години кожного тікера (серед усіх версій моделі)."""
    df = query_df(conn, """
        SELECT DISTINCT ON (base_asset, ts) base_asset, ts, model_version, probability, threshold, signal, live, in_sample
        FROM signals WHERE model = %s
        ORDER BY base_asset, ts, scored_at""", (model,))
    if not df.empty:
        df["kind"] = np.where(df["live"], "live", np.where(df["in_sample"], "in-sample", "history"))
    return df


def auto_outcome(h: pd.DataFrame, entry_ts: pd.Timestamp, rule: dict, now: pd.Timestamp) -> str:
    """Автоправило: чи виросли ціна і OI за горизонт після входу. 'wait', якщо горизонт ще не минув."""
    end = entry_ts + rule["horizon"] * H
    if end > now:
        return "wait"
    w = h[(h.index >= entry_ts) & (h.index < end)]
    if w.empty or pd.isna(w["price"].iloc[0]) or pd.isna(w["oi"].iloc[0]) or not w["oi"].iloc[0]:
        return "fp"
    price_up = w["high"].max() / w["price"].iloc[0] - 1
    oi_up = w["oi"].max() / w["oi"].iloc[0] - 1
    return "pump" if price_up >= rule["price"] and oi_up >= rule["oi"] else "fp"


def label_outcome(pumps: pd.DataFrame, base: str, start: pd.Timestamp, end: pd.Timestamp,
                  trained: set, data_end) -> str | None:
    own = pumps[(pumps["base_asset"] == base) & (pumps["start_ts"] < end) & (pumps["end_ts"] > start)]
    if (own["pattern_type"] == "pump").any():
        return "pump"
    if len(own):
        return "fp"
    if base in trained and data_end is not None and end <= data_end:
        return "fp"          # тікер розмічено повністю до кінця навчальних даних: пампу тут не було
    return None


def build_trades(conn, model: str, bundle: dict | None, rule: dict) -> pd.DataFrame:
    """Усі угоди моделі по всіх тікерах з результатом і джерелом результату."""
    scores = first_scores(conn, model)
    cols = ["ticker", "entry_ts", "exit_ts", "open", "kind", "p_max", "result_pct", "reason", "outcome", "source", "entry", "exit"]
    if scores.empty:
        return pd.DataFrame(columns=cols)
    pumps = load_pumps(conn)
    raw = load_raw(conn, sorted(scores["base_asset"].unique()))
    k = (bundle or {}).get("risk", {}).get("k")
    trained = set((bundle or {}).get("tickers", []))
    data_end = (bundle or {}).get("data_end")
    now = pd.Timestamp(datetime.now(timezone.utc))
    rows = []
    for base, g in scores.groupby("base_asset"):
        h = aggregate(raw, base)
        if h.empty:
            continue
        g = g.set_index("ts").sort_index()
        # версії моделі мають різні пороги: нормуємо ймовірність на поріг, тоді «сигнал» = score ≥ 1
        norm = (g["probability"] / g["threshold"]).reindex(h.index)
        pos = position_state(h, norm, 1.0, k)
        trades = list(pos.get("trades", [])) + ([pos] if pos.get("open") else [])
        for t in trades:
            start = pd.Timestamp(t["entry_ts"])
            end = pd.Timestamp(t["exit_ts"]) + H if t.get("exit_ts") is not None else now
            span = g[(g.index >= start) & (g.index < end)]
            lab = label_outcome(pumps, base, start, end, trained, data_end)
            outcome, source = (lab, "label") if lab else (auto_outcome(h, start, rule, now), "auto")
            if outcome == "wait":
                source = "wait"
            rows.append({
                "ticker": base, "entry_ts": start, "exit_ts": t.get("exit_ts"), "open": bool(t.get("open")),
                "kind": g["kind"].get(start, "history"), "p_max": float(span["probability"].max()) if len(span) else None,
                "result_pct": t.get("result_pct"), "reason": t.get("reason"), "outcome": outcome, "source": source,
                "entry": t.get("entry"), "exit": t.get("exit"),
            })
    return pd.DataFrame(rows, columns=cols).sort_values("entry_ts", ascending=False).reset_index(drop=True)


KIND_SCOPE = {"live": {"live"}, "hist": {"live", "history"}, "all": {"live", "history", "in-sample"}}


def model_stats(conn, model: str, trades: pd.DataFrame, scores: pd.DataFrame, since, kinds: set) -> dict:
    """Зведені метрики моделі за період: precision за результатами, спіймані розмічені пампи, хибні на 30 днів."""
    t = trades[(trades["entry_ts"] >= since) & trades["kind"].isin(kinds)]
    decided = t[t["outcome"].isin(["pump", "fp"])]
    n_pump = int((decided["outcome"] == "pump").sum())
    s = scores[(scores["ts"] >= since) & scores["kind"].isin(kinds)] if not scores.empty else scores
    # розмічені пампи на оцінених тікерах у періоді: чи був сигнал усередині, з якою затримкою
    pumps = load_pumps(conn)
    pumps = pumps[(pumps["pattern_type"] == "pump") & (pumps["start_ts"] >= since)]
    caught, total, delays = 0, 0, []
    for p in pumps.itertuples():
        hours = s[(s["base_asset"] == p.base_asset) & (s["ts"] >= p.start_ts) & (s["ts"] < p.end_ts)] if not s.empty else s
        if hours.empty:
            continue                      # памп поза оціненими годинами цього обсягу
        total += 1
        sig = hours[hours["signal"]]
        if len(sig):
            caught += 1
            delays.append((sig["ts"].min() - p.start_ts) / H)
    ticker_days = (len(s) / 24) if not s.empty else 0
    fp = int((decided["outcome"] == "fp").sum())
    res = t["result_pct"].dropna()
    return {
        "alerts": int(len(t)),
        "precision": n_pump / len(decided) if len(decided) else None, "pumps": n_pump, "decided": int(len(decided)),
        "caught": caught, "pump_total": total, "delay_h": float(np.median(delays)) if delays else None,
        "avg_trade": float(res.mean()) if len(res) else None,
        "false_per_30d": fp / ticker_days * 30 if ticker_days else None,
    }


def journal(conn, bundles: dict, model: str, days: int | None, kind: str, rule: dict) -> dict:
    since = pd.Timestamp(datetime.now(timezone.utc) - timedelta(days=days)) if days else pd.Timestamp("1970-01-01", tz="UTC")
    kinds = KIND_SCOPE.get(kind, KIND_SCOPE["live"])
    per_model, trades_sel = {}, None
    for name, bundle in bundles.items():
        trades = build_trades(conn, name, bundle, rule)
        scores = first_scores(conn, name)
        per_model[name] = model_stats(conn, name, trades, scores, since, kinds)
        if name == model:
            trades_sel = trades
    if trades_sel is None:
        trades_sel = pd.DataFrame(columns=["ticker", "entry_ts", "kind", "outcome", "source", "result_pct"])
    t = trades_sel[(trades_sel["entry_ts"] >= since) & trades_sel["kind"].isin(kinds)]
    closed = t[~t["open"]] if "open" in t else t
    res = closed["result_pct"].dropna()
    by_ticker = []
    for tk, g in t[t["outcome"].isin(["pump", "fp"])].groupby("ticker"):
        by_ticker.append({"ticker": tk, "pumps": int((g["outcome"] == "pump").sum()), "total": int(len(g))})
    by_ticker.sort(key=lambda r: (-r["pumps"] / r["total"], -r["total"]))
    return {
        "models": per_model,
        "rows": t.to_dict("records"),
        "trades": {
            "results": list(reversed(res.tolist()))[-40:],   # хронологічно, останні 40
            "win": int((res > 0).sum()), "n": int(len(res)),
            "avg_win": float(res[res > 0].mean()) if (res > 0).any() else None,
            "avg_loss": float(res[res <= 0].mean()) if (res <= 0).any() else None,
            "stops": int((closed["reason"] == "stop").sum()) if "reason" in closed else 0,
        },
        "by_ticker": by_ticker,
        "labels": {
            "total": int(len(t)), "label": int((t["source"] == "label").sum()),
            "auto": int((t["source"] == "auto").sum()), "wait": int((t["source"] == "wait").sum()),
        },
        "rule": rule,
    }
