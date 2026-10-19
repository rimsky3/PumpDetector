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

Віртуальні моделі (15m-контур, tiers.py): «tier15» — угоди за рівнем «Alert» (15m + підтвердження 1h),
«cand15» — за рівнем «Кандидат» (лише 15m). Вони рахуються на 15m-барах: вихід — EXIT_BARS_15M барів нижче порогу.
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import footprint
import market
import orderbook
import strategies
from features import aggregate, in_pump, load_pumps, load_raw, query_df
from risk import EXIT_BARS, LIMIT_ENTRY, PARTIAL_EXIT, position_state
from tiers import BAR_15M, CAND_MODEL, LIMIT_MODEL, START_MODEL, START_TIER, TIER_MODEL, split_scores, tier_frame

M15 = "lightgbm_15m"          # 15m-модель у таблиці signals
CONFIRM_MODEL = "lightgbm"    # годинна модель, що підтверджує «Alert»

H = pd.Timedelta(hours=1)
DEFAULT_RULE = {"horizon": 24, "price": 0.15, "oi": 0.20}


def is_15m(model: str) -> bool:
    return model in (TIER_MODEL, CAND_MODEL, LIMIT_MODEL, M15, START_TIER, START_MODEL)


def first_scores(conn, model: str, share: float = 0.75, cache: dict | None = None) -> pd.DataFrame:
    """Перша оцінка кожного бару кожного тікера (серед усіх версій моделі).
    Для tier15 / cand15 — оцінка рівня: probability = alert_score або s15 (уже нормовані), threshold = 1.
    cache — спільний словник на один перерахунок журналу: однакові запити й ряди не читаються двічі."""
    if cache is not None:
        key = ("scores", model, share)
        if key not in cache:
            cache[key] = _first_scores(conn, model, share, cache)
        return cache[key]
    return _first_scores(conn, model, share, None)


def _first_scores(conn, model: str, share: float, cache: dict | None) -> pd.DataFrame:
    if model in (TIER_MODEL, CAND_MODEL, LIMIT_MODEL, START_TIER):
        # START_TIER — рівень моделі «старт» (режим split): та сама годинна, свій 15m-поріг
        s15 = first_scores(conn, START_MODEL if model == START_TIER else M15, cache=cache)
        s1h = first_scores(conn, CONFIRM_MODEL, cache=cache)
        if s15.empty:
            return s15
        t = tier_frame(s15, s1h, share)
        score = t["s15"] if model == CAND_MODEL else t["alert_score"]
        return t[["base_asset", "ts", "model_version", "live", "in_sample", "kind"]].assign(
            probability=score.to_numpy(), threshold=1.0, signal=(score >= 1).to_numpy())
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


SERIES_WARMUP = pd.Timedelta(days=5)      # історія до першої оцінки: ATR(14) Уайлдера встигає «забути» старт


def _series(conn, base: str, interval: str, freq: str, cache: dict) -> pd.DataFrame:
    """Зведений ряд тікера (ціна, OI…) для симуляції угод — один на ТФ для всіх моделей журналу.
    Сирі дані беремо ПО ОДНОМУ тікеру і лише з першої оцінки (мінус SERIES_WARMUP), у кеші — лише зведений ряд:
    раніше журнал вантажив повну історію всіх тікерів одразу (~1.4 ГБ піку; сервер з 2 ГБ RAM цього не витримав)."""
    hk = ("h", interval, base)
    if hk not in cache:
        if "first_ts" not in cache:            # перша оцінка кожного тікера будь-якою моделлю
            cache["first_ts"] = {b: pd.Timestamp(t) for b, t in
                                 conn.execute("SELECT base_asset, min(ts) FROM signals GROUP BY 1").fetchall()}
        first = cache["first_ts"].get(base)
        # журналу з ряду потрібні лише свічки ціни (угоди, ATR) і сума OI (автоправило) — «легкий» ряд дає ті самі
        # значення; повний (funding, ліквідації, L/S, свічки OI) забирав 2/3 часу розрахунку журналу (13.10)
        raw = load_raw(conn, [base], interval, since=None if first is None else (first - SERIES_WARMUP).to_pydatetime(), light=True)
        cache[hk] = aggregate(raw, base, freq, light=True)
    return cache[hk]


def build_trades(conn, model: str, bundle: dict | None, rule: dict, cache: dict | None = None,
                 strat: dict | None = None) -> pd.DataFrame:
    """Усі угоди моделі по всіх тікерах з результатом і джерелом результату.
    strat — режим (strategies.py): вхід по ринку чи лімітом, фільтр OB; None — як раніше (за назвою моделі)."""
    scores = first_scores(conn, model, (bundle or {}).get("confirm", {}).get("share", 0.75), cache)
    tf15 = is_15m(model)
    interval, freq, bar = ("15min", "15min", BAR_15M) if tf15 else ("1hour", "h", H)
    exit_bars = (bundle or {}).get("exit_bars", EXIT_BARS)
    cols = ["ticker", "entry_ts", "exit_ts", "open", "kind", "p_max", "result_pct", "reason", "outcome", "source", "entry", "exit"]
    if scores.empty:
        return pd.DataFrame(columns=cols)
    pumps = cache.setdefault("pumps", load_pumps(conn)) if cache is not None else load_pumps(conn)
    cache = cache if cache is not None else {}
    k = (bundle or {}).get("risk", {}).get("k")
    trained = set((bundle or {}).get("tickers", []))
    data_end = (bundle or {}).get("data_end")
    now = pd.Timestamp(datetime.now(timezone.utc))
    rows = []
    for base, g in scores.groupby("base_asset"):
        h = _series(conn, base, interval, freq, cache)
        if h.empty:
            continue
        g = g.set_index("ts").sort_index()
        # версії моделі мають різні пороги: нормуємо ймовірність на поріг, тоді «сигнал» = score ≥ 1
        norm = (g["probability"] / g["threshold"]).reindex(h.index)
        # 15m-контур: частковий вихід; вхід і фільтр — за режимом (strategies.py)
        if strat is not None:
            entry = strategies.entry_rule(strat)
            allow = None
            if strat.get("ob") and tf15:
                try:
                    allow = orderbook.allow_fn(orderbook.history(conn, base, g.index.min().to_pydatetime()))
                except Exception:
                    conn.rollback()
            allow = market.both(allow, market.strategy_allow(conn, strat, 15 if tf15 else 60))   # «ринок не гарячий»
        else:
            entry, allow = (LIMIT_ENTRY if model == LIMIT_MODEL else None), None
        p_marks = None
        if strat is not None and strat.get("p_be") and tf15:          # режим «б/з на p»: мітки p — один запит на всі тікери
            if "p_marks" not in cache:
                cache["p_marks"] = footprint.p_times(conn, None, scores["ts"].min() - pd.Timedelta(days=1))
            p_marks = cache["p_marks"].get(base, set())
        entry_p = None
        if strat is not None and tf15 and strategies.is_split(strat):      # режим split: вхід — «старт», утримання — обидві
            if "start_by_base" not in cache:
                ss = first_scores(conn, START_TIER, (bundle or {}).get("confirm", {}).get("share", 0.75), cache)
                cache["start_by_base"] = {b: x.set_index("ts")["probability"] for b, x in ss.groupby("base_asset")} if not ss.empty else {}
            st = cache["start_by_base"].get(base)
            entry_p, norm = split_scores(norm, st.reindex(h.index) if st is not None else None)
        pos = position_state(h, norm, 1.0, k, exit_bars, PARTIAL_EXIT if tf15 else None, entry, allow, p_marks, entry_prob=entry_p)
        trades = list(pos.get("trades", [])) + ([pos] if pos.get("open") else [])
        for t in trades:
            start = pd.Timestamp(t["entry_ts"])
            end = pd.Timestamp(t["exit_ts"]) + bar if t.get("exit_ts") is not None else now
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


def prepare(conn, bundles: dict, rule: dict) -> dict:
    """Угоди й оцінки всіх моделей / режимів — найдовша частина журналу (~20 с). Від періоду, дат і обсягу оцінок не
    залежить: дашборд кешує результат і лише фільтрує його, тож перемикання періоду — миттєве.
    bundles: {назва: бандл} або {назва режиму: {"scores": назва оцінок, "bundle": бандл, "strat": режим}}."""
    cache: dict = {}
    out = {}
    for name, spec in bundles.items():
        scores_name, bundle, strat = (spec["scores"], spec["bundle"], spec["strat"]) if "scores" in (spec or {}) else (name, spec, None)
        share = (bundle or {}).get("confirm", {}).get("share", 0.75)
        out[name] = (build_trades(conn, scores_name, bundle, rule, cache, strat), first_scores(conn, scores_name, share, cache))
    out["_pumps"] = cache.get("pumps")
    return out


def model_stats(conn, model: str, trades: pd.DataFrame, scores: pd.DataFrame, since, kinds: set, until=None,
                pumps: pd.DataFrame | None = None) -> dict:
    """Зведені метрики моделі за період [since, until): precision за результатами, спіймані розмічені пампи, хибні на 30 днів."""
    bar = BAR_15M if is_15m(model) else H
    until = until if until is not None else pd.Timestamp("2100-01-01", tz="UTC")
    t = trades[(trades["entry_ts"] >= since) & (trades["entry_ts"] < until) & trades["kind"].isin(kinds)]
    decided = t[t["outcome"].isin(["pump", "fp"])]
    n_pump = int((decided["outcome"] == "pump").sum())
    s = scores[(scores["ts"] >= since) & (scores["ts"] < until) & scores["kind"].isin(kinds)] if not scores.empty else scores
    # розмічені пампи на оцінених тікерах у періоді: чи був сигнал усередині, з якою затримкою
    pumps = load_pumps(conn) if pumps is None else pumps
    pumps = pumps[(pumps["pattern_type"] == "pump") & (pumps["start_ts"] >= since) & (pumps["start_ts"] < until)]
    caught, total, delays = 0, 0, []
    for p in pumps.itertuples():
        hours = s[(s["base_asset"] == p.base_asset) & in_pump(s["ts"], p.start_ts, p.end_ts, bar)] if not s.empty else s
        if hours.empty:
            continue                      # памп поза оціненими годинами цього обсягу
        total += 1
        sig = hours[hours["signal"]]
        if len(sig):
            caught += 1
            delays.append(max(0.0, (sig["ts"].min() - p.start_ts) / H))
    ticker_days = (len(s) * (bar / H) / 24) if not s.empty else 0
    fp = int((decided["outcome"] == "fp").sum())
    res = t["result_pct"].dropna()
    return {
        "alerts": int(len(t)),
        "precision": n_pump / len(decided) if len(decided) else None, "pumps": n_pump, "decided": int(len(decided)),
        "caught": caught, "pump_total": total, "delay_h": float(np.median(delays)) if delays else None,
        "avg_trade": float(res.mean()) if len(res) else None,
        "false_per_30d": fp / ticker_days * 30 if ticker_days else None,
    }


def journal(conn, bundles: dict, model: str, days: int | None, kind: str, rule: dict,
            date_from: pd.Timestamp | None = None, date_to: pd.Timestamp | None = None, prepared: dict | None = None) -> dict:
    """date_from / date_to — дні UTC включно (вибір дат у журналі); якщо задано, days не враховується.
    prepared — результат prepare() (кеш дашборду); без нього угоди рахуються тут."""
    if prepared is None:
        prepared = prepare(conn, bundles, rule)
    since = pd.Timestamp(datetime.now(timezone.utc) - timedelta(days=days)) if days else pd.Timestamp("1970-01-01", tz="UTC")
    until = None
    if date_from is not None:
        since = date_from
        until = (date_to if date_to is not None else date_from) + pd.Timedelta(days=1)
    kinds = KIND_SCOPE.get(kind, KIND_SCOPE["live"])
    per_model, trades_sel = {}, None
    for name, spec in bundles.items():
        trades, scores = prepared[name]
        scores_name = spec["scores"] if "scores" in (spec or {}) else name
        per_model[name] = model_stats(conn, scores_name, trades, scores, since, kinds, until, prepared.get("_pumps"))
        if name == model:
            trades_sel = trades
    if trades_sel is None:
        trades_sel = pd.DataFrame(columns=["ticker", "entry_ts", "kind", "outcome", "source", "result_pct"])
    t = trades_sel[(trades_sel["entry_ts"] >= since) & trades_sel["kind"].isin(kinds)]
    if until is not None:
        t = t[t["entry_ts"] < until]
    # astype(bool): на ПОРОЖНІЙ таблиці колонка open має тип object, і t[~t["open"]] pandas читає як вибір КОЛОНОК, а не
    # рядків — виходить таблиця без колонок і KeyError 'result_pct' (19.10: журнал режиму / періоду без угод — помилка 500)
    closed = t[~t["open"].astype(bool)] if "open" in t else t
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
