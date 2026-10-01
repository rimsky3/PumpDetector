"""
risk.py — вихід з позиції: стоп і сигнал виходу (план з CLAUDE.md; trailing прибрано 2026-10-01 — власник
виходить за стаканом, а на симульовані угоди trailing і так не впливав).

    Стоп        entry − k·ATR, k з бандла моделі: для 15m-контуру — p90 просідання від ФАКТИЧНОГО входу моделі
                (entry_stop, ≈ 2.5 ATR 15m); для годинних моделей — 75-й перцентиль MAE розмічених пампів (stop_multiplier)
    Вихід       сигнал моделі нижче порогу exit_bars барів поспіль (15m: 4 бари = 1 год; 1h: 2 години)

Таймфрейм задають бари h, які передає викликач: на 15m ATR — 14 барів по 15 хв (3.5 год), на 1h — 14 годин.

MAE (max adverse excursion) пампу — наскільки ціна просідала нижче ціни входу, перш ніж дійти до піку.
Вхід — закриття першого бару розміченого інтервалу; вимірюємо до бару з максимальним high,
бо обвал після піку — це вже вихід, а не просідання, яке мав витримати стоп.
Ділимо на ATR у момент входу, тож одиниці порівнянні між монетами з різною волатильністю.
"""

import numpy as np
import pandas as pd

from features import aggregate, in_pump, load_pumps, load_raw

ATR_PERIOD = 14         # барів (15m — 3.5 год, 1h — 14 год); згладжування Уайлдера
STOP_PERCENTILE = 75
EXIT_BARS = 2           # стільки барів поспіль нижче порогу = вихід (1h)
EXIT_BARS_15M = 4       # 15m: година нижче порогу (імпульси короткі, 2 години лагу — половина руху)
H = pd.Timedelta(hours=1)


def atr(h: pd.DataFrame) -> pd.Series:
    """Average True Range з погодинного ряду features.aggregate() (price = close, high, low)."""
    prev = h["price"].shift(1)
    tr = pd.concat([h["high"] - h["low"], (h["high"] - prev).abs(), (h["low"] - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean()


def pump_mae(conn, interval: str = "1hour") -> pd.DataFrame:
    """MAE кожного розміченого пампу в одиницях NATR таймфрейму interval ('1hour' або '15min')."""
    pumps = load_pumps(conn)
    pumps = pumps[pumps["pattern_type"] == "pump"]
    if pumps.empty:
        return pd.DataFrame(columns=["base_asset", "start_ts", "mae_natr", "peak_gain_natr"])
    raw = load_raw(conn, sorted(pumps["base_asset"].unique()), interval)
    freq, bar = ("h", H) if interval == "1hour" else ("15min", pd.Timedelta(minutes=15))
    rows = []
    for base, group in pumps.groupby("base_asset"):
        h = aggregate(raw, base, freq)
        if h.empty:
            continue
        a = atr(h)
        for p in group.itertuples():
            window = h[in_pump(h.index, p.start_ts, p.end_ts, bar)]
            if len(window) < 2 or pd.isna(a.get(window.index[0])) or pd.isna(window["price"].iloc[0]):
                continue
            entry, atr0 = window["price"].iloc[0], a[window.index[0]]
            peak_ts = window["high"].idxmax()
            before_peak = window[(window.index > window.index[0]) & (window.index <= peak_ts)]
            low = before_peak["low"].min() if len(before_peak) else entry
            rows.append({
                "base_asset": base, "start_ts": p.start_ts,
                "mae_natr": max(0.0, (entry - low) / atr0),
                "peak_gain_natr": (window["high"].max() - entry) / atr0,
            })
    return pd.DataFrame(rows)


ENTRY_STOP_PERCENTILE = 90   # 15m: стоп, що не зачепить 9 з 10 відкатів після входу моделі (experiments/stop_from_entry.py)


def entry_stop(tiers: pd.DataFrame, raw: dict, pumps: pd.DataFrame, percentile: float = ENTRY_STOP_PERCENTILE) -> dict:
    """k для 15m-стопу від ФАКТИЧНОГО входу моделі: для кожного розміченого пампу вхід — перший бар Alert
    усередині (tiers — рівні з out-of-fold оцінок, tiers.tier_frame), просідання від ціни входу до піку — в ATR(15m)
    на барі входу. Стоп з розміченого старту (pump_mae) для 15m затісний: модель заходить через ~45 хв, коли
    волатильність у кілька разів більша, і вибивало ~60% угод на звичайних відкатах."""
    bar = pd.Timedelta(minutes=15)
    mae = []
    for base, g in tiers.groupby("base_asset"):
        h = aggregate(raw, base, "15min")
        if h.empty:
            continue
        a, g = atr(h), g.set_index("ts")
        for p in pumps[(pumps["base_asset"] == base) & (pumps["pattern_type"] == "pump")].itertuples():
            inside = g[in_pump(g.index, p.start_ts, p.end_ts, bar) & (g["tier"] == "alert")]
            if inside.empty:
                continue
            e = inside.index[0]
            entry, atr_e = h["price"].get(e), a.get(e)
            w = h[(h.index > e) & in_pump(h.index, p.start_ts, p.end_ts, bar)]
            if pd.isna(entry) or pd.isna(atr_e) or not atr_e or w.empty:
                continue
            low = w.loc[:w["high"].idxmax(), "low"].min()
            mae.append(max(0.0, (entry - low) / atr_e))
    if not mae:
        return {"k": None, "n": 0}
    return {"k": float(np.percentile(mae, percentile)), "n": len(mae), "method": f"p{percentile:g} drawdown from model entry",
            "median_mae_atr": float(np.median(mae)), "p75_mae_atr": float(np.percentile(mae, 75))}


def stop_multiplier(conn, interval: str = "1hour") -> dict:
    """k для стопу: 75-й перцентиль MAE. Повертає також вибірку, щоб було видно, на чому він порахований."""
    mae = pump_mae(conn, interval)
    if mae.empty:
        return {"k": None, "n": 0}
    return {
        "k": float(np.percentile(mae["mae_natr"], STOP_PERCENTILE)),
        "n": int(len(mae)),
        "median_mae_natr": float(mae["mae_natr"].median()),
        "median_peak_gain_natr": float(mae["peak_gain_natr"].median()),
    }


def trade_stats(trades: list[dict]) -> dict:
    """Підсумок закритих угод position_state (відкрита не рахується): кількість, прибуткові,
    середній виграш і програш, очікування на угоду, скільки закрито по стопу."""
    closed = [t for t in trades if not t.get("open") and t.get("result_pct") is not None]
    res = [t["result_pct"] for t in closed]
    wins, losses = [r for r in res if r > 0], [r for r in res if r <= 0]
    avg = lambda xs: sum(xs) / len(xs) if xs else None  # noqa: E731
    return {"n": len(res), "wins": len(wins), "avg_win": avg(wins), "avg_loss": avg(losses), "expectancy": avg(res),
            "stops": sum(1 for t in closed if t.get("reason") == "stop")}


def position_state(h: pd.DataFrame, prob: pd.Series, threshold: float, k: float | None,
                   exit_bars: int = EXIT_BARS) -> dict:
    """Стан «угоди» за сигналами моделі на погодинному ряді тікера.

    Угода починається з першого бару з ймовірністю ≥ порогу і закінчується,
    коли ймовірність EXIT_BARS барів поспіль нижче порогу або ціна торкнулась стопу.
    Повертає поточну угоду (якщо вона відкрита) або останню закриту;
    у ключі "trades" — усі закриті угоди для міток на графіку.
    """
    a = atr(h)
    prob = prob.reindex(h.index)
    trade, last_closed_trade, closed = None, None, []
    below = 0
    for ts, p in prob.items():
        if pd.isna(p):
            continue
        if trade is None:
            if p >= threshold:
                entry, atr0 = h.at[ts, "price"], a.get(ts)
                if pd.isna(entry) or pd.isna(atr0):
                    continue
                trade = {"entry_ts": ts, "entry": float(entry), "atr_entry": float(atr0),
                         "stop": float(entry - k * atr0) if k is not None else None, "high": float(entry)}
                below = 0
            continue
        trade["high"] = max(trade["high"], float(h.at[ts, "high"]))
        below = below + 1 if p < threshold else 0
        stop_hit = trade["stop"] is not None and h.at[ts, "low"] <= trade["stop"]
        if below >= exit_bars or stop_hit:
            exit_price = trade["stop"] if stop_hit else float(h.at[ts, "price"])
            trade.update({"exit_ts": ts, "exit": exit_price, "reason": "stop" if stop_hit else "model",
                          "result_pct": exit_price / trade["entry"] - 1})
            closed.append(dict(trade))
            last_closed_trade, trade = trade, None
            below = 0

    last_ts = prob.dropna().index.max() if prob.notna().any() else None
    if trade is not None:
        trade.update({
            "open": True, "below_bars": below,
            "hours": round((last_ts - trade["entry_ts"]) / H + (h.index[1] - h.index[0]) / H, 2) if len(h) > 1 else 1,
            "result_pct": float(h.at[last_ts, "price"] / trade["entry"] - 1),
        })
        return {**trade, "trades": closed}
    if last_closed_trade is not None:
        return {**last_closed_trade, "open": False, "trades": closed}
    return {"open": False, "trades": closed}
