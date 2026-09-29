"""
risk.py — вихід з позиції: стоп, trailing stop і сигнал виходу (план з CLAUDE.md).

    Стоп        entry − k·ATR, де k — 75-й перцентиль MAE розмічених пампів в одиницях NATR
    Trailing    найвищий high від входу − 1.5·ATR (страховка)
    Вихід       ймовірність моделі нижче порогу EXIT_BARS барів поспіль

MAE (max adverse excursion) пампу — наскільки ціна просідала нижче ціни входу, перш ніж дійти до піку.
Вхід — закриття першого бару розміченого інтервалу; вимірюємо до бару з максимальним high,
бо обвал після піку — це вже вихід, а не просідання, яке мав витримати стоп.
Ділимо на ATR у момент входу, тож одиниці порівнянні між монетами з різною волатильністю.
"""

import numpy as np
import pandas as pd

from features import aggregate, in_pump, load_pumps, load_raw

ATR_PERIOD = 14         # годин; згладжування Уайлдера
STOP_PERCENTILE = 75
TRAIL_NATR = 1.5
EXIT_BARS = 2           # стільки барів поспіль нижче порогу = вихід
H = pd.Timedelta(hours=1)


def atr(h: pd.DataFrame) -> pd.Series:
    """Average True Range з погодинного ряду features.aggregate() (price = close, high, low)."""
    prev = h["price"].shift(1)
    tr = pd.concat([h["high"] - h["low"], (h["high"] - prev).abs(), (h["low"] - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean()


def pump_mae(conn) -> pd.DataFrame:
    """MAE кожного розміченого пампу в одиницях NATR."""
    pumps = load_pumps(conn)
    pumps = pumps[pumps["pattern_type"] == "pump"]
    if pumps.empty:
        return pd.DataFrame(columns=["base_asset", "start_ts", "mae_natr", "peak_gain_natr"])
    raw = load_raw(conn, sorted(pumps["base_asset"].unique()))
    rows = []
    for base, group in pumps.groupby("base_asset"):
        h = aggregate(raw, base)
        if h.empty:
            continue
        a = atr(h)
        for p in group.itertuples():
            window = h[in_pump(h.index, p.start_ts, p.end_ts)]
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


def stop_multiplier(conn) -> dict:
    """k для стопу: 75-й перцентиль MAE. Повертає також вибірку, щоб було видно, на чому він порахований."""
    mae = pump_mae(conn)
    if mae.empty:
        return {"k": None, "n": 0}
    return {
        "k": float(np.percentile(mae["mae_natr"], STOP_PERCENTILE)),
        "n": int(len(mae)),
        "median_mae_natr": float(mae["mae_natr"].median()),
        "median_peak_gain_natr": float(mae["peak_gain_natr"].median()),
    }


def position_state(h: pd.DataFrame, prob: pd.Series, threshold: float, k: float | None) -> dict:
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
        if below >= EXIT_BARS or stop_hit:
            exit_price = trade["stop"] if stop_hit else float(h.at[ts, "price"])
            trade.update({"exit_ts": ts, "exit": exit_price, "reason": "stop" if stop_hit else "model",
                          "result_pct": exit_price / trade["entry"] - 1})
            closed.append(dict(trade))
            last_closed_trade, trade = trade, None
            below = 0

    last_ts = prob.dropna().index.max() if prob.notna().any() else None
    if trade is not None:
        atr_now = a.get(last_ts, np.nan)
        trade.update({
            "open": True, "below_bars": below,
            "trailing": float(trade["high"] - TRAIL_NATR * atr_now) if not pd.isna(atr_now) else None,
            "hours": int((last_ts - trade["entry_ts"]) / H) + 1,
            "result_pct": float(h.at[last_ts, "price"] / trade["entry"] - 1),
        })
        return {**trade, "trades": closed}
    if last_closed_trade is not None:
        return {**last_closed_trade, "open": False, "trades": closed}
    return {"open": False, "trades": closed}
