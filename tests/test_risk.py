"""Правила угоди моделі (risk.position_state, intrabar_stop): вхід за закриттям, стоп, вихід за моделлю, частковий вихід."""

import numpy as np
import pandas as pd

import risk


def prob_series(index, on: int, off_from: int | None = None, p_on: float = 0.9, p_off: float = 0.1):
    """Ймовірність: нижче порогу до бару on, вище з on, знову нижче з off_from."""
    p = pd.Series(p_off, index=index)
    p.iloc[on:] = p_on
    if off_from is not None:
        p.iloc[off_from:] = p_off
    return p


def test_atr_constant_range(hourly):
    a = risk.atr(hourly)
    assert np.isclose(a.dropna().iloc[-1], 1.0)


def test_entry_at_close_and_stop(hourly):
    """Вхід — за закриттям сигнального бару; стоп = вхід − k·ATR; лоу нижче стопу → вихід за ціною стопу."""
    h = hourly.copy()
    h.iloc[22, h.columns.get_loc("low")] = 97.9
    pos = risk.position_state(h, prob_series(h.index, on=20), threshold=0.5, k=2.0)
    t = pos["trades"][0]
    assert t["entry_ts"] == h.index[20] and t["entry"] == 100.0
    assert t["exit_ts"] == h.index[22] and t["reason"] == "stop" and t["exit"] == 98.0
    assert np.isclose(t["result_pct"], -0.02)


def test_exit_after_bars_below_threshold(hourly):
    """Вихід за моделлю — коли ймовірність exit_bars барів поспіль нижче порогу, за закриттям останнього."""
    h = hourly.copy()
    h["price"] = 100.0 + np.arange(len(h)) * 0.1                   # ціна повільно росте
    pos = risk.position_state(h, prob_series(h.index, on=20, off_from=24), threshold=0.5, k=5.0, exit_bars=2)
    t = pos["trades"][0]
    assert t["reason"] == "model" and t["exit_ts"] == h.index[25]
    assert np.isclose(t["result_pct"], h["price"].iloc[25] / h["price"].iloc[20] - 1)


def test_partial_exit_then_breakeven(hourly):
    """½ на +3 ATR, далі стоп у беззбиток: відкат до входу дає +1.5% на всю позицію, а не збиток."""
    h = hourly.copy()
    h.iloc[21, h.columns.get_loc("high")] = 103.2                  # ціль 100 + 3·1 = 103
    h.iloc[23, h.columns.get_loc("low")] = 99.9                    # відкат нижче входу → стоп у беззбиток
    pos = risk.position_state(h, prob_series(h.index, on=20), threshold=0.5, k=2.0, partial=risk.PARTIAL_EXIT)
    t = pos["trades"][0]
    assert t["partial_ts"] == h.index[21] and t["partial_price"] == 103.0
    assert t["reason"] == "stop" and t["exit"] == 100.0
    assert np.isclose(t["result_pct"], 0.5 * 0.03)


def test_stop_checked_before_target(hourly):
    """У барі, де зачеплено і стоп, і ціль, рахуємо стоп (консервативно)."""
    h = hourly.copy()
    h.iloc[21, h.columns.get_loc("high")] = 104.0
    h.iloc[21, h.columns.get_loc("low")] = 97.0
    t = risk.position_state(h, prob_series(h.index, on=20), threshold=0.5, k=2.0, partial=risk.PARTIAL_EXIT)["trades"][0]
    assert t["reason"] == "stop" and t["partial_ts"] is None and np.isclose(t["result_pct"], -0.02)


def test_open_trade_reported(hourly):
    pos = risk.position_state(hourly, prob_series(hourly.index, on=30), threshold=0.5, k=2.0)
    assert pos["open"] and pos["entry_ts"] == hourly.index[30] and pos["trades"] == []


def test_intrabar_stop_closes_before_bar_close(hourly):
    """Лоу 5m-бару торкнувся стопу — угода закрита одразу за ціною стопу (той самий результат, що дасть закритий бар)."""
    pos = risk.position_state(hourly, prob_series(hourly.index, on=30), threshold=0.5, k=2.0)
    lows = pd.Series([99.0, 97.5], index=pd.date_range(hourly.index[-1] + pd.Timedelta(hours=1), periods=2, freq="5min"))
    out = risk.intrabar_stop(pos, lows, pd.Timedelta(hours=1))
    assert not out["open"] and out["intrabar"] and out["exit"] == pos["stop"]
    assert np.isclose(out["result_pct"], pos["stop"] / pos["entry"] - 1)
    assert risk.intrabar_stop(pos, lows.iloc[:1], pd.Timedelta(hours=1))["open"]      # лоу вище стопу — без змін
