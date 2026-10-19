"""Журнал (journal.journal): підсумок не падає на режимі / періоді без угод (19.10: `KeyError: 'result_pct'` — на порожній
таблиці t[~t["open"]] pandas читав як вибір колонок, бо колонка open мала тип object)."""

import pandas as pd

import journal as J


COLS = ["ticker", "entry_ts", "exit_ts", "open", "kind", "p_max", "result_pct", "reason", "outcome", "source", "entry", "exit"]


def run(trades):
    scores = pd.DataFrame(columns=["ts", "kind"])
    pumps = pd.DataFrame(columns=["base_asset", "start_ts", "end_ts", "pattern_type"])
    return J.journal(None, {"split": {}}, "split", 30, "live", {}, prepared={"split": (trades, scores), "_pumps": pumps})


def test_summary_mode_without_trades():
    """Як build_trades для режиму без угод: колонки є, рядків немає (локально — split 19.10)."""
    out = run(pd.DataFrame(columns=COLS))
    assert out["trades"]["n"] == 0 and out["rows"] == []


def test_summary_counts_closed_only():
    ts = pd.Timestamp.now(tz="UTC").floor("15min") - pd.Timedelta(hours=3)
    rows = [{"ticker": "X", "entry_ts": ts, "open": False, "kind": "live", "result_pct": 0.02, "outcome": "pump", "source": "label"},
            {"ticker": "Y", "entry_ts": ts, "open": True, "kind": "live", "result_pct": 0.01, "outcome": None, "source": "live"}]
    out = run(pd.DataFrame(rows, columns=COLS))
    assert out["trades"]["n"] == 1 and out["trades"]["win"] == 1

