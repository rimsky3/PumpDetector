"""Позиції власника (my_positions.py): пін конуса при відкритті, добір і частковий вихід його не рухають, закриття знімає,
розворот — новий пін; звірка після перезапуску; запис виконань."""

import my_positions as M

FC = lambda base: (0.02, 0.05)          # noqa: E731 — прогноз Volatility (exp_1, exp_4)


def upd(sym, pa, ep, ps="BOTH"):
    return {"s": sym, "pa": str(pa), "ep": str(ep), "ps": ps}


def test_open_scale_partial_close():
    st = {}
    assert M.apply_positions(st, [upd("AGTUSDT", 100, 0.09)], 1000.0, FC) == ["open AGT long"]
    p = st["AGTUSDT|BOTH"]
    assert p["cone"] == {"at": 1000.0, "p": 0.09, "e1": 0.02, "e4": 0.05} and p["base"] == "AGT"
    assert M.apply_positions(st, [upd("AGTUSDT", 250, 0.092)], 1300.0, FC) == []            # добір: пін на місці
    assert st["AGTUSDT|BOTH"]["cone"]["at"] == 1000.0 and st["AGTUSDT|BOTH"]["qty"] == 250
    assert M.apply_positions(st, [upd("AGTUSDT", 50, 0.092)], 1600.0, FC) == []             # частковий вихід
    assert st["AGTUSDT|BOTH"]["cone"]["p"] == 0.09
    assert M.apply_positions(st, [upd("AGTUSDT", 0, 0)], 1900.0, FC) == ["close AGT"]
    assert st == {}


def test_flip_is_close_and_new_pin():
    st = {}
    M.apply_positions(st, [upd("QUSDT", 10, 1.0)], 100.0, FC)
    ch = M.apply_positions(st, [upd("QUSDT", -5, 1.1)], 200.0, FC)
    assert ch == ["close Q", "open Q short"] and st["QUSDT|BOTH"]["cone"]["at"] == 200.0


def test_reconcile_after_restart():
    """Збережений стан: AGT (ще відкритий) і Q (закрили, поки сервіс лежав); нова позиція S — пін від updateTime, approx."""
    st = {}
    M.apply_positions(st, [upd("AGTUSDT", 100, 0.09), upd("QUSDT", 10, 1.0)], 1000.0, FC)
    snap = [{"s": "AGTUSDT", "pa": "100", "ep": "0.09", "ps": "BOTH", "t": 1000.0},
            {"s": "SUSDT", "pa": "-3", "ep": "0.5", "ps": "BOTH", "t": 1500.0}]
    ch = M.reconcile(st, snap, FC)
    assert "close Q" in ch and "open S short" in ch
    assert st["AGTUSDT|BOTH"]["cone"]["at"] == 1000.0 and not st["AGTUSDT|BOTH"]["approx"]
    assert st["SUSDT|BOTH"]["cone"]["at"] == 1500.0 and st["SUSDT|BOTH"]["approx"]


def test_fill_row_and_save_load(tmp_path):
    o = {"s": "AGTUSDT", "x": "TRADE", "t": 77, "T": 1792450000000, "S": "BUY", "ps": "BOTH", "o": "MARKET",
         "L": "0.09", "l": "100", "rp": "0", "n": "0.0036", "N": "USDT", "i": 5, "m": False, "R": False}
    r = M.fill_row(o)
    assert r[0] == "AGTUSDT" and r[1] == 77 and abs(r[8] - 9.0) < 1e-12 and r[3] == "BUY"
    assert M.fill_row({**o, "x": "NEW"}) is None
    st = {}
    M.apply_positions(st, [upd("AGTUSDT", 100, 0.09)], 1000.0, FC)
    path = str(tmp_path / "pos.json")
    M.save(st, True, path)
    assert M.load_state(path) == st


def test_rest_poll_and_trade_row():
    """Страховка через REST: нова позиція з опитування — пін від updateTime без «~» (approx=False); виконання з userTrades."""
    st = {}
    ch = M.reconcile(st, [{"s": "AGTUSDT", "pa": "100", "ep": "0.0857", "ps": "BOTH", "t": 1500.0}], FC, approx=False)
    assert ch == ["open AGT long"] and st["AGTUSDT|BOTH"]["cone"]["at"] == 1500.0 and not st["AGTUSDT|BOTH"]["approx"]
    t = {"symbol": "AGTUSDT", "id": 9, "orderId": 3, "side": "BUY", "price": "0.0857", "qty": "100", "quoteQty": "8.57",
         "realizedPnl": "0", "commission": "0.003", "commissionAsset": "USDT", "time": 1792459293714, "positionSide": "BOTH",
         "maker": False}
    r = M.trade_row(t)
    assert r[0] == "AGTUSDT" and r[1] == 9 and r[6] == 0.0857 and r[8] == 8.57 and r[5] is None
