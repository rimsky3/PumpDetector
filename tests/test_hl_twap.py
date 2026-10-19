"""TWAP Hyperliquid (hl_twap.py): пошук порцій кожні 30 с, підтвердження з історії адреси, виконання, кінець."""

import time

import hl_twap as H


def watcher():
    w = H.TwapWatcher()
    w.coins = ["PUMP", "kPEPE"]
    w.binance = {"PUMPUSDT", "1000PEPEUSDT"}
    return w


def test_binance_name():
    assert H.binance_name("kPEPE") == "1000PEPE" and H.binance_name("PUMP") == "PUMP"


def test_slices_every_30s_queue_address_once():
    w = watcher()
    t0 = time.time() - 200
    for k in range(5):
        w.add_fill("PUMP", "0xabc", "A", t0 + 30 * k, 0.004, 1000)
        w.add_fill("PUMP", "0xabc", "A", t0 + 30 * k + 1, 0.004, 500)    # та сама порція (≤ 3 с)
        w.add_fill("PUMP", "0xrnd", "B", t0 + 7 * k * k, 0.004, 100)    # нерегулярні угоди — не TWAP
    assert w.queue.qsize() == 1 and w.queue.get_nowait() == "0xabc"
    assert len(w.slices[("PUMP", "0xabc", "A")]) == 5


def test_adopt_active_only_and_track_execution():
    w = watcher()
    now = time.time()
    st = lambda coin, side, ts, mins: {"coin": coin, "side": side, "sz": "1000000", "minutes": mins, "reduceOnly": False,   # noqa: E731
                                       "randomize": False, "timestamp": int(ts * 1000)}
    hist = [
        {"twapId": 1, "status": {"status": "activated"}, "state": st("PUMP", "A", now - 600, 60)},      # активний
        {"twapId": 2, "status": {"status": "activated"}, "state": st("PUMP", "B", now - 900, 30)},
        {"twapId": 2, "status": {"status": "finished"}, "state": st("PUMP", "B", now - 900, 30)},       # завершений
        {"twapId": 3, "status": {"status": "activated"}, "state": st("@151", "B", now - 60, 30)},       # спот — не наш ринок
        {"status": {"status": "activated"}, "state": st("PUMP", "B", now - 60, 30)},                    # старий запис без id
    ]
    w.adopt("0xabc", hist)
    assert list(w.active) == [1]
    a = w.active[1]
    assert a["binance"] == "PUMPUSDT" and a["open"] and abs(a["executed_sz"] / a["sz"] - 600 / 3600) < 0.01
    w.add_fill("PUMP", "0xabc", "A", now, 0.004, 5000)                  # порція після виявлення — додається до виконаного
    assert abs(w.active[1]["executed_sz"] - (1e6 * 600 / 3600 + 5000)) < 1


def test_expire_when_slices_stop():
    w = watcher()
    now = time.time()
    w.adopt("0xabc", [{"twapId": 7, "status": {"status": "activated"},
                       "state": {"coin": "PUMP", "side": "B", "sz": "100", "minutes": 120, "timestamp": int((now - 60) * 1000)}}])
    w.active[7]["last_fill"] = now - (H.STALE_MIN + 1) * 60             # порцій немає довше за STALE_MIN
    w.expire()
    assert 7 not in w.active and not w.by_key
