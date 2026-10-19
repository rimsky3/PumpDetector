"""Сканер (зміна OI за 4 год без збоїв біржі) і «прокидається» на 5m (wake.metrics: чотири умови разом)."""

import numpy as np
import pandas as pd

import scanner
import wake


def scan_bars(oi, volume: float = 50_000.0):
    """15m-бари сканера: ≥ 3 доби історії, стала ціна 1.0, обсяг за бар у $; oi — $ на останніх len(oi) барах."""
    n = 24 * 4 * 3 + 20
    idx = pd.date_range("2026-10-01", periods=n, freq="15min", tz="UTC")
    h = pd.DataFrame({"volume": volume, "price": 1.0, "oi": np.nan}, index=idx)
    h.iloc[-len(oi):, h.columns.get_loc("oi")] = oi
    return h


def test_scanner_oi_change_clean():
    oi = np.linspace(1e6, 1.3e6, 18)
    m = scanner.metrics_from_bars(scan_bars(oi), scanner.BARS_PER_HOUR)
    assert np.isclose(m["oi_pct"], (oi[-1] / oi[-17] - 1) * 100)


def test_scanner_oi_glitch_ignored():
    """Перші знімки OI занижені втричі (як EPIC 15.10: WOKE показав +223% при справжніх +5%) — стрибок не рахуємо.
    Обсяг за бар більший за сам стрибок OI (як в EPIC) — тож ловить саме правило «крок у монетах > ×2», а не «більше за обсяг»."""
    oi = np.full(18, 1e6)
    oi[:4] /= 3.2
    m = scanner.metrics_from_bars(scan_bars(oi, volume=2_000_000.0), scanner.BARS_PER_HOUR)
    assert abs(m["oi_pct"]) < 1


def test_scanner_oi_glitch_low_volume():
    """Той самий збій на тихій монеті — ловить правило «OI змінився більше, ніж наторгували»."""
    oi = np.full(18, 1e6)
    oi[:4] /= 3.2
    assert abs(scanner.metrics_from_bars(scan_bars(oi), scanner.BARS_PER_HOUR)["oi_pct"]) < 1


def test_scanner_real_pump_kept():
    """Справжній ріст OI ×1.7 за бар разом із ціною ×1.6 (STBL 16.10) — не збій: у монетах крок < ×2."""
    h = scan_bars(np.r_[np.full(10, 1e6), np.full(8, 1e6 * 1.7 * 1.6)])
    h.iloc[-8:, h.columns.get_loc("price")] = 1.6
    h.iloc[-8:, h.columns.get_loc("volume")] = 5e6
    m = scanner.metrics_from_bars(h, scanner.BARS_PER_HOUR)
    assert m["oi_pct"] > 150


def wake_frame(spike_vol: float, ret: float, oi_up: float):
    n = 288 * 4                                                    # 4 доби 5m — норма обсягу з ≥ 3 діб
    idx = pd.date_range("2026-10-01", periods=n, freq="5min", tz="UTC")
    h = pd.DataFrame({"volume": 1000.0, "price": 1.0, "oi_log_idx": 0.0}, index=idx)
    h.iloc[-3:, h.columns.get_loc("volume")] = spike_vol
    h.iloc[-1, h.columns.get_loc("price")] = 1.0 + ret
    h.iloc[-1, h.columns.get_loc("oi_log_idx")] = np.log1p(oi_up)
    return h


def test_wake_fires_only_when_all_conditions_hold():
    assert wake.metrics(wake_frame(10_000, 0.03, 0.03))["hit"].iloc[-1]
    assert not wake.metrics(wake_frame(10_000, -0.03, 0.03))["hit"].iloc[-1]        # ціна вниз — не «прокидається»
    assert not wake.metrics(wake_frame(10_000, 0.03, 0.0))["hit"].iloc[-1]          # без OI
    assert not wake.metrics(wake_frame(2_000, 0.03, 0.03))["hit"].iloc[-1]          # обсяг ×2 < ×5
