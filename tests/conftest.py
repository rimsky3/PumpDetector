"""Спільне для тестів: корінь проєкту в sys.path і штучні ряди (без бази й мережі — тести проходять за секунди)."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def hourly():
    """40 годинних барів: ціна 100, діапазон бару 1.0 (high 100.5 / low 99.5) — ATR рівно 1.0 після розігріву."""
    idx = pd.date_range("2026-10-01", periods=40, freq="h", tz="UTC")
    return pd.DataFrame({"open": 100.0, "price": 100.0, "high": 100.5, "low": 99.5}, index=idx)


def make_raw(n: int = 96, oi: float = 1_000_000.0, price: float = 1.0, vol: float = 50_000.0, freq: str = "15min",
             market: str = "XUSDT", exchange: str = "Binance") -> dict:
    """Сирі ряди у форматі features.load_raw(..., light=True) для одного контракту: стала ціна, обсяг і OI ($)."""
    ts = pd.date_range("2026-10-01", periods=n, freq=freq, tz="UTC")
    base = {"base_asset": "X", "market": market, "exchange": exchange, "quote_asset": "USDT", "ts": ts}
    ohlcv = pd.DataFrame({**base, "o": price, "h": price, "l": price, "c": price, "v": vol, "bv": vol / 2,
                          "volume_denominated": "QUOTE_ASSET"})
    oi_df = pd.DataFrame({**base, "o": oi, "h": oi, "l": oi, "c": oi})
    return {"ohlcv": ohlcv, "oi": oi_df}


@pytest.fixture
def raw15():
    return make_raw()


def series(values, start="2026-10-01", freq="15min") -> pd.Series:
    return pd.Series(np.asarray(values, dtype=float), index=pd.date_range(start, periods=len(values), freq=freq, tz="UTC"))
