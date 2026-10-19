"""OI в ознаках: зв'язаний індекс (лістинг не ріст), неможливий OI (збій біржі, AGT 15.10), короткі дірки (OI_GAP_HOURS)."""

import numpy as np
import pandas as pd

import features as F
from conftest import make_raw


def test_chained_index_ignores_new_listing():
    """Нова біржа з'являється з OI — сума OI стрибає, а індекс ні: позиції не відкривались."""
    idx = pd.date_range("2026-10-01", periods=4, freq="h", tz="UTC")
    w = pd.DataFrame({"binance": [100.0, 100.0, 110.0, 110.0], "bybit": [np.nan, 50.0, 50.0, 50.0]}, index=idx)
    li = F.chained_log_index(w)
    assert li.iloc[1] == 0.0                                   # лістинг Bybit — без кроку
    assert np.isclose(li.iloc[2], np.log(160 / 150))           # далі — справжній ріст по обох
    assert np.isclose(li.iloc[3], li.iloc[2])


def test_chained_index_skips_zero_and_gaps():
    """OI = 0 (архів Binance) — пропуск, а не log(0) = −∞ на весь ряд."""
    idx = pd.date_range("2026-10-01", periods=4, freq="h", tz="UTC")
    li = F.chained_log_index(pd.DataFrame({"m": [100.0, 0.0, 100.0, 110.0]}, index=idx))
    assert np.isnan(li.iloc[1])
    assert np.isfinite(li.iloc[3])


def test_impossible_oi_flags_glitch_not_real_moves():
    """OI змінився більше, ніж наторгували за бар (і > 20%), — збій; звичайні зміни — ні."""
    idx = pd.date_range("2026-10-01", periods=5, freq="15min", tz="UTC")
    oi = pd.DataFrame({"m": [1000.0, 1010.0, 3400.0, 3390.0, 1000.0]}, index=idx)      # ×3.4 і назад — як AGT
    close = pd.DataFrame({"m": [1.0] * 5}, index=idx)
    vol = pd.DataFrame({"m": [50.0] * 5}, index=idx)                                       # у монетах
    bad = F.impossible_oi(oi, close, vol)["m"].tolist()
    assert bad == [False, False, True, False, True]


def test_impossible_oi_allows_large_move_with_volume():
    """Великий справжній ріст OI при великому обсязі — не збій."""
    idx = pd.date_range("2026-10-01", periods=2, freq="15min", tz="UTC")
    oi = pd.DataFrame({"m": [1000.0, 1800.0]}, index=idx)
    bad = F.impossible_oi(oi, pd.DataFrame({"m": [1.0, 1.0]}, index=idx), pd.DataFrame({"m": [5000.0, 5000.0]}, index=idx))
    assert not bad["m"].any()


def test_aggregate_drops_impossible_oi(raw15):
    """У зведеному ряді стрибок-збій OI не дає кроку індексу: індекс після збою той самий, що до нього."""
    raw = raw15
    ts = raw["oi"]["ts"]
    glitch = ts.isin(ts.iloc[40:46])
    for col in ("o", "h", "l", "c"):
        raw["oi"].loc[glitch, col] *= 3.4
    h = F.aggregate(raw, "X", "15min", light=True)
    assert h["oi_log_idx"].dropna().abs().max() < 1e-9      # жодного кроку: вхід і вихід зі збою — пропуски


def test_compute_features_survives_short_oi_gap():
    """Дірка в OI на 1 год не робить oi_z_7d порожнім на тиждень (AGT 15.10: тікер без оцінок і сповіщень)."""
    n = 4 * 24 * 12                                                  # 7 днів розігріву + запас
    rng = np.random.default_rng(0)
    raw = make_raw(n=n)
    oi = 1e6 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    for col in ("o", "h", "l", "c"):
        raw["oi"][col] = oi
    raw["oi"] = raw["oi"].drop(index=range(n - 200, n - 196))        # 4 бари (1 год) без OI за ~2 доби до кінця
    key = raw["ohlcv"][["base_asset", "market", "exchange", "quote_asset", "ts"]]
    raw["funding"] = key.assign(c=0.0001)                             # повний aggregate чекає й ці таблиці
    raw["ls"] = key.assign(long_pct=60.0)
    raw["liq"] = key.iloc[0:0].assign(long_liq=[], short_liq=[])
    h = F.aggregate(raw, "X", "15min")
    f = F.compute_features(h, bars_per_hour=4)
    after = f["oi_z_7d"].iloc[n - 196:]
    assert after.notna().all()
