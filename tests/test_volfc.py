"""Прогноз волатильності (volfc.py): без погляду вперед, масштаб прогнозу, «звичний» рух, режими."""

import numpy as np
import pandas as pd

import volfc


def walk(days: float, sigma: float, seed: int, start="2026-10-01") -> pd.DataFrame:
    """5m-свічки випадкового блукання зі сталою волатильністю sigma (частка ціни на бар)."""
    n = int(days * 288)
    r = np.random.default_rng(seed).normal(0, sigma, n)
    c = 100 * np.exp(np.cumsum(r))
    idx = pd.date_range(start, periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({"c": c, "h": c * (1 + sigma / 2), "l": c * (1 - sigma / 2), "q": 1e5}, index=idx)


def clustered(days: float, seed: int, start="2026-10-01") -> pd.DataFrame:
    """5m-свічки з кластерами волатильності, як на ринку: лог σ години — AR(1) (φ = 0.95) навколо log 0.003."""
    rng = np.random.default_rng(seed)
    hours = int(days * 24)
    ls = np.zeros(hours)
    for t in range(1, hours):
        ls[t] = 0.95 * ls[t - 1] + rng.normal(0, 0.25)
    sig = np.repeat(0.003 * np.exp(ls), 12)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 1, len(sig)) * sig))
    idx = pd.date_range(start, periods=len(sig), freq="5min", tz="UTC")
    return pd.DataFrame({"c": c, "h": c, "l": c, "q": 1e5}, index=idx)


def test_regimes():
    assert [volfc.regime(x) for x in (0.5, 0.8, 1.49, 1.5, 2.49, 2.5, 7)] == \
        ["Calm", "Normal", "Normal", "Active", "Active", "Hot", "Hot"]
    assert volfc.regime(None) is None and volfc.regime(float("nan")) is None


def test_no_lookahead_and_last_point():
    """Стрибок у барі, що відкривається о T, НЕ входить в ознаки точки T, але входить у точку T + 1 год і в її майбутнє;
    остання точка — закриття останнього бару."""
    b = walk(9, 0.002, 1)
    t = pd.Timestamp("2026-10-09 12:00", tz="UTC")
    c = b["c"].copy()
    c[c.index >= t] *= 1.10                                     # +10% у барі [12:00, 12:05)
    p = volfc.hourly_points(c)
    assert p.loc[t, "x1"] < np.log(0.0001)                     # година до T — звичайна (σ² = 4e-6)
    assert p.loc[t + pd.Timedelta(hours=1), "x1"] > np.log(0.0005)
    assert p.loc[t, "y1"] > np.log(0.0005)                     # майбутнє точки T — зі стрибком
    assert p.index[-1] == b.index[-1] + pd.Timedelta(minutes=5)
    assert np.isnan(p["y4"].iloc[-1]) and np.isnan(p["y1"].iloc[-1])


def test_forecast_scale_and_vs_on_constant_vol():
    """Стала волатильність σ на 5m → прогноз руху за h год ≈ σ·√(12h), «звичний» рух такий самий, vs ≈ 1 (Normal)."""
    sig = 0.003
    bars = {f"C{i}USDT": walk(16, sig, 10 + i) for i in range(5)}
    out = volfc.compute(bars)
    assert out["n"] == 5
    for r in out["rows"]:
        for h in volfc.HORIZONS:
            want = sig * np.sqrt(12 * h)
            assert abs(r[f"exp_{h}"] / want - 1) < 0.15, (h, r[f"exp_{h}"], want)
            assert abs(r[f"usual_{h}"] / want - 1) < 0.1
            assert r[f"regime_{h}"] == "Normal"


def test_hot_coin_detected():
    """Волатильність іде кластерами (модель вчиться, що бурхливе триває); монета, що останні 4 год рухається вчетверо
    сильніше, ніж звичайно, — вище за інших за vs і не Calm / Normal."""
    bars = {f"C{i}USDT": clustered(16, 20 + i) for i in range(5)}
    hot = bars["C0USDT"].copy()
    tail = hot.index >= hot.index[-48]
    r = np.log(hot["c"]).diff().fillna(0).to_numpy().copy()
    r[tail] *= 4
    hot["c"] = 100 * np.exp(np.cumsum(r))
    bars["C0USDT"] = hot
    out = volfc.compute(bars)
    top = out["rows"][0]
    assert top["base"] == "C0" and top["vs_4"] > 1.5 and top["regime_4"] in ("Active", "Hot")


def test_view_turnover_filter_keeps_list_tickers():
    """Фільтр обороту за 24 год [min, max]; тікери списку показуються завжди; Hot рахується по показаних."""
    rows = [{"base": "BIG", "turnover_24h": 2e9, "regime_1": "Hot", "regime_4": "Hot"},
            {"base": "MID", "turnover_24h": 50e6, "regime_1": "Normal", "regime_4": "Hot"},
            {"base": "SMALL", "turnover_24h": 1e6, "regime_1": "Calm", "regime_4": "Calm"}]
    data = {"ts": 0, "n": 3, "rows": rows}
    v = volfc.view(data, {"min_turnover_m": 5, "max_turnover_m": None})
    assert [r["base"] for r in v["rows"]] == ["BIG", "MID"] and v["hot_4"] == 2 and v["n_all"] == 3
    v = volfc.view(data, {"min_turnover_m": 5, "max_turnover_m": 1000})
    assert [r["base"] for r in v["rows"]] == ["MID"] and v["hot_4"] == 1
    v = volfc.view(data, {"min_turnover_m": 5, "max_turnover_m": 1000}, keep={"SMALL"})
    assert [r["base"] for r in v["rows"]] == ["MID", "SMALL"]


def test_save_config_validates(tmp_path, monkeypatch):
    monkeypatch.setattr(volfc, "CONFIG_FILE", str(tmp_path / "vol_config.json"))
    assert volfc.load_config() == volfc.CONFIG_DEFAULTS
    assert volfc.save_config({"min_turnover_m": 20, "max_turnover_m": 1500}) == {"min_turnover_m": 20.0, "max_turnover_m": 1500.0}
    assert volfc.save_config({"max_turnover_m": ""})["max_turnover_m"] is None          # порожнє — без стелі
    for bad in ({"min_turnover_m": -1}, {"min_turnover_m": 30, "max_turnover_m": 10}):
        try:
            volfc.save_config(bad)
            raise AssertionError(bad)
        except ValueError:
            pass
    assert volfc.load_config() == {"min_turnover_m": 20.0, "max_turnover_m": None}
