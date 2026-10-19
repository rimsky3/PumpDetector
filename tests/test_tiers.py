"""Рівні сигналу (tiers.tier_frame): Candidate — 15m ≥ порогу; Alert — ще й остання закрита година ≥ share × її порогу."""

import numpy as np
import pandas as pd

import tiers


def frames(p15: float, p1h: float | None, hour_offset: int = 1):
    ts = pd.Timestamp("2026-10-01 13:30", tz="UTC")
    s15 = pd.DataFrame({"base_asset": ["X"], "ts": [ts], "probability": [p15], "threshold": [0.8]})
    rows = [] if p1h is None else [{"base_asset": "X", "ts": pd.Timestamp("2026-10-01 13:00", tz="UTC") - pd.Timedelta(hours=hour_offset),
                                    "probability": p1h, "threshold": 0.8}]
    return s15, pd.DataFrame(rows, columns=["base_asset", "ts", "probability", "threshold"])


def test_context_hour_is_last_closed_hour():
    ts = pd.Series(pd.to_datetime(["2026-10-01 13:30", "2026-10-01 13:45"], utc=True))
    # бар 13:30–13:45 закривається до кінця години 13:00 → остання закрита — 12:00; бар 13:45–14:00 — уже 13:00
    assert tiers.context_hour(ts).tolist() == [pd.Timestamp("2026-10-01 12:00", tz="UTC"), pd.Timestamp("2026-10-01 13:00", tz="UTC")]


def test_calm_candidate_alert():
    assert tiers.tier_frame(*frames(0.5, 0.9), share=0.75)["tier"].iloc[0] == "calm"
    assert tiers.tier_frame(*frames(0.85, 0.5), share=0.75)["tier"].iloc[0] == "candidate"     # 1h 0.5 < 0.75 × 0.8
    out = tiers.tier_frame(*frames(0.85, 0.65), share=0.75)                                    # 0.65 ≥ 0.6
    assert out["tier"].iloc[0] == "alert" and out["alert_score"].iloc[0] >= 1


def test_previous_hour_used_when_last_hour_missing():
    """Годинної оцінки щойно закритої години ще немає — береться попередня (з минулого), Alert не «гасне»."""
    out = tiers.tier_frame(*frames(0.85, 0.7, hour_offset=2), share=0.75)
    assert out["tier"].iloc[0] == "alert"


def test_no_hourly_score_means_no_alert():
    out = tiers.tier_frame(*frames(0.95, None), share=0.75)
    assert out["tier"].iloc[0] == "candidate" and out["s1h"].iloc[0] == 0.0


def test_split_scores():
    idx = pd.date_range("2026-10-01", periods=3, freq="15min", tz="UTC")
    main, start = pd.Series([0.5, 1.2, 0.9], index=idx), pd.Series([1.1, 0.3, np.nan], index=idx)
    entry, hold = tiers.split_scores(main, start)
    assert entry.tolist()[:2] == [1.1, 0.3]
    assert hold.tolist() == [1.1, 1.2, 0.9]                     # утримання — max(основна, «старт»)
    e2, h2 = tiers.split_scores(main, None)
    assert e2.isna().all() and h2.equals(main)
