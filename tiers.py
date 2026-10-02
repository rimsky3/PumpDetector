"""
tiers.py — рівні сигналу з оцінок 15m- і годинної моделі (одна логіка для дашборду і журналу).

Для кожного 15m-бару:
    s15  = P(15m) / поріг 15m-моделі
    s1h  = P(1h) / поріг годинного LightGBM — оцінка ОСТАННЬОЇ ЗАКРИТОЇ години на момент закриття 15m-бару
           (15m-бар 13:30–13:45 → годинний бар 12:00–13:00): лише те, що вже було відомо
    alert_score = min(s15, s1h / share) — ≥ 1 тоді й лише тоді, коли виконано обидві умови
    tier: «alert» (s15 ≥ 1 і s1h ≥ share), «candidate» (s15 ≥ 1), «calm»

share (0.75) і назва годинної моделі зберігаються в 15m-бандлі (bundle['confirm']); див. experiments/two_tier.py.
"""

import pandas as pd

H = pd.Timedelta(hours=1)
BAR_15M = pd.Timedelta(minutes=15)
TIER_MODEL = "tier15"      # віртуальна модель у дашборді й журналі: «Alert» (15m + підтвердження 1h)
CAND_MODEL = "cand15"      # віртуальна модель для журналу: «Кандидат» (лише 15m)
LIMIT_MODEL = "tier15_lim"    # журнал: ті самі Alert, але вхід лімітом −0.5 ATR — порівняння на живих сигналах (02.10:
                              # на 6 міс. бектесту трохи кращий, у журналі 24.09–02.10 гірший — основним лишився вхід по ринку)
# 15m-модель на КОВЗНИХ 15m-барах (зсув 5/10 хв, скринер кожні 5 хв для «гарячих» тікерів). Окрема назва в signals:
# журнал, позиції й графік лишаються на звичайних 15m-барах (як у навчанні і бектесті), а список і сповіщення
# беруть найсвіжішу оцінку — так Alert приходить на ~5 хв раніше (experiments/early_5m.py)
ROLL_MODEL = "lightgbm_15m_roll"


def context_hour(ts: pd.Series) -> pd.Series:
    """Відкриття останньої годинної свічки, закритої на момент закриття 15m-бару ts."""
    return (ts + BAR_15M - H).dt.floor("h")


def tier_frame(s15: pd.DataFrame, s1h: pd.DataFrame, share: float) -> pd.DataFrame:
    """s15, s1h — рядки signals (base_asset, ts, probability, threshold, …) 15m- і годинної моделі.
    Повертає рядки s15 з колонками p15, thr15, s15, p1h, thr1h, s1h, alert_score, tier."""
    if s15.empty:
        return s15.assign(p15=[], thr15=[], s15=[], p1h=[], thr1h=[], s1h=[], alert_score=[], tier=[])
    out = s15.copy()
    out["p15"], out["thr15"] = out["probability"], out["threshold"]
    out["s15"] = out["p15"] / out["thr15"]
    ctx = s1h[["base_asset", "ts", "probability", "threshold"]].rename(
        columns={"ts": "ctx_ts", "probability": "p1h", "threshold": "thr1h"})
    out = out.assign(ctx_ts=context_hour(out["ts"])).merge(ctx, on=["base_asset", "ctx_ts"], how="left")
    # Годинну оцінку щойно закритої години скринер рахує лише в годинній частині циклу (кілька хвилин після HH:00).
    # Поки її немає, беремо попередню годину — останнє відоме підтвердження (з минулого, не з майбутнього).
    # Інакше на початку кожної години Alert «гасне» на 1–2 бари і позиція хибно закривається (CAP 2026-10-01 12:45).
    prev = ctx.assign(ctx_ts=ctx["ctx_ts"] + H).rename(columns={"p1h": "p1h_prev", "thr1h": "thr1h_prev"})
    out = out.merge(prev, on=["base_asset", "ctx_ts"], how="left")
    miss = out["p1h"].isna()
    out.loc[miss, "p1h"], out.loc[miss, "thr1h"] = out.loc[miss, "p1h_prev"], out.loc[miss, "thr1h_prev"]
    out = out.drop(columns=["p1h_prev", "thr1h_prev"])
    out["s1h"] = (out["p1h"] / out["thr1h"]).fillna(0.0)          # немає ні цієї, ні попередньої — підтвердження немає
    out["alert_score"] = out[["s15"]].assign(c=out["s1h"] / share).min(axis=1)
    out["tier"] = "calm"
    out.loc[out["s15"] >= 1, "tier"] = "candidate"
    out.loc[out["alert_score"] >= 1, "tier"] = "alert"
    return out.drop(columns="ctx_ts")
