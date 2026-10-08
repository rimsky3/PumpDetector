"""
monitor.py — здоров'я 15m-моделі наживо (2026-10-08): чи її ймовірність щось означає на нових даних і чи модель бачить те
саме, на чому вчилась. Бектест відповідає «чи модель заробляла б», а тут — ранні ознаки того, що щось зламалось, ще до
того, як це видно в PF за місяць.

    1. Частота сигналів по днях: тікерів оцінено, барів Кандидата / Alert, тікерів з Alert. Різкий ріст або спад без
       зміни ринку — підозра (дані, індекс альтів, зміна списку).
    2. Калібрування: оцінені поза навчанням бари поточної версії, згруповані за s15 = P / поріг → що ціна робила далі:
       максимальний ріст і просідання за 4 год (MFE / MAE в ATR15), ціна через 1 і 4 год, частка барів, після яких був
       ріст ≥ 2 ATR, частка всередині розмічених імпульсів. Якщо модель «тримає зміст», усе це росте разом з s15.
    3. Дрейф ознак: PSI (population stability index) кожної ознаки моделі — бари тікерів списку за останні дні проти
       навчальної вибірки (bundle['feature_ref']: децилі навчальних значень + частка NaN). PSI < 0.1 — стабільно,
       0.1–0.25 — помітний зсув, > 0.25 — сильний: модель бачить те, чого в навчанні майже не було.
       Обережно: список — не випадкові монети (сканер бере ті, що «прокинулись»), тож частина зсуву — відбір, а не поломка.

    python monitor.py                 # звіт за 7 днів → reports/monitor.md + reports/monitor.json
    python monitor.py --days 14
    python monitor.py --add-ref       # дописати еталон ознак у робочу 15m-модель (навчену до 08.10, без feature_ref)
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_MD = os.path.join(ROOT, "reports", "monitor.md")
OUT_JSON = os.path.join(ROOT, "reports", "monitor.json")
# s15 = P / поріг; при порозі ~0.82 максимум ≈ 1.22, тож верхній кошик — від 1.1 (P ≈ 0.9)
BUCKETS = [(0.0, 0.5, "< 0.5"), (0.5, 0.75, "0.5–0.75"), (0.75, 1.0, "0.75–1 (майже)"),
           (1.0, 1.1, "1–1.1 (Кандидат)"), (1.1, np.inf, "≥ 1.1 (сильний)")]
HORIZON = 16                    # 15m-барів уперед = 4 год
BIG_MOVE_ATR = 2.0              # «був рух»: максимум за 4 год ≥ 2 ATR від закриття бару
PSI_WARN, PSI_BAD = 0.1, 0.25
FEATURE_DAYS = 15               # історії на тижневі вікна ознак (як screener.score_15m)


# ---------------------------------------------------------------- еталон і PSI (train.py викликає feature_reference)

def feature_reference(X: pd.DataFrame) -> dict:
    """Навчальний розподіл кожної ознаки: межі децилів (без повторів) і частки навчальних значень у кожному кошику + NaN."""
    ref = {}
    for c in X.columns:
        v = pd.to_numeric(X[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        ok = v.dropna().to_numpy(dtype=float)
        if len(ok) == 0:
            ref[c] = {"edges": [], "shares": [], "nan": 1.0}
            continue
        edges = np.unique(np.quantile(ok, np.linspace(0.1, 0.9, 9)))
        counts = np.bincount(np.searchsorted(edges, ok, side="right"), minlength=len(edges) + 1)
        ref[c] = {"edges": edges.tolist(), "shares": (counts / len(v)).tolist(), "nan": float(v.isna().mean()),
                  "median": float(np.median(ok))}
    return ref


def psi(ref: dict, values: pd.Series) -> float:
    """PSI між навчальним розподілом (feature_reference) і values; NaN — окремий кошик."""
    v = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if len(v) == 0 or not ref.get("shares"):
        return np.nan
    edges = np.asarray(ref["edges"], dtype=float)
    ok = v.dropna().to_numpy(dtype=float)
    counts = np.bincount(np.searchsorted(edges, ok, side="right"), minlength=len(edges) + 1)
    actual = np.append(counts / len(v), v.isna().mean())
    expected = np.append(np.asarray(ref["shares"], dtype=float), ref["nan"])
    a, e = np.clip(actual, 1e-4, None), np.clip(expected, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


# ---------------------------------------------------------------- дані

def _scores(conn, model: str, version: str | None, since) -> pd.DataFrame:
    """Оцінки поза навчанням. version=None — найсвіжіша оцінка кожного бару серед усіх версій (годинне підтвердження:
    одразу після перенавчання нова годинна версія ще не має історії, а наживо Alert підтверджувала та, що працювала)."""
    from features import query_df
    if version is None:
        df = query_df(conn, """SELECT DISTINCT ON (base_asset, ts) base_asset, ts, probability, threshold, live FROM signals
                               WHERE model = %s AND NOT in_sample AND ts >= %s ORDER BY base_asset, ts, scored_at DESC""",
                      (model, since))
    else:
        df = query_df(conn, """SELECT base_asset, ts, probability, threshold, live FROM signals
                               WHERE model = %s AND model_version = %s AND NOT in_sample AND ts >= %s""", (model, version, since))
    if len(df):
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def _forward(h: pd.DataFrame) -> pd.DataFrame:
    """Що ціна робила після закриття кожного бару: ret 1 / 4 год, MFE / MAE за 4 год (частка і в ATR15)."""
    from risk import atr
    c = h["price"]
    fut_hi = h["high"][::-1].rolling(HORIZON, min_periods=HORIZON).max()[::-1].shift(-1)
    fut_lo = h["low"][::-1].rolling(HORIZON, min_periods=HORIZON).min()[::-1].shift(-1)
    a = atr(h) / c
    return pd.DataFrame({"ret_1h": c.shift(-4) / c - 1, "ret_4h": c.shift(-HORIZON) / c - 1,
                         "mfe_atr": (fut_hi / c - 1) / a, "mae_atr": (fut_lo / c - 1) / a}, index=h.index)


def _in_pump(pumps: pd.DataFrame, base: str, ts: pd.Series) -> pd.Series:
    from features import in_pump
    p = pumps[(pumps["base_asset"] == base) & (pumps["pattern_type"] == "pump")]
    out = pd.Series(False, index=ts.index)
    for s, e in zip(p["start_ts"], p["end_ts"]):
        out |= in_pump(ts, pd.Timestamp(s), pd.Timestamp(e), pd.Timedelta(minutes=15))
    return out


# ---------------------------------------------------------------- звіт

def report(conn, days: int = 7, log=print) -> dict:
    import train
    from features import INTERVAL_15M, aggregate, build_15m, load_pumps, load_raw
    from tiers import tier_frame
    m15 = train.load_model(train.M15_NAME)
    conf = m15.get("confirm", {"model": "lightgbm", "share": 0.75})
    now = pd.Timestamp(datetime.now(timezone.utc))
    since = (now - pd.Timedelta(days=days)).floor("D")
    s15 = _scores(conn, train.M15_NAME, m15["trained_at"], since)
    if s15.empty:
        raise SystemExit(f"немає оцінок версії {m15['trained_at']} поза навчанням з {since:%Y-%m-%d}")
    s1h = _scores(conn, conf["model"], None, since - pd.Timedelta(hours=2))
    t = tier_frame(s15, s1h, conf["share"])
    bases = sorted(t["base_asset"].unique())
    log(f"  оцінок: {len(t)} барів, {len(bases)} тікерів, з {since:%Y-%m-%d}")

    # 1. частота сигналів
    t["day"] = t["ts"].dt.strftime("%Y-%m-%d")
    daily = t.groupby("day").agg(tickers=("base_asset", "nunique"), bars=("ts", "size"),
                                 cand=("tier", lambda x: int((x != "calm").sum())),
                                 alert=("tier", lambda x: int((x == "alert").sum())))
    daily["alert_tickers"] = t[t["tier"] == "alert"].groupby("day")["base_asset"].nunique()
    daily = daily.fillna(0).astype(int)

    # 2. калібрування: майбутнє ціни після кожного оціненого бару
    raw = load_raw(conn, bases, INTERVAL_15M, since - pd.Timedelta(days=2))
    pumps = load_pumps(conn)
    parts = []
    for b, g in t.groupby("base_asset"):
        h = aggregate(raw, b, "15min", light=True)
        if h.empty:
            continue
        fw = _forward(h).reindex(g["ts"]).set_axis(g.index)
        parts.append(g.join(fw).assign(in_pump=_in_pump(pumps, b, g["ts"]).to_numpy(),
                                       labeled=bool((pumps["base_asset"] == b).any())))
    cal = pd.concat(parts)
    cal = cal[cal["mfe_atr"].notna()]                  # лише бари з повними 4 год після них

    def summary(g: pd.DataFrame) -> dict:
        lab = g[g["labeled"]]
        return {"bars": int(len(g)), "tickers": int(g["base_asset"].nunique()),
                "ret_1h_med": float(g["ret_1h"].median()), "ret_4h_med": float(g["ret_4h"].median()),
                "ret_4h_mean": float(g["ret_4h"].mean()),
                "mfe_atr_med": float(g["mfe_atr"].median()), "mae_atr_med": float(g["mae_atr"].median()),
                "big_move": float((g["mfe_atr"] >= BIG_MOVE_ATR).mean()),
                "in_pump": float(lab["in_pump"].mean()) if len(lab) else None, "labeled_bars": int(len(lab))}

    calib = {"all": summary(cal)}
    for lo, hi, name in BUCKETS:
        g = cal[(cal["s15"] >= lo) & (cal["s15"] < hi)]
        if len(g):
            calib[name] = summary(g)
    for name, g in (("Alert", cal[cal["tier"] == "alert"]), ("Кандидат без Alert", cal[cal["tier"] == "candidate"])):
        if len(g):
            calib[name] = summary(g)
    order = [n for _, _, n in BUCKETS if n in calib]
    mono = all(calib[a]["mfe_atr_med"] <= calib[b]["mfe_atr_med"] + 1e-9 for a, b in zip(order, order[1:]))

    # 3. дрейф ознак
    ref = m15.get("feature_ref")
    drift, medians = {}, {}
    if ref:
        f = build_15m(conn, bases, since=since - pd.Timedelta(days=FEATURE_DAYS))
        f = f[f["ts"] >= since]
        drift = {c: psi(ref[c], f[c]) for c in m15["features"] if c in ref and c in f}
        medians = {c: [ref[c].get("median"), float(f[c].median()) if f[c].notna().any() else None] for c in drift}
    log(f"  калібрування: {len(cal)} барів з повними 4 год; дрейф: {len(drift)} ознак")
    return {"generated": now.isoformat(), "days": days, "since": since.isoformat(), "version": m15["trained_at"],
            "threshold": float(m15["threshold"]), "tickers": bases, "daily": daily.reset_index().to_dict("records"),
            "calibration": calib, "monotonic_mfe": mono, "drift": drift, "medians": medians, "has_ref": bool(ref)}


def to_markdown(r: dict) -> str:
    import train
    pct = lambda v: "—" if v is None or pd.isna(v) else f"{v:+.2%}"   # noqa: E731
    sh = lambda v: "—" if v is None or pd.isna(v) else f"{v:.0%}"     # noqa: E731
    L = [f"# Здоров'я 15m-моделі наживо", "",
         f"Версія {r['version']}, поріг {r['threshold']:.3f}; оцінки поза навчанням з {r['since'][:10]} ({r['days']} дн), "
         f"{len(r['tickers'])} тікерів; згенеровано {r['generated'][:16].replace('T', ' ')} UTC.", "",
         "## 1. Частота сигналів", "", "| день | тікерів | барів | Кандидат+ | Alert | тікерів з Alert |", "|---|---|---|---|---|---|"]
    for d in r["daily"]:
        L.append(f"| {d['day']} | {d['tickers']} | {d['bars']} | {d['cand']} | {d['alert']} | {d['alert_tickers']} |")
    L += ["", "## 2. Калібрування: що ціна робила після бару", "",
          f"s15 = P / поріг. MFE / MAE — максимальний ріст / просідання за 4 год у ATR15; «рух ≥ {BIG_MOVE_ATR:g} ATR» — частка барів, "
          "після яких ціна за 4 год піднімалась хоча б на 2 ATR; «в імпульсі» — лише тікери з розміткою.", "",
          "| рівень | барів | тікерів | ціна +1 год (мед.) | +4 год (мед.) | +4 год (сер.) | MFE, ATR | MAE, ATR | рух ≥ 2 ATR | в імпульсі |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for name, c in r["calibration"].items():
        lab = f"{sh(c['in_pump'])} з {c['labeled_bars']}" if c["labeled_bars"] else "—"
        L.append(f"| {'усі бари' if name == 'all' else name} | {c['bars']} | {c['tickers']} | {pct(c['ret_1h_med'])} | "
                 f"{pct(c['ret_4h_med'])} | {pct(c['ret_4h_mean'])} | {c['mfe_atr_med']:.2f} | {c['mae_atr_med']:.2f} | "
                 f"{sh(c['big_move'])} | {lab} |")
    L += ["", ("✅ MFE росте разом з рівнем s15 — модель «тримає зміст»." if r["monotonic_mfe"] else
               "⚠ MFE НЕ росте монотонно з рівнем s15 — перевірити (мало барів у вищих рівнях — теж причина)."), ""]
    L += ["## 3. Дрейф ознак (PSI: тікери списку проти навчання)", ""]
    if not r["has_ref"]:
        L += ["Еталону ознак у моделі немає — `python monitor.py --add-ref` (або перенавчити: train.py з 08.10 зберігає його сам).", ""]
    elif r["drift"]:
        d = pd.Series(r["drift"]).sort_values(ascending=False)
        bad, warn = d[d > PSI_BAD], d[(d > PSI_WARN) & (d <= PSI_BAD)]
        L += [f"Сильний зсув (> {PSI_BAD}): {len(bad)} · помітний ({PSI_WARN}–{PSI_BAD}): {len(warn)} · стабільні: {int((d <= PSI_WARN).sum())} з {len(d)}.",
              "Список — не випадкові монети (сканер бере ті, що «прокинулись»), тож помірний зсув обсягу / волатильності очікуваний.", "",
              "| ознака | група | PSI | | медіана: навчання → зараз |", "|---|---|---|---|---|"]
        num = lambda x: "—" if x is None or pd.isna(x) else f"{x:.3g}"   # noqa: E731
        for k, v in d.items():
            flag = "🔴" if v > PSI_BAD else "🟡" if v > PSI_WARN else ""
            m = r.get("medians", {}).get(k) or [None, None]
            L.append(f"| {k} | {train.feature_group(k)} | {v:.3f} | {flag} | {num(m[0])} → {num(m[1])} |")
        g = d.groupby(d.index.map(train.feature_group)).mean().sort_values(ascending=False)
        L += ["", "Середній PSI по групах: " + ", ".join(f"{k} {v:.2f}" for k, v in g.items()), ""]
    return "\n".join(L) + "\n"


def add_ref(conn, log=print) -> None:
    """Еталон ознак для моделі, навченої без нього: ті самі переглянуті тікери, бари до data_end (наближення вибірки train.py —
    для децилів цього досить)."""
    import train
    from features import build_15m
    path = train.model_path(train.M15_NAME)
    b = joblib.load(path)
    from features import query_df
    d = build_15m(conn, sorted(b["tickers"]))
    # як train.py: 15m-історія тікера не раніше за його годинну (інакше — довгі лише-Bybit ряди MON / ZRO з березня)
    first_1h = query_df(conn, "SELECT base_asset, min(ts) AS t FROM features GROUP BY 1").set_index("base_asset")["t"]
    d = d[d["ts"] >= pd.to_datetime(d["base_asset"].map(first_1h), utc=True)]
    if b.get("data_end") is not None:
        d = d[d["ts"] <= pd.Timestamp(b["data_end"])]
    b["feature_ref"] = feature_reference(d[b["features"]])
    joblib.dump(b, path + ".tmp")
    os.replace(path + ".tmp", path)
    log(f"  еталон ознак: {len(d)} барів, {d['base_asset'].nunique()} тікерів → {path}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Здоров'я 15m-моделі наживо: частота сигналів, калібрування, дрейф ознак (PSI)")
    ap.add_argument("--days", type=int, default=7, help="скільки останніх днів оцінок брати (за замовчуванням 7)")
    ap.add_argument("--add-ref", action="store_true", help="дописати еталон ознак (feature_ref) у робочу 15m-модель і вийти")
    args = ap.parse_args()
    import psycopg
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        if args.add_ref:
            add_ref(conn)
            return
        r = report(conn, args.days)
    md = to_markdown(r)
    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    open(OUT_MD, "w", encoding="utf-8").write(md)
    json.dump(r, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    print(md)
    print(f"Звіт: {OUT_MD}")


if __name__ == "__main__":
    main()
