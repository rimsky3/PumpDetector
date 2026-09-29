"""
Експеримент: менший таймфрейм (15 або 5 хвилин) проти 1h — чи раніше модель помічає памп.

Дані — Binance + Bybit напряму (у Coinalyze 15m/5m історії лише на 1–3 тижні; у бірж — місяці).
Обидві моделі бачать ті самі ознаки на тих самих біржах, без ліквідацій
(їх історії в бірж немає). Вікна ознак ті самі за часом: «24h» — доба і на 1h, і на 15m (features.py, bars_per_hour).
Отже, різниця лише в тому, як часто модель оновлює оцінку: раз на годину чи раз на 15 хвилин.

Розмітка — та сама (pumps.csv): бар 15m усередині інтервалу пампу = 1.
Метрики подій однакові для обох (час — за годинником, не в барах):
    сигнал доступний     на закритті бару (1h: ts + 60 хв, 15m: ts + 15 хв)
    затримка             від початку пампу за розміткою до першого доступного сигналу, хв
    пропущено руху       (ціна на сигналі − ціна на старті) / (пік − ціна на старті); ціни — з 15m/5m-ряду для обох
    епізод               сигнальні бари з проміжками не більше години — одна «угода»; хибний, якщо не перетинає розмітку

    python experiments/timeframe.py              # 15m
    python experiments/timeframe.py --tf 5m
    python experiments/timeframe.py --refresh    # завантажити з бірж заново
    → reports/experiment_timeframe_<tf>.md
"""

import argparse
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import psycopg
from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train  # noqa: E402
from exchange_direct import EXCHANGES, direct_raw, load_direct  # noqa: E402
from exchanges import TF, stats  # noqa: E402
from features import FEATURES, build_from_raw, load_pumps, load_raw  # noqa: E402
from fetch import TICKERS_FILE, read_tickers_file  # noqa: E402

# ознаки, які біржі дають з історією (без ліквідацій)
DIRECT_FEATURES = [f for f in FEATURES if not f.startswith("liq_")]
H = pd.Timedelta(hours=1)


def episodes(df: pd.DataFrame, signal: np.ndarray, bar: pd.Timedelta) -> pd.DataFrame:
    """Сигнальні бари тікера з проміжками ≤ 1 год → один епізод [start, end)."""
    rows = []
    for base, g in df[signal].groupby("base_asset"):
        ts = g["ts"].sort_values()
        run_id = (ts.diff() > H).cumsum()
        for _, run in ts.groupby(run_id):
            rows.append((base, run.iloc[0], run.iloc[-1] + bar))
    return pd.DataFrame(rows, columns=["base_asset", "start", "end"])


def events(df: pd.DataFrame, signal: np.ndarray, bar: pd.Timedelta, pumps: pd.DataFrame,
           fine_price: pd.DataFrame) -> dict:
    """Метрики трейдера на одному таймфреймі. fine_price — ціна закриття з дрібного ряду (base_asset, ts, price),
    за нею рахуємо ціну старту, входу і піку однаково для обох моделей."""
    sig = df.loc[signal, ["base_asset", "ts"]]
    px = {b: g.set_index("ts")["price"] for b, g in fine_price.groupby("base_asset")}
    fine_bar = fine_price.groupby("base_asset")["ts"].diff().median()

    def price_at(base, t):
        """Ціна на момент t = закриття дрібного бару, що завершився в t."""
        s = px[base]
        return s.asof(t - fine_bar)

    delays, missed, left = [], [], []
    n_pump = caught = fs = fs_caught = 0
    for p in pumps.itertuples():
        inside = sig[(sig["base_asset"] == p.base_asset) & (sig["ts"] >= p.start_ts) & (sig["ts"] < p.end_ts)]
        if p.pattern_type == "false_start":
            fs += 1
            fs_caught += bool(len(inside))
            continue
        if p.pattern_type != "pump":
            continue
        n_pump += 1
        if inside.empty:
            continue
        caught += 1
        at = inside["ts"].min() + bar                              # сигнал доступний на закритті бару
        delays.append((at - p.start_ts) / pd.Timedelta(minutes=1))
        s = px[p.base_asset]
        start_px, entry = price_at(p.base_asset, p.start_ts), price_at(p.base_asset, at)
        peak = s[(s.index >= p.start_ts) & (s.index < p.end_ts)].max()
        if peak > start_px:
            missed.append((entry - start_px) / (peak - start_px))
            left.append(peak / entry - 1)

    eps = episodes(df, signal, bar)
    false_eps = 0
    for e in eps.itertuples():
        own = pumps[pumps["base_asset"] == e.base_asset]
        if not ((own["start_ts"] < e.end) & (own["end_ts"] > e.start)).any():
            false_eps += 1
    days = df.groupby("base_asset")["ts"].agg(lambda s: (s.max() - s.min()) / H / 24).sum()
    return {
        "pumps_caught": f"{caught}/{n_pump}",
        "false_starts_triggered": f"{fs_caught}/{fs}",
        "median_delay_min": float(np.median(delays)) if delays else np.nan,
        "median_move_missed": float(np.median(missed)) if missed else np.nan,
        "median_upside_left": float(np.median(left)) if left else np.nan,
        "false_alarms_per_30d": false_eps / days * 30 if days else np.nan,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Експеримент: 15m/5m проти 1h на даних бірж → reports/experiment_timeframe_<tf>.md")
    ap.add_argument("--tf", choices=["15m", "5m"], default="15m", help="менший таймфрейм для порівняння з 1h")
    ap.add_argument("--refresh", action="store_true", help="завантажити дані з бірж заново, ігноруючи кеш")
    args = ap.parse_args()
    bph = TF["1h"]["ms"] // TF[args.tf]["ms"]
    fine_bar = pd.Timedelta(milliseconds=TF[args.tf]["ms"])

    load_dotenv(os.path.join(ROOT, ".env"))
    bases = read_tickers_file(os.path.join(ROOT, TICKERS_FILE))
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        cz_raw = load_raw(conn, bases[:1])            # лише як шаблон колонок для порожніх таблиць
        pumps = load_pumps(conn)
        cur = conn.execute("""SELECT symbol, base_asset, exchange_name, quote_asset, symbol_on_exchange
                              FROM markets WHERE base_asset = ANY(%s) AND exchange_name = ANY(%s)
                              AND quote_asset = 'USDT' ORDER BY base_asset, exchange_name""", (bases, list(EXCHANGES)))
        markets = pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description])

    print(f"Біржі напряму: {len(markets)} ринків, 1h і {args.tf}")
    frames = {}
    for tf, b in (("1h", 1), (args.tf, bph)):
        raw = direct_raw(markets, load_direct(markets, args.refresh, tf), cz_raw, hybrid=False, tf=tf)
        frames[tf] = build_from_raw(raw, pumps, bars_per_hour=b)
    if stats["requests"]:
        print(f"  запитів: {stats['requests']}, {stats['seconds']:.0f} с")

    # однакове вікно для обох: перетин періодів кожного тікера; свіжі нерозмічені години — не беремо (як train.py)
    cutoff = pd.Timestamp(datetime.now(timezone.utc)) - pd.Timedelta(hours=train.FRESH_HOURS)
    a, b = frames["1h"], frames[args.tf]
    span = pd.concat([a.groupby("base_asset")["ts"].agg(["min", "max"]),
                      b.groupby("base_asset")["ts"].agg(["min", "max"])], axis=1, keys=["a", "b"]).dropna()
    lo = span[[("a", "min"), ("b", "min")]].max(axis=1).dt.ceil("h")
    hi = span[[("a", "max"), ("b", "max")]].min(axis=1)

    def window(d):
        d = d[d["base_asset"].isin(lo.index)]
        keep = (d["ts"] >= d["base_asset"].map(lo)) & (d["ts"] <= d["base_asset"].map(hi))
        keep &= (d["ts"] <= cutoff) | d["pattern_type"].notna()
        return d[keep].reset_index(drop=True)

    a, b = window(a), window(b)
    ev_pumps = pumps[pumps["base_asset"].isin(lo.index)
                     & (pumps["start_ts"] >= pumps["base_asset"].map(lo))
                     & (pumps["start_ts"] <= pumps["base_asset"].map(hi))]
    n_pump = int((ev_pumps["pattern_type"] == "pump").sum())
    fine_price = b[["base_asset", "ts", "price"]]
    print(f"Вікно: {lo.min():%m-%d}…{hi.max():%m-%d}, {len(lo)} тікерів, {n_pump} пампів; "
          f"рядків 1h {len(a)}, {args.tf} {len(b)}")

    rows = {}
    for tf, d, bar in (("1h", a, H), (args.tf, b, fine_bar)):
        y, groups = d["label"].astype(int), d["base_asset"]
        for m in train.ALL_MODELS:
            print(f"  {tf} / {m}")
            res = train.leave_ticker_out(m, d[DIRECT_FEATURES], y, groups)
            sig = res["signal"].to_numpy()
            rows[f"{m} · {tf}"] = {
                **{k: v for k, v in train.hour_metrics(y, res, m != "rule").items()},
                **events(d, sig, bar, ev_pumps, fine_price),
            }
    out = pd.DataFrame(rows).T
    print(out.to_string())

    report = os.path.join(ROOT, "reports", f"experiment_timeframe_{args.tf}.md")
    lines = [
        f"# Experiment: {args.tf} bars vs 1h", "",
        f"Binance + Bybit public APIs, {len(lo)} tickers, {lo.min():%Y-%m-%d} – {hi.max():%Y-%m-%d}, {n_pump} labeled pumps. "
        f"Both models use the same {len(DIRECT_FEATURES)} features with the same wall-clock windows "
        f"(no liquidations: exchanges keep no history of them), leave-ticker-out with nested "
        f"threshold as in train.py. The only difference is how often the model re-scores: every hour or every {args.tf}.", "",
        "- **precision / recall / f1 / pr_auc** are per bar, so they are not directly comparable across timeframes",
        "- **median_delay_min**: from the labeled pump start to the first signal, counted at bar close (when it can be acted on)",
        "- **median_move_missed**: share of the move from start to peak already gone at the signal price",
        "- **median_upside_left**: peak / signal price − 1",
        "- **false_alarms_per_30d**: signal episodes (bars ≤ 1 h apart) that overlap no label, per ticker-month", "",
        train.md_table(out), "",
    ]
    with open(report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nЗвіт: {report}")


if __name__ == "__main__":
    main()
