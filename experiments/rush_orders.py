"""
experiments/rush_orders.py — ХВИЛИННИЙ ДЕТЕКТОР СТАРТУ З ПОТОКУ УГОД (2026-10-11, план дня п. 3, тег «швидший вхід»; власник:
«давай з 1 по 3»). Джерела: La Morgia et al., «The Doge of Wall Street» (arXiv 2105.00733) — памп видно за 5–25 с з потоку
«rush orders»; HDP+ (J. Inf. Data Manag.) — rush orders + change point detection, precision 96%, recall 89%. Обидва — на
Telegram-пампах; тут перевіряємо на імпульсах власника (перпи Binance, години, а не хвилини).
Rush order = один taker-ордер: aggTrades з тим самим часом (мс) і стороною (Binance aggTrade = виконання одного ордера за однією
ціною; ордер, що «пробиває» кілька рівнів, дає кілька aggTrades з одним часом).
Етап data: для кожної події власника (імпульс / фальстарт) дні [старт − 30 год, старт + 6 год]; денний архів aggTrades
(data.binance.vision, публічний) потоково → 5-секундні підсумки: кількість і $-обсяг ордерів-покупок / продажів за розміром
(< 1k, 1–5k, 5–25k, 25–100k, ≥ 100k $), найбільший ордер, ціна (остання / макс / мін). Кеш data/rush/<монета>/<дата>.parquet,
сирі файли не зберігаються.
Етап test — детектор (зафіксовано ДО запуску), на хвилинах:
    великі покупки = $-обсяг ордерів-покупок ≥ 25k за останні 5 хв; базова лінія — СЕРЕДНЄ такої суми за попередні 24 год
    (медіана для дрібних монет = 0 — змінено до запуску) і не менше $50k
    спрацювання: великі покупки ≥ K × базової лінії (K = 5 / 10 / 20) І покупки ≥ 60% великого обсягу (купівлі + продажі)
                 І ціна за 5 хв ≥ +1%; після спрацювання — пауза 60 хв
Метрики на подіях власника (лише дні з даними): імпульс спіймано, якщо спрацювання в [старт − 60 хв, старт + 60 хв]; випередження
відносно старту власника і відносно ПЕРШОГО Кандидата base-моделі (OOF за тікерами, 15m — сигнал відомий на закритті бару);
фальстарти зі спрацюванням; хибні спрацювання на день поза розміченими інтервалами (лише переглянуті тікери; дні навколо подій —
активніші за звичайні, тож це верхня оцінка).
КРИТЕРІЙ (до запуску): хоча б за одного K — спіймано ≥ 70% імпульсів, медіана випередження першого Кандидата ≥ 15 хв, хибних
≤ 3 на день на монету.
→ reports/rush_orders.md

    python experiments/rush_orders.py --stage data [--workers 4]
    python experiments/rush_orders.py --stage test
"""

import argparse
import io
import os
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import requests

DIR = os.path.join("data", "rush")
URL = "https://data.binance.vision/data/futures/um/daily/aggTrades/{s}/{s}-aggTrades-{d}.zip"
COLS = ["agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id", "transact_time", "is_buyer_maker"]
BINS = [0, 1_000, 5_000, 25_000, 100_000, np.inf]          # розміри ордерів, $
CHUNK = 2_000_000
OUT = os.path.join("reports", "rush_orders.md")
_s = requests.Session()


def day_path(base: str, day: str) -> str:
    return os.path.join(DIR, base, f"{day}.parquet")


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """aggTrades частини дня → ордери (час мс + сторона) → 5-секундні підсумки."""
    df["usd"] = df["price"] * df["quantity"]
    o = df.groupby(["transact_time", "is_buyer_maker"], sort=False).agg(usd=("usd", "sum"), px=("price", "last"),
                                                                         n=("usd", "size")).reset_index()
    o["t5"] = o["transact_time"] // 5000 * 5000
    o["bin"] = np.digitize(o["usd"], BINS[1:-1])
    o["side"] = np.where(o["is_buyer_maker"], "sell", "buy")
    vol = o.pivot_table(index="t5", columns=["side", "bin"], values="usd", aggfunc="sum", fill_value=0.0)
    vol.columns = [f"{s}_usd{b}" for s, b in vol.columns]
    cnt = o.pivot_table(index="t5", columns="side", values="usd", aggfunc=["size", "max"], fill_value=0.0)
    cnt.columns = [f"{s}_{'n' if a == 'size' else 'max'}" for a, s in cnt.columns]
    px = o.sort_values("transact_time").groupby("t5")["px"].agg(["last", "max", "min"]).add_prefix("px_")
    return vol.join(cnt).join(px)


def fetch_day(base: str, day: str) -> str:
    path = day_path(base, day)
    if os.path.exists(path):
        return "кеш"
    sym = quote(f"{base}USDT")
    tmp = os.path.join(DIR, "_tmp", f"{base}_{day}.zip")
    os.makedirs(os.path.dirname(tmp), exist_ok=True)
    try:
        with _s.get(URL.format(s=sym, d=day), timeout=300, stream=True) as r:
            if r.status_code != 200:
                return f"немає ({r.status_code})"
            with open(tmp, "wb") as f:
                for b in r.iter_content(1 << 20):
                    f.write(b)
        parts = []
        with zipfile.ZipFile(tmp) as z, z.open(z.namelist()[0]) as fh:
            head = fh.read(20).decode("ascii", "ignore")
        with zipfile.ZipFile(tmp) as z, z.open(z.namelist()[0]) as fh:
            rd = pd.read_csv(io.TextIOWrapper(fh), header=0 if head.startswith("agg_trade_id") else None, names=COLS,
                             usecols=["price", "quantity", "transact_time", "is_buyer_maker"], chunksize=CHUNK,
                             dtype={"price": "float64", "quantity": "float64", "transact_time": "int64"})
            for ch in rd:
                ch["is_buyer_maker"] = ch["is_buyer_maker"].astype(str).str.lower().eq("true")
                parts.append(summarize(ch))
        s = pd.concat(parts)
        num = [c for c in s.columns if not c.startswith("px_")]
        s = s.groupby(level=0).agg({**{c: "sum" for c in num if not c.endswith("_max")}, **{c: "max" for c in num if c.endswith("_max")},
                                    "px_last": "last", "px_max": "max", "px_min": "min"})
        os.makedirs(os.path.dirname(path), exist_ok=True)
        s.to_parquet(path + ".tmp")
        os.replace(path + ".tmp", path)
        return f"{len(s)} інтервалів по 5 с"
    except Exception as e:                  # noqa: BLE001 — один битий день не зупиняє завантаження
        return f"помилка: {e!r}"[:120]
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def event_days() -> list[tuple[str, str]]:
    p = pd.read_csv("pumps.csv", parse_dates=["start_ts"])
    days = set()
    for r in p.itertuples():
        for d in pd.date_range((r.start_ts - pd.Timedelta(hours=30)).normalize(), (r.start_ts + pd.Timedelta(hours=6)).normalize()):
            days.add((r.base_asset, d.strftime("%Y-%m-%d")))
    return sorted(days)


def stage_data(workers: int) -> None:
    jobs = event_days()
    print(f"символо-днів {len(jobs)} → {DIR}", flush=True)
    with ThreadPoolExecutor(workers) as ex:
        for n, (job, msg) in enumerate(zip(jobs, ex.map(lambda j: fetch_day(*j), jobs)), 1):
            if n % 20 == 0 or msg.startswith(("помилка", "немає")):
                print(f"  {n}/{len(jobs)} {job[0]} {job[1]}: {msg}", flush=True)
    print("готово", flush=True)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Хвилинний детектор старту імпульсу з потоку угод (rush orders) на подіях власника")
    ap.add_argument("--stage", choices=["data", "test"], required=True, help="data — завантажити й стиснути aggTrades; test — детектор і метрики")
    ap.add_argument("--workers", type=int, default=4, help="паралельних завантажень (етап data)")
    args = ap.parse_args()
    if args.stage == "data":
        stage_data(args.workers)
    else:
        stage_test()


KS = [5, 10, 20]
FLOOR, BUY_SHARE, PX_MIN, COOLDOWN = 50_000, 0.6, 0.01, pd.Timedelta(minutes=60)
WIN = pd.Timedelta(minutes=60)


def minutes(base: str) -> pd.DataFrame:
    """Хвилинний ряд монети з кешу 5-секундних підсумків (лише завантажені дні; між ними — розриви)."""
    files = sorted(f for f in os.listdir(os.path.join(DIR, base)) if f.endswith(".parquet")) if os.path.isdir(os.path.join(DIR, base)) else []
    if not files:
        return pd.DataFrame()
    s = pd.concat([pd.read_parquet(os.path.join(DIR, base, f)) for f in files]).sort_index()
    s.index = pd.to_datetime(s.index, unit="ms", utc=True)
    for c in ("buy_usd3", "buy_usd4", "sell_usd3", "sell_usd4"):
        if c not in s:
            s[c] = 0.0
    m = pd.DataFrame({"buy_big": (s["buy_usd3"] + s["buy_usd4"]).resample("1min").sum(),
                      "sell_big": (s["sell_usd3"] + s["sell_usd4"]).resample("1min").sum(),
                      "px": s["px_last"].resample("1min").last()})
    days = {d for d in s.index.normalize()}
    m = m[m.index.normalize().isin(days)]          # лише хвилини завантажених днів
    m["px"] = m["px"].ffill()
    return m


def triggers(m: pd.DataFrame, k: float) -> pd.DatetimeIndex:
    """Моменти спрацювання (кінець хвилини) з паузою COOLDOWN."""
    b5 = m["buy_big"].rolling("5min").sum()
    s5 = m["sell_big"].rolling("5min").sum()
    base = b5.rolling("24h", min_periods=12 * 60).mean().shift(5)
    r5 = m["px"] / m["px"].shift(5) - 1
    ok = (b5 >= k * base.clip(lower=FLOOR)) & (b5 / (b5 + s5) >= BUY_SHARE) & (r5 >= PX_MIN)
    out, last = [], None
    for t in m.index[ok.to_numpy()]:
        if last is None or t - last >= COOLDOWN:
            out.append(t + pd.Timedelta(minutes=1))
            last = t
    return pd.DatetimeIndex(out, dtype="datetime64[ns, UTC]")


def stage_test() -> None:
    import early_target as ET
    import train
    from fetch import read_tickers_file
    d, pumps, px, idx, uni = ET.load_data()
    y = d["label"].to_numpy().astype(int)
    p = train.oof_proba(train.make_lightgbm, d[ET.COLS], pd.Series(y), d["base_asset"])
    thr = train.best_f1_threshold(y, p)
    cand = d.loc[p >= thr, ["base_asset", "ts"]].assign(close=lambda x: x["ts"] + ET.BAR)
    reviewed = {t.upper() for t in read_tickers_file("tickers.txt")}
    ev = pumps[pumps["pattern_type"].isin(["pump", "false_start"])].copy()
    M = {b: minutes(b) for b in sorted(ev["base_asset"].unique())}
    M = {b: m for b, m in M.items() if len(m)}
    print(f"монет з даними угод {len(M)}; поріг base {thr:.3f}", flush=True)
    rows, fa = [], []
    for k in KS:
        trig = {b: triggers(m, k) for b, m in M.items()}
        n_false, days = 0, 0.0
        for b, m in M.items():
            own = ev[ev["base_asset"] == b]
            if b in reviewed:
                days += m.index.normalize().nunique()
                for t in trig[b]:
                    n_false += not ((own["start_ts"] - WIN <= t) & (t <= own["end_ts"])).any()
        fa.append({"k": k, "false": n_false, "days": days})
        for r in ev[ev["base_asset"].isin(M)].itertuples():
            m = M[r.base_asset]
            if not ((m.index <= r.start_ts - WIN).any() and (m.index >= r.start_ts + WIN).any()):
                continue                              # подія поза завантаженими днями
            t = trig[r.base_asset]
            hit = t[(t >= r.start_ts - WIN) & (t <= (r.start_ts + WIN if r.pattern_type == "pump" else r.end_ts))]
            c = cand[(cand["base_asset"] == r.base_asset) & (cand["close"] > r.start_ts - WIN) & (cand["close"] <= r.end_ts)]["close"]
            rows.append({"k": k, "type": r.pattern_type, "base": r.base_asset, "start": r.start_ts, "hit": len(hit) > 0,
                         "lead_start": ((r.start_ts - hit.min()) / pd.Timedelta(minutes=1)) if len(hit) else np.nan,
                         "model_first": c.min() if len(c) else pd.NaT,
                         "lead_model": ((c.min() - hit.min()) / pd.Timedelta(minutes=1)) if len(hit) and len(c) else np.nan})
    t = pd.DataFrame(rows)
    fa = pd.DataFrame(fa).set_index("k")
    L = ["# Хвилинний детектор старту з потоку угод (rush orders)", "",
         "Великі покупки — $-обсяг ордерів-покупок ≥ 25k за 5 хв; спрацювання: ≥ K × середнього за 24 год (не менше $50k), покупки "
         "≥ 60% великого обсягу, ціна за 5 хв ≥ +1%; пауза 60 хв. Спіймано — спрацювання в [старт − 60 хв, старт + 60 хв]. "
         "Випередження — хвилини до старту власника / до першого Кандидата base-моделі (OOF, 15m, на закритті бару).", "",
         "**Критерій (до запуску):** хоча б за одного K — спіймано ≥ 70% імпульсів, медіана випередження першого Кандидата ≥ 15 хв, "
         "хибних ≤ 3 на день на монету.", "",
         "| K | імпульсів з даними | спіймано | випередження старту, хв (медіана) | випередження Кандидата, хв (медіана) | "
         "детектор раніше за Кандидата | фальстартів спіймано | хибних / день / монету |", "|---|---|---|---|---|---|---|---|"]
    verdict = []
    for k in KS:
        p_ = t[(t["k"] == k) & (t["type"] == "pump")]
        f_ = t[(t["k"] == k) & (t["type"] == "false_start")]
        lm = p_["lead_model"].dropna()
        fpd = fa.at[k, "false"] / max(fa.at[k, "days"], 1)
        L.append(f"| {k} | {len(p_)} | {p_['hit'].sum()} ({p_['hit'].mean():.0%}) | {p_['lead_start'].median():+.0f} | {lm.median():+.0f} | "
                 f"{(lm > 0).sum()} з {len(lm)} | {f_['hit'].sum()}/{len(f_)} | {fpd:.1f} |")
        verdict.append(p_["hit"].mean() >= 0.7 and lm.median() >= 15 and fpd <= 3)
    L += ["", "## Критерій", "", f"→ **{'ВИКОНАНО' if any(verdict) else 'НЕ виконано'}** (" +
          ", ".join(f"K={k}: {'так' if v else 'ні'}" for k, v in zip(KS, verdict)) + ")"]
    t.to_csv(OUT.replace(".md", "_events.csv"), index=False)
    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
