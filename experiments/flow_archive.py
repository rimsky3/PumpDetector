"""
experiments/flow_archive.py — потік угод і футпрінт з архіву Binance як ознаки 15m-моделі (2026-10-07).

Питання: чи бачить модель більше, якщо знає СКЛАД угод (великі / дрібні, хто агресор) і де всередині бару пройшов обсяг
(футпрінт), а не лише обсяг і частку покупок свічки? experiments/trade_count.py показав: кількість угод = обсяг іншими
словами. Тут — те, чого в свічці немає.

Дані: data.binance.vision/data/futures/um/daily/aggTrades — усі угоди (aggTrade = виконання одного taker-ордера за однією
ціною), ті самі, що в живому потоці (trade_recorder.py). День завантажується в пам'ять і одразу стискається:
  * 15m-зведення: обсяг купівель / продажів за агресором, угоди ≥ BIG ($10k) і < SMALL ($2.5k) окремо, найбільша угода;
  * футпрінт: обсяг купівель / продажів на рівнях ціни через 0.1% (рівень = floor(ln ціни / ln 1.001) — одна сітка для
    всіх барів, тож години складаються з 15m-барів простим додаванням).
Кеш — data/flow/<монета>/<дата>.pkl (сирих файлів не зберігаємо). Лише Binance (у Bybit-архіві кожен fill окремо).

Ознаки (відносні до історії монети, лише дані до бару включно):
  потік   flow_big_share_1h, flow_big_share_z (z-score частки великих за 7 днів), flow_big_delta_1h / _4h (великі купівлі −
          великі продажі, частка обсягу), flow_small_delta_1h (те саме для дрібних — «натовп»), flow_max_trade_z,
          flow_order_size_z (середній розмір ордера), flow_big_n_z (кількість великих угод за годину)
  футпрінт fp_delta_top / fp_delta_bot (дельта у верхній / нижній третині діапазону бару, частка обсягу бару),
          fp_vol_top (частка обсягу у верхній третині), fp_poc_pos (де в діапазоні рівень з найбільшим обсягом: 0 — лоу,
          1 — хай); те саме за годину (_1h). Діапазон < 3 рівнів (< 0.2%) — NaN.

Порівняння: base / + потік / + футпрінт / + обидва — leave-ticker-out (рівень «Кандидат») і бектест на монетах поза
навчанням (Alert = 15m + робоча 1h, поточний і частковий вихід). Плюс медіани ознак на першому барі імпульсу і першому
барі фальстарту (Failed impulse) — чи розрізняють вони ці два випадки.

    python experiments/flow_archive.py [--workers 8] [--no-backtest]
"""

import argparse
import glob
import io
import json
import os
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from urllib.parse import quote

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import numpy as np
import pandas as pd
import psycopg
import requests
from dotenv import load_dotenv

import train
from features import (FEATURES, FEATURES_15M, INTERVAL_15M, aggregate, build_from_raw, build_hourly_own, load_pumps,
                      load_raw, short_features, zscore)
from fetch import read_tickers_file
from tiers import tier_frame
from trade_count import binance_symbol, keep_training, lto, stats

# tabulate у .venv немає — таблиці у звіті простим текстом
pd.DataFrame.to_markdown = lambda self, *a, **k: "```\n" + self.to_string() + "\n```"

URL = "https://data.binance.vision/data/futures/um/daily/aggTrades/{s}/{s}-aggTrades-{d}.zip"
CACHE = os.path.join("data", "flow")
OUT = os.path.join("reports", "flow_archive.md")
BAR = pd.Timedelta(minutes=15)
BIG, SMALL = 10_000, 2_500
LVL = np.log(1.001)
DAYS = 30
FLOW = ["flow_big_share_1h", "flow_big_share_z", "flow_big_delta_1h", "flow_big_delta_4h", "flow_small_delta_1h",
        "flow_max_trade_z", "flow_order_size_z", "flow_big_n_z"]
FP = ["fp_delta_top", "fp_delta_bot", "fp_vol_top", "fp_poc_pos",
      "fp_delta_top_1h", "fp_delta_bot_1h", "fp_vol_top_1h", "fp_poc_pos_1h"]
_session = requests.Session()


# ---------------------------------------------------------------- архів → 15m-зведення і футпрінт

def day_flow(sym: str, day: pd.Timestamp) -> dict | None:
    """Один день угод → {"bars": 15m-зведення, "fp": рівні}. None — файлу немає (монети ще не було / день не викладено)."""
    r = _session.get(URL.format(s=quote(sym), d=f"{day:%Y-%m-%d}"), timeout=120)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        raw = z.read(z.namelist()[0])
    head = raw[:20].decode("utf-8", "ignore")
    t = pd.read_csv(io.BytesIO(raw), header=0 if head.startswith("agg_trade_id") else None,
                    names=["id", "price", "qty", "first", "last", "time", "ibm"], usecols=["price", "qty", "time", "ibm"])
    if t.empty:
        return {"bars": pd.DataFrame(), "fp": pd.DataFrame()}
    ibm = t["ibm"] if t["ibm"].dtype == bool else t["ibm"].astype(str).str.lower().eq("true")
    usd = t["price"] * t["qty"]
    buy = ~ibm                                   # покупець не maker → агресор купував
    ts = pd.to_datetime(t["time"] // 900_000 * 900_000, unit="ms", utc=True)
    big, small = usd >= BIG, usd < SMALL
    f = pd.DataFrame({"ts": ts, "v_buy": usd.where(buy, 0.0), "v_sell": usd.where(~buy, 0.0), "n": 1,
                      "big_buy": usd.where(buy & big, 0.0), "big_sell": usd.where(~buy & big, 0.0), "big_n": big.astype(int),
                      "small_buy": usd.where(buy & small, 0.0), "small_sell": usd.where(~buy & small, 0.0), "max_trade": usd})
    bars = f.groupby("ts").agg({c: ("max" if c == "max_trade" else "sum") for c in f.columns if c != "ts"})
    lvl = np.floor(np.log(t["price"]) / LVL).astype(np.int64)
    fp = pd.DataFrame({"ts": ts, "lvl": lvl, "buy": f["v_buy"], "sell": f["v_sell"]}).groupby(["ts", "lvl"]).sum().reset_index()
    return {"bars": bars, "fp": fp}


def cached_day(base: str, sym: str, day: pd.Timestamp) -> dict | None:
    path = os.path.join(CACHE, base, f"{day:%Y-%m-%d}.pkl")
    if os.path.exists(path):
        return pd.read_pickle(path)
    out = day_flow(sym, day)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.to_pickle(out, path)                       # і «файлу немає» теж — щоб не питати знову
    return out


def load_flow(jobs: list[tuple], workers: int) -> dict[str, tuple]:
    """jobs: (base, sym, day) → {base: (bars, fp)}."""
    res: dict[str, list] = {}
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(cached_day, b, s, d): b for b, s, d in jobs}
        for n, fu in enumerate(futs, 1):
            b = futs[fu]
            try:
                x = fu.result()
            except Exception as e:
                print(f"  ! {b}: {e}", flush=True)
                x = None
            if x is not None and len(x["bars"]):
                res.setdefault(b, []).append(x)
            if n % 500 == 0:
                print(f"  днів оброблено: {n}/{len(jobs)}", flush=True)
    return {b: (pd.concat([x["bars"] for x in xs]).sort_index(), pd.concat([x["fp"] for x in xs], ignore_index=True))
            for b, xs in res.items()}


# ---------------------------------------------------------------- ознаки

def fp_metrics(fp: pd.DataFrame, key: str) -> pd.DataFrame:
    """Футпрінт вікна (рядки: key, lvl, buy, sell) → дельта вгорі / внизу, обсяг угорі, позиція POC."""
    fp = fp.copy()
    fp["vol"] = fp["buy"] + fp["sell"]
    g = fp.groupby(key)
    lo, hi = g["lvl"].transform("min"), g["lvl"].transform("max")
    span = (hi - lo).where(hi - lo >= 2)
    fp["pos"] = (fp["lvl"] - lo) / span
    fp["d"] = fp["buy"] - fp["sell"]
    top, bot = fp["pos"] >= 2 / 3, fp["pos"] <= 1 / 3
    tot = g["vol"].sum()
    out = pd.DataFrame({"delta_top": fp["d"].where(top, 0).groupby(fp[key]).sum() / tot,
                        "delta_bot": fp["d"].where(bot, 0).groupby(fp[key]).sum() / tot,
                        "vol_top": fp["vol"].where(top, 0).groupby(fp[key]).sum() / tot,
                        "poc_pos": fp.loc[g["vol"].idxmax(), [key, "pos"]].set_index(key)["pos"]})
    flat = (g["lvl"].max() - g["lvl"].min()) < 2              # діапазон < 0.2% — футпрінт без сенсу
    out.loc[flat[flat].index] = np.nan
    return out


def flow_features(bars: pd.DataFrame, fp: pd.DataFrame) -> pd.DataFrame:
    b, week = 4, 7 * 96
    grid = pd.date_range(bars.index.min(), bars.index.max(), freq="15min", name="ts")
    x = bars.reindex(grid)                       # бар без угод = NaN (у архіві повний день — це справді «немає угод»)
    vol = x["v_buy"] + x["v_sell"]
    r = lambda s, n: s.rolling(n, min_periods=n).sum()
    v1, v4 = r(vol, b), r(vol, 4 * b)
    f = pd.DataFrame(index=grid)
    f["flow_big_share_1h"] = r(x["big_buy"] + x["big_sell"], b) / v1
    f["flow_big_share_z"] = zscore(f["flow_big_share_1h"], week)
    f["flow_big_delta_1h"] = r(x["big_buy"] - x["big_sell"], b) / v1
    f["flow_big_delta_4h"] = r(x["big_buy"] - x["big_sell"], 4 * b) / v4
    f["flow_small_delta_1h"] = r(x["small_buy"] - x["small_sell"], b) / v1
    f["flow_max_trade_z"] = zscore(np.log(x["max_trade"]), week)
    f["flow_order_size_z"] = zscore(np.log(v1 / r(x["n"], b)), week)
    f["flow_big_n_z"] = zscore(np.log1p(r(x["big_n"], b)), week)
    m15 = fp_metrics(fp, "ts").add_prefix("fp_")
    win = pd.concat([fp.assign(win=fp["ts"] + k * BAR) for k in range(4)]).groupby(["win", "lvl"])[["buy", "sell"]].sum().reset_index()
    m1h = fp_metrics(win, "win").add_prefix("fp_").add_suffix("_1h")
    m1h.index.name = "ts"
    f = f.join(m15).join(m1h)
    return f.replace([np.inf, -np.inf], np.nan)


def coin_features(flow: dict, base: str) -> pd.DataFrame:
    if base not in flow:
        return pd.DataFrame(columns=FLOW + FP + ["base_asset", "ts"])
    f = flow_features(*flow[base])
    return f.reset_index().assign(base_asset=base)


def days_between(a: pd.Timestamp, b: pd.Timestamp) -> list[pd.Timestamp]:
    return list(pd.date_range(a.floor("D"), b.floor("D"), freq="D"))


# ---------------------------------------------------------------- основне

def main() -> None:
    ap = argparse.ArgumentParser(description="Потік угод і футпрінт з архіву Binance як ознаки 15m-моделі")
    ap.add_argument("--workers", type=int, default=8, help="паралельних завантажень архіву (за замовчуванням 8)")
    ap.add_argument("--no-backtest", action="store_true", help="лише leave-ticker-out, без бектесту")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(".env")
    pd.set_option("display.width", 220)
    reviewed = {t.upper() for t in read_tickers_file("tickers.txt")}
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=48)
    # архів викладає день наступного дня: беремо лише повні дні з архіву
    arch_end = pd.Timestamp.now(tz="UTC").floor("D")
    lines = [f"# Потік угод і футпрінт (архів Binance) як ознаки 15m-моделі ({datetime.now(timezone.utc):%Y-%m-%d})", ""]

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        pumps = load_pumps(conn)
        trained = reviewed | set(pumps["base_asset"])
        d = build_from_raw(load_raw(conn, sorted(trained), INTERVAL_15M), pumps, bars_per_hour=4,
                           extra=lambda h, b: short_features(h, b))
        d = keep_training(d, reviewed, cutoff)
        d = d[d["ts"] < arch_end - BAR].reset_index(drop=True)
        syms = {b: binance_symbol(conn, b) for b in sorted(d["base_asset"].unique())}
        first = d.groupby("base_asset")["ts"].min()
        jobs = [(b, s, day) for b, s in syms.items() if s
                for day in days_between(first[b] - pd.Timedelta(days=8), arch_end - pd.Timedelta(days=1))]
        print(f"навчання: {len(d)} барів, {d['base_asset'].nunique()} тікерів (на Binance {sum(map(bool, syms.values()))}); "
              f"днів архіву: {len(jobs)}", flush=True)
        flow = load_flow(jobs, args.workers)
        tx = pd.concat([coin_features(flow, b) for b in syms], ignore_index=True)
        d = d.merge(tx, on=["base_asset", "ts"], how="left")
        d[FLOW + FP] = d[FLOW + FP].astype(float)
        # перевірка: обсяг з архіву проти обсягу Binance 15m у базі
        chk = []
        for b in list(flow)[:5]:
            raw = load_raw(conn, [b], INTERVAL_15M)
            o = raw["ohlcv"]
            o = o[o["exchange"] == "Binance"].set_index("ts")
            vb = o["v"] * np.where(o["volume_denominated"] == "BASE_ASSET", o["c"], 1)
            a = flow[b][0]
            j = pd.concat([vb.rename("db"), (a["v_buy"] + a["v_sell"]).rename("arch")], axis=1).dropna()
            chk.append(f"{b}: {len(j)} барів, архів / база = {(j['arch'].sum() / j['db'].sum()):.4f}")
        print("перевірка обсягу: " + "; ".join(chk), flush=True)

    cover = d[FLOW[0]].notna().mean()
    print(f"  ознаки потоку є на {cover:.0%} барів", flush=True)

    # чим перший бар імпульсу відрізняється від першого бару фальстарту
    def first_bars(kind):
        rows = []
        for p in pumps[pumps["pattern_type"] == kind].itertuples():
            g = d[(d["base_asset"] == p.base_asset) & (d["ts"] > p.start_ts - BAR) & (d["ts"] < p.end_ts)]
            if len(g):
                rows.append(g.iloc[0])
        return pd.DataFrame(rows)
    fi, ff = first_bars("pump"), first_bars("false_start")
    prof = pd.DataFrame({"звичайні бари": d[(d["label"] == 0) & d["pattern_type"].isna()][FLOW + FP].median(),
                         f"1-й бар імпульсу ({len(fi)})": fi[FLOW + FP].median(),
                         f"1-й бар фальстарту ({len(ff)})": ff[FLOW + FP].median(),
                         "усі бари імпульсу": d[d["label"] == 1][FLOW + FP].median()}).round(3)
    print(prof.to_string(), flush=True)

    variants = {"base": FEATURES_15M, "+ потік": FEATURES_15M + FLOW, "+ футпрінт": FEATURES_15M + FP,
                "+ потік + футпрінт": FEATURES_15M + FLOW + FP}
    rows, models = lto(d, variants, pumps, train.BAR_15M)
    print(pd.DataFrame(rows).T.to_string(), flush=True)
    imp = pd.Series(models["+ потік + футпрінт"][0].booster_.feature_importance("gain"), index=variants["+ потік + футпрінт"])
    imp = (imp / imp.sum()).sort_values(ascending=False)
    imp_lines = [f"- {f}: {v:.1%}" + (" ← потік" if f in FLOW else " ← футпрінт" if f in FP else "") for f, v in imp.head(15).items()]
    imp_lines.append(f"- (потік разом {imp[FLOW].sum():.1%}, футпрінт разом {imp[FP].sum():.1%})")
    print("\n".join(imp_lines), flush=True)
    lines += [f"Навчання: {len(d)} барів 15m, {d['base_asset'].nunique()} тікерів; ознаки потоку є на {cover:.0%} барів.",
              "Перевірка обсягу архів / база: " + "; ".join(chk), "", "## Медіани ознак", "", prof.to_markdown(), "",
              "## Leave-ticker-out (рівень «Кандидат»)", "", pd.DataFrame(rows).T.to_markdown(), "",
              "## Важливість (+ потік + футпрінт, gain), топ-15", ""] + imp_lines + [""]

    if not args.no_backtest:
        import exit_partial as P
        m15 = joblib.load(train.model_path(train.M15_NAME))
        m1 = joblib.load(train.model_path("lightgbm"))
        k, exit_bars, share = m15.get("risk", {}).get("k"), m15.get("exit_bars", 4), m15.get("confirm", {}).get("share", 0.75)
        coins, band = set(), {}
        for path in glob.glob(os.path.join("reports", "backtests", "2026*_alert.json")):
            meta = json.load(open(path, encoding="utf-8"))["meta"]
            for u in meta["universe"]:
                coins.add(u["base"])
                band[u["base"]] = "≥ 20 млн" if meta["min_turnover"] >= 20e6 else "5–20 млн"
        coins = sorted(coins - trained)
        with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
            end = min(pd.Timestamp(conn.execute("SELECT max(ts) FROM ohlcv WHERE interval = '15min'").fetchone()[0]), arch_end)
            start = end - pd.Timedelta(days=DAYS)
            since = start - pd.Timedelta(days=9)
            csyms = {b: binance_symbol(conn, b) for b in coins}
            jobs = [(b, s, day) for b, s in csyms.items() if s for day in days_between(since, end - pd.Timedelta(days=1))]
            print(f"бектест: {len(coins)} монет, {start:%m-%d}…{end:%m-%d}; днів архіву: {len(jobs)}", flush=True)
            cflow = load_flow(jobs, args.workers)
            res = {f"{name} · {e}": [] for name in variants for e in ("поточний вихід", "частковий вихід")}
            for n, b in enumerate(coins, 1):
                raw = load_raw(conn, [b], INTERVAL_15M, since=since.to_pydatetime())
                f15 = build_from_raw(raw, None, bars_per_hour=4, extra=lambda h, bb: short_features(h, bb))
                d1 = build_hourly_own(conn, [b], since=since.to_pydatetime())
                if f15.empty or d1.empty:
                    continue
                f15 = f15.merge(coin_features(cflow, b), on=["base_asset", "ts"], how="left")
                f15[FLOW + FP] = f15[FLOW + FP].astype(float)
                s1 = d1[["base_asset", "ts"]].assign(probability=m1["model"].predict_proba(d1[FEATURES])[:, 1], threshold=m1["threshold"])
                h = aggregate(raw, b, "15min")
                h = h[(h.index >= start - pd.Timedelta(days=1)) & (h.index < end)]
                for name, (model, thr, cols) in models.items():
                    s15 = f15[["base_asset", "ts"]].assign(probability=model.predict_proba(f15[cols])[:, 1], threshold=thr)
                    t = tier_frame(s15, s1, share).set_index("ts").sort_index()
                    sc = t[(t.index >= start) & (t.index < end)]["alert_score"]
                    for x in P.simulate(h, sc, k, exit_bars, None, 0, False):
                        res[f"{name} · поточний вихід"].append({"ticker": b, "ret": x["ret"], "stop": x["reason"] == "stop", "band": band.get(b, "")})
                    for x in P.simulate(h, sc, k, exit_bars, 3.0, 0.5, True):
                        res[f"{name} · частковий вихід"].append({"ticker": b, "ret": x["ret"], "stop": x["reason"] == "stop", "band": band.get(b, "")})
                if n % 25 == 0:
                    print(f"  {n}/{len(coins)}", flush=True)
        bt = pd.DataFrame({name: stats(pd.DataFrame(r)) if r else {"угод": 0} for name, r in res.items()}).T
        print(bt.to_string(), flush=True)
        lines += [f"## Бектест: {len(coins)} монет поза навчанням, {start:%m-%d}…{end:%m-%d}", "", bt.to_markdown(), ""]

    os.makedirs("reports", exist_ok=True)
    open(OUT, "w", encoding="utf-8").write("\n".join(lines))
    print(f"\nзвіт: {OUT}")


if __name__ == "__main__":
    main()
