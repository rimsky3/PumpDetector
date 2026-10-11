"""
experiments/fe_spot.py — СПОТ ПРОТИ Ф'ЮЧЕРСА і ВІК МОНЕТИ як ознаки (2026-10-11; власник: «що ще можна взяти з Binance?», «так,
запускай»). Перевірено раніше лише на 6 з 10 тікерів (volume_delta.py); тут — архів Binance для всіх монет.
Етап data (публічний архів data.binance.vision, без акаунта):
  * спот: 15m-свічки <BASE>USDT (для 1000X / 1000000X — спот X, обсяг у $ не залежить від множника) за березень–вересень (місячні
    файли) + денні 01–09.10 для тікерів власника → data/spot15/<монета>.parquet (o h l c, v $, bv $ — покупки-агресори, n);
    немає спот-пари — монета «лише ф'ючерс»;
  * лістинг ф'ючерса: перший день денних свічок архіву → data/spot15/_listing.json.
Ознаки (зафіксовано ДО запуску; лише минуле):
  spot   has_spot (є спот-пара на Binance), spot_share_24h (спот-оборот ÷ (спот + ф'ючерс) за 24 год), spot_share_x (частка за 4 год ÷
         медіана частки за 7 днів), spot_buy_1h / 4h (частка покупок-агресорів на споті), spot_lead_1h / 4h (log-рух спот-ціни − рух
         ф'ючерса), spot_vol_z (спот-оборот 1 год, z-score за 7 днів)
  age    log_days_listed — log днів від лістингу ф'ючерса на Binance
Варіанти: base, +spot, +age, +both; random seeds 42 / 43 / 44; LTO і бектест 6 міс. як fe_anchor.py.
КРИТЕРІЙ (до запуску): як fe_anchor — проти base у парах за seed: ≥ 20 млн середня різниця ≥ +0.02 і краще на ≥ 2 з 3; середній
загальний PF ≥ base; обидві половини ≥ base; feature importance групи ≥ 3%.
→ reports/fe_spot.md

    python experiments/fe_spot.py --stage data
    python experiments/fe_spot.py --stage test
"""

import argparse
import io
import json
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

DIR = os.path.join("data", "spot15")
LISTING = os.path.join(DIR, "_listing.json")
URL = "https://data.binance.vision/data/spot/{k}/klines/{s}/15m/{s}-15m-{d}.zip"
URL_FUT1D = "https://data.binance.vision/data/futures/um/monthly/klines/{s}/1d/{s}-1d-{m}.zip"
MONTHS = ["2026-03", "2026-04", "2026-05", "2026-06", "2026-07", "2026-08", "2026-09"]
OCT_DAYS = [f"2026-10-{d:02d}" for d in range(1, 10)]
KCOLS = ["open_time", "o", "h", "l", "c", "vol", "close_time", "v", "n", "bvol", "bv", "ignore"]
OUT = os.path.join("reports", "fe_spot.md")
_s = requests.Session()


def spot_base(base: str) -> str:
    for p in ("1000000", "1000"):
        if base.startswith(p) and len(base) > len(p):
            return base[len(p):]
    return base


def _zip_csv(url: str) -> pd.DataFrame | None:
    try:
        r = _s.get(url, timeout=60)
    except requests.RequestException:
        return None
    if r.status_code != 200:
        return None
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        raw = z.read(z.namelist()[0])
    return pd.read_csv(io.BytesIO(raw), header=0 if raw[:9] == b"open_time" else None, names=KCOLS)


def fetch_spot(base: str, days: list[str]) -> str:
    path = os.path.join(DIR, f"{base}.parquet")
    if os.path.exists(path):
        return "кеш"
    sym = quote(f"{spot_base(base)}USDT")
    parts = [p for p in [_zip_csv(URL.format(k="monthly", s=sym, d=m)) for m in MONTHS] + [_zip_csv(URL.format(k="daily", s=sym, d=d)) for d in days]
             if p is not None and len(p)]
    os.makedirs(DIR, exist_ok=True)
    if not parts:
        pd.DataFrame(columns=["o", "h", "l", "c", "v", "bv", "n"]).to_parquet(path)      # спот-пари немає
        return "немає споту"
    k = pd.concat(parts)
    t = k["open_time"].astype("int64")
    t = np.where(t > 1e14, t // 1000, t)                     # спот-архів з 2025 р. — мікросекунди
    out = pd.DataFrame({c: k[c].astype(float).to_numpy() for c in ("o", "h", "l", "c", "v", "bv", "n")},
                       index=pd.to_datetime(t, unit="ms", utc=True))
    out = out[~out.index.duplicated()].sort_index()
    out.to_parquet(path)
    return f"{len(out)} барів"


S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"


def listing_day(base: str) -> str | None:
    """Перший день торгів ф'ючерса <BASE>USDT: перший місячний файл денних свічок за списком архіву (S3) → його перший рядок."""
    import re
    sym = f"{base}USDT"
    try:
        r = _s.get(S3, params={"prefix": f"data/futures/um/monthly/klines/{sym}/1d/", "max-keys": 1000}, timeout=60)
    except requests.RequestException:
        return None
    months = sorted(re.findall(r"-1d-(\d{4}-\d{2})\.zip</Key>", r.text))
    if not months:
        return None
    k = _zip_csv(URL_FUT1D.format(s=quote(sym), m=months[0]))
    if k is None or not len(k):
        return f"{months[0]}-01"
    t = int(k["open_time"].astype("int64").min())
    return pd.to_datetime(t // 1000 if t > 1e14 else t, unit="ms", utc=True).strftime("%Y-%m-%d")


def stage_data() -> None:
    import early_target as ET
    import idio_features as I
    from fetch import read_tickers_file
    uni = json.load(open(os.path.join(I.DIR, "_universe.json"), encoding="utf-8"))
    own = {t.upper() for t in read_tickers_file("tickers.txt")} | set(pd.read_csv("pumps.csv")["base_asset"].str.upper())
    coins = sorted(set(uni) | own)
    print(f"монет {len(coins)} (власника {len(own)}) → {DIR}", flush=True)
    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(lambda c: (c, fetch_spot(c, OCT_DAYS if c in own else [])), coins))
    print(f"  спот є: {sum(not m.startswith('немає') for _, m in res)} з {len(res)}", flush=True)
    lst = json.load(open(LISTING, encoding="utf-8")) if os.path.exists(LISTING) else {}
    todo = [c for c in coins if c not in lst]
    with ThreadPoolExecutor(6) as ex:
        for c, d in zip(todo, ex.map(listing_day, todo)):
            lst[c] = d
    json.dump(lst, open(LISTING, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    print(f"  дат лістингу: {sum(v is not None for v in lst.values())} з {len(coins)}; готово", flush=True)
    _ = ET


GROUPS = {"spot": ["has_spot", "spot_share_24h", "spot_share_x", "spot_buy_1h", "spot_buy_4h", "spot_lead_1h", "spot_lead_4h", "spot_vol_z"],
          "age": ["log_days_listed"]}
ALL = [f for fs in GROUPS.values() for f in fs]
SEEDS = [42, 43, 44]
BIG = "≥ 20 млн"
W7 = 7 * 96


def spot_features(base: str, h: pd.DataFrame, listing: dict) -> pd.DataFrame:
    """15m-ряд ф'ючерса (price, volume у $) + спот-свічки монети + дата лістингу → ознаки на барах h (лише минуле)."""
    f = pd.DataFrame(index=h.index)
    path = os.path.join(DIR, f"{base}.parquet")
    sp = pd.read_parquet(path) if os.path.exists(path) else pd.DataFrame()
    f["has_spot"] = float(len(sp) > 0)
    pv = h["volume"].astype(float)
    if len(sp):
        s = sp.reindex(h.index)
        sv, sb, sc = s["v"], s["bv"], s["c"]
        roll = lambda x, n: x.rolling(n, min_periods=n // 2).sum()                       # noqa: E731
        f["spot_share_24h"] = roll(sv, 96) / (roll(sv, 96) + roll(pv, 96))
        sh4 = roll(sv, 16) / (roll(sv, 16) + roll(pv, 16))
        f["spot_share_x"] = sh4 / sh4.rolling(W7, min_periods=96).median().shift(1)
        f["spot_buy_1h"] = roll(sb, 4) / roll(sv, 4)
        f["spot_buy_4h"] = roll(sb, 16) / roll(sv, 16)
        lp, ls = np.log(h["price"].where(h["price"] > 0)), np.log(sc.where(sc > 0))
        f["spot_lead_1h"] = ls.diff(4) - lp.diff(4)
        f["spot_lead_4h"] = ls.diff(16) - lp.diff(16)
        v1 = roll(sv, 4)
        f["spot_vol_z"] = (v1 - v1.rolling(W7, min_periods=96).mean().shift(1)) / v1.rolling(W7, min_periods=96).std().shift(1)
    else:
        for k in GROUPS["spot"][1:]:
            f[k] = np.nan
    d0 = listing.get(base)
    f["log_days_listed"] = (np.log1p(np.clip(np.asarray((h.index - pd.Timestamp(d0, tz="UTC")) / pd.Timedelta(days=1), dtype=float), 0, None))
                            if d0 else np.nan)
    return f[ALL].replace([np.inf, -np.inf], np.nan)


def stage_test() -> None:
    import pickle
    import joblib
    import auto_train as AT
    import early_target as ET
    import exit_partial as P
    import idio_features as I
    import train
    from tiers import tier_frame
    listing = json.load(open(LISTING, encoding="utf-8"))
    variants = {"base": ET.COLS, "+spot": ET.COLS + GROUPS["spot"], "+age": ET.COLS + GROUPS["age"], "+both": ET.COLS + ALL}
    d, pumps, px, idx, uni = ET.load_data()
    trained = set(d["base_asset"])
    d = d.join(pd.concat([spot_features(b, px[b], listing).reindex(g["ts"]).set_axis(g.index) for b, g in d.groupby("base_asset")]))
    print(f"навчання: {len(d)} барів; спот є у {d.groupby('base_asset')['has_spot'].first().sum():.0f} з {d['base_asset'].nunique()} тікерів", flush=True)
    y, grp = d["label"].to_numpy().astype(int), d["base_asset"]
    lgbm = lambda sd: train.make_lightgbm().set_params(random_state=sd)       # noqa: E731
    models, thr, lto, imp = {}, {}, {}, {}
    for v, cols in variants.items():
        for sd in SEEDS:
            key = (v, sd)
            p = train.oof_proba(lambda s=sd: lgbm(s), d[cols], pd.Series(y), grp)
            thr[key] = train.best_f1_threshold(y, p)
            sig = p >= thr[key]
            lto[key] = {**train.event_metrics(d, sig, pumps, ET.BAR), **ET.early(d, sig, pumps, px)}
            models[key] = lgbm(sd).fit(d[cols], y)
            g = pd.Series(models[key].booster_.feature_importance("gain"), index=cols)
            imp[key] = g / g.sum()
            print(f"  {v} s{sd} (поріг {thr[key]:.3f}): хибних {lto[key]['false_alarms_per_30d']:.1f}, спіймано {lto[key]['caught']}", flush=True)
    m1 = joblib.load(train.model_path("lightgbm"))
    m15 = joblib.load(train.model_path(train.M15_NAME))
    exit_bars, share = m15.get("exit_bars", 4), m15.get("confirm", {}).get("share", 0.75)
    start, end = pd.Timestamp(I.START, tz="UTC"), pd.Timestamp(I.END, tz="UTC") + pd.Timedelta(days=1)
    coins = sorted(c for c in uni if c not in trained and os.path.exists(os.path.join(I.DIR, f"{c}.pkl")))
    print(f"бектест: {len(coins)} монет…", flush=True)
    rows = []
    for n, c in enumerate(coins, 1):
        x = pickle.load(open(os.path.join(I.DIR, f"{c}.pkl"), "rb"))
        f15, h1, h = x["f15"], x["h1"], x["h"]
        f15 = f15.join(I.new_features(h, idx).reindex(f15["ts"]).set_axis(f15.index))
        f15 = f15.join(spot_features(c, h, listing).reindex(f15["ts"]).set_axis(f15.index))
        hh = h[(h.index >= start - pd.Timedelta(days=1)) & (h.index < end)]
        s1 = h1[["base_asset", "ts"]].assign(probability=m1["model"].predict_proba(h1[m1["features"]])[:, 1], threshold=m1["threshold"])
        for (v, sd), model in models.items():
            s15 = f15[["base_asset", "ts"]].assign(probability=model.predict_proba(f15[variants[v]])[:, 1], threshold=thr[(v, sd)])
            t = tier_frame(s15, s1, share).set_index("ts").sort_index()
            for r in P.simulate(hh, t.loc[(t.index >= start) & (t.index < end), "alert_score"], I.K, exit_bars, 3.0, 0.5, True):
                mth = pd.Timestamp(r["ts"]).strftime("%Y-%m")
                if mth in uni[c]:
                    rows.append({"variant": v, "seed": sd, "ticker": c, "half": AT.half(c), "month": mth, "band": uni[c][mth], **r})
        if n % 50 == 0:
            print(f"  {n}/{len(coins)}", flush=True)
    tr = pd.DataFrame(rows)
    tr.to_csv(OUT.replace(".md", "_trades.csv"), index=False)
    pf = I.pf
    V = list(variants)
    cell = {}
    for v in V:
        for sd in SEEDS:
            g = tr[(tr["variant"] == v) & (tr["seed"] == sd)]
            b = g[g["band"] == BIG]
            cell[(v, sd)] = {"pf": pf(g["ret"]), "pf20": pf(b["ret"]), "A": pf(b[b["half"] == "A"]["ret"]), "B": pf(b[b["half"] == "B"]["ret"]), "n": len(g)}
    mean = lambda v, k: float(np.mean([cell[(v, sd)][k] for sd in SEEDS]))       # noqa: E731
    L = ["# Спот проти ф'ючерса і вік монети", "",
         f"Групи: spot ({len(GROUPS['spot'])}), age ({len(GROUPS['age'])}). Random seeds {SEEDS}, значення — середнє.", "",
         "**Критерій (до запуску):** проти base у парах за seed — ≥ 20 млн середня різниця ≥ +0.02 і краще на ≥ 2 з 3; загальний ≥ base; "
         "обидві половини ≥ base; feature importance групи ≥ 3%.", "",
         "| варіант | спіймано | хибних / 30 дн | gain нових | угод | PF | ≥ 20 млн | різниця ≥ 20 млн по seeds | A ≥ 20 млн | B ≥ 20 млн |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for v in V:
        new = [q for q in variants[v] if q in ALL]
        sh = np.mean([imp[(v, sd)][new].sum() for sd in SEEDS]) if new else 0.0
        diffs = [cell[(v, sd)]["pf20"] - cell[("base", sd)]["pf20"] for sd in SEEDS]
        L.append(f"| {v} | {np.mean([int(lto[(v, sd)]['caught'].split('/')[0]) for sd in SEEDS]):.0f}/128 | "
                 f"{np.mean([lto[(v, sd)]['false_alarms_per_30d'] for sd in SEEDS]):.1f} | {sh:.1%} | {mean(v, 'n'):.0f} | {mean(v, 'pf'):.3f} | "
                 f"{mean(v, 'pf20'):.3f} | " + ("—" if v == "base" else " / ".join(f"{q:+.3f}" for q in diffs)) +
                 f" | {mean(v, 'A'):.3f} | {mean(v, 'B'):.3f} |")
    allimp = pd.concat([imp[("+both", sd)] for sd in SEEDS], axis=1).mean(axis=1)
    L += ["", "Нові ознаки в моделі +both: " + ", ".join(f"{k} {allimp[k]:.1%}" for k in sorted(ALL, key=lambda q: -allimp[q])) + ".", "", "## Критерій", ""]
    for v in V[1:]:
        diffs = np.array([cell[(v, sd)]["pf20"] - cell[("base", sd)]["pf20"] for sd in SEEDS])
        new = [q for q in variants[v] if q in ALL]
        sh = np.mean([imp[(v, sd)][new].sum() for sd in SEEDS])
        ok = (diffs.mean() >= 0.02 and (diffs > 0).sum() >= 2 and mean(v, "pf") >= mean("base", "pf")
              and mean(v, "A") >= mean("base", "A") and mean(v, "B") >= mean("base", "B") and sh >= 0.03)
        L.append(f"- {v}: ≥ 20 млн {diffs.mean():+.3f} (краще на {(diffs > 0).sum()} з {len(SEEDS)}); gain {sh:.1%} → **{'ВИКОНАНО' if ok else 'НЕ виконано'}**")
    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Спот проти ф'ючерса і вік монети як ознаки 15m-моделі (архів Binance)")
    ap.add_argument("--stage", choices=["data", "test"], required=True, help="data — спот-свічки і дати лістингу; test — моделі й бектест")
    args = ap.parse_args()
    stage_data() if args.stage == "data" else stage_test()


if __name__ == "__main__":
    main()
