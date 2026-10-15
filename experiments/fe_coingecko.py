"""
experiments/fe_coingecko.py — КАПІТАЛІЗАЦІЯ І СЕКТОР з CoinGecko як ознаки 15m-моделі (2026-10-15; план 03.10, власник: «так»).
Чого в моделі немає зовсім: розміру монети (OI і оборот відносно капіталізації — «перегрів плечем») і того, чи «горить» її сектор.

Етап cg (Demo-ключ COINGECKO_API_KEY у .env; ~1–1.5 тис. запитів з 10 тис. на місяць; кеш data/coingecko/):
  * /coins/list → кандидати за символом (1000X / 1000000X → X, множник ціни); /coins/markets — капіталізація кандидатів
    (перевіряються до 3 найбільших);
  * /coins/{id}/market_chart?days=240&interval=daily — поденні ціна і капіталізація (погодинно Demo дає лише вікнами ≤ 90 днів);
    кандидат приймається, якщо медіана |log(ціна CoinGecko × множник ÷ наша ціна)| < 3% на ≥ 20 спільних точках;
  * /coins/{id} — категорії, total / max supply (ПОТОЧНІ: легкий погляд уперед, у звіті).
Ознаки (зафіксовано ДО запуску, notes/experiments.md 15.10; лише минуле):
  cap     oi_mcap      log(OI $ ÷ капіталізація); капіталізація = монет в обігу за ПОПЕРЕДНЮ добу (mcap ÷ ціна CoinGecko) × наша ціна
          vol_mcap_24h log(оборот ф'ючерса Binance за 24 год ÷ капіталізація)
          fdv_ratio    log(max supply, або total, ÷ монет в обігу) — тиск майбутніх розлоків
  sector  sec_ret_4h / sec_ret_24h  середня по категоріях монети дохідність ІНШИХ монет категорії мінус середня по всіх монетах набору;
          sec_max_4h   те саме для найгарячішої категорії монети. Категорія — ≥ 5 монет у наборі, без загальних
          («Ecosystem», «Portfolio», «Index», біржових); на барі — ≥ 3 інших монети з ціною.
Варіанти base / +cap / +sector / +both; random seeds 42 / 43 / 44; LTO і бектест 6 міс. (01.04–30.09, чесний список) як fe_spot.py.
КРИТЕРІЙ (до запуску): проти base у парах за seed — ≥ 20 млн середня різниця PF ≥ +0.02 і краще на ≥ 2 з 3; загальний PF ≥ base;
обидві половини монет ≥ base; feature importance (gain) групи ≥ 3%.
→ reports/fe_coingecko.md. Дані CoinGecko (attribution обов'язкова: «Data: CoinGecko»).

    python experiments/fe_coingecko.py --stage cg
    python experiments/fe_coingecko.py --stage test
"""

import argparse
import glob
import json
import os
import pickle
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

DIR = os.path.join("data", "coingecko")
MAP = os.path.join(DIR, "_map.json")
API = "https://api.coingecko.com/api/v3/"
PAUSE = 2.2                     # с між запитами (~27 / хв; Demo дозволяє більше, але запас від 429)
OUT = os.path.join("reports", "fe_coingecko.md")
GROUPS = {"cap": ["oi_mcap", "vol_mcap_24h", "fdv_ratio"], "sector": ["sec_ret_4h", "sec_ret_24h", "sec_max_4h"]}
ALL = [f for fs in GROUPS.values() for f in fs]
SEEDS = [42, 43, 44]
BIG = "≥ 20 млн"
B = 96
MKT_MIN = 20                    # «ринок» для сектора — середня по монетах набору, якщо на барі є ціна ≥ MKT_MIN монет
GENERIC = ("ecosystem", "portfolio", "index", "binance", "coinbase", "launch", "alpha", "made in", "proof of", "upbit", "bithumb",
           "exchange-based", "listing", "fdv", "market cap", "smart contract", "layer 1 (l1)", "alleged", "holdings", "x402")
_s = requests.Session()
_last = [0.0]


def get(path: str, **params):
    """Запит до CoinGecko з паузою і повтором на 429 / помилках мережі. Ключ — лише в заголовку, не друкується."""
    for attempt in range(6):
        wait = PAUSE - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
        try:
            r = _s.get(API + path, params=params, timeout=60, headers={"x-cg-demo-api-key": os.environ["COINGECKO_API_KEY"]})
        except requests.RequestException:
            time.sleep(10)
            continue
        if r.status_code == 429:
            time.sleep(60)
            continue
        if r.status_code == 404:
            return None
        if r.status_code >= 500:
            time.sleep(15)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"CoinGecko не відповідає: {path}")


def cached(name: str, path: str, **params):
    f = os.path.join(DIR, name)
    if os.path.exists(f):
        return json.load(open(f, encoding="utf-8"))
    j = get(path, **params)
    os.makedirs(os.path.dirname(f), exist_ok=True)
    json.dump(j, open(f, "w", encoding="utf-8"))
    return j


def split_mult(base: str) -> tuple[str, float]:
    for p in ("1000000", "1000"):
        if base.startswith(p) and len(base) > len(p):
            return base[len(p):], float(p)
    return base, 1.0


def coin_set() -> tuple[list[str], list[str]]:
    """Монети бектесту (чесний список long_bin) і навчання (tickers.txt + розмічені, без виключених)."""
    import idio_features as I
    from fetch import excluded_tickers, read_tickers_file
    uni = json.load(open(os.path.join(I.DIR, "_universe.json"), encoding="utf-8"))
    exc = excluded_tickers(".")
    own = ({t.upper() for t in read_tickers_file("tickers.txt")} | set(pd.read_csv("pumps.csv")["base_asset"].str.upper())) - exc
    return sorted(uni), sorted(own)


def our_prices(conn, coins: list[str]) -> dict[str, pd.Series]:
    """15m-ціна закриття Binance кожної монети: кеш архіву (long_bin, 24.03–30.09) + локальна база (навчальні, 25.07–…)."""
    import idio_features as I
    from features import INTERVAL_15M, aggregate, load_raw, split_raw
    out = {}
    for c in coins:
        p = os.path.join(I.DIR, f"{c}.pkl")
        if os.path.exists(p):
            out[c] = pickle.load(open(p, "rb"))["h"]["price"]
    raw = load_raw(conn, coins, INTERVAL_15M, exchanges=["Binance"], light=True)
    for c, r in split_raw(raw).items():
        a = aggregate(r, c, "15min", light=True)
        if len(a):
            out[c] = pd.concat([out[c], a["price"]]).groupby(level=0).last() if c in out else a["price"]
    return out


def stage_cg() -> None:
    import psycopg
    uni, own = coin_set()
    coins = sorted(set(uni) | set(own))
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        px = our_prices(conn, coins)
    lst = cached("_list.json", "coins/list")
    by_sym: dict[str, list[str]] = {}
    for x in lst:
        by_sym.setdefault(x["symbol"].lower(), []).append(x["id"])
    cand = {c: by_sym.get(split_mult(c)[0].lower(), []) for c in coins}
    ids = sorted({i for v in cand.values() for i in v})
    caps = {}
    for k in range(0, len(ids), 250):
        chunk = ids[k:k + 250]
        j = cached(f"_markets_{k // 250}.json", "coins/markets", vs_currency="usd", ids=",".join(chunk), per_page=250) or []
        caps.update({x["id"]: x.get("market_cap") or 0 for x in j})
    mp = json.load(open(MAP, encoding="utf-8")) if os.path.exists(MAP) else {}
    print(f"монет {len(coins)} (бектест {len(uni)}, навчання {len(own)}); з кандидатами за символом {sum(bool(v) for v in cand.values())}", flush=True)
    for n, c in enumerate(coins, 1):
        if c in mp:
            continue
        sym, mult = split_mult(c)
        best = {"id": None, "mult": mult, "err": None, "n": 0, "why": "немає символу" if not cand[c] else "ціна не збігається"}
        ours = px.get(c)
        for cid in sorted(cand[c], key=lambda i: -caps.get(i, 0))[:3]:
            j = cached(f"chart/{cid}.json", f"coins/{cid}/market_chart", vs_currency="usd", days=240, interval="daily")
            if not j or not j.get("prices") or ours is None:
                continue
            g = pd.Series([v for _, v in j["prices"]], index=pd.to_datetime([t for t, _ in j["prices"]], unit="ms", utc=True))
            o = ours.reindex(g.index.floor("15min"))
            ok = (g.to_numpy() > 0) & (o.to_numpy() > 0)
            if ok.sum() < 20:
                continue
            err = float(np.median(np.abs(np.log(g.to_numpy()[ok] * mult / o.to_numpy()[ok]))))
            if best["err"] is None or err < best["err"]:
                best.update(id=cid, err=err, n=int(ok.sum()))
        if best["id"] is not None and best["err"] >= 0.03:
            best.update(why=f"ціна не збігається ({best['err']:.1%})", id=None)
        if best["id"] is not None:
            best["why"] = ""
            info = cached(f"coin/{best['id']}.json", f"coins/{best['id']}", localization="false", tickers="false",
                          community_data="false", developer_data="false", sparkline="false")
            md = (info or {}).get("market_data", {})
            best.update(categories=(info or {}).get("categories") or [], total=md.get("total_supply"), max=md.get("max_supply"))
        mp[c] = best
        if n % 20 == 0:
            json.dump(mp, open(MAP, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
            print(f"  {n}/{len(coins)}: зіставлено {sum(v['id'] is not None for v in mp.values())}", flush=True)
    json.dump(mp, open(MAP, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    ok = [c for c in coins if mp[c]["id"]]
    miss = ", ".join(c + " (" + mp[c]["why"] + ")" for c in coins if not mp[c]["id"])
    print(f"готово: зіставлено {len(ok)} з {len(coins)} (навчання {sum(c in ok for c in own)} з {len(own)}); не зіставлено: {miss[:1500]}", flush=True)


# ---------------------------------------------------------------- ознаки

def supply_series(cid: str) -> pd.Series:
    """Монет в обігу (капіталізація ÷ ціна) з поденного ряду CoinGecko; індекс — момент, з якого значення ВІДОМЕ (+1 доба)."""
    j = json.load(open(os.path.join(DIR, "chart", f"{cid}.json"), encoding="utf-8"))
    p = pd.Series([v for _, v in j["prices"]], index=pd.to_datetime([t for t, _ in j["prices"]], unit="ms", utc=True))
    m = pd.Series([v for _, v in j["market_caps"]], index=pd.to_datetime([t for t, _ in j["market_caps"]], unit="ms", utc=True))
    s = (m / p.reindex(m.index)).where((m > 0) & (p.reindex(m.index) > 0)).dropna()
    s.index = s.index + pd.Timedelta(days=1)
    return s[~s.index.duplicated(keep="last")].sort_index()


def cap_features(c: str, h: pd.DataFrame, mp: dict) -> pd.DataFrame:
    """h: 15m-ряд монети з колонками price, volume ($, ф'ючерс Binance), oi ($) → група cap."""
    f = pd.DataFrame(index=h.index, columns=GROUPS["cap"], dtype=float)
    m = mp.get(c) or {}
    if not m.get("id"):
        return f
    s = supply_series(m["id"])
    if s.empty:
        return f
    sup = pd.Series(s.to_numpy(), index=s.index).reindex(h.index.union(s.index)).ffill().reindex(h.index)
    sup = sup.where(h.index >= s.index[0])
    mcap = sup * h["price"] / m["mult"]
    mcap = mcap.where(mcap > 0)
    f["oi_mcap"] = np.log(h["oi"].where(h["oi"] > 0) / mcap)
    f["vol_mcap_24h"] = np.log(h["volume"].rolling(B, min_periods=B // 2).sum().where(lambda x: x > 0) / mcap)
    cap_sup = m.get("max") or m.get("total")
    f["fdv_ratio"] = np.log((cap_sup / sup).clip(lower=1.0)) if cap_sup else np.nan
    return f.replace([np.inf, -np.inf], np.nan)


def sector_table(px: dict[str, pd.Series], mp: dict) -> dict[str, pd.DataFrame]:
    """Ознаки сектора для всіх монет з категоріями: {монета: DataFrame(sec_ret_4h, sec_ret_24h, sec_max_4h) на 15m-сітці}."""
    cats: dict[str, list[str]] = {}
    for c, m in mp.items():
        if m.get("id") and c in px:
            for k in m.get("categories") or []:
                if not any(g in k.lower() for g in GENERIC):
                    cats.setdefault(k, []).append(c)
    cats = {k: v for k, v in cats.items() if len(v) >= 5}
    coins = sorted({c for v in cats.values() for c in v})
    grid = pd.date_range(min(px[c].index.min() for c in coins), max(px[c].index.max() for c in coins), freq="15min")
    lp = pd.DataFrame({c: np.log(px[c].where(px[c] > 0)).reindex(grid) for c in coins})
    allp = pd.DataFrame({c: np.log(s.where(s > 0)).reindex(grid) for c, s in px.items()})
    K = list(cats)
    M = np.zeros((len(coins), len(K)))
    for j, k in enumerate(K):
        for c in cats[k]:
            M[coins.index(c), j] = 1
    out = {c: pd.DataFrame(index=grid) for c in coins}
    for w, name in ((16, "4h"), (B, "24h")):
        R = lp.diff(w).to_numpy()
        mkt_df = allp.diff(w)
        mkt = mkt_df.mean(axis=1).where(mkt_df.notna().sum(axis=1) >= MKT_MIN).to_numpy()
        have = ~np.isnan(R)
        S, N = np.nan_to_num(R) @ M, have.astype(float) @ M
        for i, c in enumerate(coins):
            ks = np.flatnonzero(M[i])
            own, oh = np.nan_to_num(R[:, i])[:, None], have[:, i].astype(float)[:, None]
            n = N[:, ks] - oh
            m = np.where(n >= 3, (S[:, ks] - own) / np.where(n > 0, n, 1), np.nan)
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)       # порожні рядки (немає ≥ 3 інших монет) → NaN
                out[c][f"sec_ret_{name}"] = np.nanmean(m, axis=1) - mkt
                if name == "4h":
                    out[c]["sec_max_4h"] = np.nanmax(m, axis=1) - mkt
    print(f"сектори: {len(K)} категорій з ≥ 5 монет, монет з сектором {len(coins)}", flush=True)
    return {c: f[GROUPS["sector"]] for c, f in out.items()}


def oct_prices(coins: list[str]) -> dict[str, pd.Series]:
    """Ціна монет бектесту за 01.10 – позавчора (архів Binance, кеш binance_archive): щоб сектор на навчальних барах жовтня мав ті самі монети."""
    import binance_archive as BA
    from concurrent.futures import ThreadPoolExecutor
    days = [d.strftime("%Y-%m-%d") for d in pd.date_range("2026-10-01", (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=2)).strftime("%Y-%m-%d"))]

    def one(c):
        parts = [BA._csv("klines", f"{c}USDT", d, "15m") for d in days]
        parts = [p for p in parts if len(p)]
        if not parts:
            return c, None
        k = pd.concat(parts)
        t = k["open_time"].astype("int64")
        return c, pd.Series(k["close"].astype(float).to_numpy(), index=pd.to_datetime(np.where(t > 1e14, t // 1000, t), unit="ms", utc=True))

    with ThreadPoolExecutor(8) as ex:
        return {c: s for c, s in ex.map(one, coins) if s is not None}


def archive_oi(c: str, idx: pd.DatetimeIndex) -> pd.Series:
    """OI $ монети бектесту на закритті 15m-бару (знімок metrics архіву Binance на ts + 15 хв)."""
    files = sorted(glob.glob(os.path.join("data", "binance_archive", "metrics", f"{c}USDT", "*.parquet")))
    if not files:
        return pd.Series(np.nan, index=idx)
    m = pd.concat([pd.read_parquet(f, columns=["create_time", "sum_open_interest_value"]) for f in files])
    s = pd.Series(m["sum_open_interest_value"].astype(float).to_numpy(), index=pd.to_datetime(m["create_time"], utc=True))
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return pd.Series(s.reindex(idx + pd.Timedelta(minutes=15)).to_numpy(), index=idx)


def stage_test() -> None:
    import joblib
    import psycopg
    import auto_train as AT
    import early_target as ET
    import exit_partial as P
    import idio_features as I
    import train
    from features import INTERVAL_15M, aggregate, load_raw, split_raw
    from tiers import tier_frame
    mp = json.load(open(MAP, encoding="utf-8"))
    variants = {"base": ET.COLS, "+cap": ET.COLS + GROUPS["cap"], "+sector": ET.COLS + GROUPS["sector"], "+both": ET.COLS + ALL}
    d, pumps, px, idx, uni = ET.load_data()
    trained = sorted(set(d["base_asset"]))
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        allpx = our_prices(conn, sorted(set(uni) | set(trained)))
        raw = load_raw(conn, trained, INTERVAL_15M, exchanges=["Binance"], light=True)
    octp = oct_prices(sorted(c for c in uni if c not in trained))
    for c, s in octp.items():
        allpx[c] = pd.concat([allpx[c], s]).groupby(level=0).last() if c in allpx else s
    sec = sector_table(allpx, mp)

    def feats(c: str, h: pd.DataFrame) -> pd.DataFrame:
        f = cap_features(c, h, mp)
        s = sec.get(c)
        f = f.join(s.reindex(h.index) if s is not None else pd.DataFrame(np.nan, index=h.index, columns=GROUPS["sector"]))
        return f[ALL]

    parts = []
    for c, r in split_raw(raw).items():
        a = aggregate(r, c, "15min", light=True)
        g = d[d["base_asset"] == c]
        parts.append(feats(c, a[["price", "volume", "oi"]]).reindex(g["ts"]).set_axis(g.index))
    d = d.join(pd.concat(parts))
    cov = d[ALL].notna().mean()
    print(f"навчання: {len(d)} барів, {d['base_asset'].nunique()} тікерів; заповнено: " + ", ".join(f"{k} {v:.0%}" for k, v in cov.items()), flush=True)
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
            gi = pd.Series(models[key].booster_.feature_importance("gain"), index=cols)
            imp[key] = gi / gi.sum()
            print(f"  {v} s{sd} (поріг {thr[key]:.3f}): хибних {lto[key]['false_alarms_per_30d']:.1f}, спіймано {lto[key]['caught']}", flush=True)
    m1 = joblib.load(train.model_path("lightgbm"))
    m15 = joblib.load(train.model_path(train.M15_NAME))
    exit_bars, share = m15.get("exit_bars", 4), m15.get("confirm", {}).get("share", 0.75)
    start, end = pd.Timestamp(I.START, tz="UTC"), pd.Timestamp(I.END, tz="UTC") + pd.Timedelta(days=1)
    coins = sorted(c for c in uni if c not in trained and os.path.exists(os.path.join(I.DIR, f"{c}.pkl")))
    print(f"бектест: {len(coins)} монет, зіставлено з CoinGecko {sum(bool((mp.get(c) or {}).get('id')) for c in coins)}…", flush=True)
    rows, bt_cov = [], []
    for n, c in enumerate(coins, 1):
        x = pickle.load(open(os.path.join(I.DIR, f"{c}.pkl"), "rb"))
        f15, h1, h = x["f15"], x["h1"], x["h"]
        f15 = f15.join(I.new_features(h, idx).reindex(f15["ts"]).set_axis(f15.index))
        hx = h.assign(oi=archive_oi(c, h.index))
        f15 = f15.join(feats(c, hx).reindex(f15["ts"]).set_axis(f15.index))
        bt_cov.append(f15[ALL].notna().mean())
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
    btc = pd.concat(bt_cov, axis=1).mean(axis=1)
    pf = I.pf
    V = list(variants)
    cell = {}
    for v in V:
        for sd in SEEDS:
            g = tr[(tr["variant"] == v) & (tr["seed"] == sd)]
            b = g[g["band"] == BIG]
            cell[(v, sd)] = {"pf": pf(g["ret"]), "pf20": pf(b["ret"]), "A": pf(b[b["half"] == "A"]["ret"]), "B": pf(b[b["half"] == "B"]["ret"]), "n": len(g)}
    mean = lambda v, k: float(np.mean([cell[(v, sd)][k] for sd in SEEDS]))       # noqa: E731
    nmap = sum(bool((mp.get(c) or {}).get("id")) for c in coins)
    L = ["# CoinGecko: капіталізація і сектор як ознаки", "",
         f"Групи: cap ({', '.join(GROUPS['cap'])}), sector ({', '.join(GROUPS['sector'])}). Random seeds {SEEDS}, значення — середнє. "
         f"Зіставлено з CoinGecko: бектест {nmap} з {len(coins)}, навчання {sum(bool((mp.get(c) or {}).get('id')) for c in trained)} з {len(trained)}. "
         "Заповнення ознак (навчання / бектест): " + ", ".join(f"{k} {cov[k]:.0%} / {btc[k]:.0%}" for k in ALL) + ".", "",
         "Застереження: категорії й max / total supply — поточні (на дату збору), тобто легкий погляд уперед; капіталізація — поденна, "
         "з лагом доба. Data: CoinGecko.", "",
         "**Критерій (до запуску):** проти base у парах за seed — ≥ 20 млн середня різниця ≥ +0.02 і краще на ≥ 2 з 3; загальний ≥ base; "
         "обидві половини ≥ base; feature importance групи ≥ 3%.", "",
         "| варіант | спіймано | хибних / 30 дн | gain нових | угод | PF | ≥ 20 млн | різниця ≥ 20 млн по seeds | A ≥ 20 млн | B ≥ 20 млн |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for v in V:
        new = [q for q in variants[v] if q in ALL]
        sh = np.mean([imp[(v, sd)][new].sum() for sd in SEEDS]) if new else 0.0
        diffs = [cell[(v, sd)]["pf20"] - cell[("base", sd)]["pf20"] for sd in SEEDS]
        caught = np.mean([int(str(lto[(v, sd)]["caught"]).split("/")[0]) for sd in SEEDS])
        total = str(lto[(v, SEEDS[0])]["caught"]).split("/")[-1]
        L.append(f"| {v} | {caught:.0f}/{total} | "
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
    ap = argparse.ArgumentParser(description="Капіталізація і сектор з CoinGecko як ознаки 15m-моделі (6 міс. архіву Binance)")
    ap.add_argument("--stage", choices=["cg", "test"], required=True, help="cg — зіставлення й завантаження CoinGecko (кеш); test — моделі й бектест")
    args = ap.parse_args()
    load_dotenv(".env")
    stage_cg() if args.stage == "cg" else stage_test()


if __name__ == "__main__":
    main()
