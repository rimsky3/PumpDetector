"""
Експеримент: дані напряму з бірж (Binance, Bybit) проти Coinalyze.

Питання:
    1. Чи збігаються сирі ряди (ціна, обсяг, OI, funding, long/short) і як вирівняти їх у часі.
    2. Чи збігаються ознаки, пораховані тим самим features.py з біржових даних.
    3. Чи можна подавати поточній моделі (навченій на Coinalyze) ознаки з бірж — без перенавчання.
    4. Чи біржові дані так само інформативні, якщо навчати на них (у тому самому короткому вікні).

Варіанти ознак (для тих самих тікерів і годин):
    coinalyze      усі біржі з Coinalyze — як у робочій моделі
    coinalyze_bb   Coinalyze, лише ринки Binance і Bybit — відділяє «інше джерело» від «менше бірж»
    direct_hybrid  Binance + Bybit напряму; ліквідації — з Coinalyze (у бірж немає їх історії)
    direct         лише біржі: без ліквідацій (так було б без Coinalyze і без власного websocket-збору)

Історія OI і long/short у Binance — лише 30 днів, тому порівняння йде в останніх ≈ 22 днях
(30 днів мінус 7 днів на вікна ознак).

    python experiments/exchange_direct.py             # завантажити (або взяти з кешу) і порахувати
    python experiments/exchange_direct.py --refresh   # завантажити з бірж заново
    → reports/experiment_exchange_direct.md
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
import train  # noqa: E402
from exchanges import MS_H, TF, download, get, normalize, stats, to_ts  # noqa: E402
from features import FEATURES, build_from_raw, load_pumps, load_raw, query_df  # noqa: E402
from fetch import TICKERS_FILE, read_tickers_file  # noqa: E402

REPORT = os.path.join(ROOT, "reports", "experiment_exchange_direct.md")
CACHE = os.path.join(ROOT, "experiments", "cache")
DAYS = 60
EXCHANGES = ("Binance", "Bybit")
VARIANTS = ["coinalyze", "coinalyze_bb", "direct_hybrid", "direct"]
H = pd.Timedelta(hours=1)
def load_direct(markets: pd.DataFrame, refresh: bool, tf: str = "1h") -> dict[str, dict[str, pd.DataFrame]]:
    """{market: {series: df}} з кешем у experiments/cache (parquet)."""
    os.makedirs(CACHE, exist_ok=True)
    end = int(datetime.now(timezone.utc).timestamp() * 1000)
    start = end - DAYS * 24 * MS_H
    out = {}
    for m in markets.itertuples():
        suffix = "" if tf == "1h" else f"_{tf}"
        paths = {k: os.path.join(CACHE, f"{m.exchange_name}_{m.symbol_on_exchange}_{k}{suffix}.parquet")
                 for k in ("klines", "oi", "ls", "funding")}
        if not refresh and all(os.path.exists(p) for p in paths.values()):
            out[m.symbol] = {k: pd.read_parquet(p) for k, p in paths.items()}
            continue
        print(f"  {m.exchange_name:<8} {m.symbol_on_exchange} {tf}")
        data = download(m.exchange_name, m.symbol_on_exchange, start, end, tf)
        for k, df in data.items():
            df.to_parquet(paths[k], index=False)
        out[m.symbol] = data
    for data in out.values():             # одна точність часу з даними з БД, інакше merge не зіставляє ключі
        for df in data.values():
            df["ts"] = df["ts"].astype("datetime64[us, UTC]")
    return out


# ---------------------------------------------------------------- біржові дані → формат load_raw()

def direct_raw(markets: pd.DataFrame, direct: dict, coinalyze_raw: dict, hybrid: bool,
               tf: str = "1h") -> dict[str, pd.DataFrame]:
    """Біржові ряди → формат load_raw(). Одиниці й вирівнювання часу — exchanges.normalize();
    buy volume у Bybit немає → частка покупок береться з Binance того самого бару (наближення)."""
    ohlcv, oi, fund, ls = [], [], [], []
    for m in markets.itertuples():
        n = normalize(direct[m.symbol], tf)
        meta = dict(base_asset=m.base_asset, market=m.symbol, exchange=m.exchange_name, quote_asset=m.quote_asset)
        ohlcv.append(n["ohlcv"].assign(**meta, volume_denominated="BASE_ASSET"))
        oi.append(n["oi"].assign(**meta))
        fund.append(n["funding"][["ts", "c"]].assign(**meta))
        ls.append(n["ls"][["ts", "long_pct"]].assign(**meta))

    ohlcv = pd.concat(ohlcv, ignore_index=True)
    # Bybit: bv = v × частка покупок Binance тієї самої години (наближення; Bybit її не віддає)
    bn = ohlcv[ohlcv["exchange"] == "Binance"].assign(share=lambda x: x["bv"] / x["v"])[["base_asset", "ts", "share"]]
    ohlcv = ohlcv.merge(bn, on=["base_asset", "ts"], how="left")
    ohlcv["bv"] = ohlcv["bv"].fillna(ohlcv["v"] * ohlcv["share"])
    ohlcv = ohlcv.drop(columns="share")

    keep = set(markets["symbol"])
    empty = lambda name: coinalyze_raw[name].iloc[0:0]  # noqa: E731
    return {
        "ohlcv": ohlcv,
        "oi": pd.concat(oi, ignore_index=True),
        "funding": pd.concat(fund, ignore_index=True),
        "ls": pd.concat(ls, ignore_index=True),
        "liq": coinalyze_raw["liq"].query("market in @keep") if hybrid else empty("liq"),
    }


# ---------------------------------------------------------------- порівняння

def raw_agreement(cz: dict, dr: dict, markets: pd.DataFrame) -> pd.DataFrame:
    """Сирі ряди контракт до контракту: кореляція погодинних змін при зсуві −1/0/+1 год
    (найвища має бути при 0 — тоді вирівнювання правильне) і медіанна відносна різниця рівнів."""
    series = {
        "close": ("ohlcv", "c"), "volume": ("ohlcv", "v"), "buy volume": ("ohlcv", "bv"),
        "OI (USD)": ("oi", "c"), "funding": ("funding", "c"), "long %": ("ls", "long_pct"),
    }
    rows = []
    for ex in EXCHANGES:
        mk = set(markets.loc[markets["exchange_name"] == ex, "symbol"])
        for name, (table, col) in series.items():
            if name == "buy volume" and ex == "Bybit":
                continue      # у Bybit це наближення з Binance, порівнювати нема з чим
            a = cz[table].query("market in @mk")[["market", "ts", col]]
            b = dr[table].query("market in @mk")[["market", "ts", col]]
            j = a.merge(b, on=["market", "ts"], suffixes=("_cz", "_ex")).dropna().sort_values(["market", "ts"])
            if j.empty:
                continue
            rel = ((j[f"{col}_ex"] - j[f"{col}_cz"]).abs() / j[f"{col}_cz"].abs().replace(0, np.nan)).median()
            row = {"exchange": ex, "series": name, "hours": len(j), "median_rel_diff": rel}
            for shift in (-1, 0, 1):
                ca = j.groupby("market")[f"{col}_cz"].diff()
                cb = j.groupby("market")[f"{col}_ex"].shift(shift).groupby(j["market"]).diff()
                row[f"corr_chg_shift_{shift:+d}h"] = ca.corr(cb)
            rows.append(row)
    return pd.DataFrame(rows).set_index(["exchange", "series"])


def feature_agreement(V: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Рангова кореляція кожної ознаки: варіант проти робочого coinalyze, на тих самих годинах."""
    base = V["coinalyze"]
    out = {}
    for v in VARIANTS[1:]:
        out[f"spearman vs coinalyze · {v}"] = {
            f: base[f].corr(V[v][f], method="spearman") for f in FEATURES}
    out["NaN share · direct"] = V["direct"][FEATURES].isna().mean()
    return pd.DataFrame(out)


def transfer(train_df: pd.DataFrame, V: dict[str, pd.DataFrame], pumps: pd.DataFrame) -> pd.DataFrame:
    """Модель вчиться на Coinalyze (усі 60 днів, без тестового тікера), поріг — вкладеним leave-ticker-out,
    як у train.py; тестовий тікер оцінюється на кожному варіанті ознак у вікні порівняння."""
    ev = V["coinalyze"]
    tickers = sorted(ev["base_asset"].unique())
    scores = {(m, v): np.full(len(ev), np.nan) for m in train.ALL_MODELS for v in VARIANTS}
    for g in tickers:
        print(f"  {g}")
        tr = train_df[train_df["base_asset"] != g]
        Xtr, ytr, gtr = tr[FEATURES], tr["label"].astype(int), tr["base_asset"]
        test = (ev["base_asset"] == g).to_numpy()
        rule = train.RuleModel().fit(Xtr, ytr)
        for v in VARIANTS:
            scores[("rule", v)][test] = rule.predict(V[v].loc[test, FEATURES])
        for m, make in train.PROBA_MODELS.items():
            thr = train.best_f1_threshold(ytr.to_numpy(), train.oof_proba(make, Xtr, ytr, gtr))
            model = make().fit(Xtr, ytr)
            for v in VARIANTS:
                # ймовірність ділимо на поріг: тоді сигнал = score ≥ 1 для всіх тікерів однаково
                scores[(m, v)][test] = model.predict_proba(V[v].loc[test, FEATURES])[:, 1] / thr
    return summarize(ev, scores, pumps)


def summarize(ev: pd.DataFrame, scores: dict, pumps: pd.DataFrame) -> pd.DataFrame:
    y = ev["label"].astype(int)
    rows = {}
    for (m, v), s in scores.items():
        sig = pd.Series(s >= 1 if m != "rule" else s.astype(bool), index=ev.index)
        res = pd.DataFrame({"score": s, "signal": sig})
        rows[f"{m} · {v}"] = {**train.hour_metrics(y, res, m != "rule"), **train.event_metrics(ev, sig, pumps)}
    cols = ["precision", "recall", "f1", "pr_auc", "pumps_caught", "false_starts_triggered", "false_alarms_per_30d"]
    return pd.DataFrame(rows).T[cols]


def retrain_in_window(V: dict[str, pd.DataFrame], pumps: pd.DataFrame) -> pd.DataFrame:
    """Інформативність джерела: leave-ticker-out, навчання й перевірка лише у вікні порівняння."""
    y, groups = V["coinalyze"]["label"].astype(int), V["coinalyze"]["base_asset"]
    scores = {}
    for v in VARIANTS:
        for m in train.PROBA_MODELS:
            print(f"  {v} / {m}")
            res = train.leave_ticker_out(m, V[v][FEATURES], y, groups)
            scores[(m, v)] = (res["score"] / res["threshold"]).to_numpy()
    return summarize(V["coinalyze"], scores, pumps)


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description="Експеримент: дані Binance/Bybit напряму проти Coinalyze "
                                             "→ reports/experiment_exchange_direct.md")
    ap.add_argument("--refresh", action="store_true", help="завантажити дані з бірж заново, ігноруючи кеш")
    args = ap.parse_args()

    load_dotenv(os.path.join(ROOT, ".env"))
    bases = read_tickers_file(os.path.join(ROOT, TICKERS_FILE))
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        cz_raw = load_raw(conn, bases)
        pumps = load_pumps(conn)
        train_df = query_df(conn, "SELECT * FROM features ORDER BY base_asset, ts")
        markets = query_df(conn, """SELECT symbol, base_asset, exchange_name, quote_asset, symbol_on_exchange
                                   FROM markets WHERE base_asset = ANY(%s) AND exchange_name = ANY(%s)
                                   AND quote_asset = 'USDT' ORDER BY base_asset, exchange_name""",
                           (bases, list(EXCHANGES)))

    print(f"Біржі напряму: {len(markets)} ринків")
    direct = load_direct(markets, args.refresh)
    if stats["requests"]:
        print(f"  запитів: {stats['requests']}, {stats['seconds']:.0f} с")

    keep = set(markets["symbol"])
    raws = {
        "coinalyze": cz_raw,
        "coinalyze_bb": {k: v.query("market in @keep") for k, v in cz_raw.items()},
        "direct_hybrid": direct_raw(markets, direct, cz_raw, hybrid=True),
        "direct": direct_raw(markets, direct, cz_raw, hybrid=False),
    }
    agree_raw = raw_agreement(cz_raw, raws["direct"], markets)

    # свіжі нерозмічені години не беремо, як і train.py (--fresh-hours 48)
    cutoff = pd.Timestamp(datetime.now(timezone.utc)) - pd.Timedelta(hours=train.FRESH_HOURS)
    confirmed = lambda d: d[(d["ts"] <= cutoff) | d["pattern_type"].notna()]  # noqa: E731
    train_df = confirmed(train_df)

    # вікно порівняння: години, де OI Binance має повну тижневу історію, і всі варіанти мають рядок
    bn_oi_start = min(direct[s]["oi"]["ts"].min() for s in markets.loc[markets["exchange_name"] == "Binance", "symbol"])
    eval_start = bn_oi_start + pd.Timedelta(days=8)
    feats = {v: confirmed(build_from_raw(r, pumps)) for v, r in raws.items()}
    key = ["base_asset", "ts"]
    common = feats["coinalyze"][key]
    for v in VARIANTS[1:]:
        common = common.merge(feats[v][key], on=key)
    common = common[common["ts"] >= eval_start]
    V = {v: common.merge(f, on=key, how="left").reset_index(drop=True) for v, f in feats.items()}
    ev_pumps = pumps[pumps["start_ts"] >= eval_start]
    n_pump = int((ev_pumps["pattern_type"] == "pump").sum())
    print(f"Вікно: з {eval_start:%Y-%m-%d %H:%M}, {len(common)} годин, {V['coinalyze']['base_asset'].nunique()} тікерів, "
          f"{n_pump} пампів")

    agree_feat = feature_agreement(V)
    print("\nПереносимість моделі (навчена на Coinalyze):")
    tr = transfer(train_df, V, ev_pumps)
    print(tr.to_string())
    print("\nНавчання у вікні на кожному джерелі:")
    rt = retrain_in_window(V, ev_pumps)
    print(rt.to_string())

    lines = [
        "# Experiment: exchange APIs directly vs Coinalyze", "",
        f"Tickers: {', '.join(sorted(V['coinalyze']['base_asset'].unique()))}. Direct data: Binance and Bybit USDT perpetuals "
        f"({len(markets)} markets), public endpoints, no API key"
        + (f"; {stats['requests']} requests in {stats['seconds']:.0f} s" if stats["requests"] else "") + ".",
        f"Comparison window: from {eval_start:%Y-%m-%d %H:%M} UTC ({len(common)} hourly bars, {n_pump} labeled pumps). "
        "Binance serves open interest and long/short history for the last 30 days only; features need 7 days of warm-up.", "",
        "- **coinalyze**: production features, all exchanges",
        "- **coinalyze_bb**: Coinalyze restricted to Binance and Bybit, separating the source effect from exchange coverage",
        "- **direct_hybrid**: Binance and Bybit APIs; liquidations still from Coinalyze "
        "(exchanges keep no history of them)",
        "- **direct**: exchange APIs only; no liquidations (their features are 0)", "",
        "## 1. Raw series, contract by contract", "",
        "Correlation of hourly changes with the exchange series shifted by −1/0/+1 h: the peak at 0 confirms the time "
        "alignment. Exchange OI snapshots at T+1h are the close of Coinalyze's bar T; long/short snapshots at T and funding paid at T match bar T.", "",
        train.md_table(agree_raw), "",
        "## 2. Features", "",
        "Spearman correlation of each feature with the production (coinalyze) value on the same hours.", "",
        train.md_table(agree_feat), "",
        "## 3. Can the current model take exchange data?", "",
        "Models trained on Coinalyze features (60 days, leave-ticker-out, nested threshold as in train.py) and scored on "
        "the held-out ticker's features from each source.", "",
        train.md_table(tr), "",
        "## 4. Is exchange data as informative?", "",
        "Leave-ticker-out trained and tested inside the comparison window only, separately for each source. "
        "Few pumps: differences of a few points are noise.", "",
        train.md_table(rt), "",
    ]
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nЗвіт: {REPORT}")


if __name__ == "__main__":
    main()
