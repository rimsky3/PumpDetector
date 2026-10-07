"""
backtest.py — широкий бектест моделі на монетах, яких вона не бачила в навчанні.

Список монет — ПРАВИЛОМ до прогону (без вибору «де був памп»): усі USDT-перпи Binance із середнім оборотом за останні
--days днів ≥ --min-turnover на добу, без тікерів навчання (tickers.txt і розмічені) і без виключених (excluded.txt).
Дані: 15m з Binance і Bybit (безкоштовно) + 15m-ліквідації Coinalyze (~3 виклики на монету); контракти монет —
один запит Coinalyze на весь список. Годинне підтвердження — з наших 15m-даних (features.build_hourly_own), як у скринері.
Угоди — тими самими правилами, що й дашборд (risk.position_state: вхід на Alert, вихід — згас exit_bars барів, стоп k·ATR),
комісія 0.1%. Результат — reports/backtests/<час>_<модель>.json; дивитись у дашборді, вкладка «Бектест».

    python backtest.py                         # поточна модель, оборот ≥ 20 млн $/добу, 30 днів
    python backtest.py --model candidate       # кандидат (models/candidate/)
    python backtest.py --entry candidate       # вхід на Кандидаті (лише 15m) замість Alert
    python backtest.py --no-fetch              # без завантаження: на даних, що вже є в базі
    python backtest.py --limit 10              # перші 10 монет (перевірка)

Обмеження: той самий період ринку, що й навчання; без прослизання і стакану; без твого фільтру стаканом.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import joblib
import numpy as np
import pandas as pd
import psycopg
from dotenv import load_dotenv

import exchanges as X
import fetch_exchange
import train
from features import (FEATURES, FEATURES_15M, INTERVAL_15M, INTRADAY_EXCHANGES, aggregate, build_15m, build_hourly_own,
                      load_raw)
from fetch import Coinalyze, collect_liquidations, excluded_tickers, find_markets, read_tickers_file, upsert_markets
import footprint
import market
import orderbook
import strategies
from risk import PARTIAL_EXIT, position_state, trade_stats
from tiers import tier_frame

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "reports", "backtests")
FEE = 0.001
WARMUP_DAYS = 8                # тижневі вікна ознак + запас
STATUS_FILE = os.path.join(OUT_DIR, "_status.json")    # прогрес для дашборду


def status(stage: str, step: int, steps: int, **kw) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump({"stage": stage, "step": step, "steps": steps, "at": time.time(), **kw}, f, ensure_ascii=False)
    print(f"[{step}/{steps}] {stage}", flush=True)


def universe(min_turnover: float, days: int, max_oi: float, log=print, max_turnover: float = float("inf")) -> list[dict]:
    """USDT-перпи Binance із середнім денним оборотом за days днів ≥ min_turnover (денні свічки, 1 запит на монету)
    і OI ≤ max_oi — той самий скоуп, що в сканері ринку (великі монети на кшталт BTC/ETH — не наш скоуп)."""
    info = X.get(X.BINANCE + "/fapi/v1/exchangeInfo", {})
    tick = X.get(X.BINANCE + "/fapi/v1/ticker/24hr", {})
    day = {t["symbol"]: float(t.get("quoteVolume") or 0) for t in tick}
    last = {t["symbol"]: float(t.get("lastPrice") or 0) for t in tick}
    cand = [s for s in info["symbols"] if s.get("contractType") == "PERPETUAL" and s.get("quoteAsset") == "USDT"
            and s.get("status") == "TRADING" and day.get(s["symbol"], 0) >= 0.3 * min_turnover]

    def avg_turnover(s):
        try:
            k = X.get(X.BINANCE + "/fapi/v1/klines", dict(symbol=s["symbol"], interval="1d", limit=days + 1))[:-1]
            t = float(np.mean([float(r[7]) for r in k])) if k else 0.0
            oi = float(X.get(X.BINANCE + "/fapi/v1/openInterest", dict(symbol=s["symbol"]))["openInterest"]) * last.get(s["symbol"], 0)                 if t >= min_turnover else 0.0
            return s["baseAsset"], s["symbol"], t, oi
        except Exception:
            return s["baseAsset"], s["symbol"], 0.0, 0.0
    with ThreadPoolExecutor(6) as ex:
        rows = list(ex.map(avg_turnover, cand))
    liquid = [(b, sym, t, oi) for b, sym, t, oi in rows if t >= min_turnover]
    out = [{"base": b, "symbol": sym, "turnover": t, "oi": oi} for b, sym, t, oi in liquid if oi <= max_oi and t < max_turnover]
    log(f"Binance: {len(info['symbols'])} символів; оборот ≥ {min_turnover / 1e6:.0f} млн $/добу за {days} дн.: {len(liquid)}; "
        f"з них OI ≤ {max_oi / 1e6:.0f} млн $: {len(out)}")
    return sorted(out, key=lambda x: -x["turnover"])


def exclusions(conn) -> set[str]:
    """Монети, на яких модель вчилась (tickers.txt + розмічені інтервали) і виключені."""
    trained = {t.upper() for t in read_tickers_file(os.path.join(ROOT, "tickers.txt"))}
    labeled = {r[0] for r in conn.execute("SELECT DISTINCT base_asset FROM pumps")}
    return trained | labeled | excluded_tickers(ROOT)


def ensure_data(conn, client: Coinalyze, bases: list[str], days: int, log=print) -> list[str]:
    """Контракти (один запит Coinalyze), 15m з Binance і Bybit і 15m-ліквідації за days + прогрів. Повертає монети з даними."""
    have = {r[0] for r in conn.execute("SELECT DISTINCT base_asset FROM markets WHERE base_asset = ANY(%s)", (bases,))}
    new = [b for b in bases if b not in have]
    if new:
        upsert_markets(conn, find_markets(client, new))
    span = days + WARMUP_DAYS
    # дотягуємо лише бракуюче: монети, де 15m уже є з потрібної дати й до останньої години, пропускаємо
    since = pd.Timestamp(datetime.now(timezone.utc)) - pd.Timedelta(days=span)
    full = {r[0] for r in conn.execute("""
        SELECT m.base_asset FROM ohlcv o JOIN markets m USING (symbol)
        WHERE o.interval = '15min' AND m.base_asset = ANY(%s) AND m.exchange_name = ANY(%s)
        GROUP BY 1 HAVING min(o.ts) <= %s AND max(o.ts) >= now() - interval '2 hours'""",
        (bases, list(INTRADAY_EXCHANGES), since + pd.Timedelta(days=1)))}
    todo = [b for b in bases if b not in full]
    for i in range(0, len(todo), 10):
        part = todo[i:i + 10]
        fetch_exchange.collect(conn, part, ["15m"], span, log=lambda m: None, exchanges=list(INTRADAY_EXCHANGES))
        collect_liquidations(client, conn, part, ["15m"], span, log=lambda m: None)
        log(f"  дані: {min(i + 10, len(todo))}/{len(todo)}")
    got = {r[0] for r in conn.execute("""SELECT DISTINCT m.base_asset FROM ohlcv o JOIN markets m USING (symbol)
                                         WHERE o.interval = '15min' AND m.base_asset = ANY(%s)""", (bases,))}
    return [b for b in bases if b in got]


def tiers_and_bars(conn, base: str, m15: dict, m1: dict, days: int) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Рівні сигналу (tier_frame, індекс — ts) за останні days днів і 15m-бари тікера. Спільне для прогону
    і експериментів з правилами виходу (experiments/exit_structure.py)."""
    since = pd.Timestamp(datetime.now(timezone.utc)) - pd.Timedelta(days=days + WARMUP_DAYS)
    d15 = build_15m(conn, [base], since=since)
    if d15.empty:
        return None
    d1 = build_hourly_own(conn, [base], since=since)
    s15 = d15[["base_asset", "ts"]].assign(probability=m15["model"].predict_proba(d15[m15["features"]])[:, 1], threshold=m15["threshold"])
    s1 = d1[["base_asset", "ts"]].assign(probability=m1["model"].predict_proba(d1[m1["features"]])[:, 1], threshold=m1["threshold"]) \
        if not d1.empty else pd.DataFrame(columns=["base_asset", "ts", "probability", "threshold"])
    t = tier_frame(s15, s1, m15.get("confirm", {}).get("share", 0.75)).set_index("ts").sort_index()
    start = pd.Timestamp(datetime.now(timezone.utc)) - pd.Timedelta(days=days)
    t = t[t.index >= start]
    if t.empty:
        return None
    h = aggregate(load_raw(conn, [base], INTERVAL_15M, since=since), base, "15min")
    return t, h[h.index >= start - pd.Timedelta(days=1)]


def run_ticker(conn, base: str, m15: dict, m1: dict, days: int, entry: str, strat: dict | None = None) -> dict | None:
    """strat — режим (strategies.py): сигнал Alert / Кандидат, вхід ринок / ліміт, фільтр OB (стакан з архіву Binance).
    Без режиму — як раніше: entry 'alert' / 'candidate', вхід по ринку."""
    th = tiers_and_bars(conn, base, m15, m1, days)
    if th is None:
        return None
    t, h = th
    if strat is not None:
        entry = "candidate" if strat.get("signal") == "candidate" else "alert"
    score = t["alert_score"] if entry == "alert" else t["s15"]
    allow = None
    if strat is not None and strat.get("ob"):
        row = conn.execute("""SELECT symbol_on_exchange FROM markets WHERE base_asset = %s AND exchange_name = 'Binance'
                              AND quote_asset = 'USDT' LIMIT 1""", (base,)).fetchone()
        if row:
            allow = orderbook.archive_allow_fn(row[0], list(score.index[score >= 1]))
    allow = market.both(allow, market.strategy_allow(conn, strat))     # режим «ринок не гарячий»
    # режим «б/з на p»: мітки з fp_marks (є лише там, де писався футпрінт; для історичної перевірки — experiments/p_exit.py)
    p_marks = footprint.p_times(conn, [base], h.index.min()).get(base, set()) if strat is not None and strat.get("p_be") else None
    pos = position_state(h, score, 1.0, m15.get("risk", {}).get("k"), m15.get("exit_bars", 4), PARTIAL_EXIT,
                         strategies.entry_rule(strat) if strat is not None else None, allow, p_marks)
    trades = []
    for tr in list(pos.get("trades", [])) + ([pos] if pos.get("open") else []):
        trades.append({"ticker": base, "entry_ts": int(pd.Timestamp(tr["entry_ts"]).timestamp()),
                       "exit_ts": int(pd.Timestamp(tr["exit_ts"]).timestamp()) if tr.get("exit_ts") is not None else None,
                       "entry": tr["entry"], "exit": tr.get("exit"), "reason": "open" if tr.get("open") else tr.get("reason"),
                       "ret": (tr.get("result_pct") or 0) - (FEE if not tr.get("open") else 0)})
    alert = (t["tier"] == "alert").to_numpy()
    eps = len(train.signal_episodes(t.reset_index()[["base_asset", "ts"]], alert, train.BAR_15M))
    return {"ticker": base, "trades": trades, "alert_episodes": eps, "bars": len(t),
            "days": (t.index.max() - t.index.min()) / pd.Timedelta(days=1)}


def summarize(results: list[dict], days: int) -> dict:
    trades = [x for r in results for x in r["trades"]]
    closed = [x for x in trades if x["reason"] != "open"]
    r = pd.Series([x["ret"] for x in closed], dtype=float).sort_values()
    st = trade_stats([{"result_pct": x["ret"], "reason": x["reason"]} for x in closed])
    coins = len(results)
    eps = sum(x["alert_episodes"] for x in results)
    per = []
    for res in results:
        c = [x["ret"] for x in res["trades"] if x["reason"] != "open"]
        per.append({"ticker": res["ticker"], "trades": len(c), "wins": sum(1 for v in c if v > 0),
                    "sum": float(np.sum(c)) if c else 0.0, "median": float(np.median(c)) if c else None,
                    "alert_episodes": res["alert_episodes"], "open": any(x["reason"] == "open" for x in res["trades"])})
    eq = {}
    for x in closed:
        dkey = pd.Timestamp(x["exit_ts"], unit="s", tz="UTC").strftime("%Y-%m-%d")
        eq[dkey] = eq.get(dkey, 0.0) + x["ret"]
    days_sorted = sorted(eq)
    cum = np.cumsum([eq[k] for k in days_sorted]).tolist()
    return {
        "coins": coins, "trades": len(closed), "open": len(trades) - len(closed),
        "win_rate": st["wins"] / st["n"] if st["n"] else None, "avg_win": st["avg_win"], "avg_loss": st["avg_loss"],
        "mean": st["expectancy"], "median": float(r.median()) if len(r) else None,
        "sum": float(r.sum()), "sum_wo_top3": float(r.iloc[:-3].sum()) if len(r) > 3 else float(r.sum()),
        "stops": st["stops"], "stop_rate": st["stops"] / st["n"] if st["n"] else None,
        "alerts_per_day": eps / days if days else None, "alert_episodes_per_coin_30d": eps / coins * 30 / days if coins and days else None,
        "equity": [{"day": k, "sum": v} for k, v in zip(days_sorted, cum)],
        "per_ticker": sorted(per, key=lambda x: -x["sum"]),
        "trades_list": sorted(trades, key=lambda x: -x["entry_ts"]),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Широкий бектест моделі на монетах поза навчанням (вкладка «Бектест» у дашборді)")
    p.add_argument("--min-turnover", type=float, default=20e6, help="середній оборот за добу, $ (за замовчуванням 20 млн)")
    p.add_argument("--days", type=int, default=30, help="вікно бектесту, днів (за замовчуванням 30; глибше Binance не дає OI)")
    p.add_argument("--max-turnover", type=float, default=float("inf"),
                   help="середній оборот за добу менше, $ (смуга обороту, напр. --min-turnover 5e6 --max-turnover 20e6)")
    p.add_argument("--max-oi", type=float, default=None,
                   help="OI не більше, $ (за замовчуванням — як у сканері, scan_config.json, 500 млн): без BTC/ETH")
    p.add_argument("--model", choices=["current", "candidate"], default="current", help="яку модель тестувати")
    p.add_argument("--entry", choices=["alert", "candidate"], default="alert", help="рівень входу: Alert або Кандидат (без --strategy)")
    p.add_argument("--strategy", default=None,
                   help="режим зі strategies.json (market, limit, ob, ob_limit, cand…): сигнал, вхід і фільтр OB; перекриває --entry")
    p.add_argument("--no-fetch", action="store_true", help="не завантажувати дані, лише те, що вже є в базі")
    p.add_argument("--k", type=float, default=None, help="стоп у ATR замість збереженого в моделі (порівняти моделі з однаковим стопом)")
    p.add_argument("--limit", type=int, default=0, help="лише перші N монет (перевірка)")
    args = p.parse_args()
    strat = strategies.get(args.strategy) if args.strategy else None
    if strat is not None and not strategies.is15(strat):
        sys.exit(f"режим {strat['id']} — годинна модель; бектест рахує лише 15m-режими")
    load_dotenv(os.path.join(ROOT, ".env"))
    t0 = time.time()
    mdir = train.MODELS_DIR if args.model == "current" else train.CANDIDATE_DIR
    m15 = joblib.load(train.model_path(train.M15_NAME, mdir))
    m1 = joblib.load(train.model_path("lightgbm", mdir))
    if args.k is not None:
        m15 = {**m15, "risk": {**m15.get("risk", {}), "k": args.k}}
    steps = 4
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        status("список монет (Binance)", 1, steps)
        import scanner
        max_oi = args.max_oi or scanner.load_config()["max_oi"]
        uni = universe(args.min_turnover, args.days, max_oi, max_turnover=args.max_turnover)
        skip = exclusions(conn)
        bases = [u["base"] for u in uni if u["base"].upper() not in skip]
        if args.limit:
            bases = bases[:args.limit]
        print(f"поза навчанням і виключеними: {len(bases)}")
        status(f"дані: {len(bases)} монет", 2, steps, coins=len(bases))
        if not args.no_fetch:
            bases = ensure_data(conn, Coinalyze(os.environ["COINALYZE_API_KEY"]), bases, args.days)
        status(f"модель: {len(bases)} монет", 3, steps, coins=len(bases))
        results = []
        for i, b in enumerate(bases, 1):
            try:
                r = run_ticker(conn, b, m15, m1, args.days, args.entry, strat)
            except Exception as e:
                print(f"  {b}: помилка {e}")
                r = None
            if r:
                results.append(r)
            if i % 10 == 0:
                status(f"модель: {i}/{len(bases)}", 3, steps, coins=len(bases))
    status("підсумок", 4, steps)
    s = summarize(results, args.days)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M") + f"_{args.model}_{strat['id'] if strat else args.entry}" \
        + (f"_k{args.k:g}" if args.k is not None else "")
    meta = {"id": run_id, "at": datetime.now(timezone.utc).isoformat(timespec="minutes"), "model": args.model, "max_oi": max_oi,
            "max_turnover": None if args.max_turnover == float("inf") else args.max_turnover,
            "model_version": m15["trained_at"], "entry": (("candidate" if strat.get("signal") == "candidate" else "alert") if strat else args.entry),
            "strategy": strat["id"] if strat else None, "strategy_title": strat["title"] if strat else None, "min_turnover": args.min_turnover, "days": args.days,
            "fee": FEE, "k": m15.get("risk", {}).get("k"), "universe": [u for u in uni if u["base"] in {r["ticker"] for r in results}],
            "excluded_as_trained": len(skip), "minutes": round((time.time() - t0) / 60, 1)}
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, run_id + ".json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, **s}, f, ensure_ascii=False)
    try:
        os.remove(STATUS_FILE)
    except OSError:
        pass
    print(f"\n{run_id}: {s['coins']} монет, {s['trades']} угод (+{s['open']} відкритих), у плюсі {s['win_rate'] or 0:.0%}, "
          f"медіана {s['median'] or 0:+.2%}, сума {s['sum']:+.0%}, без топ-3 {s['sum_wo_top3']:+.0%}, "
          f"Alert на день {s['alerts_per_day'] or 0:.1f} · {meta['minutes']} хв")


if __name__ == "__main__":
    main()
