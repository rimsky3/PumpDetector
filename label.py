"""
label.py — локальний інструмент розмітки імпульсів.

Піднімає вебсервер на 127.0.0.1 (доступний лише з цього комп'ютера) і відкриває
сторінку з графіками ціни, open interest і обсягу по кожному тікеру з бази.
Таймфрейми: 1h (Coinalyze, усі біржі) і 15m / 5m (Binance + Bybit, python fetch_exchange.py) — межі імпульсу
можна ставити на будь-якому, це моменти часу.

Права панель — усі USDT-перпи Binance. «Завантажити» ставить тікер у чергу: 1h з Coinalyze (fetch.collect),
потім 15m/5m з бірж (fetch_exchange.collect).
«Прогнати модель 15m» (/api/model15): 15m-модель і годинний LightGBM оцінюють УСЮ історію тікера на льоту
(у signals не пишемо — журнал forward-тесту лишається чистим), рівні сигналу — tiers.py, підсумок проти поточної
розмітки — train.event_metrics. Тікер, якого модель не бачила, — чесний тест; розмітка його помилок — нові приклади.

Розмічені інтервали будь-якого тікера йдуть у навчання (features.py). Решта годин тікера — лише коли він у
tickers.txt: кнопка «У навчання» = «переглянув увесь тікер, де імпульсу не розмічено — його не було».
Виділені інтервали одразу записуються в pumps.csv і синхронізуються з таблицею pumps.
Якщо є reports/oof_predictions.csv (python train.py), показує сигнали моделей і їхні помилки.

    python label.py
    python label.py --days 30 --port 8765
"""

import argparse
import csv
import json
import os
import queue
import re
import sys
import threading
import time
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pandas as pd
import psycopg
import requests
from dotenv import load_dotenv

import fetch_exchange
from exchanges import INTERVAL
from features import FEATURES, FEATURES_15M, INTRADAY_EXCHANGES, INTERVAL_15M, aggregate, build, build_15m, build_hourly_own, load_pumps, load_raw, query_df
from risk import PARTIAL_EXIT, atr, position_state, trade_stats as summarize_trades
from tiers import tier_frame
import train
import volume_profile
from fetch import (EXCLUDED_FILE, TICKERS_FILE, Coinalyze, collect, collect_liquidations, excluded_tickers,
                   read_tickers_file, set_ticker_in_file)
from load_pumps import (SETUPS, check_overlaps, parse_row, parse_ts, read_pumps,
                        sync_db, write_pumps, TS_FORMAT)

ROOT = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.join(ROOT, "tools", "label.html")
VENDOR = os.path.join(ROOT, "app", "vendor")                          # бібліотека графіків, спільна з дашбордом
JS_DIR = os.path.join(ROOT, "app", "js")                              # спільні скрипти сторінок
PREDICTIONS = os.path.join(ROOT, "reports", "oof_predictions.csv")   # пише train.py
TICKERS_PATH = os.path.join(ROOT, TICKERS_FILE)
lock = threading.Lock()  # запис у pumps.csv — по одному


# ---------------------------------------------------------------- усі перпи Binance

_universe = {"at": 0.0, "rows": []}


def binance_universe() -> list[dict]:
    """USDT-перпи Binance, що торгуються, зі зміною ціни і обсягом за 24 год (кеш 10 хв; два публічні запити)."""
    if time.time() - _universe["at"] < 600 and _universe["rows"]:
        return _universe["rows"]
    info = requests.get("https://fapi.binance.com/fapi/v1/exchangeInfo", timeout=20).json()
    day = {t["symbol"]: t for t in requests.get("https://fapi.binance.com/fapi/v1/ticker/24hr", timeout=20).json()}
    rows = []
    for s in info["symbols"]:
        if s.get("contractType") != "PERPETUAL" or s.get("quoteAsset") != "USDT" or s.get("status") != "TRADING":
            continue
        t = day.get(s["symbol"], {})
        rows.append({"base": s["baseAsset"], "chg": float(t.get("priceChangePercent") or 0) / 100,
                     "vol": float(t.get("quoteVolume") or 0)})
    _universe.update(at=time.time(), rows=rows)
    return rows


# ---------------------------------------------------------------- 15m-модель на історії тікера

_models: dict = {"mtime": None, "m15": None, "m1": None}
_score_cache: dict = {}


def models_15m() -> tuple[dict, dict]:
    """15m-модель і годинний LightGBM (підтвердження Alert). Перечитуються, якщо train.py перезаписав файли."""
    paths = [train.model_path(train.M15_NAME), train.model_path("lightgbm")]
    if not all(os.path.exists(p) for p in paths):
        raise ValueError("немає models/lightgbm_15m.joblib або lightgbm.joblib — спершу python train.py")
    mtime = tuple(os.path.getmtime(p) for p in paths)
    if _models["mtime"] != mtime:
        _models.update(mtime=mtime, m15=train.load_model(train.M15_NAME), m1=train.load_model("lightgbm"))
        _score_cache.clear()
    return _models["m15"], _models["m1"]


def model15(conn, base: str) -> dict:
    """Оцінки 15m-моделі й годинного LightGBM по всій історії тікера, рівні сигналу й підсумок проти розмітки."""
    m15, m1 = models_15m()
    last = conn.execute("""SELECT max(o.ts) FROM ohlcv o JOIN markets m USING (symbol)
                           WHERE m.base_asset = %s AND o.interval = '15min'""", (base,)).fetchone()[0]
    key = (base, m15["trained_at"], last)
    if key not in _score_cache:
        d15 = build_15m(conn, [base])
        if d15.empty:
            raise ValueError(f"для {base} замало 15m-історії: потрібно ≥ 7 днів (кнопка «Завантажити» праворуч)")
        d1 = build(conn, [base], labels=False)
        if d1.empty:                  # немає 1h Coinalyze (напр., монета з бектесту) — година з власних 15m, як у скринері
            d1 = build_hourly_own(conn, [base])
        s15 = d15[["base_asset", "ts"]].assign(probability=m15["model"].predict_proba(d15[FEATURES_15M])[:, 1],
                                                threshold=m15["threshold"])
        s1 = d1[["base_asset", "ts"]].assign(probability=m1["model"].predict_proba(d1[FEATURES])[:, 1],
                                              threshold=m1["threshold"]) if not d1.empty else \
            pd.DataFrame(columns=["base_asset", "ts", "probability", "threshold"])
        _score_cache.clear()          # тримаємо лише останній тікер — оцінки дешево перерахувати
        _score_cache[key] = tier_frame(s15, s1, m15.get("confirm", {}).get("share", 0.75))
    t = _score_cache[key]
    share = m15.get("confirm", {}).get("share", 0.75)

    # підсумок проти ПОТОЧНОЇ розмітки (після кожної зміни в pumps.csv — свіжий)
    pumps = load_pumps(conn)
    pumps = pumps[pumps["base_asset"] == base]
    df = t[["base_asset", "ts"]]
    alert, cand = (t["tier"] == "alert").to_numpy(), (t["tier"] != "calm").to_numpy()
    epoch = lambda x: int(pd.Timestamp(x).timestamp())  # noqa: E731
    summary = {name: train.event_metrics(df, sig, pumps, train.BAR_15M) for name, sig in (("alert", alert), ("candidate", cand))}
    eps = []
    for kind, sig in (("alert", alert), ("candidate", cand & ~alert)):
        for e in train.signal_episodes(df, sig, train.BAR_15M).itertuples():
            own = pumps[(pumps["start_ts"] < e.end) & (pumps["end_ts"] > e.start)]
            span = t[(t["ts"] >= e.start) & (t["ts"] < e.end)]
            eps.append({"kind": kind, "start": e.start, "end": e.end, "p_max": float(span["p15"].max()),
                        "label": ("pump" if (own["pattern_type"] == "pump").any() else own["pattern_type"].iloc[0]) if len(own) else None})
    eps.sort(key=lambda e: e["start"])
    # угоди за правилами дашборду: вхід на Alert, стоп k·ATR(15m), вихід — Alert згас exit_bars барів
    k, exit_bars = m15.get("risk", {}).get("k"), m15.get("exit_bars", 4)
    h = aggregate(load_raw(conn, [base], INTERVAL_15M), base, "15min")
    pos = position_state(h, t.set_index("ts")["alert_score"], 1.0, k, exit_bars, PARTIAL_EXIT) if not h.empty else {"trades": []}
    trades = []
    for tr in list(pos.get("trades", [])) + ([pos] if pos.get("open") else []):
        a = pd.Timestamp(tr["entry_ts"])
        b = pd.Timestamp(tr["exit_ts"]) + train.BAR_15M if tr.get("exit_ts") is not None else h.index.max() + train.BAR_15M
        own = pumps[(pumps["start_ts"] < b) & (pumps["end_ts"] > a)]
        trades.append({"entry_ts": epoch(a), "exit_ts": epoch(tr["exit_ts"]) if tr.get("exit_ts") is not None else None,
                       "entry": tr["entry"], "exit": tr.get("exit"), "result": tr.get("result_pct"),
                       "reason": "open" if tr.get("open") else tr.get("reason"),
                       "label": ("pump" if (own["pattern_type"] == "pump").any() else own["pattern_type"].iloc[0]) if len(own) else None})
    avg = lambda xs: sum(xs) / len(xs) if xs else None  # noqa: E731
    trade_stats = {   # спільний підсумок з дашбордом (risk.trade_stats) + розбивка за розміткою
        **summarize_trades(pos.get("trades", [])), "k": k, "exit_bars": exit_bars,
        "in_pump": avg([x["result"] for x in trades if x["reason"] != "open" and x["label"] == "pump"]),
        "outside": avg([x["result"] for x in trades if x["reason"] != "open" and x["label"] != "pump"]),
    }
    trained = base in m15["tickers"] or any(b == base for b, *_ in m15.get("in_sample_extra", []))
    code = {"alert": 2, "candidate": 1, "calm": 0}
    return {
        "t": [epoch(x) for x in t["ts"]], "p15": [round(float(x), 4) for x in t["p15"]],
        "p1h": [None if pd.isna(x) else round(float(x), 4) for x in t["p1h"]],
        "tier": [code[x] for x in t["tier"]], "thr15": float(m15["threshold"]),
        "need1h": float(share * m1["threshold"]), "share": share, "model_version": m15["trained_at"],
        "in_training": trained,
        "summary": {k: {kk: (vv if isinstance(vv, str) else (None if pd.isna(vv) else float(vv))) for kk, vv in v.items()}
                    for k, v in summary.items()},
        "episodes": [{**e, "start": epoch(e["start"]), "end": epoch(e["end"])} for e in eps],
        "trades": trades, "trade_stats": trade_stats,
    }


# ---------------------------------------------------------------- стан моделі і перенавчання

RETRAIN: dict = {"state": "idle"}      # idle / running / done / error; step, stage, started, log
_retrain_lock = threading.Lock()


def _bundle(path: str) -> dict | None:
    import joblib
    return joblib.load(path) if os.path.exists(path) else None


def _summary(b15: dict | None, b1: dict | None) -> dict | None:
    """Ключові цифри валідації моделі (leave-ticker-out при її навчанні)."""
    if not b15:
        return None
    tv, v15 = b15.get("tier_validation", {}), b15.get("validation", {})
    return {"trained_at": b15.get("trained_at"), "tickers": len(b15.get("tickers", [])),
            "pumps": b15.get("pump_intervals"), "stop_k": (b15.get("risk") or {}).get("k"),
            "alert": {k: tv.get(k) for k in ("pumps_caught", "false_starts_triggered", "false_alarms_per_30d", "f1", "precision")},
            "candidate": {k: v15.get(k) for k in ("pumps_caught", "false_starts_triggered", "false_alarms_per_30d", "f1", "pr_auc")},
            "f1_1h": (b1 or {}).get("validation", {}).get("f1")}


def model_status(conn) -> dict:
    """Які тікери вже в поточній моделі, які змінились після навчання, які чекають перенавчання; кандидат, якщо є."""
    cur15, cur1 = _bundle(train.model_path(train.M15_NAME)), _bundle(train.model_path("lightgbm"))
    excluded = excluded_tickers(ROOT)
    reviewed = {t.upper() for t in read_tickers_file(TICKERS_PATH)} - excluded
    pumps = load_pumps(conn)
    fp_now = {b: v for b, v in train.label_fingerprint(pumps).items() if b not in excluded}
    tickers = {}
    if cur15:
        in_full = set(cur15.get("tickers", []))
        in_part = {b for b, *_ in cur15.get("in_sample_extra", [])} - in_full
        fp_then = cur15.get("labels_fp")          # немає у моделей, навчених до 2026-09-30
        for t in sorted(reviewed | set(fp_now) | in_full | in_part):
            wanted = t in reviewed or t in fp_now            # піде в наступне навчання
            if t not in in_full and t not in in_part:
                st = "new" if wanted else None
            elif not wanted:
                st = "removed"
            elif (t in reviewed) != (t in in_full) or (fp_then is not None and fp_then.get(t) != fp_now.get(t)):
                st = "changed"                               # розмітка чи «У навчання» змінились після навчання
            else:
                st = "in"
            if st:
                tickers[t] = st
    cand15 = _bundle(train.model_path(train.M15_NAME, train.CANDIDATE_DIR))
    cand1 = _bundle(train.model_path("lightgbm", train.CANDIDATE_DIR))
    return {"current": _summary(cur15, cur1), "candidate": _summary(cand15, cand1), "tickers": tickers,
            "fingerprints": cur15 is not None and cur15.get("labels_fp") is not None,
            "pending": sorted(t for t, s in tickers.items() if s in ("new", "changed", "removed")),
            "retrain": {k: v for k, v in RETRAIN.items() if k != "log"} | {"log": RETRAIN.get("log", [])[-3:]}}


def retrain_worker() -> None:
    """features.py → train.py --candidate у фоні. Нова модель — лише кандидат, у роботу — після «Прийняти»."""
    import subprocess
    steps = [("ознаки (features.py)", [sys.executable, "features.py"]),
             ("навчання кандидата (train.py)", [sys.executable, "train.py", "--candidate"])]
    RETRAIN.update(state="running", started=time.time(), step=0, steps=len(steps), log=[])
    try:
        for i, (stage, cmd) in enumerate(steps, 1):
            RETRAIN.update(step=i, stage=stage)
            env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
            p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                 encoding="utf-8", errors="replace", env=env)
            for line in p.stdout:
                if line.strip():
                    RETRAIN["log"] = (RETRAIN.get("log", []) + [line.rstrip()])[-40:]
            if p.wait() != 0:
                raise RuntimeError(f"{stage}: помилка (код {p.returncode}) — " + " | ".join(RETRAIN["log"][-3:]))
        RETRAIN.update(state="done", finished=time.time())
    except Exception as e:
        RETRAIN.update(state="error", error=str(e))


def retrain_accept() -> str:
    """Кандидат → у роботу; поточні моделі — у models/backup_<час>/ (відкат — скопіювати назад)."""
    import shutil
    cand = os.path.join(ROOT, train.CANDIDATE_DIR)
    files = [f for f in os.listdir(cand) if f.endswith((".joblib", ".parquet"))] if os.path.isdir(cand) else []
    if not files:
        raise ValueError("кандидата немає — спершу «Перенавчити»")
    backup = os.path.join(ROOT, "models", "backup_" + datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M"))
    os.makedirs(backup, exist_ok=True)
    for f in files:
        live = os.path.join(ROOT, "models", f)
        if os.path.exists(live):
            shutil.copy2(live, os.path.join(backup, f))
        shutil.move(os.path.join(cand, f), live)
    shutil.rmtree(cand, ignore_errors=True)
    RETRAIN.update(state="idle")
    return os.path.relpath(backup, ROOT)


# ---------------------------------------------------------------- порівняння поточної моделі і кандидата

COMPARE_PAGE = os.path.join(ROOT, "tools", "compare.html")
REVIEW_PAGE = os.path.join(ROOT, "tools", "review.html")
_prefetching: set = set()


def prefetch_review(db_url: str) -> None:
    """Фоном — 5m наступних сигналів сліпого перегляду (монетам з бектесту 5m з бірж вантажиться ~30 с)."""
    import review as R

    def work():
        try:
            with psycopg.connect(db_url) as conn:
                for it in R.upcoming(conn):
                    if it["id"] in _prefetching:
                        continue
                    _prefetching.add(it["id"])
                    try:
                        R.bars(conn, it["ticker"], it["entry_ts"], "5m")
                    except Exception:
                        pass
        except Exception:
            pass
    threading.Thread(target=work, daemon=True).start()
_compare_cache: dict = {}


def score_tiers(conn, base: str, m15: dict, m1: dict) -> pd.DataFrame:
    """Рівні сигналу (tiers.py) заданих 15m- і годинної моделей на всій історії тікера — на льоту."""
    d15 = build_15m(conn, [base])
    if d15.empty:
        return pd.DataFrame()
    d1 = build(conn, [base], labels=False)
    s15 = d15[["base_asset", "ts"]].assign(probability=m15["model"].predict_proba(d15[FEATURES_15M])[:, 1], threshold=m15["threshold"])
    s1 = d1[["base_asset", "ts"]].assign(probability=m1["model"].predict_proba(d1[FEATURES])[:, 1], threshold=m1["threshold"]) \
        if not d1.empty else pd.DataFrame(columns=["base_asset", "ts", "probability", "threshold"])
    return tier_frame(s15, s1, m15.get("confirm", {}).get("share", 0.75))


def _oof_tiers(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    o = pd.read_parquet(path)
    o["ts"] = pd.to_datetime(o["ts"], utc=True)
    return o.rename(columns={"p15": "p15"})


def _ticker_eval(t: pd.DataFrame, pumps: pd.DataFrame) -> tuple[dict, list]:
    """Метрики подій одного тікера (рівні Alert і Кандидат) і розбір кожного розміченого інтервалу."""
    df = t[["base_asset", "ts"]]
    alert, cand = (t["tier"] == "alert").to_numpy(), (t["tier"] != "calm").to_numpy()
    m = {"alert": train.event_metrics(df, alert, pumps, train.BAR_15M),
         "candidate": train.event_metrics(df, cand, pumps, train.BAR_15M),
         "days": len(t) / 96}
    per = []
    for p in pumps.itertuples():
        inside = in_pump_mask(t["ts"], p.start_ts, p.end_ts)
        a = t[inside & alert]
        c = t[inside & cand]
        first = lambda x: (x["ts"].min() + train.BAR_15M - p.start_ts) / pd.Timedelta(minutes=1) if len(x) else None  # noqa: E731
        per.append({"start": p.start_ts, "end": p.end_ts, "type": p.pattern_type,
                    "alert": bool(len(a)), "alert_delay": first(a), "cand": bool(len(c)), "cand_delay": first(c),
                    "p_max": float(t.loc[inside, "p15"].max()) if inside.any() else None})
    return m, per


def in_pump_mask(ts: pd.Series, a, b):
    from features import in_pump
    return in_pump(ts, a, b, train.BAR_15M).to_numpy()


def _importance(cur: dict | None, new: dict | None) -> dict | None:
    """На що спирається модель: частки gain LightGBM по групах і по ознаках, поточна проти кандидата.
    Розмітка змінює, які ознаки описують «імпульс» (30.09: ліквідації 18% → 3%), — видно після кожного перенавчання."""
    c, n = train.gain_shares(cur), train.gain_shares(new)
    if n is None:
        return None
    gc = train.group_shares(c) if c else pd.Series(dtype=float)
    gn = train.group_shares(n)
    groups = sorted(set(gn.index) | set(gc.index), key=lambda g: -max(gn.get(g, 0), gc.get(g, 0)))
    return {"groups": [{"group": g, "cur": float(gc[g]) if g in gc else None, "new": float(gn.get(g, 0))} for g in groups],
            "features": [{"name": f, "group": train.feature_group(f), "cur": (c or {}).get(f), "new": v}
                         for f, v in sorted(n.items(), key=lambda x: -x[1])]}


def compare(conn) -> dict:
    """Поточна модель проти кандидата: зведення, по тікерах, по кожному розміченому інтервалу.
    Кандидат — out-of-fold (leave-ticker-out) з його навчання. Поточна — її out-of-fold, якщо збережені;
    інакше оцінки на льоту: чесні лише на тікерах, яких вона не бачила в навчанні (fair=False — in-sample)."""
    cur_dir, cand_dir = train.MODELS_DIR, train.CANDIDATE_DIR
    p15c, p15n = train.model_path(train.M15_NAME), train.model_path(train.M15_NAME, cand_dir)
    if not os.path.exists(p15n):
        raise ValueError("кандидата немає — натисни «Перенавчити» в розмітці")
    oof_new = _oof_tiers(os.path.join(ROOT, cand_dir, "oof_15m.parquet"))
    if oof_new is None:
        raise ValueError("кандидат навчений старою версією train.py (без out-of-fold прогнозів) — натисни «Перенавчити» ще раз")
    labels_mtime = os.path.getmtime(os.path.join(ROOT, "pumps.csv"))
    key = (os.path.getmtime(p15c) if os.path.exists(p15c) else 0, os.path.getmtime(p15n), labels_mtime)
    if key in _compare_cache:
        return _compare_cache[key]

    import joblib
    cur15, cur1 = joblib.load(p15c), joblib.load(train.model_path("lightgbm"))
    new15 = joblib.load(p15n)
    oof_cur = _oof_tiers(os.path.join(ROOT, cur_dir, "oof_15m.parquet"))
    cur_seen = set(cur15.get("tickers", [])) | {b for b, *_ in cur15.get("in_sample_extra", [])}
    pumps = load_pumps(conn)

    tickers, pump_rows = [], []
    for base in sorted(oof_new["base_asset"].unique()):
        own = pumps[pumps["base_asset"] == base]
        tn = oof_new[oof_new["base_asset"] == base]
        mn, pn = _ticker_eval(tn, own)
        if oof_cur is not None and base in set(oof_cur["base_asset"]):
            tc, source = oof_cur[oof_cur["base_asset"] == base], "oof"
        else:
            tc, source = score_tiers(conn, base, cur15, cur1), ("unseen" if base not in cur_seen else "in_sample")
            tc = tc[tc["ts"].isin(tn["ts"])] if len(tc) else tc       # те саме вікно, що в кандидата
        mc, pc = _ticker_eval(tc, own) if len(tc) else (None, [None] * len(pn))
        fair = source in ("oof", "unseen")
        tickers.append({"ticker": base, "source": source, "fair": fair, "new": mn, "cur": mc,
                        "pumps": int((own["pattern_type"] == "pump").sum()), "false_starts": int((own["pattern_type"] == "false_start").sum())})
        for a, b in zip(pn, pc):
            pump_rows.append({"ticker": base, "fair": fair, "start": int(a["start"].timestamp()), "end": int(a["end"].timestamp()),
                              "type": a["type"], "new": {k: a[k] for k in ("alert", "alert_delay", "cand", "cand_delay", "p_max")},
                              "cur": ({k: b[k] for k in ("alert", "alert_delay", "cand", "cand_delay", "p_max")} if b else None)})
    new1 = joblib.load(train.model_path("lightgbm", cand_dir))
    res = {"current": _summary(cur15, cur1), "candidate": _summary(new15, new1),
           "importance": {"15m": _importance(cur15, new15), "1h": _importance(cur1, new1)},
           "current_delay": {"alert": cur15.get("tier_validation", {}).get("median_delay_h"), "candidate": cur15.get("validation", {}).get("median_delay_h")},
           "candidate_delay": {"alert": new15.get("tier_validation", {}).get("median_delay_h"), "candidate": new15.get("validation", {}).get("median_delay_h")},
           "current_oof": oof_cur is not None, "tickers": tickers, "pumps": pump_rows}
    _compare_cache.clear()
    _compare_cache[key] = res
    return res


# ---------------------------------------------------------------- черга завантажень

JOBS: dict[str, dict] = {}          # тікер → {state: queued/running/done/error, msg}
_jobs = queue.Queue()


def load_worker(db_url: str, api_key: str, days: int) -> None:
    """По одному тікеру: 1h з Coinalyze, потім 15m/5m з Binance і Bybit. Ліміт Coinalyze спільний зі скринером,
    тож черга, а не паралельно."""
    client = Coinalyze(api_key) if api_key else None
    while True:
        base = _jobs.get()
        job = JOBS[base]
        try:
            if client is None:
                raise ValueError("немає COINALYZE_API_KEY у .env")
            # прогрес для сторінки: крок з 3, і всередині кроку — скільки частин готово
            job.update(state="running", step=1, steps=3, stage="годинні дані (Coinalyze)", done=0, total=0, msg="1h з Coinalyze…")
            with psycopg.connect(db_url) as conn:
                if not collect(client, conn, [base], days, verbose=False):
                    raise ValueError("не знайдено в Coinalyze (буває інша назва: 1000… чи k…)")
                n_mk = conn.execute("""SELECT count(*) FROM markets WHERE base_asset = %s AND quote_asset = 'USDT'
                                       AND exchange_name = ANY(%s)""", (base, fetch_exchange.EXCHANGES)).fetchone()[0]
                job.update(step=2, stage="15m/5m з бірж", done=0, total=n_mk * 2, msg="15m/5m з бірж…")

                def part(m):          # рядок журналу fetch_exchange = один ринок × ТФ готовий (або з помилкою)
                    job.update(msg=m.strip())
                    if "свічок" in m or "помилка" in m:
                        job["done"] = job.get("done", 0) + 1
                n = fetch_exchange.collect(conn, [base], ["15m", "5m"], days, log=part)
                job.update(step=3, stage="ліквідації (Coinalyze)", done=0, total=2, msg="ліквідації 15m/5m…")

                def liq(m):
                    job.update(msg=m.strip(), done=job.get("done", 0) + 1)
                collect_liquidations(client, conn, [base], ["15m", "5m"], days, log=liq)
            job.update(state="done", step=3, done=2, msg="готово" if n else "готово (лише 1h: на Binance/Bybit/OKX немає)")
        except Exception as e:
            job.update(state="error", msg=str(e))


def predictions(base: str) -> dict:
    """Out-of-fold передбачення моделей для тікера (з останнього запуску train.py)."""
    if not os.path.exists(PREDICTIONS):
        return {"available": False}
    with open(PREDICTIONS, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        models = [c[:-len("_score")] for c in reader.fieldnames if c.endswith("_score")]
        rows = [r for r in reader if r["base_asset"] == base]
    num = lambda s: float(s) if s not in ("", None) else None
    out = {
        "available": True,
        "updated": datetime.fromtimestamp(os.path.getmtime(PREDICTIONS), timezone.utc).strftime(TS_FORMAT),
        "t": [int(datetime.fromisoformat(r["ts"]).timestamp()) for r in rows],
        "models": {},
    }
    for m in models:
        thr = next((num(r[f"{m}_threshold"]) for r in rows if r.get(f"{m}_threshold")), None)
        out["models"][m] = {
            "score": [num(r[f"{m}_score"]) for r in rows],
            "signal": [int(r[f"{m}_signal"]) for r in rows],
            "threshold": thr,
        }
    return out


def ticker_data(conn, base: str, days: int, tf: str = "1h") -> dict:
    """Ряди тікера на таймфреймі tf: ціна Binance (або перша біржа з OHLCV), OI і обсяг — сума по біржах у USD.
    1h — з Coinalyze (усі біржі); 15m/5m — з Binance і Bybit (fetch_exchange.py)."""
    interval = INTERVAL[tf]
    since = f"now() - interval '{int(days)} days'"
    # 15m/5m — лише біржі моделі (features.INTRADAY_EXCHANGES); 1h — усі біржі Coinalyze
    if tf != "1h":
        since += " AND m.exchange_name IN (" + ", ".join(f"'{e}'" for e in INTRADAY_EXCHANGES) + ")"
    q = lambda sql: conn.execute(sql, (base, interval)).fetchall()

    price_ex = q("""SELECT m.exchange_name FROM markets m WHERE m.base_asset = %s AND m.has_ohlcv
                    AND EXISTS (SELECT 1 FROM ohlcv o WHERE o.symbol = m.symbol AND o.interval = %s)
                    ORDER BY m.exchange_name <> 'Binance', m.exchange_name LIMIT 1""")
    price_ex = price_ex[0][0] if price_ex else None
    # свічки: open, high, low, close з еталонної біржі
    candles = {r[0]: r[1:] for r in conn.execute(f"""
        SELECT extract(epoch FROM o.ts)::bigint, o.o, o.h, o.l, o.c FROM ohlcv o JOIN markets m USING (symbol)
        WHERE m.base_asset = %s AND m.exchange_name = %s AND o.interval = %s AND o.ts >= {since}""",
        (base, price_ex, interval)).fetchall()}
    price = {t: c[3] for t, c in candles.items()}

    oi_rows = q(f"""
        SELECT extract(epoch FROM o.ts)::bigint, m.exchange_name, o.o, o.h, o.l, o.c, p.c
        FROM open_interest o JOIN markets m USING (symbol)
        LEFT JOIN ohlcv p ON p.symbol = o.symbol AND p.interval = o.interval AND p.ts = o.ts
        WHERE m.base_asset = %s AND o.interval = %s AND o.ts >= {since}""")
    # обсяг OHLCV буває в монетах — переводимо в USD через ціну закриття
    vol = dict(q(f"""
        SELECT extract(epoch FROM o.ts)::bigint,
               sum(CASE WHEN m.volume_denominated = 'BASE_ASSET' THEN o.v * o.c ELSE o.v END)
        FROM ohlcv o JOIN markets m USING (symbol)
        WHERE m.base_asset = %s AND o.interval = %s AND o.ts >= {since} GROUP BY 1"""))

    # свічки OI — сума по біржах: open і close точні, high/low — сума екстремумів бірж
    # (верхня/нижня межа: піки на різних біржах могли бути в різні хвилини години)
    # OI у монетах: Coinalyze множить усю годинну свічку OI на close години, тож ділимо назад
    # на close того самого контракту — зникають стрибки між годинами, які дає рух ціни
    oi, oi_ohl, coin, first = {}, {}, {}, {}
    for t, ex, o, h, l, c, px in oi_rows:
        oi.setdefault(t, {})[ex] = c
        agg = oi_ohl.setdefault(t, [0.0, 0.0, 0.0])
        agg[0] += o if o is not None else c
        agg[1] += h if h is not None else c
        agg[2] += l if l is not None else c
        if px:
            ca = coin.setdefault(t, [0.0, 0.0, 0.0, 0.0])
            for k, v in enumerate((o, h, l, c)):
                ca[k] += (v if v is not None else c) / px
        first[ex] = min(first.get(ex, t), t)
    exs = sorted(first)
    ts = sorted(set(price) | set(oi))
    get = lambda t, ex: oi.get(t, {}).get(ex)
    return {
        "tf": tf,
        "exchanges": exs,
        "priceExchange": price_ex,
        "t": ts,
        "price": [price.get(t) for t in ts],
        "open": [candles[t][0] if t in candles else None for t in ts],
        "high": [candles[t][1] if t in candles else None for t in ts],
        "low": [candles[t][2] if t in candles else None for t in ts],
        "oi": [round(sum(oi[t].values())) if t in oi else None for t in ts],
        "oiO": [round(oi_ohl[t][0]) if t in oi_ohl else None for t in ts],
        "oiH": [round(oi_ohl[t][1]) if t in oi_ohl else None for t in ts],
        "oiL": [round(oi_ohl[t][2]) if t in oi_ohl else None for t in ts],
        "coinO": [coin[t][0] if t in coin else None for t in ts],
        "coinH": [coin[t][1] if t in coin else None for t in ts],
        "coinL": [coin[t][2] if t in coin else None for t in ts],
        "coin": [coin[t][3] if t in coin else None for t in ts],
        "oiBy": {ex: [round(get(t, ex)) if get(t, ex) is not None else None for t in ts] for ex in exs},
        "vol": [round(vol[t]) if vol.get(t) is not None else None for t in ts],
        "oiStart": first,
    }


def _json_safe(o):
    """NaN/inf → null, numpy → Python (json.dumps сам не вміє)."""
    import math
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if hasattr(o, "item") and not isinstance(o, (str, bytes)):
        o = o.item()
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    return o


def rows_to_json(rows: list[tuple]) -> list[dict]:
    return [{"base_asset": b, "start_ts": s.strftime(TS_FORMAT), "end_ts": e.strftime(TS_FORMAT),
             "pattern_type": p, "notes": n or "", "entry_ts": et.strftime(TS_FORMAT) if et else None,
             "entry_price": ep, "stop_price": sp, "setup": su} for b, s, e, p, n, et, ep, sp, su in rows]


def suggest_setups(conn) -> list[dict]:
    """Підказка сетапу для розмічених імпульсів без setup (власник підтверджує або виправляє).
    За правилом розмітки (2026-10-01) відкат, на якому OI тримався, — частина того самого імпульсу, тож сетап описує,
    з чого ПОЧАВСЯ імпульс: pullback — за 24 год до старту вже був рух ≥ 8% (або розмічений імпульс закінчився
    ≤ 24 год тому), ціна відкотилась, але втримала ≥ 50% того росту; інакше breakout (старт з рейнджу)."""
    rows = [r for r in read_pumps() if r[3] == "pump" and r[8] is None]
    out = []
    by_base: dict = {}
    for r in rows:
        by_base.setdefault(r[0], []).append(r)
    all_rows = read_pumps()
    for base, items in by_base.items():
        since = min(r[1] for r in items) - pd.Timedelta(days=2)
        h = aggregate(load_raw(conn, [base], INTERVAL_15M, since=since), base, "15min")
        if h.empty:
            continue
        for r in items:
            start = pd.Timestamp(r[1])
            w = h[(h.index >= start - pd.Timedelta(hours=24)) & (h.index + pd.Timedelta(minutes=15) <= start)]
            if len(w) < 16:
                continue
            p_start, p_first = w["price"].iloc[-1], w["price"].iloc[0]
            i_hi = w["high"].idxmax()
            hi = w.at[i_hi, "high"]
            oi_hi, oi_start = w.at[i_hi, "oi"], w["oi"].iloc[-1]
            prior = [x for x in all_rows if x[0] == base and x[3] == "pump" and x[2] <= r[1] and r[1] - x[2] <= pd.Timedelta(hours=24)]
            run = hi / p_first - 1
            pulled = p_start < hi * 0.98
            kept = (p_start - p_first) / (hi - p_first) if hi > p_first else 0      # частка росту, яку втримала ціна
            if (prior or run >= 0.08) and pulled and kept >= 0.5:
                setup = "pullback"
                why = (f"розмічений імпульс закінчився {(r[1] - prior[-1][2]) / pd.Timedelta(hours=1):.0f} год тому · " if prior
                       else f"рух +{run:.0%} за добу · ") + f"відкат {p_start / hi - 1:.0%} від максимуму · ціна втримала {kept:.0%} росту"
            else:
                setup = "breakout"
                why = (f"ціна віддала більшу частину росту (втримала {kept:.0%}) — старт знову з рейнджу" if (prior or run >= 0.08) and pulled
                       else f"за добу до старту максимум лише {run:+.0%} від її початку — попереднього руху не було")
            out.append({"base_asset": base, "start_ts": r[1].strftime(TS_FORMAT), "setup": setup, "reason": why})
    return out


# ---------------------------------------------------------------- чернетки імпульсів і черга розмітки
# Модель НЕ розмічає сама: до навчання потрапляє лише те, що власник підтвердив. Чернетка — підказка меж,
# черга — де розмітка дасть моделі найбільше (active learning).

OI_END_DROP = 0.10          # кінець імпульсу: OI −10% від піку …
END_GIVEBACK = 1 / 3        # … або закриття нижче «пік − ⅓ руху» (правило розмітки 2026-10-01)
END_MIN_MOVE = 0.05         # правило ⅓ діє, коли рух від старту ≥ 5%
PAUSE_LOOK = pd.Timedelta(hours=3)   # відкат на ⅓, після якого за 3 год є новий максимум, — пауза (розмітка бачить майбутнє)


def impulse_end(fw: pd.DataFrame, after: pd.Timestamp) -> pd.Timestamp | None:
    """Кінець імпульсу за правилом розмітки на 15m-барах fw (з бару старту): перше закриття нижче «пік − ⅓ руху»
    (рух ≥ 5%), після якого ціна НЕ оновила максимум протягом PAUSE_LOOK, або OI −10% від свого піку — що раніше;
    не раніше бару after (перший сигнал). Повертає час кінця (закриття бару) або None — рух ще триває.
    Перевірено на 64 імпульсах власника: медіана |різниці| з його кінцем 35 хв (без погляду вперед — 95 хв)."""
    import numpy as np
    B = train.BAR_15M
    p0 = fw["low"].iloc[0]
    hi, cl, oi, idx = fw["high"].to_numpy(), fw["price"].to_numpy(), fw["oi"].to_numpy(), fw.index
    pk, oimax, i = hi[0], 0.0, 0
    while i < len(fw):
        pk = max(pk, hi[i])
        if not np.isnan(oi[i]):
            oimax = max(oimax, oi[i])
        if idx[i] >= after:
            if not np.isnan(oi[i]) and oimax and oi[i] <= (1 - OI_END_DROP) * oimax:
                return idx[i] + B
            mv = pk - p0
            if mv >= END_MIN_MOVE * p0 and cl[i] < pk - END_GIVEBACK * mv:
                j, new_high = i + 1, False
                while j < len(fw) and idx[j] <= idx[i] + PAUSE_LOOK:
                    if hi[j] > pk:
                        new_high = True
                        break
                    j += 1
                if not new_high:
                    return idx[i] + B
                i = j
                continue
        i += 1
    return None


def draft_intervals(conn, base: str) -> list[dict]:
    """Чернетки імпульсів на тікері з епізодів Alert, які ще не розмічено.
    Правило розмітки (CLAUDE.md, impulse_end): кінець — закриття 15m нижче «пік − ⅓ руху» (рух ≥ 5%), після якого за
    3 год немає нового максимуму, або OI −10% від піку. Старт — мінімум ціни за ≤ 6 год до першого сигналу.
    Якщо кінця ще немає — рух триває, кінець = останній бар («кінець приблизний»).
    Епізоди Alert усередині чернетки — той самий рух. Межі не заходять на розмічене."""
    model15(conn, base)                          # оцінки тікера (кеш)
    t = next(iter(_score_cache.values()))
    h = aggregate(load_raw(conn, [base], INTERVAL_15M), base, "15min").dropna(subset=["price", "high", "low"])
    if t.empty or h.empty:
        return []
    labeled = [(r[1], r[2]) for r in read_pumps() if r[0] == base]
    B = train.BAR_15M
    alert = (t["tier"] == "alert").to_numpy()
    out = []
    for e in train.signal_episodes(t[["base_asset", "ts"]], alert, B).itertuples():
        if any(a < e.end and b > e.start for a, b in labeled):
            continue
        w0 = h[(h.index >= e.start - pd.Timedelta(hours=6)) & (h.index <= e.start)]
        start = w0["low"].idxmin() if len(w0) else e.start
        w1 = h[(h.index >= e.start) & (h.index < e.end + pd.Timedelta(hours=6))]
        if w1.empty:
            continue
        # кінець — правило розмітки (impulse_end); немає — рух триває, кінець = останній бар
        fw = h[(h.index >= start) & (h.index < e.end + pd.Timedelta(hours=36))]
        found = impulse_end(fw, e.start)
        stop_at = found - B if found is not None else fw.index[-1]
        peak = fw.loc[:stop_at, "high"].idxmax()
        drop = [found] if found is not None else []
        end = stop_at + B
        nxt = [a for a, b in labeled if a >= start]          # не налазити на наступний розмічений інтервал
        if nxt:
            end = min(end, min(nxt))
        prev = [b for a, b in labeled if b <= e.start]
        if prev:
            start = max(start, max(prev))
        if end <= start:
            continue
        span = t[(t["ts"] >= e.start) & (t["ts"] < e.end)]
        p0, pk = h.at[start, "price"] if start in h.index else w0["price"].iloc[0], fw.loc[:stop_at, "high"].max()
        oi0, oik = h["oi"].asof(start), h["oi"].asof(peak)
        d = {"start": start, "end": end, "alert_start": e.start, "alert_min": int((e.end - e.start) / pd.Timedelta(minutes=1)),
             "p_max": float(span["p15"].max()), "move": float(pk / p0 - 1) if p0 else None,
             "oi_chg": float(oik / oi0 - 1) if oi0 and pd.notna(oi0) and pd.notna(oik) else None,
             "oi_drop": bool(len(drop))}
        if out and d["start"] < out[-1]["end"]:       # два епізоди одного руху — одна чернетка
            out[-1].update(end=max(out[-1]["end"], d["end"]), p_max=max(out[-1]["p_max"], d["p_max"]))
            continue
        out.append(d)
    for d in out:
        d["reason"] = (f"Alert {d['alert_min'] // 60} год {d['alert_min'] % 60} хв, P до {d['p_max']:.2f}"
                       + (f" · ціна +{d['move']:.0%}" if d["move"] is not None else "")
                       + (f" · OI {d['oi_chg']:+.0%}" if d["oi_chg"] is not None else "")
                       + ("" if d["oi_drop"] else " · розвороту ще немає — рух триває, кінець приблизний"))
        for k in ("start", "end", "alert_start"):
            d[k] = int(pd.Timestamp(d[k]).timestamp())
    return out


_queue_cache: dict = {}


def label_queue(conn) -> list[dict]:
    """Де розмітка дасть моделі найбільше: тікери скринера з нерозміченими епізодами (оцінки з таблиці signals).
    Бали: Alert без розмітки ×3 (або памп, або хибний — обидва цінні), «лише Кандидат» (15m так, 1h ні) ×2,
    майже-сигнал (15m 0.85–1 порогу) ×1; тікер поза навчанням — +2 (свіжий приклад для моделі або holdout).
    Для тікерів з tickers.txt — лише останні 7 днів (давнішу історію ти вже переглядав)."""
    key = int(time.time() // 300)
    if key in _queue_cache:
        return _queue_cache[key]
    sig = query_df(conn, """
        SELECT s.base_asset, s.ts, s.model, s.probability, s.threshold FROM signals s
        JOIN (SELECT model, max(model_version) AS v FROM signals GROUP BY model) m ON s.model = m.model AND s.model_version = m.v
        WHERE s.model IN ('lightgbm_15m', 'lightgbm')""")
    if sig.empty:
        return []
    m15, _ = models_15m()
    share = m15.get("confirm", {}).get("share", 0.75)
    t = tier_frame(sig[sig["model"] == "lightgbm_15m"], sig[sig["model"] == "lightgbm"], share)
    reviewed = {x.upper() for x in read_tickers_file(TICKERS_PATH)}
    trained = set(m15.get("tickers", []))
    excluded = excluded_tickers(ROOT)
    pumps = load_pumps(conn)
    B = train.BAR_15M
    recent = pd.Timestamp(datetime.now(timezone.utc)) - pd.Timedelta(days=7)
    out = []
    for base, g in t.groupby("base_asset"):
        if base in excluded:
            continue
        if base in reviewed:
            g = g[g["ts"] >= recent]
        own = pumps[pumps["base_asset"] == base]
        unl = lambda e: not ((own["start_ts"] < e.end) & (own["end_ts"] > e.start)).any()  # noqa: E731
        df = g[["base_asset", "ts"]]
        al = [e for e in train.signal_episodes(df, (g["tier"] == "alert").to_numpy(), B).itertuples() if unl(e)]
        co = [e for e in train.signal_episodes(df, (g["tier"] == "candidate").to_numpy(), B).itertuples() if unl(e)]
        sig_eps = al + co
        near = [e for e in train.signal_episodes(df, ((g["s15"] >= 0.85) & (g["s15"] < 1)).to_numpy(), B).itertuples()
                if unl(e) and not any(x.start < e.end + pd.Timedelta(hours=1) and x.end > e.start - pd.Timedelta(hours=1) for x in sig_eps)]
        fresh = base not in trained and base not in reviewed
        score = 3 * len(al) + 2 * len(co) + len(near) + (2 if fresh and (al or co) else 0)
        if not (al or co or near):
            continue
        first = min([e.start for e in al + co + near])
        why = [f"{len(al)} Alert без розмітки"] if al else []
        why += [f"{len(co)} лише Кандидат (1h не підтвердила)"] if co else []
        why += [f"{len(near)} майже-сигнал"] if near else []
        out.append({"ticker": base, "score": score, "alert": len(al), "cand_only": len(co), "near": len(near),
                    "first": int(pd.Timestamp(first).timestamp()), "fresh": fresh, "reviewed": base in reviewed, "why": " · ".join(why)})
    out.sort(key=lambda x: -x["score"])
    _queue_cache.clear()
    _queue_cache[key] = out[:25]
    return _queue_cache[key]


def user_stops(conn) -> dict:
    """Стопи власника з розмітки → k = (вхід − стоп) / ATR(15m) на барі входу; зведення проти k моделі."""
    rows = [r for r in read_pumps() if r[7] is not None]
    out, ks = [], []
    for base, start, end, ptype, notes, et, ep, sp, _setup in rows:
        h = aggregate(load_raw(conn, [base], INTERVAL_15M, since=et - pd.Timedelta(days=2)), base, "15min")
        a = atr(h)
        bar = pd.Timestamp(et).floor("15min")
        k = (ep - sp) / a.get(bar) if pd.notna(a.get(bar)) and a.get(bar) else None
        if k is not None:
            ks.append(k)
        out.append({"base_asset": base, "start_ts": start.strftime(TS_FORMAT), "k": k, "stop_pct": sp / ep - 1})
    q = lambda p: float(pd.Series(ks).quantile(p)) if ks else None  # noqa: E731
    try:
        k_model = models_15m()[0].get("risk", {}).get("k")
    except ValueError:
        k_model = None
    return {"rows": out, "n": len(ks), "median": q(.5), "p25": q(.25), "p75": q(.75), "k_model": k_model}


class Handler(BaseHTTPRequestHandler):
    db_url: str = ""
    days: int = 60

    def log_message(self, fmt, *args):  # без логу кожного запиту в консоль
        pass

    def _send(self, code: int, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _error(self, code: int, msg: str):
        self._send(code, {"error": msg})

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        url = urlparse(self.path)
        try:
            if url.path == "/":
                with open(PAGE, "rb") as f:
                    return self._send(200, f.read(), "text/html; charset=utf-8")
            if url.path.startswith(("/vendor/", "/js/")):
                base_dir = VENDOR if url.path.startswith("/vendor/") else JS_DIR
                path = os.path.normpath(os.path.join(base_dir, os.path.basename(url.path)))
                if not path.startswith(base_dir) or not os.path.isfile(path):
                    return self._error(404, "не знайдено")
                with open(path, "rb") as f:
                    return self._send(200, f.read(), "text/javascript; charset=utf-8")
            if url.path == "/api/tickers":
                with psycopg.connect(self.db_url) as conn:
                    tickers = [r[0] for r in conn.execute(
                        "SELECT DISTINCT base_asset FROM markets ORDER BY 1").fetchall()]
                return self._send(200, {"tickers": tickers, "days": self.days})
            if url.path == "/api/data":
                qs = parse_qs(url.query)
                base = (qs.get("ticker") or [""])[0].upper()
                tf = (qs.get("tf") or ["1h"])[0]
                if tf not in INTERVAL:
                    raise ValueError(f"невідомий таймфрейм {tf}")
                with psycopg.connect(self.db_url) as conn:
                    return self._send(200, ticker_data(conn, base, self.days, tf))
            if url.path == "/api/profile":      # профіль обсягу на графіку (app/js/profile.js, volume_profile.py)
                q = {k: v[0] for k, v in parse_qs(url.query).items()}
                with psycopg.connect(self.db_url) as conn:
                    return self._send(200, volume_profile.api(conn, q))
            if url.path == "/api/predictions":
                base = (parse_qs(url.query).get("ticker") or [""])[0].upper()
                return self._send(200, predictions(base))
            if url.path == "/api/pumps":
                with lock:
                    return self._send(200, {"rows": rows_to_json(read_pumps())})
            if url.path == "/api/universe":
                with psycopg.connect(self.db_url) as conn:
                    db = [r[0] for r in conn.execute("SELECT DISTINCT base_asset FROM markets ORDER BY 1")]
                try:
                    binance, err = binance_universe(), None
                except Exception as e:           # без інтернету — лише тікери з бази
                    binance, err = [], str(e)
                return self._send(200, {"binance": binance, "db": db, "error": err,
                                        "training": [t.upper() for t in read_tickers_file(TICKERS_PATH)],
                                        "excluded": sorted(excluded_tickers(ROOT))})
            if url.path == "/api/jobs":
                return self._send(200, JOBS)
            if url.path == "/compare":
                with open(COMPARE_PAGE, "rb") as f:
                    return self._send(200, f.read(), "text/html; charset=utf-8")
            if url.path == "/review":
                with open(REVIEW_PAGE, "rb") as f:
                    return self._send(200, f.read(), "text/html; charset=utf-8")
            if url.path.startswith("/api/review/"):
                import review as R
                qs = {k: v[0] for k, v in parse_qs(url.query).items()}
                with psycopg.connect(self.db_url) as conn:
                    if url.path == "/api/review/next":
                        res = R.next_item(conn, qs.get("after"))
                        prefetch_review(self.db_url)
                        return self._send(200, _json_safe(res))
                    if url.path == "/api/review/item":
                        return self._send(200, _json_safe(R.get_item(conn, qs.get("id", ""))))
                    if url.path == "/api/review/bars":
                        tf = qs.get("tf", "15m")
                        if tf not in ("5m", "15m", "1h"):
                            raise ValueError("ТФ: 5m / 15m / 1h")
                        tk, ts = qs.get("id", "|").split("|")
                        return self._send(200, _json_safe(R.bars(conn, tk, int(ts), tf)))
                    if url.path == "/api/review/summary":
                        return self._send(200, _json_safe(R.summary(conn)))
                return self._error(404, "не знайдено")
            if url.path == "/api/compare":
                with psycopg.connect(self.db_url) as conn:
                    return self._send(200, _json_safe(compare(conn)))
            if url.path == "/api/model_status":
                with psycopg.connect(self.db_url) as conn:
                    return self._send(200, model_status(conn))
            if url.path == "/api/pumps/suggest_setup":
                with psycopg.connect(self.db_url) as conn:
                    return self._send(200, {"items": suggest_setups(conn)})
            if url.path == "/api/drafts":
                base = (parse_qs(url.query).get("ticker") or [""])[0].upper()
                with psycopg.connect(self.db_url) as conn:
                    items = draft_intervals(conn, base)
                    t = next(iter(_score_cache.values()), pd.DataFrame())
                    info = {"from": int(t["ts"].min().timestamp()) if len(t) else None,
                            "alert_bars": int((t["tier"] == "alert").sum()) if len(t) else 0,
                            "p_max": float(t["p15"].max()) if len(t) else None}
                    return self._send(200, _json_safe({"items": items, **info}))
            if url.path == "/api/queue":
                with psycopg.connect(self.db_url) as conn:
                    return self._send(200, _json_safe({"items": label_queue(conn)}))
            if url.path == "/api/stops":
                with psycopg.connect(self.db_url) as conn:
                    return self._send(200, user_stops(conn))
            if url.path == "/api/model15":
                base = (parse_qs(url.query).get("ticker") or [""])[0].upper()
                with psycopg.connect(self.db_url) as conn:
                    return self._send(200, model15(conn, base))
            return self._error(404, "не знайдено")
        except ValueError as e:
            return self._error(400, str(e))

    def _modify(self, change):
        """Прочитати pumps.csv, змінити, перевірити, записати і синхронізувати з БД."""
        with lock:
            rows = read_pumps()
            rows = change(rows)
            errors = check_overlaps(rows)
            if errors:
                raise ValueError("; ".join(errors))
            write_pumps(rows)
            with psycopg.connect(self.db_url) as conn:
                sync_db(conn, rows)
            return rows

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            body = self._body()
            if path == "/api/pumps":
                new = parse_row(body)
                rows = self._modify(lambda rows: rows + [new])
                return self._send(200, {"rows": rows_to_json(rows)})
            if path == "/api/review":                # рішення сліпого перегляду → reviews.csv
                import review as R
                return self._send(200, R.save_review(body))
            if path == "/api/review/rebuild":
                import review as R
                with psycopg.connect(self.db_url) as conn:
                    pool = R.build_pool(conn)
                return self._send(200, {"total": len(pool)})
            if path == "/api/load":
                base = str(body.get("ticker", "")).upper()
                if not re.fullmatch(r"[^\W_]{1,20}", base):
                    raise ValueError("невірна назва тікера")
                if JOBS.get(base, {}).get("state") not in ("queued", "running"):
                    JOBS[base] = {"state": "queued", "msg": "у черзі", "step": 0, "steps": 3, "stage": "у черзі"}
                    _jobs.put(base)
                return self._send(200, JOBS[base])
            if path == "/api/retrain":
                with _retrain_lock:
                    if RETRAIN.get("state") == "running":
                        raise ValueError("перенавчання вже йде")
                    RETRAIN.update(state="running", step=0, stage="старт", started=time.time(), log=[])
                threading.Thread(target=retrain_worker, daemon=True).start()
                return self._send(200, {"ok": True})
            if path == "/api/retrain/accept":
                backup = retrain_accept()
                return self._send(200, {"ok": True, "backup": backup})
            if path == "/api/retrain/discard":
                import shutil
                shutil.rmtree(os.path.join(ROOT, train.CANDIDATE_DIR), ignore_errors=True)
                RETRAIN.update(state="idle")
                return self._send(200, {"ok": True})
            if path == "/api/pumps/update":        # редагування інтервалу на графіку: межі, тип, нотатка, вхід, стоп
                base, orig = str(body.get("base_asset", "")).upper(), parse_ts(body["orig_start_ts"])
                new = parse_row(body)
                if new[0] != base:
                    raise ValueError("тікер інтервалу змінювати не можна")
                found = []

                def replace(rows):
                    out = []
                    for r in rows:
                        if r[0] == base and r[1] == orig:
                            found.append(r)
                            r = new
                        out.append(r)
                    return out
                rows = self._modify(replace)
                if not found:
                    raise ValueError("інтервал не знайдено — онови сторінку")
                return self._send(200, {"rows": rows_to_json(rows)})
            if path == "/api/pumps/risk":          # вхід і стоп власника для вже розміченого інтервалу
                base, start = str(body.get("base_asset", "")).upper(), parse_ts(body["start_ts"])

                def change(rows):
                    out = []
                    for r in rows:
                        if r[0] == base and r[1] == start:
                            d = dict(zip(["base_asset", "start_ts", "end_ts", "pattern_type", "notes"],
                                         [r[0], r[1].strftime(TS_FORMAT), r[2].strftime(TS_FORMAT), r[3], r[4] or ""]))
                            d.update(entry_ts=body.get("entry_ts") or "", entry_price=body.get("entry_price") or "",
                                     stop_price=body.get("stop_price") or "", setup=r[8] or "")
                            r = parse_row(d)
                        out.append(r)
                    return out
                rows = self._modify(change)
                return self._send(200, {"rows": rows_to_json(rows)})
            if path == "/api/pumps/setup":
                items = body.get("items") or [body]
                want = {(str(x.get("base_asset", "")).upper(), parse_ts(x["start_ts"])): (x.get("setup") or None) for x in items}
                bad = [v for v in want.values() if v is not None and v not in SETUPS]
                if bad:
                    raise ValueError(f"невідомий setup: {bad[0]}")

                def change(rows):
                    return [r[:8] + (want[(r[0], r[1])],) if (r[0], r[1]) in want else r for r in rows]
                rows = self._modify(change)
                return self._send(200, {"rows": rows_to_json(rows)})
            if path == "/api/exclude":        # тікер поза скоупом: не навчання, не скан, не скринер (excluded.txt)
                base = str(body.get("ticker", "")).upper()
                if not re.fullmatch(r"[^\W_]{1,20}", base):
                    raise ValueError("невірна назва тікера")
                on = bool(body.get("on"))
                reason = re.sub(r"[#\r\n]", " ", str(body.get("reason", ""))).strip()[:120]
                set_ticker_in_file(os.path.join(ROOT, EXCLUDED_FILE), base, on, reason or "поза скоупом")
                if on:
                    set_ticker_in_file(TICKERS_PATH, base, False)      # і з навчання цілком
                return self._send(200, {"excluded": sorted(excluded_tickers(ROOT)),
                                        "training": [t.upper() for t in read_tickers_file(TICKERS_PATH)]})
            if path == "/api/training":
                base = str(body.get("ticker", "")).upper()
                if not re.fullmatch(r"[^\W_]{1,20}", base):
                    raise ValueError("невірна назва тікера")
                set_ticker_in_file(TICKERS_PATH, base, bool(body.get("on")), "переглянуто повністю в label.py")
                return self._send(200, {"training": [t.upper() for t in read_tickers_file(TICKERS_PATH)]})
            return self._error(404, "не знайдено")
        except (ValueError, json.JSONDecodeError) as e:
            return self._error(400, str(e))

    def do_DELETE(self):
        if urlparse(self.path).path != "/api/pumps":
            return self._error(404, "не знайдено")
        try:
            body = self._body()
            base, start = body["base_asset"].upper(), parse_ts(body["start_ts"])
            rows = self._modify(lambda rows: [r for r in rows if not (r[0] == base and r[1] == start)])
            return self._send(200, {"rows": rows_to_json(rows)})
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            return self._error(400, str(e))


def main() -> None:
    parser = argparse.ArgumentParser(description="Локальний інструмент розмітки імпульсів")
    parser.add_argument("--days", type=int, default=60, help="скільки днів історії показувати")
    parser.add_argument("--port", type=int, default=8765, help="порт на 127.0.0.1")
    parser.add_argument("--no-browser", action="store_true", help="не відкривати браузер автоматично")
    args = parser.parse_args()

    load_dotenv()
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        sys.exit("Заповни DATABASE_URL у файлі .env")
    try:
        read_pumps()
    except ValueError as e:
        sys.exit(f"{e}\nВиправ pumps.csv і запусти знову.")

    Handler.db_url, Handler.days = db_url, args.days
    threading.Thread(target=load_worker, args=(db_url, os.getenv("COINALYZE_API_KEY"), args.days), daemon=True).start()
    # На Windows SO_REUSEADDR дозволяє двом процесам зайняти один порт: другий «стартує», але запитів не отримує.
    # Вимикаємо його, щоб повторний запуск одразу казав, що порт зайнятий.
    ThreadingHTTPServer.allow_reuse_address = False
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError:
        sys.exit(f"Порт {args.port} зайнятий: мабуть, уже запущено інший екземпляр. "
                 f"Зупини його (Ctrl+C у тому терміналі) або запусти з --port 8766.")
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Розмітка імпульсів: {url}  (Ctrl+C — зупинити)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nЗупинено.")


if __name__ == "__main__":
    main()
