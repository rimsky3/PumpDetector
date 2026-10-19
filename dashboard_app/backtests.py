"""
dashboard_app/backtests.py — вкладка «Бектест»: збережені прогони backtest.py, запуск нового; переоцінка при старті дашборда.
"""

import json
import os
import sys
import threading

import psycopg

import journal as J
import strategies
from fetch import read_tickers_file
from screener import WATCHLIST_FILE, score_15m, score_bases

from dashboard_app.common import ROOT, State, refresh_lock


# ---------------------------------------------------------------- бектест (backtest.py) — вкладка «Бектест»

BT_DIR = os.path.join(ROOT, "reports", "backtests")
BT_JOB: dict = {"proc": None, "log": [], "args": None}


def backtest_runs() -> list[dict]:
    """Список прогонів: мета й підсумок (без угод) з reports/backtests/*.json, нові першими."""
    out = []
    if not os.path.isdir(BT_DIR):
        return out
    for f in sorted(os.listdir(BT_DIR), reverse=True):
        if not f.endswith(".json") or f.startswith("_"):
            continue
        try:
            with open(os.path.join(BT_DIR, f), encoding="utf-8") as fh:
                d = json.load(fh)
        except (OSError, ValueError):
            continue
        out.append({**d["meta"], **{k: d.get(k) for k in ("coins", "trades", "sum", "sum_wo_top3", "median", "win_rate")}})
    return out


def backtest_run(run_id: str) -> dict:
    if not run_id or "/" in run_id or "\\" in run_id or ".." in run_id:
        raise ValueError("невірний id прогону")
    path = os.path.join(BT_DIR, run_id + ".json")
    if not os.path.exists(path):
        raise ValueError(f"прогону {run_id} немає")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def backtest_status() -> dict:
    p = BT_JOB["proc"]
    running = p is not None and p.poll() is None
    st = {}
    try:
        with open(os.path.join(BT_DIR, "_status.json"), encoding="utf-8") as fh:
            st = json.load(fh)
    except (OSError, ValueError):
        pass
    return {"running": running, "code": None if p is None or running else p.returncode, **st,
            "log": BT_JOB["log"][-4:], "args": BT_JOB["args"]}


def backtest_start(body: dict) -> dict:
    """Запуск backtest.py у фоні (окремий процес — довгий: дані для ~100 монет ~30–60 хв першого разу)."""
    import subprocess
    if backtest_status()["running"]:
        raise ValueError("бектест уже йде")
    args = ["--min-turnover", str(float(body.get("min_turnover", 20e6))), "--days", str(int(body.get("days", 30))),
            *(["--max-turnover", str(float(body["max_turnover"]))] if body.get("max_turnover") else []),
            "--model", "candidate" if body.get("model") == "candidate" else "current",
            "--strategy", strategies.get(body.get("strategy"))["id"]]
    BT_JOB.update(log=[], args=args)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    p = subprocess.Popen([sys.executable, os.path.join(ROOT, "backtest.py"), *args], cwd=ROOT, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", env=env)
    BT_JOB["proc"] = p

    def pump():
        for line in p.stdout:
            if line.strip():
                BT_JOB["log"] = (BT_JOB["log"] + [line.rstrip()])[-60:]
    threading.Thread(target=pump, daemon=True).start()
    return {"ok": True}


def rescore_on_start() -> None:
    """Після перенавчання в signals ще немає оцінок нової версії моделі, а дашборд показує лише її.
    Оцінюємо наявні в базі дані одразу, не чекаючи наступного циклу скринера (без запитів до API)."""
    try:
        with psycopg.connect(State.db_url) as conn:
            known = {r[0] for r in conn.execute("SELECT DISTINCT base_asset FROM markets")}
            bases = [t for t in read_tickers_file(os.path.join(ROOT, WATCHLIST_FILE)) if t.upper() in known]
            if not bases:
                return
            versions = [m["trained_at"] for m in State.models]
            have = {r[0] for r in conn.execute(
                "SELECT DISTINCT base_asset FROM signals WHERE model_version = ANY(%s)", (versions,))}
            missing = [b.upper() for b in bases if b.upper() not in have]
            if missing:
                with refresh_lock:
                    score_bases(conn, missing, State.models)
                print(f"Оцінено новою моделлю: {', '.join(missing)}")
            if State.m15 is not None:
                have15 = {r[0] for r in conn.execute(
                    "SELECT DISTINCT base_asset FROM signals WHERE model = %s AND model_version = %s",
                    (J.M15, State.m15["trained_at"]))}
                missing15 = [b.upper() for b in bases if b.upper() not in have15]
                if missing15:
                    with refresh_lock:
                        score_15m(conn, missing15, State.m15)
                    print(f"Оцінено 15m-моделлю: {', '.join(missing15)}")
    except Exception as e:
        print(f"! не вдалося оцінити тікери при старті: {e}")
