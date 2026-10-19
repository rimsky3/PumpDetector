"""
dashboard_app/journal_view.py — вкладка «Журнал» (/api/journal): угоди всіх режимів на барах поза навчанням (journal.py); важкий розрахунок —
в окремому процесі, у фоні лише 2 год після останнього звернення (JOURNAL_KEEP_WARM).
"""

import os
import threading
import time

import pandas as pd
import psycopg

import journal as J
import strategies
import tiers as T
from screener import load_model_15m, load_models

from dashboard_app.common import ROOT, State, virtual_models


_journal_cache: dict = {}
# угоди всіх моделей (journal.prepare) — ключ: правило «Impulse», версія оцінок, розмітка; період лише фільтрує їх
_prepared: dict = {"key": None, "data": None, "rule": dict(J.DEFAULT_RULE)}
_prepared_lock = threading.Lock()


def journal_bundles() -> dict:
    """Режими для журналу: {id режиму: {scores, bundle, strat}} (strategies.json); режим без своєї моделі — пропускається."""
    hourly = {m["name"]: m for m in State.models}
    out = {}
    for s in strategies.load():
        bundle = State.m15 if strategies.is15(s) else hourly.get(s["model"])
        if bundle is not None:
            out[s["id"]] = {"scores": strategies.scores_name(s), "bundle": bundle, "strat": s}
    return out


def journal_version(conn) -> tuple:
    """Коли журнал треба перерахувати: оцінено НОВИЙ бар (без ковзних 5-хв — вони в журнал не йдуть) або змінена розмітка.
    Ключ — останній оцінений бар і скільки тікерів на ньому, а не max(scored_at): скринер постійно дописує дооцінки
    (нові тікери, історія, годинна частина), і журнал на сервері (~80 с на 2 vCPU) перераховувався майже без перерви —
    дашборд їв ~60% процесора і гальмував скринер (2026-10-06)."""
    last = conn.execute("""SELECT ts, count(*) FROM signals
                           WHERE model <> %s AND ts = (SELECT max(ts) FROM signals WHERE model <> %s) GROUP BY ts""",
                        (T.ROLL_MODEL, T.ROLL_MODEL)).fetchone()
    p = os.path.join(ROOT, "pumps.csv")
    return last, os.path.getmtime(p) if os.path.exists(p) else 0


_journal_pool = None


def _prepare_in_child(db_url: str, rule: dict) -> dict:
    """J.prepare в окремому процесі: важкий pandas-розрахунок журналу (до хвилини на сервері) інакше тримає GIL
    процесу дашборду, і графік тікера чи список, запитані в цей момент, чекали в черзі (2026-10-06).
    Моделі дочірній процес читає сам — один раз (дашборд теж читає їх лише при старті)."""
    if not State.models and State.m15 is None:
        State.models, State.m15 = load_models(), load_model_15m()
    with psycopg.connect(db_url) as conn:
        return J.prepare(conn, journal_bundles(), rule)


def _prepare(conn, rule: dict) -> dict:
    global _journal_pool
    try:
        if _journal_pool is None:
            from concurrent.futures import ProcessPoolExecutor
            _journal_pool = ProcessPoolExecutor(max_workers=1)
        return _journal_pool.submit(_prepare_in_child, State.db_url, dict(rule)).result()
    except Exception as e:                    # процес не стартував / впав — рахуємо як раніше, у цьому ж процесі
        print(f"  ! журнал в окремому процесі: {type(e).__name__}: {e} — рахую тут")
        _journal_pool = None
        return J.prepare(conn, journal_bundles(), rule)


def prepared_for(conn, rule: dict) -> tuple[dict, tuple]:
    key = (tuple(rule.values()),) + journal_version(conn)
    with _prepared_lock:                      # фоновий прогрів і запит не рахують одне й те саме двічі
        if _prepared["key"] != key:
            _prepared["data"] = _prepare(conn, rule)
            _prepared["key"] = key
        _prepared["rule"] = dict(rule)
        return _prepared["data"], key


JOURNAL_KEEP_WARM = 2 * 3600                   # с: стільки після останнього звернення до журналу фон тримає його свіжим
_journal_used = {"at": 0.0}                    # коли журнал востаннє запитували (вкладка «Журнал», «Результати» бота)


def journal_warmer() -> None:
    """Фоновий перерахунок угод — ЛИШЕ поки журналом користуються (13.10, власник: «завантажувати тільки коли потрібно»):
    перше відкриття після перерви рахує журнал на запит (на сервері ~1–2 хв), далі JOURNAL_KEEP_WARM секунд після
    останнього звернення фон оновлює його з кожним новим баром, і вкладка відкривається одразу. Ніхто не дивиться —
    нічого не рахується (раніше журнал рахувався цілодобово, двічі на кожен 15m-бар)."""
    seen = None
    while True:
        try:
            if time.time() - _journal_used["at"] < JOURNAL_KEEP_WARM:
                with psycopg.connect(State.db_url) as conn:
                    # Скринер оцінює бар двома порціями (гарячі тікери першими, решта — за 1–4 хв): версія журналу
                    # змінюється двічі на бар. Рахуємо, коли вона однакова на двох перевірках поспіль.
                    v = journal_version(conn)
                    if v == seen:
                        prepared_for(conn, _prepared["rule"])
                    seen = v
            else:
                seen = None
        except Exception as e:  # журнал не має валити дашборд
            print(f"  ! журнал (фон): {e}")
        time.sleep(60)


def journal_view(conn, q: dict) -> dict:
    _journal_used["at"] = time.time()           # журналом користуються — фон тримає його свіжим ще JOURNAL_KEEP_WARM
    rule = {
        "horizon": max(1, min(168, int(q.get("horizon", J.DEFAULT_RULE["horizon"])))),
        "price": float(q.get("price", J.DEFAULT_RULE["price"])),
        "oi": float(q.get("oi", J.DEFAULT_RULE["oi"])),
    }
    days = None if q.get("days", "30") == "all" else max(1, int(q.get("days", 30)))
    model, kind = strategies.get(q.get("model"))["id"], q.get("kind", "hist")       # model = id режиму
    # вибір дат (YYYY-MM-DD, дні UTC включно) — замість «останніх N днів»
    day = lambda s: pd.Timestamp(s, tz="UTC").normalize() if s else None  # noqa: E731
    d_from, d_to = day(q.get("from")), day(q.get("to"))
    if d_from is not None and d_to is not None and d_to < d_from:
        d_from, d_to = d_to, d_from
    prepared, pkey = prepared_for(conn, rule)
    key = (model, days, kind, d_from, d_to) + pkey
    if key not in _journal_cache:
        if len(_journal_cache) > 32:
            _journal_cache.clear()
        res = J.journal(conn, journal_bundles(), model, days, kind, rule, d_from, d_to, prepared)
        val = {m["name"]: m["validation"] for m in State.models}
        for v in virtual_models():
            val[v["name"]] = v["validation"]
        res["validation"] = {sid: val.get(spec["scores"]) for sid, spec in journal_bundles().items()}
        _journal_cache[key] = res
    return _journal_cache[key]
