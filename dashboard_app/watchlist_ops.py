"""
dashboard_app/watchlist_ops.py — зміни списку: «＋ у watchlist» (окремий процес dashboard.py --quick-add), «×» і «Повернути», картка «Сканер».
"""

import os
import sys

import fetch_exchange
import scanner
from features import INTRADAY_EXCHANGES
from fetch import Coinalyze
from screener import WATCHLIST_FILE, score_15m, score_bases_own

from dashboard_app.common import ROOT, State, binance_bases, list_sources


# ---------------------------------------------------------------- сканер ринку і додавання тікерів

ADD_JOBS: dict = {}          # тікер → "loading" / "done" / текст помилки (дашборд показує прогрес)


def scan_view(q: dict) -> dict:
    """Картка «Сканер»: налаштування, скільки пройшло б зараз, результат набору на розмічених пампах, авто-список.
    Параметри запиту (preset або пороги «Свій») — попередній перегляд ДО збереження."""
    cfg = scanner.load_config()
    manual, auto = list_sources()
    preset = q.get("preset", cfg["preset"])
    if preset not in scanner.PRESET_NAMES:
        preset = cfg["preset"]
    p = dict(cfg["custom"] if preset == "custom" else scanner.PRESETS[preset])
    for k in scanner.PRESETS["balance"]:
        if k in q:
            p[k] = scanner._clip(k, q[k])
    p["max_oi"] = cfg["max_oi"]
    return {"config": cfg, "preset": preset, "params": p,
            "presets": {k: {"name": v, "params": scanner.PRESETS.get(k, cfg["custom"])} for k, v in scanner.PRESET_NAMES.items()},
            "preview": scanner.preview(p, set(manual)), "recall": scanner.recall_for(p, cfg["ttl_hours"]),
            "auto": [{"ticker": b, **r} for b, r in sorted(auto.items(), key=lambda x: x[1]["last_hit"], reverse=True)],
            "manual": manual, "jobs": {b: add_status(b) for b in set(ADD_JOBS) | set(ADD_PROCS)}}


def add_to_watchlist(base: str) -> dict:
    """«＋ у watchlist»: дописати у watchlist.txt і одразу (у фоні) — лише швидку частину: контракти, 15m/5m з бірж,
    15m- і годинна оцінка з власних 15m (quick_add). Годинну історію Coinalyze скринер дозбирає сам у своїй черзі —
    раніше тут був повний Refresh (60 днів Coinalyze), і дашборд зі скринером ловили 429 на спільному ліміті (07.10, GRIFFAIN)."""
    from fetch import set_ticker_in_file
    base = base.upper().removesuffix("USDT").strip()
    if not base.isalnum():
        raise ValueError("введи базовий актив, напр. BANK")
    if base not in binance_bases():
        raise ValueError(f"{base}: немає серед USDT-перпів Binance")
    set_ticker_in_file(os.path.join(ROOT, WATCHLIST_FILE), base, True, "додано з дашборду")
    # окремим процесом з нижчим пріоритетом: потік у процесі дашборда (ознаки, модель) ділив з запитами графіків
    # один GIL — поки додавався тікер, графіки гальмували (07.10, BEAT)
    import subprocess
    import tempfile
    log = open(os.path.join(tempfile.gettempdir(), f"pd_add_{base}.log"), "w", encoding="utf-8")
    kw = {"preexec_fn": lambda: os.nice(10)} if hasattr(os, "nice") else {}
    ADD_PROCS[base] = (subprocess.Popen([sys.executable, os.path.join(ROOT, "dashboard.py"), "--quick-add", base],
                                        stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, **kw), log.name)
    ADD_JOBS[base] = "loading"
    return {"ok": True, "ticker": base}


ADD_PROCS: dict = {}         # тікер → (процес quick_add, файл журналу)


def add_status(base: str) -> str | None:
    """Стан «＋ у watchlist»: loading / done / остання строка помилки процесу quick_add."""
    item = ADD_PROCS.get(base)
    if item is not None:
        p, log = item
        rc = p.poll()
        if rc is None:
            return "loading"
        ADD_PROCS.pop(base, None)
        if rc == 0:
            ADD_JOBS[base] = "done"
        else:
            try:
                lines = [x.strip() for x in open(log, encoding="utf-8", errors="replace") if x.strip()]
            except OSError:
                lines = []
            ADD_JOBS[base] = lines[-1][:200] if lines else f"помилка (код {rc})"
    return ADD_JOBS.get(base)


def quick_add(conn, base: str) -> None:
    """Новий тікер без годинної історії Coinalyze: контракти бірж (2 виклики Coinalyze, якщо їх ще немає в markets),
    10 днів 15m/5m з бірж (без ліміту Coinalyze), 15m-оцінка і годинна з власних 15m — графік і сигнали за хвилину."""
    from features import INTRADAY_EXCHANGES
    from fetch import find_markets, upsert_markets
    have = conn.execute("SELECT 1 FROM markets WHERE base_asset = %s AND exchange_name = ANY(%s) LIMIT 1",
                        (base, list(INTRADAY_EXCHANGES))).fetchone()
    if not have:
        if State.client is None:
            State.client = Coinalyze(State.api_key)
        ms = find_markets(State.client, [base])
        if not ms:
            raise ValueError(f"{base}: контрактів не знайдено")
        upsert_markets(conn, ms)
        conn.commit()
    fetch_exchange.collect(conn, [base], ["15m", "5m"], 10, log=lambda m: None)
    if State.m15 is not None:
        score_15m(conn, [base], State.m15)
        score_bases_own(conn, [base], State.models)


REMOVED: dict = {}      # «×» у списку: base → ("manual" | "auto", запис авто-списку) — для «Повернути»


def remove_from_watchlist(base: str) -> dict:
    """«×» у рядку списку: ручний тікер — геть з watchlist.txt, авто — з авто-списку; обидва блокуються «до нового
    спрацювання» (scanner.block): той самий сплеск, що й досі проходить фільтр, їх не поверне, а новий — поверне як авто.
    Скринер не оцінює тікер з наступного циклу."""
    from fetch import set_ticker_in_file
    base = base.upper()
    manual, _ = list_sources()
    out = {"ok": True, "ticker": base, "source": None}
    if base in manual:
        set_ticker_in_file(os.path.join(ROOT, WATCHLIST_FILE), base, False)
        REMOVED[base] = ("manual", None)
        out["source"] = "manual"
    rec = scanner.remove_auto(base)
    if rec is not None:
        REMOVED[base] = ("auto", rec)
        out["source"] = "auto"
    scanner.block(base)
    ADD_JOBS.pop(base, None)
    return out


def undo_remove(base: str) -> dict:
    """«Повернути» після «×»: ручний — назад у watchlist.txt (без перезавантаження даних), авто — запис і без паузи."""
    from fetch import set_ticker_in_file
    base = base.upper()
    item = REMOVED.pop(base, None)
    if item is None:
        raise ValueError(f"{base}: нічого повертати")
    kind, rec = item
    if kind == "manual":
        set_ticker_in_file(os.path.join(ROOT, WATCHLIST_FILE), base, True, "повернено з дашборду")
        scanner.unblock(base)
    else:
        scanner.restore_auto(base, rec)
    return {"ok": True, "ticker": base}
