"""
screener.py — скринер пампів для тікерів з watchlist.txt, цикл кожні 15 хвилин.

Кожні 15 хв (о HH:02, :17, :32, :47 UTC — 15m-бар щойно закрився):
    1. 15m-дані з Binance і Bybit (fetch_exchange.collect) і 15m-ліквідації з Coinalyze
    2. ознаки features.build_15m() → 15m-модель (models/lightgbm_15m.joblib) → таблиця signals
Раз на годину (перший цикл нової години) — як і раніше:
    3. 1h-дані Coinalyze (останні 10 днів) → features.build() → годинні моделі → signals

Рівні сигналу рахуються з цих оцінок при читанні (tiers.py): «Кандидат» — 15m-модель вище порогу,
«Alert» — плюс годинний LightGBM ≥ 0.75 свого порогу.

    python screener.py              # один цикл (і годинний, і 15m)
    python screener.py --loop       # безперервно, кожні 15 хв
    python screener.py --no-fetch   # без збору даних: оцінити те, що вже є в базі

Ліміт Coinalyze — 40 символів на хвилину: годинний збір ~15 викликів на тікер, 15m-ліквідації ~3 на тікер.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import psycopg
import queue
import threading
from dotenv import load_dotenv

import fetch_exchange
import orderbook
import wake
import scanner
from features import (INTRADAY_EXCHANGES, build, build_15m, build_from_raw, build_hourly_own, load_raw, oi_seen_share,
                      roll_intraday, short_features)
from fetch import Coinalyze, collect, collect_liquidations, find_markets, read_tickers_file, upsert_markets
from tiers import ROLL_MODEL
from train import M15_NAME, PROBA_MODELS, load_model, model_path

WATCHLIST_FILE = "watchlist.txt"
STATUS_FILE = os.path.join("reports", "screener_status.json")   # читає дашборд, блок «Стан системи»
# мітка «є нові оцінки»: дашборд перевіряє її mtime кожні 5 с (/api/pulse) і одразу оновлює список і графік
SIGNALS_STAMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports", "signals_stamp")
FETCH_DAYS = 10          # 7 днів базового вікна + доба + запас
BACKFILL_HOURS = 7 * 24  # скільки годин історії оцінювати, якщо їх ще немає в signals
RUN_MINUTE = 5           # (застаріле) хвилина годинного циклу — лишено для сумісності
RUN_MINUTES = (2, 17, 32, 47)   # 15m-цикл: бар закрився ~2 хв тому, біржі й Coinalyze його вже віддають
RUN_MINUTES_5 = tuple(range(2, 60, 5))   # + швидкі цикли між ними (:07, :12, :22…) — лише «гарячі» тікери
HOT_SHARE = 0.8                 # «гарячий»: 15m-оцінка ≥ 0.8 порогу за останні 2 год
LIQ5_DAYS = 10                  # 5m-ліквідації гарячим — завжди 10 днів: виклик Coinalyze той самий (за символом),
                                # а дірки в історії (тікер був «холодним») інакше стали б «нулем ліквідацій» в ознаках
BAR_15M = pd.Timedelta(minutes=15)
BACKFILL_15M = 7 * 24 * 4       # скільки 15m-барів історії оцінювати, якщо їх ще немає в signals
INTRADAY_DAYS = 10              # новому тікеру — 10 днів 15m-історії (тижневі вікна ознак + запас)
INTRADAY_TAIL_DAYS = 0.25       # мінімальний хвіст — 6 годин (upsert, дублікатів немає)
MAX_GAP_DAYS = 29               # найдовший пропуск, який дозаповнюємо (історія OI Binance — 30 днів)
OWN_HOURLY_MIN_SHARE = 0.70     # годинна модель з власних 15m, якщо Binance+Bybit тримають ≥ 70% OI тікера
COINALYZE_OWN_REFRESH_H = 6     # таким тікерам годинні дані Coinalyze — раз на 6 год (графік 1h, історія), не щогодини
_coin_refreshed: dict = {}      # тікер → коли востаннє тягнули годинні дані Coinalyze
BACKGROUND_HOURS = 3             # тікери навчання поза списком: історія бірж (з OKX) раз на 3 год
NO_COINALYZE: set[str] = set()   # тікери, яких Coinalyze не має (годинних даних не буде)


def load_models() -> list[dict]:
    """Годинні моделі (logreg, lightgbm)."""
    models = []
    for name in PROBA_MODELS:
        if os.path.exists(model_path(name)):
            models.append(load_model(name))
    if not models:
        sys.exit("Немає збережених моделей у models/ — спершу python train.py")
    return models


def load_model_15m() -> dict | None:
    """15m-модель (None, якщо ще не навчена: тоді працює лише годинний контур)."""
    return load_model(M15_NAME) if os.path.exists(model_path(M15_NAME)) else None


def last_closed_15m(now: datetime | None = None) -> pd.Timestamp:
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    return now.floor("15min") - BAR_15M


def first_gap_15m(conn, bases: list[str]) -> dict:
    """Для кожного тікера — з якого моменту дозавантажувати 15m: перший ПРОПУЩЕНИЙ бар свічок моделі (Binance, Bybit)
    за останні INTRADAY_DAYS днів, а якщо пропусків немає — останній бар. Саме «останній бар у базі» не годиться:
    дашборд при відкритті тікера дотягує лише короткий свіжий хвіст, і дірка між ним і старими даними лишалась
    назавжди — ознакам бракує барів, і модель на тікері замовкає (так було з NIGHT 2026-09-30 14:30–15:30).
    Тікера без даних у словнику немає — він новий."""
    from features import INTRADAY_EXCHANGES
    since = pd.Timestamp(datetime.now(timezone.utc)) - pd.Timedelta(days=INTRADAY_DAYS)
    rows = conn.execute("""
        SELECT m.base_asset, m.exchange_name, o.ts FROM ohlcv o JOIN markets m USING (symbol)
        WHERE o.interval = '15min' AND m.base_asset = ANY(%s) AND m.exchange_name = ANY(%s) AND o.ts >= %s""",
        (bases, list(INTRADAY_EXCHANGES), since)).fetchall()
    df = pd.DataFrame(rows, columns=["base", "ex", "ts"])
    out = {}
    last_closed = last_closed_15m()
    for b, g in df.groupby("base"):
        first = None
        for _, ge in g.groupby("ex"):
            ts = pd.DatetimeIndex(pd.to_datetime(ge["ts"], utc=True)).unique().sort_values()
            grid = pd.date_range(ts[0], last_closed, freq="15min")
            miss = grid.difference(ts)
            cand = miss[0] if len(miss) else ts[-1]
            first = cand if first is None else min(first, cand)
        out[b] = first
    return out


def update_15m(conn, client: Coinalyze | None, bases: list[str], log=print) -> None:
    """Свіжі 15m/5m-дані з бірж і 15m-ліквідації з Coinalyze. 5m моделі не потрібен — лише графіку дашборда.

    Скільки брати — за фактичним пропуском: від останнього бару в базі до зараз плюс запас (не менше 6 год),
    новому тікеру — 10 днів. Так нічний простій ноутбука дозаповнюється повністю, без дірок посередині.
    Стеля — 29 днів: далі історія OI на Binance недоступна (на OKX — ~14 днів, на Bybit — рік+)."""
    have = first_gap_15m(conn, bases)
    now = pd.Timestamp(datetime.now(timezone.utc))
    groups: dict[float, list[str]] = {}
    for b in bases:
        if b not in have:
            days = INTRADAY_DAYS
        else:
            gap = (now - pd.Timestamp(have[b])) / pd.Timedelta(days=1)
            days = min(MAX_GAP_DAYS, max(INTRADAY_TAIL_DAYS, np.ceil((gap + 0.05) * 4) / 4))   # крок 6 год — менше груп
        groups.setdefault(float(days), []).append(b)
    quiet = lambda m: log(m) if "помилка" in m else None  # noqa: E731
    for days, group in sorted(groups.items()):
        fetch_exchange.collect(conn, group, ["15m", "5m"], days, log=quiet)
        if client is not None:
            collect_liquidations(client, conn, group, ["15m"], days, log=lambda m: None)


def score_15m(conn, bases: list[str], m15: dict) -> tuple[pd.DataFrame, pd.Timestamp]:
    """15m-ознаки → ймовірність 15m-моделі → signals. Спільне для скринера і кнопки «Прогнозувати»."""
    last = last_closed_15m()
    # тижневі вікна ознак «з'їдають» 7 днів історії: щоб оцінити BACKFILL_15M барів, беремо їх + 8 днів
    df = build_15m(conn, bases, None, since=last - BACKFILL_15M * BAR_15M - pd.Timedelta(days=8))
    if df.empty:
        return pd.DataFrame(), last
    df = df[(df["ts"] > last - BACKFILL_15M * BAR_15M) & (df["ts"] <= last)]
    scored = score(df, [m15])
    save_signals(conn, scored, last)
    return scored, last


def hot_bases(conn, bases: list[str], hours: int = 2) -> list[str]:
    """Тікери, де 15m-модель за останні hours годин була ≥ HOT_SHARE порогу, або що «прокинулись» (wake.py) за останні
    wake.HOT_HOURS: їм 5m-ліквідації й оцінка кожні 5 хв."""
    if not bases:
        return []
    rows = conn.execute("""SELECT DISTINCT base_asset FROM signals
                           WHERE model = ANY(%s) AND base_asset = ANY(%s) AND ts > now() - make_interval(hours => %s)
                             AND probability >= %s * threshold""", ([M15_NAME, ROLL_MODEL], bases, hours, HOT_SHARE)).fetchall()
    woke = set(wake.last_wake(minutes=wake.HOT_HOURS * 60)) & set(bases)
    return sorted({r[0] for r in rows} | woke)


def score_15m_rolling(conn, bases: list[str], m15: dict) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    """15m-модель на КОВЗНОМУ 15m-барі, що щойно закрився (з 5m-даних бірж і 5m-ліквідацій). Бар зі зсувом 5 або 10 хв,
    рядок у signals — з часом його відкриття і моделлю ROLL_MODEL. Модель та сама, без перенавчання."""
    open_ = pd.Timestamp(datetime.now(timezone.utc)).floor("5min") - BAR_15M
    off = open_.minute % 15
    if off == 0 or not bases:              # вирівняний бар рахує звичайний 15m-цикл
        return pd.DataFrame(), None
    raw5 = load_raw(conn, bases, "5min", since=open_ - pd.Timedelta(days=8))     # + тижневі вікна ознак
    # лише тікери, де КОЖЕН ринок уже має останній 5m-бар і свічки, і OI (OI біржі віддають після закриття бару):
    # неповний бар = «обвал OI» або половина обсягу → хибно низька чи висока P
    last5, recent = open_ + 2 * pd.Timedelta(minutes=5), open_ - pd.Timedelta(hours=1)
    ok = set(bases)
    for k in ("ohlcv", "oi"):
        d = raw5[k][raw5[k]["ts"] >= recent]
        newest = d.groupby(["base_asset", "market"])["ts"].max()
        ok &= set(newest.index.get_level_values(0)) - set(newest[newest < last5].index.get_level_values(0))
    if not ok:
        return pd.DataFrame(), open_
    raw5 = {k: (v[v["base_asset"].isin(ok)] if "base_asset" in v else v) for k, v in raw5.items()}
    f = build_from_raw(roll_intraday(raw5, "15min", off, 3), bars_per_hour=4, extra=short_features)
    f = f[f["ts"] == open_] if len(f) else f
    if len(f):                             # без тижневих ознак (мало 5m-історії) — не оцінюємо: P була б ≈ 0
        f = f.dropna(subset=["oi_z_7d", "volume_z_7d", "oi_chg_24h"])
    if f.empty:
        return pd.DataFrame(), open_
    scored = score(f, [m15]).assign(model=ROLL_MODEL)
    save_signals(conn, scored, open_)
    return scored, open_


def quick_cycle(conn, client: Coinalyze | None, bases: list[str], m15: dict, fetch: bool) -> None:
    """Швидкий цикл між 15-хвилинними: хвіст 5m з бірж для всього списку → «прокидається» (wake.py; хто прокинувся —
    одразу «гарячий») → для гарячих 5m-ліквідації й оцінка ковзного бару."""
    started = datetime.now(timezone.utc)
    if fetch:                 # хвіст 5m усім (біржі напряму, без Coinalyze): правилу «прокидається» потрібен щойно закритий бар
        fetch_exchange.collect(conn, bases, ["5m"], 0.1, log=lambda m: None, exchanges=list(INTRADAY_EXCHANGES))
    try:
        wake.check(conn, bases)
    except Exception as e:
        print(f"  ! прокидається: {e}")
    hot = hot_bases(conn, bases)
    if not hot:
        print(f"[{started:%H:%M} UTC] 5-хв цикл: гарячих тікерів немає")
        return
    if fetch:
        # ознакам потрібен тиждень історії: у кого 5m коротша (авто-тікер, раніше — лише хвіст) — разово 9 днів
        start = conn.execute("""SELECT m.base_asset, min(o.ts) FROM ohlcv o JOIN markets m USING (symbol)
                                WHERE o.interval = '5min' AND m.base_asset = ANY(%s) AND o.ts > now() - interval '9 days'
                                  AND m.exchange_name = ANY(%s) GROUP BY 1""", (hot, list(INTRADAY_EXCHANGES))).fetchall()
        start = {b: pd.Timestamp(t) for b, t in start}
        short = [b for b in hot if b not in start or start[b] > pd.Timestamp(started) - pd.Timedelta(days=8, hours=6)]
        if short:
            fetch_exchange.collect(conn, short, ["5m"], 9, log=lambda m: None, exchanges=list(INTRADAY_EXCHANGES))
        if client is not None:
            collect_liquidations(client, conn, hot, ["5m"], LIQ5_DAYS, log=lambda m: None)
    s, open_ = score_15m_rolling(conn, hot, m15)
    try:
        orderbook.record(conn, hot)       # стакан гарячих — кожні 5 хв (історія «за 15 хв» і OB у момент сигналу)
    except Exception as e:
        print(f"  ! стакан: {e}")
    sig = s[s["signal"]]["base_asset"].tolist() if len(s) else []
    try:
        scanner.touch(sig)
    except Exception as e:
        print(f"  ! авто-список: {e}")
    took = (datetime.now(timezone.utc) - started).seconds
    bar = f"ковзний бар {open_:%H:%M}" if open_ is not None else "вирівняний бар — пропуск"
    print(f"[{started:%H:%M} UTC] 5-хв цикл: гарячі {', '.join(hot)}; {bar}, оцінено {len(s)}"
          + (f"; вище порогу: {', '.join(sig)}" if sig else "") + f" · {took} с")
    write_status(cycle_5m_at=started.isoformat(timespec="seconds"), hot_5m=hot)


_score_lock = threading.Lock()      # моделі оцінюють і основний цикл, і фоновий потік — по черзі


def score(df: pd.DataFrame, models: list[dict]) -> pd.DataFrame:
    """Ймовірності всіх моделей для кожного рядка ознак."""
    out = []
    for m in models:
        with _score_lock:
            p = m["model"].predict_proba(df[m["features"]])[:, 1]
        # in_sample: ця година цього тікера була в навчанні моделі — оцінка не є чесним передбаченням
        seen = df["base_asset"].isin(m["tickers"]).to_numpy() & (df["ts"] <= m["data_end"]).to_numpy()
        for base, a, b in m.get("in_sample_extra", []):     # свіжі розмічені інтервали теж були в навчанні
            seen |= ((df["base_asset"] == base) & (df["ts"] >= a) & (df["ts"] <= b)).to_numpy()
        out.append(df[["base_asset", "ts"]].reset_index(drop=True).assign(
            model=m["name"], model_version=m["trained_at"],
            probability=p, threshold=m["threshold"], signal=p >= m["threshold"], in_sample=seen,
        ))
    return pd.concat(out, ignore_index=True)


def save_signals(conn, scored: pd.DataFrame, last_closed: pd.Timestamp) -> int:
    """Нові оцінки → signals. Уже записані години не чіпаємо (ON CONFLICT DO NOTHING)."""
    # live — чесна оцінка в реальному часі: остання закрита година, якої модель не бачила
    scored = scored.assign(live=(scored["ts"] == last_closed) & ~scored["in_sample"])
    rows = [(r.base_asset, r.ts.to_pydatetime(), r.model, r.model_version, float(r.probability),
             float(r.threshold), bool(r.signal), bool(r.live), bool(r.in_sample)) for r in scored.itertuples()]
    cur = conn.cursor()
    cur.executemany("""
        INSERT INTO signals (base_asset, ts, model, model_version, probability, threshold, signal, live, in_sample)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (base_asset, ts, model, model_version) DO NOTHING""", rows)
    conn.commit()
    try:
        with open(SIGNALS_STAMP, "w") as f:
            f.write(datetime.now(timezone.utc).isoformat())
    except OSError:
        pass
    return len(rows)


def score_bases(conn, bases: list[str], models: list[dict]) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Ознаки → ймовірності → signals для тікерів. Спільне для скринера і кнопки «Прогнозувати» в дашборді."""
    # остання закрита година — на момент оцінки, а не старту: збір може тривати кілька хвилин і перейти годину
    now = datetime.now(timezone.utc)
    last_closed = pd.Timestamp(now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1))
    df = build(conn, bases, labels=False)
    if df.empty:
        return pd.DataFrame(), last_closed
    df = df[df["ts"] > last_closed - pd.Timedelta(hours=BACKFILL_HOURS)]
    scored = score(df, models)
    save_signals(conn, scored, last_closed)
    return scored, last_closed


def split_hourly(conn, bases: list[str]) -> tuple[list[str], list[str]]:
    """Звідки годинне підтвердження: own — з наших 15m-даних (Binance+Bybit ≥ OWN_HOURLY_MIN_SHARE OI, або тікера немає
    в Coinalyze), coin — як раніше, з годинних даних Coinalyze (усі біржі: напр., MON — 42% OI на Hyperliquid)."""
    share = oi_seen_share(conn, bases)
    own = [b for b in bases if share.get(b, 1.0) >= OWN_HOURLY_MIN_SHARE]
    return own, [b for b in bases if b not in own]


def score_bases_own(conn, bases: list[str], models: list[dict]) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Як score_bases, але годинні ознаки — з наших 15m-даних: оцінка закритої години готова одразу о HH:02,
    без годинного збору Coinalyze (~23 виклики на тікер, на 27 тікерах ~19 хв і 429)."""
    now = datetime.now(timezone.utc)
    last_closed = pd.Timestamp(now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1))
    since = last_closed - pd.Timedelta(hours=BACKFILL_HOURS) - pd.Timedelta(days=8)    # + тижневі вікна ознак
    df = build_hourly_own(conn, bases, since=since) if bases else pd.DataFrame()
    if df.empty:
        return pd.DataFrame(), last_closed
    df = df[df["ts"] > last_closed - pd.Timedelta(hours=BACKFILL_HOURS)]
    scored = score(df, models)
    save_signals(conn, scored, last_closed)
    return scored, last_closed


_status_lock = threading.Lock()


def write_status(**fields) -> None:
    """Оновити reports/screener_status.json: нові поля поверх наявних (пишуть і основний цикл, і фоновий годинний
    потік — під замком, щоб не затирати одне одного). Успішний цикл (ok=True) знімає стару помилку."""
    os.makedirs(os.path.dirname(STATUS_FILE), exist_ok=True)
    if fields.get("ok") is True:
        fields.setdefault("error", None)
    with _status_lock:
        prev = screener_status_file()
        with open(STATUS_FILE, "w", encoding="utf-8") as f:
            json.dump({**prev, "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"), **fields}, f, ensure_ascii=False)


class Background:
    """Фоновий потік для довгих завантажень з Coinalyze (годинні дані, історія нових тікерів, тікери навчання поза
    списком): 15m- і 5m-цикли більше не чекають їх (AIN 03.10: годинна частина тривала 27 хв — 15m-цикл стояв).
    Свій конект до бази; виклики Coinalyze — з нижчим пріоритетом (Coinalyze.background). Завдання з однаковим
    ключем, що вже в черзі чи виконується, не дублюються."""

    def __init__(self, db_url: str, client: Coinalyze):
        self.db_url, self.client = db_url, client
        self.q: queue.Queue = queue.Queue()
        self.keys: set = set()
        self.lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True, name="screener-background").start()

    def submit(self, key: str, fn, *args) -> bool:
        with self.lock:
            if key in self.keys:
                return False
            self.keys.add(key)
        self.q.put((key, fn, args))
        return True

    def busy(self, key: str) -> bool:
        with self.lock:
            return key in self.keys

    def _run(self) -> None:
        while True:
            key, fn, args = self.q.get()
            t0 = datetime.now(timezone.utc)
            try:
                with psycopg.connect(self.db_url) as conn, self.client.background():
                    fn(conn, *args)
                print(f"  [фон] {key}: готово за {(datetime.now(timezone.utc) - t0).seconds} с")
            except Exception as e:
                print(f"  ! [фон] {key}: {type(e).__name__}: {e}")
            finally:
                with self.lock:
                    self.keys.discard(key)


BG: Background | None = None      # створюється в main() у режимі --loop; без нього все — як раніше, послідовно
_onboarding: dict = {}             # тікер → коли його завели без годинної історії (вона докачується у фоні)


def onboard(conn, client: Coinalyze, bases: list[str], models: list[dict], m15: dict | None, all15: bool = False) -> list[str]:
    """Нові тікери (сканер, «＋ у watchlist»): годинні дані Coinalyze (заодно заводять біржові контракти в markets —
    без них 15m-дані з бірж не завантажаться), годинна оцінка, 15m-дані й 15m-оцінка — одразу, не чекаючи
    годинної частини циклу, інакше до кінця години є лише «Кандидат» без Alert.
    «Новий» = немає ні свіжих 15m-даних бірж (2 год), ні годинних Coinalyze (COINALYZE_OWN_REFRESH_H + 2 год; напр.,
    тікер колись вантажили в розмітці). Раніше — лише «годинні старші за 2 год»: тікери з годинною оцінкою з власних
    15m оновлюють Coinalyze раз на 6 год, тож 4 години з 6 виглядали «новими» і раз на 3 год заново тягнули всю
    історію (на сервері 06.10 — 19 тікерів, цикл 25 хв замість кількох, пропущені 15m-бари)."""
    now_h = pd.Timestamp(datetime.now(timezone.utc)).floor("h")
    cutoff = now_h - pd.Timedelta(hours=2)                                   # для перевірки після збору (BG is None)
    sql = """SELECT m.base_asset FROM ohlcv o JOIN markets m USING (symbol)
             WHERE o.interval = '1hour' AND m.base_asset = ANY(%s) GROUP BY 1 HAVING max(o.ts) >= %s"""
    from features import INTRADAY_EXCHANGES
    have = {r[0] for r in conn.execute(sql, (bases, now_h - pd.Timedelta(hours=COINALYZE_OWN_REFRESH_H + 2)))}
    have |= {r[0] for r in conn.execute("""
        SELECT m.base_asset FROM ohlcv o JOIN markets m USING (symbol)
        WHERE o.interval = '15min' AND m.base_asset = ANY(%s) AND m.exchange_name = ANY(%s) AND o.ts >= %s
        GROUP BY 1""", (bases, list(INTRADAY_EXCHANGES), cutoff))}
    now = datetime.now(timezone.utc)
    recent = {b for b, t in _onboarding.items() if now - t < timedelta(hours=3)}   # уже заведені, історія — у фоні
    fresh = [b for b in bases if b not in have and b not in NO_COINALYZE and b not in recent]
    if all15 and m15 is not None and not fresh:      # від сканера: годинні дані свіжі, але 15m ще не рахували
        update_15m(conn, client, bases)
        score_15m(conn, bases, m15)
    if not fresh:
        return []
    if BG is None:                                   # без фонового потоку — як раніше: годинна історія одразу
        collect(client, conn, fresh, FETCH_DAYS, verbose=False)
        got = {r[0] for r in conn.execute(sql, (fresh, cutoff))}
        NO_COINALYZE.update(set(fresh) - got)     # не знайдено в Coinalyze — не питати щоциклу (до перезапуску)
        fresh = [b for b in fresh if b in got]
        if fresh:
            score_bases(conn, fresh, models)
            if m15 is not None:
                b15 = bases if all15 else fresh
                update_15m(conn, client, b15)
                score_15m(conn, b15, m15)
        return fresh
    # контракти бірж — двома викликами на весь список (без них 15m з бірж не завантажаться), далі 15m-дані, 15m- і
    # годинна оцінка з власних 15m — одразу; годинна історія Coinalyze (~23 виклики на тікер) — у фоні
    markets = find_markets(client, fresh)
    found = {m["base_asset"].upper() for m in markets}
    NO_COINALYZE.update(set(fresh) - found)
    fresh = [b for b in fresh if b in found]
    if not fresh:
        return []
    upsert_markets(conn, markets)
    for b in fresh:
        _onboarding[b] = now
    if m15 is not None:
        b15 = bases if all15 else fresh
        update_15m(conn, client, b15)
        score_15m(conn, b15, m15)
        try:
            score_bases_own(conn, fresh, models)
        except Exception as e:
            print(f"  ! годинна з власних 15m для нових: {e}")
    BG.submit(f"історія нових: {', '.join(fresh)}", coin_history, fresh, models)
    print(f"  нові тікери: {', '.join(fresh)} — 15m одразу, годинна історія Coinalyze у фоні")
    return fresh


def run_cycle(conn, bases: list[str], models: list[dict], fetch: bool, client: Coinalyze | None,
              m15: dict | None = None, hourly: bool = True) -> None:
    started = datetime.now(timezone.utc)
    calls0, limited0 = (client.calls_made, client.rate_limited) if client else (0, 0)
    print(f"\n[{started:%Y-%m-%d %H:%M} UTC] цикл: {len(bases)} тікерів" + ("" if hourly else " (15m)"))

    info15, hot = {}, []
    own, coin = split_hourly(conn, bases) if hourly else ([], [])
    scored_own, own_done = pd.DataFrame(), set()
    if m15 is not None:
        # ГАРЯЧІ ПЕРШИМИ (2026-10-06, RLC: Alert бару 10:45 з'явився лише о 11:11): тікери біля порогу за 2 год або ті, що
        # «прокинулись», — свіжі дані, 15m-оцінка і годинне підтвердження з власних 15m ДО решти списку. Раніше всі чекали
        # збору 15m-ліквідацій Coinalyze на весь список (~200 викликів, 40/хв) — оцінка була на ~7–9 хв після закриття бару.
        first = hot_bases(conn, bases)
        groups = [g for g in (first, [b for b in bases if b not in set(first)]) if g]
        parts, last15 = [], last_closed_15m()
        for group in groups:
            if fetch:
                update_15m(conn, client, group)
            s, last15 = score_15m(conn, group, m15)
            parts.append(s)
            mine = [b for b in group if b in own]
            if mine:                           # годинна оцінка щойно закритої години — одразу, щоб Alert не чекав решти
                so, _ = score_bases_own(conn, mine, models)
                scored_own = pd.concat([d for d in (scored_own, so) if len(d)], ignore_index=True) if len(so) else scored_own
                own_done |= set(mine)
            if group is first and len(groups) > 1:
                print(f"  гарячі першими ({', '.join(first)}): оцінено за {(datetime.now(timezone.utc) - started).seconds} с")
        parts = [p for p in parts if len(p)]
        s15 = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if fetch:
            try:
                wake.check(conn, bases)        # 5m щойно оновлено для всього списку
            except Exception as e:
                print(f"  ! прокидається: {e}")
            hot5 = hot_bases(conn, bases)
            if hot5 and client is not None:    # для наступних 5-хв циклів; 15m-оцінці не потрібні
                collect_liquidations(client, conn, hot5, ["5m"], LIQ5_DAYS, log=lambda m: None)
        latest15 = s15[s15["ts"] == last15] if len(s15) else s15
        info15 = {"last_closed_15m": last15.isoformat(), "scored_15m": int(latest15["base_asset"].nunique()),
                  "missing_15m": sorted(set(bases) - set(latest15["base_asset"])) if len(s15) else bases}
        hot = latest15[latest15["signal"]]["base_asset"].tolist() if len(latest15) else []
        try:
            scanner.touch(hot)            # авто-тікер із сигналом моделі не випадає зі списку
        except Exception as e:
            print(f"  ! авто-список: {e}")
        print(f"  15m: бар {last15:%H:%M} UTC, оцінено {info15['scored_15m']} тікерів"
              + (f"; вище порогу: {', '.join(hot)}" if hot else ""))
        # знімок стакану (біди / аски ±2%) для тікерів біля сигналу — «OB» у момент Alert і live-перевірка правила
        try:
            orderbook.record(conn, sorted(set(hot) | set(hot_bases(conn, bases))))
        except Exception as e:
            print(f"  ! стакан: {e}")
    fresh = onboard(conn, client, bases, models, m15) if fetch and client is not None else []
    if not hourly:
        write_status(**info15, ok=True, cycle_15m_at=started.isoformat(timespec="seconds"),
                     api_calls_15m=(client.calls_made - calls0) if client else 0)
        return

    # годинне підтвердження з власних 15m — до будь-якого збору Coinalyze (секунди); гарячим уже пораховано вище
    left = [b for b in own if b not in own_done]
    if left:
        so, _ = score_bases_own(conn, left, models)
        scored_own = pd.concat([d for d in (scored_own, so) if len(d)], ignore_index=True) if len(so) else scored_own
    if own:
        print(f"  годинна оцінка з власних 15m: {len(own)} тікерів за {(datetime.now(timezone.utc) - started).seconds} с"
              + (f"; з Coinalyze (покриття < {OWN_HOURLY_MIN_SHARE:.0%}): {', '.join(coin)}" if coin else ""))
    if BG is not None:
        # збір Coinalyze і годинна оцінка решти — у фоні: наступні 15m- і 5m-цикли не чекають
        if not BG.submit("годинна частина (Coinalyze)", hourly_coinalyze, bases, models, fetch, client, own, coin, hot, fresh,
                         started, scored_own, calls0, limited0, info15, m15 is not None):
            print("  годинна частина ще триває з минулого разу — цю годину пропускаємо")
        return
    hourly_coinalyze(conn, bases, models, fetch, client, own, coin, hot, fresh, started, scored_own, calls0, limited0,
                     info15, m15 is not None)


def hourly_coinalyze(conn, bases, models, fetch, client, own, coin, hot, fresh, started, scored_own, calls0, limited0,
                     info15, has15) -> None:
    """Годинні дані Coinalyze (тікери з покриттям Binance+Bybit < 70% OI — щогодини, решта — раз на 6 год) і годинна
    оцінка з них; звіт у консоль і статус. У режимі --loop — у фоновому потоці (Background)."""
    if fetch:
        # зазвичай 10 днів; після довгого простою — увесь пропуск (Coinalyze тримає ~60 днів; ліміт — за символами, не днями)
        last1h = conn.execute("""SELECT min(t) FROM (SELECT m.base_asset, max(o.ts) AS t FROM ohlcv o JOIN markets m USING (symbol)
                                 WHERE o.interval = '1hour' AND m.base_asset = ANY(%s) GROUP BY 1) x""", (bases,)).fetchone()[0]
        gap = (pd.Timestamp(started) - pd.Timestamp(last1h)) / pd.Timedelta(days=1) if last1h else 0
        days = int(min(60, max(FETCH_DAYS, np.ceil(gap) + 1)))
        # спершу тікери, де горить 15m-«Кандидат»: їм годинна оцінка потрібна для Alert — вона буде за ~1–2 хв
        # після старту циклу, а не після збору всього списку (Coinalyze: ~23 виклики на тікер, 40 на хвилину)
        first = [b for b in hot if b not in fresh and b in coin]
        if first:
            collect(client, conn, first, days, verbose=False)
            score_bases(conn, first, models)
            print(f"  годинна оцінка першими для гарячих: {', '.join(first)}")
        # тікери з власною годинною оцінкою — Coinalyze лише раз на COINALYZE_OWN_REFRESH_H (графік 1h, історія)
        # «давно» — за базою (останній годинний бар Coinalyze), а не лише за пам'яттю процесу: інакше після кожного
        # перезапуску скринер перетягував годинні дані всіх тікерів і перший цикл тривав ~18 хв (пропускав 15m-цикл)
        if not _coin_refreshed and own:
            for b_, t_ in conn.execute("""SELECT m.base_asset, max(o.ts) FROM ohlcv o JOIN markets m USING (symbol)
                                          WHERE o.interval = '1hour' AND m.base_asset = ANY(%s) GROUP BY 1""", (own,)):
                _coin_refreshed[b_] = pd.Timestamp(t_).to_pydatetime() + timedelta(hours=1)
        due = [b for b in own if started - _coin_refreshed.get(b, datetime.min.replace(tzinfo=timezone.utc))
               >= timedelta(hours=COINALYZE_OWN_REFRESH_H)]
        rest = [b for b in coin + due if b not in fresh and b not in first]     # нові вже зібрано в onboard()
        if rest:
            collect(client, conn, rest, days, verbose=False)
            for b in due:
                _coin_refreshed[b] = started
        print(f"  дані оновлено за {(datetime.now(timezone.utc) - started).seconds} с")
    scored, last_closed = score_bases(conn, coin, models) if coin else (pd.DataFrame(), pd.Timestamp(
        datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)))
    scored = pd.concat([scored_own, scored], ignore_index=True)
    print(f"  остання закрита година: {last_closed:%m-%d %H:%M} UTC")
    if scored.empty:
        print("  немає тікерів з повною історією для ознак (потрібно ≥ 7 днів)")
        return

    # звіт: остання година по кожному тікеру
    latest = scored[scored["ts"] == scored.groupby("base_asset")["ts"].transform("max")]
    table = latest.pivot_table(index="base_asset", columns="model", values="probability")
    table["signal"] = latest.groupby("base_asset")["signal"].any()
    table["in_sample"] = latest.groupby("base_asset")["in_sample"].any()
    last_ts = latest.groupby("base_asset")["ts"].max()
    table["last_bar"] = last_ts.dt.strftime("%m-%d %H:%M")
    table = table.sort_values(table.columns[0], ascending=False)
    stale = table[last_ts.reindex(table.index) < last_closed]
    missing = sorted(set(bases) - set(table.index))

    active = table[table["signal"]]
    print(f"  активних сигналів: {len(active)}" + (f" — {', '.join(active.index)}" if len(active) else ""))
    print(table.head(10).round(3).to_string())
    if len(stale):
        print(f"  ! дані відстають (немає останньої години): {', '.join(stale.index)}")
    if missing:
        print(f"  ! без оцінки (мало історії або не знайдено): {', '.join(missing)}")
    write_status(ok=True, cycle_started=started.isoformat(timespec="seconds"),
                 seconds=int((datetime.now(timezone.utc) - started).total_seconds()),
                 tickers=len(bases), scored=len(table), stale=list(stale.index), missing=missing,
                 hourly_own=len(own), hourly_coinalyze=coin,
                 api_calls=(client.calls_made - calls0) if client else 0,
                 api_429=(client.rate_limited - limited0) if client else 0, fetched=fetch,
                 cycle_15m_at=started.isoformat(timespec="seconds") if has15 else None, **info15)




def screener_status_file() -> dict:
    try:
        with open(STATUS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def next_run(now: datetime, minutes: tuple = RUN_MINUTES) -> datetime:
    """Найближчий момент циклу: HH:02, :17, :32, :47 (або кожні 5 хв — RUN_MINUTES_5)."""
    base = now.replace(second=0, microsecond=0)
    for m in minutes:
        t = base.replace(minute=m)
        if t > now:
            return t
    return base.replace(minute=minutes[0]) + timedelta(hours=1)


def next_hourly(now: datetime) -> datetime:
    nxt = now.replace(minute=RUN_MINUTE, second=0, microsecond=0)
    return nxt if nxt > now else nxt + timedelta(hours=1)


def watch_bases() -> list[str]:
    return [t.upper() for t in read_tickers_file(WATCHLIST_FILE)]


def coin_history(conn, bases: list[str], models: list[dict]) -> None:
    """Фон: годинна історія Coinalyze нових тікерів (графік 1h, частка OI бірж) і годинна оцінка з неї."""
    if BG is None:
        return
    collect(BG.client, conn, bases, FETCH_DAYS, verbose=False)
    have = {r[0] for r in conn.execute("""SELECT DISTINCT m.base_asset FROM ohlcv o JOIN markets m USING (symbol)
                                          WHERE o.interval = '1hour' AND m.base_asset = ANY(%s)""", (bases,))}
    if have:
        score_bases(conn, sorted(have), models)


def cycle_bases() -> list[str]:
    """Твій watchlist.txt + авто-список сканера (reports/auto_watchlist.json), без повторів."""
    from fetch import excluded_tickers
    excluded = excluded_tickers()
    manual = [b for b in watch_bases() if b not in excluded]
    return manual + sorted(b for b in scanner.active_auto() if b not in set(manual))


def main() -> None:
    parser = argparse.ArgumentParser(description="Скринер пампів: 15m-цикл і годинний")
    parser.add_argument("--loop", action="store_true",
                        help="працювати безперервно: кожні 15 хв (HH:02/17/32/47 UTC) усі тікери, між ними кожні 5 хв — "
                             "лише «гарячі» (ковзні 15m-бари); годинні моделі — раз на годину")
    parser.add_argument("--no-fetch", action="store_true", help="не збирати дані, оцінити наявні")
    parser.add_argument("--no-scan", action="store_true", help="не запускати сканер ринку (лише watchlist.txt і вже наявний авто-список)")
    args = parser.parse_args()

    load_dotenv()
    api_key, db_url = os.getenv("COINALYZE_API_KEY"), os.getenv("DATABASE_URL")
    if not db_url or (not api_key and not args.no_fetch):
        sys.exit("Заповни COINALYZE_API_KEY і DATABASE_URL у файлі .env")
    if not watch_bases() and not scanner.load_config()["enabled"]:
        sys.exit(f"Додай тікери в {WATCHLIST_FILE} (один на рядок) або ввімкни сканер ринку")

    models, m15 = load_models(), load_model_15m()
    for m in models + ([m15] if m15 else []):
        print(f"Модель {m['name']}: навчена {m['trained_at']}, поріг {m['threshold']:.3f}, "
              f"{m['pump_intervals']} пампів на {len(m['tickers'])} тікерах")
    if m15 is None:
        print("15m-моделі немає (python train.py) — працює лише годинний контур")
    client = None if args.no_fetch else Coinalyze(api_key)
    global BG
    if args.loop and client is not None:
        BG = Background(db_url, client)

    last_hour, last_bg, last_full15 = None, None, None
    while True:
        hour = pd.Timestamp(datetime.now(timezone.utc)).floor("h")
        hourly = hour != last_hour          # перший цикл нової години (або перший запуск) — ще й годинний
        bar15 = last_closed_15m()
        # новий 15m-бар ще не оцінено — повний цикл (навіть якщо попередній затягнувся й «проспав» :17/:32/:47);
        # інакше — швидкий 5-хв цикл лише для гарячих тікерів
        if args.loop and m15 is not None and not hourly and bar15 == last_full15:
            try:
                with psycopg.connect(db_url) as conn:
                    quick_cycle(conn, client, cycle_bases(), m15, not args.no_fetch)
            except Exception as e:
                print(f"  ! 5-хв цикл: {e}")
            now = datetime.now(timezone.utc)
            time.sleep(max(1.0, (next_run(now, RUN_MINUTES_5) - now).total_seconds()))
            continue
        last_full15 = bar15
        bases = cycle_bases()               # watchlist перечитується щоциклу: новий тікер — без перезапуску
        try:
            with psycopg.connect(db_url) as conn:
                run_cycle(conn, bases, models, not args.no_fetch, client, m15, hourly)
                if hourly:
                    last_hour = hour
                # сканер ринку — ПІСЛЯ оцінки (сигнали по списку не чекають ~50 с скану); нові тікери — одразу в роботу
                if not args.no_fetch and not args.no_scan and m15 is not None:
                    try:
                        before = set(bases)
                        scanner.apply(scanner.scan(), set(watch_bases()))
                        new = [b for b in cycle_bases() if b not in before]
                        if new:
                            onboard(conn, client, new, models, m15, all15=True)
                    except Exception as e:
                        print(f"  ! сканер ринку: {e}")
                # тікери навчання, яких немає в списку: 15m/5m з бірж (включно з OKX) і 15m-ліквідації — щоб
                # історія накопичувалась (OKX тримає OI лише ~14 днів, Coinalyze 15m-ліквідації — 1–2 міс.)
                # і згодом було на чому навчити модель з OKX. Моделлю не оцінюються.
                now_bg = datetime.now(timezone.utc)
                if not args.no_fetch and (last_bg is None or now_bg - last_bg >= timedelta(hours=BACKGROUND_HOURS)):
                    bg = [t.upper() for t in read_tickers_file("tickers.txt") if t.upper() not in set(cycle_bases())]
                    try:
                        if bg and BG is not None:
                            BG.submit("історія тікерів навчання", lambda c, b: update_15m(c, BG.client, b, log=lambda m: None), bg)
                        elif bg:
                            update_15m(conn, client, bg, log=lambda m: None)
                            print(f"  історія бірж для тікерів навчання поза списком: {len(bg)}")
                        last_bg = now_bg
                    except Exception as e:
                        print(f"  ! історія тікерів навчання: {e}")
        except Exception as e:  # у режимі --loop одна невдала година не має зупиняти скринер
            write_status(ok=False, error=f"{type(e).__name__}: {e}")
            if not args.loop:
                raise
            print(f"  ! помилка циклу: {e}")
        if not args.loop:
            break
        now = datetime.now(timezone.utc)
        nxt = next_run(now, RUN_MINUTES_5) if m15 is not None else next_hourly(now)
        print(f"  наступний цикл о {nxt:%H:%M} UTC")
        time.sleep((nxt - now).total_seconds())


if __name__ == "__main__":
    main()
