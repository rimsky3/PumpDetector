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
from dotenv import load_dotenv

import fetch_exchange
import scanner
from features import build, build_15m
from fetch import Coinalyze, collect, collect_liquidations, read_tickers_file
from train import M15_NAME, PROBA_MODELS, load_model, model_path

WATCHLIST_FILE = "watchlist.txt"
STATUS_FILE = os.path.join("reports", "screener_status.json")   # читає дашборд, блок «Стан системи»
FETCH_DAYS = 10          # 7 днів базового вікна + доба + запас
BACKFILL_HOURS = 7 * 24  # скільки годин історії оцінювати, якщо їх ще немає в signals
RUN_MINUTE = 5           # (застаріле) хвилина годинного циклу — лишено для сумісності
RUN_MINUTES = (2, 17, 32, 47)   # 15m-цикл: бар закрився ~2 хв тому, біржі й Coinalyze його вже віддають
BAR_15M = pd.Timedelta(minutes=15)
BACKFILL_15M = 7 * 24 * 4       # скільки 15m-барів історії оцінювати, якщо їх ще немає в signals
INTRADAY_DAYS = 10              # новому тікеру — 10 днів 15m-історії (тижневі вікна ознак + запас)
INTRADAY_TAIL_DAYS = 0.25       # мінімальний хвіст — 6 годин (upsert, дублікатів немає)
MAX_GAP_DAYS = 29               # найдовший пропуск, який дозаповнюємо (історія OI Binance — 30 днів)
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


def score(df: pd.DataFrame, models: list[dict]) -> pd.DataFrame:
    """Ймовірності всіх моделей для кожного рядка ознак."""
    out = []
    for m in models:
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


def write_status(**fields) -> None:
    os.makedirs(os.path.dirname(STATUS_FILE), exist_ok=True)
    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump({"updated": datetime.now(timezone.utc).isoformat(timespec="seconds"), **fields}, f, ensure_ascii=False)


def onboard(conn, client: Coinalyze, bases: list[str], models: list[dict], m15: dict | None, all15: bool = False) -> list[str]:
    """Нові тікери (сканер, «＋ у watchlist»): годинні дані Coinalyze (заодно заводять біржові контракти в markets —
    без них 15m-дані з бірж не завантажаться), годинна оцінка, 15m-дані й 15m-оцінка — одразу, не чекаючи
    годинної частини циклу, інакше до кінця години є лише «Кандидат» без Alert.
    «Новий» = годинних даних немає або вони старші за 2 год (напр., тікер колись вантажили в розмітці)."""
    cutoff = pd.Timestamp(datetime.now(timezone.utc)).floor("h") - pd.Timedelta(hours=2)
    sql = """SELECT m.base_asset FROM ohlcv o JOIN markets m USING (symbol)
             WHERE o.interval = '1hour' AND m.base_asset = ANY(%s) GROUP BY 1 HAVING max(o.ts) >= %s"""
    have = {r[0] for r in conn.execute(sql, (bases, cutoff))}
    fresh = [b for b in bases if b not in have and b not in NO_COINALYZE]
    if all15 and m15 is not None and not fresh:      # від сканера: годинні дані свіжі, але 15m ще не рахували
        update_15m(conn, client, bases)
        score_15m(conn, bases, m15)
    if not fresh:
        return []
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
        print(f"  нові тікери: дані й оцінки — {', '.join(fresh)}")
    return fresh


def run_cycle(conn, bases: list[str], models: list[dict], fetch: bool, client: Coinalyze | None,
              m15: dict | None = None, hourly: bool = True) -> None:
    started = datetime.now(timezone.utc)
    calls0, limited0 = (client.calls_made, client.rate_limited) if client else (0, 0)
    print(f"\n[{started:%Y-%m-%d %H:%M} UTC] цикл: {len(bases)} тікерів" + ("" if hourly else " (15m)"))

    info15, hot = {}, []
    if m15 is not None:
        if fetch:
            update_15m(conn, client, bases)
        s15, last15 = score_15m(conn, bases, m15)
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
    fresh = onboard(conn, client, bases, models, m15) if fetch and client is not None else []
    if not hourly:
        prev = screener_status_file()
        write_status(**{**prev, **info15, "ok": True, "cycle_15m_at": started.isoformat(timespec="seconds"),
                        "api_calls_15m": (client.calls_made - calls0) if client else 0})
        return

    if fetch:
        # зазвичай 10 днів; після довгого простою — увесь пропуск (Coinalyze тримає ~60 днів; ліміт — за символами, не днями)
        last1h = conn.execute("""SELECT min(t) FROM (SELECT m.base_asset, max(o.ts) AS t FROM ohlcv o JOIN markets m USING (symbol)
                                 WHERE o.interval = '1hour' AND m.base_asset = ANY(%s) GROUP BY 1) x""", (bases,)).fetchone()[0]
        gap = (pd.Timestamp(started) - pd.Timestamp(last1h)) / pd.Timedelta(days=1) if last1h else 0
        days = int(min(60, max(FETCH_DAYS, np.ceil(gap) + 1)))
        # спершу тікери, де горить 15m-«Кандидат»: їм годинна оцінка потрібна для Alert — вона буде за ~1–2 хв
        # після старту циклу, а не після збору всього списку (Coinalyze: ~23 виклики на тікер, 40 на хвилину)
        first = [b for b in hot if b not in fresh]
        if first:
            collect(client, conn, first, days, verbose=False)
            score_bases(conn, first, models)
            print(f"  годинна оцінка першими для гарячих: {', '.join(first)}")
        rest = [b for b in bases if b not in fresh and b not in first]     # нові вже зібрано в onboard()
        if rest:
            collect(client, conn, rest, days, verbose=False)
        print(f"  дані оновлено за {(datetime.now(timezone.utc) - started).seconds} с")
    scored, last_closed = score_bases(conn, bases, models)
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
                 api_calls=(client.calls_made - calls0) if client else 0,
                 api_429=(client.rate_limited - limited0) if client else 0, fetched=fetch,
                 cycle_15m_at=started.isoformat(timespec="seconds") if m15 is not None else None, **info15)


def screener_status_file() -> dict:
    try:
        with open(STATUS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def next_run(now: datetime) -> datetime:
    """Найближчий момент циклу: HH:02, :17, :32, :47."""
    base = now.replace(second=0, microsecond=0)
    for m in RUN_MINUTES:
        t = base.replace(minute=m)
        if t > now:
            return t
    return base.replace(minute=RUN_MINUTES[0]) + timedelta(hours=1)


def next_hourly(now: datetime) -> datetime:
    nxt = now.replace(minute=RUN_MINUTE, second=0, microsecond=0)
    return nxt if nxt > now else nxt + timedelta(hours=1)


def watch_bases() -> list[str]:
    return [t.upper() for t in read_tickers_file(WATCHLIST_FILE)]


def cycle_bases() -> list[str]:
    """Твій watchlist.txt + авто-список сканера (reports/auto_watchlist.json), без повторів."""
    manual = watch_bases()
    return manual + sorted(b for b in scanner.active_auto() if b not in set(manual))


def main() -> None:
    parser = argparse.ArgumentParser(description="Скринер пампів: 15m-цикл і годинний")
    parser.add_argument("--loop", action="store_true",
                        help="працювати безперервно: кожні 15 хв (HH:02/17/32/47 UTC), годинні моделі — раз на годину")
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

    last_hour, last_bg = None, None
    while True:
        hour = pd.Timestamp(datetime.now(timezone.utc)).floor("h")
        hourly = hour != last_hour          # перший цикл нової години (або перший запуск) — ще й годинний
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
                        if bg:
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
        nxt = next_run(now) if m15 is not None else next_hourly(now)
        print(f"  наступний цикл о {nxt:%H:%M} UTC")
        time.sleep((nxt - now).total_seconds())


if __name__ == "__main__":
    main()
