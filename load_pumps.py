"""
load_pumps.py — завантаження ручної розмітки з pumps.csv у таблицю pumps.

pumps.csv — джерело істини: таблиця щоразу повністю перезаписується з файлу,
тож виправлення робляться у CSV (або через label.py), а не в базі.

Формат: base_asset,start_ts,end_ts,pattern_type,notes[,entry_ts,entry_price,stop_price,setup]
Часи — UTC, "YYYY-MM-DD HH:MM", з точністю до хвилини: start_ts — момент початку пампу, end_ts — момент кінця
(виключна межа). Межі не прив'язані до таймфрейму: можна ставити на 5m, 15m чи 1h. Який бар якого таймфрейму
вважається пампом, вирішує features.in_pump (памп триває на момент закриття бару).
До 2026-09-29 end_ts був «останнім баром включно» на 1h; тодішні рядки перераховано (+1 год).
Необов'язково (з 2026-09-30): entry_ts, entry_price, stop_price — де власник зайшов би в памп і де поставив би стоп.
З них рахується «k власника» = (вхід − стоп) / ATR(15m) на момент входу — порівнюється зі стопом моделі.
Необов'язково (з 2026-10-01): setup — який це сетап імпульсу: breakout (вихід з рейнджу), pullback (підбір після
відкату, OI під час відкату не обвалився), other. Модель на ньому не вчиться (ціль та сама), він потрібен, щоб
рахувати метрики окремо для кожного сетапу.

    python load_pumps.py
"""

import argparse
import csv
import os
import sys
from datetime import datetime, timedelta, timezone

import psycopg
from dotenv import load_dotenv

# Абсолютний шлях (2026-10-11): 11.10 о 02:59 під час розмітки CHIP у label.py pumps.csv перезаписався лише 6 новими рядками
# (153 інших зникли; відновлено з git 319ef94). Ймовірна причина — відносний шлях: якщо файл «не знайшовся», read_pumps
# повертав [], і label.py записував розмітку заново з одного рядка. Тепер файл шукається поруч зі скриптом, а write_pumps
# робить резервну копію і відмовляється зменшувати розмітку більш ніж на 1 рядок за раз.
ROOT = os.path.dirname(os.path.abspath(__file__))
PUMPS_FILE = os.path.join(ROOT, "pumps.csv")
BACKUP_DIR = os.path.join(ROOT, "data", "backups")
BACKUP_KEEP = 50
PATTERN_TYPES = {"pump", "false_start", "other"}
SETUPS = {"breakout", "pullback", "other"}
COLUMNS = ["base_asset", "start_ts", "end_ts", "pattern_type", "notes", "entry_ts", "entry_price", "stop_price", "setup"]
TS_FORMAT = "%Y-%m-%d %H:%M"


def parse_ts(s: str) -> datetime:
    return datetime.strptime(s.strip(), TS_FORMAT).replace(tzinfo=timezone.utc)


def _opt_float(v):
    v = (str(v) if v is not None else "").strip()
    return float(v) if v else None


def parse_row(r: dict) -> tuple:
    """Рядок CSV/JSON → (base, start, end, type, notes, entry_ts, entry_price, stop_price, setup). ValueError, якщо щось
    не так. Вхід, стоп і сетап — необов'язкові (None)."""
    try:
        base = r["base_asset"].strip().upper()
        start, end = parse_ts(r["start_ts"]), parse_ts(r["end_ts"])
        ptype = r["pattern_type"].strip()
    except KeyError as e:
        raise ValueError(f"немає поля {e}") from None
    if not base:
        raise ValueError("порожній base_asset")
    if ptype not in PATTERN_TYPES:
        raise ValueError(f"невідомий pattern_type '{ptype}'")
    if end <= start:
        raise ValueError("end_ts має бути пізніше за start_ts")
    entry_ts = parse_ts(r["entry_ts"]) if (r.get("entry_ts") or "").strip() else None
    try:
        entry_px, stop_px = _opt_float(r.get("entry_price")), _opt_float(r.get("stop_price"))
    except ValueError:
        raise ValueError("entry_price / stop_price мають бути числами") from None
    if (entry_ts is None) != (entry_px is None) or (stop_px is not None and entry_px is None):
        raise ValueError("вхід задається часом і ціною разом, стоп — лише разом із входом")
    if entry_ts is not None and not (start - timedelta(days=1) <= entry_ts < end):
        raise ValueError("вхід має бути в межах пампу")
    if stop_px is not None and stop_px >= entry_px:
        raise ValueError("стоп для лонгу має бути нижче ціни входу")
    setup = (r.get("setup") or "").strip() or None
    if setup is not None and setup not in SETUPS:
        raise ValueError(f"невідомий setup '{setup}' (breakout / pullback / other)")
    return base, start, end, ptype, (r.get("notes") or "").strip() or None, entry_ts, entry_px, stop_px, setup


def check_overlaps(rows: list[tuple]) -> list[str]:
    """Інтервали одного тікера не мають перетинатися."""
    errors, by_base = [], {}
    for base, start, end, *_ in rows:
        by_base.setdefault(base, []).append((start, end))
    for base, spans in by_base.items():
        spans.sort()
        for (s1, e1), (s2, e2) in zip(spans, spans[1:]):
            if s2 < e1:                      # кінець виключний: інтервал може починатися там, де закінчився попередній
                errors.append(f"{base}: інтервали перетинаються "
                              f"({s1:%m-%d %H:%M}–{e1:%m-%d %H:%M} і {s2:%m-%d %H:%M}–{e2:%m-%d %H:%M})")
    return errors


def read_pumps(path: str = PUMPS_FILE) -> list[tuple]:
    if not os.path.exists(path):
        return []
    rows, errors = [], []
    with open(path, encoding="utf-8", newline="") as f:
        for n, r in enumerate(csv.DictReader(f), start=2):
            try:
                rows.append(parse_row(r))
            except ValueError as e:
                errors.append(f"рядок {n}: {e}")
    errors += check_overlaps(rows)
    if errors:
        raise ValueError(f"Помилки в {path}:\n  " + "\n  ".join(errors))
    return rows


def _count_rows(path: str) -> int:
    with open(path, encoding="utf-8", newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def _latest_backup() -> str | None:
    if not os.path.isdir(BACKUP_DIR):
        return None
    files = sorted(f for f in os.listdir(BACKUP_DIR) if f.startswith("pumps-") and f.endswith(".csv"))
    return os.path.join(BACKUP_DIR, files[-1]) if files else None


def backup_pumps(path: str = PUMPS_FILE) -> str | None:
    """Копія поточного pumps.csv у data/backups/pumps-<час UTC>.csv; лишаються останні BACKUP_KEEP."""
    if not os.path.exists(path):
        return None
    os.makedirs(BACKUP_DIR, exist_ok=True)
    dst = os.path.join(BACKUP_DIR, f"pumps-{datetime.now(timezone.utc):%Y%m%d-%H%M%S-%f}.csv")
    with open(path, "rb") as src, open(dst, "wb") as out:
        out.write(src.read())
    old = sorted(f for f in os.listdir(BACKUP_DIR) if f.startswith("pumps-") and f.endswith(".csv"))
    for f in old[:-BACKUP_KEEP]:
        os.remove(os.path.join(BACKUP_DIR, f))
    return dst


def write_pumps(rows: list[tuple], path: str = PUMPS_FILE, allow_shrink: bool = False) -> None:
    """Записати розмітку. Запобіжник: розмітка не може зменшитись більш ніж на 1 рядок за один запис (одне видалення
    в label.py — рівно мінус один) — ні проти поточного файлу, ні (якщо файлу немає) проти останньої резервної копії.
    allow_shrink=True — свідоме масове видалення."""
    if not allow_shrink:
        ref = path if os.path.exists(path) else _latest_backup()
        if ref is not None:
            before = _count_rows(ref)
            if len(rows) < before - 1:
                raise ValueError(f"запис скасовано: розмітка зменшилась би з {before} до {len(rows)} рядків "
                                 f"({'поточний файл' if ref == path else 'остання резервна копія ' + os.path.basename(ref)}). "
                                 "Перевір pumps.csv; резервні копії — data/backups/.")
    backup_pumps(path)
    rows = sorted(rows, key=lambda r: (r[0], r[1]))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(COLUMNS)
        for base, start, end, ptype, notes, entry_ts, entry_px, stop_px, setup in rows:
            w.writerow([base, start.strftime(TS_FORMAT), end.strftime(TS_FORMAT), ptype, notes or "",
                        entry_ts.strftime(TS_FORMAT) if entry_ts else "",
                        repr(entry_px) if entry_px is not None else "", repr(stop_px) if stop_px is not None else "",
                        setup or ""])
    os.replace(tmp, path)  # атомарна заміна: файл не лишиться напівзаписаним


def sync_db(conn, rows: list[tuple]) -> None:
    # і в CSV, і в таблиці end_ts — виключна межа (момент кінця пампу)
    for col, typ in (("entry_ts", "TIMESTAMPTZ"), ("entry_price", "DOUBLE PRECISION"), ("stop_price", "DOUBLE PRECISION"),
                     ("setup", "TEXT")):
        conn.execute(f"ALTER TABLE pumps ADD COLUMN IF NOT EXISTS {col} {typ}")   # база, створена до 2026-09-30
    conn.execute("TRUNCATE pumps RESTART IDENTITY")
    conn.cursor().executemany(
        "INSERT INTO pumps (base_asset, start_ts, end_ts, pattern_type, notes, entry_ts, entry_price, stop_price, setup) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
        rows,
    )
    conn.commit()


def main() -> None:
    argparse.ArgumentParser(description="Перевірити pumps.csv і перезаписати з нього таблицю pumps").parse_args()
    load_dotenv()
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        sys.exit("Заповни DATABASE_URL у файлі .env")

    try:
        rows = read_pumps()
    except ValueError as e:
        sys.exit(str(e))
    with psycopg.connect(db_url) as conn:
        sync_db(conn, rows)

    counts = {}
    for ptype in (r[3] for r in rows):
        counts[ptype] = counts.get(ptype, 0) + 1
    print(f"Завантажено {len(rows)} інтервалів: " + ", ".join(f"{k} — {v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
