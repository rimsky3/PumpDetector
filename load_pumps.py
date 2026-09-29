"""
load_pumps.py — завантаження ручної розмітки з pumps.csv у таблицю pumps.

pumps.csv — джерело істини: таблиця щоразу повністю перезаписується з файлу,
тож виправлення робляться у CSV (або через label.py), а не в базі.

Формат: base_asset,start_ts,end_ts,pattern_type,notes
Часи — UTC, "YYYY-MM-DD HH:MM", час відкриття бару; end_ts включно (останній бар пампу).

    python load_pumps.py
"""

import argparse
import csv
import os
import sys
from datetime import datetime, timezone

import psycopg
from dotenv import load_dotenv

PUMPS_FILE = "pumps.csv"
PATTERN_TYPES = {"pump", "false_start", "other"}
COLUMNS = ["base_asset", "start_ts", "end_ts", "pattern_type", "notes"]
TS_FORMAT = "%Y-%m-%d %H:%M"


def parse_ts(s: str) -> datetime:
    return datetime.strptime(s.strip(), TS_FORMAT).replace(tzinfo=timezone.utc)


def parse_row(r: dict) -> tuple:
    """Рядок CSV/JSON → (base, start, end, type, notes). ValueError, якщо щось не так."""
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
    if end < start:
        raise ValueError("end_ts раніше за start_ts")
    return base, start, end, ptype, (r.get("notes") or "").strip() or None


def check_overlaps(rows: list[tuple]) -> list[str]:
    """Інтервали одного тікера не мають перетинатися."""
    errors, by_base = [], {}
    for base, start, end, *_ in rows:
        by_base.setdefault(base, []).append((start, end))
    for base, spans in by_base.items():
        spans.sort()
        for (s1, e1), (s2, e2) in zip(spans, spans[1:]):
            if s2 <= e1:
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


def write_pumps(rows: list[tuple], path: str = PUMPS_FILE) -> None:
    rows = sorted(rows, key=lambda r: (r[0], r[1]))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(COLUMNS)
        for base, start, end, ptype, notes in rows:
            w.writerow([base, start.strftime(TS_FORMAT), end.strftime(TS_FORMAT), ptype, notes or ""])
    os.replace(tmp, path)  # атомарна заміна: файл не лишиться напівзаписаним


def sync_db(conn, rows: list[tuple]) -> None:
    # end_ts у CSV — останній бар включно, у таблиці — межа інтервалу (бар + 1 год),
    # щоб виконувалась умова end_ts > start_ts і для однобарних пампів
    conn.execute("TRUNCATE pumps RESTART IDENTITY")
    conn.cursor().executemany(
        "INSERT INTO pumps (base_asset, start_ts, end_ts, pattern_type, notes) "
        "VALUES (%s, %s, %s + interval '1 hour', %s, %s)",
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
    for *_, ptype, _ in rows:
        counts[ptype] = counts.get(ptype, 0) + 1
    print(f"Завантажено {len(rows)} інтервалів: " + ", ".join(f"{k} — {v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
