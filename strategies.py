"""
strategies.py — режими: модель + правила угод. Один список для «Прогнозу», «Журналу» і «Бектесту» (2026-10-02, власник).

Режим не змінює сигналів моделі (Alert / Кандидат рахує скринер, однаково для всіх режимів однієї моделі) — лише те,
як сигнал стає угодою:
    model    tier15 — 15m + 1h LightGBM (робоча); lightgbm / logreg — годинні, для порівняння
    signal   alert — вхід на Alert (15m + підтвердження 1h); candidate — на Кандидаті (лише 15m, раніше, але шумніше)
    entry    market — по ринку за закриттям бару сигналу; limit — лімітна заявка −0.5 ATR, дійсна 1 год (risk.LIMIT_ENTRY)
    ob       true — сигнал береться лише при біди / аски ±2% ≥ 1 у момент сигналу (невідомо — береться)
    market_max  0.6 — сигнал береться лише, коли ≤ 60% альтів дорожчі, ніж тиждень тому (market.py; невідомо — береться)
    p_be     true — мітка p («залочка» на хаях) у відкритому лонгу → стоп у беззбиток; після такої угоди — без повторного
             входу в тому ж епізоді Alert (risk.position_state(p_marks=); experiments/p_exit.py)
    split    true — вхід лише за Alert моделі «старт» (перші 30 хв імпульсу, train.START_NAME), утримання — поки горить Alert
             основної або «старту» (tiers.split_scores, risk.position_state(entry_prob=)); 2026-10-12, experiments/purged_split.py
Вихід для 15m-режимів спільний: risk.PARTIAL_EXIT (½ на +3 ATR, далі беззбиток; решта — 4 бари без сигналу або стоп).

Список — strategies.json (новий експеримент = новий рядок; без файлу — DEFAULT). Результати режимів — notes/experiments.md.
"""

import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(ROOT, "strategies.json")

DEFAULT = [
    {"id": "market", "title": "15m + 1h · ринок", "model": "tier15", "signal": "alert", "entry": "market", "ob": False, "main": True,
     "note": "основний: вхід по ринку на Alert"},
    {"id": "limit", "title": "15m + 1h · ліміт −0.5 ATR", "model": "tier15", "signal": "alert", "entry": "limit", "ob": False,
     "note": "6 міс. бектесту PF 1.02 проти 0.96; журнал 24.09–02.10 гірше за ринок"},
    {"id": "ob", "title": "15m + 1h · OB ≥ 1", "model": "tier15", "signal": "alert", "entry": "market", "ob": True,
     "note": "вересень PF 0.96 проти 0.86, але на 6 міс. не підтвердився (3 з 7 місяців)"},
    {"id": "ob_limit", "title": "15m + 1h · OB + ліміт", "model": "tier15", "signal": "alert", "entry": "limit", "ob": True,
     "note": "вересень PF 1.08; 6 міс. — 1.00"},
    {"id": "market_cool", "title": "15m + 1h · ринок не гарячий", "model": "tier15", "signal": "alert", "entry": "market",
     "ob": False, "market_max": 0.6,
     "note": "Alert лише коли ≤ 60% альтів у плюсі за 7 дн: 6 міс. PF 1.09 проти 0.78 у гарячому ринку, 6 з 6 місяців"},
    {"id": "p_be", "title": "15m + 1h · б/з на p", "model": "tier15", "signal": "alert", "entry": "market", "ob": False,
     "p_be": True, "note": "p у позиції → стоп у б/з, без повторного входу в епізоді: 40 днів PF 0.94 проти 0.93, великих не ріже — жива перевірка"},
    {"id": "split", "title": "15m + 1h · вхід на старті", "model": "tier15", "signal": "alert", "entry": "market", "ob": False,
     "split": True, "note": "вхід лише на свіжому старті (модель «старт»), утримання — основна або «старт»: ≥ 20 млн — 6 міс. 0.84 → 0.95, purged ~70 днів 0.79 → 0.82"},
    {"id": "cand", "title": "15m · вхід на Кандидаті", "model": "tier15", "signal": "candidate", "entry": "market", "ob": False,
     "note": "раніший вхід (без підтвердження 1h), більше фальстартів"},
    {"id": "lightgbm", "title": "LightGBM 1h · порівняння", "model": "lightgbm"},
    {"id": "logreg", "title": "Регресія 1h · порівняння", "model": "logreg"},
]


def load() -> list[dict]:
    try:
        with open(PATH, encoding="utf-8") as f:
            items = json.load(f)
        return [s for s in items if s.get("id") and s.get("model")] or DEFAULT
    except (OSError, ValueError):
        return DEFAULT


def get(sid: str | None) -> dict:
    items = load()
    return next((s for s in items if s["id"] == sid), next((s for s in items if s.get("main")), items[0]))


def is15(s: dict) -> bool:
    return s.get("model") == "tier15"


def scores_name(s: dict) -> str:
    """Яку оцінку брати як «сигнал ≥ 1»: tier15 (Alert), cand15 (Кандидат) або годинну модель."""
    from tiers import CAND_MODEL, TIER_MODEL
    if not is15(s):
        return s["model"]
    return CAND_MODEL if s.get("signal") == "candidate" else TIER_MODEL


def entry_rule(s: dict) -> dict | None:
    from risk import LIMIT_ENTRY
    return LIMIT_ENTRY if is15(s) and s.get("entry") == "limit" else None


def public(s: dict) -> dict:
    return {k: s.get(k) for k in ("id", "title", "model", "signal", "entry", "ob", "market_max", "p_be", "split", "main", "note")}


def is_split(s: dict | None) -> bool:
    return bool(s and is15(s) and s.get("split"))


if __name__ == "__main__":
    import argparse
    argparse.ArgumentParser(description="Список режимів (модель + правила угод) зі strategies.json").parse_args()
    for s in load():
        print(f"{s['id']:<10} {s['title']:<32} {s.get('note', '')}")
