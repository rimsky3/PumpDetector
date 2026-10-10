"""
experiments/label_teacher.py — РОЗМІТНИК-«ВЧИТЕЛЬ»: чи можна ВИВЧИТИ вибірковість власника (2026-10-10, власник: «має бути метод
робити розмітку автоматично»; «давай»). Авторозмітка за правилом (рух ≥ 8 ATR і ≥ 1.5 год) знаходить імпульси власника, але
називає імпульсом утричі більше рухів — модель на ній вчиться «будь-якій волатильності» (auto_walkforward: гірше за base).
Ідея: розмітка робиться ЗАДНІМ ЧИСЛОМ, тож розмітнику можна дивитись у майбутнє. «Вчитель» бачить рух цілком (старт → кінець
і після) і вчиться на рішеннях власника «імпульс / ні»; торгова модель («учень») лишається причинною і вчиться на його розмітці
всіх монет (наступний етап — лише якщо цей пройде).
Етап 1 — таблиця подій на тікерах власника: кандидати = події авторозмітки (auto_label.auto_events, тригер T1) будь-якого
класу; опис заднім числом (OPIS, зафіксовано ДО запуску): розмір і тривалість руху, час до піку, OI за рух і після піку, обсяг,
плавність (Кауфман), найглибший відкат усередині, скільки віддано за 6 / 24 год після піку, що було до старту (рух за добу,
діапазон 96 год), власний рух проти ринку (ALT-EW з бази), волатильність і оборот монети, кількість тригерів.
Мітка: подія перетинає імпульс власника → 1; перетинає лише фальстарт або на ПЕРЕГЛЯНУТОМУ тікері не перетинає нічого → 0;
на непереглянутих тікерах без розмітки — невідомо (не в навчанні).
Етап 2 — leave-ticker-out вчителя (LightGBM train.make_lightgbm): на відкладеному тікері порівнюємо з розміткою власника.
    знайдено   імпульс власника перетинає хоч одну подію з P ≥ порогу (у правила — подію класу «pump»)
    зайві      події-«імпульси» на переглянутих тікерах, що не перетинають жодного імпульсу власника (з них фальстарти окремо)
КРИТЕРІЙ (до запуску): на порозі, де вчитель знаходить НЕ МЕНШЕ імпульсів власника, ніж правило, зайвих у нього має бути
≤ ½ від правила. Інакше вибірковість власника з цих описів не вивчається — на всі монети не масштабуємо.
→ reports/label_teacher.md, reports/label_teacher_events.csv (усі події з P вчителя),
  reports/label_teacher_review.csv (спірні для власника: без розмітки, а вчитель певен, що імпульс; і навпаки)

    python experiments/label_teacher.py
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import psycopg
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score

import auto_label as AL
import train
from features import INTERVAL_15M, alt_index_bars, load_alt_index, load_pumps, load_raw
from fetch import excluded_tickers, read_tickers_file

B = AL.B
TRIG = AL.TRIGGERS["T1"]
OPIS = ["move_atr", "move_pct", "dur_h", "peak_h", "peak_frac", "trig_delay_h", "oi_move", "oi_after_peak", "oi_after_6h",
        "vol_x_move", "vol_peak_x", "eff", "dd_in", "giveback_6h", "giveback_24h", "pre_ret_24h", "pre_range_96h", "pre_vol_x",
        "idio_share", "mkt_ret", "natr", "log_turnover", "n_trig", "green_frac"]
OUT = os.path.join("reports", "label_teacher.md")


def describe(h: pd.DataFrame, ev, mkt: pd.Series, trig: np.ndarray) -> dict:
    """Опис події заднім числом: від старту до кінця і після (h — 15m-ряд auto_label.series)."""
    idx = h.index
    s, e = idx.get_indexer([ev.start])[0], idx.searchsorted(ev.end_ts) - 1
    if s < 0 or e <= s:
        return {}
    c, hi, lo, v, oi = (h[k].to_numpy() for k in ("price", "high", "low", "volume", "oi"))
    a0 = h["atr"].to_numpy()[max(s - 1, 0)]
    vm = h["vol_med"].to_numpy()[s]
    p0 = lo[s]
    pk = s + int(np.nanargmax(hi[s:e + 1]))
    peak = hi[pk]
    move = peak - p0
    if not (move > 0 and a0 > 0):
        return {}
    n = len(c)
    seg = c[s:pk + 1]
    path = np.nansum(np.abs(np.diff(seg))) if len(seg) > 1 else np.nan
    run = np.maximum.accumulate(hi[s:pk + 1])
    after = lambda k: np.nanmin(lo[pk + 1:min(n, pk + 1 + k)]) if pk + 1 < n else np.nan          # noqa: E731
    lr = np.log(c[pk] / c[max(s - 1, 0)]) if c[max(s - 1, 0)] > 0 else np.nan
    m = float(np.nansum(mkt.to_numpy()[s:pk + 1]))
    out = {
        "move_atr": move / a0, "move_pct": peak / p0 - 1, "dur_h": (e - s + 1) / 4, "peak_h": (pk - s) / 4,
        "peak_frac": (pk - s) / max(e - s, 1), "trig_delay_h": (idx.get_indexer([ev.trigger])[0] - s) / 4,
        "oi_move": oi[pk] / oi[s] - 1 if oi[s] > 0 else np.nan,
        "oi_after_peak": oi[e] / oi[pk] - 1 if oi[pk] > 0 else np.nan,
        "oi_after_6h": oi[min(n - 1, pk + 24)] / oi[pk] - 1 if oi[pk] > 0 else np.nan,
        "vol_x_move": np.nanmean(v[s:pk + 1]) / vm if vm > 0 else np.nan,
        "vol_peak_x": np.nanmax(v[s:pk + 1]) / vm if vm > 0 else np.nan,
        "eff": (c[pk] - c[s]) / path if path and path > 0 else np.nan,
        "dd_in": float(np.nanmax((run - lo[s:pk + 1]) / move)),
        "giveback_6h": (peak - after(24)) / move, "giveback_24h": (peak - after(96)) / move,
        "pre_ret_24h": c[s] / c[s - 96] - 1 if s >= 96 and c[s - 96] > 0 else np.nan,
        "pre_range_96h": np.log(np.nanmax(hi[s - 384:s]) / np.nanmin(lo[s - 384:s])) if s >= 384 else np.nan,
        "pre_vol_x": np.nanmean(v[s - 96:s]) / vm if s >= 96 and vm > 0 else np.nan,
        "idio_share": (lr - m) / lr if lr and lr > 0 and np.isfinite(m) else np.nan,
        "mkt_ret": m, "natr": a0 / c[s] if c[s] > 0 else np.nan, "log_turnover": np.log(vm) if vm > 0 else np.nan,
        "n_trig": int(trig[s:e + 1].sum()), "green_frac": float(np.mean(np.diff(seg) > 0)) if len(seg) > 1 else np.nan,
    }
    return out


def overlaps(ev: pd.DataFrame, own: pd.DataFrame) -> np.ndarray:
    """Для кожної події — чи перетинає хоч один інтервал own того самого тікера."""
    out = np.zeros(len(ev), dtype=bool)
    for b, g in own.groupby("base_asset"):
        m = (ev["base"] == b).to_numpy()
        if not m.any():
            continue
        st, en = ev.loc[m, "start"].to_numpy()[:, None], ev.loc[m, "end_ts"].to_numpy()[:, None]
        out[m] = ((st < g["end_ts"].to_numpy()[None, :]) & (en > g["start_ts"].to_numpy()[None, :])).any(axis=1)
    return out


def score(ev: pd.DataFrame, pos: np.ndarray, own_p: pd.DataFrame, reviewed: set) -> dict:
    """Знайдено імпульсів власника і зайві події-«імпульси» на переглянутих тікерах для набору pos."""
    imp = ev[pos]
    found = 0
    for r in own_p.itertuples():
        x = imp[(imp["base"] == r.base_asset) & (imp["start"] < r.end_ts) & (imp["end_ts"] > r.start_ts)]
        found += bool(len(x))
    rv = imp[imp["base"].isin(reviewed)]
    return {"found": found, "extra": int((~rv["in_pump"]).sum()), "fs": int((rv["in_fs"] & ~rv["in_pump"]).sum()), "n": int(pos.sum())}


def owner_events():
    """Події на тікерах власника з описом і мітками → (k — події з відомою міткою, own_p, own_f, reviewed, days)."""
    load_dotenv(".env")
    excl = excluded_tickers(".")
    reviewed = {t.upper() for t in read_tickers_file("tickers.txt")} - excl
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        pumps = load_pumps(conn)
        pumps = pumps[~pumps["base_asset"].isin(excl)]
        bases = sorted(reviewed | set(pumps["base_asset"]))
        raw = load_raw(conn, bases, INTERVAL_15M, exchanges=["Binance"])
        idx5 = load_alt_index(conn)
    own_p = pumps[pumps["pattern_type"] == "pump"]
    own_f = pumps[pumps["pattern_type"] == "false_start"]
    rows, days = [], 0.0
    for b in bases:
        h = AL.series(raw, b)
        if len(h) <= 2 * AL.DAY:
            continue
        if b in reviewed:
            days += len(h) / AL.DAY
        rows += [{"base": b, **r} for r in event_rows(h, alt_index_bars(idx5, h.index, B))]
    ev = pd.DataFrame(rows)
    ev["in_pump"], ev["in_fs"] = overlaps(ev, own_p), overlaps(ev, own_f)
    ev["known"] = ev["in_pump"] | ev["in_fs"] | ev["base"].isin(reviewed)
    ev["y"] = ev["in_pump"].astype(int)
    return ev[ev["known"]].reset_index(drop=True), own_p, own_f, reviewed, days


def event_rows(h: pd.DataFrame, mkt: pd.Series) -> list[dict]:
    """Події авторозмітки ряду h (будь-якого класу) з описом заднім числом."""
    e = AL.auto_events(h, TRIG)
    if e.empty:
        return []
    trig = AL.trigger_mask(h, *TRIG)
    out = []
    for r in e.itertuples():
        d = describe(h, r, mkt, trig)
        if d:
            out.append({"start": r.start, "end_ts": r.end_ts, "trigger": r.trigger, "cls": r.cls, **d})
    return out


def fit_teacher(k: pd.DataFrame):
    """Вчитель на всіх подіях з відомою міткою."""
    return train.make_lightgbm().fit(k[OPIS], k["y"])


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    argparse.ArgumentParser(description="Розмітник-«вчитель» із поглядом заднім числом: чи вивчається вибірковість власника (LTO подій)").parse_args()
    k, own_p, own_f, reviewed, days = owner_events()
    print(f"подій з відомою міткою {len(k)}: імпульс власника {k['y'].sum()}, ні {len(k) - k['y'].sum()} "
          f"(з них фальстартів {(k['in_fs'] & ~k['in_pump']).sum()}); тікерів {k['base'].nunique()}", flush=True)

    p = train.oof_proba(train.make_lightgbm, k[OPIS], k["y"], k["base"])
    k["p"] = p
    auc = roc_auc_score(k["y"], p)
    rule = score(k, (k["cls"] == "pump").to_numpy(), own_p, reviewed)
    grid = np.unique(np.round(p, 3))[::-1]
    curve = [(t, score(k, p >= t, own_p, reviewed)) for t in grid]
    same = next(((t, s) for t, s in curve if s["found"] >= rule["found"]), curve[-1])
    model = fit_teacher(k)
    gain = pd.Series(model.booster_.feature_importance("gain"), index=OPIS)
    imp = (gain / gain.sum()).sort_values(ascending=False)
    ok = same[1]["found"] >= rule["found"] and same[1]["extra"] <= 0.5 * rule["extra"]

    k.to_csv(OUT.replace(".md", "_events.csv"), index=False)
    rv = k[k["base"].isin(reviewed)]
    cand = pd.concat([rv[~rv["in_pump"] & ~rv["in_fs"]].nlargest(40, "p").assign(why="без розмітки, вчитель: імпульс"),
                      rv[rv["in_pump"]].nsmallest(15, "p").assign(why="імпульс власника, вчитель: ні")])
    cand[["why", "base", "start", "end_ts", "p", "cls", "move_pct", "move_atr", "dur_h", "oi_move", "giveback_24h"]].to_csv(
        OUT.replace(".md", "_review.csv"), index=False)

    n_p, n_f = len(own_p), len(own_f)
    L = ["# Розмітник-«вчитель»: чи вивчається вибірковість власника", "",
         f"Події — тригер T1 авторозмітки на тікерах власника: {len(k)} з відомою міткою ({k['y'].sum()} перетинають імпульс власника, "
         f"{len(k) - k['y'].sum()} — ні), {k['base'].nunique()} тікерів, переглянутих — {days:.0f} тікеро-днів. Вчитель — LightGBM на "
         f"{len(OPIS)} описах заднім числом, leave-ticker-out. AUC на відкладених тікерах **{auc:.3f}**.", "",
         "**Критерій (до запуску):** при не меншій кількості знайдених імпульсів власника, ніж у правила, зайвих ≤ ½ від правила.", "",
         "## Вчитель проти правила (відкладені тікери)", "",
         "| розмітник | подій-«імпульсів» | знайдено імпульсів власника | зайвих (переглянуті тікери) | зайвих / 30 дн | з них фальстартів власника |",
         "|---|---|---|---|---|---|",
         f"| правило (рух ≥ 8 ATR і ≥ 1.5 год) | {rule['n']} | {rule['found']}/{n_p} | {rule['extra']} | {rule['extra'] / days * 30:.1f} | {rule['fs']}/{n_f} |",
         f"| вчитель, поріг {same[0]:.3f} (стільки ж знайдено) | {same[1]['n']} | {same[1]['found']}/{n_p} | {same[1]['extra']} | "
         f"{same[1]['extra'] / days * 30:.1f} | {same[1]['fs']}/{n_f} |"]
    L += ["", "## Крива вчителя: скільки зайвих за різної повноти", "", "| знайдено імпульсів власника | поріг | подій | зайвих | зайвих / 30 дн | фальстартів |",
          "|---|---|---|---|---|---|"]
    seen = set()
    for target in (rule["found"], 100, 90, 80, 70, 60):
        hit = next(((t, s) for t, s in curve if s["found"] >= target), None)
        if hit and hit[0] not in seen:
            seen.add(hit[0])
            L.append(f"| ≥ {target} ({hit[1]['found']}) | {hit[0]:.3f} | {hit[1]['n']} | {hit[1]['extra']} | {hit[1]['extra'] / days * 30:.1f} | {hit[1]['fs']} |")
    L += ["", "## На що дивиться вчитель (частка gain)", "", ", ".join(f"{f} {imp[f]:.1%}" for f in imp.index[:12]) + ".", "",
          "## Критерій", "",
          f"- правило: знайдено {rule['found']}, зайвих {rule['extra']}; вчитель: знайдено {same[1]['found']}, зайвих {same[1]['extra']} "
          f"({same[1]['extra'] / max(rule['extra'], 1):.0%} від правила) → **{'ВИКОНАНО' if ok else 'НЕ виконано'}**", "",
          f"Спірні події для перегляду власником — reports/label_teacher_review.csv ({len(cand)} рядків)."]
    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
