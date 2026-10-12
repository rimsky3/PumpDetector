"""
experiments/random_entry.py — тест значущості рандомізацією входів (2026-10-12; ідея: Chan, «Algorithmic Trading», розд. 1,
Example 1.1 — notes/books/ideas/chan_algorithmic_trading_2013.md, п. 1).

Питання: чи вхід моделі кращий за випадковий, якщо лишити ті самі монети, ті самі дні, ту саму кількість входів і той самий
вихід (½ на +3 ATR → беззбиток; 4 бари без Alert; стоп k·ATR)? Null hypothesis — «переваги у виборі моменту немає».
Два «нулі», для кожного режиму (split і market) окремо:
  N1  випадковий момент усередині Alert: для кожної реальної угоди — випадковий бар серед барів з Alert цього режиму
      тієї ж монети того ж дня UTC (для split «бар з Alert» = горить основна або «старт», для market — основна);
  N2  випадковий вхід: те саме, але серед УСІХ оцінених барів того ж дня (вихід за тим самим правилом).
Кількість входів на кожну пару (монета, день) збережена — тому «поганий день для всього ринку» однаковий у всіх світах
(програші йдуть серіями: experiments/position_sizing.py, Z −7.6; перемішування по одній угоді дало б надто красивий p-value).
Статистика — profit factor усіх закритих угод (комісія 0.1%); p-value — частка світів з PF не гіршим за справжній.

Критерій ДО запуску: PF split ≥ 95-го перцентиля N1 і PF обох режимів ≥ 95-го перцентиля N2. Не пройшло — «не відрізнити
від випадкового вибору моменту»; рішення про основний режим тоді лишається за живим порівнянням у журналі.

Дані — ті самі, що в контрольному бектесті 12.10: 87 монет ≥ 20 млн поза навчанням, локальна база (без завантажень),
вікно з 2026-09-11 22:30 UTC. Угоди рахує власний швидкий симулятор; він звіряється з risk.position_state (має збігтися).
Оцінки монет кешуються в data/random_entry/tiers.pkl. Звіт — reports/random_entry.md.

    python experiments/random_entry.py                 # ~5 хв перший раз (оцінки 87 монет), далі ~1–2 хв
    python experiments/random_entry.py --worlds 2000   # швидша прикидка
"""

import argparse
import json
import os
import pickle
import sys

import joblib
import numpy as np
import pandas as pd
import psycopg
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import backtest  # noqa: E402
import strategies  # noqa: E402
import train  # noqa: E402
from risk import PARTIAL_EXIT, atr, position_state  # noqa: E402
from tiers import split_scores  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(ROOT, "reports", "backtests", "20261011_2228_current_split.json")
CACHE = os.path.join(ROOT, "data", "random_entry", "tiers.pkl")
OUT = os.path.join(ROOT, "reports", "random_entry.md")
START = pd.Timestamp("2026-09-11 22:30", tz="UTC")        # початок вікна контрольного бектесту 12.10


def pf(r: np.ndarray) -> float:
    loss = -r[r < 0].sum()
    return float(r[r > 0].sum() / loss) if loss > 0 else float("nan")


def load_coins(days: int) -> dict:
    """{монета: (t, h)} — рівні сигналу й 15m-бари, як у бектесті; з кешу, якщо він є."""
    if os.path.exists(CACHE):
        return pickle.load(open(CACHE, "rb"))
    coins = [x["ticker"] for x in json.load(open(REF, encoding="utf-8"))["per_ticker"]]
    m15 = joblib.load(train.model_path(train.M15_NAME, "models"))
    m15["start"] = joblib.load(train.model_path(train.START_NAME, "models"))
    m1 = joblib.load(train.model_path("lightgbm", "models"))
    out = {}
    with psycopg.connect(os.environ["DATABASE_URL"], connect_timeout=15) as conn:
        for i, base in enumerate(coins, 1):
            th = backtest.tiers_and_bars(conn, base, m15, m1, days)
            if th is not None:
                t, h = th
                out[base] = (t[t.index >= START][["alert_score", "start_score"]], h[h.index >= START - pd.Timedelta(days=1)])
            if i % 10 == 0:
                print(f"  оцінки: {i}/{len(coins)}", flush=True)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    tmp = CACHE + ".tmp"
    pickle.dump(out, open(tmp, "wb"))
    os.replace(tmp, CACHE)
    return out


def trades_from(price, high, low, a, hold, k, exit_bars, part):
    """Для кожного бару i — угода з входом на його закритті: (індекс бару виходу, результат з комісією); -1 — входу немає
    або угода не закрилась до кінця даних. Правила — ті самі, що в risk.position_state (вхід по ринку)."""
    n = len(price)
    ex, ret = np.full(n, -1), np.full(n, np.nan)
    frac = part["frac"]
    for i in range(n):
        if np.isnan(hold[i]) or np.isnan(price[i]) or np.isnan(a[i]):
            continue
        entry, stop, target = price[i], price[i] - k * a[i], price[i] + part["atr"] * a[i]
        below, pp = 0, None
        for j in range(i + 1, n):
            hit = low[j] <= stop
            if not hit and pp is None and high[j] >= target:
                pp = target
                if part.get("breakeven"):
                    stop = max(stop, entry)
            if not np.isnan(hold[j]):
                below = below + 1 if hold[j] < 1 else 0
            if below >= exit_bars or hit:
                px = stop if hit else price[j]
                rest = px / entry - 1
                ex[i] = j
                ret[i] = (rest if pp is None else frac * (pp / entry - 1) + (1 - frac) * rest) - backtest.FEE
                break
    return ex, ret


def walk(flags: np.ndarray, ex: np.ndarray, ret: np.ndarray, opn: np.ndarray) -> list[float]:
    """Послідовні угоди за відсортованими індексами дозволених барів входу: наступний вхід — не раніше бару після виходу.
    opn[i] — вхід на барі i можливий, але угода не закрилась до кінця даних: вона лишається відкритою і блокує подальші входи."""
    out, free = [], 0
    for i in flags:
        if i < free:
            continue
        if opn[i]:
            break
        if ex[i] < 0:
            continue
        out.append(ret[i])
        free = ex[i] + 1
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Тест значущості: вхід моделі проти випадкового входу в ті самі монети й дні")
    p.add_argument("--worlds", type=int, default=10000, help="кількість випадкових світів на кожен «нуль» (за замовчуванням 10000)")
    p.add_argument("--days", type=int, default=31, help="скільки днів оцінювати (вікно далі обрізається до початку контрольного бектесту)")
    p.add_argument("--seed", type=int, default=42, help="random seed перемішувань")
    args = p.parse_args()
    load_dotenv(os.path.join(ROOT, ".env"))
    data = load_coins(args.days)
    m15 = joblib.load(train.model_path(train.M15_NAME, "models"))
    k, exit_bars = m15["risk"]["k"], m15.get("exit_bars", 4)
    rule = strategies.entry_rule(strategies.get("market"))
    rng = np.random.default_rng(args.seed)
    modes = {"split": [], "market": []}                    # режим → [дані по монеті]
    real = {"split": [], "market": []}
    mismatch = 0
    for base, (t, h) in data.items():
        idx = h.index
        price, high, low = (h[c].to_numpy(dtype=float) for c in ("price", "high", "low"))
        a = atr(h).reindex(idx).to_numpy(dtype=float)
        day = np.array([ts.strftime("%Y-%m-%d") for ts in idx])
        main_s = t["alert_score"]
        entry_split, hold_split = split_scores(main_s, t["start_score"])
        for mode, entry_p, hold_s in (("split", entry_split, hold_split), ("market", None, main_s)):
            hold = hold_s.reindex(idx).to_numpy(dtype=float)
            ent = hold if entry_p is None else entry_p.reindex(idx).to_numpy(dtype=float)
            ex, ret = trades_from(price, high, low, a, hold, k, exit_bars, PARTIAL_EXIT)
            valid = ~np.isnan(hold) & ~np.isnan(price) & ~np.isnan(a)
            opn = valid & (ex < 0)
            flags = np.flatnonzero(valid & (np.nan_to_num(ent, nan=0.0) >= 1))
            mine = walk(flags, ex, ret, opn)
            # звірка зі справжнім рушієм угод
            pos = position_state(h, hold_s, 1.0, k, exit_bars, PARTIAL_EXIT, rule, None, None, entry_prob=entry_p)
            ref = [(x.get("result_pct") or 0) - backtest.FEE for x in pos.get("trades", [])]
            if len(ref) != len(mine) or (ref and not np.allclose(ref, mine, atol=1e-9)):
                mismatch += 1
            real[mode] += ref
            # реальні входи по днях → скільки випадкових входів брати в кожному дні
            entries, free = [], 0
            for i in flags:
                if i >= free and ex[i] >= 0:
                    entries.append(i)
                    free = ex[i] + 1
                elif i >= free and opn[i]:
                    break
            counts = pd.Series(day[entries]).value_counts().to_dict() if entries else {}
            modes[mode].append({"ex": ex, "ret": ret, "opn": opn, "counts": counts, "day": day,
                                "alert": valid & (hold >= 1), "any": valid})
    print(f"звірка швидкого симулятора з risk.position_state: розбіжностей {mismatch} з {2 * len(data)} (монета × режим)")
    res = {}
    for mode, coins in modes.items():
        r_real = np.array(real[mode])
        res[mode] = {"n": len(r_real), "pf": pf(r_real), "mean": float(r_real.mean())}
        for null, key in (("N1", "alert"), ("N2", "any")):
            pfs, means, ns = np.empty(args.worlds), np.empty(args.worlds), np.empty(args.worlds)
            # для кожної монети й дня — матриця випадкових барів на всі світи одразу
            picks = []
            for c in coins:
                per_world = [[] for _ in range(args.worlds)]
                for d, cnt in c["counts"].items():
                    pool = np.flatnonzero(c[key] & (c["day"] == d))
                    m = min(cnt, len(pool))
                    if m == 0:
                        continue
                    order = rng.random((args.worlds, len(pool))).argsort(axis=1)[:, :m]
                    chosen = pool[order]
                    for w in range(args.worlds):
                        per_world[w].extend(chosen[w])
                picks.append(per_world)
            for w in range(args.worlds):
                rr = []
                for c, per_world in zip(coins, picks):
                    if per_world[w]:
                        rr += walk(np.sort(np.array(per_world[w])), c["ex"], c["ret"], c["opn"])
                rr = np.array(rr)
                pfs[w], means[w], ns[w] = pf(rr), rr.mean() if len(rr) else np.nan, len(rr)
                if (w + 1) % 2000 == 0:
                    print(f"  {mode} {null}: {w + 1}/{args.worlds}", flush=True)
            res[mode][null] = {"p5": float(np.nanpercentile(pfs, 5)), "p50": float(np.nanpercentile(pfs, 50)),
                               "p95": float(np.nanpercentile(pfs, 95)), "p_value": float(np.mean(pfs >= res[mode]["pf"])),
                               "mean50": float(np.nanmedian(means)), "p_mean": float(np.mean(means >= res[mode]["mean"])),
                               "n": float(np.mean(ns))}
    s, m = res["split"], res["market"]
    ok1 = s["pf"] >= s["N1"]["p95"]
    ok2 = s["pf"] >= s["N2"]["p95"] and m["pf"] >= m["N2"]["p95"]
    lines = ["# Тест значущості: вхід моделі проти випадкового входу (12.10)", "",
             f"{len(data)} монет ≥ 20 млн поза навчанням, вікно контрольного бектесту (з 11.09), {args.worlds} випадкових світів, "
             "вихід як у робочому режимі, комісія 0.1%.",
             "N1 — випадковий бар серед барів з Alert тієї ж монети того ж дня; N2 — серед усіх барів того ж дня. "
             "Кількість входів на (монета, день) — як у справжніх угод.", "",
             "| Режим | Угод | PF справжній | «Нуль» | PF «нуля»: 5% / медіана / 95% | p-value (PF) | Сер. угода: справжня / «нуль» | p-value (сер.) | Угод у «нулі» |",
             "|---|---|---|---|---|---|---|---|---|"]
    for mode in ("split", "market"):
        r = res[mode]
        for null in ("N1", "N2"):
            z = r[null]
            lines.append(f"| {mode} | {r['n']} | {r['pf']:.2f} | {null} | {z['p5']:.2f} / {z['p50']:.2f} / {z['p95']:.2f} | "
                         f"{z['p_value']:.3f} | {r['mean']:+.2%} / {z['mean50']:+.2%} | {z['p_mean']:.3f} | {z['n']:.0f} |")
    lines += ["", f"**Критерій 1 (PF split ≥ 95-го перцентиля N1): {'ВИКОНАНО' if ok1 else 'НЕ ВИКОНАНО'}** "
              f"({s['pf']:.2f} проти {s['N1']['p95']:.2f})",
              f"**Критерій 2 (PF обох режимів ≥ 95-го перцентиля N2): {'ВИКОНАНО' if ok2 else 'НЕ ВИКОНАНО'}** "
              f"(split {s['pf']:.2f} проти {s['N2']['p95']:.2f}; market {m['pf']:.2f} проти {m['N2']['p95']:.2f})",
              "", f"Звірка симулятора з risk.position_state: розбіжностей {mismatch} з {2 * len(data)}."]
    open(OUT, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
