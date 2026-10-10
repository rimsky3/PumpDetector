"""
experiments/position_sizing.py — розмір позиції за Р. Вінсом («Математика управління капіталом»), 2026-10-09, власник: пункти 1 і 2.

1) Optimal f + Монте-Карло. Одиниця ризику — найгірша угода набору: HPR = 1 + f · (r / |r_min|), TWR = ΠHPR, f* — максимум TWR.
   f = 10% означає «повтор найгіршої угоди коштує 10% рахунку»; номінал позиції = рахунок × f / |r_min|. Від'ємне очікування →
   f* = 0 (за Вінсом розміром позиції це не виправити). Симуляція: 10 000 послідовностей (вибір угод з поверненням) довжиною
   як сам набір; для f*, ½ f*, ¼ f* і фіксованих 1 / 2 / 5%: множник рахунку (медіана, 5% найгірших), макс. просідання
   (медіана, 95-й перцентиль), P(просідання ≥ 30% / 50%), найдовша серія програшів.
2) Серійний тест: Z серій (формула 1.01 у Вінса; Z < −1.96 — серії довші, ніж випадково, «програш тягне програш»;
   Z > +1.96 — чергування), кореляція сусідніх результатів, P(програш | попередній програш) проти P(програш | попередній виграш).

Набори: угоди власника (reports/my_trades/positions.parquet — усі закриті і «системні»: ≥ 10 хв, без «відігратися» / «серії»;
r = net / max_usd, тобто % номіналу з комісією); угоди моделі — 6 міс. архіву Binance (reports/long_binance_trades.csv, чесний
список; усі і ≥ 20 млн) і останній бектест 30 днів (reports/backtests/*_current_alert_k2.13.json). Порядок — за часом входу.
→ reports/position_sizing.md

    python experiments/position_sizing.py [--sims 10000]
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "reports", "position_sizing.md")
FIXED = (0.01, 0.02, 0.05)
RNG = np.random.default_rng(42)


def datasets() -> dict[str, pd.Series]:
    out = {}
    p = pd.read_parquet(os.path.join(ROOT, "reports", "my_trades", "positions.parquet"))
    p = p[p["exit_time"].notna() & p["ret"].notna()].sort_values("entry_time")
    out["Власник: усі позиції"] = p["ret"].reset_index(drop=True)
    sysm = (p["minutes"] >= 10) & ~p["revenge"] & ~p["burst"]
    out["Власник: системні (≥ 10 хв, без тільту)"] = p[sysm]["ret"].reset_index(drop=True)
    t = pd.read_csv(os.path.join(ROOT, "reports", "long_binance_trades.csv")).sort_values("ts")
    out["Модель: 6 міс. архіву, усі монети"] = t["ret"].reset_index(drop=True)
    out["Модель: 6 міс. архіву, ≥ 20 млн"] = t[t["band"] == "≥ 20 млн"]["ret"].reset_index(drop=True)
    bt = sorted(glob.glob(os.path.join(ROOT, "reports", "backtests", "*_current_alert_k2.13.json")))
    if bt:
        d = json.load(open(bt[-1], encoding="utf-8"))
        r = pd.DataFrame(d["trades_list"]).dropna(subset=["ret"]).sort_values("entry_ts")["ret"]
        out[f"Модель: бектест 30 дн ({os.path.basename(bt[-1])[:13]})"] = r.reset_index(drop=True)
    return out


def twr(r: np.ndarray, f: float) -> float:
    hpr = 1 + f * r / abs(r.min())
    return np.exp(np.log(np.clip(hpr, 1e-12, None)).sum())


def optimal_f(r: np.ndarray) -> tuple[float, float]:
    if r.mean() <= 0 or r.min() >= 0:
        return 0.0, 1.0
    grid = np.arange(0.005, 1.0, 0.005)
    v = [twr(r, f) for f in grid]
    i = int(np.argmax(v))
    return float(grid[i]), float(v[i])


def monte_carlo(r: np.ndarray, f: float, sims: int) -> dict:
    n = len(r)
    idx = RNG.integers(0, n, size=(sims, n))
    hpr = 1 + f * r[idx] / abs(r.min())
    eq = np.cumprod(np.clip(hpr, 0, None), axis=1)
    peak = np.maximum.accumulate(np.concatenate([np.ones((sims, 1)), eq], axis=1), axis=1)[:, 1:]
    dd = (1 - eq / peak).max(axis=1)
    loss = r[idx] <= 0
    # найдовша серія програшів у кожній послідовності
    streak = np.zeros(sims, dtype=int)
    cur = np.zeros(sims, dtype=int)
    for j in range(n):
        cur = np.where(loss[:, j], cur + 1, 0)
        streak = np.maximum(streak, cur)
    fin = eq[:, -1]
    return {"fin_med": np.median(fin), "fin_p5": np.quantile(fin, 0.05), "dd_med": np.median(dd), "dd_p95": np.quantile(dd, 0.95),
            "p_dd30": (dd >= 0.30).mean(), "p_dd50": (dd >= 0.50).mean(), "streak_med": np.median(streak), "streak_p95": np.quantile(streak, 0.95)}


def runs_test(r: np.ndarray) -> dict:
    w = r > 0
    n, nw = len(w), int(w.sum())
    nl = n - nw
    runs = 1 + int((w[1:] != w[:-1]).sum())
    x = 2 * nw * nl
    z = (n * (runs - 0.5) - x) / np.sqrt(x * (x - n) / (n - 1)) if x > n else np.nan
    prev_l = ~w[:-1]
    return {"n": n, "wins": nw, "runs": runs, "z": z, "ac1": float(np.corrcoef(r[1:], r[:-1])[0, 1]),
            "p_loss": float((~w).mean()), "p_loss_after_loss": float((~w[1:])[prev_l].mean()) if prev_l.any() else np.nan,
            "p_loss_after_win": float((~w[1:])[~prev_l].mean()) if (~prev_l).any() else np.nan}


def main() -> None:
    ap = argparse.ArgumentParser(description="Optimal f, Монте-Карло просідань і серійний тест за Вінсом на угодах власника і моделі")
    ap.add_argument("--sims", type=int, default=10_000, help="скільки симуляцій Монте-Карло (за замовчуванням 10 000)")
    args = ap.parse_args()
    L = ["# Розмір позиції за Вінсом: optimal f, просідання, серії", "",
         "Одиниця ризику — найгірша угода набору (% номіналу). **f = частка рахунку, яку втратиш, якщо повториться найгірша угода.** "
         "Номінал позиції = рахунок × f ÷ |найгірша угода|. Симуляції — вибір угод з поверненням, довжина = кількість угод набору.", ""]
    for name, s in datasets().items():
        r = s.to_numpy(dtype=float)
        pfv = r[r > 0].sum() / -r[r <= 0].sum()
        f_star, t_star = optimal_f(r)
        L += [f"## {name}", "",
              f"{len(r)} угод · PF {pfv:.2f} · середня {r.mean():+.2%} · медіана {np.median(r):+.2%} · найгірша {r.min():+.1%} · "
              f"найкраща {r.max():+.1%}", ""]
        if f_star == 0:
            L += ["**Optimal f = 0**: очікування ≤ 0 — за Вінсом будь-який розмір позиції в довгу програє, збільшувати ризик немає сенсу.", ""]
        else:
            g = t_star ** (1 / len(r))
            L += [f"**Optimal f = {f_star:.1%}** (номінал = {f_star / abs(r.min()):.1f}× рахунку): TWR {t_star:.3g} за {len(r)} угод, "
                  f"геометричне середнє на угоду {g - 1:+.3%}.", ""]
        fs = ([("f*", f_star), ("½ f*", f_star / 2), ("¼ f*", f_star / 4)] if f_star > 0 else []) + [(f"{x:.0%}", x) for x in FIXED]
        L += ["| ризик f | номінал / рахунок | рахунок ×, медіана | 5% найгірших | просідання, медіана | 95-й перц. | P(≥ 30%) | P(≥ 50%) | серія програшів, медіана / 95% |",
              "|---|---|---|---|---|---|---|---|---|"]
        for lab, f in fs:
            m = monte_carlo(r, f, args.sims)
            L.append(f"| {lab} = {f:.1%} | {f / abs(r.min()):.2f}× | {m['fin_med']:.2f} | {m['fin_p5']:.2f} | {m['dd_med']:.0%} | "
                     f"{m['dd_p95']:.0%} | {m['p_dd30']:.0%} | {m['p_dd50']:.0%} | {m['streak_med']:.0f} / {m['streak_p95']:.0f} |")
        rt = runs_test(r)
        verdict = ("серії довші за випадкові — «програш тягне програш»" if rt["z"] < -1.96 else
                   "результати чергуються частіше, ніж випадково" if rt["z"] > 1.96 else "залежності між сусідніми угодами не видно")
        L += ["", f"**Серійний тест:** Z = {rt['z']:+.2f} ({verdict}); кореляція сусідніх результатів {rt['ac1']:+.3f} "
              f"(поріг значущості ≈ ±{1.96 / np.sqrt(rt['n']):.3f}); P(програш) {rt['p_loss']:.0%}, після програшу "
              f"{rt['p_loss_after_loss']:.0%}, після виграшу {rt['p_loss_after_win']:.0%}.", ""]
        print(f"{name}: n {len(r)}, PF {pfv:.2f}, f* {f_star:.1%}, Z {rt['z']:+.2f}")
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
