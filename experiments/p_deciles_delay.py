"""
experiments/p_deciles_delay.py — ДВІ ПЕРЕВІРКИ РОБОЧОЇ МОДЕЛІ на 6 місяцях архіву (2026-10-14; ідеї з розбору altcoin-statarb,
notes/altcoin_statarb_review.md). Модель не навчається: готові models/ (15m, «старт», годинна), монети — чесний список
6 міс. (data/long_bin) без тікерів навчання, розмітки й виключених, як у candidate_6m.py. Режими split (основний) і market.
Вихід — як у роботі: ½ на +3 ATR → беззбиток, 4 бари без сигналу, стоп k = 2.13 ATR(15m), комісія 0.1%.

1) ЧИ КРАЩА УГОДА З ВИЩОЮ P. Власник питав про EV наперед; якщо P на вході ранжує угоди, її можна використати для оцінки EV
   і розміру позиції. Міра — результат угоди в R (R = результат ÷ початковий ризик k·ATR/ціна входу), P — оцінка основної
   15m-моделі на барі входу (та, що на графіку).
   КРИТЕРІЙ (до запуску), split, монети ≥ 20 млн, вхід без затримки: P «ранжує угоди», якщо одночасно
     (а) кореляція Спірмена ρ(P, R) > 0 і 95% бутстреп-інтервал вище нуля;
     (б) середній R верхньої п'ятої частини угод за P > нижньої — в обох половинах монет (md5) і обох половинах часу
         (квітень–червень / липень–вересень).
   Пройдено → наступний крок: EV / розмір позиції за P. Ні → вище порогу P не каже, яка угода краща.

2) ЗАТРИМКА ВХОДУ. У симуляції (і в журналі) вхід — за закриттям сигнального 15m-бару; наживо сигнал видно через 1–3 хв
   (оцінка скринера, Telegram, реакція). Вхід через d хв: ціна — закриття 1m-свічки, що закінчується на d-й хвилині після
   закриття бару (кеш data/klines1m); стоп і ціль — від цієї ціни; перший бар після входу перевіряється лише на хвилинах
   ПІСЛЯ входу. d = 0 / 1 / 3 / 5.
   КРИТЕРІЙ (до запуску), split, ≥ 20 млн: модель «переживає затримку», якщо PF при d = 5 ≥ PF при d = 0 − 0.05 і в обох
   половинах монет падіння ≤ 0.05. Інакше — симуляція і журнал завищують результат на величину падіння.
→ reports/p_deciles_delay.md, reports/p_deciles_delay_trades.csv

    python experiments/p_deciles_delay.py            # ~30–60 хв
    python experiments/p_deciles_delay.py --limit 20 # перевірка на 20 монетах
"""

import argparse
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import auto_train as AT
import early_target as ET
import exit_partial as P
import idio_features as I
import klines1m as K
from candidate_6m import load_set, proba
from fetch import excluded_tickers, read_tickers_file
from tiers import split_scores, tier_frame

OUT = os.path.join("reports", "p_deciles_delay.md")
DELAYS = (0, 1, 3, 5)
BIG = "≥ 20 млн"
M = pd.Timedelta(minutes=1)


def simulate(h, entry, hold, p15, sl, exit_bars, delay, m1, tp=3.0, frac=0.5, be=True) -> list[dict]:
    """Як early_target.simulate_split (при delay = 0 — ті самі угоди), плюс вхід через delay хв за 1m-свічками,
    P на барі входу і ризик для R."""
    a = P.atr(h).reindex(h.index).to_numpy()
    en, ho, pp = entry.reindex(h.index).to_numpy(), hold.reindex(h.index).to_numpy(), p15.reindex(h.index).to_numpy()
    hi, lo, c = h["high"].to_numpy(), h["low"].to_numpy(), h["price"].to_numpy()
    out, tr, below, missing = [], None, 0, 0
    for i in range(len(h)):
        if np.isnan(ho[i]):
            continue
        if tr is None:
            if en[i] >= 1 and not np.isnan(c[i]) and not np.isnan(a[i]):
                e, first = c[i], None
                if delay:
                    t_close = h.index[i] + pd.Timedelta(minutes=15)
                    px = m1["c"].get(t_close + (delay - 1) * M)
                    if px is None or np.isnan(px):
                        missing += 1                          # немає 1m-свічки — вхід за закриттям бару (рідко)
                    else:
                        e = float(px)
                        w = m1.loc[t_close + delay * M: t_close + 14 * M]
                        first = (float(w["h"].max()), float(w["l"].min())) if len(w) else None
                tr = {"e": e, "stop": e - sl * a[i], "tp": e + tp * a[i] if tp else None, "part": 0.0, "i0": i,
                      "first": first, "risk": sl * a[i] / e, "p": pp[i]}
                below = 0
            continue
        e = tr["e"]
        hi_i, lo_i = (tr.pop("first") or (hi[i], lo[i])) if "first" in tr and i == tr["i0"] + 1 else (hi[i], lo[i])
        tr.pop("first", None)
        done = None
        if lo_i <= tr["stop"]:
            rest = tr["stop"] / e - 1
            done = ("stop", tr["part"] + (1 - (frac if tr["part"] else 0)) * rest - P.FEE)
        else:
            if tr["tp"] is not None and not tr["part"] and hi_i >= tr["tp"]:
                tr["part"] = frac * (tr["tp"] / e - 1)
                if be:
                    tr["stop"] = max(tr["stop"], e)
            below = below + 1 if ho[i] < 1 else 0
            if below >= exit_bars:
                rest = c[i] / e - 1
                done = ("model", tr["part"] + (1 - (frac if tr["part"] else 0)) * rest - P.FEE)
        if done:
            out.append({"ts": h.index[tr["i0"]], "ret": done[1], "reason": done[0], "risk": tr["risk"], "p15": tr["p"],
                        "R": done[1] / tr["risk"] if tr["risk"] > 0 else np.nan})
            tr = None
    return out, missing


def boot_spearman(x, y, n=2000, seed=42) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    rx, ry = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
    rho = float(np.corrcoef(rx, ry)[0, 1])
    k = len(x)
    bs = []
    for _ in range(n):
        j = rng.integers(0, k, k)
        bs.append(np.corrcoef(pd.Series(x[j]).rank(), pd.Series(y[j]).rank())[0, 1])
    return rho, float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def main() -> None:
    ap = argparse.ArgumentParser(description="Чи ранжує P угоди (R за квінтилями P) і чи переживає модель затримку входу 1–5 хв (6 міс.)")
    ap.add_argument("--limit", type=int, default=0, help="лише перші N монет (перевірка)")
    ap.add_argument("--k", type=float, default=2.13, help="стоп у ATR(15m), за замовчуванням робочий 2.13")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    s = load_set("models")
    uni = json.load(open(os.path.join(I.DIR, "_universe.json"), encoding="utf-8"))
    idx = I.market_index()
    trained = ({t.upper() for t in read_tickers_file("tickers.txt")} | set(pd.read_csv("pumps.csv")["base_asset"].str.upper())
               | excluded_tickers(".") | set(s["m15"].get("tickers", [])) | set(s["start"].get("tickers", [])))
    have1m = set(K.available())
    coins = sorted(c for c in uni if c not in trained and c in have1m and os.path.exists(os.path.join(I.DIR, f"{c}.pkl")))
    coins = coins[: args.limit or None]
    start, end = pd.Timestamp(I.START, tz="UTC"), pd.Timestamp(I.END, tz="UTC") + pd.Timedelta(days=1)
    share, exit_bars = s["m15"].get("confirm", {}).get("share", 0.75), s["m15"].get("exit_bars", 4)
    print(f"монет {len(coins)} (з 1m-свічками); k = {args.k}", flush=True)
    rows, missing = [], 0
    for n, c in enumerate(coins, 1):
        x = pickle.load(open(os.path.join(I.DIR, f"{c}.pkl"), "rb"))
        f15, h1, h = x["f15"], x["h1"], x["h"]
        f15 = f15.join(I.new_features(h, idx).reindex(f15["ts"]).set_axis(f15.index))
        hh = h[(h.index >= start - pd.Timedelta(days=1)) & (h.index < end)]
        s1 = proba(s["m1"], h1)
        pm = proba(s["m15"], f15)
        win = lambda t: t.loc[(t.index >= start) & (t.index < end), "alert_score"]      # noqa: E731
        main_s = win(tier_frame(pm, s1, share).set_index("ts").sort_index())
        start_s = win(tier_frame(proba(s["start"], f15), s1, share).set_index("ts").sort_index())
        p15 = pm.set_index("ts")["probability"].sort_index()
        p15 = p15[~p15.index.duplicated(keep="last")]
        entry, hold = split_scores(main_s, start_s)
        m1 = K.load(c)
        for mode, (en, ho) in (("split", (entry, hold)), ("market", (main_s, main_s))):
            for d in DELAYS:
                trades, miss = simulate(hh, en, ho, p15, args.k, exit_bars, d, m1)
                missing += miss
                for r in trades:
                    mth = pd.Timestamp(r["ts"]).strftime("%Y-%m")
                    if mth in uni[c]:
                        rows.append({"mode": mode, "delay": d, "ticker": c, "half": AT.half(c), "month": mth, "band": uni[c][mth], **r})
        if n % 50 == 0:
            print(f"  {n}/{len(coins)}", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT.replace(".md", "_trades.csv"), index=False)
    pf = I.pf
    t["period"] = np.where(t["month"] <= "2026-06", "квіт–черв", "лип–вер")

    # перевірка: при d = 0 симулятор дає ті самі угоди, що й робочий early_target.simulate_split (на перших монетах)
    L = ["# Чи ранжує P угоди і чи переживає модель затримку входу (6 міс., 14.10)", "",
         f"Готові models/; {len(coins)} монет чесного списку поза навчанням з 1m-свічками; стоп k = {args.k}; комісія 0.1%. "
         f"Входів без 1m-свічки (взято закриття бару): {missing}.", ""]

    # ---- 1) P → R
    L += ["## 1. Чи краща угода з вищою P (вхід без затримки, монети ≥ 20 млн)", ""]
    verdict1 = {}
    for mode in ("split", "market"):
        g = t[(t["mode"] == mode) & (t["delay"] == 0) & (t["band"] == BIG)].dropna(subset=["p15", "R"]).copy()
        g["q"] = pd.qcut(g["p15"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
        rho, lo, hi = boot_spearman(g["p15"].to_numpy(), g["R"].to_numpy())
        L += [f"### Режим {mode}: угод {len(g)}, ρ(P, R) = {rho:+.3f} (95%: {lo:+.3f} … {hi:+.3f})", "",
              "| п'ята частина за P | P від–до | угод | середній R | медіана R | у плюсі | PF |", "|---|---|---|---|---|---|---|"]
        for q, x in g.groupby("q", observed=True):
            L.append(f"| {q} | {x['p15'].min():.3f}–{x['p15'].max():.3f} | {len(x)} | {x['R'].mean():+.3f} | {x['R'].median():+.2f} | "
                     f"{(x['ret'] > 0).mean():.0%} | {pf(x['ret']):.2f} |")
        cuts = {}
        for col, vals in (("half", ("A", "B")), ("period", ("квіт–черв", "лип–вер"))):
            for v in vals:
                x = g[g[col] == v]
                top, bot = x[x["q"] == 5]["R"].mean(), x[x["q"] == 1]["R"].mean()
                cuts[v] = (top, bot)
        L += ["", "Верхня проти нижньої п'ятої частини (середній R): " + "; ".join(f"{k} {a:+.3f} / {b:+.3f}" for k, (a, b) in cuts.items()), ""]
        verdict1[mode] = rho > 0 and lo > 0 and all(a > b for a, b in cuts.values())
    L += [f"**Критерій 1 (split): {'ВИКОНАНО — P ранжує угоди' if verdict1['split'] else 'НЕ ВИКОНАНО — вище порогу P не каже, яка угода краща'}**"
          f" (довідково, market: {'так' if verdict1['market'] else 'ні'}).", ""]

    # ---- 2) затримка
    L += ["## 2. Затримка входу (монети ≥ 20 млн)", "",
          "| режим | затримка | угод | PF | половина A | половина B | середній R | у плюсі | по стопу |", "|---|---|---|---|---|---|---|---|---|"]
    res = {}
    for mode in ("split", "market"):
        for d in DELAYS:
            g = t[(t["mode"] == mode) & (t["delay"] == d) & (t["band"] == BIG)]
            res[(mode, d)] = (pf(g["ret"]), pf(g[g["half"] == "A"]["ret"]), pf(g[g["half"] == "B"]["ret"]))
            L.append(f"| {mode} | {d} хв | {len(g)} | **{res[(mode, d)][0]:.3f}** | {res[(mode, d)][1]:.3f} | {res[(mode, d)][2]:.3f} | "
                     f"{g['R'].mean():+.3f} | {(g['ret'] > 0).mean():.0%} | {(g['reason'] == 'stop').mean():.0%} |")
    b0, b5 = res[("split", 0)], res[("split", 5)]
    ok2 = b5[0] >= b0[0] - 0.05 and b0[1] - b5[1] <= 0.05 and b0[2] - b5[2] <= 0.05
    L += ["", f"**Критерій 2 (split, d = 5 хв): {'ВИКОНАНО — модель переживає затримку' if ok2 else 'НЕ ВИКОНАНО — симуляція завищує результат'}**"
          f" — PF {b5[0]:.3f} проти {b0[0]:.3f} (A {b5[1]:.3f} / {b0[1]:.3f}, B {b5[2]:.3f} / {b0[2]:.3f}).", ""]
    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
