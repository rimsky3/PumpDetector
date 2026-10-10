"""
experiments/short_exhaustion.py — ШОРТ ПІСЛЯ ВИСНАЖЕННЯ (2026-10-10, план «інша логіка», крок 3; власник: «тестуй так поки»).
Чому: найстабільніший ефект у даних — згасання після сплеску (in_play: після Alert ціна −0.4 / −0.7 / −1.1% за 1 / 4 / 12 год;
style_backtest: монети після +20% за добу в середньому падають далі, а з Alert — сильніше). Лонгу PF > 1 на нових монетах не
дала жодна логіка (авторозмітка, ціль «потенціал», подієва). Питання: чи ЗАРОБЛЯЄ шорт на згасанні з урахуванням стопу,
комісії, funding і сквізів. Без нової моделі — правила на сигналах РОБОЧОЇ моделі (models/, Alert = 15m ≥ порогу + годинна ≥ 0.75).
Дані — кеш архіву Binance 6 міс. (data/long_bin), чесний список без тікерів власника, 15m.
Епізод Alert — бари Alert з розривом < 4 барів (як вихід лонгу). Старт руху — мінімум low за 3 год до першого Alert (як авторозмітка);
пік — максимум high від старту епізоду до поточного бару (лише минуле).
Входи (зафіксовано ДО запуску; один вхід на епізод / подію, шорт за закриттям бару):
    A  проти Alert            бар Alert, монета вже ≥ +20% за 24 год
    B  лонг-модель вийшла     4-й бар поспіль без Alert після епізоду, рух епізоду ≥ 8 ATR
    C  кінець за правилом     закриття < пік − ⅓ руху, рух ≥ 8 ATR і ≥ 5% (правило кінця розмітки без 3-год перевірки),
                              від старту епізоду до 12 год після його кінця
    D  OI злився              OI −10% від піку OI з початку епізоду, рух ≥ 8 ATR, те саме вікно
    K1 контроль: випадковий шорт — на кожній монеті стільки ж барів, скільки входів A, випадково (seed 42), пік — max high 4 год
    K2 контроль: монета перетнула +20% за 24 год (без умови Alert), пауза 12 год, пік — max high 4 год
Вихід (дзеркало лонгу): стоп = min(пік + 0.5 ATR, вхід + 2.5 ATR) (не ближче за вхід + 0.5 ATR); ½ на вхід − 3 ATR → стоп у б/з;
решта — закриття через 12 год або стоп. Стоп перевіряється раніше за ціль (обережно); бар відкрився вище стопу — виконання за
відкриттям (геп). Комісія 0.1% на угоду; funding — за кожну виплату 00 / 08 / 16 UTC під час угоди (додатний — шорт отримує).
Міряємо: PF (з funding і без), у плюсі, медіана, без топ-3, БЕЗ НАЙГІРШИХ 3, ≥ 20 млн / 5–20 млн, половини монет, місяці;
сквізи — частка угод, де за 12 год після входу high ≥ вхід + 3 ATR (і ≥ +10%), частка гепів через стоп, найгірші угоди;
сирий дрейф — медіана (вхід / закриття через 12 год − 1) без стопу.
КРИТЕРІЙ (до запуску), на ≥ 20 млн: PF > 1.05 (з комісією і funding); PF > 1 щонайменше в 4 з 6 місяців; PF > 1 на обох половинах;
кращий за K1 і K2; без найгірших 3 PF > 1; сквізів (≥ 3 ATR за 12 год) ≤ 15%.
→ reports/short_exhaustion.md, reports/short_exhaustion_trades.csv

    python experiments/short_exhaustion.py
"""

import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import numpy as np
import pandas as pd

import auto_train as AT
import idio_features as I
import train
from fetch import read_tickers_file
from tiers import tier_frame

FEE = 0.001
HOT = 0.20                 # +20% за 24 год
MOVE_ATR, MOVE_PCT = 8.0, 0.05
GAP = 4                    # бари без Alert, після яких епізод закінчено (і вихід лонгу)
PRE = 12                   # 3 год до першого Alert — старт руху
AFTER = 48                 # вікно C / D: до 12 год після кінця епізоду
HOLD = 48                  # 12 год утримання
STOP_PAD, STOP_MAX, STOP_MIN, TP = 0.5, 2.5, 0.5, 3.0
SQUEEZE_ATR, SQUEEZE_PCT = 3.0, 0.10
VARIANTS = ["A", "B", "C", "D", "K1", "K2"]
NAMES = {"A": "проти Alert (+20% за добу)", "B": "лонг-модель вийшла (рух ≥ 8 ATR)", "C": "кінець за правилом (віддав ⅓)",
         "D": "OI −10% від піку", "K1": "контроль: випадковий шорт", "K2": "контроль: +20% за добу без Alert"}
BIG, MID = "≥ 20 млн", "5–20 млн"
OUT = os.path.join("reports", "short_exhaustion.md")


def owner_tickers() -> set:
    t = {x.upper() for x in read_tickers_file("tickers.txt")}
    return t | set(pd.read_csv("pumps.csv")["base_asset"].str.upper())


def episodes(alert: np.ndarray) -> list[tuple[int, int]]:
    """Епізоди Alert: (перший, останній бар з Alert); розрив < GAP барів — той самий епізод."""
    out, s, last = [], None, None
    for i in np.flatnonzero(alert):
        if s is None or i - last >= GAP:
            if s is not None:
                out.append((s, last))
            s = i
        last = i
    if s is not None:
        out.append((s, last))
    return out


def entries(s: pd.DataFrame, alert: np.ndarray, rng, i0: int) -> dict:
    """Входи кожного варіанта: список (бар, пік для стопу); i0 — перший бар періоду тесту (для випадкових входів K1)."""
    c, hi, lo, a, oi = (s[k].to_numpy() for k in ("price", "high", "low", "atr", "oi"))
    n = len(s)
    r24 = np.full(n, np.nan)
    r24[96:] = c[96:] / c[:-96] - 1
    out = {v: [] for v in VARIANTS}
    for s0, s1 in episodes(alert):
        p0 = np.nanmin(lo[max(0, s0 - PRE):s0 + 1])
        a0 = a[s0]
        end = min(n - 1, s1 + GAP + AFTER)
        peak = np.maximum.accumulate(hi[s0:end + 1])          # пік від старту епізоду, лише минуле
        oipk = np.fmax.accumulate(oi[s0:end + 1])
        done = {"A": False, "C": False, "D": False}
        for j, i in enumerate(range(s0, end + 1)):
            if not np.isfinite(c[i]) or not np.isfinite(a[i]) or not a0:
                continue
            move = peak[j] - p0
            big = move / a0 >= MOVE_ATR and move / p0 >= MOVE_PCT
            if not done["A"] and i <= s1 and alert[i] and r24[i] >= HOT:
                out["A"].append((i, peak[j]))
                done["A"] = True
            if not done["C"] and big and c[i] < peak[j] - move / 3:
                out["C"].append((i, peak[j]))
                done["C"] = True
            if not done["D"] and big and np.isfinite(oi[i]) and oi[i] <= 0.9 * oipk[j]:
                out["D"].append((i, peak[j]))
                done["D"] = True
        b = s1 + GAP                                            # 4-й бар без Alert
        if b < n and (peak[min(b, end) - s0] - p0) / a0 >= MOVE_ATR:
            out["B"].append((b, peak[min(b, end) - s0]))
    pk4 = pd.Series(hi).rolling(16, min_periods=1).max().to_numpy()
    ok = np.flatnonzero(np.isfinite(c) & np.isfinite(a))
    ok = ok[ok >= i0]
    k = sum(i >= i0 for i, _ in out["A"])
    if k and len(ok):
        for i in np.sort(rng.choice(ok, size=min(k, len(ok)), replace=False)):
            out["K1"].append((i, pk4[i]))
    last = -10 ** 9
    for i in range(97, n):
        if r24[i] >= HOT and not r24[i - 1] >= HOT and i - last >= HOLD:
            out["K2"].append((i, pk4[i]))
            last = i
    return out


def simulate(s: pd.DataFrame, fund: np.ndarray, i: int, peak: float) -> dict | None:
    """Шорт за закриттям бару i: стоп над піком (обмежений), ½ на −3 ATR → б/з, решта — 12 год або стоп."""
    o, c, hi, lo, a = (s[k].to_numpy() for k in ("open", "price", "high", "low", "atr"))
    e, at = c[i], a[i]
    if not (np.isfinite(e) and np.isfinite(at) and at > 0) or i + 1 >= len(c):
        return None
    stop = min(max(peak + STOP_PAD * at, e + STOP_MIN * at), e + STOP_MAX * at)
    tp = e - TP * at
    part, gap, fund_sum, reason, px, j = 0.0, False, 0.0, "time", None, i
    last = min(len(c) - 1, i + HOLD)
    idx = s.index
    for j in range(i + 1, last + 1):
        if idx[j].minute == 0 and idx[j].hour % 8 == 0 and np.isfinite(fund[j]):
            fund_sum += fund[j] / 100 * (1 - (0.5 if part else 0))
        if hi[j] >= stop:
            px = max(stop, o[j]) if np.isfinite(o[j]) else stop
            gap = np.isfinite(o[j]) and o[j] > stop
            reason = "stop"
            break
        if not part and lo[j] <= tp:
            part = 0.5 * (1 - tp / e)
            stop = min(stop, e)
    if px is None:
        px, j = c[last], last
    rest = 1 - px / e
    ret_nof = part + (1 - (0.5 if part else 0)) * rest - FEE
    w = slice(i + 1, min(len(c), i + 1 + HOLD))
    mx = np.nanmax(hi[w]) if len(hi[w]) else e
    drift = 1 - c[min(len(c) - 1, i + HOLD)] / e
    return {"ts": idx[i], "entry": e, "ret": ret_nof + fund_sum, "ret_nofund": ret_nof, "funding": fund_sum, "reason": reason,
            "took": bool(part), "gap": bool(gap), "bars": j - i, "stop_atr": (stop - e) / at if not part else np.nan,
            "squeeze": bool(mx >= e + SQUEEZE_ATR * at and mx >= e * (1 + SQUEEZE_PCT)), "mae_atr": (mx - e) / at, "drift12": drift}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    argparse.ArgumentParser(description="Шорт після виснаження: правила на сигналах робочої моделі, 6 міс. архіву, контролі, сквізи").parse_args()
    import json
    uni = json.load(open(os.path.join(I.DIR, "_universe.json"), encoding="utf-8"))
    idx = I.market_index()
    m15 = joblib.load(train.model_path(train.M15_NAME))
    m1 = joblib.load(train.model_path("lightgbm"))
    share = m15.get("confirm", {}).get("share", 0.75)
    own = owner_tickers()
    coins = sorted(c for c in uni if c not in own and os.path.exists(os.path.join(I.DIR, f"{c}.pkl")))
    start, end = pd.Timestamp(I.START, tz="UTC"), pd.Timestamp(I.END, tz="UTC") + pd.Timedelta(days=1)
    rng = np.random.default_rng(42)
    rows = []
    print(f"монет {len(coins)}; модель 15m поріг {m15['threshold']:.3f}", flush=True)
    for n, c in enumerate(coins, 1):
        x = pickle.load(open(os.path.join(I.DIR, f"{c}.pkl"), "rb"))
        f15, h1, h = x["f15"], x["h1"], x["h"]
        f15 = f15.join(I.new_features(h, idx).reindex(f15["ts"]).set_axis(f15.index))
        s15 = f15[["base_asset", "ts"]].assign(probability=m15["model"].predict_proba(f15[m15["features"]])[:, 1], threshold=m15["threshold"])
        s1 = h1[["base_asset", "ts"]].assign(probability=m1["model"].predict_proba(h1[m1["features"]])[:, 1], threshold=m1["threshold"])
        t = tier_frame(s15, s1, share).set_index("ts")["alert_score"]
        s = AT.coin_series(c, h).join(h[["open"]])
        s = s[(s.index >= start - pd.Timedelta(days=2)) & (s.index < end)]
        alert = (t.reindex(s.index) >= 1).to_numpy()
        fund = f15.set_index("ts")["funding"].reindex(s.index).to_numpy()
        months = uni[c]
        for v, lst in entries(s, alert, rng, int(np.searchsorted(s.index, start))).items():
            seen = set()                     # вікна C / D сусідніх епізодів перекриваються — той самий бар входу один раз
            for i, peak in lst:
                if s.index[i] < start or i in seen:
                    continue
                seen.add(i)
                r = simulate(s, fund, i, peak)
                if r is None:
                    continue
                mth = r["ts"].strftime("%Y-%m")
                if mth in months:
                    rows.append({"variant": v, "ticker": c, "half": AT.half(c), "month": mth, "band": months[mth], **r})
        if n % 50 == 0:
            print(f"  {n}/{len(coins)}", flush=True)
    tr = pd.DataFrame(rows)
    tr.to_csv(OUT.replace(".md", "_trades.csv"), index=False)
    write_report(tr)


def write_report(tr: pd.DataFrame) -> None:
    pf = I.pf

    def pf_wo_worst(r, k=3):
        r = r.sort_values()
        return pf(r.iloc[k:])
    L = ["# Шорт після виснаження", "",
         "Правила на сигналах робочої моделі, 6 міс. архіву Binance, чесний список без тікерів власника. Шорт за закриттям бару; "
         f"стоп — над піком + {STOP_PAD} ATR (не далі {STOP_MAX} ATR від входу); ½ на −{TP:g} ATR → б/з; решта — 12 год або стоп; "
         "комісія 0.1%, funding за виплати під час угоди.", "",
         "**Критерій (до запуску), ≥ 20 млн:** PF > 1.05; PF > 1 у ≥ 4 з 6 місяців; PF > 1 на обох половинах; кращий за обидва "
         "контролі; без найгірших 3 PF > 1; сквізів ≤ 15%.", "",
         "Сквіз — за 12 год після входу ціна йшла проти ≥ 3 ATR і ≥ 10% (незалежно від стопу). Геп — бар відкрився вище стопу. "
         "Сирий дрейф — падіння ціни за 12 год без стопу (медіана; плюс = на користь шорту).", "",
         "## Усі монети", "",
         "| вхід | угод | у плюсі | медіана | PF | PF без funding | без топ-3 | без найгірших 3 | сквізів | гепів | сирий дрейф 12 год | funding / угоду |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for v in VARIANTS:
        g = tr[tr["variant"] == v]
        if g.empty:
            continue
        r = g["ret"].sort_values()
        L.append(f"| {v}: {NAMES[v]} | {len(g)} | {(r > 0).mean():.0%} | {r.median():+.2%} | {pf(r):.2f} | {pf(g['ret_nofund']):.2f} | "
                 f"{pf(r.iloc[:-3]):.2f} | {pf_wo_worst(r):.2f} | {g['squeeze'].mean():.0%} | {g['gap'].mean():.1%} | "
                 f"{g['drift12'].median():+.2%} | {g['funding'].mean():+.3%} |")
    L += ["", "## За оборотом і половинами монет (PF)", "", "| вхід | ≥ 20 млн | 5–20 млн | A ≥ 20 млн | B ≥ 20 млн | ≥ 20 млн без найгірших 3 | сквізів ≥ 20 млн |",
          "|---|---|---|---|---|---|---|"]
    for v in VARIANTS:
        g = tr[tr["variant"] == v]
        if g.empty:
            continue
        b = g[g["band"] == BIG]
        L.append(f"| {v} | {pf(b['ret']):.2f} ({len(b)}) | {pf(g[g['band'] == MID]['ret']):.2f} | {pf(b[b['half'] == 'A']['ret']):.2f} | "
                 f"{pf(b[b['half'] == 'B']['ret']):.2f} | {pf_wo_worst(b['ret']):.2f} | {b['squeeze'].mean():.0%} |")
    V = [v for v in VARIANTS if v in set(tr["variant"])]
    piv = tr.groupby(["month", "variant"])["ret"].apply(pf).unstack().reindex(columns=V)
    piv20 = tr[tr["band"] == BIG].groupby(["month", "variant"])["ret"].apply(pf).unstack().reindex(columns=V)
    L += ["", "## По місяцях (PF; у дужках ≥ 20 млн)", "", "| місяць | " + " | ".join(V) + " |", "|---|" + "---|" * len(V)]
    for mth in piv.index:
        L.append(f"| {mth} | " + " | ".join(f"{piv.at[mth, v]:.2f} ({piv20.at[mth, v]:.2f})" for v in V) + " |")
    L += ["", "## Найгірші угоди (усі монети)", "", "| вхід | монета | час входу | результат | причина | геп | рух проти, ATR |", "|---|---|---|---|---|---|---|"]
    for v in ("A", "B", "C", "D"):
        g = tr[tr["variant"] == v].nsmallest(3, "ret")
        for r in g.itertuples():
            L.append(f"| {v} | {r.ticker} | {pd.Timestamp(r.ts):%Y-%m-%d %H:%M} | {r.ret:+.1%} | {r.reason} | {'так' if r.gap else ''} | {r.mae_atr:.1f} |")
    L += ["", "## Критерій", ""]
    big = tr[tr["band"] == BIG]
    ctrl = max(pf(big[big["variant"] == k]["ret"]) for k in ("K1", "K2"))
    for v in ("A", "B", "C", "D"):
        b = big[big["variant"] == v]
        if b.empty:
            L.append(f"- {v}: угод немає")
            continue
        p = pf(b["ret"])
        months = int((piv20[v] > 1).sum())
        halves = all(pf(b[b["half"] == hf]["ret"]) > 1 for hf in ("A", "B"))
        worst = pf_wo_worst(b["ret"]) > 1
        sq = b["squeeze"].mean() <= 0.15
        ok = p > 1.05 and months >= 4 and halves and p > ctrl and worst and sq
        L.append(f"- {v} ({NAMES[v]}): PF {p:.2f} {'>' if p > 1.05 else '≤'} 1.05; місяців з PF > 1: {months} з {len(piv20)}; "
                 f"половини {'так' if halves else 'ні'}; краще за контролі ({ctrl:.2f}) {'так' if p > ctrl else 'ні'}; "
                 f"без найгірших 3 {'так' if worst else 'ні'}; сквізів {b['squeeze'].mean():.0%} → **{'ВИКОНАНО' if ok else 'НЕ виконано'}**")
    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
