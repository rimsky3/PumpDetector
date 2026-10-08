"""
experiments/exit_5m.py — ранній вихід власника на 5m-барах, 6 місяців, чесний список монет (2026-10-08).

Правило власника (розбір стопу MOVR 01.10: «кожна 5-хвилинка червона, OI виходить — сенсу чекати стоп немає»): у відкритій
угоді n 5m-свічок поспіль закрились червоними (close < open), OI у МОНЕТАХ падав на кожній, і ціна за цей час не оновила
максимум з моменту входу → вихід за закриттям n-ї свічки. На 15m (exit_structure.py, 01.10) переваги не дав: медіана трохи
краща, сума та сама, ріже великі виграші. На 5m не перевірялось — у монет бектесту не було 5m-даних; тепер є архів Binance.

Входи — Alert РОБОЧОЇ моделі (15m з власним рухом + годинна, models/), монети чесного списку станом на кожен місяць
(data/long_bin, як idio_features.py), квітень–вересень, лише Binance; угоди — лише в місяцях, коли монета була в списку.
Позиція — як у дашборді: стоп entry − k·ATR15, ½ на +3 ATR і стоп у беззбиток, решта — 4 15m-бари без Alert або стоп;
після виходу новий вхід на наступному барі з Alert. Усе перевіряється на 5m-барах (стоп раніше за ціль усередині бару).

Варіанти:
    A15        поточна симуляція на 15m (exit_partial.simulate) — звірка рушія
    A5         те саме на 5m-барах без раннього виходу — точка відліку
    E2/E3/E4   + ранній вихід: 2 / 3 / 4 червоні 5m з падінням OI (монети), без нового хаю
    E3p        E3 без умови OI (лише ціна) — чи OI щось додає
    E2w/E3w    + після раннього виходу не заходити знову, доки Alert не згасне (бар без Alert)
5m-свічки — архів Binance (місячні файли, кеш data/binance_archive), OI 5m — архів metrics (вже в кеші); зведення на монету —
data/exit5m/<монета>.pkl.

    python experiments/exit_5m.py --stage data     # завантажити 5m (довго, ~15–25 хв)
    python experiments/exit_5m.py                  # дані (якщо ще немає) + прогін → reports/exit_5m.md
"""

import argparse
import glob
import json
import os
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import numpy as np
import pandas as pd

import binance_archive as BA
import exit_partial as P
import idio_features as E
import train
from features import load_pumps, move_features
from fetch import excluded_tickers, read_tickers_file
from risk import atr
from tiers import tier_frame

DIR = os.path.join("data", "long_bin")
OUT_DIR = os.path.join("data", "exit5m")
START, END = "2026-04-01", "2026-09-30"
OUT = os.path.join("reports", "exit_5m.md")
FIVE = pd.Timedelta(minutes=5)
# funding тут не потрібен, а API для делістованих символів його не знає — download не питає біржу взагалі
BA.binance_funding = lambda *a, **k: pd.DataFrame({"ts": pd.to_datetime([], utc=True), "value": []})
VARIANTS = {"A5": dict(early=0), "E2": dict(early=2), "E3": dict(early=3), "E4": dict(early=4),
            "E3p": dict(early=3, oi=False), "E2w": dict(early=2, wait=True), "E3w": dict(early=3, wait=True)}


# ---------------------------------------------------------------- дані: 5m-свічки + OI у монетах

def load_5m(coin: str) -> pd.DataFrame | None:
    """5m o / h / l / c і OI у монетах на закритті бару (ts — відкриття бару); кеш data/exit5m/<монета>.pkl."""
    path = os.path.join(OUT_DIR, f"{coin}.pkl")
    if os.path.exists(path):
        try:
            return pickle.load(open(path, "rb"))
        except Exception:
            os.remove(path)
    d = BA.download(f"{coin}USDT", START, END, "5m")
    k = d["klines"]
    if k.empty:
        return None
    df = k.set_index("ts")[["o", "h", "l", "c"]].astype("float32")
    # OI біржі з міткою T+5хв — знімок на закритті бару T (як exchanges.normalize)
    oi = d["oi"].assign(ts=d["oi"]["ts"] - FIVE).set_index("ts")["value"]
    df["oi"] = oi.reindex(df.index).astype("float32")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    os.makedirs(OUT_DIR, exist_ok=True)
    pickle.dump(df, open(path + ".tmp", "wb"))
    os.replace(path + ".tmp", path)
    return df


def coins_list() -> tuple[list[str], dict]:
    uni = json.load(open(os.path.join(DIR, "_universe.json"), encoding="utf-8"))
    m15 = joblib.load(train.model_path(train.M15_NAME))
    excluded = excluded_tickers(".")
    trained = {t.upper() for t in read_tickers_file("tickers.txt")} | set(m15["tickers"])
    try:
        import psycopg
        from dotenv import load_dotenv
        load_dotenv(".env")
        with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
            trained |= set(load_pumps(conn)["base_asset"])
    except Exception:
        pass
    coins = sorted(c for c in uni if c not in trained and c not in excluded and os.path.exists(os.path.join(DIR, f"{c}.pkl")))
    return coins, uni


def stage_data(coins: list[str]) -> None:
    done = [0]

    def job(c):
        try:
            load_5m(c)
        except Exception as e:
            print(f"  ! {c}: {type(e).__name__}: {e}", flush=True)
        done[0] += 1
        if done[0] % 25 == 0:
            print(f"  5m: {done[0]}/{len(coins)}", flush=True)
    with ThreadPoolExecutor(3) as pool:          # усередині download ще 8 потоків
        list(pool.map(job, coins))


# ---------------------------------------------------------------- симуляція на 5m

def sim5(h15: pd.DataFrame, score: pd.Series, h5: pd.DataFrame, k: float, exit_bars: int, early: int = 0,
         oi: bool = True, wait: bool = False, tp_atr: float = 3.0, frac: float = 0.5) -> list[dict]:
    """Позиція як risk.position_state з частковим виходом, перевірки — на 5m-барах; вхід і вихід за моделлю — на
    закриттях 15m. early > 0 — ранній вихід за правилом власника."""
    a = atr(h15).reindex(h15.index).to_numpy()
    sc = score.reindex(h15.index).to_numpy()
    c15 = h15["price"].to_numpy()
    grid = h15.index.repeat(3) + pd.to_timedelta(np.tile([0, 5, 10], len(h15)), unit="min")   # три 5m-бари кожного 15m
    o5, hi5, lo5, c5, oi5 = (h5[x].reindex(grid).to_numpy(dtype=float).reshape(-1, 3) for x in ("o", "h", "l", "c", "oi"))
    prev_oi = np.concatenate([[np.nan], oi5.ravel()[:-1]]).reshape(-1, 3)
    out, tr, below, cnt, waiting = [], None, 0, 0, False
    idx = h15.index
    for i in range(len(idx)):
        p = sc[i]
        if tr is None:
            if waiting:
                if not np.isnan(p) and p < 1:
                    waiting = False
                continue
            if not np.isnan(p) and p >= 1 and not np.isnan(c15[i]) and not np.isnan(a[i]):
                e = c15[i]
                tr = {"e": e, "stop": e - k * a[i], "tp": e + tp_atr * a[i], "part": 0.0, "i0": i,
                      "high": np.nanmax(hi5[i]) if np.isfinite(hi5[i]).any() else e}
                below, cnt = 0, 0
            continue
        exit_px = reason = None
        for j in range(3):
            if np.isnan(lo5[i, j]):
                cnt = 0
                continue
            if lo5[i, j] <= tr["stop"]:
                exit_px, reason = tr["stop"], "stop"
                break
            if not tr["part"] and hi5[i, j] >= tr["tp"]:
                tr["part"] = frac * (tr["tp"] / tr["e"] - 1)
                tr["stop"] = max(tr["stop"], tr["e"])
            if early:
                new_high = hi5[i, j] > tr["high"]
                red = c5[i, j] < o5[i, j]
                oi_down = oi5[i, j] < prev_oi[i, j] if (np.isfinite(oi5[i, j]) and np.isfinite(prev_oi[i, j])) else False
                cnt = cnt + 1 if (red and (oi_down or not oi) and not new_high) else 0
                if cnt >= early:
                    exit_px, reason = c5[i, j], "early"
            tr["high"] = max(tr["high"], hi5[i, j])
            if exit_px is not None:
                break
        if exit_px is None and not np.isnan(p):
            below = below + 1 if p < 1 else 0
            if below >= exit_bars:
                exit_px, reason = c15[i], "model"
        if exit_px is not None:
            rest = exit_px / tr["e"] - 1
            out.append({"ret": tr["part"] + (1 - (frac if tr["part"] else 0)) * rest - P.FEE, "reason": reason,
                        "took": bool(tr["part"]), "bars": i - tr["i0"], "ts": idx[tr["i0"]]})
            waiting = wait and reason == "early" and not np.isnan(p) and p >= 1
            tr = None
    return out


# ---------------------------------------------------------------- прогін

def pf(v) -> float:
    v = pd.Series(v, dtype=float)
    gl = -v[v <= 0].sum()
    return v[v > 0].sum() / gl if gl else np.nan


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Ранній вихід власника (червоні 5m + OI у монетах падає, без нового хаю): 6 міс., чесний список")
    ap.add_argument("--stage", choices=["data", "all"], default="all", help="data — лише завантажити 5m; all — дані + прогін")
    ap.add_argument("--limit", type=int, help="лише перші N монет (перевірка)")
    args = ap.parse_args()
    coins, uni = coins_list()
    if args.limit:
        coins = coins[:args.limit]
    print(f"монет: {len(coins)}; 5m-дані…", flush=True)
    stage_data(coins)
    if args.stage == "data":
        return
    m15 = joblib.load(train.model_path(train.M15_NAME))
    m1 = joblib.load(train.model_path(m15.get("confirm", {}).get("model", "lightgbm")))
    k, exit_bars, share = m15["risk"]["k"], m15.get("exit_bars", 4), m15.get("confirm", {}).get("share", 0.75)
    print(f"модель {m15['trained_at']}, поріг {m15['threshold']:.3f}, k {k:.2f}; індекс ALT-EW з кешу…", flush=True)
    idx = E.market_index()
    start, end = pd.Timestamp(START, tz="UTC"), pd.Timestamp(END, tz="UTC") + pd.Timedelta(days=1)
    rows = []
    for n, c in enumerate(coins, 1):
        h5 = load_5m(c)
        if h5 is None or h5.empty:
            continue
        x = pickle.load(open(os.path.join(DIR, f"{c}.pkl"), "rb"))
        f15, h1, h = x["f15"], x["h1"], x["h"]
        f15 = f15.join(move_features(h, idx.reindex(h.index), 4).reindex(f15["ts"]).set_axis(f15.index))
        s15 = f15[["base_asset", "ts"]].assign(probability=m15["model"].predict_proba(f15[m15["features"]])[:, 1], threshold=m15["threshold"])
        s1 = h1[["base_asset", "ts"]].assign(probability=m1["model"].predict_proba(h1[m1["features"]])[:, 1], threshold=m1["threshold"])
        t = tier_frame(s15, s1, share).set_index("ts").sort_index()
        sc = t.loc[(t.index >= start) & (t.index < end), "alert_score"]
        hh = h[(h.index >= start - pd.Timedelta(days=1)) & (h.index < end)]
        months = uni[c]
        res = {"A15": P.simulate(hh, sc, k, exit_bars, 3.0, 0.5, True)}
        for name, kw in VARIANTS.items():
            res[name] = sim5(hh, sc, h5, k, exit_bars, **kw)
        for name, trades in res.items():
            for r in trades:
                mth = pd.Timestamp(r["ts"]).strftime("%Y-%m")
                if mth in months:
                    rows.append({"variant": name, "ticker": c, "month": mth, "band": months[mth], **r})
        if n % 50 == 0:
            print(f"  {n}/{len(coins)}", flush=True)
    tr = pd.DataFrame(rows)
    tr.to_csv(os.path.join("reports", "exit_5m_trades.csv"), index=False)
    names = ["A15"] + list(VARIANTS)

    base = tr[tr["variant"] == "A5"].set_index(["ticker", "ts"])["ret"]
    lines = ["# Ранній вихід власника на 5m: 6 місяців, чесний список монет", "",
             f"{tr['ticker'].nunique()} монет поза навчанням (список станом на кожен місяць), {START} – {END}, лише Binance. Входи — Alert "
             f"робочої моделі {m15['trained_at']} (поріг {m15['threshold']:.3f}) + годинна ≥ {share}; стоп {k:.2f} ATR15, ½ на +3 ATR і б/з, "
             f"решта — {exit_bars} бари 15m без Alert; комісія 0.1%. Ранній вихід: n червоних 5m поспіль, OI у монетах падає на кожній, "
             "без нового хаю з входу. «Зрізано» — угоди A5 з результатом ≥ +5%, що з тим самим входом дали < +1%.", "",
             "| варіант | угод | у плюсі | медіана | PF | PF без топ-3 | ≥ 20 млн | 5–20 млн | рано | по стопу | ≥ +5% | зрізано | утрим., год |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for v in names:
        g = tr[tr["variant"] == v]
        r = g["ret"].sort_values()
        same = g.set_index(["ticker", "ts"])["ret"]
        j = pd.concat([base.rename("a"), same.rename("b")], axis=1, join="inner")
        cut = int(((j["a"] >= 0.05) & (j["b"] < 0.01)).sum()) if v not in ("A5", "A15") else 0
        lines.append(f"| {v} | {len(r)} | {(r > 0).mean():.0%} | {r.median():+.2%} | {pf(r):.2f} | {pf(r.iloc[:-3]):.2f} | "
                     f"{pf(g[g['band'] == '≥ 20 млн']['ret']):.2f} | {pf(g[g['band'] == '5–20 млн']['ret']):.2f} | "
                     f"{(g['reason'] == 'early').mean():.0%} | {(g['reason'] == 'stop').mean():.0%} | {int((r >= 0.05).sum())} | {cut} | "
                     f"{g['bars'].median() / 4:.1f} |")
    piv = tr.groupby(["month", "variant"])["ret"].apply(pf).unstack()[names]
    lines += ["", "## По місяцях (PF)", "", "| місяць | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for m, r in piv.iterrows():
        lines.append(f"| {m} | " + " | ".join(f"{r[v]:.2f}" for v in names) + " |")
    lines += ["", "Місяців, де варіант кращий за A5: " + ", ".join(f"{v} {int((piv[v] > piv['A5'] + 1e-9).sum())} з {len(piv)}"
                                                          for v in names if v not in ("A5", "A15")), ""]
    open(OUT, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Звіт: {OUT}")


if __name__ == "__main__":
    main()
