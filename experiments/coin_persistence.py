"""
experiments/coin_persistence.py — ЧИ ТРИМАЄТЬСЯ УСПІХ МОДЕЛІ НА МОНЕТІ і чим найкращі монети відрізняються від найгірших
(2026-10-18; власник: картка AGT «13 угод, 10 у плюсі, +2.6%» — «проаналізуй тікери з найкращим результатом і спільні риси»).

Дані — угоди робочих моделей за 6 міс. (reports/p_deciles_delay_trades.csv, затримка 0; split і market; 416 монет поза навчанням,
квітень–вересень). R — результат угоди в одиницях початкового ризику (k·ATR), тож монети з різною волатильністю порівнянні.

(1) Стійкість (КРИТЕРІЙ ДО ЗАПУСКУ, notes/experiments.md 18.10). Щомісяця червень–вересень: у кожної монети — середній R усіх її
    угод ДО цього місяця (≥ MIN_PAST угод), терцилі монет за ним, PF угод цього місяця у верхній і нижній третині. Стійкість є,
    якщо в split PF верхньої − нижньої ≥ +0.10, той самий знак в обох половинах монет (md5) і в market, ≥ 3 з 4 місяців того
    самого знаку. Окремо — Spearman середнього R монети квітень–червень проти липня–вересня (≥ 5 угод у кожній половині), 95% CI.
(2) Профіль (лише гіпотези): найкраща і найгірша чверть монет за середнім R (≥ MIN_COIN угод) за ознаками, відомими
    заздалегідь: оборот, волатильність, вік ф'ючерса, OI / оборот, капіталізація, FDV / в обігу, мем-категорія (кеш CoinGecko,
    data/coingecko/ — fe_coingecko.py), частота угод.
→ reports/coin_persistence.md

    python experiments/coin_persistence.py
    python experiments/coin_persistence.py --no-profile     # лише стійкість (швидко)
"""

import argparse
import glob
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

TRADES = os.path.join("reports", "p_deciles_delay_trades.csv")
OUT = os.path.join("reports", "coin_persistence.md")
MIN_PAST, MIN_COIN, MIN_HALF = 5, 8, 5
BIG = "≥ 20 млн"
B = 96


def pf(v) -> float:
    v = pd.Series(v, dtype=float)
    loss = -v[v <= 0].sum()
    return float(v[v > 0].sum() / loss) if loss > 0 else np.nan


def persistence(t: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Щомісячні терцилі монет за середнім R попередніх місяців → угоди місяця з позначкою групи."""
    months = sorted(t["month"].unique())
    rows = []
    for m in months[2:]:
        past = t[t["month"] < m].groupby("ticker")["R"].agg(["mean", "count"])
        past = past[past["count"] >= MIN_PAST]
        if len(past) < 9:
            continue
        grp = pd.qcut(past["mean"].rank(method="first"), 3, labels=["низ", "середина", "верх"])
        cur = t[(t["month"] == m) & t["ticker"].isin(grp.index)].copy()
        cur["group"] = cur["ticker"].map(grp).astype(str)
        cur["past_mean"] = cur["ticker"].map(past["mean"])
        rows.append(cur)
    g = pd.concat(rows) if rows else pd.DataFrame()
    res = {}
    if len(g):
        res["pf"] = {k: pf(g[g["group"] == k]["ret"]) for k in ("низ", "середина", "верх")}
        res["n"] = {k: int((g["group"] == k).sum()) for k in ("низ", "середина", "верх")}
        res["diff"] = res["pf"]["верх"] - res["pf"]["низ"]
        res["halves"] = {h: pf(g[(g["group"] == "верх") & (g["half"] == h)]["ret"]) - pf(g[(g["group"] == "низ") & (g["half"] == h)]["ret"])
                         for h in ("A", "B")}
        res["months"] = {m: pf(g[(g["group"] == "верх") & (g["month"] == m)]["ret"]) - pf(g[(g["group"] == "низ") & (g["month"] == m)]["ret"])
                         for m in sorted(g["month"].unique())}
        b = g[g["band"] == BIG]
        res["big"] = pf(b[b["group"] == "верх"]["ret"]) - pf(b[b["group"] == "низ"]["ret"])
    return g, res


def half_corr(t: pd.DataFrame, seed: int = 0) -> tuple[float, float, float, int]:
    """Spearman середнього R монети квітень–червень проти липня–вересня + 95% bootstrap CI по монетах."""
    h1 = t[t["month"] <= "2026-06"].groupby("ticker")["R"].agg(["mean", "count"])
    h2 = t[t["month"] >= "2026-07"].groupby("ticker")["R"].agg(["mean", "count"])
    j = h1.join(h2, lsuffix="1", rsuffix="2", how="inner")
    j = j[(j["count1"] >= MIN_HALF) & (j["count2"] >= MIN_HALF)]
    if len(j) < 10:
        return np.nan, np.nan, np.nan, len(j)
    rho = j["mean1"].rank().corr(j["mean2"].rank())
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(2000):
        s = j.iloc[rng.integers(0, len(j), len(j))]
        bs.append(s["mean1"].rank().corr(s["mean2"].rank()))
    return float(rho), float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5)), len(j)


def coin_traits(coins: list[str]) -> pd.DataFrame:
    """Ознаки монети за квітень–вересень (відомі заздалегідь або повільні): оборот, волатильність, вік, OI / оборот, капіталізація."""
    import idio_features as I
    mp = json.load(open(os.path.join("data", "coingecko", "_map.json"), encoding="utf-8")) if os.path.exists(os.path.join("data", "coingecko", "_map.json")) else {}
    listing = json.load(open(os.path.join("data", "spot15", "_listing.json"), encoding="utf-8")) if os.path.exists(os.path.join("data", "spot15", "_listing.json")) else {}
    start, end = pd.Timestamp(I.START, tz="UTC"), pd.Timestamp(I.END, tz="UTC") + pd.Timedelta(days=1)
    out = []
    for n, c in enumerate(coins, 1):
        p = os.path.join(I.DIR, f"{c}.pkl")
        if not os.path.exists(p):
            continue
        h = pickle.load(open(p, "rb"))["h"]
        h = h[(h.index >= start) & (h.index < end)]
        if len(h) < 10 * B:
            continue
        day_vol = h["volume"].resample("1D").sum()
        lr = np.log(h["price"].where(h["price"] > 0)).diff()
        r = {"ticker": c, "turnover_m": float(day_vol.median() / 1e6),
             "vol_daily": float(lr.groupby(h.index.floor("1D")).std().median() * np.sqrt(B))}
        d0 = listing.get(c)
        r["age_days"] = float((start - pd.Timestamp(d0, tz="UTC")).days) if d0 else np.nan
        files = sorted(glob.glob(os.path.join("data", "binance_archive", "metrics", f"{c}USDT", "*.parquet")))
        if files:
            files = files[:: max(1, len(files) // 30)]                   # ~30 днів рівномірно — для медіани досить
            m = pd.concat([pd.read_parquet(f, columns=["sum_open_interest_value"]) for f in files])
            r["oi_turnover"] = float(m["sum_open_interest_value"].median() / 1e6 / r["turnover_m"]) if r["turnover_m"] > 0 else np.nan
        info = mp.get(c) or {}
        if info.get("id"):
            j = json.load(open(os.path.join("data", "coingecko", "chart", f"{info['id']}.json"), encoding="utf-8"))
            caps = pd.Series([v for _, v in j["market_caps"]], index=pd.to_datetime([t for t, _ in j["market_caps"]], unit="ms", utc=True))
            prices = pd.Series([v for _, v in j["prices"]], index=pd.to_datetime([t for t, _ in j["prices"]], unit="ms", utc=True))
            caps = caps[(caps.index >= start) & (caps.index < end) & (caps > 0)]
            r["mcap_m"] = float(caps.median() / 1e6) if len(caps) else np.nan
            sup = (caps / prices.reindex(caps.index)).median() if len(caps) else np.nan
            cap_sup = info.get("max") or info.get("total")
            r["fdv_ratio"] = float(cap_sup / sup) if cap_sup and sup and sup > 0 else np.nan
            r["meme"] = float(any("meme" in k.lower() for k in info.get("categories") or []))
        out.append(r)
        if n % 100 == 0:
            print(f"  ознаки: {n}/{len(coins)}", flush=True)
    return pd.DataFrame(out).set_index("ticker")


TRAIT_NAMES = {"turnover_m": "оборот, млн $/добу (медіана)", "vol_daily": "волатильність за добу", "age_days": "вік ф'ючерса на 01.04, днів",
               "oi_turnover": "OI ÷ денний оборот", "mcap_m": "капіталізація, млн $", "fdv_ratio": "FDV ÷ капіталізація",
               "meme": "частка мем-монет", "trades_m": "угод на місяць"}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Чи тримається успіх моделі на монеті (6 міс.) і профіль найкращих / найгірших монет")
    ap.add_argument("--no-profile", action="store_true", help="лише перевірка стійкості, без ознак монет")
    args = ap.parse_args()
    t = pd.read_csv(TRADES)
    t = t[t["delay"] == 0]
    L = ["# Чи тримається успіх моделі на монеті", "",
         f"Угоди робочих моделей за 6 міс. (`{TRADES}`, затримка 0), монети поза навчанням. R — результат угоди в одиницях початкового "
         "ризику k·ATR (порівнянний між монетами), PF — на результатах угод у %.", "",
         "**Критерій (до запуску):** split — PF верхньої третини монет (за середнім R попередніх місяців) − нижньої ≥ +0.10, той самий знак "
         "в обох половинах монет і в market, ≥ 3 з 4 місяців того самого знаку; Spearman між половинами періоду — CI не перетинає 0.", "",
         "## 1. Стійкість: монети, найкращі в минулих місяцях, — у наступному місяці", "",
         "| режим | угод | PF низ / середина / верх | верх − низ | ≥ 20 млн верх − низ | половини A / B | по місяцях (червень–вересень) | Spearman H1 ↔ H2 (монет) |",
         "|---|---|---|---|---|---|---|---|"]
    verdict = {}
    groups = {}
    for mode in ("split", "market"):
        tm = t[t["mode"] == mode]
        g, res = persistence(tm)
        groups[mode] = g
        rho, lo, hi, nc = half_corr(tm)
        verdict[mode] = (res, rho, lo, hi)
        L.append(f"| {mode} | {len(g)} | " + " / ".join(f"{res['pf'][k]:.2f}" for k in ("низ", "середина", "верх"))
                 + f" | {res['diff']:+.2f} | {res['big']:+.2f} | {res['halves']['A']:+.2f} / {res['halves']['B']:+.2f} | "
                 + ", ".join(f"{v:+.2f}" for v in res["months"].values()) + f" | {rho:+.2f} [{lo:+.2f}; {hi:+.2f}] ({nc}) |")
        print(L[-1], flush=True)
    rs, rm = verdict["split"][0], verdict["market"][0]
    same = lambda x: np.sign(x) == np.sign(rs["diff"])                      # noqa: E731
    ok = (rs["diff"] >= 0.10 and all(same(v) for v in rs["halves"].values()) and same(rm["diff"])
          and sum(same(v) for v in rs["months"].values()) >= 3)
    ci_ok = verdict["split"][2] > 0
    L += ["", f"**Критерій стійкості: {'ВИКОНАНО' if ok else 'НЕ виконано'}**; Spearman split між половинами періоду — "
          f"{'CI вище нуля' if ci_ok else 'CI перетинає нуль або нижче'}.", ""]

    # картка «як AGT»: монети з ≥ 10 угод за останній місяць-півтора і середнім результатом ≥ +2% — що далі
    sp = t[t["mode"] == "split"].sort_values("ts")
    rows = []
    for m in sorted(sp["month"].unique())[1:-1]:
        win = sp[sp["month"] == m].groupby("ticker")["ret"].agg(["mean", "count", lambda v: (v > 0).mean()])
        win.columns = ["mean", "count", "win"]
        good = win[(win["count"] >= 6) & (win["mean"] >= 0.02)].index
        nxt = sp[(sp["month"] > m) & sp["ticker"].isin(good)]
        nxt_m = sorted(sp["month"].unique())
        nxt = nxt[nxt["month"] == nxt_m[nxt_m.index(m) + 1]]
        rows.append((m, len(good), len(nxt), pf(nxt["ret"]), pf(sp[sp["month"] == nxt_m[nxt_m.index(m) + 1]]["ret"])))
    L += ["## 2. «Як AGT»: монети, де за місяць було ≥ 6 угод split із середнім ≥ +2%, — наступного місяця", "",
          "| місяць відбору | монет | їх угод наступного місяця | PF цих монет | PF усіх монет того місяця |", "|---|---|---|---|---|"]
    for m, ng, nn, p1, p0 in rows:
        L.append(f"| {m} | {ng} | {nn} | {p1:.2f} | {p0:.2f} |")
    L.append("")

    if not args.no_profile:
        coin = sp.groupby("ticker").agg(R=("R", "mean"), n=("R", "size"), ret=("ret", "mean"),
                                        pf=("ret", pf), months=("month", "nunique"))
        coin = coin[coin["n"] >= MIN_COIN]
        coin["trades_m"] = coin["n"] / coin["months"]
        tr = coin_traits(list(coin.index))
        d = coin.join(tr, how="inner")
        q = d["R"].quantile([0.25, 0.75])
        top, bot = d[d["R"] >= q[0.75]], d[d["R"] <= q[0.25]]
        L += ["## 3. Профіль: найкраща і найгірша чверть монет (split, ≥ 8 угод) — лише гіпотези", "",
              f"Монет: {len(d)}; верхня чверть — середній R ≥ {q[0.75]:+.2f} ({len(top)} монет, PF {pf(sp[sp['ticker'].isin(top.index)]['ret']):.2f}), "
              f"нижня — ≤ {q[0.25]:+.2f} ({len(bot)}, PF {pf(sp[sp['ticker'].isin(bot.index)]['ret']):.2f}). Spearman — по всіх монетах, ознака ↔ середній R.", "",
              "| ознака | нижня чверть (медіана) | верхня чверть (медіана) | Spearman з R (усі монети) |", "|---|---|---|---|"]
        for k, name in TRAIT_NAMES.items():
            if k not in d:
                continue
            x = d[[k, "R"]].dropna()
            rho = x[k].rank().corr(x["R"].rank()) if len(x) > 10 else np.nan
            fmt = (lambda v: f"{v:.0%}") if k == "meme" else (lambda v: f"{v:,.2f}" if abs(v) < 100 else f"{v:,.0f}")
            L.append(f"| {name} | {fmt(bot[k].median() if k != 'meme' else bot[k].mean())} | {fmt(top[k].median() if k != 'meme' else top[k].mean())} | "
                     f"{rho:+.2f} (n={len(x)}) |")
        L += ["", "Найкращі монети (верхня чверть): " + ", ".join(f"{c} ({r.n:.0f} угод, {r.ret:+.1%})" for c, r in top.sort_values("R", ascending=False).head(15).iterrows()) + ".", ""]
    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
