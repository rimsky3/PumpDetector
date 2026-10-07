"""
experiments/market_regime.py — режим РИНКУ (risk on / risk off) і результат угод моделі (2026-10-07).

Питання: PF скаче між місяцями (6-міс. бектест 0.79…1.12). Чи пояснює це стан ринку — тренд і волатильність BTC,
ширина ринку альтів, альти проти BTC? Модель цього не бачить: усі її ознаки — про саму монету.

Дані (нічого з Coinalyze):
    • угоди: reports/long_backtest_trades.csv (6 міс., Bybit, вхід по ринку на Alert, стара модель) і вересневий бектест
      поточної моделі (reports/backtests/<прогін>.json, Binance, частковий вихід);
    • ширина ринку — ціни 205 монет з кешу довгого бектесту (data/long_bt, 15m Bybit, 04.04–02.10);
    • BTC — 1h Binance (REST, кеш data/market_btc.pkl).
Без погляду вперед: ознаки годинні, відомі на закритті години; угода бачить останню годину, що закрилась до входу
(вхід = закриття сигнального 15m-бару). Межі третин — за погодинним рядом усього періоду (для розвідки; правило-фільтр
мусило б брати межі лише з минулого) — тому головний критерій СТАБІЛЬНІСТЬ ПО МІСЯЦЯХ, а не загальний PF.

    python experiments/market_regime.py
    python experiments/market_regime.py --bt reports/backtests/20261003_1216_candidate_market_k2.48.json
"""

import argparse
import glob
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from exchanges import BINANCE, get

LONG_TRADES = os.path.join("reports", "long_backtest_trades.csv")
LONG_BT = os.path.join("data", "long_bt")
BTC_CACHE = os.path.join("data", "market_btc.pkl")
OUT = os.path.join("reports", "market_regime.md")
B15 = pd.Timedelta(minutes=15)
H = pd.Timedelta(hours=1)
START = pd.Timestamp("2026-03-01", tz="UTC")

FEATS = {
    "btc_ret_24h": "BTC за 24 год",
    "btc_vs_ma7d": "BTC до середньої 7 дн",
    "btc_dd_7d": "BTC від хаю 7 дн",
    "btc_vol_x": "волатильність BTC 24 год / норма 30 дн",
    "breadth_24h": "частка альтів у плюсі за 24 год",
    "breadth_7d": "частка альтів у плюсі за 7 дн",
    "alt_med_24h": "медіана альтів за 24 год",
    "alt_vs_btc_24h": "альти мінус BTC за 24 год",
}


def btc_hourly(refresh: bool = False) -> pd.DataFrame:
    if os.path.exists(BTC_CACHE) and not refresh:
        return pd.read_pickle(BTC_CACHE)
    rows, t = [], int(START.timestamp() * 1000)
    end = int(pd.Timestamp.now(tz="UTC").floor("h").timestamp() * 1000)
    while t < end:
        page = get(BINANCE + "/fapi/v1/klines", dict(symbol="BTCUSDT", interval="1h", startTime=t, endTime=end, limit=1500))
        if not page:
            break
        rows += page
        t = page[-1][0] + 3_600_000
    df = pd.DataFrame({"ts": pd.to_datetime([r[0] for r in rows], unit="ms", utc=True), "c": [float(r[4]) for r in rows]})
    df = df.drop_duplicates("ts").set_index("ts").sort_index()
    df = df[df.index + H <= pd.Timestamp.now(tz="UTC")]          # лише закриті години
    df.to_pickle(BTC_CACHE)
    return df


def alt_hourly() -> pd.DataFrame:
    """Закриття години кожної монети (15m Bybit з кешу довгого бектесту) на спільній годинній сітці."""
    closes = {}
    for p in sorted(glob.glob(os.path.join(LONG_BT, "*.pkl"))):
        got = pickle.load(open(p, "rb"))
        if got is None:
            continue
        price = got[1]["price"]
        closes[os.path.basename(p)[:-4]] = price.resample("1h").last()
    w = pd.DataFrame(closes)
    return w.reindex(pd.date_range(w.index.min(), w.index.max(), freq="1h"))


def regime_features(btc: pd.DataFrame, alts: pd.DataFrame) -> pd.DataFrame:
    """Індекс — відкриття години; значення відомі на її закритті."""
    c = btc["c"].reindex(pd.date_range(btc.index.min(), btc.index.max(), freq="1h")).ffill()
    lr = np.log(c).diff()
    vol24 = lr.rolling(24).std()
    f = pd.DataFrame({
        "btc_ret_24h": c / c.shift(24) - 1,
        "btc_vs_ma7d": c / c.rolling(168).mean() - 1,
        "btc_dd_7d": c / c.rolling(168).max() - 1,
        "btc_vol_x": vol24 / vol24.rolling(720, min_periods=240).median(),
    })
    r24, r7 = alts / alts.shift(24) - 1, alts / alts.shift(168) - 1
    n24, n7 = r24.notna().sum(axis=1), r7.notna().sum(axis=1)
    a = pd.DataFrame({
        "breadth_24h": (r24 > 0).sum(axis=1) / n24,
        "breadth_7d": (r7 > 0).sum(axis=1) / n7,
        "alt_med_24h": r24.median(axis=1),
    }).where(n24 >= 50)
    a.loc[n7 < 50, "breadth_7d"] = np.nan
    f = f.join(a, how="left")
    f["alt_vs_btc_24h"] = f["alt_med_24h"] - f["btc_ret_24h"]
    return f


def attach(trades: pd.DataFrame, f: pd.DataFrame) -> pd.DataFrame:
    """Кожній угоді — режим останньої закритої години до входу (вхід = закриття сигнального 15m-бару)."""
    known = f.copy()
    known.index = known.index + H                                  # момент, коли година закрилась
    t = trades.sort_values("decide").reset_index(drop=True)
    return pd.merge_asof(t, known.rename_axis("known_at").reset_index(), left_on="decide", right_on="known_at",
                         direction="backward")


def stats(r: pd.Series, reason: pd.Series | None = None) -> dict:
    win, loss = r[r > 0].sum(), -r[r < 0].sum()
    out = {"n": len(r), "win%": 100 * (r > 0).mean() if len(r) else np.nan, "mean%": 100 * r.mean(),
           "median%": 100 * r.median(), "PF": win / loss if loss > 0 else np.nan}
    if reason is not None and len(r):
        out["stop%"] = 100 * (reason == "stop").mean()
    return out


def fmt_row(name: str, s: dict, days: int | None = None) -> str:
    cells = [name, f"{s['n']}", "" if days is None else f"{days}", f"{s['win%']:.0f}%", f"{s['mean%']:+.2f}%",
             f"{s['median%']:+.2f}%", f"{s['PF']:.2f}", f"{s.get('stop%', np.nan):.0f}%"]
    return "| " + " | ".join(cells) + " |"


HEAD = "| | угод | днів | у плюсі | середня | медіана | PF | по стопу |\n|---|---|---|---|---|---|---|---|"


def by_terciles(t: pd.DataFrame, f: pd.DataFrame, col: str) -> tuple[list[str], dict]:
    """Третини ознаки (межі — з погодинного ряду ринку) → метрики і стабільність по місяцях (нижня vs верхня третина)."""
    q1, q2 = f[col].quantile([1 / 3, 2 / 3])
    b = pd.cut(t[col], [-np.inf, q1, q2, np.inf], labels=["нижня", "середня", "верхня"])
    lines = [f"**{FEATS[col]}** (`{col}`; межі третин {q1:+.3f} / {q2:+.3f})", "", HEAD]
    res = {}
    for lab in ["нижня", "середня", "верхня"]:
        g = t[b == lab]
        s = stats(g["ret"], g.get("reason"))
        res[lab] = s
        lines.append(fmt_row(lab, s, g["decide"].dt.date.nunique()))
    # по місяцях: PF верхньої і нижньої третини; у скількох місяцях верхня краща
    months = []
    for m, g in t.groupby(t["decide"].dt.strftime("%Y-%m")):
        bm = b.loc[g.index]
        lo, hi = g[bm == "нижня"]["ret"], g[bm == "верхня"]["ret"]
        if len(lo) >= 20 and len(hi) >= 20:
            months.append((m, stats(lo)["PF"], stats(hi)["PF"]))
    better = sum(h > l for _, l, h in months)
    lines += ["", "По місяцях (PF нижня → верхня, місяці з ≥ 20 угодами в обох): " +
              "; ".join(f"{m[5:]} {l:.2f} → {h:.2f}" for m, l, h in months) +
              (f" — верхня краща в {better} з {len(months)}" if months else ""), ""]
    res["months_better_hi"], res["months"] = better, len(months)
    return lines, res


def combo(t: pd.DataFrame) -> pd.Series:
    """Проста комбінація з «природними» межами (не з даних): risk on — BTC вище середньої 7 дн і ≥ 50% альтів у плюсі
    за добу; risk off — BTC нижче середньої і < 40% альтів у плюсі; решта — нейтрально."""
    on = (t["btc_vs_ma7d"] > 0) & (t["breadth_24h"] >= 0.5)
    off = (t["btc_vs_ma7d"] < 0) & (t["breadth_24h"] < 0.4)
    return pd.Series(np.where(on, "risk on", np.where(off, "risk off", "нейтрально")), index=t.index)


def report_set(name: str, t: pd.DataFrame, f: pd.DataFrame) -> tuple[list[str], dict]:
    t = t.dropna(subset=["btc_vs_ma7d", "breadth_24h"])
    lines = [f"## {name}", "", f"{len(t)} угод, {t['ticker'].nunique()} монет, {t['decide'].min():%Y-%m-%d} – "
             f"{t['decide'].max():%Y-%m-%d}. Усі разом: " + ", ".join(f"{k} {v:.2f}" for k, v in stats(t['ret']).items() if k != 'n'), ""]
    summary = {}
    for col in FEATS:
        ls, res = by_terciles(t, f, col)
        lines += ls
        summary[col] = res
    c = combo(t)
    lines += ["**Комбінація** (risk on: BTC вище середньої 7 дн і ≥ 50% альтів у плюсі за 24 год; risk off: BTC нижче і "
              "< 40%; межі задано заздалегідь, не з даних)", "", HEAD]
    for lab in ["risk on", "нейтрально", "risk off"]:
        g = t[c == lab]
        lines.append(fmt_row(lab, stats(g["ret"], g.get("reason")), g["decide"].dt.date.nunique()))
    months = []
    for m, g in t.groupby(t["decide"].dt.strftime("%Y-%m")):
        cm = c.loc[g.index]
        on, off = g[cm == "risk on"]["ret"], g[cm == "risk off"]["ret"]
        months.append(f"{m[5:]} on {stats(on)['PF']:.2f} ({len(on)}) / off {stats(off)['PF']:.2f} ({len(off)})")
    lines += ["", "По місяцях: " + "; ".join(months), ""]
    # фільтр «ринок альтів гарячий» з круглими порогами, заданими заздалегідь (не третини з даних)
    lines += ["**Фільтр: пропускати Alert, коли > X% альтів у плюсі за 7 дн** (пороги задано заздалегідь)", "",
              "| поріг | лишаємо: угод / PF / сума / без топ-3 / ≥ +10% | пропускаємо: угод / PF | місяців, де лишаємо краще |",
              "|---|---|---|---|"]
    tb = t.dropna(subset=["breadth_7d"])
    for th in (0.5, 0.6, 0.7):
        k, s = tb[tb["breadth_7d"] <= th], tb[tb["breadth_7d"] > th]
        good = tot = 0
        for _, g in tb.groupby(tb["decide"].dt.strftime("%Y-%m")):
            gk, gs = g[g["breadth_7d"] <= th]["ret"], g[g["breadth_7d"] > th]["ret"]
            if len(gk) >= 20 and len(gs) >= 20:
                tot += 1
                good += stats(gk)["PF"] > stats(gs)["PF"]
        r = np.sort(k["ret"].to_numpy())
        lines.append(f"| {th:.0%} | {len(k)} / {stats(k['ret'])['PF']:.2f} / {100 * r.sum():+.0f}% / {100 * r[:-3].sum():+.0f}% / "
                     f"{int((k['ret'] >= 0.10).sum())} з {int((tb['ret'] >= 0.10).sum())} | {len(s)} / {stats(s['ret'])['PF']:.2f} | {good} з {tot} |")
    lines.append("")
    share = c.value_counts(normalize=True)
    hours = combo(f.dropna(subset=["btc_vs_ma7d", "breadth_24h"]))
    lines += [f"Частка угод: " + ", ".join(f"{k} {100 * v:.0f}%" for k, v in share.items()) +
              " · частка годин ринку: " + ", ".join(f"{k} {100 * v:.0f}%" for k, v in hours.value_counts(normalize=True).items()), ""]
    return lines, summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Режим ринку (BTC, ширина альтів) і результат угод моделі")
    ap.add_argument("--bt", help="вересневий прогін бектесту (за замовчуванням — найновіший current/candidate market ≥ 20 млн)")
    ap.add_argument("--refresh-btc", action="store_true", help="перезавантажити BTC з Binance")
    sys.stdout.reconfigure(encoding="utf-8")         # до parse_args: --help з «≥» у консолі cp1251
    args = ap.parse_args()

    btc = btc_hourly(args.refresh_btc)
    alts = alt_hourly()
    f = regime_features(btc, alts)
    print(f"BTC {btc.index.min():%m-%d}…{btc.index.max():%m-%d}, альтів {alts.shape[1]}, годин {len(f)}")

    lt = pd.read_csv(LONG_TRADES, parse_dates=["ts"])
    lt = lt[lt["variant"] == "як зараз (ринок)"].assign(decide=lambda d: d["ts"] + B15)
    sets = [("6 місяців (стара модель, Bybit, вхід по ринку)", attach(lt, f))]

    bt = args.bt
    if not bt:
        cand = [p for p in glob.glob(os.path.join("reports", "backtests", "*.json")) if "_market" in p]
        meta = {p: json.load(open(p, encoding="utf-8"))["meta"] for p in cand}
        cand = [p for p in cand if meta[p].get("min_turnover", 0) >= 20e6 and not meta[p].get("max_turnover")]
        bt = max(cand, key=os.path.getmtime) if cand else None
    if bt:
        d = json.load(open(bt, encoding="utf-8"))
        tl = pd.DataFrame(d["trades_list"])
        tl = tl.assign(decide=pd.to_datetime(tl["entry_ts"], unit="s", utc=True) + B15)
        sets.append((f"Вересень, {os.path.basename(bt)} (Binance, частковий вихід)", attach(tl, f)))

    lines = ["# Режим ринку (risk on / risk off) і угоди моделі", "",
             "Ознаки — на останній закритій годині до входу. Межі третин — з погодинного ряду всього періоду (розвідка); "
             "головне — чи «краща» третина краща в більшості місяців. «днів» — скільки різних днів у групі: угоди "
             "одного дня не незалежні (ринок рухає всі монети разом).", ""]
    summ = {}
    for name, t in sets:
        ls, s = report_set(name, t, f)
        lines += ls
        summ[name] = s
    # коротко: яка третина краща і чи стабільно
    lines += ["## Коротко", "", "| ознака | " + " | ".join(f"набір {i + 1}: PF нижня / верхня, місяців верхня краща" for i in range(len(sets))) + " |",
              "|---|" + "---|" * len(sets)]
    for col in FEATS:
        cells = []
        for name, _ in sets:
            r = summ[name][col]
            cells.append(f"{r['нижня']['PF']:.2f} / {r['верхня']['PF']:.2f}, {r['months_better_hi']} з {r['months']}")
        lines.append(f"| {FEATS[col]} | " + " | ".join(cells) + " |")
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nЗвіт: {OUT}")


if __name__ == "__main__":
    main()
