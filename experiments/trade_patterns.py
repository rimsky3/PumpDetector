"""
experiments/trade_patterns.py — чим прибуткові угоди моделі відрізняються від збиткових (2026-10-09, власник: «переглянемо
рухи з хорошим прибутком і пошукаємо неочевидні правила — напр. більше трейдів на тікері або більший NATR = монета в грі»).

Угоди — reports/long_binance_trades.csv (experiments/long_binance.py --universe monthly: робоча модель, 6 міс. архіву Binance,
чесний список монет з делістованими, частковий вихід, комісія; вхід — закриття бару ts). Ознаки на момент входу — з місячних
15m-свічок архіву (data/binance_archive/monthly, є count — кількість угод), строго по бар входу включно.
Набір ознак зафіксовано ДО запуску (не підбирали після). Перевірка: квінтилі ознаки (межі — з першої половини, квіт–черв),
PF у кожному квінтилі окремо для квіт–черв (A) і лип–вер (B); патерн — лише якщо крайній квінтиль відрізняється від загального
PF в один бік в ОБОХ половинах. Звіт — reports/trade_patterns.md, ознаки угод — reports/trade_patterns.csv.

    python experiments/trade_patterns.py
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

TRADES = os.path.join(ROOT, "reports", "long_binance_trades.csv")
KL = os.path.join(ROOT, "data", "binance_archive", "monthly", "klines")
OUT_MD = os.path.join(ROOT, "reports", "trade_patterns.md")
OUT_CSV = os.path.join(ROOT, "reports", "trade_patterns.csv")
D, B = 96, pd.Timedelta(minutes=15)          # 15m-барів на добу
SPLIT = pd.Timestamp("2026-07-01", tz="UTC")

# ознака → опис (порядок = порядок у звіті)
FEATURES = {
    "trades_x_1h": "кількість угод за 1 год ÷ норма (медіана за 7 дн)",
    "trades_x_24h": "кількість угод за 24 год ÷ середня доба за попередні 7 дн",
    "avg_size_x": "середній розмір угоди за 4 год ÷ середній за 7 дн",
    "natr": "NATR: ATR(14) 15m у % ціни",
    "natr_x": "NATR ÷ медіана NATR за 7 дн",
    "range_24h": "діапазон 24 год (хай − лоу) у % ціни",
    "ret_4h": "ріст за 4 год",
    "ret_24h": "ріст за 24 год",
    "dist_hi_24h": "відстань від хаю 24 год",
    "taker_buy_1h": "частка покупок-агресорів за 1 год",
    "turnover_24h": "оборот за 24 год, $",
    "hour": "година входу, UTC",
    "prev_loss_6h": "за 6 год до входу на монеті був збиток (1 / 0)",
}


def pf(r: pd.Series) -> float:
    w, l = r[r > 0].sum(), -r[r <= 0].sum()
    return w / l if l > 0 else np.nan


def klines(ticker: str) -> pd.DataFrame | None:
    files = sorted(glob.glob(os.path.join(KL, f"{ticker}USDT", "15m", "*.parquet")))
    if not files:
        return None
    k = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True).drop_duplicates("open_time")
    k.index = pd.to_datetime(k["open_time"].astype("int64"), unit="ms", utc=True)
    k = k.sort_index()
    return k[["high", "low", "close", "quote_volume", "count", "taker_buy_quote_volume"]].astype(float)


def bar_features(k: pd.DataFrame) -> pd.DataFrame:
    """Ознаки на закритті кожного 15m-бару — лише з цього й попередніх барів."""
    h, l, c, qv, n, tb = (k[x] for x in ("high", "low", "close", "quote_volume", "count", "taker_buy_quote_volume"))
    f = pd.DataFrame(index=k.index)
    n4, n96 = n.rolling(4).sum(), n.rolling(D).sum()
    f["trades_x_1h"] = n4 / (n.shift(4).rolling(7 * D, min_periods=3 * D).median() * 4)
    f["trades_x_24h"] = n96 / (n.shift(D).rolling(7 * D, min_periods=3 * D).mean() * D)
    size_now = qv.rolling(16).sum() / n.rolling(16).sum()
    size_ref = qv.shift(16).rolling(7 * D, min_periods=3 * D).sum() / n.shift(16).rolling(7 * D, min_periods=3 * D).sum()
    f["avg_size_x"] = size_now / size_ref
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()           # Уайлдер, як risk.atr
    f["natr"] = atr / c * 100
    f["natr_x"] = f["natr"] / f["natr"].shift(1).rolling(7 * D, min_periods=3 * D).median()
    f["range_24h"] = (h.rolling(D).max() / l.rolling(D).min() - 1) * 100
    f["ret_4h"] = (c / c.shift(16) - 1) * 100
    f["ret_24h"] = (c / c.shift(D) - 1) * 100
    f["dist_hi_24h"] = (c / h.rolling(D).max() - 1) * 100
    f["taker_buy_1h"] = tb.rolling(4).sum() / qv.rolling(4).sum()
    f["turnover_24h"] = qv.rolling(D).sum()
    return f


def main() -> None:
    argparse.ArgumentParser(description="Патерни прибуткових угод моделі: ознаки на вході, квінтилі, дві половини 6 міс.").parse_args()
    t = pd.read_csv(TRADES)
    t["ts"] = pd.to_datetime(t["ts"], utc=True)
    t["exit_ts"] = t["ts"] + B * (t["bars"] + 1)
    rows, missing = [], 0
    for tk, g in t.groupby("ticker"):
        k = klines(tk)
        if k is None:
            missing += len(g)
            continue
        f = bar_features(k)
        g = g.sort_values("ts").copy()
        g = g.join(f, on="ts")
        # збиток на тій самій монеті, що закрився за 6 год до входу
        prev = []
        for i, r in enumerate(g.itertuples()):
            ex = g.iloc[:i]
            ex = ex[(ex["exit_ts"] <= r.ts + B) & (ex["exit_ts"] > r.ts + B - pd.Timedelta(hours=6))]
            prev.append(int((ex["ret"] < 0).any()))
        g["prev_loss_6h"] = prev
        g["hour"] = (g["ts"] + B).dt.hour
        rows.append(g)
    a = pd.concat(rows, ignore_index=True)
    a["half"] = np.where(a["ts"] < SPLIT, "A", "B")
    a.to_csv(OUT_CSV, index=False)

    lines = ["# Патерни прибуткових угод моделі", "",
             f"{len(a)} угод робочої моделі (6 міс. архіву Binance, чесний список монет, частковий вихід, комісія), "
             f"без свічок архіву — {missing}. A = квітень–червень, B = липень–вересень.", "",
             f"Загальний PF: усе {pf(a['ret']):.2f} · A {pf(a[a.half == 'A']['ret']):.2f} · B {pf(a[a.half == 'B']['ret']):.2f}.", "",
             "## Виграші проти програшів (медіани на вході)", "",
             "| ознака | великі виграші (≥ +5%) | усі виграші | програші | усі |", "|---|---|---|---|---|"]
    big, win, lose = a[a["ret"] >= 0.05], a[a["ret"] > 0], a[a["ret"] <= 0]
    for f, desc in FEATURES.items():
        lines.append(f"| {desc} | {big[f].median():.3g} | {win[f].median():.3g} | {lose[f].median():.3g} | {a[f].median():.3g} |")
    lines += ["", f"Великих виграшів {len(big)}, усіх виграшів {len(win)}, програшів {len(lose)}.", "",
              "## PF за квінтилями (межі — з половини A)", "",
              "Q1 — найменші значення ознаки, Q5 — найбільші. ✓ — крайній квінтиль кращий / гірший за загальний PF в обох половинах (на ≥ 0.10).", ""]
    verdict = []
    for f, desc in FEATURES.items():
        x = a.dropna(subset=[f])
        if f in ("hour", "prev_loss_6h"):
            if f == "hour":
                x = x.assign(q=pd.cut(x[f], [-1, 5, 11, 17, 23], labels=["00–05", "06–11", "12–17", "18–23"]))
            else:
                x = x.assign(q=x[f].map({0: "ні", 1: "так"}))
        else:
            edges = np.unique(np.nanquantile(x[x.half == "A"][f], [0.2, 0.4, 0.6, 0.8]))
            x = x.assign(q=pd.cut(x[f], [-np.inf, *edges, np.inf], labels=[f"Q{i + 1}" for i in range(len(edges) + 1)]))
        lines += [f"### {desc} (`{f}`)", "", "| група | межі | угод A / B | PF A | PF B | медіана ret |", "|---|---|---|---|---|---|"]
        res = []
        for q, gq in x.groupby("q", observed=True):
            ga, gb = gq[gq.half == "A"], gq[gq.half == "B"]
            lo, hi = gq[f].min(), gq[f].max()
            lines.append(f"| {q} | {lo:.3g} … {hi:.3g} | {len(ga)} / {len(gb)} | {pf(ga['ret']):.2f} | {pf(gb['ret']):.2f} | {gq['ret'].median():+.2%} |")
            res.append((str(q), pf(ga["ret"]), pf(gb["ret"]), len(gq)))
        pa, pb = pf(x[x.half == "A"]["ret"]), pf(x[x.half == "B"]["ret"])
        for q, qa, qb, n in (res[0], res[-1]):
            if qa - pa >= 0.10 and qb - pb >= 0.10:
                verdict.append(f"✓ **{desc}: {q} КРАЩЕ** — PF A {qa:.2f} (усе {pa:.2f}), B {qb:.2f} (усе {pb:.2f}), {n} угод")
            elif pa - qa >= 0.10 and pb - qb >= 0.10:
                verdict.append(f"✓ **{desc}: {q} ГІРШЕ** — PF A {qa:.2f} (усе {pa:.2f}), B {qb:.2f} (усе {pb:.2f}), {n} угод")
        lines.append("")
    lines[lines.index("## PF за квінтилями (межі — з половини A)"):lines.index("## PF за квінтилями (межі — з половини A)")] = \
        ["## Що повторилось в обох половинах", ""] + (verdict or ["Нічого: жоден крайній квінтиль не відрізняється в обох половинах."]) + [""]
    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(verdict) or "нічого стабільного")
    print(f"→ {OUT_MD}")


if __name__ == "__main__":
    main()
