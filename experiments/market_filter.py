"""
experiments/market_filter.py — коли моделі краще не торгувати взагалі (2026-10-09, власник: «фільтр для всього ринку, де
моделі краще зачекати»; причина — програші моделі йдуть серіями, experiments/position_sizing.py: Z −7.6).

Угоди — reports/long_binance_trades.csv (20 701 угода робочої моделі, 6 міс. архіву Binance, чесний список монет, частковий
вихід, комісія; вхід — закриття бару ts, вихід — закриття бару ts + bars). Ознаки на момент входу, лише з минулого
(зафіксовано ДО запуску):
    1) крива капіталу моделі: частка виграшів серед останніх 20 угод, що вже закрились (усі монети); PF і середній
       результат угод, закритих за останні 24 год (менше 5 угод — немає значення);
    2) натовп: скільки різних монет отримали вхід моделі за останні 1 / 4 год;
    3) ширина ринку: частка монет архіву (data/binance_archive/monthly, 15m → 1h), дорожчих ніж 24 год / 7 днів тому,
       на останню закриту годину.
Перевірка: квінтилі (межі з квіт–черв), PF окремо квіт–черв (A) і лип–вер (B); для правил «пауза» — PF, кількість угод, сума,
найбільше просідання суми результатів і найдовша серія програшів (у часовому порядку) з фільтром і без.
→ reports/market_filter.md, reports/market_filter.csv

    python experiments/market_filter.py
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
OUT_MD = os.path.join(ROOT, "reports", "market_filter.md")
OUT_CSV = os.path.join(ROOT, "reports", "market_filter.csv")
B = pd.Timedelta(minutes=15)
SPLIT = pd.Timestamp("2026-07-01", tz="UTC")
FEATS = {
    "wr20": "частка виграшів серед останніх 20 закритих угод моделі",
    "pf24": "PF угод моделі, закритих за 24 год",
    "mean24": "середній результат угод, закритих за 24 год",
    "crowd_1h": "монет із входом моделі за останню 1 год",
    "crowd_4h": "монет із входом моделі за останні 4 год",
    "breadth_24h": "частка монет, дорожчих ніж 24 год тому",
    "breadth_7d": "частка монет, дорожчих ніж 7 днів тому",
}


def pf(r: pd.Series) -> float:
    w, l = r[r > 0].sum(), -r[r <= 0].sum()
    return w / l if l > 0 else np.nan


def breadth() -> pd.DataFrame:
    """Годинні закриття всіх монет архіву → частка в плюсі за 24 год і 7 днів (індекс — кінець години)."""
    closes = {}
    for d in glob.glob(os.path.join(KL, "*USDT")):
        files = sorted(glob.glob(os.path.join(d, "15m", "*.parquet")))
        if not files:
            continue
        k = pd.concat([pd.read_parquet(f, columns=["open_time", "close"]) for f in files], ignore_index=True)
        s = pd.Series(k["close"].astype(float).to_numpy(), index=pd.to_datetime(k["open_time"].astype("int64"), unit="ms", utc=True))
        s = s[~s.index.duplicated()].sort_index()
        h = s.resample("1h").last()
        h.index = h.index + pd.Timedelta(hours=1)          # кінець години — коли закриття вже відоме
        closes[os.path.basename(d)] = h
    c = pd.DataFrame(closes).sort_index()
    out = pd.DataFrame(index=c.index)
    for name, lag in (("breadth_24h", 24), ("breadth_7d", 168)):
        r = c / c.shift(lag) - 1
        n = r.notna().sum(axis=1)
        out[name] = ((r > 0).sum(axis=1) / n).where(n >= 50)
    return out


def max_dd(r: pd.Series) -> float:
    eq = r.cumsum()
    return float((eq.cummax() - eq).max())


def longest_loss(r: pd.Series) -> int:
    best = cur = 0
    for x in r:
        cur = cur + 1 if x <= 0 else 0
        best = max(best, cur)
    return best


def main() -> None:
    argparse.ArgumentParser(description="Фільтр «зараз не торгувати»: крива капіталу моделі, натовп сигналів, ширина ринку").parse_args()
    t = pd.read_csv(TRADES)
    t["ts"] = pd.to_datetime(t["ts"], utc=True)
    t["entry"] = t["ts"] + B
    t["exit"] = t["ts"] + B * (t["bars"] + 1)
    t = t.sort_values("entry").reset_index(drop=True)
    # 1) крива капіталу: угоди, закриті до входу
    ex = t.sort_values("exit")
    ex_t, ex_r = ex["exit"].values, ex["ret"].to_numpy()      # .values — datetime64 без поясу (UTC)
    wr20, pf24, mean24 = [], [], []
    for e in t["entry"].values:
        j = np.searchsorted(ex_t, e, side="right")              # закриті не пізніше за момент входу
        last = ex_r[max(0, j - 20):j]
        wr20.append((last > 0).mean() if len(last) == 20 else np.nan)
        i0 = np.searchsorted(ex_t, e - np.timedelta64(24, "h"), side="right")
        win = ex_r[i0:j]
        if len(win) >= 5:
            l = -win[win <= 0].sum()
            pf24.append(win[win > 0].sum() / l if l > 0 else np.nan)
            mean24.append(win.mean())
        else:
            pf24.append(np.nan)
            mean24.append(np.nan)
    t["wr20"], t["pf24"], t["mean24"] = wr20, pf24, mean24
    # 2) натовп: різні монети з входом за 1 / 4 год (до цього входу, без нього самого)
    en = t["entry"].values
    tick = t["ticker"].to_numpy()
    for name, hrs in (("crowd_1h", 1), ("crowd_4h", 4)):
        lo = np.searchsorted(en, en - np.timedelta64(hrs, "h"), side="left")
        t[name] = [len(set(tick[a:i])) for i, a in enumerate(lo)]
    # 3) ширина ринку на останню закриту годину
    br = breadth()
    hour_end = t["entry"].dt.floor("h")
    t = t.assign(**{c: br[c].reindex(hour_end).to_numpy() for c in br.columns})
    t["half"] = np.where(t["entry"] < SPLIT, "A", "B")
    t.to_csv(OUT_CSV, index=False)

    L = ["# Коли моделі краще не торгувати: крива капіталу, натовп, ширина ринку", "",
         f"{len(t)} угод робочої моделі (6 міс. архіву Binance, чесний список). A = квітень–червень, B = липень–вересень. "
         f"PF усіх: A {pf(t[t.half == 'A']['ret']):.2f} · B {pf(t[t.half == 'B']['ret']):.2f}.", ""]
    verdict, body, rules = [], [], []
    for f, desc in FEATS.items():
        x = t.dropna(subset=[f])
        edges = np.unique(np.nanquantile(x[x.half == "A"][f], [0.2, 0.4, 0.6, 0.8]))
        x = x.assign(q=pd.cut(x[f], [-np.inf, *edges, np.inf], labels=[f"Q{i + 1}" for i in range(len(edges) + 1)]))
        body += [f"### {desc} (`{f}`)", "", "| група | межі | угод A / B | PF A | PF B | медіана ret |", "|---|---|---|---|---|---|"]
        res = []
        for q, g in x.groupby("q", observed=True):
            ga, gb = g[g.half == "A"], g[g.half == "B"]
            body.append(f"| {q} | {g[f].min():.3g} … {g[f].max():.3g} | {len(ga)} / {len(gb)} | {pf(ga['ret']):.2f} | "
                        f"{pf(gb['ret']):.2f} | {g['ret'].median():+.2%} |")
            res.append((str(q), pf(ga["ret"]), pf(gb["ret"]), len(g), g[f].min(), g[f].max()))
        pa, pb = pf(x[x.half == "A"]["ret"]), pf(x[x.half == "B"]["ret"])
        for side, (q, qa, qb, n, lo_, hi_) in (("low", res[0]), ("high", res[-1])):
            worse = pa - qa >= 0.10 and pb - qb >= 0.10
            better = qa - pa >= 0.10 and qb - pb >= 0.10
            if worse or better:
                verdict.append(f"✓ **{desc}: {q} {'ГІРШЕ' if worse else 'КРАЩЕ'}** — PF A {qa:.2f} (усе {pa:.2f}), B {qb:.2f} "
                               f"(усе {pb:.2f}), {n} угод")
            if worse:                               # правило «пауза» = не входити в цьому крайньому квінтилі
                cut = edges[0] if side == "low" else edges[-1]
                rules.append((f"пауза, якщо {f} {'≤' if side == 'low' else '>'} {cut:.3g}",
                              (t[f] <= cut) if side == "low" else (t[f] > cut)))
        body.append("")
    # правила «пауза»: кожне окремо і разом
    if len(rules) > 1:
        both = np.zeros(len(t), dtype=bool)
        for _, m in rules:
            both |= m.fillna(False).to_numpy()
        rules.append(("усі паузи разом", pd.Series(both, index=t.index)))
    L += ["## Що повторилось в обох половинах", ""] + (verdict or ["Жоден крайній квінтиль не відрізняється в обох половинах."]) + [""]
    if rules:
        L += ["## Правила «пауза» (угоди в часовому порядку, сума результатів при однаковому розмірі)", "",
              "| правило | половина | угод | PF | сума | найбільше просідання суми | найдовша серія програшів |", "|---|---|---|---|---|---|---|"]
        for name, m in [("без фільтра", pd.Series(False, index=t.index))] + rules:
            m = m.fillna(False)
            for h in ("A", "B"):
                k = t[(t.half == h) & ~m]["ret"]
                L.append(f"| {name} | {h} | {len(k)} | {pf(k):.2f} | {k.sum():+.0%} | {max_dd(k):.0%} | {longest_loss(k)} |")
        L.append("")
    L += ["## Квінтилі", ""] + body
    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(verdict) or "нічого стабільного")
    print(f"→ {OUT_MD}")


if __name__ == "__main__":
    main()
