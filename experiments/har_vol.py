"""
experiments/har_vol.py — крок 0 «прогнозу волатильності»: чи прогнозує HAR рух найближчих годин краще за ATR, 2026-10-20.

Питання власника: панель «очікуваний рух» (як Movers) і розмір позиції за прогнозом волатильності. Але стоп у нас уже
k · ATR(15m), тобто розмір за фіксованого ризику вже обернено пропорційний ATR. Панель і розмір за HAR мають сенс, лише
якщо HAR прогнозує майбутній рух КРАЩЕ за поточний ATR. Цей скрипт перевіряє саме це, без угод.

Дані: 1m-свічки Binance з архіву (experiments/klines1m.py, квітень – 11.10.2026), усі монети кешу.
Точки прогнозу — кожна закрита година T. Волатильність — з 5m log-доходностей r: v(вікно) = середнє r² (дисперсія на 5m-бар).
    ціль:      log v за [T, T + 4 год) (і за 1 год — опис);
    ATR:       log v ~ a + b · log(ATR(14) 15m ÷ ціна), ATR — як risk.atr (Wilder, 14);
    RV 4h:     log v ~ a + b · log v за останні 4 год (наївний прогноз «як було, так і буде»);
    HAR:       log v ~ a + b1 · log v_1h + b2 · log v_4h + b3 · log v_24h + b4 · log v_7d  (Corsi 2009, у логах);
    HAR-S, HAR + ATR — опис: HAR + година доби (сезонність), HAR + log ATR.
Коефіцієнти — одні на всі монети (pooled OLS), walk-forward: для місяця M — навчання на всіх точках до M (з ембарго 4 год,
щоб ціль навчання не заходила в M), прогноз на M. Тестові місяці: травень – жовтень (6).
Метрика — QLIKE (Patton 2011): mean(RV/F − log(RV/F) − 1), F — прогноз дисперсії (exp(прогнозу логу) з поправкою
логнормального зсуву, оціненою на навчанні); менше — краще. Поруч — MSE логу і кореляція.

КРИТЕРІЙ (записано до запуску, 20.10): HAR краще за ATR (менший QLIKE) в ОБОХ половинах монет (md5) і у ≥ 5 з 6 тестових
місяців. Інакше — панель і розмір за HAR не робимо (вони повторювали б ATR). Смуга ≥ 20 млн — окремим рядком.

    python experiments/har_vol.py                 # → reports/har_vol.md
    python experiments/har_vol.py --limit 20      # перевірка на 20 монетах
"""

import argparse
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import klines1m as K

ATR_PERIOD = 14
MIN_FILL = 0.8            # частка наявних 5m-доходностей у вікні
TEST_MONTHS = ["2026-05", "2026-06", "2026-07", "2026-08", "2026-09", "2026-10"]
EMBARGO = pd.Timedelta(hours=4)
BAND = 20e6
OUT_MD = os.path.join("reports", "har_vol.md")
WIN = {"1h": 12, "4h": 48, "24h": 288, "7d": 2016}      # у 5m-барах


def coin_rows(base: str) -> pd.DataFrame:
    """Точки прогнозу (кожна закрита година) однієї монети: минулі й майбутні дисперсії, ATR, оборот."""
    m1 = K.load(base)
    if len(m1) < 20 * 1440:
        return pd.DataFrame()
    b5 = K.resample(m1, "5min")
    grid = pd.date_range(b5.index[0], b5.index[-1], freq="5min", tz="UTC")
    c = b5["c"].reindex(grid)
    sq = np.log(c).diff() ** 2
    have = sq.notna().astype(float)
    out = {}
    for name, n in WIN.items():                                   # минуле: бари з відкриттям у [T − w, T) → рядок T − 5 хв
        s, k = sq.rolling(n, min_periods=1).sum(), have.rolling(n, min_periods=1).sum()
        out[f"v_{name}"] = (s / k).where(k >= MIN_FILL * n).shift(1)
    for name in ("1h", "4h"):                                    # майбутнє: [T, T + w) → рядок T
        n = WIN[name]
        s = sq[::-1].rolling(n, min_periods=1).sum()[::-1]
        k = have[::-1].rolling(n, min_periods=1).sum()[::-1]
        out[f"y_{name}"] = (s / k).where(k >= MIN_FILL * n)
    d = pd.DataFrame(out, index=grid)
    b15 = K.resample(m1, "15min")
    prev = b15["c"].shift(1)
    tr = pd.concat([b15["h"] - b15["l"], (b15["h"] - prev).abs(), (b15["l"] - prev).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean() / b15["c"]
    atr.index = atr.index + pd.Timedelta(minutes=15)              # відомий на закритті 15m-бару
    d["atr"] = atr.reindex(grid, method="ffill", tolerance=pd.Timedelta(minutes=15))
    turn = m1["v"].resample("1h").sum()
    d["turn30"] = (turn.rolling(30 * 24, min_periods=10 * 24).sum() / 30).shift(1).reindex(grid, method="ffill")
    d = d[d.index.minute == 0]
    d = d.replace(0, np.nan).dropna(subset=["v_1h", "v_4h", "v_24h", "v_7d", "atr", "y_4h"])
    if d.empty:
        return d
    lg = np.log(d[["v_1h", "v_4h", "v_24h", "v_7d", "y_4h", "y_1h", "atr"]])
    lg.columns = ["x1", "x4", "x24", "x168", "y4", "y1", "xatr"]
    lg["base"], lg["turn30"], lg["hour"] = base, d["turn30"], d.index.hour
    return lg.reset_index(names="ts")


MODELS = {"ATR": ["xatr"], "RV 4h": ["x4"], "HAR": ["x1", "x4", "x24", "x168"],
          "HAR-S": ["x1", "x4", "x24", "x168", "HOUR"], "HAR + ATR": ["x1", "x4", "x24", "x168", "xatr"]}


def design(d: pd.DataFrame, cols: list[str]) -> np.ndarray:
    parts = [np.ones(len(d))]
    for c in cols:
        if c == "HOUR":
            parts += [(d["hour"].to_numpy() == h).astype(float) for h in range(1, 24)]
        else:
            parts.append(d[c].to_numpy())
    return np.column_stack(parts)


def walk_forward(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Прогноз дисперсії кожної моделі для тестових місяців (навчання — усе раніше, з ембарго)."""
    df = df.dropna(subset=[target])
    out = []
    for m in TEST_MONTHS:
        start = pd.Timestamp(m + "-01", tz="UTC")
        end = start + pd.offsets.MonthBegin(1)
        tr = df[df["ts"] < start - EMBARGO]
        te = df[(df["ts"] >= start) & (df["ts"] < end)].copy()
        if te.empty or len(tr) < 1000:
            continue
        y = tr[target].to_numpy()
        for name, cols in MODELS.items():
            X = design(tr, cols)
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            s2 = np.var(y - X @ beta)
            te[name] = design(te, cols) @ beta + s2 / 2         # лог прогнозу дисперсії з поправкою на зсув
        te["month"] = m
        out.append(te)
    return pd.concat(out, ignore_index=True)


def qlike(y: np.ndarray, f: np.ndarray) -> float:
    q = np.exp(y - f)                                             # RV / F
    return float(np.mean(q - np.log(q) - 1))


def scores(d: pd.DataFrame, target: str) -> dict:
    y = d[target].to_numpy()
    row = {"точок": len(d), "монет": d["base"].nunique()}
    for name in MODELS:
        row[f"QLIKE {name}"] = round(qlike(y, d[name].to_numpy()), 4)
    row["HAR vs ATR"] = f"{1 - row['QLIKE HAR'] / row['QLIKE ATR']:+.1%}"
    row["corr ATR"] = round(np.corrcoef(y, d["ATR"])[0, 1], 3)
    row["corr HAR"] = round(np.corrcoef(y, d["HAR"])[0, 1], 3)
    return row


def table(rows: list[dict]) -> str:
    cols = list(rows[0])
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join(out)


def half(base: str) -> str:
    return "A" if int(hashlib.md5(base.encode()).hexdigest(), 16) % 2 == 0 else "B"


def main() -> None:
    ap = argparse.ArgumentParser(description="Крок 0: HAR проти ATR як прогноз волатильності наступних 4 год (walk-forward, QLIKE).")
    ap.add_argument("--limit", type=int, default=0, help="лише перші N монет (перевірка)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    bases = K.available()[: args.limit or None]
    parts = []
    for n, base in enumerate(bases, 1):
        parts.append(coin_rows(base))
        if n % 50 == 0:
            print(f"  {n}/{len(bases)} монет", flush=True)
    df = pd.concat([p for p in parts if len(p)], ignore_index=True)
    print(f"точок: {len(df)}, монет: {df['base'].nunique()}", flush=True)

    md = ["# Прогноз волатильності: HAR проти ATR (крок 0)", "",
          f"{df['base'].nunique()} монет Binance, 1m-архів {df['ts'].min():%d.%m} – {df['ts'].max():%d.%m.%Y}; точка прогнозу — кожна "
          "закрита година; ціль — дисперсія 5m-доходностей за наступні 4 год. Walk-forward по місяцях, коефіцієнти спільні для "
          "всіх монет. QLIKE — менше краще; «HAR vs ATR» — на скільки HAR зменшує похибку ATR.", ""]
    res = {}
    for target, label in (("y4", "4 год"), ("y1", "1 год (опис)")):
        p = walk_forward(df, target)
        p["coins"], p["band"] = p["base"].map(half), np.where(p["turn30"] >= BAND, "≥ 20 млн", "< 20 млн")
        res[target] = p
        rows = [{"зріз": "усі", **scores(p, target)}]
        rows += [{"зріз": f"монети {h}", **scores(p[p["coins"] == h], target)} for h in "AB"]
        rows += [{"зріз": b, **scores(p[p["band"] == b], target)} for b in ("≥ 20 млн", "< 20 млн")]
        rows += [{"зріз": m, **scores(p[p["month"] == m], target)} for m in TEST_MONTHS if (p["month"] == m).any()]
        md += [f"## Горизонт {label}", "", table(rows), ""]

    p = res["y4"]
    # «гарячі» моменти: остання година ≥ 3× звичайного для монети за тиждень (опис — саме тут власник торгує)
    hot = p[(p["x1"] - p["x168"]) >= np.log(9)]
    md += ["## Гарячі моменти (опис): остання година рухається ≥ 3× від звичайного тижневого", "",
           table([{"зріз": "гарячі", **scores(hot, "y4")}, {"зріз": "решта", **scores(p.drop(hot.index), "y4")}]), ""]

    beta, *_ = np.linalg.lstsq(design(df.dropna(subset=["y4"]), MODELS["HAR"]), df.dropna(subset=["y4"])["y4"].to_numpy(), rcond=None)
    md += ["## Коефіцієнти HAR (усі дані, 4 год)", "",
           f"log v₄ₕ = {beta[0]:.2f} + {beta[1]:.2f}·log v₁ₕ + {beta[2]:.2f}·log v₄ₕ(мин.) + {beta[3]:.2f}·log v₂₄ₕ + "
           f"{beta[4]:.2f}·log v₇д", ""]

    q = lambda d, m: qlike(d["y4"].to_numpy(), d[m].to_numpy())
    halves = {h: q(p[p["coins"] == h], "HAR") < q(p[p["coins"] == h], "ATR") for h in "AB"}
    months = {m: q(p[p["month"] == m], "HAR") < q(p[p["month"] == m], "ATR") for m in TEST_MONTHS if (p["month"] == m).any()}
    ok = all(halves.values()) and sum(months.values()) >= 5
    md += ["## Критерій (записано до запуску)", "",
           f"- HAR краще за ATR у половинах монет: A — {'так' if halves['A'] else 'НІ'}, B — {'так' if halves['B'] else 'НІ'}",
           f"- місяців, де HAR краще: {sum(months.values())} з {len(months)} (потрібно ≥ 5)", "",
           f"**Висновок за критерієм: {'проходить' if ok else 'НЕ проходить'}.**", ""]
    os.makedirs("reports", exist_ok=True)
    open(OUT_MD, "w", encoding="utf-8").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
