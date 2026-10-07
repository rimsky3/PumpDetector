"""
movers.py — монети з власним (ідіосинкратичним) рухом: кнопка «Movers ▾» у дашборді (2026-10-07).

Рух монети = ринкова частина (β × індекс альтів) + власна. Тут для ~250 ліквідних альтів (ті самі годинні закриття й
оборот, що market.py) на кожній закритій годині рахуємо:
    beta, r2, resid_sd   OLS годинних log-доходностей монети на рівноважений індекс альтів ALT-EW за 168 год (7 днів),
                         вікно закінчується за EXCL год до «зараз» — поточний сплеск не псує оцінку бети
    tot_H / sys_H / idio_H   доходність за H годин (1 / 4 / 24): уся, систематична (β × індекс), власна (уся − систематична)
    z_H                  власна доходність у σ залишків моделі: idio_H / (resid_sd · √H) — наскільки рух незвичний для монети
    illiq / illiq_x      Amihud ILLIQ: середнє |r_1h| ÷ $-оборот години за 24 год (на $1 млн) і відносно медіани
                         власної ковзної ILLIQ за попередні 30 днів (> 1 — тонше за норму, < 1 — глибше, напр. сплеск обороту)
    turnover_24h         $-оборот за 24 год
Пояснення для власника — notes/movers.md. Без погляду вперед: лише закриті години до моменту розрахунку.
Скринер (market.watcher, щогодини о HH:01) пише reports/movers.json; дашборд — /api/movers.

    python movers.py            # порахувати з кешу market.py і показати топ
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "reports", "movers.json")
BETA_WIN = 168            # 7 днів годинних барів для бети
EXCL = 24                 # вікно бети закінчується за добу до «зараз»
MIN_OBS = 120             # менше спостережень — бета ненадійна, монету пропускаємо
ILLIQ_NORM = 720          # 30 днів для норми ILLIQ
HORIZONS = (1, 4, 24)
TOP = 25                  # скільки монет на кожен горизонт у файлі
Z_MOVER = 2.5             # |z| від цього — «mover» (число на кнопці)
BTC = "BTCUSDT"


def compute(closes: pd.DataFrame, qvol: pd.DataFrame, at: pd.Timestamp | None = None) -> tuple[pd.DataFrame, dict]:
    """Метрики кожної монети на останній закритій годині (або at — відкриття години). Повертає (таблиця, індекс)."""
    idx_t = pd.date_range(closes.index.min(), closes.index.max(), freq="1h")
    alts = closes.drop(columns=[BTC], errors="ignore").reindex(idx_t)
    lr = np.log(alts.where(alts > 0)).diff()
    n = lr.notna().sum(axis=1)
    index = lr.mean(axis=1, skipna=True).where(n >= 50)          # ALT-EW: рівноважене середнє log-доходностей
    if at is not None:
        lr, index = lr[lr.index <= at], index[index.index <= at]
    end = len(lr)
    win = slice(end - EXCL - BETA_WIN, end - EXCL)
    q = qvol.reindex(lr.index) if not qvol.empty else pd.DataFrame(index=lr.index)
    rows = []
    for s in lr.columns:
        y, x = lr[s].iloc[win], index.iloc[win]
        m = y.notna() & x.notna()
        if m.sum() < MIN_OBS or x[m].var() == 0:
            continue
        beta = float(np.cov(x[m], y[m])[0, 1] / x[m].var())
        alpha = float(y[m].mean() - beta * x[m].mean())
        res = y[m] - alpha - beta * x[m]
        sd = float(res.std())
        r2 = float(1 - res.var() / y[m].var()) if y[m].var() > 0 else np.nan
        r = {"symbol": s, "base": s[:-4] if s.endswith("USDT") else s, "beta": beta, "r2": r2, "resid_sd": sd}
        for H in HORIZONS:
            ry, rx = lr[s].iloc[-H:], index.iloc[-H:]
            if ry.isna().any() or rx.isna().any() or not sd > 0:
                r.update({f"tot_{H}": None, f"sys_{H}": None, f"idio_{H}": None, f"z_{H}": None})
                continue
            ty, tx = float(ry.sum()), float(rx.sum())
            r.update({f"tot_{H}": float(np.expm1(ty)), f"sys_{H}": float(np.expm1(beta * tx)),
                      f"idio_{H}": float(np.expm1(ty - beta * tx)), f"z_{H}": float((ty - beta * tx) / (sd * np.sqrt(H)))})
        if s in q:
            il = (lr[s].abs() / q[s].where(q[s] > 0)) * 1e6          # ціновий вплив на $1 млн обороту
            roll = il.rolling(24, min_periods=18).mean()
            norm = roll.iloc[-24 - ILLIQ_NORM:-24].median()
            cur = roll.iloc[-1]
            r.update({"illiq": float(cur) if pd.notna(cur) else None,
                      "illiq_x": float(cur / norm) if pd.notna(cur) and pd.notna(norm) and norm > 0 else None,
                      "turnover_24h": float(q[s].iloc[-24:].sum())})
        rows.append(r)
    ix = {f"idx_{H}": (float(np.expm1(index.iloc[-H:].sum())) if index.iloc[-H:].notna().all() else None) for H in HORIZONS}
    ix["n"] = int(n.iloc[-1]) if len(n) else 0
    ix["ts"] = int((lr.index[-1] + pd.Timedelta(hours=1)).timestamp())    # момент, коли година закрилась
    return pd.DataFrame(rows), ix


def liquidity_label(x: float | None) -> str:
    """Підпис ILLIQ відносно норми монети."""
    if x is None or not np.isfinite(x):
        return "n/a"
    return "Deep" if x < 0.6 else "Very thin" if x >= 2.5 else "Thin" if x >= 1.5 else "Normal"


def save(closes: pd.DataFrame, qvol: pd.DataFrame) -> dict:
    """Порахувати й записати reports/movers.json (топ за |z| для кожного горизонту)."""
    df, ix = compute(closes, qvol)
    clean = lambda v: None if v is None or (isinstance(v, float) and not np.isfinite(v)) else v   # noqa: E731
    out = {"ts": ix["ts"], "n": ix["n"], "index": {str(H): ix[f"idx_{H}"] for H in HORIZONS},
           "beta_window_h": BETA_WIN, "z_mover": Z_MOVER, "rows": {}}
    for H in HORIZONS:
        col = f"z_{H}"
        d = df[df[col].notna()].reindex(df[col].abs().sort_values(ascending=False).index).dropna(subset=[col]).head(TOP)
        out["rows"][str(H)] = [{k: clean(v) for k, v in r.items()} | {"liq": liquidity_label(r.get("illiq_x"))}
                               for r in d.to_dict("records")]
        out[f"movers_{H}"] = int((df[col].abs() >= Z_MOVER).sum())
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT + ".tmp", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    os.replace(OUT + ".tmp", OUT)
    return out


def load() -> dict:
    try:
        with open(OUT, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {"error": "немає даних (market.py ще не порахував movers)"}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Монети з власним рухом (бета, ідіосинкратична доходність, Amihud ILLIQ) з кешу market.py")
    ap.add_argument("--h", type=int, choices=HORIZONS, default=4, help="горизонт для показу (за замовчуванням 4 год)")
    args = ap.parse_args()
    import market
    out = save(market.load_closes(), market.load_qvol())
    print(f"{datetime.fromtimestamp(out['ts'], timezone.utc):%Y-%m-%d %H:%M} UTC · {out['n']} альтів · ALT-EW "
          + " · ".join(f"{h}H {v:+.2%}" for h, v in out["index"].items() if v is not None)
          + f" · movers (|z| ≥ {Z_MOVER}, {args.h}H): {out[f'movers_{args.h}']}")
    for r in out["rows"][str(args.h)][:15]:
        print(f"  {r['base']:<10} idio {r[f'idio_{args.h}']:+7.2%}  z {r[f'z_{args.h}']:+5.1f}  tot {r[f'tot_{args.h}']:+7.2%}  "
              f"β {r['beta']:.2f}  R² {r['r2']:.2f}  ILLIQ {r['liq']}" + (f" ×{r['illiq_x']:.1f}" if r.get("illiq_x") else ""))


if __name__ == "__main__":
    main()
