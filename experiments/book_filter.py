"""
experiments/book_filter.py — фільтр входів моделі за стаканом і спотом: усі ознаки разом, а не поодинці (2026-10-02).

experiments/book_spot.py: поодинці чесну перевірку пройшов лише перекіс стакану ±2% (PF 1.05 проти 0.85 на другій половині).
Тут — невелика модель-фільтр на всіх ознаках стакану й споту з reports/book_spot_features.csv (1431 угода робочої моделі,
частковий вихід, монети поза навчанням): ціль — угода в плюсі. Чесна перевірка за часом, в обидва боки:
    fold А: навчання — угоди до середини періоду, тест — після;   fold Б: навпаки
Моделі: логістична регресія (сильна регуляризація — угод лише ~700 на навчання) і неглибокий LightGBM.
На тесті — угоди, де прогноз фільтра у верхніх 70 / 50 / 30% (пороги — квантилі прогнозів на НАВЧАННІ).
Для порівняння — усі угоди і правило «перекіс ±2% ≥ порогу з навчання».

    python experiments/book_filter.py
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA = os.path.join("reports", "book_spot_features.csv")
COLS = ["imb_02", "imb_1", "imb_2", "imb_5", "d_imb_1_15m", "depth_1_rel", "ask_1_rel", "bid_1_rel",
        "has_spot", "spot_share_4h", "spot_buy_1h", "premium", "premium_chg_4h"]
MAKERS = {
    "логрегресія": lambda: Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                                     ("m", LogisticRegression(C=0.05, max_iter=2000))]),
    "LightGBM": lambda: LGBMClassifier(n_estimators=150, learning_rate=0.03, num_leaves=4, max_depth=2, min_child_samples=40,
                                       subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=42, verbose=-1),
}


def pf(v: pd.Series) -> float:
    gl = -v[v <= 0].sum()
    return v[v > 0].sum() / gl if gl else np.nan


def line(v: pd.Series) -> str:
    return f"{len(v):>4} угод, у плюсі {(v > 0).mean():.0%}, середня {v.mean():+.2%}, PF {pf(v):.2f}, без топ-3 PF {pf(v.sort_values().iloc[:-3]):.2f}"


def main() -> None:
    argparse.ArgumentParser(description="Модель-фільтр входів за стаканом і спотом (чесна перевірка за часом)").parse_args()
    d = pd.read_csv(DATA)
    d["T"] = pd.to_datetime(d["entry_ts"], unit="s", utc=True)
    d = d.sort_values("T").reset_index(drop=True)
    d["y"] = (d["ret"] > 0).astype(int)
    mid = d["T"].iloc[len(d) // 2]
    halves = {"А (вчимо до, тест після)": (d[d["T"] < mid], d[d["T"] >= mid]),
              "Б (вчимо після, тест до)": (d[d["T"] >= mid], d[d["T"] < mid])}
    for fold, (tr, te) in halves.items():
        print(f"\n=== fold {fold}: навчання {len(tr)}, тест {len(te)} (межа {mid:%m-%d %H:%M})")
        print(f"  усі угоди тесту:            {line(te['ret'])}")
        thr = tr["imb_2"].quantile(0.4)
        print(f"  правило перекіс ±2% ≥ {thr:+.3f}: {line(te[te['imb_2'] >= thr]['ret'])}")
        for name, make in MAKERS.items():
            m = make().fit(tr[COLS], tr["y"])
            p_tr, p_te = m.predict_proba(tr[COLS])[:, 1], m.predict_proba(te[COLS])[:, 1]
            print(f"  {name}: AUC тест {roc_auc_score(te['y'], p_te):.3f}")
            for top in (0.7, 0.5, 0.3):
                cut = np.quantile(p_tr, 1 - top)
                print(f"    верхні {top:.0%} прогнозів:    {line(te[p_te >= cut]['ret'])}")
            if name == "логрегресія":
                coef = pd.Series(m.named_steps["m"].coef_[0], index=COLS).sort_values(key=abs, ascending=False)
                print("    найсильніші коефіцієнти: " + ", ".join(f"{k} {v:+.2f}" for k, v in coef.head(5).items()))


if __name__ == "__main__":
    main()
