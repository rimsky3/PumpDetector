"""
experiments/entry_model.py — «модель входу» на футпрінті (meta-labeling, 2026-10-07).

Основна модель відповідає «чи триває імпульс» (монета в грі). Модель входу — друга, маленька: лише на барах, де за 6 год
був Кандидат або Alert, відповідає «якщо зайти на закритті цього бару зі стопом під його лоу, чи дійде ціна до +2R раніше
за стоп (12 год)?». Ціль бере історія — ручної розмітки не треба. Правило b (footprint.py) — окремий випадок того, що
модель могла б знайти сама.

Стоп: під лоу бару (−0.1%), але не ближче 0.5 ATR(15m) від закриття — інакше «стоп» вибиває шум. Те саме для всіх
порівнянь. Результат угоди — footprint.outcome (+2R / стоп / 12 год, комісія 0.1%).

Ознаки (усі з даних до закриття бару): футпрінт бару (частки обсягу під лоу / над хаєм 2 год і дельта там, верхня / нижня
третина, POC, місце закриття), прокол і відстані до лоу / хаю 2 год в ATR, обсяг проти норми, дельта бару і двох попередніх,
склад угод (великі ≥ $10k, дрібні < $2.5k, найбільша угода проти норми), стан моделі (s15, s1h, рівень, барів від Alert /
Кандидата), відкат від хаю 6 год і ріст за 1 / 4 год в ATR / %.

Перевірка: монети ділимо на дві групи; вчимо на групі A за першу половину періоду → перевіряємо на групі B за другу, і
навпаки (бари навчання, чиї 12 год заходять у другу половину, викидаємо). Угоди — по одній на монету одночасно (новий вхід
лише після виходу з попередньої), однаково для моделі (топ-10/20/33% оцінок порогом з навчання), правила b, входу
основної моделі (перший бар Alert) і випадкового бару контексту.

Дані: кеш архіву data/flow/ (experiments/flow_archive.py), 15m Binance з бази, оцінки — backtest.tiers_and_bars (робочі
моделі; монети бектесту поза навчанням).

    python experiments/entry_model.py [--days 40]
"""

import argparse
import glob
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import psycopg
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score

import footprint as FP
import train
from backtest import tiers_and_bars
from features import load_pumps
from fetch import read_tickers_file

pd.DataFrame.to_markdown = lambda self, *a, **k: "```\n" + self.to_string() + "\n```"
CACHE = os.path.join("data", "flow")
OUT = os.path.join("reports", "entry_model.md")
MIN_STOP_ATR = 0.5
FEATS = ["below_share", "below_delta", "above_share", "above_delta", "top_share", "bot_share", "poc_pos", "close_pos",
         "pierce_low", "pierce_atr", "dist_L_atr", "dist_H_atr", "range_atr", "vol_ratio", "delta_share", "delta_share_1",
         "delta_share_2", "below_share_1", "big_share", "big_delta", "small_delta", "max_trade_x", "s15", "s1h", "tier",
         "bars_since_alert", "bars_since_cand", "pullback_6h_atr", "ret_1h", "ret_4h", "is_b"]


def load_coin(conn, base: str):
    days = sorted(glob.glob(os.path.join(CACHE, base, "*.pkl")))
    parts = [x for x in (pd.read_pickle(p) for p in days) if x is not None and len(x["bars"])]
    if not parts:
        return None
    flow = pd.concat([x["bars"] for x in parts]).sort_index()
    fp = pd.concat([x["fp"] for x in parts], ignore_index=True)
    sym = FP._binance_symbol(conn, base)
    if sym is None:
        return None
    from features import query_df
    o = query_df(conn, "SELECT ts, o, h, l, c FROM ohlcv WHERE symbol = %s AND interval = '15min' ORDER BY ts", (sym[0],))
    if o.empty:
        return None
    o["ts"] = pd.to_datetime(o["ts"], utc=True)
    b = o.drop_duplicates("ts").set_index("ts").rename(columns={"o": "open", "h": "high", "l": "low", "c": "price"})
    b = b.join(flow, how="inner")
    return b, fp[fp["ts"].isin(b.index)]


def poc_pos(fp: pd.DataFrame, b: pd.DataFrame) -> pd.Series:
    f = fp.assign(vol=fp["buy"] + fp["sell"])
    top = f.loc[f.groupby("ts")["vol"].idxmax(), ["ts", "lvl"]].set_index("ts")["lvl"]
    mid = np.exp((top + 0.5) * FP.LVL)
    return ((mid - b["low"].reindex(top.index)) / (b["high"] - b["low"]).reindex(top.index)).reindex(b.index)


def coin_frame(conn, base: str, m15, m1, days: int) -> pd.DataFrame | None:
    x = load_coin(conn, base)
    th = tiers_and_bars(conn, base, m15, m1, days) if x is not None else None
    if th is None:
        return None
    b = FP.bar_metrics(*x)
    t = th[0]
    b = b.join(t[["s15", "s1h", "tier"]], how="left")
    b = b[b.index >= t.index.min()].copy()
    a = b["atr"]
    sig = b["tier"].isin(["candidate", "alert"])
    alert = b["tier"] == "alert"
    b["ctx"] = sig.astype(int).rolling(FP.CTX_BARS, min_periods=1).max().astype(bool)
    b["tier"] = b["tier"].map({"calm": 0, "candidate": 1, "alert": 2}).fillna(0)
    idx = pd.Series(np.arange(len(b)), index=b.index, dtype=float)

    def since(m: pd.Series) -> pd.Series:           # барів від останнього бару, де m (0 — цей бар); давно / ніколи — 48
        return (idx - idx.where(m).ffill()).fillna(48).clip(upper=48)
    b["bars_since_alert"] = since(alert)
    b["bars_since_cand"] = since(sig)
    b["poc_pos"] = poc_pos(x[1], b)
    rng = (b["high"] - b["low"]).replace(0, np.nan)
    b["close_pos"] = (b["price"] - b["low"]) / rng
    b["pierce_low"] = (b["low"] < b["L"]).astype(int)
    b["pierce_atr"] = ((b["L"] - b["low"]).clip(lower=0)) / a
    b["dist_L_atr"] = (b["price"] - b["L"]) / a
    b["dist_H_atr"] = (b["H"] - b["price"]) / a
    b["range_atr"] = rng / a
    b["delta_share"] = b["delta"] / b["vol"]
    b["delta_share_1"], b["delta_share_2"] = b["delta_share"].shift(1), b["delta_share"].shift(2)
    b["below_share_1"] = b["below_share"].shift(1)
    b["big_share"] = (b["big_buy"] + b["big_sell"]) / b["vol"]
    b["big_delta"] = (b["big_buy"] - b["big_sell"]) / b["vol"]
    b["small_delta"] = (b["small_buy"] - b["small_sell"]) / b["vol"]
    b["max_trade_x"] = np.log(b["max_trade"] / b["max_trade"].shift(1).rolling(96, min_periods=48).median())
    b["pullback_6h_atr"] = (b["high"].rolling(24).max() - b["price"]) / a
    b["ret_1h"], b["ret_4h"] = b["price"].pct_change(4), b["price"].pct_change(16)
    b["is_b"] = FP.patterns(b)["b"].astype(int)
    first_alert = alert & (alert.astype(int).shift(1, fill_value=0).rolling(4, min_periods=1).max() == 0)
    b["first_alert"] = first_alert
    # ціль: +2R до стопу під лоу бару (не ближче 0.5 ATR), 12 год
    rows = []
    for i in np.flatnonzero(b["ctx"].to_numpy() & b["atr"].notna().to_numpy()):
        if i + FP.HORIZON >= len(b):
            continue
        c = b["price"].iloc[i]
        stop = min(b["low"].iloc[i] * (1 - FP.STOP_PAD), c - MIN_STOP_ATR * a.iloc[i])
        r = FP.outcome(b, i, stop)
        if r:
            rows.append({"ts": b.index[i], "y": int(r["hit2R"]), "R": r["R"], "exit_ts": r["exit_ts"]})
    if not rows:
        return None
    o = pd.DataFrame(rows).set_index("ts")
    out = b.loc[o.index, FEATS + ["first_alert"]].join(o)
    return out.reset_index().assign(base=base)


def trades(df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """По одній угоді на монету одночасно: вхід на першому барі маски після виходу з попередньої."""
    out = []
    for base, g in df[mask].sort_values("ts").groupby("base"):
        free = None
        for r in g.itertuples():
            if free is not None and r.ts < free:
                continue
            out.append(r)
            free = r.exit_ts
    return pd.DataFrame(out)


def stat(t: pd.DataFrame) -> dict:
    if t.empty:
        return {"угод": 0}
    gl = -t["R"][t["R"] <= 0].sum()
    return {"угод": len(t), "монет": t["base"].nunique(), "+2R до стопу": f"{t['y'].mean():.0%}",
            "R сер.": f"{t['R'].mean():+.3f}", "R медіана": f"{t['R'].median():+.2f}",
            "PF (у R)": f"{t['R'][t['R'] > 0].sum() / gl:.2f}" if gl else "∞"}


def main() -> None:
    ap = argparse.ArgumentParser(description="Модель входу: LightGBM на барах контексту, ціль +2R до стопу під лоу бару")
    ap.add_argument("--days", type=int, default=40, help="днів оцінок (кеш архіву — з 28.08)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(".env")
    pd.set_option("display.width", 250)
    m15 = joblib.load(train.model_path(train.M15_NAME))
    m1 = joblib.load(train.model_path("lightgbm"))
    reviewed = {t.upper() for t in read_tickers_file("tickers.txt")}
    coins = set()
    for path in glob.glob(os.path.join("reports", "backtests", "2026*_alert.json")):
        coins |= {u["base"] for u in json.load(open(path, encoding="utf-8"))["meta"]["universe"]}
    cache = os.path.join("reports", "entry_model_frame.pkl")
    if os.path.exists(cache):
        df = pd.read_pickle(cache)
        print(f"рамка з кешу: {len(df)} барів", flush=True)
    else:
        parts = []
        with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
            trained = reviewed | set(load_pumps(conn)["base_asset"])
            coins = sorted((coins - trained) & set(os.listdir(CACHE)))
            print(f"{len(coins)} монет", flush=True)
            for n, base in enumerate(coins, 1):
                try:
                    f = coin_frame(conn, base, m15, m1, args.days)
                except Exception as e:
                    print(f"  ! {base}: {e}", flush=True)
                    conn.rollback()
                    f = None
                if f is not None:
                    parts.append(f)
                if n % 40 == 0:
                    print(f"  {n}/{len(coins)}", flush=True)
        df = pd.concat(parts, ignore_index=True)
        df.to_pickle(cache)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["exit_ts"] = pd.to_datetime(df["exit_ts"], utc=True)
    mid = df["ts"].min() + (df["ts"].max() - df["ts"].min()) / 2
    grp = df["base"].map(lambda s: sum(map(ord, s)) % 2)
    print(f"барів контексту: {len(df)}, монет {df['base'].nunique()}, +2R до стопу в середньому {df['y'].mean():.1%}; "
          f"межа половин {mid:%m-%d %H:%M}", flush=True)

    params = dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=200, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, random_state=42)
    results, imps, aucs = {}, [], []
    test_parts = []
    for g_train in (0, 1):
        tr = df[(grp == g_train) & (df["ts"] < mid) & (df["exit_ts"] < mid)]          # без барів, чиї 12 год у тесті
        te = df[(grp != g_train) & (df["ts"] >= mid)].copy()
        m = lgb.LGBMClassifier(**params).fit(tr[FEATS], tr["y"])
        p_tr = m.predict_proba(tr[FEATS])[:, 1]
        te["score"] = m.predict_proba(te[FEATS])[:, 1]
        for q in (0.90, 0.80, 0.67):
            te[f"top{round((1 - q) * 100)}"] = te["score"] >= np.quantile(p_tr, q)
        aucs.append(roc_auc_score(te["y"], te["score"]))
        imps.append(pd.Series(m.booster_.feature_importance("gain"), index=FEATS))
        test_parts.append(te)
        print(f"  фолд {g_train}: навчання {len(tr)} барів (+2R {tr['y'].mean():.1%}), тест {len(te)}, AUC {aucs[-1]:.3f}", flush=True)
    te = pd.concat(test_parts, ignore_index=True)
    rnd = np.random.default_rng(42).random(len(te)) < 0.125
    variants = {"модель входу · топ-10%": te["top10"], "модель входу · топ-20%": te["top20"], "модель входу · топ-33%": te["top33"],
                "правило b": te["is_b"] == 1, "b І модель входу топ-33%": (te["is_b"] == 1) & te["top33"],
                "вхід основної моделі (1-й бар Alert)": te["first_alert"], "випадковий бар контексту (1/8)": pd.Series(rnd, index=te.index)}
    for name, msk in variants.items():
        results[name] = stat(trades(te, msk))
    # стійкість: окремо кожен тестовий фолд
    by_fold = {}
    for k, part in enumerate(test_parts):
        part = part.copy()
        for name, col in (("модель топ-20%", "top20"),):
            by_fold[f"{name} · фолд {k}"] = stat(trades(part, part[col]))
        by_fold[f"правило b · фолд {k}"] = stat(trades(part, part["is_b"] == 1))
        by_fold[f"вхід основної моделі · фолд {k}"] = stat(trades(part, part["first_alert"]))
    imp = sum(imps)
    imp = (imp / imp.sum()).sort_values(ascending=False)
    res = pd.DataFrame(results).T
    bf = pd.DataFrame(by_fold).T
    print(f"\nAUC на тесті: {aucs[0]:.3f} / {aucs[1]:.3f}\n" + res.to_string() + "\n\nПо фолдах:\n" + bf.to_string(), flush=True)
    print("\nВажливість, топ-15:\n" + "\n".join(f"  {f:<18} {v:.1%}" for f, v in imp.head(15).items()), flush=True)
    lines = [f"# Модель входу на футпрінті ({datetime.now(timezone.utc):%Y-%m-%d})", "",
             f"{df['base'].nunique()} монет поза навчанням, бари контексту (Кандидат / Alert за 6 год): {len(df)}; "
             f"+2R до стопу в середньому {df['y'].mean():.1%}. AUC на тесті {aucs[0]:.3f} / {aucs[1]:.3f}.", "",
             "## Угоди на тесті (інші монети, друга половина періоду)", "", res.to_markdown(), "", "## По фолдах", "", bf.to_markdown(), "",
             "## Важливість (gain), топ-15", ""] + [f"- {f}: {v:.1%}" for f, v in imp.head(15).items()]
    os.makedirs("reports", exist_ok=True)
    open(OUT, "w", encoding="utf-8").write("\n".join(lines))
    print(f"\nзвіт: {OUT}")


if __name__ == "__main__":
    main()
