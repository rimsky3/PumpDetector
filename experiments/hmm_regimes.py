"""
experiments/hmm_regimes.py — приховані марковські режими (HMM) як ознаки 15m-моделі (2026-10-07).

Ідея (notes «Марковські моделі для класифікатора пампів»): ринок перебуває в прихованих станах (спокій → акумуляція →
імпульс → злив), видно лише прояви — ціна, OI, обсяг. HMM без розмітки вчить, як виглядає кожен стан і як вони
переходять один в одного; ймовірність кожного стану на барі — нова ознака для LightGBM.

Чесність (головні пастки):
  * лише ПРЯМИЙ прохід (filtering): ймовірність стану на барі t — з даних до t включно. Згладжування forward-backward
    (hmmlearn predict_proba, Вітербі) дивиться в майбутнє — на історії «блискуче», наживо не працює;
  * HMM навчається лише на монетах бектесту (не тікерах навчання) і лише на періоді ДО вікна бектесту — тож ні
    leave-ticker-out, ні угоди бектесту вона не бачила;
  * номери станів довільні — впорядковуємо за «імпульсом» (ціна + OI за годину).
hmmlearn під Python 3.14 не збирається — своя реалізація (діагональні гауссіани, Баум–Велш із масштабуванням).

Спостереження на 15m-бар (усі — до бару включно, перетворені в нормальний розподіл квантилями з навчальних даних):
  ret_1h, oi_chg_1h, ret_4h, vol_ratio (обсяг години / медіана годинного обсягу за 7 днів, log), buy_imb (частка купівель − ½).

Порівняння (як level_features.py): base = FEATURES_15M; hmm = + ймовірності станів. 1) leave-ticker-out на тікерах
навчання (рівень «Кандидат»); 2) бектест на монетах поза навчанням за 30 днів (Alert = 15m + робоча 1h, поточний і
частковий вихід); 3) чи передує якийсь стан імпульсам (рання ознака): ймовірності станів за 2 год до старту імпульсу
проти середнього.

    python experiments/hmm_regimes.py [--states 4]
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
import numpy as np
import pandas as pd
import psycopg
from dotenv import load_dotenv
from sklearn.preprocessing import QuantileTransformer

import train
from features import (FEATURES, FEATURES_15M, INTERVAL_15M, aggregate, build_from_raw, build_hourly_own, load_pumps,
                      load_raw, short_features)
from fetch import read_tickers_file
from tiers import tier_frame

# tabulate у .venv немає — таблиці у звіті простим текстом
pd.DataFrame.to_markdown = lambda self, *a, **k: "```\n" + self.to_string() + "\n```"

FEE, DAYS, FIT_DAYS = 0.001, 30, 25
OBS = ["ret_1h", "oi_chg_1h", "ret_4h", "vol_ratio", "buy_imb"]
OUT = os.path.join("reports", "hmm_regimes.md")


# ---------------------------------------------------------------- HMM (діагональні гауссіани)

class DiagHMM:
    def __init__(self, k: int, seed: int = 42):
        self.k, self.rng = k, np.random.default_rng(seed)

    def _logb(self, X):
        d = X[:, None, :] - self.mu[None]
        return -0.5 * ((d ** 2 / self.var[None]).sum(-1) + np.log(2 * np.pi * self.var).sum(-1)[None])

    def _b(self, X):
        lb = self._logb(X)
        m = lb.max(1, keepdims=True)
        return np.exp(lb - m), m[:, 0]

    def fit(self, seqs: list, n_iter: int = 40, tol: float = 1e-4, log=print):
        X = np.vstack(seqs)
        k, dim = self.k, X.shape[1]
        # старт: центри — квантилі «імпульсу» (сума ознак), щоб стани одразу розійшлися; дисперсія — загальна
        score = X[:, :2].sum(1)
        qs = np.quantile(score, np.linspace(0.1, 0.9, k))
        self.mu = np.array([X[np.argmin(np.abs(score - q))] for q in qs]) + self.rng.normal(0, 0.05, (k, dim))
        self.var = np.tile(X.var(0) + 1e-3, (k, 1))
        self.A = np.full((k, k), 0.05 / (k - 1)) + np.eye(k) * (0.95 - 0.05 / (k - 1))
        self.pi = np.full(k, 1.0 / k)
        prev = -np.inf
        for it in range(n_iter):
            g_sum = np.zeros(k); g_x = np.zeros((k, dim)); g_x2 = np.zeros((k, dim)); xi_sum = np.zeros((k, k))
            pi_acc = np.zeros(k); ll = 0.0
            for S in seqs:
                T = len(S)
                b, mx = self._b(S)
                alpha = np.empty((T, k)); c = np.empty(T)
                a = self.pi * b[0]; c[0] = a.sum(); alpha[0] = a / c[0]
                A = self.A
                for t in range(1, T):
                    a = (alpha[t - 1] @ A) * b[t]
                    c[t] = a.sum(); alpha[t] = a / c[t]
                beta = np.empty((T, k)); beta[-1] = 1.0
                for t in range(T - 2, -1, -1):
                    beta[t] = A @ (b[t + 1] * beta[t + 1]) / c[t + 1]
                gamma = alpha * beta
                gamma /= gamma.sum(1, keepdims=True)
                # ξ: сума по t переходів i→j
                bb = b[1:] * beta[1:] / c[1:, None]
                xi_sum += A * (alpha[:-1].T @ bb)
                g_sum += gamma.sum(0); g_x += gamma.T @ S; g_x2 += gamma.T @ (S ** 2)
                pi_acc += gamma[0]
                ll += np.log(c).sum() + mx.sum()
            self.pi = pi_acc / pi_acc.sum()
            self.A = xi_sum / xi_sum.sum(1, keepdims=True)
            self.mu = g_x / g_sum[:, None]
            self.var = np.maximum(g_x2 / g_sum[:, None] - self.mu ** 2, 1e-3)
            log(f"    EM {it + 1}: log-правдоподібність {ll:,.0f}")
            if ll - prev < tol * abs(ll):
                break
            prev = ll
        return self

    def filter(self, X) -> np.ndarray:
        """Ймовірності станів на кожному барі лише з даних до нього включно (прямий прохід)."""
        b, _ = self._b(X)
        out = np.empty((len(X), self.k))
        a = self.pi * b[0]; out[0] = a / a.sum()
        for t in range(1, len(X)):
            a = (out[t - 1] @ self.A) * b[t]
            s = a.sum()
            out[t] = a / s if s > 0 else self.pi
        return out


# ---------------------------------------------------------------- спостереження і ознаки

def observations(h: pd.DataFrame, b: int) -> pd.DataFrame:
    lp, oi = np.log(h["price"]), h["oi_log_idx"]
    v1 = h["volume"].rolling(b).sum()
    norm = v1.shift(b).rolling(7 * 24 * b, min_periods=3 * 24 * b).median()
    br = h["buy_volume"].rolling(b).sum() / h["volume_bv"].rolling(b).sum()
    o = pd.DataFrame({"ret_1h": lp.diff(b), "oi_chg_1h": oi.diff(b), "ret_4h": lp.diff(4 * b),
                      "vol_ratio": np.log(v1 / norm.where(norm > 0)), "buy_imb": br - 0.5}, index=h.index)
    return o.replace([np.inf, -np.inf], np.nan)


class Regimes:
    def __init__(self, k: int):
        self.k = k
        self.qt = QuantileTransformer(output_distribution="normal", n_quantiles=1000, subsample=400_000, random_state=42)
        self.order = None

    def transform(self, o: pd.DataFrame) -> np.ndarray:
        X = self.qt.transform(o[OBS].to_numpy())
        return np.nan_to_num(np.clip(X, -4, 4), nan=0.0)        # немає даних — «звичайне» значення

    def fit(self, obs_list: list, log=print):
        allo = pd.concat(obs_list)
        self.qt.fit(allo[OBS].dropna().to_numpy())
        seqs = [self.transform(o) for o in obs_list if len(o) > 50]
        self.hmm = DiagHMM(self.k).fit(seqs, log=log)
        # впорядкувати стани за «імпульсом» (ціна + OI за годину): 0 — найсильніше падіння, k−1 — найсильніший ріст
        self.order = np.argsort(self.hmm.mu[:, 0] + self.hmm.mu[:, 1])
        return self

    def probs(self, h: pd.DataFrame, b: int) -> pd.DataFrame:
        o = observations(h, b)
        p = self.hmm.filter(self.transform(o))[:, self.order]
        return pd.DataFrame(p, index=h.index, columns=[f"hmm_s{i}" for i in range(self.k)])

    def profile(self) -> pd.DataFrame:
        mu = self.hmm.mu[self.order]
        stay = np.diag(self.hmm.A)[self.order]
        df = pd.DataFrame(mu, columns=OBS, index=[f"hmm_s{i}" for i in range(self.k)]).round(2)
        df["тримається, барів"] = (1 / (1 - stay)).round(1)
        return df


# ---------------------------------------------------------------- угоди (як level_features.py)

def _pf(v: pd.Series) -> str:
    gl = -v[v <= 0].sum()
    return f"{v[v > 0].sum() / gl:.2f}" if gl else "∞"


def stats(df: pd.DataFrame) -> dict:
    v = df["ret"].sort_values()
    if v.empty:
        return {"угод": 0}
    return {"угод": len(v), "у плюсі": f"{(v > 0).mean():.0%}", "медіана": f"{v.median():+.2%}", "без топ-3": f"{v.iloc[:-3].sum():+.0%}",
            "PF": _pf(v), "PF ≥ 20 млн": _pf(df[df["band"] == "≥ 20 млн"]["ret"]),
            "PF 5–20 млн": _pf(df[df["band"] == "5–20 млн"]["ret"]), "по стопу": f"{df['stop'].mean():.0%}"}


def main() -> None:
    ap = argparse.ArgumentParser(description="HMM-режими (ймовірності прихованих станів) як ознаки 15m-моделі")
    ap.add_argument("--states", type=int, default=4, help="скільки прихованих станів (за замовчуванням 4)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(".env")
    reviewed = {t.upper() for t in read_tickers_file("tickers.txt")}
    m15 = joblib.load(train.model_path(train.M15_NAME))
    m1 = joblib.load(train.model_path("lightgbm"))
    k, exit_bars, share = m15.get("risk", {}).get("k"), m15.get("exit_bars", 4), m15.get("confirm", {}).get("share", 0.75)

    # монети бектесту (поза навчанням) і їхні смуги обороту
    coins, band = set(), {}
    for path in glob.glob(os.path.join("reports", "backtests", "2026*_alert.json")):
        try:
            meta = json.load(open(path, encoding="utf-8"))["meta"]
        except Exception:
            continue
        for u in meta.get("universe", []):
            coins.add(u["base"])
            band.setdefault(u["base"], "≥ 20 млн" if meta.get("min_turnover", 0) >= 20e6 else "5–20 млн")

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        pumps = load_pumps(conn)
        trained = reviewed | set(pumps["base_asset"])
        coins = sorted(coins - trained)
        last = conn.execute("SELECT max(ts) FROM ohlcv WHERE interval = '15min'").fetchone()[0]
        end = pd.Timestamp(last)
        start = end - pd.Timedelta(days=DAYS)                     # вікно бектесту
        # 1) HMM — на монетах бектесту, лише ДО вікна бектесту
        print(f"HMM: {args.states} стани, навчання на {len(coins)} монетах бектесту до {start:%m-%d}…", flush=True)
        obs_list = []
        for b in coins:
            raw = load_raw(conn, [b], INTERVAL_15M, since=(start - pd.Timedelta(days=FIT_DAYS + 8)).to_pydatetime())
            h = aggregate(raw, b, "15min")
            if h.empty:
                continue
            o = observations(h, 4)
            o = o[(o.index < start) & (o.index >= start - pd.Timedelta(days=FIT_DAYS))].dropna(subset=["vol_ratio"])
            if len(o) > 200:
                obs_list.append(o)
        print(f"  послідовностей: {len(obs_list)}, барів: {sum(len(o) for o in obs_list):,}", flush=True)
        reg = Regimes(args.states).fit(obs_list)
        prof = reg.profile()
        print("Профілі станів (середні у нормованих одиницях):\n" + prof.to_string(), flush=True)

        def extra_base(h, bph):
            return short_features(h, bph)

        def extra_hmm(h, bph):
            return short_features(h, bph).join(reg.probs(h, bph))

        d = build_from_raw(load_raw(conn, sorted(trained), INTERVAL_15M), pumps, bars_per_hour=4, extra=extra_hmm)
    hmm_cols = [f"hmm_s{i}" for i in range(args.states)]
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=48)
    d = d[d["base_asset"].isin(reviewed) | d["pattern_type"].notna()]
    d = d[(d["ts"] <= cutoff) | d["pattern_type"].notna()].reset_index(drop=True)
    y, groups = d["label"].astype(int), d["base_asset"]
    print(f"навчання: {len(d)} барів 15m, {groups.nunique()} тікерів, імпульсів {int((pumps['pattern_type'] == 'pump').sum())}", flush=True)

    # 3) чи передує якийсь стан імпульсам: за 2 год до старту / перша година імпульсу / усі бари
    pre, first = [], []
    for p in pumps[pumps["pattern_type"] == "pump"].itertuples():
        g = d[d["base_asset"] == p.base_asset].set_index("ts")
        w = g[(g.index >= p.start_ts - pd.Timedelta(hours=2)) & (g.index < p.start_ts)]
        f1 = g[(g.index >= p.start_ts) & (g.index < p.start_ts + pd.Timedelta(hours=1))]
        if len(w):
            pre.append(w[hmm_cols].mean())
        if len(f1):
            first.append(f1[hmm_cols].mean())
    early = pd.DataFrame({"усі бари": d[hmm_cols].mean(), "2 год до імпульсу": pd.DataFrame(pre).mean(),
                          "перша година імпульсу": pd.DataFrame(first).mean(),
                          "бари імпульсу": d[d["label"] == 1][hmm_cols].mean()}).round(3)
    print("Середні ймовірності станів:\n" + early.to_string(), flush=True)

    variants = {"base": FEATURES_15M, "hmm": FEATURES_15M + hmm_cols}
    oof_rows, models = {}, {}
    for name, cols in variants.items():
        print(f"  {name}: {len(cols)} ознак, leave-ticker-out…", flush=True)
        p = train.oof_proba(train.make_lightgbm, d[cols], y, groups)
        thr = train.best_f1_threshold(y.to_numpy(), p)
        ev = train.event_metrics(d, p >= thr, pumps, train.BAR_15M)
        oof_rows[name] = {"поріг": round(thr, 3), "спіймано": ev["pumps_caught"],
                          "затримка, хв": round(ev["median_delay_h"] * 60) if pd.notna(ev["median_delay_h"]) else None,
                          "фальстарти": ev["false_starts_triggered"], "хибних / 30 д": round(ev["false_alarms_per_30d"], 1)}
        models[name] = (train.make_lightgbm().fit(d[cols], y), thr, cols)
    imp = pd.Series(models["hmm"][0].booster_.feature_importance("gain"), index=variants["hmm"])
    imp = (imp / imp.sum()).sort_values(ascending=False)
    print(pd.DataFrame(oof_rows).T.to_string(), flush=True)

    # 2) бектест — вікно після навчання HMM
    import exit_partial as P
    res = {f"{name} · {e}": [] for name in variants for e in ("поточний вихід", "частковий вихід")}
    since = start - pd.Timedelta(days=9)
    print(f"бектест: {len(coins)} монет поза навчанням, {start:%m-%d}…{end:%m-%d}…", flush=True)
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        for n, b in enumerate(coins, 1):
            raw = load_raw(conn, [b], INTERVAL_15M, since=since.to_pydatetime())
            f15 = build_from_raw(raw, None, bars_per_hour=4, extra=extra_hmm)
            d1 = build_hourly_own(conn, [b], since=since.to_pydatetime())
            if f15.empty or d1.empty:
                continue
            s1 = d1[["base_asset", "ts"]].assign(probability=m1["model"].predict_proba(d1[FEATURES])[:, 1], threshold=m1["threshold"])
            h = aggregate(raw, b, "15min")
            h = h[h.index >= start - pd.Timedelta(days=1)]
            for name, (model, thr, cols) in models.items():
                s15 = f15[["base_asset", "ts"]].assign(probability=model.predict_proba(f15[cols])[:, 1], threshold=thr)
                t = tier_frame(s15, s1, share).set_index("ts").sort_index()
                sc = t[t.index >= start]["alert_score"]
                for x in P.simulate(h, sc, k, exit_bars, None, 0, False):
                    res[f"{name} · поточний вихід"].append({"ticker": b, "ret": x["ret"], "stop": x["reason"] == "stop", "band": band.get(b, "")})
                for x in P.simulate(h, sc, k, exit_bars, 3.0, 0.5, True):
                    res[f"{name} · частковий вихід"].append({"ticker": b, "ret": x["ret"], "stop": x["reason"] == "stop", "band": band.get(b, "")})
            if n % 25 == 0:
                print(f"  {n}/{len(coins)}", flush=True)
    bt = {name: (stats(pd.DataFrame(rows)) if rows else {"угод": 0}) for name, rows in res.items()}

    pd.set_option("display.width", 220)
    print("\n1) Leave-ticker-out:\n" + pd.DataFrame(oof_rows).T.to_string())
    print("\n2) Бектест:\n" + pd.DataFrame(bt).T.to_string())
    print("\nВажливість (hmm), топ-12:\n" + "\n".join(f"  {f:<18} {v:.1%}" + ("  ← HMM" if f in hmm_cols else "") for f, v in imp.head(12).items()), flush=True)
    lines = [f"# HMM-режими як ознаки 15m-моделі ({datetime.now(timezone.utc):%Y-%m-%d})", "",
             f"{args.states} стани; HMM навчена на {len(obs_list)} монетах бектесту до {start:%Y-%m-%d} (прямий прохід, без погляду вперед).", "",
             "## Профілі станів", "", prof.to_markdown(), "", "## Середні ймовірності станів", "", early.to_markdown(), "",
             "## Leave-ticker-out (рівень «Кандидат»)", "", pd.DataFrame(oof_rows).T.to_markdown(), "",
             f"## Бектест: {len(coins)} монет поза навчанням, {start:%m-%d}…{end:%m-%d}", "", pd.DataFrame(bt).T.to_markdown(), "",
             "## Важливість ознак (hmm, gain), топ-12", ""]
    lines += [f"- {f}: {v:.1%}" + (" ← HMM" if f in hmm_cols else "") for f, v in imp.head(12).items()]
    os.makedirs("reports", exist_ok=True)
    open(OUT, "w", encoding="utf-8").write("\n".join(lines))
    print(f"\nзвіт: {OUT}")


if __name__ == "__main__":
    main()
