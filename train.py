"""
train.py — навчання і чесне порівняння моделей на таблиці features.

Драбина складності (кожен наступний щабель має обіграти попередній):
    rule      просте правило: OI і обсяг вище норми (пороги підбираються на тренувальних тікерах)
    logreg    логістична регресія з class weights
    lightgbm  градієнтний бустинг неглибоких дерев

Валідація — leave-ticker-out: кожен тікер оцінює модель, яка його не бачила.
Поріг ймовірності для сигналу теж підбирається без тестового тікера (вкладена leave-ticker-out
на тренувальних тікерах), інакше метрики були б завищені.

    python train.py              # усі моделі, звіт у reports/train_report.md, моделі для скринера в models/
    python train.py --models logreg lightgbm
    python train.py --no-save    # лише оцінка, без збереження моделей

Для скринера зберігається модель, навчена на всіх розмічених тікерах. Її поріг — найкращий F1
на out-of-fold передбаченнях (кожен тікер оцінений моделлю, що його не бачила).
"""

import argparse
import os
import sys
from datetime import datetime, timezone
from itertools import product

import joblib
import numpy as np
import pandas as pd
import psycopg
from dotenv import load_dotenv
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURES, load_pumps, query_df
from risk import stop_multiplier

REPORT_FILE = os.path.join("reports", "train_report.md")
FRESH_HOURS = 48   # останні години без розмітки не йдуть у навчання (ще не підтверджені)
PREDICTIONS_FILE = os.path.join("reports", "oof_predictions.csv")
MODELS_DIR = "models"


def model_path(name: str) -> str:
    return os.path.join(MODELS_DIR, f"{name}.joblib")
H = pd.Timedelta(hours=1)


# ---------------------------------------------------------------- моделі

class RuleModel:
    """Baseline трейдера: сигнал, коли OI і обсяг одночасно вище своєї тижневої норми.
    «Навчання» — перебір порогів z-score і вибір пари з найкращим F1 на тренувальних даних."""

    OI_GRID = np.arange(1.0, 6.01, 0.5)
    VOL_GRID = np.arange(0.0, 4.01, 0.5)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        oi, vol = X["oi_z_7d"].to_numpy(), X["volume_z_7d"].to_numpy()
        best = (-1.0, None)
        for a, b in product(self.OI_GRID, self.VOL_GRID):
            f1 = f1_score(y, (oi > a) & (vol > b), zero_division=0)
            if f1 > best[0]:
                best = (f1, (a, b))
        self.oi_thr, self.vol_thr = best[1]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return ((X["oi_z_7d"] > self.oi_thr) & (X["volume_z_7d"] > self.vol_thr)).to_numpy()


def make_logreg():
    # пропуски → медіана, масштабування (регресія чутлива до масштабу ознак),
    # class_weight="balanced": помилка на рідкісному класі «коштує» дорожче
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", C=0.1, max_iter=5000)),
    ])


def make_lightgbm():
    # неглибокі дерева і сильна регуляризація: пампів мало, модель легко запам'ятати конкретні випадки
    return LGBMClassifier(
        n_estimators=300, learning_rate=0.03, num_leaves=7, max_depth=3, min_child_samples=40,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
        class_weight="balanced", random_state=42, verbose=-1,
    )


PROBA_MODELS = {"logreg": make_logreg, "lightgbm": make_lightgbm}
ALL_MODELS = ["rule", *PROBA_MODELS]


# ---------------------------------------------------------------- валідація

def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    return float(thr[np.argmax(f1)])


def oof_proba(make, X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> np.ndarray:
    """Out-of-fold ймовірності: кожен тікер передбачає модель, навчена на решті."""
    p = np.zeros(len(y))
    for g in groups.unique():
        test = (groups == g).to_numpy()
        p[test] = make().fit(X[~test], y[~test]).predict_proba(X[test])[:, 1]
    return p


def leave_ticker_out(name: str, X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> pd.DataFrame:
    """Передбачення для кожного рядка від моделі, яка не бачила його тікера.
    Повертає score (ймовірність або 0/1 для правила), signal і використаний поріг."""
    score = np.full(len(X), np.nan)
    signal = np.zeros(len(X), dtype=bool)
    threshold = np.full(len(X), np.nan)
    for g in sorted(groups.unique()):
        test = (groups == g).to_numpy()
        Xtr, ytr, gtr = X[~test], y[~test], groups[~test]
        if name == "rule":
            m = RuleModel().fit(Xtr, ytr)
            signal[test] = m.predict(X[test])
            score[test] = signal[test]
            print(f"    {g:<9} пороги: OI z > {m.oi_thr:.1f}, обсяг z > {m.vol_thr:.1f}")
            continue
        make = PROBA_MODELS[name]
        # поріг — з out-of-fold передбачень на тренувальних тікерах (вкладена валідація)
        thr = best_f1_threshold(ytr.to_numpy(), oof_proba(make, Xtr, ytr, gtr))
        score[test] = make().fit(Xtr, ytr).predict_proba(X[test])[:, 1]
        signal[test] = score[test] >= thr
        threshold[test] = thr
        print(f"    {g:<9} поріг ймовірності {thr:.3f}")
    return pd.DataFrame({"score": score, "signal": signal, "threshold": threshold}, index=X.index)


# ---------------------------------------------------------------- метрики

def signal_episodes(df: pd.DataFrame, signal: pd.Series) -> pd.DataFrame:
    """Сусідні години з сигналом одного тікера → один епізод (одна «угода»)."""
    rows = []
    for base, g in df[signal.to_numpy()].groupby("base_asset"):
        ts = g["ts"].sort_values()
        run_id = (ts.diff() != H).cumsum()
        for _, run in ts.groupby(run_id):
            rows.append((base, run.iloc[0], run.iloc[-1] + H))
    return pd.DataFrame(rows, columns=["base_asset", "start", "end"])


def event_metrics(df: pd.DataFrame, signal: pd.Series, pumps: pd.DataFrame) -> dict:
    """Метрики з погляду трейдера: спіймані пампи, затримка, хибні епізоди."""
    covered = pumps[pumps["base_asset"].isin(df["base_asset"].unique())]
    sig_ts = df.loc[signal.to_numpy(), ["base_asset", "ts"]]

    delays, caught_fs = [], 0
    n_pump = n_fs = 0
    for p in covered.itertuples():
        inside = sig_ts[(sig_ts["base_asset"] == p.base_asset) & (sig_ts["ts"] >= p.start_ts) & (sig_ts["ts"] < p.end_ts)]
        if p.pattern_type == "pump":
            n_pump += 1
            if len(inside):
                delays.append((inside["ts"].min() - p.start_ts) / H)
        elif p.pattern_type == "false_start":
            n_fs += 1
            caught_fs += bool(len(inside))

    eps = signal_episodes(df, signal)
    labeled = covered.rename(columns={"start_ts": "ps", "end_ts": "pe"})
    false_eps = 0
    for e in eps.itertuples():
        own = labeled[labeled["base_asset"] == e.base_asset]
        if not ((own["ps"] < e.end) & (own["pe"] > e.start)).any():
            false_eps += 1
    days = df.groupby("base_asset")["ts"].agg(lambda s: (s.max() - s.min()) / H / 24).sum()

    return {
        "pumps_caught": f"{len(delays)}/{n_pump}",
        "median_delay_h": float(np.median(delays)) if delays else np.nan,
        "false_starts_triggered": f"{caught_fs}/{n_fs}",
        "false_alarms_per_30d": false_eps / days * 30 if days else np.nan,
    }


def hour_metrics(y: pd.Series, res: pd.DataFrame, has_proba: bool) -> dict:
    return {
        "precision": precision_score(y, res["signal"], zero_division=0),
        "recall": recall_score(y, res["signal"], zero_division=0),
        "f1": f1_score(y, res["signal"], zero_division=0),
        "pr_auc": average_precision_score(y, res["score"].astype(float)) if has_proba else np.nan,
    }


# ---------------------------------------------------------------- важливість ознак

def feature_importance(X: pd.DataFrame, y: pd.Series, models: list[str]) -> pd.DataFrame:
    """Моделі, навчені на всіх даних: ваги регресії (на стандартизованих ознаках) і gain LightGBM."""
    imp = pd.DataFrame(index=X.columns)
    if "logreg" in models:
        imp["logreg_coef"] = make_logreg().fit(X, y).named_steps["clf"].coef_[0]
    if "lightgbm" in models:
        m = make_lightgbm().fit(X, y)
        gain = m.booster_.feature_importance(importance_type="gain")
        imp["lightgbm_gain_%"] = 100 * gain / gain.sum()
    key = "lightgbm_gain_%" if "lightgbm_gain_%" in imp else "logreg_coef"
    return imp.reindex(imp[key].abs().sort_values(ascending=False).index)


# ---------------------------------------------------------------- звіт

def fmt(v, digits=3):
    if isinstance(v, str):
        return v
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{digits}f}"


def md_table(df: pd.DataFrame, digits=3) -> str:
    head = "| " + " | ".join([df.index.name or ""] + list(df.columns)) + " |"
    sep = "|" + "---|" * (len(df.columns) + 1)
    rows = ["| " + " | ".join([str(i)] + [fmt(v, digits) for v in r]) + " |" for i, r in zip(df.index, df.to_numpy())]
    return "\n".join([head, sep, *rows])


def main() -> None:
    parser = argparse.ArgumentParser(description="Leave-ticker-out порівняння моделей")
    parser.add_argument("--models", nargs="+", choices=ALL_MODELS, default=ALL_MODELS)
    parser.add_argument("--no-save", action="store_true", help="не зберігати моделі для скринера")
    parser.add_argument("--fresh-hours", type=int, default=FRESH_HOURS,
                        help="скільки останніх годин без розмітки не брати в навчання (0 — брати все)")
    args = parser.parse_args()

    load_dotenv()
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        sys.exit("Заповни DATABASE_URL у файлі .env")
    with psycopg.connect(db_url) as conn:
        df = query_df(conn, "SELECT * FROM features ORDER BY base_asset, ts")
        pumps = load_pumps(conn)
        risk_info = stop_multiplier(conn)   # k для стопу: 75-й перцентиль MAE пампів у NATR
    if df.empty:
        sys.exit("Таблиця features порожня — спершу python features.py")

    # Свіжі години без розмітки ще не підтверджені: памп, що саме розвивається, пішов би в навчання як «0»,
    # і модель «розвчилась» би бачити його (так зник live-Alert QNT після перенавчання). Години всередині
    # розмічених інтервалів лишаємо — їх підтвердив власник розмітки.
    cutoff = pd.Timestamp(datetime.now(timezone.utc)) - pd.Timedelta(hours=args.fresh_hours)
    fresh_unlabeled = (df["ts"] > cutoff) & df["pattern_type"].isna()
    n_fresh_dropped = int(fresh_unlabeled.sum()) if args.fresh_hours else 0
    if args.fresh_hours and fresh_unlabeled.any():
        print(f"Не беру в навчання {int(fresh_unlabeled.sum())} свіжих нерозмічених годин (після {cutoff:%m-%d %H:%M} UTC)")
        df = df[~fresh_unlabeled].reset_index(drop=True)
    labeled_recent = df[(df["ts"] > cutoff) & df["pattern_type"].notna()]
    extra = [(b, g["ts"].min(), g["ts"].max()) for b, g in labeled_recent.groupby("base_asset")]

    X, y, groups = df[FEATURES], df["label"].astype(int), df["base_asset"]
    tickers = sorted(groups.unique())
    # з якими налаштуваннями навчено — у файл моделі, звіт і дашборд, щоб потім було видно, чому модель така
    settings = {
        "fresh_hours": args.fresh_hours, "fresh_cutoff": cutoff if args.fresh_hours else None,
        "fresh_hours_dropped": n_fresh_dropped, "models": args.models, "n_features": len(FEATURES),
        "hours": int(len(df)), "pump_hours": int(y.sum()), "tickers": len(tickers),
    }
    print(f"Дані: {len(df)} годин, {len(tickers)} тікерів, пампів {y.sum()} год ({y.mean():.1%})\n")

    summary, per_ticker, oof = {}, {}, {}
    preds = df[["base_asset", "ts", "label", "pattern_type"]].copy()
    for name in args.models:
        print(f"  {name}")
        res = leave_ticker_out(name, X, y, groups)
        oof[name] = res["score"].to_numpy()
        preds[f"{name}_score"], preds[f"{name}_signal"] = res["score"], res["signal"].astype(int)
        preds[f"{name}_threshold"] = res["threshold"]
        summary[name] = {**hour_metrics(y, res, name != "rule"), **event_metrics(df, res["signal"], pumps)}
        per_ticker[name] = pd.DataFrame({
            t: {"f1": f1_score(y[groups == t], res["signal"][groups == t], zero_division=0),
                **event_metrics(df[groups == t], res["signal"][groups == t], pumps)}
            for t in tickers}).T

    summary = pd.DataFrame(summary).T
    summary.index.name = "model"
    imp = feature_importance(X, y, [m for m in args.models if m != "rule"])
    imp.index.name = "feature"

    print("\nПідсумок (leave-ticker-out):")
    print(summary.to_string())
    for name, t in per_ticker.items():
        print(f"\n{name} по тікерах:")
        print(t.to_string())
    if len(imp.columns):
        print("\nВажливість ознак (моделі на всіх даних):")
        print(imp.head(12).round(3).to_string())

    os.makedirs(os.path.dirname(REPORT_FILE), exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    parts = [
        "# Training report", "",
        f"Generated {now}. Leave-ticker-out validation: every ticker is scored by a model that never saw it; "
        "the probability threshold is chosen by nested leave-ticker-out on the training tickers only.", "",
        f"Data: {len(df)} hourly bars, {len(tickers)} tickers ({', '.join(tickers)}), "
        f"{y.sum()} pump hours ({y.mean():.1%}), {(pumps['pattern_type'] == 'pump').sum()} pump intervals.", "",
        "## Summary", "",
        "Hour level: precision / recall / F1 of the `pump in progress` class; PR-AUC is threshold-free. "
        "Event level: pumps with at least one signal inside, median hours from pump start to the first signal, "
        "false starts that triggered a signal, and signal episodes outside any labeled interval per 30 days of one ticker.", "",
        md_table(summary),
        "", "## Training settings", "",
        f"- `--fresh-hours {args.fresh_hours}`: " + (
            f"{n_fresh_dropped} unlabeled hours after {cutoff:%Y-%m-%d %H:%M} UTC left out of training"
            if args.fresh_hours else "all hours used, including fresh unlabeled ones"),
        f"- models: {', '.join(args.models)}; {len(FEATURES)} features; saved for the screener: {'no' if args.no_save else 'yes'}",
    ]
    for name, t in per_ticker.items():
        t.index.name = "ticker"
        parts += ["", f"## {name} by ticker", "", md_table(t)]
    if len(imp.columns):
        parts += ["", "## Feature importance", "",
                  "Models fitted on all data. Logistic regression coefficients are on standardised features "
                  "(sign = direction of effect). LightGBM gain is the share of total split gain.", "",
                  md_table(imp)]
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(parts) + "\n")
    # out-of-fold передбачення: для аналізу помилок (кожен рядок передбачила модель, що не бачила тікер)
    preds.to_csv(PREDICTIONS_FILE, index=False)
    print(f"\nЗвіт: {REPORT_FILE}\nПередбачення: {PREDICTIONS_FILE}")

    if not args.no_save:
        os.makedirs(MODELS_DIR, exist_ok=True)
        for name in [m for m in args.models if m in PROBA_MODELS]:
            bundle = {
                "name": name,
                "model": PROBA_MODELS[name]().fit(X, y),
                "features": FEATURES,
                "threshold": best_f1_threshold(y.to_numpy(), oof[name]),
                "trained_at": now,
                "tickers": tickers,
                # до data_end години тікерів з навчання — in-sample; після — лише розмічені інтервали з in_sample_extra
                "data_end": min(df["ts"].max(), cutoff) if args.fresh_hours else df["ts"].max(),
                "in_sample_extra": extra,
                "pump_intervals": int((pumps["pattern_type"] == "pump").sum()),
                "validation": {k: (v if isinstance(v, str) else float(v)) for k, v in summary.loc[name].items()},
                "risk": risk_info,
                "settings": settings,
            }
            joblib.dump(bundle, model_path(name))
            print(f"Модель для скринера: {model_path(name)} (поріг {bundle['threshold']:.3f})")
        if risk_info.get("k") is not None:
            print(f"Стоп: {risk_info['k']:.2f} NATR (75-й перцентиль MAE, {risk_info['n']} пампів)")


def load_model(name: str) -> dict:
    """Збережена модель з метаданими: model, features, threshold, trained_at, validation."""
    bundle = joblib.load(model_path(name))
    if bundle["features"] != FEATURES:
        raise ValueError(f"{name}: модель навчена на іншому наборі ознак — перезапусти python train.py")
    return bundle


if __name__ == "__main__":
    main()
