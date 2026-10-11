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
    python train.py --no-15m     # без 15m-моделі (швидше: лише годинні моделі)

Крім годинних моделей навчається 15m-LightGBM (models/lightgbm_15m.joblib) для двох рівнів сигналу:
    «Кандидат» — 15m-модель вище свого порогу (найраніше, шумніше);
    «Alert»    — 15m-сигнал, підтверджений годинним LightGBM ≥ CONFIRM_SHARE його порогу
                 (оцінка останньої закритої години; experiments/two_tier.py).

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

from features import (FEATURES, FEATURES_15M, INTERVAL_15M, MODEL_FEATURES, MODEL_FEATURES_15M, MOVE_FEATURES, build_15m, in_pump, load_pumps, load_raw, query_df)
from monitor import feature_reference
from risk import EXIT_BARS_15M, entry_stop, stop_multiplier

REPORT_FILE = os.path.join("reports", "train_report.md")
FRESH_HOURS = 48   # останні години без розмітки не йдуть у навчання (ще не підтверджені)
PREDICTIONS_FILE = os.path.join("reports", "oof_predictions.csv")
MODELS_DIR = "models"
CANDIDATE_DIR = os.path.join("models", "candidate")   # --candidate: нова модель тут, у роботу — лише після «Прийняти»


M15_NAME = "lightgbm_15m"
# модель «старт» (режим split, 2026-10-12): ціль — перші START_MINUTES імпульсу; вхід за нею, утримання — за основною
# (experiments/early_target.py, early_target_check.py, purged_split.py; notes/experiments.md)
START_NAME = "lightgbm_15m_start"
START_MINUTES = 30
BAR_15M = pd.Timedelta(minutes=15)
CONFIRM_SHARE = 0.75      # «Alert»: годинна оцінка ≥ 0.75 × її порогу (experiments/two_tier.py)


def model_path(name: str, out_dir: str | None = None) -> str:
    return os.path.join(out_dir or MODELS_DIR, f"{name}.joblib")


def label_fingerprint(pumps: pd.DataFrame) -> dict:
    """Відбиток розмітки кожного тікера (межі й типи інтервалів) — щоб потім бачити, чи змінилась розмітка
    після навчання моделі. Вхід і стоп не входять: модель на них не вчиться."""
    import hashlib
    out = {}
    for base, g in pumps.sort_values(["base_asset", "start_ts"]).groupby("base_asset"):
        key = ";".join(f"{a:%Y-%m-%d %H:%M}|{b:%Y-%m-%d %H:%M}|{t}" for a, b, t in zip(g["start_ts"], g["end_ts"], g["pattern_type"]))
        out[base] = hashlib.md5(key.encode()).hexdigest()[:12]
    return out
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

def signal_episodes(df: pd.DataFrame, signal: pd.Series, bar: pd.Timedelta = H) -> pd.DataFrame:
    """Сигнальні бари одного тікера з проміжками не більше години → один епізод (одна «угода»).
    На 1h це сусідні години; на 15m сигнал, що згас на 15–45 хв і повернувся, — та сама ситуація, не новий епізод."""
    rows = []
    for base, g in df[np.asarray(signal)].groupby("base_asset"):
        ts = g["ts"].sort_values()
        run_id = (ts.diff() > max(bar, H)).cumsum()
        for _, run in ts.groupby(run_id):
            rows.append((base, run.iloc[0], run.iloc[-1] + bar))
    return pd.DataFrame(rows, columns=["base_asset", "start", "end"])


def event_metrics(df: pd.DataFrame, signal: pd.Series, pumps: pd.DataFrame, bar: pd.Timedelta = H) -> dict:
    """Метрики з погляду трейдера: спіймані пампи, затримка, хибні епізоди. bar — довжина бару рядків df."""
    covered = pumps[pumps["base_asset"].isin(df["base_asset"].unique())]
    sig_ts = df.loc[np.asarray(signal), ["base_asset", "ts"]]

    delays, caught_fs = [], 0
    n_pump = n_fs = 0
    for p in covered.itertuples():
        inside = sig_ts[(sig_ts["base_asset"] == p.base_asset) & in_pump(sig_ts["ts"], p.start_ts, p.end_ts, bar)]
        if p.pattern_type == "pump":
            n_pump += 1
            if len(inside):
                delays.append(max(0.0, (inside["ts"].min() - p.start_ts) / H))   # старт усередині бару — затримка 0
        elif p.pattern_type == "false_start":
            n_fs += 1
            caught_fs += bool(len(inside))

    eps = signal_episodes(df, signal, bar)
    labeled = covered.rename(columns={"start_ts": "ps", "end_ts": "pe"})
    false_eps = 0
    for e in eps.itertuples():
        own = labeled[labeled["base_asset"] == e.base_asset]
        if not ((own["ps"] < e.end) & (own["pe"] > e.start)).any():
            false_eps += 1
    days = len(df) * (bar / H) / 24    # оцінених барів у днях (у тікерів лише з інтервалами між інтервалами є розриви)

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


# Схожі ознаки (oi_chg_24h і oi_z_24h — обидві «OI за добу вище норми») ділять gain між собою і від навчання до навчання
# «перескакують» одна на одну; сума групи стабільніша. Порядок важить: oi_minus_ret_24h — група OI.
FEATURE_GROUPS = [("Власний рух", ("beta", "r2", "idio_")), ("Ліквідність (Amihud)", ("illiq_",)),
                  ("OI", ("oi_",)), ("Ціна", ("ret_",)), ("Обсяг", ("volume_", "vol_burst")),
                  ("Волатильність", ("volatility_", "range_", "park_", "gk_")), ("Купівлі / продажі", ("buy_ratio_",)),
                  ("Funding", ("funding",)), ("Ліквідації", ("liq_",)), ("Long / Short", ("ls_",))]


def feature_group(name: str) -> str:
    return next((g for g, prefixes in FEATURE_GROUPS if name.startswith(prefixes)), "Інше")


def gain_shares(bundle: dict | None) -> dict[str, float] | None:
    """Частка gain кожної ознаки LightGBM збереженої моделі (сума = 1); None — не LightGBM або моделі немає."""
    booster = getattr((bundle or {}).get("model"), "booster_", None)
    if booster is None:
        return None
    gain = booster.feature_importance(importance_type="gain")
    names = bundle.get("features") or booster.feature_name()
    return {n: float(g / gain.sum()) for n, g in zip(names, gain)} if gain.sum() > 0 else None


def group_shares(shares: dict[str, float]) -> pd.Series:
    """Частки ознак → сума по групах, від найбільшої."""
    s = pd.Series(shares)
    return s.groupby(s.index.map(feature_group)).sum().sort_values(ascending=False)


def group_table(shares: dict[str, float]) -> str:
    """Таблиця «група → % gain, найважливіша ознака групи» для звіту."""
    s = pd.Series(shares)
    groups = group_shares(shares)
    top = s.groupby(s.index.map(feature_group)).idxmax()
    t = pd.DataFrame({"gain_%": (100 * groups).round(1),
                      "top feature": [f"{top[g]} ({100 * s[top[g]]:.1f}%)" for g in groups.index]})
    t.index.name = "group"
    return md_table(t)


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
    parser.add_argument("--no-15m", action="store_true", help="не навчати 15m-модель (лише годинні)")
    parser.add_argument("--with-liq", action="store_true",
                        help="ознаки ліквідацій у моделях (до 2026-10-07; без них не гірше — experiments/no_liq.py)")
    parser.add_argument("--no-move", action="store_true",
                        help="15m-модель без ознак власного руху (beta / idio / illiq / park…; як до 2026-10-08)")
    parser.add_argument("--candidate", action="store_true",
                        help=f"зберегти моделі як кандидата в {CANDIDATE_DIR} (у роботу — після порівняння і «Прийняти» в label.py)")
    parser.add_argument("--no-start", action="store_true",
                        help=f"не навчати модель «старт» ({START_NAME}: ціль — перші {START_MINUTES} хв імпульсу; вхід режиму split)")
    parser.add_argument("--start-only", action="store_true",
                        help="навчити ЛИШЕ модель «старт» на поточній розмітці (основна 15m і годинні моделі, звіт — без змін); "
                             "поріг — та сама кількість хибних, що в 15m-моделі з тими самими налаштуваннями")
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
    if "reviewed" not in df:                       # таблиця від старої версії features.py
        df["reviewed"] = True
    labeled_recent = df[df["reviewed"] & (df["ts"] > cutoff) & df["pattern_type"].notna()]
    extra = [(b, g["ts"].min(), g["ts"].max()) for b, g in labeled_recent.groupby("base_asset")]
    # тікери лише з розміченими інтервалами: у навчанні були тільки ці інтервали (кожен окремим відрізком)
    for b, g in df[~df["reviewed"]].groupby("base_asset"):
        run = (g["ts"].diff() != H).cumsum()
        extra += [(b, r["ts"].min(), r["ts"].max()) for _, r in g.groupby(run)]

    feats = FEATURES if args.with_liq else MODEL_FEATURES
    X, y, groups = df[feats], df["label"].astype(int), df["base_asset"]
    tickers = sorted(groups.unique())
    reviewed = sorted(df.loc[df["reviewed"], "base_asset"].unique())    # нерозмічена година тут = «пампу не було»
    # з якими налаштуваннями навчено — у файл моделі, звіт і дашборд, щоб потім було видно, чому модель така
    settings = {
        "fresh_hours": args.fresh_hours, "fresh_cutoff": cutoff if args.fresh_hours else None,
        "fresh_hours_dropped": n_fresh_dropped, "models": args.models, "n_features": len(feats), "liquidations": bool(args.with_liq),
        "hours": int(len(df)), "pump_hours": int(y.sum()), "tickers": len(tickers),
        "labeled_only_tickers": len(tickers) - len(reviewed),
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
        f"- models: {', '.join(args.models)}; {len(feats)} features ({'with' if args.with_liq else 'without'} liquidations); saved for the screener: {'no' if args.no_save else 'yes'}",
    ]
    for name, t in per_ticker.items():
        t.index.name = "ticker"
        parts += ["", f"## {name} by ticker", "", md_table(t)]
    if len(imp.columns):
        parts += ["", "## Feature importance", "",
                  "Models fitted on all data. Logistic regression coefficients are on standardised features "
                  "(sign = direction of effect). LightGBM gain is the share of total split gain.", "",
                  md_table(imp)]
        if "lightgbm_gain_%" in imp:
            parts += ["", "### LightGBM gain by feature group", "",
                      "Correlated features share gain and can swap places between runs; the group total is more stable.", "",
                      group_table((imp["lightgbm_gain_%"] / 100).to_dict())]
    m15 = None
    if not args.no_15m:
        m15 = train_15m(db_url, df, pumps, cutoff if args.fresh_hours else None, set(reviewed), extra,
                        oof.get("lightgbm"), preds, now, args,
                        CANDIDATE_DIR if args.candidate else MODELS_DIR, label_fingerprint(pumps))
        parts += m15["report"]
    if args.start_only:
        print("\n--start-only: годинні моделі, 15m-модель і звіт не змінено")
        return
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(parts) + "\n")
    # out-of-fold передбачення: для аналізу помилок (кожен рядок передбачила модель, що не бачила тікер)
    preds.to_csv(PREDICTIONS_FILE, index=False)
    print(f"\nЗвіт: {REPORT_FILE}\nПередбачення: {PREDICTIONS_FILE}")

    out_dir = CANDIDATE_DIR if args.candidate else MODELS_DIR
    labels_fp = label_fingerprint(pumps)
    if not args.no_save:
        os.makedirs(out_dir, exist_ok=True)
        for name in [m for m in args.models if m in PROBA_MODELS]:
            bundle = {
                "name": name,
                "model": PROBA_MODELS[name]().fit(X, y),
                "features": feats,
                "threshold": best_f1_threshold(y.to_numpy(), oof[name]),
                "trained_at": now,
                "tickers": reviewed,
                # до data_end години переглянутих тікерів — in-sample; крім того — розмічені інтервали з in_sample_extra
                # (свіжі розмічені і всі інтервали тікерів поза tickers.txt)
                "data_end": min(df["ts"].max(), cutoff) if args.fresh_hours else df["ts"].max(),
                "in_sample_extra": extra,
                "pump_intervals": int((pumps["pattern_type"] == "pump").sum()),
                "validation": {k: (v if isinstance(v, str) else float(v)) for k, v in summary.loc[name].items()},
                "risk": risk_info,
                "settings": settings,
            }
            bundle["labels_fp"] = labels_fp
            joblib.dump(bundle, model_path(name, out_dir))
            print(f"Модель для скринера: {model_path(name, out_dir)} (поріг {bundle['threshold']:.3f})")
        if risk_info.get("k") is not None:
            print(f"Стоп: {risk_info['k']:.2f} NATR (75-й перцентиль MAE, {risk_info['n']} пампів)")


def train_15m(db_url: str, df1h: pd.DataFrame, pumps: pd.DataFrame, cutoff, reviewed: set, extra: list,
              oof_1h, preds_1h: pd.DataFrame, now: str, args, out_dir: str = MODELS_DIR, labels_fp: dict | None = None) -> dict:
    """15m-LightGBM: leave-ticker-out як для годинних, валідація обох рівнів сигналу, бандл для скринера.
    Тікери й правила ті самі: переглянуті — усі бари, інші — лише розмічені інтервали; свіжі нерозмічені — ні."""
    print("\n15m-модель: ознаки…")
    bases = sorted(df1h["base_asset"].unique())
    with psycopg.connect(db_url) as conn:
        d = build_15m(conn, bases, pumps)
        raw15 = load_raw(conn, bases, INTERVAL_15M)
        risk15 = stop_multiplier(conn, "15min")      # довідково: від розміченого старту
    # 15m-історія тікера — не раніше за його годинну: розмітка показує цей період (годинна модель має ту саму межу).
    # Інакше довга 15m-історія лише з Bybit (з довгого бектесту, напр. MON / ZRO з 26.03) іде в навчання як
    # «переглянуто, імпульсів немає», хоча ти її не бачив, а на стику бірж стрибає сума обсягу.
    first_1h = df1h.groupby("base_asset")["ts"].min()
    cut = d["base_asset"].map(first_1h)
    dropped = int((d["ts"] < cut).sum())
    d = d[d["ts"] >= cut]
    if dropped:
        print(f"  відкинуто {dropped} барів 15m раніше за годинну історію тікера")
    d["reviewed"] = d["base_asset"].isin(reviewed)
    d = d[d["reviewed"] | d["pattern_type"].notna()]
    if cutoff is not None:
        d = d[(d["ts"] <= cutoff) | d["pattern_type"].notna()]
    d = d.reset_index(drop=True)
    feats15 = (FEATURES_15M if args.with_liq else MODEL_FEATURES_15M) + ([] if args.no_move else MOVE_FEATURES)
    X, y, groups = d[feats15], d["label"].astype(int), d["base_asset"]
    print(f"  {len(d)} барів 15m, {groups.nunique()} тікерів")
    move_cov = float(d["idio_4h"].notna().mean()) if not args.no_move else None
    if move_cov is not None:
        print(f"  ознаки власного руху є в {move_cov:.0%} барів (решта — до початку індексу альтів, NaN)")
        if move_cov < 0.8:
            print("  ! мало індексу альтів — python alt_index.py (історія з архіву)")
    res = leave_ticker_out("lightgbm", X, y, groups)
    s15 = (res["score"] / res["threshold"]).to_numpy()
    rows = {"15m candidate": {**hour_metrics(y, res, True), **event_metrics(d, s15 >= 1, pumps, BAR_15M)}}
    if oof_1h is not None:
        # «Alert»: 15m-сигнал і годинна оцінка останньої ЗАКРИТОЇ години ≥ CONFIRM_SHARE її порогу
        hour = preds_1h[["base_asset", "ts"]].assign(s1h=oof_1h / preds_1h["lightgbm_threshold"].to_numpy())
        key = (d["ts"] + BAR_15M - H).dt.floor("h")
        s1h = d[["base_asset"]].assign(ts=key).merge(hour, on=["base_asset", "ts"], how="left")["s1h"].fillna(0).to_numpy()
        alert = (s15 >= 1) & (s1h >= CONFIRM_SHARE)
        # стоп 15m-контуру — від фактичного входу моделі (перший out-of-fold Alert у пампі), не від старту
        from tiers import tier_frame
        oof_t = tier_frame(d[["base_asset", "ts"]].assign(probability=res["score"].to_numpy(), threshold=res["threshold"].to_numpy()),
                           hour.assign(probability=hour["s1h"], threshold=1.0)[["base_asset", "ts", "probability", "threshold"]],
                           CONFIRM_SHARE)
        es = entry_stop(oof_t, raw15, pumps)
        if es.get("k") is not None:
            risk15 = {**es, "k_from_start": risk15.get("k")}
        rows[f"15m + 1h ≥ {CONFIRM_SHARE:g} (alert)"] = {
            **hour_metrics(y, pd.DataFrame({"score": s15, "signal": alert}), False),
            **event_metrics(d, alert, pumps, BAR_15M)}
    summary = pd.DataFrame(rows).T
    summary.index.name = "tier"
    # out-of-fold прогнози 15m-контуру — для сторінки порівняння моделей (label.py /compare)
    oof15 = d[["base_asset", "ts", "label", "pattern_type"]].assign(
        p15=res["score"].to_numpy(), thr15=res["threshold"].to_numpy(),
        s1h=s1h if oof_1h is not None else np.nan)
    oof15["tier"] = np.where(s15 >= 1, np.where(oof15["s1h"] >= CONFIRM_SHARE, "alert", "candidate"), "calm")
    print(summary.to_string())
    report = ["", "## 15m model and signal tiers", "",
              f"LightGBM on {len(d)} 15-minute bars ({len(feats15)} features: the hourly set with the same wall-clock "
              "windows plus short bar features"
              + ("" if args.no_move else f" and own-move features vs the ALT-EW index — present on {move_cov:.0%} of bars")
              + "), leave-ticker-out with nested threshold. *Candidate* = 15m model alone; "
              f"*alert* = 15m signal confirmed by the hourly LightGBM at ≥ {CONFIRM_SHARE:g} of its threshold "
              "(score of the last closed hour). Event delays are in hours from the labeled start; "
              f"stop multiplier on 15m bars: k = {risk15.get('k') or float('nan'):.2f} ATR.", "",
              md_table(summary)]
    if getattr(args, "start_only", False):
        ref = {"trained_at": now, "tickers": sorted(reviewed & set(bases)),
               "data_end": min(d["ts"].max(), cutoff) if cutoff is not None else d["ts"].max(), "in_sample_extra": extra,
               "pump_intervals": int((pumps["pattern_type"] == "pump").sum()), "confirm": {"model": "lightgbm", "share": CONFIRM_SHARE},
               "labels_fp": labels_fp or {}}
        if os.path.exists(model_path(M15_NAME, out_dir)):
            ref["main_version"] = joblib.load(model_path(M15_NAME, out_dir))["trained_at"]
        report += train_start(d, X, groups, pumps, s15, feats15, None if args.no_save else ref, out_dir, args)
        return {"report": report, "summary": summary}
    if not args.no_save:
        bundle = {
            "name": M15_NAME, "tf": "15m", "bar_minutes": 15,
            "model": make_lightgbm().fit(X, y), "features": feats15,
            "threshold": best_f1_threshold(y.to_numpy(), res["score"].to_numpy()),
            "trained_at": now, "tickers": sorted(reviewed & set(bases)),
            "data_end": min(d["ts"].max(), cutoff) if cutoff is not None else d["ts"].max(),
            "in_sample_extra": extra,
            "pump_intervals": int((pumps["pattern_type"] == "pump").sum()),
            "validation": {k: (v if isinstance(v, str) else float(v)) for k, v in summary.iloc[0].items()},
            "tier_validation": {k: (v if isinstance(v, str) else float(v)) for k, v in summary.iloc[-1].items()},
            "risk": risk15, "exit_bars": EXIT_BARS_15M,
            "confirm": {"model": "lightgbm", "share": CONFIRM_SHARE},
            "settings": {"fresh_hours": args.fresh_hours, "bars": int(len(d)), "tickers": int(groups.nunique()),
                         "n_features": len(feats15), "liquidations": bool(args.with_liq), "from_1h_start": True, "dropped_before_1h": dropped,
                         "move_features": not args.no_move, "move_coverage": move_cov},
        }
        bundle["labels_fp"] = labels_fp or {}
        bundle["feature_ref"] = feature_reference(X)   # навчальний розподіл ознак — monitor.py рахує дрейф наживо (PSI)
        report += ["", "### 15m LightGBM gain by feature group", "", group_table(gain_shares(bundle))]
        os.makedirs(out_dir, exist_ok=True)
        joblib.dump(bundle, model_path(M15_NAME, out_dir))
        oof15.to_parquet(os.path.join(out_dir, "oof_15m.parquet"), index=False)
        print(f"Збережено {model_path(M15_NAME, out_dir)}: поріг {bundle['threshold']:.3f}, стоп {risk15.get('k')} ATR (15m)")
    if not getattr(args, "no_start", False):
        report += train_start(d, X, groups, pumps, s15, feats15, bundle if not args.no_save else None, out_dir, args)
    return {"report": report, "summary": summary}


def start_target(d: pd.DataFrame, pumps: pd.DataFrame, minutes: int = START_MINUTES) -> tuple[np.ndarray, np.ndarray]:
    """Ціль моделі «старт»: 1 — бар, що закривається в перші minutes хвилин імпульсу; решта барів імпульсу — поза навчанням
    (не «0»: це не звичайний ринок); фальстарти й звичайний ринок — 0. Повертає (ціль, маска «в навчанні»)."""
    close = d["ts"] + BAR_15M
    pos = np.zeros(len(d), dtype=bool)
    for r in pumps[pumps["pattern_type"] == "pump"].itertuples():
        m = (d["base_asset"] == r.base_asset) & (close > r.start_ts) & (close <= min(r.start_ts + pd.Timedelta(minutes=minutes), r.end_ts))
        pos |= m.to_numpy()
    keep = pos | (d["label"].to_numpy() == 0)
    return pos.astype(int), keep


def matched_threshold(d: pd.DataFrame, p: np.ndarray, pumps: pd.DataFrame, target_fa: float) -> float:
    """Поріг, за якого хибних епізодів на 30 днів найближче до target_fa (сітка квантилів 0.90…0.999)."""
    best, gap = None, np.inf
    for q in np.linspace(0.90, 0.999, 45):
        thr = float(np.quantile(p, q))
        fa = event_metrics(d, p >= thr, pumps, BAR_15M)["false_alarms_per_30d"]
        if abs(fa - target_fa) < gap:
            best, gap = thr, abs(fa - target_fa)
    return best


def train_start(d, X, groups, pumps, s15, feats15, main_bundle: dict | None, out_dir: str, args) -> list[str]:
    """Модель «старт» для режиму split: leave-ticker-out (вчиться лише на рядках у навчанні), поріг — та сама кількість
    хибних на 30 днів, що в основної 15m-моделі на її порозі (як experiments/early_target.py); бандл поруч з основним."""
    print(f"\nМодель «старт» (перші {START_MINUTES} хв імпульсу)…")
    ys, keep = start_target(d, pumps)
    p = np.zeros(len(d))
    for g in groups.unique():
        te = (groups == g).to_numpy()
        tr = ~te & keep
        p[te] = make_lightgbm().fit(X[tr], ys[tr]).predict_proba(X[te])[:, 1]
    fa0 = event_metrics(d, s15 >= 1, pumps, BAR_15M)["false_alarms_per_30d"]
    thr = matched_threshold(d, p, pumps, fa0)
    m = event_metrics(d, p >= thr, pumps, BAR_15M)
    print(f"  позитивів {ys.sum()}, у навчанні {keep.sum()} з {len(d)}; поріг {thr:.3f} (хибних {m['false_alarms_per_30d']:.1f} — як "
          f"в основної {fa0:.1f}); спіймано {m['pumps_caught']}, фальстартів {m['false_starts_triggered']}")
    report = ["", "## 15m start model (split mode entry)", "",
              f"Target: bars closing in the first {START_MINUTES} minutes of an impulse (the rest of the impulse is left out of training). "
              f"Threshold matched to the main 15m model's false-alarm rate ({fa0:.1f} per 30 days): {thr:.3f}. "
              f"Leave-ticker-out: caught {m['pumps_caught']}, false starts {m['false_starts_triggered']}, "
              f"false alarms {m['false_alarms_per_30d']:.1f} per 30 days, median delay {m['median_delay_h']:.2f} h."]
    if main_bundle is not None:
        bundle = {
            "name": START_NAME, "tf": "15m", "bar_minutes": 15, "role": "entry", "target": f"start{START_MINUTES}",
            "model": make_lightgbm().fit(X[keep], ys[keep]), "features": feats15, "threshold": thr,
            "trained_at": main_bundle["trained_at"], "tickers": main_bundle["tickers"], "data_end": main_bundle["data_end"],
            "in_sample_extra": main_bundle["in_sample_extra"], "pump_intervals": main_bundle["pump_intervals"],
            "validation": {k: (v if isinstance(v, str) else float(v)) for k, v in m.items()},
            "confirm": main_bundle["confirm"],
            "settings": {"minutes": START_MINUTES, "positives": int(ys.sum()), "train_rows": int(keep.sum()),
                         "matched_false_alarms": float(fa0)},
            "labels_fp": main_bundle.get("labels_fp", {}),
            # версія основної 15m-моделі, з якою «старт» у парі (поріг підібрано під її рівень хибних)
            "main_version": main_bundle.get("main_version", main_bundle["trained_at"]),
        }
        joblib.dump(bundle, model_path(START_NAME, out_dir))
        print(f"Збережено {model_path(START_NAME, out_dir)}: поріг {thr:.3f}")
    return report


def load_model(name: str) -> dict:
    """Збережена модель з метаданими: model, features, threshold, trained_at, validation."""
    bundle = joblib.load(model_path(name))
    expected = ([b + m for b in (FEATURES_15M, MODEL_FEATURES_15M) for m in ([], MOVE_FEATURES)] if bundle.get("tf") == "15m"
                else (FEATURES, MODEL_FEATURES))
    if bundle["features"] not in expected:            # з ліквідаціями (до 2026-10-07) або без; з власним рухом (з 10-08) або без
        raise ValueError(f"{name}: модель навчена на іншому наборі ознак — перезапусти python train.py")
    return bundle


if __name__ == "__main__":
    main()
