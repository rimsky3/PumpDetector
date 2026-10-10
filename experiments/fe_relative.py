"""
experiments/fe_relative.py — НОВІ ОЗНАКИ: «відносно ринку» і дані, яких модель ще не бачила (2026-10-10, власник: «подумай про
feature engineering — так ми знайшли власний рух»; «так»). Принцип: «власний рух» спрацював, бо відняв ринок від уже наявної
ознаки (ціна); решта спроб (рівні, профіль, POC, Боллінджер, перша хвилина бару) — ще один погляд на ту саму ціну / обсяг монети.
Тому (1) очистити від ринку найважливіше — OI (~39% ваги) і обсяг (~11%); (2) додати нове: топ-трейдери, премія, набір позицій.
Групи (ознаки зафіксовано ДО запуску; значення бару — стан на його закриття, лише минуле):
  mkt   OI і обсяг відносно ринку. Ринок — усі монети кешу архіву (≥ 50 на барі):
        oi_rel_1h / 4h / 24h   зміна log OI (USD) монети − середня така зміна по ринку
        oi_rank_24h            місце зміни OI за добу серед усіх монет (0…1)
        vol_rel_4h / 24h       log(обсяг 4 год / 24 год ÷ власна медіана за 7 днів) − медіана цього по ринку
        vol_rank_4h            місце «сплеску обсягу 4 год» серед усіх монет (0…1)
  top   Топ-трейдери проти натовпу (архів Binance metrics, колонки, яких ми досі не брали):
        tt_pos_long            частка лонгів топ-трейдерів за позиціями
        tt_gap                 tt_pos_long − частка лонг-акаунтів усіх (великі проти натовпу)
        tt_acc_gap             частка лонг-акаунтів топ-трейдерів − усіх
        tt_gap_chg_4h          зміна tt_gap за 4 год
        tt_pos_chg_24h         зміна tt_pos_long за добу
  prem  Премія ф'ючерса до споту (архів premiumIndexKlines 15m; funding — її повільне відображення):
        prem                   премія на закритті бару, %
        prem_z_7d              премія відносно власного тижня, у σ
        prem_chg_1h / 4h       зміна премії за 1 / 4 год
  turn  Скільки обороту стало новими позиціями:
        oi_turn_1h / 4h / 24h  приріст OI в монетах × ціна ÷ оборот за той самий час
Варіанти (усі свіжонавчені однаково, розмітка власника): base (ознаки робочої моделі), base_s43 (те саме, зерно 43 — міра шуму),
+mkt, +top, +prem, +turn, +all. LTO (Кандидат 15m) і бектест 6 міс. на чесному списку (Alert = 15m + робоча годинна, частковий
вихід, стоп I.K), половини монет, важливість груп. Ринкові ряди є лише до 30.09 — у жовтневих барах навчання ознаки mkt порожні
(як «власний рух» в усіх експериментах на кеші архіву).
КРИТЕРІЙ (до запуску), для кожного варіанта: ≥ 20 млн PF − base ≥ 0.03 і більше за |base_s43 − base|; кращий за base у ≥ 4 з 6
місяців; кращий на обох половинах монет; загальний PF ≥ base.
→ reports/fe_relative.md, reports/fe_relative_trades.csv; кеш data/fe_rel/

    python experiments/fe_relative.py
"""

import argparse
import glob
import io
import os
import pickle
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import numpy as np
import pandas as pd
import requests

import auto_train as AT
import binance_archive as BA
import early_target as ET
import exit_partial as P
import idio_features as I
import train
from tiers import tier_frame

DIR = os.path.join("data", "fe_rel")
ARCH = os.path.join("data", "binance_archive", "metrics")
PREM_URL = "https://data.binance.vision/data/futures/um/{k}/premiumIndexKlines/{s}/15m/{s}-15m-{d}.zip"
BAR = pd.Timedelta(minutes=15)
D1, W1 = 96, 7 * 96
MONTHS = ["2026-03", "2026-04", "2026-05", "2026-06", "2026-07", "2026-08", "2026-09"]
OWN_FROM, OWN_TO = "2026-07-25", "2026-10-08"
GROUPS = {
    "mkt": ["oi_rel_1h", "oi_rel_4h", "oi_rel_24h", "oi_rank_24h", "vol_rel_4h", "vol_rel_24h", "vol_rank_4h"],
    "top": ["tt_pos_long", "tt_gap", "tt_acc_gap", "tt_gap_chg_4h", "tt_pos_chg_24h"],
    "prem": ["prem", "prem_z_7d", "prem_chg_1h", "prem_chg_4h"],
    "turn": ["oi_turn_1h", "oi_turn_4h", "oi_turn_24h"],
}
ALL = [f for g in GROUPS.values() for f in g]
VARIANTS = {"base": (ET.COLS, 42), "base_s43": (ET.COLS, 43), **{f"+{g}": (ET.COLS + fs, 42) for g, fs in GROUPS.items()},
            "+all": (ET.COLS + ALL, 42)}
BIG = "≥ 20 млн"
SEEDS = [42, 43, 44, 45, 46]
OUT = os.path.join("reports", "fe_relative.md")


# ---------- дані ----------

def _prem_file(coin: str, kind: str, d: str) -> pd.Series | None:
    s = quote(f"{coin}USDT")
    try:
        r = requests.get(PREM_URL.format(k=kind, s=s, d=d), timeout=60)
    except requests.RequestException:
        return None
    if r.status_code != 200:
        return None
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        raw = z.read(z.namelist()[0])
    df = pd.read_csv(io.BytesIO(raw), header=0 if raw[:9] == b"open_time" else None, names=BA.KLINE_COLS, usecols=[0, 4])
    return pd.Series(df.iloc[:, 1].astype(float).to_numpy(), index=pd.to_datetime(df.iloc[:, 0].astype("int64"), unit="ms", utc=True))


def premium(coin: str, days: list[str] = ()) -> pd.Series:
    """Премія (закриття 15m-свічки індексу премії, частка) на відкриттях барів; кеш data/fe_rel/premium/."""
    path = os.path.join(DIR, "premium", f"{coin}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)["prem"]
    parts = [p for p in ([_prem_file(coin, "monthly", m) for m in MONTHS] + [_prem_file(coin, "daily", d) for d in days]) if p is not None]
    s = pd.concat(parts).sort_index() if parts else pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
    s = s[~s.index.duplicated()]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    s.rename("prem").to_frame().to_parquet(path)
    return s


def metrics(coin: str) -> pd.DataFrame:
    """Знімки архіву metrics монети (час = create_time + 5 хв): OI у монетах і частки лонгів; кеш data/fe_rel/metrics/."""
    path = os.path.join(DIR, "metrics", f"{coin}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    files = sorted(glob.glob(os.path.join(ARCH, f"{coin}USDT", "*.parquet")))
    cols = ["create_time", "sum_open_interest", "count_toptrader_long_short_ratio", "sum_toptrader_long_short_ratio", "count_long_short_ratio"]
    if not files:
        return pd.DataFrame(columns=["oi", "top_acc", "top_pos", "all"])
    m = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
    t = pd.to_datetime(m["create_time"], utc=True) + pd.Timedelta(minutes=5)
    share = lambda c: (m[c].astype(float) / (1 + m[c].astype(float))).where(m[c].astype(float) > 0).to_numpy()      # noqa: E731
    out = pd.DataFrame({"oi": m["sum_open_interest"].astype(float).where(m["sum_open_interest"].astype(float) > 0).to_numpy(),
                        "top_acc": share("count_toptrader_long_short_ratio"), "top_pos": share("sum_toptrader_long_short_ratio"),
                        "all": share("count_long_short_ratio")}, index=t)
    out = out[~out.index.duplicated()].sort_index()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_parquet(path)
    return out


def at_bars(m: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Стан на закриття бару: останній знімок не старший за 10 хв."""
    return m.reindex(index + BAR, method="ffill", tolerance=pd.Timedelta(minutes=10)).set_axis(index)


def vol_x(v: pd.Series, n: int) -> pd.Series:
    """log(обсяг за n барів ÷ медіана таких сум за попередні 7 днів)."""
    s = v.rolling(n, min_periods=n).sum()
    return np.log(s / s.shift(n).rolling(W1, min_periods=W1 // 2).median()).replace([np.inf, -np.inf], np.nan)


def market() -> dict:
    """Ринкові ряди з усіх монет кешу: середня зміна log OI (USD) за 1 / 4 / 24 год, медіана сплеску обсягу; матриці для місць."""
    path = os.path.join(DIR, "market.pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    doi = {4: {}, 16: {}, 96: {}}
    vx = {16: {}, 96: {}}
    files = sorted(glob.glob(os.path.join(I.DIR, "*.pkl")))
    for n, p in enumerate(files, 1):
        c = os.path.basename(p)[:-4]
        h = pickle.load(open(p, "rb"))["h"]
        met = metrics(c)
        if not len(met):
            continue
        oi = np.log((at_bars(met[["oi"]], h.index)["oi"] * h["price"]).where(lambda s: s > 0))
        for k in doi:
            doi[k][c] = oi.diff(k)
        for k in vx:
            vx[k][c] = vol_x(h["volume"], k)
        if n % 100 == 0:
            print(f"  ринок: {n}/{len(files)}", flush=True)
    out = {}
    for k, dct in doi.items():
        w = pd.DataFrame(dct)
        out[f"oi_{k}"] = w.mean(axis=1).where(w.notna().sum(axis=1) >= 50)
        if k == 96:
            out["M_oi_96"] = w.astype("float32")
    for k, dct in vx.items():
        w = pd.DataFrame(dct)
        out[f"vol_{k}"] = w.median(axis=1).where(w.notna().sum(axis=1) >= 50)
        if k == 16:
            out["M_vol_16"] = w.astype("float32")
    os.makedirs(DIR, exist_ok=True)
    pickle.dump(out, open(path + ".tmp", "wb"))
    os.replace(path + ".tmp", path)
    return out


def rank_in(M: pd.DataFrame, s: pd.Series) -> pd.Series:
    """Частка монет ринку зі значенням меншим за s на тому самому барі."""
    m = M.reindex(s.index)
    n = m.notna().sum(axis=1)
    return (m.lt(s, axis=0).sum(axis=1) / n.where(n >= 50)).where(s.notna())


def features(h: pd.DataFrame, met: pd.DataFrame, prem: pd.Series, mk: dict) -> pd.DataFrame:
    """15m-ряд монети (price, volume у $) + знімки metrics + премія + ринок → нові ознаки на барах h."""
    f = pd.DataFrame(index=h.index)
    b = at_bars(met, h.index) if len(met) else pd.DataFrame(np.nan, index=h.index, columns=["oi", "top_acc", "top_pos", "all"])
    px, v = h["price"].where(h["price"] > 0), h["volume"]
    oi = np.log((b["oi"] * px).where(lambda s: s > 0))
    for name, k in (("1h", 4), ("4h", 16), ("24h", 96)):
        f[f"oi_rel_{name}"] = oi.diff(k) - mk[f"oi_{k}"].reindex(h.index)
    f["oi_rank_24h"] = rank_in(mk["M_oi_96"], oi.diff(96))
    vx4, vx24 = vol_x(v, 16), vol_x(v, 96)
    f["vol_rel_4h"] = vx4 - mk["vol_16"].reindex(h.index)
    f["vol_rel_24h"] = vx24 - mk["vol_96"].reindex(h.index)
    f["vol_rank_4h"] = rank_in(mk["M_vol_16"], vx4)
    f["tt_pos_long"] = b["top_pos"]
    f["tt_gap"] = b["top_pos"] - b["all"]
    f["tt_acc_gap"] = b["top_acc"] - b["all"]
    f["tt_gap_chg_4h"] = f["tt_gap"].diff(16)
    f["tt_pos_chg_24h"] = b["top_pos"].diff(96)
    p = prem.reindex(h.index) * 100
    f["prem"] = p
    f["prem_z_7d"] = (p - p.rolling(W1, min_periods=W1 // 2).mean()) / p.rolling(W1, min_periods=W1 // 2).std()
    f["prem_chg_1h"], f["prem_chg_4h"] = p.diff(4), p.diff(16)
    for name, k in (("1h", 4), ("4h", 16), ("24h", 96)):
        f[f"oi_turn_{name}"] = (b["oi"] - b["oi"].shift(k)) * px / v.rolling(k, min_periods=k).sum().where(lambda s: s > 0)
    return f[ALL].replace([np.inf, -np.inf], np.nan)


def prepare(own: list[str], coins: list[str]) -> None:
    """Докачати з публічного архіву Binance: metrics тікерів власника (до OWN_TO) і премію всіх монет."""
    days = BA._days(OWN_FROM, OWN_TO)
    oct_days = [d for d in days if d >= "2026-10-01"]
    print(f"архів: metrics {len(own)} тікерів власника × {len(days)} дн., премія {len(own) + len(coins)} монет…", flush=True)
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda a: BA._csv("metrics", f"{a[0]}USDT", a[1]), [(b, d) for b in own for d in days]))
        list(ex.map(lambda b: premium(b, oct_days), own))
        for n, _ in enumerate(ex.map(premium, coins), 1):
            if n % 100 == 0:
                print(f"  премія: {n}/{len(coins)}", flush=True)


def lgbm(seed: int):
    m = train.make_lightgbm()
    m.set_params(random_state=seed)
    return m


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Нові ознаки: OI / обсяг відносно ринку, топ-трейдери, премія, набір позицій — LTO і бектест 6 міс.")
    ap.add_argument("--seeds", choices=list(GROUPS), help="перевірка шуму: base і base + ця група на 5 зернах (42–46) "
                                                          "→ reports/fe_relative_seeds_<група>.md")
    args = ap.parse_args()
    global VARIANTS
    if args.seeds:
        VARIANTS = {f"{n}_s{sd}": (cols, sd) for sd in SEEDS for n, cols in (("base", ET.COLS), (f"+{args.seeds}", ET.COLS + GROUPS[args.seeds]))}
    d, pumps, px, idx, uni = ET.load_data()
    trained = set(d["base_asset"])
    coins = sorted(c for c in uni if c not in trained and os.path.exists(os.path.join(I.DIR, f"{c}.pkl")))
    prepare(sorted(trained), coins)
    mk = market()
    parts = []
    for b, g in d.groupby("base_asset"):
        parts.append(features(px[b], metrics(b), premium(b), mk).reindex(g["ts"]).set_axis(g.index))
    d = d.join(pd.concat(parts))
    cov = {g: float(d[fs[0]].notna().mean()) for g, fs in GROUPS.items()}
    print(f"навчання: {len(d)} барів, {d['base_asset'].nunique()} тікерів; покриття: " + ", ".join(f"{g} {c:.0%}" for g, c in cov.items()), flush=True)

    y, grp = d["label"].to_numpy().astype(int), d["base_asset"]
    thr, lto, models = {}, {}, {}
    for v, (cols, seed) in VARIANTS.items():
        p = train.oof_proba(lambda s=seed: lgbm(s), d[cols], pd.Series(y), grp)
        thr[v] = train.best_f1_threshold(y, p)
        sig = p >= thr[v]
        lto[v] = {**train.event_metrics(d, sig, pumps, ET.BAR), **ET.early(d, sig, pumps, px)}
        models[v] = lgbm(seed).fit(d[cols], y)
        print(f"  {v} (поріг {thr[v]:.3f}): {lto[v]}", flush=True)
    if args.seeds:
        imp = None
    else:
        gain = pd.Series(models["+all"].booster_.feature_importance("gain"), index=VARIANTS["+all"][0])
        imp = (gain / gain.sum()).sort_values(ascending=False)

    m1 = joblib.load(train.model_path("lightgbm"))
    m15 = joblib.load(train.model_path(train.M15_NAME))
    exit_bars, share = m15.get("exit_bars", 4), m15.get("confirm", {}).get("share", 0.75)
    start, end = pd.Timestamp(I.START, tz="UTC"), pd.Timestamp(I.END, tz="UTC") + pd.Timedelta(days=1)
    print(f"бектест: {len(coins)} монет чесного списку…", flush=True)
    rows = []
    for n, c in enumerate(coins, 1):
        x = pickle.load(open(os.path.join(I.DIR, f"{c}.pkl"), "rb"))
        f15, h1, h = x["f15"], x["h1"], x["h"]
        f15 = f15.join(I.new_features(h, idx).reindex(f15["ts"]).set_axis(f15.index))
        f15 = f15.join(features(h, metrics(c), premium(c), mk).reindex(f15["ts"]).set_axis(f15.index))
        hh = h[(h.index >= start - pd.Timedelta(days=1)) & (h.index < end)]
        s1 = h1[["base_asset", "ts"]].assign(probability=m1["model"].predict_proba(h1[m1["features"]])[:, 1], threshold=m1["threshold"])
        for v, (cols, _) in VARIANTS.items():
            s15 = f15[["base_asset", "ts"]].assign(probability=models[v].predict_proba(f15[cols])[:, 1], threshold=thr[v])
            t = tier_frame(s15, s1, share).set_index("ts").sort_index()
            for r in P.simulate(hh, t.loc[(t.index >= start) & (t.index < end), "alert_score"], I.K, exit_bars, 3.0, 0.5, True):
                mth = pd.Timestamp(r["ts"]).strftime("%Y-%m")
                if mth in uni[c]:
                    rows.append({"variant": v, "ticker": c, "half": AT.half(c), "month": mth, "band": uni[c][mth], **r})
        if n % 50 == 0:
            print(f"  {n}/{len(coins)}", flush=True)
    tr = pd.DataFrame(rows)
    if args.seeds:
        write_seeds(tr, lto, args.seeds)
        return
    tr.to_csv(OUT.replace(".md", "_trades.csv"), index=False)
    write_report(tr, lto, thr, imp, cov)


def write_seeds(tr, lto, group) -> None:
    """base проти base + група на кількох зернах: чи різниця більша за розкид між зернами."""
    pf = I.pf
    big = tr[tr["band"] == BIG]
    rows = []
    for sd in SEEDS:
        r = {"seed": sd}
        for n in ("base", f"+{group}"):
            v = f"{n}_s{sd}"
            g, b = tr[tr["variant"] == v], big[big["variant"] == v]
            r[n] = {"pf": pf(g["ret"]), "pf20": pf(b["ret"]), "A": pf(b[b["half"] == "A"]["ret"]), "B": pf(b[b["half"] == "B"]["ret"]),
                    "n": len(g), "fa": lto[v]["false_alarms_per_30d"], "caught": lto[v]["caught"]}
        rows.append(r)
    new = f"+{group}"
    L = [f"# Перевірка шуму: base проти base + {group} на {len(SEEDS)} зернах", "",
         "Кожне зерно — окреме навчання (свій поріг) і бектест 6 міс. на чесному списку. Різниця справжня, якщо нова група краща "
         "на більшості зерен і середня різниця більша за розкид base між зернами.", "",
         f"| зерно | base PF | {new} PF | base ≥ 20 млн | {new} ≥ 20 млн | різниця ≥ 20 млн | base A / B | {new} A / B | хибних / 30 дн base / {new} |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        a, b = r["base"], r[new]
        L.append(f"| {r['seed']} | {a['pf']:.2f} | {b['pf']:.2f} | {a['pf20']:.3f} | {b['pf20']:.3f} | {b['pf20'] - a['pf20']:+.3f} | "
                 f"{a['A']:.2f} / {a['B']:.2f} | {b['A']:.2f} / {b['B']:.2f} | {a['fa']:.1f} / {b['fa']:.1f} |")
    a20 = np.array([r["base"]["pf20"] for r in rows])
    b20 = np.array([r[new]["pf20"] for r in rows])
    at, bt = np.array([r["base"]["pf"] for r in rows]), np.array([r[new]["pf"] for r in rows])
    L += ["", f"- ≥ 20 млн: base {a20.mean():.3f} (від {a20.min():.3f} до {a20.max():.3f}), {new} {b20.mean():.3f} (від {b20.min():.3f} до "
              f"{b20.max():.3f}); середня різниця {(b20 - a20).mean():+.3f}, розкид base між зернами {a20.max() - a20.min():.3f}; "
              f"{new} краща на {int((b20 > a20).sum())} з {len(rows)} зерен; найгірша {new} проти найкращої base: {b20.min() - a20.max():+.3f}",
          f"- усі монети: base {at.mean():.3f}, {new} {bt.mean():.3f}, різниця {(bt - at).mean():+.3f}; краща на {int((bt > at).sum())} з {len(rows)} зерен"]
    out = OUT.replace(".md", f"_seeds_{group}.md")
    open(out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


def write_report(tr, lto, thr, imp, cov) -> None:
    pf = I.pf
    V = list(VARIANTS)
    names = {"mkt": "OI і обсяг відносно ринку", "top": "топ-трейдери проти натовпу", "prem": "премія ф'ючерса до споту",
             "turn": "оборот → нові позиції"}
    L = ["# Нові ознаки: відносно ринку і нові дані", "",
         "Групи: " + "; ".join(f"**{g}** — {names[g]} ({len(fs)})" for g, fs in GROUPS.items()) + ". "
         "Покриття навчальних барів: " + ", ".join(f"{g} {c:.0%}" for g, c in cov.items()) + ". base_s43 — ті самі ознаки, зерно 43 (шум).", "",
         "**Критерій (до запуску):** ≥ 20 млн PF − base ≥ 0.03 і більше за |base_s43 − base|; ≥ 4 з 6 місяців; обидві половини; "
         "загальний PF ≥ base.", "",
         "## Leave-ticker-out (Кандидат 15m)", "",
         "| варіант | поріг | спіймано | затримка, хв | пройдено руху | фальстартів | хибних / 30 дн |", "|---|---|---|---|---|---|---|"]
    for v in V:
        m = lto[v]
        L.append(f"| {v} | {thr[v]:.3f} | {m['caught']} | {m['delay_min']:.0f} | {m['done']:.0%} | {m['false_starts_triggered']} | {m['false_alarms_per_30d']:.1f} |")
    rank = {k: i + 1 for i, k in enumerate(imp.index)}
    L += ["", "## Важливість у моделі +all (частка gain)", "", "| група | разом | ознаки (частка, місце з " + str(len(imp)) + ") |", "|---|---|---|"]
    for g, fs in GROUPS.items():
        L.append(f"| {g}: {names[g]} | **{imp[fs].sum():.1%}** | " + ", ".join(f"{k} {imp[k]:.1%} (№{rank[k]})" for k in sorted(fs, key=lambda k: -imp[k])) + " |")
    L += ["", "Топ-10 ознак моделі +all: " + ", ".join(f"{k} {imp[k]:.1%}" for k in imp.index[:10]) + ".", "",
          f"## Бектест 6 міс., {tr['ticker'].nunique()} монет", "",
          "| варіант | угод | у плюсі | медіана | PF | без топ-3 | ≥ 20 млн | 5–20 млн | A ≥ 20 млн | B ≥ 20 млн |", "|---|---|---|---|---|---|---|---|---|---|"]
    for v in V:
        g = tr[tr["variant"] == v]
        r = g["ret"].sort_values()
        big = g["band"] == BIG
        L.append(f"| {v} | {len(r)} | {(r > 0).mean():.0%} | {r.median():+.2%} | {pf(r):.2f} | {pf(r.iloc[:-3]):.2f} | {pf(g[big]['ret']):.2f} | "
                 f"{pf(g[g['band'] == '5–20 млн']['ret']):.2f} | {pf(g[big & (g['half'] == 'A')]['ret']):.2f} | {pf(g[big & (g['half'] == 'B')]['ret']):.2f} |")
    piv = tr.groupby(["month", "variant"])["ret"].apply(pf).unstack().reindex(columns=V)
    piv20 = tr[tr["band"] == BIG].groupby(["month", "variant"])["ret"].apply(pf).unstack().reindex(columns=V)
    L += ["", "## По місяцях (PF; у дужках ≥ 20 млн)", "", "| місяць | " + " | ".join(V) + " |", "|---|" + "---|" * len(V)]
    for mth in piv.index:
        L.append(f"| {mth} | " + " | ".join(f"{piv.at[mth, v]:.2f} ({piv20.at[mth, v]:.2f})" for v in V) + " |")
    big = tr[tr["band"] == BIG]
    p20 = {v: pf(big[big["variant"] == v]["ret"]) for v in V}
    noise = abs(p20["base_s43"] - p20["base"])
    L += ["", "## Критерій", "", f"Шум між зернами на ≥ 20 млн: {noise:.3f}.", ""]
    for v in V[2:]:
        diff = p20[v] - p20["base"]
        months = int((piv20[v] > piv20["base"] + 1e-9).sum())
        halves = all(pf(big[(big["variant"] == v) & (big["half"] == hf)]["ret"]) > pf(big[(big["variant"] == "base") & (big["half"] == hf)]["ret"])
                     for hf in ("A", "B"))
        total = pf(tr[tr["variant"] == v]["ret"]) >= pf(tr[tr["variant"] == "base"]["ret"])
        ok = diff >= 0.03 and diff > noise and months >= 4 and halves and total
        L.append(f"- {v}: ≥ 20 млн {diff:+.3f}; місяців краще {months} з {len(piv20)}; обидві половини {'так' if halves else 'ні'}; "
                 f"загальний ≥ base {'так' if total else 'ні'} → **{'ВИКОНАНО' if ok else 'НЕ виконано'}**")
    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
