"""
experiments/liq_cascade.py — «лонг після каскаду ліквідацій» (mean reversion), 2026-10-20.

Питання власника: чи можна заробити на поверненні ціни після примусового продажу (ліквідацій лонгів)?
L-shock (03.10) ловив злив лише за ціною й обсягом: PF 1.04, після комісії ≈ 0. Тут — те саме, але подію визначають
ЛІКВІДАЦІЇ (нове джерело, не похідне від ціни): чи додають вони інформацію понад «просто злив»?

Дані: ліквідації лонгів Coinalyze 15m (Binance + Bybit, USDT-перпи, USD) з локальної бази — з covered_from кожного
ринку до останньої точки (локальна копія до ~11.10); ціни — 1m-свічки Binance з архіву (experiments/klines1m.py).

Пороги ЗАФІКСОВАНО ДО ПРОГОНУ (не підбираються):
    каскад:   15m-бар, ліквідації лонгів ≥ LIQ_X × середнє за 7 днів (нулі враховано, ≥ 3 дні покриття) і ≥ LIQ_MIN $;
              сусідні бари каскаду (проміжок ≤ 1 бар) — одна подія, рахуємо від останнього;
    покриття: бар вважається покритим, якщо в ринку є будь-яка точка ліквідацій у межах ±6 год (Coinalyze віддає лише
              непорожні бари, «рядка немає» = 0 лише там, де збір ішов);
    угода:    вхід — закриття останнього бару каскаду; стоп — лоу каскаду; ½ на +1R; решта — на відкритті першого бару
              каскаду (не нижче +1R) / стоп / 4 год; комісія 0.1% (як l_shock.simulate, 1m-бари).
    варіанти: V1 каскад; V2 каскад + перша зелена 15m за ≤ 2 бари (вхід на її закритті); V3 = V1 із затримкою 5 хв
              (Coinalyze 15m приходить із запізненням).
    контроль: C1 — злив тієї самої сили БЕЗ каскаду (червоний 15m, падіння ≥ DROP_ATR ATR, ліквідації ≤ 2× середнього),
              порівнюється з V1 при падінні ≥ DROP_ATR ATR; C2 — випадкові входи тих самих монет (стоп і ціль — медіани V1 в ATR).

КРИТЕРІЙ (записано до запуску). Ідея варта продовження, лише якщо для V1 (усі монети) ВИКОНАНО ВСЕ:
    1) PF після комісії ≥ 1.10 і медіана > 0;  2) PF ≥ 1.0 в обох половинах монет (md5) і обох половинах часу;
    3) PF(V1, падіння ≥ 1.5 ATR) − PF(C1) ≥ 0.10 — ліквідації дають інформацію понад ціну;
    4) сума без топ-3 > 0;  5) угод ≥ 300.
Окремо — смуга ≥ 20 млн $/добу (сер. оборот Binance за 30 днів): якщо загалом проходить, а там ні — «не для власника».
Інакше ідею закрито, як L-shock.

    python experiments/liq_cascade.py --data   # докачати 1m-свічки (07–09 міс. + 01–11.10) для монет з ліквідаціями
    python experiments/liq_cascade.py          # → reports/liq_cascade.md, reports/liq_cascade_trades.csv
"""

import argparse
import hashlib
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import klines1m as K
from l_shock import atr, fwd, simulate, table

LIQ_X, LIQ_MIN = 10.0, 50_000
DROP_ATR, CTRL_LIQ_X = 1.5, 2.0
COVER = pd.Timedelta(hours=6)
WARM = pd.Timedelta(days=3)
RESP_BARS, DELAY = 2, pd.Timedelta(minutes=5)
FEE = 0.001
BAND = 20e6
RANDOM_PER_EVENT, SEED = 3, 42
OUT_MD = os.path.join("reports", "liq_cascade.md")
OUT_CSV = os.path.join("reports", "liq_cascade_trades.csv")
BAR = pd.Timedelta(minutes=15)


def load_liq(conn) -> tuple[dict, dict]:
    """Ліквідації лонгів 15m Binance + Bybit (USDT-перпи) за монетою Binance → (серії, covered_from)."""
    q = """SELECT m.symbol_on_exchange, m.exchange_name, l.ts, l.long_liq, l.short_liq
           FROM liquidations l JOIN markets m USING (symbol)
           WHERE l.interval = '15min' AND m.quote_asset = 'USDT' AND m.exchange_name IN ('Binance', 'Bybit')
             AND (m.exchange_name = 'Bybit' OR m.symbol LIKE '%%\\_PERP.A')"""
    df = pd.DataFrame(conn.execute(q).fetchall(), columns=["sym", "ex", "ts", "long", "short"])
    cov = pd.DataFrame(conn.execute(
        """SELECT m.symbol_on_exchange, m.exchange_name, c.covered_from FROM liquidation_coverage c JOIN markets m USING (symbol)
           WHERE c.interval = '15min' AND m.quote_asset = 'USDT' AND m.exchange_name IN ('Binance', 'Bybit')
             AND (m.exchange_name = 'Bybit' OR m.symbol LIKE '%%\\_PERP.A')""").fetchall(), columns=["sym", "ex", "cf"])
    bn = set(cov.loc[cov["ex"] == "Binance", "sym"])           # монети з покриттям Binance; Bybit — додатком
    series, covered = {}, {}
    for sym, g in df[df["sym"].isin(bn)].groupby("sym"):
        base = sym[:-4] if sym.endswith("USDT") else None
        if not base:
            continue
        g = g.assign(ts=pd.to_datetime(g["ts"], utc=True))
        s = g.groupby("ts")[["long", "short"]].sum().fillna(0)
        series[base] = s.sort_index()
        covered[base] = pd.to_datetime(cov.loc[(cov["sym"] == sym) & (cov["ex"] == "Binance"), "cf"].min(), utc=True)
    return series, covered


def bars_with_liq(m1: pd.DataFrame, liq: pd.DataFrame, cf: pd.Timestamp) -> pd.DataFrame:
    """15m-бари Binance + ліквідації лонгів, покриття, середнє за 7 днів, ATR, оборот 30 днів."""
    b = K.resample(m1, "15min")
    end = liq.index.max() + COVER
    b = b[(b.index >= cf - pd.Timedelta(days=8)) & (b.index <= end)]
    if len(b) < 200:
        return pd.DataFrame()
    b["atr"] = atr(b).shift(1)
    b["turn30"] = b["v"].rolling(30 * 96, min_periods=10 * 96).sum().shift(1) / 30   # сер. оборот за добу
    lq = liq.reindex(b.index).fillna(0)
    b["liq"] = lq["long"]
    pts = pd.Series(1.0, index=liq.index[(liq["long"] + liq["short"]) > 0]).reindex(b.index).fillna(0)
    near = pts.rolling(int(COVER / BAR) + 1, min_periods=1).max()                        # точка за 6 год до
    near_fwd = pts[::-1].rolling(int(COVER / BAR) + 1, min_periods=1).max()[::-1]         # і після
    b["covered"] = (b.index >= cf) & ((near > 0) | (near_fwd > 0))
    liq_cov = b["liq"].where(b["covered"])
    b["liq_mean"] = liq_cov.rolling(7 * 96, min_periods=int(WARM / BAR)).mean().shift(1)
    b.loc[b.index < cf + WARM, "liq_mean"] = np.nan
    return b


def make_trade(m1, b, i0, j, base, kind, delay=pd.Timedelta(0)) -> dict | None:
    """Угода: каскад — бари i0..(кінець), вхід на закритті бару j (+ затримка)."""
    t_e = b.index[j] + BAR + delay
    if delay > pd.Timedelta(0):
        k = m1.index.searchsorted(t_e - pd.Timedelta(minutes=1))
        if k >= len(m1) or m1.index[k] != t_e - pd.Timedelta(minutes=1):
            return None
        entry = m1["c"].iloc[k]
        stop = min(b["l"].iloc[i0:j + 1].min(), m1["l"].loc[b.index[j] + BAR:t_e - pd.Timedelta(minutes=1)].min())
    else:
        entry = b["c"].iloc[j]
        stop = b["l"].iloc[i0:j + 1].min()
    origin, a = b["o"].iloc[i0], b["atr"].iloc[i0]
    if not np.isfinite(a) or a <= 0:
        return None
    sim = simulate(m1, t_e, entry, stop, origin)
    if sim is None:
        return None
    return {"base": base, "kind": kind, "ts": b.index[i0], "entry_ts": t_e, "entry": entry, "stop": stop,
            "origin": origin, "atr": a, "r_atr": (entry - stop) / a, "orig_atr": (origin - entry) / a,
            "drop_atr": (origin - b["c"].iloc[j if kind != "V2" else i0:j + 1].min()) / a,
            "liq": b["liq"].iloc[i0:j + 1].sum(), "liq_x": b["liq"].iloc[i0:j + 1].max() / max(b["liq_mean"].iloc[i0], 1),
            "turn30": b["turn30"].iloc[i0], **sim, **fwd(m1["c"], t_e, entry)}


def events(m1, b, base) -> list[dict]:
    ok = b["covered"] & b["liq_mean"].notna() & b["atr"].notna()
    casc = ok & (b["liq"] >= LIQ_X * b["liq_mean"]) & (b["liq"] >= LIQ_MIN)
    idx = np.flatnonzero(casc.to_numpy())
    groups, cur = [], []
    for i in idx:                                          # сусідні бари (проміжок ≤ 1 бар) — один каскад
        if cur and i - cur[-1] > 2:
            groups.append(cur)
            cur = []
        cur.append(i)
    if cur:
        groups.append(cur)
    out = []
    o, c = b["o"].to_numpy(), b["c"].to_numpy()
    for g in groups:
        i0, i1 = g[0], g[-1]
        if i1 + 1 >= len(b):
            continue
        for kind, j, d in (("V1", i1, pd.Timedelta(0)), ("V3", i1, DELAY)):
            t = make_trade(m1, b, i0, j, base, kind, d)
            if t:
                out.append(t)
        for k in range(1, RESP_BARS + 1):                  # V2: перша зелена після каскаду
            j = i1 + k
            if j < len(b) and c[j] > o[j]:
                t = make_trade(m1, b, i0, j, base, "V2")
                if t:
                    out.append(t)
                break
    # C1: злив тієї самої сили без каскаду ліквідацій
    drop = (b["o"] - b["c"]) / b["atr"]
    ctrl = ok & (b["c"] < b["o"]) & (drop >= DROP_ATR) & (b["liq"] <= CTRL_LIQ_X * b["liq_mean"])
    for i in np.flatnonzero(ctrl.to_numpy()):
        if i + 1 < len(b):
            t = make_trade(m1, b, i, i, base, "C1")
            if t:
                out.append(t)
    return out


def random_trades(m1, b, base, n, r_atr, o_atr, rng) -> list[dict]:
    ok = np.flatnonzero((b["covered"] & b["liq_mean"].notna() & b["atr"].notna()).to_numpy())
    ok = ok[ok < len(b) - 1]
    out = []
    for j in rng.choice(ok, size=min(n, len(ok)), replace=False) if len(ok) and n > 0 else []:
        t_e, entry, a = b.index[j] + BAR, b["c"].iloc[j], b["atr"].iloc[j]
        sim = simulate(m1, t_e, entry, entry - r_atr * a, entry + o_atr * a)
        if sim:
            out.append({"base": base, "kind": "C2", "ts": b.index[j], "entry_ts": t_e, "entry": entry, "atr": a,
                        "turn30": b["turn30"].iloc[j], **sim, **fwd(m1["c"], t_e, entry)})
    return out


def stats(d: pd.DataFrame) -> dict:
    r = d["res"].dropna() * 100
    if r.empty:
        return {"n": 0}
    pos, neg = r[r > 0].sum(), -r[r < 0].sum()
    g = r + FEE * 100
    gpos, gneg = g[g > 0].sum(), -g[g < 0].sum()
    top3 = r.sort_values(ascending=False).iloc[:3].sum()
    return {"n": len(r), "монет": d["base"].nunique(), "у плюсі": f"{(r > 0).mean():.0%}",
            "PF": round(pos / neg, 2) if neg else np.inf, "PF без ком.": round(gpos / gneg, 2) if gneg else np.inf,
            "сер. %": round(r.mean(), 2), "медіана %": round(r.median(), 2), "сума %": round(r.sum()),
            "без топ-3 %": round(r.sum() - top3), "1год %": round(d["f1h"].median() * 100, 2),
            "4год %": round(d["f4h"].median() * 100, 2)}


def pf(d) -> float:
    r = d["res"].dropna()
    neg = -r[r < 0].sum()
    return r[r > 0].sum() / neg if neg else np.inf


def half(base: str) -> str:
    return "A" if int(hashlib.md5(base.encode()).hexdigest(), 16) % 2 == 0 else "B"


def stage_data(series: dict) -> None:
    bases = sorted(series)
    print(f"докачую 1m-свічки для {len(bases)} монет: 07–09 міс. і 01–11.10", flush=True)
    days = [d.strftime("%Y-%m-%d") for d in pd.date_range("2026-10-01", "2026-10-11")]
    with ThreadPoolExecutor(8) as ex:
        for i, msg in enumerate(ex.map(lambda b: K.fetch(b, ["2026-07", "2026-08", "2026-09"]) + " " + K.fetch_days(b, days),
                                       bases), 1):
            if i % 25 == 0 or i == len(bases):
                print(f"  {i}/{len(bases)}  {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Лонг після каскаду ліквідацій лонгів (Coinalyze 15m) з контролем «злив без каскаду».")
    ap.add_argument("--data", action="store_true", help="лише докачати 1m-свічки для монет з ліквідаціями")
    ap.add_argument("--limit", type=int, default=0, help="лише перші N монет (перевірка)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    import psycopg
    from dotenv import load_dotenv
    load_dotenv(os.path.join(K.ROOT, ".env"))
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        series, covered = load_liq(conn)
    if args.data:
        stage_data(series)
        return
    bases = sorted(series)[: args.limit or None]
    rng = np.random.default_rng(SEED)
    rows, bars = [], {}
    for n, base in enumerate(bases, 1):
        m1 = K.load(base)
        if m1.empty:
            continue
        b = bars_with_liq(m1, series[base], covered[base])
        if b.empty:
            continue
        bars[base] = b
        rows += events(m1, b, base)
        if n % 25 == 0:
            print(f"  {n}/{len(bases)} монет, угод {len(rows)}", flush=True)
    ev = pd.DataFrame(rows)
    v1 = ev[ev["kind"] == "V1"]
    r_atr, o_atr = v1["r_atr"].median(), v1["orig_atr"].clip(lower=0).median()
    rnd = []
    for base, k in v1.groupby("base").size().items():
        rnd += random_trades(K.load(base), bars[base], base, int(k * RANDOM_PER_EVENT), r_atr, o_atr, rng)
    allr = pd.concat([ev, pd.DataFrame(rnd)], ignore_index=True)
    allr["coins"] = allr["base"].map(half)
    mid = pd.to_datetime(v1["entry_ts"], utc=True).median()
    allr["period"] = np.where(pd.to_datetime(allr["entry_ts"], utc=True) < mid, "ранній", "пізній")
    allr["band"] = np.where(allr["turn30"] >= BAND, "≥ 20 млн", "< 20 млн")
    os.makedirs("reports", exist_ok=True)
    allr.to_csv(OUT_CSV, index=False)

    K_ = {k: allr[allr["kind"] == k] for k in ("V1", "V2", "V3", "C1", "C2")}
    v1s = K_["V1"][K_["V1"]["drop_atr"] >= DROP_ATR]
    cols = ["група", "n", "монет", "у плюсі", "PF", "PF без ком.", "сер. %", "медіана %", "сума %", "без топ-3 %", "1год %", "4год %"]
    t0, t1 = pd.to_datetime(allr["entry_ts"], utc=True).min(), pd.to_datetime(allr["entry_ts"], utc=True).max()
    md = ["# Лонг після каскаду ліквідацій — подієвий тест", "",
          f"{allr['base'].nunique()} монет Binance, {t0:%d.%m} – {t1:%d.%m.%Y}; ліквідації лонгів Coinalyze 15m (Binance + Bybit). "
          f"Каскад: ≥ {LIQ_X:g}× середнього за 7 днів і ≥ ${LIQ_MIN / 1e3:g}K. Угода: стоп — лоу каскаду, ½ на 1R, решта — початок "
          f"каскаду / стоп / 4 год; комісія {FEE:.1%}. Середина часу — {mid:%d.%m}.", "",
          "## Усі групи", "",
          table([{"група": "V1 каскад", **stats(K_["V1"])}, {"група": "V2 каскад + зелена", **stats(K_["V2"])},
                 {"група": "V3 каскад, вхід +5 хв", **stats(K_["V3"])},
                 {"група": f"V1, падіння ≥ {DROP_ATR:g} ATR", **stats(v1s)},
                 {"група": f"C1 злив ≥ {DROP_ATR:g} ATR без каскаду", **stats(K_["C1"])},
                 {"група": "C2 випадковий вхід", **stats(K_["C2"])}], cols), ""]
    split = []
    for name, d in (("V1", K_["V1"]), ("V2", K_["V2"]), ("C1", K_["C1"]), ("C2", K_["C2"])):
        for col, vals in (("coins", "AB"), ("period", ("ранній", "пізній")), ("band", ("≥ 20 млн", "< 20 млн"))):
            for v in vals:
                split.append({"група": f"{name} · {v if col != 'coins' else 'монети ' + v}", **stats(d[d[col] == v])})
    md += ["## Половини монет, часу і смуги обороту", "", table(split, cols), ""]
    sens = []
    for x in (5, 20, 40):
        d = K_["V1"][K_["V1"]["liq_x"] >= x]
        sens.append({"група": f"V1, каскад ≥ {x}× (опис, не вибір)", **stats(d)})
    md += ["## Чутливість до сили каскаду (лише опис)", "", table(sens, cols), ""]
    v1b = K_["V1"][K_["V1"]["band"] == "≥ 20 млн"]
    checks = [
        ("1) PF ≥ 1.10 і медіана > 0", pf(K_["V1"]) >= 1.10 and K_["V1"]["res"].median() > 0, f"PF {pf(K_['V1']):.2f}, медіана {K_['V1']['res'].median() * 100:.2f}%"),
        ("2) обидві половини монет і часу ≥ 1.0",
         all(pf(K_["V1"][K_["V1"][c] == v]) >= 1.0 for c, vs in (("coins", "AB"), ("period", ("ранній", "пізній"))) for v in vs),
         ", ".join(f"{v} {pf(K_['V1'][K_['V1'][c] == v]):.2f}" for c, vs in (("coins", "AB"), ("period", ("ранній", "пізній"))) for v in vs)),
        ("3) PF(V1 ≥ 1.5 ATR) − PF(C1) ≥ 0.10", pf(v1s) - pf(K_["C1"]) >= 0.10, f"{pf(v1s):.2f} − {pf(K_['C1']):.2f}"),
        ("4) сума без топ-3 > 0", stats(K_["V1"])["без топ-3 %"] > 0, f"{stats(K_['V1'])['без топ-3 %']}%"),
        ("5) угод ≥ 300", len(K_["V1"]) >= 300, str(len(K_["V1"]))),
    ]
    md += ["## Критерій (записано до запуску)", "", "| умова | виконано | значення |", "|---|---|---|"]
    md += [f"| {a} | {'так' if b_ else 'НІ'} | {c} |" for a, b_, c in checks]
    md += ["", f"Смуга ≥ 20 млн: V1 — {len(v1b)} угод, PF {pf(v1b):.2f}.", "",
           f"**Висновок за критерієм: {'проходить' if all(b_ for _, b_, _ in checks) else 'НЕ проходить'}.**", ""]
    open(OUT_MD, "w", encoding="utf-8").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
