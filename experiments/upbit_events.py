"""
experiments/upbit_events.py — ПОДІЄВЕ ДОСЛІДЖЕННЯ (event study): що робить перп Binance після анонсу лістингу на Upbit (2026-10-13).
Ідея «Анонси бірж» (08.10; Chan, розд. про події): лістинг на Upbit — відомий привід для пампу. Подій мало (≈ 40 за 7.5 міс.
на монетах з перпом Binance), тож порівнювати PF угод моделі «з анонсом / без» — шум. Тому спершу без моделі: чи є рух після
анонсу, і чи встигає людина. Це перевірка анонсу як ПОДІЇ; ознаки з Upbit у модель це не додає.

Події: анонси Upbit категорії «거래» з лістингом на ринок KRW («신규 거래지원» — нова монета; «디지털 자산 추가» / «마켓 추가» —
ринок додано монеті, що вже торгувалась) з 2025-01-01 (глибше — щоб подій було більше; вирішено до запуску). Час — first_listed_at
(перша публікація, KST → UTC). Монета має перп USDT на Binance з ≥ 2 днями 1m-історії до анонсу; повтор тієї самої монети
ближче ніж за 3 дні — не подія. T0 — перша межа хвилини не раніше анонсу; ціна в момент T0 + d — закриття 1m-бару, що
закінчується в T0 + d (тобто d хвилин після анонсу, з точністю до хвилини в гірший для нас бік).

Контроль: для кожної події — CONTROLS випадкових моментів тієї самої монети в [T0 − 7 днів, T0 − 6 год]; ті самі виміри.
Тест — рандомізація (як у random_entry.py): замість кожної події беремо один її випадковий момент, середнє по всіх; B разів →
розподіл «якби анонс нічого не значив»; p — частка, де воно не менше за фактичне.

КРИТЕРІЙ (записано ДО запуску). Основна міра: вхід T0 + 3 хв → вихід T0 + 60 хв, мінус комісія FEE (0.1%).
ПРОЙДЕНО («анонс Upbit — нагода для людини»), якщо одночасно:
    (а) середнє > 0 і вище за контроль, p < 0.05;
    (б) медіана > 0 і середнє без трьох найкращих подій > 0;
    (в) той самий знак середнього в обох половинах подій за часом;
    (г) вхід T0 + 5 хв → T0 + 60 хв: середнє > 0.
Якщо (а) виконується для входу в T0 (до 1 хв після анонсу), але не для T0 + 3 хв → «РУХ Є, ЛЮДИНА НЕ ВСТИГАЄ»: лише чіп-пояснення
на графіку («Upbit listing … тому»), без сповіщення на вхід. Якщо й від T0 руху немає → ідею закриваємо.
Додатково (без критерію, для розуміння): рух ДО анонсу (−30 … −1 хв: витік / неточний час), профіль середньої ціни по хвилинах,
тип події, період з 2026-03-01, горизонти 30 хв / 4 год / 24 год.
→ reports/upbit_events.md, reports/upbit_events.csv; кеш — data/upbit_events/

    python experiments/upbit_events.py                 # ~10 хв перший раз (1m-свічки Binance з REST), далі з кешу
    python experiments/upbit_events.py --since 2026-03-01
"""

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import requests

import exchanges as X

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data", "upbit_events")
OUT_MD = os.path.join(ROOT, "reports", "upbit_events.md")
OUT_CSV = os.path.join(ROOT, "reports", "upbit_events.csv")
UPBIT = "https://api-manager.upbit.com/api/v1/announcements"
FEE = 0.001
DELAYS = (0, 1, 3, 5, 15)                 # хв після анонсу до входу
HORIZONS = (30, 60, 240, 1440)            # хв після анонсу до виходу
MAIN = (3, 60)
CONTROLS, B, SEED = 300, 10000, 42
BEFORE, AFTER = pd.Timedelta(days=7), pd.Timedelta(hours=24)
MIN_HISTORY = pd.Timedelta(days=2)
PROFILE = (-30, -10, -1, 0, 1, 2, 3, 5, 10, 15, 30, 60, 120, 240, 1440)
M = pd.Timedelta(minutes=1)


def upbit_listings(since: pd.Timestamp, refresh: bool) -> pd.DataFrame:
    """Анонси лістингів Upbit на ринок KRW: час першої публікації (UTC), тікер, тип."""
    path = os.path.join(CACHE, "announcements.json")
    if os.path.exists(path) and not refresh:
        raw = json.load(open(path, encoding="utf-8"))
    else:
        s = requests.Session()
        s.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/126.0 Safari/537.36", "accept": "application/json"})
        raw, pg = [], 1
        while True:
            d = s.get(UPBIT, params={"os": "web", "page": pg, "per_page": 20, "category": "trade"}, timeout=25).json()["data"]
            raw += d["notices"]
            if pg >= d["total_pages"] or pd.Timestamp(d["notices"][-1]["first_listed_at"]).tz_convert("UTC") < since - pd.Timedelta(days=5):
                break
            pg += 1
            time.sleep(0.35)
        json.dump(raw, open(path, "w", encoding="utf-8"), ensure_ascii=False)
    rows = []
    for n in raw:
        t = n["title"]
        if "KRW" not in t or not re.search("신규 거래지원|디지털 자산 추가|마켓 추가", t):
            continue
        tick = [x for grp in re.findall(r"\(([A-Z0-9, ]{1,60})\)", t) for x in re.split(r",\s*", grp)]
        tick = [x for x in dict.fromkeys(tick) if x and x not in ("KRW", "BTC", "USDT")]
        for c in tick:
            rows.append({"ts": pd.Timestamp(n["first_listed_at"]).tz_convert("UTC"), "coin": c,
                         "kind": "нова монета" if "신규" in t else "додано ринок KRW", "title": t})
    d = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
    d = d[d["ts"] >= since]
    keep = d.groupby("coin")["ts"].diff().fillna(pd.Timedelta(days=99)) >= pd.Timedelta(days=3)     # уточнення того самого анонсу
    return d[keep].reset_index(drop=True)


def klines(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Закриття 1m-барів перпа Binance (індекс — час ВІДКРИТТЯ бару); порожньо — символу тоді не було."""
    out, cur = [], int(start.timestamp() * 1000)
    stop = int(end.timestamp() * 1000)
    while cur < stop:
        try:
            k = X.get(X.BINANCE + "/fapi/v1/klines", {"symbol": symbol, "interval": "1m", "startTime": cur, "endTime": stop, "limit": 1500})
        except requests.HTTPError:
            return pd.Series(dtype=float)
        if not k:
            break
        out += [(int(r[0]), float(r[4])) for r in k]
        cur = int(k[-1][0]) + 60_000
        time.sleep(0.35)
        if len(k) < 1500:
            break
    if not out:
        return pd.Series(dtype=float)
    s = pd.Series(dict(out))
    s.index = pd.to_datetime(s.index, unit="ms", utc=True)
    return s[~s.index.duplicated()].sort_index()


def event_prices(coin: str, t0: pd.Timestamp, perps: set, log) -> tuple[str, pd.Series] | None:
    sym = next((s for s in (coin + "USDT", "1000" + coin + "USDT") if s in perps), None)
    if sym is None:
        return None
    path = os.path.join(CACHE, f"{sym}_{t0:%Y%m%d%H%M}.pkl")
    if os.path.exists(path):
        return sym, pd.read_pickle(path)
    s = klines(sym, t0 - BEFORE, t0 + AFTER + 2 * M)
    s.to_pickle(path)
    log(f"  {sym} {t0:%Y-%m-%d %H:%M}: {len(s)} барів")
    return sym, s


def measures(p: np.ndarray, i0: int) -> dict | None:
    """p — закриття на суцільній хвилинній сітці; i0 — індекс бару, що ЗАКІНЧУЄТЬСЯ в T0. Ціна в T0 + d = p[i0 + d]."""
    if i0 - 31 < 0 or i0 + max(HORIZONS) >= len(p) or not np.isfinite(p[i0]):
        return None
    out = {"pre30": p[i0 - 1] / p[i0 - 31] - 1}                    # бар i0 − 1 закрився ДО анонсу
    for d in DELAYS:
        for h in HORIZONS:
            if h > d:
                out[f"r{d}_{h}"] = p[i0 + h] / p[i0 + d] - 1 - FEE
    seg = p[i0 + MAIN[0]: i0 + 241]
    out["mfe"], out["mae"] = np.nanmax(seg) / seg[0] - 1, np.nanmin(seg) / seg[0] - 1
    out["peak_min"] = MAIN[0] + int(np.nanargmax(seg))
    for m in PROFILE:
        out[f"p{m}"] = p[i0 + m] / p[i0 - 1] - 1
    return out if all(np.isfinite(v) for v in out.values()) else None


def main() -> None:
    ap = argparse.ArgumentParser(description="Подієве дослідження: перп Binance після анонсу лістингу на Upbit (ринок KRW)")
    ap.add_argument("--since", default="2025-01-01", help="з якої дати брати анонси (за замовчуванням 2025-01-01)")
    ap.add_argument("--refresh", action="store_true", help="перечитати список анонсів з Upbit (інакше — з кешу data/upbit_events)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(CACHE, exist_ok=True)
    log = lambda m: print(m, flush=True)                              # noqa: E731
    since = pd.Timestamp(args.since, tz="UTC")
    ann = upbit_listings(since, args.refresh)
    ann = ann[ann["ts"] <= pd.Timestamp.now(tz="UTC") - AFTER - pd.Timedelta(hours=1)]
    info = X.get(X.BINANCE + "/fapi/v1/exchangeInfo", {})
    perps = {s["symbol"] for s in info["symbols"] if s.get("contractType") == "PERPETUAL" and s.get("quoteAsset") == "USDT"}
    log(f"анонсів Upbit з ринком KRW з {since:%Y-%m-%d}: {len(ann)} (монет {ann['coin'].nunique()})")

    rng = np.random.default_rng(SEED)
    ev, ctrl, drop = [], [], {"немає перпа Binance": 0, "перп молодший за 2 дні / немає даних": 0, "пропуски в даних": 0}
    for a in ann.itertuples():
        got = event_prices(a.coin, a.ts, perps, log)
        if got is None:
            drop["немає перпа Binance"] += 1
            continue
        sym, s = got
        t0 = a.ts.ceil("min")
        if s.empty or s.index.min() > t0 - MIN_HISTORY or s.index.max() < t0 + AFTER - M:
            drop["перп молодший за 2 дні / немає даних"] += 1
            continue
        grid = pd.date_range(s.index.min(), s.index.max(), freq="min")
        p = s.reindex(grid).ffill(limit=5).to_numpy()
        i0 = grid.get_loc(t0 - M)                                    # бар з відкриттям T0 − 1 хв закінчується в T0
        m = measures(p, i0)
        if m is None:
            drop["пропуски в даних"] += 1
            continue
        lo, hi = 31, i0 - 6 * 60 - max(HORIZONS)                     # контроль закінчується до T0 − 6 год
        cs = [c for j in rng.integers(lo, max(lo + 1, hi), CONTROLS) if (c := measures(p, int(j))) is not None] if hi > lo + 100 else []
        if len(cs) < 50:
            drop["пропуски в даних"] += 1
            continue
        ev.append({"ts": a.ts, "coin": a.coin, "symbol": sym, "kind": a.kind, **m})
        ctrl.append(pd.DataFrame(cs))
    E = pd.DataFrame(ev)
    E.to_csv(OUT_CSV, index=False)
    n = len(E)
    log(f"подій у тесті: {n}; відкинуто: {drop}")

    def test(col: str) -> dict:
        obs = E[col].to_numpy()
        cm = np.array([c[col].mean() for c in ctrl])
        null = np.zeros(B)
        for c in ctrl:
            null += rng.choice(c[col].to_numpy(), B)
        null /= n
        srt = np.sort(obs)
        half = n // 2
        return {"mean": obs.mean(), "median": np.median(obs), "ctrl": cm.mean(), "p": (np.sum(null >= obs.mean()) + 1) / (B + 1),
                "no_top3": srt[:-3].mean() if n > 3 else np.nan, "win": (obs > 0).mean(), "h1": obs[:half].mean(), "h2": obs[half:].mean()}

    pc = lambda v, d=2: "—" if not np.isfinite(v) else f"{v * 100:+.{d}f}%"       # noqa: E731
    res = {(d, h): test(f"r{d}_{h}") for d in DELAYS for h in HORIZONS if h > d}
    md = ["# Анонси лістингів Upbit (ринок KRW): що робить перп Binance", "",
          f"Подій: **{n}** ({E['coin'].nunique()} монет), {E['ts'].min():%Y-%m-%d} … {E['ts'].max():%Y-%m-%d}; "
          f"нова монета — {(E['kind'] == 'нова монета').sum()}, додано ринок KRW — {(E['kind'] != 'нова монета').sum()}. "
          f"Відкинуто: {', '.join(f'{k} {v}' for k, v in drop.items())}. Комісія {FEE:.1%} врахована; контроль — {CONTROLS} "
          f"випадкових моментів тієї самої монети за 7 днів до анонсу; p — рандомізація, {B} повторів.", "",
          "## Вхід через d хвилин після анонсу → вихід через h хвилин після анонсу", "",
          "| вхід | вихід | середнє | медіана | у плюсі | без топ-3 | контроль | p | 1-ша половина | 2-га половина |", "|---|---|---|---|---|---|---|---|---|---|"]
    for (d, h), r in res.items():
        b = "**" if (d, h) == MAIN else ""
        md.append(f"| {b}+{d} хв{b} | {b}+{h} хв{b} | {pc(r['mean'])} | {pc(r['median'])} | {r['win']:.0%} | {pc(r['no_top3'])} | {pc(r['ctrl'])} | "
                  f"{r['p']:.3f} | {pc(r['h1'])} | {pc(r['h2'])} |")
    md += ["", "## Профіль: середня і медіанна зміна ціни відносно останньої хвилини ДО анонсу", "",
           "| хвилина | " + " | ".join(f"{m:+d}" for m in PROFILE) + " |", "|---|" + "---|" * len(PROFILE),
           "| середнє | " + " | ".join(pc(E[f"p{m}"].mean(), 1) for m in PROFILE) + " |",
           "| медіана | " + " | ".join(pc(E[f"p{m}"].median(), 1) for m in PROFILE) + " |",
           "| контроль, середнє | " + " | ".join(pc(np.mean([c[f"p{m}"].mean() for c in ctrl]), 1) for m in PROFILE) + " |", ""]
    pre = test("pre30")
    md += [f"Рух ДО анонсу (−31 … −1 хв): середнє {pc(pre['mean'])}, медіана {pc(pre['median'])}, контроль {pc(pre['ctrl'])}, p {pre['p']:.3f}.",
           f"Після входу +{MAIN[0]} хв за 4 год: найкраща точка — середнє {pc(E['mfe'].mean())} (медіана {pc(E['mfe'].median())}), "
           f"найгірша — середнє {pc(E['mae'].mean())} (медіана {pc(E['mae'].median())}); пік у медіані на {E['peak_min'].median():.0f}-й хв.", "",
           "## Зрізи (вхід +3 хв → +60 хв; для розуміння, без критерію)", "", "| зріз | подій | середнє | медіана | у плюсі |", "|---|---|---|---|---|"]
    col = f"r{MAIN[0]}_{MAIN[1]}"
    for name, mask in (("нова монета", E["kind"] == "нова монета"), ("додано ринок KRW", E["kind"] != "нова монета"),
                       ("з 2026-03-01", E["ts"] >= pd.Timestamp("2026-03-01", tz="UTC")), ("до 2026-03-01", E["ts"] < pd.Timestamp("2026-03-01", tz="UTC"))):
        x = E.loc[mask, col]
        md.append(f"| {name} | {len(x)} | {pc(x.mean()) if len(x) else '—'} | {pc(x.median()) if len(x) else '—'} | {(x > 0).mean():.0%} |" if len(x) else f"| {name} | 0 | — | — | — |")
    m3, m5, m0 = res[MAIN], res[(5, 60)], res[(0, 60)]
    a = m3["mean"] > 0 and m3["mean"] > m3["ctrl"] and m3["p"] < 0.05
    b = m3["median"] > 0 and m3["no_top3"] > 0
    c = np.sign(m3["h1"]) == np.sign(m3["h2"]) and m3["h1"] > 0
    g = m5["mean"] > 0
    a0 = m0["mean"] > 0 and m0["mean"] > m0["ctrl"] and m0["p"] < 0.05
    verdict = ("**ПРОЙДЕНО** — анонс Upbit дає рух, який людина встигає взяти" if a and b and c and g else
               "**РУХ Є, ЛЮДИНА НЕ ВСТИГАЄ** — від моменту анонсу рух значущий, від +3 хв — ні: лише чіп-пояснення на графіку" if a0 else
               "**НЕ ПРОЙДЕНО** — значущого руху після анонсу немає навіть від першої хвилини; ідею закриваємо")
    md += ["", "## Критерій (записано до запуску)", "",
           f"- (а) +3 → +60 хв: середнє > 0, вище за контроль, p < 0.05 — {'так' if a else 'ні'} ({pc(m3['mean'])} проти {pc(m3['ctrl'])}, p {m3['p']:.3f})",
           f"- (б) медіана > 0 і середнє без топ-3 > 0 — {'так' if b else 'ні'} ({pc(m3['median'])}, {pc(m3['no_top3'])})",
           f"- (в) обидві половини подій за часом у плюсі — {'так' if c else 'ні'} ({pc(m3['h1'])} / {pc(m3['h2'])})",
           f"- (г) +5 → +60 хв: середнє > 0 — {'так' if g else 'ні'} ({pc(m5['mean'])})",
           f"- від моменту анонсу (+0 → +60 хв): {pc(m0['mean'])} проти {pc(m0['ctrl'])}, p {m0['p']:.3f} — {'значущо' if a0 else 'ні'}", "", verdict, ""]
    open(OUT_MD, "w", encoding="utf-8").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
