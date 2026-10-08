"""
drivers.py — Drivers: що саме дало ймовірність 15m-моделі на конкретному барі (2026-10-08).

Мета проєкту — кожен сигнал пояснює, які метрики його дали. Досі була лише загальна важливість ознак моделі
(train.FEATURE_GROUPS, /compare), а не «чому саме цей Alert». LightGBM розкладає кожен прогноз точно (TreeSHAP,
booster.predict(..., pred_contrib=True)):

    log-odds прогнозу = базове значення + Σ внесків ознак          P = 1 / (1 + e^(−log-odds))

Внесок у log-odds людині нічого не каже, тож показуємо **share** — частку внеску від суми всіх |внесків| (зі знаком):
«OI дав 38% усього поштовху». Не залежить від того, наскільки P близька до 1 (при P 0.98 «на скільки впала б P без
ознаки» майже нуль — сигмоїда впирається в стелю). Додатково impact — sigmoid(L) − sigmoid(L − внесок), для довідки.

Ознаки рахуються так само, як у скринері (features.build_15m: 15 днів історії + індекс альтів), тож P тут = P у signals.

    python drivers.py NMR             # drivers останнього оціненого 15m-бару
    python drivers.py NMR --ts "2026-10-07 11:00"
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

HISTORY = pd.Timedelta(days=16)          # як screener.score_15m для останнього бару (тижневі вікна + бета / Amihud)
TOP_FOR, TOP_AGAINST = 4, 2

# групи train.FEATURE_GROUPS → англійські назви для інтерфейсу
GROUP_EN = {"Власний рух": "Own move", "Ліквідність (Amihud)": "Liquidity", "OI": "Open interest", "Ціна": "Price",
            "Обсяг": "Volume", "Волатильність": "Volatility", "Купівлі / продажі": "Taker flow", "Funding": "Funding",
            "Ліквідації": "Liquidations", "Long / Short": "Long / short", "Інше": "Other"}

# ознака → (назва англійською, формат значення): pct — log-зміна → %, pct_raw — уже частка → %, x — множник,
# z — сигми, num — число, pp — відсоткові пункти, fund — funding у %, logx — log множника → ×
FEATURE_EN = {
    "oi_chg_1h": ("OI 1h", "pct"), "oi_chg_4h": ("OI 4h", "pct"), "oi_chg_24h": ("OI 24h", "pct"),
    "oi_z_24h": ("OI vs 24h", "z"), "oi_z_7d": ("OI vs 7d", "z"), "oi_minus_ret_24h": ("OI − price 24h", "pct"),
    "oi_chg_bar": ("OI last bar", "pct"), "oi_chg_2bar": ("OI 30m", "pct"), "oi_accel_1h": ("OI acceleration", "pp"),
    "ret_1h": ("Price 1h", "pct"), "ret_4h": ("Price 4h", "pct"), "ret_24h": ("Price 24h", "pct"),
    "ret_bar": ("Price last bar", "pct"), "ret_2bar": ("Price 30m", "pct"), "ret_accel_1h": ("Price acceleration", "pp"),
    "ret_4h_vol": ("Move 4h in σ", "z"), "ret_24h_vol": ("Move 24h in σ", "z"),
    "volume_z_7d": ("Volume vs 7d", "z"), "volume_ratio_24h": ("Volume 24h vs avg", "x"), "vol_burst": ("Volume burst", "x"),
    "volatility_24h": ("Volatility 24h", "pct_raw"), "range_1h": ("Range 1h", "pct_raw"), "range_bar": ("Range last bar", "pct_raw"),
    "park_24h": ("Parkinson vol 24h", "pct_raw"), "gk_24h": ("Garman–Klass vol 24h", "pct_raw"), "park_x": ("Vol vs 7d", "x"),
    "buy_ratio_1h": ("Taker buys 1h", "pct_raw"), "buy_ratio_4h": ("Taker buys 4h", "pct_raw"), "buy_ratio_bar": ("Taker buys bar", "pct_raw"),
    "funding": ("Funding", "fund"), "funding_z_7d": ("Funding vs 7d", "z"), "funding_chg_24h": ("Funding Δ24h", "fund"),
    "ls_long_pct": ("Long accounts", "pct100"), "ls_chg_24h": ("Long accounts Δ24h", "pp100"),
    "beta": ("Beta", "num"), "r2": ("R²", "num"), "idio_1h": ("Idio. ret 1h", "pct"), "idio_4h": ("Idio. ret 4h", "pct"),
    "idio_24h": ("Idio. ret 24h", "pct"), "idio_z_4h": ("Resid. z 4h", "z"), "illiq_x": ("ILLIQ vs 7d", "logx"),
    "liq_long_4h_oi": ("Long liqs 4h / OI", "pct_raw"), "liq_short_4h_oi": ("Short liqs 4h / OI", "pct_raw"),
    "liq_imbalance_4h": ("Liq. imbalance 4h", "num"), "liq_long_1h_oi": ("Long liqs 1h / OI", "pct_raw"),
    "liq_short_1h_oi": ("Short liqs 1h / OI", "pct_raw"),
}

# коротке пояснення для підказки в дашборді (українською, як решта «?»)
FEATURE_DESC = {
    "oi_chg_1h": "зміна відкритого інтересу за годину", "oi_chg_4h": "зміна OI за 4 год", "oi_chg_24h": "зміна OI за добу",
    "oi_z_24h": "OI відносно своєї доби, у σ", "oi_z_7d": "OI відносно свого тижня, у σ",
    "oi_minus_ret_24h": "OI росте швидше за ціну — набирають позиції", "oi_chg_bar": "зміна OI за останній 15m-бар",
    "oi_chg_2bar": "зміна OI за 30 хв", "oi_accel_1h": "OI за останню годину проти попередньої",
    "ret_1h": "зміна ціни за годину", "ret_4h": "зміна ціни за 4 год", "ret_24h": "зміна ціни за добу",
    "ret_bar": "зміна ціни за останній бар", "ret_2bar": "зміна ціни за 30 хв", "ret_accel_1h": "ціна за останню годину проти попередньої",
    "ret_4h_vol": "рух за 4 год у власних коливаннях монети", "ret_24h_vol": "рух за добу у власних коливаннях монети",
    "volume_z_7d": "обсяг бару відносно тижня, у σ", "volume_ratio_24h": "обсяг доби до середньої доби тижня",
    "vol_burst": "обсяг бару до середнього бару доби", "volatility_24h": "розкид годинних змін ціни за добу",
    "range_1h": "діапазон ціни за годину", "range_bar": "діапазон ціни останнього бару",
    "park_24h": "волатильність за high/low за добу", "gk_24h": "волатильність за open/high/low/close за добу",
    "park_x": "волатильність доби до звичайної (тиждень)", "buy_ratio_1h": "частка ринкових покупок за годину",
    "buy_ratio_4h": "частка ринкових покупок за 4 год", "buy_ratio_bar": "частка ринкових покупок за бар",
    "funding": "ставка фінансування зараз", "funding_z_7d": "funding відносно тижня, у σ", "funding_chg_24h": "зміна funding за добу",
    "ls_long_pct": "частка акаунтів у лонгу", "ls_chg_24h": "зміна частки лонгів за добу",
    "beta": "наскільки монета зазвичай ходить за ринком альтів", "r2": "яку частку коливань монети пояснює ринок",
    "idio_1h": "власний рух за годину (понад ринок)", "idio_4h": "власний рух за 4 год", "idio_24h": "власний рух за добу",
    "idio_z_4h": "наскільки власний рух за 4 год незвичний для монети", "illiq_x": "наскільки ціну зараз легко штовхнути, до норми",
}


def fmt_value(feature: str, v) -> str:
    if v is None or not np.isfinite(v):
        return "n/a"
    kind = FEATURE_EN.get(feature, (feature, "num"))[1]
    if kind == "pct":
        return f"{np.expm1(v):+.1%}"
    if kind == "pct_raw":
        return f"{v:.1%}" if abs(v) >= 0.001 else f"{v:.2%}"
    if kind == "pct100":
        return f"{v:.0f}%"
    if kind == "pp":
        return f"{v * 100:+.1f} pp"
    if kind == "pp100":
        return f"{v:+.1f} pp"
    if kind == "x":
        return f"×{v:.1f}"
    if kind == "logx":
        return f"×{np.exp(v):.1f}"
    if kind == "z":
        return f"{v:+.1f}σ"
    if kind == "fund":
        return f"{v:+.3f}%"
    return f"{v:.2f}"


def _sig(x: float) -> float:
    return float(1 / (1 + np.exp(-x)))


def explain(bundle: dict, row: pd.Series) -> dict:
    """Внески ознак у P моделі bundle для одного рядка ознак (row містить bundle["features"])."""
    import train
    feats = bundle["features"]
    X = pd.DataFrame([row[feats].astype(float).to_numpy()], columns=feats)
    c = bundle["model"].booster_.predict(X, pred_contrib=True)[0]
    contrib, bias = c[:-1], float(c[-1])
    L = bias + float(contrib.sum())
    p = _sig(L)

    total = float(np.abs(contrib).sum()) or 1.0

    def impact(v: float) -> float:          # на скільки P нижча (вища), якщо прибрати цей внесок
        return p - _sig(L - v)

    items = [{"feature": f, "label": FEATURE_EN.get(f, (f, ""))[0], "desc": FEATURE_DESC.get(f, ""), "value": None if not np.isfinite(row[f]) else float(row[f]),
              "text": fmt_value(f, row[f]), "group": GROUP_EN.get(train.feature_group(f), train.feature_group(f)),
              "contrib": float(v), "share": float(v) / total, "impact": impact(float(v))} for f, v in zip(feats, contrib)]
    groups: dict[str, float] = {}
    for it in items:
        groups[it["group"]] = groups.get(it["group"], 0.0) + it["contrib"]
    grp = sorted(({"group": g, "contrib": v, "share": v / total, "impact": impact(v)} for g, v in groups.items()),
                 key=lambda g: -abs(g["contrib"]))
    pos = sorted((i for i in items if i["contrib"] > 0), key=lambda i: -i["contrib"])
    neg = sorted((i for i in items if i["contrib"] < 0), key=lambda i: i["contrib"])
    warnings = []
    # сигнал без обсягу (власник обрав, 08.10): P ≥ порогу, а обсяг тягне вниз або нічого не дає — не той «сплеск з
    # обсягом», на який він входить (CHIP 06.10 08:45: Alert з R², funding і лонгів, обсяг −2%)
    if p >= float(bundle["threshold"]) and groups.get("Volume", 0.0) <= 0:
        warnings.append({"code": "no_volume", "text": "No volume behind this signal"})
    return {"p": p, "threshold": float(bundle["threshold"]), "base_p": _sig(bias),
            "for": pos[:TOP_FOR], "against": neg[:TOP_AGAINST], "groups": grp, "warnings": warnings}


_cache: dict = {}


def for_bar(conn, bundle: dict, base: str, ts=None) -> dict | None:
    """Drivers 15m-моделі для тікера на барі ts (відкриття 15m-бару; None — останній закритий бар з ознаками).
    Бар з конкретним ts кешується (ознаки закритого бару не змінюються): дашборд і бот питають той самий бар."""
    if ts is not None:
        key = (base, pd.Timestamp(ts), bundle.get("trained_at"))
        if key in _cache:
            return _cache[key]
        out = _for_bar(conn, bundle, base, ts)
        if len(_cache) > 500:
            _cache.clear()
        _cache[key] = out
        return out
    return _for_bar(conn, bundle, base, None)


def _for_bar(conn, bundle: dict, base: str, ts=None) -> dict | None:
    from features import build_15m
    end = pd.Timestamp(ts) if ts is not None else pd.Timestamp.now(tz="UTC")
    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    d = build_15m(conn, [base], since=end - HISTORY)
    if d.empty:
        return None
    d = d[d["ts"] <= end]
    if ts is not None:
        d = d[d["ts"] == end]
    if d.empty:
        return None
    row = d.iloc[-1]
    out = explain(bundle, row)
    out["ts"] = pd.Timestamp(row["ts"])
    return out


def text(dr: dict, n_for: int = 3, n_against: int = 1) -> str:
    """Коротко для Telegram: «Drivers: Idio. ret 24h +28% · OI 24h +35% · Volume vs 7d +3.1σ | against: Funding +0.05%»."""
    s = " · ".join(f"{i['label']} {i['text']}" for i in dr["for"][:n_for])
    a = " · ".join(f"{i['label']} {i['text']}" for i in dr["against"][:n_against])
    w = "".join(f"⚠ {x['text']}\n" for x in dr.get("warnings", []))
    return w + f"Drivers: {s}" + (f" | against: {a}" if a else "")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Drivers: внески ознак у ймовірність 15m-моделі на барі (SHAP, LightGBM pred_contrib)")
    ap.add_argument("ticker", help="базовий актив, напр. NMR")
    ap.add_argument("--ts", help="відкриття 15m-бару UTC, напр. '2026-10-07 11:00' (за замовчуванням — останній)")
    args = ap.parse_args()
    import psycopg
    from dotenv import load_dotenv
    import train
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
    bundle = train.load_model(train.M15_NAME)
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        dr = for_bar(conn, bundle, args.ticker.upper(), args.ts)
    if dr is None:
        sys.exit("немає ознак для цього бару")
    print(f"{args.ticker.upper()} {dr['ts']:%Y-%m-%d %H:%M} UTC · P {dr['p']:.3f} (поріг {dr['threshold']:.3f}, база {dr['base_p']:.3f})")
    print("за:")
    for i in dr["for"]:
        print(f"  {i['label']:<22} {i['text']:>10}   {i['share']:+.0%}  [{i['group']}]")
    print("проти:")
    for i in dr["against"]:
        print(f"  {i['label']:<22} {i['text']:>10}   {i['share']:+.0%}  [{i['group']}]")
    print("групи: " + ", ".join(f"{g['group']} {g['share']:+.0%}" for g in dr["groups"]))
    print(text(dr))


if __name__ == "__main__":
    main()
