"""
tg_bot.py — Telegram-бот Pump Detector (2026-10-06): сповіщення на телефон і результати журналу кнопками.

Бот — ще один «клієнт» дашборда, як браузер: сигнали бере з /api/list (той самий кеш, що й список тікерів), подробиці
тікера (P, стоп, ціль, ліміт) — з /api/ticker, результати — з /api/journal. Нічого не рахує сам — цифри ті самі, що в
дашборді. Скринер і дашборд не змінюються.

Що надсилає (вмикається кнопками в /settings): 🔴 Alert (вхід моделі), ⚪ Late (Alert без входу моделі, з 12.10),
🟠 Кандидат (за замовчуванням вимкнено), 🟣 «прокидається».
Одне повідомлення на епізод; під ним — 📈 Графік (посилання на дашборд), 🔕 заглушити тікер на 2 год, 📌 у watchlist
(для авто-тікерів сканера). Кнопки внизу чату: 📋 Статус, 📊 Результати (сьогодні / вчора / тиждень / місяць),
⚙️ Налаштування.

Приватний: власник — той, хто першим натиснув /start (або TELEGRAM_CHAT_ID у .env); іншим бот не відповідає.
Стан — reports/tg_state.json (власник, налаштування, заглушені тікери, що вже надіслано).

    python tg_bot.py                 # працювати (на сервері — контейнер tgbot)
    python tg_bot.py --check         # перевірити токен і доступ до дашборда, нічого не надсилати

Змінні (.env / docker-compose): TELEGRAM_BOT_TOKEN (обов'язково), DASH_URL (де дашборд; у Docker — http://dashboard:8766),
PUBLIC_URL (посилання «Графік», напр. https://45-76-90-13.sslip.io), TELEGRAM_CHAT_ID (необов'язково).
"""

import argparse
import html
import json
import os
import sys
import threading
import time
from datetime import datetime, timedelta, timezone

import requests
from dotenv import load_dotenv

import strategies

ROOT = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(ROOT, "reports", "tg_state.json")
MODEL = "tier15"                         # основна модель; режим — основний зі strategies.json (main_mode)


def main_mode() -> str:
    """Основний режим (\"main\": true у strategies.json) — той самий, за яким дашборд рахує список і сповіщення;
    читається щоразу, тож перемикання режиму не потребує перезапуску бота (12.10: market → split)."""
    return strategies.get(None)["id"]


SEND_MAX_AGE_MIN = 45                     # старіші епізоди не надсилаємо (після простою — без «зливи» старих сигналів)
MUTE_HOURS = 2
POLL_SEC = 5
# alert — лише ВХІД моделі (угода, як мітка «Вхід» на графіку); realert — новий спалах Alert, поки угода моделі відкрита
# (до 2026-10-08 такі приходили як звичайний Alert; власник: «заходжу з моделлю», тож за замовчуванням вимкнено)
DEFAULTS = {"alert": True, "realert": False, "late": True, "cand": False, "wake": True, "fp": True, "woke": True}
TITLES = {"alert": "🔴 Alert", "realert": "🔁 Re-alert", "late": "⚪ Late", "cand": "🟠 Candidate", "wake": "🟣 Volume 5m", "fp": "🟢 b / p",
          "woke": "🟪 WOKE"}
MENU = [["📋 Статус", "📊 Результати", "⚙️ Налаштування"]]


def now() -> datetime:
    return datetime.now(timezone.utc)


def ts_of(v) -> datetime | None:
    """Час з API: секунди епохи (як віддає дашборд) або ISO-рядок."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return datetime.fromtimestamp(v, tz=timezone.utc)
    try:
        return datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except ValueError:
        return None


def pct(v, d=1) -> str:
    return "—" if v is None else f"{'+' if v >= 0 else '−'}{abs(v) * 100:.{d}f}%"


def px(v) -> str:
    if v is None:
        return "—"
    a = abs(v)
    return f"{v:.1f}" if a >= 1000 else f"{v:.2f}" if a >= 100 else f"{v:.3f}" if a >= 1 else f"{v:.4g}"


def p2(v) -> str:
    return "—" if v is None else f"{v:.2f}"


class State:
    def __init__(self):
        self.lock = threading.Lock()
        try:
            with open(STATE_FILE, encoding="utf-8") as f:
                self.d = json.load(f)
        except (OSError, ValueError):
            self.d = {}
        self.d.setdefault("owner", None)
        self.d.setdefault("settings", dict(DEFAULTS))
        self.d.setdefault("muted", {})
        self.d.setdefault("sent", {})
        self.d.setdefault("offset", 0)

    def save(self):
        with self.lock:
            cut = time.time() - 3 * 86400
            self.d["sent"] = {k: v for k, v in self.d["sent"].items() if v >= cut}
            self.d["muted"] = {k: v for k, v in self.d["muted"].items() if v > time.time()}
            os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
            tmp = STATE_FILE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.d, f, ensure_ascii=False, indent=1)
            os.replace(tmp, STATE_FILE)


class Bot:
    def __init__(self, token: str, dash: str, public: str, chat_id: str | None):
        self.api = f"https://api.telegram.org/bot{token}/"
        self.dash = dash.rstrip("/")
        self.public = public.rstrip("/")
        self.st = State()
        if chat_id:
            self.st.d["owner"] = int(chat_id)
        self.http = requests.Session()

    # ---------- Telegram ----------
    def tg(self, method: str, http_timeout: float = 30, **params):
        r = self.http.post(self.api + method, json=params, timeout=http_timeout)
        j = r.json()
        if not j.get("ok"):
            raise RuntimeError(f"{method}: {j.get('description')}")
        return j["result"]

    def send(self, text: str, buttons: list | None = None, chat: int | None = None, menu: bool = False):
        markup = {"inline_keyboard": buttons} if buttons else ({"keyboard": [[{"text": t} for t in row] for row in MENU],
                                                                 "resize_keyboard": True, "is_persistent": True} if menu else None)
        return self.tg("sendMessage", chat_id=chat or self.st.d["owner"], text=text, parse_mode="HTML",
                       disable_web_page_preview=True, **({"reply_markup": markup} if markup else {}))

    # ---------- дашборд ----------
    def get(self, path: str, timeout: float = 120):
        r = self.http.get(self.dash + path, timeout=timeout)
        r.raise_for_status()
        return r.json()

    def chart_url(self, tk: str) -> str:
        return f"{self.public}/#forecast/{tk}"

    # ---------- сповіщення ----------
    def signal_buttons(self, tk: str, auto: bool) -> list:
        row = [{"text": "📈 Графік", "url": self.chart_url(tk)}, {"text": f"🔕 {tk} на {MUTE_HOURS} год", "callback_data": f"mute:{tk}"}]
        rows = [row]
        if auto:
            rows.append([{"text": "📌 У watchlist (не випаде зі списку)", "callback_data": f"watch:{tk}"}])
        return rows

    def levels(self, tk: str) -> tuple[str, dict]:
        """Рядки про ціну, P, стоп і ціль з даних тікера (ті самі правила, що в дашборді)."""
        d = self.get(f"/api/ticker?symbol={requests.utils.quote(tk)}&model={MODEL}&strategy={main_mode()}&days=3", timeout=60)
        c, pos, ctx = d.get("cards") or {}, d.get("position") or {}, d.get("ctx") or {}
        price = d.get("price")
        lines = [f"<b>{html.escape(tk)}</b> {px(price)} ({pct(d.get('chg_24h'))} за добу)"]
        need = ctx["share"] * ctx["thr1h"] if ctx.get("thr1h") is not None and ctx.get("share") is not None else None
        lines.append(f"P {p2(c.get('probability'))} · 1h {p2(ctx.get('p1h'))}" + (f" (для Alert ≥ {p2(need)})" if need else "")
                     + f" · OI 4 год {pct(c.get('oi_chg_4h'))}")
        if pos.get("open") and pos.get("entry"):
            e = pos["entry"]
            s = f"Вхід {px(e)} · стоп {px(pos.get('stop'))} ({pct(pos['stop'] / e - 1) if pos.get('stop') else '—'})"
            if pos.get("partial_ts"):
                s += " · ½ уже закрито, стоп у беззбитку"
            elif pos.get("target") is not None:
                s += f" · ціль ½ {px(pos['target'])} ({pct(pos['target'] / e - 1)})"
            lines.append(s)
        elif price and d.get("atr") and d.get("k") is not None:
            stop = price - d["k"] * d["atr"]
            lines.append(f"Стоп, якщо вхід зараз: {px(stop)} ({pct(stop / price - 1)})")
        lim = d.get("limit")
        if lim and lim.get("price"):
            until = ts_of(lim.get("until"))
            lines.append(f"Ліміт {px(lim['price'])} (−0.5 ATR{f', до {until:%H:%M} UTC' if until else ''})")
        # Drivers (drivers.py): що дало P на цьому барі; ⚠ — сигнал без обсягу
        dr = d.get("drivers")
        if dr and dr.get("for"):
            lines += [f"⚠ <b>{html.escape(w['text'])}</b>" for w in dr.get("warnings", [])]
            lines.append("<b>Drivers:</b> " + " · ".join(f"{html.escape(i['label'])} {html.escape(i['text'])}" for i in dr["for"][:3]))
            if dr.get("against"):
                lines.append("<i>against: " + " · ".join(f"{html.escape(i['label'])} {html.escape(i['text'])}"
                                                         for i in dr["against"][:1]) + "</i>")
        return "\n".join(lines), d

    def notify_once(self, first: bool):
        lst = self.get(f"/api/list?model={MODEL}", timeout=180)
        rows = {r["ticker"]: r for r in lst.get("rows", [])}
        s = self.st.d
        fresh = []
        for kind, items in (("alert", lst.get("alerts", [])), ("realert", lst.get("realerts", [])),
                            ("late", lst.get("lates", [])), ("cand", lst.get("cands", [])), ("wake", lst.get("wakes", [])),
                            ("fp", lst.get("fps", [])), ("woke", lst.get("wokeups", []))):
            for e in items:
                key = f"{kind}|{e['ticker']}|{e['ts']}" + (f"|{e['kind']}" if kind == "fp" else "")
                if key in s["sent"]:
                    continue
                with self.st.lock:
                    s["sent"][key] = time.time()
                t = ts_of(e["ts"])
                if first or t is None:                      # при старті — лише запам'ятати, без надсилання
                    continue
                age = (now() - t).total_seconds() / 60 - (15 if kind not in ("wake", "woke") else 0)  # від закриття бару
                if age > SEND_MAX_AGE_MIN:
                    continue
                fresh.append((kind, e))
        alerted = {e["ticker"] for k, e in fresh if k == "alert"}
        for kind, e in fresh:
            tk = e["ticker"]
            if not s["settings"].get(kind, DEFAULTS[kind]) or s["muted"].get(tk, 0) > time.time():
                continue
            if kind == "cand" and tk in alerted and s["settings"].get("alert"):
                continue                                    # на тому ж тікері одразу Alert — лише Alert
            try:
                self.send_signal(kind, e, rows.get(tk, {}))
            except Exception as ex:
                print(f"! надсилання {kind} {tk}: {ex}", flush=True)
        self.st.save()

    def send_signal(self, kind: str, e: dict, row: dict):
        tk = e["ticker"]
        t = ts_of(e["ts"])
        if kind == "fp":
            text = self.fp_text(e)
        elif kind == "woke":
            text = self.woke_text(e, row)
        elif kind == "realert":
            tr = e.get("trade") or {}
            et = ts_of(tr.get("entry_ts"))
            text = (f"🔁 <b>Re-alert</b> · <b>{html.escape(tk)}</b> · P {p2(e.get('probability'))}\n"
                    f"Модель уже в угоді" + (f" з {et:%H:%M} UTC" if et else "") + f" (вхід {px(tr.get('entry'))}, зараз "
                    f"{pct(tr.get('result_pct'))}) — Alert спалахнув знову, нового входу немає"
                    + (f"\n15m-бар {t:%H:%M}–{(t + timedelta(minutes=15)):%H:%M} UTC" if t else ""))
        elif kind == "wake":
            head = f"🟣 <b>Прокидається</b> · <b>{html.escape(tk)}</b>"
            body = (f"обсяг ×{e.get('vol_x', 0):.1f} до тижня · ціна {pct(e.get('ret15'))} за 15 хв · OI {pct(e.get('oi1h'))} за год\n"
                    f"5m-бар до {t:%H:%M} UTC" if t else "")
            text = head + "\n" + body + "\n<i>Не сигнал на вхід — монета ожила, подивись.</i>"
        elif kind == "late":
            # Alert без входу основного режиму (split: «старт» і годинна не збіглися на одному барі, як CHIP 10-10;
            # або повторний сигнал після стопу) — модель не бере, рішення за власником (12.10)
            lv, _ = self.levels(tk)
            text = (f"⚪ <b>Late</b> · {lv}\n<i>Alert без входу моделі — вирішуй сам: відкат і стакан.</i>"
                    + (f"\n15m-бар {t:%H:%M}–{(t + timedelta(minutes=15)):%H:%M} UTC" if t else ""))
        else:
            lv, _ = self.levels(tk)
            head = (("🔴 <b>Alert</b> · вхід моделі" + (" (повторний, після стопу)" if e.get("reentry") else ""))
                    if kind == "alert" else "🟠 <b>Кандидат</b> — перевір стакан")
            text = f"{head} · {lv}\n15m-бар {t:%H:%M}–{(t + timedelta(minutes=15)):%H:%M} UTC" if t else f"{head} · {lv}"
        self.send(text, self.signal_buttons(tk, (row.get("source") or e.get("source")) == "auto"))
        print(f"[{now():%H:%M:%S}] надіслано {kind} {tk}", flush=True)

    @staticmethod
    def woke_text(e: dict, row: dict) -> str:
        """Канал WOKE (scanner.py): тиха монета ожила — перше спрацювання. Не сигнал входу (experiments/wake_channel.py)."""
        mln = lambda v: "—" if v is None else f"{v / 1e6:.1f}" if v < 10e6 else f"{v / 1e6:.0f}"      # noqa: E731
        sg = lambda v: "—" if v is None else f"{v:+.0f}%"                                         # noqa: E731
        p, thr = row.get("probability"), row.get("threshold")
        pl = (f"P моделі {p2(p)} ({'вище' if thr is not None and p >= thr else 'нижче'} порогу {p2(thr)})"
              if p is not None else "P моделі — ще не оцінено (монета щойно в списку)")
        return (f"🟪 <b>WOKE</b> · <b>{html.escape(e['ticker'])}</b> — тиха монета ожила\n"
                f"Оборот: сер. <b>{mln(e.get('turnover_avg'))} млн</b>/добу → зараз <b>{mln(e.get('pace'))} млн</b>/добу\n"
                f"Обсяг 4 год <b>×{e.get('vol_x') or 0:.0f}</b> до норми · ціна <b>{sg(e.get('price_pct'))}</b> · "
                f"OI <b>{sg(e.get('oi_pct'))}</b> за 4 год\n{pl}\n"
                "<i>Не сигнал входу: такі монети в 42% випадків дають +10% за добу, але в медіані за добу ціна −5%. "
                "Вхід — за стаканом.</i>")

    @staticmethod
    def fp_text(e: dict) -> str:
        """Мітка футпрінту (footprint.py): b — «залочка» на лоу (підказка на лонг), p — те саме на хаях."""
        tk, t, c, stop = html.escape(e["ticker"]), ts_of(e["ts"]), e.get("close"), e.get("stop")
        ctx = "Alert" if e.get("ctx") == "alert" else "Кандидат"
        bar = f"15m-бар {t:%H:%M}–{(t + timedelta(minutes=15)):%H:%M} UTC" if t else ""
        if e.get("kind") == "b":
            return (f"🟢 <b>b · залочка</b> · <b>{tk}</b>\n"
                    f"Прокол лоу {px(e.get('level'))} і викуп: під лоу {e.get('beyond', 0):.0%} обсягу бару "
                    f"(дельта {e.get('beyond_delta', 0):+.0%}), обсяг ×{e.get('vol_x', 0):.1f}, закриття {px(c)}\n"
                    f"Контекст: {ctx} за 6 год · стоп під лоу {px(stop)} ({pct(stop / c - 1) if c and stop else '—'})\n"
                    f"{bar}\n<i>Підказка з футпрінту, не сигнал моделі — перевір стакан.</i>")
        return (f"🟠 <b>p · на хаях</b> · <b>{tk}</b>\n"
                f"Прокол хаю {px(e.get('level'))} і продаж: над хаєм {e.get('beyond', 0):.0%} обсягу бару "
                f"(дельта {e.get('beyond_delta', 0):+.0%}), обсяг ×{e.get('vol_x', 0):.1f}, закриття {px(c)} — нижче хаю\n"
                f"Контекст: {ctx} за 6 год · покупців на хаях поглинули — обережно з входом, подумай про вихід\n{bar}")

    def notifier(self):
        last_pulse, first, last_full = None, True, 0.0
        while True:
            try:
                if self.st.d["owner"]:
                    p = self.get("/api/pulse", timeout=20)
                    key = f"{p.get('sig')}|{p.get('wake')}|{p.get('woke')}"
                    if key != last_pulse or time.time() - last_full > 60:
                        self.notify_once(first)
                        first, last_pulse, last_full = False, key, time.time()
            except Exception as ex:
                print(f"! сповіщення: {type(ex).__name__}: {ex}", flush=True)
            time.sleep(POLL_SEC)

    # ---------- команди і кнопки ----------
    def settings_markup(self) -> list:
        s = self.st.d["settings"]
        row = [{"text": ("✅ " if s.get(k, DEFAULTS[k]) else "⬜ ") + TITLES[k], "callback_data": f"set:{k}"}
               for k in ("alert", "realert", "cand", "wake", "fp", "woke")]
        return [row[:2], row[2:4], row[4:], [{"text": "🔕 Заглушені", "callback_data": "muted"}]]

    def results_markup(self) -> list:
        return [[{"text": "Сьогодні", "callback_data": "res:today"}, {"text": "Вчора", "callback_data": "res:yday"},
                 {"text": "Тиждень", "callback_data": "res:week"}, {"text": "Місяць", "callback_data": "res:month"}]]

    def results(self, period: str) -> str:
        today = now().date()
        a, b, title = {
            "today": (today, today, "Сьогодні"), "yday": (today - timedelta(days=1), today - timedelta(days=1), "Вчора"),
            "week": (today - timedelta(days=6), today, "Тиждень"), "month": (today - timedelta(days=29), today, "Місяць"),
        }[period]
        j = self.get(f"/api/journal?model={main_mode()}&kind=hist&from={a:%Y-%m-%d}&to={b:%Y-%m-%d}", timeout=180)
        rows = j.get("rows", [])
        closed = [r for r in rows if not r.get("open") and r.get("result_pct") is not None]
        opened = [r for r in rows if r.get("open")]
        span = f"{a:%d.%m}" if a == b else f"{a:%d.%m}–{b:%d.%m}"
        lines = [f"📊 <b>{title}</b> ({span}, UTC) · 15m + 1h · ринок"]
        if not closed:
            lines.append("Закритих угод немає.")
        else:
            res = [r["result_pct"] for r in closed]
            wins, losses = [v for v in res if v > 0], [v for v in res if v <= 0]
            pf = sum(wins) / -sum(losses) if losses and sum(losses) < 0 else None
            best = max(closed, key=lambda r: r["result_pct"])
            worst = min(closed, key=lambda r: r["result_pct"])
            stops = sum(1 for r in closed if r.get("reason") == "stop")
            lines.append(f"Закритих угод {len(closed)} · у плюсі {len(wins) / len(closed):.0%}"
                         + (f" · PF {pf:.2f}" if pf is not None else "") + f" · сума {pct(sum(res))}")
            lines.append(f"Середня {pct(sum(res) / len(res))} · по стопу {stops}")
            lines.append(f"Найкраща {html.escape(best['ticker'])} {pct(best['result_pct'])} · найгірша "
                         f"{html.escape(worst['ticker'])} {pct(worst['result_pct'])}")
        if opened:
            op = ", ".join(f"{html.escape(r['ticker'])} {pct(r.get('result_pct'))}" for r in opened[:8])
            lines.append(f"Відкриті ({len(opened)}): {op}")
        lines.append("<i>live + історія, як у журналі дашборда</i>")
        try:                                            # мітки футпрінту b / p: результат «+2R / стоп / 12 год»
            fs = self.get(f"/api/fp_stats?from={a:%Y-%m-%d}&to={b:%Y-%m-%d}", timeout=60)
            for k, name in (("b", "🟢 b (залочка)"), ("p", "🟠 p (на хаях)"), ("b_hist", "▫️ b · історія"), ("p_hist", "▫️ p · історія")):
                x = fs.get(k) or {}
                if x.get("n"):
                    lines.append(f"{name}: {x['n']} міток, завершено {x['done']}: +2R {x['target']}, стоп {x['stop']}, "
                                 f"12 год {x['time']}" + (f" · сер. {x['avg_R']:+.2f}R" if x.get("avg_R") is not None else ""))
        except Exception as ex:
            print(f"! b/p статистика: {ex}", flush=True)
        return "\n".join(lines)

    def status(self) -> str:
        lst = self.get(f"/api/list?model={MODEL}", timeout=180)
        rows = lst.get("rows", [])
        al = [r for r in rows if r.get("status") == "alert"]
        ca = [r for r in rows if r.get("status") == "candidate"]
        fmt = lambda rs: ", ".join(f"{html.escape(r['ticker'])} {p2(r.get('probability'))}" for r in rs[:12])  # noqa: E731
        lines = ["📋 <b>Зараз</b>"]
        lines.append(f"🔴 Alert ({len(al)}): {fmt(al)}" if al else "🔴 Alert: немає")
        lines.append(f"🟠 Кандидати ({len(ca)}): {fmt(ca)}" if ca else "🟠 Кандидати: немає")
        lines.append(f"Усього тікерів у списку: {len(rows)}")
        return "\n".join(lines)

    def on_message(self, m: dict):
        chat, text = m["chat"]["id"], (m.get("text") or "").strip()
        s = self.st.d
        if s["owner"] is None and text.startswith("/start"):
            s["owner"] = chat
            self.st.save()
            print(f"власник бота: chat {chat}", flush=True)
        if chat != s["owner"]:
            self.send("Це приватний бот.", chat=chat)
            return
        if text.startswith("/start") or text.startswith("/help"):
            self.send("Pump Detector на зв'язку. Надсилатиму сигнали, щойно скринер їх побачить.\n"
                      "Кнопки внизу: 📋 Статус — що горить зараз, 📊 Результати — журнал за період, ⚙️ Налаштування — які сповіщення надсилати.",
                      menu=True)
            self.send("⚙️ Що надсилати:", self.settings_markup())
        elif text in ("📋 Статус", "/status"):
            self.send(self.status())
        elif text in ("📊 Результати", "/results"):
            self.send("📊 За який період?", self.results_markup())
        elif text in ("⚙️ Налаштування", "/settings"):
            self.send("⚙️ Що надсилати:", self.settings_markup())

    def on_callback(self, cq: dict):
        chat = cq["message"]["chat"]["id"]
        if chat != self.st.d["owner"]:
            self.tg("answerCallbackQuery", callback_query_id=cq["id"])
            return
        data, s, note = cq.get("data", ""), self.st.d, ""
        kind, _, arg = data.partition(":")
        if kind == "set" and arg in DEFAULTS:
            s["settings"][arg] = not s["settings"].get(arg, DEFAULTS[arg])
            self.st.save()
            self.tg("editMessageReplyMarkup", chat_id=chat, message_id=cq["message"]["message_id"],
                    reply_markup={"inline_keyboard": self.settings_markup()})
            note = f"{TITLES[arg]}: {'увімкнено' if s['settings'][arg] else 'вимкнено'}"
        elif kind == "mute":
            s["muted"][arg] = time.time() + MUTE_HOURS * 3600
            self.st.save()
            note = f"{arg}: тиша на {MUTE_HOURS} год"
        elif kind == "muted":
            m = {k: v for k, v in s["muted"].items() if v > time.time()}
            note = "Заглушених немає" if not m else "Заглушені: " + ", ".join(
                f"{k} до {datetime.fromtimestamp(v, tz=timezone.utc):%H:%M}" for k, v in m.items())
        elif kind == "watch":
            try:
                r = self.http.post(self.dash + "/api/watchlist/add", json={"symbol": arg}, timeout=60)
                note = f"{arg} у watchlist" if r.ok else f"Не вдалось: {r.text[:80]}"
            except Exception as ex:
                note = f"Не вдалось: {ex}"
        elif kind == "res":
            self.tg("answerCallbackQuery", callback_query_id=cq["id"], text="Рахую…")
            try:
                self.send(self.results(arg), self.results_markup())
            except Exception as ex:
                self.send(f"Журнал не відповів: {ex}")
            return
        self.tg("answerCallbackQuery", callback_query_id=cq["id"], text=note[:190])

    def poll_updates(self):
        while True:
            try:
                # long polling: Telegram тримає запит до 25 с, поки не з'явиться повідомлення чи натискання кнопки
                ups = self.tg("getUpdates", http_timeout=40, offset=self.st.d["offset"], timeout=25,
                              allowed_updates=["message", "callback_query"])
            except Exception as ex:
                print(f"! getUpdates: {ex}", flush=True)
                time.sleep(5)
                continue
            for u in ups:
                self.st.d["offset"] = u["update_id"] + 1
                try:
                    if "message" in u:
                        self.on_message(u["message"])
                    elif "callback_query" in u:
                        self.on_callback(u["callback_query"])
                except Exception as ex:
                    print(f"! оновлення {u.get('update_id')}: {type(ex).__name__}: {ex}", flush=True)
            if ups:
                self.st.save()

    def run(self):
        self.tg("setMyCommands", commands=[
            {"command": "status", "description": "Що горить зараз"},
            {"command": "results", "description": "Результати журналу за період"},
            {"command": "settings", "description": "Які сповіщення надсилати"},
        ])
        me = self.tg("getMe")
        print(f"Бот @{me['username']} запущено; власник: {self.st.d['owner'] or 'ще немає — натисни /start у Telegram'}", flush=True)
        threading.Thread(target=self.notifier, daemon=True).start()
        self.poll_updates()


def main():
    ap = argparse.ArgumentParser(description="Telegram-бот Pump Detector: сповіщення про сигнали і результати журналу")
    ap.add_argument("--check", action="store_true", help="перевірити токен і доступ до дашборда, нічого не надсилати")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(os.path.join(ROOT, ".env"))
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        sys.exit("Заповни TELEGRAM_BOT_TOKEN у .env (токен від @BotFather)")
    bot = Bot(token, os.getenv("DASH_URL", "http://127.0.0.1:8766"), os.getenv("PUBLIC_URL", "http://127.0.0.1:8766"),
              os.getenv("TELEGRAM_CHAT_ID"))
    if args.check:
        me = bot.tg("getMe")
        p = bot.get("/api/pulse", timeout=20)
        print(f"Токен OK: @{me['username']}; дашборд відповідає (pulse {p}); власник: {bot.st.d['owner']}")
        return
    bot.run()


if __name__ == "__main__":
    main()
