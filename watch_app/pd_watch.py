"""
PD Watchlist — маленьке вікно Windows лише зі списком тікерів Pump Detector (2026-10-15).

Обгортка pywebview (WebView2 — той самий рушій, що в Edge) над сторінкою дашборда `#list`: ті самі секції NOW / WOKE UP /
WATCHLIST / QUIET, чіпи, значки, звук. Чого не вміє вкладка браузера, а вміє застосунок:
  • вікна «поверх усіх» — список і графіки разом (перемикач у треї, запам'ятовується);
  • іконка в треї: клік — показати / сховати; хрестик вікна ховає його в трей, а не закриває;
  • системні сповіщення (Alert, Spread…) — через трей, бо в WebView2 сповіщень браузера немає;
  • звук без першого кліку по сторінці;
  • клік по тікеру відкриває його в окремому вікні графіка «PD Chart» (одне на всі кліки — новий тікер підміняє попередній;
    браузер і дашборд у ньому не чіпає: у застосунку своє сховище налаштувань); Shift + клік — тікер у НОВОМУ окремому
    вікні (кілька графіків одночасно; хрестик закриває його насовсім); пункт трею «Відкрити дашборд» — браузер;
  • Alt + клік по тікеру — окреме вікно «PD Flow» (лише картка Flow: хто зараз діє на монеті, сторінка /flow#TK; 22.10),
    так само одне на всі Alt + кліки, Alt + Shift + клік — ще одне; кнопка ⇱ на картці Flow у вікні графіка — те саме;
  • Ctrl + / Ctrl − / Ctrl 0 — масштаб сторінки і розмір вікна разом, у кожному вікні окремо (запам'ятовується).

Налаштування — %APPDATA%\\PD Watchlist\\config.json (створюється при першому запуску):
  url       — адреса дашборда (за замовчуванням https://45-76-90-13.sslip.io/; через тунель — http://localhost:8766/);
  user, password — логін дашборда, якщо не хочеш вводити його щоразу (інакше WebView2 спитає сам);
  on_top, x, y, width, height, zoom — вікно списку; chart_x, chart_y — основне вікно графіка; chart_width, chart_height,
  chart_zoom — розмір і масштаб графіка, який змінювали останнім (з ними відкривається кожне нове вікно графіка);
  flow_x, flow_y, flow_width, flow_height, flow_zoom — те саме для вікон PD Flow.

    python watch_app/pd_watch.py                       # запуск без збирання
    python watch_app/pd_watch.py --url http://127.0.0.1:8799/
    watch_app\\build.cmd                                # → dist\\PD Watchlist.exe
"""

import argparse
import json
import os
import re
import sys
import threading
import webbrowser
from urllib.parse import quote, urlsplit, urlunsplit

APP = "PD Watchlist"
CFG_DIR = os.path.join(os.environ.get("APPDATA", os.path.expanduser("~")), APP)
CFG_FILE = os.path.join(CFG_DIR, "config.json")
DEFAULTS = {"url": "https://45-76-90-13.sslip.io/", "user": "", "password": "", "on_top": True,
            "x": None, "y": None, "width": 420, "height": 900, "zoom": 1.0,
            "chart_x": None, "chart_y": None, "chart_width": 1100, "chart_height": 720, "chart_zoom": 1.0,
            "flow_x": None, "flow_y": None, "flow_width": 380, "flow_height": 330, "flow_zoom": 1.0}
ZOOM_MIN, ZOOM_MAX, ZOOM_STEP = 0.5, 2.0, 0.1
HERE = os.path.dirname(os.path.abspath(__file__))
ICON_FILE = os.path.join(HERE, "pd_watch.ico")


# ---------------------------------------------------------------- налаштування

def load_config() -> dict:
    try:
        with open(CFG_FILE, encoding="utf-8") as f:
            saved = json.load(f)
    except (OSError, ValueError):
        saved = {}
    cfg = {**DEFAULTS, **{k: v for k, v in saved.items() if k in DEFAULTS}}
    if not os.path.exists(CFG_FILE):
        save_config(cfg)
    return cfg


def save_config(cfg: dict) -> None:
    os.makedirs(CFG_DIR, exist_ok=True)
    tmp = CFG_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=1)
    os.replace(tmp, CFG_FILE)


def base_url(cfg: dict) -> str:
    u = cfg["url"].strip()
    return u if u.endswith("/") else u + "/"


def page_url(cfg: dict, view: str = "list", page: str = "") -> str:
    """Адреса вікна: сторінка дашборда з #view («list» — лише список, «chart/TK» — лише графік) або інша сторінка page
    («flow» → /flow#TK — вікно PD Flow); логін, якщо заданий, — в адресі."""
    parts = urlsplit(base_url(cfg))
    netloc = parts.netloc
    if cfg.get("user") and cfg.get("password"):
        netloc = f"{quote(cfg['user'], safe='')}:{quote(cfg['password'], safe='')}@{parts.hostname}" + (f":{parts.port}" if parts.port else "")
    path = parts.path or "/"
    if page:
        path = path.rstrip("/") + "/" + page
    return urlunsplit((parts.scheme, netloc, path, "", view))


# ---------------------------------------------------------------- іконка

def icon_image(size: int = 64):
    """Помаранчевий квадрат з «P» — як логотип дашборда."""
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, size - 1, size - 1), radius=size // 5, fill=(216, 90, 48, 255))
    try:
        font = ImageFont.truetype("segoeuib.ttf", int(size * 0.62))
    except OSError:
        font = ImageFont.load_default()
    d.text((size / 2, size / 2), "P", fill=(15, 15, 14, 255), font=font, anchor="mm")
    return img


def make_icon() -> None:
    icon_image(256).save(ICON_FILE, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    print("іконка:", ICON_FILE)


# ---------------------------------------------------------------- застосунок

class Api:
    """Функції, які сторінка викликає як window.pywebview.api.<назва>()."""

    def __init__(self, app: "WatchApp", kind: str = "list"):
        self._app = app
        self._kind = kind

    def open_dashboard(self, ticker: str = "") -> None:
        webbrowser.open(base_url(self._app.cfg) + (f"#forecast/{ticker}" if ticker else "#forecast"))

    def open_chart(self, ticker: str, new: bool = False) -> None:
        """Клік по тікеру у списку: вікно графіка (браузер не чіпаємо); new=True (Shift + клік) — нове окреме вікно."""
        self._app.open_chart(ticker, bool(new))

    def open_flow(self, ticker: str, new: bool = False) -> None:
        """Alt + клік по тікеру у списку (або ⇱ на картці Flow): вікно PD Flow; new=True (Alt + Shift + клік) — ще одне."""
        self._app.open_chart(ticker, bool(new), what="flow")

    def chart_title(self, ticker: str) -> None:
        self._app.chart_title(ticker, self._kind)

    def notify(self, title: str, body: str = "") -> None:
        self._app.notify(title, body)

    def zoom(self, step: int) -> float:
        """Ctrl + / Ctrl − / Ctrl 0 зі сторінки: +1 більше, −1 менше, 0 — 100% (того вікна, звідки натиснуто)."""
        return self._app.zoom(step, self._kind)


class ChartWin:
    """Вікно графіка (what="chart") або PD Flow (what="flow", 22.10 — лише картка Flow, свої розмір / місце / масштаб у
    config flow_*): основне (звичайний клік / Alt + клік; хрестик ховає, позиція запам'ятовується) або додаткове (з Shift;
    хрестик закриває насовсім, стає драбинкою від основного). Нове вікно відкривається з розміром і масштабом вікна того ж
    типу, яке змінювали ОСТАННІМ (мишею чи Ctrl + / −): config <what>_width / _height / _zoom — «останні», їх пише кожне вікно.
    У кожного вікна свій поточний масштаб — щоб Ctrl + в одному не рахувався від масштабу іншого."""

    TITLE = {"chart": "PD Chart", "flow": "PD Flow"}

    def __init__(self, app: "WatchApp", kind: str, ticker: str, main: bool, n: int = 0, what: str = "chart"):
        import webview
        c = app.cfg
        self.app, self.kind, self.main, self.what = app, kind, main, what
        self.zoom = float(c.get(f"{what}_zoom") or 1.0)
        self.width, self.height = int(c[f"{what}_width"]), int(c[f"{what}_height"])
        shift = 0 if main else 32 * n                     # додаткові — драбинкою від основного, щоб не лягали одне на одне
        x = c[f"{what}_x"] + shift if c[f"{what}_x"] is not None else None
        y = c[f"{what}_y"] + shift if c[f"{what}_y"] is not None else None
        url = page_url(c, f"chart/{ticker}") if what == "chart" else page_url(c, ticker, page="flow")
        self.win = webview.create_window(
            f"{ticker} — {self.TITLE[what]}", url, js_api=Api(app, kind),
            width=self.width, height=self.height, x=x, y=y, on_top=bool(c["on_top"]),
            min_size=(480, 360) if what == "chart" else (280, 160), background_color="#141413", text_select=True)
        self.win.events.resized += self.on_resized
        self.win.events.loaded += self.on_loaded
        if main:
            self.win.events.moved += self.on_moved
            self.win.events.closing += self.on_closing
        else:
            self.win.events.closed += self.on_closed

    def remember(self) -> None:
        """Цей графік змінили останнім — його розмір і масштаб стають стартовими для нових вікон."""
        self.app.cfg.update({f"{self.what}_width": self.width, f"{self.what}_height": self.height, f"{self.what}_zoom": self.zoom})

    def on_resized(self, width, height):
        self.width, self.height = width, height
        self.remember()

    def on_moved(self, x, y):
        self.app.cfg[f"{self.what}_x"], self.app.cfg[f"{self.what}_y"] = x, y

    def on_loaded(self):
        if abs(self.zoom - 1.0) > 1e-6:
            self.app.set_factor(self.win, self.zoom)

    def on_closing(self):
        if self.app.quitting:
            return True
        self.win.hide()                            # хрестик ховає; наступний клік по тікеру покаже те саме вікно
        return False

    def on_closed(self):
        self.app.extra.pop(self.kind, None)

    def set_zoom(self, step: int) -> float:
        new = 1.0 if step == 0 else min(ZOOM_MAX, max(ZOOM_MIN, round(self.zoom + step * ZOOM_STEP, 2)))
        if new != self.zoom:
            k = new / self.zoom
            self.zoom = new
            self.app.set_factor(self.win, new)
            self.width, self.height = int(self.width * k), int(self.height * k)
            self.win.resize(self.width, self.height)
            self.remember()
            save_config(self.app.cfg)
        return new


class WatchApp:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.window = None                        # список
        self.chart: ChartWin | None = None        # основне вікно графіка (створюється при першому кліку по тікеру)
        self.flow: ChartWin | None = None         # основне вікно PD Flow (перший Alt + клік)
        self.extra: dict[str, ChartWin] = {}      # додаткові вікна графіка і Flow (з Shift)
        self._n_extra = 0
        self.tray = None
        self.visible = True
        self.quitting = False

    # --- трей
    def notify(self, title: str, body: str) -> None:
        if self.tray is not None:
            try:
                self.tray.notify(body or title, title)
            except Exception:
                pass

    def toggle(self, *_):
        if self.visible:
            self.window.hide()
        else:
            self.window.show()
            self.window.restore()
        self.visible = not self.visible

    def set_on_top(self, *_):
        """«Поверх усіх вікон» у треї — для списку і всіх вікон графіка разом (власник, 15.10)."""
        self.cfg["on_top"] = not self.cfg["on_top"]
        for w in [self.window, *(x.win for x in self._charts())]:
            if w is not None:
                w.on_top = self.cfg["on_top"]
        save_config(self.cfg)

    # --- масштаб: WebView2.ZoomFactor (шрифт і все на сторінці) + розмір вікна в тій самій пропорції
    def _charts(self) -> list:
        return [x for x in [self.chart, self.flow, *self.extra.values()] if x is not None]

    def _chart(self, kind: str):
        return {"chart": self.chart, "flow": self.flow}.get(kind) or self.extra.get(kind)

    @staticmethod
    def set_factor(win, z: float) -> None:
        try:
            wv = win.native.browser.webview
        except AttributeError:
            return
        from System import Action
        win.native.Invoke(Action(lambda: setattr(wv, "ZoomFactor", float(z))))

    def zoom(self, step: int, kind: str = "list") -> float:
        ch = self._chart(kind)
        if ch is not None:                            # вікно графіка — свій масштаб, «останній» пишеться в config
            return ch.set_zoom(step)
        old = float(self.cfg.get("zoom") or 1.0)
        new = 1.0 if step == 0 else min(ZOOM_MAX, max(ZOOM_MIN, round(old + step * ZOOM_STEP, 2)))
        if new == old:
            return new
        self.cfg["zoom"] = new
        self.set_factor(self.window, new)
        k = new / old
        w, h = int(self.cfg["width"] * k), int(self.cfg["height"] * k)
        self.cfg["width"], self.cfg["height"] = w, h
        self.window.resize(w, h)
        save_config(self.cfg)
        return new

    def on_list_loaded(self):
        z = float(self.cfg.get("zoom") or 1.0)
        if abs(z - 1.0) > 1e-6:
            self.set_factor(self.window, z)

    # --- вікно графіка
    def chart_title(self, ticker: str, kind: str = "chart") -> None:
        t = re.sub(r"[^A-Z0-9]", "", str(ticker).upper())
        ch = self._chart(kind)
        if ch is not None and t:
            ch.win.set_title(f"{t} — {ch.TITLE[ch.what]}")

    def open_chart(self, ticker: str, new: bool = False, what: str = "chart") -> None:
        """Вікно графіка (what="chart") або PD Flow (what="flow") для тікера."""
        t = re.sub(r"[^A-Z0-9]", "", str(ticker).upper())
        if not t:
            return
        if new:                                       # з Shift: ще одне вікно, основне не чіпаємо
            self._n_extra += 1
            kind = f"x{self._n_extra}"
            n = sum(1 for x in self.extra.values() if x.what == what) + 1
            self.extra[kind] = ChartWin(self, kind, t, main=False, n=n, what=what)
            return
        main = self.chart if what == "chart" else self.flow
        if main is None:
            main = ChartWin(self, what, t, main=True, what=what)
            if what == "chart":
                self.chart = main
            else:
                self.flow = main
            return
        # вікно вже є: сторінка лишається завантаженою, міняємо лише тікер — без перезавантаження
        w = main.win
        w.evaluate_js(f"location.hash = '#chart/{t}'" if what == "chart" else f"location.hash = '#{t}'")
        self.chart_title(t, what)
        w.show()
        w.restore()
        try:
            from System import Action
            w.native.Invoke(Action(w.native.Activate))      # на передній план
        except Exception:
            pass

    def quit(self, *_):
        self.quitting = True
        save_config(self.cfg)
        if self.tray is not None:
            self.tray.stop()
        for x in self._charts():
            x.win.destroy()
        self.window.destroy()

    def start_tray(self) -> None:
        import pystray
        menu = pystray.Menu(
            pystray.MenuItem("Показати / сховати", self.toggle, default=True),
            pystray.MenuItem("Поверх усіх вікон", self.set_on_top, checked=lambda item: self.cfg["on_top"]),
            pystray.MenuItem("Відкрити дашборд", lambda *_: Api(self).open_dashboard()),
            pystray.MenuItem("Налаштування (config.json)", lambda *_: os.startfile(CFG_FILE)),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Вийти", self.quit),
        )
        self.tray = pystray.Icon(APP, icon_image(64), APP, menu)
        self.tray.run_detached()

    # --- вікно
    def on_closing(self):
        if self.quitting:
            return True
        self.window.hide()                         # хрестик — у трей; вийти — пункт «Вийти» в треї
        self.visible = False
        return False

    def on_moved(self, x, y):
        self.cfg["x"], self.cfg["y"] = x, y

    def on_resized(self, width, height):
        self.cfg["width"], self.cfg["height"] = width, height

    def run(self) -> None:
        import webview
        # звук без першого кліку по сторінці (у браузері його дозволено лише після дії користувача)
        os.environ.setdefault("WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS", "--autoplay-policy=no-user-gesture-required")
        c = self.cfg
        self.window = webview.create_window(
            APP, page_url(c, "list"), js_api=Api(self, "list"), width=int(c["width"]), height=int(c["height"]),
            x=c["x"], y=c["y"], on_top=bool(c["on_top"]), min_size=(320, 360), background_color="#141413", text_select=True)
        self.window.events.closing += self.on_closing
        self.window.events.moved += self.on_moved
        self.window.events.resized += self.on_resized
        self.window.events.loaded += self.on_list_loaded
        self.start_tray()
        try:
            # private_mode=False + свій каталог: localStorage (звук, обрані, згорнуті секції) зберігається між запусками
            webview.start(gui="edgechromium", private_mode=False, storage_path=os.path.join(CFG_DIR, "webview"))
        finally:
            save_config(self.cfg)
            if self.tray is not None and not self.quitting:
                self.tray.stop()


def main() -> None:
    ap = argparse.ArgumentParser(description="PD Watchlist — вікно Windows лише зі списком тікерів дашборда Pump Detector")
    ap.add_argument("--url", help="адреса дашборда для цього запуску (інакше — з config.json)")
    ap.add_argument("--make-icon", action="store_true", help="лише створити watch_app/pd_watch.ico (для збирання .exe) і вийти")
    args = ap.parse_args()
    if args.make_icon:
        make_icon()
        return
    cfg = load_config()
    if args.url:
        cfg = {**cfg, "url": args.url}
    app = WatchApp(cfg)
    app.run()


if __name__ == "__main__":
    sys.exit(main())
