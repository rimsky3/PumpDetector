"""
PD Watchlist — маленьке вікно Windows лише зі списком тікерів Pump Detector (2026-10-15).

Обгортка pywebview (WebView2 — той самий рушій, що в Edge) над сторінкою дашборда `#list`: ті самі секції NOW / WOKE UP /
WATCHLIST / QUIET, чіпи, значки, звук. Чого не вміє вкладка браузера, а вміє застосунок:
  • вікно «поверх усіх» (перемикач у треї, запам'ятовується);
  • іконка в треї: клік — показати / сховати; хрестик вікна ховає його в трей, а не закриває;
  • системні сповіщення (Alert, Spread…) — через трей, бо в WebView2 сповіщень браузера немає;
  • звук без першого кліку по сторінці;
  • клік по тікеру відкриває його в окремому вікні графіка «PD Chart» (одне на всі кліки — новий тікер підміняє попередній;
    браузер і дашборд у ньому не чіпає: у застосунку своє сховище налаштувань); Shift + клік — тікер у НОВОМУ окремому
    вікні (кілька графіків одночасно; хрестик закриває його насовсім); пункт трею «Відкрити дашборд» — браузер;
  • Ctrl + / Ctrl − / Ctrl 0 — масштаб сторінки і розмір вікна разом, у кожному вікні окремо (запам'ятовується).

Налаштування — %APPDATA%\\PD Watchlist\\config.json (створюється при першому запуску):
  url       — адреса дашборда (за замовчуванням https://45-76-90-13.sslip.io/; через тунель — http://localhost:8766/);
  user, password — логін дашборда, якщо не хочеш вводити його щоразу (інакше WebView2 спитає сам);
  on_top, x, y, width, height, zoom — вікно списку; chart_x, chart_y, chart_width, chart_height, chart_zoom — вікно графіка
  (запам'ятовуються).

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
            "chart_x": None, "chart_y": None, "chart_width": 1100, "chart_height": 720, "chart_zoom": 1.0}
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


def page_url(cfg: dict, view: str = "list") -> str:
    """Адреса вікна: сторінка дашборда з #view («list» — лише список, «chart/TK» — лише графік); логін, якщо заданий, — в адресі."""
    parts = urlsplit(base_url(cfg))
    netloc = parts.netloc
    if cfg.get("user") and cfg.get("password"):
        netloc = f"{quote(cfg['user'], safe='')}:{quote(cfg['password'], safe='')}@{parts.hostname}" + (f":{parts.port}" if parts.port else "")
    return urlunsplit((parts.scheme, netloc, parts.path or "/", "", view))


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

    def chart_title(self, ticker: str) -> None:
        self._app.chart_title(ticker, self._kind)

    def notify(self, title: str, body: str = "") -> None:
        self._app.notify(title, body)

    def zoom(self, step: int) -> float:
        """Ctrl + / Ctrl − / Ctrl 0 зі сторінки: +1 більше, −1 менше, 0 — 100% (того вікна, звідки натиснуто)."""
        return self._app.zoom(step, self._kind)


class ExtraChart:
    """Додаткове вікно графіка (Shift + клік): хрестик закриває його насовсім; розмір і масштаб — лише на час життя вікна
    (у config.json не пишуться, стартові — як у основного вікна графіка)."""

    def __init__(self, app: "WatchApp", kind: str, ticker: str, n: int):
        import webview
        c = app.cfg
        self.app, self.kind = app, kind
        self.zoom = float(c.get("chart_zoom") or 1.0)
        self.width, self.height = int(c["chart_width"]), int(c["chart_height"])
        shift = 32 * n                                    # драбинкою від основного вікна графіка, щоб не лягали одне на одне
        x = c["chart_x"] + shift if c["chart_x"] is not None else None
        y = c["chart_y"] + shift if c["chart_y"] is not None else None
        self.win = webview.create_window(
            f"{ticker} — PD Chart", page_url(c, f"chart/{ticker}"), js_api=Api(app, kind),
            width=self.width, height=self.height, x=x, y=y, min_size=(480, 360), background_color="#141413", text_select=True)
        self.win.events.resized += self.on_resized
        self.win.events.loaded += self.on_loaded
        self.win.events.closed += self.on_closed

    def on_resized(self, width, height):
        self.width, self.height = width, height

    def on_loaded(self):
        if abs(self.zoom - 1.0) > 1e-6:
            self.app.set_factor(self.win, self.zoom)

    def on_closed(self):
        self.app.extra.pop(self.kind, None)


class WatchApp:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.window = None                        # список
        self.chart = None                         # вікно графіка (створюється при першому кліку по тікеру)
        self.extra: dict[str, ExtraChart] = {}    # додаткові вікна графіка (Shift + клік)
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
        self.cfg["on_top"] = not self.cfg["on_top"]
        self.window.on_top = self.cfg["on_top"]
        save_config(self.cfg)

    # --- масштаб: WebView2.ZoomFactor (шрифт і все на сторінці) + розмір вікна в тій самій пропорції
    def _win(self, kind: str):
        if kind in self.extra:
            return self.extra[kind].win
        return self.chart if kind == "chart" else self.window

    @staticmethod
    def _key(kind: str, name: str) -> str:
        return f"chart_{name}" if kind == "chart" else name

    def _set_factor(self, kind: str, z: float) -> None:
        self.set_factor(self._win(kind), z)

    @staticmethod
    def set_factor(win, z: float) -> None:
        try:
            wv = win.native.browser.webview
        except AttributeError:
            return
        from System import Action
        win.native.Invoke(Action(lambda: setattr(wv, "ZoomFactor", float(z))))

    def zoom(self, step: int, kind: str = "list") -> float:
        x = self.extra.get(kind)
        if x is not None:                             # додаткове вікно графіка: масштаб лише в пам'яті
            new = 1.0 if step == 0 else min(ZOOM_MAX, max(ZOOM_MIN, round(x.zoom + step * ZOOM_STEP, 2)))
            if new != x.zoom:
                k = new / x.zoom
                x.zoom = new
                self.set_factor(x.win, new)
                x.win.resize(int(x.width * k), int(x.height * k))
            return new
        k_zoom, k_w, k_h = (self._key(kind, n) for n in ("zoom", "width", "height"))
        old = float(self.cfg.get(k_zoom) or 1.0)
        new = 1.0 if step == 0 else min(ZOOM_MAX, max(ZOOM_MIN, round(old + step * ZOOM_STEP, 2)))
        if new == old:
            return new
        self.cfg[k_zoom] = new
        self._set_factor(kind, new)
        k = new / old
        w, h = int(self.cfg[k_w] * k), int(self.cfg[k_h] * k)
        self.cfg[k_w], self.cfg[k_h] = w, h
        self._win(kind).resize(w, h)
        save_config(self.cfg)
        return new

    def on_loaded(self, kind: str = "list"):
        z = float(self.cfg.get(self._key(kind, "zoom")) or 1.0)
        if abs(z - 1.0) > 1e-6:
            self._set_factor(kind, z)

    # --- вікно графіка
    def chart_title(self, ticker: str, kind: str = "chart") -> None:
        t = re.sub(r"[^A-Z0-9]", "", str(ticker).upper())
        win = self._win(kind) if kind != "list" else None
        if win is not None and t:
            win.set_title(f"{t} — PD Chart")

    def open_chart(self, ticker: str, new: bool = False) -> None:
        import webview
        t = re.sub(r"[^A-Z0-9]", "", str(ticker).upper())
        if not t:
            return
        if new:                                       # Shift + клік: ще одне вікно, основне не чіпаємо
            self._n_extra += 1
            kind = f"x{self._n_extra}"
            self.extra[kind] = ExtraChart(self, kind, t, len(self.extra) + 1)
            return
        if self.chart is None:
            c = self.cfg
            self.chart = webview.create_window(
                f"{t} — PD Chart", page_url(c, f"chart/{t}"), js_api=Api(self, "chart"),
                width=int(c["chart_width"]), height=int(c["chart_height"]), x=c["chart_x"], y=c["chart_y"],
                min_size=(480, 360), background_color="#141413", text_select=True)
            self.chart.events.closing += self.on_chart_closing
            self.chart.events.moved += self.on_chart_moved
            self.chart.events.resized += self.on_chart_resized
            self.chart.events.loaded += self.on_chart_loaded
            return
        # вікно вже є: сторінка лишається завантаженою, міняємо лише тікер — без перезавантаження
        self.chart.evaluate_js(f"location.hash = '#chart/{t}'")
        self.chart_title(t)
        self.chart.show()
        self.chart.restore()
        try:
            from System import Action
            self.chart.native.Invoke(Action(self.chart.native.Activate))      # на передній план
        except Exception:
            pass

    def on_chart_closing(self):
        if self.quitting:
            return True
        self.chart.hide()                          # хрестик ховає; наступний клік по тікеру покаже те саме вікно
        return False

    def on_chart_moved(self, x, y):
        self.cfg["chart_x"], self.cfg["chart_y"] = x, y

    def on_chart_resized(self, width, height):
        self.cfg["chart_width"], self.cfg["chart_height"] = width, height

    def on_chart_loaded(self):
        self.on_loaded("chart")

    def on_list_loaded(self):
        self.on_loaded("list")

    def quit(self, *_):
        self.quitting = True
        save_config(self.cfg)
        if self.tray is not None:
            self.tray.stop()
        for x in list(self.extra.values()):
            x.win.destroy()
        if self.chart is not None:
            self.chart.destroy()
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
