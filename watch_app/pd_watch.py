"""
PD Watchlist — маленьке вікно Windows лише зі списком тікерів Pump Detector (2026-10-15).

Обгортка pywebview (WebView2 — той самий рушій, що в Edge) над сторінкою дашборда `#list`: ті самі секції NOW / WOKE UP /
WATCHLIST / QUIET, чіпи, значки, звук. Чого не вміє вкладка браузера, а вміє застосунок:
  • вікно «поверх усіх» (перемикач у треї, запам'ятовується);
  • іконка в треї: клік — показати / сховати; хрестик вікна ховає його в трей, а не закриває;
  • системні сповіщення (Alert, Spread…) — через трей, бо в WebView2 сповіщень браузера немає;
  • звук без першого кліку по сторінці;
  • клік по тікеру відкриває його в повному дашборді у звичайному браузері.

Налаштування — %APPDATA%\\PD Watchlist\\config.json (створюється при першому запуску):
  url       — адреса дашборда (за замовчуванням https://45-76-90-13.sslip.io/; через тунель — http://localhost:8766/);
  user, password — логін дашборда, якщо не хочеш вводити його щоразу (інакше WebView2 спитає сам);
  on_top, x, y, width, height — вікно (запам'ятовуються).

    python watch_app/pd_watch.py                       # запуск без збирання
    python watch_app/pd_watch.py --url http://127.0.0.1:8799/
    watch_app\\build.cmd                                # → dist\\PD Watchlist.exe
"""

import argparse
import json
import os
import sys
import threading
import webbrowser
from urllib.parse import quote, urlsplit, urlunsplit

APP = "PD Watchlist"
CFG_DIR = os.path.join(os.environ.get("APPDATA", os.path.expanduser("~")), APP)
CFG_FILE = os.path.join(CFG_DIR, "config.json")
DEFAULTS = {"url": "https://45-76-90-13.sslip.io/", "user": "", "password": "", "on_top": True,
            "x": None, "y": None, "width": 420, "height": 900}
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


def list_url(cfg: dict) -> str:
    """Адреса вікна: сторінка дашборда в режимі «лише список»; логін, якщо заданий, — у самій адресі."""
    parts = urlsplit(base_url(cfg))
    netloc = parts.netloc
    if cfg.get("user") and cfg.get("password"):
        netloc = f"{quote(cfg['user'], safe='')}:{quote(cfg['password'], safe='')}@{parts.hostname}" + (f":{parts.port}" if parts.port else "")
    return urlunsplit((parts.scheme, netloc, parts.path or "/", "", "list"))


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

    def __init__(self, app: "WatchApp"):
        self._app = app

    def open_dashboard(self, ticker: str = "") -> None:
        webbrowser.open(base_url(self._app.cfg) + (f"#forecast/{ticker}" if ticker else "#forecast"))

    def notify(self, title: str, body: str = "") -> None:
        self._app.notify(title, body)


class WatchApp:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.window = None
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

    def quit(self, *_):
        self.quitting = True
        save_config(self.cfg)
        if self.tray is not None:
            self.tray.stop()
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
            APP, list_url(c), js_api=Api(self), width=int(c["width"]), height=int(c["height"]),
            x=c["x"], y=c["y"], on_top=bool(c["on_top"]), min_size=(320, 360), background_color="#141413", text_select=True)
        self.window.events.closing += self.on_closing
        self.window.events.moved += self.on_moved
        self.window.events.resized += self.on_resized
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
