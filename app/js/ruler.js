/*
 * Лінійка як у TradingView для Lightweight Charts: вимір між двома точками графіка.
 * Спільна для дашборду (app/dashboard.html) і розмітки (tools/label.html).
 *
 *   Shift + клік — почати вимір, рух миші — друга точка, клік — зафіксувати,
 *   ще один клік або Esc — прибрати. Кнопка лінійки (toggle) робить те саме без Shift.
 *
 * const ruler = PDRuler.create({
 *   chart, el,                 // графік LWC і його контейнер (position: relative)
 *   series,                    // серія ціни (свічки) — для перерахунку y ↔ ціна на верхній панелі
 *   times: () => number[],     // час барів (секунди) — для прив'язки до свічок
 *   info: (i1, i2, p1, p2) => [{ k: "Ціна", v: "+12%", cls: "up" }, …],   // рядки підказки
 *   onModeChange: (armed) => {},                                          // для кнопки
 * });
 */
(function () {
  "use strict";

  const css = `
.pdr-layer { position: absolute; inset: 0; pointer-events: none; z-index: 4; overflow: hidden; }
.pdr-box { position: absolute; border: 1px solid; }
.pdr-box.up { background: rgba(41,98,255,.16); border-color: rgba(41,98,255,.7); }
.pdr-box.down { background: rgba(226,75,74,.16); border-color: rgba(226,75,74,.7); }
.pdr-v, .pdr-h { position: absolute; }
.pdr-v { width: 0; border-left: 1px dashed; } .pdr-h { height: 0; border-top: 1px dashed; }
.pdr-info { position: absolute; transform: translateX(-50%); padding: 7px 10px; border-radius: 6px; color: #fff;
  font: 12px/1.45 -apple-system, "Segoe UI", system-ui, sans-serif; font-variant-numeric: tabular-nums; white-space: nowrap;
  box-shadow: 0 6px 18px rgba(0,0,0,.4); text-align: center; }
.pdr-info.up { background: #2962ff; } .pdr-info.down { background: #c93a39; }
.pdr-info .main { font-weight: 600; font-size: 13px; }
.pdr-info .row { opacity: .92; } .pdr-info .row b { font-weight: 600; }
.pdr-info .sub { margin-top: 3px; padding-top: 3px; border-top: 1px solid rgba(255,255,255,.25); }
.pdr-armed { cursor: crosshair !important; }
`;
  let styled = false;
  const addStyle = () => {
    if (styled) return; styled = true;
    const s = document.createElement("style"); s.textContent = css; document.head.appendChild(s);
  };

  function create(opts) {
    addStyle();
    const { chart, el, series } = opts;
    const layer = document.createElement("div"); layer.className = "pdr-layer";
    el.appendChild(layer);
    let mode = "off";          // off | armed | measuring | done
    let a = null, b = null;    // { i, price }

    const setScroll = (on) => chart.applyOptions({ handleScroll: on, handleScale: on });
    const setMode = (m) => {
      mode = m;
      el.classList.toggle("pdr-armed", m === "armed" || m === "measuring");
      setScroll(!(m === "armed" || m === "measuring"));
      if (opts.onModeChange) opts.onModeChange(m === "armed" || m === "measuring");
    };

    // точка під курсором: найближча свічка і ціна на верхній панелі
    const pointAt = (ev) => {
      const r = el.getBoundingClientRect();
      const x = ev.clientX - r.left, y = ev.clientY - r.top;
      const times = opts.times();
      if (!times.length) return null;
      const logical = chart.timeScale().coordinateToLogical(x);
      if (logical == null) return null;
      const i = Math.max(0, Math.min(times.length - 1, Math.round(logical)));
      const paneH = chart.panes()[0].getHeight();
      const price = series.coordinateToPrice(Math.min(y, paneH - 1));
      if (price == null) return null;
      return { i, price };
    };

    const clear = () => { a = b = null; layer.replaceChildren(); };

    const draw = () => {
      layer.replaceChildren();
      if (!a || !b) return;
      const times = opts.times();
      const ts = chart.timeScale();
      const x1 = ts.timeToCoordinate(times[a.i]), x2 = ts.timeToCoordinate(times[b.i]);
      const y1 = series.priceToCoordinate(a.price), y2 = series.priceToCoordinate(b.price);
      if (x1 == null || x2 == null || y1 == null || y2 == null) return;
      const up = b.price >= a.price, cls = up ? "up" : "down";
      const L = Math.min(x1, x2), T = Math.min(y1, y2), W = Math.max(1, Math.abs(x2 - x1)), Hh = Math.max(1, Math.abs(y2 - y1));
      const col = up ? "rgba(41,98,255,.9)" : "rgba(226,75,74,.9)";
      const box = document.createElement("div"); box.className = "pdr-box " + cls;
      Object.assign(box.style, { left: L + "px", top: T + "px", width: W + "px", height: Hh + "px" });
      const v = document.createElement("div"); v.className = "pdr-v";
      Object.assign(v.style, { left: (L + W / 2) + "px", top: T + "px", height: Hh + "px", borderColor: col });
      const h = document.createElement("div"); h.className = "pdr-h";
      Object.assign(h.style, { left: L + "px", top: (T + Hh / 2) + "px", width: W + "px", borderColor: col });
      layer.append(box, v, h);

      // підпис: над рамкою, якщо ціна росла, під нею — якщо падала (як у TradingView)
      const i1 = Math.min(a.i, b.i), i2 = Math.max(a.i, b.i);
      const rows = opts.info(i1, i2, a.i <= b.i ? a.price : b.price, a.i <= b.i ? b.price : a.price, a.price, b.price) || [];
      const info = document.createElement("div"); info.className = "pdr-info " + cls;
      rows.forEach((r, k) => {
        const d = document.createElement("div");
        d.className = k === 0 ? "main" : (r.sub ? "row sub" : "row");
        if (r.k) { d.append(r.k + " "); const bb = document.createElement("b"); bb.textContent = r.v; d.appendChild(bb); }
        else d.textContent = r.v;
        info.appendChild(d);
      });
      layer.appendChild(info);
      const ih = info.offsetHeight, elH = el.clientHeight;
      let top = up ? T - ih - 8 : T + Hh + 8;
      if (top < 4) top = T + Hh + 8;
      if (top + ih > elH - 4) top = Math.max(4, T - ih - 8);
      info.style.top = top + "px";
      const iw = info.offsetWidth, cx = Math.max(iw / 2 + 4, Math.min(el.clientWidth - iw / 2 - 60, L + W / 2));
      info.style.left = cx + "px";
    };

    // клік: у режимі очікування — перша точка, під час виміру — зафіксувати, у готовому — прибрати
    el.addEventListener("click", (ev) => {
      if (mode === "off" && !ev.shiftKey) return;
      if (mode === "done") { clear(); setMode("off"); if (!ev.shiftKey) return; }
      const p = pointAt(ev); if (!p) return;
      ev.stopPropagation(); ev.preventDefault();
      if (mode === "off" || mode === "armed" || mode === "done") { a = p; b = p; setMode("measuring"); draw(); return; }
      if (mode === "measuring") { b = p; setMode("done"); draw(); }
    }, true);
    el.addEventListener("mousemove", (ev) => {
      if (mode !== "measuring") return;
      const p = pointAt(ev); if (!p) return;
      b = p; draw();
    });
    window.addEventListener("keydown", (ev) => { if (ev.key === "Escape" && mode !== "off") { clear(); setMode("off"); } });
    chart.timeScale().subscribeVisibleLogicalRangeChange(() => draw());
    chart.subscribeCrosshairMove(() => { if (mode === "done") draw(); });   // масштаб ціни змінюється без окремої події
    new ResizeObserver(() => draw()).observe(el);

    // Shift відстежуємо окремо: подія кліку графіка (subscribeClick) приходить раніше за DOM-click
    let shiftDown = false;
    window.addEventListener("keydown", (ev) => { if (ev.key === "Shift") shiftDown = true; });
    window.addEventListener("keyup", (ev) => { if (ev.key === "Shift") shiftDown = false; });
    window.addEventListener("blur", () => { shiftDown = false; });

    return {
      /** true, якщо клік по графіку зараз належить лінійці (не треба обробляти його як звичайний) */
      busy() { return mode !== "off" || shiftDown; },
      toggle() { if (mode === "off" || mode === "done") { clear(); setMode("armed"); } else { clear(); setMode("off"); } },
      clear() { clear(); setMode("off"); },
      /** показати зафіксований вимір між барами i1 і i2 (ціни p1, p2) — напр. для посилань або перевірки */
      show(i1, p1, i2, p2) { a = { i: i1, price: p1 }; b = { i: i2, price: p2 }; setMode("done"); draw(); },
      redraw: draw,
    };
  }

  window.PDRuler = { create };
})();
