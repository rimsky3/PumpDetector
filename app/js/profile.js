/*
 * Профіль обсягу (volume profile) на графіку Lightweight Charts, як в ATAS / TradingView.
 * Спільний для дашборду (app/dashboard.html) і розмітки (tools/label.html). Дані — /api/profile (volume_profile.py).
 *
 *   Статичний: профіль на кожен день або тиждень UTC (setPeriod("day" | "week" | "off")).
 *   Фрагмент (Fixed Range): кнопка або клавіша V → клік — початок, рух миші, клік — кінець.
 *     Фрагментів може бути кілька; межі тягнуться мишею; × або Del (курсор над фрагментом) — прибрати; Esc — скасувати.
 *     Фрагменти зберігаються для кожного тікера (localStorage) і не залежать від ТФ графіка.
 *   На кожному профілі: гістограма обсягу за рівнями (value area яскравіше), POC, VAH/VAL;
 *   POC фрагмента продовжується праворуч пунктиром — видно тест рівня.
 *
 * const vp = PDProfile.create({
 *   chart, el, series,            // графік, його контейнер (position: relative), серія свічок
 *   times: () => number[],        // час барів (секунди)
 *   bar: () => number,            // тривалість бару, с
 *   symbol: () => string,         // тікер
 *   onModeChange: (armed) => {},  // для кнопки фрагмента
 *   onInfo: (text) => {},         // необов'язково: статус («профіль з 15m» тощо)
 * });
 * vp.setPeriod("day"); vp.reload();   // після зміни тікера / даних графіка
 */
(function () {
  "use strict";

  const css = `
.pdp-canvas { position: absolute; left: 0; top: 0; pointer-events: none; z-index: 3; }
.pdp-ui { position: absolute; inset: 0; pointer-events: none; z-index: 5; overflow: hidden; }
.pdp-x { position: absolute; pointer-events: auto; width: 18px; height: 18px; border-radius: 4px; border: 0; padding: 0;
  font: 13px/18px -apple-system, "Segoe UI", system-ui, sans-serif; text-align: center; cursor: pointer;
  background: rgba(41,98,255,.85); color: #fff; }
.pdp-x:hover { background: #2962ff; }
.pdp-armed { cursor: crosshair !important; }
.pdp-edge { cursor: ew-resize !important; }
`;
  let styled = false;
  const addStyle = () => {
    if (styled) return; styled = true;
    const s = document.createElement("style"); s.textContent = css; document.head.appendChild(s);
  };
  const LS = {
    get(k) { try { return JSON.parse(localStorage.getItem(k) || "null"); } catch (e) { return null; } },
    set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) {} },
  };
  const cssVar = (name, fb) => (getComputedStyle(document.documentElement).getPropertyValue(name) || "").trim() || fb;
  const rgba = (c, a) => {
    if (c.startsWith("#") && c.length === 7) { const n = parseInt(c.slice(1), 16); return `rgba(${n >> 16 & 255},${n >> 8 & 255},${n & 255},${a})`; }
    return c;
  };
  const getJson = async (url) => {
    const r = await fetch(url, { cache: "no-store" });
    const j = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(j.error || `HTTP ${r.status}`);
    return j;
  };

  function create(opts) {
    addStyle();
    const { chart, el, series } = opts;
    if (el._pdp) el._pdp.destroy();            // графік перестворено — прибираємо попередній екземпляр
    const ac = new AbortController(), sig = { signal: ac.signal }, cap = { signal: ac.signal, capture: true };
    const cv = document.createElement("canvas"); cv.className = "pdp-canvas";
    const ui = document.createElement("div"); ui.className = "pdp-ui";
    el.append(cv, ui);

    let period = "off";            // статичні профілі: off | day | week
    let statics = [], staticKey = null;
    let frags = [];                // [{ from, to, prof }]
    let fragSym = null;
    let mode = "off";              // off | armed | measuring
    let ta = null, tb = null;      // час барів початку / кінця під час виділення
    let hover = -1, drag = null, dragEnd = 0;   // drag: { i, side: "from" | "to" }
    const info = (t) => { if (opts.onInfo) opts.onInfo(t); };

    // ---------- час ↔ координата ----------
    const times = () => opts.times() || [];
    // логічний індекс моменту t (може бути дробовим і поза даними); межа бару i — (i − 0.5)
    const logicalOf = (t) => {
      const T = times(), b = opts.bar() || 900;
      if (!T.length) return null;
      if (t <= T[0]) return (t - T[0]) / b;
      let lo = 0, hi = T.length - 1;
      while (lo < hi) { const m = (lo + hi + 1) >> 1; if (T[m] <= t) lo = m; else hi = m - 1; }
      return lo + Math.min(1, (t - T[lo]) / b);
    };
    // logicalToCoordinate коректний лише для цілих індексів — рахуємо від цілого опорного бару і відстані між барами
    const xAt = (t) => {
      const l = logicalOf(t); if (l == null) return null;
      const ts = chart.timeScale(), r = ts.getVisibleLogicalRange(); if (!r) return null;
      const a = Math.round(r.from), xa = ts.logicalToCoordinate(a), xb = ts.logicalToCoordinate(a + 1);
      if (xa == null || xb == null) return null;
      return xa + (l - 0.5 - a) * (xb - xa);
    };
    const barAt = (x) => {         // час бару під курсором
      const T = times(); if (!T.length) return null;
      const l = chart.timeScale().coordinateToLogical(x); if (l == null) return null;
      return T[Math.max(0, Math.min(T.length - 1, Math.round(l)))];
    };
    const paneH = () => chart.panes()[0].getHeight();
    const plotW = () => chart.timeScale().width();

    // ---------- малювання ----------
    const drawProfile = (ctx, p, x0, x1, frac, isFrag, hi) => {
      const W = plotW(), H = paneH();
      const avail = x1 - x0; if (avail < 2) return;
      const maxW = avail * frac, maxV = Math.max(...p.vol) || 1;
      const C = colors();
      const ya = (price) => series.priceToCoordinate(price);
      for (let i = 0; i < p.vol.length; i++) {
        if (!p.vol[i]) continue;
        const lo = (p.k0 + i) * p.step, up = lo + p.step;
        const y1 = ya(up), y2 = ya(lo); if (y1 == null || y2 == null) continue;
        if (y2 < 0 || y1 > H) continue;
        const h = Math.max(1, y2 - y1 - (y2 - y1 > 3 ? 1 : 0));
        const w = Math.max(1, p.vol[i] / maxV * maxW);
        const mid = lo + p.step / 2, inVa = mid >= p.val && mid <= p.vah;
        const isPoc = p.poc >= lo && p.poc < up;
        ctx.fillStyle = isPoc ? C.poc : isFrag ? (inVa ? C.fva : C.fout) : inVa ? C.va : C.out;
        ctx.fillRect(x0, y1, w, h);
      }
      const hline = (price, color, dash, xa, xb, lw) => {
        const y = ya(price); if (y == null || y < 0 || y > H) return;
        ctx.save(); ctx.strokeStyle = color; ctx.lineWidth = lw || 1; ctx.setLineDash(dash);
        ctx.beginPath(); ctx.moveTo(xa, Math.round(y) + 0.5); ctx.lineTo(xb, Math.round(y) + 0.5); ctx.stroke(); ctx.restore();
      };
      hline(p.vah, C.vaLine, [3, 3], x0, x1); hline(p.val, C.vaLine, [3, 3], x0, x1);
      hline(p.poc, C.pocLine, [], x0, x1, isFrag ? 1.5 : 1);
      if (isFrag) {
        hline(p.poc, C.pocLine, [5, 4], x1, W);                         // POC праворуч — для тесту рівня
        ctx.fillStyle = hi ? C.bandHi : C.band; ctx.fillRect(x0, 0, avail, H);
        ctx.fillStyle = C.edge; ctx.fillRect(Math.round(x0), 0, 1, H); ctx.fillRect(Math.round(x1) - 1, 0, 1, H);
      }
    };
    let C0 = null;
    const colors = () => {
      if (C0) return C0;
      const fg = cssVar("--fg2", "#9aa3b0"), poc = cssVar("--warm", cssVar("--sel-edge", "#e0703f"));
      C0 = { va: rgba(fg, .42), out: rgba(fg, .2), poc: rgba(poc, .85), pocLine: poc, vaLine: rgba(fg, .7),
             fva: "rgba(92,140,255,.6)", fout: "rgba(92,140,255,.3)",
             band: "rgba(41,98,255,.05)", bandHi: "rgba(41,98,255,.11)", edge: "rgba(41,98,255,.55)" };
      return C0;
    };
    const draw = () => {
      const dpr = window.devicePixelRatio || 1, W = el.clientWidth, H = el.clientHeight;
      if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) {
        cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); cv.style.width = W + "px"; cv.style.height = H + "px";
      }
      const ctx = cv.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
      if (dead) return;
      const shown = [];
      if (!times().length) { syncX(shown); return; }
      ctx.save(); ctx.beginPath(); ctx.rect(0, 0, plotW(), paneH()); ctx.clip();
      for (const p of statics) {
        const x0 = xAt(p.from), x1 = xAt(p.to);
        if (x0 == null || x1 == null || x1 < 0 || x0 > plotW()) continue;
        // початок періоду за лівим краєм — гістограму притискаємо до краю, щоб її було видно
        drawProfile(ctx, p, Math.max(x0, 0), x1, 0.45, false, false);
      }
      frags.forEach((f, i) => {
        const x0 = xAt(f.from), x1 = xAt(f.to);
        if (x0 == null || x1 == null || x1 < 0 || x0 > plotW()) return;
        if (f.prof) drawProfile(ctx, f.prof, x0, x1, 0.85, true, i === hover || (drag && drag.i === i));
        else { ctx.fillStyle = colors().band; ctx.fillRect(x0, 0, x1 - x0, paneH()); }
        shown.push({ i, left: Math.max(2, Math.min(plotW() - 20, x1 - 25)) });
      });
      syncX(shown);
      if (mode === "measuring" && ta != null && tb != null) {        // виділення фрагмента
        const b = opts.bar() || 900, a = Math.min(ta, tb), z = Math.max(ta, tb) + b;
        const x0 = xAt(a), x1 = xAt(z);
        if (x0 != null && x1 != null) {
          ctx.fillStyle = "rgba(41,98,255,.14)"; ctx.fillRect(x0, 0, x1 - x0, paneH());
          ctx.fillStyle = colors().edge; ctx.fillRect(Math.round(x0), 0, 1, paneH()); ctx.fillRect(Math.round(x1) - 1, 0, 1, paneH());
        }
      }
      ctx.restore();
    };
    // кнопки × — постійні елементи (не перестворюються на кожен рух миші, інакше клік губиться)
    const xs = [];
    const syncX = (shown) => {
      while (xs.length < shown.length) {
        const b = document.createElement("button"); b.className = "pdp-x"; b.textContent = "×";
        b.title = "Прибрати профіль (або Del, коли курсор над ним)";
        b.addEventListener("click", (e) => { e.stopPropagation(); e.preventDefault(); removeFrag(+b.dataset.i); });
        ui.appendChild(b); xs.push(b);
      }
      xs.forEach((b, k) => {
        const s = shown[k]; b.hidden = !s;
        if (s) { b.dataset.i = s.i; b.style.left = s.left + "px"; b.style.top = "4px"; }
      });
    };
    let raf = 0;
    let dead = false;
    const redraw = () => { if (!dead && !raf) raf = requestAnimationFrame(() => { raf = 0; draw(); }); };

    // ---------- дані ----------
    const sym = () => (opts.symbol() || "").toUpperCase();
    const url = (from, to, m) => `/api/profile?symbol=${encodeURIComponent(sym())}&from=${from}&to=${to}&mode=${m}`;
    const loadStatics = async () => {
      const T = times();
      if (period === "off" || !T.length || !sym()) { statics = []; staticKey = null; redraw(); return; }
      const from = T[0], to = T[T.length - 1] + (opts.bar() || 900);
      const key = `${sym()}|${period}|${from}|${to}`;
      if (key === staticKey) return;
      staticKey = key;
      try {
        const r = await getJson(url(from, to, period));
        if (staticKey !== key) return;
        statics = r.profiles || [];
        info(r.src ? `Профіль: з ${r.src}-свічок` : "Профіль: немає даних");
      } catch (e) { statics = []; staticKey = null; info("Профіль: " + e.message); }
      redraw();
    };
    const storeKey = () => `pd-vp-frag-${sym()}`;
    const save = () => LS.set(storeKey(), frags.map((f) => ({ from: f.from, to: f.to })));
    const fetchFrag = async (f) => {
      f.prof = null; const tag = f.tag = Math.random();
      try {
        const r = await getJson(url(f.from, f.to, "range"));
        if (f.tag === tag) { f.prof = (r.profiles || [])[0] || null; if (r.src) info(`Профіль: з ${r.src}-свічок`); }
      } catch (e) { info("Профіль: " + e.message); }
      redraw();
    };
    const loadFrags = () => {
      if (fragSym === sym()) return;
      fragSym = sym();
      frags = (LS.get(storeKey()) || []).filter((f) => f && f.to > f.from).map((f) => ({ from: f.from, to: f.to, prof: null }));
      frags.forEach(fetchFrag);
    };
    const removeFrag = (i) => { frags.splice(i, 1); hover = -1; save(); redraw(); };

    // ---------- виділення фрагмента ----------
    const setScroll = (on) => chart.applyOptions({ handleScroll: on, handleScale: on });
    const setMode = (m) => {
      mode = m;
      el.classList.toggle("pdp-armed", m !== "off");
      setScroll(m === "off");
      if (opts.onModeChange) opts.onModeChange(m !== "off");
    };
    const local = (ev) => { const r = el.getBoundingClientRect(); return { x: ev.clientX - r.left, y: ev.clientY - r.top }; };
    const inPlot = (p) => p.x >= 0 && p.x <= plotW() && p.y >= 0 && p.y <= paneH();
    const onX = (ev) => ev.target.closest && ev.target.closest(".pdp-x");
    el.addEventListener("click", (ev) => {
      if (mode === "off" || onX(ev)) return;
      const p = local(ev); if (!inPlot(p)) return;
      const t = barAt(p.x); if (t == null) return;
      ev.stopPropagation(); ev.preventDefault();
      if (mode === "armed") { ta = tb = t; setMode("measuring"); redraw(); return; }
      tb = t;
      const b = opts.bar() || 900, f = { from: Math.min(ta, tb), to: Math.max(ta, tb) + b, prof: null };
      frags.push(f); save(); fetchFrag(f);
      ta = tb = null; setMode("off"); redraw();
    }, cap);
    // межа фрагмента під курсором (±5 px)
    const edgeAt = (x) => {
      for (let i = frags.length - 1; i >= 0; i--) {
        const x0 = xAt(frags[i].from), x1 = xAt(frags[i].to);
        if (x0 != null && Math.abs(x - x0) <= 5) return { i, side: "from" };
        if (x1 != null && Math.abs(x - x1) <= 5) return { i, side: "to" };
      }
      return null;
    };
    const fragAt = (x) => {
      for (let i = frags.length - 1; i >= 0; i--) {
        const x0 = xAt(frags[i].from), x1 = xAt(frags[i].to);
        if (x0 != null && x1 != null && x >= x0 && x <= x1) return i;
      }
      return -1;
    };
    el.addEventListener("pointerdown", (ev) => {
      if (mode !== "off" || ev.button !== 0 || onX(ev)) return;
      const p = local(ev); if (!inPlot(p)) return;
      const e = edgeAt(p.x); if (!e) return;
      drag = e; setScroll(false); ev.stopPropagation(); ev.preventDefault();
    }, cap);
    el.addEventListener("mousemove", (ev) => {
      const p = local(ev);
      if (mode === "measuring") { const t = barAt(p.x); if (t != null && t !== tb) { tb = t; redraw(); } return; }
      if (drag) return;
      const h = inPlot(p) ? fragAt(p.x) : -1;
      el.classList.toggle("pdp-edge", inPlot(p) && !!edgeAt(p.x));
      if (h !== hover) { hover = h; redraw(); }
    }, sig);
    el.addEventListener("mouseleave", () => { if (hover !== -1 && !drag) { hover = -1; redraw(); } }, sig);
    window.addEventListener("pointermove", (ev) => {
      if (!drag) return;
      const t = barAt(local(ev).x); if (t == null) return;
      const f = frags[drag.i], b = opts.bar() || 900;
      if (drag.side === "from") f.from = Math.min(t, f.to - b); else f.to = Math.max(t + b, f.from + b);
      redraw();
    }, sig);
    window.addEventListener("pointerup", () => {
      if (!drag) return;
      const f = frags[drag.i]; drag = null; dragEnd = performance.now(); setScroll(true); save(); fetchFrag(f);
    }, sig);
    window.addEventListener("keydown", (ev) => {
      if (ev.target.closest && ev.target.closest("input, select, textarea")) return;
      if (opts.keys && !opts.keys()) return;     // кілька графіків на сторінці: клавіші — лише активному (opts.keys)
      if (ev.key === "Escape" && mode !== "off") { ta = tb = null; setMode("off"); redraw(); return; }
      if ((ev.key === "Delete" || ev.key === "Backspace") && hover >= 0 && mode === "off") { ev.preventDefault(); removeFrag(hover); return; }
      if (ev.ctrlKey || ev.metaKey || ev.altKey) return;
      const k = ev.key.toLowerCase();
      if (k === "v" || k === "м") { ev.preventDefault(); api.toggle(); }
    }, sig);
    chart.timeScale().subscribeVisibleLogicalRangeChange(redraw);
    chart.subscribeCrosshairMove(redraw);              // масштаб ціни змінюється без окремої події
    const ro = new ResizeObserver(redraw); ro.observe(el);

    const api = {
      /** статичні профілі: "off" | "day" | "week" */
      setPeriod(p) { period = p; staticKey = null; loadStatics(); },
      period: () => period,
      /** почати / скасувати виділення фрагмента */
      toggle() { if (mode === "off") setMode("armed"); else { ta = tb = null; setMode("off"); redraw(); } },
      /** true, якщо клік по графіку зараз належить профілю */
      busy: () => mode !== "off" || !!drag || performance.now() - dragEnd < 300,
      /** після зміни тікера або даних графіка */
      reload() { loadFrags(); loadStatics(); redraw(); },
      clearFragments() { frags = []; save(); redraw(); },
      redraw,
      /** діагностика: розміри області графіка, кількість профілів, координати першого */
      state: () => ({ plotW: plotW(), paneH: paneH(), statics: statics.length, frags: frags.length,
        loaded: frags.filter((f) => f.prof).length }),
      destroy() { dead = true; ac.abort(); ro.disconnect(); cv.remove(); ui.remove(); el.classList.remove("pdp-armed", "pdp-edge"); if (el._pdp === api) el._pdp = null; },
    };
    el._pdp = api;
    return api;
  }

  window.PDProfile = { create };
})();
