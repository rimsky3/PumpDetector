// Конус очікуваного руху на графіку (кнопка Cone, 2026-10-20; власник обрав варіант A макета): віяло на 4 год уперед —
// яскраво ±1σ (ціна в ньому приблизно у 2 з 3 випадків), бліде ±2σ (вихід — рідко).
// Дані — прогноз volfc.py (HAR, «від зараз» кожні 5 хв): exp_1 і exp_4 з /api/ticker → vol. Між ними дисперсія росте
// лінійно з часом (рух — як корінь з часу): σ(t) = exp_1·√t до 1 год, далі √(exp_1² + (exp_4² − exp_1²)·(t − 1)/3).
// Межі — у логах: ціна · e^(±kσ). Це опис розмаху, НЕ рівні: ціна від краю конуса не відскакує.
// Режими кнопки (20.10, власник: «конус весь час рухається за ціною — як з ним працювати?»):
//   live — від останньої ціни (план угоди «якщо зайти зараз»);
//   pin  — закріплений у момент натискання: та ціна, той прогноз, на 4 год (контроль угоди після входу); зберігається
//          для тікера (localStorage pd-cone-pin-<TK>) і зникає через 4 год; тікер без закріплення — як live.
// Позиція власника (my_positions.py, 20.10): поки Cone увімкнено, на тікері з відкритою позицією конус сам закріплюється від
// входу (ціна входу, прогноз на ту мить; мітка «long entry 14:05 UTC», «~» — вхід до старту сервісу) і знімається після виходу.
(function () {
  "use strict";

  const HOURS = 4;
  const cssVar = (name, fb) => (getComputedStyle(document.documentElement).getPropertyValue(name) || "").trim() || fb;
  const rgba = (c, a) => {
    if (c.startsWith("#") && c.length === 7) { const n = parseInt(c.slice(1), 16); return `rgba(${n >> 16 & 255},${n >> 8 & 255},${n & 255},${a})`; }
    return c;
  };
  const pinKey = (tk) => `pd-cone-pin-${(tk || "").toUpperCase()}`;
  const loadPin = (tk) => {
    try {
      const p = JSON.parse(localStorage.getItem(pinKey(tk)) || "null");
      if (p && Date.now() / 1000 < p.at + HOURS * 3600) return p;
      localStorage.removeItem(pinKey(tk));
    } catch (e) { /* немає сховища */ }
    return null;
  };

  /** σ (частка ціни) через t год за прогнозами на 1 і 4 год */
  const sigmaAt = (t, e1, e4) => {
    if (t <= 0) return 0;
    if (e1 == null) return e4 * Math.sqrt(t / HOURS);
    if (t <= 1 || e4 == null) return e1 * Math.sqrt(t);
    return Math.sqrt(e1 * e1 + Math.max(0, e4 * e4 - e1 * e1) * (t - 1) / (HOURS - 1));
  };

  function create(opts) {
    const { chart, el, series } = opts;
    if (el._pdc) el._pdc.destroy();
    const cv = document.createElement("canvas");
    cv.style.cssText = "position:absolute;left:0;top:0;pointer-events:none;z-index:2";
    el.appendChild(cv);
    let mode = "off", dead = false, raf = 0, baseOffset = null;
    const bar = () => opts.bar() || 900;
    const sym = () => (opts.symbol ? opts.symbol() : "") || "";
    const paneH = () => chart.panes()[0].getHeight();
    const plotW = () => chart.timeScale().width();
    const ahead = () => Math.ceil(HOURS * 3600 / bar());

    // місце праворуч від останньої свічки під 4 год (5m — 48 барів, 15m — 16, 1h — 4); вимкнено — як було
    const applyOffset = () => {
      const ts = chart.timeScale();
      if (baseOffset == null) baseOffset = ts.options().rightOffset;
      ts.applyOptions({ rightOffset: mode !== "off" ? Math.max(baseOffset, ahead() + 3) : baseOffset });
    };
    const lastCandle = () => {
      const data = series.data();
      let li = data.length - 1;
      while (li >= 0 && !(data[li].close > 0)) li--;           // наприкінці можуть бути порожні (whitespace) бари
      return li >= 1 ? data[li] : null;
    };
    // x моменту at (с): бар, що його містить, + частка всередині бару; номер у серії ≠ логічний індекс шкали —
    // позиція за часом (timeToIndex), відстань між барами — з двох сусідніх логічних індексів
    const xOf = (at) => {
      const ts = chart.timeScale(), b = bar(), t0 = Math.floor(at / b) * b;
      const L = ts.timeToIndex(t0, true);
      if (L == null) return null;
      const xa = ts.logicalToCoordinate(L), xb = ts.logicalToCoordinate(L + 1);
      if (xa == null || xb == null) return null;
      return { x: xa + (xb - xa) * ((at - t0) / b), sp: xb - xa };
    };

    const draw = () => {
      const dpr = window.devicePixelRatio || 1, W = el.clientWidth, Hh = el.clientHeight;
      if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(Hh * dpr)) {
        cv.width = Math.round(W * dpr); cv.height = Math.round(Hh * dpr); cv.style.width = W + "px"; cv.style.height = Hh + "px";
      }
      const ctx = cv.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, Hh);
      if (mode === "off" || dead) return;
      // позиція власника (my_positions.py): конус від входу — пріоритетніший за ручний пін і живий (лише коли Cone увімкнено)
      const pos = opts.pos ? opts.pos() : null, pc = pos && pos.cone && pos.cone.p > 0 ? pos.cone : null;
      const pin = pc ? null : mode === "pin" ? loadPin(sym()) : null;
      let A;                                                     // якір: момент, ціна, прогнози 1 і 4 год
      if (pc) {
        const v = opts.vol() || {};
        A = { at: pc.at, p: pc.p, e1: pc.e1 ?? v.exp_1, e4: pc.e4 ?? v.exp_4, entry: pos.side, approx: pos.approx };
        if (A.e4 == null) return;
      } else if (pin) A = pin;
      else {
        const v = opts.vol(), lc = lastCandle();
        if (!v || v.exp_4 == null || !lc) {
          ctx.font = "12px -apple-system, Segoe UI, sans-serif"; ctx.fillStyle = cssVar("--fg3", "#6b6a64");
          ctx.fillText("Cone: прогнозу для цієї монети ще немає", 8, paneH() - 10); return;
        }
        A = { at: lc.time + bar() * 0.999, p: lc.close, e1: v.exp_1, e4: v.exp_4 };   // кінець останньої свічки
      }
      const xy = xOf(A.at); if (!xy) return;
      const perH = 3600 / bar(), xAt = (t) => xy.x + xy.sp * perH * t, yAt = (p) => series.priceToCoordinate(p);
      const amber = cssVar("--amber", "#ef9f27"), fg2 = cssVar("--fg2", "#a8a79f"), link = cssVar("--link", "#85b7eb");
      const fixed = !!(pin || pc), col = fixed ? link : amber;   // закріплений — блакитний, щоб не сплутати з живим
      ctx.save(); ctx.beginPath(); ctx.rect(0, 0, plotW(), paneH()); ctx.clip();
      const STEPS = 32;
      const band = (k, fill, stroke) => {
        const up = [], dn = [];
        for (let i = 0; i <= STEPS; i++) {
          const t = HOURS * i / STEPS, s = sigmaAt(t, A.e1, A.e4), x = xAt(t);
          const yu = yAt(A.p * Math.exp(k * s)), yd = yAt(A.p * Math.exp(-k * s));
          if (yu == null || yd == null) return;
          up.push([x, yu]); dn.push([x, yd]);
        }
        ctx.beginPath(); up.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
        for (let i = dn.length - 1; i >= 0; i--) ctx.lineTo(dn[i][0], dn[i][1]);
        ctx.closePath(); ctx.fillStyle = fill; ctx.fill();
        if (stroke) {
          ctx.strokeStyle = stroke; ctx.lineWidth = 1;
          [up, dn].forEach((pts) => { ctx.beginPath(); pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y))); ctx.stroke(); });
        }
      };
      band(2, rgba(col, 0.07), null);
      band(1, rgba(col, 0.2), rgba(col, 0.55));
      ctx.font = "10.5px -apple-system, Segoe UI, sans-serif"; ctx.textBaseline = "top";
      [[1, A.e1], [HOURS, A.e4]].forEach(([t, e]) => {
        const x = xAt(t); if (e == null || x > plotW()) return;
        ctx.strokeStyle = rgba(fg2, 0.35); ctx.setLineDash([2, 3]); ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, paneH()); ctx.stroke(); ctx.setLineDash([]);
        const yl = yAt(A.p * Math.exp(-sigmaAt(t, A.e1, A.e4))); if (yl == null) return;
        const txt = `+${t}H ±${(e * 100).toFixed(e >= 0.1 ? 0 : 1)}%`;
        ctx.fillStyle = col; ctx.textAlign = t === HOURS ? "right" : "center"; ctx.fillText(txt, t === HOURS ? x - 2 : x, yl + 4);
      });
      if (fixed) {                                                // мітка якоря: вхід (позиція) або ручний пін, час UTC
        const y0 = yAt(A.p);
        if (y0 != null) {
          ctx.fillStyle = col; ctx.beginPath(); ctx.arc(xy.x, y0, 3, 0, 2 * Math.PI); ctx.fill();
          const d = new Date(A.at * 1000), hh = String(d.getUTCHours()).padStart(2, "0"), mm = String(d.getUTCMinutes()).padStart(2, "0");
          const what = A.entry ? `${A.entry} entry${A.approx ? " ~" : ""}` : "pin";
          ctx.textAlign = "left"; ctx.textBaseline = "bottom"; ctx.fillText(`${what} ${hh}:${mm} UTC`, xy.x + 5, y0 - 4);
        }
      }
      ctx.restore();
    };
    const redraw = () => { if (!dead && !raf) raf = requestAnimationFrame(() => { raf = 0; draw(); }); };

    chart.timeScale().subscribeVisibleLogicalRangeChange(redraw);
    chart.subscribeCrosshairMove(redraw);              // масштаб ціни змінюється без окремої події
    const ro = new ResizeObserver(redraw); ro.observe(el);
    const tick = setInterval(() => { if (mode !== "off") redraw(); }, 2000);    // жива свічка рухає вершину конуса

    const api = {
      /** "off" | "live" | "pin"; перехід у pin закріплює конус від поточної ціни й прогнозу (якщо є дані) */
      setMode(m, fresh = false) {
        const was = mode; mode = ["live", "pin"].includes(m) ? m : "off";
        if (mode === "pin" && fresh) {
          const v = opts.vol(), lc = lastCandle();
          if (v && v.exp_4 != null && lc) {
            try { localStorage.setItem(pinKey(sym()), JSON.stringify({ at: Math.floor(Date.now() / 1000), p: lc.close, e1: v.exp_1, e4: v.exp_4 })); }
            catch (e) { /* немає сховища */ }
          }
        }
        if ((was === "off") !== (mode === "off")) applyOffset();
        redraw();
      },
      mode: () => mode,
      pinned: () => !!loadPin(sym()),
      /** після нових даних графіка (тікер, ТФ — інша кількість барів на 4 год) */
      reload() { if (mode !== "off") applyOffset(); redraw(); },
      redraw,
      destroy() {
        dead = true; clearInterval(tick); ro.disconnect(); cv.remove();
        chart.timeScale().unsubscribeVisibleLogicalRangeChange(redraw); chart.unsubscribeCrosshairMove(redraw);
        if (el._pdc === api) el._pdc = null;
      },
    };
    el._pdc = api;
    return api;
  }

  window.PDCone = { create, sigmaAt };
})();
