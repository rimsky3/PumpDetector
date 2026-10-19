// Конус очікуваного руху на графіку (кнопка Cone, 2026-10-20; власник обрав варіант A макета): від останньої ціни вправо
// розходиться віяло на 4 год уперед — яскраво ±1σ (ціна в ньому приблизно у 2 з 3 випадків), бліде ±2σ (вихід — рідко).
// Дані — прогноз volfc.py (HAR, «від зараз» кожні 5 хв): exp_1 і exp_4 з /api/ticker → vol. Між ними дисперсія росте
// лінійно з часом (рух — як корінь з часу): σ(t) = exp_1·√t до 1 год, далі √(exp_1² + (exp_4² − exp_1²)·(t − 1)/3).
// Межі — у логах: ціна · e^(±kσ). Це опис розмаху, НЕ рівні: ціна від краю конуса не відскакує.
(function () {
  "use strict";

  const HOURS = 4;
  const cssVar = (name, fb) => (getComputedStyle(document.documentElement).getPropertyValue(name) || "").trim() || fb;
  const rgba = (c, a) => {
    if (c.startsWith("#") && c.length === 7) { const n = parseInt(c.slice(1), 16); return `rgba(${n >> 16 & 255},${n >> 8 & 255},${n & 255},${a})`; }
    return c;
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
    let on = false, dead = false, raf = 0, baseOffset = null;
    const bar = () => opts.bar() || 900;
    const paneH = () => chart.panes()[0].getHeight();
    const plotW = () => chart.timeScale().width();
    const ahead = () => Math.ceil(HOURS * 3600 / bar());

    // місце праворуч від останньої свічки під 4 год (5m — 48 барів, 15m — 16, 1h — 4); вимкнено — як було
    const applyOffset = () => {
      const ts = chart.timeScale();
      if (baseOffset == null) baseOffset = ts.options().rightOffset;
      ts.applyOptions({ rightOffset: on ? Math.max(baseOffset, ahead() + 3) : baseOffset });
    };

    const draw = () => {
      const dpr = window.devicePixelRatio || 1, W = el.clientWidth, Hh = el.clientHeight;
      if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(Hh * dpr)) {
        cv.width = Math.round(W * dpr); cv.height = Math.round(Hh * dpr); cv.style.width = W + "px"; cv.style.height = Hh + "px";
      }
      const ctx = cv.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, Hh);
      if (!on || dead) return;
      const v = opts.vol();
      const data = series.data();
      let li = data.length - 1;
      while (li >= 0 && !(data[li].close > 0)) li--;           // наприкінці можуть бути порожні (whitespace) бари
      if (!v || v.exp_4 == null || li < 1) {
        ctx.font = "12px -apple-system, Segoe UI, sans-serif"; ctx.fillStyle = cssVar("--fg3", "#6b6a64");
        ctx.fillText("Cone: прогнозу для цієї монети ще немає", 8, paneH() - 10); return;
      }
      // позиція останньої свічки на шкалі часу: номер у серії свічок ≠ логічний індекс шкали (інші панелі — довша історія)
      const ts = chart.timeScale(), P = data[li].close;
      const L = ts.timeToIndex(data[li].time, true);
      const xa = L == null ? null : ts.logicalToCoordinate(L - 1), xb = L == null ? null : ts.logicalToCoordinate(L);
      if (xa == null || xb == null || !P) return;
      const sp = xb - xa, perH = 3600 / bar();                 // px між барами; барів на годину
      const xAt = (t) => xb + sp * perH * t;
      const yAt = (p) => series.priceToCoordinate(p);
      const amber = cssVar("--amber", "#ef9f27"), fg2 = cssVar("--fg2", "#a8a79f");
      ctx.save(); ctx.beginPath(); ctx.rect(0, 0, plotW(), paneH()); ctx.clip();
      const STEPS = 32;
      const band = (k, fill, stroke) => {
        const up = [], dn = [];
        for (let i = 0; i <= STEPS; i++) {
          const t = HOURS * i / STEPS, s = sigmaAt(t, v.exp_1, v.exp_4), x = xAt(t);
          const yu = yAt(P * Math.exp(k * s)), yd = yAt(P * Math.exp(-k * s));
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
      band(2, rgba(amber, 0.07), null);
      band(1, rgba(amber, 0.2), rgba(amber, 0.55));
      // позначки +1H і +4H: пунктир і підпис ±σ
      ctx.font = "10.5px -apple-system, Segoe UI, sans-serif"; ctx.textBaseline = "top";
      [[1, v.exp_1], [HOURS, v.exp_4]].forEach(([t, e]) => {
        const x = xAt(t); if (e == null || x > plotW()) return;
        ctx.strokeStyle = rgba(fg2, 0.35); ctx.setLineDash([2, 3]); ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, paneH()); ctx.stroke(); ctx.setLineDash([]);
        const yl = yAt(P * Math.exp(-sigmaAt(t, v.exp_1, v.exp_4))); if (yl == null) return;
        const txt = `+${t}H ±${(e * 100).toFixed(e >= 0.1 ? 0 : 1)}%`;
        ctx.fillStyle = amber; ctx.textAlign = t === HOURS ? "right" : "center"; ctx.fillText(txt, t === HOURS ? x - 2 : x, yl + 4);
      });
      ctx.restore();
    };
    const redraw = () => { if (!dead && !raf) raf = requestAnimationFrame(() => { raf = 0; draw(); }); };

    chart.timeScale().subscribeVisibleLogicalRangeChange(redraw);
    chart.subscribeCrosshairMove(redraw);              // масштаб ціни змінюється без окремої події
    const ro = new ResizeObserver(redraw); ro.observe(el);
    const tick = setInterval(() => { if (on) redraw(); }, 2000);    // жива свічка рухає вершину конуса

    const api = {
      setOn(x) { const was = on; on = !!x; if (was !== on) applyOffset(); redraw(); },
      on: () => on,
      /** після нових даних графіка (тікер, ТФ — інша кількість барів на 4 год) */
      reload() { if (on) applyOffset(); redraw(); },
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
