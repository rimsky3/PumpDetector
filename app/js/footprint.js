// Футпрінт на графіку (кнопка FP, 2026-10-07): у кожній свічці — обсяг агресивних продажів (ліворуч від центру, червоне)
// і купівель (праворуч, зелене) на рівнях ціни; рамка — POC бару (рівень з найбільшим обсягом); на широких свічках —
// числа, як у кластерах ATAS. Дані — /api/footprint (5m-бари записувача угод і архіву Binance, рівні через 0.1%:
// рівень k = [e^(k·LVL), e^((k+1)·LVL)), footprint.py), лише для видимих барів і лише коли кнопка увімкнена.
// Вузькі свічки (графік віддалено) не малюються — там рівнів однаково не прочитати; підказка «наблизь».
(function () {
  "use strict";

  const LVL = Math.log(1.001);
  const MIN_SPACING = 16;          // px між свічками, з яких малюємо рівні
  const MIN_ROW = 3;               // px — мінімальна висота рядка; дрібніші рівні зливаємо
  const NUM_SPACING = 70, NUM_ROW = 11;   // з яких розмірів пишемо числа

  const cssVar = (name, fb) => (getComputedStyle(document.documentElement).getPropertyValue(name) || "").trim() || fb;
  const rgba = (c, a) => {
    if (c.startsWith("#") && c.length === 7) { const n = parseInt(c.slice(1), 16); return `rgba(${n >> 16 & 255},${n >> 8 & 255},${n & 255},${a})`; }
    return c;
  };
  const fmtK = (v) => v >= 1e6 ? (v / 1e6).toFixed(1) + "M" : v >= 1e3 ? Math.round(v / 1e3) + "k" : v >= 1 ? String(Math.round(v)) : "";

  function create(opts) {
    const { chart, el, series } = opts;
    if (el._pdf) el._pdf.destroy();
    const cv = document.createElement("canvas");
    cv.style.cssText = "position:absolute;left:0;top:0;pointer-events:none;z-index:3";
    el.appendChild(cv);
    let on = false, dead = false, raf = 0, timer = 0;
    let data = new Map(), dataKey = null, have = null;      // have — [from, to] уже завантаженого діапазону
    const times = () => opts.times() || [];
    const bar = () => opts.bar() || 900;
    const key = () => `${(opts.symbol() || "").toUpperCase()}|${opts.tf() || ""}`;
    const info = (t) => { if (opts.onInfo) opts.onInfo(t); };
    const paneH = () => chart.panes()[0].getHeight();
    const plotW = () => chart.timeScale().width();

    // ---------- дані ----------
    const visibleSpan = () => {
      const T = times(), r = chart.timeScale().getVisibleLogicalRange();
      if (!T.length || !r) return null;
      const a = Math.max(0, Math.floor(r.from) - 5), b = Math.min(T.length - 1, Math.ceil(r.to) + 5);
      return a > b ? null : [T[a], T[b] + bar()];
    };
    const load = async () => {
      if (!on || dead) return;
      const k = key();
      if (k !== dataKey) { data = new Map(); dataKey = k; have = null; }
      const span = visibleSpan(); if (!span) return;
      if (have && span[0] >= have[0] && span[1] <= have[1]) return;
      const from = Math.min(span[0], have ? have[0] : span[0]), to = Math.max(span[1], have ? have[1] : span[1]);
      const tf = opts.tf() || "15m", sym = (opts.symbol() || "").toUpperCase();
      try {
        const r = await fetch(`/api/footprint?symbol=${encodeURIComponent(sym)}&tf=${tf}&from=${span[0]}&to=${span[1]}`, { cache: "no-store" });
        const j = await r.json();
        if (!r.ok) throw new Error(j.error || `HTTP ${r.status}`);
        if (key() !== k) return;
        (j.bars || []).forEach((b) => data.set(b.t, b));
        have = [from, to];
        info(data.size ? "" : "Футпрінт: для цих барів даних немає (записувач угод / архів — монети зі списку)");
      } catch (e) { info("Футпрінт: " + e.message); }
      redraw();
    };
    const schedule = () => { if (!on) return; clearTimeout(timer); timer = setTimeout(load, 250); };

    // ---------- малювання ----------
    const draw = () => {
      const dpr = window.devicePixelRatio || 1, W = el.clientWidth, H = el.clientHeight;
      if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) {
        cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); cv.style.width = W + "px"; cv.style.height = H + "px";
      }
      const ctx = cv.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
      if (!on || dead) return;
      const T = times(), ts = chart.timeScale(), r = ts.getVisibleLogicalRange();
      if (!T.length || !r) return;
      const a0 = Math.max(0, Math.floor(r.from)), a1 = Math.min(T.length - 1, Math.ceil(r.to));
      const xa = ts.logicalToCoordinate(a0), xb = ts.logicalToCoordinate(a0 + 1);
      const spacing = xa != null && xb != null ? xb - xa : 0;
      ctx.save(); ctx.beginPath(); ctx.rect(0, 0, plotW(), paneH()); ctx.clip();
      if (spacing < MIN_SPACING) {
        ctx.font = "12px -apple-system, Segoe UI, sans-serif"; ctx.fillStyle = cssVar("--fg2", "#a8a79f");
        ctx.fillText("FP: наблизь графік (колесо миші), щоб побачити футпрінт", 8, 16);
        ctx.restore(); return;
      }
      const sellC = rgba(cssVar("--hot", "#e24b4a"), .6), buyC = rgba(cssVar("--ok", "#97c459"), .6);
      const pocC = cssVar("--warm", "#f0997b"), txt = cssVar("--fg", "#f0efec");
      const half = spacing * 0.46;
      const y = (lvl) => series.priceToCoordinate(Math.exp(lvl * LVL));
      for (let i = a0; i <= a1; i++) {
        const b = data.get(T[i]); if (!b) continue;
        const x = ts.logicalToCoordinate(i); if (x == null) continue;
        const n = b.buy.length;
        const h1 = Math.abs((y(b.lvl0) ?? 0) - (y(b.lvl0 + 1) ?? 0));
        const g = Math.max(1, Math.ceil(MIN_ROW / Math.max(h1, 0.01)));        // скільки рівнів зливаємо в рядок
        const rows = new Map();
        for (let k = 0; k < n; k++) {
          const lv = Math.floor((b.lvl0 + k) / g) * g;
          const row = rows.get(lv) || [0, 0];
          row[0] += b.sell[k]; row[1] += b.buy[k]; rows.set(lv, row);
        }
        let maxSide = 1, poc = null, pocV = -1;
        rows.forEach(([s, u], lv) => { maxSide = Math.max(maxSide, s, u); if (s + u > pocV) { pocV = s + u; poc = lv; } });
        rows.forEach(([s, u], lv) => {
          const yTop = y(lv + g), yBot = y(lv); if (yTop == null || yBot == null) return;
          const top = Math.min(yTop, yBot), hh = Math.max(1, Math.abs(yBot - yTop) - (Math.abs(yBot - yTop) > 4 ? 1 : 0));
          if (top > paneH() || top + hh < 0) return;
          const ws = s / maxSide * half, wb = u / maxSide * half;
          ctx.fillStyle = sellC; ctx.fillRect(x - 1 - ws, top, ws, hh);
          ctx.fillStyle = buyC; ctx.fillRect(x + 1, top, wb, hh);
          if (lv === poc) { ctx.strokeStyle = pocC; ctx.lineWidth = 1; ctx.strokeRect(Math.round(x - half) + 0.5, Math.round(top) + 0.5, Math.round(2 * half), Math.max(1, Math.round(hh) - 1)); }
          if (spacing >= NUM_SPACING && hh >= NUM_ROW) {
            ctx.font = `${Math.min(11, Math.floor(hh) - 1)}px -apple-system, Segoe UI, sans-serif`; ctx.fillStyle = txt;
            ctx.textBaseline = "middle";
            ctx.textAlign = "right"; ctx.fillText(fmtK(s), x - 3, top + hh / 2);
            ctx.textAlign = "left"; ctx.fillText(fmtK(u), x + 3, top + hh / 2);
          }
        });
      }
      ctx.restore();
    };
    const redraw = () => { if (!dead && !raf) raf = requestAnimationFrame(() => { raf = 0; draw(); }); };

    const onRange = () => { redraw(); schedule(); };
    chart.timeScale().subscribeVisibleLogicalRangeChange(onRange);
    chart.subscribeCrosshairMove(redraw);              // масштаб ціни змінюється без окремої події
    const ro = new ResizeObserver(redraw); ro.observe(el);

    const api = {
      setOn(v) { on = !!v; if (on) schedule(); redraw(); },
      on: () => on,
      /** після зміни тікера, таймфрейму або даних графіка */
      reload() { if (key() !== dataKey) { data = new Map(); dataKey = key(); have = null; } schedule(); redraw(); },
      redraw,
      destroy() {
        dead = true; clearTimeout(timer); ro.disconnect(); cv.remove();
        chart.timeScale().unsubscribeVisibleLogicalRangeChange(onRange); chart.unsubscribeCrosshairMove(redraw);
        if (el._pdf === api) el._pdf = null;
      },
    };
    el._pdf = api;
    return api;
  }

  window.PDFootprint = { create };
})();
