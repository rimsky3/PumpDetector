// Очікуваний рух на графіку — смужка біля шкали ціни (кнопка Range; 21.10 замість конуса-віяла: «конус займає дуже багато
// місця, не практично», варіант A компактного макета + числа з варіанта C у підказці). Свічки займають усю ширину: місця
// праворуч під прогноз не треба, як у ADR-рівнях / expected move професійних платформ.
//   смужка (≈ 12 px біля шкали): темніше — ±1σ на наступну 1 год, світліше — ±1σ на 4 год; риска — якір (ціна / вхід / пін),
//   біла риска — поточна ціна (коли якір не вона); мітки на шкалі ціни ±1H (блідіші) і ±4H (яскравіші), без підписів;
//   наведення на смужку — підказка: зона ±1σ на 1H / 4H, де ціна зараз у σ, скільки лишилось до −1σ і +1σ.
// Дані — прогноз volfc.py (HAR, «від зараз» кожні 5 хв): exp_1 і exp_4 з /api/ticker → vol; межі — ціна · e^(±σ).
// Між 1 і 4 год дисперсія росте лінійно з часом (sigmaAt). Це опис розмаху, НЕ рівні: від краю ціна не відскакує.
// Режими кнопки: live — від останньої ціни (план «якщо зайти зараз»); pin — закріплено в момент натискання: та ціна, той
// прогноз, на 4 год (localStorage pd-cone-pin-<TK>), блакитне. Позиція власника (my_positions.py): поки кнопка увімкнена,
// на тікері з відкритою позицією якір — твій вхід (блакитне, «long entry 14:05 UTC», «~» — вхід до старту сервісу).
// Внутрішні назви (cone, P.cone, #coneBtn, PDCone) лишились від конуса — не перейменовували.
(function () {
  "use strict";

  const HOURS = 4;
  // смужка впритул до шкали ціни (власник 21.10: «смужка далеко від шкали»); тому мітки на шкалі — без підписів +4H / +1H
  // (бібліотека малює підпис усередині графіка біля шкали — саме там, де смужка): 4H яскравіші, 1H блідіші, як і на смужці
  const BAR_W = 12, BAR_GAP = 2;
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
  const hhmm = (at) => { const d = new Date(at * 1000); return `${String(d.getUTCHours()).padStart(2, "0")}:${String(d.getUTCMinutes()).padStart(2, "0")}`; };
  const pct = (x) => (x >= 0 ? "+" : "−") + Math.abs(x * 100).toFixed(1) + "%";

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
    const tip = document.createElement("div");
    tip.style.cssText = "position:absolute;z-index:7;pointer-events:none;display:none;padding:6px 9px;border-radius:7px;font:11.5px/1.45 -apple-system,'Segoe UI',system-ui,sans-serif;"
      + "background:rgba(28,28,26,.94);border:1px solid #33332e;color:#a8a79f;white-space:nowrap;font-variant-numeric:tabular-nums";
    el.appendChild(tip);
    let mode = "off", dead = false, raf = 0, last = null;            // last — що намальовано (для підказки)
    const sym = () => (opts.symbol ? opts.symbol() : "") || "";
    const paneH = () => chart.panes()[0].getHeight();
    const plotW = () => chart.timeScale().width();
    const fmt = (p) => { try { return series.priceFormatter().format(p); } catch (e) { return p.toPrecision(5); } };
    const lastCandle = () => {
      // остання свічка разом з живою (WebSocket дошиває її через update): dataByIndex з «найближча зліва» — надійніше за data()
      try {
        const b = series.dataByIndex(1e9, -1);
        if (b && b.close > 0) return b;
      } catch (e) { /* стара версія бібліотеки — нижче */ }
      const data = series.data();
      let li = data.length - 1;
      while (li >= 0 && !(data[li].close > 0)) li--;           // наприкінці можуть бути порожні (whitespace) бари
      return li >= 1 ? data[li] : null;
    };

    // межі на шкалі ціни: чотири вбудовані мітки Lightweight Charts без ліній через графік; оновлюються лише при зміні
    const lines = {};
    const axis = (A, col) => {
      const s1 = A ? sigmaAt(1, A.e1, A.e4) : 0, s4 = A ? sigmaAt(HOURS, A.e1, A.e4) : 0;
      const want = A ? { u4: [A.p * Math.exp(s4), "+4H", col], u1: [A.p * Math.exp(s1), "+1H", rgba(col, 0.6)],
                         d1: [A.p * Math.exp(-s1), "−1H", rgba(col, 0.6)], d4: [A.p * Math.exp(-s4), "−4H", col] } : {};
      Object.keys(lines).forEach((k) => { if (!want[k]) { try { series.removePriceLine(lines[k].line); } catch (e) { /* серію вже прибрано */ } delete lines[k]; } });
      Object.entries(want).forEach(([k, [price, title, color]]) => {
        const old = lines[k];
        if (old && Math.abs(old.price - price) <= price * 1e-7 && old.color === color) return;
        if (old) old.line.applyOptions({ price, color, title: "" });
        else lines[k] = { line: series.createPriceLine({ price, color, title: "", lineVisible: false, axisLabelVisible: true }) };
        Object.assign(lines[k], { price, color });
      });
    };

    const anchor = () => {
      // позиція власника — пріоритетніша за ручний пін і живий режим
      const pos = opts.pos ? opts.pos() : null, pc = pos && pos.cone && pos.cone.p > 0 ? pos.cone : null;
      const v = opts.vol() || {};
      if (pc) return { at: pc.at, p: pc.p, e1: pc.e1 ?? v.exp_1, e4: pc.e4 ?? v.exp_4, what: `${pos.side} entry${pos.approx ? " ~" : ""}`, fixed: true };
      const pin = mode === "pin" ? loadPin(sym()) : null;
      if (pin) return { ...pin, what: "pin", fixed: true };
      const lc = lastCandle();
      return lc && v.exp_4 != null ? { at: null, p: lc.close, e1: v.exp_1, e4: v.exp_4, what: "price", fixed: false } : null;
    };

    const draw = () => {
      const dpr = window.devicePixelRatio || 1, W = el.clientWidth, Hh = el.clientHeight;
      if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(Hh * dpr)) {
        cv.width = Math.round(W * dpr); cv.height = Math.round(Hh * dpr); cv.style.width = W + "px"; cv.style.height = Hh + "px";
      }
      const ctx = cv.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, Hh);
      last = null;
      if (mode === "off" || dead) { if (!dead) axis(null); tip.style.display = "none"; return; }
      const A = anchor();
      if (!A || A.e4 == null) {
        axis(null);
        ctx.font = "12px -apple-system, Segoe UI, sans-serif"; ctx.fillStyle = cssVar("--fg3", "#6b6a64");
        ctx.textAlign = "right"; ctx.fillText("Range: прогнозу для цієї монети ще немає", plotW() - 8, paneH() - 10); return;
      }
      const col = A.fixed ? cssVar("--link", "#85b7eb") : cssVar("--amber", "#ef9f27");   // закріплене — блакитне
      axis(A, col);
      const s1 = sigmaAt(1, A.e1, A.e4), s4 = sigmaAt(HOURS, A.e1, A.e4), y = (p) => series.priceToCoordinate(p);
      const x0 = plotW() - BAR_GAP - BAR_W;
      const seg = (s, a) => {
        const yt = y(A.p * Math.exp(s)), yb = y(A.p * Math.exp(-s)); if (yt == null || yb == null) return;
        ctx.fillStyle = rgba(col, a); ctx.beginPath();
        if (ctx.roundRect) ctx.roundRect(x0, yt, BAR_W, yb - yt, 3); else ctx.rect(x0, yt, BAR_W, yb - yt);
        ctx.fill();
      };
      ctx.save(); ctx.beginPath(); ctx.rect(0, 0, plotW(), paneH()); ctx.clip();
      seg(s4, 0.22); seg(s1, 0.55);
      const tick = (p, c, w) => { const yy = y(p); if (yy == null) return; ctx.fillStyle = c; ctx.fillRect(x0 - 4, Math.round(yy) - w / 2, BAR_W + 8, w); };
      const lc = lastCandle(), now = lc ? lc.close : A.p;
      if (A.fixed) tick(A.p, col, 2);
      tick(now, cssVar("--fg", "#f0efec"), 2);
      ctx.restore();
      last = { A, s1, s4, now, col, x0 };
    };
    const redraw = () => { if (!dead && !raf) raf = requestAnimationFrame(() => { raf = 0; draw(); }); };

    // підказка при наведенні на смужку: числа з варіанта C макета
    const showTip = (pt) => {
      if (!last || !pt || pt.x < last.x0 - 6 || pt.x > last.x0 + BAR_W + 6) { tip.style.display = "none"; return; }
      const { A, s1, s4, now, col } = last;
      const z = Math.log(now / A.p);
      const row = (k, a, b) => `<tr><td style="padding-right:12px">${k}</td><td style="padding-right:12px;color:#f0efec">${a}</td><td style="color:#f0efec">${b}</td></tr>`;
      const head = A.fixed ? `<span style="color:${col}">${A.what} ${fmt(A.p)} · ${hhmm(A.at)} UTC</span>` : `<span style="color:${col}">from price ${fmt(A.p)}</span>`;
      tip.innerHTML = head + `<table style="border-collapse:collapse;margin-top:3px">`
        + row("", "1H", "4H")
        + row("±1σ zone", `${fmt(A.p * Math.exp(-s1))} – ${fmt(A.p * Math.exp(s1))}`, `${fmt(A.p * Math.exp(-s4))} – ${fmt(A.p * Math.exp(s4))}`)
        + (A.fixed ? row("price now", `${pct(Math.expm1(z))} · ${(z / s1).toFixed(2)}σ`, `${(z / s4).toFixed(2)}σ`) : "")
        + row("room to −1σ", pct(A.p * Math.exp(-s1) / now - 1), pct(A.p * Math.exp(-s4) / now - 1))
        + row("room to +1σ", pct(A.p * Math.exp(s1) / now - 1), pct(A.p * Math.exp(s4) / now - 1))
        + `</table><div style="margin-top:3px;font-size:10.5px;color:#6b6a64">1σ — inside about 2 of 3 times; a range, not levels</div>`;
      tip.style.display = "block";
      const w = tip.offsetWidth, h = tip.offsetHeight;
      tip.style.left = Math.max(4, last.x0 - w - 10) + "px";
      tip.style.top = Math.min(Math.max(4, pt.y - h / 2), paneH() - h - 4) + "px";
    };

    chart.timeScale().subscribeVisibleLogicalRangeChange(redraw);
    const onMove = (p) => { redraw(); showTip(p && p.point); };
    chart.subscribeCrosshairMove(onMove);              // масштаб ціни змінюється без окремої події
    const ro = new ResizeObserver(redraw); ro.observe(el);
    const tickT = setInterval(() => { if (mode !== "off") redraw(); }, 2000);    // жива свічка рухає поточну ціну

    const api = {
      /** "off" | "live" | "pin"; перехід у pin закріплює від поточної ціни й прогнозу (якщо є дані) */
      setMode(m, fresh = false) {
        mode = ["live", "pin"].includes(m) ? m : "off";
        if (mode === "pin" && fresh) {
          const v = opts.vol(), lc = lastCandle();
          if (v && v.exp_4 != null && lc) {
            try { localStorage.setItem(pinKey(sym()), JSON.stringify({ at: Math.floor(Date.now() / 1000), p: lc.close, e1: v.exp_1, e4: v.exp_4 })); }
            catch (e) { /* немає сховища */ }
          }
        }
        redraw();
      },
      mode: () => mode,
      pinned: () => !!loadPin(sym()),
      /** після нових даних графіка */
      reload() { redraw(); },
      redraw,
      destroy() {
        axis(null); dead = true; clearInterval(tickT); ro.disconnect(); cv.remove(); tip.remove();
        chart.timeScale().unsubscribeVisibleLogicalRangeChange(redraw); chart.unsubscribeCrosshairMove(onMove);
        if (el._pdc === api) el._pdc = null;
      },
    };
    el._pdc = api;
    return api;
  }

  window.PDCone = { create, sigmaAt };
})();
