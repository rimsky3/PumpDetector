// Картка Flow на графіку (кнопка Flow, 22.10; варіант A макета + шкала з 12 клітинок з варіанта B): хто зараз діє на
// монеті — живий опис, НЕ сигнал. Власник: «інформаційне віконечко, яке в реальному часі аналізує ці показники».
//   стан за 5 хв — квадрант ціна × OI: ↑↑ нові лонги, ↑↓ шорти закриваються (сквіз), ↓↑ нові шорти, ↓↓ лонги виходять;
//     OI росте / падає при пласкій ціні — позиції набирають / закривають; ціна йде без OI — «price only»;
//   шкала — 12 клітинок по 5 хв UTC (остання година, остання — поточна, ще не закрита), наведення — числа клітинки;
//   рядки: ціна × OI за 15 хв / 1 год, частка ринкових покупок (5 хв / 1 год), ліквідації лонгів / шортів
//     (5 / 15 хв), натовп (усі акаунти, Binance + Bybit) проти топ-трейдерів Binance (за розміром позицій);
//   ⚠ — гіпотези з розбору MAGMA 21.10 (сквіз «спалює паливо» шортів, Top ≈ 50 — великі гроші не в лонгу), НЕ перевірені.
// «Рух» — ≥ ¾ звичного для монети за останню годину (σ хвилинних змін ціни / 5-хв змін OI; OI — не менше 0.15%), щоб
// на сонній монеті і на гарячій поріг був свій; зміна стану на картці — коли новий стан тримається ≥ 6 с (гістерезис).
// Дані — напряму з браузера (Binance дозволяє CORS і WebSocket з будь-якої сторінки), сервер не навантажує:
//   1m-свічки (ціна, обсяг, ринкові покупки): REST за годину + WebSocket /market/ws/<sym>@kline_1m;
//   OI у монетах: /futures/data/openInterestHist 5m за годину + /fapi/v1/openInterest кожні 10 с (лише Binance);
//   ліквідації: /api/liqs — власний записувач liq_recorder.py (Binance + Bybit + OKX) кожні 15 с;
//   натовп / топ: ls_flow і top_ls з /api/ticker (15m-бари, як чіп Longs · Top у рядку над графіком).
// Згорнути до одного рядка (стан + шкала) — клік по заголовку картки (pd-flow-compact, для всіх вікон).
(function () {
  "use strict";

  const FAPI = "https://fapi.binance.com", WS = "wss://fstream.binance.com/market/ws/";
  const OI_POLL = 10000, LIQ_POLL = 15000, TICK = 2000, HOLD = 3;          // HOLD тіків по 2 с — гістерезис стану
  const CELLS = 12, CELL = 300;
  const MOVE = 0.75;         // «рух» — ≥ ¾ звичного σ монети за годину (з ½ σ на SOL рухом рахувалась майже кожна клітинка)
  // стани: колір шкали, назва, фраза; P / B / U / F — нейтральні (сірі на шкалі)
  const ST = {
    L: { c: "--ok", name: "New longs", why: "buyers opening: price ↑ OI ↑", short: "new longs" },
    S: { c: "--link", name: "Shorts closing", why: "squeeze: price ↑ OI ↓ — up-move on short covering", short: "squeeze" },
    N: { c: "--amber", name: "New shorts", why: "sellers opening: price ↓ OI ↑", short: "new shorts" },
    X: { c: "--hot-soft", name: "Longs exiting", why: "long unwind: price ↓ OI ↓", short: "longs out" },
    B: { c: null, name: "Positions building", why: "OI ↑ while price is flat — someone absorbs", short: "OI building" },
    U: { c: null, name: "Positions closing", why: "OI ↓ while price is flat", short: "OI closing" },
    P: { c: null, name: "Price only", why: "price moves, OI flat — no new positions", short: "price only" },
    F: { c: null, name: "Quiet", why: "no move bigger than usual for this coin", short: "flat" },
    "?": { c: null, name: "Waiting for OI", why: "Binance publishes 5m OI a few minutes late — fills in by itself", short: "no data" },
  };
  const cssVar = (n, fb) => (getComputedStyle(document.documentElement).getPropertyValue(n) || "").trim() || fb;
  const LS = {
    get: (k, d) => { try { const v = localStorage.getItem(k); return v == null ? d : JSON.parse(v); } catch (e) { return d; } },
    set: (k, v) => { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) { /* немає сховища */ } },
  };
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const pct = (x, d = 1) => x == null || !isFinite(x) ? "—" : (x >= 0 ? "+" : "−") + Math.abs(x * 100).toFixed(d) + "%";
  const usd = (v) => v == null ? "—" : v >= 1e6 ? "$" + (v / 1e6).toFixed(1) + "M" : v >= 1e3 ? "$" + Math.round(v / 1e3) + "K" : "$" + Math.round(v);
  const hhmm = (t) => { const d = new Date(t * 1000); return String(d.getUTCHours()).padStart(2, "0") + ":" + String(d.getUTCMinutes()).padStart(2, "0"); };
  const arrow = (x, thr) => x == null ? "·" : x >= thr ? "↑" : x <= -thr ? "↓" : "→";
  const std = (a) => { if (a.length < 3) return null; const m = a.reduce((s, x) => s + x, 0) / a.length; return Math.sqrt(a.reduce((s, x) => s + (x - m) ** 2, 0) / (a.length - 1)); };
  const getJson = async (u) => { const r = await fetch(u); if (!r.ok) throw new Error(r.status); return r.json(); };

  /** стан за зміною ціни dp і OI doi (частки) при порогах tp / to */
  const classify = (dp, doi, tp, to) => {
    if (dp == null || doi == null) return "F";
    const pm = Math.abs(dp) >= tp, om = Math.abs(doi) >= to;
    if (pm && om) return dp > 0 ? (doi > 0 ? "L" : "S") : (doi > 0 ? "N" : "X");
    if (om) return doi > 0 ? "B" : "U";
    return pm ? "P" : "F";
  };

  function create(opts) {
    const { el, chart } = opts;
    if (el._pdflow) el._pdflow.destroy();
    const card = document.createElement("div");
    card.className = "flow-card"; card.hidden = true;
    el.appendChild(card);

    let on = false, dead = false, sym = null, sock = null, wsRetry = 0, timers = [];
    let k1m = new Map(), oi = [], liq = { m: [], live: false }, shown = null, cand = null, candN = 0, gen = 0;
    let compact = !!LS.get("pd-flow-compact", false);
    // зсув годинника комп'ютера відносно Binance (с): усі часи картки — біржові. На ноутбуці 22.10 годинник відставав на 35 с,
    // і свіжий знімок OI виглядав «з майбутнього» — картка писала «Waiting for OI»
    let skew = 0;
    const nowS = () => Date.now() / 1000 + skew;

    const reset = () => {
      if (sock) { sock.onclose = null; try { sock.close(); } catch (e) { /* уже закрито */ } }
      clearTimeout(wsRetry); timers.forEach(clearInterval);
      sock = null; timers = []; k1m = new Map(); oi = []; liq = { m: [], live: false }; shown = null; cand = null; candN = 0; gen++;
    };

    // ---------- дані ----------
    const putK = (t, o, h, l, c, q, qb) => k1m.set(t, { t, o, h, l, c, q, qb });
    const loadHist = async (g) => {
      const [kl, oh] = await Promise.all([
        getJson(`${FAPI}/fapi/v1/klines?symbol=${sym}&interval=1m&limit=75`).catch(() => []),
        getJson(`${FAPI}/futures/data/openInterestHist?symbol=${sym}&period=5m&limit=15`).catch(() => [])]);
      if (g !== gen) return;
      kl.forEach((k) => putK(Math.floor(k[0] / 1000), +k[1], +k[2], +k[3], +k[4], +k[7], +k[10]));
      const hs = oh.map((r) => [Math.floor(r.timestamp / 1000), +r.sumOpenInterest]).filter((r) => r[1] > 0);
      oi = hs.concat(oi.filter((r) => !hs.length || r[0] > hs[hs.length - 1][0])).sort((a, b) => a[0] - b[0]);
    };
    const pollOi = async (g) => {
      try {
        const j = await getJson(`${FAPI}/fapi/v1/openInterest?symbol=${sym}`);
        if (g !== gen || !(+j.openInterest > 0)) return;
        skew = j.time / 1000 - Date.now() / 1000;
        oi.push([Math.floor(j.time / 1000), +j.openInterest]);
        const cut = nowS() - 4000;
        while (oi.length && oi[0][0] < cut) oi.shift();
      } catch (e) { /* наступне опитування */ }
    };
    const pollLiq = async (g) => {
      if (!opts.base()) return;
      try { const j = await getJson(`/api/liqs?symbol=${encodeURIComponent(opts.base())}`); if (g === gen) liq = j; } catch (e) { /* лишаємо попередні */ }
    };
    const connect = (g) => {
      if (g !== gen || dead || !on) return;
      const w = new WebSocket(`${WS}${sym.toLowerCase()}@kline_1m`);
      sock = w;
      w.onmessage = (ev) => {
        try {
          const k = JSON.parse(ev.data).k; if (!k) return;
          putK(Math.floor(k.t / 1000), +k.o, +k.h, +k.l, +k.c, +k.q, +k.Q);
        } catch (e) { /* не JSON */ }
      };
      w.onclose = () => { if (sock === w && g === gen) wsRetry = setTimeout(() => connect(g), 3000); };
    };
    const start = () => {
      reset();
      sym = opts.symbol();
      if (!on || dead) { card.hidden = true; return; }
      card.hidden = false;
      if (!sym) { card.innerHTML = `<div class="fl-h"><b>FLOW</b></div><div class="fl-note">Binance only — this coin has no Binance perp</div>`; return; }
      const g = gen;
      card.innerHTML = `<div class="fl-h"><b>FLOW · LIVE</b><span>loading…</span></div>`;
      loadHist(g).then(() => { if (g === gen) { connect(g); render(); } });
      pollOi(g); pollLiq(g);
      timers.push(setInterval(() => { if (!document.hidden) pollOi(g); }, OI_POLL));
      timers.push(setInterval(() => { if (!document.hidden) pollLiq(g); }, LIQ_POLL));
      timers.push(setInterval(() => { if (!document.hidden) render(); }, TICK));
    };

    // ---------- розрахунок ----------
    const bars = () => [...k1m.values()].sort((a, b) => a.t - b.t);
    // ціна в момент t: закриття хвилини, що закрилась до t (на межі хвилини — її відкриття наступної)
    const priceAt = (bs, t) => { let p = null; for (const b of bs) { if (b.t + 60 <= t) p = b.c; else if (b.t <= t) { p = p ?? b.o; break; } } return p; };
    // OI в момент t: між сусідніми знімками (≤ 11 хв один від одного: перші хвилини після відкриття картки між 5-хв знімком і опитуванням) — лінійно; після останнього — лише свіжий (≤ tol с).
    // Інакше «не знаю», а не «OI не змінився» (Binance публікує 5-хв знімки із запізненням ~3 хв)
    const oiAt = (t, tol = 60) => {
      let i = 0; while (i < oi.length && oi[i][0] <= t) i++;
      const a = oi[i - 1], b = oi[i];
      if (a && b && b[0] - a[0] <= 660) return a[1] + (b[1] - a[1]) * (t - a[0]) / (b[0] - a[0]);
      if (b && !a && b[0] - t <= 5) return b[1];
      return a && t - a[0] <= tol ? a[1] : null;
    };
    const sumK = (bs, a, b, f) => bs.reduce((s, x) => s + (x.t >= a && x.t < b ? f(x) : 0), 0);
    const sumL = (a, b) => liq.m.reduce((s, [t, l, sh]) => (t >= a && t < b ? [s[0] + l, s[1] + sh] : s), [0, 0]);

    const compute = () => {
      const bs = bars(); if (bs.length < 10 || oi.length < 2) return null;
      const now = nowS(), last = bs[bs.length - 1], pNow = last.c, oNow = oiAt(now, 60);
      // звичний розмах монети за годину: σ хвилинних змін ціни (закриті хвилини) і 5-хв змін OI
      const r1 = []; for (let i = 1; i < bs.length - 1; i++) r1.push(Math.log(bs[i].c / bs[i - 1].c));
      const s1 = std(r1) || 0.002;
      const o5 = []; for (let t = Math.floor(now / CELL) * CELL - CELL * CELLS; t < now - CELL; t += CELL) { const a = oiAt(t), b = oiAt(t + CELL); if (a && b) o5.push(Math.log(b / a)); }
      const so = std(o5) || 0.004;
      const thr = (sec) => ({ p: Math.max(0.001, MOVE * s1 * Math.sqrt(sec / 60)), o: Math.max(0.0015, MOVE * so * Math.sqrt(sec / CELL)) });
      const win = (sec) => {
        const p0 = priceAt(bs, now - sec), o0 = oiAt(now - sec), t = thr(sec);
        const dp = p0 ? pNow / p0 - 1 : null, doi = o0 ? oNow / o0 - 1 : null;
        return { dp, doi, t, k: dp == null || doi == null ? "?" : classify(dp, doi, t.p, t.o) };
      };
      const buy = (sec) => { const q = sumK(bs, now - sec - 59, now + 1, (x) => x.q); return q > 0 ? sumK(bs, now - sec - 59, now + 1, (x) => x.qb) / q : null; };
      const w5 = win(CELL), w15 = win(900), w60 = win(3600);
      // шкала: 12 клітинок по 5 хв UTC, остання — поточна
      const c0 = Math.floor(now / CELL) * CELL - CELL * (CELLS - 1), t5 = thr(CELL), cells = [];
      for (let i = 0; i < CELLS; i++) {
        const a = c0 + i * CELL, b = Math.min(a + CELL, now);
        const pa = priceAt(bs, a), pb = i === CELLS - 1 ? pNow : priceAt(bs, b), oa = oiAt(a), ob = i === CELLS - 1 ? oNow : oiAt(b);
        const dp = pa && pb ? pb / pa - 1 : null, doi = oa && ob ? ob / oa - 1 : null;
        const q = sumK(bs, a, a + CELL, (x) => x.q), lq = sumL(a, a + CELL);
        cells.push({ a, dp, doi, k: dp == null || doi == null ? "?" : classify(dp, doi, t5.p, t5.o),
                     buy: q > 0 ? sumK(bs, a, a + CELL, (x) => x.qb) / q : null, liq: lq });
      }
      return { now, w5, w15, w60, buy5: buy(CELL), buy15: buy(900), buy60: buy(3600), l5: sumL(now - CELL, now + 60), l15: sumL(now - 900, now + 60),
               l60: sumL(now - 3600, now + 60), cells };
    };

    // гіпотези (⚠, не перевірені): перша, що підходить
    const hint = (R, crowd, top) => {
      const sq = R.cells.filter((c) => c.k === "S");
      const lAvg = R.l60[0] / 12;
      if (R.l5[0] >= 50000 && R.l5[0] >= 3 * lAvg && ["X", "U", "N"].includes(R.w5.k))
        return "Long liquidations cascade — forced selling; it usually stops when liquidations dry up";
      if (R.l5[1] >= 50000 && R.l5[1] >= 3 * (R.l60[1] / 12) && ["S", "P"].includes(R.w5.k))
        return "Short liquidations — forced buying; this fuel is spent once they stop";
      if (sq.length >= 2 && R.w5.k !== "S")
        return `Short fuel spent at ${hhmm(sq[sq.length - 1].a)} (up-move with OI falling) — crowd-short ▲ is weaker now`;
      if (top != null && top >= 45 && top <= 55 && crowd != null && crowd < 40)
        return "Top ≈ 50 — big money is not long; the ▲ gap comes from crowd shorts only";
      return null;
    };

    // ---------- вигляд ----------
    // колір стану: на шкалі нейтральні — приглушені, у тексті — читабельний сірий
    const col = (k) => (ST[k] && ST[k].c ? cssVar(ST[k].c) : cssVar("--muted", "#5f5e5a"));
    const tcol = (k) => (ST[k] && ST[k].c ? cssVar(ST[k].c) : cssVar("--fg2", "#a8a79f"));
    const buyCls = (x) => x == null ? "" : x >= 0.55 ? "up" : x <= 0.45 ? "dn" : "";
    const pxo = (w) => w.k === "?" ? "" : `${arrow(w.dp, w.t.p)}${arrow(w.doi, w.t.o)} `;
    const place = () => {
      let ph = el.clientHeight;
      try { ph = chart.panes()[0].getHeight(); } catch (e) { /* одна панель */ }
      card.style.top = Math.max(4, ph - card.offsetHeight - 8) + "px";
    };
    const render = () => {
      if (!on || dead || !sym) return;
      const R = compute();
      if (!R) { card.innerHTML = `<div class="fl-h"><b>FLOW · LIVE</b><span>waiting for data…</span></div>`; place(); return; }
      // гістерезис: новий стан показуємо, коли він тримається HOLD тіків поспіль
      const k = R.w5.k;
      if (shown == null || k === shown) { shown = k; cand = null; candN = 0; } else if (k === cand) { if (++candN >= HOLD) { shown = k; cand = null; candN = 0; } } else { cand = k; candN = 1; }
      const s = ST[shown], d = opts.crowd() || {}, crowd = d.ls?.ls, top = d.top?.pos;
      const gap = crowd != null && top != null ? top - crowd : null;
      const topNote = top == null ? "" : top >= 45 && top <= 55 ? "Top ≈ 50 → neutral" : top > 55 ? "big money long" : "big money short";
      const h = hint(R, crowd, top);
      const cells = R.cells.map((c) => {
        const t = c.k === "?" ? `${hhmm(c.a)} · no data` : `${hhmm(c.a)}–${hhmm(c.a + CELL)} · ${ST[c.k].short}\nprice ${pct(c.dp)} · OI ${pct(c.doi)}`
          + (c.buy != null ? ` · buy ${Math.round(c.buy * 100)}%` : "") + (c.liq[0] + c.liq[1] > 0 ? `\nliq L ${usd(c.liq[0])} · S ${usd(c.liq[1])}` : "");
        return `<i style="background:${c.k === "?" ? "transparent" : col(c.k)}" title="${esc(t)}"></i>`;
      }).join("");
      const sq = R.cells.findLastIndex((c) => c.k === "S");
      const mid = sq >= 0 && sq < CELLS - 2 ? `${hhmm(R.cells[sq].a)} squeeze` : "";
      const head = `<div class="fl-h" title="Click to ${compact ? "expand" : "collapse"}"><b>FLOW · LIVE</b><span>${hhmm(R.now)} UTC ${compact ? "▸" : "▾"}</span></div>`
        + `<div class="fl-st"><span class="fl-pill" style="--c:${tcol(shown)}">${s.name}</span><span>price ${pct(R.w5.dp)} · OI ${pct(R.w5.doi)} (5m)</span></div>`;
      const strip = `<div class="fl-tl">${cells}</div><div class="fl-tla"><span>${hhmm(R.cells[0].a)}</span><span>${mid}</span><span>now</span></div>`;
      if (compact) { card.innerHTML = head + strip; place(); return; }
      card.innerHTML = head + `<div class="fl-why">${esc(s.why)}</div>` + strip
        + `<div class="fl-g"><span></span><span class="fl-k">5m</span><span class="fl-k">15m</span><span class="fl-k">1h</span>`
        + `<span class="fl-l">Price × OI</span>` + [R.w5, R.w15, R.w60].map((w) => `<span style="color:${tcol(w.k)}" title="price ${pct(w.dp)} · OI ${pct(w.doi)}">${pxo(w)}${ST[w.k].short}</span>`).join("")
        + `<span class="fl-l" title="Share of market (taker) buys in turnover, Binance 1m bars">Buy flow</span>`
        + [R.buy5, R.buy15, R.buy60].map((x) => `<span class="${buyCls(x)}">${x == null ? "—" : Math.round(x * 100) + "%"}</span>`).join("")
        + `<span class="fl-l" title="Forced closes, longs / shorts: L — forced selling, S — forced buying. Binance + Bybit + OKX, own recorder (liq_recorder.py)">Liq. L / S</span>`
        + (liq.live ? [R.l5, R.l15, R.l60].map(([l, sh]) => `<span><b class="${l > sh * 2 && l > 0 ? "dn" : ""}">${usd(l)}</b>/<b class="${sh > l * 2 && sh > 0 ? "up" : ""}">${usd(sh)}</b></span>`).join("")
          : `<span class="fl-m fl-span">recorder offline — no data ≠ zero</span>`)
        + `<span class="fl-l" title="Crowd — long share of all accounts (Binance + Bybit); Top — long share of top traders' positions (Binance); last 15m bar">Crowd / Top</span>`
        + `<span>${crowd == null ? "—" : Math.round(crowd) + "%"} / ${top == null ? "—" : Math.round(top) + "%"}</span>`
        + `<span class="fl-m fl-span2">${topNote}${gap != null && Math.abs(gap) >= 10 ? ` · ${gap > 0 ? "▲" : "▼"}${Math.abs(Math.round(gap))}` : ""}</span></div>`
        + (h ? `<div class="fl-hint">⚠ ${esc(h)}</div>` : "")
        + `<div class="fl-note">description, not a signal · ⚠ untested hypotheses</div>`;
      place();
    };

    card.addEventListener("click", (e) => {
      if (!e.target.closest(".fl-h")) return;
      compact = !compact; LS.set("pd-flow-compact", compact); render();
    });
    const ro = new ResizeObserver(place); ro.observe(el);

    const api = {
      setOn(v) { const was = on; on = !!v; if (on !== was || (on && opts.symbol() !== sym)) start(); },
      /** нові дані графіка: інший тікер — заново, той самий — лише перемалювати (натовп / топ з /api/ticker) */
      reload() { if (!on) return; if (opts.symbol() !== sym) start(); else render(); },
      destroy() { dead = true; reset(); ro.disconnect(); card.remove(); if (el._pdflow === api) el._pdflow = null; },
    };
    el._pdflow = api;
    return api;
  }

  window.PDFlow = { create, classify };
})();
