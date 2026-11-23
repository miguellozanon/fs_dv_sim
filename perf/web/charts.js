// Small SVG charts for the Performance Lab (no libraries, works offline).
// Colours come from CSS custom properties (perf.css), so light and dark themes both work.
(() => {
"use strict";

const SVGNS = "http://www.w3.org/2000/svg";
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const SERIES = () => [1, 2, 3, 4, 5, 6].map((i) => css(`--series-${i}`));

function el(tag, attrs = {}, parent) {
  const e = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}

function niceTicks(lo, hi, n = 5) {
  if (!isFinite(lo) || !isFinite(hi)) return [];
  if (hi - lo <= 1e-6 * Math.max(1, Math.abs(hi))) { const c = (hi + lo) / 2; hi = c + 1; lo = c - 1; }   // flat data
  const span = hi - lo, step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const f = step0 / mag, step = (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-6; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  out.dec = Math.max(0, -Math.floor(Math.log10(step) + 1e-9));
  return out;
}

function fmt(v, d) {
  if (v === null || v === undefined || v === "" || !isFinite(v)) return "–";
  v = Number(v);
  const a = Math.abs(v);
  const s = d !== undefined ? v.toFixed(d) : a >= 1000 ? v.toFixed(0) : a >= 100 ? v.toFixed(1) : a >= 1 ? v.toFixed(2) : v.toFixed(3);
  return /^-0(\.0*)?$/.test(s) ? s.slice(1) : s;
}

// shared tooltip
let TIP;
function tip() {
  if (!TIP) { TIP = document.createElement("div"); TIP.className = "ctip"; TIP.hidden = true; document.body.appendChild(TIP); }
  return TIP;
}
function showTip(evt, html) {
  const t = tip();
  t.innerHTML = html; t.hidden = false;
  const pad = 14, w = t.offsetWidth, h = t.offsetHeight;
  let x = evt.clientX + pad, y = evt.clientY + pad;
  if (x + w > window.innerWidth - 8) x = evt.clientX - w - pad;
  if (y + h > window.innerHeight - 8) y = evt.clientY - h - pad;
  t.style.left = `${x}px`; t.style.top = `${y}px`;
}
function hideTip() { tip().hidden = true; }

function legend(container, series, swatch) {
  if (series.length < 2) return;
  const ul = document.createElement("ul");
  ul.className = "clegend";
  for (const s of series) {
    const li = document.createElement("li");
    li.innerHTML = swatch ? `<i class="sw" style="background:${s.color}"></i>` : `<i class="lnk${s.dash ? " dash" : ""}" style="border-color:${s.color}"></i>`;
    li.append(s.name);
    ul.appendChild(li);
  }
  container.appendChild(ul);
}

function bisect(xs, x) {
  let lo = 0, hi = xs.length - 1;
  if (hi < 0) return -1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (xs[m] <= x) lo = m; else hi = m; }
  return Math.abs(xs[hi] - x) < Math.abs(xs[lo] - x) ? hi : lo;
}

// ------------------------------------------------------------------ line chart
// opts: {series: [{name, x, y, color?, dash?, points? (scatter)}], xLabel, yLabel, height, yFmt, xFmt, zero, markers}
function lineChart(container, opts) {
  container.innerHTML = "";
  container.classList.add("chart");
  const pal = SERIES();
  const series = opts.series.filter((s) => s && s.x && s.x.length).map((s, i) => ({ ...s, color: s.color || pal[i % pal.length] }));
  if (opts.legend !== false) legend(container, series);
  else if (opts.caption) container.insertAdjacentHTML("beforeend", `<p class="ccap">${opts.caption}</p>`);
  if (!series.length) { container.insertAdjacentHTML("beforeend", `<p class="empty">no data</p>`); return; }
  const W = Math.max(container.clientWidth || 600, 280), H = opts.height || 220;
  const m = { l: 56, r: 14, t: 10, b: 38 };
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const s of series) for (let i = 0; i < s.x.length; i++) {
    const x = s.x[i], y = s.y[i];
    if (x === null || y === null || !isFinite(x) || !isFinite(y)) continue;
    x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y);
  }
  if (opts.zero) { y0 = Math.min(y0, 0); y1 = Math.max(y1, 0); }
  if (opts.yMin !== undefined) y0 = Math.min(y0, opts.yMin);
  if (!isFinite(x0)) { container.insertAdjacentHTML("beforeend", `<p class="empty">no data</p>`); return; }
  if (y1 - y0 <= 1e-6 * Math.max(1, Math.abs(y1))) { const c = (y0 + y1) / 2; y0 = c - 1; y1 = c + 1; }   // flat data
  const yp = (y1 - y0) * 0.06 || 1; y0 -= opts.zero && y0 === 0 ? 0 : yp; y1 += yp;
  const X = (x) => m.l + (x - x0) / ((x1 - x0) || 1) * (W - m.l - m.r);
  const Y = (y) => H - m.b - (y - y0) / ((y1 - y0) || 1) * (H - m.t - m.b);
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", height: H, role: "img",
    "aria-label": `${opts.yLabel || ""} against ${opts.xLabel || ""}` }, container);
  const gGrid = el("g", { class: "grid" }, svg);
  const yt = niceTicks(y0, y1, 4), xt = niceTicks(x0, x1, Math.max(3, Math.floor(W / 110)));
  for (const t of yt) {
    el("line", { x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t) }, gGrid);
    el("text", { x: m.l - 6, y: Y(t) + 4, "text-anchor": "end", class: "tick" }, svg).textContent = fmt(t, yt.dec);
  }
  for (const t of xt) {
    el("text", { x: X(t), y: H - m.b + 16, "text-anchor": "middle", class: "tick" }, svg).textContent = fmt(t, xt.dec);
  }
  el("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, class: "axis" }, svg);
  if (opts.zero && y0 < 0 && y1 > 0) el("line", { x1: m.l, x2: W - m.r, y1: Y(0), y2: Y(0), class: "axis" }, svg);
  if (opts.xLabel) el("text", { x: (m.l + W - m.r) / 2, y: H - 4, "text-anchor": "middle", class: "lab" }, svg).textContent = opts.xLabel;
  if (opts.yLabel) el("text", { x: 12, y: (H - m.b + m.t) / 2, "text-anchor": "middle", class: "lab",
    transform: `rotate(-90 12 ${(H - m.b + m.t) / 2})` }, svg).textContent = opts.yLabel;
  for (const s of series) {
    if (s.points) {                    // scatter series (measured samples): small dots, no line, no hover
      const g = el("g", { fill: s.color, "fill-opacity": 0.45 }, svg);
      for (let i = 0; i < s.x.length; i++) if (isFinite(s.x[i]) && isFinite(s.y[i]) && s.x[i] !== null && s.y[i] !== null)
        el("circle", { cx: X(s.x[i]).toFixed(1), cy: Y(s.y[i]).toFixed(1), r: 2 }, g);
      continue;
    }
    let d = "", pen = false;
    for (let i = 0; i < s.x.length; i++) {
      const x = s.x[i], y = s.y[i];
      if (x === null || y === null || !isFinite(x) || !isFinite(y)) { pen = false; continue; }
      d += `${pen ? "L" : "M"}${X(x).toFixed(1)},${Y(y).toFixed(1)}`; pen = true;
    }
    el("path", { d, fill: "none", stroke: s.color, "stroke-width": s.width || 2, "stroke-linejoin": "round",
      "stroke-dasharray": s.dash ? "6 4" : "none" }, svg);
    if (opts.markers) for (let i = 0; i < s.x.length; i++) if (isFinite(s.y[i]) && s.y[i] !== null)
      el("circle", { cx: X(s.x[i]), cy: Y(s.y[i]), r: 4, fill: s.color, stroke: css("--panel"), "stroke-width": 2 }, svg);
  }
  if (opts.mark !== undefined && isFinite(opts.mark)) {
    el("line", { x1: X(opts.mark), x2: X(opts.mark), y1: m.t, y2: H - m.b, class: "markline" }, svg);
  }
  // hover: crosshair + values of every series at the nearest x
  const cross = el("line", { y1: m.t, y2: H - m.b, class: "cross", visibility: "hidden" }, svg);
  const dots = series.map((s) => el("circle", { r: 4, fill: s.color, stroke: css("--panel"), "stroke-width": 2, visibility: "hidden" }, svg));
  const hit = el("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent" }, svg);
  hit.addEventListener("mousemove", (evt) => {
    const r = svg.getBoundingClientRect(), px = (evt.clientX - r.left) * W / r.width;
    const xv = x0 + (px - m.l) / (W - m.l - m.r) * (x1 - x0);
    cross.setAttribute("x1", px); cross.setAttribute("x2", px); cross.setAttribute("visibility", "visible");
    let rows = "";
    series.forEach((s, k) => {
      if (s.points) return;
      const i = bisect(s.x, xv), y = s.y[i];
      if (i < 0 || y === null || !isFinite(y)) { dots[k].setAttribute("visibility", "hidden"); return; }
      dots[k].setAttribute("cx", X(s.x[i])); dots[k].setAttribute("cy", Y(y)); dots[k].setAttribute("visibility", "visible");
      rows += `<div><span><i class="sw" style="background:${s.color}"></i>${s.name || opts.yLabel || ""}</span><span>${(opts.yFmt || fmt)(y)}</span></div>`;
    });
    showTip(evt, `<b>${opts.xLabel || "x"}: ${(opts.xFmt || fmt)(xv)}</b>${rows}`);
  });
  hit.addEventListener("mouseleave", () => { hideTip(); cross.setAttribute("visibility", "hidden"); dots.forEach((d) => d.setAttribute("visibility", "hidden")); });
}

// ------------------------------------------------------------------ horizontal bars around zero
// rows: [{label, values: [v per series]}], series: [{name, color?}]
function barChart(container, opts) {
  container.innerHTML = "";
  container.classList.add("chart");
  const pal = SERIES();
  const series = opts.series.map((s, i) => ({ ...s, color: s.color || pal[i % pal.length] }));
  legend(container, series, true);
  const rows = opts.rows;
  const W = Math.max(container.clientWidth || 600, 280), bh = 12, gap = 2, rowH = series.length * (bh + gap) + 12;
  const m = { l: opts.labelWidth || 130, r: 70, t: 6, b: 30 }, H = m.t + m.b + rows.length * rowH;
  let lo = 0, hi = 0;
  for (const r of rows) for (const v of r.values) if (isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  if (lo === hi) hi = lo + 1;
  const X = (v) => m.l + (v - lo) / (hi - lo) * (W - m.l - m.r);
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", height: H, role: "img", "aria-label": opts.xLabel || "bars" }, container);
  const bt = niceTicks(lo, hi, 5);
  for (const t of bt) {
    el("line", { x1: X(t), x2: X(t), y1: m.t, y2: H - m.b, class: "gridv" }, svg);
    el("text", { x: X(t), y: H - m.b + 14, "text-anchor": "middle", class: "tick" }, svg).textContent = fmt(t, bt.dec);
  }
  el("line", { x1: X(0), x2: X(0), y1: m.t, y2: H - m.b, class: "axis" }, svg);
  if (opts.xLabel) el("text", { x: (m.l + W - m.r) / 2, y: H - 2, "text-anchor": "middle", class: "lab" }, svg).textContent = opts.xLabel;
  rows.forEach((r, i) => {
    const yRow = m.t + i * rowH;
    el("text", { x: m.l - 8, y: yRow + rowH / 2 + 3, "text-anchor": "end", class: "rowlab" }, svg).textContent = r.label;
    r.values.forEach((v, k) => {
      if (!isFinite(v)) return;
      const y = yRow + 6 + k * (bh + gap), xa = X(Math.min(0, v)), xb = X(Math.max(0, v));
      const w = Math.max(xb - xa, 1);
      const b = el("rect", { x: xa, y, width: w, height: bh, rx: Math.min(3, w / 2), fill: series[k].color }, svg);
      const label = (opts.valFmt || fmt)(v), tw = label.length * 6.5 + 6;
      // outside the bar, unless it would run into the row labels (left) or off the chart (right): then inside
      const inside = v >= 0 ? xb + tw > W : xa - tw < m.l;
      const t = el("text", { x: v >= 0 ? (inside ? xb - 4 : xb + 4) : (inside ? xa + 4 : xa - 4), y: y + bh - 2,
        "text-anchor": (v >= 0) !== inside ? "start" : "end", class: inside ? "val in" : "val" }, svg);
      t.textContent = label;
      const h = el("rect", { x: m.l, y: y - 1, width: W - m.l - m.r, height: bh + 2, fill: "transparent" }, svg);
      h.addEventListener("mousemove", (evt) => showTip(evt, `<b>${r.label}</b><div><span><i class="sw" style="background:${series[k].color}"></i>${series[k].name}</span><span>${(opts.valFmt || fmt)(v)}</span></div>${r.note ? `<div><span>${r.note}</span></div>` : ""}`));
      h.addEventListener("mouseleave", hideTip);
      void b;
    });
  });
}

// ------------------------------------------------------------------ sequential ramp (one hue, light -> dark)
function ramp(t) {
  t = Math.max(0, Math.min(1, t));
  const a = css("--seq-lo").match(/\w\w/g).map((h) => parseInt(h, 16)), b = css("--seq-hi").match(/\w\w/g).map((h) => parseInt(h, 16));
  return `rgb(${a.map((v, i) => Math.round(v + (b[i] - v) * t)).join(",")})`;
}

// ------------------------------------------------------------------ heatmap
// opts: {xs, ys, z[i][j] (i over ys rows? no: z[ix][iy]), xLabel, yLabel, zLabel, best: {x, y}}
function heatmap(container, opts) {
  container.innerHTML = "";
  container.classList.add("chart");
  const { xs, ys, z } = opts;
  const W = Math.max(container.clientWidth || 600, 280), H = opts.height || 320;
  const m = { l: 60, r: 90, t: 8, b: 40 };
  const cw = (W - m.l - m.r) / xs.length, ch = (H - m.t - m.b) / ys.length;
  let lo = Infinity, hi = -Infinity;
  z.forEach((row) => row.forEach((v) => { if (v !== null && isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); } }));
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", height: H, role: "img", "aria-label": opts.zLabel || "map" }, container);
  xs.forEach((x, i) => ys.forEach((y, j) => {
    const v = z[i][j], cx = m.l + i * cw, cy = H - m.b - (j + 1) * ch;
    const r = el("rect", { x: cx + 1, y: cy + 1, width: Math.max(cw - 2, 1), height: Math.max(ch - 2, 1), rx: 2,
      fill: v === null || !isFinite(v) ? "transparent" : ramp((v - lo) / ((hi - lo) || 1)) }, svg);
    if (opts.best && opts.best.x === x && opts.best.y === y) r.setAttribute("class", "bestcell");
    r.addEventListener("mousemove", (evt) => showTip(evt, `<b>${opts.zLabel}: ${fmt(v)}</b><div><span>${opts.xLabel}</span><span>${fmt(x)}</span></div><div><span>${opts.yLabel}</span><span>${fmt(y)}</span></div>`));
    r.addEventListener("mouseleave", hideTip);
  }));
  const every = (n, cnt) => Math.max(1, Math.ceil(n / cnt));
  xs.forEach((x, i) => { if (i % every(xs.length, W / 70) === 0) el("text", { x: m.l + (i + 0.5) * cw, y: H - m.b + 15, "text-anchor": "middle", class: "tick" }, svg).textContent = fmt(x); });
  ys.forEach((y, j) => { if (j % every(ys.length, H / 26) === 0) el("text", { x: m.l - 6, y: H - m.b - (j + 0.5) * ch + 4, "text-anchor": "end", class: "tick" }, svg).textContent = fmt(y); });
  el("text", { x: (m.l + W - m.r) / 2, y: H - 4, "text-anchor": "middle", class: "lab" }, svg).textContent = opts.xLabel;
  el("text", { x: 12, y: (H - m.b) / 2, "text-anchor": "middle", class: "lab", transform: `rotate(-90 12 ${(H - m.b) / 2})` }, svg).textContent = opts.yLabel;
  scaleBar(svg, W - m.r + 24, m.t, H - m.t - m.b, lo, hi, opts.zLabel);
}

function scaleBar(svg, x, y, h, lo, hi, label) {
  const id = `g${Math.random().toString(36).slice(2)}`;
  const defs = el("defs", {}, svg), g = el("linearGradient", { id, x1: 0, y1: 1, x2: 0, y2: 0 }, defs);
  for (let k = 0; k <= 4; k++) el("stop", { offset: k / 4, "stop-color": ramp(k / 4) }, g);
  el("rect", { x, y, width: 12, height: h, rx: 2, fill: `url(#${id})` }, svg);
  el("text", { x: x + 16, y: y + 10, class: "tick" }, svg).textContent = fmt(hi);
  el("text", { x: x + 16, y: y + h, class: "tick" }, svg).textContent = fmt(lo);
  if (label) el("text", { x: x + 16, y: y + h / 2, class: "tick" }, svg).textContent = label;
}

// ------------------------------------------------------------------ track map coloured by a value
// opts: {x, y, v, label, unit, path: {x, y} (optional second line, e.g. the driven path)}
function trackMap(container, opts) {
  container.innerHTML = "";
  container.classList.add("chart");
  const W = Math.max(container.clientWidth || 400, 240), H = opts.height || 320, pad = 12, rightPad = 70;
  const xs = opts.x.concat(opts.path ? opts.path.x : []), ys = opts.y.concat(opts.path ? opts.path.y : []);
  const xmin = Math.min(...xs), xmax = Math.max(...xs), ymin = Math.min(...ys), ymax = Math.max(...ys);
  const k = Math.min((W - 2 * pad - rightPad) / ((xmax - xmin) || 1), (H - 2 * pad) / ((ymax - ymin) || 1));
  const ox = pad + ((W - 2 * pad - rightPad) - k * (xmax - xmin)) / 2, oy = pad + ((H - 2 * pad) - k * (ymax - ymin)) / 2;
  const X = (x) => ox + (x - xmin) * k, Y = (y) => H - oy - (y - ymin) * k;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", height: H, role: "img", "aria-label": `track map, ${opts.label}` }, container);
  const v = opts.v, finite = v.filter((a) => a !== null && isFinite(a));
  const lo = Math.min(...finite), hi = Math.max(...finite);
  el("polyline", { points: opts.x.map((x, i) => `${X(x).toFixed(1)},${Y(opts.y[i]).toFixed(1)}`).join(" "), class: "trackbase" }, svg);
  const n = opts.x.length;
  for (let i = 0; i < n - 1; i++) {
    el("line", { x1: X(opts.x[i]), y1: Y(opts.y[i]), x2: X(opts.x[i + 1]), y2: Y(opts.y[i + 1]),
      stroke: ramp((v[i] - lo) / ((hi - lo) || 1)), "stroke-width": 4, "stroke-linecap": "round" }, svg);
  }
  if (opts.path) el("polyline", { points: opts.path.x.map((x, i) => `${X(x).toFixed(1)},${Y(opts.path.y[i]).toFixed(1)}`).join(" "), class: "drivenpath" }, svg);
  el("circle", { cx: X(opts.x[0]), cy: Y(opts.y[0]), r: 6, class: "startdot" }, svg);
  scaleBar(svg, W - rightPad + 14, pad, H - 2 * pad, lo, hi, opts.unit);
  const hit = el("rect", { x: 0, y: 0, width: W - rightPad, height: H, fill: "transparent" }, svg);
  hit.addEventListener("mousemove", (evt) => {
    const r = svg.getBoundingClientRect(), px = (evt.clientX - r.left) * W / r.width, py = (evt.clientY - r.top) * H / r.height;
    let best = -1, bd = 400;
    for (let i = 0; i < n; i++) { const d = (X(opts.x[i]) - px) ** 2 + (Y(opts.y[i]) - py) ** 2; if (d < bd) { bd = d; best = i; } }
    if (best < 0) return hideTip();
    showTip(evt, `<b>${opts.label}: ${fmt(v[best])} ${opts.unit || ""}</b>${opts.s ? `<div><span>distance</span><span>${fmt(opts.s[best], 0)} m</span></div>` : ""}`);
  });
  hit.addEventListener("mouseleave", hideTip);
}

// ------------------------------------------------------------------ CSV
function downloadCSV(name, header, rows) {
  const q = (v) => (v === null || v === undefined ? "" : typeof v === "string" && /[",\n]/.test(v) ? `"${v.replace(/"/g, '""')}"` : String(v));
  const text = [header.map(q).join(","), ...rows.map((r) => r.map(q).join(","))].join("\n");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type: "text/csv" }));
  a.download = name;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}

window.Charts = { lineChart, barChart, heatmap, trackMap, downloadCSV, fmt, ramp, SERIES };
})();
