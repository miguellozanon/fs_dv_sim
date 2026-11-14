// Performance Lab page logic.
"use strict";

const { lineChart, barChart, heatmap, trackMap, downloadCSV, fmt } = window.Charts;
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const EVENT_LABEL = { acceleration: "Acceleration", skidpad: "Skidpad", autocross: "Autocross", endurance: "Endurance" };
const evLabel = (n) => EVENT_LABEL[n] || n;

const S = {
  info: null, comp: "", changes: [], setupName: "", base: null, cur: null,
  ovEvent: "autocross", cmEvent: "autocross", results: {},
};

// ------------------------------------------------------------------ API
async function api(path, body) {
  const opt = body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json", "X-Perf-Lab": "1" }, body: JSON.stringify(body) };
  const r = await fetch(path, opt);
  let j;
  try { j = await r.json(); } catch { throw new Error(`HTTP ${r.status}`); }
  if (j && j.ok === false) throw new Error(j.message || `HTTP ${r.status}`);
  return j;
}

// a long study: progress bar, cancel, result
async function runJob(boxId, kind, args) {
  const box = $(boxId);
  box.hidden = false; box.classList.remove("err");
  box.innerHTML = `<div class="bar"><i></i></div><div class="txt">starting …</div><button type="button" class="btn small">Cancel</button>`;
  const bar = box.querySelector(".bar i"), txt = box.querySelector(".txt"), btn = box.querySelector("button");
  const { job } = await api("/api/job", { kind, args: { competition: S.comp, ...args } });
  btn.onclick = () => api("/api/job/cancel", { id: job.id }).catch(() => {});
  for (;;) {
    await new Promise((r) => setTimeout(r, 250));
    const j = await api(`/api/job/${job.id}`);
    bar.style.width = `${Math.round(j.progress * 100)}%`;
    txt.textContent = j.state === "queued" ? "waiting for the study before …" : `${j.text}  (${j.elapsed.toFixed(1)} s)`;
    if (j.state === "done") { box.hidden = true; return j.result; }
    if (j.state === "failed" || j.state === "cancelled") {
      box.classList.add("err"); txt.textContent = j.state === "cancelled" ? "cancelled" : j.error; btn.remove();
      return null;
    }
  }
}

function showErr(id, e) { const m = $(id); m.textContent = e.message || String(e); m.classList.add("err"); }
function showMsg(id, text) { const m = $(id); m.textContent = text; m.classList.remove("err"); }

// ------------------------------------------------------------------ navigation
function route() {
  let page = (location.hash || "#overview").slice(1);
  if (page.startsWith("guide-")) {                    // a section of the guide
    const target = page;
    page = "guide";
    setTimeout(() => { const el = document.getElementById(target); if (el) el.scrollIntoView({ block: "start" }); }, 30);
  }
  if (page === "guide") { try { localStorage.setItem("perf-guide-seen", "1"); } catch { /* ignore */ } }
  let found = false;
  document.querySelectorAll(".page").forEach((p) => { p.hidden = p.dataset.page !== page; found ||= !p.hidden; });
  if (!found) { $("page-overview").hidden = false; }
  document.querySelectorAll(".tabs a").forEach((a) => a.toggleAttribute("aria-current", a.getAttribute("href") === `#${found ? page : "overview"}`));
  document.querySelectorAll(".tabs a[aria-current]").forEach((a) => a.setAttribute("aria-current", "page"));
  redrawVisible();
}
window.addEventListener("hashchange", route);
let resizeT;
window.addEventListener("resize", () => { clearTimeout(resizeT); resizeT = setTimeout(redrawVisible, 150); });
const DRAW = {};   // page -> redraw function (charts need the page visible to know their width)
function redrawVisible() {
  const p = document.querySelector(".page:not([hidden])");
  if (p && DRAW[p.dataset.page]) DRAW[p.dataset.page]();
}

// ------------------------------------------------------------------ context bar
async function loadInfo() {
  S.info = await api(`/api/info?competition=${encodeURIComponent(S.comp)}`);
  S.comp = S.info.competition;
  $("ctx-comp").innerHTML = S.info.competitions.map((c) => `<option ${c === S.comp ? "selected" : ""}>${esc(c)}</option>`).join("");
  $("ctx-eta").innerHTML = `<option value="">constant eta_motor</option>` +
    S.info.eta_maps.map((m) => `<option ${m === S.info.eta_map ? "selected" : ""} value="${esc(m)}">${esc(m.replace(/^eta_map_?|\.json$/g, "") || m)}</option>`).join("");
  $("foot-src").textContent = `LapSim: ${S.info.params_file} · ${S.info.competition}`;
  $("su-src").textContent = S.info.params_file;
  const evs = S.info.events.map((e) => e.name);
  for (const id of ["tr-event", "lg-event", "oc-event", "mc-event"]) $(id).innerHTML = evs.map((n) => `<option value="${n}">${evLabel(n)}</option>`).join("");
  $("tr-event").value = evs.includes("autocross") ? "autocross" : evs[0];
  $("lg-event").value = evs.includes("autocross") ? "autocross" : evs[0];
  $("oc-event").value = $("mc-event").value = evs.includes("skidpad") ? "skidpad" : evs[0];
  $("sw-trace").innerHTML = `<option value="">no traces</option>` + evs.map((n) => `<option value="${n}">${evLabel(n)}</option>`).join("");
  $("gg-event").innerHTML = `<option value="">the tyres (utilisation 1)</option>` + evs.map((n) => `<option value="${n}">${evLabel(n)} (driver, endurance limits)</option>`).join("");
  const keys = S.info.groups.flatMap((g) => g.rows.map((r) => r.key));
  $("param-list").innerHTML = keys.concat(["mu", "mass", "power", "cog", "cl", "cd"]).map((k) => `<option value="${k}">`).join("");
  buildSetupPage();
  buildComparePicks();
  buildGuide();
  await buildMethods();
}

$("ctx-comp").addEventListener("change", async () => {
  S.comp = $("ctx-comp").value; S.base = null;
  await loadInfo();
  await runOverview();
});
$("ctx-eta").addEventListener("change", async () => {
  await api("/api/eta_map", { name: $("ctx-eta").value });
  S.base = null;
  await runOverview();
});
$("quit").addEventListener("click", async () => {
  if (!confirm("Stop the Performance Lab?")) return;
  try { await api("/api/quit", {}); } catch { /* already gone */ }
  document.body.innerHTML = `<main><section class="panel"><h2>Performance Lab stopped</h2><p class="hint">Start it again with perf.bat.</p></section></main>`;
});

function setContext() {
  $("ctx-name").textContent = S.setupName || (S.changes.length ? "Modified" : "Baseline");
  $("ctx-changes").textContent = S.changes.length ? S.changes.join("  ") : "lr26_params.m as it is";
  const t = S.cur ? S.cur.total : null, b = S.base ? S.base.total : null;
  $("ctx-total").textContent = t === null ? "–" : `${fmt(t, 2)} P`;
  const d = t !== null && b !== null ? t - b : null;
  $("ctx-delta").textContent = d === null || Math.abs(d) < 0.005 || !S.changes.length ? "" : `${d > 0 ? "+" : ""}${fmt(d, 2)} vs baseline`;
  $("ctx-delta").className = `ctx-delta ${d > 0 ? "up" : "down"}`;
}

// ------------------------------------------------------------------ overview
async function runOverview() {
  try {
    if (!S.base) S.base = (await api("/api/run", { changes: [], competition: S.comp })).result;
    S.cur = S.changes.length ? (await api("/api/run", { changes: S.changes, competition: S.comp })).result : S.base;
  } catch (e) {
    $("ov-note").textContent = e.message; return;
  }
  setContext();
  drawOverview();
}

function drawOverview() {
  const r = S.cur, b = S.base;
  if (!r) return;
  const evs = Object.keys(r.events);
  if (!evs.includes(S.ovEvent)) S.ovEvent = evs[0];
  const delta = (a, c, d = 2, inv = false) => {
    if (!S.changes.length || a === null || c === null) return "";
    const x = a - c; if (Math.abs(x) < Math.pow(10, -d) / 2) return "";
    const good = inv ? x < 0 : x > 0;
    return `<span class="${good ? "up" : "down"}">${x > 0 ? "+" : ""}${fmt(x, d)}</span>`;
  };
  $("ov-kpis").innerHTML = `<dl class="kpi"><dt>Dynamic events</dt><dd>${fmt(r.total, 1)}<small>points</small></dd><div class="sub">${delta(r.total, b.total) || "&nbsp;"}</div></dl>` +
    evs.map((n) => { const e = r.events[n], eb = b.events[n];
      return `<dl class="kpi"><dt>${evLabel(n)}</dt><dd>${fmt(e.T, n === "endurance" ? 1 : 3)}<small>s</small></dd><div class="sub">${fmt(e.points, 2)} P ${delta(e.points, eb.points)}${delta(e.T, eb.T, 3, true) ? ` · Δt ${delta(e.T, eb.T, 3, true)}` : ""}</div></dl>`; }).join("");
  $("ov-table").innerHTML = `<thead><tr><th>Event</th><th>Track</th><th class="num">Time [s]</th><th class="num">Lap [s]</th><th class="num">Laps</th><th class="num">Points</th><th class="num">Δ points</th><th class="num">v avg [km/h]</th><th class="num">v max [km/h]</th><th class="num">Energy [kWh]</th><th class="num">Driver util. x / y</th></tr></thead><tbody>` +
    evs.map((n) => { const e = r.events[n], eb = b.events[n], dp = e.points - eb.points;
      return `<tr><td>${evLabel(n)}</td><td>${esc(e.track)}</td><td class="num">${fmt(e.T, 3)}</td><td class="num">${fmt(e.t_lap, 3)}</td><td class="num">${e.laps}</td><td class="num">${fmt(e.points, 2)}</td><td class="num ${dp > 0.005 ? "pos" : dp < -0.005 ? "neg" : ""}">${S.changes.length ? fmt(dp, 2) : "–"}</td><td class="num">${fmt(e.v_avg_kmh, 1)}</td><td class="num">${fmt(e.v_max_kmh, 1)}</td><td class="num">${fmt(e.E_total_kwh, 3)}</td><td class="num">${e.util.map((u) => fmt(u, 2)).join(" / ")}</td></tr>`; }).join("") +
    `<tr><td><b>Total</b></td><td></td><td></td><td></td><td></td><td class="num"><b>${fmt(r.total, 2)}</b></td><td class="num">${S.changes.length ? fmt(r.total - b.total, 2) : "–"}</td><td colspan="4"></td></tr></tbody>`;
  $("ov-note").textContent = `Quasi-steady-state LapSim (port of LR26_LapSim), ${r.wall_s.toFixed(2)} s. Points = dynamic events only (no efficiency, no static events).`;
  $("ov-event").innerHTML = evs.map((n) => `<button type="button" data-ev="${n}" aria-pressed="${n === S.ovEvent}">${evLabel(n)}</button>`).join("");
  $("ov-event").querySelectorAll("button").forEach((bt) => bt.onclick = () => { S.ovEvent = bt.dataset.ev; drawOverview(); });
  const e = r.events[S.ovEvent].trace, eb = b.events[S.ovEvent].trace, two = S.changes.length > 0;
  const ser = (k, name) => [{ name: two ? "this setup" : name, x: e.s, y: e[k] }].concat(two ? [{ name: "baseline", x: eb.s, y: eb[k], dash: true, color: getComputedStyle(document.documentElement).getPropertyValue("--muted") }] : []);
  lineChart($("ov-v"), { series: ser("v", "speed"), xLabel: "distance [m]", yLabel: "km/h", height: 200 });
  lineChart($("ov-ax"), { series: ser("ax", "ax"), xLabel: "distance [m]", yLabel: "g", height: 150, zero: true });
  lineChart($("ov-ay"), { series: ser("ay", "ay"), xLabel: "distance [m]", yLabel: "g", height: 150, zero: true });
  lineChart($("ov-p"), { series: ser("P_batt", "power"), xLabel: "distance [m]", yLabel: "kW", height: 150, zero: true });
  if (S.ovEvent === "acceleration") $("ov-map").innerHTML = `<p class="empty">a straight line</p>`;
  else trackMap($("ov-map"), { x: e.x, y: e.y, v: e.v, s: e.s, label: "speed", unit: "km/h", height: 360 });
}
DRAW.overview = drawOverview;

$("ov-csv").onclick = () => {
  const r = S.cur; if (!r) return;
  downloadCSV("perf_events.csv", ["event", "time_s", "lap_s", "laps", "points", "points_baseline", "v_avg_kmh", "v_max_kmh", "energy_kWh", "changes"],
    Object.entries(r.events).map(([n, e]) => [n, e.T, e.t_lap, e.laps, e.points, S.base.events[n].points, e.v_avg_kmh, e.v_max_kmh, e.E_total_kwh, S.changes.join(" ")]));
};
$("ov-trace-csv").onclick = () => {
  const e = S.cur && S.cur.events[S.ovEvent].trace; if (!e) return;
  downloadCSV(`perf_${S.ovEvent}_trace.csv`, ["s_m", "t_s", "v_kmh", "ax_g", "ay_g", "P_batt_kW", "E_kWh", "x_m", "y_m", "k_1pm"],
    e.s.map((_, i) => ["s", "t", "v", "ax", "ay", "P_batt", "E", "x", "y", "k"].map((k) => e[k][i])));
};

// ------------------------------------------------------------------ setup
function buildSetupPage() {
  const g = $("su-groups"), keep = currentInputs();
  g.innerHTML = S.info.groups.map((grp, gi) => `<details class="pgroup" ${gi < 2 ? "open" : ""}><summary>${esc(grp.title)}<span class="count" hidden></span></summary>
    <table class="ptable"><tbody>${grp.rows.map((r) => `<tr><td><span class="p-label">${esc(r.label)}</span><span class="p-key">${esc(r.key)} · ${esc(r.desc)}</span></td>
    <td class="base">${r.value === null ? "–" : fmt(r.value)}</td><td class="unit">${esc(r.unit)}</td>
    <td class="inp"><input class="p-in" data-key="${esc(r.key)}" placeholder="${r.value === null ? "" : r.value}" spellcheck="false" aria-label="${esc(r.label)}"></td></tr>`).join("")}</tbody></table></details>`).join("") +
    `<details class="pgroup"><summary>Transient lap only (simulator)</summary><table class="ptable"><tbody>
     ${[["yaw_inertia", "Yaw inertia", "kg m²"], ["actuators.max_steer", "Max steering angle", "rad"], ["actuators.max_accel", "Max acceleration command", "m/s²"], ["actuators.max_decel", "Max deceleration command", "m/s²"]]
       .map(([k, l, u]) => `<tr><td><span class="p-label">${l}</span><span class="p-key">${k} · not in the QSS LapSim</span></td><td class="base">sim</td><td class="unit">${u}</td><td class="inp"><input class="p-in" data-key="${k}" spellcheck="false" aria-label="${l}"></td></tr>`).join("")}
     </tbody></table></details>`;
  g.querySelectorAll(".p-in").forEach((inp) => { if (keep[inp.dataset.key]) inp.value = keep[inp.dataset.key]; inp.addEventListener("input", setupChanged); });
  setupChanged();
  const saved = S.info.setups;
  $("su-saved").innerHTML = saved.length ? saved.map((s, i) => `<li><span class="nm">${esc(s.name)}</span><span><button type="button" class="btn small" data-load="${i}">Load</button></span>
    <span class="ch">${s.changes.length ? esc(s.changes.join("  ")) : "no changes"}${s.notes ? ` · ${esc(s.notes)}` : ""}</span></li>`).join("") : `<li class="muted">none yet</li>`;
  $("su-saved").querySelectorAll("[data-load]").forEach((b) => b.onclick = () => loadSetup(saved[+b.dataset.load]));
}

function currentInputs() {
  const o = {};
  document.querySelectorAll("#su-groups .p-in").forEach((i) => { if (i.value.trim()) o[i.dataset.key] = i.value.trim(); });
  return o;
}

function inputToChange(key, v) {
  // "1.5" or "=1.5" -> absolute value (also negative numbers); "*1.05" / "+10" / "+-5" -> relative
  v = v.trim();
  if (!v) return null;
  if (/^[*+]/.test(v)) return `${key}${v}`;
  return `${key}=${v.replace(/^=/, "")}`;
}

function pendingChanges() {
  const out = [];
  document.querySelectorAll("#su-groups .p-in").forEach((i) => { const c = inputToChange(i.dataset.key, i.value); if (c) out.push(c); });
  return out.concat($("su-extra").value.split(/[\s,]+/).map((x) => x.trim()).filter(Boolean));
}

function setupChanged() {
  document.querySelectorAll("#su-groups .p-in").forEach((i) => i.classList.toggle("changed", !!i.value.trim()));
  document.querySelectorAll("#su-groups details").forEach((d) => {
    const n = [...d.querySelectorAll(".p-in")].filter((i) => i.value.trim()).length, c = d.querySelector(".count");
    if (c) { c.hidden = !n; c.textContent = n; }
  });
  const ch = pendingChanges();
  $("su-list").innerHTML = ch.length ? ch.map((c) => `<code>${esc(c)}</code>`).join("") : `<span class="muted">no changes: the baseline</span>`;
}
$("su-extra").addEventListener("input", setupChanged);

function loadSetup(s) {
  document.querySelectorAll("#su-groups .p-in").forEach((i) => { i.value = ""; });
  const extra = [];
  for (const c of s.changes) {
    const m = c.match(/^\s*([A-Za-z_][\w.]*)\s*([=*+-])\s*(.+)$/);
    const inp = m && m[2] !== "-" && document.querySelector(`#su-groups .p-in[data-key="${CSS.escape(m[1])}"]`);
    if (inp && !inp.value) inp.value = m[2] === "=" ? m[3] : `${m[2]}${m[3]}`;
    else extra.push(c);
  }
  $("su-extra").value = extra.join("\n");
  $("su-name").value = s.name; $("su-notes").value = s.notes || "";
  setupChanged();
  S.loaded = { name: s.name, changes: pendingChanges().join("|") };
  applySetup();
}

async function applySetup() {
  const ch = pendingChanges();
  const name = S.loaded && S.loaded.changes === ch.join("|") ? S.loaded.name : "";
  try {
    const r = await api("/api/validate", { changes: ch, competition: S.comp });
    S.changes = r.changes; S.setupName = name || "";
    showMsg("su-msg", ch.length ? `${ch.length} change(s) applied` : "baseline");
    await runOverview();
  } catch (e) { showErr("su-msg", e); }
}
$("su-apply").onclick = () => applySetup();
$("su-base").onclick = () => { document.querySelectorAll("#su-groups .p-in").forEach((i) => { i.value = ""; }); $("su-extra").value = ""; setupChanged(); applySetup(); };
$("su-clear").onclick = () => { document.querySelectorAll("#su-groups .p-in").forEach((i) => { i.value = ""; }); $("su-extra").value = ""; setupChanged(); };
$("su-save").onclick = async () => {
  try {
    const r = await api("/api/setup/save", { name: $("su-name").value, notes: $("su-notes").value, changes: pendingChanges(), overwrite: $("su-overwrite").checked, competition: S.comp });
    showMsg("su-save-msg", r.message);
    S.info.setups = r.setups;
    buildSetupPage(); buildComparePicks();
  } catch (e) { showErr("su-save-msg", e); }
};

// ------------------------------------------------------------------ studies
document.querySelectorAll("#st-mode button").forEach((b) => b.onclick = () => {
  document.querySelectorAll("#st-mode button").forEach((x) => x.setAttribute("aria-pressed", x === b));
  document.querySelectorAll(".panel.study").forEach((p) => { p.hidden = p.dataset.mode !== b.dataset.mode; });
  DRAW.studies();
});
$("se-params").value = "mu m i_trans h_cog c_z c_w P_max aero_bal_f T_motor A";

$("se-run").onclick = async () => {
  const params = $("se-params").value.split(/[\s,]+/).filter(Boolean);
  const r = await runJob("se-job", "sensitivity", { params, delta: +$("se-delta").value / 100, changes: S.changes });
  if (!r) return;
  S.results.sens = r; $("se-out").hidden = false; drawSens();
};
function drawSens() {
  const r = S.results.sens; if (!r || $("se-out").hidden) return;
  const d = Math.round(r.delta * 100);
  barChart($("se-bars"), {
    series: [{ name: `+${d} %` }, { name: `−${d} %` }],
    rows: r.rows.map((x) => ({ label: x.param, values: [x.total_plus - r.baseline.total, x.total_minus - r.baseline.total],
      note: x.no_effect ? "no effect (value 0 or not used)" : `${fmt(x.points_per_pct, 3)} P/% (${x.better})` })),
    xLabel: "points against the active setup", labelWidth: 120,
  });
  const evs = Object.keys(r.baseline.T);
  $("se-table").innerHTML = `<thead><tr><th>Parameter</th><th class="num">Value</th><th class="num">P / %</th>${evs.map((n) => `<th class="num">${evLabel(n)} Δt +${d} % [s]</th><th class="num">Δt −${d} % [s]</th>`).join("")}</tr></thead><tbody>` +
    r.rows.map((x) => `<tr><td>${esc(x.param)}${x.points_per_kg !== undefined ? ` <span class="muted">(${fmt(x.points_per_kg, 3)} P/kg, ${fmt(x.autocross_s_per_kg, 4)} s/kg AutoX)</span>` : ""}</td><td class="num">${fmt(x.value)}</td><td class="num">${fmt(x.points_per_pct, 3)}</td>${evs.map((n) => `<td class="num">${fmt(x.dT_plus[n], 3)}</td><td class="num">${fmt(x.dT_minus[n], 3)}</td>`).join("")}</tr>`).join("") + "</tbody>";
}
$("se-csv").onclick = () => {
  const r = S.results.sens; if (!r) return;
  const evs = Object.keys(r.baseline.T);
  downloadCSV("sensitivity.csv", ["param", "value", "delta", "total_plus", "total_minus", "points_per_pct", ...evs.flatMap((n) => [`${n}_dT_plus`, `${n}_dT_minus`, `${n}_dP_plus`, `${n}_dP_minus`])],
    r.rows.map((x) => [x.param, x.value, r.delta, x.total_plus, x.total_minus, x.points_per_pct, ...evs.flatMap((n) => [x.dT_plus[n], x.dT_minus[n], x.dP_plus[n], x.dP_minus[n]])]));
};

$("sw-run").onclick = async () => {
  const r = await runJob("sw-job", "sweep", { param: $("sw-param").value.trim(), values: { from: +$("sw-from").value, to: +$("sw-to").value, n: +$("sw-n").value },
    changes: S.changes, trace_event: $("sw-trace").value });
  if (!r) return;
  S.results.sweep = r; $("sw-out").hidden = false;
  const evs = Object.keys(r.rows[0].T);
  $("sw-tev").innerHTML = evs.map((n) => `<option value="${n}">${evLabel(n)}</option>`).join("");
  $("sw-tev").value = evs.includes("autocross") ? "autocross" : evs[0];
  drawSweep();
};
$("sw-tev").onchange = () => drawSweep();
function drawSweep() {
  const r = S.results.sweep; if (!r || $("sw-out").hidden) return;
  const xs = r.rows.map((x) => x.value), evs = Object.keys(r.rows[0].T), lab = r.param;
  lineChart($("sw-total"), { series: [{ name: "total", x: xs, y: r.rows.map((x) => x.total) }], xLabel: lab, yLabel: "points", markers: true, mark: r.base_value, height: 220 });
  lineChart($("sw-events"), { series: evs.map((n) => ({ name: evLabel(n), x: xs, y: r.rows.map((x) => x.points[n]) })), xLabel: lab, yLabel: "points", markers: true, mark: r.base_value, height: 220 });
  const tev = $("sw-tev").value;
  lineChart($("sw-time"), { series: [{ name: evLabel(tev), x: xs, y: r.rows.map((x) => x.T[tev]) }], xLabel: lab, yLabel: "s", markers: true, mark: r.base_value, height: 200 });
  const tr = r.rows.filter((x) => x.trace);
  $("sw-trace-box").hidden = !tr.length;
  if (tr.length) {
    const n = tr.length;
    lineChart($("sw-traces"), { series: tr.map((x, i) => ({ name: `${lab} = ${fmt(x.value)}`, x: x.trace.s, y: x.trace.v, color: window.Charts.ramp(n > 1 ? i / (n - 1) : 1), width: 1.5 })),
      xLabel: "distance [m]", yLabel: "km/h", height: 200, legend: n <= 6,
      caption: `light → dark: ${lab} ${fmt(tr[0].value)} → ${fmt(tr[n - 1].value)}` });
  }
  $("sw-table").innerHTML = `<thead><tr><th class="num">${esc(lab)}</th><th class="num">Total</th>${evs.map((n) => `<th class="num">${evLabel(n)} [s]</th><th class="num">P</th>`).join("")}</tr></thead><tbody>` +
    r.rows.map((x) => `<tr class="${Math.abs(x.value - r.base_value) < 1e-9 ? "sel" : ""}"><td class="num">${fmt(x.value)}</td><td class="num">${fmt(x.total, 2)}</td>${evs.map((n) => `<td class="num">${fmt(x.T[n], 3)}</td><td class="num">${fmt(x.points[n], 2)}</td>`).join("")}</tr>`).join("") + "</tbody>";
}
$("sw-csv").onclick = () => {
  const r = S.results.sweep; if (!r) return;
  const evs = Object.keys(r.rows[0].T);
  downloadCSV(`sweep_${r.param}.csv`, [r.param, "total", ...evs.flatMap((n) => [`${n}_T_s`, `${n}_points`, `${n}_E_kWh`])],
    r.rows.map((x) => [x.value, x.total, ...evs.flatMap((n) => [x.T[n], x.points[n], x.E_kwh[n]])]));
};

$("m2-run").onclick = async () => {
  const r = await runJob("m2-job", "sweep2d", {
    p1: $("m2-p1").value.trim(), v1: { from: +$("m2-f1").value, to: +$("m2-t1").value, n: +$("m2-n1").value },
    p2: $("m2-p2").value.trim(), v2: { from: +$("m2-f2").value, to: +$("m2-t2").value, n: +$("m2-n2").value }, changes: S.changes });
  if (!r) return;
  S.results.map2 = r; $("m2-out").hidden = false;
  $("m2-show").innerHTML = `<option value="">total</option>` + Object.keys(r.points).map((n) => `<option value="${n}">${evLabel(n)}</option>`).join("");
  drawMap2();
};
$("m2-show").onchange = () => drawMap2();
function drawMap2() {
  const r = S.results.map2; if (!r || $("m2-out").hidden) return;
  const k = $("m2-show").value, z = k ? r.points[k] : r.total;
  heatmap($("m2-heat"), { xs: r.v1, ys: r.v2, z, xLabel: r.p1, yLabel: r.p2, zLabel: k ? `${evLabel(k)} points` : "total points", best: k ? null : { x: r.best.v1, y: r.best.v2 }, height: 360 });
  $("m2-best").textContent = `best: ${r.p1} = ${fmt(r.best.v1)}, ${r.p2} = ${fmt(r.best.v2)} → ${fmt(r.best.total, 2)} P (active setup: ${r.p1} = ${fmt(r.base_values[0])}, ${r.p2} = ${fmt(r.base_values[1])})`;
}
$("m2-csv").onclick = () => {
  const r = S.results.map2; if (!r) return;
  const rows = [];
  r.v1.forEach((a, i) => r.v2.forEach((b, j) => rows.push([a, b, r.total[i][j], ...Object.keys(r.points).map((n) => r.points[n][i][j])])));
  downloadCSV(`map_${r.p1}_${r.p2}.csv`, [r.p1, r.p2, "total", ...Object.keys(r.points)], rows);
};

function buildComparePicks() {
  const s = S.info.setups;
  $("cm-pick").innerHTML = `<li><label class="check"><input type="checkbox" data-i="-1" checked> Active setup</label></li>` +
    s.map((x, i) => `<li><label class="check"><input type="checkbox" data-i="${i}"> ${esc(x.name)}</label></li>`).join("");
}
$("cm-run").onclick = async () => {
  const picks = [...document.querySelectorAll("#cm-pick input:checked")].map((i) => +i.dataset.i);
  const setups = picks.map((i) => (i < 0 ? { name: S.setupName || (S.changes.length ? "Active setup" : "Baseline"), changes: S.changes } : S.info.setups[i]));
  if (setups.length < 2) { alert("Choose at least two setups"); return; }
  if (setups.length > 8) { alert("At most eight setups"); return; }
  const r = await runJob("cm-job", "compare", { setups });
  if (!r) return;
  S.results.cmp = r; $("cm-out").hidden = false; drawCompare();
};
function drawCompare() {
  const r = S.results.cmp; if (!r || $("cm-out").hidden) return;
  const evs = Object.keys(r.setups[0].T), b = r.setups[0];
  if (!evs.includes(S.cmEvent)) S.cmEvent = evs[0];
  $("cm-table").innerHTML = `<thead><tr><th>Setup</th><th class="num">Total</th><th class="num">Δ</th>${evs.map((n) => `<th class="num">${evLabel(n)} [s]</th><th class="num">P</th>`).join("")}<th>Changes</th></tr></thead><tbody>` +
    r.setups.map((x, i) => `<tr><td><i class="sw" style="background:${window.Charts.SERIES()[i % 6]}"></i>${esc(x.name)}</td><td class="num">${fmt(x.total, 2)}</td><td class="num ${x.total - b.total > 0.005 ? "pos" : x.total - b.total < -0.005 ? "neg" : ""}">${i ? fmt(x.total - b.total, 2) : "–"}</td>${evs.map((n) => `<td class="num">${fmt(x.T[n], 3)}</td><td class="num">${fmt(x.points[n], 2)}</td>`).join("")}<td class="muted">${esc(x.changes.join(" ") || "–")}</td></tr>`).join("") + "</tbody>";
  $("cm-event").innerHTML = evs.map((n) => `<button type="button" data-ev="${n}" aria-pressed="${n === S.cmEvent}">${evLabel(n)}</button>`).join("");
  $("cm-event").querySelectorAll("button").forEach((bt) => bt.onclick = () => { S.cmEvent = bt.dataset.ev; drawCompare(); });
  lineChart($("cm-v"), { series: r.setups.map((x) => ({ name: x.name, x: x.traces[S.cmEvent].s, y: x.traces[S.cmEvent].v })), xLabel: "distance [m]", yLabel: "km/h", height: 240 });
  const t0 = b.traces[S.cmEvent];
  const interp = (xs, ys, x) => { let i = 0; while (i < xs.length - 2 && xs[i + 1] < x) i++; const f = (x - xs[i]) / ((xs[i + 1] - xs[i]) || 1); return ys[i] + f * (ys[i + 1] - ys[i]); };
  lineChart($("cm-dt"), { series: r.setups.map((x) => ({ name: x.name, x: x.traces[S.cmEvent].s, y: x.traces[S.cmEvent].s.map((s, i) => x.traces[S.cmEvent].t[i] - interp(t0.s, t0.t, s)) })),
    xLabel: "distance [m]", yLabel: "s", height: 240, zero: true });
}
$("cm-csv").onclick = () => {
  const r = S.results.cmp; if (!r) return;
  const evs = Object.keys(r.setups[0].T);
  downloadCSV("compare.csv", ["setup", "changes", "total", ...evs.flatMap((n) => [`${n}_T_s`, `${n}_points`, `${n}_E_kWh`])],
    r.setups.map((x) => [x.name, x.changes.join(" "), x.total, ...evs.flatMap((n) => [x.T[n], x.points[n], x.E_kwh[n]])]));
};
DRAW.studies = () => { drawSens(); drawSweep(); drawMap2(); drawCompare(); };

// ------------------------------------------------------------------ transient
$("tr-run").onclick = async () => {
  const r = await runJob("tr-job", "transient", { event: $("tr-event").value, dt: +$("tr-dt").value, track_width: +$("tr-w").value, changes: S.changes });
  if (!r) return;
  S.results.tr = r; $("tr-out").hidden = false; drawTransient();
};
function drawTransient() {
  const r = S.results.tr; if (!r || $("tr-out").hidden) return;
  const q = r.qss, dT = r.T - q.T, timed = r.closed ? "flying lap" : "run";
  const kpi = (t, v, u, sub, warn) => `<dl class="kpi${warn ? " warn" : ""}"><dt>${t}</dt><dd>${v}<small>${u}</small></dd><div class="sub">${sub}</div></dl>`;
  $("tr-kpis").innerHTML =
    kpi(`${evLabel(r.event)} time`, fmt(r.T, 3), "s", `QSS ${fmt(q.T, 3)} s · ${isFinite(dT) ? `+${fmt(dT, 3)} s (${fmt(100 * dT / q.T, 1)} %)` : r.status}`, r.status !== "finished") +
    kpi("Points", fmt(r.points, 2), "P", `QSS ${fmt(q.points, 2)} P`) +
    kpi(`Time of the ${timed}`, fmt(r.t_lap, 3), "s", `QSS ${fmt(q.t_lap, 3)} s`) +
    kpi("Top speed", fmt(r.v_max_kmh, 1), "km/h", `QSS ${fmt(q.v_max_kmh, 1)} km/h`) +
    kpi("Max sideslip", fmt(r.beta_max, 1), "°", `max lateral offset ${fmt(r.lat_max, 2)} m`, r.off_track_s > 0) +
    kpi("Off the track", fmt(r.off_track_s, 1), "s", `more than half the track width (${fmt($("tr-w").value)} m) from the line`, r.off_track_s > 0);
  const t = r.trace, lap = r.lap;
  lineChart($("tr-v"), { series: [{ name: "transient", x: r.closed ? lap.s : t.s, y: r.closed ? lap.v : t.v }, { name: "QSS LapSim", x: q.trace.s, y: q.trace.v, dash: true }], xLabel: "distance [m]", yLabel: "km/h", height: 220 });
  lineChart($("tr-dt-chart"), { series: [{ name: "transient − QSS", x: lap.s, y: lap.dt }], xLabel: "distance [m]", yLabel: "s", height: 160, zero: true });
  trackMap($("tr-map"), { x: r.track.x, y: r.track.y, v: interpOnTrack(r), s: null, label: "speed", unit: "km/h", height: 400, path: { x: t.x, y: t.y } });
  lineChart($("tr-steer"), { series: [{ name: "steering", x: t.s, y: t.steer }], xLabel: "distance [m]", yLabel: "°", height: 160, zero: true });
  lineChart($("tr-beta"), { series: [{ name: "sideslip", x: t.s, y: t.beta }], xLabel: "distance [m]", yLabel: "°", height: 160, zero: true });
  lineChart($("tr-lat"), { series: [{ name: "offset (left +)", x: t.s, y: t.lat }], xLabel: "distance [m]", yLabel: "m", height: 160, zero: true });
  lineChart($("tr-ay"), { series: [{ name: "ay", x: t.s, y: t.ay }], xLabel: "distance [m]", yLabel: "g", height: 160, zero: true });
  $("tr-note").innerHTML = `Driver: grip utilisation ${fmt(r.driver.grip_util, 2)} (the event's util_y), longitudinal ${fmt(r.driver.long_util, 2)}; vehicle model ${esc(r.driver.handling)}, yaw inertia ${fmt(r.driver.yaw_inertia)} kg m², steering limit ${fmt(r.driver.max_steer_deg, 1)}°; time step ${r.dt} s; ${fmt(r.wall_s, 1)} s to compute.` +
    (r.closure_gap_m > 0 ? ` The track's x, y did not close (${fmt(r.closure_gap_m, 1)} m gap, a curvature-only track): the gap was spread along the lap.` : "") +
    (r.closed ? " Distance of the charts: the timed (second) lap." : "");
}
function interpOnTrack(r) {
  // speed of the transient lap at the track points (nearest recorded sample on distance)
  const L = r.L, lap = r.closed ? r.lap : { s: r.trace.s, v: r.trace.v };
  const n = r.track.x.length, out = [];
  let acc = 0;
  for (let i = 0; i < n; i++) {
    if (i) acc += Math.hypot(r.track.x[i] - r.track.x[i - 1], r.track.y[i] - r.track.y[i - 1]);
    let k = 0, best = Infinity;
    for (let j = 0; j < lap.s.length; j += 1) { const d = Math.abs(lap.s[j] - Math.min(acc, L)); if (d < best) { best = d; k = j; } }
    out.push(lap.v[k]);
  }
  return out;
}
$("tr-csv").onclick = () => {
  const r = S.results.tr; if (!r) return;
  const t = r.trace, keys = ["t", "s", "v", "ax", "ay", "steer", "r", "beta", "lat", "E", "P", "acc_cmd", "x", "y"];
  downloadCSV(`transient_${r.event}.csv`, ["t_s", "s_m", "v_kmh", "ax_g", "ay_g", "steer_deg", "yaw_rate_degps", "beta_deg", "lateral_offset_m", "E_kWh", "P_kW", "accel_cmd_mps2", "x_m", "y_m"],
    t.t.map((_, i) => keys.map((k) => t[k][i])));
};
DRAW.transient = drawTransient;

// ------------------------------------------------------------------ methods
async function buildMethods() {
  if (!S.methods) S.methods = await api("/api/methods");
  const M = S.methods;
  $("me-ocp").innerHTML = M.ocp.available ? `Optimal control: CasADi ${esc(M.ocp.detail)} found.`
    : `<b>Optimal control is not available</b>: ${esc(M.ocp.detail)}. The launcher's <b>Repair</b> installs CasADi.`;
  $("me-guide").innerHTML = `<thead><tr><th>Question</th><th>Method</th><th>Why</th></tr></thead><tbody>` +
    M.guidance.map((g) => `<tr><td>${esc(g.question)}</td><td><b>${esc(g.method)}</b></td><td class="muted">${esc(g.why)}</td></tr>`).join("") + "</tbody>";
  $("mc-methods").innerHTML = M.methods.map((m) => {
    const off = m.name.startsWith("ocp_") && !M.ocp.available;
    return `<label class="check" title="${esc(m.description)}"><input type="checkbox" value="${m.name}" ${M.default.includes(m.name) && !off ? "checked" : ""} ${off ? "disabled" : ""}> ${esc(m.label)}</label>`;
  }).join("");
  $("mo-unc").innerHTML = Object.entries(M.uncertainty).map(([k, v]) =>
    `<label class="field">σ ${esc(k)} [%] <input type="number" data-k="${k}" value="${+(100 * v).toFixed(2)}" min="0" max="50" step="0.5" size="4"></label>`).join("");
  for (const id of ["oc-run", "oc-tv", "oc-free"]) $(id).disabled = !M.ocp.available;
}
const kpiBox = (t, v, u, sub, warn) => `<dl class="kpi${warn ? " warn" : ""}"><dt>${t}</dt><dd>${v}<small>${u}</small></dd><div class="sub">${sub}</div></dl>`;
$("oc-run").onclick = async () => {
  const r = await runJob("oc-job", "ocp", { event: $("oc-event").value, model: $("oc-model").value, torque_vectoring: $("oc-tv").checked,
    fixed_line: !$("oc-free").checked, track_width: +$("oc-w").value, changes: S.changes });
  if (!r) return;
  S.results.oc = r; $("oc-out").hidden = false; drawOcp();
};
function drawOcp() {
  const r = S.results.oc; if (!r || $("oc-out").hidden) return;
  const q = r.qss, dT = r.ok ? r.T - q.T : NaN;
  $("oc-kpis").innerHTML =
    kpiBox(`${evLabel(r.event)} time`, r.ok ? fmt(r.T, 3) : "–", "s", r.ok ? `QSS ${fmt(q.T, 3)} s · ${dT >= 0 ? "+" : ""}${fmt(dT, 3)} s (${fmt(100 * dT / q.T, 2)} %)` : "did not converge", !r.ok) +
    kpiBox("Points", r.ok ? fmt(r.points, 2) : "–", "P", `QSS ${fmt(q.points, 2)} P`) +
    kpiBox(r.laps > 1 ? "Lap time" : "Run time", r.ok ? fmt(r.t_lap, 3) : "–", "s", `QSS on the same 1 m grid ${fmt(q.t_lap_same_grid, 3)} s`) +
    kpiBox("Top speed", fmt(r.v_max_kmh, 1), "km/h", `${r.nodes} points, ${fmt(r.ds, 2)} m`) +
    kpiBox("Compute time", fmt(r.wall_s, 1), "s", r.stages.map((x) => `${x.model.replace("_", " ")} ${x.iterations} it.`).join(" → "));
  $("oc-warn").hidden = !r.warnings.length; $("oc-warn").textContent = r.warnings.join(" ");
  const t = r.trace;
  lineChart($("oc-v"), { series: [{ name: r.model.replace("_", " ") + (r.torque_vectoring ? " + TV" : ""), x: t.s, y: t.v }], xLabel: "distance [m]", yLabel: "km/h", height: 220 });
  lineChart($("oc-a"), { series: [{ name: "ax", x: t.s, y: t.ax }, { name: "ay", x: t.s, y: t.ay }], xLabel: "distance [m]", yLabel: "g", height: 180, zero: true });
  trackMap($("oc-map"), { x: t.x, y: t.y, v: t.v, s: null, label: "speed", unit: "km/h", height: 400, path: r.fixed_line ? null : { x: r.centre.x, y: r.centre.y } });
  $("oc-dyn").hidden = !t.delta;
  if (t.delta) {
    lineChart($("oc-steer"), { series: [{ name: "steering (wheel)", x: t.s, y: t.delta }], xLabel: "distance [m]", yLabel: "°", height: 160, zero: true });
    lineChart($("oc-beta"), { series: [{ name: "sideslip", x: t.s, y: t.beta }], xLabel: "distance [m]", yLabel: "°", height: 160, zero: true });
  }
  const e = r.envelope;
  $("oc-note").textContent = `Line: ${r.fixed_line ? "fixed (the event's line)" : `free within ${fmt(r.track_width, 1)} m (±${fmt(r.half_width, 2)} m for the car's centre)`}. ` +
    `Driver utilisation ${r.util.map((u) => fmt(u, 2)).join(" / ")} (x / y). ggV envelope fit: brake exponent ${fmt(e.p_brake, 2)}, drive ${fmt(e.p_traction, 2)}. ` +
    `Stages: ${r.stages.map((x) => `${x.model} ${x.status} (${fmt(x.solve_s, 1)} s)`).join(", ")}.`;
}
$("oc-csv").onclick = () => {
  const r = S.results.oc; if (!r) return;
  const t = r.trace, keys = ["s", "t", "v", "n", "ax", "ay", "x", "y"].concat(t.delta ? ["delta", "beta", "r"] : []);
  downloadCSV(`ocp_${r.event}_${r.model}.csv`, keys, t.s.map((_, i) => keys.map((k) => t[k][i])));
};
$("mc-run").onclick = async () => {
  const ms = [...document.querySelectorAll("#mc-methods input:checked")].map((x) => x.value);
  if (!ms.length) return;
  const r = await runJob("mc-job", "methods", { event: $("mc-event").value, change: $("mc-change").value.trim(), methods: ms, changes: S.changes });
  if (!r) return;
  S.results.mc = r; $("mc-out").hidden = false; drawMethods();
};
function drawMethods() {
  const r = S.results.mc; if (!r || $("mc-out").hidden) return;
  const ch = !!r.change;
  $("mc-title").textContent = `${evLabel(r.event)}${ch ? `: baseline and ${r.change}` : ""}`;
  $("mc-table").innerHTML = `<thead><tr><th>Method</th><th class="num">Time [s]</th><th class="num">vs QSS</th><th class="num">Points</th>${ch ? `<th class="num">Δ time [s]</th><th class="num">Δ points</th>` : ""}<th class="num">Compute [s]</th><th>Notes</th></tr></thead><tbody>` +
    r.methods.map((m) => `<tr><td title="${esc(m.description)}">${esc(m.label)}</td><td class="num">${m.base.ok ? fmt(m.base.T, 3) : "failed"}</td><td class="num">${m.vs_qss == null ? "–" : (m.vs_qss >= 0 ? "+" : "") + fmt(100 * m.vs_qss, 2) + " %"}</td><td class="num">${m.base.ok ? fmt(m.base.points, 2) : "–"}</td>` +
      (ch ? `<td class="num">${m.dT == null ? "–" : fmt(m.dT, 4)}</td><td class="num">${m.dpoints == null ? "–" : fmt(m.dpoints, 3)}</td>` : "") +
      `<td class="num">${fmt(m.wall_s, 1)}</td><td class="muted">${esc(m.family)}${m.base.ok ? "" : " · " + esc(m.base.note)}</td></tr>`).join("") + "</tbody>";
  const ok = r.methods.filter((m) => m.base.ok);
  if (ch) barChart($("mc-dt"), { series: [{ name: "Δ time" }], rows: r.methods.filter((m) => m.dT != null).map((m) => ({ label: m.label.replace("Optimal control: ", "OCP "), values: [m.dT] })), labelWidth: 200, valFmt: (v) => fmt(v, 4), xLabel: "s (negative = faster)" });
  else $("mc-dt").innerHTML = `<p class="hint">No change given.</p>`;
  barChart($("mc-rel"), { series: [{ name: "vs QSS" }], rows: ok.map((m) => ({ label: m.label.replace("Optimal control: ", "OCP "), values: [100 * (m.vs_qss || 0)] })), labelWidth: 200, valFmt: (v) => fmt(v, 2) + " %", xLabel: "% (positive = slower than QSS)" });
  $("mc-note").textContent = ch ? (r.agree ? "All methods agree on the sign of the change." : "The methods do NOT agree on the sign of the change: look at what the change acts on (see the guidance table).") : "";
}
$("mc-csv").onclick = () => {
  const r = S.results.mc; if (!r) return;
  downloadCSV(`methods_${r.event}.csv`, ["method", "T_base_s", "points_base", "T_change_s", "points_change", "dT_s", "dpoints", "vs_qss", "compute_s"],
    r.methods.map((m) => [m.method, m.base.T, m.base.points, m.change ? m.change.T : "", m.change ? m.change.points : "", m.dT, m.dpoints, m.vs_qss, m.wall_s]));
};
$("mo-run").onclick = async () => {
  const unc = {}; document.querySelectorAll("#mo-unc input").forEach((x) => { unc[x.dataset.k] = (+x.value || 0) / 100; });
  const r = await runJob("mo-job", "monte_carlo", { n: +$("mo-n").value, change: $("mo-change").value.trim(), uncertainty: unc, seed: +$("mo-seed").value, changes: S.changes });
  if (!r) return;
  S.results.mo = r; $("mo-out").hidden = false; drawMonte();
};
function hist(xs, bins = Math.max(8, Math.min(30, Math.round(Math.sqrt(xs.length))))) {
  const lo = Math.min(...xs), hi = Math.max(...xs), w = (hi - lo) / bins || 1, c = new Array(bins).fill(0);
  for (const x of xs) c[Math.min(bins - 1, Math.floor((x - lo) / w))]++;
  return { x: c.map((_, i) => lo + (i + 0.5) * w), y: c.map((v) => v / xs.length * 100) };
}
function drawMonte() {
  const r = S.results.mo; if (!r || $("mo-out").hidden) return;
  const t = r.total, g = r.gain;
  $("mo-kpis").innerHTML = kpiBox("Total points", fmt(t.mean, 1), "P", `nominal ${fmt(r.nominal, 1)} · 90 % in ${fmt(t.p5, 1)} … ${fmt(t.p95, 1)}`) +
    kpiBox("Spread (1σ)", fmt(t.sd, 1), "P", `${r.n} draws`) +
    (g ? kpiBox("Gain of the change", fmt(g.mean, 2), "P", `nominal ${fmt(g.nominal, 2)} · 90 % in ${fmt(g.p5, 2)} … ${fmt(g.p95, 2)}`) +
      kpiBox("Probability it gains", fmt(100 * g.p_gain, 0), "%", "on the same draws (paired)", g.p_gain < 0.9) : "") +
    kpiBox("Compute time", fmt(r.wall_s, 0), "s", "");
  const h = hist(r.samples);
  lineChart($("mo-hist"), { series: [{ name: "share of draws", x: h.x, y: h.y }], xLabel: "total points", yLabel: "%", height: 200, legend: false });
  if (g) { const hg = hist(g.samples); lineChart($("mo-ghist"), { series: [{ name: "share of draws", x: hg.x, y: hg.y }], xLabel: "points gained", yLabel: "%", height: 200, legend: false }); }
  else $("mo-ghist").innerHTML = `<p class="hint">No change given.</p>`;
  $("mo-table").innerHTML = `<thead><tr><th>Event</th><th class="num">Mean</th><th class="num">σ</th><th class="num">5 %</th><th class="num">95 %</th></tr></thead><tbody>` +
    Object.entries(r.events).map(([n, e]) => `<tr><td>${evLabel(n)}</td><td class="num">${fmt(e.mean, 2)}</td><td class="num">${fmt(e.sd, 2)}</td><td class="num">${fmt(e.p5, 2)}</td><td class="num">${fmt(e.p95, 2)}</td></tr>`).join("") + "</tbody>";
  $("mo-note").textContent = `σ used: ${Object.entries(r.uncertainty).map(([k, v]) => `${k} ${fmt(100 * v, 1)} %`).join(", ")}. util_y is one factor on every event's driver utilisation. Points are relative to the competition's reference times, so the spread includes the scoring's sensitivity.`;
}
DRAW.methods = () => { drawOcp(); drawMethods(); drawMonte(); };

// ------------------------------------------------------------------ energy
$("en-run").onclick = async () => {
  const caps = $("en-caps").value.trim() ? $("en-caps").value.split(/[\s,;]+/).filter(Boolean).map(Number) : null;
  if (caps && caps.some((c) => !isFinite(c) || c <= 0)) { alert("power caps: positive numbers in kW"); return; }
  const r = await runJob("en-job", "energy", { changes: S.changes, budget_kwh: +$("en-budget").value || null, caps_kw: caps });
  if (!r) return;
  S.results.en = r; $("en-out").hidden = false; drawEnergy();
};
function drawEnergy() {
  const r = S.results.en; if (!r || $("en-out").hidden) return;
  const kpi = (t, v, u, sub) => `<dl class="kpi"><dt>${t}</dt><dd>${v}<small>${u}</small></dd><div class="sub">${sub}</div></dl>`;
  $("en-kpis").innerHTML = kpi("Endurance energy", fmt(r.E_total_kwh, 2), "kWh", `${r.laps} laps · ${fmt(r.E_lap_kwh, 3)} kWh per lap`) +
    kpi("Power cap", fmt(r.cap_kw, 1), "kW", `motor limit ${fmt(r.P_max_kw, 0)} kW`) +
    kpi("Endurance", fmt(r.T, 1), "s", `${fmt(r.points, 2)} P · lap ${fmt(r.t_lap, 2)} s`) +
    kpi("Efficiency factor", fmt(r.EF / 1e6, 2), "M s²kWh", `T² · E (LionsLTS EFmin ${fmt(r.EF_lionslts_min / 1e6, 2)} M)`);
  const b = r.breakdown;
  barChart($("en-break"), { series: [{ name: "kWh per lap" }], rows: [
    { label: "battery out", values: [b.battery_out] }, { label: "drivetrain losses", values: [b.drivetrain_losses] },
    { label: "drag", values: [b.drag] }, { label: "rolling resistance", values: [b.rolling] },
    { label: "friction brakes", values: [b.friction_brakes] }, { label: "recuperated", values: [b.battery_back] }],
    xLabel: "kWh per lap", labelWidth: 140, valFmt: (v) => fmt(v, 3) });
  $("en-break-note").textContent = `Wheel work ${fmt(b.wheel_drive, 3)} kWh = drag + rolling + braking (the speed at the end of a flying lap equals the start). Motor efficiency: ${r.eta_map}.`;
  $("en-events").innerHTML = `<thead><tr><th>Event</th><th class="num">Energy [kWh]</th></tr></thead><tbody>` +
    Object.entries(r.per_event).map(([n, e]) => `<tr><td>${evLabel(n)}</td><td class="num">${fmt(e, 3)}</td></tr>`).join("") + "</tbody>";
  const bu = r.budget;
  $("en-budget-out").innerHTML = bu ? `<h3 class="ch">For ${fmt(bu.budget_kwh, 2)} kWh</h3><p class="hint">${!bu.feasible ? "Not even the lowest cap fits the budget." :
    !bu.limited ? `The full motor power fits: ${fmt(bu.E_total_kwh, 2)} kWh needed.` :
    `Power cap <b>${fmt(bu.cap_kw, 2)} kW</b>: endurance ${fmt(bu.T, 1)} s, ${fmt(bu.points, 2)} points, ${fmt(bu.E_total_kwh, 3)} kWh.`}</p>` : "";
  const cap = r.sweep.map((x) => x.cap_kw);
  lineChart($("en-pts"), { series: [{ name: "points", x: cap, y: r.sweep.map((x) => x.points) }], xLabel: "power cap [kW]", yLabel: "points", markers: true, mark: bu && bu.limited ? bu.cap_kw : r.cap_kw, height: 200 });
  lineChart($("en-e"), { series: [{ name: "energy", x: cap, y: r.sweep.map((x) => x.E_total_kwh) }], xLabel: "power cap [kW]", yLabel: "kWh", markers: true, mark: bu ? bu.cap_kw : undefined, height: 200 });
  lineChart($("en-t"), { series: [{ name: "lap time", x: cap, y: r.sweep.map((x) => x.t_lap) }], xLabel: "power cap [kW]", yLabel: "s", markers: true, height: 200 });
  lineChart($("en-p"), { series: [{ name: "power", x: r.trace.s, y: r.trace.P_batt }], xLabel: "distance [m]", yLabel: "kW", zero: true, height: 200 });
}
$("en-csv").onclick = () => {
  const r = S.results.en; if (!r) return;
  downloadCSV("endurance_power_cap.csv", ["cap_kW", "lap_s", "endurance_s", "E_lap_kWh", "E_total_kWh", "EF", "points", "v_max_mps"],
    r.sweep.map((x) => [x.cap_kw, x.t_lap, x.T, x.E_lap_kwh, x.E_total_kwh, x.EF, x.points, x.v_max]));
};
DRAW.energy = drawEnergy;

// ------------------------------------------------------------------ ggv
$("gg-run").onclick = async () => {
  try {
    S.results.gg = (await api("/api/ggv", { changes: S.changes, event: $("gg-event").value, competition: S.comp })).result;
    drawGG();
  } catch (e) { $("gg-gg").innerHTML = `<p class="msg err">${esc(e.message)}</p>`; }
};
function drawGG() {
  const g = S.results.gg; if (!g) return;
  const n = g.v_kmh.length, idx = [...Array(n).keys()].filter((i) => n <= 6 || i % Math.ceil(n / 6) === 0 || i === n - 1);
  const ser = [];
  idx.forEach((i, k) => {
    const c = window.Charts.ramp(idx.length > 1 ? k / (idx.length - 1) : 1);
    const ay = g.ay_g[i], acc = g.ax_acc_g[i], dec = g.ax_dec_g[i];
    // closed outline: right half (ay >= 0), mirrored to the left
    const x = ay.concat(ay.slice().reverse()).concat(ay.map((a) => -a)).concat(ay.slice().reverse().map((a) => -a));
    const y = acc.concat(dec.slice().reverse()).concat(dec).concat(acc.slice().reverse());
    ser.push({ name: `${fmt(g.v_kmh[i], 0)} km/h`, x, y, color: c, width: 1.5 });
  });
  // the gg outline is not a function of x: draw without the nearest-x tooltip by keeping points in order
  lineChart($("gg-gg"), { series: ser, xLabel: "lateral [g]", yLabel: "longitudinal [g]", height: 360, zero: true });
  lineChart($("gg-v"), { series: [
    { name: "max lateral", x: g.v_kmh, y: g.ay_g.map((r) => r[r.length - 1]) },
    { name: "max drive", x: g.v_kmh, y: g.ax_acc_g.map((r) => r[0]) },
    { name: "max brake", x: g.v_kmh, y: g.ax_dec_g.map((r) => r[0]) }], xLabel: "speed [km/h]", yLabel: "g", height: 360, zero: true, markers: true });
}
$("gg-csv").onclick = () => {
  const g = S.results.gg; if (!g) return;
  const rows = [];
  g.v_kmh.forEach((v, i) => g.ay_g[i].forEach((a, j) => rows.push([v, a, g.ax_acc_g[i][j], g.ax_dec_g[i][j]])));
  downloadCSV("ggv.csv", ["v_kmh", "ay_g", "ax_drive_g", "ax_brake_g"], rows);
};
DRAW.ggv = () => { if (!S.results.gg) $("gg-run").click(); else drawGG(); };

// ------------------------------------------------------------------ logs
const UNIT_SPEED = [["kmh", "km/h"], ["mps", "m/s"], ["mph", "mph"], ["rpm_motor", "motor rpm (gear, tyre)"]];
const UNIT_ACC = [["g", "g"], ["mps2", "m/s²"]];
async function loadLogs(select) {
  const r = await api("/api/logs");
  $("lg-src").innerHTML = `<option value="">choose a log …</option>` + r.logs.map((l) => `<option value="${esc(l.id)}">${esc(l.name)} (${l.kind === "reference" ? "LapSim reference" : `${fmt(l.mb, 1)} MB`})</option>`).join("");
  if (select) { $("lg-src").value = select; inspectLog(); }
}
$("lg-src").onchange = inspectLog;
$("lg-file").onchange = async () => {
  const f = $("lg-file").files[0]; if (!f) return;
  showMsg("lg-msg", `uploading ${f.name} …`);
  try {
    const r = await fetch(`/api/logs/upload?name=${encodeURIComponent(f.name)}`, { method: "POST", headers: { "X-Perf-Lab": "1" }, body: f });
    const j = await r.json();
    if (!j.ok) throw new Error(j.message);
    showMsg("lg-msg", j.message);
    await loadLogs(j.id);
  } catch (e) { showErr("lg-msg", e); }
  $("lg-file").value = "";
};
async function inspectLog() {
  const src = $("lg-src").value;
  $("lg-map").hidden = !src; $("pt-out").hidden = true;
  if (!src) return;
  try {
    const i = (await api("/api/logs/inspect", { source: src })).result;
    S.logInfo = i;
    const opts = (sel, none) => `${none ? `<option value="">${none}</option>` : ""}` + i.channels.map((c, k) => `<option value="${k}" ${c.name === sel ? "selected" : ""}>${esc(c.name)}${c.group ? ` · ${esc(c.group)}` : ""}</option>`).join("");
    $("lg-v").innerHTML = opts(i.suggest.speed, "–");
    $("lg-ax").innerHTML = opts(i.suggest.ax, "none");
    $("lg-ay").innerHTML = opts(i.suggest.ay, "none");
    $("lg-t").innerHTML = opts(i.suggest.time, "none");
    $("lg-s").innerHTML = opts(i.suggest.distance, "none");
    const uo = (list, sel) => list.map(([v, l]) => `<option value="${v}" ${v === sel ? "selected" : ""}>${l}</option>`).join("");
    const vname = (i.suggest.speed || "").toLowerCase();
    $("lg-vu").innerHTML = uo(UNIT_SPEED, /kmh|km_h/.test(vname) ? "kmh" : /mps|m_s/.test(vname) ? "mps" : /act_speed|rpm/.test(vname) ? "rpm_motor" : "kmh");
    $("lg-axu").innerHTML = uo(UNIT_ACC, "g"); $("lg-ayu").innerHTML = uo(UNIT_ACC, "g");
    document.querySelectorAll(".csvonly").forEach((e) => { e.hidden = i.format !== "csv"; });
    showMsg("lg-msg", `${i.format.toUpperCase()}, ${i.channels.length} channels${i.rows ? `, ${i.rows} rows` : ""}${i.powertrain ? `, ${i.powertrain} of 9 powertrain channels` : ""}`);
  } catch (e) { showErr("lg-msg", e); }
}
function logSpec() {
  const ch = (id) => { const v = $(id).value; return v === "" ? null : S.logInfo.channels[+v]; };
  const sig = (id, uid) => { const c = ch(id); return c ? { channel: c.name, group: c.group || "", unit: $(uid).value } : null; };
  return { speed: sig("lg-v", "lg-vu"), ax: sig("lg-ax", "lg-axu"), ay: sig("lg-ay", "lg-ayu"),
    time: ch("lg-t") ? ch("lg-t").name : "", distance: ch("lg-s") ? ch("lg-s").name : "", t0: $("lg-t0").value, t1: $("lg-t1").value };
}
$("lg-run").onclick = async () => {
  try {
    const sh = $("lg-shift").value.trim();
    const r = (await api("/api/logs/correlate", { source: $("lg-src").value, spec: logSpec(), event: $("lg-event").value, changes: S.changes, competition: S.comp,
      shift: sh === "" || sh === "auto" ? "auto" : +sh })).result;
    S.results.lg = r; $("lg-out").hidden = false; drawLog();
    showMsg("lg-msg", "");
  } catch (e) { showErr("lg-msg", e); }
};
function drawLog() {
  const r = S.results.lg; if (!r || $("lg-out").hidden) return;
  const kpi = (t, v, u, sub) => `<dl class="kpi"><dt>${t}</dt><dd>${v}<small>${u}</small></dd><div class="sub">${sub}</div></dl>`;
  $("lg-kpis").innerHTML = kpi("Speed RMSE", fmt(r.rmse_v_kmh, 2), "km/h", `bias ${fmt(r.bias_v_kmh, 2)} km/h (log − sim)`) +
    kpi("Time over the compared part", fmt(r.t_log, 2), "s", `simulation ${fmt(r.t_sim, 2)} s · Δ ${fmt(r.t_log - r.t_sim, 2)} s`) +
    kpi("Compared", fmt(100 * r.coverage, 0), "%", `${fmt(r.s_from, 0)}–${fmt(r.s_to, 0)} m of ${fmt(r.lap_length, 0)} m · offset ${fmt(r.shift_m, 1)} m${r.shift_auto ? " (searched)" : ""}`) +
    kpi("Top speed", fmt(r.v_max_log_kmh, 1), "km/h", `simulation ${fmt(r.v_max_sim_kmh, 1)} km/h`) +
    (r.rmse_ax_g !== undefined ? kpi("ax RMSE", fmt(r.rmse_ax_g, 3), "g", "") : "") + (r.rmse_ay_g !== undefined ? kpi("|ay| RMSE", fmt(r.rmse_ay_g, 3), "g", "") : "");
  const t = r.trace;
  lineChart($("lg-cv"), { series: [{ name: "log", x: t.s, y: t.v_log }, { name: "simulation", x: t.s, y: t.v_sim, dash: true }], xLabel: "distance [m]", yLabel: "km/h", height: 230 });
  lineChart($("lg-cdt"), { series: [{ name: "log − simulation", x: t.s, y: t.dt }], xLabel: "distance [m]", yLabel: "s", height: 230, zero: true });
  const pair = (log, sim) => (log.length ? [{ name: "log", x: t.s, y: log }, { name: "simulation", x: t.s, y: sim, dash: true }] : [{ name: "simulation (no log channel)", x: t.s, y: sim }]);
  lineChart($("lg-cax"), { series: pair(t.ax_log, t.ax_sim), xLabel: "distance [m]", yLabel: "g", height: 180, zero: true });
  lineChart($("lg-cay"), { series: pair(t.ay_log, t.ay_sim), xLabel: "distance [m]", yLabel: "g", height: 180 });
}
$("lg-csv").onclick = () => {
  const r = S.results.lg; if (!r) return;
  const t = r.trace;
  downloadCSV(`correlation_${r.event}.csv`, ["s_m", "v_log_kmh", "v_sim_kmh", "dt_s", "ax_log_g", "ax_sim_g", "ay_log_g", "ay_sim_g"],
    t.s.map((s, i) => [s, t.v_log[i], t.v_sim[i], t.dt[i], t.ax_log[i] ?? "", t.ax_sim[i], t.ay_log[i] ?? "", t.ay_sim[i]]));
};
$("pt-run").onclick = async () => {
  const out = $("pt-out"); out.hidden = false;
  if (!$("lg-src").value) { out.innerHTML = `<p class="msg err">choose an .mf4 log first</p>`; return; }
  out.innerHTML = `<p class="msg">analysing …</p>`;
  try {
    const r = (await api("/api/logs/powertrain", { source: $("lg-src").value, changes: S.changes, competition: S.comp, motors_per_axle: +$("pt-n").value })).result;
    const m = r.map, k = r.fit.k;
    const head = m.n_edges.slice(0, -1).map((n, j) => `<th>${fmt(n / 1e3, 0)}–${fmt(Math.min(m.n_edges[j + 1], 20000) / 1e3, 0)}k</th>`).join("");
    const rows = m.t_edges.slice(0, -1).map((t, i) => `<tr><th>${fmt(t, 1)}–${fmt(m.t_edges[i + 1], 1)} Nm</th>${m.eff[i].map((e, j) => e === null ? `<td class="muted">–</td>` :
      `<td class="cell" style="background:${window.Charts.ramp((e - 0.6) / 0.4)}" title="${m.count[i][j]} samples">${fmt(e, 3)}</td>`).join("")}</tr>`).join("");
    out.innerHTML = `<div class="kpis">
      <dl class="kpi"><dt>Efficiency, median</dt><dd>${fmt(r.eff_median, 3)}</dd><div class="sub">p10 ${fmt(r.eff_p10, 3)} · p90 ${fmt(r.eff_p90, 3)} · LapSim ${fmt(r.eta_motor_lapsim, 2)}</div></dl>
      <dl class="kpi${r.rollout.decel_median > 0.6 ? " warn" : ""}"><dt>Roll-out deceleration</dt><dd>${fmt(r.rollout.decel_median, 2)}<small>m/s²</small></dd><div class="sub">${esc(r.rollout_verdict)}</div></dl></div>
      <h3 class="ch">Median efficiency (motor + inverter) by torque and speed</h3>
      <div class="tablewrap"><table class="efftab"><thead><tr><th></th>${head}</tr></thead><tbody>${rows}</tbody></table></div>
      <p class="note">${r.samples} samples, ${fmt(r.duration_s, 0)} s, up to ${fmt(r.T_max, 1)} Nm and ${fmt(r.n_max, 0)} rpm. Loss per motor [kW] = ${fmt(k[0], 3)} + ${k[1].toExponential(2)} T² + ${fmt(k[2], 4)} n + ${fmt(k[3], 5)} n² (n in krpm; rms ${fmt(r.fit.rms_kw, 2)} kW). Assumes act_Torque is shaft torque [Nm] and act_Power the DC power [kW].</p>
      <h3 class="sub">Save as an efficiency map for LapSim</h3>
      <div class="row"><label class="field">Name <input id="pt-name" placeholder="e.g. D1F58" maxlength="40"></label>
      <button type="button" class="btn" id="pt-save">Save map</button><label class="check"><input type="checkbox" id="pt-over"> replace</label></div>
      <p class="msg" id="pt-msg"></p>`;
    $("pt-save").onclick = async () => {
      try {
        const s = await api("/api/logs/save_map", { source: $("lg-src").value, name: $("pt-name").value.trim(), changes: S.changes, competition: S.comp, motors_per_axle: +$("pt-n").value, overwrite: $("pt-over").checked });
        showMsg("pt-msg", `${s.message}. Choose it under Motor efficiency at the top.`);
        await loadInfo();
      } catch (e) { showErr("pt-msg", e); }
    };
  } catch (e) { out.innerHTML = `<p class="msg err">${esc(e.message)}</p>`; }
};
DRAW.logs = drawLog;

// ------------------------------------------------------------------ guide
function buildGuide() {
  if (!S.info) return;
  const rows = [];
  for (const g of S.info.groups) for (const r of g.rows) rows.push([g.title, r]);
  $("g-params").innerHTML = `<thead><tr><th>Group</th><th>Parameter</th><th>Name</th><th class="num">Baseline</th><th>Unit</th><th>Meaning</th></tr></thead><tbody>` +
    rows.map(([g, r]) => `<tr data-q="${esc((g + " " + r.label + " " + r.key + " " + r.desc).toLowerCase())}"><td class="muted">${esc(g)}</td><td>${esc(r.label)}</td><td><code>${esc(r.key)}</code></td><td class="num">${r.value === null ? "–" : fmt(r.value)}</td><td>${esc(r.unit)}</td><td class="muted">${esc(r.desc)}</td></tr>`).join("") + "</tbody>";
}
$("g-filter").addEventListener("input", () => {
  const q = $("g-filter").value.trim().toLowerCase();
  document.querySelectorAll("#g-params tbody tr").forEach((tr) => { tr.hidden = q && !tr.dataset.q.includes(q); });
  document.querySelectorAll("#g-terms dt").forEach((dt) => {
    const hit = !q || (dt.textContent + " " + dt.nextElementSibling.textContent).toLowerCase().includes(q);
    dt.hidden = !hit; dt.nextElementSibling.hidden = !hit;
  });
});
// "? Help" in the top bar: the guide section of the page you are on
function helpTarget() {
  const page = (location.hash || "#overview").slice(1);
  $("help-page").href = page === "guide" ? "#guide" : `#guide-${page}`;
}
window.addEventListener("hashchange", helpTarget);
try {
  if (!localStorage.getItem("perf-guide-seen")) $("ov-new").hidden = false;
} catch { /* storage blocked: no banner */ }
$("ov-new-close").onclick = () => { $("ov-new").hidden = true; try { localStorage.setItem("perf-guide-seen", "1"); } catch { /* ignore */ } };

// ------------------------------------------------------------------ start
(async () => {
  try {
    await loadInfo();
    route();
    helpTarget();
    await runOverview();
    await loadLogs();
  } catch (e) {
    document.querySelector("main").insertAdjacentHTML("afterbegin", `<section class="panel"><h2>Cannot start</h2><p class="msg err">${esc(e.message)}</p></section>`);
  }
})();
