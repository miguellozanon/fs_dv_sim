// DV launcher page: polls /api/state, posts actions with the launcher header.
"use strict";

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const S = { state: null, lon: "reference", percep: "none", logKey: null, wantDash: false, filled: false };

async function post(path, body) {
  try {
    const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json", "X-FSG-Launcher": "1" }, body: JSON.stringify(body || {}) });
    return await r.json();
  } catch (e) {
    return { ok: false, message: "the launcher is not reachable" };
  }
}

function fmtTime(t) {
  return t ? new Date(t * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }) : "";
}
function fmtDur(s) {
  s = Math.max(0, Math.round(s));
  return s < 60 ? `${s} s` : s < 3600 ? `${Math.floor(s / 60)} min ${s % 60} s` : `${Math.floor(s / 3600)} h ${Math.round((s % 3600) / 60)} min`;
}

// ------------------------------------------------------------------ simulation form
const form = $("sim-form");
const backend = () => form.querySelector('input[name="backend"]:checked').value;

function updateFormVisibility() {
  const b = backend(), m = $("f-mission").value;
  $("seed-wrap").hidden = !["autocross", "trackdrive"].includes(m) || b === "demo";
  $("rtf-wrap").hidden = b === "isaac";
  $("rviz-wrap").hidden = b === "demo";
  $("headless-wrap").hidden = b !== "isaac";
  $("pol-wrap").hidden = S.lon !== "rl";
  // cone sources: the offline demo has none; YOLO alone needs Isaac's rendered images
  $("cones-wrap").hidden = b === "demo";
  document.querySelector('[data-percep="yolo"]').hidden = b !== "isaac";
  if (b === "demo" || (b !== "isaac" && S.percep === "yolo")) setPercep("none");
  const own = ["lidar", "fusion", "camera"].includes(S.percep), circuit = ["autocross", "trackdrive"].includes(m);
  const needsYolo = S.percep === "yolo" || (b === "isaac" && ["fusion", "camera"].includes(S.percep));
  $("record-wrap").hidden = b !== "isaac";
  $("cone-model-wrap").hidden = !needsYolo;
  $("slam-wrap").hidden = !own;
  $("race-wrap").hidden = !(own && m === "trackdrive");
  // racing line: from the start on ground truth; with perceived cones only on the mapped track (trackdrive, race after)
  $("line-wrap").hidden = !circuit || S.lon === "rl" || (own && m !== "trackdrive");
  const pi = S.state?.perception || {};
  $("cones-hint").textContent = S.percep === "none" ? "The driver gets the simulator's cone positions."
    : needsYolo && !(pi.models || []).length ? "No cone detector yet: record images (tick below), then RL & Performance → Train the cone detector."
    : S.percep === "lidar" ? (b === "isaac" ? "Cones from the LiDAR points. Isaac's RTX LiDAR gives no stripe colours, so the car cannot drive a lap on them: use LiDAR + camera."
      : "Cones from the LiDAR points (own car filtered out, colour from the stripes).")
    : S.percep === "fusion" ? "LiDAR positions with the camera's colours; the cones build a map."
    : S.percep === "camera" ? "Camera cones only, collected in a map (less accurate than with the LiDAR)."
    : S.lon === "rl" ? "With the RL driver the car follows the known path; the camera cones steer the reference driver."
    : "The driver steers from the cones YOLO finds in the camera image (map from depth + car pose).";
}
function setPercep(v) {
  S.percep = v;
  document.querySelectorAll("[data-percep]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.percep === v)));
}
form.addEventListener("change", updateFormVisibility);
document.querySelectorAll("[data-percep]").forEach((btn) => btn.addEventListener("click", () => {
  S.percep = btn.dataset.percep;
  document.querySelectorAll("[data-percep]").forEach((b) => b.setAttribute("aria-pressed", String(b === btn)));
  updateFormVisibility();
}));
document.querySelectorAll("[data-lon]").forEach((btn) => btn.addEventListener("click", () => {
  S.lon = btn.dataset.lon;
  document.querySelectorAll("[data-lon]").forEach((b) => b.setAttribute("aria-pressed", String(b === btn)));
  updateFormVisibility();
  renderVehSummary();
}));
// ------------------------------------------------------------------ vehicle setup (own page, used at the start)
// One source of truth: the Changes field of the simulation form (#f-changes). The Vehicle setup page edits it.
const V = { info: null, check: null, timer: 0, seq: 0, inputs: {} };
const CH_RE = /^\s*([A-Za-z_][\w.]*)\s*([=*+-])\s*(.+?)\s*$/;
const NUM_RE = /^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$/;
const chList = () => $("f-changes").value.split(/[\s,;]+/).filter(Boolean);

function fmtNum(x) {
  if (x === null || x === undefined) return "—";
  const a = Math.abs(x);
  return a !== 0 && (a >= 1e5 || a < 1e-3) ? x.toExponential(3) : String(+x.toPrecision(6));
}
function fmtT(t) {
  if (t === null || t === undefined) return "—";
  return t < 120 ? `${t.toFixed(3)} s` : `${Math.floor(t / 60)} min ${(t % 60).toFixed(1)} s`;
}
const sign = (x, d) => `${x >= 0 ? "+" : "−"}${Math.abs(x).toFixed(d)}`;

// pages (like the dashboard): #setup shows the Vehicle setup, anything else the launcher
function showPage() {
  const setup = location.hash === "#setup";
  $("page-main").hidden = setup;
  $("page-setup").hidden = !setup;
  document.querySelectorAll(".tabs a[href^='#']").forEach((a) => a.removeAttribute("aria-current"));
  const cur = setup ? $("nav-setup") : document.querySelector(`.tabs a[href="${location.hash || "#sim"}"]`) || $("nav-sim");
  cur.setAttribute("aria-current", "page");
  if (setup) { window.scrollTo(0, 0); if (!V.info) loadVehicle(); }
  else if (location.hash) document.querySelector(location.hash)?.scrollIntoView();
}
window.addEventListener("hashchange", showPage);

async function loadVehicle() {
  try {
    V.info = await (await fetch("/api/vehicle")).json();
  } catch (e) {
    V.info = { ok: false, message: "the launcher is not reachable" };
  }
  if (!V.info.ok) {
    $("vs-groups").innerHTML = `<p class="msg err">${esc(V.info.message)}</p>`;
    $("veh-brief").textContent = V.info.message;
    return;
  }
  $("vs-src").textContent = `${V.info.params_file} · ${V.info.competition} · changed ${V.info.modified}`;
  const host = $("vs-groups");
  host.innerHTML = "";
  V.inputs = {};
  V.info.schema.forEach((g, gi) => {
    const det = document.createElement("details");
    det.className = "pgroup";
    det.open = gi < 5;
    const tag = g.kind === "event" ? "LapSim prediction only" : g.kind === "sim" ? "simulator only" : "";
    det.innerHTML = `<summary>${esc(g.title)}${tag ? ` <span class="tag">${tag}</span>` : ""}<span class="count" hidden></span></summary>
      <div class="tbl-scroll"><table class="ptable"><thead><tr><th scope="col">Parameter</th><th scope="col" class="num">Baseline</th>
      <th scope="col" class="num">New value</th><th scope="col" class="unit">Unit</th></tr></thead><tbody></tbody></table></div>`;
    const tb = det.querySelector("tbody");
    g.items.forEach((it) => {
      const tr = document.createElement("tr");
      const id = "vp-" + it.key.replace(/\W/g, "_");
      tr.innerHTML = `<td><label for="${id}"><span class="p-label">${esc(it.label)}</span>
          <span class="p-key">${esc(it.key)}${it.desc ? " · " + esc(it.desc) : ""}</span></label></td>
        <td class="base">${esc(it.base_text || fmtNum(it.base))}</td>
        <td class="inp"><input class="p-in" id="${id}" inputmode="decimal" autocomplete="off" spellcheck="false" placeholder="${esc(it.base_text ? "NaN" : fmtNum(it.base))}"></td>
        <td class="unit">${esc(it.unit === "-" ? "" : it.unit)}</td>`;
      const inp = tr.querySelector("input");
      inp.addEventListener("input", onFormInput);
      V.inputs[it.key] = { inp, tr, det, it };
      tb.appendChild(tr);
    });
    host.appendChild(det);
  });
  fillForm(chList());
  scheduleCheck(0);
}

// the table and "Other changes" -> the list of changes
function collectChanges() {
  const out = [], bad = [];
  Object.entries(V.inputs).forEach(([k, x]) => {
    const v = x.inp.value.trim().replace(",", ".");
    x.tr.classList.remove("bad", "changed");
    if (!v) return;
    const fac = v.match(/^[x×*]\s*(.+)$/i);
    if (fac && NUM_RE.test(fac[1])) { out.push(`${k}*${fac[1]}`); x.tr.classList.add("changed"); }
    else if (NUM_RE.test(v) || /^nan$/i.test(v)) { out.push(`${k}=${/^nan$/i.test(v) ? "NaN" : v}`); x.tr.classList.add("changed"); }
    else { bad.push(k); x.tr.classList.add("bad"); }
  });
  $("vs-other").value.split(/[\s,;]+/).map((t) => t.trim()).filter(Boolean).forEach((t) => out.push(t));
  countChanged();
  return { changes: out, bad };
}

function countChanged() {
  Object.values(V.inputs).reduce((m, x) => m.set(x.det, (m.get(x.det) || 0) + (x.tr.classList.contains("changed") ? 1 : 0)), new Map())
    .forEach((n, det) => { const c = det.querySelector(".count"); c.hidden = !n; c.textContent = `${n} changed`; });
}

// the list of changes -> the table and "Other changes"
function fillForm(changes) {
  Object.values(V.inputs).forEach((x) => { x.inp.value = ""; x.tr.classList.remove("bad", "changed"); });
  const rest = [];
  (changes || []).forEach((c) => {
    const m = String(c).match(CH_RE);
    const x = m && V.inputs[m[1]];
    if (x && m[2] === "=") { x.inp.value = m[3]; x.tr.classList.add("changed"); }
    else if (x && m[2] === "*") { x.inp.value = "*" + m[3]; x.tr.classList.add("changed"); }
    else rest.push(c);
  });
  $("vs-other").value = rest.join(" ");
  countChanged();
  showChangesLine([]);
}

function showChangesLine(bad) {
  const ch = chList(), line = $("vs-changes-line");
  if (bad.length) line.innerHTML = `Check ${bad.map((k) => `<code>${esc(k)}</code>`).join(" ")}: enter a number or a factor like <code>*1.05</code>.`;
  else line.innerHTML = ch.length ? ch.map((c) => `<code>${esc(c)}</code>`).join("") : "No changes: the LapSim baseline.";
  $("btn-edit-setup").textContent = ch.length ? `Edit parameters (${ch.length})` : "Edit parameters";
}

function onFormInput() {
  const { changes, bad } = collectChanges();
  $("f-changes").value = changes.join(" ");
  matchConfig();
  showChangesLine(bad);
  scheduleCheck(400);
}

function setConfigSelects(name) {
  document.querySelectorAll(".cfg-select").forEach((s) => { s.value = name; });
  $("vs-delete").disabled = !name;
}

function matchConfig() {
  // the configuration stays selected only while the changes are exactly its changes
  const c = (S.state?.configs || []).find((x) => x.name === $("f-config").value);
  if (c && (c.changes || []).join(" ") !== chList().join(" ")) setConfigSelects("");
}

function useConfig(name) {
  const c = (S.state?.configs || []).find((x) => x.name === name);
  $("f-changes").value = c ? (c.changes || []).join(" ") : "";
  setConfigSelects(c ? c.name : "");
  $("vs-name").value = c ? c.name : "";
  fillForm(chList());
  scheduleCheck(0);
}

function scheduleCheck(ms) {
  clearTimeout(V.timer);
  V.timer = setTimeout(runCheck, ms);
}

async function runCheck() {
  const seq = ++V.seq;
  const r = await post("/api/vehicle/check", { changes: $("f-changes").value, longitudinal: S.lon, policy: $("f-policy").value || "auto" });
  if (seq !== V.seq) return;
  V.check = r;
  renderVehSummary();
}

function rlText(rl) {
  if (!rl) return null;
  if (rl.chosen_problem) return { t: rl.chosen_problem, warn: true };
  if (rl.exact) return { t: `RL driver: policy “${rl.exact}”, trained for exactly this car`, warn: false };
  if (rl.universal) return { t: `RL driver: universal policy “${rl.universal}” covers this car`, warn: false };
  return { t: "RL driver: no policy covers this car" + (rl.why_not?.length ? ` (${rl.why_not.join("; ")})` : "") + "; the reference driver drives it", warn: true };
}

function renderVehSummary() {
  const r = V.check;
  if (!r) return;
  const errs = (r.errors || []).length ? `<ul class="veh-errors">${r.errors.map((e) => `<li>${esc(e)}</li>`).join("")}</ul>` : "";
  const b = r.baseline, l = r.lapsim, has = (r.changes || []).length > 0;
  const rl = rlText(r.rl);
  const rlHtml = rl ? `<span class="${rl.warn && S.lon === "rl" ? "warn" : ""}">${esc(rl.t)}</span>` : "";

  // simulation page: one line
  let brief = errs;
  if (b && l) {
    brief += `<p class="hint small">LapSim: <b>${l.total.toFixed(1)}</b> points` +
      (has ? ` (${sign(l.total - b.total, 1)} vs the baseline)` : " (baseline)") + (rlHtml ? ` · ${rlHtml}` : "") + `</p>`;
  } else if (rlHtml) brief += `<p class="hint small">${rlHtml}</p>`;
  $("veh-brief").innerHTML = brief;

  // vehicle setup page: the table
  let h = errs;
  const mf = matFor(r.changes);
  const matCol = !M.st?.available ? "" : mf ? "ok" : ["starting", "running"].includes(M.st.state) && sameList(M.st.changes, r.changes) ? "…" : "–";
  if (b && l) {
    const rows = Object.keys(b.events).map((ev) => {
      const e0 = b.events[ev], e1 = l.events[ev] || {};
      const dT = e1.T - e0.T, dP = e1.points - e0.points, same = Math.abs(dT) < 5e-4;
      const cls = !has || same ? "" : dT < 0 ? "better" : "worse";
      const m = mf?.results?.[ev];
      return `<tr><th scope="row">${esc(ev[0].toUpperCase() + ev.slice(1))}</th>` +
        `<td class="num">${fmtT(e0.T)}<small>${e0.points?.toFixed(1)} pts</small></td>` +
        `<td class="num">${fmtT(e1.T)}<small>${e1.points?.toFixed(1)} pts</small>` +
        (has ? `<small class="${cls}">${same ? "±0 s" : sign(dT, 3) + " s"} · ${sign(dP, 1)} pts</small>` : "") + `</td>` +
        (matCol ? `<td class="num">${m ? fmtT(m.T) + `<small>${m.points?.toFixed(1) ?? "–"} pts</small>` : matCol}</td>` : "") + `</tr>`;
    }).join("");
    h += `<div class="tbl-scroll"><table class="vtable vpred">
      <thead><tr><th>Event</th><th class="num">Baseline</th><th class="num">${has ? "This setup" : "LapSim"}</th>${matCol ? `<th class="num">MATLAB</th>` : ""}</tr></thead>
      <tbody>${rows}</tbody>
      <tfoot><tr><th scope="row">Total points</th><td class="num">${b.total.toFixed(1)}</td>
        <td class="num"><b>${l.total.toFixed(1)}</b>${has ? `<small>${sign(l.total - b.total, 1)}</small>` : ""}</td>${matCol ? `<td class="num">${mf && mf.total != null ? `<b>${mf.total.toFixed(1)}</b>` : ""}</td>` : ""}</tr></tfoot>
    </table></div>`;
  }
  const notes = [];
  if ((r.lapsim_only || []).length) notes.push(`Only in the LapSim prediction (driver model): <code>${esc(r.lapsim_only.join(" "))}</code>`);
  if (has && (r.sim_changes || []).length) notes.push(`The simulated car gets: <code>${esc(r.sim_changes.join(" "))}</code>`);
  if (rlHtml) notes.push(rlHtml);
  if (notes.length) h += `<p class="hint small veh-notes">${notes.join("<br>")}</p>`;
  $("veh-summary").innerHTML = h;
}

// ------------------------------------------------------------------ MATLAB (vehicle setup page)
const M = { st: null, timer: 0 };
const sameList = (a, b) => (a || []).join(" ") === (b || []).join(" ");
const matFor = (changes) => (M.st?.state === "done" && sameList(M.st.changes, changes) ? M.st : null);

async function loadMatlab() {
  try { M.st = await (await fetch("/api/vehicle/matlab")).json(); } catch (e) { return; }
  renderMatlab();
  renderVehSummary();
  clearTimeout(M.timer);
  const busy = ["starting", "running"].includes(M.st.state);
  if (busy || location.hash === "#setup") M.timer = setTimeout(loadMatlab, busy ? 2000 : 6000);
}

function renderMatlab() {
  const st = M.st || {};
  const line = $("vs-mat-status");
  const ago = (t) => (t ? fmtTime(t) : "");
  let txt;
  if (!st.available) txt = st.message || "MATLAB not found";
  else if (st.state === "idle") txt = `Ready: ${st.exe.split(/[\\/]/).slice(-3, -2)[0] || "MATLAB"}. Runs the real LR26_LapSim with the setup above.`;
  else if (["starting", "running"].includes(st.state)) txt = `${st.message} · ${Math.round(Date.now() / 1000 - (st.started || Date.now() / 1000))} s`;
  else if (st.state === "done") txt = `${st.message} (${ago(st.finished)}) for ${(st.changes || []).join(" ") || "the baseline"}`;
  else if (st.state === "error") txt = st.message;
  else txt = st.message || st.state;
  line.textContent = txt;
  line.className = "mat-status " + (st.state === "error" ? "err" : st.state === "done" ? "good" : "");
  const busy = ["starting", "running"].includes(st.state);
  $("vs-mat-run").disabled = !st.available;
  $("vs-mat-run").textContent = busy ? "Run again with this setup" : "Run this setup in MATLAB";
  $("vs-mat-cancel").hidden = !busy;
  const files = Object.entries(st.files || {});
  const btn = ([k, f]) => `<button type="button" class="btn small-btn" data-open="${esc(k)}" ${f.exists ? "" : "disabled"} title="${esc(f.path)}">` +
    `${esc(f.label)}${f.kind === "file" && f.name !== f.label ? ` <span class="muted">${esc(f.name)}</span>` : ""}</button>`;
  $("vs-files").innerHTML = files.filter(([, f]) => f.group === "files").map(btn).join("");
  const res = files.filter(([, f]) => f.group === "results" && f.exists);
  $("vs-files-results").innerHTML = res.length ? res.map(btn).join("") : `<span class="muted">None yet: run a setup in MATLAB.</span>`;
}

document.addEventListener("click", async (e) => {
  const b = e.target.closest("[data-open]");
  if (!b) return;
  setMsg("vs-msg", await post("/api/vehicle/open", { target: b.dataset.open }));
});
$("vs-mat-run").addEventListener("click", async () => {
  const r = await post("/api/vehicle/matlab", { changes: $("f-changes").value });
  setMsg("vs-msg", r);
  loadMatlab();
});
$("vs-mat-cancel").addEventListener("click", async () => { setMsg("vs-msg", await post("/api/vehicle/matlab", { action: "cancel" })); loadMatlab(); });
window.addEventListener("hashchange", () => { if (location.hash === "#setup") loadMatlab(); });

$("vs-other").addEventListener("input", onFormInput);
$("f-changes").addEventListener("input", () => { matchConfig(); fillForm(chList()); scheduleCheck(500); });
document.querySelectorAll(".cfg-select").forEach((sel) => sel.addEventListener("change", () => useConfig(sel.value)));
$("f-policy").addEventListener("change", () => scheduleCheck(0));
$("vs-reload").addEventListener("click", async () => { V.info = null; await loadVehicle(); loadMatlab(); setMsg("vs-msg", { ok: true, message: "LapSim files re-read" }); });
$("vs-clear").addEventListener("click", () => useConfig(""));
$("vs-save").addEventListener("click", async () => {
  const name = $("vs-name").value.trim();
  if (!name) { setMsg("vs-msg", { ok: false, message: "give the configuration a name" }); return; }
  const r = await post("/api/vehicle/save", { name, changes: $("f-changes").value });
  setMsg("vs-msg", r);
  if (r.ok) { S.selectConfig = name; refresh(); }
});
$("vs-delete").addEventListener("click", async () => {
  const name = $("vs-config").value;
  if (!name || !confirm(`Delete the configuration “${name}” (lapsim/configs)? The changes stay in the form.`)) return;
  const r = await post("/api/vehicle/delete", { name });
  setMsg("vs-msg", r);
  if (r.ok) { setConfigSelects(""); refresh(); }
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const fd = new FormData(form);
  const req = {
    backend: backend(), mission: fd.get("mission"), seed: fd.get("seed") || -1, rtf: fd.get("rtf") || 1,
    changes: fd.get("changes") || "", longitudinal: S.lon, policy: fd.get("policy") || "auto",
    rviz: form.elements.rviz.checked, headless: form.elements.headless.checked,
    setup_name: $("f-config").value || "",
    perception: backend() === "demo" ? "none" : S.percep, cone_model: $("f-cone-model").value || "",
    slam: form.elements.slam.checked, line: $("line-wrap").hidden ? "center" : fd.get("line"),
    race_after: $("race-wrap").hidden ? 0 : Number(fd.get("race_after") || 0),
    faults: fd.get("faults") || "", handling: fd.get("handling") || "single_track",
    record_cones: backend() === "isaac" && form.elements.record_cones.checked,
  };
  $("btn-start").disabled = true;
  const r = await post("/api/sim/start", req);
  setMsg("sim-msg", r);
  S.wantDash = r.ok && S.state?.settings?.open_dashboard;
  refresh();
});
const stopSim = async () => { setMsg("sim-msg", await post("/api/sim/stop")); refresh(); };
$("btn-stop").addEventListener("click", stopSim);
$("hero-stop").addEventListener("click", stopSim);
$("btn-router").addEventListener("click", async () => { setMsg("sim-msg", await post("/api/router/start")); refresh(); });

function setMsg(id, r) {
  const el = $(id);
  el.textContent = r.message || "";
  el.classList.toggle("err", !r.ok);
}

// ------------------------------------------------------------------ tools
document.querySelectorAll("form.tool").forEach((f) => f.addEventListener("submit", async (e) => {
  e.preventDefault();
  const params = {};
  for (const el of f.elements) {
    if (!el.name) continue;
    if (el.type === "checkbox") {
      if (el.name === "events") { (params.events ||= []); if (el.checked) params.events.push(el.value); }
      else if (el.value && el.value !== "on") { if (el.checked) params[el.name] = el.value; }
      else params[el.name] = el.checked;
    } else if (el.value !== "") params[el.name] = el.value;
  }
  const btn = f.querySelector("button[type=submit]");
  btn.disabled = true;
  const r = await post("/api/job/start", { type: f.dataset.job, params });
  btn.disabled = false;
  toast(r);
  refresh();
}));
document.querySelectorAll("[data-job-btn]").forEach((b) => b.addEventListener("click", async () => {
  if (b.dataset.jobBtn === "update" && !confirm("Install the zip over this copy of the project and build it?")) return;
  if (b.dataset.jobBtn === "shell" && !confirm("Point fsgsim, fsgrouter and fsgbuild in your WSL terminals (~/.bashrc, backed up first) to this copy of the project?")) return;
  toast(await post("/api/job/start", { type: b.dataset.jobBtn, params: JSON.parse(b.dataset.jobParams || "{}") }));
  refresh();
}));

function toast(r) {
  const el = $("sim-msg");
  el.textContent = r.message || "";
  el.classList.toggle("err", !r.ok);
  if (!r.ok) alert(r.message);
}

// ------------------------------------------------------------------ settings / quit
$("settings-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const f = e.target;
  toast(await post("/api/settings", {
    isaac_path: f.elements.isaac_path.value, isaac_python: f.elements.isaac_python.value,
    update_zip: f.elements.update_zip.value, win_python: f.elements.win_python.value,
    dashboard_port: Number(f.elements.dashboard_port.value) || 8080,
    open_dashboard: f.elements.open_dashboard.checked,
  }));
  refresh();
});
// ------------------------------------------------------------------ GitHub / GitLab
function gitFields() {
  const f = $("git-form").elements;
  return { git_github: f.git_github.value.trim(), git_gitlab: f.git_gitlab.value.trim(), git_branch: f.git_branch.value.trim(),
           git_name: f.git_name.value.trim(), git_email: f.git_email.value.trim() };
}
async function gitSave() {
  const r = await post("/api/settings", gitFields());
  if (!r.ok) toast(r);
  return r.ok;
}
$("git-form").addEventListener("submit", (e) => e.preventDefault());   // Enter in a field must not reload the page
$("git-save").addEventListener("click", async () => { toast(await post("/api/settings", gitFields())); refresh(); });
$("git-check").addEventListener("click", async () => {
  if (!(await gitSave())) return;
  toast(await post("/api/job/start", { type: "git_check" }));
  refresh();
});
$("git-push").addEventListener("click", async () => {
  const g = gitFields();
  const msg = $("git-form").elements.git_message.value.trim();
  const where = [g.git_github && "GitHub", g.git_gitlab && "GitLab"].filter(Boolean).join(" and ");
  if (!where) { toast({ ok: false, message: "set the GitHub and / or GitLab repository first" }); return; }
  if (!confirm(`Commit all changes${msg ? ` ("${msg}")` : ""} and push to ${where} (branch ${g.git_branch || "main"})?`)) return;
  if (!(await gitSave())) return;
  toast(await post("/api/job/start", { type: "git_push", params: { message: msg } }));
  refresh();
});

$("btn-quit").addEventListener("click", async () => {
  if (!confirm("Stop the simulation, all jobs and the launcher?")) return;
  const r = await post("/api/quit");
  document.body.innerHTML = `<main class="lgrid"><section class="panel"><h2>Launcher stopped</h2><p>${esc(r.message)}</p>
    <p class="hint">Start it again with run.bat.</p></section></main>`;
});

// ------------------------------------------------------------------ copy / share
async function copyText(text, btn) {
  let ok = false;
  try { await navigator.clipboard.writeText(text); ok = true; } catch (e) {
    const ta = document.createElement("textarea");
    ta.value = text; ta.style.position = "fixed"; ta.style.opacity = "0";
    document.body.appendChild(ta); ta.select();
    try { ok = document.execCommand("copy"); } catch (e2) { ok = false; }
    ta.remove();
  }
  if (btn) {
    const old = btn.textContent;
    btn.textContent = ok ? "Copied" : "Copy failed";
    btn.classList.toggle("copied", ok);
    setTimeout(() => { btn.textContent = old; btn.classList.remove("copied"); }, 1600);
  }
  return ok;
}
$("btn-report-copy").addEventListener("click", async (e) => {
  const txt = await (await fetch("/api/report?n=200", { cache: "no-store" })).text();
  const ok = await copyText(txt, e.currentTarget);
  $("share-msg").textContent = ok ? `Report copied (${Math.round(txt.length / 1024)} kB): paste it into the chat.` : "Could not copy: use Save instead.";
});
$("btn-report-save").addEventListener("click", async () => {
  const r = await post("/api/report/save");
  $("share-msg").textContent = r.message;
  $("share-msg").classList.toggle("err", !r.ok);
});
$("btn-wincheck").addEventListener("click", async () => { toast(await post("/api/winpython/check")); setTimeout(refresh, 6000); });
$("log-copy").addEventListener("click", (e) => copyText($("log-text").textContent, e.currentTarget));

// ------------------------------------------------------------------ logs
async function showLog(key, title) {
  S.logKey = key;
  $("log-title").textContent = title;
  $("logview").hidden = false;
  await loadLog();
}
async function loadLog() {
  if (!S.logKey) return;
  try {
    const txt = await (await fetch(`/api/log?key=${encodeURIComponent(S.logKey)}&n=1500`)).text();
    const pre = $("log-text");
    pre.textContent = txt || "(empty)";
    if ($("log-follow").checked) pre.scrollTop = pre.scrollHeight;
  } catch (e) { /* ignore */ }
}
$("log-close").addEventListener("click", () => { $("logview").hidden = true; S.logKey = null; });
$("logview").addEventListener("click", (e) => { if (e.target.id === "logview") $("log-close").click(); });
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !$("logview").hidden) $("log-close").click(); });

function procItem(p, withLinks) {
  const dur = p.started ? fmtDur((p.ended || Date.now() / 1000) - p.started) : "";
  return `<li><span class="t">${esc(p.title)}<span class="badge ${esc(p.state)}">${esc(p.state)}</span></span>
    <span class="pbtns"><button type="button" class="btn" data-log="${esc(p.key)}" data-title="${esc(p.title)}">Log</button>
    <button type="button" class="btn" data-copylog="${esc(p.key)}" title="Copy the complete log">Copy</button>
    ${withLinks && p.links.length && p.state !== "running" ? `<button type="button" class="btn" data-copyres="${esc(p.key)}" title="Copy the result tables">Copy results</button>` : ""}
    ${p.state === "running" ? `<button type="button" class="btn" data-stop="${esc(p.key)}">Stop</button>` : ""}</span>
    <span class="last">${esc(dur)}${dur && p.last ? " · " : ""}${esc(p.last)}</span>
    ${withLinks && p.links.length ? `<span class="links">${p.links.map((l) => `<a href="${esc(l.href)}" target="_blank" rel="noopener">${esc(l.label)}</a>`).join("")}</span>` : ""}</li>`;
}
document.addEventListener("click", async (e) => {
  const lg = e.target.closest("[data-log]");
  if (lg) return showLog(lg.dataset.log, lg.dataset.title);
  const cl = e.target.closest("[data-copylog]");
  if (cl) {
    const txt = await (await fetch(`/api/log?key=${encodeURIComponent(cl.dataset.copylog)}&n=5000`, { cache: "no-store" })).text();
    return copyText(txt, cl);
  }
  const cr = e.target.closest("[data-copyres]");
  if (cr) {
    const txt = await (await fetch(`/api/results?key=${encodeURIComponent(cr.dataset.copyres)}`, { cache: "no-store" })).text();
    return copyText(txt || "(no result tables)", cr);
  }
  const st = e.target.closest("[data-stop]");
  if (st) { toast(await post("/api/proc/stop", { key: st.dataset.stop })); refresh(); }
});

// ------------------------------------------------------------------ render
const CHECKS = [
  ["ros", "ROS 2 Jazzy", "sudo apt install ros-jazzy-desktop"],
  ["zenoh", "Zenoh RMW", "sudo apt install ros-jazzy-rmw-zenoh-cpp"],
  ["built", "Workspace built", "builds by itself before the first start"],
  ["router", "Zenoh router", "starts with the simulation (or the button below)"],
  ["isaac", "Isaac Sim found", "set its folder under Maintenance → Settings"],
  ["relay", "Isaac relay (port 47800)", "open while isaac.launch.py runs"],
  ["dashboard", "Dashboard", "open while a simulation runs"],
  ["torch", "PyTorch in WSL", "optional (the Windows Python is used for GPU jobs)"],
  ["winpy", "Windows Python (GPU jobs)", "set it under Settings"],
  ["gpu", "GPU memory", "no NVIDIA GPU visible from WSL"],
  ["yolo", "Camera cones (YOLO)", "no detector yet"],
];

function render(d) {
  S.state = d;
  const st = d.status, sess = d.session || {}, procs = d.procs || [];
  $("ver").textContent = `Simulator ${d.version}`;
  $("repo").textContent = d.repo_windows || d.repo;
  $("conn").textContent = "Connected";
  // fill selects once / when lists change
  const mis = $("f-mission");
  if (!mis.options.length) {
    mis.innerHTML = d.missions.map((m) => `<option value="${m}">${m[0].toUpperCase() + m.slice(1).replace("_", " ")}</option>`).join("");
    mis.value = "trackdrive";
  }
  const cfgSig = d.configs.map((c) => c.name + "=" + (c.changes || []).join(" ")).join("|");
  if ($("f-config").dataset.sig !== cfgSig) {
    const cur = $("f-config").value;
    const opts = `<option value="">LapSim baseline (lr26_params.m)</option>` +
      d.configs.map((c) => `<option value="${esc(c.name)}">${esc(c.name)}</option>`).join("");
    document.querySelectorAll(".cfg-select").forEach((sel) => { sel.innerHTML = opts; });
    setConfigSelects(S.selectConfig && d.configs.some((c) => c.name === S.selectConfig) ? S.selectConfig
      : d.configs.some((c) => c.name === cur) ? cur : "");
    S.selectConfig = null;
    $("f-config").dataset.sig = cfgSig;
  }
  const pi = d.perception || {};
  const cmSig = (pi.models || []).map((m) => m.name).join("|");
  if ($("f-cone-model").dataset.sig !== cmSig) {
    const cur = $("f-cone-model").value;
    $("f-cone-model").innerHTML = `<option value="">Newest</option>` +
      (pi.models || []).map((m) => `<option value="${esc(m.name)}">${esc(m.name)}${m.mAP50 != null ? ` · mAP50 ${m.mAP50.toFixed(2)}` : ""}</option>`).join("");
    if ((pi.models || []).some((m) => m.name === cur)) $("f-cone-model").value = cur;
    $("f-cone-model").dataset.sig = cmSig;
  }
  updateFormVisibility();
  $("cone-data-line").textContent = `${pi.images || 0} images recorded in ${pi.sessions || 0} runs` +
    ((pi.images || 0) < 300 ? " (record at least a few hundred: Simulation → Isaac Sim → Record camera images)." : ".");
  const polSig = d.policies.map((p) => p.name).join("|");
  if ($("f-policy").dataset.sig !== polSig) {
    const opts = (uniOnly, auto) => (auto ? `<option value="auto">Automatic (trained for the setup, else universal)</option><option value="universal">Universal</option>` : "") +
      d.policies.filter((p) => !uniOnly || p.universal).map((p) => `<option value="${esc(p.name)}">${esc(p.name)}${p.universal ? " (universal)" : ""}</option>`).join("");
    $("f-policy").innerHTML = opts(false, true);
    document.querySelectorAll(".pol-select").forEach((s) => {
      const uni = s.classList.contains("uni-only");
      s.innerHTML = (uni ? `<option value="">— new policy —</option>` : "") + opts(uni, false);
      if (!uni && d.policies.some((p) => p.name === "universal")) s.value = "universal";
    });
    $("f-policy").dataset.sig = polSig;
  }
  if (!S.filled) {
    const f = $("settings-form").elements;
    f.isaac_path.value = d.settings.isaac_path || "";
    f.isaac_python.value = d.settings.isaac_python || "";
    f.update_zip.value = d.settings.update_zip || "";
    f.win_python.value = d.settings.win_python || "";
    f.dashboard_port.value = d.settings.dashboard_port;
    f.open_dashboard.checked = !!d.settings.open_dashboard;
    const g = $("git-form").elements;
    for (const k of ["git_github", "git_gitlab", "git_branch", "git_name", "git_email"]) g[k].value = d.settings[k] || "";
    S.filled = true;
    // a simulation started earlier (another tab, before a reload): show its options
    const o0 = (d.session || {}).opts;
    if (o0 && o0.backend) {
      const r = form.querySelector(`input[name="backend"][value="${o0.backend}"]`);
      if (r) r.checked = true;
      if (o0.mission) $("f-mission").value = o0.mission;
      $("f-changes").value = (o0.changes || []).join(" ");
      if (o0.setup_name && d.configs.some((c) => c.name === o0.setup_name)) setConfigSelects(o0.setup_name);
      fillForm(chList());
      scheduleCheck(0);
      if (o0.seed >= 0) form.elements.seed.value = o0.seed;
      form.elements.rtf.value = o0.rtf || 1;
      form.elements.rviz.checked = o0.rviz !== false;
      form.elements.headless.checked = !!o0.headless;
      S.lon = o0.longitudinal === "rl" ? "rl" : "reference";
      document.querySelectorAll("[data-lon]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.lon === S.lon)));
      if (o0.policy) $("f-policy").value = o0.policy;
      setPercep(["yolo", "lidar", "fusion", "camera"].includes(o0.perception) ? o0.perception : "none");
      form.elements.slam.checked = o0.slam !== false;
      $("f-line").value = o0.line === "racing" ? "racing" : "center";
      $("f-handling").value = o0.handling === "double_track" ? "double_track" : "single_track";
      $("f-race-after").value = String(Math.min(2, o0.race_after || 0));
      $("f-faults").value = o0.faults || "";
      form.elements.record_cones.checked = !!o0.record_cones;
    }
    updateFormVisibility();
  }
  $("zip-path").textContent = st.update_zip || "Downloads/fsg-dv-sim.zip";

  // system checks
  $("checks").innerHTML = CHECKS.map(([k, label, hint]) => {
    let ok = !!st[k];
    let cls = ok ? "good" : ["router", "relay", "dashboard", "torch", "gpu"].includes(k) ? "none" : "warning";
    let extra = "";
    if (k === "torch") { extra = st.torch === null ? "checking…" : st.torch ? (st.cuda ? `${st.torch.split(" ")[0]}, CUDA` : `${st.torch.split(" ")[0]}, no CUDA`) : hint; }
    if (k === "winpy") {
      const w = d.win_python || {};
      ok = !!(w.path && w.torch && !(w.missing || []).length);
      cls = ok ? "good" : w.path ? "warning" : "none";
      extra = !w.checked ? "checking…" : !w.path ? hint
        : `${w.path} · Python ${w.python || "?"} · ` + (w.torch ? `torch ${w.torch}${w.cuda ? ` · CUDA: ${w.gpu}` : " · no CUDA"}` : "no PyTorch")
          + ((w.missing || []).length ? ` · missing ${w.missing.join(", ")}: Maintenance → Install packages for GPU jobs` : "");
    }
    if (k === "gpu") { ok = !!(st.gpu && st.gpu.total_mb); cls = ok ? "good" : "none"; }
    if (k === "yolo") {
      const pi = d.perception || {}, ms = pi.models || [];
      const last = ms.slice().sort((a, b) => String(a.trained).localeCompare(String(b.trained))).pop();
      ok = ms.length > 0;
      cls = ok ? (pi.runtime === "" ? "warning" : "good") : "none";
      extra = (ok ? `${ms.length} detector${ms.length > 1 ? "s" : ""}, newest ${last.name}` + (last.mAP50 != null ? ` (mAP50 ${last.mAP50.toFixed(2)})` : "")
        : "no detector yet") + ` · ${pi.images || 0} training images in ${pi.sessions || 0} recordings` +
        (pi.runtime ? ` · ${pi.runtime}` : pi.runtime === "" ? " · no ONNX Runtime in WSL (Maintenance → Install ONNX Runtime)" : "");
    }
    if (k === "gpu" && ok) {
      const g = st.gpu, frac = g.used_mb / g.total_mb;
      if (frac >= 0.85) cls = "warning";
      extra = `${g.name} · ${(g.used_mb / 1024).toFixed(1)} of ${(g.total_mb / 1024).toFixed(1)} GB in use`
        + (g.util != null ? ` · ${g.util} % busy` : "")
        + (frac >= 0.85 ? " · little left for GPU training: close other GPU programs (Isaac Sim, games, video)" : "");
    }
    if (k === "isaac" && !st.windows) { cls = "warning"; extra = "no Windows interop from this WSL"; }
    if (k === "built" && st.building) { ok = false; cls = "busy"; extra = "building now (see Jobs)"; }
    else if (k === "built" && ok && st.needs_build) { cls = "warning"; extra = "sources changed: rebuilds before the next start"; }
    const icon = cls === "good" ? "✓" : cls === "warning" ? "!" : cls === "busy" ? "" : "–";
    return `<li><span class="st ${cls}"><i aria-hidden="true">${icon}</i></span><span>${esc(label)}${!ok || extra ? `<small>${esc(extra || hint)}</small>` : ""}</span></li>`;
  }).join("");
  $("btn-router").disabled = st.router;

  // running services
  const svc = procs.filter((p) => p.kind === "service" && (p.state === "running" || Date.now() / 1000 - (p.ended || 0) < 600));
  $("procs").innerHTML = svc.length ? svc.map((p) => procItem(p, false)).join("") : `<li class="empty">Nothing running.</li>`;
  const jobs = procs.filter((p) => p.kind === "job").reverse();
  $("jobs").innerHTML = jobs.length ? jobs.map((p) => procItem(p, true)).join("") : `<li class="empty">No jobs yet.</li>`;

  // session / hero
  const simRunning = procs.some((p) => (p.key === "sim" || p.key === "isaac") && p.state === "running");
  const state = sess.state === "starting" ? "starting" : simRunning ? "running" : sess.state === "failed" ? "failed" : "idle";
  $("hero").dataset.state = state;
  const o = sess.opts || {};
  const what = o.backend === "isaac" ? "Isaac Sim" : o.backend === "lite" ? "Lite simulator" : o.backend === "demo" ? "Offline demo" : "";
  $("hero-title").textContent = state === "running" ? `${what} running` : state === "starting" ? `Starting ${what}…` : state === "failed" ? "Start failed" : "Ready";
  $("hero-sub").textContent = state === "idle" ? "Choose how to run the simulator and press Start."
    : `${o.mission || ""}${o.changes && o.changes.length ? " · " + o.changes.join(" ") : " · LapSim baseline"} · ${o.longitudinal === "rl" ? "RL driver" : "reference driver"}`
      + (state === "failed" ? " — see the steps below and the logs." : "");
  const dashUp = st.dashboard && simRunning;
  for (const id of ["hero-dash", "dash-link"]) { $(id).hidden = !dashUp; $(id).href = d.dashboard_url; }
  const simUp = procs.some((p) => p.key === "sim" && p.state === "running");
  $("vs-running").hidden = !simUp;
  if (simUp) {
    $("vs-running").innerHTML = "A simulation is running: this setup is used at the next start." +
      (dashUp ? ` To change the running car, use the <a href="${esc(d.dashboard_url)}#setup" target="_blank" rel="noopener">dashboard's Vehicle setup</a>.` : "");
  }
  $("hero-stop").hidden = !simRunning;
  $("btn-stop").disabled = !simRunning && state !== "starting";
  $("btn-start").disabled = simRunning || state === "starting";
  if (dashUp && S.wantDash) {
    S.wantDash = false;
    window.open(d.dashboard_url, "_blank", "noopener");
  }
  $("steps").innerHTML = (sess.steps || []).map((s) => `<li class="${esc(s.kind)}"><time>${fmtTime(s.t)}</time><span>${esc(s.text)}</span></li>`).join("");
  if (S.logKey) loadLog();
}

async function refresh() {
  try {
    const r = await fetch("/api/state", { cache: "no-store" });
    render(await r.json());
  } catch (e) {
    $("conn").textContent = "Launcher not reachable - start run.bat";
    $("hero").dataset.state = "failed";
    $("hero-title").textContent = "Launcher stopped";
    $("hero-sub").textContent = "Start it again with run.bat.";
  }
}

refresh();
setInterval(refresh, 1500);
showPage();
loadVehicle();
loadMatlab();
