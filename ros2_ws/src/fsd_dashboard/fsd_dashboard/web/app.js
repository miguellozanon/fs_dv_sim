// DV dashboard client: one EventSource stream, canvas map and charts, no libraries.
"use strict";

const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

const STATE_KEY = { 1: "off", 2: "ready", 3: "driving", 4: "emergency", 5: "finished" };
const ASSI_TEXT = {
  off: "ASSI off", ready: "ASSI yellow, steady", driving: "ASSI yellow, flashing",
  emergency: "ASSI blue, flashing", finished: "ASSI blue, steady",
};
const MISSION_KEY = { 1: "acceleration", 2: "skidpad", 3: "trackdrive", 4: "ebs_test", 5: "inspection", 6: "autocross" };
const CHART_SPAN = 30; // s

const S = {
  connected: false, track: null, cones: [], trail: [], lidar: null, car: null, pose: null,
  status: {}, tele: [], laps: [], images: {}, cam: "left", camSeq: -1, camLoading: false,
  view: "track", showLidar: true, hoverT: null, mapDirty: true, chartsDirty: true, bounds: null,
  missionSynced: false,
};

// ------------------------------------------------------------------ theme
const themeBtn = $("theme-btn");
function currentTheme() {
  const forced = document.documentElement.dataset.theme;
  if (forced) return forced;
  return matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
}
function applyTheme(t) {
  document.documentElement.dataset.theme = t;
  try { localStorage.setItem("fsd-theme", t); } catch (e) { /* storage unavailable */ }
  themeBtn.textContent = t === "dark" ? "Switch to light theme" : "Switch to dark theme";
  S.mapDirty = S.chartsDirty = true;
}
try { const t = localStorage.getItem("fsd-theme"); if (t) document.documentElement.dataset.theme = t; } catch (e) { /* ignore */ }
themeBtn.textContent = currentTheme() === "dark" ? "Switch to light theme" : "Switch to dark theme";
themeBtn.addEventListener("click", () => applyTheme(currentTheme() === "dark" ? "light" : "dark"));
matchMedia("(prefers-color-scheme: light)").addEventListener("change", () => { S.mapDirty = S.chartsDirty = true; });

// ------------------------------------------------------------------ stream
function connect() {
  const es = new EventSource("events");
  es.onmessage = (m) => { S.connected = true; onDelta(JSON.parse(m.data)); };
  es.onerror = () => { S.connected = false; renderAssi(); };
}

function onDelta(d) {
  if (d.reset) { S.tele = []; S.trail = []; }
  if (d.track) { S.track = d.track; S.bounds = null; S.mapDirty = true; }
  if (d.cones) { S.cones = d.cones; S.bounds = null; S.mapDirty = true; }
  if (d.trail) { S.trail = d.trail; S.mapDirty = true; }
  else if (d.trail_tail && d.trail_tail.length) {
    // the server sends its last few path points; append the ones after our newest point
    const last = S.trail[S.trail.length - 1];
    const tail = d.trail_tail;
    const at = last ? tail.findIndex((p) => p[0] === last[0] && p[1] === last[1]) : -1;
    const fresh = at >= 0 ? tail.slice(at + 1) : tail;
    if (fresh.length) { S.trail.push(...fresh); S.mapDirty = true; }
  }
  if (d.lidar) { S.lidar = decodeLidar(d.lidar); S.mapDirty = true; }
  if (d.pose) { S.pose = d.pose; S.mapDirty = true; }
  S.car = d.car || S.car;
  if (d.telemetry) {
    if (S.tele.length && d.telemetry[0][0] + 1 < S.tele[S.tele.length - 1][0]) S.tele = [];
    S.tele.push(...d.telemetry);
    const tEnd = S.tele[S.tele.length - 1][0];
    let cut = 0;
    while (cut < S.tele.length && S.tele[cut][0] < tEnd - 62) cut++;
    if (cut) S.tele.splice(0, cut);
    S.chartsDirty = true;
  }
  S.status = d.status || {};
  S.laps = d.laps || [];
  S.images = d.images || {};
  S.perception = d.perception || null;
  if (d.pcones) { S.pcones = d.pcones; S.mapDirty = true; }
  renderAssi(); renderControls(); renderVitals(d); renderLaps(); renderPrediction(d); renderHealth(d.health || []);
  if (d.events) renderEvents(d.events);
  renderCamera();
  noteSetupVersion(d.setup_version);
  $("source").textContent = d.source || "";
  $("clock").textContent = `Simulation time ${fmt(d.t, 1)} s`;
}

function decodeLidar(l) {
  const bin = atob(l.data);
  const buf = new ArrayBuffer(bin.length);
  const u8 = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return { n: l.n, res: l.res, pts: new Int16Array(buf) };
}

// ------------------------------------------------------------------ helpers
function fmt(v, nd = 1) { return v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toFixed(nd); }
const DEG = 180 / Math.PI;

// ------------------------------------------------------------------ ASSI band + controls
function renderAssi() {
  const band = $("assi");
  const st = S.status || {};
  let key = STATE_KEY[st.state] || "none";
  if (!S.connected) key = "offline";
  band.dataset.state = key;
  const word = { offline: "Not connected", none: "Waiting", off: "AS off" }[key] || st.state_name;
  $("state-word").textContent = word;
  let detail;
  if (key === "offline") detail = "The dashboard server is not reachable. Check that the launch is still running.";
  else if (key === "none") detail = "Connected. Waiting for a simulator or the car to report its AS state.";
  else {
    const msg = st.message && st.message !== `AS ${st.state_name}` ? `${st.message}. ` : "";
    detail = `${msg}${ASSI_TEXT[key]}.`;
  }
  $("state-detail").textContent = detail;
  $("lamp").setAttribute("aria-label", ASSI_TEXT[key] || "ASSI unknown");
  $("run-mission").textContent = st.mission_name && st.mission ? st.mission_name : "–";
  $("run-lap").textContent = st.laps_target ? `${Math.min(st.lap_count, st.laps_target)} of ${st.laps_target}` : (st.lap_count ? String(st.lap_count) : "–");
  $("run-time").textContent = st.state ? `${fmt(st.mission_time, 1)} s` : "–";
  const pen = st.penalty_time || 0;
  $("run-penalty").textContent = st.state ? (pen > 0 ? `+${fmt(pen, 0)} s` : "None") : "–";
}

const ENABLED = {
  off: ["select_mission", "activate", "reset"],
  ready: ["go", "emergency", "reset"],
  driving: ["emergency", "reset"],
  emergency: ["reset"],
  finished: ["reset"],
};
function renderControls() {
  const key = S.connected ? STATE_KEY[S.status.state] : null;
  const allowed = new Set(ENABLED[key] || []);
  document.querySelectorAll(".controls button[data-cmd]").forEach((b) => { b.disabled = !allowed.has(b.dataset.cmd); });
  $("mission-select").disabled = key !== "off";
  if (!S.missionSynced && S.status.mission && MISSION_KEY[S.status.mission]) {
    $("mission-select").value = MISSION_KEY[S.status.mission];
    S.missionSynced = true;
  }
}
document.querySelectorAll(".controls button[data-cmd]").forEach((b) => {
  b.addEventListener("click", async () => {
    const body = { cmd: b.dataset.cmd };
    if (body.cmd === "select_mission") body.mission = $("mission-select").value;
    const msg = $("controls-msg");
    msg.textContent = `${b.textContent}…`;
    try {
      const r = await fetch("api/command", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      const j = await r.json();
      msg.textContent = j.ok ? `${b.textContent}: ${j.message || "done"}` : `${b.textContent} refused: ${j.message}`;
    } catch (e) {
      msg.textContent = `${b.textContent} failed: the dashboard server did not answer.`;
    }
  });
});

// ------------------------------------------------------------------ vitals, laps, health, log
function renderVitals(d) {
  const p = d.pose, v = d.vehicle || {}, imu = d.imu || {};
  $("v-speed").textContent = p ? fmt(Math.hypot(p.vx, p.vy) * 3.6, 0) : "–";
  const steer = v.steering;
  $("v-steer").textContent = steer === undefined || steer === null ? "–" : `${fmt(steer * DEG, 1)}°`;
  const bar = $("v-steer-bar");
  if (steer !== undefined && steer !== null) {
    const frac = Math.min(Math.abs(steer) / 0.45, 1) * 50;
    bar.style.width = `${frac}%`;
    bar.style.left = steer > 0 ? `${50 - frac}%` : "50%"; // left steer drawn to the left
  }
  $("v-ay").textContent = imu.ay === undefined ? "–" : `${fmt(imu.ay, 1)} m/s²`;
  $("v-r").textContent = imu.r === undefined ? "–" : `${fmt(imu.r * DEG, 0)} °/s`;
  const rpm = v.wheel_rpm || [];
  ["w-fl", "w-fr", "w-rl", "w-rr"].forEach((id, i) => { $(id).textContent = rpm[i] === undefined ? "–" : fmt(rpm[i], 0); });
}

function renderLaps() {
  const st = S.status || {};
  const laps = S.laps;
  const best = laps.length ? Math.min(...laps.map((l) => l.time)) : null;
  $("l-best").textContent = best === null ? "–" : `${fmt(best, 3)} s`;
  $("l-cones").textContent = st.state ? String(st.cones_hit ?? 0) : "–";
  $("l-off").textContent = st.state ? String(st.off_course ?? 0) : "–";
  const body = $("lap-rows");
  const key = laps.map((l) => `${l.lap}:${l.time}`).join("|");
  if (body.dataset.key === key) return;
  body.dataset.key = key;
  if (!laps.length) { body.innerHTML = '<tr class="placeholder"><td colspan="3">No completed laps yet.</td></tr>'; return; }
  const bestIdx = laps.findIndex((l) => l.time === best);
  body.innerHTML = laps.map((l, i) => {
    const dlt = l.time - best;
    return `<tr class="${i === bestIdx ? "best" : ""}"><td>${l.lap}</td><td class="num">${fmt(l.time, 3)}</td>` +
      `<td class="num">${dlt === 0 ? "–" : "+" + fmt(dlt, 3)}</td></tr>`;
  }).join("");
}

function renderPrediction(d) {
  const p = d.prediction || {}, vi = d.vehicle_info || {};
  const box = $("pred");
  if (!p.lapsim) { box.hidden = true; return; }
  box.hidden = false;
  const circuit = p.event === "autocross" || p.event === "trackdrive";
  const show = (x) => {
    if (!x) return "–";
    if (!circuit) return `${fmt(x.time, 3)} s`;
    if (p.event === "trackdrive") return `${fmt(x.time, 1)} s <small>${fmt(x.flying_lap, 2)} s per lap</small>`;
    return `${fmt(x.first_lap, 2)} s`;
  };
  $("p-lapsim").innerHTML = show(p.lapsim);
  $("p-dv").innerHTML = show(p.dv_target);
  $("p-dv-label").textContent = `DV target (${Math.round((p.grip_util || 0) * 100)} % grip)`;
  const lu = p.lapsim_util;
  $("p-lapsim-label").textContent = lu === undefined ? "LapSim"
    : lu >= 0.999 ? "LapSim, tyre limit" : `LapSim, driver at ${Math.round(lu * 100)} % grip`;
  const ch = (vi.changes || []).join(" ");
  $("p-veh").textContent = vi.name
    ? `Vehicle ${vi.name}, ${vi.model === "performance" ? "Performance (LapSim) model" : vi.model}; changes: ${ch || "none"}.`
    : "";
}

const ST_ICON = { good: "✓", warning: "!", critical: "✕", none: "" };
function renderHealth(rows) {
  const html = rows.filter((r) => !r.optional || r.seen).map((r) => {
    const age = r.level === "critical" && r.age !== null ? `, ${fmt(r.age, 0)} s ago` : "";
    return `<tr class="${r.seen ? "" : "dim"}"><td>${r.label}<span class="topic">${r.topic}</span></td>` +
      `<td class="num">${r.rate === null ? "–" : fmt(r.rate, r.rate < 20 ? 1 : 0)}</td>` +
      `<td class="num">${r.expected ? fmt(r.expected, 0) : "–"}</td>` +
      `<td class="num">${r.delay_ms == null ? "–" : r.delay_ms}</td>` +
      `<td><span class="st ${r.level}"><i aria-hidden="true">${ST_ICON[r.level]}</i>${r.text}${age}</span></td></tr>`;
  }).join("");
  const body = $("health-rows");
  if (body.dataset.html !== html) { body.innerHTML = html; body.dataset.html = html; }
}

function renderEvents(evs) {
  const list = $("log");
  for (const e of evs) {
    const li = document.createElement("li");
    li.className = e.kind;
    const t = document.createElement("time");
    t.textContent = `${fmt(e.t, 1)} s`;
    const s = document.createElement("span");
    s.textContent = e.text;
    li.append(t, s);
    list.prepend(li);
  }
  while (list.children.length > 200) list.lastChild.remove();
}

// ------------------------------------------------------------------ camera
document.querySelectorAll("[data-cam]").forEach((b) => b.addEventListener("click", () => {
  S.cam = b.dataset.cam; S.camSeq = -1;
  document.querySelectorAll("[data-cam]").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
  renderCamera();
}));
const camImg = $("cam");
camImg.addEventListener("load", () => { S.camLoading = false; camImg.hidden = false; $("cam-empty").hidden = true; });
camImg.addEventListener("error", () => { S.camLoading = false; });
function renderCamera() {
  const info = S.images[S.cam];
  if (!info) {
    camImg.hidden = true; $("cam-empty").hidden = false;
    $("cam-meta").textContent = S.cam === "depth" && S.images.left ? "No depth stream (start Isaac with --camera-streams left,depth)." : "";
    return;
  }
  $("cam-meta").textContent = S.cam === "depth"
    ? `Depth, ${info.w}×${info.h} preview. Lighter is nearer; 40 m and beyond is darkest.`
    : `Left image, ${info.w}×${info.h} preview.`;
  renderYolo(info);
  if (S.camLoading || info.seq === S.camSeq) return;
  S.camSeq = info.seq; S.camLoading = true;
  camImg.src = `api/image/${S.cam}?s=${info.seq}`;
}

// camera cone perception (YOLO): its boxes on the colour image, its speed and accuracy under it
const YOLO_COL = ["--cone-blue", "--cone-yellow", "--cone-orange", "--cone-orange"];
const YOLO_NAME = ["blue", "yellow", "orange", "big orange"];
function renderYolo(info) {
  const P = S.perception, svg = $("cam-boxes"), meta = $("yolo-meta");
  const d = P && P.dets, st = P && P.stats, mp = P && P.map;
  const mm = $("map-meta");
  mm.hidden = !mp;
  if (mp) {
    // cone map and SLAM pose of the fsd_perception node against the ground truth
    const a = mp.map || {}, src = { lidar: "LiDAR", fusion: "LiDAR + camera", camera: "camera" }[mp.mode] || mp.mode;
    const pct = (x) => `${Math.round(100 * x)} %`;
    mm.textContent = `Cone map (${src}): ${a.mapped ?? 0} cones` +
      (a.seen_share != null ? `, ${pct(a.seen_share)} of the track's ${a.truth} found` : "") +
      (a.pos_rms != null ? `, position error ${fmt(a.pos_rms, 2)} m` : "") +
      (a.false ? `, ${a.false} false` : "") + (a.colour_ok != null ? `, colour right ${pct(a.colour_ok)}` : "") +
      (mp.slam ? ` · pose from SLAM, error ${mp.pose_err == null ? "–" : fmt(mp.pose_err, 2) + " m"}` +
        (mp.pose_err_max != null ? ` (max ${fmt(mp.pose_err_max, 2)} m)` : "") : " · ground-truth pose") +
      (mp.scan_ms != null ? ` · ${fmt(mp.scan_ms, 0)} ms per scan` : "") +
      (mp.lidar_timing ? ` · LiDAR timing: best offset ${Math.round(1000 * mp.lidar_timing.best_offset)} ms ` +
        `(now ${Math.round(1000 * mp.lidar_timing.now)} ms, error ${fmt(mp.lidar_timing.rms_now, 2)} → ${fmt(mp.lidar_timing.rms_best, 2)} m)` +
        (mp.lidar_timing.bias ? `, cone offset in the car frame x ${fmt(mp.lidar_timing.bias.x, 2)} y ${fmt(mp.lidar_timing.bias.y, 2)} m, ` +
          `along the ray ${fmt(mp.lidar_timing.bias.along_ray, 2)} m, yaw ${fmt(mp.lidar_timing.bias.yaw_deg, 1)}°` : "") : "") +
      (mp.frozen_frames ? ` · ${mp.frozen_frames} frozen frames skipped` : "");
  }
  meta.hidden = !st;
  if (st) {
    const pc = (x) => (x == null ? "–" : `${Math.round(100 * x)} %`);
    const f = st.frame || {}, m = st.map || {};
    meta.textContent = `YOLO ${st.model} (${st.backend}): ${fmt(st.fps, 1)} frames/s, ${st.ms == null ? "–" : fmt(st.ms, 0)} ms · ` +
      `found ${pc(f.recall)} of the visible cones, ${pc(f.precision)} of the boxes are cones, colour right ${pc(f.colour)}, ` +
      `position error ${f.error_m == null ? "–" : fmt(f.error_m, 2) + " m"} · map ${m.mapped ?? 0} cones` +
      (m.false != null ? ` (${m.false} false, ${m.colour_ok ?? 0} right colour)` : "");
  }
  // boxes only on the colour image, and only when they belong to (about) the frame shown
  if (!d || S.cam !== "left" || !info || Math.abs((d.t ?? 0) - (info.t ?? 0)) > 0.35) { svg.innerHTML = ""; return; }
  svg.setAttribute("viewBox", `0 0 ${d.w} ${d.h}`);
  const lw = Math.max(2, d.w / 400);
  svg.innerHTML = d.dets.map((b) => {
    const [x1, y1, x2, y2] = b.box, c = css(YOLO_COL[b.cls] || "--cone-down");
    return `<rect x="${x1}" y="${y1}" width="${x2 - x1}" height="${y2 - y1}" fill="none" stroke="${c}" stroke-width="${lw}"/>` +
      `<text x="${x1}" y="${Math.max(y1 - 4, 12)}" fill="${c}" font-size="${Math.max(12, d.w / 70)}" font-weight="600" paint-order="stroke" stroke="#000" stroke-width="3">` +
      `${YOLO_NAME[b.cls] || "?"} ${Math.round(100 * b.conf)}</text>`;
  }).join("");
}

// ------------------------------------------------------------------ map
const mapCv = $("map");
document.querySelectorAll("[data-view]").forEach((b) => b.addEventListener("click", () => {
  S.view = b.dataset.view; S.mapDirty = true;
  document.querySelectorAll("[data-view]").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
}));
$("show-lidar").addEventListener("change", (e) => { S.showLidar = e.target.checked; S.mapDirty = true; });

function fitCanvas(cv) {
  const r = cv.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr));
  if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
  return { w, h, dpr };
}

function trackBounds() {
  if (S.bounds) return S.bounds;
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  const add = (x, y) => { x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y); };
  (S.track ? S.track.centerline : []).forEach((p) => add(p[0], p[1]));
  S.cones.forEach((c) => add(c[2], c[3]));
  if (!Number.isFinite(x0)) return null;
  S.bounds = { x0, y0, x1, y1 };
  return S.bounds;
}

function drawMap() {
  const { w, h, dpr } = fitCanvas(mapCv);
  const g = mapCv.getContext("2d");
  g.setTransform(1, 0, 0, 1, 0, 0);
  g.clearRect(0, 0, w, h);
  const b = trackBounds();
  $("map-empty").hidden = !!b;
  if (!b) return;
  const pad = 24 * dpr;
  let cx, cy, s;
  if (S.view === "follow" && S.pose) {
    s = Math.min(w, h) / 45; // ~45 m across the short side
    cx = S.pose.x; cy = S.pose.y;
  } else {
    s = Math.min((w - 2 * pad) / Math.max(b.x1 - b.x0, 1), (h - 2 * pad) / Math.max(b.y1 - b.y0, 1));
    cx = (b.x0 + b.x1) / 2; cy = (b.y0 + b.y1) / 2;
  }
  const X = (x) => w / 2 + (x - cx) * s;
  const Y = (y) => h / 2 - (y - cy) * s;

  // track surface and centreline
  const cl = S.track ? S.track.centerline : [];
  if (cl.length > 1) {
    g.lineJoin = "round"; g.lineCap = "round";
    g.beginPath();
    cl.forEach((p, i) => (i ? g.lineTo(X(p[0]), Y(p[1])) : g.moveTo(X(p[0]), Y(p[1]))));
    if (S.track.closed) g.closePath();
    g.strokeStyle = css("--road");
    g.lineWidth = Math.max(4 * dpr, (S.track.width || 3) * s);
    g.stroke();
    g.setLineDash([4 * dpr, 6 * dpr]);
    g.strokeStyle = css("--line");
    g.lineWidth = 1 * dpr;
    g.stroke();
    g.setLineDash([]);
  }
  // timing / finish lines
  g.strokeStyle = css("--ink-2"); g.lineWidth = 2 * dpr;
  for (const key of ["timing_line", "finish_line"]) {
    const l = S.track && S.track[key];
    if (l) { g.beginPath(); g.moveTo(X(l[0][0]), Y(l[0][1])); g.lineTo(X(l[1][0]), Y(l[1][1])); g.stroke(); }
  }
  // driven path
  if (S.trail.length > 1) {
    g.beginPath();
    S.trail.forEach((p, i) => (i ? g.lineTo(X(p[0]), Y(p[1])) : g.moveTo(X(p[0]), Y(p[1]))));
    g.strokeStyle = css("--path"); g.lineWidth = 1.5 * dpr; g.globalAlpha = 0.8; g.stroke(); g.globalAlpha = 1;
  }
  // LiDAR: ground returns faint, object returns (cones) bright
  if (S.showLidar && S.lidar) {
    const P = S.lidar.pts, res = S.lidar.res, sz = Math.max(1.5, 1.6 * dpr);
    const ground = css("--lidar-ground"), obj = css("--lidar-obj");
    g.fillStyle = ground; g.globalAlpha = 0.55;
    for (let i = 0; i < P.length; i += 3) if (P[i + 2] * res <= 0.06) g.fillRect(X(P[i] * res) - sz / 2, Y(P[i + 1] * res) - sz / 2, sz, sz);
    g.fillStyle = obj; g.globalAlpha = 1;
    for (let i = 0; i < P.length; i += 3) if (P[i + 2] * res > 0.06) g.fillRect(X(P[i] * res) - sz / 2, Y(P[i + 1] * res) - sz / 2, sz, sz);
  }
  // cones
  const col = { 0: css("--cone-blue"), 1: css("--cone-yellow"), 2: css("--cone-orange"), 3: css("--cone-orange"), 4: css("--cone-down") };
  const ring = css("--bg");
  for (const c of S.cones) {
    const r = Math.max(2.6 * dpr, (c[1] === 3 ? 0.21 : 0.15) * s);
    g.beginPath(); g.arc(X(c[2]), Y(c[3]), r, 0, 2 * Math.PI);
    g.fillStyle = c[4] ? css("--cone-down") : col[c[1]] || col[4];
    g.fill();
    g.lineWidth = 1.5 * dpr; g.strokeStyle = ring; g.stroke();
    if (c[4]) { // knocked: slash
      g.beginPath(); g.moveTo(X(c[2]) - r, Y(c[3]) + r); g.lineTo(X(c[2]) + r, Y(c[3]) - r);
      g.strokeStyle = css("--ink"); g.lineWidth = 1.2 * dpr; g.stroke();
    }
  }
  // cones the camera (YOLO) put on its map: rings around / next to the true cones
  if (S.pcones && S.pcones.length) {
    for (const c of S.pcones) {
      const r = Math.max(4 * dpr, 0.32 * s);
      g.beginPath(); g.arc(X(c[2]), Y(c[3]), r, 0, 2 * Math.PI);
      g.lineWidth = 2 * dpr; g.strokeStyle = col[c[1]] || col[4]; g.stroke();
    }
  }
  // car
  if (S.pose) {
    const car = S.car || { front: 1.55, rear: 1.35, width: 1.4 };
    const minPx = 7 * dpr;
    const k = Math.max(1, minPx / (car.width * s));
    const f = car.front * k, rr = car.rear * k, hw = (car.width / 2) * k;
    const c = Math.cos(S.pose.yaw), sn = Math.sin(S.pose.yaw);
    const pt = (lx, ly) => [X(S.pose.x + c * lx - sn * ly), Y(S.pose.y + sn * lx + c * ly)];
    const poly = [pt(f, hw), pt(f, -hw), pt(-rr, -hw), pt(-rr, hw)];
    g.beginPath(); poly.forEach((q, i) => (i ? g.lineTo(q[0], q[1]) : g.moveTo(q[0], q[1]))); g.closePath();
    g.fillStyle = css("--car"); g.fill();
    g.lineWidth = 2 * dpr; g.strokeStyle = ring; g.stroke();
    const nose = [pt(f, 0), pt(f - Math.min(1.0 * k, f), hw * 0.75), pt(f - Math.min(1.0 * k, f), -hw * 0.75)];
    g.beginPath(); nose.forEach((q, i) => (i ? g.lineTo(q[0], q[1]) : g.moveTo(q[0], q[1]))); g.closePath();
    g.fillStyle = ring; g.fill();
  }
  // scale bar
  const target = 120 * dpr / s;
  const nice = [1, 2, 5, 10, 20, 50, 100].find((v) => v >= target / 2) || 100;
  const len = nice * s;
  const bx = 14 * dpr, by = h - 14 * dpr;
  g.strokeStyle = css("--ink-2"); g.lineWidth = 2 * dpr;
  g.beginPath(); g.moveTo(bx, by - 5 * dpr); g.lineTo(bx, by); g.lineTo(bx + len, by); g.lineTo(bx + len, by - 5 * dpr); g.stroke();
  g.fillStyle = css("--ink-2");
  g.font = `${12 * dpr}px ${css("--font")}`;
  g.fillText(`${nice} m`, bx + len + 6 * dpr, by);
}

// ------------------------------------------------------------------ charts
const CHARTS = [
  { el: document.querySelector('[data-chart="speed"]'), series: [{ i: 1, k: 3.6, c: "--s1", name: "Speed", unit: "km/h", nd: 1 }], zero: true },
  { el: document.querySelector('[data-chart="steer"]'), series: [
    { i: 2, k: DEG, c: "--s1", name: "Measured", unit: "°", nd: 1 },
    { i: 3, k: DEG, c: "--s2", name: "Command", unit: "°", nd: 1, dash: true }], symmetric: true },
  { el: document.querySelector('[data-chart="accel"]'), series: [
    { i: 5, k: 1, c: "--s1", name: "IMU", unit: "m/s²", nd: 2 },
    { i: 4, k: 1, c: "--s2", name: "Command", unit: "m/s²", nd: 2, dash: true }], symmetric: true },
];
CHARTS.forEach((ch) => {
  ch.cv = ch.el.querySelector("canvas");
  ch.cv.addEventListener("mousemove", (e) => hover(ch, e));
  ch.cv.addEventListener("mouseleave", () => { S.hoverT = null; $("tip").hidden = true; S.chartsDirty = true; });
});

function niceStep(range, n) {
  const raw = range / n, p = Math.pow(10, Math.floor(Math.log10(raw))), m = raw / p;
  return (m < 1.5 ? 1 : m < 3 ? 2 : m < 7 ? 5 : 10) * p;
}

function chartGeom(ch) {
  const { w, h, dpr } = fitCanvas(ch.cv);
  const L = 40 * dpr, R = 104 * dpr, T = 8 * dpr, B = 18 * dpr;
  const tEnd = S.tele.length ? S.tele[S.tele.length - 1][0] : 0;
  return { w, h, dpr, L, R, T, B, t0: tEnd - CHART_SPAN, t1: tEnd };
}

function drawChart(ch) {
  const G = chartGeom(ch);
  const { w, h, dpr, L, R, T, B, t0, t1 } = G;
  const g = ch.cv.getContext("2d");
  g.clearRect(0, 0, w, h);
  const rows = S.tele.filter((r) => r[0] >= t0);
  let lo = Infinity, hi = -Infinity;
  for (const r of rows) for (const s of ch.series) { const v = r[s.i]; if (v !== null && v !== undefined) { lo = Math.min(lo, v * s.k); hi = Math.max(hi, v * s.k); } }
  if (!Number.isFinite(lo)) { lo = 0; hi = 1; }
  if (ch.zero) { lo = 0; hi = Math.max(hi * 1.1, 10); }
  if (ch.symmetric) { const m = Math.max(Math.abs(lo), Math.abs(hi), 1) * 1.15; lo = -m; hi = m; }
  const step = niceStep(hi - lo, 3);
  lo = Math.floor(lo / step) * step; hi = Math.ceil(hi / step) * step;
  const X = (t) => L + (t - t0) / CHART_SPAN * (w - L - R);
  const Y = (v) => T + (1 - (v - lo) / (hi - lo)) * (h - T - B);
  ch.geom = { ...G, X, Y, lo, hi };

  g.font = `${11 * dpr}px ${css("--font")}`;
  g.fillStyle = css("--muted");
  g.strokeStyle = css("--grid");
  g.lineWidth = 1 * dpr;
  g.textAlign = "right"; g.textBaseline = "middle";
  for (let v = lo; v <= hi + 1e-9; v += step) {
    const y = Math.round(Y(v)) + 0.5;
    g.beginPath(); g.moveTo(L, y); g.lineTo(w - R, y); g.stroke();
    g.fillText(String(+v.toFixed(2)), L - 6 * dpr, y);
  }
  if (lo < 0 && hi > 0) { g.strokeStyle = css("--line"); g.beginPath(); g.moveTo(L, Y(0)); g.lineTo(w - R, Y(0)); g.stroke(); }
  g.textAlign = "center"; g.textBaseline = "top";
  for (let k = 0; k <= CHART_SPAN; k += 10) g.fillText(k === 0 ? "now" : `−${k} s`, X(t1 - k), h - B + 4 * dpr);

  const labels = [];
  for (const s of ch.series) {
    g.beginPath();
    let started = false, last = null;
    for (const r of rows) {
      const v = r[s.i];
      if (v === null || v === undefined) { started = false; continue; }
      const x = X(r[0]), y = Y(v * s.k);
      if (!started) { g.moveTo(x, y); started = true; } else g.lineTo(x, y);
      last = [x, y, v * s.k];
    }
    g.strokeStyle = css(s.c); g.lineWidth = 2 * dpr; g.lineJoin = "round";
    g.setLineDash(s.dash ? [6 * dpr, 4 * dpr] : []);
    g.stroke(); g.setLineDash([]);
    if (last) labels.push({ y: last[1], text: `${s.name} ${fmt(last[2], s.nd)}`, c: css("--ink-2") });
  }
  // direct labels at the line ends, nudged apart
  labels.sort((a, b) => a.y - b.y);
  for (let i = 1; i < labels.length; i++) if (labels[i].y - labels[i - 1].y < 13 * dpr) labels[i].y = labels[i - 1].y + 13 * dpr;
  g.textAlign = "left"; g.textBaseline = "middle";
  for (const lb of labels) { g.fillStyle = lb.c; g.fillText(ch.series.length > 1 ? lb.text : lb.text.split(" ").pop(), w - R + 6 * dpr, Math.min(Math.max(lb.y, T + 5 * dpr), h - B - 5 * dpr)); }

  if (S.hoverT !== null && S.hoverT >= t0) {
    const x = Math.round(X(S.hoverT)) + 0.5;
    g.strokeStyle = css("--ink-2"); g.lineWidth = 1 * dpr;
    g.beginPath(); g.moveTo(x, T); g.lineTo(x, h - B); g.stroke();
    const r = nearestRow(S.hoverT);
    if (r) for (const s of ch.series) {
      const v = r[s.i]; if (v === null || v === undefined) continue;
      g.beginPath(); g.arc(X(r[0]), Y(v * s.k), 4 * dpr, 0, 2 * Math.PI);
      g.fillStyle = css(s.c); g.fill(); g.lineWidth = 2 * dpr; g.strokeStyle = css("--panel"); g.stroke();
    }
  }
}

function nearestRow(t) {
  let best = null, bd = Infinity;
  for (let i = S.tele.length - 1; i >= 0; i--) {
    const d = Math.abs(S.tele[i][0] - t);
    if (d < bd) { bd = d; best = S.tele[i]; } else if (S.tele[i][0] < t) break;
  }
  return best;
}

function hover(ch, e) {
  if (!ch.geom || !S.tele.length) return;
  const rect = ch.cv.getBoundingClientRect();
  const { dpr, L, R, w, t0 } = ch.geom;
  const px = (e.clientX - rect.left) * dpr;
  const t = t0 + Math.min(Math.max((px - L) / (w - L - R), 0), 1) * CHART_SPAN;
  S.hoverT = t; S.chartsDirty = true;
  const r = nearestRow(t);
  const tip = $("tip");
  if (!r) { tip.hidden = true; return; }
  const ago = S.tele[S.tele.length - 1][0] - r[0];
  const line = (name, v, k, nd, unit) => `<div><span>${name}</span><span>${v === null || v === undefined ? "–" : fmt(v * k, nd) + " " + unit}</span></div>`;
  tip.innerHTML = `<b>${ago < 0.05 ? "Now" : fmt(ago, 1) + " s ago"} (t = ${fmt(r[0], 2)} s)</b>` +
    line("Speed", r[1], 3.6, 1, "km/h") + line("Steering measured", r[2], DEG, 1, "°") + line("Steering command", r[3], DEG, 1, "°") +
    line("Accel. IMU", r[5], 1, 2, "m/s²") + line("Accel. command", r[4], 1, 2, "m/s²");
  tip.hidden = false;
  const host = document.querySelector(".charts").getBoundingClientRect();
  let left = e.clientX - host.left + 16;
  if (left + tip.offsetWidth > host.width - 8) left = e.clientX - host.left - tip.offsetWidth - 16;
  tip.style.left = `${left}px`;
  tip.style.top = `${e.clientY - host.top + 12}px`;
}

// ------------------------------------------------------------------ pages
function showPage() {
  const setup = location.hash === "#setup";
  $("page-live").hidden = setup;
  $("page-setup").hidden = !setup;
  $("tab-live").toggleAttribute("aria-current", !setup);
  $("tab-setup").toggleAttribute("aria-current", setup);
  if (setup) { $("tab-setup").setAttribute("aria-current", "page"); fetchSetup(); }
  else { $("tab-live").setAttribute("aria-current", "page"); S.mapDirty = S.chartsDirty = true; }
}
window.addEventListener("hashchange", showPage);

// ------------------------------------------------------------------ vehicle setup
const esc = (t) => String(t ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const SU = { data: null, version: null, fetching: false, built: "", inputs: {}, timer: null };
const EV_LABEL = { acceleration: "Acceleration", skidpad: "Skidpad", autocross: "Autocross", endurance: "Endurance" };
const NUM_RE = /^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/;

function noteSetupVersion(v) {
  if (v === undefined) return;
  if (v === null) { if (SU.version !== null || !SU.data) { SU.version = null; renderSetupOff(true); } return; }
  if (v !== SU.version) { SU.version = v; if (location.hash === "#setup" || !SU.data) fetchSetup(); }
}
function renderSetupOff(off) {
  $("setup-off").hidden = !off;
  $("setup-form").hidden = off;
  document.querySelector(".setup-side").hidden = off;
}
async function fetchSetup() {
  if (SU.fetching) { SU.again = true; return; }
  SU.fetching = true;
  try {
    const r = await fetch("api/setup", { cache: "no-store" });
    const d = await r.json();
    if (d.available === false) { renderSetupOff(true); return; }
    renderSetupOff(false);
    SU.data = d;
    renderSetup();
  } catch (e) { /* server restarting: the next delta retries */ }
  finally {
    SU.fetching = false;
    if (SU.again) { SU.again = false; fetchSetup(); }
  }
}
async function setupPost(body) {
  const msg = $("setup-msg");
  msg.classList.remove("err");
  msg.textContent = "…";
  try {
    const r = await fetch("api/setup", { method: "POST", headers: { "Content-Type": "application/json", "X-FSG-Dashboard": "1" }, body: JSON.stringify(body) });
    const d = await r.json();
    msg.textContent = d.message || (d.ok ? "Done" : "Failed");
    msg.classList.toggle("err", !d.ok);
    fetchSetup();
    return d;
  } catch (e) {
    msg.textContent = "The dashboard server did not answer.";
    msg.classList.add("err");
    return { ok: false };
  }
}

function fmtBase(v, it) {
  if (it && it.base_text) return it.base_text;
  if (v === null || v === undefined) return "–";
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e6)) return v.toExponential(3);
  return String(+v.toPrecision(6));
}

function buildGroups(schema) {
  const host = $("groups");
  const draft = readDraft();
  host.innerHTML = "";
  SU.inputs = {};
  schema.forEach((g, gi) => {
    const det = document.createElement("details");
    det.className = "pgroup";
    det.open = gi < 5;
    const tag = g.kind === "event" ? "LapSim and MATLAB only" : g.kind === "sim" ? "simulator only" : "";
    det.innerHTML = `<summary>${esc(g.title)}${tag ? ` <span class="tag">${tag}</span>` : ""}<span class="count" hidden></span></summary>
      <table class="ptable"><thead><tr><th scope="col">Parameter</th><th scope="col" class="num">Baseline</th>
      <th scope="col" class="num">New value</th><th scope="col" class="unit">Unit</th></tr></thead><tbody></tbody></table>`;
    const tb = det.querySelector("tbody");
    g.items.forEach((it) => {
      const tr = document.createElement("tr");
      const id = "p-" + it.key.replace(/\W/g, "_");
      tr.innerHTML = `<td><label for="${id}"><span class="p-label">${esc(it.label)}</span>
          <span class="p-key">${esc(it.key)}${it.desc ? " · " + esc(it.desc) : ""}</span></label></td>
        <td class="base"></td>
        <td class="inp"><input class="p-in" id="${id}" inputmode="decimal" autocomplete="off" spellcheck="false"></td>
        <td class="unit">${esc(it.unit)}</td>`;
      const inp = tr.querySelector("input");
      inp.value = draft.form?.[it.key] ?? "";
      inp.addEventListener("input", onFormInput);
      SU.inputs[it.key] = { inp, tr, det, it };
      tb.appendChild(tr);
    });
    host.appendChild(det);
  });
  $("other").value = draft.other ?? "";
  if (draft.name) $("cfg-name").value = draft.name;
}

function updateBaselines(schema) {
  schema.forEach((g) => g.items.forEach((it) => {
    const x = SU.inputs[it.key];
    if (!x) return;
    x.it = it;
    x.tr.querySelector(".base").textContent = fmtBase(it.base, it);
    x.inp.placeholder = fmtBase(it.base, it);
  }));
}

// MATLAB files and the last MATLAB results: buttons that open them on this PC
function renderFiles(files) {
  const btn = ([k, f]) => `<button type="button" class="btn small-btn" data-open="${esc(k)}" ${f.exists ? "" : "disabled"} title="${esc(f.path)}">` +
    `${esc(f.label)}${f.kind === "file" && f.name !== f.label ? ` <span class="muted">${esc(f.name)}</span>` : ""}</button>`;
  const all = Object.entries(files);
  const sig = JSON.stringify(all.map(([k, f]) => [k, f.exists, f.path]));
  if (sig === SU.filesSig) return;
  SU.filesSig = sig;
  $("files-lapsim").innerHTML = all.filter(([, f]) => f.group === "files").map(btn).join("");
  const res = all.filter(([, f]) => f.group === "results" && f.exists);
  $("files-results").innerHTML = res.length ? res.map(btn).join("") : `<span class="muted">None yet: apply a setup with “Run in MATLAB”.</span>`;
}
document.addEventListener("click", (e) => {
  const b = e.target.closest("[data-open]");
  if (b) setupPost({ action: "open", target: b.dataset.open });
});

function readDraft() {
  try { return JSON.parse(localStorage.getItem("fsd-setup-draft") || "{}"); } catch (e) { return {}; }
}
function writeDraft() {
  const form = {};
  Object.entries(SU.inputs).forEach(([k, x]) => { if (x.inp.value.trim()) form[k] = x.inp.value.trim(); });
  try { localStorage.setItem("fsd-setup-draft", JSON.stringify({ form, other: $("other").value, name: $("cfg-name").value, applied: SU.formFor ?? null })); } catch (e) { /* ignore */ }
}

function collectChanges() {
  const out = [], bad = [];
  Object.entries(SU.inputs).forEach(([k, x]) => {
    const v = x.inp.value.trim().replace(",", ".");
    x.tr.classList.remove("bad", "changed");
    if (!v) return;
    const fac = v.match(/^[x×*]\s*(.+)$/i);
    if (fac && NUM_RE.test(fac[1])) { out.push(`${k}*${fac[1]}`); x.tr.classList.add("changed"); }
    else if (NUM_RE.test(v) || /^nan$/i.test(v)) { out.push(`${k}=${/^nan$/i.test(v) ? "NaN" : v}`); x.tr.classList.add("changed"); }
    else { bad.push(k); x.tr.classList.add("bad"); }
  });
  $("other").value.split(/[\s,;]+/).map((t) => t.trim()).filter(Boolean).forEach((t) => out.push(t));
  Object.values(SU.inputs).reduce((m, x) => m.set(x.det, (m.get(x.det) || 0) + (x.tr.classList.contains("changed") ? 1 : 0)), new Map())
    .forEach((n, det) => { const c = det.querySelector(".count"); c.hidden = !n; c.textContent = `${n} changed`; });
  return { changes: out, bad };
}

function onFormInput() {
  const { changes, bad } = collectChanges();
  const line = $("changes-line");
  if (bad.length) line.innerHTML = `Check ${bad.map((k) => `<code>${esc(k)}</code>`).join(" ")}: enter a number or a factor like <code>*1.05</code>.`;
  else line.innerHTML = changes.length ? changes.map((c) => `<code>${esc(c)}</code>`).join("") : "No changes: the LapSim baseline.";
  $("btn-apply").disabled = bad.length > 0;
  $("btn-save").disabled = bad.length > 0;
  writeDraft();
}

function fillForm(changes) {
  Object.values(SU.inputs).forEach((x) => { x.inp.value = ""; });
  const rest = [];
  (changes || []).forEach((c) => {
    const m = String(c).match(/^\s*([A-Za-z_][\w.]*)\s*([=*+-])\s*(.+?)\s*$/);
    const x = m && SU.inputs[m[1]];
    if (x && m[2] === "=") x.inp.value = m[3];
    else if (x && m[2] === "*") x.inp.value = "*" + m[3];
    else rest.push(c);
  });
  $("other").value = rest.join(" ");
  onFormInput();
}

function fmtT(ev, t) { return t === null || t === undefined ? "–" : fmt(t, ev === "endurance" ? 1 : 3); }
function cell(ev, r, base, kind) {
  if (!r) return `<td class="num pending">${kind || "–"}</td>`;
  let d = "";
  if (base && r.T !== null && base.T !== null && Math.abs(r.T - base.T) > 5e-4) {
    const dt = r.T - base.T, dp = (r.points ?? 0) - (base.points ?? 0);
    d = `<small class="${dt < 0 ? "d-good" : "d-bad"}">${dt > 0 ? "+" : "−"}${fmt(Math.abs(dt), ev === "endurance" ? 1 : 3)} s · ${dp >= 0 ? "+" : "−"}${fmt(Math.abs(dp), 2)} pts</small>`;
  }
  return `<td class="num">${fmtT(ev, r.T)}<small>${r.points === null || r.points === undefined ? "" : fmt(r.points, 2) + " pts"}</small>${d}</td>`;
}

function sameChanges(a, b) { return JSON.stringify(a || []) === JSON.stringify(b || []); }

function renderSetup() {
  const d = SU.data;
  if (!d) return;
  const sig = JSON.stringify(d.schema.map((g) => [g.title, g.items.map((i) => i.key)]));
  if (sig !== SU.built) { buildGroups(d.schema); SU.built = sig; }
  updateBaselines(d.schema);
  const ls = d.lapsim;
  $("src-line").textContent = `${ls.params_file.split(/[\\/]/).slice(-3).join("/")} · ${ls.competition} · modified ${ls.modified}`;

  // saved configurations
  const sel = $("cfg-select"), keep = sel.value;
  sel.innerHTML = `<option value="">LapSim baseline (no changes)</option>` +
    d.configs.map((c) => `<option value="${esc(c.name)}">${esc(c.name)}${c.saved ? " · " + esc(c.saved) : ""}</option>`).join("");
  const want = SU.selectAfter || keep;
  if ([...sel.options].some((o) => o.value === want)) { sel.value = want; if (want === SU.selectAfter) SU.selectAfter = null; }

  // a new active setup (the launcher started the simulation with it, or it was applied here or in another tab):
  // the form shows it, instead of an older draft of this browser
  const a = d.active, port = d.port, mat = d.matlab, sim = d.sim;
  SU.formFor ??= readDraft().applied ?? null;
  if (a.applied && a.applied !== SU.formFor) {
    SU.formFor = a.applied;
    fillForm(a.changes || []);
    const named = !["custom", "set at start", "LapSim baseline"].includes(a.name);
    $("cfg-name").value = named ? a.name : "";
    if (named && d.configs.some((c) => c.name === a.name)) { sel.value = a.name; SU.selectAfter = null; }
    writeDraft();
    if (a.start) {
      $("setup-msg").classList.remove("err");
      $("setup-msg").textContent = `The simulation was started with “${a.name}” (DV launcher). Change it here and press Apply to restart the run with another car.`;
    }
  }

  // results
  $("res-for").textContent = a.applied ? `for “${a.name}”` : "LapSim baseline";
  const matOk = mat.state === "done" && sameChanges(mat.changes, a.changes);
  const matKind = !a.applied ? "–" : mat.state === "unavailable" ? "n/a"
    : ["starting", "running"].includes(mat.state) && sameChanges(mat.changes, a.changes) ? "running…"
    : mat.state === "error" && sameChanges(mat.changes, a.changes) ? "error" : "–";
  const portOk = port.state === "done" && sameChanges(port.changes, a.changes);
  const evs = (ls.events || []).filter((e) => d.baseline.events[e]);
  $("res-rows").innerHTML = evs.map((ev) => {
    const b = d.baseline.events[ev];
    const dvKey = Object.keys(d.dv_events).find((k) => d.dv_events[k] === ev);
    const dv = sim.results && dvKey ? sim.results[dvKey] : null;
    const dvCell = dv ? `<td class="num">${fmtT(ev, dv.T)}${dv.driver === "rl" ? '<span class="drv-tag" title="RL driver">RL</span>' : ""}<small>${dvKey === ev ? "" : esc(dvKey)}${dv.penalty ? ` +${fmt(dv.penalty, 0)} s pen.` : ""}</small></td>`
      : `<td class="num pending">${dvKey === ev ? "–" : "–<small>" + esc(dvKey || "") + "</small>"}</td>`;
    return `<tr><th scope="row" class="ev">${EV_LABEL[ev] || esc(ev)}<small>${esc(b.track || "")}</small></th>
      <td class="num">${fmtT(ev, b.T)}<small>${fmt(b.points, 2)} pts</small></td>
      ${portOk ? cell(ev, port.events[ev], b) : cell(ev, null, null, port.state === "running" ? "…" : "–")}
      ${matOk ? cell(ev, mat.results[ev], b) : cell(ev, null, null, matKind)}
      ${dvCell}</tr>`;
  }).join("");
  const tot = (x) => (x === null || x === undefined ? "–" : fmt(x, 2));
  $("res-foot").innerHTML = `<tr><th scope="row">Total points</th><td class="num">${tot(d.baseline.total)}</td>
    <td class="num">${portOk ? tot(port.total) : "–"}</td><td class="num">${matOk ? tot(mat.total) : "–"}</td><td></td></tr>`;

  // status
  const now = Date.now() / 1000;
  const st = (lvl, text) => `<span class="st ${lvl}"><i>${{ good: "✓", warning: "!", critical: "✕", none: "", busy: "" }[lvl]}</i><span>${esc(text)}</span></span>`;
  const portSt = port.state === "done" ? st("good", `computed in ${port.seconds} s (Python port of LR26_LapSim, identical results)`)
    : port.state === "running" ? st("busy", "computing…") : port.state === "error" ? st("critical", port.message) : st("none", "not run yet");
  let matSt;
  if (mat.state === "unavailable") matSt = st("none", mat.message);
  else if (["starting", "running"].includes(mat.state)) matSt = st("busy", `${mat.message} · ${fmt(now - (mat.started || now), 0)} s`);
  else if (mat.state === "done") matSt = st("good", `${mat.message}${sameChanges(mat.changes, a.changes) ? "" : " (for an earlier setup)"}`);
  else if (mat.state === "error") matSt = st("critical", mat.message);
  else matSt = st("none", `ready (${(mat.matlab || "").split(/[\\/]/).slice(-3, -2)[0] || "MATLAB"})`);
  const simSt = sim.state === "active" ? st("good", sim.message) : sim.state === "sent" || sim.state === "sending" ? st("busy", sim.message || "waiting for the simulator")
    : sim.state === "error" ? st("warning", sim.message) : st("none", "no setup sent yet");
  $("status-list").innerHTML = `<li><b>LapSim</b>${portSt}</li><li><b>MATLAB</b>${matSt}</li><li><b>Simulation</b>${simSt}</li>`;
  renderFiles(d.files || {});
  $("opt-matlab").disabled = mat.state === "unavailable";
  if (mat.state === "unavailable") $("opt-matlab").checked = false;
  renderRL(d);
  clearInterval(SU.timer);
  if (["starting", "running"].includes(mat.state) || d.rl?.job?.state === "running") {
    SU.timer = setInterval(() => (d.rl?.job?.state === "running" ? fetchSetup() : renderSetup()), d.rl?.job?.state === "running" ? 2000 : 1000);
  }
  onFormInput();
}

const MIS_SHORT = { acceleration: "Accel", skidpad: "Skidpad", autocross: "Autocross" };
function fmtDur(s) { if (s === null || s === undefined) return "–"; s = Math.round(s); return s >= 3600 ? `${Math.floor(s / 3600)} h ${Math.round((s % 3600) / 60)} min` : s >= 60 ? `${Math.floor(s / 60)} min ${s % 60} s` : `${s} s`; }
function renderRL(d) {
  const rl = d.rl;
  $("rl-panel").hidden = !rl;
  if (!rl) return;
  const drv = rl.driver || {};
  document.querySelectorAll("[data-lon]").forEach((b) => b.setAttribute("aria-pressed", String((drv.requested || "reference") === b.dataset.lon)));
  $("drv-desc").textContent = drv.description
    ? `Active: ${drv.description}.` : "Steering always follows the path; this chooses who works the throttle and brakes.";
  const pols = rl.policies || [];
  const m = pols.find((p) => p.name === rl.match);
  const gain = (p) => {
    const ev = p.evaluation || [];
    const ok = ev.filter((e) => e.ref !== null && e.rl !== null);
    if (!ok.length) return "";
    const dt = ok.reduce((a, e) => a + (e.rl - e.ref), 0);
    return ` (RL ${dt <= 0 ? "−" : "+"}${fmt(Math.abs(dt), 2)} s over ${ok.length} evaluation runs)`;
  };
  const u = pols.find((p) => p.name === rl.universal);
  const trainHint = `about ${rl.cpus >= 12 ? "10 min" : rl.cpus >= 6 ? "20 min" : "40 min"} for 50 generations on ${rl.cpus} CPU threads`;
  $("rl-match").innerHTML = m ? `Policy <b>${esc(m.name)}</b>, trained for this setup ${esc(m.trained || "")}${esc(gain(m))}.`
      + (u ? ` The universal policy <b>${esc(u.name)}</b> also covers it.` : "")
    : u ? `The universal policy <b>${esc(u.name)}</b> drives this setup: it is inside its training range, no training needed`
      + `${u.validation && u.validation.gain !== undefined ? ` (validation: ${u.validation.gain <= 0 ? "−" : "+"}${fmt(Math.abs(100 * u.validation.gain), 1)} % vs the reference driver)` : ""}.`
      + ` A policy trained for exactly this setup is optional: ${trainHint}.`
    : `No policy for this setup yet${pols.length ? ` (${pols.length} for other setups)` : ""}.`
      + (rl.universal_why ? ` ${esc(rl.universal_why)}.` : "") + ` Train one: ${trainHint}.`;
  // policy choice
  const sel = $("rl-policy");
  const want = drv.policy_requested || "auto";
  const opts = [["auto", "Automatic: trained for this setup, else universal"], ["universal", "Universal policy (best covering this setup)"],
    ...pols.map((p) => [p.name, `${p.name}${p.universal ? " (universal)" : ""}`])];
  const sig = opts.map((o) => o[0]).join("|");
  if (sel.dataset.sig !== sig) {
    sel.innerHTML = opts.map(([v, t]) => `<option value="${esc(v)}">${esc(t)}</option>`).join("");
    sel.dataset.sig = sig;
  }
  if (document.activeElement !== sel) sel.value = opts.some((o) => o[0] === want) ? want : "auto";
  // training job
  const job = rl.job || {}, prog = job.progress || {};
  const running = job.state === "running";
  $("btn-rl-train").disabled = running || !rl.available;
  $("btn-rl-stop").hidden = !running;
  $("rl-prog").hidden = !job.state || job.state === "idle";
  if (!$("rl-prog").hidden) {
    const g = prog.generation || 0, G = prog.generations || job.generations || 1;
    const pct = job.state === "done" ? 100 : prog.phase === "imitation" ? 3 : Math.min(99, Math.round((100 * g) / G));
    $("rl-bar").style.width = `${pct}%`;
    document.querySelector("#rl-prog .bar").setAttribute("aria-valuenow", String(pct));
    let txt;
    if (job.state === "running") {
      txt = prog.phase === "imitation" ? `“${job.name}”: imitating the reference driver…`
        : prog.phase === "evaluating" ? `“${job.name}”: evaluating on complete missions…`
        : `“${job.name}”: generation ${g} of ${G}` + (prog.best_val !== undefined ? ` · score ${fmt(prog.best_val, 1)} (reference ${fmt(prog.ref_val, 1)})` : "")
          + (prog.eta_s ? ` · about ${fmtDur(prog.eta_s)} left` : "");
    } else if (job.state === "done") {
      txt = `“${job.name}” trained${job.stopped ? " (stopped early)" : ""} in ${fmtDur((job.finished || 0) - (job.started || 0))}.`;
    } else {
      txt = `Training “${job.name}” failed.\n${job.log_tail || ""}`;
    }
    $("rl-prog-text").textContent = txt;
  }
  // policy list
  $("rl-count").textContent = `(${pols.length})`;
  const cellFor = (p, mis) => {
    const ev = (p.evaluation || []).filter((e) => e.mission === mis);
    if (!ev.length) return `<td class="num">–</td>`;
    const ok = ev.filter((e) => e.ref !== null && e.rl !== null);
    if (!ok.length) return `<td class="num d-bad">no time</td>`;
    const dt = ok.reduce((a, e) => a + (e.rl - e.ref), 0) / ok.length;
    const pen = ev.reduce((a, e) => a + (e.rl_pen || 0), 0);
    const rlT = ok.reduce((a, e) => a + e.rl, 0) / ok.length;
    return `<td class="num"><span class="${dt < 0 ? "d-good" : "d-bad"}">${dt <= 0 ? "−" : "+"}${fmt(Math.abs(dt), 3)} s</span><small>${fmt(rlT, 2)} s${pen ? ` · ${pen} pen.` : ""}</small></td>`;
  };
  $("rl-rows").innerHTML = pols.length ? pols.map((p) => `<tr><td>${esc(p.name)}${p.name === rl.match || p.name === rl.universal ? ' <span class="drv-tag">this setup</span>' : ""}<small>${esc(p.trained || "")}</small></td>
      <td>${p.universal ? `<b>Universal</b><span class="rl-uni">${esc(p.ranges || "")}</span>`
        : p.changes.length ? p.changes.map((c) => `<code>${esc(c)}</code>`).join(" ") : '<span class="muted">LapSim baseline</span>'}</td>
      ${["acceleration", "skidpad", "autocross"].map((mi) => cellFor(p, mi)).join("")}</tr>`).join("")
    : `<tr class="placeholder"><td colspan="5">No trained policies yet.</td></tr>`;
}

document.querySelectorAll("[data-lon]").forEach((b) => b.addEventListener("click", () =>
  setupPost({ action: "driver", longitudinal: b.dataset.lon, policy: $("rl-policy").value || "auto" })));
$("rl-policy").addEventListener("change", () => {
  const cur = document.querySelector('[data-lon][aria-pressed="true"]');
  setupPost({ action: "driver", longitudinal: cur ? cur.dataset.lon : "rl", policy: $("rl-policy").value || "auto" });
});
$("btn-rl-train").addEventListener("click", () =>
  setupPost({ action: "rl_train", generations: Number($("rl-gens").value) || 50 }));
$("btn-rl-stop").addEventListener("click", () => setupPost({ action: "rl_stop" }));

$("btn-apply").addEventListener("click", () => {
  const { changes, bad } = collectChanges();
  if (bad.length) return;
  setupPost({ action: "apply", changes, name: $("cfg-name").value.trim(), matlab: $("opt-matlab").checked, sim: $("opt-sim").checked });
});
$("btn-save").addEventListener("click", () => {
  const name = $("cfg-name").value.trim();
  if (!name) { $("setup-msg").textContent = "Give the configuration a name first."; $("setup-msg").classList.add("err"); $("cfg-name").focus(); return; }
  const { changes } = collectChanges();
  SU.selectAfter = name;
  setupPost({ action: "save", name, changes });
});
$("btn-load").addEventListener("click", () => {
  const name = $("cfg-select").value;
  const cfg = (SU.data?.configs || []).find((c) => c.name === name);
  fillForm(cfg ? cfg.changes : []);
  $("cfg-name").value = cfg ? cfg.name : "";
  writeDraft();
  $("setup-msg").classList.remove("err");
  $("setup-msg").textContent = cfg ? `Loaded “${cfg.name}”${cfg.notes ? ": " + cfg.notes : ""}. Press Apply to run it.` : "Form cleared (LapSim baseline).";
});
$("btn-delete").addEventListener("click", () => {
  const name = $("cfg-select").value;
  if (name) setupPost({ action: "delete", name });
});
$("btn-clear").addEventListener("click", () => { fillForm([]); $("cfg-name").value = ""; writeDraft(); });
$("btn-reload").addEventListener("click", () => setupPost({ action: "reload" }));
$("other").addEventListener("input", onFormInput);
$("cfg-name").addEventListener("input", writeDraft);

// ------------------------------------------------------------------ loop
new ResizeObserver(() => { S.mapDirty = S.chartsDirty = true; }).observe(document.body);
function frame() {
  if (S.mapDirty) { S.mapDirty = false; drawMap(); }
  if (S.chartsDirty) { S.chartsDirty = false; CHARTS.forEach(drawChart); }
  requestAnimationFrame(frame);
}
renderAssi(); renderControls();
showPage();
connect();
requestAnimationFrame(frame);
