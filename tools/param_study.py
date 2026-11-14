#!/usr/bin/env python3
"""Parameter studies: how a vehicle parameter changes LapSim's prediction AND the driverless simulation.

For every parameter value and event it runs
  * LapSim (QSS, port of the Performance department's model) with the event's driver utilisation
    from performance_lr26.yaml, i.e. the number the Performance documentation would give,
  * the DV target: the same QSS model at the reference driver's grip margin (what it aims for),
  * the closed-loop driverless simulation (fsdsim dynamics in performance mode + reference driver),
and writes a CSV, a JSON and a self-contained HTML report with charts.

    python3 tools/param_study.py --param m --values 260,284,310
    python3 tools/param_study.py --param mu --scale 0.9,1.0,1.1            # factors on the baseline
    python3 tools/param_study.py --param c_z --values 3.5,4.15,4.8 --events acceleration skidpad autocross trackdrive
    python3 tools/param_study.py --table                                    # Table 2.1 style: +-10 % on mu, m, i_trans, h_cog, c_z, c_w
    python3 tools/param_study.py --param i_trans --values 12,14.5,17,19.5 --no-sim   # LapSim only (seconds)

Base changes for every run: --set "m-50" "mux*1.02". Parameter names: LapSim fields (m, mux, muy,
c_z, c_w, i_trans, T_motor, P_max, h_cog, aero_bal_f, brake_bias, ...), the aliases of LapSim's
lts_set_param (mu, mass, gear, cd/cw, cl/cz, cog, power, cop/aero_balance, torque) or vehicle.yaml
fields (yaw_inertia, actuators.max_steer, ...).

Every study also writes matlab_study.m: the same cases for LR26_LapSim in MATLAB (on the FSA 2025
tracks, with points). For acceleration and skidpad, the "LapSim" column here equals MATLAB's result.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import html
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ros2_ws", "src", "fsdsim"))

import numpy as np  # noqa: E402

from fsdsim.config import load_all_configs  # noqa: E402
from fsdsim.control import make_driver, make_vehicle_driver  # noqa: E402
from fsdsim.mission import ASState  # noqa: E402
from fsdsim.perf import PerfParams, qss  # noqa: E402
from fsdsim.perf.params import ALIASES  # noqa: E402
from fsdsim.sim import Simulation  # noqa: E402

CONFIG = os.path.join(ROOT, "ros2_ws", "src", "fsd_sim_bringup", "config")
EVENTS = ["acceleration", "skidpad", "autocross", "trackdrive"]
TIMEOUT = {"acceleration": 40, "skidpad": 90, "autocross": 120, "trackdrive": 700}
TABLE_PARAMS = ["mu", "m", "i_trans", "h_cog", "c_z", "c_w"]
EVENT_LABEL = {"acceleration": "Acceleration (75 m)", "skidpad": "Skidpad (timed lap)",
               "autocross": "Autocross (1 lap)", "trackdrive": "Trackdrive (10 laps)"}


# --------------------------------------------------------------------------- #
# one run
# --------------------------------------------------------------------------- #
def _qss_event(perf, event, track, perf_brake=None):
    """LapSim-style event time on the simulator's own track (QSS)."""
    if event == "acceleration":
        r = qss.acceleration(perf)
        return r.time, r
    if event == "skidpad":
        r = qss.skidpad(perf)
        return r.time, r
    ev = "endurance" if event == "trackdrive" and "endurance" in (perf.events or {}) else event
    r = qss.run_path(perf, track.centerline, track.closed, event=ev, veh_brake=perf_brake)
    if event == "trackdrive":
        t = r.lap_times[0] + (track.laps - 1) * r.lap_times[1]
    else:
        t = r.lap_times[0]                  # standing-start lap
    return t, r


def run_case(case: dict) -> dict:
    """case: {label, changes, event, seed, grip_util, long_util, sim}"""
    t0 = time.time()
    veh, sens, simcfg = load_all_configs(case["config"], vehicle_overrides=case["changes"])
    simcfg.sim.mission = case["event"]
    if case.get("seed") is not None:
        simcfg.events.track_generator.seed = case["seed"]
    sim = Simulation(veh, sens, simcfg, enable_lidar=False, enable_camera=False)
    perf = veh.perf()
    out = {k: case[k] for k in ("label", "value", "event")}
    out["changes"] = " ".join(case["changes"])
    if perf is None:
        raise SystemExit("vehicle.yaml has no performance (LapSim) parameters: set model: performance")
    t_lap, r_lap = _qss_event(perf, case["event"], sim.track)
    u, ul = case["grip_util"], case["long_util"]
    dv_perf = perf.copy(util_x=u, util_y=u, events={})
    t_dv, r_dv = _qss_event(dv_perf, case["event"], sim.track, dv_perf.copy(util_x=u * ul))
    out.update(lapsim_time=t_lap, lapsim_vmax=r_lap.v_max * 3.6, dv_target_time=t_dv)
    v_trace = r_lap.v_first if len(r_lap.v_first) else r_lap.v      # first lap = standing start, like the sim
    out["qss_trace"] = [np.round(r_lap.s, 2).tolist(), np.round(v_trace * 3.6, 2).tolist()]
    if not case["sim"]:
        out["wall"] = time.time() - t0
        return out

    rl = case.get("driver") == "rl"
    if rl:
        drv, out["driver"] = make_vehicle_driver(veh, case["event"], "map", sim.track.width, longitudinal="rl",
                                                 policy=case.get("policy", "auto"), grip_util=u, long_util=ul)
        rl = hasattr(drv, "idle")
    else:
        drv, out["driver"] = make_driver(veh, case["event"], "map", sim.track.width, grip_util=u, long_util=ul), \
            "reference driver"
    drv.set_reference(sim.track.centerline, sim.track.closed)
    path = drv.follower.path
    seg = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    dt, vmax, beta_max = 0.02, 0.0, 0.0
    trace_s, trace_v, s_off, last_i, s0 = [], [], 0.0, 0, None
    while sim.t < TIMEOUT[case["event"]]:
        sim.step(dt)
        s = sim.vehicle.state
        st = sim.mission.status(sim.t)
        if rl and sim.mission.state != ASState.DRIVING:
            drv.idle()
        if sim.mission.state in (ASState.READY, ASState.DRIVING):
            if rl:
                v = sim.vehicle
                steer, acc, info = drv.update(s.pose, s.vx, stop=st["mission_complete"],
                                              state={"vy": s.vy, "r": s.r, "ax": v.ax_body, "ay": v.ay_body})
            else:
                steer, acc, info = drv.update(s.pose, s.vx, stop=st["mission_complete"])
            sim.set_command(steer, acc)
            if sim.mission.state == ASState.DRIVING:
                vmax = max(vmax, s.vx)
                beta_max = max(beta_max, abs(math.degrees(math.atan2(s.vy, max(s.vx, 0.5)))))
                # speed trace from the start line on (where LapSim's distance starts), first lap
                if sim.mission.start_time is not None:
                    i = drv.follower.idx
                    if s0 is None:
                        s0, last_i = seg[i], i
                    if sim.track.closed and i < last_i - len(path) // 2:
                        s_off += seg[-1]
                    last_i = i
                    if len(trace_s) == 0 or sim.t - trace_s[-1][0] >= 0.1:
                        trace_s.append((sim.t, s_off + seg[i] - s0))
                        trace_v.append(s.vx * 3.6)
        if sim.mission.state in (ASState.FINISHED, ASState.EMERGENCY):
            break
    r = sim.result_dict()
    if case["event"] == "autocross" and r["lap_times"]:
        sim_time = r["lap_times"][0]
    else:
        sim_time = r["raw_time"]
    out.update(sim_time=sim_time, sim_final=r["final_time"], state=r["state"], cones=r["cones_hit"],
               off_course=r["off_course"], sim_vmax=vmax * 3.6, beta_max=beta_max, laps=r["laps"],
               lap_times=r["lap_times"], wall=time.time() - t0)
    out["sim_trace"] = [[round(x[1], 2) for x in trace_s], [round(v, 2) for v in trace_v]]
    return out


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
COLORS = {"lapsim": "#2a78d6", "dv": "#1baf7a", "sim": "#eb6834"}


def svg_lines(series, xlabel, ylabel, width=400, height=260, xcat=None):
    """Tiny SVG line chart. series: [(name, color, [(x, y)], dashed)]."""
    pts = [p for _, _, s, _ in series for p in s if p[1] is not None and math.isfinite(p[1])]
    if not pts:
        return "<p class='muted'>no data</p>"
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    if x1 == x0:
        x0, x1 = x0 - 1, x1 + 1
    pad = (y1 - y0) * 0.12 or abs(y1) * 0.05 or 1.0
    y0, y1 = y0 - pad, y1 + pad
    L, R, T, B = 58, 16, 14, 42
    X = lambda x: L + (x - x0) / (x1 - x0) * (width - L - R)  # noqa: E731
    Y = lambda y: T + (1 - (y - y0) / (y1 - y0)) * (height - T - B)  # noqa: E731
    step = _nice((y1 - y0) / 4)
    g = [f'<svg viewBox="0 0 {width} {height}" class="chart" role="img" aria-label="{html.escape(ylabel)} over {html.escape(xlabel)}">']
    v = math.ceil(y0 / step) * step
    while v <= y1:
        g.append(f'<line x1="{L}" x2="{width-R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" class="grid"/>'
                 f'<text x="{L-6}" y="{Y(v)+4:.1f}" class="tick" text-anchor="end">{_fmt(v, step)}</text>')
        v += step
    xticks = sorted(set(xs)) if xcat is None and len(set(xs)) <= 9 else None
    if xticks is None:
        xs_step = _nice((x1 - x0) / 5)
        xv = math.ceil(x0 / xs_step) * xs_step
        xticks = []
        while xv <= x1 + 1e-9:
            xticks.append(xv)
            xv += xs_step
    for xv in xticks:
        g.append(f'<text x="{X(xv):.1f}" y="{height-B+16}" class="tick" text-anchor="middle">{_fmt(xv, None)}</text>')
    g.append(f'<text x="{(L + width - R)/2:.0f}" y="{height-6}" class="axis" text-anchor="middle">{html.escape(xlabel)}</text>')
    g.append(f'<text x="14" y="{T + (height-T-B)/2:.0f}" class="axis" text-anchor="middle" '
             f'transform="rotate(-90 14 {T + (height-T-B)/2:.0f})">{html.escape(ylabel)}</text>')
    for name, color, s, dashed in series:
        s = [p for p in s if p[1] is not None and math.isfinite(p[1])]
        if not s:
            continue
        d = " ".join(f"{'M' if i == 0 else 'L'}{X(x):.1f},{Y(y):.1f}" for i, (x, y) in enumerate(s))
        dash = ' stroke-dasharray="6 4"' if dashed else ""
        g.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2"{dash}/>')
        if len(s) <= 12:
            for x, y in s:
                g.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="4" fill="{color}" stroke="var(--panel)" '
                         f'stroke-width="2"><title>{html.escape(name)}: {y:.3f} at {x:g}</title></circle>')
    g.append("</svg>")
    return "".join(g)


def _nice(raw):
    if raw <= 0:
        return 1.0
    p = 10 ** math.floor(math.log10(raw))
    m = raw / p
    return (1 if m < 1.5 else 2 if m < 3 else 5 if m < 7 else 10) * p


def _fmt(v, step):
    if step is not None and step >= 1:
        return f"{v:.0f}"
    return f"{v:.3g}" if abs(v) < 1000 else f"{v:.0f}"


def legend():
    return ('<ul class="legend"><li><i style="background:%s"></i>LapSim (event utilisation)</li>'
            '<li><i class="dash" style="color:%s"></i>DV target (QSS at the driver\'s grip margin)</li>'
            '<li><i style="background:%s"></i>Closed-loop DV simulation</li></ul>'
            % (COLORS["lapsim"], COLORS["dv"], COLORS["sim"]))


def write_report(out_dir, title, param, rows, args, xlabel):
    events = [e for e in EVENTS if any(r["event"] == e for r in rows)]
    labels = list(dict.fromkeys(r["label"] for r in rows))
    numeric = all(isinstance(r["value"], (int, float)) for r in rows)
    parts = []
    # summary table
    head = "".join(f"<th colspan='{3 if args.sim else 1}'>{html.escape(EVENT_LABEL[e])}</th>" for e in events)
    sub = "".join(("<th>LapSim</th><th>DV target</th><th>Sim</th>" if args.sim else "<th>LapSim</th>") for _ in events)
    body = []
    base_label = labels[len(labels) // 2] if args.table is False else None
    for lab in labels:
        cells = []
        for e in events:
            r = next((x for x in rows if x["label"] == lab and x["event"] == e), None)
            if r is None:
                cells.append("<td></td>" * (3 if args.sim else 1))
                continue
            cells.append(f"<td class='num'>{r['lapsim_time']:.3f}</td>")
            if args.sim:
                cells.append(f"<td class='num'>{r['dv_target_time']:.3f}</td>")
                st = r.get("sim_time")
                note = []
                if r.get("cones"):
                    note.append(f"{r['cones']} cone{'s' if r['cones'] > 1 else ''}")
                if r.get("off_course"):
                    note.append(f"{r['off_course']} off")
                if r.get("state") not in ("FINISHED",):
                    note.append(r.get("state", "").lower())
                cells.append(f"<td class='num'>{'–' if st is None else f'{st:.3f}'}"
                             f"{'<small> ' + ', '.join(note) + '</small>' if note else ''}</td>")
        mark = " class='base'" if lab == base_label else ""
        body.append(f"<tr{mark}><th scope='row'>{html.escape(lab)}</th>{''.join(cells)}</tr>")
    parts.append(f"<section><h2>Event times (s)</h2><div class='scroll'><table><thead><tr><th rowspan='2'>{html.escape(xlabel)}</th>"
                 f"{head}</tr><tr>{sub}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"
                 "<p class='muted'>LapSim = Performance model with the event's driver utilisation (performance_lr26.yaml, "
                 f"events). DV target = the same model at the reference driver's margin (lateral {args.grip_util:.2f}, "
                 f"braking {args.grip_util*args.long_util:.2f}). Sim = closed loop: dynamic vehicle (performance mode), "
                 + ("pure pursuit steering with the learned (RL) throttle / brake policy - "
                    + html.escape(sorted({r.get('driver', '') for r in rows if r.get('driver')})[0] if any(r.get('driver') for r in rows) else '')
                    if getattr(args, "driver", "reference") == "rl" else "reference driver")
                 + ", timing and penalties as in the DV rules; penalties are listed, not added.</p></section>")
    # charts: time vs parameter per event
    if numeric and len(labels) > 1:
        charts = []
        for e in events:
            rr = sorted([r for r in rows if r["event"] == e], key=lambda r: r["value"])
            series = [("LapSim", COLORS["lapsim"], [(r["value"], r["lapsim_time"]) for r in rr], False)]
            if args.sim:
                series.append(("DV target", COLORS["dv"], [(r["value"], r["dv_target_time"]) for r in rr], True))
                series.append(("Sim", COLORS["sim"], [(r["value"], r.get("sim_time")) for r in rr], False))
            charts.append(f"<figure><figcaption>{html.escape(EVENT_LABEL[e])}</figcaption>"
                          f"{svg_lines(series, xlabel, 'time (s)')}</figure>")
        parts.append(f"<section><h2>Time vs {html.escape(xlabel)}</h2>{legend()}<div class='grid'>{''.join(charts)}</div></section>")
    # speed traces for the circuit events
    for e in [x for x in events if x in ("autocross", "trackdrive", "acceleration")]:
        rr = [r for r in rows if r["event"] == e]
        if not rr:
            continue
        palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
        pick = rr if len(rr) <= 3 else [rr[0], rr[len(rr) // 2], rr[-1]]
        series = []
        for i, r in enumerate(pick):
            c = palette[i % len(palette)]
            s, v = r["qss_trace"]
            series.append((f"LapSim {r['label']}", c, list(zip(s, v))[::2], True))
            if args.sim and r.get("sim_trace") and r["sim_trace"][0]:
                ss, vv = r["sim_trace"]
                lap = r["qss_trace"][0][-1] if e != "acceleration" else 1e9
                sel = [(x, y) for x, y in zip(ss, vv) if x <= lap]
                series.append((f"Sim {r['label']}", c, sel, False))
        key = "".join(f"<li><i style='background:{palette[i % len(palette)]}'></i>{html.escape(r['label'])}</li>"
                      for i, r in enumerate(pick))
        parts.append(f"<section><h2>Speed over distance, {html.escape(EVENT_LABEL[e].split(' (')[0].lower())} "
                     f"(first lap)</h2><ul class='legend'>{key}<li><i class='dash' style='color:var(--muted)'></i>"
                     f"LapSim</li><li><i style='background:var(--muted)'></i>Sim</li></ul>"
                     f"<figure class='wide'>{svg_lines(series, 'distance (m)', 'speed (km/h)', 1100, 320)}</figure></section>")
    css = """
:root{--bg:#f5f6f7;--panel:#fff;--ink:#15181c;--ink2:#48505a;--muted:#67707a;--line:#d0d5da;--grid:#e4e7ea}
@media (prefers-color-scheme:dark){:root{--bg:#16191d;--panel:#1f2328;--ink:#eef0f2;--ink2:#b3b9c0;--muted:#8a929b;--line:#343a42;--grid:#2c3238}}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 "Bahnschrift","DIN Alternate","Barlow","Segoe UI",sans-serif;font-variant-numeric:tabular-nums}
main{max-width:1180px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:1.8rem;margin:0 0 4px} h2{font-size:1.1rem;margin:0 0 12px}
section{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:16px 18px;margin-top:16px}
table{border-collapse:collapse;width:100%;font-size:.92rem} th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
thead th{color:var(--muted);font-weight:500} td.num{text-align:right;font-weight:600} small{color:var(--muted);font-weight:400}
tr.base th,tr.base td{background:color-mix(in srgb,var(--panel) 85%,#2a78d6)} .scroll{overflow-x:auto}
.muted{color:var(--muted);font-size:.88rem;max-width:90ch} .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:12px}
figure{margin:0} figcaption{font-weight:600;margin-bottom:4px} .chart{width:100%;height:auto;display:block}
.chart .grid{stroke:var(--grid)} .chart .tick{fill:var(--muted);font-size:11px} .chart .axis{fill:var(--ink2);font-size:12px}
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;list-style:none;padding:0;margin:0 0 10px;color:var(--ink2);font-size:.88rem}
.legend li{display:inline-flex;align-items:center;gap:6px} .legend i{display:inline-block;width:18px;height:3px;border-radius:2px}
.legend i.dash{height:3px;background:repeating-linear-gradient(90deg,currentColor 0 6px,transparent 6px 10px)!important}
"""
    meta = (f"{len(rows)} runs, {_dt.datetime.now():%Y-%m-%d %H:%M}. Base changes: {' '.join(args.set) or 'none'}. "
            f"Track seed: {args.seed if args.seed is not None else 'sim.yaml'}.")
    doc = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
           f"content='width=device-width,initial-scale=1'><title>{html.escape(title)}</title><style>{css}</style></head>"
           f"<body><main><h1>{html.escape(title)}</h1><p class='muted'>{html.escape(meta)}</p>{''.join(parts)}</main></body></html>")
    with open(os.path.join(out_dir, "report.html"), "w", encoding="utf-8") as f:
        f.write(doc)


def write_table_report(out_dir, rows, args):
    """Table 2.1 style: event time change for +-10 % of each parameter."""
    events = [e for e in EVENTS if any(r["event"] == e for r in rows)]
    lines = ["<section><h2>Sensitivity: change of event time for +10 % / -10 % (s)</h2><div class='scroll'><table><thead><tr>"
             "<th rowspan='2'>Parameter</th>" + "".join(f"<th colspan='{2 if args.sim else 1}'>{html.escape(EVENT_LABEL[e])}</th>" for e in events)
             + "</tr><tr>" + "".join(("<th>LapSim s/%</th><th>Sim s/%</th>" if args.sim else "<th>LapSim s/%</th>") for _ in events)
             + "</tr></thead><tbody>"]
    for p in TABLE_PARAMS:
        cells = []
        for e in events:
            get = lambda lab, k: next((r.get(k) for r in rows if r["label"] == lab and r["event"] == e), None)  # noqa: E731
            for key in (["lapsim_time", "sim_time"] if args.sim else ["lapsim_time"]):
                hi, lo = get(f"{p} +10%", key), get(f"{p} -10%", key)
                cells.append("<td class='num'>–</td>" if hi is None or lo is None else f"<td class='num'>{(hi - lo) / 20:+.4f}</td>")
        lines.append(f"<tr><th scope='row'>{html.escape(p)}</th>{''.join(cells)}</tr>")
    lines.append("</tbody></table></div><p class='muted'>Seconds per percent change of the parameter (central difference "
                 "over +-10 %). Negative: the event gets faster when the parameter increases.</p></section>")
    return "".join(lines)


# --------------------------------------------------------------------------- #
MATLAB_EVENTS = {"acceleration": "acceleration", "skidpad": "skidpad", "autocross": "autocross",
                 "trackdrive": "endurance"}


def matlab_change(change: str) -> str:
    """One simulator change ('m=300', 'mu*1.1', 'c_z+0.5') as MATLAB code for LR26_LapSim."""
    from fsdsim.perf.lapsim import matlab_change as _mc  # noqa: PLC0415

    return _mc(change)


def write_matlab_study(out_dir: str, variants, base_changes, events) -> str:
    """matlab_study.m: the same variants in LR26_LapSim (MATLAB), results to matlab_results.csv."""
    evs = [MATLAB_EVENTS[e] for e in events]
    lines = [
        "% The same cases as this parameter study, in LR26_LapSim (MATLAB). Generated by tools/param_study.py.",
        "% In MATLAB:  cd <LR26_LapSim folder>; startup_lapsim; run('<this file>')",
        "% Writes matlab_results.csv next to this file. The simulator's 'trackdrive' is LapSim's 'endurance'.",
        "if ~exist('lr26_params', 'file') || ~exist('lts_run_all', 'file')",
        "    error('LR26_LapSim is not on the path: cd to the LR26_LapSim folder and run startup_lapsim first.');",
        "end",
        "comp = competition_fsa2025();",
        f"events = {{{', '.join(repr(e) for e in evs)}}};",
        "here = fileparts(mfilename('fullpath'));",
        "rows = {};",
    ]
    for label, _val, ch in variants:
        lines.append(f"\n% {label}")
        lines.append("veh = lr26_params();")
        lines += [matlab_change(c) for c in list(base_changes) + list(ch)]
        lines.append("[~, tbl] = lts_run_all(veh, comp, struct('events', {events}));")
        lines.append(f"rows(end+1, :) = {{{label!r}, tbl}};  %#ok<SAGROW>")
    lines += [
        "",
        "out = table();",
        "for i = 1:size(rows, 1)",
        "    t = rows{i, 2};",
        "    t.Variant = repmat(string(rows{i, 1}), height(t), 1);",
        "    out = [out; t(:, [end, 1:end-1])];  %#ok<AGROW>",
        "end",
        "disp(out(:, {'Variant', 'Event', 'Time_s', 'Points'}));",
        "writetable(out, fullfile(here, 'matlab_results.csv'));",
        "fprintf('wrote %s\\n', fullfile(here, 'matlab_results.csv'));",
    ]
    path = os.path.join(out_dir, "matlab_study.m")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def matlab_one_liner(param: str, values, mode: str, base_changes, events) -> str | None:
    """lts_sweep call for a plain one-parameter study (no base changes), else None."""
    if base_changes or not all(f in PerfParams().numeric_fields() for f in ALIASES.get(param.lower(), (param,))):
        return None
    evs = ", ".join(repr(MATLAB_EVENTS[e]) for e in events)
    vals = " ".join(f"{v:g}" for v in values)
    return (f"S = lts_sweep(lr26_params(), competition_fsa2025(), '{param}', [{vals}], '{mode}', "
            f"struct('events', {{{{{evs}}}}})); lts_plot_sweep(S)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--param", help="parameter to vary (LapSim name, alias or vehicle.yaml field)")
    ap.add_argument("--values", help="comma separated absolute values")
    ap.add_argument("--scale", help="comma separated factors on the baseline value (e.g. 0.9,1,1.1)")
    ap.add_argument("--table", action="store_true", help=f"+-10 %% on {', '.join(TABLE_PARAMS)} (Table 2.1 style)")
    ap.add_argument("--events", nargs="+", default=["acceleration", "skidpad", "autocross"], choices=EVENTS)
    ap.add_argument("--set", nargs="*", default=[], help="base changes for all runs, e.g. m-50 mux*1.02")
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--seed", type=int, default=None, help="track generator seed (autocross / trackdrive)")
    ap.add_argument("--grip-util", type=float, default=0.80, help="reference driver: share of the lateral grip")
    ap.add_argument("--long-util", type=float, default=0.70, help="reference driver: extra factor on longitudinal grip")
    ap.add_argument("--no-sim", dest="sim", action="store_false", help="LapSim / QSS only (fast)")
    ap.add_argument("--driver", choices=["reference", "rl"], default="reference",
                    help="who works throttle and brake in the simulation: the reference driver or the RL policy "
                         "(a universal policy drives every setup inside its training range)")
    ap.add_argument("--policy", default="auto", help="with --driver rl: auto, universal, or a policy name")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default="", help="output folder (default results/<param>_<time>)")
    args = ap.parse_args()

    if args.table:
        variants = [("baseline", None, [])]
        for p in TABLE_PARAMS:
            variants += [(f"{p} -10%", None, [f"{p}*0.9"]), (f"{p} +10%", None, [f"{p}*1.1"])]
        param, xlabel = "table", "variant"
    else:
        if not args.param or not (args.values or args.scale):
            ap.error("give --param with --values or --scale, or use --table")
        param = args.param
        if args.values:
            vals = [float(v) for v in args.values.split(",")]
            variants = [(f"{param} = {v:g}", v, [f"{param}={v!r}"]) for v in vals]
            xlabel = param
        else:
            vals = [float(v) for v in args.scale.split(",")]
            variants = [(f"{param} x {v:g}", v, [f"{param}*{v!r}"]) for v in vals]
            xlabel = f"{param} (factor on baseline)"

    out_dir = args.out or os.path.join(ROOT, "results", f"{param}_{_dt.datetime.now():%Y%m%d-%H%M%S}")
    os.makedirs(out_dir, exist_ok=True)
    cases = [{"label": lab, "value": val if val is not None else lab, "changes": list(args.set) + ch, "event": ev,
              "seed": args.seed, "grip_util": args.grip_util, "long_util": args.long_util, "sim": args.sim,
              "driver": args.driver, "policy": args.policy,
              "config": args.config}
             for lab, val, ch in variants for ev in args.events]
    print(f"[study] {len(cases)} runs ({len(variants)} variants x {len(args.events)} events), {args.jobs} workers -> {out_dir}")
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for r in ex.map(run_case, cases):
            rows.append(r)
            sim = f"  sim {r['sim_time']:.3f}" if r.get("sim_time") is not None else ("  sim DNF" if args.sim else "")
            pen = f" ({r['cones']} cones, {r['off_course']} off)" if r.get("cones") or r.get("off_course") else ""
            print(f"  {r['label']:22s} {r['event']:12s} LapSim {r['lapsim_time']:8.3f}  DV target {r['dv_target_time']:8.3f}"
                  f"{sim}{pen}", flush=True)
    print(f"[study] done in {time.time() - t0:.0f} s")

    keys = ["label", "value", "event", "changes", "lapsim_time", "lapsim_vmax", "dv_target_time", "sim_time",
            "sim_final", "state", "cones", "off_course", "sim_vmax", "beta_max", "laps", "wall"]
    with open(os.path.join(out_dir, "results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k in keys})
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "rows": rows}, f)
    title = "Sensitivity table (+-10 %)" if args.table else f"Parameter study: {param}"
    write_report(out_dir, title, param, rows, args, xlabel)
    if args.table:
        rep = os.path.join(out_dir, "report.html")
        doc = open(rep, encoding="utf-8").read()
        doc = doc.replace("<section>", write_table_report(out_dir, rows, args) + "<section>", 1)
        open(rep, "w", encoding="utf-8").write(doc)
    print(f"[study] {os.path.join(out_dir, 'report.html')}")

    mfile = write_matlab_study(out_dir, variants, args.set, args.events)
    print(f"[study] same cases in MATLAB (after cd LR26_LapSim; startup_lapsim):  run('{windows_path(mfile)}')")
    if not args.table:
        one = matlab_one_liner(param, vals, "abs" if args.values else "scale", args.set, args.events)
        if one:
            print(f"[study] or as a LapSim sweep:  {one}")


def windows_path(path: str) -> str:
    """The path as Windows (MATLAB) sees a WSL file: \\\\wsl.localhost\\<distro>\\..."""
    path = os.path.abspath(path)
    if path.startswith("/mnt/") and len(path) > 6 and path[6] == "/":
        return f"{path[5].upper()}:" + path[6:].replace("/", "\\")
    distro = os.environ.get("WSL_DISTRO_NAME")
    return f"\\\\wsl.localhost\\{distro}" + path.replace("/", "\\") if distro else path


if __name__ == "__main__":
    main()
