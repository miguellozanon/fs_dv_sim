#!/usr/bin/env python3
"""Figures and numbers for the two papers (docs/paper/*.tex), computed with the project's own code.

    python3 docs/paper/figures/make_figures.py          # writes docs/paper/figures/*.pdf and numbers.json

Needs the LR26_LapSim folder (lapsim/LR26_LapSim) and matplotlib. Takes about 3 minutes.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path[:0] = [os.path.join(REPO, "ros2_ws", "src", "fsdsim"), os.path.join(REPO, "perf"), os.path.join(REPO, "tools")]

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# categorical palette (validated, light surface) and a one-hue sequential ramp
C1, C2, C3, C4, C5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
GREY = "#6b7280"
SEQ = LinearSegmentedColormap.from_list("seq", ["#cfe0f6", "#2a78d6", "#0b2f5e"])
W1, W2 = 6.3, 3.0            # full text width [in]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["LM Roman 10", "Latin Modern Roman", "DejaVu Serif"], "mathtext.fontset": "cm",
    "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9, "legend.fontsize": 7.5, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e5e7eb", "grid.linewidth": 0.6, "axes.edgecolor": "#6b7280", "axes.linewidth": 0.6,
    "lines.linewidth": 1.4, "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
NUM: dict = {}


def save(fig, name):
    fig.savefig(os.path.join(HERE, name))
    plt.close(fig)
    print("  ", name, flush=True)


def colored_line(ax, x, y, c, vmin=None, vmax=None, lw=2.2):
    pts = np.array([x, y]).T.reshape(-1, 1, 2)
    seg = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(seg, cmap=SEQ, norm=plt.Normalize(vmin if vmin is not None else np.min(c),
                                                          vmax if vmax is not None else np.max(c)))
    lc.set_array(np.asarray(c)[:-1])
    lc.set_linewidth(lw)
    ax.add_collection(lc)
    return lc


# =============================================================================================== #
# Paper I: driverless simulator
# =============================================================================================== #
def dv_figures():
    from closed_loop_eval import run
    from fsdsim.config import load_all_configs
    from fsdsim.sim import Simulation

    cfg = os.path.join(REPO, "ros2_ws", "src", "fsd_sim_bringup", "config")
    colours = {0: C1, 1: C4, 2: C2, 3: C2}

    # --- generated layouts
    fig, axs = plt.subplots(1, 4, figsize=(W1, 1.9))
    for ax, seed in zip(axs, (7, 11, 23, 42)):
        veh, sens, simcfg = load_all_configs(cfg)
        simcfg.sim.mission = "trackdrive"
        simcfg.events.track_generator.seed = seed
        sim = Simulation(veh, sens, simcfg, enable_lidar=False, enable_camera=False)
        tr = sim.track
        xy, col = tr.cone_xy(), tr.cone_colors()
        for c in (0, 1, 2, 3):
            m = col == c
            ax.scatter(xy[m, 0], xy[m, 1], s=4 if c < 3 else 9, c=colours[c], linewidths=0)
        ax.plot(tr.centerline[:, 0], tr.centerline[:, 1], color=GREY, lw=0.5, ls="--")
        ax.set_aspect("equal")
        ax.set_title(f"seed {seed}: {tr.length:.0f} m", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    save(fig, "dv_tracks.pdf")

    # --- tyre model: Pacejka shape (peak 1) and friction-ellipse remainder
    B, Cc, E = 11.0, 1.45, 0.3
    a = np.radians(np.linspace(0, 20, 400))
    ba = B * a
    sh = np.sin(Cc * np.arctan(ba - E * (ba - np.arctan(ba))))
    ipk = int(np.argmax(sh))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W1, 2.2))
    a1.plot(np.degrees(a), sh, color=C1)
    a1.axvline(np.degrees(a[ipk]), color=GREY, lw=0.8, ls=":")
    a1.annotate(f"peak at {np.degrees(a[ipk]):.1f}°", (np.degrees(a[ipk]), 1.0), xytext=(9.5, 0.62), fontsize=7.5,
                arrowprops=dict(arrowstyle="-", color=GREY, lw=0.6))
    a1.set_xlabel(r"slip angle $\alpha$ [deg]"); a1.set_ylabel(r"$F_y / p_y$ [-]")
    a1.set_title("Lateral shape (B = 11, C = 1.45, E = 0.3)")
    u = np.linspace(0, 1, 300)
    for e, c in ((1.5, C2), (2.0, C1), (3.0, C3)):
        a2.plot(u, (1 - u ** e) ** (1 / e), color=c, label=f"e = {e:g}")
    a2.set_xlabel(r"lateral utilisation $u = F_y / p_y$"); a2.set_ylabel(r"$F_{x,\mathrm{rem}} / p_x$")
    a2.set_title("Longitudinal remainder (friction ellipse)")
    a2.legend()
    save(fig, "dv_tyre.pdf")
    NUM["alpha_peak_deg"] = float(np.degrees(a[ipk]))

    # --- closed loop: autocross seed 7, map mode, centre and racing line
    runs = {}
    for line in ("center", "racing"):
        sim, res, log, _ = run("autocross", cfg, mode="map", seed=7, line=line, verbose=False)
        runs[line] = (sim, res, log)
        NUM[f"dv_autocross7_{line}"] = {"time": res.raw_time, "cones": res.cones_hit, "state": res.state,
                                        "energy_kwh": sim.vehicle.energy_j / 3.6e6, "vmax_kmh": max(log["v"]) * 3.6}
    sim, res, log = runs["racing"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W1, 2.6))
    xy, col = sim.track.cone_xy(), sim.track.cone_colors()
    for c in (0, 1, 2, 3):
        m = col == c
        a1.scatter(xy[m, 0], xy[m, 1], s=3, c=colours[c], linewidths=0, zorder=1)
    v = np.array(log["v"]) * 3.6
    st = np.array(log["state"])
    m = st == 3
    lc = colored_line(a1, np.array(log["x"])[m], np.array(log["y"])[m], v[m], lw=1.6)
    acb = a1.inset_axes([0.3, -0.06, 0.4, 0.035])
    cb = fig.colorbar(lc, cax=acb, orientation="horizontal")
    cb.set_label("speed [km/h]", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    a1.set_aspect("equal"); a1.set_xticks([]); a1.set_yticks([]); a1.grid(False)
    for s in a1.spines.values():
        s.set_visible(False)
    a1.set_title(f"Racing line, driven: {res.raw_time:.2f} s, {res.cones_hit} cones")
    for line, c in (("center", C2), ("racing", C1)):
        s_, r_, lg = runs[line]
        stt = np.array(lg["state"])
        mm = stt == 3
        t = np.array(lg["t"])[mm]
        a2.plot(t - t[0], np.array(lg["v"])[mm] * 3.6, color=c, label=f"{'centre' if line == 'center' else 'racing'} line ({r_.raw_time:.2f} s)")
    a2.set_xlabel("time since start [s]"); a2.set_ylabel("speed [km/h]"); a2.legend(loc="lower right")
    a2.set_title("Speed, reference driver")
    save(fig, "dv_closedloop.pdf")

    # --- events at 50 Hz (reference driver, ground truth)
    ev = {}
    for mis in ("acceleration", "skidpad", "autocross", "trackdrive"):
        sim, res, log, _ = run(mis, cfg, verbose=False)
        ev[mis] = {"raw": res.raw_time, "laps": res.lap_times, "cones": res.cones_hit, "state": res.state,
                   "oc": res.off_course, "energy_kwh": sim.vehicle.energy_j / 3.6e6}
    NUM["dv_events"] = ev

    # --- perception: fusion + SLAM on autocross seed 7, one LiDAR scan and the SLAM error
    t0 = time.time()
    sim, res, log, last = run("autocross", cfg, seed=7, perception="fusion", slam=True, verbose=False)
    NUM["dv_fusion_slam"] = {"time": res.raw_time, "cones": res.cones_hit, "state": res.state, "wall": time.time() - t0}
    acc = sim.pipeline.accuracy([(c.x, c.y, int(c.color)) for c in sim.track.cones if not c.knocked])
    NUM["dv_fusion_slam"]["accuracy"] = acc
    sl = np.array(sim.slam_log)
    NUM["dv_fusion_slam"]["slam_mean_m"] = float(sl[:, 1].mean())
    NUM["dv_fusion_slam"]["slam_max_m"] = float(sl[:, 1].max())
    NUM["dv_fusion_slam"]["slam_head_max_deg"] = float(np.degrees(sl[:, 2].max()))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W1, 2.5), gridspec_kw={"width_ratios": [1, 1.25]})
    if last is not None:
        evl, pose = last
        cloud = evl["cloud"]
        xyz = np.asarray(cloud["xyz"])
        from fsdsim.perception.lidar_cones import LidarConeDetector
        det = LidarConeDetector(sim.sc.lidar, sim.vp)
        cones = det.detect(cloud)
        Rm, tv = det.R, det.t
        p = xyz @ Rm.T + tv
        keep = (np.hypot(p[:, 0], p[:, 1]) < 22) & (p[:, 2] < 1.2)
        p = p[keep]
        a1.scatter(p[:, 1], p[:, 0], s=0.15, c=np.clip(p[:, 2], -0.1, 0.5), cmap="Greys", vmin=-0.4, vmax=0.6, linewidths=0, rasterized=True)
        cx = np.array([c["x"] for c in cones]); cy = np.array([c["y"] for c in cones])
        a1.scatter(cy, cx, s=28, facecolors="none", edgecolors=C2, linewidths=0.9, label=f"detected ({len(cones)})")
        x0, y0, yaw = pose
        tc = sim.track.cone_xy()
        rel = tc - np.array([x0, y0])
        cc, ss = math.cos(yaw), math.sin(yaw)
        lx, ly = cc * rel[:, 0] + ss * rel[:, 1], -ss * rel[:, 0] + cc * rel[:, 1]
        vis = (np.hypot(lx, ly) < 22)
        a1.scatter(ly[vis], lx[vis], s=6, marker="x", c=C1, linewidths=0.6, label="true cones")
        a1.set_xlim(12, -12); a1.set_ylim(-4, 22); a1.set_aspect("equal")
        a1.set_xlabel("y (left) [m]"); a1.set_ylabel("x (forward) [m]")
        a1.set_title("One analytic OS1-64 scan, car frame"); a1.legend(loc="upper left", markerscale=0.8, fontsize=6.5)
        NUM["dv_scan_points"] = int(len(xyz)); NUM["dv_scan_cones"] = len(cones)
    a2.plot(sl[:, 0] - sl[0, 0], sl[:, 1], color=C1, label="position error [m]")
    a2.set_xlabel("time [s]"); a2.set_ylabel("SLAM position error [m]")
    a2.set_title(f"Fusion + SLAM, autocross: mean {sl[:, 1].mean():.3f} m")
    save(fig, "dv_perception.pdf")


# =============================================================================================== #
# Paper II: Performance Lab
# =============================================================================================== #
def perf_figures():
    from fsdsim.perf import lapsim as L
    from perflab import energy, logs, studies, transient
    from perflab.lab import Lab

    lab = Lab()
    base = lab.raw([])
    NUM["pl_base"] = {n: {"T": e["T"], "t_lap": e["t_lap"], "points": e["points"], "vmax": e["v_max"] * 3.6,
                          "E_kwh": e["E_total"] / 3.6e6} for n, e in base["events"].items()}
    NUM["pl_base_total"] = base["total"]
    real = lab.raw([], "competition_fsa2025_real.m")
    NUM["pl_real"] = {n: {"T": e["T"], "t_lap": e["t_lap"], "points": e["points"]} for n, e in real["events"].items()}
    NUM["pl_real_total"] = real["total"]

    # --- QSS construction on the autocross: corner limit, forward pass, backward pass, result
    ls = lab.lapsim()
    ev = ls.comp.events["autocross"]
    vehE = L.event_vehicle(ls.veh, ev)
    k, ds = ev.track.k, ev.track.ds
    vc, limit = L.corner_speed(vehE, k)
    f = L._Fast(vehE)
    n = len(k)
    vf = np.zeros(n); vf[0] = 0.0; ax = 0.0
    for i in range(n - 1):
        ax = f.ax_acc(vf[i], k[i], ax)
        vf[i + 1] = min(math.sqrt(max(vf[i] ** 2 + 2 * ax * ds, 0)), vc[i + 1])
    vb = np.zeros(n); vb[-1] = vc[-1]; axb = 0.0
    for i in range(n - 1, 0, -1):
        axb = f.ax_dec(vb[i], k[i], axb)
        vb[i - 1] = min(math.sqrt(vb[i] ** 2 - 2 * axb * ds), vc[i - 1])
    v = np.minimum(vf, vb)
    s = ev.track.s
    m = s <= 300
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(W1, 3.6), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    a1.plot(s[m], np.minimum(vc[m], 40) * 3.6, color=GREY, lw=0.9, label=r"corner limit $v_c(s)$")
    a1.plot(s[m], vf[m] * 3.6, color=C2, lw=1.0, ls="--", label=r"forward pass $v_f$ (traction / power)")
    a1.plot(s[m], vb[m] * 3.6, color=C3, lw=1.0, ls="--", label=r"backward pass $v_b$ (braking)")
    a1.plot(s[m], v[m] * 3.6, color=C1, lw=1.8, label=r"$v = \min(v_f, v_b)$")
    a1.set_ylim(0, 125); a1.set_ylabel("speed [km/h]"); a1.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=7)
    lim = np.where(v[m] >= vc[m] - 1e-6, 1, np.where(vf[m] <= vb[m], 0, 2))
    names = ["accelerating (forward pass)", "at the corner limit", "braking (backward pass)"]
    for j, c in zip(range(3), (C2, GREY, C3)):
        a2.fill_between(s[m], 0, 1, where=lim == j, color=c, alpha=0.75, step="mid", lw=0, label=names[j])
    a2.set_yticks([]); a2.set_ylim(0, 1); a2.set_xlabel("distance s [m]"); a2.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.55))
    a2.grid(False)
    save(fig, "pl_qss.pdf")
    full_lim = np.where(v >= vc - 1e-6, 1, np.where(vf <= vb, 0, 2))
    NUM["pl_autocross_share"] = {"accel": float(np.mean(full_lim == 0)), "corner": float(np.mean(full_lim == 1)),
                                 "brake": float(np.mean(full_lim == 2))}
    NUM["pl_autocross_limit_axle"] = {"front": int(np.sum(limit == 1)), "rear": int(np.sum(limit == 2)), "power": int(np.sum(limit == 0))}

    # --- ggV
    g = energy.ggv_map(lab, [], "")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W1, 2.5))
    vk = np.array(g["v_kmh"]); idx = [0, 3, 6, 9, len(vk) - 1]
    for jj, i in enumerate(idx):
        ay = np.array(g["ay_g"][i]); acc = np.array(g["ax_acc_g"][i]); dec = np.array(g["ax_dec_g"][i])
        c = SEQ(0.25 + 0.75 * jj / (len(idx) - 1))
        a1.plot(np.r_[-ay[::-1], ay], np.r_[acc[::-1], acc], color=c, lw=1.1, label=f"{vk[i]:.0f} km/h")
        a1.plot(np.r_[-ay[::-1], ay], np.r_[dec[::-1], dec], color=c, lw=1.1)
        a1.plot([ay[-1], ay[-1]], [dec[-1], acc[-1]], color=c, lw=1.1); a1.plot([-ay[-1], -ay[-1]], [dec[-1], acc[-1]], color=c, lw=1.1)
    a1.set_xlabel(r"$a_y$ [g]"); a1.set_ylabel(r"$a_x$ [g]"); a1.set_aspect("equal"); a1.legend(fontsize=6.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.28), columnspacing=0.8)
    a1.set_title("gg diagram per speed (tyre limit)")
    a2.plot(vk, [r[-1] for r in g["ay_g"]], color=C1, label="max lateral")
    a2.plot(vk, [r[0] for r in g["ax_acc_g"]], color=C2, label="max drive")
    a2.plot(vk, [-r[0] for r in g["ax_dec_g"]], color=C3, label="max brake (magnitude)")
    a2.set_xlabel("speed [km/h]"); a2.set_ylabel("acceleration [g]"); a2.legend(); a2.set_title("Limits against speed")
    save(fig, "pl_ggv.pdf")
    NUM["pl_ggv"] = {"ay_low": g["ay_g"][0][-1], "ay_high": g["ay_g"][-1][-1], "v_low": vk[0], "v_high": vk[-1],
                     "acc_low": g["ax_acc_g"][0][0], "acc_high": g["ax_acc_g"][-1][0], "dec_low": g["ax_dec_g"][0][0],
                     "dec_high": g["ax_dec_g"][-1][0]}

    # --- sensitivity
    sens = studies.sensitivity(lab, ["mu", "m", "i_trans", "T_motor", "c_z", "A", "aero_bal_f", "c_w", "h_cog", "P_max"], 0.10)
    NUM["pl_sens"] = [{k: r[k] for k in ("param", "value", "points_per_pct", "better", "total_plus", "total_minus", "dT_plus", "dT_minus")}
                      | ({"points_per_kg": r["points_per_kg"], "autocross_s_per_kg": r["autocross_s_per_kg"]} if "points_per_kg" in r else {})
                      for r in sens["rows"]]
    rows = sens["rows"][::-1]
    b0 = sens["baseline"]["total"]
    fig, ax = plt.subplots(figsize=(W1, 2.8))
    y = np.arange(len(rows))
    ax.barh(y + 0.19, [r["total_plus"] - b0 for r in rows], height=0.36, color=C1, label="+10 %")
    ax.barh(y - 0.19, [r["total_minus"] - b0 for r in rows], height=0.36, color=C2, label="−10 %")
    ax.set_yticks(y); ax.set_yticklabels([r["param"] for r in rows])
    ax.axvline(0, color=GREY, lw=0.8); ax.set_xlabel(f"change of total points (baseline {b0:.2f})"); ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    save(fig, "pl_sensitivity.pdf")

    # --- sweep of the gear ratio
    sw = studies.sweep(lab, "i_trans", {"from": 11.6, "to": 23.2, "n": 25})
    xs = [r["value"] for r in sw["rows"]]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W1, 2.3))
    a1.plot(xs, [r["total"] for r in sw["rows"]], color=C1); a1.axvline(14.5, color=GREY, ls=":", lw=0.8)
    a1.set_xlabel(r"gear ratio $i$"); a1.set_ylabel("total points"); a1.set_title("Total of the four dynamic events")
    for n_, c in (("acceleration", C1), ("autocross", C2), ("endurance", C3), ("skidpad", C4)):
        a2.plot(xs, [r["points"][n_] for r in sw["rows"]], color=c, label=n_)
    a2.axvline(14.5, color=GREY, ls=":", lw=0.8); a2.set_xlabel(r"gear ratio $i$"); a2.set_ylabel("points"); a2.legend(fontsize=6.5)
    a2.set_title("Per event")
    save(fig, "pl_sweep_gear.pdf")
    best = max(sw["rows"], key=lambda r: r["total"])
    NUM["pl_gear_best"] = {"i": best["value"], "total": best["total"]}

    # --- 2-D map c_z x c_w
    mp = studies.sweep2d(lab, "c_z", {"from": 3.0, "to": 5.4, "n": 13}, "c_w", {"from": 1.0, "to": 1.8, "n": 9})
    Z = np.array(mp["total"])
    fig, ax = plt.subplots(figsize=(W1 * 0.62, 2.6))
    im = ax.pcolormesh(mp["v1"], mp["v2"], Z.T, cmap=SEQ, shading="nearest")
    cs = ax.contour(mp["v1"], mp["v2"], Z.T, levels=8, colors="white", linewidths=0.5)
    ax.clabel(cs, fontsize=6, fmt="%.0f")
    ax.plot(4.15, 1.38, marker="o", mfc="none", mec="black", ms=6)
    ax.annotate("LR26", (4.15, 1.38), xytext=(4.3, 1.45), fontsize=7)
    fig.colorbar(im, ax=ax, label="total points")
    ax.set_xlabel(r"downforce coefficient $c_z$"); ax.set_ylabel(r"drag coefficient $c_w$"); ax.grid(False)
    save(fig, "pl_map_aero.pdf")
    # efficiency of aero: points per +0.1 c_z vs +0.1 c_w around the LR26
    r1 = lab.raw(["c_z+0.1"])["total"] - b0
    r2 = lab.raw(["c_w+0.1"])["total"] - b0
    NUM["pl_aero_dp"] = {"cz_plus_0.1": r1, "cw_plus_0.1": r2, "ratio": -r1 / r2 if r2 else None}

    # --- energy
    en = energy.analyse(lab, [], budget_kwh=5.0, caps_kw=list(np.arange(8, 41, 1.0)))
    NUM["pl_energy"] = {k: en[k] for k in ("cap_kw", "t_lap", "T", "points", "E_lap_kwh", "E_total_kwh", "EF")}
    NUM["pl_energy"]["breakdown"] = en["breakdown"]
    NUM["pl_energy"]["budget"] = en["budget"]
    caps = [x["cap_kw"] for x in en["sweep"]]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W1, 2.4))
    a1.plot(caps, [x["points"] for x in en["sweep"]], color=C1)
    a1.set_xlabel("endurance power cap [kW]"); a1.set_ylabel("endurance points")
    a1.axvline(en["budget"]["cap_kw"], color=GREY, ls=":", lw=0.8)
    a1.annotate(f"5 kWh budget → {en['budget']['cap_kw']:.1f} kW", (en["budget"]["cap_kw"], 60), xytext=(20, 45), fontsize=7,
                arrowprops=dict(arrowstyle="-", color=GREY, lw=0.6))
    a1.set_title("Points against the cap")
    a2.plot(caps, [x["E_total_kwh"] for x in en["sweep"]], color=C2)
    a2.axhline(5.0, color=GREY, ls=":", lw=0.8)
    a2.set_xlabel("endurance power cap [kW]"); a2.set_ylabel("energy, 16 laps [kWh]"); a2.set_title("Energy against the cap")
    save(fig, "pl_energy_cap.pdf")
    b = en["breakdown"]
    items = [("battery out", b["battery_out"]), ("drivetrain losses", b["drivetrain_losses"]), ("aerodynamic drag", b["drag"]),
             ("rolling resistance", b["rolling"]), ("friction brakes", b["friction_brakes"])]
    fig, ax = plt.subplots(figsize=(W1 * 0.62, 1.8))
    ax.barh(range(len(items))[::-1], [v for _, v in items], color=[C1, C2, C3, C4, C5], height=0.6)
    for i, (nme, val) in enumerate(items):
        ax.text(val + 0.004, len(items) - 1 - i, f"{val:.3f}", va="center", fontsize=7)
    ax.set_yticks(range(len(items))[::-1]); ax.set_yticklabels([nm for nm, _ in items]); ax.set_xlabel("energy per endurance lap [kWh]")
    ax.grid(axis="y", visible=False); ax.set_xlim(0, max(v for _, v in items) * 1.18)
    save(fig, "pl_energy_breakdown.pdf")

    # --- transient lap vs QSS
    tr_res = {}
    for evn in ("acceleration", "skidpad", "autocross", "endurance"):
        t0 = time.time()
        r = transient.run(lab, evn)
        tr_res[evn] = r
        NUM.setdefault("pl_transient", {})[evn] = {"T": r["T"], "t_lap": r["t_lap"], "qss_T": r["qss"]["T"], "qss_t_lap": r["qss"]["t_lap"],
                                                  "lat_max": r["lat_max"], "beta_max": r["beta_max"], "off": r["off_track_s"],
                                                  "vmax": r["v_max_kmh"], "qss_vmax": r["qss"]["v_max_kmh"],
                                                  "E_lap": r["E_lap_kwh"], "qss_E_lap": r["qss"]["E_lap_kwh"], "wall": time.time() - t0,
                                                  "closure": r["closure_gap_m"]}
    r = tr_res["autocross"]
    fig, axs = plt.subplots(3, 1, figsize=(W1, 4.6), sharex=True, gridspec_kw={"height_ratios": [2, 1, 1]})
    axs[0].plot(r["qss"]["trace"]["s"], r["qss"]["trace"]["v"], color=C2, ls="--", lw=1.1, label=f"QSS LapSim ({r['qss']['T']:.2f} s)")
    axs[0].plot(r["lap"]["s"], r["lap"]["v"], color=C1, label=f"transient, model + driver ({r['T']:.2f} s)")
    axs[0].set_ylabel("speed [km/h]"); axs[0].legend(loc="lower center", ncol=2); axs[0].set_title("FSA 2025 autocross")
    axs[1].plot(r["lap"]["s"], r["lap"]["dt"], color=C1); axs[1].set_ylabel(r"$t_{tr} - t_{QSS}$ [s]")
    axs[2].plot(r["trace"]["s"], r["trace"]["beta"], color=C3, lw=1.0); axs[2].set_ylabel(r"sideslip $\beta$ [deg]")
    axs[2].set_xlabel("distance s [m]")
    save(fig, "pl_transient.pdf")

    # --- correlation with the documentation's speed trace
    src = [x["id"] for x in logs.sources(lab) if "autocross_doc" in x["id"]][0]
    c = logs.correlate(lab, src, {"speed": {"channel": "v_kmh", "unit": "kmh"}, "distance": "s_m"}, "autocross")
    NUM["pl_corr"] = {k: c[k] for k in ("rmse_v_kmh", "bias_v_kmh", "t_log", "t_sim", "shift_m", "coverage", "v_max_log_kmh", "v_max_sim_kmh")}
    src2 = [x["id"] for x in logs.sources(lab) if "endurance_doc" in x["id"]][0]
    c2 = logs.correlate(lab, src2, {"speed": {"channel": "v_kmh", "unit": "kmh"}, "distance": "s_m"}, "endurance", shift=0)
    NUM["pl_corr_end"] = {k: c2[k] for k in ("rmse_v_kmh", "bias_v_kmh", "t_log", "t_sim", "coverage")}
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(W1, 3.0), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    t = c["trace"]
    a1.plot(t["s"], t["v_log"], color=C2, lw=1.1, label="documentation trace (Python LapSim)")
    a1.plot(t["s"], t["v_sim"], color=C1, lw=1.1, ls="--", label="this work (port of LR26_LapSim)")
    a1.set_ylabel("speed [km/h]"); a1.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    a2.plot(t["s"], t["dt"], color=C1); a2.set_ylabel(r"$\Delta t$ [s]"); a2.set_xlabel("distance s [m]")
    save(fig, "pl_correlation.pdf")

    # --- solvers: qss vs point mass
    try:
        pm = ls.run([], solver="point_mass")
        NUM["pl_point_mass_total"] = pm["total"]
        NUM["pl_point_mass"] = {n: e["T"] for n, e in pm["events"].items()}
    except Exception as e:  # noqa: BLE001
        NUM["pl_point_mass_error"] = str(e)

    # --- timing of one full run
    t0 = time.time()
    for _ in range(5):
        lab.raw([])
    NUM["pl_run_wall_s"] = (time.time() - t0) / 5


def methods_figures():
    """Lap time methods (paper II, section "Lap time methods"): OCP models, method comparison, Monte-Carlo."""
    from perflab import methods, ocp
    from perflab.lab import Lab

    lab = Lab()
    q = lab.raw([], None, ["autocross"])["events"]["autocross"]
    res = {}
    for key, model, tv in (("pm", "point_mass", False), ("st", "single_track", False), ("dt", "double_track", False),
                           ("tv", "double_track", True)):
        for evn in ("acceleration", "skidpad", "autocross", "endurance"):
            if key == "tv" and evn in ("acceleration", "endurance"):
                continue
            r = ocp.run(lab, evn, model, torque_vectoring=tv)
            NUM.setdefault("pl_ocp", {}).setdefault(key, {})[evn] = {
                "ok": r["ok"], "T": r["T"], "t_lap": r["t_lap"], "points": r["points"], "wall": r["wall_s"],
                "qss_t_lap": r["qss"]["t_lap"], "qss_same_grid": r["qss"]["t_lap_same_grid"]}
            if evn == "autocross":
                res[key] = r
            print("    ocp", key, evn, r["t_lap"], f"{r['wall_s']:.1f} s", flush=True)
    NUM["pl_ocp_envelope"] = res["pm"]["envelope"]
    fig, axs = plt.subplots(2, 1, figsize=(W1, 3.9), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    axs[0].plot(q["s"], q["v"] * 3.6, color=GREY, ls="--", lw=1.0, label=f"LapSim QSS ({q['t_lap']:.2f} s)")
    for key, c, lab_ in (("pm", C1, "OCP point mass"), ("st", C2, "OCP single track"), ("tv", C3, "OCP double track + TV")):
        r = res[key]
        axs[0].plot(r["trace"]["s"], r["trace"]["v"], color=c, lw=1.1, label=f"{lab_} ({r['t_lap']:.2f} s)")
    axs[0].set_ylabel("speed [km/h]"); axs[0].legend(loc="lower center", ncol=2); axs[0].set_title("FSA 2025 autocross, fixed line")
    for key, c, lab_ in (("st", C2, "single track"), ("tv", C3, "double track + TV")):
        axs[1].plot(res[key]["trace"]["s"], res[key]["trace"]["beta"], color=c, lw=0.9, label=lab_)
    axs[1].set_ylabel(r"sideslip $\beta$ [deg]"); axs[1].set_xlabel("distance s [m]"); axs[1].legend(loc="upper right", ncol=2)
    save(fig, "pl_methods_autocross.pdf")

    c = methods.compare(lab, "skidpad", "mass-5", list(methods.METHODS))
    NUM["pl_methods_skidpad"] = {m["method"]: {"T": m["base"]["T"], "vs_qss": m["vs_qss"], "dT": m["dT"],
                                               "dpoints": m["dpoints"], "wall": m["wall_s"]} for m in c["methods"]}
    NUM["pl_methods_agree"] = c["agree"]
    pm = lab.lapsim().run([], None, solver="point_mass")
    NUM["pl_qss_point_mass"] = {"T": {n: e["T"] for n, e in pm["events"].items()}, "total": pm["total"]}

    mc = methods.monte_carlo(lab, 200, "mass-5")
    NUM["pl_mc"] = {k: mc[k] for k in ("n", "uncertainty", "nominal", "total", "events", "wall_s")}
    NUM["pl_mc"]["gain"] = {k: v for k, v in mc["gain"].items() if k != "samples"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W1, 2.2))
    a1.hist(mc["samples"], bins=20, color=C1, edgecolor="white", linewidth=0.6)
    a1.axvline(mc["nominal"], color=GREY, ls="--", lw=0.8)
    a1.set_xlabel("total dynamic points"); a1.set_ylabel("draws"); a1.set_title("Baseline under uncertainty")
    a2.hist(mc["gain"]["samples"], bins=20, color=C3, edgecolor="white", linewidth=0.6)
    a2.axvline(mc["gain"]["nominal"], color=GREY, ls="--", lw=0.8)
    a2.set_xlabel("points gained by 5 kg less"); a2.set_title(f"Gain, P(gain > 0) = {100 * mc['gain']['p_gain']:.0f} %")
    save(fig, "pl_montecarlo.pdf")


if __name__ == "__main__":
    t0 = time.time()
    only = sys.argv[1] if len(sys.argv) > 1 else ""        # "dv", "perf" or "methods": a part only (numbers.json is updated)
    old = os.path.join(HERE, "numbers.json")
    if only and os.path.exists(old):
        with open(old) as f:
            NUM.update(json.load(f))
    if only in ("", "perf"):
        print("Paper II figures ...", flush=True)
        perf_figures()
    if only in ("", "perf", "methods"):
        print("Paper II, lap time methods ...", flush=True)
        methods_figures()
    if only in ("", "dv"):
        print("Paper I figures ...", flush=True)
        dv_figures()
    NUM["generated"] = time.strftime("%Y-%m-%d %H:%M")
    with open(os.path.join(HERE, "numbers.json"), "w") as f:
        json.dump(NUM, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"done in {time.time() - t0:.0f} s")
