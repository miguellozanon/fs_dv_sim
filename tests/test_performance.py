"""Performance (LapSim) model inside the simulator: port accuracy against the team's LapSim results,
parameter handling, MATLAB import, fused dynamics, parameter studies, Zenoh shell setup."""

import json
import math
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "ros2_ws", "src")
sys.path[:0] = [os.path.join(SRC, "fsdsim"), os.path.join(SRC, "fsd_dashboard"), os.path.join(SRC, "fsd_sim_bridge"),
                os.path.join(ROOT, "tools")]

import numpy as np  # noqa: E402

from fsdsim.config import load_all_configs, to_dict, vehicle_from_dict  # noqa: E402
from fsdsim.control import make_driver  # noqa: E402
from fsdsim.mission import ASState  # noqa: E402
from fsdsim.perf import PerfParams, qss  # noqa: E402
from fsdsim.perf.params import _eval, matlab_statements  # noqa: E402
from fsdsim.sim import Simulation  # noqa: E402

CONFIG = os.path.join(SRC, "fsd_sim_bringup", "config")
PERF_YAML = os.path.join(CONFIG, "performance_lr26.yaml")
SWEEPS = json.load(open(os.path.join(ROOT, "tests", "data", "lapsim_lr26_sweeps.json")))["sweeps"]


def lr26():
    return PerfParams.from_yaml(PERF_YAML)


class TestLapSimPort(unittest.TestCase):
    """The Python port with the LR26 set reproduces LapSim's own results (lts_results_sensitivities.mat)."""

    def test_baseline(self):
        p = lr26()
        self.assertAlmostEqual(qss.acceleration(p).time, 4.6619, delta=0.002)
        self.assertAlmostEqual(qss.skidpad(p).time, 5.1543, delta=0.002)

    def test_skidpad_sweeps(self):
        p, err = lr26(), []
        for name in ("m", "mux", "muy", "c_w", "c_z", "h_cog", "aero_bal_f", "brake_bias", "i_trans", "P_max"):
            sw = SWEEPS[name]
            for val, T in zip(sw["values"], sw["T"]):
                err.append(qss.skidpad(p.copy(**{name: float(val)})).time - T[1])
        err = np.array(err)
        self.assertLess(float(np.sqrt(np.mean(err ** 2))), 0.002, "skidpad RMS vs LapSim")
        self.assertLess(float(np.abs(err).max()), 0.005)

    def test_acceleration_sweeps(self):
        p, err = lr26(), []
        for name in ("i_trans", "m", "mux", "P_max", "c_w"):
            sw = SWEEPS[name]
            for val, T in list(zip(sw["values"], sw["T"]))[::3]:
                err.append(qss.acceleration(p.copy(**{name: float(val)}), ds=0.1).time - T[0])
        err = np.array(err)
        self.assertLess(float(np.sqrt(np.mean(err ** 2))), 0.003, "acceleration RMS vs LapSim")
        self.assertLess(float(np.abs(err).max()), 0.006)

    def test_physics_sanity(self):
        p = lr26()
        from fsdsim.perf import model

        Fz = model.wheel_loads(p, 20.0, 0.0, 0.0)
        self.assertAlmostEqual(float(Fz.sum()), p.m * p.g + model.downforce(p, 20.0), places=6)
        Fz2 = model.wheel_loads(p, 20.0, 0.0, 10.0)   # left-hand corner: right (outer) wheels gain load
        self.assertGreater(Fz2[1], Fz[1])
        self.assertLess(Fz2[0], Fz[0])
        F_set, F_cap, v_max = model.powertrain(p, v_max_probe := 30.0)
        self.assertGreater(v_max, 25.0)
        self.assertEqual(model.powertrain(p, v_max + 1.0)[0], 0.0)
        self.assertGreater(qss.corner_speed(p, 1 / 9.125), 10.0)
        self.assertLess(v_max_probe, 31)

    def test_path_solver_closed_track(self):
        p = lr26()
        r = 20.0
        th = np.linspace(0, 2 * np.pi, 400, endpoint=False)
        res = qss.run_path(p, np.c_[r * np.cos(th), r * np.sin(th)], True)
        v_c = qss.corner_speed(p.for_event("autocross"), 1 / r)   # same driver utilisation as run_path
        self.assertAlmostEqual(float(np.median(res.v)), v_c, delta=0.3)
        self.assertGreater(res.lap_times[0], res.lap_times[1])   # standing start is slower

    def test_matches_matlab_sweeps_exactly(self):
        """Acceleration and skidpad of every swept value equal LapSim's own .mat results."""
        p, ea, es = lr26(), [], []
        for name, sw in SWEEPS.items():
            for val, T in zip(sw["values"], sw["T"]):
                q = p.copy(**{name: float(val)})
                ea.append(qss.acceleration(q).time - T[0])
                es.append(qss.skidpad(q).time - T[1])
        self.assertLess(float(np.abs(ea).max()), 1e-4, "acceleration vs MATLAB")
        self.assertLess(float(np.abs(es).max()), 1e-4, "skidpad vs MATLAB")

    def test_matlab_study_script(self):
        sys.path.insert(0, os.path.join(ROOT, "tools"))
        import param_study as ps

        self.assertEqual(ps.matlab_change("m=300"), "veh = lts_set_param(veh, 'm', 300);")
        self.assertEqual(ps.matlab_change("mu*1.1"), "veh = lts_set_param(veh, 'mu', 1.1, 'scale');")
        self.assertEqual(ps.matlab_change("mu+0.1"), "veh.mux = veh.mux + 0.1; veh.muy = veh.muy + 0.1;")
        self.assertTrue(ps.matlab_change("yaw_inertia=130").startswith("% (simulator only"))
        with tempfile.TemporaryDirectory() as d:
            path = ps.write_matlab_study(d, [("m = 300", 300.0, ["m=300.0"])], ["c_z+0.2"], ["skidpad", "trackdrive"])
            text = open(path).read()
        self.assertIn("events = {'skidpad', 'endurance'};", text)
        self.assertIn("veh.c_z = veh.c_z + 0.2;\nveh = lts_set_param(veh, 'm', 300);", text)
        self.assertIn("writetable(out, fullfile(here, 'matlab_results.csv'));", text)

    def test_aliases_follow_lts_set_param(self):
        p = lr26()
        q = p.with_changes("mu=1.8")                   # lts_set_param(veh, 'mu', 1.8): muy = 1.8, ratio kept
        self.assertAlmostEqual(q.muy, 1.8)
        self.assertAlmostEqual(q.mux / q.muy, p.mux / p.muy)
        q = p.with_changes("Gear=16", "cop*1.1", "cl+0.5", "torque=40", "cog=0.28")
        self.assertEqual((q.i_trans, q.T_motor, q.h_cog), (16.0, 40.0, 0.28))
        self.assertAlmostEqual(q.aero_bal_f, p.aero_bal_f * 1.1)
        self.assertAlmostEqual(q.c_z, p.c_z + 0.5)


class TestParams(unittest.TestCase):
    def test_changes_and_aliases(self):
        p = lr26().with_changes("m=300", "mu*1.1", "c_z+0.5", "h_cog-0.015")
        self.assertEqual(p.m, 300.0)
        self.assertAlmostEqual(p.mux, 1.43 * 1.1)
        self.assertAlmostEqual(p.muy, 1.64 * 1.1)
        self.assertAlmostEqual(p.c_z, 4.65)
        self.assertAlmostEqual(p.h_cog, 0.25)
        with self.assertRaises(KeyError):
            p.with_changes("not_a_param=1")
        with self.assertRaises(ValueError):
            p.with_changes("m300")

    def test_matlab_import(self):
        text = """function veh = lr26_params()
% Table 3.1
veh.m   = 284;   % kg
veh.l   = 1.53;
veh.lf  = 0.53*veh.l; veh.lr = veh.l - veh.lf;
veh.P_max = 80e3;
veh.brake_bias = NaN;
veh.c_z = 4.15; veh.c_w = 1.4 % no semicolon
veh.v = [1 2 3];
veh.bad = foo(2);
end
"""
        self.assertEqual(list(matlab_statements("a = 1; b = [1; 2]; % c = 3"))[:2], ["a = 1", "b = [1; 2]"])
        self.assertEqual(_eval("1.25", {}), 1.25)
        self.assertEqual(_eval("2^3 + veh.x", {"x": 1.0}), 9.0)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "lr26_params.m")
            open(path, "w").write(text)
            p, rep = PerfParams.from_matlab(path, base=lr26())
        self.assertAlmostEqual(p.lf, 0.8109)
        self.assertAlmostEqual(p.lr, 1.53 - 0.8109)
        self.assertEqual(p.P_max, 80000.0)
        self.assertTrue(math.isnan(p.brake_bias))
        self.assertEqual(p.c_w, 1.4)
        self.assertIn("bad", rep["failed"])
        self.assertIn("l", rep["unknown"])

    def test_importer_cli_finds_event_utilisation(self):
        from import_lapsim_params import event_utils, parse_assignments

        with tempfile.TemporaryDirectory() as d:
            comp = os.path.join(d, "competition_fsa2025.m")
            open(comp, "w").write("comp.events.acceleration.util = [0.97 0.96];\ncomp.events.skidpad.util_y = 0.8;\n"
                                  "comp.events.endurance.util_x = 0.9; comp.events.endurance.util_y = 0.91;\n")
            u = event_utils(parse_assignments(comp))
        self.assertEqual(u["acceleration"], {"util_x": 0.97, "util_y": 0.96})
        self.assertEqual(u["skidpad"], {"util_y": 0.8})
        self.assertEqual(u["endurance"], {"util_x": 0.9, "util_y": 0.91})


class TestVehicleConfig(unittest.TestCase):
    def test_performance_vehicle_is_default_and_synced(self):
        veh, _, _ = load_all_configs(CONFIG)
        p = veh.perf()
        self.assertEqual(veh.model, "performance")
        self.assertAlmostEqual(veh.mass, p.m)
        self.assertAlmostEqual(veh.wheelbase, p.lf + p.lr)
        self.assertAlmostEqual(veh.wheel_radius, p.circ_ref / (2 * math.pi))

    def test_overrides_reach_perf_and_vehicle_fields(self):
        veh, _, _ = load_all_configs(CONFIG, vehicle_overrides=["m=300", "mux*1.05", "yaw_inertia*1.5",
                                                                "actuators.max_steer=0.4"])
        self.assertEqual(veh.mass, 300.0)
        self.assertAlmostEqual(veh.perf().mux, 1.43 * 1.05)
        self.assertAlmostEqual(veh.yaw_inertia, 180.0)
        self.assertAlmostEqual(veh.actuators.max_steer, 0.4)
        # what the Isaac -> relay hello carries reproduces the same car
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            back = vehicle_from_dict(json.loads(json.dumps(to_dict(veh), default=str)))
        self.assertEqual(back.mass, 300.0)
        self.assertAlmostEqual(back.perf().mux, 1.43 * 1.05)

    def test_generic_vehicle_still_available(self):
        veh, _, _ = load_all_configs(CONFIG, vehicle_file="vehicle_generic.yaml")
        self.assertEqual(veh.model, "single_track")
        self.assertIsNone(veh.perf())


def _run(mission, changes=(), grip_util=None, vehicle_file="", tmax=90):
    veh, sens, simcfg = load_all_configs(CONFIG, vehicle_overrides=list(changes), vehicle_file=vehicle_file)
    simcfg.sim.mission = mission
    sim = Simulation(veh, sens, simcfg, enable_lidar=False, enable_camera=False)
    drv = make_driver(veh, mission, "map", sim.track.width, grip_util=grip_util)
    drv.set_reference(sim.track.centerline, sim.track.closed)
    beta = 0.0
    while sim.t < tmax:
        sim.step(0.02)
        s = sim.vehicle.state
        st = sim.mission.status(sim.t)
        if sim.mission.state in (ASState.READY, ASState.DRIVING):
            steer, acc, _ = drv.update(s.pose, s.vx, stop=st["mission_complete"])
            sim.set_command(steer, acc)
            beta = max(beta, abs(math.degrees(math.atan2(s.vy, max(s.vx, 0.5)))))
        if sim.mission.state in (ASState.FINISHED, ASState.EMERGENCY):
            break
    return sim.result_dict(), beta


class TestFusedDynamics(unittest.TestCase):
    def test_acceleration_close_to_lapsim(self):
        r, _ = _run("acceleration")
        self.assertEqual(r["state"], "FINISHED")
        self.assertAlmostEqual(r["raw_time"], qss.acceleration(lr26()).time, delta=0.2)

    def test_skidpad_clean_and_stable(self):
        r, beta = _run("skidpad")
        self.assertEqual((r["state"], r["cones_hit"], r["off_course"]), ("FINISHED", 0, 0))
        self.assertLess(beta, 6.0)
        self.assertLess(r["raw_time"], 6.0)

    def test_parameters_change_the_closed_loop_result(self):
        light, _ = _run("acceleration", ["m-30"])
        heavy, _ = _run("acceleration", ["m+30"])
        self.assertLess(light["raw_time"] + 0.1, heavy["raw_time"])

    def test_timing_uses_front_of_car(self):
        # acceleration starts 0.30 m behind the line: a standing start, not a 1.4 m rolling one
        r, _ = _run("acceleration")
        self.assertGreater(r["raw_time"], qss.acceleration(lr26()).time - 0.05)

    def test_generic_model_unchanged(self):
        r, _ = _run("acceleration", vehicle_file="vehicle_generic.yaml")
        self.assertEqual(r["state"], "FINISHED")


class TestStudyAndPrediction(unittest.TestCase):
    def test_param_study_case_without_sim(self):
        import param_study

        out = param_study.run_case({"label": "m = 300", "value": 300.0, "changes": ["m=300"], "event": "skidpad",
                                    "seed": None, "grip_util": 0.8, "long_util": 0.7, "sim": False,
                                    "config": CONFIG})
        self.assertAlmostEqual(out["lapsim_time"], qss.skidpad(lr26().with_changes("m=300")).time, places=6)
        self.assertGreater(out["dv_target_time"], 0)

    def test_dashboard_prediction(self):
        from fsd_dashboard.predict import predict

        p = lr26()
        a = predict(p, "acceleration")
        self.assertAlmostEqual(a["lapsim"]["time"], qss.acceleration(p).time)
        th = np.linspace(0, 2 * np.pi, 300, endpoint=False)
        c = predict(p, "trackdrive", np.c_[25 * np.cos(th), 15 * np.sin(th)], True, laps=10)
        self.assertAlmostEqual(c["lapsim"]["time"], c["lapsim"]["first_lap"] + 9 * c["lapsim"]["flying_lap"])
        self.assertGreater(c["dv_target"]["time"], c["lapsim"]["time"])


class TestZenohSetup(unittest.TestCase):
    @unittest.skipIf(shutil.which("bash") is None, "bash needed")
    def test_setup_shell_is_idempotent_and_keeps_other_aliases(self):
        with tempfile.TemporaryDirectory() as home:
            rc = os.path.join(home, ".bashrc")
            open(rc, "w").write("alias ll='ls -l'\nsource /opt/ros/jazzy/setup.bash\n"
                                "[ -f ~/fsg-dv-sim/ros2_ws/install/setup.bash ] && source ~/fsg-dv-sim/ros2_ws/install/setup.bash\n"
                                "alias go2sim='source /opt/ros/jazzy/setup.bash && export ROS_DOMAIN_ID=17'\n")
            env = dict(os.environ, HOME=home)
            for _ in range(2):
                subprocess.run(["bash", os.path.join(ROOT, "tools", "wsl", "setup_shell.sh")], env=env, check=True,
                               capture_output=True)
            text = open(rc).read()
        self.assertEqual(text.count(">>> fsg-dv-sim >>>"), 1)
        self.assertIn("alias go2sim='source /opt/ros/jazzy/setup.bash && export ROS_DOMAIN_ID=17'", text)
        self.assertIn("# [fsg-dv-sim: now in fsgsim] source /opt/ros/jazzy/setup.bash", text)
        self.assertIn("alias fsgsim=", text)
        self.assertIn("alias fsgrouter='fsgsim && ros2 run rmw_zenoh_cpp rmw_zenohd'", text)
        self.assertFalse([ln for ln in text.splitlines() if ln.startswith("source ")], "no global sourcing left")

    @unittest.skipIf(shutil.which("bash") is None, "bash needed")
    def test_setup_shell_leaves_multiline_alias_intact_and_repairs_it(self):
        fs = ("alias fs='source /opt/ros/jazzy/setup.bash\n"
              "[ -f ~/fsg-dv-sim/ros2_ws/install/setup.bash ] && source ~/fsg-dv-sim/ros2_ws/install/setup.bash\n"
              "export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST'\n")
        broken = ("alias fs='source /opt/ros/jazzy/setup.bash\n"
                  "# [fsg-dv-sim: now in fsgsim] [ -f ~/fsg-dv-sim/ros2_ws/install/setup.bash ] && "
                  "source ~/fsg-dv-sim/ros2_ws/install/setup.bash\n"
                  "# [fsg-dv-sim: not used with Zenoh] export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST'\n")
        for start in (fs, broken):
            with tempfile.TemporaryDirectory() as home:
                rc = os.path.join(home, ".bashrc")
                open(rc, "w").write("# don't\n" + start + "export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST\n")
                subprocess.run(["bash", os.path.join(ROOT, "tools", "wsl", "setup_shell.sh")],
                               env=dict(os.environ, HOME=home), check=True, capture_output=True)
                text = open(rc).read()
                subprocess.run(["bash", "-n", rc], check=True)
            self.assertIn(fs, text)
            self.assertIn("# [fsg-dv-sim: not used with Zenoh] export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST\n", text)

    def test_router_check(self):
        from fsd_sim_bridge.common import zenoh_router_problem

        old = os.environ.get("RMW_IMPLEMENTATION")
        try:
            os.environ["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"
            self.assertIsNone(zenoh_router_problem())
            os.environ["RMW_IMPLEMENTATION"] = "rmw_zenoh_cpp"
            srv = socket.socket()
            srv.bind(("127.0.0.1", 0))
            srv.listen(1)
            port = srv.getsockname()[1]
            self.assertIsNone(zenoh_router_problem(port=port))
            srv.close()
            self.assertIn("rmw_zenohd", zenoh_router_problem(port=port))
        finally:
            if old is None:
                os.environ.pop("RMW_IMPLEMENTATION", None)
            else:
                os.environ["RMW_IMPLEMENTATION"] = old

    def test_launch_files_check_router_and_pass_vehicle_changes(self):
        for name in ("isaac.launch.py", "lite.launch.py"):
            text = open(os.path.join(SRC, "fsd_sim_bringup", "launch", name), encoding="utf-8").read()
            self.assertIn("zenoh_router_problem()", text, name)
        self.assertIn('"vehicle_set"', open(os.path.join(SRC, "fsd_sim_bringup", "launch", "lite.launch.py")).read())
        ps1 = open(os.path.join(ROOT, "isaac", "run_isaac.ps1"), encoding="utf-8").read()
        self.assertIn('$env:RMW_IMPLEMENTATION = "rmw_zenoh_cpp"', ps1)
        self.assertIn("ZENOH_SESSION_CONFIG_URI", ps1)
        self.assertIn('"--set"', ps1)


if __name__ == "__main__":
    unittest.main()
