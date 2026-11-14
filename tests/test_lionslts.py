"""Ideas taken over from the Performance department's LionsLTS code: spline tracks, the real FSA 2025 tracks,
endurance energy budget, vehicle-file import, MF4 logs, CSV export and the real-track importer."""

import csv
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "ros2_ws", "src")
sys.path[:0] = [os.path.join(SRC, "fsdsim"), os.path.join(ROOT, "tools")]

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from fsdsim.logdata.mdf4 import MDF4, resample  # noqa: E402
from fsdsim.perf import lapsim as L  # noqa: E402
from fsdsim.perf import strategy  # noqa: E402
from fsdsim.perf.export import COLUMNS, run_columns, write_csv  # noqa: E402
from fsdsim.perf.lionslts import to_changes  # noqa: E402
from fsdsim.perf.params import PerfParams  # noqa: E402

LAPSIM = os.path.join(ROOT, "lapsim", "LR26_LapSim")
DATA = os.path.join(ROOT, "tests", "data", "lionslts")


def circle_xy(r=30.0, n=60):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return r * np.cos(a), r * np.sin(a)


class TestSplineTrack(unittest.TestCase):
    def test_circle_curvature_and_length(self):
        x, y = circle_xy(30.0)
        tr = L._spline_track(x, y, 0.5, 5.0, True, "circle")
        self.assertAlmostEqual(tr.L, 2 * np.pi * 30.0, delta=0.5)
        self.assertAlmostEqual(float(np.median(tr.k)), 1 / 30.0, delta=1e-3)
        self.assertLess(float(np.max(np.abs(tr.k - 1 / 30.0))), 3e-3)
        self.assertEqual(tr.k[0], tr.k[-1])               # closed: periodic

    def test_open_line_is_straight(self):
        x = np.linspace(0, 100, 11)
        tr = L._spline_track(x, np.zeros_like(x), 0.5, 5.0, False, "line")
        self.assertAlmostEqual(tr.L, 100.0, delta=0.2)
        self.assertLess(float(np.max(np.abs(tr.k))), 1e-6)

    def test_track_csv_method_spline(self):
        x, y = circle_xy(25.0, 80)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "c.csv")
            np.savetxt(p, np.stack([x, y], 1), delimiter=",", header="x_m,y_m", comments="")
            tr = L.track_csv(p, closed=True, method="spline", curv_filter=5.0)
        self.assertAlmostEqual(float(np.median(tr.k)), 1 / 25.0, delta=1e-3)

    def test_movmean_keeps_length_and_mean_of_constant(self):
        z = np.full(20, 3.0)
        out = L._movmean(z, 5)
        self.assertEqual(len(out), 20)
        np.testing.assert_allclose(out, 3.0)


class TestRealTracks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.comp = L.load_competition(os.path.join(LAPSIM, "config", "competition_fsa2025_real.m"))
        cls.veh = L.LapSim.load(LAPSIM).veh

    def test_files_exist(self):
        for f in ("FSA2025_autocross_real.csv", "FSA2025_endurance_real.csv"):
            self.assertTrue(os.path.isfile(os.path.join(LAPSIM, "tracks", f)), f)

    def test_events_and_lengths(self):
        self.assertIn("endurance", self.comp.events)
        self.assertAlmostEqual(self.comp.events["endurance"].track.L, 1312.4, delta=15)

    def test_lap_times_match_documentation(self):
        """Autocross 49.9 s and endurance lap 95.4 s in the team's documentation, within 2 / 3 %."""
        sim = L.LapSim.load(LAPSIM, competition="competition_fsa2025_real.m")
        res = sim.run()
        self.assertAlmostEqual(res["events"]["autocross"]["T"], 49.90, delta=1.0)
        e = L.run_event(sim.veh, sim.comp.events["endurance"])
        self.assertAlmostEqual(e["t_lap"], 95.4, delta=3.0)

    def test_env_variable_selects_competition(self):
        old = os.environ.get("FSG_LAPSIM_COMPETITION")
        os.environ["FSG_LAPSIM_COMPETITION"] = "competition_fsa2025_real.m"
        try:
            self.assertTrue(L.LapSim.load(LAPSIM).competition_file.endswith("competition_fsa2025_real.m"))
        finally:
            if old is None:
                os.environ.pop("FSG_LAPSIM_COMPETITION")
            else:
                os.environ["FSG_LAPSIM_COMPETITION"] = old

    def test_run_event_exposes_signals(self):
        r = L.run_event(self.veh, self.comp.events["acceleration"])
        for k in ("k", "ax", "ay", "P_wheel", "P_batt", "E", "veh"):
            self.assertIn(k, r)


class TestStrategy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim = L.LapSim.load(LAPSIM, competition="competition_fsa2025_real.m")
        cls.ev = cls.sim.comp.events["endurance"]

    def test_budget_fits_and_is_tight(self):
        r = strategy.power_for_energy(self.sim.veh, self.ev, 5.5)
        self.assertTrue(r["limited"] and r["feasible"])
        self.assertLessEqual(r["E_total_kwh"], 5.5 + 1e-6)
        self.assertGreater(r["E_total_kwh"], 5.5 - 0.05)
        self.assertAlmostEqual(r["cap_kw"], 18.0, delta=0.8)

    def test_big_budget_not_limited_and_tiny_budget_infeasible(self):
        self.assertFalse(strategy.power_for_energy(self.sim.veh, self.ev, 50.0)["limited"])
        self.assertFalse(strategy.power_for_energy(self.sim.veh, self.ev, 0.5)["feasible"])

    def test_less_power_is_slower_and_cheaper(self):
        a, b = strategy.sweep_power(self.sim.veh, self.ev, [15, 30])
        self.assertGreater(a["T"], b["T"])
        self.assertLess(a["E_total_kwh"], b["E_total_kwh"])


class TestVehicleImport(unittest.TestCase):
    def load(self, name):
        with open(os.path.join(DATA, name), encoding="utf-8") as f:
            return yaml.safe_load(f)

    def test_emrax_file(self):
        base = L.LapSim.load(LAPSIM).veh
        ch, notes = to_changes(self.load("LR26_emrax.yml"), base)
        d = dict(c.split("=") for c in ch)
        self.assertAlmostEqual(float(d["m"]), 290.0)
        self.assertAlmostEqual(float(d["c_w"]), 1.769636015, places=6)
        self.assertAlmostEqual(float(d["aero_bal_f"]), 0.5299881235, places=6)
        self.assertAlmostEqual(float(d["T_motor"]), 200.0)
        self.assertNotIn("circ_ref", d)                   # 1.5651 vs 1.57: below tolerance
        self.assertTrue(notes)

    def test_changes_apply_and_baseline_gives_none(self):
        base = L.LapSim.load(LAPSIM).veh
        ch, _ = to_changes(self.load("LR25.yml"), base)
        p = base.with_changes(*ch)
        self.assertIsInstance(p, PerfParams)
        again, _ = to_changes(self.load("LR25.yml"), p)
        self.assertEqual(again, [])

    def test_saved_configs_are_valid(self):
        names = [f for f in os.listdir(os.path.join(ROOT, "lapsim", "configs")) if f.startswith("LionsLTS")]
        self.assertGreaterEqual(len(names), 6)
        base = L.LapSim.load(LAPSIM).veh
        for n in names:
            with open(os.path.join(ROOT, "lapsim", "configs", n), encoding="utf-8") as fh:
                cfg = json.load(fh)
            self.assertIn("changes", cfg)
            base.with_changes(*L.lapsim_changes(cfg["changes"]))


# ---- a synthetic MDF 4 writer ------------------------------------------------------------------ #
class _Mdf:
    """Just enough MDF 4.1 to test the reader: one DG per channel group, a time master and N value channels."""

    def __init__(self):
        self.buf = bytearray(b"MDF     4.10    " + b"test    " + b"\0" * 40)   # ID block, 64 bytes
        self.buf[:64] = (b"MDF     4.10    asam    " + struct.pack("<HHHHH", 0, 410, 0, 0, 0)).ljust(64, b"\0")

    def block(self, bid, links, data=b""):
        pos = len(self.buf)
        length = 24 + 8 * len(links) + len(data)
        self.buf += struct.pack("<4sIQQ", bid, 0, length, len(links)) + b"".join(struct.pack("<Q", x) for x in links) + data
        self.buf += b"\0" * (-len(self.buf) % 8)
        return pos

    def tx(self, s):
        return self.block(b"##TX", [], s.encode() + b"\0")

    def patch(self, pos, i, val):
        struct.pack_into("<Q", self.buf, pos + 24 + 8 * i, val)


def write_mdf(path, groups, compress=False):
    """groups: [(name, t, {chan: (raw uint16 array, unit, (offset, factor) or None)})]"""
    m = _Mdf()
    hd = m.block(b"##HD", [0] * 6, b"\0" * 32)
    assert hd == 64
    prev_dg = 0
    first_dg = 0
    for name, t, chans in groups:
        n = len(t)
        recb = 8 + 2 * len(chans)
        rec = np.zeros(n, dtype=[("t", "<f8")] + [(c, "<u2") for c in chans])
        rec["t"] = t
        for c, (raw, _u, _cv) in chans.items():
            rec[c] = raw
        body = rec.tobytes()
        if compress:
            z = zlib.compress(body)
            dt = m.block(b"##DZ", [], b"DT" + struct.pack("<BBIQQ", 0, 0, 0, len(body), len(z)) + z)
        else:
            dt = m.block(b"##DT", [], body)
        cns = []
        specs = [("t", 2, 4, 0, 64, None, "s")] + [(c, 0, 0, 8 + 2 * i, 16, cv, u) for i, (c, (_r, u, cv)) in enumerate(chans.items())]
        for cname, cn_type, dtype, boff, bits, cv, unit in specs:
            cc = 0
            if cv:
                cc = m.block(b"##CC", [0, 0, 0, 0], struct.pack("<BBHHH", 1, 0, 0, 0, 2) + b"\0" * 0
                             + struct.pack("<2d", 0.0, 0.0) + struct.pack("<2d", cv[0], cv[1]))
            # CC data: type, prec, flags, nref, nval = 8 bytes, then range min/max (16), then vals
            cns.append(m.block(b"##CN", [0, 0, m.tx(cname), 0, cc, 0, m.tx(unit), 0],
                               struct.pack("<BBBBIIIIBBH", cn_type, 0, dtype, 0, boff, bits, 0, 0, 0, 0, 0) + b"\0" * 8))
        for a, b in zip(cns, cns[1:]):
            m.patch(a, 0, b)
        cg = m.block(b"##CG", [0, cns[0], m.tx(name), 0, 0, 0],
                     struct.pack("<QQHHIII", 0, n, 0, 0, 0, recb, 0) + b"\0" * 8)
        dg = m.block(b"##DG", [0, cg, dt, 0], b"\0" * 8)
        if prev_dg:
            m.patch(prev_dg, 0, dg)
        else:
            first_dg = dg
        prev_dg = dg
    m.patch(64, 0, first_dg)
    with open(path, "wb") as f:
        f.write(bytes(m.buf))


class TestMdf4(unittest.TestCase):
    def make(self, d, compress):
        p = os.path.join(d, "x.mf4")
        t = np.arange(0, 1.0, 0.1)
        rpm = (np.arange(10) * 100).astype(np.uint16)
        rpm[3] = 0xFFFF                                    # "not available"
        write_mdf(p, [("CAN1 grp", t, {"rpm": (rpm, "rpm", (0.0, 1.0))}),
                      ("CAN3 grp", t[::2], {"rpm": ((np.arange(5) * 10).astype(np.uint16), "rpm", (5.0, 0.5)),
                                              "n": (np.arange(5).astype(np.uint16), "", None)})], compress)
        return p

    def check(self, compress):
        with tempfile.TemporaryDirectory() as d:
            f = MDF4(self.make(d, compress))
            self.assertEqual(len(f.groups), 2)
            t, x = f.signal("rpm")
            self.assertEqual(len(t), 10)
            self.assertAlmostEqual(float(x[2]), 200.0)
            self.assertTrue(np.isnan(x[3]))                # invalid value masked
            self.assertAlmostEqual(float(f.signal("rpm", mask_invalid=False)[1][3]), 65535.0)
            t3, x3 = f.signal("rpm", group="CAN3")
            self.assertEqual(len(t3), 5)
            self.assertAlmostEqual(float(x3[2]), 5.0 + 0.5 * 20)   # linear conversion
            self.assertEqual(len(f.signals("rpm")), 2)
            with self.assertRaises(KeyError):
                f.signal("nothing")

    def test_uncompressed(self):
        self.check(False)

    def test_zlib(self):
        self.check(True)

    def test_not_mdf(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.mf4")
            with open(p, "wb") as fh:
                fh.write(b"hello" * 100)
            with self.assertRaises(ValueError):
                MDF4(p)

    def test_resample_holds_last_value(self):
        grid, out = resample({"a": (np.array([0.0, 1.0, 2.0]), np.array([1.0, 2.0, 3.0])),
                              "b": (np.array([0.5]), np.array([9.0]))}, 4.0)
        self.assertAlmostEqual(grid[1] - grid[0], 0.25)
        self.assertEqual(float(out["a"][4]), 2.0)          # t = 1.0
        self.assertTrue(np.isnan(out["b"][0]))             # before the first sample
        self.assertEqual(float(out["b"][-1]), 9.0)

    def test_real_file_if_present(self):
        p = os.environ.get("FSG_TEST_MF4")
        if not p or not os.path.isfile(p):
            self.skipTest("set FSG_TEST_MF4 to a real .mf4 to run this")
        f = MDF4(p)
        self.assertGreater(len(f.groups), 0)


class TestExport(unittest.TestCase):
    def test_csv_columns_and_energy(self):
        sim = L.LapSim.load(LAPSIM)
        r = L.run_event(sim.veh, sim.comp.events["autocross"])
        cols = run_columns(r, energy_start_j=1e6)
        self.assertEqual(list(cols), COLUMNS)
        n = len(cols["time"])
        for c in COLUMNS:
            self.assertEqual(len(cols[c]), n, c)
        np.testing.assert_allclose(cols["energy storage state"], 1e6 - cols["consumed energy"])
        self.assertTrue(np.all(np.diff(cols["time"]) >= 0))
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "run.csv")
            self.assertEqual(write_csv(r, p), n)
            with open(p, encoding="utf-8") as fh:
                rows = list(csv.reader(fh))
        self.assertEqual(rows[0], COLUMNS)
        self.assertEqual(len(rows), n + 1)


class TestTrackImport(unittest.TestCase):
    def run_tool(self, *args):
        return subprocess.run([sys.executable, os.path.join(ROOT, "tools", "track_import.py"), *args],
                              capture_output=True, text=True)

    def test_read_xy_latlon_and_header(self):
        import track_import as T

        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "g.csv")
            with open(p, "w") as fh:
                fh.write("# comment\nlat,lon\n48.0,11.0\n48.001,11.0\n48.001,11.001\n48.0,11.001\n")
            xy = T.read_xy(p, True)
        self.assertEqual(xy.shape, (4, 2))
        self.assertAlmostEqual(xy[1, 1], 111.2, delta=0.5)         # 0.001 deg of latitude ~ 111 m
        self.assertAlmostEqual(xy[3, 0], 111.2 * np.cos(np.radians(48.0)), delta=0.5)

    def test_line_and_cones_from_real_endurance(self):
        with tempfile.TemporaryDirectory() as d:
            line, cones = os.path.join(d, "line.csv"), os.path.join(d, "cones.csv")
            r = self.run_tool(os.path.join(LAPSIM, "tracks", "FSA2025_endurance_real.csv"), "--closed",
                              "--line", line, "--cones", cones, "--width", "3.5")
            self.assertEqual(r.returncode, 0, r.stderr)
            xy = np.loadtxt(line, delimiter=",", skiprows=1)
            self.assertGreater(len(xy), 2000)
            self.assertIn("1312", r.stdout)
            self.assertGreater(os.path.getsize(cones), 1000)

    def test_cones_need_closed(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "l.csv")
            np.savetxt(p, np.stack([np.linspace(0, 50, 20), np.zeros(20)], 1), delimiter=",")
            r = self.run_tool(p, "--cones", os.path.join(d, "c.csv"))
            self.assertEqual(r.returncode, 2)



class TestPowertrainLog(unittest.TestCase):
    """Efficiency / roll-out analysis of a car log, on a synthetic log with known efficiency."""

    EFF = 0.88

    def synth(self, decel=None):
        """Ramps of torque and speed with DC power = shaft power / EFF, then a zero-torque roll-out."""
        t = np.arange(0, 60, 0.05)
        n = np.where(t < 40, 300 * t, np.maximum(0.0, 12000 - decel * 14.5 * 60 / 1.57 * (t - 40)) if decel else 12000.0)
        T = np.where(t < 40, 4 + 6 * np.sin(t / 5) ** 2, 0.0)
        P = T * n * 2 * np.pi / 60 / 1e3 / self.EFF
        z = np.zeros_like(t)
        return t, {"nA": n, "nB": n, "TA": T, "TB": T, "PA": P, "PB": P, "thr": z, "bpf": z, "bpr": z, "t": t}

    def test_efficiency_and_loss_fit(self):
        from fsdsim.logdata import powertrain as PT

        t, d = self.synth()
        s = PT.motor_samples(d)
        self.assertGreater(len(s["eff"]), 500)
        self.assertAlmostEqual(float(np.median(s["eff"])), self.EFF, places=3)
        te, ne, tab, cnt = PT.efficiency_map(s)
        self.assertTrue(np.all(np.abs(tab[np.isfinite(tab)] - self.EFF) < 0.01))
        fit = PT.loss_fit(s)
        self.assertLess(fit["rms_kw"], 0.15)           # the synthetic loss is proportional to T*n, outside the fit's terms
        k = fit["k"]
        T0, n0 = 8.0, 6.0                                # Nm, krpm
        true = T0 * n0 * 1e3 * 2 * np.pi / 60 / 1e3 * (1 / self.EFF - 1)
        self.assertAlmostEqual(k[0] + k[1] * T0 ** 2 + k[2] * n0 + k[3] * n0 ** 2, true, delta=0.15)

    def test_rollout_detects_non_road_deceleration(self):
        from fsdsim.logdata import powertrain as PT

        _, slow = self.synth(decel=0.3)
        _, fast = self.synth(decel=3.0)
        a = PT.rollout(slow, 14.5, 1.57)["decel_median"]
        b = PT.rollout(fast, 14.5, 1.57)["decel_median"]
        self.assertAlmostEqual(a, 0.3, delta=0.1)
        self.assertAlmostEqual(b, 3.0, delta=0.3)

    def test_load_from_mf4_and_cli(self):
        from fsdsim.logdata import powertrain as PT

        t, d = self.synth(decel=3.0)
        ch = {}
        for key, name, fac in (("nA", "act_Speed_R_DCUA", 1.0), ("nB", "act_Speed_R_DCUB", 1.0), ("TA", "act_Torque_R_DCUA", 0.1),
                               ("TB", "act_Torque_R_DCUB", 0.1), ("PA", "act_Power_R_DCUA", 0.01), ("PB", "act_Power_R_DCUB", 0.01),
                               ("thr", "PDB_Pedal_Throttle", 1.0), ("bpf", "PDB_Pedal_BPF", 1.0), ("bpr", "PDB_Pedal_BPR", 1.0)):
            ch[name] = (np.round(d[key] / fac).astype(np.uint16), "", (0.0, fac))
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "car.mf4")
            write_mdf(p, [("CAN3 message", t, ch)])
            got = PT.load(p, 20.0)
            self.assertAlmostEqual(float(np.nanmax(got["nA"])), 12000.0, delta=1)
            r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "log_correlate.py"), p, "--csv", os.path.join(tmp, "m.csv")],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("median 0.88", r.stdout)
            self.assertIn("far above a rolling car", r.stdout)
            self.assertTrue(os.path.isfile(os.path.join(tmp, "m.csv")))


if __name__ == "__main__":
    unittest.main()
