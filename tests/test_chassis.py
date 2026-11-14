"""Body attitude, wheel slip, double-track handling and the energy meter (fsdsim.chassis, dynamics)."""

import math
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "ros2_ws", "src", "fsdsim"), os.path.join(ROOT, "tools")]
CONFIG = os.path.join(ROOT, "ros2_ws", "src", "fsd_sim_bringup", "config")

import numpy as np  # noqa: E402

from fsdsim.chassis import Chassis  # noqa: E402
from fsdsim.config import ChassisParams, load_all_configs  # noqa: E402
from fsdsim.dynamics import VehicleModel  # noqa: E402
from fsdsim.sensors.lidar import LidarModel  # noqa: E402
from fsdsim.sensors.proprioceptive import ImuModel  # noqa: E402


def drive(vm, steer, accel, t, dt=0.01):
    vm.set_command(steer, accel)
    for _ in range(int(t / dt)):
        vm.step(dt)


class TestChassis(unittest.TestCase):
    def test_steady_state_follows_gradients(self):
        c = Chassis(ChassisParams(), 1.6)
        for _ in range(300):
            c.step(0.01, -9.81, 9.81, 0.0, 0.5)          # 1 g braking, 1 g left
        self.assertAlmostEqual(math.degrees(c.state.roll), 0.7, places=3)       # right side down
        self.assertAlmostEqual(math.degrees(c.state.pitch), 0.45, places=3)     # nose down
        self.assertAlmostEqual(c.state.heave, 0.0, places=6)

    def test_downforce_lowers_the_car(self):
        c = Chassis(ChassisParams(), 1.6)
        for _ in range(300):
            c.step(0.01, 0.0, 0.0, 1900.0, 0.5)
        self.assertAlmostEqual(c.state.heave, -1900 / 2 / 90000 * 0.5 - 1900 / 2 / 100000 * 0.5, places=5)
        self.assertLess(c.state.heave, -0.005)

    def test_off_switch_and_slip_curve(self):
        c = Chassis(ChassisParams(attitude=False, wheel_slip=False), 1.6)
        c.step(0.1, 5.0, 5.0, 1000.0)
        self.assertEqual(c.state.attitude, (0.0, 0.0, 0.0))
        self.assertEqual(c.slip_ratio(500.0, 1000.0), 0.0)
        c = Chassis(ChassisParams(), 1.6)
        k = c.slip_ratio(500.0, 1000.0)
        self.assertAlmostEqual(math.sin(1.5 * math.atan(12.0 * k)), 0.5, places=9)
        self.assertAlmostEqual(c.slip_ratio(-500.0, 1000.0), -k)
        self.assertLess(c.slip_ratio(5000.0, 1000.0), 0.15)                     # saturates at the peak


class TestVehicleWithChassis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.veh, cls.sens, cls.sim = load_all_configs(CONFIG)

    def test_wheel_slip_under_drive_and_brake(self):
        vm = VehicleModel(self.veh)
        vm.reset(0, 0, 0)
        drive(vm, 0.0, 8.0, 1.5)
        w = vm.wheel_speeds()
        v = vm.state.vx / self.veh.wheel_radius
        self.assertGreater(w[2], v * 1.02)               # driven rear wheels turn faster than the road
        self.assertAlmostEqual(w[0], v, delta=1e-9)      # RWD: free-rolling fronts
        drive(vm, 0.0, -10.0, 0.5)
        w = vm.wheel_speeds()
        v = vm.state.vx / self.veh.wheel_radius
        self.assertLess(w[0], v * 0.99)
        self.assertLess(w[2], v * 0.995)

    def test_attitude_in_a_corner_and_imu(self):
        vm = VehicleModel(self.veh)
        vm.reset(0, 0, 0)
        drive(vm, 0.0, 6.0, 2.5)
        drive(vm, 0.12, 0.0, 1.5)                        # left corner
        ch = vm.chassis.state
        self.assertGreater(vm.ay_body, 5.0)
        self.assertAlmostEqual(math.degrees(ch.roll), 0.7 * vm.ay_body / 9.81, delta=0.15)
        self.assertLess(ch.heave, 0.0)
        cfg = self.sens.imu
        imu = ImuModel(cfg, np.random.default_rng(1))
        m = np.mean([imu.measure(vm)["linear_acceleration"] for _ in range(400)], axis=0)
        # the rolled IMU feels part of gravity sideways, on top of the lateral acceleration
        self.assertAlmostEqual(m[1], vm.ay_body * math.cos(ch.roll) + 9.81 * math.sin(ch.roll), delta=0.4)

    def test_planar_motion_does_not_depend_on_attitude_or_slip(self):
        res = []
        for att in (True, False):
            veh = load_all_configs(CONFIG, vehicle_overrides=[f"chassis.attitude={att}", f"chassis.wheel_slip={att}"])[0]
            vm = VehicleModel(veh)
            vm.reset(0, 0, 0)
            drive(vm, 0.0, 6.0, 2.0)
            drive(vm, 0.1, 1.0, 2.0)
            res.append(vm.state.to_array())
        np.testing.assert_allclose(res[0], res[1], atol=1e-12)

    def test_double_track_close_to_single_track(self):
        import closed_loop_eval as C
        t = []
        for ch in ([], ["chassis.handling=double_track"]):
            _, res, _, _ = C.run("skidpad", CONFIG, verbose=False, changes=ch)
            self.assertEqual((res.state, res.cones_hit, res.off_course), ("FINISHED", 0, 0), ch)
            t.append(res.raw_time)
        self.assertAlmostEqual(t[0], t[1], delta=0.06)
        self.assertNotEqual(t[0], t[1])

    def test_double_track_inner_wheel_carries_less(self):
        veh = load_all_configs(CONFIG, vehicle_overrides=["chassis.handling=double_track"])[0]
        vm = VehicleModel(veh)
        self.assertTrue(vm.double_track)
        vm.reset(0, 0, 0)
        drive(vm, 0.0, 6.0, 2.5)
        drive(vm, 0.12, 6.0, 1.0)                        # left corner under power
        fx, px = vm._derivatives(vm.state.to_array(), 0.12, 6.0)[1][4][5]
        self.assertLess(px[2], px[3])                    # rear left (inner) has less grip than rear right
        self.assertLessEqual(fx[2], fx[3] + 1e-9)

    def test_energy_meter_matches_lapsim_definition(self):
        vm = VehicleModel(self.veh)
        vm.reset(0, 0, 0)
        pp, E = vm.perf, 0.0
        vm.set_command(0.0, 6.0)
        for _ in range(300):
            vm.step(0.01)
            f = vm.last_forces
            E += max((f["Fx_f"] + f["Fx_r"]) * vm.state.vx, 0.0) / (pp.eta_g * pp.eta_motor) * 0.01
        self.assertGreater(vm.energy_j, 1e4)
        self.assertAlmostEqual(vm.energy_j, E, delta=1e-6 * E)

    def test_lidar_sees_the_ground_move_when_the_car_pitches(self):
        lid = LidarModel(self.sens.lidar, np.random.default_rng(0))
        flat = lid.scan((0, 0, 0), (0, 0, 0), [])
        dive = lid.scan((0, 0, 0), (0, 0, 0), [], attitude=(0.0, math.radians(0.5), -0.005))
        def ground_range(c, ring):
            m = (c["ring"] == ring) & (np.abs(c["xyz"][:, 1]) < 0.05) & (c["xyz"][:, 0] > 0)
            return float(np.median(np.linalg.norm(c["xyz"][m], axis=1)))
        ring = 40                                         # a beam that looks down a few degrees
        self.assertLess(ground_range(dive, ring), ground_range(flat, ring) * 0.97)   # nose down: ground comes closer


if __name__ == "__main__":
    unittest.main()
