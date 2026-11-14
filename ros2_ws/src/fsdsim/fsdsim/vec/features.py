"""What the universal throttle / brake policy sees (features version 2), for N cars at once.

Shared by the training environment (fsdsim.vec.env, NumPy or PyTorch) and the RL driver in the
simulator (fsdsim.control.rl_driver, NumPy, one car), so training and driving compute exactly the
same inputs.

    1   speed
    12  speed envelope at 0, 2.5, 5, ... 52 m ahead: the highest speed at that point from which THIS
        car (its LapSim limits) can still brake down to every corner further ahead, at the tyre limit
    12  max |curvature| in each of the 12 intervals ahead (where the corners are)
    1   speed minus the envelope here
    2   full traction and full braking acceleration this car has now (speed, curvature here)
    7   lateral velocity, yaw rate, longitudinal / lateral acceleration, side-slip, lateral and heading
        error to the path
    1   previous action
    15  the vehicle specification (fsdsim.vec.spec.SpecBatch.descriptor)
"""

from __future__ import annotations

from .backend import Backend
from .limits import LimitTables, lookup_k, lookup_vk
from .spec import N_DESCRIPTOR
from .tracks import LOOKAHEAD, N_INTERVALS, TrackBank

FEATURES_VERSION = 2
N_FEATURES = 1 + N_INTERVALS + N_INTERVALS + 1 + 2 + 7 + 1 + N_DESCRIPTOR
# sensor noise (1 sigma) the policy is trained with, x Realism.obs_noise
NOISE = {"v": 0.05, "vy": 0.05, "r": 0.01, "ax": 0.3, "ay": 0.3, "e_lat": 0.03, "e_head": 0.01}


def envelope(B: Backend, vc_table, dec_table, km):
    """(N, N_INTERVALS) speed envelope from per-car tables and interval curvatures km (N, N_INTERVALS)."""
    vc = lookup_k(B, vc_table, km)
    cols = [None] * N_INTERVALS
    e = vc[:, N_INTERVALS - 1]
    cols[-1] = e
    for j in range(N_INTERVALS - 2, -1, -1):
        dsj = LOOKAHEAD[j + 1] - LOOKAHEAD[j]
        kj = km[:, j:j + 1]
        d1 = -lookup_vk(B, dec_table, e[:, None], kj)[:, 0]
        vt = B.sqrt(e * e + 2.0 * B.maximum(d1, 0.0) * dsj)
        d2 = -lookup_vk(B, dec_table, (0.5 * (vt + e))[:, None], kj)[:, 0]
        vt = B.sqrt(e * e + 2.0 * B.maximum(d2, 0.0) * dsj)
        e = B.minimum(vc[:, j], vt)
        cols[j] = e
    return B.stack(cols, axis=1)


def interp_ahead(B: Backend, env, d):
    """Envelope value at distance d ahead (N,) (piecewise linear between the lookahead points)."""
    out = env[:, 0]
    for j in range(N_INTERVALS - 1):
        a, b = LOOKAHEAD[j], LOOKAHEAD[j + 1]
        w = B.clip((d - a) / (b - a), 0.0, 1.0)
        seg = env[:, j] + w * (env[:, j + 1] - env[:, j])
        out = B.where((d >= a) & (d < b), seg, out)
    return B.where(d >= LOOKAHEAD[N_INTERVALS - 1], env[:, -1], out)


def features(B: Backend, bank: TrackBank, tables: LimitTables, desc, tid, idx, v, vy, r, ax, ay, e_lat, e_head,
             prev_a, km=None):
    """(N, N_FEATURES) - all inputs (N,) arrays; desc = spec.descriptor() (N, 15)."""
    if km is None:
        km = bank.kmax[tid, idx]
    env = envelope(B, tables.vc, tables.dec, km)
    k0 = km[:, 0:1]
    a_acc = lookup_vk(B, tables.acc, v[:, None], k0)[:, 0]
    a_dec = lookup_vk(B, tables.dec, v[:, None], k0)[:, 0]
    beta = B.atan2(vy, B.maximum(B.abs(v), 1.0))
    cols = [v / 30.0]
    cols += [env[:, j] / 30.0 for j in range(N_INTERVALS)]
    cols += [km[:, j] * 5.0 for j in range(N_INTERVALS)]
    cols += [(v - env[:, 0]) / 5.0, a_acc / 15.0, a_dec / 15.0,
             vy / 3.0, r / 1.5, ax / 15.0, ay / 15.0, beta * 3.0, e_lat / 1.5, e_head * 2.0, prev_a]
    f = B.concat([B.stack(cols, axis=1), desc], axis=1)
    return B.clip(f, -5.0, 5.0), env


def teacher_action(B: Backend, tables: LimitTables, km, v, a_max, d_max, kp: float = 2.5):
    """The reference driver's longitudinal law on the same envelope with its margins (80 % grip,
    braking x 0.7): P control on the target speed with a 0.4 s preview. Imitation warm start only."""
    env = envelope(B, tables.vc_ref, tables.dec_ref, km)
    v_t = B.minimum(env[:, 0], interp_ahead(B, env, B.maximum(v, 0.0) * 0.4))
    acc = B.clip(kp * (v_t - v), -d_max, a_max)
    return B.where(acc >= 0.0, acc / a_max, acc / d_max)
