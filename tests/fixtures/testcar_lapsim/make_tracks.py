"""Fictional tracks of the CI test car (tests/fixtures/testcar_lapsim). Run once; the CSVs are committed.

Built from curvature pieces: (length [m], curvature [1/m]). The endurance is two identical halves joined by two
180-degree turns, so it closes exactly."""

import math
import os

DS = 0.5


def arc(angle_deg, radius, left=True):
    return [(abs(math.radians(angle_deg)) * radius, (1 if left else -1) / radius)]


def slalom(n, pitch, amp_k):
    """n gates; curvature +k, -k, ... with a half piece at both ends: net heading 0."""
    out = [(pitch / 2, amp_k)]
    for i in range(n - 1):
        out.append((pitch, -amp_k if i % 2 == 0 else amp_k))
    out.append((pitch / 2, -amp_k if (n - 1) % 2 == 0 else amp_k))
    return out


def build(pieces, x0=0.0, y0=0.0, psi0=math.pi / 2):
    rows, s, x, y, psi = [], 0.0, x0, y0, psi0
    for length, k in pieces:
        n = max(1, round(length / DS))
        h = length / n
        for _ in range(n):
            rows.append((s, x, y, k))
            # exact arc step
            if abs(k) < 1e-12:
                x += h * math.cos(psi); y += h * math.sin(psi)
            else:
                x += (math.sin(psi + k * h) - math.sin(psi)) / k
                y -= (math.cos(psi + k * h) - math.cos(psi)) / k
            psi += k * h
            s += h
    rows.append((s, x, y, pieces[-1][1]))
    return rows


def write(path, rows):
    with open(path, "w", newline="\n") as f:
        f.write("s_m,x_m,y_m,kappa_1pm\n")
        for s, x, y, k in rows:
            f.write(f"{s:.2f},{x:.3f},{y:.3f},{k:.6f}\n")


here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tracks")
auto = ([(30, 0)] + arc(90, 15) + [(40, 0)] + slalom(6, 12, 1 / 14) + [(25, 0)] + arc(180, 9, left=False)
        + [(60, 0)] + arc(45, 18) + arc(45, 18, left=False) + [(35, 0)] + arc(135, 20) + [(50, 0)])
write(os.path.join(here, "TEST_autocross.csv"), build(auto))
half = [(80, 0)] + slalom(5, 13, 1 / 15) + [(30, 0)] + arc(60, 25, left=False) + arc(60, 25) + [(40, 0)] + arc(180, 12)
rows = build(half + half)
write(os.path.join(here, "TEST_endurance.csv"), rows)
print("closure gap of the endurance [m]:", math.hypot(rows[-1][1] - rows[0][1], rows[-1][2] - rows[0][2]))
