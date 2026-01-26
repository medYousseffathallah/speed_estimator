from __future__ import annotations

import math
from typing import Optional, Tuple


def heading_deg_from_delta(dx: float, dy: float) -> float:
    ang = math.degrees(math.atan2(dy, dx))
    if ang < 0.0:
        ang += 360.0
    return float(ang)


def wrap_angle_deg(angle: float) -> float:
    a = float(angle) % 360.0
    if a >= 180.0:
        a -= 360.0
    return a


def angle_between_deg(a_dx: float, a_dy: float, b_dx: float, b_dy: float) -> float:
    a_norm = math.hypot(a_dx, a_dy)
    b_norm = math.hypot(b_dx, b_dy)
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    dot = (a_dx * b_dx + a_dy * b_dy) / (a_norm * b_norm)
    dot = max(-1.0, min(1.0, dot))
    return float(math.degrees(math.acos(dot)))


def speed_mps(p0: Tuple[float, float], t0_s: float, p1: Tuple[float, float], t1_s: float) -> Optional[float]:
    dt = float(t1_s - t0_s)
    if dt <= 0.0:
        return None
    dx = float(p1[0] - p0[0])
    dy = float(p1[1] - p0[1])
    dist = math.hypot(dx, dy)
    return float(dist / dt)

