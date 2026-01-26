from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import math

from speedestimation.speed_estimation.math import angle_between_deg, heading_deg_from_delta


@dataclass(frozen=True)
class TurningMetrics:
    heading_deg: float
    turn_angle_deg: float
    curvature_1pm: float


def compute_turning_metrics(
    world_xy_m: Sequence[Tuple[float, float]],
    timestamps_s: Sequence[float],
    window: int,
    window_s: float | None = None,
) -> TurningMetrics:
    if len(world_xy_m) < 2 or len(world_xy_m) != len(timestamps_s):
        return TurningMetrics(heading_deg=float("nan"), turn_angle_deg=0.0, curvature_1pm=0.0)
    n = len(world_xy_m)
    pts = list(world_xy_m)
    ts = list(timestamps_s)
    dx = pts[-1][0] - pts[-2][0]
    dy = pts[-1][1] - pts[-2][1]
    heading = heading_deg_from_delta(dx, dy)

    if window_s is not None and window_s > 0.0:
        t_target = ts[-1] - float(window_s)
        idx = None
        for i in range(n - 1, -1, -1):
            if ts[i] <= t_target:
                idx = i
                break
        if idx is not None and idx >= 1:
            v0_dx = pts[idx][0] - pts[idx - 1][0]
            v0_dy = pts[idx][1] - pts[idx - 1][1]
            v1_dx = pts[-1][0] - pts[-2][0]
            v1_dy = pts[-1][1] - pts[-2][1]
            turn_angle = angle_between_deg(v0_dx, v0_dy, v1_dx, v1_dy)
            arc_len = 0.0
            for i in range(idx, n):
                arc_len += math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
            curvature = 0.0 if arc_len <= 1e-6 else math.radians(turn_angle) / arc_len
            return TurningMetrics(heading_deg=heading, turn_angle_deg=float(turn_angle), curvature_1pm=float(curvature))

    w = max(2, min(int(window), n))
    pts_w = pts[-w:]
    if len(pts_w) < 3:
        return TurningMetrics(heading_deg=heading, turn_angle_deg=0.0, curvature_1pm=0.0)

    v0_dx = pts_w[-2][0] - pts_w[-3][0]
    v0_dy = pts_w[-2][1] - pts_w[-3][1]
    v1_dx = pts_w[-1][0] - pts_w[-2][0]
    v1_dy = pts_w[-1][1] - pts_w[-2][1]
    turn_angle = angle_between_deg(v0_dx, v0_dy, v1_dx, v1_dy)

    arc_len = 0.0
    for i in range(1, len(pts_w)):
        arc_len += math.hypot(pts_w[i][0] - pts_w[i - 1][0], pts_w[i][1] - pts_w[i - 1][1])
    if arc_len <= 1e-6:
        curvature = 0.0
    else:
        curvature = math.radians(turn_angle) / arc_len
    return TurningMetrics(heading_deg=heading, turn_angle_deg=float(turn_angle), curvature_1pm=float(curvature))

