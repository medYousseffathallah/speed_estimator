from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TurnSpeedLimitConfig:
    enabled: bool
    a_lat_max_mps2: float
    v_max_mps: float
    v_min_mps: float
    angle_max_deg: float
    mode: str
    alpha: float


def turn_limited_speed_mps(v_raw_mps: float, curvature_1pm: float, turn_angle_deg: float, cfg: TurnSpeedLimitConfig) -> float:
    if not cfg.enabled:
        return float(v_raw_mps)
    mode = str(cfg.mode).lower()
    if mode == "linear_angle":
        angle = float(abs(turn_angle_deg))
        if cfg.angle_max_deg <= 0.0:
            v_cap = float(cfg.v_max_mps)
        else:
            ratio = min(1.0, max(0.0, angle / float(cfg.angle_max_deg)))
            v_cap = float(cfg.v_max_mps) - ratio * (float(cfg.v_max_mps) - float(cfg.v_min_mps))
            v_cap = max(float(cfg.v_min_mps), min(float(cfg.v_max_mps), v_cap))
    else:
        k = float(abs(curvature_1pm))
        if k <= 0.0:
            v_cap = float(cfg.v_max_mps)
        else:
            v_curve = math.sqrt(max(0.0, float(cfg.a_lat_max_mps2) / k))
            v_cap = min(float(cfg.v_max_mps), float(v_curve))
    v_limited = min(float(v_raw_mps), float(v_cap))
    a = max(0.0, min(1.0, float(cfg.alpha)))
    return float((1.0 - a) * float(v_raw_mps) + a * float(v_limited))


def accel_limited_speed_mps(
    v_prev_mps: Optional[float],
    t_prev_s: Optional[float],
    v_curr_mps: float,
    t_curr_s: float,
    enabled: bool,
    a_max_mps2: float,
) -> float:
    if not enabled or v_prev_mps is None or t_prev_s is None:
        return float(v_curr_mps)
    dt = float(t_curr_s - t_prev_s)
    if dt <= 0.0:
        return float(v_curr_mps)
    dv_max = float(a_max_mps2) * dt
    dv = float(v_curr_mps - v_prev_mps)
    if dv > dv_max:
        return float(v_prev_mps + dv_max)
    if dv < -dv_max:
        return float(v_prev_mps - dv_max)
    return float(v_curr_mps)

