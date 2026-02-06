from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import logging
import math
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

BBoxXYXY = Tuple[float, float, float, float]
TrajectoryDot = Tuple[float, float, float]  # (x_m, y_m, t_s)


@dataclass(frozen=True)
class Detection:
    bbox_xyxy: BBoxXYXY
    score: float
    class_id: int
    class_name: str


@dataclass
class TrackState:
    track_id: int
    camera_id: str
    class_id: int
    class_name: str
    bbox_xyxy: BBoxXYXY
    score: float
    age_frames: int = 0
    hits: int = 0
    time_since_update: int = 0


@dataclass(frozen=True)
class Track:
    camera_id: str
    frame_index: int
    timestamp_s: float
    state: TrackState
    world_xy_m: Optional[Tuple[float, float]] = None


@dataclass(frozen=True)
class SpeedSample:
    camera_id: str
    track_id: int
    timestamp_s: float
    frame_index: int
    world_xy_m: Tuple[float, float]
    speed_mps_raw: float
    speed_mps_limited: float
    speed_mps_smoothed: float
    heading_deg: float
    turn_angle_deg: float
    turn_angle_signed_deg: float
    curvature_1pm: float
    angular_rate_deg_s: float
    turn_angle_dot_cross_signed_deg: float
    angular_rate_dot_cross_signed_deg_s: float
    trajectory_dots: Tuple[TrajectoryDot, ...]
    metadata: Dict[str, float]


def heading_deg_from_delta(dx: float, dy: float) -> float:
    ang = math.degrees(math.atan2(dy, dx))
    if ang < 0.0:
        ang += 360.0
    return float(ang)


def wrap_angle_deg(angle: float) -> float:
    a = float(angle) % 360.0
    if a >= 180.0:
        a -= 360.0
    return float(a)


def angle_between_deg(a_dx: float, a_dy: float, b_dx: float, b_dy: float) -> float:
    # Equivalent to legacy angle_between_deg: normalized dot, clamped acos in degrees.
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


def _signed_angle_deg_from_vectors(a_dx: float, a_dy: float, b_dx: float, b_dy: float) -> float:
    # Equivalent to legacy dot/cross signed angle: magnitude from acos(dot), sign from 2D cross.
    mag = float(angle_between_deg(a_dx, a_dy, b_dx, b_dy))
    cross = float(a_dx * b_dy - a_dy * b_dx)
    if cross > 0.0:
        return mag
    if cross < 0.0:
        return -mag
    return 0.0


def turning_dot_cross_metrics(dots: Sequence[TrajectoryDot]) -> Tuple[float, float]:
    # NOTE:
    # Curvature-based turning is authoritative.
    # Dot/Cross turning is diagnostic-only.
    #
    # Equivalent to v5 pre-refactor _turn_from_dots_dot_cross: start->mid and mid->end vectors.
    if len(dots) < 4:
        return (0.0, 0.0)
    mid = len(dots) // 2
    x0, y0, t0 = dots[0]
    xm, ym, _tm = dots[mid]
    x1, y1, t1 = dots[-1]
    v0x = float(xm - x0)
    v0y = float(ym - y0)
    v1x = float(x1 - xm)
    v1y = float(y1 - ym)
    ang = _signed_angle_deg_from_vectors(v0x, v0y, v1x, v1y)
    dt = float(t1 - t0)
    if dt <= 0.0:
        return (float(ang), 0.0)
    return (float(ang), float(ang / dt))


def _angular_rate_deg_s(v_mps: float, curvature_1pm: float) -> float:
    # Equivalent to legacy pipeline omega_deg_s: degrees(v * curvature).
    return float(math.degrees(float(v_mps) * float(curvature_1pm)))


def mps_to_kmh(v_mps: float) -> float:
    return float(float(v_mps) * 3.6)


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


@dataclass
class EmaSmoother:
    alpha: float
    max_gap_s: float
    _value: Optional[float] = None
    _t_last_s: Optional[float] = None

    def update(self, value: float, t_s: float) -> float:
        if self._value is None or self._t_last_s is None:
            self._value = float(value)
            self._t_last_s = float(t_s)
            return float(self._value)
        dt = float(t_s - self._t_last_s)
        if dt < 0.0 or dt > self.max_gap_s or value == 0.0:
            self._value = float(value)
            self._t_last_s = float(t_s)
            return float(self._value)
        if float(self._value) == 0.0:
            self._value = float(value)
            self._t_last_s = float(t_s)
            return float(self._value)
        a = float(self.alpha)
        self._value = a * float(value) + (1.0 - a) * float(self._value)
        self._t_last_s = float(t_s)
        return float(self._value)


@dataclass(frozen=True)
class TurningMetrics:
    heading_deg: float
    turn_angle_deg: float
    curvature_1pm: float
    arc_len_m: float


class TurningConfig:
    def __init__(
        self,
        min_speed_mps: float = 0.5,
        min_arc_length_m: float = 0.1,
        max_angular_rate_deg_s: float = 45.0,
        max_angular_accel_deg_s2: float = 180.0,
        min_curvature_1pm: float = 0.005,
        smoothing_window_min: int = 3,
        smoothing_window_max: int = 7,
        curvature_filter_sigma: float = 1.0,
        enable_adaptive_thresholds: bool = True,
        low_speed_threshold_mps: float = 2.0,
        high_speed_threshold_mps: float = 15.0,
    ):
        self.min_speed_mps = min_speed_mps
        self.min_arc_length_m = min_arc_length_m
        self.max_angular_rate_deg_s = max_angular_rate_deg_s
        self.max_angular_accel_deg_s2 = max_angular_accel_deg_s2
        self.min_curvature_1pm = min_curvature_1pm
        self.smoothing_window_min = smoothing_window_min
        self.smoothing_window_max = smoothing_window_max
        self.curvature_filter_sigma = curvature_filter_sigma
        self.enable_adaptive_thresholds = enable_adaptive_thresholds
        self.low_speed_threshold_mps = low_speed_threshold_mps
        self.high_speed_threshold_mps = high_speed_threshold_mps

    def get_speed_dependent_min_curvature(self, speed_mps: float) -> float:
        if not isinstance(speed_mps, (int, float)) or speed_mps < 0:
            return float(self.min_curvature_1pm)
        if not bool(self.enable_adaptive_thresholds):
            return float(self.min_curvature_1pm)
        if speed_mps < float(self.low_speed_threshold_mps):
            return float(self.min_curvature_1pm) * 0.5
        if speed_mps > float(self.high_speed_threshold_mps):
            return float(self.min_curvature_1pm) * 2.0
        denom = float(self.high_speed_threshold_mps) - float(self.low_speed_threshold_mps)
        if denom <= 0:
            return float(self.min_curvature_1pm)
        speed_ratio = (float(speed_mps) - float(self.low_speed_threshold_mps)) / denom
        return float(self.min_curvature_1pm) * (0.5 + speed_ratio * 1.5)


def _compute_derivatives(positions: np.ndarray, times: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = len(positions)
    velocities = np.zeros_like(positions, dtype=float)
    accelerations = np.zeros_like(positions, dtype=float)
    for i in range(1, n - 1):
        dt_central = times[i + 1] - times[i - 1]
        if dt_central > 0:
            velocities[i] = (positions[i + 1] - positions[i - 1]) / dt_central
    if n >= 2:
        dt_start = times[1] - times[0]
        dt_end = times[-1] - times[-2]
        if dt_start > 0:
            velocities[0] = (positions[1] - positions[0]) / dt_start
        if dt_end > 0:
            velocities[-1] = (positions[-1] - positions[-2]) / dt_end
    if n >= 3:
        for i in range(1, n - 1):
            dt_vel = times[i + 1] - times[i - 1]
            if dt_vel > 0 and i < n - 2:
                accelerations[i] = (velocities[i + 1] - velocities[i - 1]) / dt_vel
    return velocities, accelerations


def _compute_curvature_and_angular_rate(velocities: np.ndarray, accelerations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = len(velocities)
    curvatures = np.zeros(n, dtype=float)
    angular_rates = np.zeros(n, dtype=float)
    for i in range(n):
        vx, vy = velocities[i]
        ax, ay = accelerations[i]
        speed = math.sqrt(vx**2 + vy**2)
        if speed < 1e-6:
            continue
        cross_product = vx * ay - vy * ax
        speed_cubed = speed**3
        if speed_cubed > 1e-6:
            curvature = cross_product / speed_cubed
            angular_rate_rad_s = speed * abs(curvature)
            curvatures[i] = curvature
            angular_rates[i] = math.degrees(angular_rate_rad_s)
    return curvatures, angular_rates


def _temporal_smoothing(values: np.ndarray, window_size: int, *, sigma: float) -> np.ndarray:
    if len(values) < window_size or window_size < 3:
        return values.copy()
    smoothed = np.zeros_like(values)
    half_window = window_size // 2
    for i in range(len(values)):
        start_idx = max(0, i - half_window)
        end_idx = min(len(values), i + half_window + 1)
        window = values[start_idx:end_idx]
        finite = window[np.isfinite(window)]
        if finite.size == 0:
            smoothed[i] = 0.0
            continue
        s = float(sigma)
        if s > 0.0 and finite.size >= 3:
            med = float(np.median(finite))
            mad = float(np.median(np.abs(finite - med)))
            if mad > 0.0:
                z = np.abs(finite - med) / (1.4826 * mad)
                finite = finite[z <= s]
                if finite.size == 0:
                    smoothed[i] = med
                    continue
        smoothed[i] = float(np.mean(finite))
    return smoothed


def _apply_physical_constraints(
    curvatures: np.ndarray, angular_rates: np.ndarray, speeds: np.ndarray, config: TurningConfig
) -> Tuple[np.ndarray, np.ndarray]:
    angular_rates_clamped = np.clip(angular_rates, 0, config.max_angular_rate_deg_s)
    valid_speed_mask = speeds >= config.min_speed_mps
    curvatures_constrained = curvatures.copy()
    angular_rates_constrained = angular_rates_clamped.copy()
    curvatures_constrained[~valid_speed_mask] = 0.0
    angular_rates_constrained[~valid_speed_mask] = 0.0
    for i in range(len(curvatures_constrained)):
        if not bool(valid_speed_mask[i]):
            continue
        if abs(float(curvatures_constrained[i])) <= 0.0:
            continue
        thr = float(config.get_speed_dependent_min_curvature(float(speeds[i])))
        if abs(float(curvatures_constrained[i])) < thr:
            curvatures_constrained[i] = 0.0
            angular_rates_constrained[i] = 0.0
    return curvatures_constrained, angular_rates_constrained


def compute_turning_metrics_improved(dots: Sequence[TrajectoryDot], config: Optional[TurningConfig] = None) -> TurningMetrics:
    logger = logging.getLogger(__name__)
    if config is None:
        config = TurningConfig()
    if len(dots) < 3:
        return TurningMetrics(heading_deg=float("nan"), turn_angle_deg=0.0, curvature_1pm=0.0, arc_len_m=0.0)
    positions = np.array([(x, y) for x, y, _t in dots], dtype=float)
    times = np.array([t for _x, _y, t in dots], dtype=float)
    velocities, accelerations = _compute_derivatives(positions, times)
    speeds = np.linalg.norm(velocities, axis=1)
    curvatures, angular_rates = _compute_curvature_and_angular_rate(velocities, accelerations)
    curvatures_constrained, angular_rates_constrained = _apply_physical_constraints(curvatures, angular_rates, speeds, config)
    avg_speed = np.mean(speeds[speeds >= config.min_speed_mps]) if np.any(speeds >= config.min_speed_mps) else 1.0
    smoothing_window = max(
        config.smoothing_window_min,
        min(config.smoothing_window_max, int(2.0 / avg_speed + config.smoothing_window_min)),
    )
    sigma = float(getattr(config, "curvature_filter_sigma", 0.0) or 0.0)
    curvatures_smooth = _temporal_smoothing(curvatures_constrained, smoothing_window, sigma=sigma)
    angular_rates_smooth = _temporal_smoothing(angular_rates_constrained, smoothing_window, sigma=sigma)
    latest_valid_idx = -1
    for i in range(len(dots) - 1, -1, -1):
        if speeds[i] >= config.min_speed_mps and abs(curvatures_smooth[i]) >= config.min_curvature_1pm:
            latest_valid_idx = i
            break
    if latest_valid_idx == -1:
        final_heading = math.degrees(math.atan2(velocities[-1, 1], velocities[-1, 0]))
        if final_heading < 0:
            final_heading += 360.0
        return TurningMetrics(
            heading_deg=float(final_heading),
            turn_angle_deg=0.0,
            curvature_1pm=0.0,
            arc_len_m=float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))),
        )
    final_curvature = curvatures_smooth[latest_valid_idx]
    final_angular_rate = angular_rates_smooth[latest_valid_idx]
    final_heading = math.degrees(math.atan2(velocities[-1, 1], velocities[-1, 0]))
    if final_heading < 0:
        final_heading += 360.0
    dt = np.diff(times)
    turn_angle = np.sum(angular_rates_smooth[1:] * dt)
    mean_curvature = np.mean(curvatures_smooth)
    if mean_curvature < 0:
        turn_angle = -turn_angle
    arc_length = float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)))
    if arc_length < config.min_arc_length_m:
        turn_angle = 0.0
        final_curvature = 0.0
        final_angular_rate = 0.0
        final_heading = float("nan")
        logger.debug(
            "arc_length_guard_enforced: arc_length=%.6fm < %.6fm → zero motion enforced",
            arc_length,
            config.min_arc_length_m,
        )
    elif avg_speed < config.min_speed_mps:
        final_heading = float("nan")
    logger.debug(
        "turning_metrics_improved: heading=%.2f° turn_angle=%.2f° curvature=%.6f 1/m angular_rate=%.2f°/s arc_length=%.3fm avg_speed=%.2f m/s valid_points=%d/%d",
        final_heading,
        turn_angle,
        final_curvature,
        final_angular_rate,
        arc_length,
        avg_speed,
        int(np.sum(speeds >= config.min_speed_mps)),
        len(dots),
    )
    return TurningMetrics(
        heading_deg=float(final_heading),
        turn_angle_deg=float(turn_angle),
        curvature_1pm=float(final_curvature),
        arc_len_m=arc_length,
    )


def _detect_zero_velocity_segments_authoritative(
    positions: np.ndarray,
    times: np.ndarray,
    movement_threshold_m: float,
    min_time_window_s: float,
    min_arc_length_m: float,
) -> np.ndarray:
    if len(positions) < 2 or len(times) < 2:
        return np.zeros(len(positions), dtype=bool)
    distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    cumulative_dist = np.concatenate([[0.0], np.cumsum(distances)])
    zero_velocity_mask = np.zeros(len(positions), dtype=bool)
    for i in range(len(positions)):
        if i == 0:
            zero_velocity_mask[i] = False
            continue
        current_time = times[i]
        window_start_idx = i
        for j in range(i - 1, -1, -1):
            if current_time - times[j] >= min_time_window_s:
                window_start_idx = j
                break
            window_start_idx = j
        if window_start_idx < i:
            movement_in_window = cumulative_dist[i] - cumulative_dist[window_start_idx]
            time_in_window = times[i] - times[window_start_idx]
            if (
                movement_in_window < movement_threshold_m
                and time_in_window >= min_time_window_s
                and movement_in_window < min_arc_length_m
            ):
                zero_velocity_mask[i] = True
    return zero_velocity_mask


def _detect_zero_velocity_segments(
    positions: np.ndarray,
    times: np.ndarray,
    movement_threshold_m: float,
    min_time_window_s: float,
) -> np.ndarray:
    return _detect_zero_velocity_segments_authoritative(positions, times, movement_threshold_m, min_time_window_s, 0.0)


@dataclass(frozen=True)
class SpeedEstimatorConfig:
    min_dt_s: float
    max_dt_s: float
    min_displacement_m: float
    min_speed_mps: float
    stop_time_s: float
    min_arc_length_m: float
    turning_min_arc_len_m: float
    turning_persist_s: float
    max_turn_rate_deg_per_s: float
    theta_min_deg: float
    curvature_min_1pm: float
    turn_limit: TurnSpeedLimitConfig
    accel_limit_enabled: bool
    a_max_mps2: float
    smoothing_enabled: bool
    smoothing_alpha: float
    smoothing_max_gap_s: float
    disable_turn_limit: bool
    disable_accel_limit: bool
    disable_smoothing: bool
    dot_min_distance_m: float
    dot_min_dt_s: float
    dot_buffer_size: int
    mode: str
    simple_window_s: float
    simple_axis: str
    simple_method: str
    turning_config: TurningConfig

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "SpeedEstimatorConfig":
        mode = str(d.get("mode", "advanced")).lower()
        if mode not in {"advanced", "simple"}:
            raise ValueError("mode must be one of: advanced, simple")
        simple = d.get("simple_speed", {}) or {}
        simple_window_s = float(simple.get("window_s", 1.0))
        simple_axis = str(simple.get("axis", "y")).lower()
        if simple_axis not in {"x", "y", "xy"}:
            raise ValueError("simple_speed.axis must be one of: x, y, xy")
        simple_method = str(simple.get("method", "mean")).lower()
        if simple_method not in {"mean", "displacement"}:
            raise ValueError("simple_speed.method must be one of: mean, displacement")
        raw = d.get("raw_speed", {})
        turning = d.get("turning", {})
        tsl = d.get("turn_speed_limit", {})
        accel = d.get("accel_limit", {})
        smoothing = d.get("smoothing", {})
        ab = d.get("ablations", {})
        dots = d.get("dots", {})
        units = d.get("units", {})
        output_units = str(units.get("output", "kmh")).lower()
        if output_units not in {"kmh", "mph", "mps"}:
            raise ValueError("units.output must be one of: kmh, mph, mps")
        v_max_kmh = float(tsl.get("v_max_kmh", 180.0))
        v_max_mps = v_max_kmh / 3.6
        v_min_kmh = float(tsl.get("v_min_kmh", 30.0))
        v_min_mps = v_min_kmh / 3.6
        min_speed_mps = float(raw.get("min_speed_mps", 0.03))
        dot_min_distance_m = float(dots.get("min_distance_m", 0.20))
        dot_min_dt_s = float(dots.get("min_dt_s", 0.2))
        dot_buffer_size = int(dots.get("buffer_size", 5))
        dot_buffer_size = max(4, min(6, dot_buffer_size))
        return SpeedEstimatorConfig(
            min_dt_s=float(raw.get("min_dt_s", 0.04)),
            max_dt_s=float(raw.get("max_dt_s", 1.0)),
            min_displacement_m=float(raw.get("min_displacement_m", 0.15)),
            min_speed_mps=min_speed_mps,
            stop_time_s=float(d.get("stop_time_s", raw.get("stop_time_s", 0.5))),
            min_arc_length_m=float(d.get("min_arc_length_m", raw.get("min_arc_length_m", 0.1))),
            turning_min_arc_len_m=float(turning.get("min_arc_len_m", 0.05)),
            turning_persist_s=float(turning.get("persist_s", 0.25)),
            max_turn_rate_deg_per_s=float(turning.get("max_turn_rate_deg_per_s", 45.0)),
            theta_min_deg=float(turning.get("theta_min_deg", 8.0)),
            curvature_min_1pm=float(turning.get("curvature_min_1pm", 0.015)),
            turn_limit=TurnSpeedLimitConfig(
                enabled=bool(tsl.get("enabled", True)),
                a_lat_max_mps2=float(tsl.get("a_lat_max_mps2", 2.5)),
                v_max_mps=float(v_max_mps),
                v_min_mps=float(v_min_mps),
                angle_max_deg=float(tsl.get("angle_max_deg", 60.0)),
                mode=str(tsl.get("mode", "curvature")),
                alpha=float(tsl.get("alpha", 0.5)),
            ),
            accel_limit_enabled=bool(accel.get("enabled", True)),
            a_max_mps2=float(accel.get("a_max_mps2", 60.0)),
            smoothing_enabled=bool(smoothing.get("method", "ema") == "ema"),
            smoothing_alpha=float(smoothing.get("ema_alpha", 0.25)),
            smoothing_max_gap_s=float(smoothing.get("max_gap_s", 1.0)),
            disable_turn_limit=bool(ab.get("disable_turn_limit", False)),
            disable_accel_limit=bool(ab.get("disable_accel_limit", False)),
            disable_smoothing=bool(ab.get("disable_smoothing", False)),
            dot_min_distance_m=float(dot_min_distance_m),
            dot_min_dt_s=float(dot_min_dt_s),
            dot_buffer_size=int(dot_buffer_size),
            mode=mode,
            simple_window_s=simple_window_s,
            simple_axis=simple_axis,
            simple_method=simple_method,
            turning_config=TurningConfig(
                min_speed_mps=float(turning.get("min_speed_mps", 0.5)),
                min_arc_length_m=float(turning.get("min_arc_length_m", 0.1)),
                max_angular_rate_deg_s=float(turning.get("max_angular_rate_deg_s", 45.0)),
                max_angular_accel_deg_s2=float(turning.get("max_angular_accel_deg_s2", 180.0)),
                min_curvature_1pm=float(turning.get("min_curvature_1pm", 0.005)),
                smoothing_window_min=int(turning.get("smoothing_window_min", 3)),
                smoothing_window_max=int(turning.get("smoothing_window_max", 7)),
                curvature_filter_sigma=float(turning.get("curvature_filter_sigma", 1.0)),
                enable_adaptive_thresholds=bool(turning.get("enable_adaptive_thresholds", True)),
                low_speed_threshold_mps=float(turning.get("low_speed_threshold_mps", 2.0)),
                high_speed_threshold_mps=float(turning.get("high_speed_threshold_mps", 15.0)),
            ),
        )


@dataclass
class _TrackHistory:
    world_xy_m: Deque[Tuple[float, float]]
    timestamps_s: Deque[float]
    frame_indices: Deque[int]
    dots: Deque[TrajectoryDot]
    smoother: Optional[EmaSmoother]
    v_prev_mps: Optional[float] = None
    t_prev_s: Optional[float] = None
    prev_turn_angle_deg: Optional[float] = None
    prev_turn_t_s: Optional[float] = None


class SpeedEstimator:
    def __init__(self, cfg: SpeedEstimatorConfig, max_history: int = 64) -> None:
        self._cfg = cfg
        self._max_history = int(max_history)
        self._hist: Dict[Tuple[str, int], _TrackHistory] = {}
        self._logger = logging.getLogger(__name__)

    def get_trajectory_dots(self, camera_id: str, track_id: int) -> Tuple[TrajectoryDot, ...]:
        h = self._hist.get((str(camera_id), int(track_id)))
        if h is None:
            return ()
        return tuple(h.dots)

    def reset(self) -> None:
        self._hist.clear()

    def update(self, tracks: List[Track]) -> List[SpeedSample]:
        out: List[SpeedSample] = []
        for tr in tracks:
            if tr.world_xy_m is None:
                continue
            key = (tr.camera_id, tr.state.track_id)
            h = self._hist.get(key)
            if h is None:
                smoother = None
                if self._cfg.smoothing_enabled and not self._cfg.disable_smoothing:
                    smoother = EmaSmoother(alpha=self._cfg.smoothing_alpha, max_gap_s=self._cfg.smoothing_max_gap_s)
                h = _TrackHistory(
                    world_xy_m=deque(maxlen=self._max_history),
                    timestamps_s=deque(maxlen=self._max_history),
                    frame_indices=deque(maxlen=self._max_history),
                    dots=deque(maxlen=self._cfg.dot_buffer_size),
                    smoother=smoother,
                )
                self._hist[key] = h
            h.world_xy_m.append(tr.world_xy_m)
            h.timestamps_s.append(tr.timestamp_s)
            h.frame_indices.append(tr.frame_index)
            curr_x, curr_y = float(tr.world_xy_m[0]), float(tr.world_xy_m[1])
            curr_t = float(tr.timestamp_s)
            dot_added = False
            if not h.dots:
                h.dots.append((curr_x, curr_y, curr_t))
                dot_added = True
                if self._logger.isEnabledFor(logging.DEBUG):
                    self._logger.debug(
                        "dot_added: cam=%s track=%d first_dot=(%.2f, %.2f, %.3f)",
                        tr.camera_id,
                        tr.state.track_id,
                        curr_x,
                        curr_y,
                        curr_t,
                    )
            else:
                last_x, last_y, last_t = h.dots[-1]
                if curr_t > last_t:
                    dx = curr_x - last_x
                    dy = curr_y - last_y
                    dt_dot = curr_t - last_t
                    dist = (dx * dx + dy * dy) ** 0.5
                    if dist >= self._cfg.dot_min_distance_m and dt_dot >= self._cfg.dot_min_dt_s:
                        h.dots.append((curr_x, curr_y, curr_t))
                        dot_added = True
                        if self._logger.isEnabledFor(logging.DEBUG):
                            self._logger.debug(
                                "dot_added: cam=%s track=%d dist=%.3fm dt=%.3fs dots=%d",
                                tr.camera_id,
                                tr.state.track_id,
                                dist,
                                dt_dot,
                                len(h.dots),
                            )
                    else:
                        if self._logger.isEnabledFor(logging.DEBUG):
                            self._logger.debug(
                                "dot_rejected: cam=%s track=%d dist=%.3fm (need %.3f) dt=%.3fs (need %.3f)",
                                tr.camera_id,
                                tr.state.track_id,
                                dist,
                                self._cfg.dot_min_distance_m,
                                dt_dot,
                                self._cfg.dot_min_dt_s,
                            )
            if self._cfg.mode == "simple":
                if not dot_added:
                    continue
                if len(h.dots) < 2:
                    continue
                p1 = h.dots[-1]
                t1 = float(p1[2])
                target_t = t1 - max(0.0, float(self._cfg.simple_window_s))
                idx0 = 0
                for i in range(len(h.dots) - 1, -1, -1):
                    if h.dots[i][2] <= target_t:
                        idx0 = i
                        break
                if idx0 >= len(h.dots) - 1:
                    continue
                disp_sum = 0.0
                dt_sum = 0.0
                speeds: List[float] = []
                for i in range(idx0 + 1, len(h.dots)):
                    p_prev = h.dots[i - 1]
                    p_curr = h.dots[i]
                    t_prev = float(p_prev[2])
                    t_curr = float(p_curr[2])
                    dt = float(t_curr - t_prev)
                    if dt <= 0.0:
                        continue
                    if self._cfg.simple_axis == "x":
                        disp = abs(p_curr[0] - p_prev[0])
                    elif self._cfg.simple_axis == "y":
                        disp = abs(p_curr[1] - p_prev[1])
                    else:
                        disp = ((p_curr[0] - p_prev[0]) ** 2 + (p_curr[1] - p_prev[1]) ** 2) ** 0.5
                    disp_sum += disp
                    dt_sum += dt
                    speeds.append(float(disp / dt))
                if not speeds or dt_sum <= 0.0:
                    continue
                positions_simple = np.array([(p[0], p[1]) for p in h.dots], dtype=float)
                times_simple = np.array([p[2] for p in h.dots], dtype=float)
                zero_velocity_mask_simple = _detect_zero_velocity_segments_authoritative(
                    positions_simple,
                    times_simple,
                    movement_threshold_m=float(self._cfg.min_displacement_m),
                    min_time_window_s=float(self._cfg.stop_time_s),
                    min_arc_length_m=float(self._cfg.min_arc_length_m),
                )
                zero_velocity_detected = False
                if zero_velocity_mask_simple[-1]:
                    v_raw = 0.0
                    zero_velocity_detected = True
                    if self._logger.isEnabledFor(logging.DEBUG):
                        self._logger.debug(
                            "zero_velocity_detected_simple: cam=%s track=%d t=%.3f disp_sum=%.6fm (below threshold %.6fm)",
                            tr.camera_id,
                            tr.state.track_id,
                            float(t1),
                            disp_sum,
                            max(self._cfg.min_displacement_m, self._cfg.dot_min_distance_m),
                        )
                elif self._cfg.simple_method == "mean":
                    v_raw = float(sum(speeds) / len(speeds))
                else:
                    v_raw = float(disp_sum / dt_sum)
                total_arc_length = 0.0
                for i in range(1, len(h.dots)):
                    p_prev = h.dots[i - 1]
                    p_curr = h.dots[i]
                    segment_length = ((p_curr[0] - p_prev[0]) ** 2 + (p_curr[1] - p_prev[1]) ** 2) ** 0.5
                    total_arc_length += segment_length
                if total_arc_length < self._cfg.turning_config.min_arc_length_m:
                    v_raw = 0.0
                    zero_velocity_detected = True
                    if self._logger.isEnabledFor(logging.DEBUG):
                        self._logger.debug(
                            "arc_length_zero_speed_override_simple: cam=%s track=%d t=%.3f arc_len=%.6fm < %.6fm → speed forced to 0",
                            tr.camera_id,
                            tr.state.track_id,
                            float(t1),
                            total_arc_length,
                            self._cfg.turning_config.min_arc_length_m,
                        )
                if disp_sum < max(self._cfg.min_displacement_m, self._cfg.dot_min_distance_m) or v_raw < self._cfg.min_speed_mps:
                    v_raw = 0.0
                    zero_velocity_detected = True
                if h.smoother is None:
                    v_smooth = float(v_raw)
                else:
                    v_smooth = h.smoother.update(float(v_raw), float(t1))
                if zero_velocity_detected or v_smooth < self._cfg.min_speed_mps:
                    v_smooth = 0.0
                if zero_velocity_detected:
                    h.v_prev_mps = 0.0
                    if h.smoother is not None:
                        h.smoother._value = 0.0
                        h.smoother._t_last_s = float(t1)
                else:
                    h.v_prev_mps = float(v_smooth)
                h.t_prev_s = float(t1)
                dots_now = tuple(h.dots)
                turn_vec, omega_vec = turning_dot_cross_metrics(dots_now)
                out.append(
                    SpeedSample(
                        camera_id=tr.camera_id,
                        track_id=tr.state.track_id,
                        timestamp_s=float(t1),
                        frame_index=int(tr.frame_index),
                        world_xy_m=(float(p1[0]), float(p1[1])),
                        speed_mps_raw=float(v_raw),
                        speed_mps_limited=float(v_raw),
                        speed_mps_smoothed=float(v_smooth),
                        heading_deg=0.0,
                        turn_angle_deg=0.0,
                        turn_angle_signed_deg=0.0,
                        curvature_1pm=0.0,
                        angular_rate_deg_s=0.0,
                        turn_angle_dot_cross_signed_deg=float(turn_vec),
                        angular_rate_dot_cross_signed_deg_s=float(omega_vec),
                        trajectory_dots=dots_now,
                        metadata={
                            "dt_s": float(dt_sum),
                            "disp_m": float(disp_sum),
                            "window_s": float(self._cfg.simple_window_s),
                        },
                    )
                )
                continue
            if not dot_added:
                continue
            if len(h.dots) < 2:
                continue
            p0 = h.dots[-2]
            p1 = h.dots[-1]
            t0 = float(p0[2])
            t1 = float(p1[2])
            dt = float(t1 - t0)
            if dt < self._cfg.min_dt_s or dt > self._cfg.max_dt_s:
                continue
            v_raw = speed_mps((p0[0], p0[1]), t0, (p1[0], p1[1]), t1)
            if v_raw is None:
                continue
            disp = ((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2) ** 0.5
            positions = np.array([(p[0], p[1]) for p in h.dots], dtype=float)
            times = np.array([p[2] for p in h.dots], dtype=float)
            zero_velocity_mask = _detect_zero_velocity_segments_authoritative(
                positions,
                times,
                movement_threshold_m=float(self._cfg.min_displacement_m),
                min_time_window_s=float(self._cfg.stop_time_s),
                min_arc_length_m=float(self._cfg.min_arc_length_m),
            )
            zero_velocity_detected = False
            if zero_velocity_mask[-1]:
                v_raw = 0.0
                zero_velocity_detected = True
                if self._logger.isEnabledFor(logging.DEBUG):
                    self._logger.debug(
                        "zero_velocity_detected: cam=%s track=%d t=%.3f disp=%.6fm (below threshold %.6fm)",
                        tr.camera_id,
                        tr.state.track_id,
                        float(t1),
                        disp,
                        max(self._cfg.min_displacement_m, self._cfg.dot_min_distance_m),
                    )
            elif disp < max(self._cfg.min_displacement_m, self._cfg.dot_min_distance_m):
                v_raw = 0.0
                zero_velocity_detected = True
            turning = compute_turning_metrics_improved(h.dots, self._cfg.turning_config)
            turn_angle_signed_deg = float(turning.turn_angle_deg)
            curvature_1pm = float(turning.curvature_1pm)
            if turning.arc_len_m < self._cfg.turning_config.min_arc_length_m:
                v_raw = 0.0
                zero_velocity_detected = True
                if self._logger.isEnabledFor(logging.DEBUG):
                    self._logger.debug(
                        "arc_length_zero_speed_override: cam=%s track=%d t=%.3f arc_len=%.6fm < %.6fm → speed forced to 0",
                        tr.camera_id,
                        tr.state.track_id,
                        float(t1),
                        turning.arc_len_m,
                        self._cfg.turning_config.min_arc_length_m,
                    )
            if self._logger.isEnabledFor(logging.DEBUG):
                self._logger.debug(
                    "turning_computed: cam=%s track=%d t=%.3f angle_signed=%.3f° curvature=%.6f arc=%.3fm dots=%d",
                    tr.camera_id,
                    tr.state.track_id,
                    float(t1),
                    float(turn_angle_signed_deg),
                    float(curvature_1pm),
                    float(turning.arc_len_m),
                    len(h.dots),
                )
                if abs(turn_angle_signed_deg) > 0.1:
                    self._logger.debug(
                        "turning_math_validation: angle_signed=%.3f° angle_abs=%.3f° direction=%s",
                        float(turn_angle_signed_deg),
                        abs(float(turn_angle_signed_deg)),
                        "left" if turn_angle_signed_deg > 0 else "right",
                    )
            arc_too_small = turning.arc_len_m < self._cfg.turning_min_arc_len_m
            speed_too_small = v_raw < self._cfg.min_speed_mps
            apply_turn_thresh = (abs(turn_angle_signed_deg) >= self._cfg.theta_min_deg) or (abs(curvature_1pm) >= self._cfg.curvature_min_1pm)
            if self._logger.isEnabledFor(logging.DEBUG):
                self._logger.debug(
                    "turning_decision: arc_too_small=%s (%.3f < %.3f) speed_too_small=%s (%.3f < %.3f) apply_thresh=%s (|angle|=%.3f° >= %.3f° OR |curvature|=%.6f >= %.6f)",
                    arc_too_small,
                    turning.arc_len_m,
                    self._cfg.turning_min_arc_len_m,
                    speed_too_small,
                    v_raw,
                    self._cfg.min_speed_mps,
                    apply_turn_thresh,
                    abs(turn_angle_signed_deg),
                    self._cfg.theta_min_deg,
                    abs(curvature_1pm),
                    self._cfg.curvature_min_1pm,
                )
            if zero_velocity_detected:
                turn_angle_signed_deg = 0.0
                curvature_1pm = 0.0
                h.prev_turn_angle_deg = None
                h.prev_turn_t_s = None
            original_heading_deg = float(turning.heading_deg)
            heading_deg_final = 0.0 if zero_velocity_detected else original_heading_deg
            if h.prev_turn_t_s is not None:
                dt_since_last_turn = float(t1 - h.prev_turn_t_s)
                if dt_since_last_turn > 2.0:
                    h.prev_turn_angle_deg = None
                    h.prev_turn_t_s = None
            max_rate = float(self._cfg.max_turn_rate_deg_per_s)
            if max_rate > 0.0 and h.prev_turn_angle_deg is not None and h.prev_turn_t_s is not None:
                dt_turn = float(t1 - h.prev_turn_t_s)
                if dt_turn > 0.0:
                    max_delta = max_rate * dt_turn
                    lo = h.prev_turn_angle_deg - max_delta
                    hi = h.prev_turn_angle_deg + max_delta
                    capped = min(max(turn_angle_signed_deg, lo), hi)
                    if abs(turn_angle_signed_deg) > 1e-6:
                        curvature_1pm = curvature_1pm * (capped / turn_angle_signed_deg)
                    else:
                        curvature_1pm = 0.0
                    turn_angle_signed_deg = capped
            persist_ok = False
            if h.prev_turn_angle_deg is not None and h.prev_turn_t_s is not None:
                dt_turn_prev = float(t1 - h.prev_turn_t_s)
                if dt_turn_prev <= self._cfg.turning_persist_s and abs(h.prev_turn_angle_deg) >= (0.8 * self._cfg.theta_min_deg):
                    persist_ok = True
            apply_turn_thresh = (abs(turn_angle_signed_deg) >= self._cfg.theta_min_deg) or (abs(curvature_1pm) >= self._cfg.curvature_min_1pm)
            apply_turn = (not arc_too_small) and (not speed_too_small) and (apply_turn_thresh or persist_ok)
            if self._logger.isEnabledFor(logging.DEBUG):
                self._logger.debug(
                    "turning_final: apply_turn=%s (not arc_too_small=%s AND not speed_too_small=%s AND (apply_thresh=%s OR persist_ok=%s))",
                    apply_turn,
                    (not arc_too_small),
                    (not speed_too_small),
                    apply_turn_thresh,
                    persist_ok,
                )
            curvature_for_limit = curvature_1pm if apply_turn else 0.0
            angle_for_limit = turn_angle_signed_deg if apply_turn else 0.0
            angle_display_deg = abs(float(turn_angle_signed_deg))
            if angle_display_deg < self._cfg.theta_min_deg and h.prev_turn_angle_deg is not None and h.prev_turn_t_s is not None:
                dt_display_prev = float(t1 - h.prev_turn_t_s)
                if dt_display_prev <= self._cfg.turning_persist_s:
                    angle_display_deg = abs(float(h.prev_turn_angle_deg))
            turn_cfg = self._cfg.turn_limit
            if self._cfg.disable_turn_limit:
                v_turn_limited = float(v_raw)
            else:
                v_turn_limited = turn_limited_speed_mps(
                    v_raw_mps=v_raw,
                    curvature_1pm=curvature_for_limit,
                    turn_angle_deg=angle_for_limit,
                    cfg=turn_cfg,
                )
            if self._cfg.disable_accel_limit:
                v_acc_limited = float(v_turn_limited)
            else:
                v_acc_limited = accel_limited_speed_mps(
                    v_prev_mps=h.v_prev_mps,
                    t_prev_s=h.t_prev_s,
                    v_curr_mps=v_turn_limited,
                    t_curr_s=float(t1),
                    enabled=self._cfg.accel_limit_enabled,
                    a_max_mps2=self._cfg.a_max_mps2,
                )
            if h.smoother is None:
                v_smooth = float(v_acc_limited)
            else:
                v_smooth = h.smoother.update(v_acc_limited, float(t1))
            if zero_velocity_detected or v_smooth < self._cfg.min_speed_mps:
                v_smooth = 0.0
            if zero_velocity_detected:
                h.v_prev_mps = 0.0
                if h.smoother is not None:
                    h.smoother._value = 0.0
                    h.smoother._t_last_s = float(t1)
            else:
                h.v_prev_mps = float(v_smooth)
            h.t_prev_s = float(t1)
            dots_now = tuple(h.dots)
            turn_vec, omega_vec = turning_dot_cross_metrics(dots_now)
            out.append(
                SpeedSample(
                    camera_id=tr.camera_id,
                    track_id=tr.state.track_id,
                    timestamp_s=float(t1),
                    frame_index=int(tr.frame_index),
                    world_xy_m=(float(p1[0]), float(p1[1])),
                    speed_mps_raw=float(v_raw),
                    speed_mps_limited=float(v_acc_limited),
                    speed_mps_smoothed=float(v_smooth),
                    heading_deg=heading_deg_final,
                    turn_angle_deg=float(angle_display_deg),
                    turn_angle_signed_deg=float(turn_angle_signed_deg),
                    curvature_1pm=float(curvature_1pm),
                    angular_rate_deg_s=_angular_rate_deg_s(v_smooth, curvature_1pm),
                    turn_angle_dot_cross_signed_deg=float(turn_vec),
                    angular_rate_dot_cross_signed_deg_s=float(omega_vec),
                    trajectory_dots=dots_now,
                    metadata={
                        "dt_s": float(dt),
                        "disp_m": float(disp),
                        "turn_applied": 1.0 if apply_turn else 0.0,
                        "arc_len_m": float(turning.arc_len_m),
                        "turn_angle_signed_deg": float(turn_angle_signed_deg),
                    },
                )
            )
            if zero_velocity_detected:
                h.prev_turn_angle_deg = None
                h.prev_turn_t_s = None
            else:
                h.prev_turn_angle_deg = float(turn_angle_signed_deg)
                h.prev_turn_t_s = float(t1)
        return out

    def prune_missing(self, alive_keys: List[Tuple[str, int]]) -> None:
        alive = set(alive_keys)
        to_del = [k for k in self._hist.keys() if k not in alive]
        for k in to_del:
            del self._hist[k]
