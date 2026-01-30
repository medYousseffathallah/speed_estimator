from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

from collections import deque
import logging
import numpy as np

from speedestimation.speed_estimation.limits import TurnSpeedLimitConfig, accel_limited_speed_mps, turn_limited_speed_mps
from speedestimation.speed_estimation.math import speed_mps
from speedestimation.speed_estimation.smoothing import EmaSmoother
from speedestimation.turning_model.turning_improved import compute_turning_metrics_improved, TurningConfig, _detect_zero_velocity_segments, _detect_zero_velocity_segments_authoritative
from speedestimation.utils.types import SpeedSample, Track


@dataclass(frozen=True)
class SpeedEstimatorConfig:
    min_dt_s: float
    max_dt_s: float
    min_displacement_m: float
    min_speed_mps: float
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
        v_min_kmh = float(tsl.get("v_min_kmh", v_max_kmh))
        v_min_mps = v_min_kmh / 3.6

        min_speed_mps = float(raw.get("min_speed_mps", 1.0))

        dot_min_distance_m = float(dots.get("min_distance_m", 0.45))
        dot_min_dt_s = float(dots.get("min_dt_s", 0.3))
        dot_buffer_size = int(dots.get("buffer_size", 5))
        dot_buffer_size = max(4, min(6, dot_buffer_size))

        return SpeedEstimatorConfig(
            min_dt_s=float(raw.get("min_dt_s", 0.05)),
            max_dt_s=float(raw.get("max_dt_s", 1.0)),
            min_displacement_m=float(raw.get("min_displacement_m", 0.05)),
            min_speed_mps=min_speed_mps,
            turning_min_arc_len_m=float(turning.get("min_arc_len_m", 0.05)),
            turning_persist_s=float(turning.get("persist_s", 0.25)),
            max_turn_rate_deg_per_s=float(turning.get("max_turn_rate_deg_per_s", 0.0)),
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
            a_max_mps2=float(accel.get("a_max_mps2", 6.0)),
            smoothing_enabled=bool(smoothing.get("method", "ema") == "ema"),
            smoothing_alpha=float(smoothing.get("ema_alpha", 0.35)),
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
                smoothing_window_max=int(turning.get("smoothing_window_max", 7))
            ),
        )


@dataclass
class _TrackHistory:
    world_xy_m: Deque[Tuple[float, float]]
    timestamps_s: Deque[float]
    frame_indices: Deque[int]
    dots: Deque[Tuple[float, float, float]]
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
                        tr.camera_id, tr.state.track_id, curr_x, curr_y, curr_t
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
                                tr.camera_id, tr.state.track_id, dist, dt_dot, len(h.dots)
                            )
                    else:
                        if self._logger.isEnabledFor(logging.DEBUG):
                            self._logger.debug(
                                "dot_rejected: cam=%s track=%d dist=%.3fm (need %.3f) dt=%.3fs (need %.3f)",
                                tr.camera_id, tr.state.track_id, dist, self._cfg.dot_min_distance_m, 
                                dt_dot, self._cfg.dot_min_dt_s
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
                
                # Zero-velocity detection for simple mode using authoritative rule
                positions_simple = np.array([(p[0], p[1]) for p in h.dots], dtype=float)
                times_simple = np.array([p[2] for p in h.dots], dtype=float)
                zero_velocity_mask_simple = _detect_zero_velocity_segments_authoritative(
                    positions_simple, times_simple, 
                    movement_threshold_m=max(self._cfg.min_displacement_m, self._cfg.dot_min_distance_m),
                    min_time_window_s=getattr(self._cfg, 'stop_time_s', 0.5),
                    min_arc_length_m=getattr(self._cfg, 'min_arc_length_m', 0.1)
                )
                
                # Track if zero-velocity was detected to ensure state consistency
                zero_velocity_detected = False
                
                # Check if current point (last one) is in zero-velocity state
                if zero_velocity_mask_simple[-1]:
                    v_raw = 0.0
                    zero_velocity_detected = True
                    if self._logger.isEnabledFor(logging.DEBUG):
                        self._logger.debug(
                            "zero_velocity_detected_simple: cam=%s track=%d t=%.3f disp_sum=%.6fm (below threshold %.6fm)",
                            tr.camera_id, tr.state.track_id, float(t1), disp_sum, max(self._cfg.min_displacement_m, self._cfg.dot_min_distance_m)
                        )
                elif self._cfg.simple_method == "mean":
                    v_raw = float(sum(speeds) / len(speeds))
                else:
                    v_raw = float(disp_sum / dt_sum)
                
                # CRITICAL FIX: Hard arc-length check for simple mode consistency
                # Compute total arc length from dot trajectory
                total_arc_length = 0.0
                for i in range(1, len(h.dots)):
                    p_prev = h.dots[i-1]
                    p_curr = h.dots[i]
                    segment_length = ((p_curr[0] - p_prev[0])**2 + (p_curr[1] - p_prev[1])**2)**0.5
                    total_arc_length += segment_length
                
                if total_arc_length < self._cfg.turning_config.min_arc_length_m:
                    v_raw = 0.0
                    zero_velocity_detected = True
                    if self._logger.isEnabledFor(logging.DEBUG):
                        self._logger.debug(
                            "arc_length_zero_speed_override_simple: cam=%s track=%d t=%.3f arc_len=%.6fm < %.6fm → speed forced to 0",
                            tr.camera_id, tr.state.track_id, float(t1), 
                            total_arc_length, self._cfg.turning_config.min_arc_length_m
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
                
                # CRITICAL FIX: When zero-velocity is detected, ensure state is reset to 0
                # This prevents speed persistence from previous moving states
                if zero_velocity_detected:
                    h.v_prev_mps = 0.0
                    if h.smoother is not None:
                        h.smoother._value = 0.0
                else:
                    h.v_prev_mps = float(v_smooth)
                
                h.t_prev_s = float(t1)
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
                        curvature_1pm=0.0,
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
                
            # Speed and turning use the same dot buffer for consistency
            # Speed uses last 2 dots, turning uses first 2 and last 2 dots for direction vectors
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
            
            # Zero-velocity detection using authoritative stop rule
            positions = np.array([(p[0], p[1]) for p in h.dots], dtype=float)
            times = np.array([p[2] for p in h.dots], dtype=float)
            zero_velocity_mask = _detect_zero_velocity_segments_authoritative(
                positions, times, 
                movement_threshold_m=max(self._cfg.min_displacement_m, self._cfg.dot_min_distance_m),
                min_time_window_s=getattr(self._cfg, 'stop_time_s', 0.5),
                min_arc_length_m=getattr(self._cfg, 'min_arc_length_m', 0.1)
            )
            
            # Track if zero-velocity was detected to ensure state consistency
            zero_velocity_detected = False
            
            # Check if current point (last one) is in zero-velocity state
            if zero_velocity_mask[-1]:
                v_raw = 0.0
                zero_velocity_detected = True
                if self._logger.isEnabledFor(logging.DEBUG):
                    self._logger.debug(
                        "zero_velocity_detected: cam=%s track=%d t=%.3f disp=%.6fm (below threshold %.6fm)",
                        tr.camera_id, tr.state.track_id, float(t1), disp, max(self._cfg.min_displacement_m, self._cfg.dot_min_distance_m)
                    )
            elif disp < max(self._cfg.min_displacement_m, self._cfg.dot_min_distance_m):
                v_raw = 0.0
                zero_velocity_detected = True

            turning = compute_turning_metrics_improved(h.dots, self._cfg.turning_config)
            turn_angle_signed_deg = float(turning.turn_angle_deg)  # Keep signed for internal logic
            curvature_1pm = float(turning.curvature_1pm)
            
            # CRITICAL FIX: Hard zero-speed override based on arc length
            # This ensures consistency between turning math and speed computation
            if turning.arc_len_m < self._cfg.turning_config.min_arc_length_m:
                v_raw = 0.0
                zero_velocity_detected = True
                if self._logger.isEnabledFor(logging.DEBUG):
                    self._logger.debug(
                        "arc_length_zero_speed_override: cam=%s track=%d t=%.3f arc_len=%.6fm < %.6fm → speed forced to 0",
                        tr.camera_id, tr.state.track_id, float(t1), 
                        turning.arc_len_m, self._cfg.turning_config.min_arc_length_m
                    )
            
            # Debug turning computation details with signed angle and math validation
            if self._logger.isEnabledFor(logging.DEBUG):
                self._logger.debug(
                    "turning_computed: cam=%s track=%d t=%.3f angle_signed=%.3f° curvature=%.6f arc=%.3fm dots=%d",
                    tr.camera_id,
                    tr.state.track_id,
                    float(t1),
                    float(turn_angle_signed_deg),
                    float(curvature_1pm),
                    float(turning.arc_len_m),
                    len(h.dots)
                )
                # Additional debug for angle math validation
                if abs(turn_angle_signed_deg) > 0.1:  # Only log for meaningful angles
                    self._logger.debug(
                        "turning_math_validation: angle_signed=%.3f° angle_abs=%.3f° direction=%s",
                        float(turn_angle_signed_deg),
                        abs(float(turn_angle_signed_deg)),
                        "left" if turn_angle_signed_deg > 0 else "right"
                    )
            
            arc_too_small = (turning.arc_len_m < self._cfg.turning_min_arc_len_m)
            speed_too_small = (v_raw < self._cfg.min_speed_mps)
            apply_turn_thresh = (abs(turn_angle_signed_deg) >= self._cfg.theta_min_deg) or (abs(curvature_1pm) >= self._cfg.curvature_min_1pm)
            
            # Debug turning decision factors - use absolute angle for threshold comparison
            if self._logger.isEnabledFor(logging.DEBUG):
                self._logger.debug(
                    "turning_decision: arc_too_small=%s (%.3f < %.3f) speed_too_small=%s (%.3f < %.3f) apply_thresh=%s (|angle|=%.3f° >= %.3f° OR |curvature|=%.6f >= %.6f)",
                    arc_too_small, turning.arc_len_m, self._cfg.turning_min_arc_len_m,
                    speed_too_small, v_raw, self._cfg.min_speed_mps,
                    apply_turn_thresh, abs(turn_angle_signed_deg), self._cfg.theta_min_deg, abs(curvature_1pm), self._cfg.curvature_min_1pm
                )
            arc_too_small = (turning.arc_len_m < self._cfg.turning_min_arc_len_m)
            speed_too_small = (v_raw < self._cfg.min_speed_mps)
            
            # CRITICAL FIX: When zero-velocity is detected, force zero turning values
            # This ensures physical consistency - stopped vehicles have no turning motion
            if zero_velocity_detected:
                turn_angle_signed_deg = 0.0
                curvature_1pm = 0.0
                # Reset turning state to prevent persistence
                h.prev_turn_angle_deg = None
                h.prev_turn_t_s = None
            
            # Store the original heading for potential zeroing
            original_heading_deg = float(turning.heading_deg)
            heading_deg_final = 0.0 if zero_velocity_detected else original_heading_deg
            
            # Reset turn state after extended stops (>2 seconds)
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
            
            # Debug final turning decision
            if self._logger.isEnabledFor(logging.DEBUG):
                self._logger.debug(
                    "turning_final: apply_turn=%s (not arc_too_small=%s AND not speed_too_small=%s AND (apply_thresh=%s OR persist_ok=%s))",
                    apply_turn, (not arc_too_small), (not speed_too_small), apply_turn_thresh, persist_ok
                )
            
            curvature_for_limit = curvature_1pm if apply_turn else 0.0
            angle_for_limit = turn_angle_signed_deg if apply_turn else 0.0
            angle_display_deg = abs(float(turn_angle_signed_deg))  # Use absolute value for display
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

            # CRITICAL FIX: When zero-velocity is detected, ensure state is reset to 0
            # This prevents speed persistence from previous moving states
            if zero_velocity_detected:
                h.v_prev_mps = 0.0
                if h.smoother is not None:
                    h.smoother._value = 0.0
            else:
                h.v_prev_mps = float(v_smooth)
            
            h.t_prev_s = float(t1)

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
                    turn_angle_deg=float(angle_display_deg),  # Display absolute value
                    curvature_1pm=float(curvature_1pm),
                    metadata={
                        "dt_s": float(dt),
                        "disp_m": float(disp),
                        "turn_applied": 1.0 if apply_turn else 0.0,
                        "arc_len_m": float(turning.arc_len_m),
                        "turn_angle_signed_deg": float(turn_angle_signed_deg),  # Store signed angle for direction
                    },
                )
            )
            h.prev_turn_angle_deg = float(turn_angle_signed_deg)  # Store signed angle for state tracking
            h.prev_turn_t_s = float(t1)
        return out

    def prune_missing(self, alive_keys: List[Tuple[str, int]]) -> None:
        alive = set(alive_keys)
        to_del = [k for k in self._hist.keys() if k not in alive]
        for k in to_del:
            del self._hist[k]
