"""
Isolated v5 speed estimation components.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple
from collections import deque
import logging
import numpy as np

from turning_improved import compute_turning_metrics_improved, TurningConfig, _detect_zero_velocity_segments, _detect_zero_velocity_segments_authoritative

BBoxXYXY = Tuple[float, float, float, float]

@dataclass(frozen=True)
class Detection:
    bbox_xyxy: BBoxXYXY
    score: float
    class_id: int
    class_name: str

    @property
    def centroid_xy(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

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

    def centroid_xy(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

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
    curvature_1pm: float
    metadata: Dict[str, float]

@dataclass(frozen=True)
class DetectorInput:
    frame_index: int
    image_bgr: np.ndarray

@dataclass(frozen=True)
class TrackerInput:
    camera_id: str
    frame_index: int
    timestamp_s: float
    detections: List[Detection]

@dataclass(frozen=True)
class TrackerOutput:
    camera_id: str
    frame_index: int
    timestamp_s: float
    tracks: List[TrackState]

class EmaSmoother:
    """Exponential moving average smoother for speed values."""
    
    def __init__(self, alpha: float, max_gap_s: float):
        self.alpha = alpha
        self.max_gap_s = max_gap_s
        self._value: Optional[float] = None
        self._last_t: Optional[float] = None

    def update(self, value: float, timestamp_s: float) -> float:
        if self._value is None or self._last_t is None or (timestamp_s - self._last_t) > self.max_gap_s:
            self._value = value
        else:
            self._value = self.alpha * value + (1.0 - self.alpha) * self._value
        self._last_t = timestamp_s
        return self._value

@dataclass
class _TrackHistory:
    world_xy_m: Deque[Tuple[float, float]]
    timestamps_s: Deque[float]
    frame_indices: Deque[int]
    dots: Deque[Tuple[float, float, float]]
    smoother: Optional[EmaSmoother]
    v_prev_mps: float = 0.0
    t_prev_s: Optional[float] = None
    prev_turn_angle_deg: Optional[float] = None
    prev_turn_t_s: Optional[float] = None

class SpeedEstimatorConfig:
    """Configuration for speed estimation."""
    
    def __init__(self,
                 min_dt_s: float = 0.05,
                 max_dt_s: float = 1.0,
                 min_displacement_m: float = 0.05,
                 min_speed_mps: float = 1.0,
                 turning_min_arc_len_m: float = 0.05,
                 turning_persist_s: float = 0.25,
                 max_turn_rate_deg_per_s: float = 0.0,
                 theta_min_deg: float = 8.0,
                 curvature_min_1pm: float = 0.015,
                 smoothing_enabled: bool = True,
                 smoothing_alpha: float = 0.3,
                 smoothing_max_gap_s: float = 1.0,
                 dot_min_distance_m: float = 0.45,
                 dot_min_dt_s: float = 0.3,
                 dot_buffer_size: int = 5,
                 mode: str = "advanced",
                 simple_window_s: float = 1.0,
                 simple_axis: str = "y",
                 simple_method: str = "mean"):
        
        self.min_dt_s = min_dt_s
        self.max_dt_s = max_dt_s
        self.min_displacement_m = min_displacement_m
        self.min_speed_mps = min_speed_mps
        self.turning_min_arc_len_m = turning_min_arc_len_m
        self.turning_persist_s = turning_persist_s
        self.max_turn_rate_deg_per_s = max_turn_rate_deg_per_s
        self.theta_min_deg = theta_min_deg
        self.curvature_min_1pm = curvature_min_1pm
        self.smoothing_enabled = smoothing_enabled
        self.smoothing_alpha = smoothing_alpha
        self.smoothing_max_gap_s = smoothing_max_gap_s
        self.dot_min_distance_m = dot_min_distance_m
        self.dot_min_dt_s = dot_min_dt_s
        self.dot_buffer_size = dot_buffer_size
        self.mode = mode
        self.simple_window_s = simple_window_s
        self.simple_axis = simple_axis
        self.simple_method = simple_method
        
        # Create turning config
        self.turning_config = TurningConfig(
            min_speed_mps=min_speed_mps,
            min_arc_length_m=turning_min_arc_len_m,
            max_angular_rate_deg_s=max_turn_rate_deg_per_s,
            min_curvature_1pm=curvature_min_1pm
        )

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
        dots = d.get("dots", {})

        return SpeedEstimatorConfig(
            min_dt_s=float(raw.get("min_dt_s", 0.05)),
            max_dt_s=float(raw.get("max_dt_s", 1.0)),
            min_displacement_m=float(raw.get("min_displacement_m", 0.05)),
            min_speed_mps=float(raw.get("min_speed_mps", 1.0)),
            turning_min_arc_len_m=float(turning.get("min_arc_len_m", 0.05)),
            turning_persist_s=float(turning.get("persist_s", 0.25)),
            max_turn_rate_deg_per_s=float(turning.get("max_turn_rate_deg_per_s", 0.0)),
            theta_min_deg=float(turning.get("theta_min_deg", 8.0)),
            curvature_min_1pm=float(turning.get("curvature_min_1pm", 0.015)),
            smoothing_enabled=bool(d.get("smoothing", {}).get("enabled", True)),
            smoothing_alpha=float(d.get("smoothing", {}).get("alpha", 0.3)),
            smoothing_max_gap_s=float(d.get("smoothing", {}).get("max_gap_s", 1.0)),
            dot_min_distance_m=float(dots.get("min_distance_m", 0.45)),
            dot_min_dt_s=float(dots.get("min_dt_s", 0.3)),
            dot_buffer_size=int(dots.get("buffer_size", 5)),
            mode=mode,
            simple_window_s=simple_window_s,
            simple_axis=simple_axis,
            simple_method=simple_method
        )

def speed_mps(p1: Tuple[float, float], t1: float, p2: Tuple[float, float], t2: float) -> Optional[float]:
    """Compute speed in m/s between two points."""
    if t2 <= t1:
        return None
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dist = (dx * dx + dy * dy) ** 0.5
    dt = t2 - t1
    if dt <= 0:
        return None
    return dist / dt

class SpeedEstimator:
    """Speed and turning estimation engine."""
    
    def __init__(self, cfg: SpeedEstimatorConfig, max_history: int = 64) -> None:
        self._cfg = cfg
        self._max_history = int(max_history)
        self._hist: Dict[Tuple[str, int], _TrackHistory] = {}
        self._logger = logging.getLogger(__name__)

    def update(self, tracks: List[Track]) -> List[SpeedSample]:
        """Update estimator with new tracks and return speed samples."""
        out: List[SpeedSample] = []
        
        for tr in tracks:
            if tr.world_xy_m is None:
                continue
                
            key = (tr.camera_id, tr.state.track_id)
            h = self._hist.get(key)
            
            if h is None:
                smoother = None
                if self._cfg.smoothing_enabled:
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

            if self._cfg.mode == "simple":
                if not dot_added or len(h.dots) < 2:
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
                
                # Zero-velocity detection for simple mode
                positions_simple = np.array([(p[0], p[1]) for p in h.dots], dtype=float)
                times_simple = np.array([p[2] for p in h.dots], dtype=float)
                zero_velocity_mask_simple = _detect_zero_velocity_segments_authoritative(
                    positions_simple, times_simple, 
                    movement_threshold_m=max(self._cfg.min_displacement_m, self._cfg.dot_min_distance_m),
                    min_time_window_s=0.5,
                    min_arc_length_m=self._cfg.turning_min_arc_len_m
                )
                
                zero_velocity_detected = False
                if zero_velocity_mask_simple[-1]:
                    v_raw = 0.0
                    zero_velocity_detected = True
                elif self._cfg.simple_method == "mean":
                    v_raw = float(sum(speeds) / len(speeds))
                else:
                    v_raw = float(disp_sum / dt_sum)
                
                # Hard arc-length check for simple mode
                total_arc_length = 0.0
                for i in range(1, len(h.dots)):
                    p_prev = h.dots[i-1]
                    p_curr = h.dots[i]
                    segment_length = ((p_curr[0] - p_prev[0])**2 + (p_curr[1] - p_prev[1])**2)**0.5
                    total_arc_length += segment_length
                
                if total_arc_length < self._cfg.turning_config.min_arc_length_m:
                    v_raw = 0.0
                    zero_velocity_detected = True
                
                if disp_sum < max(self._cfg.min_displacement_m, self._cfg.dot_min_distance_m) or v_raw < self._cfg.min_speed_mps:
                    v_raw = 0.0
                    zero_velocity_detected = True
                
                if h.smoother is None:
                    v_smooth = float(v_raw)
                else:
                    v_smooth = h.smoother.update(float(v_raw), float(t1))
                
                if zero_velocity_detected or v_smooth < self._cfg.min_speed_mps:
                    v_smooth = 0.0
                
                # Reset state when zero-velocity detected
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

            if not dot_added or len(h.dots) < 2:
                continue
                
            # Speed and turning use the same dot buffer for consistency
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
                min_time_window_s=0.5,
                min_arc_length_m=self._cfg.turning_min_arc_len_m
            )
            
            zero_velocity_detected = False
            if zero_velocity_mask[-1]:
                v_raw = 0.0
                zero_velocity_detected = True
            elif disp < max(self._cfg.min_displacement_m, self._cfg.dot_min_distance_m):
                v_raw = 0.0
                zero_velocity_detected = True

            # Compute turning metrics
            turning = compute_turning_metrics_improved(h.dots, self._cfg.turning_config)
            turn_angle_signed_deg = float(turning.turn_angle_deg)
            curvature_1pm = float(turning.curvature_1pm)
            
            # Hard zero-speed override based on arc length
            if turning.arc_len_m < self._cfg.turning_config.min_arc_length_m:
                v_raw = 0.0
                zero_velocity_detected = True
            
            # Store original heading for potential zeroing
            original_heading_deg = float(turning.heading_deg)
            heading_deg_final = 0.0 if zero_velocity_detected else original_heading_deg
            
            # Reset turn state after extended stops (>2 seconds)
            if h.prev_turn_t_s is not None:
                dt_since_last_turn = float(t1 - h.prev_turn_t_s)
                if dt_since_last_turn > 2.0:
                    h.prev_turn_angle_deg = None
                    h.prev_turn_t_s = None
            
            # Apply turn rate limiting if configured
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
            
            # Persist turn angle if below threshold
            persist_ok = False
            if h.prev_turn_angle_deg is not None and h.prev_turn_t_s is not None:
                dt_since_last_turn = float(t1 - h.prev_turn_t_s)
                if dt_since_last_turn <= self._cfg.turning_persist_s:
                    if abs(turn_angle_signed_deg) < self._cfg.theta_min_deg:
                        turn_angle_signed_deg = h.prev_turn_angle_deg
                        curvature_1pm = 0.0 if zero_velocity_detected else curvature_1pm
                        persist_ok = True
            
            if not persist_ok:
                h.prev_turn_angle_deg = float(turn_angle_signed_deg)
                h.prev_turn_t_s = float(t1)
            
            # Apply smoothing
            if h.smoother is None:
                v_smooth = float(v_raw)
            else:
                v_smooth = h.smoother.update(float(v_raw), float(t1))
            
            if zero_velocity_detected or v_smooth < self._cfg.min_speed_mps:
                v_smooth = 0.0
            
            # Reset state when zero-velocity detected
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
                    heading_deg=float(heading_deg_final),
                    turn_angle_deg=float(turn_angle_signed_deg),
                    curvature_1pm=float(curvature_1pm),
                    metadata={
                        "dt_s": float(dt),
                        "disp_m": float(disp),
                        "arc_len_m": float(turning.arc_len_m),
                    },
                )
            )
        
        return out