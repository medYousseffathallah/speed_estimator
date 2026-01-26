from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

from collections import deque

from speedestimation.speed_estimation.limits import TurnSpeedLimitConfig, accel_limited_speed_mps, turn_limited_speed_mps
from speedestimation.speed_estimation.math import speed_mps
from speedestimation.speed_estimation.smoothing import EmaSmoother
from speedestimation.turning_model.turning import compute_turning_metrics
from speedestimation.utils.types import SpeedSample, Track


@dataclass(frozen=True)
class SpeedEstimatorConfig:
    min_dt_s: float
    max_dt_s: float
    min_displacement_m: float
    min_speed_mps: float
    turning_window: int
    turning_window_s: float
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

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "SpeedEstimatorConfig":
        raw = d.get("raw_speed", {})
        turning = d.get("turning", {})
        tsl = d.get("turn_speed_limit", {})
        accel = d.get("accel_limit", {})
        smoothing = d.get("smoothing", {})
        ab = d.get("ablations", {})
        units = d.get("units", {})
        output_units = str(units.get("output", "kmh")).lower()
        if output_units not in {"kmh", "mph", "mps"}:
            raise ValueError("units.output must be one of: kmh, mph, mps")

        v_max_kmh = float(tsl.get("v_max_kmh", 180.0))
        v_max_mps = v_max_kmh / 3.6
        v_min_kmh = float(tsl.get("v_min_kmh", v_max_kmh))
        v_min_mps = v_min_kmh / 3.6
        
        # Parse min speed threshold (default 1.0 m/s ~ 3.6 km/h)
        min_speed_mps = float(raw.get("min_speed_mps", 1.0))
        
        return SpeedEstimatorConfig(
            min_dt_s=float(raw.get("min_dt_s", 0.05)),
            max_dt_s=float(raw.get("max_dt_s", 1.0)),
            min_displacement_m=float(raw.get("min_displacement_m", 0.05)),
            min_speed_mps=min_speed_mps,
            turning_window=int(turning.get("window", 6)),
            turning_window_s=float(turning.get("window_s", 0.0)),
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
        )


@dataclass
class _TrackHistory:
    world_xy_m: Deque[Tuple[float, float]]
    timestamps_s: Deque[float]
    frame_indices: Deque[int]
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
                    smoother=smoother,
                )
                self._hist[key] = h

            h.world_xy_m.append(tr.world_xy_m)
            h.timestamps_s.append(tr.timestamp_s)
            h.frame_indices.append(tr.frame_index)
            if len(h.world_xy_m) < 2:
                continue

            p0 = h.world_xy_m[-2]
            p1 = h.world_xy_m[-1]
            t0 = h.timestamps_s[-2]
            t1 = h.timestamps_s[-1]
            dt = float(t1 - t0)
            if dt < self._cfg.min_dt_s or dt > self._cfg.max_dt_s:
                continue
            v_raw = speed_mps(p0, t0, p1, t1)
            if v_raw is None:
                continue
            disp = ((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2) ** 0.5
            if disp < self._cfg.min_displacement_m:
                v_raw = 0.0

            window_s = self._cfg.turning_window_s if self._cfg.turning_window_s > 0.0 else None
            turning = compute_turning_metrics(h.world_xy_m, h.timestamps_s, window=self._cfg.turning_window, window_s=window_s)
            turn_angle_deg = float(turning.turn_angle_deg)
            curvature_1pm = float(turning.curvature_1pm)
            max_rate = float(self._cfg.max_turn_rate_deg_per_s)
            if max_rate > 0.0 and h.prev_turn_angle_deg is not None and h.prev_turn_t_s is not None:
                dt_turn = float(t1 - h.prev_turn_t_s)
                if dt_turn > 0.0:
                    max_delta = max_rate * dt_turn
                    lo = max(0.0, h.prev_turn_angle_deg - max_delta)
                    hi = h.prev_turn_angle_deg + max_delta
                    capped = min(max(turn_angle_deg, lo), hi)
                    if turn_angle_deg > 1e-6:
                        curvature_1pm = curvature_1pm * (capped / turn_angle_deg)
                    else:
                        curvature_1pm = 0.0
                    turn_angle_deg = capped

            apply_turn = (turn_angle_deg >= self._cfg.theta_min_deg) or (abs(curvature_1pm) >= self._cfg.curvature_min_1pm)
            curvature_for_limit = curvature_1pm if apply_turn else 0.0
            angle_for_limit = turn_angle_deg if apply_turn else 0.0

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

            if v_smooth < self._cfg.min_speed_mps:
                v_smooth = 0.0

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
                    heading_deg=float(turning.heading_deg),
                    turn_angle_deg=float(turn_angle_deg),
                    curvature_1pm=float(curvature_1pm),
                    metadata={
                        "dt_s": float(dt),
                        "disp_m": float(disp),
                        "turn_applied": 1.0 if apply_turn else 0.0,
                    },
                )
            )
            h.prev_turn_angle_deg = float(turn_angle_deg)
            h.prev_turn_t_s = float(t1)
        return out

    def prune_missing(self, alive_keys: List[Tuple[str, int]]) -> None:
        alive = set(alive_keys)
        to_del = [k for k in self._hist.keys() if k not in alive]
        for k in to_del:
            del self._hist[k]

