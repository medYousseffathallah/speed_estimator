import math

from speedestimation.speed_estimation.estimator import SpeedEstimator, SpeedEstimatorConfig
from speedestimation.speed_estimation.limits import TurnSpeedLimitConfig, accel_limited_speed_mps, turn_limited_speed_mps
from speedestimation.speed_estimation.math import angle_between_deg, heading_deg_from_delta, speed_mps
from speedestimation.turning_model.turning import compute_turning_metrics
from speedestimation.utils.types import Track, TrackState


def test_speed_mps_basic() -> None:
    v = speed_mps((0.0, 0.0), 0.0, (3.0, 4.0), 2.0)
    assert v is not None
    assert abs(v - 2.5) < 1e-9


def test_heading_deg_quadrants() -> None:
    assert heading_deg_from_delta(1.0, 0.0) == 0.0
    assert heading_deg_from_delta(0.0, 1.0) == 90.0
    assert heading_deg_from_delta(-1.0, 0.0) == 180.0
    assert heading_deg_from_delta(0.0, -1.0) == 270.0


def test_angle_between_deg() -> None:
    assert abs(angle_between_deg(1.0, 0.0, 0.0, 1.0) - 90.0) < 1e-9


def test_turning_metrics_straight() -> None:
    pts = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
    ts = [0.0, 1.0, 2.0]
    m = compute_turning_metrics(pts, ts, window=3)
    assert abs(m.turn_angle_deg - 0.0) < 1e-9
    assert abs(m.curvature_1pm - 0.0) < 1e-9


def test_turn_speed_limit_reduces_speed_on_curvature() -> None:
    cfg = TurnSpeedLimitConfig(
        enabled=True,
        a_lat_max_mps2=2.0,
        v_max_mps=100.0,
        v_min_mps=0.0,
        angle_max_deg=60.0,
        mode="curvature",
        alpha=1.0,
    )
    v = turn_limited_speed_mps(v_raw_mps=20.0, curvature_1pm=0.5, turn_angle_deg=0.0, cfg=cfg)
    assert v < 20.0


def test_accel_limit_clamps_delta_v() -> None:
    v = accel_limited_speed_mps(v_prev_mps=10.0, t_prev_s=0.0, v_curr_mps=30.0, t_curr_s=1.0, enabled=True, a_max_mps2=5.0)
    assert abs(v - 15.0) < 1e-9


def test_turn_angle_rate_limiter_caps_jump() -> None:
    cfg = SpeedEstimatorConfig.from_dict(
        {
            "raw_speed": {"min_dt_s": 0.01, "max_dt_s": 1.0, "min_displacement_m": 0.0, "min_speed_mps": 0.0},
            "turning": {"window": 3, "window_s": 0.0, "max_turn_rate_deg_per_s": 30.0, "theta_min_deg": 0.0, "curvature_min_1pm": 0.0},
            "turn_speed_limit": {"enabled": False},
            "accel_limit": {"enabled": False},
            "smoothing": {"method": "ema", "ema_alpha": 1.0, "max_gap_s": 1.0},
            "ablations": {"disable_turn_limit": True, "disable_accel_limit": True, "disable_smoothing": True},
        }
    )
    est = SpeedEstimator(cfg)
    ts = TrackState(track_id=1, camera_id="cam", class_id=0, class_name="car", bbox_xyxy=(0.0, 0.0, 1.0, 1.0), score=1.0)
    tracks = [
        Track(camera_id="cam", frame_index=0, timestamp_s=0.0, state=ts, world_xy_m=(0.0, 0.0)),
        Track(camera_id="cam", frame_index=1, timestamp_s=0.1, state=ts, world_xy_m=(1.0, 0.0)),
        Track(camera_id="cam", frame_index=2, timestamp_s=0.2, state=ts, world_xy_m=(1.0, 1.0)),
    ]
    out = []
    for tr in tracks:
        out = est.update([tr])
    assert out
    assert out[-1].turn_angle_deg <= 3.1

