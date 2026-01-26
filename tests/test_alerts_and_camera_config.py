from speedestimation.io.camera_handler import CameraHandlerConfig
from speedestimation.output.alerts import SpeedAlertConfig, SpeedAlertEngine
from speedestimation.utils.types import SpeedSample


def _sample(track_id: int, t: float, v_raw: float, v_lim: float, v_smooth: float) -> SpeedSample:
    return SpeedSample(
        camera_id="cam_01",
        track_id=track_id,
        timestamp_s=t,
        frame_index=int(t * 10),
        world_xy_m=(0.0, 0.0),
        speed_mps_raw=v_raw,
        speed_mps_limited=v_lim,
        speed_mps_smoothed=v_smooth,
        heading_deg=0.0,
        turn_angle_deg=0.0,
        curvature_1pm=0.0,
        metadata={},
    )


def test_speed_alert_min_consecutive_and_cooldown() -> None:
    cfg = SpeedAlertConfig.from_dict(
        {
            "enabled": True,
            "speed_limit_kmh": 10.0,
            "threshold_kmh": 0.0,
            "min_consecutive_samples": 2,
            "cooldown_s": 1.0,
            "use_speed": "smoothed",
        }
    )
    eng = SpeedAlertEngine(cfg)

    s0 = _sample(track_id=1, t=0.0, v_raw=3.0, v_lim=3.0, v_smooth=3.0)
    s1 = _sample(track_id=1, t=0.1, v_raw=3.0, v_lim=3.0, v_smooth=3.0)
    assert eng.update([s0]) == []
    ev = eng.update([s1])
    assert len(ev) == 1
    assert ev[0].track_id == 1

    s2 = _sample(track_id=1, t=0.2, v_raw=3.0, v_lim=3.0, v_smooth=3.0)
    assert eng.update([s2]) == []

    s3 = _sample(track_id=1, t=1.3, v_raw=3.0, v_lim=3.0, v_smooth=3.0)
    ev2 = eng.update([s3])
    assert len(ev2) == 1


def test_speed_alert_uses_selected_speed_field() -> None:
    cfg = SpeedAlertConfig.from_dict(
        {
            "enabled": True,
            "speed_limit_kmh": 10.0,
            "threshold_kmh": 0.0,
            "min_consecutive_samples": 1,
            "cooldown_s": 0.0,
            "use_speed": "raw",
        }
    )
    eng = SpeedAlertEngine(cfg)
    s = _sample(track_id=1, t=0.0, v_raw=3.0, v_lim=1.0, v_smooth=1.0)
    ev = eng.update([s])
    assert len(ev) == 1
    assert abs(ev[0].speed_mps - 3.0) < 1e-9


def test_camera_handler_config_merges_overrides() -> None:
    global_cfg = {"reconnect": {"enabled": True, "max_retries": 0, "backoff_s": 1.0}, "rtsp": {"use_gstreamer": False, "latency_ms": 120}}
    per_cam = {"reconnect": {"backoff_s": 0.25}, "rtsp": {"use_gstreamer": True, "latency_ms": 50}}
    cfg = CameraHandlerConfig.from_sources(
        camera_source_type="rtsp",
        camera_uri="rtsp://example/stream",
        fps_hint=30.0,
        resize_enabled=False,
        resize_width=0,
        resize_height=0,
        global_cfg=global_cfg,
        per_camera_params=per_cam,
    )
    assert cfg.reconnect.enabled is True
    assert abs(cfg.reconnect.backoff_s - 0.25) < 1e-9
    assert cfg.rtsp.use_gstreamer is True
    assert cfg.rtsp.latency_ms == 50

