from __future__ import annotations

from dataclasses import dataclass
import copy
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union


JsonDict = Dict[str, Any]
CameraSource = Union[str, int]


@dataclass(frozen=True)
class V5PipelineConfig:
    camera_id: str
    calibration: Dict[str, Any]
    detection: Dict[str, Any]
    tracking: Dict[str, Any]
    speed: Dict[str, Any]
    runtime: Dict[str, Any]
    base_dir: Optional[str] = None


def _env_float(key: str, default: float) -> float:
    v = str(os.environ.get(key, "") or "").strip()
    if v == "":
        return float(default)
    try:
        return float(v)
    except Exception:
        return float(default)


def _env_int(key: str, default: int) -> int:
    v = str(os.environ.get(key, "") or "").strip()
    if v == "":
        return int(default)
    try:
        return int(float(v))
    except Exception:
        return int(default)


def _env_str(key: str, default: str = "") -> str:
    v = str(os.environ.get(key, "") or "").strip()
    return v if v != "" else str(default)


_ENV_BRACE_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_env_str(s: str) -> str:
    s2 = os.path.expandvars(str(s))
    def repl(m: re.Match[str]) -> str:
        return str(os.environ.get(m.group(1), "") or "")
    s2 = _ENV_BRACE_RE.sub(repl, s2)
    return os.path.expandvars(s2)


def _expand_env(obj: Any) -> Any:
    if isinstance(obj, str):
        return _expand_env_str(obj)
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_expand_env(v) for v in obj)
    return obj


def _load_json(path: str) -> JsonDict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a dict at root of JSON config: {path}")
    return dict(_expand_env(data))


def _resolve_path(p: str, base_dir: Optional[str]) -> str:
    s = str(p or "").strip()
    if s == "":
        return ""
    pp = Path(s)
    if pp.is_absolute() or base_dir in (None, ""):
        return str(pp)
    return str((Path(str(base_dir)) / pp).resolve())


def default_config() -> Tuple[V5PipelineConfig, str]:
    rtsp_url = _env_str("V5_RTSP_URL", "")
    camera_id = _env_str("V5_CAMERA_ID", "cam1")

    model_path = _env_str("V5_MODEL_PATH", "yolov8n.pt")
    conf = _env_float("V5_CONF", 0.25)
    iou = _env_float("V5_IOU", 0.7)

    homography_npy = _env_str("V5_HOMOGRAPHY_NPY", "")
    meters_per_pixel = _env_float("V5_METERS_PER_PIXEL", 1.0)

    dot_min_distance_m = _env_float("V5_DOT_MIN_DISTANCE_M", 0.20)
    dot_min_dt_s = _env_float("V5_DOT_MIN_DT_S", 0.2)
    dot_buffer_size = _env_int("V5_DOT_BUFFER_SIZE", 5)

    tracking_params = {
        "iou_threshold": _env_float("V5_TRACK_IOU", 0.5),
        "max_missing_frames": _env_int("V5_MAX_MISSING", 30),
        "max_age_frames": _env_int("V5_MAX_MISSING", 30),
        "min_hits": 4,
        "high_conf_threshold": 0.7,
        "low_conf_threshold": 0.3,
    }

    calib: Dict[str, Any]
    if homography_npy:
        calib = {"homography_npy": homography_npy, "meters_per_pixel": meters_per_pixel}
    else:
        calib = {"meters_per_pixel": meters_per_pixel}

    cfg = V5PipelineConfig(
        camera_id=camera_id,
        base_dir=None,
        calibration=calib,
        detection={
            "enabled": True,
            "model_path": model_path,
            "conf": conf,
            "iou": iou,
            "classes": [1, 2, 3, 5, 7],
            "class_whitelist": ["car", "truck", "bus", "motorcycle", "bicycle"],
        },
        tracking={"backend": "bytetrack", "params": tracking_params},
        speed={
            "mode": "advanced",
            "units": {"output": "kmh"},
            "raw_speed": {
                "min_dt_s": 0.04,
                "max_dt_s": 1.0,
                "min_displacement_m": 0.15,
                "min_speed_mps": 0.03,
                "stop_time_s": 0.5,
                "min_arc_length_m": 0.1,
            },
            "dots": {"min_distance_m": dot_min_distance_m, "min_dt_s": dot_min_dt_s, "buffer_size": dot_buffer_size},
            "turning": {
                "min_speed_mps": 0.5,
                "min_arc_length_m": 0.1,
                "max_angular_rate_deg_s": 45.0,
                "max_angular_accel_deg_s2": 180.0,
                "min_curvature_1pm": 0.005,
                "smoothing_window_min": 3,
                "smoothing_window_max": 7,
                "curvature_filter_sigma": 1.0,
                "enable_adaptive_thresholds": True,
                "low_speed_threshold_mps": 2.0,
                "high_speed_threshold_mps": 15.0,
                "max_turn_rate_deg_per_s": 45.0,
                "theta_min_deg": 8.0,
                "curvature_min_1pm": 0.015,
                "min_arc_len_m": 0.05,
                "persist_s": 0.25,
            },
            "turn_speed_limit": {
                "enabled": True,
                "mode": "curvature",
                "a_lat_max_mps2": 2.5,
                "v_max_kmh": 180.0,
                "v_min_kmh": 30.0,
                "angle_max_deg": 60.0,
                "alpha": 0.5,
            },
            "accel_limit": {"enabled": True, "a_max_mps2": 60.0},
            "smoothing": {"method": "ema", "ema_alpha": 0.25, "max_gap_s": 1.0},
            "ablations": {"disable_turn_limit": False, "disable_accel_limit": False, "disable_smoothing": False},
        },
        runtime={},
    )
    return cfg, rtsp_url


def load_pipeline_config_from_file(path: str) -> Tuple[V5PipelineConfig, CameraSource, JsonDict]:
    p = str(path)
    d = _load_json(p)
    base_dir = d.get("base_dir")
    base_dir = None if base_dir in (None, "") else str(base_dir)
    if base_dir is None:
        base_dir = os.getcwd()

    if "camera" in d:
        unified = apply_env_overrides(copy.deepcopy(d))
        validate_unified_config(unified, base_dir=base_dir)
        cfg, src = unified_to_v5_config(unified, base_dir=base_dir)
        return cfg, src, unified

    legacy = apply_env_overrides(copy.deepcopy(d))
    cfg, src = legacy_to_v5_config(legacy, base_dir=base_dir)
    validate_v5_config(cfg, src)
    return cfg, src, legacy


def apply_env_overrides(cfg: JsonDict) -> JsonDict:
    rtsp_url = _env_str("V5_RTSP_URL", "")
    camera_id = _env_str("V5_CAMERA_ID", "")
    model_path = _env_str("V5_MODEL_PATH", "")
    homography_npy = _env_str("V5_HOMOGRAPHY_NPY", "")

    if "camera" in cfg:
        cam = dict(cfg.get("camera", {}) or {})
        if camera_id:
            cam["id"] = camera_id
        if rtsp_url:
            cam_type = str(cam.get("type", "rtsp") or "rtsp").strip().lower()
            if cam_type == "rtsp" or "type" not in cam:
                cam["type"] = "rtsp"
                cam["source"] = rtsp_url
        cfg["camera"] = cam

        det = dict(cfg.get("detection", {}) or {})
        if model_path:
            det["model_path"] = model_path
        if str(os.environ.get("V5_CONF", "") or "").strip() != "":
            det["conf"] = _env_float("V5_CONF", float(det.get("conf", 0.25)))
        if str(os.environ.get("V5_IOU", "") or "").strip() != "":
            det["iou"] = _env_float("V5_IOU", float(det.get("iou", 0.7)))
        cfg["detection"] = det

        calib = dict(cfg.get("calibration", {}) or {})
        if homography_npy:
            calib["method"] = "homography"
            calib["homography_npy"] = homography_npy
        if str(os.environ.get("V5_METERS_PER_PIXEL", "") or "").strip() != "":
            calib["meters_per_pixel"] = _env_float("V5_METERS_PER_PIXEL", float(calib.get("meters_per_pixel", 1.0)))
        cfg["calibration"] = calib

        speed = dict(cfg.get("speed_estimation", {}) or {})
        dots = dict(speed.get("trajectory_dots", {}) or {})
        if str(os.environ.get("V5_DOT_MIN_DISTANCE_M", "") or "").strip() != "":
            dots["min_distance_m"] = _env_float("V5_DOT_MIN_DISTANCE_M", float(dots.get("min_distance_m", 0.8)))
        if str(os.environ.get("V5_DOT_MIN_DT_S", "") or "").strip() != "":
            dots["min_dt_s"] = _env_float("V5_DOT_MIN_DT_S", float(dots.get("min_dt_s", 0.5)))
        if str(os.environ.get("V5_DOT_BUFFER_SIZE", "") or "").strip() != "":
            dots["buffer_size"] = _env_int("V5_DOT_BUFFER_SIZE", int(dots.get("buffer_size", 6)))
        if dots:
            speed["trajectory_dots"] = dots
            cfg["speed_estimation"] = speed

        tr = dict(cfg.get("tracking", {}) or {})
        params = dict(tr.get("params", {}) or {})
        if str(os.environ.get("V5_TRACK_IOU", "") or "").strip() != "":
            params["iou_threshold"] = _env_float("V5_TRACK_IOU", float(params.get("iou_threshold", 0.3)))
        if str(os.environ.get("V5_MAX_MISSING", "") or "").strip() != "":
            mm = _env_int("V5_MAX_MISSING", int(params.get("max_missing_frames", 30)))
            params["max_missing_frames"] = mm
            params.setdefault("max_age_frames", mm)
        if params:
            tr["params"] = params
            cfg["tracking"] = tr

        return cfg

    if camera_id:
        cfg["camera_id"] = camera_id
    if rtsp_url:
        cfg["rtsp_url"] = rtsp_url
    if model_path:
        det = dict(cfg.get("detection", {}) or {})
        det["model_path"] = model_path
        cfg["detection"] = det
    if homography_npy:
        calib = dict(cfg.get("calibration", {}) or {})
        calib["homography_npy"] = homography_npy
        cfg["calibration"] = calib
    if str(os.environ.get("V5_METERS_PER_PIXEL", "") or "").strip() != "":
        calib = dict(cfg.get("calibration", {}) or {})
        calib["meters_per_pixel"] = _env_float("V5_METERS_PER_PIXEL", float(calib.get("meters_per_pixel", 1.0)))
        cfg["calibration"] = calib
    if str(os.environ.get("V5_CONF", "") or "").strip() != "":
        det = dict(cfg.get("detection", {}) or {})
        det["conf"] = _env_float("V5_CONF", float(det.get("conf", 0.25)))
        cfg["detection"] = det
    if str(os.environ.get("V5_IOU", "") or "").strip() != "":
        det = dict(cfg.get("detection", {}) or {})
        det["iou"] = _env_float("V5_IOU", float(det.get("iou", 0.7)))
        cfg["detection"] = det
    if str(os.environ.get("V5_DOT_MIN_DISTANCE_M", "") or "").strip() != "":
        sp = dict(cfg.get("speed", {}) or {})
        dots = dict(sp.get("dots", {}) or {})
        dots["min_distance_m"] = _env_float("V5_DOT_MIN_DISTANCE_M", float(dots.get("min_distance_m", 0.8)))
        sp["dots"] = dots
        cfg["speed"] = sp
    if str(os.environ.get("V5_DOT_MIN_DT_S", "") or "").strip() != "":
        sp = dict(cfg.get("speed", {}) or {})
        dots = dict(sp.get("dots", {}) or {})
        dots["min_dt_s"] = _env_float("V5_DOT_MIN_DT_S", float(dots.get("min_dt_s", 0.5)))
        sp["dots"] = dots
        cfg["speed"] = sp
    if str(os.environ.get("V5_DOT_BUFFER_SIZE", "") or "").strip() != "":
        sp = dict(cfg.get("speed", {}) or {})
        dots = dict(sp.get("dots", {}) or {})
        dots["buffer_size"] = _env_int("V5_DOT_BUFFER_SIZE", int(dots.get("buffer_size", 6)))
        sp["dots"] = dots
        cfg["speed"] = sp
    if str(os.environ.get("V5_TRACK_IOU", "") or "").strip() != "":
        tr = dict(cfg.get("tracking", {}) or {})
        params = dict(tr.get("params", {}) or {})
        params["iou_threshold"] = _env_float("V5_TRACK_IOU", float(params.get("iou_threshold", 0.3)))
        tr["params"] = params
        cfg["tracking"] = tr
    if str(os.environ.get("V5_MAX_MISSING", "") or "").strip() != "":
        tr = dict(cfg.get("tracking", {}) or {})
        params = dict(tr.get("params", {}) or {})
        mm = _env_int("V5_MAX_MISSING", int(params.get("max_missing_frames", 30)))
        params["max_missing_frames"] = mm
        params.setdefault("max_age_frames", mm)
        tr["params"] = params
        cfg["tracking"] = tr
    return cfg


def legacy_to_v5_config(d: JsonDict, *, base_dir: Optional[str]) -> Tuple[V5PipelineConfig, CameraSource]:
    rtsp_url = str(d.get("rtsp_url", "") or "").strip()
    cfg = V5PipelineConfig(
        camera_id=str(d.get("camera_id", "cam1")),
        base_dir=str(d.get("base_dir")) if d.get("base_dir") not in (None, "") else base_dir,
        calibration=dict(d.get("calibration", {}) or {}),
        detection=dict(d.get("detection", {}) or {}),
        tracking=dict(d.get("tracking", {}) or {}),
        speed=dict(d.get("speed", {}) or {}),
        runtime=dict(d.get("runtime", {}) or {}),
    )
    return cfg, rtsp_url


def unified_to_v5_config(d: JsonDict, *, base_dir: Optional[str]) -> Tuple[V5PipelineConfig, CameraSource]:
    cam = dict(d.get("camera", {}) or {})
    cam_id = str(cam.get("id", "cam1") or "cam1").strip()
    cam_type = str(cam.get("type", "rtsp") or "rtsp").strip().lower()
    cam_source_raw = cam.get("source", "")
    if cam_type == "webcam":
        if isinstance(cam_source_raw, int):
            cam_source: CameraSource = int(cam_source_raw)
        else:
            cam_source = int(str(cam_source_raw).strip())
    else:
        cam_source = str(cam_source_raw or "").strip()

    det = dict(d.get("detection", {}) or {})
    tr = dict(d.get("tracking", {}) or {})
    calib_in = dict(d.get("calibration", {}) or {})
    speed_in = dict(d.get("speed_estimation", {}) or {})
    runtime_in = dict(d.get("runtime", {}) or {})

    calib_method = str(calib_in.get("method", "") or "").strip().lower()
    calib: Dict[str, Any] = {}
    if calib_method in {"homography", ""}:
        homography_npy = str(calib_in.get("homography_npy", "") or "").strip()
        if homography_npy:
            calib["homography_npy"] = homography_npy
            calib_method = "homography"
        else:
            calib_method = "meters_per_pixel"

    if calib_method == "homography":
        calib["homography_npy"] = str(calib_in.get("homography_npy", "") or "").strip()

    mpp = calib_in.get("meters_per_pixel")
    if mpp is None:
        mpp = 1.0
    calib["meters_per_pixel"] = float(mpp)

    mode = str(speed_in.get("mode", "advanced") or "advanced").strip().lower()
    simple_mode = dict(speed_in.get("simple_mode", {}) or {})
    adv_mode = dict(speed_in.get("advanced_mode", {}) or {})
    turning = dict(speed_in.get("turning", {}) or {})
    limits = dict(speed_in.get("limits", {}) or {})
    smoothing = dict(speed_in.get("smoothing", {}) or {})
    dots = dict(speed_in.get("trajectory_dots", {}) or {})

    turning_enabled = bool(turning.get("enabled", True))

    speed: Dict[str, Any] = {
        "mode": mode,
        "simple_speed": {
            "window_s": float(simple_mode.get("window_s", 1.0)),
            "axis": str(simple_mode.get("axis", "xy") or "xy"),
            "method": str(simple_mode.get("method", "mean") or "mean"),
        },
        "raw_speed": {
            "min_dt_s": float(adv_mode.get("min_dt_s", 0.04)),
            "max_dt_s": float(adv_mode.get("max_dt_s", 1.0)),
            "min_displacement_m": float(adv_mode.get("min_displacement_m", 0.15)),
            "stop_time_s": float(adv_mode.get("stop_time_s", 0.5)),
            "min_arc_length_m": float(adv_mode.get("min_arc_length_m", 0.1)),
            "min_speed_mps": float(adv_mode.get("min_speed_mps", 0.03)),
        },
        "turning": {
            "theta_min_deg": float(turning.get("theta_min_deg", 8.0)),
            "curvature_min_1pm": float(turning.get("curvature_min_1pm", 0.015)),
            "persist_s": float(turning.get("turning_persist_s", 0.25)),
            "max_turn_rate_deg_per_s": float(turning.get("max_turn_rate_deg_per_s", 45.0)),
            "min_arc_len_m": float(turning.get("min_arc_len_m", 0.05)),
            "min_speed_mps": float(turning.get("min_speed_mps", 0.5)),
            "min_arc_length_m": float(turning.get("min_arc_length_m", 0.1)),
            "max_angular_rate_deg_s": float(turning.get("max_angular_rate_deg_s", 45.0)),
            "max_angular_accel_deg_s2": float(turning.get("max_angular_accel_deg_s2", 180.0)),
            "min_curvature_1pm": float(turning.get("min_curvature_1pm", 0.005)),
            "smoothing_window_min": int(turning.get("smoothing_window_min", 3)),
            "smoothing_window_max": int(turning.get("smoothing_window_max", 7)),
        },
        "turn_speed_limit": dict(limits.get("turn_speed_limit", {}) or {}),
        "accel_limit": dict(limits.get("accel_limit", {}) or {}),
        "smoothing": {
            "method": str(smoothing.get("method", "ema") or "ema"),
            "ema_alpha": float(smoothing.get("ema_alpha", 0.25)),
            "max_gap_s": float(smoothing.get("max_gap_s", 1.0)),
        },
        "dots": {
            "min_distance_m": float(dots.get("min_distance_m", 0.20)),
            "min_dt_s": float(dots.get("min_dt_s", 0.2)),
            "buffer_size": int(dots.get("buffer_size", 5)),
        },
        "units": {"output": "kmh"},
    }

    if not turning_enabled:
        speed["turn_speed_limit"] = {"enabled": False}
        speed["turning"] = dict(speed.get("turning", {}) or {})
        speed["turning"]["theta_min_deg"] = 1e9
        speed["turning"]["curvature_min_1pm"] = 1e9
        speed["turning"]["max_turn_rate_deg_per_s"] = 0.0
        speed["turning"]["persist_s"] = 0.0

    runtime = {
        "fps_hint": float(runtime_in.get("fps_hint", 0.0) or 0.0),
        "resize": dict(runtime_in.get("resize", {}) or {}),
        "buffer_size": int(runtime_in.get("buffer_size", 1)),
        "read_timeout_s": float(runtime_in.get("read_timeout_s", 2.0)),
        "reconnect_delay_s": float(runtime_in.get("reconnect_delay_s", 2.0)),
    }

    cfg = V5PipelineConfig(
        camera_id=cam_id,
        base_dir=str(d.get("base_dir")) if d.get("base_dir") not in (None, "") else base_dir,
        calibration=calib,
        detection=det,
        tracking=tr,
        speed=speed,
        runtime=runtime,
    )
    return cfg, cam_source


def validate_v5_config(cfg: V5PipelineConfig, source: CameraSource) -> None:
    if str(cfg.camera_id or "").strip() == "":
        raise ValueError("camera_id is required")
    if isinstance(source, str) and source.strip() == "":
        raise ValueError("camera source is required")
    det = dict(cfg.detection or {})
    if bool(det.get("enabled", True)):
        model_path = str(det.get("model_path", "") or "").strip()
        if model_path == "":
            raise ValueError("detection.model_path is required when detection.enabled=true")
        base_dir = cfg.base_dir if cfg.base_dir not in (None, "") else os.getcwd()
        mp = _resolve_path(model_path, base_dir)
        p = Path(model_path)
        looks_like_path = p.is_absolute() or str(p.parent) not in {".", ""}
        if looks_like_path and not Path(mp).exists():
            raise ValueError(f"detection.model_path not found: {mp}")
        conf = float(det.get("conf", 0.25))
        if conf < 0.0 or conf > 1.0:
            raise ValueError("detection.conf must be in [0, 1]")
        iou = float(det.get("iou", 0.7))
        if iou < 0.0 or iou > 1.0:
            raise ValueError("detection.iou must be in [0, 1]")
    calib = dict(cfg.calibration or {})
    mpp = float(calib.get("meters_per_pixel", 1.0))
    if mpp <= 0.0:
        raise ValueError("calibration.meters_per_pixel must be > 0")
    homography_npy = str(calib.get("homography_npy", "") or "").strip()
    if homography_npy != "":
        base_dir = cfg.base_dir if cfg.base_dir not in (None, "") else os.getcwd()
        hp = _resolve_path(homography_npy, base_dir)
        if not Path(hp).exists():
            raise ValueError(f"calibration.homography_npy not found: {hp}")


def validate_unified_config(d: JsonDict, *, base_dir: Optional[str]) -> None:
    cam = dict(d.get("camera", {}) or {})
    cam_id = str(cam.get("id", "") or "").strip()
    if cam_id == "":
        raise ValueError("camera.id is required")
    cam_type = str(cam.get("type", "") or "").strip().lower()
    if cam_type not in {"rtsp", "video_file", "webcam"}:
        raise ValueError("camera.type must be one of: rtsp, video_file, webcam")
    src = cam.get("source")
    if cam_type == "webcam":
        try:
            int(src)
        except Exception:
            raise ValueError("camera.source must be an integer webcam index when camera.type=webcam")
    else:
        if str(src or "").strip() == "":
            raise ValueError("camera.source is required")
        if cam_type == "video_file":
            sp = _resolve_path(str(src), base_dir)
            if not Path(sp).exists():
                raise ValueError(f"camera.source video file not found: {sp}")

    det = dict(d.get("detection", {}) or {})
    if bool(det.get("enabled", True)):
        model_path = str(det.get("model_path", "") or "").strip()
        if model_path == "":
            raise ValueError("detection.model_path is required when detection.enabled=true")
        mp = _resolve_path(model_path, base_dir)
        p = Path(model_path)
        looks_like_path = p.is_absolute() or str(p.parent) not in {".", ""}
        if looks_like_path and not Path(mp).exists():
            raise ValueError(f"detection.model_path not found: {mp}")
        conf = float(det.get("conf", 0.25))
        if conf < 0.0 or conf > 1.0:
            raise ValueError("detection.conf must be in [0, 1]")
        iou = float(det.get("iou", 0.7))
        if iou < 0.0 or iou > 1.0:
            raise ValueError("detection.iou must be in [0, 1]")

    calib = dict(d.get("calibration", {}) or {})
    method = str(calib.get("method", "") or "").strip().lower()
    if method not in {"homography", "meters_per_pixel", ""}:
        raise ValueError("calibration.method must be one of: homography, meters_per_pixel")
    if method == "homography":
        hp = str(calib.get("homography_npy", "") or "").strip()
        if hp == "":
            raise ValueError("calibration.homography_npy is required when calibration.method=homography")
        rp = _resolve_path(hp, base_dir)
        if not Path(rp).exists():
            raise ValueError(f"calibration.homography_npy not found: {rp}")
    if method == "meters_per_pixel":
        if calib.get("meters_per_pixel") is None:
            raise ValueError("calibration.meters_per_pixel is required when calibration.method=meters_per_pixel")
    mpp = float(calib.get("meters_per_pixel", 1.0))
    if mpp <= 0.0:
        raise ValueError("calibration.meters_per_pixel must be > 0")

    tr = dict(d.get("tracking", {}) or {})
    backend = str(tr.get("backend", "simple_iou") or "simple_iou").strip().lower()
    if backend not in {"simple_iou", "bytetrack", "greedy_iou", "sort"}:
        raise ValueError("tracking.backend must be one of: simple_iou, bytetrack, greedy_iou, sort")
    params = dict(tr.get("params", {}) or {})
    iou_th = float(params.get("iou_threshold", 0.3))
    if iou_th < 0.0 or iou_th > 1.0:
        raise ValueError("tracking.params.iou_threshold must be in [0, 1]")
    max_missing = int(params.get("max_missing_frames", 30))
    if max_missing <= 0:
        raise ValueError("tracking.params.max_missing_frames must be > 0")
    max_age = int(params.get("max_age_frames", max_missing))
    if max_age <= 0:
        raise ValueError("tracking.params.max_age_frames must be > 0")

    sp = dict(d.get("speed_estimation", {}) or {})
    mode = str(sp.get("mode", "advanced") or "advanced").strip().lower()
    if mode not in {"advanced", "simple"}:
        raise ValueError("speed_estimation.mode must be one of: advanced, simple")

    simple = dict(sp.get("simple_mode", {}) or {})
    axis = str(simple.get("axis", "xy") or "xy").strip().lower()
    if axis not in {"x", "y", "xy"}:
        raise ValueError("speed_estimation.simple_mode.axis must be one of: x, y, xy")
    method = str(simple.get("method", "mean") or "mean").strip().lower()
    if method not in {"mean", "displacement"}:
        raise ValueError("speed_estimation.simple_mode.method must be one of: mean, displacement")

    adv = dict(sp.get("advanced_mode", {}) or {})
    min_dt = float(adv.get("min_dt_s", 0.05))
    max_dt = float(adv.get("max_dt_s", 1.0))
    if min_dt <= 0.0:
        raise ValueError("speed_estimation.advanced_mode.min_dt_s must be > 0")
    if max_dt < min_dt:
        raise ValueError("speed_estimation.advanced_mode.max_dt_s must be >= min_dt_s")
    min_disp = float(adv.get("min_displacement_m", 0.05))
    if min_disp < 0.0:
        raise ValueError("speed_estimation.advanced_mode.min_displacement_m must be >= 0")
    stop_t = float(adv.get("stop_time_s", 0.5))
    if stop_t < 0.0:
        raise ValueError("speed_estimation.advanced_mode.stop_time_s must be >= 0")

    turning = dict(sp.get("turning", {}) or {})
    if bool(turning.get("enabled", True)):
        max_turn_rate = float(turning.get("max_turn_rate_deg_per_s", 45.0))
        if max_turn_rate <= 0.0:
            raise ValueError("speed_estimation.turning.max_turn_rate_deg_per_s must be > 0 when turning.enabled=true")

    limits = dict(sp.get("limits", {}) or {})
    tsl = dict(limits.get("turn_speed_limit", {}) or {})
    if bool(tsl.get("enabled", True)):
        a_lat = float(tsl.get("a_lat_max_mps2", 2.5))
        if a_lat <= 0.0:
            raise ValueError("speed_estimation.limits.turn_speed_limit.a_lat_max_mps2 must be > 0")
        v_max = float(tsl.get("v_max_kmh", 180.0))
        if v_max <= 0.0:
            raise ValueError("speed_estimation.limits.turn_speed_limit.v_max_kmh must be > 0")
        v_min = float(tsl.get("v_min_kmh", v_max))
        if v_min <= 0.0 or v_min > v_max:
            raise ValueError("speed_estimation.limits.turn_speed_limit.v_min_kmh must be in (0, v_max_kmh]")
        alpha = float(tsl.get("alpha", 0.5))
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError("speed_estimation.limits.turn_speed_limit.alpha must be in [0, 1]")
        mode_tsl = str(tsl.get("mode", "curvature") or "curvature").strip().lower()
        if mode_tsl not in {"curvature", "linear_angle"}:
            raise ValueError("speed_estimation.limits.turn_speed_limit.mode must be one of: curvature, linear_angle")

    accel = dict(limits.get("accel_limit", {}) or {})
    if bool(accel.get("enabled", True)):
        a_max = float(accel.get("a_max_mps2", 6.0))
        if a_max <= 0.0:
            raise ValueError("speed_estimation.limits.accel_limit.a_max_mps2 must be > 0")

    smoothing = dict(sp.get("smoothing", {}) or {})
    sm_method = str(smoothing.get("method", "ema") or "ema").strip().lower()
    if sm_method not in {"ema"}:
        raise ValueError("speed_estimation.smoothing.method must be 'ema'")
    ema_alpha = float(smoothing.get("ema_alpha", 0.35))
    if ema_alpha < 0.0 or ema_alpha > 1.0:
        raise ValueError("speed_estimation.smoothing.ema_alpha must be in [0, 1]")

    dots = dict(sp.get("trajectory_dots", {}) or {})
    md = float(dots.get("min_distance_m", 0.8))
    if md <= 0.0:
        raise ValueError("speed_estimation.trajectory_dots.min_distance_m must be > 0")
    mdt = float(dots.get("min_dt_s", 0.5))
    if mdt <= 0.0:
        raise ValueError("speed_estimation.trajectory_dots.min_dt_s must be > 0")
    buf = int(dots.get("buffer_size", 6))
    if buf <= 0:
        raise ValueError("speed_estimation.trajectory_dots.buffer_size must be > 0")

    runtime = dict(d.get("runtime", {}) or {})
    fps_hint = float(runtime.get("fps_hint", 0.0) or 0.0)
    if fps_hint < 0.0:
        raise ValueError("runtime.fps_hint must be >= 0")
    resize = dict(runtime.get("resize", {}) or {})
    if bool(resize.get("enabled", False)):
        w = int(resize.get("width", 0) or 0)
        h = int(resize.get("height", 0) or 0)
        if w <= 0 or h <= 0:
            raise ValueError("runtime.resize.width and runtime.resize.height must be > 0 when resize.enabled=true")
    bs = int(runtime.get("buffer_size", 1) or 1)
    if bs <= 0:
        raise ValueError("runtime.buffer_size must be > 0")
    rto = float(runtime.get("read_timeout_s", 2.0) or 2.0)
    if rto <= 0.0:
        raise ValueError("runtime.read_timeout_s must be > 0")
    rcd = float(runtime.get("reconnect_delay_s", 2.0) or 2.0)
    if rcd < 0.0:
        raise ValueError("runtime.reconnect_delay_s must be >= 0")
