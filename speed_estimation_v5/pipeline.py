from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sqlite3
import sys
import time
import traceback
import csv
from collections import deque
import threading
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

try:
    from speed_estimation_v5.calibration import WorldMapping
    from speed_estimation_v5.detection import YoloV8Detector
    from speed_estimation_v5 import math as motion_math
    from speed_estimation_v5.config import CameraSource, V5PipelineConfig, default_config, load_pipeline_config_from_file
except ModuleNotFoundError:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from speed_estimation_v5.calibration import WorldMapping
    from speed_estimation_v5.detection import YoloV8Detector
    from speed_estimation_v5 import math as motion_math
    from speed_estimation_v5.config import CameraSource, V5PipelineConfig, default_config, load_pipeline_config_from_file


JsonDict = Dict[str, Any]


def _bbox_anchor_bottom_center_xy(bbox_xyxy: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox_xyxy
    return (0.5 * (float(x1) + float(x2)), float(y2))


def _detection_from_json(obj: Mapping[str, Any]) -> motion_math.Detection:
    bbox = obj.get("bbox_xyxy")
    if bbox is None:
        x = float(obj["x"])
        y = float(obj["y"])
        w = float(obj["w"])
        h = float(obj["h"])
        bbox = [x, y, x + w, y + h]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return motion_math.Detection(
        bbox_xyxy=(x1, y1, x2, y2),
        score=float(obj.get("score", 1.0)),
        class_id=int(obj.get("class_id", 0)),
        class_name=str(obj.get("class_name", "")),
    )


def _detections_from_json(seq: Sequence[Mapping[str, Any]]) -> List[motion_math.Detection]:
    return [_detection_from_json(x) for x in seq]


def _sort_detections(dets: List[motion_math.Detection]) -> List[motion_math.Detection]:
    def key_fn(d: motion_math.Detection) -> Tuple[int, float, float, float, float, float]:
        x1, y1, x2, y2 = d.bbox_xyxy
        return (
            int(d.class_id),
            float(x1),
            float(y1),
            float(x2),
            float(y2),
            -float(d.score),
        )

    dets.sort(key=key_fn)
    return dets


def _iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = a_area + b_area - inter
    if denom <= 0.0:
        return 0.0
    return float(inter / denom)


class _SimpleIoUTracker:
    def __init__(self, *, iou_threshold: float = 0.3, max_missing_frames: int = 30) -> None:
        self._iou_threshold = float(iou_threshold)
        self._max_missing_frames = int(max_missing_frames)
        self._next_id = 1
        self._tracks: Dict[int, motion_math.TrackState] = {}

    def reset(self) -> None:
        self._next_id = 1
        self._tracks.clear()

    def update(
        self,
        *,
        camera_id: str,
        detections: Sequence[motion_math.Detection],
    ) -> List[motion_math.TrackState]:
        if len(detections) == 0:
            for ts in self._tracks.values():
                ts.age_frames += 1
                ts.time_since_update += 1
            self._prune()
            return []

        track_ids = sorted(self._tracks.keys())
        cand: List[Tuple[float, int, int]] = []
        for det_idx, det in enumerate(detections):
            for tid in track_ids:
                iou = _iou_xyxy(self._tracks[tid].bbox_xyxy, det.bbox_xyxy)
                if iou >= self._iou_threshold:
                    cand.append((iou, tid, det_idx))

        cand.sort(key=lambda x: (-x[0], int(x[1]), int(x[2])))
        assigned_tracks: set[int] = set()
        assigned_dets: set[int] = set()
        matches: List[Tuple[int, int]] = []
        for _iou, tid, det_idx in cand:
            if tid in assigned_tracks or det_idx in assigned_dets:
                continue
            assigned_tracks.add(tid)
            assigned_dets.add(det_idx)
            matches.append((tid, det_idx))

        for tid in track_ids:
            ts = self._tracks[tid]
            ts.age_frames += 1
            ts.time_since_update += 1

        for tid, det_idx in matches:
            det = detections[det_idx]
            ts = self._tracks[tid]
            ts.camera_id = str(camera_id)
            ts.class_id = int(det.class_id)
            ts.class_name = str(det.class_name)
            ts.bbox_xyxy = tuple(float(v) for v in det.bbox_xyxy)
            ts.score = float(det.score)
            ts.hits += 1
            ts.time_since_update = 0

        for det_idx, det in enumerate(detections):
            if det_idx in assigned_dets:
                continue
            tid = int(self._next_id)
            self._next_id += 1
            self._tracks[tid] = motion_math.TrackState(
                track_id=tid,
                camera_id=str(camera_id),
                class_id=int(det.class_id),
                class_name=str(det.class_name),
                bbox_xyxy=tuple(float(v) for v in det.bbox_xyxy),
                score=float(det.score),
                age_frames=1,
                hits=1,
                time_since_update=0,
            )

        self._prune()

        visible = [ts for ts in self._tracks.values() if int(ts.time_since_update) == 0]
        visible.sort(key=lambda x: int(x.track_id))
        return visible

    def _prune(self) -> None:
        to_del = [tid for tid, ts in self._tracks.items() if int(ts.time_since_update) > self._max_missing_frames]
        for tid in to_del:
            del self._tracks[tid]


class SpeedEstimationPipelineV5:
    """
    Production-oriented speed estimation pipeline.

    - Detection: consumes externally provided detections (JSON or Detection list)
    - Tracking: local, simple IoU-based ID assignment
    - Calibration: maps pixel anchors to world coordinates (meters)
    - Metrics: uses local motion math (speed, turning angle, curvature, angular rate)
    """

    def __init__(self, cfg: V5PipelineConfig) -> None:
        self._cfg = cfg
        self._mapping = WorldMapping.from_config(cfg.calibration, base_dir=cfg.base_dir)

        runtime_cfg = dict(cfg.runtime or {})
        resize_cfg = dict(runtime_cfg.get("resize", {}) or {})
        self._resize_enabled = bool(resize_cfg.get("enabled", False))
        self._resize_width = int(resize_cfg.get("width", 0) or 0)
        self._resize_height = int(resize_cfg.get("height", 0) or 0)
        if self._resize_enabled and (self._resize_width <= 0 or self._resize_height <= 0):
            self._resize_enabled = False

        det_cfg = dict(cfg.detection or {})
        det_params = dict(det_cfg.get("params", {}) or {})
        det_enabled = bool(det_cfg.get("enabled", True))
        self._detector: Optional[YoloV8Detector]
        if det_enabled:
            self._detector = YoloV8Detector(
                model_path=str(det_params.get("model_path", det_cfg.get("model_path", "yolov8n.pt"))),
                base_dir=cfg.base_dir,
                conf=float(det_params.get("conf", det_cfg.get("conf", 0.25))),
                iou=float(det_params.get("iou", det_cfg.get("iou", 0.7))),
                device=det_params.get("device", det_cfg.get("device")),
                classes=det_params.get("classes", det_cfg.get("classes")),
                class_whitelist=det_params.get("class_whitelist", det_cfg.get("class_whitelist")),
            )
        else:
            self._detector = None

        tr_cfg = dict(cfg.tracking or {})
        tr_params = dict(tr_cfg.get("params", {}) or {})
        self._tracker_backend = str(tr_cfg.get("backend", "bytetrack") or "bytetrack").strip().lower()
        self._tracker = None
        self._legacy_tracker = None
        self._legacy_tracker_input_cls = None
        self._legacy_detection_cls = None
        self._legacy_tracker_backend: Optional[str] = None
        self._legacy_tracker_params: Optional[Dict[str, Any]] = None
        if self._tracker_backend in {"bytetrack", "greedy_iou", "sort"}:
            import sys
            from pathlib import Path

            root = Path(__file__).resolve().parents[1]
            src = root / "src"
            if str(src) not in sys.path:
                sys.path.insert(0, str(src))

            from speedestimation.tracking.registry import create_tracker
            from speedestimation.tracking.base import TrackerInput
            from speedestimation.utils.types import Detection as LegacyDetection

            legacy_params = dict(tr_params)
            if self._tracker_backend == "bytetrack":
                if "max_age_frames" not in legacy_params and "max_missing_frames" in legacy_params:
                    legacy_params["max_age_frames"] = int(legacy_params.get("max_missing_frames"))
                legacy_params.setdefault("iou_threshold", 0.5)
                legacy_params.setdefault("max_age_frames", 30)
                legacy_params.setdefault("min_hits", 4)
                legacy_params.setdefault("high_conf_threshold", 0.7)
                legacy_params.setdefault("low_conf_threshold", 0.3)
            else:
                if "max_age_frames" not in legacy_params and "max_missing_frames" in legacy_params:
                    legacy_params["max_age_frames"] = int(legacy_params.get("max_missing_frames"))
            self._legacy_tracker_backend = str(self._tracker_backend)
            self._legacy_tracker_params = dict(legacy_params)
            self._legacy_tracker = create_tracker(self._tracker_backend, legacy_params)
            self._legacy_tracker_input_cls = TrackerInput
            self._legacy_detection_cls = LegacyDetection
        else:
            self._tracker = _SimpleIoUTracker(
                iou_threshold=float(tr_params.get("iou_threshold", 0.3)),
                max_missing_frames=int(tr_params.get("max_missing_frames", 30)),
            )

        self._speed_cfg = motion_math.SpeedEstimatorConfig.from_dict(dict(cfg.speed or {}))
        self._estimator = motion_math.SpeedEstimator(self._speed_cfg)

        self._latest_sample_by_track: Dict[int, Any] = {}

    def reset(self) -> None:
        self._estimator.prune_missing([])
        if self._tracker is not None:
            self._tracker.reset()
        if self._legacy_tracker is not None:
            if self._legacy_tracker_backend is not None and self._legacy_tracker_params is not None:
                from speedestimation.tracking.registry import create_tracker

                self._legacy_tracker = create_tracker(self._legacy_tracker_backend, dict(self._legacy_tracker_params))
        self._latest_sample_by_track.clear()

    def process_frame(
        self,
        *,
        frame_index: int,
        timestamp_s: float,
        frame_bgr: Optional[np.ndarray] = None,
        detections: Optional[Union[Sequence[motion_math.Detection], Sequence[Mapping[str, Any]]]] = None,
    ) -> JsonDict:
        """
        Process a single frame worth of data and return JSON-serializable results.

        Inputs:
          - frame_bgr: required only when YOLO inference is used
          - detections: if provided, YOLO inference is skipped
        """
        frame_for_detection = frame_bgr
        if (
            self._resize_enabled
            and detections is None
            and frame_bgr is not None
            and self._resize_width > 0
            and self._resize_height > 0
        ):
            try:
                import cv2
            except Exception as e:
                raise RuntimeError("OpenCV (cv2) is required for runtime.resize") from e
            frame_for_detection = cv2.resize(frame_bgr, (int(self._resize_width), int(self._resize_height)))

        dets = self._get_detections(
            frame_index=frame_index,
            timestamp_s=timestamp_s,
            frame_bgr=frame_for_detection,
            detections=detections,
        )

        if self._legacy_tracker is not None:
            TrackerInput = self._legacy_tracker_input_cls
            LegacyDetection = self._legacy_detection_cls
            if TrackerInput is None or LegacyDetection is None:
                raise RuntimeError("Legacy tracker dependencies are not initialized")
            legacy_dets = [
                LegacyDetection(
                    bbox_xyxy=d.bbox_xyxy,
                    score=float(d.score),
                    class_id=int(d.class_id),
                    class_name=str(d.class_name),
                )
                for d in dets
            ]
            out0 = self._legacy_tracker.update(
                TrackerInput(
                    camera_id=str(self._cfg.camera_id),
                    frame_index=int(frame_index),
                    timestamp_s=float(timestamp_s),
                    detections=legacy_dets,
                )
            )
            track_states = list(out0.tracks)
        else:
            track_states = self._tracker.update(camera_id=str(self._cfg.camera_id), detections=dets)  # type: ignore[union-attr]
        world_tracks = self._tracks_to_world_tracks(track_states, frame_index=int(frame_index), timestamp_s=float(timestamp_s))
        samples = self._estimator.update(world_tracks)
        alive_keys = [(str(self._cfg.camera_id), int(ts.track_id)) for ts in track_states]
        self._estimator.prune_missing(alive_keys)

        alive_ids = {int(ts.track_id) for ts in track_states}
        if self._latest_sample_by_track:
            to_del = [tid for tid in self._latest_sample_by_track.keys() if int(tid) not in alive_ids]
            for tid in to_del:
                del self._latest_sample_by_track[int(tid)]

        for s in samples:
            self._latest_sample_by_track[int(s.track_id)] = s

        sample_by_id: Dict[int, Any] = {int(s.track_id): s for s in samples}
        tracks_out: List[JsonDict] = []

        for ts in sorted(track_states, key=lambda x: int(x.track_id)):
            px_anchor = _bbox_anchor_bottom_center_xy(ts.bbox_xyxy)
            w_anchor = self._mapping.pixel_to_world(px_anchor)

            sample_emitted = int(ts.track_id) in sample_by_id
            sample = sample_by_id.get(int(ts.track_id))
            if sample is None:
                sample = self._latest_sample_by_track.get(int(ts.track_id))
            used_cache = bool((not sample_emitted) and (sample is not None))
            dots = self._estimator.get_trajectory_dots(str(self._cfg.camera_id), int(ts.track_id))
            turn_angle_vec_signed_deg, angular_rate_vec_signed_deg_s = motion_math.turning_dot_cross_metrics(dots)
            dot_count = int(len(dots))
            dot_last_dist_m = float("nan")
            dot_last_dt_s = float("nan")
            if len(dots) >= 2:
                p0 = dots[-2]
                p1 = dots[-1]
                dot_last_dt_s = float(p1[2] - p0[2])
                dx = float(p1[0] - p0[0])
                dy = float(p1[1] - p0[1])
                dot_last_dist_m = float((dx * dx + dy * dy) ** 0.5)

            if sample is None:
                tracks_out.append(
                    {
                        "track_id": int(ts.track_id),
                        "class_id": int(ts.class_id),
                        "class_name": str(ts.class_name),
                        "bbox_xyxy": [float(v) for v in ts.bbox_xyxy],
                        "anchor_world_m": [float(w_anchor[0]), float(w_anchor[1])] if w_anchor is not None else None,
                        "speed": {"mps": 0.0, "kmh": 0.0},
                        "turning": {"turn_angle_signed_deg": 0.0, "angular_rate_deg_s": 0.0},
                        "diagnostics": {
                            "dot_count": float(dot_count),
                            "dot_last_dist_m": float(dot_last_dist_m),
                            "dot_last_dt_s": float(dot_last_dt_s),
                            "sample_emitted": 1.0 if sample_emitted else 0.0,
                            "sample_used_cache": 1.0 if used_cache else 0.0,
                            "dot_cross_turn_angle_deg": float(turn_angle_vec_signed_deg),
                            "dot_cross_angular_rate_deg_s": float(angular_rate_vec_signed_deg_s),
                        },
                    }
                )
                continue

            tracks_out.append(
                {
                    "track_id": int(ts.track_id),
                    "class_id": int(ts.class_id),
                    "class_name": str(ts.class_name),
                    "bbox_xyxy": [float(v) for v in ts.bbox_xyxy],
                    "anchor_world_m": [float(sample.world_xy_m[0]), float(sample.world_xy_m[1])],
                    "speed": {"mps": float(sample.speed_mps_smoothed), "kmh": float(motion_math.mps_to_kmh(sample.speed_mps_smoothed))},
                    "turning": {
                        "turn_angle_signed_deg": float(sample.turn_angle_signed_deg),
                        "angular_rate_deg_s": float(sample.angular_rate_deg_s),
                    },
                    "diagnostics": {
                        "dot_count": float(dot_count),
                        "dot_last_dist_m": float(dot_last_dist_m),
                        "dot_last_dt_s": float(dot_last_dt_s),
                        "sample_emitted": 1.0 if sample_emitted else 0.0,
                        "sample_used_cache": 1.0 if used_cache else 0.0,
                        "dot_cross_turn_angle_deg": float(sample.turn_angle_dot_cross_signed_deg),
                        "dot_cross_angular_rate_deg_s": float(sample.angular_rate_dot_cross_signed_deg_s),
                    },
                }
            )

        return {
            "camera_id": str(self._cfg.camera_id),
            "frame_index": int(frame_index),
            "timestamp_s": float(timestamp_s),
            "tracks": tracks_out,
            "num_detections": int(len(dets)),
            "num_tracks": int(len(track_states)),
        }

    def process_frames(
        self,
        frames: Iterable[Tuple[int, float, np.ndarray]],
        *,
        detections_by_frame: Optional[Mapping[int, Sequence[Union[motion_math.Detection, Mapping[str, Any]]]]] = None,
    ) -> Iterator[JsonDict]:
        """
        Process a stream of frames.

        Args:
          frames: iterator of (frame_index, timestamp_s, frame_bgr)
          detections_by_frame: optional mapping from frame_index to detections

        Yields:
          per-frame JSON results (deterministic ordering by track_id)
        """
        for frame_index, t_s, frame_bgr in frames:
            dets = None
            if detections_by_frame is not None:
                dets = detections_by_frame.get(int(frame_index))
            yield self.process_frame(frame_index=int(frame_index), timestamp_s=float(t_s), frame_bgr=frame_bgr, detections=dets)

    def _get_detections(
        self,
        *,
        frame_index: int,
        timestamp_s: float,
        frame_bgr: Optional[np.ndarray],
        detections: Optional[Union[Sequence[motion_math.Detection], Sequence[Mapping[str, Any]]]],
    ) -> List[motion_math.Detection]:
        if detections is not None:
            if len(detections) == 0:
                return []
            first = detections[0]
            if isinstance(first, motion_math.Detection):
                return _sort_detections(list(detections))  # type: ignore[arg-type]
            return _sort_detections(_detections_from_json(detections))  # type: ignore[arg-type]
        if self._detector is None:
            return []
        if frame_bgr is None:
            raise ValueError("frame_bgr is required when detections are not provided")
        det_json = self._detector.detect(frame_bgr)
        return _sort_detections(_detections_from_json(det_json))

    def _tracks_to_world_tracks(
        self, tracks: Sequence[motion_math.TrackState], *, frame_index: int, timestamp_s: float
    ) -> List[motion_math.Track]:
        out: List[motion_math.Track] = []
        for ts in tracks:
            px_anchor = _bbox_anchor_bottom_center_xy(ts.bbox_xyxy)
            w_anchor = self._mapping.pixel_to_world(px_anchor)
            out.append(
                motion_math.Track(
                    camera_id=str(self._cfg.camera_id),
                    frame_index=int(frame_index),
                    timestamp_s=float(timestamp_s),
                    state=ts,
                    world_xy_m=w_anchor,
                )
            )
        return out


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--config", type=str, default="", help="Path to a JSON config file")
    p.add_argument("--window", type=str, default="v5 overlay", help="OpenCV window name")
    p.add_argument("--verbose", action="store_true", help="Enable verbose logs")
    p.add_argument("--no-traj", action="store_true", help="Disable drawing trajectory dots")
    p.add_argument("--traj-len", type=int, default=40, help="Max trajectory dots per track")
    p.add_argument("--no-vector", action="store_true", help="Disable drawing last motion vector")
    p.add_argument("--read-timeout-s", type=float, default=2.0, help="Seconds before treating RTSP as stalled")
    p.add_argument("--buffer-size", type=int, default=1, help="OpenCV capture buffer size (best-effort)")
    return p.parse_args(list(argv) if argv is not None else None)


def _open_capture(source: CameraSource) -> Optional["cv2.VideoCapture"]:
    import cv2

    if isinstance(source, int):
        cap = cv2.VideoCapture(int(source))
        if cap.isOpened():
            return cap
        cap.release()
        return None

    s = str(source)
    cap = cv2.VideoCapture(s, cv2.CAP_FFMPEG)
    if cap.isOpened():
        return cap
    cap.release()
    cap = cv2.VideoCapture(s)
    if cap.isOpened():
        return cap
    cap.release()
    return None


class _ThreadedFrameGrabber:
    def __init__(self, *, source: CameraSource, buffer_size: int, target_fps: float = 0.0) -> None:
        self._source = source
        self._buffer_size = int(buffer_size)
        self._target_fps = float(target_fps)
        self._pace = isinstance(self._source, str) and os.path.exists(str(self._source))
        self._lock = threading.Lock()
        self._last_frame: Optional[np.ndarray] = None
        self._last_frame_t_s: Optional[float] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cap = None
        self._fps = 0.0
        self._frame_interval_s = 0.0

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="rtsp-grabber", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None:
            t.join(timeout=1.0)
        self._thread = None
        cap = self._cap
        self._cap = None
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

    def reset(self) -> None:
        with self._lock:
            self._last_frame = None
            self._last_frame_t_s = None

    def get_latest(self) -> Tuple[Optional[np.ndarray], Optional[float]]:
        with self._lock:
            if self._last_frame is None:
                return None, None
            return self._last_frame.copy(), self._last_frame_t_s

    def _run(self) -> None:
        import cv2

        while not self._stop.is_set():
            cap = _open_capture(self._source)
            if cap is None:
                time.sleep(0.5)
                continue
            self._cap = cap

            if self._pace:
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps > 0:
                    self._fps = fps
                    self._frame_interval_s = 1.0 / fps
                elif self._target_fps > 0:
                    self._fps = self._target_fps
                    self._frame_interval_s = 1.0 / self._target_fps
                else:
                    self._fps = 30.0
                    self._frame_interval_s = 1.0 / 30.0
            else:
                self._fps = 0.0
                self._frame_interval_s = 0.0

            try:
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, max(1, int(self._buffer_size)))
                except Exception:
                    pass

                last_read_time = 0.0
                while not self._stop.is_set():
                    if self._frame_interval_s > 0:
                        elapsed = time.time() - last_read_time
                        if elapsed < self._frame_interval_s:
                            time.sleep(self._frame_interval_s - elapsed)

                    ok, frame = cap.read()
                    if not ok or frame is None:
                        break
                    last_read_time = time.time()
                    with self._lock:
                        self._last_frame = frame
                        self._last_frame_t_s = time.time()
            finally:
                try:
                    cap.release()
                except Exception:
                    pass
                self._cap = None
            time.sleep(0.1)


def run_rtsp_overlay(
    *,
    rtsp_url: Optional[CameraSource] = None,
    cfg: Optional[V5PipelineConfig] = None,
    window_name: str = "v5 overlay",
    reconnect: bool = True,
    reconnect_delay_s: Optional[float] = None,
    max_consecutive_read_failures: int = 30,
    draw_traj: bool = True,
    traj_len: int = 40,
    draw_vector: bool = True,
    read_timeout_s: Optional[float] = None,
    buffer_size: Optional[int] = None,
    export_cfg: Optional[Dict[str, Any]] = None,
    event_cfg: Optional[Dict[str, Any]] = None,
) -> None:
    import cv2

    logger = logging.getLogger("speed_estimation_v5")

    if cfg is None or rtsp_url is None:
        cfg0, rtsp0 = default_config()
        if cfg is None:
            cfg = cfg0
        if rtsp_url is None:
            rtsp_url = rtsp0

    if isinstance(rtsp_url, str):
        rtsp_url = str(rtsp_url or "").strip()
        if rtsp_url == "":
            raise SystemExit("Missing camera source.")

    logger.info("Opening source: %s", rtsp_url)

    try:
        p = SpeedEstimationPipelineV5(cfg)
    except Exception:
        logger.error("Failed to initialize pipeline:\n%s", traceback.format_exc())
        raise

    speed_cfg = dict(getattr(cfg, "speed", {}) or {})
    dots_cfg = dict(speed_cfg.get("dots", {}) or {})
    dot_min_distance_m = float(dots_cfg.get("min_distance_m", 0.8))
    dot_min_dt_s = float(dots_cfg.get("min_dt_s", 0.5))

    runtime_cfg = dict(getattr(cfg, "runtime", {}) or {})
    fps_hint = float(runtime_cfg.get("fps_hint", 0.0) or 0.0)

    if buffer_size is None:
        buffer_size = int(runtime_cfg.get("buffer_size", 1))
    if read_timeout_s is None:
        read_timeout_s = float(runtime_cfg.get("read_timeout_s", 2.0))
    if reconnect_delay_s is None:
        reconnect_delay_s = float(runtime_cfg.get("reconnect_delay_s", 2.0))

    grabber = _ThreadedFrameGrabber(source=rtsp_url, buffer_size=int(buffer_size), target_fps=fps_hint)
    grabber.start()

    exporter = _ExportWriter.from_config(export_cfg) if export_cfg else None
    clipper = _EventClipper.from_config(event_cfg, camera_id=str(cfg.camera_id), fps_hint=fps_hint) if event_cfg else None

    traj_len = max(2, int(traj_len))
    history_px_by_track: Dict[int, "deque[Tuple[int, int, float, float, float]]"] = {}

    t0_frame_t_s: Optional[float] = None
    frame_index = 0
    consecutive_failures = 0
    while True:
        frame, frame_t_s = grabber.get_latest()
        now = time.time()
        if frame is None or frame_t_s is None:
            consecutive_failures += 1
            if consecutive_failures == 1:
                logger.warning("Waiting for frames...")
            time.sleep(0.02)
            continue

        if (now - float(frame_t_s)) > float(read_timeout_s):
            consecutive_failures += 1
            logger.warning("RTSP stalled (no new frames for %.2fs).", now - float(frame_t_s))
            if reconnect and consecutive_failures >= int(max_consecutive_read_failures):
                logger.warning("Restarting grabber after %d stalled checks...", consecutive_failures)
                grabber.stop()
                time.sleep(max(0.0, float(reconnect_delay_s)))
                grabber = _ThreadedFrameGrabber(source=rtsp_url, buffer_size=int(buffer_size), target_fps=fps_hint)
                grabber.start()
                consecutive_failures = 0
            time.sleep(0.02)
            continue

        consecutive_failures = 0

        if t0_frame_t_s is None:
            t0_frame_t_s = float(frame_t_s)
        t_s = float(frame_t_s) - float(t0_frame_t_s)

        try:
            out = p.process_frame(frame_index=frame_index, timestamp_s=t_s, frame_bgr=frame, detections=None)
        except Exception:
            logger.error("Pipeline error at frame_index=%d:\n%s", frame_index, traceback.format_exc())
            time.sleep(0.05)
            frame_index += 1
            continue

        if exporter is not None:
            try:
                exporter.write(out)
            except Exception:
                logger.error("Export error:\n%s", traceback.format_exc())
                exporter.close()
                exporter = None
        if clipper is not None:
            try:
                clipper.update(frame_bgr=frame, out=out)
            except Exception:
                logger.error("Event clipper error:\n%s", traceback.format_exc())
                clipper.close()
                clipper = None

        alive_ids: set[int] = set()
        for tr in out["tracks"]:
            tid = int(tr["track_id"])
            alive_ids.add(tid)
            x1, y1, x2, y2 = [int(v) for v in tr["bbox_xyxy"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if draw_traj or draw_vector:
                ax = int(round(0.5 * (x1 + x2)))
                ay = int(round(y2))
                t_s = float(out.get("timestamp_s", 0.0))
                w = p._mapping.pixel_to_world((float(ax), float(ay)))
                wx = float(w[0]) if w is not None else float("nan")
                wy = float(w[1]) if w is not None else float("nan")
                h = history_px_by_track.get(tid)
                if h is None:
                    h = deque(maxlen=traj_len)
                    history_px_by_track[tid] = h
                add = False
                if not h:
                    add = True
                else:
                    last_px, last_py, last_wx, last_wy, last_t = h[-1]
                    dt = float(t_s - last_t)
                    if dt > 0.0:
                        if np.isfinite(wx) and np.isfinite(wy) and np.isfinite(last_wx) and np.isfinite(last_wy):
                            dx = float(wx - last_wx)
                            dy = float(wy - last_wy)
                            dist = (dx * dx + dy * dy) ** 0.5
                            if dist >= dot_min_distance_m and dt >= dot_min_dt_s:
                                add = True
                        else:
                            dpx = float(ax - last_px)
                            dpy = float(ay - last_py)
                            dist_px = (dpx * dpx + dpy * dpy) ** 0.5
                            if dist_px >= 1.0 and dt >= dot_min_dt_s:
                                add = True
                if add:
                    h.append((ax, ay, wx, wy, t_s))
                if draw_traj:
                    pts = list(h)
                    for px, py, _wx, _wy, _t in pts:
                        cv2.circle(frame, (int(px), int(py)), 2, (0, 200, 255), -1)
                if draw_vector and len(h) >= 2:
                    (p0x, p0y, _w0x, _w0y, _t0) = h[-2]
                    (p1x, p1y, _w1x, _w1y, _t1) = h[-1]
                    if p0x != p1x or p0y != p1y:
                        cv2.arrowedLine(frame, (int(p0x), int(p0y)), (int(p1x), int(p1y)), (255, 0, 0), 2, tipLength=0.3)

            sp_kmh = float(tr["speed"]["kmh"])
            turn = float(tr["turning"]["turn_angle_signed_deg"])
            omega = float(tr["turning"]["angular_rate_deg_s"])
            label = f"id={tr['track_id']} {tr['class_name']} {sp_kmh:.1f}km/h turn={turn:.1f} w={omega:.1f}"
            cv2.putText(frame, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if history_px_by_track:
            to_del = [tid for tid in history_px_by_track.keys() if int(tid) not in alive_ids]
            for tid in to_del:
                del history_px_by_track[tid]

        try:
            cv2.imshow(window_name, frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
        except Exception:
            logger.error("OpenCV display error:\n%s", traceback.format_exc())
            break
        frame_index += 1

    grabber.stop()
    if exporter is not None:
        exporter.close()
    if clipper is not None:
        clipper.close()
    cv2.destroyAllWindows()


class _ExportWriter:
    def __init__(self, *, fmt: str, path: str) -> None:
        self._fmt = str(fmt).strip().lower()
        self._path = str(path)
        self._fp = open(self._path, "w", encoding="utf-8", newline="")
        self._csv: Optional[csv.DictWriter] = None
        if self._fmt == "csv":
            self._csv = csv.DictWriter(
                self._fp,
                fieldnames=[
                    "camera_id",
                    "frame_index",
                    "timestamp_s",
                    "track_id",
                    "class_name",
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                    "world_x_m",
                    "world_y_m",
                    "speed_kmh",
                    "turn_angle_signed_deg",
                    "angular_rate_deg_s",
                ],
            )
            self._csv.writeheader()

    @staticmethod
    def from_config(cfg: Optional[Dict[str, Any]]) -> Optional["_ExportWriter"]:
        if not cfg:
            return None
        if not bool(cfg.get("enabled", False)):
            return None
        fmt = str(cfg.get("format", "json") or "json").strip().lower()
        if fmt not in {"json", "csv"}:
            raise ValueError("output.export.format must be one of: json, csv")
        out_dir = str(cfg.get("output_dir", "") or "").strip()
        if out_dir == "":
            raise ValueError("output.export.output_dir is required when output.export.enabled=true")
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        out_path = str(Path(out_dir) / ("tracks.jsonl" if fmt == "json" else "tracks.csv"))
        return _ExportWriter(fmt=fmt, path=out_path)

    def write(self, out: Dict[str, Any]) -> None:
        if self._fmt == "json":
            self._fp.write(json.dumps(out, ensure_ascii=False) + "\n")
            return
        if self._csv is None:
            return
        cam_id = str(out.get("camera_id", ""))
        frame_index = int(out.get("frame_index", 0))
        ts = float(out.get("timestamp_s", 0.0))
        for tr in out.get("tracks", []) or []:
            bbox = tr.get("bbox_xyxy") or [0, 0, 0, 0]
            world = tr.get("anchor_world_m")
            wx = "" if world is None else float(world[0])
            wy = "" if world is None else float(world[1])
            sp = tr.get("speed", {}) or {}
            turning = tr.get("turning", {}) or {}
            row = {
                "camera_id": cam_id,
                "frame_index": frame_index,
                "timestamp_s": ts,
                "track_id": int(tr.get("track_id", 0)),
                "class_name": str(tr.get("class_name", "")),
                "x1": float(bbox[0]),
                "y1": float(bbox[1]),
                "x2": float(bbox[2]),
                "y2": float(bbox[3]),
                "world_x_m": wx,
                "world_y_m": wy,
                "speed_kmh": float(sp.get("kmh", 0.0)),
                "turn_angle_signed_deg": float(turning.get("turn_angle_signed_deg", 0.0)),
                "angular_rate_deg_s": float(turning.get("angular_rate_deg_s", 0.0)),
            }
            self._csv.writerow(row)

    def close(self) -> None:
        try:
            self._fp.close()
        except Exception:
            pass


class _EventClipper:
    def __init__(
        self,
        *,
        enabled: bool,
        camera_id: str,
        output_dir: str,
        db_path: str,
        pre_s: float,
        post_s: float,
        cooldown_s: float,
        min_consecutive_samples: int,
        min_duration_s: float,
        max_gap_s: float,
        speed_limit_kmh: float,
        angle_min_deg: float,
        angle_speed_limit_kmh: float,
        fps: float,
    ) -> None:
        self._enabled = bool(enabled)
        self._camera_id = str(camera_id)
        self._output_dir = str(output_dir)
        self._db_path = str(db_path)
        self._pre_s = max(0.0, float(pre_s))
        self._post_s = max(0.0, float(post_s))
        self._cooldown_s = max(0.0, float(cooldown_s))
        self._fps = float(fps) if float(fps) > 0.0 else 30.0
        self._min_consecutive_samples = max(1, int(min_consecutive_samples))
        self._min_duration_s = max(0.0, float(min_duration_s))
        self._max_gap_s = float(max_gap_s)
        if not (self._max_gap_s > 0.0):
            self._max_gap_s = max(0.2, 2.0 / float(self._fps))
        self._speed_limit_kmh = float(speed_limit_kmh)
        self._angle_min_deg = float(angle_min_deg)
        self._angle_speed_limit_kmh = float(angle_speed_limit_kmh)

        self._pre_buf: "deque[Tuple[float, Any]]" = deque()
        self._active: Dict[int, Dict[str, Any]] = {}
        self._last_event_ts: Dict[int, float] = {}
        self._over_state: Dict[int, Dict[str, Any]] = {}

        Path(self._output_dir).mkdir(parents=True, exist_ok=True)
        db_parent = Path(self._db_path).parent
        if str(db_parent) not in {".", ""}:
            db_parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS events ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "camera_id TEXT NOT NULL,"
            "track_id INTEGER NOT NULL,"
            "timestamp_s REAL NOT NULL,"
            "speed_kmh REAL NOT NULL,"
            "angle_deg REAL NOT NULL,"
            "angular_rate_deg_s REAL NOT NULL,"
            "speed_limit_kmh REAL NOT NULL,"
            "angle_min_deg REAL NOT NULL,"
            "angle_speed_limit_kmh REAL NOT NULL,"
            "clip_path TEXT NOT NULL"
            ")"
        )
        self._conn.commit()

    @staticmethod
    def from_config(cfg: Optional[Dict[str, Any]], *, camera_id: str, fps_hint: float) -> Optional["_EventClipper"]:
        if not cfg:
            return None
        if not bool(cfg.get("enabled", False)):
            return None
        out_dir = str(cfg.get("output_dir", "outputs/clips") or "outputs/clips").strip()
        db_path = str(cfg.get("db_path", "outputs/events.sqlite") or "outputs/events.sqlite").strip()
        pre_s = float(cfg.get("pre_s", 15.0))
        post_s = float(cfg.get("post_s", 15.0))
        cooldown_s = float(cfg.get("cooldown_s", 10.0))
        min_consecutive_samples = int(cfg.get("min_consecutive_samples", 3))
        min_duration_s = float(cfg.get("min_duration_s", 0.25))
        max_gap_s = float(cfg.get("max_gap_s", 0.0))
        speed_limit_kmh = float(cfg.get("speed_limit_kmh", 0.0))
        angle_min_deg = float(cfg.get("angle_min_deg", 0.0))
        angle_speed_limit_kmh = float(cfg.get("angle_speed_limit_kmh", 0.0))
        fps = float(cfg.get("fps", fps_hint))
        if out_dir == "":
            raise ValueError("output.event_clipping.output_dir is required when enabled=true")
        if db_path == "":
            raise ValueError("output.event_clipping.db_path is required when enabled=true")
        if pre_s < 0.0 or post_s < 0.0:
            raise ValueError("output.event_clipping.pre_s and post_s must be >= 0")
        if cooldown_s < 0.0:
            raise ValueError("output.event_clipping.cooldown_s must be >= 0")
        if int(min_consecutive_samples) <= 0:
            raise ValueError("output.event_clipping.min_consecutive_samples must be > 0")
        if float(min_duration_s) < 0.0:
            raise ValueError("output.event_clipping.min_duration_s must be >= 0")
        if float(max_gap_s) < 0.0:
            raise ValueError("output.event_clipping.max_gap_s must be >= 0")
        if speed_limit_kmh <= 0.0 and (angle_min_deg <= 0.0 or angle_speed_limit_kmh <= 0.0):
            raise ValueError(
                "output.event_clipping must set speed_limit_kmh > 0 or (angle_min_deg > 0 and angle_speed_limit_kmh > 0)"
            )
        if fps < 0.0:
            raise ValueError("output.event_clipping.fps must be >= 0")
        return _EventClipper(
            enabled=True,
            camera_id=camera_id,
            output_dir=out_dir,
            db_path=db_path,
            pre_s=pre_s,
            post_s=post_s,
            cooldown_s=cooldown_s,
            min_consecutive_samples=min_consecutive_samples,
            min_duration_s=min_duration_s,
            max_gap_s=max_gap_s,
            speed_limit_kmh=speed_limit_kmh,
            angle_min_deg=angle_min_deg,
            angle_speed_limit_kmh=angle_speed_limit_kmh,
            fps=fps,
        )

    def update(self, *, frame_bgr: Any, out: Dict[str, Any]) -> None:
        if not self._enabled:
            return
        t_s = float(out.get("timestamp_s", 0.0))
        if frame_bgr is None:
            return
        self._pre_buf.append((t_s, frame_bgr.copy()))
        while self._pre_buf and (t_s - float(self._pre_buf[0][0])) > self._pre_s:
            self._pre_buf.popleft()

        to_close: List[int] = []
        for tid, st in list(self._active.items()):
            end_ts = float(st["end_ts"])
            if t_s > float(st.get("last_written_ts", -1e9)):
                st["writer"].write(frame_bgr)
                st["last_written_ts"] = float(t_s)
            if t_s >= end_ts:
                to_close.append(int(tid))

        for tid in to_close:
            st = self._active.pop(int(tid), None)
            if st is not None:
                try:
                    st["writer"].release()
                except Exception:
                    pass

        tracks = out.get("tracks", []) or []
        for tr in tracks:
            tid = int(tr.get("track_id", 0))
            sp = tr.get("speed", {}) or {}
            turning = tr.get("turning", {}) or {}
            speed_kmh = float(sp.get("kmh", 0.0))
            angle_deg = float(turning.get("turn_angle_signed_deg", 0.0))
            angular_rate = float(turning.get("angular_rate_deg_s", 0.0))

            over_speed = False
            if self._speed_limit_kmh > 0.0 and speed_kmh >= self._speed_limit_kmh:
                over_speed = True

            over_angle_speed = False
            if self._angle_min_deg > 0.0 and self._angle_speed_limit_kmh > 0.0:
                if abs(angle_deg) >= self._angle_min_deg and speed_kmh >= self._angle_speed_limit_kmh:
                    over_angle_speed = True

            is_over = bool(over_speed or over_angle_speed)
            st_over = self._over_state.get(tid)
            if is_over:
                if st_over is None or (t_s - float(st_over.get("last_ts", t_s))) > float(self._max_gap_s):
                    st_over = {"first_ts": float(t_s), "last_ts": float(t_s), "count": 1}
                    self._over_state[tid] = st_over
                else:
                    st_over["last_ts"] = float(t_s)
                    st_over["count"] = int(st_over.get("count", 0)) + 1
            else:
                if st_over is not None:
                    del self._over_state[tid]
                continue

            if st_over is None:
                continue
            count = int(st_over.get("count", 0.0))
            duration = float(t_s) - float(st_over.get("first_ts", t_s))
            if count < int(self._min_consecutive_samples):
                continue
            if duration < float(self._min_duration_s):
                continue

            last_ts = self._last_event_ts.get(tid)
            if last_ts is not None and (t_s - float(last_ts)) < self._cooldown_s:
                continue
            if tid in self._active:
                continue

            clip_name = f"{self._camera_id}_track{tid}_{int(t_s * 1000.0)}.avi"
            clip_path = str(Path(self._output_dir) / clip_name)

            import cv2

            h, w = frame_bgr.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter(clip_path, fourcc, float(self._fps), (int(w), int(h)))
            if not writer.isOpened():
                writer.release()
                raise RuntimeError(f"Failed to open AVI writer: {clip_path}")

            for _ts0, f0 in list(self._pre_buf):
                writer.write(f0)

            self._active[tid] = {
                "writer": writer,
                "end_ts": float(t_s + self._post_s),
                "last_written_ts": float(t_s),
                "clip_path": clip_path,
            }
            self._last_event_ts[tid] = float(t_s)

            self._conn.execute(
                "INSERT INTO events (camera_id, track_id, timestamp_s, speed_kmh, angle_deg, angular_rate_deg_s, "
                "speed_limit_kmh, angle_min_deg, angle_speed_limit_kmh, clip_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    str(self._camera_id),
                    int(tid),
                    float(t_s),
                    float(speed_kmh),
                    float(angle_deg),
                    float(angular_rate),
                    float(self._speed_limit_kmh),
                    float(self._angle_min_deg),
                    float(self._angle_speed_limit_kmh),
                    str(clip_path),
                ),
            )
            self._conn.commit()

    def close(self) -> None:
        for st in list(self._active.values()):
            try:
                st["writer"].release()
            except Exception:
                pass
        self._active.clear()
        try:
            self._conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    args = _parse_args()
    raw_cfg: Dict[str, Any] = {}
    cfg: Optional[V5PipelineConfig] = None
    source: Optional[CameraSource] = None
    if str(args.config).strip() != "":
        try:
            cfg, source, raw_cfg = load_pipeline_config_from_file(str(args.config))
        except Exception as e:
            raise SystemExit(f"Config error: {e}")

    out_cfg = dict(raw_cfg.get("output", {}) or {})
    log_cfg = dict(out_cfg.get("logging", {}) or {})
    level_name = "DEBUG" if bool(args.verbose) else str(log_cfg.get("level", "INFO") or "INFO")
    handlers: List[Any] = []
    if bool(log_cfg.get("save_to_file", False)):
        file_path = str(log_cfg.get("file_path", "speed_estimation.log") or "speed_estimation.log")
        handlers.append(logging.FileHandler(file_path, encoding="utf-8"))
    handlers.append(logging.StreamHandler())
    logging.basicConfig(level=getattr(logging, level_name.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s", handlers=handlers, force=True)

    display_cfg = dict(out_cfg.get("display", {}) or {})
    window_name = str(args.window)
    if window_name == "v5 overlay" and str(display_cfg.get("window_name", "") or "").strip() != "":
        window_name = str(display_cfg.get("window_name"))

    draw_traj = (not bool(args.no_traj))
    if (not bool(args.no_traj)) and "draw_trajectory" in display_cfg:
        draw_traj = bool(display_cfg.get("draw_trajectory", True))

    draw_vector = (not bool(args.no_vector))
    if (not bool(args.no_vector)) and "draw_vector" in display_cfg:
        draw_vector = bool(display_cfg.get("draw_vector", True))

    traj_len = int(args.traj_len)
    if traj_len == 40 and "trajectory_length" in display_cfg:
        traj_len = int(display_cfg.get("trajectory_length", 40))

    export_cfg = dict(out_cfg.get("export", {}) or {}) if out_cfg else None
    event_cfg = dict(out_cfg.get("event_clipping", {}) or {}) if out_cfg else None

    if cfg is not None and source is not None:
        run_rtsp_overlay(
            rtsp_url=source,
            cfg=cfg,
            window_name=window_name,
            draw_traj=draw_traj,
            traj_len=traj_len,
            draw_vector=draw_vector,
            read_timeout_s=float(args.read_timeout_s) if float(args.read_timeout_s) != 2.0 else None,
            buffer_size=int(args.buffer_size) if int(args.buffer_size) != 1 else None,
            export_cfg=export_cfg,
            event_cfg=event_cfg,
        )
    else:
        run_rtsp_overlay(
            window_name=window_name,
            draw_traj=draw_traj,
            traj_len=traj_len,
            draw_vector=draw_vector,
            read_timeout_s=float(args.read_timeout_s) if float(args.read_timeout_s) != 2.0 else None,
            buffer_size=int(args.buffer_size) if int(args.buffer_size) != 1 else None,
            export_cfg=export_cfg,
            event_cfg=event_cfg,
        )
