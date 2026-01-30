from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2

from speedestimation.detection.base import Detector, DetectorInput
from speedestimation.detection.registry import create_detector
from speedestimation.geometry.homography import Homography
from speedestimation.geometry.roi import point_in_polygon
from speedestimation.io.camera_handler import CameraHandler, CameraHandlerConfig
from speedestimation.output.alerts import SpeedAlertConfig, SpeedAlertEngine
from speedestimation.output.overlay import OverlayRenderer
from speedestimation.output.notifier import Notifier, create_notifier
from speedestimation.output.sinks import CsvSink, JsonlSink, SpeedSinks
from speedestimation.output.trails import DotTrails
from speedestimation.utils.types import SpeedSample
from speedestimation.speed_estimation.estimator import SpeedEstimator, SpeedEstimatorConfig
from speedestimation.speed_estimation.smoothing import EmaSmoother, PolySmoother
from speedestimation.tracking.base import TrackerInput
from speedestimation.tracking.registry import create_tracker
from speedestimation.utils.config import resolve_path
from speedestimation.utils.types import CameraConfig, Track


logger = logging.getLogger("speedestimation.pipeline.camera")


@dataclass(frozen=True)
class CameraPipelineConfig:
    camera: Dict[str, Any]
    camera_handling: Dict[str, Any]
    tracking: Dict[str, Any]
    speed: Dict[str, Any]
    base_dir: str
    detector: Optional[Detector] = None


class CameraPipeline:
    def __init__(self, cfg: CameraPipelineConfig) -> None:
        self._cfg = cfg
        self._camera = CameraConfig.from_dict(cfg.camera)
        self._homography = self._load_homography()
        self._meters_per_pixel = max(1e-9, float(self._camera.meters_per_pixel))
        self._use_pixel_scale = self._homography is None or not self._is_homography_valid(
            self._camera.resize_width, self._camera.resize_height
        )
        if self._use_pixel_scale:
            self._homography = None

        det_cfg = cfg.camera.get("detection", {})
        self._detector = cfg.detector or create_detector(str(det_cfg.get("backend", "mock")), dict(det_cfg.get("params", {})))

        tr_cfg = cfg.tracking
        self._tracker = create_tracker(str(tr_cfg.get("backend", "greedy_iou")), dict(tr_cfg.get("params", {})))

        self._speed_cfg = SpeedEstimatorConfig.from_dict(cfg.speed)
        self._estimator = SpeedEstimator(self._speed_cfg)

        ps_cfg = dict(cfg.speed.get("position_smoothing", {}) or {})
        self._pos_smoothing_method = str(ps_cfg.get("method", "")).lower()
        self._pos_smoothing_enabled = bool(ps_cfg.get("enabled", True)) and self._pos_smoothing_method in {"ema", "poly"}
        self._pos_smoothing_alpha = float(ps_cfg.get("ema_alpha", self._speed_cfg.smoothing_alpha))
        self._pos_smoothing_max_gap_s = float(ps_cfg.get("max_gap_s", self._speed_cfg.smoothing_max_gap_s))
        self._pos_smoothing_window = int(ps_cfg.get("window", 8))
        self._pos_smoothing_poly_degree = int(ps_cfg.get("poly_degree", 2))
        self._pos_smoothers: Dict[int, Tuple[Any, Any]] = {}

        out_cfg = cfg.camera.get("output", {})
        csv_cfg = out_cfg.get("csv", {})
        jsonl_cfg = out_cfg.get("jsonl", {})
        self._sinks = SpeedSinks(
            csv=CsvSink(resolve_path(str(csv_cfg.get("path")), cfg.base_dir)) if bool(csv_cfg.get("enabled", False)) else None,
            jsonl=JsonlSink(resolve_path(str(jsonl_cfg.get("path")), cfg.base_dir)) if bool(jsonl_cfg.get("enabled", False)) else None,
        )

        self._overlay_cfg = out_cfg.get("overlay", {})
        self._overlay_renderer = OverlayRenderer(units=str(cfg.speed.get("units", {}).get("output", "kmh")).lower())
        self._latest_speed_by_track: Dict[int, SpeedSample] = {}
        self._dot_trails = DotTrails(
            max_len=int(self._speed_cfg.dot_buffer_size),
            min_distance_m=float(self._speed_cfg.dot_min_distance_m),
            min_dt_s=float(self._speed_cfg.dot_min_dt_s),
        )

        alerts_cfg = cfg.camera.get("alerts", {}) or {}
        self._alerts = SpeedAlertEngine(SpeedAlertConfig.from_dict(dict(alerts_cfg)))
        notifier_cfg = dict(alerts_cfg.get("notifier", {}) or {})
        self._notifier: Notifier = create_notifier(notifier_cfg) if bool(alerts_cfg.get("enabled", False)) else create_notifier({"type": "log", "level": "INFO"})

        roi_cfg = cfg.camera.get("roi", {})
        self._roi_enabled = bool(roi_cfg.get("enabled", False))
        self._roi_polygon = [(float(x), float(y)) for x, y in (roi_cfg.get("polygon_xy") or [])]
        roi_stage = str(roi_cfg.get("stage", "pre")).lower()
        self._roi_stage = roi_stage if roi_stage in {"pre", "post"} else "pre"

        self._video_writer: Optional[cv2.VideoWriter] = None

        self._camera_uri = self._resolve_camera_uri()

    def run(self) -> None:
        handler_cfg = CameraHandlerConfig.from_sources(
            camera_source_type=self._camera.source_type,
            camera_uri=self._camera_uri,
            fps_hint=self._camera.fps_hint,
            resize_enabled=self._camera.resize_enabled,
            resize_width=self._camera.resize_width,
            resize_height=self._camera.resize_height,
            global_cfg=dict(self._cfg.camera_handling),
            per_camera_params=dict(self._camera.source_params),
        )
        handler = CameraHandler(handler_cfg)
        self._sinks.open()
        try:
            for frame_index, t_s, frame_bgr in handler:
                dets = self._detector.detect(
                    DetectorInput(camera_id=self._camera.camera_id, frame_index=frame_index, timestamp_s=t_s, image_bgr=frame_bgr)
                )
                if self._roi_enabled and self._roi_polygon and self._roi_stage == "pre":
                    dets = [d for d in dets if point_in_polygon(d.centroid_xy[0], d.centroid_xy[1], self._roi_polygon)]

                tr_out = self._tracker.update(
                    TrackerInput(camera_id=self._camera.camera_id, frame_index=frame_index, timestamp_s=t_s, detections=dets)
                )

                output_tracks = tr_out.tracks
                if self._roi_enabled and self._roi_polygon and self._roi_stage == "post":
                    filtered = []
                    for ts in tr_out.tracks:
                        x1, y1, x2, y2 = ts.bbox_xyxy
                        cx, cy = (0.5 * (x1 + x2), y2)
                        if point_in_polygon(cx, cy, self._roi_polygon):
                            filtered.append(ts)
                    output_tracks = filtered
                self._prune_pos_smoothers([ts.track_id for ts in output_tracks])
                self._prune_latest_speeds([ts.track_id for ts in output_tracks])  # CRITICAL FIX: Remove stale speed cache

                tracks: List[Track] = []
                dot_inputs: List[Tuple[int, Tuple[float, float], Tuple[float, float], float]] = []
                for ts in output_tracks:
                    world_xy = None
                    x1, y1, x2, y2 = ts.bbox_xyxy
                    cx, cy = (0.5 * (x1 + x2), y2)
                    if self._pos_smoothing_enabled:
                        sm = self._pos_smoothers.get(ts.track_id)
                        if sm is None:
                            if self._pos_smoothing_method == "ema":
                                sm = (
                                    EmaSmoother(alpha=self._pos_smoothing_alpha, max_gap_s=self._pos_smoothing_max_gap_s),
                                    EmaSmoother(alpha=self._pos_smoothing_alpha, max_gap_s=self._pos_smoothing_max_gap_s),
                                )
                            elif self._pos_smoothing_method == "poly":
                                sm = (
                                    PolySmoother(degree=self._pos_smoothing_poly_degree, window=self._pos_smoothing_window),
                                    PolySmoother(degree=self._pos_smoothing_poly_degree, window=self._pos_smoothing_window),
                                )
                            else:
                                sm = (
                                    KalmanSmoother(process_noise=self._pos_smoothing_kalman_process_noise, measurement_noise=self._pos_smoothing_kalman_measurement_noise),
                                    KalmanSmoother(process_noise=self._pos_smoothing_kalman_process_noise, measurement_noise=self._pos_smoothing_kalman_measurement_noise),
                                )
                            self._pos_smoothers[ts.track_id] = sm
                        
                        measurement_x = float(cx)
                        measurement_y = float(cy)
                        if not is_detection_frame and self._pos_smoothing_method == "kalman":
                            measurement_x = None
                            measurement_y = None
                            
                        cx = sm[0].update(measurement_x, float(t_s))
                        cy = sm[1].update(measurement_y, float(t_s))
                    if self._homography is not None:
                        wx, wy = self._homography.transform_point((cx, cy))
                        if wx != wx or wy != wy:
                            continue
                        world_xy = (wx, wy)
                    elif self._use_pixel_scale:
                        world_xy = (cx * self._meters_per_pixel, cy * self._meters_per_pixel)
                    tracks.append(Track(camera_id=self._camera.camera_id, frame_index=frame_index, timestamp_s=t_s, state=ts, world_xy_m=world_xy))
                    if world_xy is not None:
                        dot_inputs.append((int(ts.track_id), (float(world_xy[0]), float(world_xy[1])), (float(cx), float(cy)), float(t_s)))

                samples = self._estimator.update(tracks)
                self._estimator.prune_missing([(self._camera.camera_id, ts.track_id) for ts in output_tracks])
                for s in samples:
                    self._latest_speed_by_track[s.track_id] = s
                    self._sinks.write(s)

                if samples:
                    events = self._alerts.update(samples)
                    for e in events:
                        self._notifier.notify_speed(e)
                    self._alerts.prune_missing([ts.track_id for ts in output_tracks])

                if bool(self._overlay_cfg.get("enabled", False)):
                    trails_by_track = self._dot_trails.update(dot_inputs)
                    overlay = self._overlay_renderer.draw(
                        frame_bgr,
                        output_tracks,
                        self._latest_speed_by_track,
                        trails_by_track=trails_by_track,
                        draw_vectors=bool(self._overlay_cfg.get("draw_vectors", False)),
                        show_heading=bool(self._overlay_cfg.get("show_heading", False)),
                        show_turn_angle=bool(self._overlay_cfg.get("show_turn_angle", False)),
                        trail_color_bgr=self._parse_color(self._overlay_cfg.get("trail_color_bgr"), (0, 255, 255)),
                        vector_color_bgr=self._parse_color(self._overlay_cfg.get("vector_color_bgr"), (0, 0, 255)),
                        trail_thickness=int(self._overlay_cfg.get("trail_thickness", 2)),
                        vector_thickness=int(self._overlay_cfg.get("vector_thickness", 2)),
                        vector_tip_length=float(self._overlay_cfg.get("vector_tip_length", 0.3)),
                    )
                    if bool(self._overlay_cfg.get("show", False)):
                        cv2.imshow(self._camera.camera_id, overlay)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                    if bool(self._overlay_cfg.get("write_video", False)):
                        self._write_overlay_frame(overlay, fps=self._camera.fps_hint)
        finally:
            handler.close()
            self._close_overlay_writer()
            self._sinks.close()
            try:
                cv2.destroyWindow(self._camera.camera_id)
            except Exception:
                pass
            self._pos_smoothers.clear()

    def _write_overlay_frame(self, frame_bgr, fps: float) -> None:
        path = resolve_path(str(self._overlay_cfg.get("video_path", "")), self._cfg.base_dir)
        if not path:
            return
        if self._video_writer is None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            h, w = frame_bgr.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._video_writer = cv2.VideoWriter(path, fourcc, float(fps), (w, h))
            if not self._video_writer.isOpened():
                self._video_writer = None
                raise RuntimeError(f"Failed to open video writer: {path}")
        self._video_writer.write(frame_bgr)

    def _close_overlay_writer(self) -> None:
        if self._video_writer is not None:
            self._video_writer.release()
        self._video_writer = None

    def _prune_pos_smoothers(self, alive_track_ids: List[int]) -> None:
        if not self._pos_smoothers:
            return
        alive = set(int(x) for x in alive_track_ids)
        to_del = [tid for tid in self._pos_smoothers.keys() if tid not in alive]
        for tid in to_del:
            del self._pos_smoothers[tid]

    def _prune_latest_speeds(self, alive_track_ids: List[int]) -> None:
        """CRITICAL FIX: Remove stale speed cache entries for tracks that are no longer active."""
        if not self._latest_speed_by_track:
            return
        alive = set(int(x) for x in alive_track_ids)
        to_del = [tid for tid in self._latest_speed_by_track.keys() if tid not in alive]
        for tid in to_del:
            del self._latest_speed_by_track[tid]
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("pruned stale speed cache: track_id=%d", tid)

    def _resolve_camera_uri(self) -> str:
        st = str(self._camera.source_type).lower()
        if st == "file":
            return resolve_path(str(self._camera.source_uri), self._cfg.base_dir)
        return str(self._camera.source_uri)

    def _load_homography(self) -> Optional[Homography]:
        path = str(self._camera.homography_npy or "").strip()
        if not path:
            return None
        try:
            return Homography.load_npy(resolve_path(path, self._cfg.base_dir))
        except FileNotFoundError:
            logger.warning("Homography file not found: %s. Speed estimation will be disabled for camera %s.", path, self._camera.camera_id)
            return None

    def _is_homography_valid(self, width: int, height: int) -> bool:
        if self._homography is None:
            return False
        w = max(1, int(width))
        h = max(1, int(height))
        p1 = (w * 0.25, h * 0.25)
        p2 = (w * 0.75, h * 0.25)
        p3 = (w * 0.25, h * 0.75)
        w1 = self._homography.transform_point(p1)
        w2 = self._homography.transform_point(p2)
        w3 = self._homography.transform_point(p3)
        if any(v != v for v in (w1[0], w1[1], w2[0], w2[1], w3[0], w3[1])):
            return False
        d12 = math.hypot(w1[0] - w2[0], w1[1] - w2[1])
        d13 = math.hypot(w1[0] - w3[0], w1[1] - w3[1])
        return d12 > 1e-6 and d13 > 1e-6

    def _parse_color(self, v: Any, default: Tuple[int, int, int]) -> Tuple[int, int, int]:
        if isinstance(v, (list, tuple)) and len(v) == 3:
            try:
                return (int(v[0]), int(v[1]), int(v[2]))
            except Exception:
                return default
        return default

