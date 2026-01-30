from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

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


@dataclass(frozen=True)
class FrameDetections:
    camera_id: str
    frame_index: int
    timestamp_s: float
    detections: List[Detection]


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
class CameraConfig:
    camera_id: str
    source_type: Literal["rtsp", "http", "file"]
    source_uri: str
    source_params: Dict[str, Any]
    homography_npy: str
    meters_per_pixel: float
    fps_hint: float
    resize_enabled: bool
    resize_width: int
    resize_height: int

    @staticmethod
    def from_dict(d: Dict) -> "CameraConfig":
        runtime = d.get("runtime", {})
        resize = runtime.get("resize", {})
        calibration = d.get("calibration", {})
        source = d.get("source", {})
        return CameraConfig(
            camera_id=str(d["camera_id"]),
            source_type=str(source.get("type", "file")),
            source_uri=str(source.get("uri", "")),
            source_params=dict(source.get("params", {}) or {}),
            homography_npy=str(calibration.get("homography_npy", "")),
            meters_per_pixel=float(calibration.get("meters_per_pixel", 0.5)),
            fps_hint=float(runtime.get("fps_hint", 30.0)),
            resize_enabled=bool(resize.get("enabled", False)),
            resize_width=int(resize.get("width", 1280)),
            resize_height=int(resize.get("height", 720)),
        )


def as_np_xy(points: List[Tuple[float, float]]) -> np.ndarray:
    return np.asarray(points, dtype=np.float32).reshape(-1, 2)

