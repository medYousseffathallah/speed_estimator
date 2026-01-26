from __future__ import annotations

from typing import Any, Dict

from speedestimation.tracking.base import Tracker
from speedestimation.tracking.bytetrack_adapter import ByteTrackAdapter
from speedestimation.tracking.greedy_iou import GreedyIoUTracker
from speedestimation.tracking.sort_adapter import SortAdapter


def create_tracker(backend: str, params: Dict[str, Any]) -> Tracker:
    if backend == "greedy_iou":
        return GreedyIoUTracker(
            iou_threshold=float(params.get("iou_threshold", 0.25)),
            max_age_frames=int(params.get("max_age_frames", 30)),
            min_hits=int(params.get("min_hits", 2)),
            class_whitelist=params.get("class_whitelist"),
        )
    if backend == "sort":
        return SortAdapter()
    if backend == "bytetrack":
        return ByteTrackAdapter(
            iou_threshold=float(params.get("iou_threshold", 0.2)),
            max_age_frames=int(params.get("max_age_frames", 30)),
            min_hits=int(params.get("min_hits", 2)),
            high_conf_threshold=float(params.get("high_conf_threshold", 0.6)),
            low_conf_threshold=float(params.get("low_conf_threshold", 0.1)),
            class_whitelist=params.get("class_whitelist"),
        )
    raise ValueError(f"Unknown tracker backend: {backend}")

