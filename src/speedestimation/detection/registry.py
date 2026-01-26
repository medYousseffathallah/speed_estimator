from __future__ import annotations

from typing import Any, Dict, List, Optional

from speedestimation.detection.base import Detector
from speedestimation.detection.mock import MockDetector


def create_detector(backend: str, params: Dict[str, Any]) -> Detector:
    if backend == "mock":
        class_names = params.get("class_names", ["car", "truck", "bus", "motorcycle"])
        if not isinstance(class_names, list):
            raise ValueError("class_names must be a list")
        return MockDetector(class_names=[str(x) for x in class_names])

    if backend == "ultralytics_yolo":
        from speedestimation.detection.yolo_ultralytics import UltralyticsYoloDetector

        model_path = str(params["model_path"])
        conf_threshold = float(params.get("conf_threshold", 0.25))
        iou_threshold = float(params.get("iou_threshold", 0.5))
        device = params.get("device")
        class_whitelist = params.get("class_whitelist")
        if class_whitelist is not None and not isinstance(class_whitelist, list):
            raise ValueError("class_whitelist must be a list when provided")
        return UltralyticsYoloDetector(
            model_path=model_path,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            device=str(device) if device is not None else None,
            class_whitelist=[str(x) for x in class_whitelist] if class_whitelist is not None else None,
        )

    raise ValueError(f"Unknown detector backend: {backend}")

