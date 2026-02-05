from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


JsonDetection = Dict[str, Any]


def _resolve_path(path: str, base_dir: Optional[str]) -> str:
    p = str(path)
    if base_dir is None or base_dir == "":
        return p
    if os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(str(base_dir), p))


def _as_int_list(values: Any) -> Optional[List[int]]:
    if values is None:
        return None
    if isinstance(values, (list, tuple)):
        out: List[int] = []
        for v in values:
            out.append(int(v))
        return out
    return [int(values)]


class YoloV8Detector:
    def __init__(
        self,
        *,
        model_path: str = "yolov8n.pt",
        base_dir: Optional[str] = None,
        conf: float = 0.25,
        iou: float = 0.7,
        device: Optional[str] = None,
        classes: Optional[Sequence[int]] = None,
        class_whitelist: Optional[Sequence[str]] = None,
    ) -> None:
        self._model_path = _resolve_path(model_path, base_dir)
        self._conf = float(conf)
        self._iou = float(iou)
        self._device = None if device is None or str(device).strip() == "" else str(device)
        self._classes = _as_int_list(classes)
        self._class_whitelist = None if class_whitelist is None else {str(x).strip().lower() for x in class_whitelist if str(x).strip() != ""}
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("Ultralytics is required for YOLOv8 detection (pip install ultralytics)") from e
        self._model = YOLO(self._model_path)

    def detect(self, frame_bgr: np.ndarray) -> List[JsonDetection]:
        self._ensure_loaded()
        if frame_bgr is None:
            raise ValueError("frame_bgr is required for YOLO inference")
        if not isinstance(frame_bgr, np.ndarray):
            raise TypeError("frame_bgr must be a numpy ndarray")

        kwargs: Dict[str, Any] = {
            "verbose": False,
            "conf": float(self._conf),
            "iou": float(self._iou),
        }
        if self._device is not None:
            kwargs["device"] = str(self._device)
        if self._classes is not None:
            kwargs["classes"] = list(self._classes)

        results = self._model.predict(source=frame_bgr, **kwargs)
        if not results:
            return []
        r0 = results[0]
        boxes = getattr(r0, "boxes", None)
        if boxes is None:
            return []

        xyxy = boxes.xyxy
        conf = boxes.conf
        cls = boxes.cls

        xyxy_np = xyxy.detach().cpu().numpy() if hasattr(xyxy, "detach") else np.asarray(xyxy)
        conf_np = conf.detach().cpu().numpy() if hasattr(conf, "detach") else np.asarray(conf)
        cls_np = cls.detach().cpu().numpy() if hasattr(cls, "detach") else np.asarray(cls)

        names: Dict[int, str] = {}
        model_names = getattr(getattr(self._model, "model", None), "names", None)
        if isinstance(model_names, dict):
            names = {int(k): str(v) for k, v in model_names.items()}
        elif isinstance(model_names, (list, tuple)):
            names = {int(i): str(n) for i, n in enumerate(model_names)}

        dets: List[Tuple[float, JsonDetection]] = []
        for i in range(int(xyxy_np.shape[0])):
            x1, y1, x2, y2 = [float(v) for v in xyxy_np[i].tolist()]
            score = float(conf_np[i])
            class_id = int(cls_np[i])
            class_name = str(names.get(class_id, ""))
            if self._class_whitelist is not None:
                if str(class_name).strip().lower() not in self._class_whitelist:
                    continue
            d: JsonDetection = {
                "bbox_xyxy": (x1, y1, x2, y2),
                "score": float(score),
                "class_id": int(class_id),
                "class_name": str(class_name),
            }
            dets.append((score, d))

        dets.sort(key=lambda x: (-float(x[0]), float(x[1]["bbox_xyxy"][0]), float(x[1]["bbox_xyxy"][1])))
        return [d for _s, d in dets]
