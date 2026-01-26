from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from speedestimation.detection.base import Detector, DetectorInput
from speedestimation.utils.types import Detection


@dataclass
class UltralyticsYoloDetector(Detector):
    model_path: str
    conf_threshold: float = 0.25
    iou_threshold: float = 0.5
    device: Optional[str] = None
    class_whitelist: Optional[List[str]] = None

    def __post_init__(self) -> None:
        from ultralytics import YOLO

        self._model = YOLO(self.model_path)

    def detect(self, inp: DetectorInput) -> List[Detection]:
        results = self._model.predict(
            source=inp.image_bgr,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []
        r0 = results[0]
        names = r0.names if hasattr(r0, "names") else {}
        dets: List[Detection] = []
        boxes = getattr(r0, "boxes", None)
        if boxes is None:
            return dets
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), s, c in zip(xyxy, conf, cls):
            class_name = str(names.get(int(c), str(int(c))))
            if self.class_whitelist is not None and class_name not in self.class_whitelist:
                continue
            dets.append(
                Detection(
                    bbox_xyxy=(float(x1), float(y1), float(x2), float(y2)),
                    score=float(s),
                    class_id=int(c),
                    class_name=class_name,
                )
            )
        return dets

