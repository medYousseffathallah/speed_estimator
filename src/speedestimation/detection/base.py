from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np

from speedestimation.utils.types import Detection


@dataclass(frozen=True)
class DetectorInput:
    camera_id: str
    frame_index: int
    timestamp_s: float
    image_bgr: np.ndarray


class Detector(Protocol):
    def detect(self, inp: DetectorInput) -> List[Detection]:
        ...


class DetectorFactory(Protocol):
    def create(self, backend: str, params: Dict[str, Any]) -> Detector:
        ...

