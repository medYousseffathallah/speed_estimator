from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from speedestimation.detection.base import Detector, DetectorInput
from speedestimation.utils.types import Detection


@dataclass
class MockDetector(Detector):
    class_names: List[str]

    def detect(self, inp: DetectorInput) -> List[Detection]:
        _ = inp
        return []

