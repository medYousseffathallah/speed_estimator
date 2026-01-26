from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Protocol

from speedestimation.utils.types import SpeedSample


class FusionEngine(Protocol):
    def fuse(self, samples_by_camera: Dict[str, List[SpeedSample]]) -> List[SpeedSample]:
        ...


@dataclass
class NoFusionEngine(FusionEngine):
    def fuse(self, samples_by_camera: Dict[str, List[SpeedSample]]) -> List[SpeedSample]:
        out: List[SpeedSample] = []
        for s in samples_by_camera.values():
            out.extend(s)
        return out

