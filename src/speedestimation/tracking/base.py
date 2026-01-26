from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Protocol

from speedestimation.utils.types import Detection, TrackState


@dataclass(frozen=True)
class TrackerInput:
    camera_id: str
    frame_index: int
    timestamp_s: float
    detections: List[Detection]


@dataclass(frozen=True)
class TrackerOutput:
    camera_id: str
    frame_index: int
    timestamp_s: float
    tracks: List[TrackState]


class Tracker(Protocol):
    def update(self, inp: TrackerInput) -> TrackerOutput:
        ...


class TrackerFactory(Protocol):
    def create(self, backend: str, params: Dict[str, Any]) -> Tracker:
        ...

