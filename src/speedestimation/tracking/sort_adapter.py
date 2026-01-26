from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from speedestimation.tracking.base import Tracker, TrackerInput, TrackerOutput


@dataclass
class SortAdapter(Tracker):
    def __post_init__(self) -> None:
        raise RuntimeError("SORT adapter is not installed in this environment")

    def update(self, inp: TrackerInput) -> TrackerOutput:
        raise RuntimeError("SORT adapter is not installed in this environment")

