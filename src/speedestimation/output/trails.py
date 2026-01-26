from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Tuple

from speedestimation.utils.types import TrackState


PointXY = Tuple[float, float]


@dataclass
class TrackTrails:
    max_len: int
    _by_track: Dict[int, Deque[PointXY]]

    def __init__(self, max_len: int) -> None:
        self.max_len = max(2, int(max_len))
        self._by_track = {}

    def update(self, tracks: List[TrackState]) -> Dict[int, List[PointXY]]:
        alive: List[int] = []
        for ts in tracks:
            alive.append(int(ts.track_id))
            dq = self._by_track.get(ts.track_id)
            if dq is None:
                dq = deque(maxlen=self.max_len)
                self._by_track[ts.track_id] = dq
            dq.append((float(ts.centroid_xy()[0]), float(ts.centroid_xy()[1])))
        self.prune_missing(alive)
        return {tid: list(dq) for tid, dq in self._by_track.items()}

    def prune_missing(self, alive_track_ids: List[int]) -> None:
        alive = set(int(x) for x in alive_track_ids)
        to_del = [tid for tid in self._by_track.keys() if tid not in alive]
        for tid in to_del:
            del self._by_track[tid]

