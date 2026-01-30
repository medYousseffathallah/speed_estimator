from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Tuple

from speedestimation.utils.types import TrackState


PointXY = Tuple[float, float]
DotSample = Tuple[float, float, float, float, float]


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
            x1, y1, x2, y2 = ts.bbox_xyxy
            cx = 0.5 * (float(x1) + float(x2))
            cy = float(y2)
            dq.append((float(cx), float(cy)))
        self.prune_missing(alive)
        return {tid: list(dq) for tid, dq in self._by_track.items()}

    def prune_missing(self, alive_track_ids: List[int]) -> None:
        alive = set(int(x) for x in alive_track_ids)
        to_del = [tid for tid in self._by_track.keys() if tid not in alive]
        for tid in to_del:
            del self._by_track[tid]


@dataclass
class DotTrails:
    max_len: int
    min_distance_m: float
    min_dt_s: float
    _by_track: Dict[int, Deque[DotSample]]

    def __init__(self, max_len: int, min_distance_m: float, min_dt_s: float) -> None:
        self.max_len = max(2, int(max_len))
        self.min_distance_m = float(min_distance_m)
        self.min_dt_s = float(min_dt_s)
        self._by_track = {}

    def update(self, items: List[Tuple[int, Tuple[float, float], Tuple[float, float], float]]) -> Dict[int, List[PointXY]]:
        alive: List[int] = []
        for track_id, world_xy, image_xy, t_s in items:
            alive.append(int(track_id))
            dq = self._by_track.get(track_id)
            if dq is None:
                dq = deque(maxlen=self.max_len)
                self._by_track[track_id] = dq
                dq.append((float(world_xy[0]), float(world_xy[1]), float(t_s), float(image_xy[0]), float(image_xy[1])))
                continue

            last = dq[-1]
            if float(t_s) <= float(last[2]):
                continue
            dx = float(world_xy[0]) - float(last[0])
            dy = float(world_xy[1]) - float(last[1])
            dt = float(t_s) - float(last[2])
            dist = (dx * dx + dy * dy) ** 0.5
            if dist >= self.min_distance_m and dt >= self.min_dt_s:
                dq.append((float(world_xy[0]), float(world_xy[1]), float(t_s), float(image_xy[0]), float(image_xy[1])))

        self.prune_missing(alive)
        return {tid: [(d[3], d[4]) for d in dq] for tid, dq in self._by_track.items()}

    def prune_missing(self, alive_track_ids: List[int]) -> None:
        alive = set(int(x) for x in alive_track_ids)
        to_del = [tid for tid in self._by_track.keys() if tid not in alive]
        for tid in to_del:
            del self._by_track[tid]

