from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from speedestimation.tracking.base import Tracker, TrackerInput, TrackerOutput
from speedestimation.utils.types import BBoxXYXY, Detection, TrackState


def iou_xyxy(a: BBoxXYXY, b: BBoxXYXY) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    a_area = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    b_area = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = a_area + b_area - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def greedy_match(cost: np.ndarray, cost_threshold: float) -> List[Tuple[int, int]]:
    matches: List[Tuple[int, int]] = []
    if cost.size == 0:
        return matches
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    flat = [(float(cost[r, c]), r, c) for r in range(cost.shape[0]) for c in range(cost.shape[1])]
    flat.sort(key=lambda x: x[0])
    for v, r, c in flat:
        if v > cost_threshold:
            break
        if r in used_rows or c in used_cols:
            continue
        used_rows.add(r)
        used_cols.add(c)
        matches.append((r, c))
    return matches


@dataclass
class GreedyIoUTracker(Tracker):
    iou_threshold: float = 0.25
    max_age_frames: int = 30
    min_hits: int = 2
    class_whitelist: Optional[List[str]] = None

    def __post_init__(self) -> None:
        self._next_id = 1
        self._tracks: Dict[int, TrackState] = {}

    def update(self, inp: TrackerInput) -> TrackerOutput:
        dets = inp.detections
        if self.class_whitelist is not None:
            dets = [d for d in dets if d.class_name in self.class_whitelist]

        for t in self._tracks.values():
            t.age_frames += 1
            t.time_since_update += 1

        track_ids = list(self._tracks.keys())
        track_states = [self._tracks[tid] for tid in track_ids]

        if track_states and dets:
            cost = np.zeros((len(track_states), len(dets)), dtype=np.float32)
            for i, ts in enumerate(track_states):
                for j, d in enumerate(dets):
                    cost[i, j] = 1.0 - iou_xyxy(ts.bbox_xyxy, d.bbox_xyxy)
            matches = greedy_match(cost=cost, cost_threshold=1.0 - self.iou_threshold)
        else:
            matches = []

        matched_tracks = set()
        matched_dets = set()

        for ti, dj in matches:
            ts = track_states[ti]
            d = dets[dj]
            ts.bbox_xyxy = d.bbox_xyxy
            ts.score = d.score
            ts.class_id = d.class_id
            ts.class_name = d.class_name
            ts.time_since_update = 0
            ts.hits += 1
            matched_tracks.add(ts.track_id)
            matched_dets.add(dj)

        for dj, d in enumerate(dets):
            if dj in matched_dets:
                continue
            tid = self._next_id
            self._next_id += 1
            self._tracks[tid] = TrackState(
                track_id=tid,
                camera_id=inp.camera_id,
                class_id=d.class_id,
                class_name=d.class_name,
                bbox_xyxy=d.bbox_xyxy,
                score=d.score,
                age_frames=1,
                hits=1,
                time_since_update=0,
            )

        to_delete = [tid for tid, ts in self._tracks.items() if ts.time_since_update > self.max_age_frames]
        for tid in to_delete:
            del self._tracks[tid]

        out_tracks = [
            ts
            for ts in self._tracks.values()
            if ts.hits >= self.min_hits and ts.time_since_update == 0
        ]
        return TrackerOutput(camera_id=inp.camera_id, frame_index=inp.frame_index, timestamp_s=inp.timestamp_s, tracks=out_tracks)

