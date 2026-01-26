from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from speedestimation.tracking.base import Tracker, TrackerInput, TrackerOutput
from speedestimation.tracking.greedy_iou import greedy_match, iou_xyxy
from speedestimation.utils.types import TrackState


@dataclass
class ByteTrackAdapter(Tracker):
    iou_threshold: float = 0.2
    max_age_frames: int = 30
    min_hits: int = 2
    high_conf_threshold: float = 0.6
    low_conf_threshold: float = 0.1
    class_whitelist: Optional[List[str]] = None

    def __post_init__(self) -> None:
        self._next_id = 1
        self._tracks: Dict[int, TrackState] = {}

    def update(self, inp: TrackerInput) -> TrackerOutput:
        dets = inp.detections
        if self.class_whitelist is not None:
            dets = [d for d in dets if d.class_name in self.class_whitelist]

        high_dets = [d for d in dets if d.score >= self.high_conf_threshold]
        low_dets = [d for d in dets if self.low_conf_threshold <= d.score < self.high_conf_threshold]

        for t in self._tracks.values():
            t.age_frames += 1
            t.time_since_update += 1

        track_ids = list(self._tracks.keys())
        track_states = [self._tracks[tid] for tid in track_ids]

        matched_tracks: set[int] = set()
        matched_high_dets: set[int] = set()

        if track_states and high_dets:
            cost = np.zeros((len(track_states), len(high_dets)), dtype=np.float32)
            for i, ts in enumerate(track_states):
                for j, d in enumerate(high_dets):
                    cost[i, j] = 1.0 - iou_xyxy(ts.bbox_xyxy, d.bbox_xyxy)
            matches = greedy_match(cost=cost, cost_threshold=1.0 - self.iou_threshold)
        else:
            matches = []

        for ti, dj in matches:
            ts = track_states[ti]
            d = high_dets[dj]
            ts.bbox_xyxy = d.bbox_xyxy
            ts.score = d.score
            ts.class_id = d.class_id
            ts.class_name = d.class_name
            ts.time_since_update = 0
            ts.hits += 1
            matched_tracks.add(ts.track_id)
            matched_high_dets.add(dj)

        unmatched_tracks = [ts for ts in track_states if ts.track_id not in matched_tracks]
        matched_low_dets: set[int] = set()

        if unmatched_tracks and low_dets:
            cost_low = np.zeros((len(unmatched_tracks), len(low_dets)), dtype=np.float32)
            for i, ts in enumerate(unmatched_tracks):
                for j, d in enumerate(low_dets):
                    cost_low[i, j] = 1.0 - iou_xyxy(ts.bbox_xyxy, d.bbox_xyxy)
            matches_low = greedy_match(cost=cost_low, cost_threshold=1.0 - self.iou_threshold)
        else:
            matches_low = []

        for ti, dj in matches_low:
            ts = unmatched_tracks[ti]
            d = low_dets[dj]
            ts.bbox_xyxy = d.bbox_xyxy
            ts.score = d.score
            ts.class_id = d.class_id
            ts.class_name = d.class_name
            ts.time_since_update = 0
            ts.hits += 1
            matched_tracks.add(ts.track_id)
            matched_low_dets.add(dj)

        for dj, d in enumerate(high_dets):
            if dj in matched_high_dets:
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

