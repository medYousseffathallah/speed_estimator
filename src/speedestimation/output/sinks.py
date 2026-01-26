from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from speedestimation.utils.types import SpeedSample


def _ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


@dataclass
class CsvSink:
    path: str
    _f: Optional[object] = None
    _w: Optional[csv.DictWriter] = None

    def open(self) -> None:
        _ensure_parent(self.path)
        self._f = open(self.path, "w", newline="", encoding="utf-8")
        self._w = csv.DictWriter(
            self._f,
            fieldnames=[
                "camera_id",
                "track_id",
                "timestamp_s",
                "frame_index",
                "world_x_m",
                "world_y_m",
                "speed_mps_raw",
                "speed_mps_limited",
                "speed_mps_smoothed",
                "heading_deg",
                "turn_angle_deg",
                "curvature_1pm",
                "dt_s",
                "disp_m",
                "turn_applied",
            ],
        )
        self._w.writeheader()

    def write(self, s: SpeedSample) -> None:
        if self._w is None:
            raise RuntimeError("CsvSink not opened")
        self._w.writerow(
            {
                "camera_id": s.camera_id,
                "track_id": s.track_id,
                "timestamp_s": s.timestamp_s,
                "frame_index": s.frame_index,
                "world_x_m": s.world_xy_m[0],
                "world_y_m": s.world_xy_m[1],
                "speed_mps_raw": s.speed_mps_raw,
                "speed_mps_limited": s.speed_mps_limited,
                "speed_mps_smoothed": s.speed_mps_smoothed,
                "heading_deg": s.heading_deg,
                "turn_angle_deg": s.turn_angle_deg,
                "curvature_1pm": s.curvature_1pm,
                "dt_s": s.metadata.get("dt_s", float("nan")),
                "disp_m": s.metadata.get("disp_m", float("nan")),
                "turn_applied": s.metadata.get("turn_applied", 0.0),
            }
        )

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
        self._f = None
        self._w = None


@dataclass
class JsonlSink:
    path: str
    _f: Optional[object] = None

    def open(self) -> None:
        _ensure_parent(self.path)
        self._f = open(self.path, "w", encoding="utf-8")

    def write(self, s: SpeedSample) -> None:
        if self._f is None:
            raise RuntimeError("JsonlSink not opened")
        obj = {
            "camera_id": s.camera_id,
            "track_id": s.track_id,
            "timestamp_s": s.timestamp_s,
            "frame_index": s.frame_index,
            "world_xy_m": [s.world_xy_m[0], s.world_xy_m[1]],
            "speed_mps_raw": s.speed_mps_raw,
            "speed_mps_limited": s.speed_mps_limited,
            "speed_mps_smoothed": s.speed_mps_smoothed,
            "heading_deg": s.heading_deg,
            "turn_angle_deg": s.turn_angle_deg,
            "curvature_1pm": s.curvature_1pm,
            "metadata": s.metadata,
        }
        self._f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
        self._f = None


@dataclass
class SpeedSinks:
    csv: Optional[CsvSink]
    jsonl: Optional[JsonlSink]

    def open(self) -> None:
        if self.csv is not None:
            self.csv.open()
        if self.jsonl is not None:
            self.jsonl.open()

    def write(self, s: SpeedSample) -> None:
        if self.csv is not None:
            self.csv.write(s)
        if self.jsonl is not None:
            self.jsonl.write(s)

    def close(self) -> None:
        if self.csv is not None:
            self.csv.close()
        if self.jsonl is not None:
            self.jsonl.close()

