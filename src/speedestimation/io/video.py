from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoReaderConfig:
    uri: str
    source_type: str
    fps_hint: float
    resize_enabled: bool
    resize_width: int
    resize_height: int


class VideoReader:
    def __init__(self, cfg: VideoReaderConfig) -> None:
        self._cfg = cfg
        self._cap = cv2.VideoCapture(cfg.uri)
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {cfg.uri}")
        try:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        self._t0_wall = time.monotonic()
        self._frame_index = 0
        self._fps = self._cap.get(cv2.CAP_PROP_FPS)
        if self._fps is None or self._fps <= 1e-3:
            self._fps = float(cfg.fps_hint)

    def __iter__(self) -> Iterator[Tuple[int, float, np.ndarray]]:
        while True:
            ok, frame = self._cap.read()
            if not ok:
                break
            if self._cfg.resize_enabled:
                frame = cv2.resize(frame, (self._cfg.resize_width, self._cfg.resize_height), interpolation=cv2.INTER_LINEAR)
            t_s = self._timestamp_s()
            fi = self._frame_index
            self._frame_index += 1
            yield fi, t_s, frame

    def _timestamp_s(self) -> float:
        pos_msec = self._cap.get(cv2.CAP_PROP_POS_MSEC)
        if pos_msec is not None and pos_msec > 0:
            return float(pos_msec) / 1000.0
        return float(time.monotonic() - self._t0_wall)

    def close(self) -> None:
        self._cap.release()

