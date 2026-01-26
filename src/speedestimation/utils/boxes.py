from __future__ import annotations

from typing import Tuple

from speedestimation.utils.types import BBoxXYXY


def clip_bbox_xyxy(b: BBoxXYXY, width: int, height: int) -> BBoxXYXY:
    x1, y1, x2, y2 = b
    x1 = max(0.0, min(float(width - 1), float(x1)))
    y1 = max(0.0, min(float(height - 1), float(y1)))
    x2 = max(0.0, min(float(width - 1), float(x2)))
    y2 = max(0.0, min(float(height - 1), float(y2)))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return (x1, y1, x2, y2)

