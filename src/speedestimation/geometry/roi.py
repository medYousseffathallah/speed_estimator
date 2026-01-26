from __future__ import annotations

from typing import List, Tuple


def point_in_polygon(x: float, y: float, polygon_xy: List[Tuple[float, float]]) -> bool:
    if len(polygon_xy) < 3:
        return True
    inside = False
    j = len(polygon_xy) - 1
    for i in range(len(polygon_xy)):
        xi, yi = polygon_xy[i]
        xj, yj = polygon_xy[j]
        intersects = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-12) + xi)
        if intersects:
            inside = not inside
        j = i
    return inside

