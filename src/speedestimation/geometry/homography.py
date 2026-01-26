from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Homography:
    H_img_to_world: np.ndarray

    @staticmethod
    def load_npy(path: str) -> "Homography":
        H = np.load(path)
        if H.shape != (3, 3):
            raise ValueError(f"Expected homography 3x3, got {H.shape} from {path}")
        return Homography(H_img_to_world=H.astype(np.float64))

    def transform_point(self, xy_img: Tuple[float, float]) -> Tuple[float, float]:
        x, y = float(xy_img[0]), float(xy_img[1])
        p = np.asarray([x, y, 1.0], dtype=np.float64)
        w = self.H_img_to_world @ p
        if w[2] == 0.0:
            return (float("nan"), float("nan"))
        return (float(w[0] / w[2]), float(w[1] / w[2]))

    def transform_points(self, xy_img: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if len(xy_img) == 0:
            return []
        pts = np.asarray(xy_img, dtype=np.float64).reshape(-1, 2)
        ones = np.ones((pts.shape[0], 1), dtype=np.float64)
        p = np.concatenate([pts, ones], axis=1)
        w = (self.H_img_to_world @ p.T).T
        z = w[:, 2:3]
        z[z == 0.0] = np.nan
        xy = w[:, 0:2] / z
        return [(float(x), float(y)) for x, y in xy]

