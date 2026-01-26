from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class HomographyCalibration:
    H_img_to_world: np.ndarray
    inlier_mask: np.ndarray


def compute_homography(
    img_xy: Sequence[Tuple[float, float]],
    world_xy_m: Sequence[Tuple[float, float]],
    ransac_reproj_threshold_px: float = 3.0,
) -> HomographyCalibration:
    if len(img_xy) != len(world_xy_m):
        raise ValueError("img_xy and world_xy_m must have same length")
    if len(img_xy) < 4:
        raise ValueError("Need at least 4 correspondences for homography")

    import cv2

    src = np.asarray(img_xy, dtype=np.float64).reshape(-1, 1, 2)
    dst = np.asarray(world_xy_m, dtype=np.float64).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=ransac_reproj_threshold_px)
    if H is None or mask is None:
        raise RuntimeError("cv2.findHomography failed")
    return HomographyCalibration(H_img_to_world=H.astype(np.float64), inlier_mask=mask.astype(np.uint8).reshape(-1))

