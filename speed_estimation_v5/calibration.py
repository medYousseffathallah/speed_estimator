from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

MatrixLike = Union[np.ndarray, Sequence[Sequence[float]]]


def _as_homography_matrix(value: MatrixLike) -> np.ndarray:
    H = np.asarray(value, dtype=np.float64)
    if H.shape != (3, 3):
        raise ValueError(f"Expected homography 3x3, got {H.shape}")
    return H


def _resolve_path(path: str, base_dir: Optional[str]) -> str:
    p = str(path)
    if base_dir is None or base_dir == "":
        return p
    if os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(str(base_dir), p))


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


def load_homography_matrix_from_config(cfg: Dict[str, Any], base_dir: Optional[str] = None) -> Optional[np.ndarray]:
    """
    Load a 3x3 image->world homography from a config dict.

    Supported inputs:
    - cfg['homography_npy']: path to a .npy file storing a 3x3 matrix
    - cfg['homography_matrix']: nested list/ndarray representing a 3x3 matrix

    The returned matrix maps homogeneous image points (x_px, y_px, 1) to world
    coordinates (x_m, y_m, w), where (x_m/w, y_m/w) are meters.
    """
    npy_path = str(cfg.get("homography_npy", "") or "").strip()
    if npy_path:
        p = _resolve_path(npy_path, base_dir)
        return _as_homography_matrix(np.load(p))

    mat = cfg.get("homography_matrix")
    if mat is not None:
        return _as_homography_matrix(mat)

    return None


@dataclass(frozen=True)
class WorldMapping:
    """
    Pixel→world coordinate mapping.

    Math:
      Let p = [x_px, y_px, 1]^T.
      Let w = H * p.
      Then (x_m, y_m) = (w0/w2, w1/w2).

    If homography is unavailable, a fallback pixel scale can be used:
      (x_m, y_m) = (x_px * meters_per_pixel, y_px * meters_per_pixel).
    """

    homography: Optional[Homography]
    meters_per_pixel: float = 1.0

    @staticmethod
    def from_config(cfg: Dict[str, Any], base_dir: Optional[str] = None) -> "WorldMapping":
        calibration_cfg: Dict[str, Any] = dict(cfg.get("calibration", cfg) or {})
        H = load_homography_matrix_from_config(calibration_cfg, base_dir=base_dir)
        homography = Homography(H_img_to_world=H) if H is not None else None
        meters_per_pixel = float(calibration_cfg.get("meters_per_pixel", 1.0))
        meters_per_pixel = max(1e-12, meters_per_pixel)
        return WorldMapping(homography=homography, meters_per_pixel=meters_per_pixel)

    def pixel_to_world(self, xy_px: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        if self.homography is not None:
            x_m, y_m = self.homography.transform_point(xy_px)
            if x_m != x_m or y_m != y_m:
                return None
            return (float(x_m), float(y_m))
        x_px, y_px = float(xy_px[0]), float(xy_px[1])
        return (x_px * self.meters_per_pixel, y_px * self.meters_per_pixel)

    def pixels_to_world(self, xy_px: Sequence[Tuple[float, float]]) -> List[Optional[Tuple[float, float]]]:
        if len(xy_px) == 0:
            return []
        if self.homography is not None:
            out: List[Optional[Tuple[float, float]]] = []
            for x_m, y_m in self.homography.transform_points(xy_px):
                if x_m != x_m or y_m != y_m:
                    out.append(None)
                else:
                    out.append((float(x_m), float(y_m)))
            return out
        return [self.pixel_to_world(p) for p in xy_px]


def compute_homography_calibration(
    img_xy_px: Sequence[Tuple[float, float]],
    world_xy_m: Sequence[Tuple[float, float]],
    ransac_reproj_threshold_px: float = 3.0,
) -> "HomographyCalibration":
    """
    Compute an image->world homography from point correspondences.

    This matches the legacy behavior: OpenCV RANSAC homography fitting with the
    same inputs and threshold.
    """
    if len(img_xy_px) != len(world_xy_m):
        raise ValueError("img_xy_px and world_xy_m must have same length")
    if len(img_xy_px) < 4:
        raise ValueError("Need at least 4 correspondences for homography")

    try:
        import cv2  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("compute_homography_calibration requires OpenCV (cv2) to be installed") from e

    src = np.asarray(img_xy_px, dtype=np.float64).reshape(-1, 1, 2)
    dst = np.asarray(world_xy_m, dtype=np.float64).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=float(ransac_reproj_threshold_px))
    if H is None or mask is None:
        raise RuntimeError("cv2.findHomography failed")
    return HomographyCalibration(H_img_to_world=H.astype(np.float64), inlier_mask=mask.astype(np.uint8).reshape(-1))


@dataclass(frozen=True)
class HomographyCalibration:
    H_img_to_world: np.ndarray
    inlier_mask: np.ndarray


def world_mapping_from_homography(
    H_img_to_world: MatrixLike, *, meters_per_pixel_fallback: float = 1.0
) -> WorldMapping:
    """
    Construct a WorldMapping directly from a 3x3 homography matrix.
    """
    return WorldMapping(
        homography=Homography(H_img_to_world=_as_homography_matrix(H_img_to_world)),
        meters_per_pixel=max(1e-12, float(meters_per_pixel_fallback)),
    )
