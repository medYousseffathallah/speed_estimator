import numpy as np

from speedestimation.geometry.homography import Homography


def test_homography_transform_point_affine() -> None:
    H = np.array(
        [
            [2.0, 0.0, 10.0],
            [0.0, 3.0, -5.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    h = Homography(H_img_to_world=H)
    x, y = h.transform_point((1.0, 2.0))
    assert x == 12.0
    assert y == 1.0


def test_homography_transform_points_batch() -> None:
    H = np.array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    h = Homography(H_img_to_world=H)
    pts = [(0.0, 0.0), (3.0, 4.0)]
    out = h.transform_points(pts)
    assert out == [(1.0, 2.0), (4.0, 6.0)]

