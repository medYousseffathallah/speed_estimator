from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from speedestimation.geometry.calibration import compute_homography
from speedestimation.utils.config import load_yaml, resolve_path


def _parse_world_xy(s: str) -> Tuple[float, float]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 2:
        raise ValueError("Expected: x_m,y_m")
    return float(parts[0]), float(parts[1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", required=True, help="Camera YAML path")
    ap.add_argument("--frame-index", type=int, default=0)
    ap.add_argument("--ransac-threshold-px", type=float, default=3.0)
    ap.add_argument("--out", default=None, help="Override output H.npy path")
    args = ap.parse_args()

    base_dir = os.getcwd()
    cam = load_yaml(resolve_path(args.camera, base_dir))
    cam_id = str(cam.get("camera_id", "cam"))
    src = cam.get("source", {})
    source_type = str(src.get("type", "file")).lower()
    raw_uri = str(src.get("uri", ""))
    uri = resolve_path(raw_uri, base_dir) if source_type == "file" else raw_uri
    cap = cv2.VideoCapture(uri)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open source: {uri}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.frame_index))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Failed to read frame for calibration")

    clicks_img_xy: List[Tuple[float, float]] = []
    world_xy_m: List[Tuple[float, float]] = []
    win = f"calibrate:{cam_id}"

    def _on_mouse(event, x, y, flags, param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        clicks_img_xy.append((float(x), float(y)))
        cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
        cv2.imshow(win, frame)
        w = _parse_world_xy(input(f"World XY (meters) for clicked point {len(clicks_img_xy)} (x_m,y_m): "))
        world_xy_m.append(w)

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.imshow(win, frame)
    cv2.setMouseCallback(win, _on_mouse)

    while True:
        k = cv2.waitKey(50) & 0xFF
        if k == ord("q"):
            break
        if k == ord("s") and len(clicks_img_xy) >= 4:
            calib = compute_homography(clicks_img_xy, world_xy_m, ransac_reproj_threshold_px=float(args.ransac_threshold_px))
            out_path = args.out
            if out_path is None:
                out_path = str(cam.get("calibration", {}).get("homography_npy", f"calibration/{cam_id}_H.npy"))
            out_path = resolve_path(out_path, base_dir)
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            np.save(out_path, calib.H_img_to_world)
            print(f"Saved H to: {out_path}")
            print(f"Inliers: {int(calib.inlier_mask.sum())}/{len(calib.inlier_mask)}")
            break

    cv2.destroyWindow(win)


if __name__ == "__main__":
    main()

