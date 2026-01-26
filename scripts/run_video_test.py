from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from speedestimation.pipeline.camera_pipeline import CameraPipeline, CameraPipelineConfig
from speedestimation.utils.config import load_yaml, resolve_path
from speedestimation.utils.logging import setup_logging


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to video file")
    ap.add_argument("--camera-id", default="video_test")
    ap.add_argument("--homography", default="", help="Optional homography .npy path")
    ap.add_argument("--tracking", default="configs/tracking.yaml")
    ap.add_argument("--speed", default="configs/speed_model.yaml")
    ap.add_argument("--camera-handling", default="configs/camera_handling.yaml")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    base_dir = os.getcwd()
    setup_logging(level=args.log_level, log_file=None)

    tracking = load_yaml(resolve_path(args.tracking, base_dir))
    speed = load_yaml(resolve_path(args.speed, base_dir))
    camera_handling = load_yaml(resolve_path(args.camera_handling, base_dir))

    cam_cfg = {
        "camera_id": str(args.camera_id),
        "source": {"type": "file", "uri": str(args.video), "params": {}},
        "runtime": {"fps_hint": 30.0, "resize": {"enabled": False}},
        "calibration": {"homography_npy": str(args.homography)},
        "output": {
            "csv": {"enabled": True, "path": "outputs/speeds/video_test.csv"},
            "jsonl": {"enabled": False, "path": "outputs/speeds/video_test.jsonl"},
            "overlay": {
                "enabled": True,
                "show": False,
                "write_video": True,
                "video_path": "outputs/overlays/video_test.mp4",
                "draw_vectors": True,
                "trail_length": 30,
                "trail_color_bgr": [0, 255, 255],
                "vector_color_bgr": [0, 0, 255],
                "trail_thickness": 2,
                "vector_thickness": 2,
                "vector_tip_length": 0.3,
            },
        },
        "alerts": {"enabled": False, "notifier": {"type": "log"}},
    }

    p = CameraPipeline(CameraPipelineConfig(camera=cam_cfg, camera_handling=camera_handling, tracking=tracking, speed=speed, base_dir=base_dir))
    p.run()


if __name__ == "__main__":
    main()

