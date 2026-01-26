from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from speedestimation.pipeline.multi_camera import MultiCameraRunner, MultiCameraRunnerConfig
from speedestimation.utils.config import resolve_path
from speedestimation.utils.logging import setup_logging


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cameras", required=True, help="Directory containing camera YAML files")
    ap.add_argument("--camera-handling", default="configs/camera_handling.yaml", help="Camera handling YAML")
    ap.add_argument("--tracking", required=True, help="Tracking YAML")
    ap.add_argument("--speed", required=True, help="Speed model YAML")
    ap.add_argument("--log-level", default="INFO")
    ap.add_argument("--log-file", default=None)
    args = ap.parse_args()

    base_dir = os.getcwd()
    setup_logging(level=args.log_level, log_file=args.log_file)

    runner = MultiCameraRunner(
        MultiCameraRunnerConfig(
            cameras_dir=resolve_path(args.cameras, base_dir),
            camera_handling_yaml=resolve_path(args.camera_handling, base_dir),
            tracking_yaml=resolve_path(args.tracking, base_dir),
            speed_yaml=resolve_path(args.speed, base_dir),
            base_dir=base_dir,
        )
    )
    runner.run()


if __name__ == "__main__":
    main()

