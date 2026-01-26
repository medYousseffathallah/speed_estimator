from __future__ import annotations

import glob
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from speedestimation.pipeline.camera_pipeline import CameraPipeline, CameraPipelineConfig
from speedestimation.detection.base import Detector, DetectorInput
from speedestimation.detection.registry import create_detector
from speedestimation.utils.config import load_yaml


logger = logging.getLogger("speedestimation.pipeline.multi")


@dataclass
class _LockedDetector(Detector):
    detector: Detector

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    def detect(self, inp: DetectorInput):
        with self._lock:
            return self.detector.detect(inp)


@dataclass(frozen=True)
class MultiCameraRunnerConfig:
    cameras_dir: str
    camera_handling_yaml: str
    tracking_yaml: str
    speed_yaml: str
    base_dir: str


class MultiCameraRunner:
    def __init__(self, cfg: MultiCameraRunnerConfig) -> None:
        self._cfg = cfg

    def run(self) -> None:
        camera_paths = sorted(glob.glob(str(Path(self._cfg.cameras_dir) / "*.yaml")))
        if not camera_paths:
            raise RuntimeError(f"No camera configs found in: {self._cfg.cameras_dir}")

        tracking = load_yaml(self._cfg.tracking_yaml)
        speed = load_yaml(self._cfg.speed_yaml)
        camera_handling = load_yaml(self._cfg.camera_handling_yaml)
        cam_cfgs = [(p, load_yaml(p)) for p in camera_paths]

        shared_detector: Optional[Detector] = None
        shared_enabled = True
        shared_backend: Optional[str] = None
        shared_params: Optional[Dict[str, Any]] = None
        for _, cam_cfg in cam_cfgs:
            det_cfg = cam_cfg.get("detection", {})
            if not bool(det_cfg.get("shared", False)):
                shared_enabled = False
                break
            backend = str(det_cfg.get("backend", "mock"))
            params = dict(det_cfg.get("params", {}) or {})
            if shared_backend is None:
                shared_backend = backend
                shared_params = params
            elif backend != shared_backend or params != shared_params:
                shared_enabled = False
                break

        if shared_enabled and shared_backend is not None and shared_params is not None:
            shared_detector = _LockedDetector(create_detector(shared_backend, shared_params))

        threads: List[threading.Thread] = []
        errors: List[BaseException] = []

        def _worker(cam_path: str, cam_cfg: Dict[str, Any]) -> None:
            try:
                p = CameraPipeline(
                    CameraPipelineConfig(
                        camera=cam_cfg,
                        camera_handling=camera_handling,
                        tracking=tracking,
                        speed=speed,
                        base_dir=self._cfg.base_dir,
                        detector=shared_detector,
                    )
                )
                p.run()
            except BaseException as e:
                logger.exception("Camera pipeline failed: %s", cam_path)
                errors.append(e)

        for cam_path, cam_cfg in cam_cfgs:
            t = threading.Thread(target=_worker, args=(cam_path, cam_cfg), daemon=True)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        if errors:
            logger.error(f"{len(errors)} camera pipeline(s) failed, but others may have succeeded.")
            # raise RuntimeError(f"{len(errors)} camera pipeline(s) failed") from errors[0]

