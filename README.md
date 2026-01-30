# Multi-Camera Vehicle Speed Estimation (Jetson Nano / Orin)

This repository provides a modular, configuration-driven pipeline for estimating vehicle speed and direction from multiple fixed cameras, with per-camera calibration, optional multi-camera fusion, and turn-aware speed limiting for smart-city deployments.

## Highlights

- Pluggable object detection backends (YOLO-family adapters included)
- Pluggable multi-object tracking (built-in IoU tracker; adapters for SORT/ByteTrack)
- Centroid-based motion in image and world coordinates
- Perspective transform via per-camera homography (image → world)
- Turning angle/curvature estimation and turn-aware speed limiting
- Temporal smoothing and structured outputs (JSON/CSV) plus overlay visualization

## Project Layout

```
configs/
  cameras/
    gttet.yaml
    test_normanniles.yaml
  tracking.yaml
  speed_model.yaml
  camera_handling.yaml
calibration/
  gttet_H.npy
  normanniles_H.npy
docs/
  pipeline_architecture.md
src/
  speedestimation/
    detection/
    tracking/
    geometry/
    speed_estimation/
    turning_model/
    fusion/
    io/
    output/
    utils/
scripts/
  calibrate_camera.py
  run_pipeline.py
  run_video_test.py
  evaluate.py
outputs/
  logs/
  speeds/
  visualizations/
tests/
```

## File and Folder Explanations

- configs/: All runtime configuration in YAML.
- configs/cameras/: Per-camera configuration files for standard multi-camera runs.
- configs/tracking.yaml: Default tracking backend and parameters.
- configs/speed_model.yaml: Vehicle speed estimation parameters.
- configs/camera_handling.yaml: Camera input/RTSP handling options.
- calibration/: Saved homography matrices (image → world) per camera.
- docs/pipeline_architecture.md: Full pipeline architecture and step-by-step guide.
- src/speedestimation/: Core library code.
- scripts/: CLI entry points for calibration, pipelines, and evaluation.
- outputs/: Generated logs, speed data, and visualizations.
- tests/: Unit and integration tests.

## Source Package Overview

- detection/: Detector interfaces, registry, and YOLO adapters.
- tracking/: Tracker interfaces and adapters (IoU, SORT, ByteTrack).
- geometry/: Calibration, homography, and ROI helpers.
- io/: Camera ingestion and video utilities.
- speed_estimation/: Speed math, smoothing, limits, and units.
- turning_model/: Turn angle and curvature estimation.
- output/: Overlays, trails, alerts, and output sinks.
- pipeline/: Single and multi-camera orchestration.
- fusion/: Multi-camera fusion interfaces.
- utils/: Shared utilities (config loading, logging, geometry helpers, types).

## Jetson Notes (Nano / Orin)

- Prefer system OpenCV on Jetson. If you already have `python3-opencv` installed, remove `opencv-python-headless` from requirements and use the system package.
- For YOLO on Jetson, run inference via TensorRT (recommended) or ONNX Runtime GPU. The codebase keeps detection behind an interface so you can swap backends per device.

## Quickstart

1. Install:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

2. Calibrate each camera (produces `calibration/<cam>_H.npy`):

```bash
python scripts/calibrate_camera.py --camera configs/cameras/gttet.yaml
```

3. Run multi-camera pipeline:

```bash
python scripts/run_pipeline.py --cameras configs/cameras --tracking configs/tracking.yaml --speed configs/speed_model.yaml
```

## Configuration

- `configs/cameras/*.yaml` defines per-camera ingestion, homography path, and per-camera thresholds.
- `configs/tracking.yaml` defines tracker backend and parameters.
- `configs/speed_model.yaml` defines speed math parameters, turn limits, smoothing, and ablation flags.
- `configs/camera_handling.yaml` defines RTSP, buffering, and read settings.

## Scripts

- calibrate_camera.py: Create homography from a camera YAML file.
- run_pipeline.py: Multi-camera pipeline from a folder of camera YAMLs.
- run_video_test.py: Single video run with explicit homography.
- evaluate.py: Offline evaluation helpers.
