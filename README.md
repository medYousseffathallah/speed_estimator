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
    cam_01.yaml
    cam_02.yaml
  tracking.yaml
  speed_model.yaml
calibration/
  cam_01_H.npy
  cam_02_H.npy
src/
  speedestimation/
    detection/
    tracking/
    geometry/
    speed_estimation/
    turning_model/
    fusion/
    utils/
scripts/
  calibrate_camera.py
  run_pipeline.py
  evaluate.py
outputs/
  logs/
  speeds/
  visualizations/
tests/
```

## Jetson Notes (Nano / Orin)

- Prefer system OpenCV on Jetson. If you already have `python3-opencv` installed, remove `opencv-python-headless` from requirements and use the system package.
- For YOLO on Jetson, run inference via TensorRT (recommended) or ONNX Runtime GPU. The codebase keeps detection behind an interface so you can swap backends per device.

## Quickstart

1) Install:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

2) Calibrate each camera (produces `calibration/<cam>_H.npy`):

```bash
python scripts/calibrate_camera.py --camera configs/cameras/cam_01.yaml
```

3) Run multi-camera pipeline:

```bash
python scripts/run_pipeline.py --cameras configs/cameras --tracking configs/tracking.yaml --speed configs/speed_model.yaml
```

## Configuration

- `configs/cameras/*.yaml` defines per-camera ingestion, homography path, and per-camera thresholds.
- `configs/tracking.yaml` defines tracker backend and parameters.
- `configs/speed_model.yaml` defines speed math parameters, turn limits, smoothing, and ablation flags.

