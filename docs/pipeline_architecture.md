# Pipeline Architecture & Step‑By‑Step Guide

This document explains the pipeline architecture and how to run, configure, and extend it. Use it as a guide when adding cameras, changing backends, or adjusting speed estimation logic.

## Overview

- Multi‑camera orchestration loads per‑camera YAML configs and runs each camera in its own worker.
- Each camera pipeline ingests frames, detects vehicles, tracks them, maps image points to world coordinates via homography, computes speed, renders overlays, and writes outputs.
- The system is configuration‑driven: most behavior is set in YAML files.

## Core Components

- Multi‑camera runner: [run_pipeline.py](file:///c:/Users/youss/OneDrive/Desktop/youssef/speedestimation/scripts/run_pipeline.py), [multi_camera.py](file:///c:/Users/youss/OneDrive/Desktop/youssef/speedestimation/src/speedestimation/pipeline/multi_camera.py#L38-L106)
  - Reads a directory of camera YAMLs
  - Loads shared configs (tracking, speed model, camera handling)
  - Optionally shares a detector instance across cameras
- Single camera pipeline: [camera_pipeline.py](file:///c:/Users/youss/OneDrive/Desktop/youssef/speedestimation/src/speedestimation/pipeline/camera_pipeline.py#L60-L216)
  - Ingestion via [CameraHandler](file:///c:/Users/youss/OneDrive/Desktop/youssef/speedestimation/src/speedestimation/io/camera_handler.py)
  - Detection via backend registry (YOLO adapters or mock)
  - Tracking via tracker registry (IoU, SORT/ByteTrack adapters)
  - Geometry via homography: image bottom‑center → world XY
  - Speed estimation with turn‑aware limits and smoothing: [estimator.py](file:///c:/Users/youss/OneDrive/Desktop/youssef/speedestimation/src/speedestimation/speed_estimation/estimator.py)
  - Overlay rendering: [overlay.py](file:///c:/Users/youss/OneDrive/Desktop/youssef/speedestimation/src/speedestimation/output/overlay.py)
  - Structured outputs (CSV/JSONL): [sinks.py](file:///c:/Users/youss/OneDrive/Desktop/youssef/speedestimation/src/speedestimation/output/sinks.py)
- Configuration utilities: [utils](file:///c:/Users/youss/OneDrive/Desktop/youssef/speedestimation/src/speedestimation/utils/__init__.py)

## Configuration Files

- Camera YAML (one per camera)
  - Example: [gttet.yaml](file:///c:/Users/youss/OneDrive/Desktop/youssef/speedestimation/configs/cameras/gttet.yaml)
  - Key fields:
    - `camera_id`: unique name
    - `source.type` and `source.uri`: ingestion source (typically `file` and path)
    - `calibration.homography_npy`: path to 3×3 `.npy` homography for image→world
    - `runtime.resize`: optional resize settings
    - `detection`: backend and thresholds
    - `roi`: optional polygon and stage (`pre`/`post`)
    - `output.overlay`: enable/show/write and overlay options
    - `output.csv`/`output.jsonl`: structured output paths
- Tracking config: [tracking.yaml](file:///c:/Users/youss/OneDrive/Desktop/youssef/speedestimation/configs/tracking.yaml)
- Speed model config: [speed_model.yaml](file:///c:/Users/youss/OneDrive/Desktop/youssef/speedestimation/configs/speed_model.yaml)
- Camera handling: [camera_handling.yaml](file:///c:/Users/youss/OneDrive/Desktop/youssef/speedestimation/configs/camera_handling.yaml)

## Execution Flow (per frame)

1. Read frame and timestamp from the camera handler.
2. Detect vehicles; optionally filter detections by ROI (pre stage).
3. Update the tracker to obtain persistent track IDs and bounding boxes.
4. Optionally filter tracks by ROI (post stage).
5. Compute bottom‑center of each bbox; map to world XY:
   - If valid homography present, transform to meters.
   - Otherwise, fall back to pixel scale using `meters_per_pixel`.
6. Update speed estimator:
   - Raw speed from displacement and time delta
   - Turn limit and acceleration limit
   - Temporal smoothing (EMA)
7. Write samples to CSV/JSONL; update alerts and notifier if enabled.
8. Draw overlay with bboxes, trails, vectors, and speed labels; show and/or write video.

## Calibrating Homography

- Goal: produce a per‑camera 3×3 matrix `calibration/<CAMERA_ID>_H.npy` mapping image points to world coordinates (meters).
- Command (replace placeholders):

```bash
python scripts/calibrate_camera.py --camera <CAMERA_YAML>
```

- Example:

```bash
python scripts/calibrate_camera.py --camera configs/cameras/gttet.yaml
```

## Running the Pipeline

- Multi‑camera from a folder of YAMLs:

```bash
python scripts/run_pipeline.py --cameras <CAMERA_DIR> --tracking configs/tracking.yaml --speed configs/speed_model.yaml --camera-handling configs/camera_handling.yaml --log-level INFO
```

- Examples:

```bash
python scripts/run_pipeline.py --cameras configs/cameras
python scripts/run_pipeline.py --cameras configs/cameras_test_single
python scripts/run_pipeline.py --cameras configs/cameras_gttet
```

- Single video test with explicit homography and simple camera config created on the fly:

```bash
python scripts/run_video_test.py --video <VIDEO> --camera-id <CAMERA_ID> --homography <HOMOGRAPHY_NPY> --tracking configs/tracking.yaml --speed configs/speed_model.yaml --camera-handling configs/camera_handling.yaml --log-level INFO
```

## Human Speed Testing (RTSP) — Isolated Workflow

Use a dedicated branch and config folder so human‑tracking experiments do not impact vehicle tracking defaults.

### Proposed Isolation Layout

- Branch: `human-speed-tests`
- Config folder: `configs/cameras_human_test/`
- Tracking config: `configs/tracking_human.yaml`
- Optional speed tuning: `configs/speed_model_human.yaml`
- Outputs: `outputs/human_test/<camera_id>/...`

### Human RTSP Camera YAML (key fields)

- `source.type: "rtsp"`
- `source.uri: "rtsp://<user>:<pass>@<ip>:<port>/<path>"`
- `runtime.fps_hint`: set to expected stream FPS
- `detection.params.class_whitelist: ["person"]`
- `output.overlay.show: true` while validating connectivity
- `output.csv.path` / `output.jsonl.path` under `outputs/human_test/...`

### Human Tracking Config (key fields)

- `backend`: keep current tracker unless needed to switch
- `params.class_whitelist: ["person"]`
- Adjust thresholds for lower speeds and smaller targets if needed

### Human Speed Model (optional)

- Lower `max_speed_kmh` and acceleration limits for pedestrians
- Increase smoothing if jittery; decrease if responsiveness is needed

### RTSP Connectivity Validation

1. Start with `source.type: "rtsp"`, `source.uri` from the phone app.
2. If frames drop, enable `rtsp.use_gstreamer: true` in `configs/camera_handling.yaml` or override per‑camera params.
3. Use `output.overlay.show: true` for a quick visual confirmation.

### Run Command (human test folder)

```bash
python scripts/run_pipeline.py --cameras configs/cameras_human_test --tracking configs/tracking_human.yaml --speed configs/speed_model_human.yaml --camera-handling configs/camera_handling.yaml --log-level INFO
```

## Tasks to Finish Human Tracking Without Affecting Main Project

1. Create a new branch and isolate configs under `configs/cameras_human_test/`.
2. Add a human‑focused tracking config with `class_whitelist: ["person"]`.
3. Add or tune a human speed model (limits and smoothing).
4. Add one RTSP camera YAML with correct URI and outputs under `outputs/human_test/`.
5. Validate RTSP connectivity with overlay enabled and confirm detections.
6. Calibrate homography or set `meters_per_pixel` for provisional speed scale.
7. Run the multi‑camera pipeline using the human configs and review CSV/JSONL output.

## Adding a New Camera

1. Copy an existing camera YAML into `configs/cameras` and update fields:
   - `camera_id`, `source.uri`, `calibration.homography_npy`, `output.*` paths
2. Calibrate and save the homography to `calibration/<camera_id>_H.npy`.
3. Run the pipeline pointing to the directory that contains your new YAML.

## Changing Backends or Parameters

- Detection:
  - Change `detection.backend` and tuning params (e.g., YOLO thresholds)
- Tracking:
  - Adjust thresholds in `configs/tracking.yaml` or swap backend
- Speed model:
  - Tweak smoothing, turn limits, and acceleration limits in `configs/speed_model.yaml`
- Overlay:
  - Configure `output.overlay` in the camera YAML to enable showing or writing, and to adjust visuals

## Outputs

- CSV fields include camera_id, track_id, time, world XY, raw/limited/smoothed speed, heading, turn angle, curvature, and metadata. See [sinks.py](file:///c:/Users/youss/OneDrive/Desktop/youssef/speedestimation/src/speedestimation/output/sinks.py#L23-L71).
- Overlay MP4 is written when `write_video` is true; see writer in [camera_pipeline.py](file:///c:/Users/youss/OneDrive/Desktop/youssef/speedestimation/src/speedestimation/pipeline/camera_pipeline.py#L183-L197).

## Troubleshooting

- Missing homography:
  - The pipeline logs a warning and uses pixel scale (`meters_per_pixel`) until a valid homography is provided.
- Overlay not showing:
  - Set `output.overlay.show: true` and press `q` to exit the window loop.
- Overlay not written:
  - Ensure `output.overlay.write_video: true` and `video_path` points to a writable location.
- YOLO model not found:
  - Ensure `detection.params.model_path` exists or install `ultralytics` via the `yolo` extra if needed.
