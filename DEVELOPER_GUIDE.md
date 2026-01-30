# Multi-Camera Vehicle Speed Estimation Pipeline - Developer Guide

## Overview

This document provides comprehensive documentation for developers working with the Multi-Camera Vehicle Speed Estimation pipeline. The system processes video feeds from multiple fixed cameras to detect, track, and estimate vehicle speeds with turn-aware speed limiting for smart-city deployments.

## System Architecture

### Core Components

The pipeline is built around a modular, configuration-driven architecture with the following key components:

1. **Multi-Camera Orchestration** (`src/speedestimation/pipeline/multi_camera.py`)
   - Loads per-camera YAML configurations from a directory
   - Runs each camera in its own worker thread
   - Optionally shares detector instances across cameras for efficiency

2. **Single Camera Pipeline** (`src/speedestimation/pipeline/camera_pipeline.py`)
   - Processes individual camera feeds frame-by-frame
   - Coordinates detection, tracking, speed estimation, and output generation
   - Handles ROI filtering and position smoothing

3. **Detection System** (`src/speedestimation/detection/`)
   - Pluggable backends supporting YOLO-family models
   - Registry pattern for easy backend swapping
   - Mock detector available for testing

4. **Tracking System** (`src/speedestimation/tracking/`)
   - Multiple tracker backends (IoU, SORT, ByteTrack)
   - Persistent track IDs across frames
   - Configurable tracking parameters

5. **Speed Estimation** (`src/speedestimation/speed_estimation/`)
   - Advanced speed calculation with turning detection
   - Turn-aware speed limiting based on lateral acceleration
   - Temporal smoothing using EMA and Kalman filtering

6. **Geometry Processing** (`src/speedestimation/geometry/`)
   - Homography-based perspective transformation
   - Image coordinates to real-world coordinates mapping
   - Calibration tools for camera setup

## Pipeline Execution Flow

### Frame Processing Sequence

Each frame follows this processing pipeline:

1. **Frame Ingestion**
   - Camera handler reads frame with timestamp
   - Optional resizing based on configuration
   - RTSP reconnection handling for network cameras

2. **Vehicle Detection**
   - Detector processes frame to find vehicles
   - Configurable confidence and IoU thresholds
   - Optional ROI filtering (pre-detection stage)

3. **Multi-Object Tracking**
   - Tracker updates with new detections
   - Maintains persistent track IDs
   - Handles track birth/death logic

4. **ROI Filtering** (if configured)
   - Post-tracking ROI filtering available
   - Polygon-based region of interest
   - Configurable filtering stage (pre/post tracking)

5. **Coordinate Transformation**
   - Bottom-center of bounding box calculated
   - Homography transformation to world coordinates (meters)
   - Fallback to pixel scale if no valid homography

6. **Speed Estimation**
   - Raw speed from displacement and time delta
   - Turning angle and curvature calculation
   - Turn-aware speed limiting applied
   - Acceleration limiting for realistic values

7. **Temporal Smoothing**
   - EMA smoothing for speed values
   - Optional Kalman filtering for position
   - Configurable smoothing parameters

8. **Output Generation**
   - CSV/JSONL structured data output
   - Overlay visualization with trails and vectors
   - Alert generation for speed violations

### Configuration Management

The system uses YAML configuration files for all runtime parameters:

#### Camera Configuration (`configs/cameras/*.yaml`)

```yaml
camera_id: "camera_name" # Unique identifier
source:
  type: "file" # or "rtsp"
  uri: "path/to/video.mp4" # File path or RTSP URL
  params: {} # Additional source parameters

calibration:
  homography_npy: "calibration/camera_H.npy" # Path to homography matrix

runtime:
  fps_hint: 30.0 # Expected frame rate
  resize:
    enabled: true # Enable frame resizing
    width: 1280 # Target width
    height: 720 # Target height

detection:
  interval: 5 # Detection frequency (frames)
  backend: "ultralytics_yolo" # Detection backend
  params:
    model_path: "yolov8n.pt" # Model file
    conf_threshold: 0.3 # Confidence threshold
    iou_threshold: 0.5 # IoU threshold
    device: "cpu" # or "cuda"
    class_whitelist: ["car", "truck"] # Vehicle classes

roi:
  enabled: false # Region of interest filtering
  polygon_xy: [] # Polygon vertices
  stage: "pre" # "pre" or "post" tracking

output:
  overlay:
    enabled: true
    show: false # Display window
    write_video: true # Save video with overlay
    video_path: "outputs/overlay.mp4"
    draw_vectors: true # Speed vectors
    trail_length: 30 # Trail history length

  csv:
    enabled: true
    path: "outputs/speeds.csv"

  jsonl:
    enabled: false
    path: "outputs/speeds.jsonl"

alerts:
  enabled: false # Speed alert system
```

#### Tracking Configuration (`configs/tracking.yaml`)

```yaml
backend: "bytetrack" # Tracker backend
params:
  iou_threshold: 0.5 # IoU matching threshold
  max_age_frames: 30 # Max frames without detection
  min_hits: 4 # Min detections for track
  high_conf_threshold: 0.7 # High confidence threshold
  low_conf_threshold: 0.3 # Low confidence threshold
  class_whitelist: ["car", "truck"] # Tracked classes
```

#### Speed Model Configuration (`configs/speed_model.yaml`)

```yaml
mode: "advanced" # Speed calculation mode
units:
  output: "kmh" # Output units

raw_speed:
  min_dt_s: 0.04 # Min time delta
  max_dt_s: 1.0 # Max time delta
  min_displacement_m: 0.15 # Min displacement
  min_speed_mps: 0.03 # Min speed threshold

# Dot-based tracking for turning detection
dots:
  min_distance_m: 0.45 # Min distance between dots
  min_dt_s: 0.3 # Min time between dots
  buffer_size: 5 # Dot history size

# Turning detection parameters
turning:
  max_turn_rate_deg_per_s: 300.0 # Max realistic turn rate
  theta_min_deg: 5.0 # Min angle for detection
  curvature_min_1pm: 0.005 # Min curvature
  min_arc_len_m: 0.1 # Min arc length
  persist_s: 0.25 # Turn persistence

# Turn-aware speed limiting
turn_speed_limit:
  enabled: true
  mode: "curvature" # Limiting mode
  a_lat_max_mps2: 2.5 # Max lateral acceleration
  v_max_kmh: 180.0 # Max speed limit
  v_min_kmh: 30.0 # Min speed limit
  angle_max_deg: 60.0 # Max turn angle
  alpha: 0.5 # Smoothing factor

# Acceleration limiting
accel_limit:
  enabled: true
  a_max_mps2: 60.0 # Max acceleration

# Speed smoothing
smoothing:
  method: "ema" # Smoothing method
  ema_alpha: 0.25 # EMA factor
  max_gap_s: 1.0 # Max gap for smoothing

# Position smoothing
position_smoothing:
  enabled: true
  method: "kalman" # Kalman filter
  window: 12 # Filter window
  poly_degree: 2 # Polynomial degree
  ema_alpha: 0.35 # EMA factor
  kalman_process_noise: 1.0 # Process noise
  kalman_measurement_noise: 2.0 # Measurement noise
```

## Key Algorithms

### Homography-Based Coordinate Transformation

The system uses perspective transformation to convert image coordinates to real-world coordinates:

1. **Calibration Process**: Interactive tool (`scripts/calibrate_camera.py`) collects corresponding points
2. **Transformation**: 3×3 homography matrix maps image points to world coordinates
3. **Validation**: Matrix validated against expected image dimensions
4. **Fallback**: Pixel-scale estimation if homography unavailable

### Turn-Aware Speed Limiting

The system implements sophisticated speed limiting based on vehicle turning behavior:

1. **Turning Detection**: Uses dot-based tracking to estimate turning angles and curvature
2. **Lateral Acceleration**: Calculates maximum safe speed based on turn radius
3. **Speed Limiting**: Applies limits to prevent unrealistic speeds during turns
4. **Smoothing**: Maintains smooth speed transitions during limit application

### Multi-Camera Coordination

The system supports efficient multi-camera processing:

1. **Thread Management**: Each camera runs in separate thread
2. **Shared Resources**: Optional shared detector instances
3. **Error Handling**: Individual camera failures don't affect others
4. **Configuration**: Per-camera configuration with shared global settings

## Development Guidelines

### Adding New Detection Backends

1. Implement `Detector` interface in `src/speedestimation/detection/`
2. Register in `detection/registry.py`
3. Add configuration parameters to camera YAML
4. Update requirements if new dependencies needed

### Adding New Tracking Backends

1. Implement `Tracker` interface in `src/speedestimation/tracking/`
2. Register in `tracking/registry.py`
3. Add configuration parameters to tracking YAML
4. Ensure proper track ID persistence

### Modifying Speed Estimation

1. Update `SpeedEstimator` class in `speed_estimation/estimator.py`
2. Modify configuration schema in `speed_model.yaml`
3. Update `SpeedEstimatorConfig` dataclass
4. Test with various turning scenarios

### Custom Output Formats

1. Implement new sink classes in `src/speedestimation/output/sinks.py`
2. Add configuration options to camera YAML
3. Update `SpeedSinks` dataclass
4. Handle file opening/closing properly

## Testing and Validation

### Unit Tests

- Located in `tests/` directory
- Run with pytest: `pytest tests/`
- Cover core algorithms and configuration loading

### Integration Tests

- Use `scripts/run_video_test.py` for single video testing
- Multi-camera testing with `scripts/run_pipeline.py`
- Validate outputs against ground truth data

### Performance Monitoring

- Frame processing time logging
- Memory usage monitoring
- GPU utilization tracking (if applicable)

## Troubleshooting

### Common Issues

1. **Missing Homography**
   - Warning logged, pixel scale used
   - Calibrate camera using `scripts/calibrate_camera.py`

2. **Overlay Not Displaying**
   - Check `output.overlay.show` in camera YAML
   - Ensure OpenCV GUI available
   - Press 'q' to exit window loop

3. **Video Not Saving**
   - Verify `output.overlay.write_video` enabled
   - Check `video_path` directory exists
   - Ensure sufficient disk space

4. **Detection Issues**
   - Adjust confidence thresholds
   - Check model file exists
   - Verify class whitelist matches model

5. **Tracking Problems**
   - Tune IoU thresholds
   - Adjust max age and min hits
   - Check detection frequency

### Performance Optimization

1. **GPU Acceleration**
   - Use CUDA-enabled detection backends
   - Configure device parameter in detection config

2. **Frame Skipping**
   - Increase detection interval
   - Use lower resolution inputs
   - Enable frame resizing

3. **Memory Management**
   - Monitor memory usage during long runs
   - Implement proper cleanup in custom components
   - Use shared detectors for multi-camera setups

## Deployment Considerations

### Jetson Nano/Orin Optimization

- Use system OpenCV instead of pip version
- Configure TensorRT for YOLO models
- Monitor thermal throttling
- Optimize power consumption

### Network Camera Setup

- Configure RTSP parameters in `camera_handling.yaml`
- Implement reconnection logic
- Handle network latency and packet loss
- Use appropriate transport protocols

### Production Deployment

- Implement proper logging and monitoring
- Set up alert systems for failures
- Configure appropriate output paths
- Implement backup and recovery procedures

## Code Organization

### Source Structure

```
src/speedestimation/
├── detection/          # Vehicle detection backends
├── tracking/           # Multi-object tracking
├── geometry/           # Coordinate transformations
├── speed_estimation/   # Speed calculation algorithms
├── turning_model/      # Turn detection and modeling
├── fusion/            # Multi-camera fusion (future)
├── io/                # Input/output handling
├── output/            # Results formatting and visualization
├── pipeline/          # Main pipeline orchestration
└── utils/             # Shared utilities and types
```

### Configuration Structure

```
configs/
├── cameras/           # Per-camera configurations
├── tracking.yaml      # Tracking backend settings
├── speed_model.yaml   # Speed estimation parameters
└── camera_handling.yaml # Input handling settings
```

### Scripts Structure

```
scripts/
├── calibrate_camera.py    # Interactive calibration tool
├── run_pipeline.py        # Multi-camera pipeline
├── run_video_test.py      # Single video testing
└── evaluate.py           # Performance evaluation
```

This guide provides the foundation for understanding and extending the speed estimation pipeline. For specific implementation details, refer to the source code and inline documentation.
