# Unified Configuration Guide (V5 Speed Estimation)

This project supports a unified JSON configuration format for the V5 speed-estimation runner at [pipeline.py](file:///c:/Users/youss/OneDrive/Desktop/youssef/speedestimation_version2/speed_estimation_v5/pipeline.py).

The V5 runner accepts:

- The unified JSON schema described in this guide (recommended)
- The legacy V5 JSON schema (top-level `rtsp_url`, `camera_id`, `speed`, ...)
- Environment-variable overrides (`V5_*`) on top of either schema

## Running

Unified config:

```bash
python speed_estimation_v5/pipeline.py --config config_examples/basic_setup.json
```

Legacy config (still supported):

```bash
python speed_estimation_v5/pipeline.py --config speed_estimation_v5/v5_rtsp_config.json
```

## Configuration File Structure

Top-level keys:

- `camera`
- `detection`
- `calibration`
- `tracking`
- `speed_estimation`
- `runtime`
- `output`

Example configs are provided under [config_examples](file:///c:/Users/youss/OneDrive/Desktop/youssef/speedestimation_version2/config_examples).

## When To Change Each Section

### Camera Source (`camera`)

Change this when switching input sources.

- `camera.id`: Identifies the camera in outputs/logs.
- `camera.type`:
  - `rtsp`: Live stream URL
  - `video_file`: Local path to a file
  - `webcam`: Integer index (e.g., `0`)
- `camera.source`:
  - `rtsp`: `rtsp://...`
  - `video_file`: path like `data_test/gttet.mp4`
  - `webcam`: integer index

### Calibration (`calibration`)

Change this whenever the camera position/angle changes, or when moving to a new location.

- `method`:
  - `homography`: preferred for real-world scale correctness
  - `meters_per_pixel`: quick fallback if you do not have a homography
- `homography_npy` (required when `method=homography`): path to a 3×3 `.npy` image→world homography
- `meters_per_pixel` (required when `method=meters_per_pixel`): scalar pixel→meter scale
- `reference_points` (optional): documentation/verification points

How to determine `meters_per_pixel`:

1. Pick two points on the ground plane that are visible in the image.
2. Measure their real distance in meters: `D_m`.
3. Measure pixel distance between their image coordinates: `D_px`.
4. Set `meters_per_pixel = D_m / D_px`.

### Detection (`detection`)

Change this to adjust detection accuracy, supported classes, or compute device.

- `enabled`: set `false` only if you provide detections externally (V5 runner currently runs its own YOLO inference when enabled).
- `model_path`: path or model name (e.g., `yolov8n.pt`). If you specify a path (like `models/yolov8n.pt`), it must exist.
- `conf`: lower → more detections, more false positives
- `iou`: higher → stricter NMS merging
- `classes`: numeric class IDs to keep (COCO IDs for YOLO defaults)
- `class_whitelist`: optional name filter (useful when labels are present)
- `device`: `cpu` or `cuda:0`

Performance vs accuracy:

- Lower `conf` improves recall but increases tracker noise.
- Larger models improve precision but reduce FPS.

### Tracking (`tracking`)

Change this when tracks are unstable (ID switches, flicker) or when you want a different tracker backend.

- `backend`:
  - `simple_iou`: light-weight, no extra dependencies
  - `greedy_iou`, `sort`, `bytetrack`: supported via the main package backends
- `params`:
  - `iou_threshold`: higher → fewer associations, more track fragmentation
  - `max_missing_frames`: how long to keep a track without a match
  - `max_age_frames`: same idea for some tracker implementations

### Speed Estimation (`speed_estimation`)

Change this based on expected speed ranges, road geometry, and the responsiveness vs stability you want.

#### Mode

- `mode: advanced` (recommended): uses dot gating + constraints + smoothing
- `mode: simple`: window-based speed (more responsive, typically noisier)

#### Simple Mode (`simple_mode`)

- `window_s`: larger → smoother but slower updates
- `axis`:
  - `xy`: true planar speed magnitude
  - `x`/`y`: use when motion is mostly aligned to an axis
- `method`:
  - `mean`: averages segment speeds
  - `displacement`: uses total displacement over window

#### Advanced Mode (`advanced_mode`)

- `min_dt_s`/`max_dt_s`: gating on time delta between dots
- `min_displacement_m`: filters micro-jitter
- `stop_time_s` and `min_arc_length_m`: affect stop/zero-velocity detection sensitivity

#### Turning (`turning`)

Enable for intersections/roundabouts, disable for mostly-straight highways.

- `enabled`: turning detection on/off
- `theta_min_deg`: minimum turning angle to treat as a real turn
- `curvature_min_1pm`: minimum curvature threshold (1/m)
- `turning_persist_s`: how long to keep turn state
- `max_turn_rate_deg_per_s`: physical realism cap

#### Limits (`limits`)

Turning speed limit:

- `a_lat_max_mps2`: lower → stricter turn speed cap
- `v_max_kmh`: absolute maximum output speed
- `v_min_kmh`: minimum speed where turn limiting is applied
- `mode`: `curvature` is recommended
- `alpha`: blending factor between raw and capped speeds

Acceleration limit:

- `a_max_mps2`: lower → fewer unrealistic speed jumps, but more lag

#### Smoothing (`smoothing`)

- `method: ema`
- `ema_alpha`: higher → more responsive, lower → smoother
- `max_gap_s`: resets/softens smoothing when data gaps are large

#### Trajectory Dots (`trajectory_dots`)

This directly controls how frequently new dots are accepted into the dot buffer.

- `min_distance_m`:
  - Typical roads: 0.5–1.0 m
  - Highways: 1.0–2.0 m
- `min_dt_s`:
  - Typical: 0.2–0.6 s
- `buffer_size`:
  - Larger → more stable turning metrics but more latency

Trade-offs:

- Smaller `min_distance_m` → more frequent updates, noisier speeds.
- Larger `buffer_size` → smoother turns/speeds, more delay.

### Runtime (`runtime`)

Change this to match your camera/video characteristics.

- `fps_hint`: set to the actual FPS if known; affects timestamping when pacing is used
- `resize.enabled`: enable if you want to reduce compute cost
- `buffer_size`: OpenCV capture buffer size (best-effort)
- `read_timeout_s`: how long before treating a stream as stalled
- `reconnect_delay_s`: sleep before reconnecting

### Output (`output`)

#### Display (`output.display`)

- `window_name`: OpenCV window name
- `draw_trajectory`: show dot trail
- `trajectory_length`: number of drawn points
- `draw_vector`: draw the last motion arrow

#### Logging (`output.logging`)

- `level`: `INFO`/`DEBUG`/...
- `save_to_file`: whether to write logs to a file
- `file_path`: log file path
- `format`: `text` or `json`
- `rotate_mb`: rotate the log file after this many MB (0 disables rotation)
- `backup_count`: number of rotated files to keep

#### Runtime Logging (`runtime.logging`)

Optional periodic telemetry (disabled by default):

- `enabled`: enable periodic runtime summaries
- `stats_interval_s`: seconds between `frame_stats` summaries
- `drops_interval_s`: seconds between `frame_drop_summary` summaries

#### Export (`output.export`)

If enabled, the V5 runner writes per-frame outputs:

- `format: json` → JSON Lines (`tracks.jsonl`)
- `format: csv` → CSV (`tracks.csv`)
- `output_dir`: directory to create/write

## Scenario Presets

Use these as starting points:

- Highway monitoring: [highway_monitoring.json](file:///c:/Users/youss/OneDrive/Desktop/youssef/speedestimation_version2/config_examples/highway_monitoring.json)
  - Higher `v_max_kmh`, larger `min_distance_m`, turning disabled
- Urban intersection with turning: [urban_intersection_turning.json](file:///c:/Users/youss/OneDrive/Desktop/youssef/speedestimation_version2/config_examples/urban_intersection_turning.json)
  - Turning enabled, lower `v_max_kmh`, smaller `min_distance_m`
- Video-file testing: [video_file_testing.json](file:///c:/Users/youss/OneDrive/Desktop/youssef/speedestimation_version2/config_examples/video_file_testing.json)
  - Uses `camera.type=video_file` with an included test video

## Environment Variable Overrides (V5_*)

These override values from the JSON config when set:

- `V5_RTSP_URL`
- `V5_CAMERA_ID`
- `V5_MODEL_PATH`
- `V5_CONF`, `V5_IOU`
- `V5_HOMOGRAPHY_NPY`, `V5_METERS_PER_PIXEL`
- `V5_TRACK_IOU`, `V5_MAX_MISSING`
- `V5_DOT_MIN_DISTANCE_M`, `V5_DOT_MIN_DT_S`, `V5_DOT_BUFFER_SIZE`

This lets you keep a stable config file and adjust only a few parameters per deployment.
