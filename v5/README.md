# Speed & Turning Estimation System (v5)

A minimal, deployable system for real-time vehicle speed and turning estimation using Computer Vision.

## 🚀 Features

- **Threaded Frame Ingestion**: High-performance single-threaded reading pipeline for Camera, RTSP, or Video files.
- **Object Tracking**: Robust tracking using Greedy IoU (or ByteTrack adapter).
- **Speed Estimation**: Converts pixel displacement to real-world speed (m/s -> km/h).
- **Turning Metrics**: Physically consistent turning estimation (Heading, Curvature, Turn Angle) using `turning_improved.py`.
- **Event Detection**: Automatically detects speeding violations.
- **Evidence Logging**:
  - Saves 30s video clips (centered on the event).
  - Logs metadata to SQLite database (`events.db`).

## 🛠️ Pipeline

1.  **Ingest**: Frames are captured in a dedicated thread and buffered.
2.  **Detect**: YOLOv8 detects vehicles (Car, Truck, Bus, Motorcycle).
3.  **Track**: Detections are associated across frames to form Tracks.
4.  **Estimate**:
    *   **Speed**: Calculated from centroid displacement over time, smoothed with EMA (Exponential Moving Average).
    *   **Turning**: Uses `turning_improved.py` to calculate curvature and angular rate, applying physical constraints to eliminate noise and flip errors.
5.  **Monitor**: Checks if speed > `speed_limit_kmh`.
6.  **Act**: If violation detected, triggers video recording and DB logging.

## 📐 Math & Logic

### Speed Calculation
Speed is computed by tracking the centroid of the vehicle:
1.  **Pixel Delta**: $\Delta p = \sqrt{(x_2-x_1)^2 + (y_2-y_1)^2}$
2.  **World Delta**: $\Delta d = \Delta p \times \text{scale}$ (where scale is defined in `config.yaml`)
3.  **Velocity**: $v = \Delta d / \Delta t$
4.  **Smoothing**: $v_{smooth} = \alpha \cdot v_{raw} + (1-\alpha) \cdot v_{prev}$

### Turning Estimation (`turning_improved.py`)
Turning metrics are derived from the trajectory:
1.  **Derivatives**: Velocity and acceleration vectors computed via central differences.
2.  **Curvature**: $\kappa = \frac{v_x a_y - v_y a_x}{(v_x^2 + v_y^2)^{1.5}}$
3.  **Angular Rate**: $\omega = v \cdot |\kappa|$
4.  **Constraints**: Filters out stationary noise and physically impossible turn rates.

## 📋 Configuration (`config.yaml`)

```yaml
camera:
  source: "video.mp4" # Path or 0 for webcam
  fps: 30

speed_estimation:
  pixel_to_meter_scale: 0.05 # Calibration factor
  speed_limit_kmh: 50.0      # Trigger threshold
```

## 🏃 Usage

1.  **Install Dependencies**:
    ```bash
    pip install ultralytics opencv-python pyyaml
    ```

2.  **Run**:
    ```bash
    python main.py
    ```

3.  **Output**:
    - **Live View**: Shows bounding boxes, speed, and turning metrics.
    - **`events/`**: Saved AVI clips of speeding events.
    - **`events.db`**: SQLite database of violations.

## 📂 Event Logging

When a vehicle exceeds the speed limit:
1.  The system captures the **past 15 seconds** from the buffer.
2.  It continues recording the **next 15 seconds**.
3.  The combined clip is saved to `events/event_YYYYMMDD_HHMMSS_Speed.avi`.
4.  A record is inserted into `events.db`:
    - `timestamp`: ISO8601 time
    - `camera_id`: Source ID
    - `speed_kmh`: Detected speed
    - `video_path`: Path to evidence clip

## 🔧 System Architecture

The v5 system is **fully isolated** with all components copied and adapted for standalone operation:

- **`main.py`**: Orchestrates the entire pipeline with threaded ingestion
- **`detection.py`**: YOLOv8 detector with vehicle class filtering
- **`tracking.py`**: Greedy IoU tracker for object persistence
- **`speed_estimation.py`**: Speed and turning calculation engine
- **`turning_improved.py`**: Physically consistent turning metrics (copied exactly)
- **`config.yaml`**: Centralized configuration

## 🎯 Jetson Deployment Ready

The system is optimized for edge deployment:
- Minimal dependencies (ultralytics, opencv-python, pyyaml)
- Efficient memory usage with bounded queues
- Thread-safe frame processing
- SQLite for lightweight event logging
- Configurable for different camera sources and calibration