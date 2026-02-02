# Jetson Nano/Orin Adaptation Guide

## Overview

This guide provides comprehensive instructions for adapting the Multi-Camera Vehicle Speed Estimation pipeline for optimal performance on NVIDIA Jetson Nano and Jetson Orin devices. The pipeline is designed for edge deployment and includes specific optimizations for ARM architecture and GPU acceleration.

## Key Changes Required

### 1. Dependencies and Package Management

#### Remove Incompatible Packages
The current requirements include `opencv-python-headless` which conflicts with Jetson's system OpenCV:

```bash
# Remove opencv-python-headless from requirements.txt
pip uninstall opencv-python-headless

# Install system OpenCV (recommended for Jetson)
sudo apt update
sudo apt install python3-opencv
```

#### Jetson-Specific Dependencies
Add these Jetson-optimized packages to your environment:

```bash
# Install JetPack components (if not already installed)
sudo apt install nvidia-jetpack

# Install CUDA-enabled PyTorch for Jetson
# Check your JetPack version first
jetpack_release=$(dpkg-query --showformat='${Version}' --show nvidia-jetpack 2>/dev/null | cut -d'-' -f1)
echo "JetPack version: $jetpack_release"

# Install PyTorch with CUDA support (example for JetPack 5.1+)
pip install torch torchvision --index-url https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048
```

### 2. GPU/CUDA Configuration

#### Update Camera Configuration
Change the device parameter in your camera YAML files:

```yaml
# configs/cameras/test_normanniles.yaml (line 23)
detection:
  backend: "ultralytics_yolo"
  params:
    model_path: "yolov8n.pt"
    device: "cuda:0"  # Change from "cpu" to "cuda:0"
    conf_threshold: 0.3
    iou_threshold: 0.5
```

#### TensorRT Optimization
For maximum performance, export YOLO models to TensorRT format:

```python
# Export script for Jetson
from ultralytics import YOLO

# Load your model
model = YOLO('yolov8n.pt')

# Export to TensorRT with FP16 precision (recommended for Jetson)
model.export(format='engine', device=0, half=True, imgsz=640)

# Use the exported model in your camera config
detection:
  backend: "ultralytics_yolo"
  params:
    model_path: "yolov8n.engine"  # Use .engine instead of .pt
    device: "cuda:0"
```

### 3. Detection Backend Optimization

#### Jetson-Optimized Ultralytics Configuration
The current [yolo_ultralytics.py](file:///c:/Users/youss/OneDrive/Desktop/youssef/speedestimation_version2/src/speedestimation/detection/yolo_ultralytics.py) supports device specification:

```python
# The detector already supports device parameter
@dataclass
class UltralyticsYoloDetector(Detector):
    model_path: str
    conf_threshold: float = 0.25
    iou_threshold: float = 0.5
    device: Optional[str] = None  # Supports "cuda:0", "cuda", "cpu"
    class_whitelist: Optional[List[str]] = None
```

#### Model Size Recommendations
For Jetson devices, use smaller models:
- **Jetson Nano 4GB**: `yolov8n.pt` (nano) or `yolov8s.pt` (small)
- **Jetson Orin Nano 8GB**: `yolov8s.pt` (small) or `yolov8m.pt` (medium)
- **Jetson Orin NX**: `yolov8m.pt` (medium) or `yolov8l.pt` (large)

### 4. Performance Optimizations

#### Power Mode Configuration
Set Jetson to maximum performance mode:

```bash
# Check current power mode
sudo nvpmodel -q

# Set to MAXN mode (maximum performance)
sudo nvpmodel -m 0

# Lock clocks to maximum
sudo jetson_clocks

# Enable jetson_clocks on boot
sudo systemctl enable jetson_clocks
```

#### Memory Optimization
For Jetson Nano with limited memory:

```yaml
# Reduce frame processing frequency
detection:
  interval: 10  # Process every 10th frame (increase from 5)
  
# Reduce input resolution
runtime:
  resize:
    enabled: true
    width: 640   # Reduce from 1280
    height: 480  # Reduce from 720
```

#### TensorRT Precision Settings
```python
# FP16 precision (recommended balance)
model.export(format='engine', half=True)

# INT8 precision (maximum speed, requires calibration)
# Only use if you have calibration data
model.export(format='engine', int8=True, data='path/to/calibration/images')
```

### 5. ARM Architecture Optimizations

#### NumPy and BLAS
Ensure ARM-optimized libraries are used:

```bash
# Install OpenBLAS for better ARM performance
sudo apt install libopenblas-dev

# Set OpenBLAS threads
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
```

#### Python Environment
Use Python 3.8+ for better ARM compatibility:

```bash
# Check Python version
python3 --version

# If needed, install newer Python
sudo apt install python3.9 python3.9-dev python3.9-venv
```

### 6. Docker Deployment (Optional)

#### Jetson-Compatible Docker Image
Use Ultralytics' official Jetson Docker images:

```bash
# For JetPack 5.x (Orin series)
docker pull ultralytics/ultralytics:latest-jetson-jetpack5

# For JetPack 4.x (Nano series)
docker pull ultralytics/ultralytics:latest-jetson-jetpack4

# Run with GPU access
docker run -it --runtime nvidia \
  --device /dev/video0 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY=$DISPLAY \
  ultralytics/ultralytics:latest-jetson-jetpack5
```

### 7. Configuration Examples

#### Jetson Orin Nano Configuration
```yaml
# configs/cameras/jetson_orin_nano.yaml
camera_id: "jetson_orin_nano"
source:
  type: "file"
  uri: "data_test/video.mp4"

runtime:
  fps_hint: 30.0
  resize:
    enabled: true
    width: 1280
    height: 720

detection:
  interval: 5
  backend: "ultralytics_yolo"
  params:
    model_path: "yolov8s.engine"  # TensorRT model
    device: "cuda:0"              # GPU acceleration
    conf_threshold: 0.3
    iou_threshold: 0.5
    class_whitelist: ["car", "truck", "bus", "motorcycle"]

output:
  overlay:
    enabled: true
    show: false  # Disable GUI on headless systems
    write_video: true
    video_path: "outputs/jetson_overlay.mp4"
```

#### Jetson Nano 4GB Configuration
```yaml
# configs/cameras/jetson_nano_4gb.yaml
camera_id: "jetson_nano_4gb"
source:
  type: "file"
  uri: "data_test/video.mp4"

runtime:
  fps_hint: 15.0  # Lower frame rate for Nano
  resize:
    enabled: true
    width: 640     # Smaller resolution
    height: 480

detection:
  interval: 10     # Process every 10th frame
  backend: "ultralytics_yolo"
  params:
    model_path: "yolov8n.engine"  # Smallest model
    device: "cuda:0"
    conf_threshold: 0.4
    iou_threshold: 0.5

# Reduce memory usage
speed_model:
  dots:
    buffer_size: 3  # Reduce from 5
  smoothing:
    ema_alpha: 0.35  # Slightly faster response
```

### 8. Monitoring and Debugging

#### GPU Utilization
```bash
# Monitor GPU usage
sudo tegrastats

# Or use jtop (install if needed)
pip install jetson-stats
sudo jtop
```

#### Temperature Monitoring
```bash
# Check thermal zones
cat /sys/class/thermal/thermal_zone*/temp

# Monitor during inference
watch -n 1 'cat /sys/class/thermal/thermal_zone*/temp'
```

#### Performance Benchmarking
```bash
# Benchmark your model
yolo benchmark model=yolov8s.engine device=cuda:0 imgsz=640

# Profile with TensorRT
trtexec --loadEngine=yolov8s.engine --fp16
```

### 9. Troubleshooting

#### CUDA Out of Memory
- Reduce batch size in model export
- Lower input resolution
- Increase detection interval
- Use smaller model variant

#### Slow Inference
- Ensure TensorRT model is being used
- Check power mode is set to MAXN
- Verify jetson_clocks is enabled
- Monitor thermal throttling

#### Model Loading Issues
- Verify CUDA version compatibility
- Check TensorRT version matches JetPack
- Ensure model is exported for correct architecture

### 10. Performance Expectations

#### Jetson Orin Nano (8GB)
- **YOLOv8n**: ~25-30 FPS @ 640x640
- **YOLOv8s**: ~15-20 FPS @ 640x640
- **Power consumption**: 7-15W

#### Jetson Nano (4GB)
- **YOLOv8n**: ~5-8 FPS @ 480x480
- **YOLOv8s**: ~3-5 FPS @ 480x480
- **Power consumption**: 5-10W

*Performance varies based on model complexity, input resolution, and system load.*

## Summary

The key adaptations for Jetson deployment are:
1. **Replace opencv-python-headless** with system OpenCV
2. **Configure GPU acceleration** in camera YAML files
3. **Export models to TensorRT** for optimal performance
4. **Optimize power settings** and memory usage
5. **Use appropriate model sizes** for your Jetson variant
6. **Monitor performance** and adjust settings as needed

These changes will enable efficient real-time vehicle speed estimation on Jetson edge devices while maintaining accuracy and reliability.