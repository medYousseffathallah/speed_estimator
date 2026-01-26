# Project Rules

## Run Pipeline (Camera Configs)

Use a folder that contains one or more camera YAML files. Replace <CAMERA_DIR> with your folder path.

```bash
python scripts\run_pipeline.py --cameras <CAMERA_DIR> --tracking configs\tracking.yaml --speed configs\speed_model.yaml --camera-handling configs\camera_handling.yaml --log-level INFO
```

Examples:

```bash
python scripts\run_pipeline.py --cameras configs\cameras --tracking configs\tracking.yaml --speed configs\speed_model.yaml --camera-handling configs\camera_handling.yaml --log-level INFO
```

```bash
python scripts\run_pipeline.py --cameras configs\cameras_test_single --tracking configs\tracking.yaml --speed configs\speed_model.yaml --camera-handling configs\camera_handling.yaml --log-level INFO
```

```bash
python scripts\run_pipeline.py --cameras configs\cameras_gttet --tracking configs\tracking.yaml --speed configs\speed_model.yaml --camera-handling configs\camera_handling.yaml --log-level INFO
```

## Add Homography (Calibration)

Calibrate from a camera YAML file. Replace <CAMERA_YAML> with the YAML that points to your video.

```bash
python scripts\calibrate_camera.py --camera <CAMERA_YAML>
```

Examples:

```bash
python scripts\calibrate_camera.py --camera configs\cameras\gttet.yaml
```

```bash
python scripts\calibrate_camera.py --camera configs\cameras_test_single\normanniles3.yaml
```

## Run Video Test (Single File)

Use a direct video file and an explicit homography. Replace <VIDEO>, <CAMERA_ID>, and <HOMOGRAPHY_NPY>.

```bash
python scripts\run_video_test.py --video <VIDEO> --camera-id <CAMERA_ID> --homography <HOMOGRAPHY_NPY> --tracking configs\tracking.yaml --speed configs\speed_model.yaml --camera-handling configs\camera_handling.yaml --log-level INFO
```

Examples:

```bash
python scripts\run_video_test.py --video data_test\normanniles3_2025-10-26-17-18-45.mp4 --camera-id normanniles3 --homography calibration\normanniles_H.npy --tracking configs\tracking.yaml --speed configs\speed_model.yaml --camera-handling configs\camera_handling.yaml --log-level INFO
```

```bash
python scripts\run_video_test.py --video data_test\normanniles4_2025-10-26-17-15-45.mp4 --camera-id normanniles4 --homography calibration\normanniles_H.npy --tracking configs\tracking.yaml --speed configs\speed_model.yaml --camera-handling configs\camera_handling.yaml --log-level INFO
```
