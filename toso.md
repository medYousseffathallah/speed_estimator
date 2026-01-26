Roadmap: Speed Estimation Pipeline Improvements

1. Tracking stability (High impact)

- [x] Implement ByteTrack adapter (tracking/bytetrack_adapter.py)
- [x] Wire config to allow bytetrack backend in configs/tracking.yaml
- [x] Validate ID switch reduction on multi-object scenes (200f normanniles3 sample)
- [x] Tests: compare track continuity before/after on sample videos (greedy_iou vs bytetrack)

2. Detection realism (High impact)

- [x] Switch camera configs from mock to ultralytics_yolo
- [x] Tune conf_threshold, iou_threshold, and class_whitelist
- [x] Validate detection count consistency across frames (200f normanniles3 sample)
- [x] Tests: qualitative overlay review + speed output sanity (overlay video generated, speed stats captured)

3. Geometry accuracy (Medium impact)

- [x] Use bottom-center of bbox for homography mapping
- [x] Validate speed bias on tall vehicles is reduced (gttet GT OCR ~66 km/h; 260f pxscale run, raw mean 21.76 m/s, bias +3.42 m/s)
- [x] Tests: compare speed estimates against known ground truth (gttet_pred_260f_pxscale.csv vs GT 66 km/h)

4. Estimator memory hygiene (Medium impact)

- [x] Call SpeedEstimator.prune_missing each frame
- [x] Validate memory usage is stable in long runs (1200f normanniles3 tracemalloc)
- [x] Tests: long video run, monitor memory

5. ROI policy (Medium impact)

- [x] Evaluate post-tracking ROI filtering vs pre-tracking detection filtering (200f sample)
- [x] Validate fewer ID resets at ROI boundaries (200f ROI rectangle)
- [x] Tests: scenario with vehicles entering/leaving ROI (200f sample)

6. Multi-camera efficiency (Medium impact)

- [x] Explore shared detector instance or batched detection worker
- [x] Validate GPU memory usage and throughput (CPU run, 200f x2 cameras)
- [x] Tests: multi-camera run with profiling (shared detector, 200f x2 cameras)
