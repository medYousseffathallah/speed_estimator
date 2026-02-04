import os
import time
import threading
import queue
import sqlite3
import logging
import subprocess
from dataclasses import dataclass
from collections import deque
from datetime import datetime
from typing import Optional, Dict, List, Tuple

import cv2
import numpy as np

from detection import UltralyticsYoloDetector, DetectorInput
from tracking import GreedyIoUTracker, TrackerInput
from speed_estimation import SpeedEstimator, SpeedEstimatorConfig, Track, SpeedSample

logger = logging.getLogger("v5_system")


@dataclass(frozen=True)
class FramePacket:
    camera_id: str
    frame_index: int
    timestamp_s: float
    image_bgr: np.ndarray


class _FfmpegReader:
    def __init__(self, source: str, width: int, height: int, fps: float) -> None:
        self.source = source
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self._proc: Optional[subprocess.Popen] = None
        self._frame_bytes = int(self.width * self.height * 3)

    def open(self) -> bool:
        self.close()
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-rtsp_transport",
            "tcp",
            "-stimeout",
            "5000000",
            "-fflags",
            "nobuffer",
            "-flags",
            "low_delay",
            "-max_delay",
            "0",
            "-analyzeduration",
            "0",
            "-probesize",
            "32",
            "-i",
            str(self.source),
            "-an",
            "-vf",
            f"scale={self.width}:{self.height}",
            "-pix_fmt",
            "bgr24",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
        try:
            popen_kwargs = {
                "stdout": subprocess.PIPE,
                "stderr": subprocess.DEVNULL,
                "stdin": subprocess.DEVNULL,
                "bufsize": 0,
            }
            if os.name == "nt":
                popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
            self._proc = subprocess.Popen(cmd, **popen_kwargs)
        except FileNotFoundError:
            logger.error("ffmpeg not found on PATH")
            self._proc = None
            return False
        except Exception as e:
            logger.error(f"Failed to start ffmpeg for {self.source}: {e}")
            self._proc = None
            return False
        return True

    def read(self) -> Optional[np.ndarray]:
        if self._proc is None or self._proc.stdout is None:
            return None
        raw = self._proc.stdout.read(self._frame_bytes)
        if raw is None or len(raw) != self._frame_bytes:
            return None
        frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 3))
        return frame

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.stdout:
                self._proc.stdout.close()
        except Exception:
            pass
        try:
            if self._proc.stderr:
                self._proc.stderr.close()
        except Exception:
            pass
        try:
            self._proc.terminate()
        except Exception:
            pass
        try:
            self._proc.wait(timeout=1.0)
        except Exception:
            pass
        self._proc = None


class _OpenCvReader:
    def __init__(self, source: str) -> None:
        self.source = source
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        self.close()
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened() and str(self.source).isdigit():
            cap = cv2.VideoCapture(int(self.source))
        if not cap.isOpened():
            self._cap = None
            return False
        self._cap = cap
        return True

    def read(self) -> Optional[np.ndarray]:
        if self._cap is None:
            return None
        ret, frame = self._cap.read()
        if not ret:
            return None
        return frame

    def close(self) -> None:
        if self._cap is None:
            return
        try:
            self._cap.release()
        except Exception:
            pass
        self._cap = None


class FrameIngestor:
    def __init__(
        self,
        camera_id: str,
        source: str,
        width: int,
        height: int,
        fps: float,
        queue_size: int = 30,
        backend: str = "auto",
        reconnect_backoff_s: float = 1.0,
        frozen_s: float = 3.0,
    ):
        self.camera_id = str(camera_id)
        self.source = str(source)
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.backend = str(backend).lower()
        src = self.source.lower()
        self._reconnect_on_failure = bool(
            self.backend in {"ffmpeg", "rtsp"} or src.startswith("rtsp://") or self.source.isdigit()
        )
        self.queue = queue.Queue(maxsize=int(queue_size))
        self._stop = threading.Event()
        self._reconnect = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._reconnect_backoff_s = float(reconnect_backoff_s)
        self._frozen_s = float(frozen_s)
        self._dropped = 0
        self._last_frame_t: Optional[float] = None
        self._frame_index = 0
        self._reader = None
        self._freeze_ref: Optional[np.ndarray] = None
        self._freeze_ref_t: Optional[float] = None

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._reconnect.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._close_reader()

    def request_reconnect(self) -> None:
        if not self._reconnect_on_failure:
            return
        self._reconnect.set()

    def read(self) -> Optional[FramePacket]:
        try:
            return self.queue.get(timeout=0.05)
        except queue.Empty:
            return None

    def stats(self) -> Dict[str, float]:
        t = self._last_frame_t if self._last_frame_t is not None else float("nan")
        return {
            "dropped_frames": float(self._dropped),
            "last_frame_ts": float(t),
        }

    def _make_reader(self):
        if self.backend in {"ffmpeg", "rtsp"}:
            return _FfmpegReader(self.source, self.width, self.height, self.fps)
        if self.backend in {"opencv", "cv2"}:
            return _OpenCvReader(self.source)
        if self.source.lower().startswith("rtsp://"):
            return _FfmpegReader(self.source, self.width, self.height, self.fps)
        return _OpenCvReader(self.source)

    def _open_reader(self) -> bool:
        self._reader = self._make_reader()
        try:
            ok = bool(self._reader.open())
        except Exception:
            ok = False
        if not ok:
            self._reader = None
            return False
        return True

    def _close_reader(self) -> None:
        if self._reader is None:
            return
        try:
            self._reader.close()
        except Exception:
            pass
        self._reader = None

    def _push(self, packet: FramePacket) -> None:
        try:
            self.queue.put(packet, timeout=0.02)
            return
        except queue.Full:
            self._dropped += 1
        try:
            _ = self.queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self.queue.put(packet, timeout=0.02)
        except queue.Full:
            self._dropped += 1

    def _check_frozen(self, frame: np.ndarray, t: float) -> bool:
        if self._frozen_s <= 0.0:
            return False
        try:
            small = cv2.resize(frame, (32, 32), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        except Exception:
            return False
        if self._freeze_ref is None or self._freeze_ref_t is None:
            self._freeze_ref = gray
            self._freeze_ref_t = t
            return False
        diff = float(np.mean(np.abs(gray.astype(np.int16) - self._freeze_ref.astype(np.int16))))
        if diff >= 2.0:
            self._freeze_ref = gray
            self._freeze_ref_t = t
            return False
        if (t - self._freeze_ref_t) >= self._frozen_s:
            return True
        return False

    def _run(self) -> None:
        backoff = self._reconnect_backoff_s
        while not self._stop.is_set():
            if self._reader is None:
                if not self._open_reader():
                    if not self._reconnect_on_failure:
                        logger.error(f"[{self.camera_id}] Failed to open source")
                        self._stop.set()
                        break
                    logger.error(f"[{self.camera_id}] Failed to open source, retrying in {backoff:.1f}s")
                    time.sleep(backoff)
                    backoff = min(10.0, backoff * 1.5)
                    continue
                logger.info(f"[{self.camera_id}] Ingest connected ({self.backend})")
                backoff = self._reconnect_backoff_s
                self._reconnect.clear()
                self._freeze_ref = None
                self._freeze_ref_t = None

            if self._reconnect.is_set():
                logger.warning(f"[{self.camera_id}] Reconnecting ingest")
                self._close_reader()
                continue

            try:
                frame = self._reader.read()
            except Exception:
                frame = None

            if frame is None:
                if not self._reconnect_on_failure:
                    logger.info(f"[{self.camera_id}] Stream ended")
                    self._stop.set()
                    self._close_reader()
                    break
                logger.warning(f"[{self.camera_id}] Frame read failed, reconnecting")
                self._close_reader()
                time.sleep(backoff)
                backoff = min(10.0, backoff * 1.5)
                continue

            if self.width and self.height and (frame.shape[1] != self.width or frame.shape[0] != self.height):
                try:
                    frame = cv2.resize(frame, (self.width, self.height))
                except Exception:
                    continue

            t = time.monotonic()
            self._last_frame_t = t
            if self._check_frozen(frame, t):
                logger.warning(f"[{self.camera_id}] Frozen frames detected, reconnecting")
                self._close_reader()
                continue

            packet = FramePacket(camera_id=self.camera_id, frame_index=self._frame_index, timestamp_s=t, image_bgr=frame)
            self._frame_index += 1
            self._push(packet)


class EventManager:
    def __init__(self, db_path: str, video_dir: str, buffer_seconds: int, fps: int):
        self.db_path = str(db_path)
        self.video_dir = str(video_dir)
        self.buffer_seconds = int(buffer_seconds)
        self.fps = int(fps)
        self.buffer_size = max(1, int(self.buffer_seconds * self.fps))
        self.frame_buffer = deque(maxlen=self.buffer_size)
        self.recording = False
        self.recording_frames_left = 0
        self.video_writer = None
        self.current_video_path = None
        os.makedirs(self.video_dir, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=5000")
        except Exception:
            pass
        return conn

    def _init_db(self):
        conn = self._connect()
        c = conn.cursor()
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS speed_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                camera_id TEXT,
                speed_kmh REAL,
                speed_limit_kmh REAL,
                video_path TEXT
            )
            """
        )
        conn.commit()
        conn.close()

    def update_buffer(self, frame):
        self.frame_buffer.append(frame)
        if self.recording:
            if self.video_writer:
                self.video_writer.write(frame)
            self.recording_frames_left -= 1
            if self.recording_frames_left <= 0:
                self.stop_recording()

    def trigger_event(self, camera_id: str, speed_kmh: float, limit_kmh: float, frame_width: int, frame_height: int):
        frames_needed = max(1, int(self.buffer_seconds * self.fps))
        if not self.recording:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"event_{camera_id}_{ts}_{int(speed_kmh)}kmh.avi"
            self.current_video_path = os.path.join(self.video_dir, filename)
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            self.video_writer = cv2.VideoWriter(
                self.current_video_path, fourcc, float(self.fps), (int(frame_width), int(frame_height))
            )
            for f in self.frame_buffer:
                self.video_writer.write(f)
            self.recording = True
            self.recording_frames_left = frames_needed
            self._log_db(camera_id, speed_kmh, limit_kmh, self.current_video_path)
        else:
            if self.recording_frames_left < frames_needed:
                self.recording_frames_left = frames_needed

    def _log_db(self, camera_id, speed, limit, video_path):
        try:
            conn = self._connect()
            c = conn.cursor()
            c.execute(
                "INSERT INTO speed_events (timestamp, camera_id, speed_kmh, speed_limit_kmh, video_path) VALUES (?, ?, ?, ?, ?)",
                (datetime.now().isoformat(), str(camera_id), float(speed), float(limit), str(video_path)),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"DB Error: {e}")

    def stop_recording(self):
        if self.video_writer:
            try:
                self.video_writer.release()
            except Exception:
                pass
            self.video_writer = None
        self.recording = False
        self.current_video_path = None


class Homography:
    def __init__(self, H: Optional[np.ndarray]) -> None:
        self.H = H

    def transform_point(self, xy_img: Tuple[float, float]) -> Tuple[float, float]:
        if self.H is None:
            return (float("nan"), float("nan"))
        x, y = float(xy_img[0]), float(xy_img[1])
        p = np.asarray([x, y, 1.0], dtype=np.float64)
        w = self.H @ p
        if w[2] == 0.0:
            return (float("nan"), float("nan"))
        return (float(w[0] / w[2]), float(w[1] / w[2]))


def load_homography(calib: Dict) -> Optional[Homography]:
    path = calib.get("homography_path")
    if path and isinstance(path, str):
        if os.path.exists(path):
            try:
                H = None
                if path.endswith(".npy"):
                    H = np.load(path)
                elif path.endswith(".json"):
                    import json

                    with open(path, "r", encoding="utf-8") as f:
                        H = np.array(json.load(f))
                else:
                    H = np.loadtxt(path)
                if H is not None and H.shape == (3, 3):
                    return Homography(H.astype(np.float64))
                logger.error(f"Invalid homography shape in {path}: {H.shape if H is not None else 'None'}")
            except Exception as e:
                logger.error(f"Failed to load homography from {path}: {e}")
        else:
            logger.error(f"Homography path not found: {path}")

    H_list = calib.get("homography_matrix")
    if isinstance(H_list, list) and len(H_list) == 3 and all(isinstance(row, list) and len(row) == 3 for row in H_list):
        H = np.asarray(H_list, dtype=np.float64)
        return Homography(H)

    dots = calib.get("dots", {}) or {}
    img = dots.get("image_xy") or []
    world = dots.get("world_xy_m") or []
    if isinstance(img, list) and isinstance(world, list) and len(img) >= 4 and len(img) == len(world):
        src = np.asarray(img, dtype=np.float64).reshape(-1, 2)
        dst = np.asarray(world, dtype=np.float64).reshape(-1, 2)
        H, _ = cv2.findHomography(src, dst, method=0)
        if H is not None:
            return Homography(H.astype(np.float64))
    return None


class _Aggregator:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.count_by_cam: Dict[str, int] = {}
        self.sum_speed_kmh: float = 0.0
        self.sample_count: int = 0

    def update(self, cam_id: str, samples: List[SpeedSample]) -> None:
        with self._lock:
            self.count_by_cam[cam_id] = int(self.count_by_cam.get(cam_id, 0) + len(samples))
            for s in samples:
                self.sum_speed_kmh += float(s.speed_mps_smoothed) * 3.6
                self.sample_count += 1

    def snapshot(self) -> Tuple[Dict[str, int], float]:
        with self._lock:
            avg = (self.sum_speed_kmh / self.sample_count) if self.sample_count > 0 else 0.0
            return (dict(self.count_by_cam), float(avg))


class CameraWorker:
    def __init__(self, cam_cfg: Dict, det_cfg: Dict, tr_cfg: Dict, sp_cfg: Dict, global_cfg: Dict, agg: _Aggregator) -> None:
        self.cam_id = str(cam_cfg["id"])
        self.enabled = bool(cam_cfg.get("enabled", True))
        self.width = int(cam_cfg.get("width", 0) or 0)
        self.height = int(cam_cfg.get("height", 0) or 0)
        self.fps = float(cam_cfg.get("fps", 0) or 0)
        self.source = str(cam_cfg["source"])
        self.limit_kmh = float(cam_cfg.get("speed_limit_kmh", 50.0))
        self.calib = dict(cam_cfg.get("calibration", {}) or {})
        self.pixel_scale = float(self.calib.get("pixel_to_meter_scale", cam_cfg.get("pixel_to_meter_scale", 0.0)) or 0.0)
        self.homography = load_homography(self.calib)
        self.overlay = dict(global_cfg.get("overlay", {}) or {})
        self.ingest_cfg = dict(global_cfg.get("ingestion", {}) or {})
        out_cfg = dict(cam_cfg.get("output", {}) or {})
        global_out = dict(global_cfg.get("output", {}) or {})
        db_path = str(out_cfg.get("db_path", global_out.get("db_path", "events.db")))
        self.event_mgr = EventManager(
            db_path=db_path,
            video_dir=str(out_cfg.get("video_dir", os.path.join("events", self.cam_id))),
            buffer_seconds=int(out_cfg.get("video_buffer_seconds", 15)),
            fps=int(self.fps) if self.fps > 0 else 30,
        )

        backend = str(cam_cfg.get("ingest_backend", self.ingest_cfg.get("backend", "auto")) or "auto")
        queue_size = int(cam_cfg.get("queue_size", self.ingest_cfg.get("queue_size", 30)) or 30)
        frozen_s = float(cam_cfg.get("frozen_s", self.ingest_cfg.get("frozen_s", 3.0)) or 3.0)
        reconnect_backoff_s = float(cam_cfg.get("reconnect_backoff_s", self.ingest_cfg.get("reconnect_backoff_s", 1.0)) or 1.0)

        self.ingestor = FrameIngestor(
            camera_id=self.cam_id,
            source=self.source,
            width=self.width,
            height=self.height,
            fps=self.fps if self.fps > 0 else 30.0,
            queue_size=queue_size,
            backend=backend,
            reconnect_backoff_s=reconnect_backoff_s,
            frozen_s=frozen_s,
        )

        self.detector = UltralyticsYoloDetector(
            model_path=str(det_cfg.get("model_path", "yolov8n.pt")),
            conf_threshold=float(det_cfg.get("conf_threshold", 0.25)),
            class_whitelist=None,
        )
        self.class_ids = list(det_cfg.get("class_ids", []) or [])
        self.tracker = GreedyIoUTracker(
            iou_threshold=float(tr_cfg.get("iou_threshold", 0.3)),
            max_age_frames=int(tr_cfg.get("max_age_frames", 30)),
            min_hits=int(tr_cfg.get("min_hits", 3)),
        )

        est_cfg = SpeedEstimatorConfig.from_dict(
            {
                "raw_speed": {"min_speed_mps": float(sp_cfg.get("min_speed_mps", 1.0))},
                "turning": dict(sp_cfg.get("turning", {}) or {}),
                "dots": {"min_distance_m": 0.5, "buffer_size": 5},
                "smoothing": {"enabled": True, "alpha": float(sp_cfg.get("smoothing_alpha", 0.3)), "max_gap_s": 1.0},
            }
        )
        self.estimator = SpeedEstimator(est_cfg)
        self._agg = agg
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        if not self.enabled:
            logger.info(f"[{self.cam_id}] Disabled")
            return
        self.ingestor.start()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.ingestor.stop()
        self.event_mgr.stop_recording()

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def _img_to_world(self, x1: float, y1: float, x2: float, y2: float) -> Optional[Tuple[float, float]]:
        cx = 0.5 * (x1 + x2)
        cy = y2
        if self.homography is not None:
            wx, wy = self.homography.transform_point((cx, cy))
            if wx == wx and wy == wy:
                return (float(wx), float(wy))
            return None
        if self.pixel_scale > 0.0:
            return (float(cx * self.pixel_scale), float(cy * self.pixel_scale))
        return None

    def _run(self) -> None:
        self.event_mgr.frame_buffer.clear()
        show = bool(self.overlay.get("show", True))
        while not self._stop.is_set():
            try:
                packet = self.ingestor.read()
                if packet is None:
                    time.sleep(0.005)
                    continue
                frame = packet.image_bgr
                ts = float(packet.timestamp_s)
                dets = self.detector.detect(DetectorInput(frame_index=packet.frame_index, image_bgr=frame))
                if self.class_ids:
                    dets = [d for d in dets if d.class_id in self.class_ids]
                tr_out = self.tracker.update(
                    TrackerInput(camera_id=self.cam_id, frame_index=packet.frame_index, timestamp_s=ts, detections=dets)
                )
                tracks: List[Track] = []
                for tsr in tr_out.tracks:
                    x1, y1, x2, y2 = tsr.bbox_xyxy
                    world_xy = self._img_to_world(x1, y1, x2, y2)
                    if world_xy is None:
                        continue
                    tracks.append(
                        Track(
                            camera_id=self.cam_id,
                            frame_index=packet.frame_index,
                            timestamp_s=ts,
                            state=tsr,
                            world_xy_m=world_xy,
                        )
                    )
                samples = self.estimator.update(tracks)
                self._agg.update(self.cam_id, samples)

                limit = float(self.limit_kmh)
                for s in samples:
                    v_kmh = float(s.speed_mps_smoothed) * 3.6
                    t_state = next((t for t in tr_out.tracks if t.track_id == s.track_id), None)
                    if t_state:
                        x1, y1, x2, y2 = t_state.bbox_xyxy
                        color = (0, 255, 0)
                        if v_kmh > limit:
                            color = (0, 0, 255)
                            self.event_mgr.trigger_event(self.cam_id, v_kmh, limit, frame.shape[1], frame.shape[0])
                        if bool(self.overlay.get("draw_speed", True)) or bool(self.overlay.get("draw_turn", True)) or bool(
                            self.overlay.get("draw_curvature", True)
                        ):
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        if bool(self.overlay.get("draw_speed", True)):
                            label = f"ID:{s.track_id} {v_kmh:.1f}km/h"
                            cv2.putText(
                                frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                            )
                        if bool(self.overlay.get("draw_turn", True)):
                            turn_label = f"Turn:{s.turn_angle_deg:.1f}°"
                            cv2.putText(
                                frame,
                                turn_label,
                                (int(x1), int(y1) - 25),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 255, 0),
                                1,
                            )
                        if bool(self.overlay.get("draw_curvature", True)):
                            curv_label = f"Curv:{s.curvature_1pm:.3f}"
                            cv2.putText(
                                frame,
                                curv_label,
                                (int(x1), int(y1) - 40),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 255, 255),
                                1,
                            )

                self.event_mgr.update_buffer(frame)
                if show:
                    title_prefix = str(self.overlay.get("window_title_prefix", "Speed & Turning"))
                    cv2.imshow(f"{title_prefix} - {self.cam_id}", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        self._stop.set()
                        break
            except Exception as e:
                logger.error(f"[{self.cam_id}] Worker error: {e}")
                time.sleep(0.2)


class CameraManager:
    def __init__(self, cfg: Dict) -> None:
        det_cfg = dict(cfg.get("detection", {}) or {})
        tr_cfg = dict(cfg.get("tracking", {}) or {})
        sp_cfg = dict(cfg.get("speed_estimation", {}) or {})
        global_cfg = dict(cfg.get("global", {}) or {})
        cams = list(cfg.get("cameras", []) or [])
        if not cams:
            raise ValueError("No cameras configured")

        enabled_cams = [cam for cam in cams if bool(cam.get("enabled", True))]
        overlay = dict(global_cfg.get("overlay", {}) or {})
        if len(enabled_cams) > 1 and bool(overlay.get("show", True)) and not bool(overlay.get("allow_multi_window", False)):
            overlay["show"] = False
            global_cfg["overlay"] = overlay
            logger.warning("Disabled overlay.show for multi-camera (HighGUI is not thread-safe). Set overlay.allow_multi_window=true to override.")

        self._agg = _Aggregator()
        self._workers: List[CameraWorker] = []
        for cam in enabled_cams:
            self._workers.append(CameraWorker(cam, det_cfg, tr_cfg, sp_cfg, global_cfg, self._agg))

        self._stop = threading.Event()
        self._mon = threading.Thread(target=self._monitor, daemon=True)

        mon_cfg = dict(global_cfg.get("monitoring", {}) or {})
        self._stale_s = float(mon_cfg.get("stale_s", 5.0))
        self._check_interval_s = float(mon_cfg.get("check_interval_s", 1.0))

    def start(self) -> None:
        for w in self._workers:
            w.start()
        self._mon.start()

    def stop(self) -> None:
        self._stop.set()
        for w in self._workers:
            w.stop()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    def snapshot(self) -> Tuple[Dict[str, int], float]:
        return self._agg.snapshot()

    def all_stopped(self) -> bool:
        return all(w.ingestor.stopped or not w.is_alive() for w in self._workers)

    def _monitor(self) -> None:
        while not self._stop.is_set():
            now = time.monotonic()
            for w in self._workers:
                st = w.ingestor.stats()
                last_ts = float(st.get("last_frame_ts", float("nan")))
                if last_ts == last_ts and (now - last_ts) > self._stale_s:
                    logger.warning(f"[{w.cam_id}] No frames for {now - last_ts:.1f}s, reconnecting")
                    w.ingestor.request_reconnect()
            time.sleep(self._check_interval_s)
