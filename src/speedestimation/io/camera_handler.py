from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class ReconnectConfig:
    enabled: bool = True
    max_retries: int = 0
    backoff_s: float = 1.0
    reset_on_success: bool = True


@dataclass(frozen=True)
class RtspConfig:
    use_gstreamer: bool = False
    transport: str = "tcp"
    latency_ms: int = 100
    drop_on_latency: bool = True
    decoder: str = "avdec_h264"


@dataclass(frozen=True)
class CameraHandlerConfig:
    uri: str
    source_type: str
    fps_hint: float
    resize_enabled: bool
    resize_width: int
    resize_height: int
    reconnect: ReconnectConfig
    rtsp: RtspConfig

    @staticmethod
    def from_sources(
        camera_source_type: str,
        camera_uri: str,
        fps_hint: float,
        resize_enabled: bool,
        resize_width: int,
        resize_height: int,
        global_cfg: Dict[str, Any],
        per_camera_params: Optional[Dict[str, Any]] = None,
    ) -> "CameraHandlerConfig":
        per_camera_params = per_camera_params or {}
        reconnect = dict(global_cfg.get("reconnect", {}))
        reconnect.update(dict(per_camera_params.get("reconnect", {})))
        rtsp = dict(global_cfg.get("rtsp", {}))
        rtsp.update(dict(per_camera_params.get("rtsp", {})))
        return CameraHandlerConfig(
            uri=str(camera_uri),
            source_type=str(camera_source_type),
            fps_hint=float(fps_hint),
            resize_enabled=bool(resize_enabled),
            resize_width=int(resize_width),
            resize_height=int(resize_height),
            reconnect=ReconnectConfig(
                enabled=bool(reconnect.get("enabled", True)),
                max_retries=int(reconnect.get("max_retries", 0)),
                backoff_s=float(reconnect.get("backoff_s", 1.0)),
                reset_on_success=bool(reconnect.get("reset_on_success", True)),
            ),
            rtsp=RtspConfig(
                use_gstreamer=bool(rtsp.get("use_gstreamer", False)),
                transport=str(rtsp.get("transport", "tcp")),
                latency_ms=int(rtsp.get("latency_ms", 100)),
                drop_on_latency=bool(rtsp.get("drop_on_latency", True)),
                decoder=str(rtsp.get("decoder", "avdec_h264")),
            ),
        )


logger = logging.getLogger("speedestimation.io.camera")


class CameraHandler:
    def __init__(self, cfg: CameraHandlerConfig) -> None:
        self._cfg = cfg
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_index = 0
        self._t0_wall = time.monotonic()
        self._fps = float(cfg.fps_hint)
        self._retries = 0
        self._open()

    def __iter__(self) -> Iterator[Tuple[int, float, np.ndarray]]:
        while True:
            cap = self._cap
            if cap is None:
                if not self._try_reconnect():
                    logger.error("Camera source unavailable and reconnect disabled: %s", self._cfg.uri)
                    break
                cap = self._cap
                if cap is None:
                    break

            ok, frame = cap.read()
            if not ok:
                logger.warning("Failed to read frame from source: %s", self._cfg.uri)
                self._close()
                if not self._try_reconnect():
                    logger.error("Reconnect failed for source: %s", self._cfg.uri)
                    break
                continue

            if self._cfg.reconnect.enabled and self._cfg.reconnect.reset_on_success:
                self._retries = 0

            if self._cfg.resize_enabled:
                frame = cv2.resize(
                    frame,
                    (self._cfg.resize_width, self._cfg.resize_height),
                    interpolation=cv2.INTER_LINEAR,
                )

            t_s = self._timestamp_s(cap)
            fi = self._frame_index
            self._frame_index += 1
            yield fi, t_s, frame

    def close(self) -> None:
        self._close()

    def _try_reconnect(self) -> bool:
        if not self._cfg.reconnect.enabled:
            logger.error("Reconnect disabled for source: %s", self._cfg.uri)
            return False
        if self._cfg.reconnect.max_retries > 0 and self._retries >= self._cfg.reconnect.max_retries:
            logger.error("Reconnect retries exhausted for source: %s", self._cfg.uri)
            return False
        self._retries += 1
        logger.info("Reconnecting to source (%d): %s", self._retries, self._cfg.uri)
        time.sleep(max(0.0, float(self._cfg.reconnect.backoff_s)))
        return self._open()

    def _open(self) -> bool:
        self._close()
        uri, api = self._build_capture()
        if api is None:
            cap = cv2.VideoCapture(uri)
        else:
            cap = cv2.VideoCapture(uri, api)
        if not cap.isOpened():
            if self._cfg.source_type.lower() == "rtsp" and self._cfg.rtsp.use_gstreamer:
                fallback = cv2.VideoCapture(self._cfg.uri)
                if fallback.isOpened():
                    logger.warning("GStreamer failed; falling back to OpenCV backend for %s", self._cfg.uri)
                    cap = fallback
                else:
                    logger.error("Failed to open RTSP stream: %s", self._cfg.uri)
                    self._cap = None
                    return False
            else:
                logger.error("Failed to open source: %s", self._cfg.uri)
                self._cap = None
                return False

        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is not None and fps > 1e-3:
            self._fps = float(fps)
        self._cap = cap
        self._t0_wall = time.monotonic()
        return True

    def _close(self) -> None:
        if self._cap is not None:
            self._cap.release()
        self._cap = None

    def _timestamp_s(self, cap: cv2.VideoCapture) -> float:
        pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        if pos_msec is not None and pos_msec > 0:
            return float(pos_msec) / 1000.0
        return float(time.monotonic() - self._t0_wall)

    def _build_capture(self) -> Tuple[str, Optional[int]]:
        st = self._cfg.source_type.lower()
        if st == "rtsp" and self._cfg.rtsp.use_gstreamer:
            return self._build_gst_rtsp(), cv2.CAP_GSTREAMER
        return self._cfg.uri, None

    def _build_gst_rtsp(self) -> str:
        protocols = "tcp" if self._cfg.rtsp.transport.lower() == "tcp" else "udp"
        latency = int(self._cfg.rtsp.latency_ms)
        drop = "true" if self._cfg.rtsp.drop_on_latency else "false"
        decoder = self._cfg.rtsp.decoder
        return (
            f"rtspsrc location={self._cfg.uri} protocols={protocols} latency={latency} ! "
            "rtph264depay ! h264parse ! "
            f"{decoder} ! "
            "videoconvert ! appsink sync=false drop=" + drop
        )

