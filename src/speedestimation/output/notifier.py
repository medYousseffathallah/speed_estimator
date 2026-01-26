from __future__ import annotations

import json
import logging
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol

from speedestimation.output.alerts import SpeedAlertEvent


logger = logging.getLogger("speedestimation.output.notifier")


class Notifier(Protocol):
    def notify_speed(self, event: SpeedAlertEvent) -> None:
        ...


@dataclass
class LogNotifier(Notifier):
    level: str = "WARNING"

    def notify_speed(self, event: SpeedAlertEvent) -> None:
        lvl = getattr(logging, str(self.level).upper(), logging.WARNING)
        logger.log(
            lvl,
            "SPEED_ALERT camera=%s track=%s v=%.2f m/s limit=%.2f m/s (+%.2f) t=%.3f",
            event.camera_id,
            event.track_id,
            event.speed_mps,
            event.speed_limit_mps,
            event.threshold_mps,
            event.timestamp_s,
        )


@dataclass
class HttpWebhookNotifier(Notifier):
    url: str
    headers: Dict[str, str]
    timeout_s: float = 2.0

    def notify_speed(self, event: SpeedAlertEvent) -> None:
        payload = {
            "type": "speed_alert",
            "camera_id": event.camera_id,
            "track_id": event.track_id,
            "timestamp_s": event.timestamp_s,
            "frame_index": event.frame_index,
            "speed_mps": event.speed_mps,
            "speed_limit_mps": event.speed_limit_mps,
            "threshold_mps": event.threshold_mps,
            "use_speed": event.use_speed,
            "world_xy_m": [event.world_xy_m[0], event.world_xy_m[1]],
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(self.url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        for k, v in self.headers.items():
            if k.lower() == "content-type":
                continue
            req.add_header(str(k), str(v))
        try:
            with urllib.request.urlopen(req, timeout=float(self.timeout_s)) as resp:
                _ = resp.read(1)
        except Exception:
            logger.exception("Failed to POST speed alert to webhook")


def create_notifier(cfg: Dict[str, Any]) -> Notifier:
    t = str(cfg.get("type", "log")).lower()
    if t == "log":
        return LogNotifier(level=str(cfg.get("level", "WARNING")))
    if t == "http":
        http = dict(cfg.get("http", {}))
        url = str(http.get("url", ""))
        if not url:
            raise ValueError("notifier.http.url is required when notifier.type=http")
        headers = http.get("headers", {}) or {}
        if not isinstance(headers, dict):
            raise ValueError("notifier.http.headers must be a dict")
        timeout_s = float(http.get("timeout_s", 2.0))
        return HttpWebhookNotifier(url=url, headers={str(k): str(v) for k, v in headers.items()}, timeout_s=timeout_s)
    raise ValueError(f"Unknown notifier.type: {t}")

