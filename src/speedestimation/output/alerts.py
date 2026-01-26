from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

from speedestimation.utils.types import SpeedSample


SpeedField = Literal["raw", "limited", "smoothed"]


@dataclass(frozen=True)
class SpeedAlertConfig:
    enabled: bool
    speed_limit_kmh: float
    threshold_kmh: float
    min_consecutive_samples: int
    cooldown_s: float
    use_speed: SpeedField

    @staticmethod
    def from_dict(d: Dict) -> "SpeedAlertConfig":
        return SpeedAlertConfig(
            enabled=bool(d.get("enabled", False)),
            speed_limit_kmh=float(d.get("speed_limit_kmh", 50.0)),
            threshold_kmh=float(d.get("threshold_kmh", 0.0)),
            min_consecutive_samples=int(d.get("min_consecutive_samples", 3)),
            cooldown_s=float(d.get("cooldown_s", 10.0)),
            use_speed=str(d.get("use_speed", "smoothed")).lower(),  # type: ignore[return-value]
        )

    def speed_limit_mps(self) -> float:
        return float(self.speed_limit_kmh) / 3.6

    def threshold_mps(self) -> float:
        return float(self.threshold_kmh) / 3.6


@dataclass(frozen=True)
class SpeedAlertEvent:
    camera_id: str
    track_id: int
    timestamp_s: float
    frame_index: int
    speed_mps: float
    speed_limit_mps: float
    threshold_mps: float
    use_speed: SpeedField
    world_xy_m: tuple[float, float]


@dataclass
class _TrackAlertState:
    consecutive: int = 0
    last_alert_ts: Optional[float] = None


class SpeedAlertEngine:
    def __init__(self, cfg: SpeedAlertConfig) -> None:
        self._cfg = cfg
        self._state: Dict[int, _TrackAlertState] = {}

    def update(self, samples: List[SpeedSample]) -> List[SpeedAlertEvent]:
        if not self._cfg.enabled:
            return []
        events: List[SpeedAlertEvent] = []
        limit = self._cfg.speed_limit_mps()
        thr = self._cfg.threshold_mps()
        min_n = max(1, int(self._cfg.min_consecutive_samples))
        cooldown = max(0.0, float(self._cfg.cooldown_s))

        for s in samples:
            v = self._select_speed(s)
            st = self._state.get(s.track_id)
            if st is None:
                st = _TrackAlertState()
                self._state[s.track_id] = st

            if v >= (limit + thr):
                st.consecutive += 1
            else:
                st.consecutive = 0
                continue

            if st.consecutive < min_n:
                continue

            if st.last_alert_ts is not None and (float(s.timestamp_s) - float(st.last_alert_ts)) < cooldown:
                continue

            st.last_alert_ts = float(s.timestamp_s)
            events.append(
                SpeedAlertEvent(
                    camera_id=s.camera_id,
                    track_id=s.track_id,
                    timestamp_s=float(s.timestamp_s),
                    frame_index=int(s.frame_index),
                    speed_mps=float(v),
                    speed_limit_mps=float(limit),
                    threshold_mps=float(thr),
                    use_speed=self._cfg.use_speed,
                    world_xy_m=(float(s.world_xy_m[0]), float(s.world_xy_m[1])),
                )
            )
        return events

    def prune_missing(self, alive_track_ids: List[int]) -> None:
        alive = set(int(t) for t in alive_track_ids)
        to_del = [tid for tid in self._state.keys() if tid not in alive]
        for tid in to_del:
            del self._state[tid]

    def _select_speed(self, s: SpeedSample) -> float:
        if self._cfg.use_speed == "raw":
            return float(s.speed_mps_raw)
        if self._cfg.use_speed == "limited":
            return float(s.speed_mps_limited)
        return float(s.speed_mps_smoothed)

