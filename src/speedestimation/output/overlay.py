from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from speedestimation.speed_estimation.units import mps_to_kmh
from speedestimation.utils.types import SpeedSample, TrackState


@dataclass
class OverlayRenderer:
    units: str = "kmh"

    def draw(
        self,
        frame_bgr: np.ndarray,
        tracks: List[TrackState],
        latest_speeds: Dict[int, SpeedSample],
        trails_by_track: Optional[Dict[int, List[Tuple[float, float]]]] = None,
        draw_vectors: bool = False,
        show_heading: bool = False,
        show_turn_angle: bool = False,
        trail_color_bgr: Tuple[int, int, int] = (0, 255, 255),
        vector_color_bgr: Tuple[int, int, int] = (0, 0, 255),
        trail_thickness: int = 2,
        vector_thickness: int = 2,
        vector_tip_length: float = 0.3,
    ) -> np.ndarray:
        img = frame_bgr
        for ts in tracks:
            x1, y1, x2, y2 = ts.bbox_xyxy
            p1 = (int(x1), int(y1))
            p2 = (int(x2), int(y2))
            cv2.rectangle(img, p1, p2, (0, 255, 0), 2)

            sp = latest_speeds.get(ts.track_id)
            if trails_by_track is not None:
                pts = trails_by_track.get(ts.track_id, [])
                if len(pts) >= 2:
                    poly = np.asarray([(int(x), int(y)) for x, y in pts], dtype=np.int32).reshape(-1, 1, 2)
                    cv2.polylines(img, [poly], isClosed=False, color=trail_color_bgr, thickness=int(trail_thickness))
                    if draw_vectors:
                        a = (int(pts[-2][0]), int(pts[-2][1]))
                        b = (int(pts[-1][0]), int(pts[-1][1]))
                        cv2.arrowedLine(
                            img,
                            a,
                            b,
                            vector_color_bgr,
                            int(vector_thickness),
                            tipLength=float(vector_tip_length),
                        )
                for x, y in pts:
                    cv2.circle(img, (int(x), int(y)), max(2, int(trail_thickness) + 1), trail_color_bgr, -1)

            label = f"id={ts.track_id} {ts.class_name}"
            if sp is not None:
                v = sp.speed_mps_smoothed
                if self.units == "kmh":
                    label += f" {mps_to_kmh(v):.1f} km/h"
                else:
                    label += f" {v:.2f} m/s"
                if show_heading:
                    label += f" hdg={sp.heading_deg:.1f}°"
                if show_turn_angle:
                    applied = bool(sp.metadata.get("turn_applied", 0.0) > 0.5)
                    state = "applied" if applied else ("detected" if sp.turn_angle_deg > 0.0 else "none")
                    # Add direction indicator based on signed angle if available
                    signed_angle = sp.metadata.get("turn_angle_signed_deg", 0.0)
                    direction = ""
                    if abs(signed_angle) > 0.1:  # Only show direction for meaningful angles
                        direction = " L" if signed_angle > 0 else " R"
                    label += f" turn={sp.turn_angle_deg:.1f}°{direction} ({state})"
            cv2.putText(img, label, (p1[0], max(0, p1[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return img

