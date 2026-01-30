"""
Physically consistent turning estimation for vehicle trajectories.

This module provides robust turning metrics based on curvature and angular rate,
eliminating angle jumps and left/right flips through physical constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple, Optional
import logging
import math
import numpy as np


def heading_deg_from_delta(dx: float, dy: float) -> float:
    """Compute heading angle in degrees from delta x, y components."""
    ang = math.degrees(math.atan2(dy, dx))
    if ang < 0.0:
        ang += 360.0
    return float(ang)


@dataclass(frozen=True)
class TurningMetrics:
    heading_deg: float
    turn_angle_deg: float
    curvature_1pm: float
    arc_len_m: float


class TurningConfig:
    """Configuration for physically consistent turning estimation."""
    
    def __init__(self,
                 min_speed_mps: float = 0.5,
                 min_arc_length_m: float = 0.1,
                 max_angular_rate_deg_s: float = 45.0,
                 max_angular_accel_deg_s2: float = 180.0,
                 min_curvature_1pm: float = 0.005,
                 smoothing_window_min: int = 3,
                 smoothing_window_max: int = 7,
                 curvature_filter_sigma: float = 1.0):
        self.min_speed_mps = min_speed_mps
        self.min_arc_length_m = min_arc_length_m
        self.max_angular_rate_deg_s = max_angular_rate_deg_s
        self.max_angular_accel_deg_s2 = max_angular_accel_deg_s2
        self.min_curvature_1pm = min_curvature_1pm
        self.smoothing_window_min = smoothing_window_min
        self.smoothing_window_max = smoothing_window_max
        self.curvature_filter_sigma = curvature_filter_sigma


def _compute_derivatives(positions: np.ndarray, times: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute velocity and acceleration using central differences."""
    n = len(positions)
    velocities = np.zeros_like(positions, dtype=float)
    accelerations = np.zeros_like(positions, dtype=float)
    
    # Central difference for interior points
    for i in range(1, n-1):
        dt_central = times[i+1] - times[i-1]
        if dt_central > 0:
            velocities[i] = (positions[i+1] - positions[i-1]) / dt_central
    
    # Forward/backward difference for endpoints
    if n >= 2:
        dt_start = times[1] - times[0]
        dt_end = times[-1] - times[-2]
        
        if dt_start > 0:
            velocities[0] = (positions[1] - positions[0]) / dt_start
        if dt_end > 0:
            velocities[-1] = (positions[-1] - positions[-2]) / dt_end
    
    # Second derivative for acceleration
    if n >= 3:
        for i in range(1, n-1):
            dt_vel = times[i+1] - times[i-1]
            if dt_vel > 0 and i < n-2:
                accelerations[i] = (velocities[i+1] - velocities[i-1]) / dt_vel
    
    return velocities, accelerations


def _compute_curvature_and_angular_rate(velocities: np.ndarray, accelerations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute curvature and angular rate from velocity and acceleration."""
    n = len(velocities)
    curvatures = np.zeros(n, dtype=float)
    angular_rates = np.zeros(n, dtype=float)
    
    for i in range(n):
        vx, vy = velocities[i]
        ax, ay = accelerations[i]
        
        speed = math.sqrt(vx**2 + vy**2)
        if speed < 1e-6:
            continue
        
        # Cross product in 2D: v × a = vx*ay - vy*ax
        cross_product = vx * ay - vy * ax
        speed_cubed = speed**3
        
        if speed_cubed > 1e-6:
            curvature = cross_product / speed_cubed
            angular_rate_rad_s = speed * abs(curvature)
            
            curvatures[i] = curvature
            angular_rates[i] = math.degrees(angular_rate_rad_s)
    
    return curvatures, angular_rates


def _temporal_smoothing(values: np.ndarray, window_size: int) -> np.ndarray:
    """Apply temporal smoothing using moving average."""
    if len(values) < window_size or window_size < 3:
        return values.copy()
    
    smoothed = np.zeros_like(values)
    half_window = window_size // 2
    
    for i in range(len(values)):
        start_idx = max(0, i - half_window)
        end_idx = min(len(values), i + half_window + 1)
        smoothed[i] = np.mean(values[start_idx:end_idx])
    
    return smoothed


def _apply_physical_constraints(curvatures: np.ndarray, angular_rates: np.ndarray, 
                               speeds: np.ndarray, config: TurningConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Apply physical constraints to curvature and angular rate."""
    # Clamp angular rates to physically reasonable values
    angular_rates_clamped = np.clip(angular_rates, 0, config.max_angular_rate_deg_s)
    
    # Zero out measurements below minimum speed
    valid_speed_mask = speeds >= config.min_speed_mps
    curvatures_constrained = curvatures.copy()
    angular_rates_constrained = angular_rates_clamped.copy()
    
    curvatures_constrained[~valid_speed_mask] = 0.0
    angular_rates_constrained[~valid_speed_mask] = 0.0
    
    # Remove measurements below minimum curvature threshold
    low_curvature_mask = np.abs(curvatures_constrained) < config.min_curvature_1pm
    curvatures_constrained[low_curvature_mask] = 0.0
    
    return curvatures_constrained, angular_rates_constrained


def compute_turning_metrics_improved(dots: Sequence[Tuple[float, float, float]], 
                                   config: Optional[TurningConfig] = None) -> TurningMetrics:
    """
    Compute physically consistent turning metrics from trajectory dots.
    
    Uses curvature-based estimation with temporal smoothing and physical constraints
    to eliminate angle jumps and left/right flips.
    
    Args:
        dots: Sequence of (x, y, t) trajectory points in meters and seconds
        config: Configuration parameters for turning estimation
        
    Returns:
        TurningMetrics with heading, turn angle, curvature, and arc length
        
    Sign convention: Left turn = positive, Right turn = negative
    """
    logger = logging.getLogger(__name__)
    
    if config is None:
        config = TurningConfig()
    
    if len(dots) < 3:
        return TurningMetrics(heading_deg=float("nan"), turn_angle_deg=0.0, 
                            curvature_1pm=0.0, arc_len_m=0.0)
    
    # Convert to numpy arrays for efficient computation
    positions = np.array([(x, y) for x, y, t in dots], dtype=float)
    times = np.array([t for x, y, t in dots], dtype=float)
    
    # Step 1: Compute derivatives
    velocities, accelerations = _compute_derivatives(positions, times)
    
    # Step 2: Compute speeds for validation
    speeds = np.linalg.norm(velocities, axis=1)
    
    # Step 3: Compute curvature and angular rate
    curvatures, angular_rates = _compute_curvature_and_angular_rate(velocities, accelerations)
    
    # Step 4: Apply physical constraints
    curvatures_constrained, angular_rates_constrained = _apply_physical_constraints(
        curvatures, angular_rates, speeds, config)
    
    # Step 5: Determine smoothing window based on average speed
    avg_speed = np.mean(speeds[speeds >= config.min_speed_mps]) if np.any(speeds >= config.min_speed_mps) else 1.0
    smoothing_window = max(config.smoothing_window_min, 
                           min(config.smoothing_window_max, 
                               int(2.0 / avg_speed + config.smoothing_window_min)))
    
    # Step 6: Apply temporal smoothing
    curvatures_smooth = _temporal_smoothing(curvatures_constrained, smoothing_window)
    angular_rates_smooth = _temporal_smoothing(angular_rates_constrained, smoothing_window)
    
    # Step 7: Find most recent valid measurement
    latest_valid_idx = -1
    for i in range(len(dots)-1, -1, -1):
        if (speeds[i] >= config.min_speed_mps and 
            abs(curvatures_smooth[i]) >= config.min_curvature_1pm):
            latest_valid_idx = i
            break
    
    if latest_valid_idx == -1:
        # No valid turning measurements, return straight motion
        final_heading = math.degrees(math.atan2(velocities[-1,1], velocities[-1,0]))
        if final_heading < 0:
            final_heading += 360.0
        
        return TurningMetrics(
            heading_deg=float(final_heading),
            turn_angle_deg=0.0,
            curvature_1pm=0.0,
            arc_len_m=float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)))
        )
    
    # Step 8: Compute final metrics
    final_curvature = curvatures_smooth[latest_valid_idx]
    final_angular_rate = angular_rates_smooth[latest_valid_idx]
    
    # Heading from final velocity vector
    final_heading = math.degrees(math.atan2(velocities[-1,1], velocities[-1,0]))
    if final_heading < 0:
        final_heading += 360.0
    
    # FIXED: Turn angle is properly integrated angular rate over trajectory duration
    # Instead of assuming constant angular rate, use discrete integration
    dt = np.diff(times)  # Time differences between consecutive points
    # Integrate angular rates (excluding first point since dt is n-1)
    turn_angle = np.sum(angular_rates_smooth[1:] * dt)
    
    # Apply sign based on overall curvature direction (left = positive)
    mean_curvature = np.mean(curvatures_smooth)
    if mean_curvature < 0:
        turn_angle = -turn_angle
    
    # Compute total arc length
    arc_length = float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)))
    
    # FIXED: Add arc length guard to enforce zero motion when dots are near-overlapping
    # This directly aligns with the "accumulated dots" logic requirement
    if arc_length < config.min_arc_length_m:
        # Near-overlapping dots imply zero motion - enforce physical constraints
        turn_angle = 0.0
        final_curvature = 0.0
        final_angular_rate = 0.0
        # FIXED: Set heading to NaN when stopped to avoid jitter-driven heading jumps
        final_heading = float("nan")
        logger.debug(
            "arc_length_guard_enforced: arc_length=%.6fm < %.6fm → zero motion enforced",
            arc_length, config.min_arc_length_m
        )
    
    # FIXED: Also check if vehicle is stopped by speed threshold and set heading to NaN
    elif avg_speed < config.min_speed_mps:
        # Vehicle is essentially stopped - heading is meaningless
        final_heading = float("nan")
    
    # Debug logging
    logger.debug(
        "turning_metrics_improved: heading=%.2f° turn_angle=%.2f° "
        "curvature=%.6f 1/m angular_rate=%.2f°/s arc_length=%.3fm "
        "avg_speed=%.2f m/s valid_points=%d/%d",
        final_heading, turn_angle, final_curvature, final_angular_rate,
        arc_length, avg_speed, np.sum(speeds >= config.min_speed_mps), len(dots)
    )
    
    return TurningMetrics(
        heading_deg=float(final_heading),
        turn_angle_deg=float(turn_angle),
        curvature_1pm=float(final_curvature),
        arc_len_m=arc_length
    )


def _detect_zero_velocity_segments_authoritative(
    positions: np.ndarray,
    times: np.ndarray,
    movement_threshold_m: float,
    min_time_window_s: float,
    min_arc_length_m: float
) -> np.ndarray:
    """
    Detect zero-velocity segments using authoritative stop detection rule.
    
    Args:
        positions: Array of (x, y) positions
        times: Array of timestamps
        movement_threshold_m: Minimum movement threshold in meters
        min_time_window_s: Minimum time window for stop detection
        min_arc_length_m: Minimum arc length threshold
        
    Returns:
        Boolean array indicating zero-velocity segments
    """
    if len(positions) < 2 or len(times) < 2:
        return np.zeros(len(positions), dtype=bool)
    
    # Compute cumulative arc length from start
    distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    cumulative_dist = np.concatenate([[0.0], np.cumsum(distances)])
    
    # Compute time differences
    time_diffs = np.diff(times)
    
    # Initialize zero-velocity mask
    zero_velocity_mask = np.zeros(len(positions), dtype=bool)
    
    # For each point, check if it's in a zero-velocity segment
    for i in range(len(positions)):
        if i == 0:
            zero_velocity_mask[i] = False
            continue
            
        # Look backward in time for min_time_window_s
        current_time = times[i]
        window_start_idx = i
        
        # Find the start of the time window
        for j in range(i-1, -1, -1):
            if current_time - times[j] >= min_time_window_s:
                window_start_idx = j
                break
            else:
                window_start_idx = j
        
        # Calculate movement in the time window
        if window_start_idx < i:
            movement_in_window = cumulative_dist[i] - cumulative_dist[window_start_idx]
            time_in_window = times[i] - times[window_start_idx]
            
            # Authoritative stop rule: zero velocity if:
            # 1. Movement in window is below threshold
            # 2. Time window is sufficient
            # 3. Arc length is below minimum
            if (movement_in_window < movement_threshold_m and 
                time_in_window >= min_time_window_s and
                cumulative_dist[i] < min_arc_length_m):
                zero_velocity_mask[i] = True
    
    return zero_velocity_mask


def _detect_zero_velocity_segments(
    positions: np.ndarray,
    times: np.ndarray,
    movement_threshold_m: float,
    min_time_window_s: float
) -> np.ndarray:
    """
    Simplified zero-velocity detection (without arc length constraint).
    
    Args:
        positions: Array of (x, y) positions
        times: Array of timestamps
        movement_threshold_m: Minimum movement threshold in meters
        min_time_window_s: Minimum time window for stop detection
        
    Returns:
        Boolean array indicating zero-velocity segments
    """
    return _detect_zero_velocity_segments_authoritative(
        positions, times, movement_threshold_m, min_time_window_s, 0.0
    )


def classify_turn_severity(curvature_1pm: float, angular_rate_deg_s: float, speed_mps: float) -> str:
    """
    Classify turn severity based on curvature and angular rate.
    
    Accounts for speed effects: same curvature produces higher angular rates at higher speeds.
    
    Args:
        curvature_1pm: Signed curvature in 1/meters
        angular_rate_deg_s: Angular rate in degrees/second
        speed_mps: Vehicle speed in meters/second
        
    Returns:
        Classification: "straight", "gentle", "moderate", or "sharp"
    """
    # Normalize for speed effects
    # At higher speeds, same curvature produces higher angular rates
    normalized_curvature = abs(curvature_1pm) * (1.0 + speed_mps / 20.0)  # Empirical scaling
    
    if normalized_curvature < 0.005:  # R > 200m equivalent
        return "straight"
    elif normalized_curvature < 0.02 and abs(angular_rate_deg_s) < 15:  # R > 50m, ω < 15°/s
        return "gentle"
    elif normalized_curvature < 0.1 and abs(angular_rate_deg_s) < 45:  # R > 10m, ω < 45°/s
        return "moderate"
    else:
        return "sharp"


def get_turn_radius_m(curvature_1pm: float) -> Optional[float]:
    """Convert curvature to turn radius, handling edge cases."""
    if abs(curvature_1pm) < 1e-6:
        return None  # Straight line (infinite radius)
    return 1.0 / abs(curvature_1pm)