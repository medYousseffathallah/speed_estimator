"""
Physically consistent turning estimation for vehicle trajectories.

Production-ready module providing robust turning metrics based on curvature 
and angular rate, eliminating angle jumps and left/right flips through physical constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple, Optional
import logging
import math
import numpy as np

logger = logging.getLogger(__name__)


def heading_deg_from_delta(dx: float, dy: float) -> float:
    """Compute heading angle in degrees from delta x, y components."""
    if not (isinstance(dx, (int, float)) and isinstance(dy, (int, float))):
        logger.warning("Invalid input types for heading calculation: dx=%s, dy=%s", type(dx), type(dy))
        return float("nan")
    
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
                 curvature_filter_sigma: float = 1.0,
                 enable_adaptive_thresholds: bool = True,
                 low_speed_threshold_mps: float = 2.0,
                 high_speed_threshold_mps: float = 15.0):
        
        # Input validation
        if min_speed_mps < 0:
            raise ValueError(f"min_speed_mps must be non-negative, got {min_speed_mps}")
        if min_arc_length_m < 0:
            raise ValueError(f"min_arc_length_m must be non-negative, got {min_arc_length_m}")
        if max_angular_rate_deg_s <= 0:
            raise ValueError(f"max_angular_rate_deg_s must be positive, got {max_angular_rate_deg_s}")
        if max_angular_accel_deg_s2 <= 0:
            raise ValueError(f"max_angular_accel_deg_s2 must be positive, got {max_angular_accel_deg_s2}")
        if min_curvature_1pm < 0:
            raise ValueError(f"min_curvature_1pm must be non-negative, got {min_curvature_1pm}")
        if smoothing_window_min < 2:
            raise ValueError(f"smoothing_window_min must be >= 2, got {smoothing_window_min}")
        if smoothing_window_max < smoothing_window_min:
            raise ValueError(f"smoothing_window_max ({smoothing_window_max}) must be >= smoothing_window_min ({smoothing_window_min})")
        if curvature_filter_sigma <= 0:
            raise ValueError(f"curvature_filter_sigma must be positive, got {curvature_filter_sigma}")
        if low_speed_threshold_mps < 0:
            raise ValueError(f"low_speed_threshold_mps must be non-negative, got {low_speed_threshold_mps}")
        if high_speed_threshold_mps < low_speed_threshold_mps:
            raise ValueError(f"high_speed_threshold_mps ({high_speed_threshold_mps}) must be >= low_speed_threshold_mps ({low_speed_threshold_mps})")
        
        self.min_speed_mps = min_speed_mps
        self.min_arc_length_m = min_arc_length_m
        self.max_angular_rate_deg_s = max_angular_rate_deg_s
        self.max_angular_accel_deg_s2 = max_angular_accel_deg_s2
        self.min_curvature_1pm = min_curvature_1pm
        self.smoothing_window_min = smoothing_window_min
        self.smoothing_window_max = smoothing_window_max
        self.curvature_filter_sigma = curvature_filter_sigma
        self.enable_adaptive_thresholds = enable_adaptive_thresholds
        self.low_speed_threshold_mps = low_speed_threshold_mps
        self.high_speed_threshold_mps = high_speed_threshold_mps
    
    def get_speed_dependent_min_curvature(self, speed_mps: float) -> float:
        """Get minimum curvature threshold based on speed for adaptive filtering."""
        if not isinstance(speed_mps, (int, float)) or speed_mps < 0:
            logger.warning("Invalid speed for adaptive curvature: %s", speed_mps)
            return self.min_curvature_1pm
        
        if not self.enable_adaptive_thresholds:
            return self.min_curvature_1pm
        
        # Higher speeds require higher minimum curvature to be considered turning
        if speed_mps < self.low_speed_threshold_mps:
            return self.min_curvature_1pm * 0.5  # More sensitive at low speeds
        elif speed_mps > self.high_speed_threshold_mps:
            return self.min_curvature_1pm * 2.0  # Less sensitive at high speeds
        else:
            # Linear interpolation between thresholds
            speed_ratio = (speed_mps - self.low_speed_threshold_mps) / (self.high_speed_threshold_mps - self.low_speed_threshold_mps)
            return self.min_curvature_1pm * (0.5 + speed_ratio * 1.5)


def _validate_trajectory_input(dots: Sequence[Tuple[float, float, float]]) -> bool:
    """Validate trajectory input data."""
    if not dots:
        logger.warning("Empty trajectory provided")
        return False
    
    if len(dots) < 2:
        logger.warning("Trajectory too short: %d points", len(dots))
        return False
    
    # Check for valid numeric values
    for i, (x, y, t) in enumerate(dots):
        if not all(isinstance(val, (int, float)) for val in [x, y, t]):
            logger.warning("Invalid numeric values at point %d: x=%s, y=%s, t=%s", i, type(x), type(y), type(t))
            return False
        if any(math.isnan(val) or math.isinf(val) for val in [x, y, t]):
            logger.warning("NaN or infinite values at point %d: x=%s, y=%s, t=%s", i, x, y, t)
            return False
    
    # Check temporal consistency
    times = [t for _, _, t in dots]
    for i in range(1, len(times)):
        if times[i] <= times[i-1]:
            logger.warning("Non-increasing timestamps at indices %d-%d: %f <= %f", i-1, i, times[i], times[i-1])
            return False
    
    return True


def _compute_derivatives(positions: np.ndarray, times: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute velocity and acceleration using central differences with safety checks."""
    n = len(positions)
    if n < 2:
        logger.error("Insufficient points for derivative computation: %d", n)
        return np.zeros_like(positions, dtype=np.float32), np.zeros_like(positions, dtype=np.float32)
    
    velocities = np.zeros_like(positions, dtype=np.float32)
    accelerations = np.zeros_like(positions, dtype=np.float32)
    
    try:
        # Central difference for interior points
        for i in range(1, n-1):
            dt_central = times[i+1] - times[i-1]
            if dt_central > 1e-6:  # Avoid division by very small numbers
                velocities[i] = (positions[i+1] - positions[i-1]) / dt_central
            else:
                logger.warning("Very small time interval at index %d: %e", i, dt_central)
        
        # Forward/backward difference for endpoints
        if n >= 2:
            dt_start = times[1] - times[0]
            dt_end = times[-1] - times[-2]
            
            if dt_start > 1e-6:
                velocities[0] = (positions[1] - positions[0]) / dt_start
            else:
                logger.warning("Very small start time interval: %e", dt_start)
                
            if dt_end > 1e-6:
                velocities[-1] = (positions[-1] - positions[-2]) / dt_end
            else:
                logger.warning("Very small end time interval: %e", dt_end)
        
        # Second derivative for acceleration (only if we have enough points)
        if n >= 3:
            for i in range(1, n-1):
                dt_vel = times[i+1] - times[i-1]
                if dt_vel > 1e-6 and i < n-2:
                    accelerations[i] = (velocities[i+1] - velocities[i-1]) / dt_vel
    
    except Exception as e:
        logger.error("Error computing derivatives: %s", str(e))
        return np.zeros_like(positions, dtype=np.float32), np.zeros_like(positions, dtype=np.float32)
    
    return velocities, accelerations


def _compute_curvature_and_angular_rate(velocities: np.ndarray, accelerations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute curvature and angular rate from velocity and acceleration with safety checks."""
    n = len(velocities)
    if n == 0:
        logger.error("Empty velocities array")
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    
    curvatures = np.zeros(n, dtype=np.float32)
    angular_rates = np.zeros(n, dtype=np.float32)
    
    try:
        for i in range(n):
            vx, vy = velocities[i]
            ax, ay = accelerations[i]
            
            # Compute speed with safety check
            speed_sq = vx**2 + vy**2
            if speed_sq < 1e-12:  # ~1e-6 m/s threshold squared
                continue
            
            speed = math.sqrt(speed_sq)
            speed_cubed = speed**3
            
            if speed_cubed < 1e-18:  # Very small speed cubed
                continue
            
            # Cross product in 2D: v × a = vx*ay - vy*ax
            cross_product = vx * ay - vy * ax
            
            # Clamp cross product to avoid numerical issues
            max_cross = speed_cubed * 10.0  # Reasonable upper bound
            cross_product = np.clip(cross_product, -max_cross, max_cross)
            
            curvature = cross_product / speed_cubed
            angular_rate_rad_s = speed * abs(curvature)
            
            # Validate results
            if math.isfinite(curvature) and math.isfinite(angular_rate_rad_s):
                curvatures[i] = curvature
                angular_rates[i] = math.degrees(angular_rate_rad_s)
            else:
                logger.warning("Non-finite curvature/angular_rate at index %d: curvature=%s, angular_rate=%s", 
                             i, curvature, angular_rate_rad_s)
    
    except Exception as e:
        logger.error("Error computing curvature and angular rate: %s", str(e))
        return np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)
    
    return curvatures, angular_rates


def _temporal_smoothing(values: np.ndarray, window_size: int) -> np.ndarray:
    """Apply temporal smoothing using moving average with safety checks."""
    n = len(values)
    if n == 0:
        logger.error("Empty values array for smoothing")
        return np.zeros(0, dtype=np.float32)
    
    if window_size < 3:
        logger.warning("Window size too small for smoothing: %d, returning original values", window_size)
        return values.astype(np.float32)
    
    if n < window_size:
        logger.warning("Not enough points for smoothing: %d < %d, returning original values", n, window_size)
        return values.astype(np.float32)
    
    try:
        smoothed = np.zeros_like(values, dtype=np.float32)
        half_window = window_size // 2
        
        for i in range(n):
            start_idx = max(0, i - half_window)
            end_idx = min(n, i + half_window + 1)
            window_values = values[start_idx:end_idx]
            
            # Only smooth if we have valid values in the window
            valid_values = window_values[np.isfinite(window_values)]
            if len(valid_values) > 0:
                smoothed[i] = np.mean(valid_values)
            else:
                smoothed[i] = values[i] if math.isfinite(values[i]) else 0.0
        
        return smoothed
    
    except Exception as e:
        logger.error("Error in temporal smoothing: %s", str(e))
        return values.astype(np.float32)


def _apply_physical_constraints(curvatures: np.ndarray, angular_rates: np.ndarray, 
                               speeds: np.ndarray, config: TurningConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Apply physical constraints to curvature and angular rate with safety checks."""
    if len(curvatures) == 0 or len(angular_rates) == 0 or len(speeds) == 0:
        logger.error("Empty arrays in physical constraints")
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    
    try:
        # Clamp angular rates to physically reasonable values
        angular_rates_clamped = np.clip(angular_rates, 0, config.max_angular_rate_deg_s)
        
        # Zero out measurements below minimum speed
        valid_speed_mask = speeds >= config.min_speed_mps
        curvatures_constrained = curvatures.copy().astype(np.float32)
        angular_rates_constrained = angular_rates_clamped.astype(np.float32)
        
        # Apply speed-based filtering
        curvatures_constrained[~valid_speed_mask] = 0.0
        angular_rates_constrained[~valid_speed_mask] = 0.0
        
        # Apply adaptive curvature thresholding
        for i in range(len(curvatures_constrained)):
            if valid_speed_mask[i] and abs(curvatures_constrained[i]) > 0:
                speed_dependent_threshold = config.get_speed_dependent_min_curvature(speeds[i])
                if abs(curvatures_constrained[i]) < speed_dependent_threshold:
                    curvatures_constrained[i] = 0.0
        
        return curvatures_constrained, angular_rates_constrained
    
    except Exception as e:
        logger.error("Error applying physical constraints: %s", str(e))
        return np.zeros_like(curvatures, dtype=np.float32), np.zeros_like(angular_rates, dtype=np.float32)


def compute_turning_metrics(dots: Sequence[Tuple[float, float, float]], 
                          config: Optional[TurningConfig] = None) -> TurningMetrics:
    """
    Compute physically consistent turning metrics from trajectory dots.
    
    Production-ready implementation with comprehensive input validation,
    error handling, and physical constraints to eliminate angle jumps 
    and left/right flips.
    
    Args:
        dots: Sequence of (x, y, t) trajectory points in meters and seconds
        config: Configuration parameters for turning estimation
        
    Returns:
        TurningMetrics with heading, turn angle, curvature, and arc length
        
    Sign convention: Left turn = positive, Right turn = negative
        
    Safety features:
        - Input validation for NaN/inf values
        - Temporal consistency checks
        - Safe numeric operations with division guards
        - Physical constraint enforcement
        - Memory-efficient float32 arrays
        - Comprehensive error handling
    """
    logger = logging.getLogger(__name__)
    
    # Use default config if none provided
    if config is None:
        config = TurningConfig()
    
    # Input validation
    if not _validate_trajectory_input(dots):
        return TurningMetrics(heading_deg=float("nan"), turn_angle_deg=0.0, 
                            curvature_1pm=0.0, arc_len_m=0.0)
    
    if len(dots) < 3:
        logger.warning("Trajectory too short for reliable turning metrics: %d points", len(dots))
        return TurningMetrics(heading_deg=float("nan"), turn_angle_deg=0.0, 
                            curvature_1pm=0.0, arc_len_m=0.0)
    
    try:
        # Convert to numpy arrays for efficient computation (use float32 for memory efficiency)
        positions = np.array([(x, y) for x, y, t in dots], dtype=np.float32)
        times = np.array([t for x, y, t in dots], dtype=np.float32)
        
        # Step 1: Compute derivatives
        velocities, accelerations = _compute_derivatives(positions, times)
        
        # Step 2: Compute speeds for validation
        speeds = np.linalg.norm(velocities, axis=1)
        
        # Check for near-zero motion
        if np.all(speeds < config.min_speed_mps):
            logger.debug("All speeds below minimum threshold: max_speed=%.3f < %.3f", 
                        np.max(speeds), config.min_speed_mps)
            return TurningMetrics(heading_deg=float("nan"), turn_angle_deg=0.0, 
                                curvature_1pm=0.0, arc_len_m=0.0)
        
        # Step 3: Compute curvature and angular rate
        curvatures, angular_rates = _compute_curvature_and_angular_rate(velocities, accelerations)
        
        # Step 4: Apply physical constraints
        curvatures_constrained, angular_rates_constrained = _apply_physical_constraints(
            curvatures, angular_rates, speeds, config)
        
        # Step 5: Determine smoothing window based on average speed
        valid_speeds = speeds[speeds >= config.min_speed_mps]
        if len(valid_speeds) == 0:
            logger.warning("No valid speeds for smoothing window calculation")
            return TurningMetrics(heading_deg=float("nan"), turn_angle_deg=0.0, 
                                curvature_1pm=0.0, arc_len_m=0.0)
        
        avg_speed = np.mean(valid_speeds)
        # Adaptive smoothing window: larger windows for lower speeds
        smoothing_window = max(config.smoothing_window_min, 
                             min(config.smoothing_window_max, 
                                 int(5.0 / avg_speed + config.smoothing_window_min)))
        
        # Step 6: Apply temporal smoothing
        curvatures_smooth = _temporal_smoothing(curvatures_constrained, smoothing_window)
        angular_rates_smooth = _temporal_smoothing(angular_rates_constrained, smoothing_window)
        
        # Step 7: Find most recent valid measurement
        latest_valid_idx = -1
        for i in range(len(dots)-1, -1, -1):
            if (speeds[i] >= config.min_speed_mps and 
                abs(curvatures_smooth[i]) >= config.min_curvature_1pm and
                math.isfinite(curvatures_smooth[i]) and math.isfinite(angular_rates_smooth[i])):
                latest_valid_idx = i
                break
        
        if latest_valid_idx == -1:
            # No valid turning measurements, return straight motion
            final_heading = math.degrees(math.atan2(velocities[-1,1], velocities[-1,0]))
            if final_heading < 0:
                final_heading += 360.0
            
            arc_length = float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)))
            
            return TurningMetrics(
                heading_deg=float(final_heading),
                turn_angle_deg=0.0,
                curvature_1pm=0.0,
                arc_len_m=arc_length
            )
        
        # Step 8: Compute final metrics
        final_curvature = curvatures_smooth[latest_valid_idx]
        final_angular_rate = angular_rates_smooth[latest_valid_idx]
        
        # Heading from final velocity vector
        final_heading = math.degrees(math.atan2(velocities[-1,1], velocities[-1,0]))
        if final_heading < 0:
            final_heading += 360.0
        
        # Turn angle integration with proper sign handling
        dt = np.diff(times)
        if len(dt) > 0 and len(angular_rates_smooth) > 1:
            # Integrate angular rates (excluding first point since dt is n-1)
            turn_angle = np.sum(angular_rates_smooth[1:] * dt)
            
            # Apply sign based on overall curvature direction (left = positive)
            mean_curvature = np.mean(curvatures_smooth[np.isfinite(curvatures_smooth)])
            if mean_curvature < 0:
                turn_angle = -turn_angle
        else:
            turn_angle = 0.0
        
        # Compute total arc length
        arc_length = float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)))
        
        # Arc length guard for near-overlapping trajectory points
        if arc_length < config.min_arc_length_m:
            logger.debug(
                "Arc length below minimum threshold: %.6fm < %.6fm → zero motion enforced",
                arc_length, config.min_arc_length_m
            )
            turn_angle = 0.0
            final_curvature = 0.0
            final_angular_rate = 0.0
            final_heading = float("nan")
        
        # Heading guard for very low speeds
        elif avg_speed < config.min_speed_mps:
            logger.debug(
                "Average speed below minimum threshold: %.3f < %.3f → heading set to NaN",
                avg_speed, config.min_speed_mps
            )
            final_heading = float("nan")
        
        # Debug logging for production monitoring
        logger.debug(
            "turning_metrics_computed: heading=%.2f° turn_angle=%.2f° "
            "curvature=%.6f 1/m angular_rate=%.2f°/s arc_length=%.3fm "
            "avg_speed=%.2f m/s valid_points=%d/%d smoothing_window=%d",
            final_heading, turn_angle, final_curvature, final_angular_rate,
            arc_length, avg_speed, np.sum(speeds >= config.min_speed_mps), len(dots),
            smoothing_window
        )
        
        return TurningMetrics(
            heading_deg=float(final_heading),
            turn_angle_deg=float(turn_angle),
            curvature_1pm=float(final_curvature),
            arc_len_m=arc_length
        )
    
    except Exception as e:
        logger.error("Unexpected error in compute_turning_metrics: %s", str(e))
        return TurningMetrics(heading_deg=float("nan"), turn_angle_deg=0.0, 
                            curvature_1pm=0.0, arc_len_m=0.0)


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
        
    Safety features:
        - Input validation for NaN/inf values
        - Speed-dependent thresholds
        - Bounded classification ranges
    """
    logger = logging.getLogger(__name__)
    
    # Input validation
    if not all(isinstance(val, (int, float)) and math.isfinite(val) 
               for val in [curvature_1pm, angular_rate_deg_s, speed_mps]):
        logger.warning("Invalid inputs for turn classification: curvature=%s, angular_rate=%s, speed=%s",
                      curvature_1pm, angular_rate_deg_s, speed_mps)
        return "straight"
    
    if speed_mps < 0:
        logger.warning("Negative speed in turn classification: %f", speed_mps)
        return "straight"
    
    try:
        # Normalize for speed effects
        if speed_mps > 0:
            normalized_curvature = abs(curvature_1pm) * math.sqrt(speed_mps)
            normalized_angular_rate = abs(angular_rate_deg_s) / max(1.0, speed_mps)
        else:
            normalized_curvature = abs(curvature_1pm)
            normalized_angular_rate = abs(angular_rate_deg_s)
        
        # Speed-dependent thresholds
        if speed_mps < 2.0:  # Low speed
            curvature_thresholds = [0.01, 0.03, 0.08]  # More sensitive
            angular_rate_thresholds = [5.0, 15.0, 35.0]
        elif speed_mps > 15.0:  # High speed
            curvature_thresholds = [0.005, 0.015, 0.04]  # Less sensitive
            angular_rate_thresholds = [3.0, 10.0, 25.0]
        else:  # Medium speed
            curvature_thresholds = [0.008, 0.02, 0.06]
            angular_rate_thresholds = [4.0, 12.0, 30.0]
        
        # Classification based on both curvature and angular rate
        if (normalized_curvature < curvature_thresholds[0] and 
            normalized_angular_rate < angular_rate_thresholds[0]):
            return "straight"
        elif (normalized_curvature < curvature_thresholds[1] and 
              normalized_angular_rate < angular_rate_thresholds[1]):
            return "gentle"
        elif (normalized_curvature < curvature_thresholds[2] and 
              normalized_angular_rate < angular_rate_thresholds[2]):
            return "moderate"
        else:
            return "sharp"
    
    except Exception as e:
        logger.error("Error in turn classification: %s", str(e))
        return "straight"