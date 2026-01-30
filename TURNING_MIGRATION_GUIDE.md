# Turning Estimation Migration Guide

## Overview

This guide explains how to migrate from the original turning estimation to the improved physically-consistent implementation that eliminates angle jumps and left/right flips.

## Key Improvements

1. **Curvature-based estimation**: Uses vehicle dynamics equations instead of angle differences
2. **Physical constraints**: Enforces maximum turn rates and speed thresholds  
3. **Temporal smoothing**: Reduces noise while preserving physical relationships
4. **Robust validation**: Handles low-speed and noisy measurements gracefully

## Migration Steps

### Step 1: Update Imports

Replace the original turning import in your speed estimation module:

```python
# OLD (in estimator.py)
from speedestimation.turning_model.turning import compute_turning_metrics

# NEW 
from speedestimation.turning_model.turning_improved import (
    compute_turning_metrics_improved, 
    TurningConfig,
    classify_turn_severity
)
```

### Step 2: Add Configuration Parameters

Add turning configuration to your speed estimator config:

```python
# In SpeedEstimatorConfig.__init__ or from_dict()
self.turning_config = TurningConfig(
    min_speed_mps=float(turning.get("min_speed_mps", 0.5)),
    min_arc_length_m=float(turning.get("min_arc_length_m", 0.1)),
    max_angular_rate_deg_s=float(turning.get("max_angular_rate_deg_s", 45.0)),
    max_angular_accel_deg_s2=float(turning.get("max_angular_accel_deg_s2", 180.0)),
    min_curvature_1pm=float(turning.get("min_curvature_1pm", 0.005)),
    smoothing_window_min=int(turning.get("smoothing_window_min", 3)),
    smoothing_window_max=int(turning.get("smoothing_window_max", 7))
)
```

### Step 3: Replace Turning Computation

Update the turning computation in your speed estimation loop:

```python
# OLD (in estimator.py around line 250)
turning = compute_turning_metrics(h.dots)

# NEW
turning = compute_turning_metrics_improved(h.dots, self._cfg.turning_config)
```

### Step 4: Optional - Add Turn Severity Classification

Add turn severity information to your output:

```python
# After computing turning metrics
if turning.curvature_1pm != 0.0 and v_smooth > 0:
    turn_severity = classify_turn_severity(
        turning.curvature_1pm,
        abs(turning.turn_angle_deg) / max(0.1, dt),
        v_smooth
    )
    # Add to metadata or output as needed
```

## Configuration Recommendations

### Urban Traffic (Low Speed, High Maneuvering)
```yaml
turning:
  min_speed_mps: 0.3          # Lower threshold for stop-and-go
  min_arc_length_m: 0.05      # Shorter minimum segments
  max_angular_rate_deg_s: 60.0  # Higher for tight turns
  min_curvature_1pm: 0.003    # More sensitive to gentle curves
```

### Highway Traffic (High Speed, Gentle Curves)
```yaml
turning:
  min_speed_mps: 1.0          # Higher minimum speed
  min_arc_length_m: 0.2       # Longer segments for stability
  max_angular_rate_deg_s: 30.0  # Lower for gentle highway curves
  min_curvature_1pm: 0.008    # Less sensitive to minor deviations
```

### General Purpose (Recommended Defaults)
```yaml
turning:
  min_speed_mps: 0.5          # Reasonable minimum for vehicle motion
  min_arc_length_m: 0.1       # Standard segment length
  max_angular_rate_deg_s: 45.0  # Typical maximum for normal driving
  max_angular_accel_deg_s2: 180.0  # Physically reasonable limit
  min_curvature_1pm: 0.005    # Filters out measurement noise
  smoothing_window_min: 3     # Minimum smoothing
  smoothing_window_max: 7     # Maximum smoothing
```

## Validation Checklist

After migration, verify:

- [ ] No more angle jumps > 90° between consecutive frames
- [ ] Turn direction (left/right) remains consistent during continuous turns
- [ ] Low-speed measurements are properly rejected
- [ ] Turn severity classification matches visual observation
- [ ] Performance impact is acceptable (< 10% increase in processing time)

## Backwards Compatibility

The improved implementation maintains the same `TurningMetrics` interface:

```python
@dataclass(frozen=True)
class TurningMetrics:
    heading_deg: float      # Same as before
    turn_angle_deg: float   # Now physically consistent
    curvature_1pm: float    # Same meaning, more accurate
    arc_len_m: float        # Same as before
```

All existing code using these fields will continue to work without modification.

## Troubleshooting

### Issue: All turns classified as "straight"
**Solution**: Reduce `min_curvature_1pm` or `min_speed_mps` thresholds

### Issue: Turn angle still shows small fluctuations
**Solution**: Increase `smoothing_window_max` or reduce `max_angular_rate_deg_s`

### Issue: Sharp turns not detected
**Solution**: Increase `max_angular_rate_deg_s` and reduce `min_curvature_1pm`

### Issue: Performance degradation
**Solution**: Reduce `smoothing_window_max` or implement numpy vectorization

## Performance Considerations

- The improved method uses ~2-3x more CPU than the original
- For real-time applications, consider:
  - Reducing smoothing window size
  - Using vectorized numpy operations
  - Processing turning estimation less frequently than speed estimation
  - Implementing curvature caching for repeated calculations

## Example Integration

See `test_turning_improvements.py` for complete working examples of both methods and comparison plots.