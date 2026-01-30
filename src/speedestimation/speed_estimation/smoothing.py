from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Deque
from collections import deque
import numpy as np


@dataclass
class EmaSmoother:
    """
    Exponential Moving Average smoother for scalar speed values only.
    
    IMPORTANT: This smoother is designed for scalar speed values only.
    It should NOT be used for angles, curvature, or other directional quantities
    as this would introduce artifacts and incorrect smoothing behavior.
    """
    alpha: float
    max_gap_s: float
    _value: Optional[float] = None
    _t_last_s: Optional[float] = None

    def update(self, value: float, t_s: float) -> float:
        if self._value is None or self._t_last_s is None:
            self._value = float(value)
            self._t_last_s = float(t_s)
            return float(self._value)
        dt = float(t_s - self._t_last_s)
        if dt < 0.0 or dt > self.max_gap_s or value == 0.0:
            self._value = float(value)
            self._t_last_s = float(t_s)
            return float(self._value)
        a = float(self.alpha)
        self._value = a * float(value) + (1.0 - a) * float(self._value)
        self._t_last_s = float(t_s)
        return float(self._value)


@dataclass
class PolySmoother:
    degree: int
    window: int
    _t: Deque[float] = None
    _v: Deque[float] = None

    def __post_init__(self) -> None:
        self.degree = max(1, int(self.degree))
        self.window = max(self.degree + 1, int(self.window))
        self._t = deque(maxlen=int(self.window))
        self._v = deque(maxlen=int(self.window))

    def update(self, value: float, t_s: float) -> float:
        self._t.append(float(t_s))
        self._v.append(float(value))
        if len(self._t) < (self.degree + 1):
            return float(value)
        t = np.asarray(list(self._t), dtype=np.float64)
        v = np.asarray(list(self._v), dtype=np.float64)
        try:
            coeffs = np.polyfit(t, v, deg=int(self.degree))
            p = np.poly1d(coeffs)
            return float(p(float(t_s)))
        except Exception:
            return float(value)


@dataclass
class KalmanSmoother:
    process_noise: float = 1.0
    measurement_noise: float = 1.0
    error_covariance: float = 1.0
    _x: Optional[np.ndarray] = None  # State vector [pos, vel]
    _P: Optional[np.ndarray] = None  # Covariance matrix
    _t_last_s: Optional[float] = None

    def update(self, value: Optional[float], t_s: float) -> float:
        z = float(value)
        
        # Initialize
        if self._x is None or self._P is None or self._t_last_s is None:
            self._x = np.array([z, 0.0])
            self._P = np.eye(2) * self.error_covariance
            self._t_last_s = float(t_s)
            return z

        dt = float(t_s - self._t_last_s)
        self._t_last_s = float(t_s)
        
        if dt <= 0.0:
            return float(self._x[0])

        # Prediction Step
        # F = [[1, dt], [0, 1]]
        F = np.array([[1.0, dt], [0.0, 1.0]])
        # Q (Process Noise) - simplified discrete white noise acceleration model
        # Assumes constant velocity with some noise in acceleration
        # Q = [[dt^4/4, dt^3/2], [dt^3/2, dt^2]] * sigma_a^2
        q_std = self.process_noise
        Q = np.array([
            [0.25 * dt**4, 0.5 * dt**3],
            [0.5 * dt**3, dt**2]
        ]) * (q_std**2)
        
        self._x = F @ self._x
        self._P = F @ self._P @ F.T + Q

        # Update Step (only if we have a measurement)
        if value is not None:
            z = float(value)
            # H = [1, 0]
            H = np.array([[1.0, 0.0]])
            # R (Measurement Noise)
            R = np.array([[self.measurement_noise**2]])
            
            # S = HPH' + R
            S = H @ self._P @ H.T + R
            
            # K = PH'S^-1
            K = self._P @ H.T @ np.linalg.inv(S)
            
            # y = z - Hx (Residual)
            y = z - H @ self._x
            
            # x = x + Ky
            self._x = self._x + K @ y
            
            # P = (I - KH)P
            I = np.eye(2)
            self._P = (I - K @ H) @ self._P

        return float(self._x[0])

