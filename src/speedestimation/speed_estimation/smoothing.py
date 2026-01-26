from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class EmaSmoother:
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
        if dt < 0.0 or dt > self.max_gap_s:
            self._value = float(value)
            self._t_last_s = float(t_s)
            return float(self._value)
        a = float(self.alpha)
        self._value = a * float(value) + (1.0 - a) * float(self._value)
        self._t_last_s = float(t_s)
        return float(self._value)

