"""Top-1 minus Top-2 probability margin baseline router."""
from __future__ import annotations
import numpy as np


class MarginRouter:
    """Invoke expert when student top-1 vs top-2 margin is smaller than a threshold."""

    def __init__(self, threshold: float = 0.35) -> None:
        self.threshold = threshold

    def fit_threshold_for_call_rate(self, cal_margins: np.ndarray, target_call_rate: float) -> float:
        if cal_margins.size == 0 or target_call_rate <= 0.0:
            self.threshold = -1.0
        elif target_call_rate >= 1.0:
            self.threshold = 2.0
        else:
            self.threshold = float(np.quantile(cal_margins, target_call_rate))
        return self.threshold

    def route(self, student_margins: np.ndarray) -> np.ndarray:
        return np.asarray(student_margins) < self.threshold
