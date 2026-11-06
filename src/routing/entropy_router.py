"""Normalized Shannon entropy baseline router."""
from __future__ import annotations
import numpy as np


class EntropyRouter:
    """Invoke expert when student predictive entropy exceeds a threshold."""

    def __init__(self, threshold: float = 0.50) -> None:
        self.threshold = threshold

    def fit_threshold_for_call_rate(self, cal_entropies: np.ndarray, target_call_rate: float) -> float:
        if cal_entropies.size == 0 or target_call_rate <= 0.0:
            self.threshold = 1e6
        elif target_call_rate >= 1.0:
            self.threshold = -1.0
        else:
            self.threshold = float(np.quantile(cal_entropies, 1.0 - target_call_rate))
        return self.threshold

    def route(self, student_entropies: np.ndarray) -> np.ndarray:
        return np.asarray(student_entropies) > self.threshold
