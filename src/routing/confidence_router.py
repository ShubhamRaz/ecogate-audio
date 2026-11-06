"""Fixed-confidence (Maximum Softmax Probability) baseline router."""
from __future__ import annotations
import numpy as np


class ConfidenceRouter:
    """Invoke expert when student maximum softmax probability falls below a threshold."""

    def __init__(self, threshold: float = 0.75) -> None:
        self.threshold = threshold

    def fit_threshold_for_call_rate(self, cal_max_probs: np.ndarray, target_call_rate: float) -> float:
        if cal_max_probs.size == 0 or target_call_rate <= 0.0:
            self.threshold = -1.0
        elif target_call_rate >= 1.0:
            self.threshold = 2.0
        else:
            self.threshold = float(np.quantile(cal_max_probs, target_call_rate))
        return self.threshold

    def route(self, student_max_probs: np.ndarray) -> np.ndarray:
        """Return boolean mask (True = invoke expert) using only student confidence."""
        return np.asarray(student_max_probs) < self.threshold
