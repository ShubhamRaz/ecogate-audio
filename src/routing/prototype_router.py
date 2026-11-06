"""Embedding prototype-distance baseline router."""
from __future__ import annotations
import numpy as np


class PrototypeRouter:
    """
    Invoke expert when student embedding is moderately far from known class prototypes
    (in the ambiguous boundary zone where expert harmonic resolution is helpful).
    """

    def __init__(self, threshold: float = 0.65) -> None:
        self.threshold = threshold

    def fit_threshold_for_call_rate(self, cal_proto_dists: np.ndarray, target_call_rate: float) -> float:
        if cal_proto_dists.size == 0 or target_call_rate <= 0.0:
            self.threshold = 1e6
        elif target_call_rate >= 1.0:
            self.threshold = -1.0
        else:
            self.threshold = float(np.quantile(cal_proto_dists, 1.0 - target_call_rate))
        return self.threshold

    def route(self, proto_dists: np.ndarray) -> np.ndarray:
        return np.asarray(proto_dists) > self.threshold
