"""
Post-hoc probability calibration via Temperature Scaling (Guo et al., ICML 2017),
Expected Calibration Error (ECE), and Brier Score evaluation.
"""
from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
from scipy.optimize import minimize_scalar

from src.models.audio_models import softmax


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) on known-class predictions."""
    if probs.shape[0] == 0:
        return 0.0
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(np.float64)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = float(probs.shape[0])
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.sum(in_bin) / n
        if prop_in_bin > 0:
            acc_in_bin = np.mean(accuracies[in_bin])
            conf_in_bin = np.mean(confidences[in_bin])
            ece += abs(acc_in_bin - conf_in_bin) * prop_in_bin
    return float(ece)


def compute_brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Compute multi-class Brier score."""
    if probs.shape[0] == 0:
        return 0.0
    n, c = probs.shape
    one_hot = np.eye(c)[labels]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


class TemperatureScaler:
    """Fit post-hoc temperature parameter T > 0 on an independent calibration split."""

    def __init__(self, init_temperature: float = 1.0) -> None:
        self.temperature: float = init_temperature

    def fit(self, cal_logits: np.ndarray, cal_labels: np.ndarray) -> Dict[str, float]:
        mask = cal_labels >= 0
        logits = cal_logits[mask]
        labels = cal_labels[mask]
        if logits.shape[0] == 0:
            return {"temperature": self.temperature, "ece_before": 0.0, "ece_after": 0.0}

        probs_before = softmax(logits, temperature=1.0)
        ece_before = compute_ece(probs_before, labels)
        brier_before = compute_brier_score(probs_before, labels)

        def nll_obj(t: float) -> float:
            p = softmax(logits, temperature=max(0.05, t))
            idx = np.arange(labels.shape[0])
            return float(-np.mean(np.log(p[idx, labels] + 1e-12)))

        res = minimize_scalar(nll_obj, bounds=(0.1, 10.0), method="bounded")
        self.temperature = float(res.x)
        probs_after = softmax(logits, temperature=self.temperature)
        ece_after = compute_ece(probs_after, labels)
        brier_after = compute_brier_score(probs_after, labels)

        return {
            "temperature": round(self.temperature, 4),
            "ece_before": round(ece_before, 4),
            "ece_after": round(ece_after, 4),
            "brier_before": round(brier_before, 4),
            "brier_after": round(brier_after, 4),
        }

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        return softmax(logits, temperature=self.temperature)
