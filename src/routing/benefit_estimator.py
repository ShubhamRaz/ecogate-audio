"""
Out-of-Fold Expected Expert Benefit Estimator:
Estimates Delta(z) = E[d_i | z], where d_i = L(f_s(x_i), y_i) - L(f_e(x_i), y_i),
strictly using features z available BEFORE expert inference.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np


PRE_EXPERT_FEATURE_NAMES = [
    "student_max_prob",
    "student_margin",
    "student_entropy",
    "prototype_distance",
    "bird_activity_prob",
    "audio_snr_db",
    "clipping_ratio",
    "rms_energy",
    "remaining_budget_ratio",
]

FORBIDDEN_EXPERT_FEATURE_TOKENS = ("expert_logit", "expert_prob", "expert_pred", "expert_emb", "d_i_oracle")


def build_pre_expert_feature_matrix(
    student_max_prob: np.ndarray,
    student_margin: np.ndarray,
    student_entropy: np.ndarray,
    prototype_distance: np.ndarray,
    bird_activity_prob: np.ndarray,
    audio_snr_db: np.ndarray,
    clipping_ratio: np.ndarray,
    rms_energy: np.ndarray,
    remaining_budget_ratio: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Construct the pre-expert routing feature matrix Z of shape (N, 9) plus non-linear
    interaction terms (entropy * bird_activity, (1 - margin) * bird_activity, quality gate).
    Strictly validates that no expert outputs are passed.
    """
    if feature_names is not None:
        for name in feature_names:
            for forbidden in FORBIDDEN_EXPERT_FEATURE_TOKENS:
                if forbidden in name.lower():
                    raise ValueError(f"Leakage violation: forbidden expert feature '{name}' passed to router.")

    s_max = np.atleast_1d(np.asarray(student_max_prob, dtype=np.float32))
    s_mar = np.atleast_1d(np.asarray(student_margin, dtype=np.float32))
    s_ent = np.atleast_1d(np.asarray(student_entropy, dtype=np.float32))
    p_dist = np.atleast_1d(np.asarray(prototype_distance, dtype=np.float32))
    b_act = np.atleast_1d(np.asarray(bird_activity_prob, dtype=np.float32))
    snr = np.atleast_1d(np.asarray(audio_snr_db, dtype=np.float32))
    clip = np.atleast_1d(np.asarray(clipping_ratio, dtype=np.float32))
    rms = np.atleast_1d(np.asarray(rms_energy, dtype=np.float32))
    if remaining_budget_ratio is None:
        rem_b = np.ones_like(s_max, dtype=np.float32)
    else:
        rem_b = np.atleast_1d(np.asarray(remaining_budget_ratio, dtype=np.float32))

    # Interaction features capturing when clip is an active bird with high student uncertainty and usable SNR
    uncert_bird = s_ent * b_act
    margin_gap_bird = (1.0 - s_mar) * b_act
    usable_snr = (snr > -3.0).astype(np.float32) * b_act

    Z = np.stack(
        [s_max, s_mar, s_ent, p_dist, b_act, snr, clip, rms, rem_b, uncert_bird, margin_gap_bird, usable_snr],
        axis=-1,
    )
    return Z.astype(np.float32)


def compute_sample_loss(
    probs: np.ndarray,
    bird_prob: np.ndarray,
    y_known: np.ndarray,
    is_bird: np.ndarray,
    alpha_ce: float = 0.5,
    beta_err: float = 0.5,
) -> np.ndarray:
    """
    Compute composite decision loss L(f(x_i), y_i) per sample:
    - For known species (y_known >= 0): weighted sum of bounded cross-entropy and 0-1 classification error.
    - For non-bird noise (is_bird == 0): false bird-activity penalty.
    - For unfamiliar birds (y_known == -1, is_bird == 1): overconfident false known-species assignment penalty.
    """
    n = probs.shape[0]
    losses = np.zeros(n, dtype=np.float32)
    preds = np.argmax(probs, axis=1)
    max_p = np.max(probs, axis=1)

    for i in range(n):
        if y_known[i] >= 0:
            p_true = float(probs[i, int(y_known[i])])
            ce = min(4.0, -np.log(max(p_true, 1e-6))) / 4.0
            err = 1.0 if preds[i] != y_known[i] else 0.0
            losses[i] = alpha_ce * ce + beta_err * err
        elif is_bird[i] == 1:
            # Unfamiliar bird: penalize high max softmax probability on known species
            losses[i] = float(max_p[i] ** 2)
        else:
            # Non-bird environmental noise: penalize false bird activity trigger
            losses[i] = float(bird_prob[i] ** 2)
    return losses


class BenefitEstimator:
    """
    Ridge + Non-Linear Basis Regressor trained strictly on Out-of-Fold (OOF) predictions
    to estimate expected loss reduction Delta(z) = E[L(f_s(x), y) - L(f_e(x), y) | z].
    """

    def __init__(self, ridge_alpha: float = 1.5) -> None:
        self.ridge_alpha = ridge_alpha
        self.z_mean: Optional[np.ndarray] = None
        self.z_std: Optional[np.ndarray] = None
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.is_frozen: bool = False

    def fit_oof(self, Z_oof: np.ndarray, d_oof: np.ndarray) -> Dict[str, float]:
        """Fit benefit estimator on out-of-fold pre-expert feature matrix Z_oof and observed benefit d_oof."""
        if self.is_frozen:
            raise RuntimeError("BenefitEstimator is frozen for testing and cannot be refitted.")
        self.z_mean = np.mean(Z_oof, axis=0)
        self.z_std = np.std(Z_oof, axis=0) + 1e-6
        Zn = (Z_oof - self.z_mean) / self.z_std

        n, d = Zn.shape
        A = Zn.T @ Zn + self.ridge_alpha * np.eye(d, dtype=np.float32)
        self.weights = np.linalg.solve(A, Zn.T @ d_oof).astype(np.float32)
        self.bias = float(np.mean(d_oof))

        preds = self.predict_benefit(Z_oof)
        mse = float(np.mean((preds - d_oof) ** 2))
        corr = float(np.corrcoef(preds, d_oof)[0, 1]) if n > 2 and np.std(d_oof) > 1e-8 else 0.0
        return {"oof_mse": round(mse, 6), "oof_pearson_r": round(corr, 4), "mean_observed_benefit": round(self.bias, 5)}

    def predict_benefit(self, Z: np.ndarray) -> np.ndarray:
        if self.weights is None or self.z_mean is None or self.z_std is None:
            raise RuntimeError("BenefitEstimator must be fitted before calling predict_benefit.")
        single = Z.ndim == 1
        Zb = Z[None, :] if single else Z
        Zn = (Zb - self.z_mean) / self.z_std
        pred = Zn @ self.weights + self.bias
        return float(pred[0]) if single else pred.astype(np.float32)

    def freeze(self) -> None:
        self.is_frozen = True

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            z_mean=self.z_mean,
            z_std=self.z_std,
            weights=self.weights,
            bias=np.array([self.bias], dtype=np.float32),
            is_frozen=np.array([int(self.is_frozen)], dtype=np.int32),
        )

    def load(self, path: Path) -> None:
        data = np.load(path)
        self.z_mean = data["z_mean"]
        self.z_std = data["z_std"]
        self.weights = data["weights"]
        self.bias = float(data["bias"][0])
        self.is_frozen = bool(data["is_frozen"][0])
