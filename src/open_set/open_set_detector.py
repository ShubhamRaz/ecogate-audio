"""
Open-Set Recognition Module for EcoGate-Audio.
Implements six independently configurable open-set / out-of-distribution scoring methods:
1. Maximum Softmax Probability (MSP)
2. Temperature-scaled confidence
3. Entropy-based rejection
4. Prediction margin
5. Prototype-distance rejection (Euclidean & Cosine distance to class centroids)
6. Energy-based OOD scoring (Liu et al., NeurIPS 2020)

Score Direction Convention:
- `novelty_score`: HIGHER values indicate GREATER likelihood of being unknown/OOD (`bird_unfamiliar` or `nonbird`).
- `knownness_score`: HIGHER values indicate GREATER likelihood of belonging to `known_species`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.models.audio_models import softmax


@dataclass
class OpenSetScores:
    method: str
    novelty_score: np.ndarray      # Higher = more likely unknown/OOD
    knownness_score: np.ndarray    # Higher = more likely known species
    is_rejected_unknown: np.ndarray
    threshold: float


class OpenSetDetector:
    """
    Configurable Open-Set Recognition module supporting 6 scoring mechanisms and
    class prototype computation on known training embeddings.
    """

    SUPPORTED_METHODS = (
        "msp",
        "temperature_scaled",
        "entropy",
        "margin",
        "prototype_distance",
        "energy",
    )

    def __init__(
        self,
        num_known_classes: int = 4,
        temperature: float = 1.5,
        energy_temp: float = 1.0,
    ) -> None:
        self.num_known_classes = num_known_classes
        self.temperature = temperature
        self.energy_temp = energy_temp
        self.prototypes: Optional[np.ndarray] = None
        self.thresholds: Dict[str, float] = {m: 0.5 for m in self.SUPPORTED_METHODS}

    def fit_prototypes(self, embeddings: np.ndarray, y_known: np.ndarray) -> np.ndarray:
        """Compute L2-normalized class prototypes from training known-species embeddings."""
        d = embeddings.shape[1]
        protos = np.zeros((self.num_known_classes, d), dtype=np.float32)
        for c in range(self.num_known_classes):
            mask = y_known == c
            if np.any(mask):
                mean_vec = np.mean(embeddings[mask], axis=0)
                norm = np.linalg.norm(mean_vec) + 1e-8
                protos[c] = mean_vec / norm
        self.prototypes = protos
        return protos

    def compute_prototype_distances(self, embeddings: np.ndarray) -> np.ndarray:
        """Return minimum Euclidean distance to any known class prototype for each sample."""
        if self.prototypes is None:
            raise RuntimeError("Prototypes not fitted. Call fit_prototypes first.")
        single = embeddings.ndim == 1
        emb = embeddings[None, :] if single else embeddings
        # emb: (N, D), prototypes: (C, D)
        diffs = emb[:, None, :] - self.prototypes[None, :, :]
        dists = np.linalg.norm(diffs, axis=-1)  # (N, C)
        min_dist = np.min(dists, axis=1)
        return float(min_dist[0]) if single else min_dist

    def compute_all_novelty_features(
        self,
        logits: np.ndarray,
        embeddings: np.ndarray,
        calibrated_temp: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Compute all 6 novelty scores (higher = more likely unknown/OOD) and auxiliary pre-expert
        confidence/uncertainty statistics.
        """
        single = logits.ndim == 1
        log_b = logits[None, :] if single else logits
        emb_b = embeddings[None, :] if single else embeddings
        temp = calibrated_temp if calibrated_temp is not None else self.temperature

        probs = softmax(log_b, temperature=1.0)
        cal_probs = softmax(log_b, temperature=temp)

        # 1. MSP novelty: 1 - max_c P(y=c|x)
        max_prob = np.max(probs, axis=1)
        msp_novelty = 1.0 - max_prob

        # 2. Temperature-scaled novelty: 1 - max_c P_T(y=c|x)
        cal_max_prob = np.max(cal_probs, axis=1)
        temp_novelty = 1.0 - cal_max_prob

        # 3. Normalized Shannon entropy novelty: -sum P log P / log(C)
        c = log_b.shape[1]
        entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1) / np.log(max(2, c))

        # 4. Margin novelty: 1 - (P_(1) - P_(2))
        sorted_p = np.sort(probs, axis=1)
        margin = sorted_p[:, -1] - sorted_p[:, -2]
        margin_novelty = 1.0 - margin

        # 5. Prototype distance novelty
        proto_dist = np.asarray(self.compute_prototype_distances(emb_b), dtype=np.float32)

        # 6. Energy-based OOD score: -T * log sum_c exp(f_c(x) / T)
        # Shifted for numerical stability; higher energy = more OOD
        t_e = max(0.1, self.energy_temp)
        max_l = np.max(log_b / t_e, axis=1)
        energy_ood = -t_e * (max_l + np.log(np.sum(np.exp(log_b / t_e - max_l[:, None]), axis=1) + 1e-12))

        out = {
            "msp": msp_novelty.astype(np.float32),
            "temperature_scaled": temp_novelty.astype(np.float32),
            "entropy": entropy.astype(np.float32),
            "margin": margin_novelty.astype(np.float32),
            "prototype_distance": proto_dist.astype(np.float32),
            "energy": energy_ood.astype(np.float32),
            "raw_max_prob": max_prob.astype(np.float32),
            "raw_margin": margin.astype(np.float32),
        }
        if single:
            return {k: float(v[0]) for k, v in out.items()}
        return out

    def calibrate_thresholds(
        self,
        cal_logits: np.ndarray,
        cal_embeddings: np.ndarray,
        is_known_species: np.ndarray,
        target_known_recall: float = 0.90,
    ) -> Dict[str, float]:
        """
        Calibrate rejection thresholds on the calibration split so that at least
        `target_known_recall` (e.g., 90% or 95%) of known-species samples fall below the
        novelty threshold. Never touches test data.
        """
        feats = self.compute_all_novelty_features(cal_logits, cal_embeddings)
        known_mask = is_known_species.astype(bool)
        for m in self.SUPPORTED_METHODS:
            scores = feats[m]
            if np.any(known_mask):
                self.thresholds[m] = float(np.quantile(scores[known_mask], target_known_recall))
            else:
                self.thresholds[m] = float(np.median(scores))
        return dict(self.thresholds)

    def score(
        self,
        logits: np.ndarray,
        embeddings: np.ndarray,
        method: str = "prototype_distance",
    ) -> OpenSetScores:
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(f"Unsupported open-set method '{method}'. Choose from {self.SUPPORTED_METHODS}")
        feats = self.compute_all_novelty_features(logits, embeddings)
        nov = np.asarray(feats[method], dtype=np.float32)
        thresh = self.thresholds.get(method, 0.5)
        rejected = nov > thresh
        return OpenSetScores(
            method=method,
            novelty_score=nov,
            knownness_score=-nov,
            is_rejected_unknown=rejected,
            threshold=thresh,
        )
