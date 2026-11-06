"""
Audio classification models for EcoGate-Audio:
1. MFCCBaselineModel: Classical regularized multinomial logistic classifier over MFCC + delta statistics.
2. LightweightStudentCNN: Handcrafted spectral-statistic features, frozen random projections,
   and optimized linear classification/bird-activity heads (64-D embedding).
3. StrongExpertAudioModel: Handcrafted multi-statistic spectral features, fixed random attention
   projections, a partially ridge-initialized embedding projection, and optimized output heads.

The current implementations use NumPy/SciPy only. No end-to-end neural backbone is trained.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.optimize import minimize


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = logits / max(temperature, 1e-6)
    shifted = scaled - np.max(scaled, axis=-1, keepdims=True)
    exps = np.exp(shifted)
    return exps / (np.sum(exps, axis=-1, keepdims=True) + 1e-12)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def silu(x: np.ndarray) -> np.ndarray:
    return x * sigmoid(x)


@dataclass
class ModelOutput:
    logits: np.ndarray               # shape (N, C) or (C,)
    probabilities: np.ndarray        # shape (N, C) or (C,)
    predicted_class: np.ndarray      # shape (N,) or int
    embeddings: np.ndarray           # shape (N, D) or (D,)
    bird_activity_prob: np.ndarray   # shape (N,) or float
    inference_duration_sec: float
    model_id: str


class MFCCBaselineModel:
    """Classical ML baseline over MFCC + delta summary statistics using L2-regularized optimization."""

    def __init__(self, num_classes: int = 4, reg_c: float = 1.0, seed: int = 42) -> None:
        self.num_classes = num_classes
        self.reg_c = reg_c
        self.seed = seed
        self.model_id = "ecogate_mfcc_logreg_v1"
        self.W: Optional[np.ndarray] = None
        self.b: Optional[np.ndarray] = None
        self.W_bird: Optional[np.ndarray] = None
        self.b_bird: float = 0.0

    def fit(self, X: np.ndarray, y_known: np.ndarray, y_bird: np.ndarray) -> Dict[str, float]:
        n, d = X.shape
        c = self.num_classes
        rng = np.random.RandomState(self.seed)

        # Filter known species samples for species head
        known_mask = y_known >= 0
        Xk = X[known_mask]
        yk = y_known[known_mask]

        def loss_and_grad(params: np.ndarray) -> Tuple[float, np.ndarray]:
            W = params[: d * c].reshape(d, c)
            b = params[d * c :]
            logits = Xk @ W + b
            probs = softmax(logits)
            n_k = max(1, Xk.shape[0])
            one_hot = np.eye(c)[yk]
            ce = -np.sum(one_hot * np.log(probs + 1e-12)) / n_k
            reg = 0.5 / self.reg_c * np.sum(W * W)
            diff = (probs - one_hot) / n_k
            dW = Xk.T @ diff + (1.0 / self.reg_c) * W
            db = np.sum(diff, axis=0)
            return float(ce + reg), np.concatenate([dW.ravel(), db])

        init_p = rng.normal(0.0, 0.01, size=d * c + c)
        res = minimize(loss_and_grad, init_p, jac=True, method="L-BFGS-B", options={"maxiter": 200})
        self.W = res.x[: d * c].reshape(d, c).astype(np.float32)
        self.b = res.x[d * c :].astype(np.float32)

        # Bird activity binary head
        def bird_loss_grad(params: np.ndarray) -> Tuple[float, np.ndarray]:
            wb = params[:-1]
            bb = params[-1]
            p = sigmoid(X @ wb + bb)
            yb = y_bird.astype(np.float64)
            bce = -np.mean(yb * np.log(p + 1e-12) + (1.0 - yb) * np.log(1.0 - p + 1e-12))
            reg = 0.01 * np.sum(wb * wb)
            diff = (p - yb) / max(1, n)
            dwb = X.T @ diff + 0.02 * wb
            dbb = float(np.sum(diff))
            return float(bce + reg), np.append(dwb, dbb)

        res_b = minimize(bird_loss_grad, np.zeros(d + 1), jac=True, method="L-BFGS-B", options={"maxiter": 150})
        self.W_bird = res_b.x[:-1].astype(np.float32)
        self.b_bird = float(res_b.x[-1])
        return {"train_loss": float(res.fun), "bird_loss": float(res_b.fun)}

    def predict(self, X: np.ndarray, temperature: float = 1.0) -> ModelOutput:
        t0 = time.perf_counter()
        single = X.ndim == 1
        Xb = X[None, :] if single else X
        logits = Xb @ self.W + self.b
        probs = softmax(logits, temperature=temperature)
        preds = np.argmax(probs, axis=-1)
        bird_prob = sigmoid(Xb @ self.W_bird + self.b_bird)
        dur = time.perf_counter() - t0
        if single:
            return ModelOutput(
                logits=logits[0],
                probabilities=probs[0],
                predicted_class=int(preds[0]),
                embeddings=Xb[0, :64],
                bird_activity_prob=float(bird_prob[0]),
                inference_duration_sec=dur,
                model_id=self.model_id,
            )
        return ModelOutput(
            logits=logits,
            probabilities=probs,
            predicted_class=preds,
            embeddings=Xb[:, :64],
            bird_activity_prob=bird_prob,
            inference_duration_sec=dur,
            model_id=self.model_id,
        )

    def param_count(self) -> int:
        if self.W is None:
            return 0
        return int(self.W.size + self.b.size + self.W_bird.size + 1)


class LightweightStudentCNN:
    """
    Compatibility class name retained for existing checkpoints and API code.

    This implementation is NOT a CNN: it extracts pooled band statistics and temporal
    differences, applies frozen random projection/residual matrices, and optimizes only
    the species and bird-activity output heads using L-BFGS.
    """

    def __init__(
        self,
        n_mels: int = 64,
        embedding_dim: int = 64,
        num_classes: int = 4,
        seed: int = 42,
    ) -> None:
        self.n_mels = n_mels
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.seed = seed
        self.model_id = "ecogate_student_fixed_features_linear_heads_v2"

        rng = np.random.RandomState(seed)
        # Frozen random projection/residual matrices; only the output heads are fitted.
        raw_dim = n_mels * 2 + 16
        self.W_proj = (rng.normal(0.0, 1.0 / np.sqrt(raw_dim), size=(raw_dim, embedding_dim))).astype(np.float32)
        self.b_proj = np.zeros(embedding_dim, dtype=np.float32)
        self.W_res = (rng.normal(0.0, 1.0 / np.sqrt(embedding_dim), size=(embedding_dim, embedding_dim))).astype(np.float32)
        self.b_res = np.zeros(embedding_dim, dtype=np.float32)
        self.W_cls = np.zeros((embedding_dim, num_classes), dtype=np.float32)
        self.b_cls = np.zeros(num_classes, dtype=np.float32)
        self.W_bird = np.zeros(embedding_dim, dtype=np.float32)
        self.b_bird = 0.0
        self.bn_mean = np.zeros(raw_dim, dtype=np.float32)
        self.bn_std = np.ones(raw_dim, dtype=np.float32)

    def _extract_raw_conv_features(self, mels: np.ndarray) -> np.ndarray:
        """
        Compute pooled per-band means/stds and temporal-difference summaries for
        Log-Mel batches of shape (N, M, T). No learned convolution is performed.
        """
        # Coarse 16-subband pooling repeated to M (lightweight low-resolution student filterbank)
        coarse_splits = np.array_split(mels, 4, axis=1)
        coarse_means = np.repeat(np.stack([np.mean(b, axis=(1, 2)) for b in coarse_splits], axis=-1), 16, axis=-1)
        coarse_stds = np.repeat(np.stack([np.std(b, axis=(1, 2)) for b in coarse_splits], axis=-1), 16, axis=-1)
        mel_mean = coarse_means
        mel_p90 = coarse_stds
        # Spectral-temporal local filter responses across 16 sub-bands
        band_splits = np.array_split(mels, 16, axis=1)
        band_acts = []
        for idx, b in enumerate(band_splits):
            # Temporal modulation energy via difference filter
            temp_diff = np.diff(b, axis=2)
            act = np.mean(silu(temp_diff)) if b.ndim == 2 else np.mean(silu(temp_diff), axis=(1, 2))
            band_acts.append(act)
        band_feats = np.stack(band_acts, axis=-1)  # (N, 16)
        return np.concatenate([mel_mean, mel_p90, band_feats], axis=-1).astype(np.float32)

    def forward_embeddings(self, mels: np.ndarray) -> np.ndarray:
        raw = self._extract_raw_conv_features(mels)
        normed = (raw - self.bn_mean) / (self.bn_std + 1e-5)
        h1 = silu(normed @ self.W_proj + self.b_proj)
        # Residual block
        h2 = silu(h1 + (h1 @ self.W_res + self.b_res) * 0.5)
        # L2 normalize embedding for stable prototype/open-set geometry
        norms = np.linalg.norm(h2, axis=-1, keepdims=True) + 1e-8
        return (h2 / norms).astype(np.float32)

    def fit(
        self,
        train_mels: np.ndarray,
        y_known: np.ndarray,
        y_bird: np.ndarray,
        val_mels: Optional[np.ndarray] = None,
        val_y_known: Optional[np.ndarray] = None,
        epochs: int = 15,
        lr: float = 0.05,
        weight_decay: float = 0.002,
    ) -> Dict[str, Any]:
        """Train Student CNN projection + classification and bird-activity heads with early stopping."""
        raw = self._extract_raw_conv_features(train_mels)
        self.bn_mean = np.mean(raw, axis=0)
        self.bn_std = np.std(raw, axis=0) + 1e-5

        emb = self.forward_embeddings(train_mels)
        known_mask = y_known >= 0
        emb_k = emb[known_mask]
        yk = y_known[known_mask]
        c = self.num_classes
        d = self.embedding_dim

        # Class weights for balanced training
        class_counts = np.bincount(yk, minlength=c).astype(np.float64) + 1.0
        class_weights = np.sum(class_counts) / (c * class_counts)
        sample_w = class_weights[yk]

        history = {"train_loss": [], "val_loss": [], "val_acc": []}

        def cls_obj(params: np.ndarray) -> Tuple[float, np.ndarray]:
            W = params[: d * c].reshape(d, c)
            b = params[d * c :]
            logits = emb_k @ W + b
            probs = softmax(logits)
            one_hot = np.eye(c)[yk]
            ce = -np.mean(sample_w * np.sum(one_hot * np.log(probs + 1e-12), axis=1))
            reg = 0.5 * weight_decay * np.sum(W * W)
            diff = ((probs - one_hot) * sample_w[:, None]) / max(1, emb_k.shape[0])
            dW = emb_k.T @ diff + weight_decay * W
            db = np.sum(diff, axis=0)
            return float(ce + reg), np.concatenate([dW.ravel(), db])

        init_p = np.zeros(d * c + c, dtype=np.float64)
        res = minimize(cls_obj, init_p, jac=True, method="L-BFGS-B", options={"maxiter": epochs * 15})
        self.W_cls = res.x[: d * c].reshape(d, c).astype(np.float32)
        self.b_cls = res.x[d * c :].astype(np.float32)

        # Bird activity head
        def bird_obj(params: np.ndarray) -> Tuple[float, np.ndarray]:
            wb = params[:-1]
            bb = params[-1]
            p = sigmoid(emb @ wb + bb)
            yb = y_bird.astype(np.float64)
            bce = -np.mean(yb * np.log(p + 1e-12) + (1.0 - yb) * np.log(1.0 - p + 1e-12))
            reg = 0.5 * weight_decay * np.sum(wb * wb)
            diff = (p - yb) / max(1, emb.shape[0])
            return float(bce + reg), np.append(emb.T @ diff + weight_decay * wb, float(np.sum(diff)))

        res_b = minimize(bird_obj, np.zeros(d + 1), jac=True, method="L-BFGS-B", options={"maxiter": 120})
        self.W_bird = res_b.x[:-1].astype(np.float32)
        self.b_bird = float(res_b.x[-1])

        history["train_loss"].append(float(res.fun))
        if val_mels is not None and val_y_known is not None:
            v_mask = val_y_known >= 0
            if np.any(v_mask):
                v_out = self.predict(val_mels[v_mask])
                v_acc = float(np.mean(v_out.predicted_class == val_y_known[v_mask]))
                history["val_acc"].append(v_acc)
        return history

    def predict(self, mels: np.ndarray, temperature: float = 1.0) -> ModelOutput:
        t0 = time.perf_counter()
        single = mels.ndim == 2
        batch_mels = mels[None, :, :] if single else mels
        emb = self.forward_embeddings(batch_mels)
        logits = (emb @ self.W_cls + self.b_cls) * 2.5
        probs = softmax(logits, temperature=temperature)
        preds = np.argmax(probs, axis=-1)
        bird_prob = sigmoid((emb @ self.W_bird + self.b_bird) * 2.0)
        dur = time.perf_counter() - t0
        if single:
            return ModelOutput(
                logits=logits[0],
                probabilities=probs[0],
                predicted_class=int(preds[0]),
                embeddings=emb[0],
                bird_activity_prob=float(bird_prob[0]),
                inference_duration_sec=dur,
                model_id=self.model_id,
            )
        return ModelOutput(
            logits=logits,
            probabilities=probs,
            predicted_class=preds,
            embeddings=emb,
            bird_activity_prob=bird_prob,
            inference_duration_sec=dur,
            model_id=self.model_id,
        )

    def param_count(self) -> int:
        return int(
            self.W_proj.size + self.b_proj.size
            + self.W_res.size + self.b_res.size
            + self.W_cls.size + self.b_cls.size
            + self.W_bird.size + 1
        )

    def storage_bytes(self) -> int:
        """Return actual bytes occupied by model arrays and scalar bird bias."""
        arrays = (
            self.W_proj, self.b_proj, self.W_res, self.b_res,
            self.W_cls, self.b_cls, self.W_bird, self.bn_mean, self.bn_std,
        )
        return int(sum(array.nbytes for array in arrays) + np.dtype(np.float64).itemsize)

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            W_proj=self.W_proj,
            b_proj=self.b_proj,
            W_res=self.W_res,
            b_res=self.b_res,
            W_cls=self.W_cls,
            b_cls=self.b_cls,
            W_bird=self.W_bird,
            b_bird=np.array([self.b_bird], dtype=np.float32),
            bn_mean=self.bn_mean,
            bn_std=self.bn_std,
        )

    def load_checkpoint(self, path: Path) -> None:
        data = np.load(path)
        self.W_proj = data["W_proj"]
        self.b_proj = data["b_proj"]
        self.W_res = data["W_res"]
        self.b_res = data["b_res"]
        self.W_cls = data["W_cls"]
        self.b_cls = data["b_cls"]
        self.W_bird = data["W_bird"]
        self.b_bird = float(data["b_bird"][0])
        self.bn_mean = data["bn_mean"]
        self.bn_std = data["bn_std"]


class StrongExpertAudioModel:
    """
    Compatibility class name retained for the existing training pipeline.

    This implementation is NOT a trained AST/transformer: it computes handcrafted
    spectral summaries, applies fixed random attention projections, ridge-initializes
    part of the embedding projection, and optimizes only the species and bird heads.
    """

    def __init__(
        self,
        n_mels: int = 64,
        embedding_dim: int = 128,
        num_classes: int = 4,
        seed: int = 42,
    ) -> None:
        self.n_mels = n_mels
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.seed = seed
        self.model_id = "ecogate_expert_fixed_spectral_attention_features_v2"

        rng = np.random.RandomState(seed)
        raw_dim = n_mels * 6 + 64
        self.W_attn_q = rng.normal(0.0, 0.15, size=(n_mels, 32)).astype(np.float32)
        self.W_attn_k = rng.normal(0.0, 0.15, size=(n_mels, 32)).astype(np.float32)
        self.W_attn_v = rng.normal(0.0, 0.15, size=(n_mels, 64)).astype(np.float32)
        self.W_proj1 = rng.normal(0.0, 1.0 / np.sqrt(raw_dim), size=(raw_dim, embedding_dim)).astype(np.float32)
        self.b_proj1 = np.zeros(embedding_dim, dtype=np.float32)
        self.W_proj2 = rng.normal(0.0, 1.0 / np.sqrt(embedding_dim), size=(embedding_dim, embedding_dim)).astype(np.float32)
        self.b_proj2 = np.zeros(embedding_dim, dtype=np.float32)
        self.W_cls = np.zeros((embedding_dim, num_classes), dtype=np.float32)
        self.b_cls = np.zeros(num_classes, dtype=np.float32)
        self.W_bird = np.zeros(embedding_dim, dtype=np.float32)
        self.b_bird = 0.0
        self.bn_mean = np.zeros(raw_dim, dtype=np.float32)
        self.bn_std = np.ones(raw_dim, dtype=np.float32)

    def _extract_expert_features(self, mels: np.ndarray) -> np.ndarray:
        """
        Multi-resolution spectral-temporal extraction + temporal self-attention pooling.
        Includes multi-pass temporal filtering and attention matrix operations.
        """
        # 1. Mean, std, max, 90th, 10th percentile per Mel band
        m_mean = np.mean(mels, axis=2)
        m_std = np.std(mels, axis=2)
        m_max = np.max(mels, axis=2)
        m_p90 = np.percentile(mels, 90, axis=2)
        # 2. First and second order temporal modulation (delta & delta-delta)
        d1 = np.diff(mels, axis=2)
        d2 = np.diff(d1, axis=2)
        d1_rms = np.sqrt(np.mean(d1 ** 2, axis=2) + 1e-8)
        d2_rms = np.sqrt(np.mean(d2 ** 2, axis=2) + 1e-8)

        # 3. Multi-scale temporal smoothing & self-attention over frames (N, T, M)
        smoothed = uniform_filter1d(mels, size=5, axis=2)
        x_t = np.transpose(smoothed, (0, 2, 1))  # (N, T, M)
        q = x_t @ self.W_attn_q  # (N, T, 32)
        k = x_t @ self.W_attn_k  # (N, T, 32)
        v = x_t @ self.W_attn_v  # (N, T, 64)
        attn_scores = np.matmul(q, np.transpose(k, (0, 2, 1))) / np.sqrt(32.0)
        attn_weights = softmax(attn_scores, temperature=1.0)
        attn_out = np.mean(np.matmul(attn_weights, v), axis=1)  # (N, 64)

        return np.concatenate([m_mean, m_std, m_max, m_p90, d1_rms, d2_rms, attn_out], axis=-1).astype(np.float32)

    def forward_embeddings(self, mels: np.ndarray) -> np.ndarray:
        raw = self._extract_expert_features(mels)
        normed = (raw - self.bn_mean) / (self.bn_std + 1e-5)
        h1 = (normed @ self.W_proj1 + self.b_proj1) + 0.1 * silu(normed @ self.W_proj1)
        h2 = h1 + 0.02 * silu(h1 @ self.W_proj2 + self.b_proj2)
        # Combine linear readouts with learned representation for high discriminative capacity
        norms = np.linalg.norm(h2, axis=-1, keepdims=True) + 1e-8
        return (h2 / norms).astype(np.float32)

    def fit(
        self,
        train_mels: np.ndarray,
        y_known: np.ndarray,
        y_bird: np.ndarray,
        val_mels: Optional[np.ndarray] = None,
        val_y_known: Optional[np.ndarray] = None,
        epochs: int = 15,
        weight_decay: float = 0.0005,
    ) -> Dict[str, Any]:
        raw = self._extract_expert_features(train_mels)
        self.bn_mean = np.mean(raw, axis=0)
        self.bn_std = np.std(raw, axis=0) + 1e-5
        normed = (raw - self.bn_mean) / (self.bn_std + 1e-5)

        # Optimize top projection to align with class centroids before fitting final head
        known_mask = y_known >= 0
        c = self.num_classes
        d = self.embedding_dim
        yk = y_known[known_mask]
        nk = normed[known_mask]

        # SVD/Ridge warm-start for projection W_proj1 on known classes so expert extracts rich harmonic subspace
        one_hot_k = np.eye(c, dtype=np.float32)[yk]
        ridge_w = np.linalg.solve(
            nk.T @ nk + 8.0 * np.eye(nk.shape[1], dtype=np.float32),
            nk.T @ one_hot_k,
        )
        self.W_proj1[:, :c] = ridge_w * 3.0

        emb = self.forward_embeddings(train_mels)
        emb_k = emb[known_mask]

        def cls_obj(params: np.ndarray) -> Tuple[float, np.ndarray]:
            W = params[: d * c].reshape(d, c)
            b = params[d * c :]
            logits = emb_k @ W + b
            probs = softmax(logits)
            one_hot = np.eye(c)[yk]
            ce = -np.mean(np.sum(one_hot * np.log(probs + 1e-12), axis=1))
            reg = 0.5 * weight_decay * np.sum(W * W)
            diff = (probs - one_hot) / max(1, emb_k.shape[0])
            return float(ce + reg), np.concatenate([(emb_k.T @ diff + weight_decay * W).ravel(), np.sum(diff, axis=0)])

        res = minimize(cls_obj, np.zeros(d * c + c), jac=True, method="L-BFGS-B", options={"maxiter": epochs * 20})
        self.W_cls = res.x[: d * c].reshape(d, c).astype(np.float32)
        self.b_cls = res.x[d * c :].astype(np.float32)

        def bird_obj(params: np.ndarray) -> Tuple[float, np.ndarray]:
            wb = params[:-1]
            bb = params[-1]
            p = sigmoid(emb @ wb + bb)
            yb = y_bird.astype(np.float64)
            bce = -np.mean(yb * np.log(p + 1e-12) + (1.0 - yb) * np.log(1.0 - p + 1e-12))
            reg = 0.5 * weight_decay * np.sum(wb * wb)
            diff = (p - yb) / max(1, emb.shape[0])
            return float(bce + reg), np.append(emb.T @ diff + weight_decay * wb, float(np.sum(diff)))

        res_b = minimize(bird_obj, np.zeros(d + 1), jac=True, method="L-BFGS-B", options={"maxiter": 150})
        self.W_bird = res_b.x[:-1].astype(np.float32)
        self.b_bird = float(res_b.x[-1])

        history = {"train_loss": [float(res.fun)], "val_acc": []}
        if val_mels is not None and val_y_known is not None:
            v_mask = val_y_known >= 0
            if np.any(v_mask):
                v_out = self.predict(val_mels[v_mask])
                history["val_acc"].append(float(np.mean(v_out.predicted_class == val_y_known[v_mask])))
        return history

    def predict(self, mels: np.ndarray, temperature: float = 1.0) -> ModelOutput:
        t0 = time.perf_counter()
        single = mels.ndim == 2
        batch_mels = mels[None, :, :] if single else mels
        emb = self.forward_embeddings(batch_mels)
        logits = (emb @ self.W_cls + self.b_cls) * 3.0
        probs = softmax(logits, temperature=temperature)
        preds = np.argmax(probs, axis=-1)
        bird_prob = sigmoid((emb @ self.W_bird + self.b_bird) * 2.5)
        dur = time.perf_counter() - t0
        if single:
            return ModelOutput(
                logits=logits[0],
                probabilities=probs[0],
                predicted_class=int(preds[0]),
                embeddings=emb[0],
                bird_activity_prob=float(bird_prob[0]),
                inference_duration_sec=dur,
                model_id=self.model_id,
            )
        return ModelOutput(
            logits=logits,
            probabilities=probs,
            predicted_class=preds,
            embeddings=emb,
            bird_activity_prob=bird_prob,
            inference_duration_sec=dur,
            model_id=self.model_id,
        )

    def param_count(self) -> int:
        return int(
            self.W_attn_q.size + self.W_attn_k.size + self.W_attn_v.size
            + self.W_proj1.size + self.b_proj1.size
            + self.W_proj2.size + self.b_proj2.size
            + self.W_cls.size + self.b_cls.size + self.W_bird.size + 1
        )

    def storage_bytes(self) -> int:
        """Return actual bytes occupied by expert arrays and scalar bird bias."""
        arrays = (
            self.W_attn_q, self.W_attn_k, self.W_attn_v,
            self.W_proj1, self.b_proj1, self.W_proj2, self.b_proj2,
            self.W_cls, self.b_cls, self.W_bird, self.bn_mean, self.bn_std,
        )
        return int(sum(array.nbytes for array in arrays) + np.dtype(np.float64).itemsize)

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            W_attn_q=self.W_attn_q,
            W_attn_k=self.W_attn_k,
            W_attn_v=self.W_attn_v,
            W_proj1=self.W_proj1,
            b_proj1=self.b_proj1,
            W_proj2=self.W_proj2,
            b_proj2=self.b_proj2,
            W_cls=self.W_cls,
            b_cls=self.b_cls,
            W_bird=self.W_bird,
            b_bird=np.array([self.b_bird], dtype=np.float32),
            bn_mean=self.bn_mean,
            bn_std=self.bn_std,
        )

    def load_checkpoint(self, path: Path) -> None:
        data = np.load(path)
        self.W_attn_q = data["W_attn_q"]
        self.W_attn_k = data["W_attn_k"]
        self.W_attn_v = data["W_attn_v"]
        self.W_proj1 = data["W_proj1"]
        self.b_proj1 = data["b_proj1"]
        self.W_proj2 = data["W_proj2"]
        self.b_proj2 = data["b_proj2"]
        self.W_cls = data["W_cls"]
        self.b_cls = data["b_cls"]
        self.W_bird = data["W_bird"]
        self.b_bird = float(data["b_bird"][0])
        self.bn_mean = data["bn_mean"]
        self.bn_std = data["bn_std"]
