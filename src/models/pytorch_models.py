"""
Genuine trainable PyTorch neural architectures for EcoGate-Audio.

Models
------
StudentCNN
    Lightweight 2-D convolutional network operating on Log-Mel spectrograms.
    Approximately 150 K–300 K trainable parameters; suitable for edge deployment.

ExpertTransformer
    Transformer-based audio model with patch tokenisation of Log-Mel spectrograms.
    Approximately 1–4 M trainable parameters; higher capacity than the student.

Both expose the same public interface as the legacy NumPy models in audio_models.py
(``fit()``, ``predict()``, ``save_checkpoint()``, ``load_checkpoint()``,
``param_count()``, ``storage_bytes()``) so they are drop-in replacements.

Training
--------
``train_model()`` runs a full reproducible training loop including:
  - AdamW optimiser with cosine-annealing LR schedule
  - Gradient clipping
  - Early stopping on validation macro-F1
  - Multi-seed reproducibility via ``seed_everything()``
  - Checkpoint saved whenever validation F1 improves
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False

from src.models.audio_models import ModelOutput, softmax, sigmoid


# ---------------------------------------------------------------------------
# Reproducibility helpers
# ---------------------------------------------------------------------------

def seed_everything(seed: int = 42) -> None:
    """Set all relevant random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    if _TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _require_torch(name: str) -> None:
    if not _TORCH_AVAILABLE:
        raise RuntimeError(
            f"{name} requires PyTorch. Install it with:\n"
            "  uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121"
        )


# ---------------------------------------------------------------------------
# Shared metric helpers (NumPy, no PyTorch needed)
# ---------------------------------------------------------------------------

def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    f1s = []
    for c in range(num_classes):
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        f1s.append(f1)
    return float(np.mean(f1s))


# ---------------------------------------------------------------------------
# StudentCNN — genuine trainable lightweight 2-D CNN
# ---------------------------------------------------------------------------

class _ConvBlock(nn.Module):
    """Conv2d → BatchNorm2d → GELU → MaxPool2d block."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int = 3,
        stride: int = 1,
        pool: Tuple[int, int] = (2, 2),
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=kernel // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d(pool)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.pool(F.gelu(self.bn(self.conv(x))))


class _StudentBackbone(nn.Module):
    """
    Lightweight 2-D CNN backbone for log-Mel spectrograms.

    Input  : (B, 1, n_mels=64, n_frames)
    Output : (B, embedding_dim=64)
    """

    def __init__(self, n_mels: int = 64, embedding_dim: int = 64) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
        )
        # Four conv blocks; each halves both frequency and time axes.
        self.block1 = _ConvBlock(16, 32, pool=(2, 2))   # → (32, 32, T/2)
        self.block2 = _ConvBlock(32, 64, pool=(2, 2))   # → (64, 16, T/4)
        self.block3 = _ConvBlock(64, 96, pool=(2, 2))   # → (96,  8, T/8)
        self.block4 = _ConvBlock(96, 128, pool=(2, 2))  # → (128, 4, T/16)
        self.gap = nn.AdaptiveAvgPool2d(1)              # → (128, 1, 1)
        self.proj = nn.Linear(128, embedding_dim)
        self.drop = nn.Dropout(0.3)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        h = self.stem(x)
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.gap(h).flatten(1)       # (B, 128)
        h = self.drop(h)
        emb = F.normalize(self.proj(h), dim=-1)  # (B, embedding_dim)
        return emb


class _StudentNet(nn.Module):
    def __init__(self, n_mels: int = 64, embedding_dim: int = 64, num_classes: int = 4) -> None:
        super().__init__()
        self.backbone = _StudentBackbone(n_mels, embedding_dim)
        self.cls_head = nn.Linear(embedding_dim, num_classes)
        self.bird_head = nn.Linear(embedding_dim, 1)

    def forward(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        emb = self.backbone(x)
        return self.cls_head(emb), self.bird_head(emb).squeeze(-1), emb


class StudentCNN:
    """
    Genuinely trainable lightweight CNN over Log-Mel spectrograms.

    Public interface matches the legacy ``LightweightStudentCNN`` so all
    dependent scripts, tests and API code work without modification.
    """

    def __init__(
        self,
        n_mels: int = 64,
        embedding_dim: int = 64,
        num_classes: int = 4,
        seed: int = 42,
    ) -> None:
        _require_torch("StudentCNN")
        self.n_mels = n_mels
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.seed = seed
        self.model_id = "ecogate_student_cnn_pytorch_v1"
        self._training_log: List[Dict[str, Any]] = []

        seed_everything(seed)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._net = _StudentNet(n_mels, embedding_dim, num_classes).to(self._device)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        train_mels: np.ndarray,       # (N, n_mels, T)
        y_known: np.ndarray,           # (N,) — -1 for unknown/open-set
        y_bird: np.ndarray,            # (N,) — 0/1 binary
        val_mels: Optional[np.ndarray] = None,
        val_y_known: Optional[np.ndarray] = None,
        epochs: int = 30,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        patience: int = 8,
        checkpoint_path: Optional[Path] = None,
        **_kwargs: Any,
    ) -> Dict[str, Any]:
        """Train with AdamW + cosine-LR, early stopping on validation macro-F1."""
        seed_everything(self.seed)
        net = self._net
        net.train()

        # --- Dataset tensors --------------------------------------------------
        X_t = torch.from_numpy(train_mels[:, None, :, :])  # (N,1,M,T)
        known_mask = y_known >= 0
        X_k = X_t[known_mask]
        yk_t = torch.from_numpy(y_known[known_mask].astype(np.int64))
        yb_t = torch.from_numpy(y_bird[known_mask].astype(np.float32))
        # bird head uses all samples (including unknown-class samples)
        X_all = X_t
        yb_all = torch.from_numpy(y_bird.astype(np.float32))

        # Class weights for imbalanced species counts
        class_counts = np.bincount(y_known[known_mask], minlength=self.num_classes).astype(np.float32)
        class_weights = torch.tensor(
            float(np.sum(class_counts)) / (self.num_classes * np.maximum(class_counts, 1)),
            dtype=torch.float32,
        ).to(self._device)

        ds_cls = TensorDataset(X_k, yk_t, yb_t)
        dl_cls = DataLoader(ds_cls, batch_size=batch_size, shuffle=True, drop_last=False)

        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.05)

        best_val_f1 = -1.0
        patience_ctr = 0
        self._training_log = []

        for epoch in range(epochs):
            net.train()
            epoch_cls_loss = 0.0
            epoch_bird_loss = 0.0
            n_batches = 0

            for Xb, ykb, ybb in dl_cls:
                Xb = Xb.to(self._device)
                ykb = ykb.to(self._device)
                ybb = ybb.to(self._device)

                opt.zero_grad()
                logits_cls, logits_bird, _ = net(Xb)
                loss_cls = F.cross_entropy(logits_cls, ykb, weight=class_weights)
                loss_bird = F.binary_cross_entropy_with_logits(logits_bird, ybb)
                loss = loss_cls + 0.3 * loss_bird
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                opt.step()
                epoch_cls_loss += loss_cls.item()
                epoch_bird_loss += loss_bird.item()
                n_batches += 1

            sched.step()
            avg_cls = epoch_cls_loss / max(1, n_batches)
            avg_bird = epoch_bird_loss / max(1, n_batches)

            # Validation
            val_f1 = -1.0
            val_acc = -1.0
            val_loss = float("inf")
            if val_mels is not None and val_y_known is not None:
                v_mask = val_y_known >= 0
                if np.any(v_mask):
                    v_out = self.predict(val_mels[v_mask])
                    val_f1 = _macro_f1(val_y_known[v_mask], v_out.predicted_class, self.num_classes)
                    val_acc = float(np.mean(v_out.predicted_class == val_y_known[v_mask]))
                    # Compute val loss
                    with torch.no_grad():
                        Xv = torch.from_numpy(val_mels[v_mask][:, None, :, :]).to(self._device)
                        yv = torch.from_numpy(val_y_known[v_mask].astype(np.int64)).to(self._device)
                        logits_v, _, _ = net(Xv)
                        val_loss = F.cross_entropy(logits_v, yv).item()

            current_lr = sched.get_last_lr()[0]
            log_entry = {
                "epoch": epoch + 1,
                "train_cls_loss": avg_cls,
                "train_bird_loss": avg_bird,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_macro_f1": val_f1,
                "lr": current_lr,
            }
            self._training_log.append(log_entry)

            # Early stopping / best checkpoint
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_ctr = 0
                if checkpoint_path is not None:
                    self.save_checkpoint(checkpoint_path)
            else:
                patience_ctr += 1
                if patience > 0 and patience_ctr >= patience:
                    break

        # Reload best checkpoint if saved
        if checkpoint_path is not None and checkpoint_path.exists():
            self.load_checkpoint(checkpoint_path)

        return {
            "training_log": self._training_log,
            "best_val_macro_f1": best_val_f1,
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _forward_batch(
        self, mels: np.ndarray, chunk_size: int = 128
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run batched inference; returns (logits, embeddings, bird_logits) as numpy arrays."""
        self._net.eval()
        device = next(self._net.parameters()).device
        n = mels.shape[0]
        if n == 0:
            return (
                np.zeros((0, self.num_classes), dtype=np.float32),
                np.zeros((0, self.embedding_dim), dtype=np.float32),
                np.zeros(0, dtype=np.float32),
            )
        logits_list = []
        embs_list = []
        bird_list = []
        for i in range(0, n, chunk_size):
            chunk = torch.from_numpy(mels[i : i + chunk_size, None, :, :]).to(device)
            logits_cls, logits_bird, emb = self._net(chunk)
            logits_list.append(logits_cls.cpu().numpy().astype(np.float32))
            embs_list.append(emb.cpu().numpy().astype(np.float32))
            bird_list.append(logits_bird.cpu().numpy().astype(np.float32))

        return (
            np.concatenate(logits_list, axis=0),
            np.concatenate(embs_list, axis=0),
            np.concatenate(bird_list, axis=0),
        )

    def forward_embeddings(self, mels: np.ndarray) -> np.ndarray:
        """Return L2-normalised embeddings (N, D). Called by edge_engine."""
        single = mels.ndim == 2
        batch = mels[None, :, :] if single else mels
        _, emb, _ = self._forward_batch(batch)
        return emb[0] if single else emb

    def predict(self, mels: np.ndarray, temperature: float = 1.0) -> ModelOutput:
        t0 = time.perf_counter()
        single = mels.ndim == 2
        batch = mels[None, :, :] if single else mels

        logits, emb, bird_logits = self._forward_batch(batch)
        probs = softmax(logits, temperature=temperature)
        preds = np.argmax(probs, axis=-1)
        bird_prob = sigmoid(bird_logits)
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

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def param_count(self) -> int:
        return int(sum(p.numel() for p in self._net.parameters()))

    def storage_bytes(self) -> int:
        total = 0
        for p in self._net.parameters():
            total += p.numel() * p.element_size()
        return total

    def save_checkpoint(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self._net.state_dict(),
                "config": {
                    "n_mels": self.n_mels,
                    "embedding_dim": self.embedding_dim,
                    "num_classes": self.num_classes,
                    "seed": self.seed,
                    "model_id": self.model_id,
                },
                "training_log": self._training_log,
            },
            path,
        )

    def load_checkpoint(self, path: Path) -> None:
        path = Path(path)
        ckpt = torch.load(path, map_location=self._device, weights_only=False)
        self._net.load_state_dict(ckpt["model_state_dict"])
        cfg = ckpt.get("config", {})
        self.model_id = cfg.get("model_id", self.model_id)
        self._training_log = ckpt.get("training_log", [])
        self._net.eval()


# ---------------------------------------------------------------------------
# ExpertTransformer — genuine trainable patch-based Transformer
# ---------------------------------------------------------------------------

class _PatchEmbed(nn.Module):
    """
    Split a Log-Mel spectrogram into non-overlapping patches and linearly embed them.

    Input  : (B, 1, n_mels, n_frames)
    Output : (B, n_patches, hidden_dim)
    """

    def __init__(
        self,
        n_mels: int = 64,
        n_frames: int = 301,
        patch_mel: int = 8,
        patch_time: int = 8,
        hidden_dim: int = 192,
    ) -> None:
        super().__init__()
        self.patch_mel = patch_mel
        self.patch_time = patch_time
        # Pad n_frames to be divisible by patch_time
        self.n_patches_mel = n_mels // patch_mel
        self.n_patches_time = math.ceil(n_frames / patch_time)
        self.n_patches = self.n_patches_mel * self.n_patches_time
        patch_dim = patch_mel * patch_time
        self.proj = nn.Linear(patch_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self._n_frames = n_frames
        self._n_mels = n_mels

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        B, C, M, T = x.shape
        # Pad time to multiple of patch_time
        pad_t = self.n_patches_time * self.patch_time - T
        if pad_t > 0:
            x = F.pad(x, (0, pad_t))
        x = x.squeeze(1)  # (B, M, T_pad)
        # Unfold into patches
        patches = x.unfold(1, self.patch_mel, self.patch_mel).unfold(
            2, self.patch_time, self.patch_time
        )  # (B, n_mel_patches, n_time_patches, patch_mel, patch_time)
        B, nm, nt, pm, pt = patches.shape
        patches = patches.contiguous().view(B, nm * nt, pm * pt)  # (B, n_patches, patch_dim)
        return self.norm(self.proj(patches))


class _TransformerEncoderLayer(nn.Module):
    """Pre-norm Transformer encoder layer with GELU feed-forward."""

    def __init__(self, hidden_dim: int, n_heads: int, ff_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # Self-attention with pre-norm
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + self.drop(attn_out)
        # Feed-forward with pre-norm
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class _ExpertNet(nn.Module):
    """
    Transformer-based audio classification network.

    Architecture:
      PatchEmbed → learnable CLS token + positional encoding →
      N × TransformerEncoderLayer → LayerNorm → CLS head →
      classification logits + bird-activity logit + embedding
    """

    def __init__(
        self,
        n_mels: int = 64,
        n_frames: int = 301,
        patch_mel: int = 8,
        patch_time: int = 8,
        hidden_dim: int = 192,
        n_layers: int = 4,
        n_heads: int = 4,
        ff_dim: int = 384,
        embedding_dim: int = 128,
        num_classes: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_embed = _PatchEmbed(n_mels, n_frames, patch_mel, patch_time, hidden_dim)
        n_patches = self.patch_embed.n_patches

        # CLS token + learnable positional encoding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches + 1, hidden_dim) * 0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            _TransformerEncoderLayer(hidden_dim, n_heads, ff_dim, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, embedding_dim)
        self.cls_head = nn.Linear(embedding_dim, num_classes)
        self.bird_head = nn.Linear(embedding_dim, 1)

    def forward(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        # x: (B, 1, n_mels, n_frames)
        tokens = self.patch_embed(x)                        # (B, n_patches, hidden_dim)
        B = tokens.shape[0]
        cls = self.cls_token.expand(B, -1, -1)              # (B, 1, hidden_dim)
        tokens = torch.cat([cls, tokens], dim=1)            # (B, n_patches+1, hidden_dim)
        tokens = self.drop(tokens + self.pos_embed[:, :tokens.shape[1], :])

        for layer in self.layers:
            tokens = layer(tokens)
        tokens = self.norm(tokens)

        cls_rep = tokens[:, 0]                              # (B, hidden_dim)
        emb = F.normalize(self.proj(cls_rep), dim=-1)       # (B, embedding_dim)
        return self.cls_head(emb), self.bird_head(emb).squeeze(-1), emb


class ExpertTransformer:
    """
    Genuinely trainable Transformer-based audio classification model.

    Public interface matches the legacy ``StrongExpertAudioModel`` so all
    dependent scripts, tests and API code work without modification.
    """

    def __init__(
        self,
        n_mels: int = 64,
        embedding_dim: int = 128,
        num_classes: int = 4,
        seed: int = 42,
        # Transformer hyper-parameters
        n_frames: int = 301,
        patch_mel: int = 8,
        patch_time: int = 8,
        hidden_dim: int = 192,
        n_layers: int = 4,
        n_heads: int = 4,
        ff_dim: int = 384,
        dropout: float = 0.1,
    ) -> None:
        _require_torch("ExpertTransformer")
        self.n_mels = n_mels
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.seed = seed
        self.model_id = "ecogate_expert_transformer_pytorch_v1"
        self._training_log: List[Dict[str, Any]] = []

        seed_everything(seed)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._net = _ExpertNet(
            n_mels=n_mels,
            n_frames=n_frames,
            patch_mel=patch_mel,
            patch_time=patch_time,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            ff_dim=ff_dim,
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            dropout=dropout,
        ).to(self._device)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        train_mels: np.ndarray,
        y_known: np.ndarray,
        y_bird: np.ndarray,
        val_mels: Optional[np.ndarray] = None,
        val_y_known: Optional[np.ndarray] = None,
        epochs: int = 30,
        lr: float = 1e-4,
        weight_decay: float = 1e-3,
        batch_size: int = 32,
        patience: int = 8,
        checkpoint_path: Optional[Path] = None,
        **_kwargs: Any,
    ) -> Dict[str, Any]:
        """Train with AdamW + cosine LR, early stopping on validation macro-F1."""
        seed_everything(self.seed)
        net = self._net
        net.train()

        known_mask = y_known >= 0
        X_t = torch.from_numpy(train_mels[:, None, :, :])  # (N,1,M,T)
        X_k = X_t[known_mask]
        yk_t = torch.from_numpy(y_known[known_mask].astype(np.int64))
        yb_t = torch.from_numpy(y_bird[known_mask].astype(np.float32))

        class_counts = np.bincount(y_known[known_mask], minlength=self.num_classes).astype(np.float32)
        class_weights = torch.tensor(
            float(np.sum(class_counts)) / (self.num_classes * np.maximum(class_counts, 1)),
            dtype=torch.float32,
        ).to(self._device)

        ds = TensorDataset(X_k, yk_t, yb_t)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.05)

        best_val_f1 = -1.0
        patience_ctr = 0
        self._training_log = []

        for epoch in range(epochs):
            net.train()
            epoch_cls_loss = 0.0
            n_batches = 0

            for Xb, ykb, ybb in dl:
                Xb = Xb.to(self._device)
                ykb = ykb.to(self._device)
                ybb = ybb.to(self._device)

                opt.zero_grad()
                logits_cls, logits_bird, _ = net(Xb)
                loss_cls = F.cross_entropy(logits_cls, ykb, weight=class_weights)
                loss_bird = F.binary_cross_entropy_with_logits(logits_bird, ybb)
                loss = loss_cls + 0.3 * loss_bird
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                opt.step()
                epoch_cls_loss += loss_cls.item()
                n_batches += 1

            sched.step()
            avg_cls = epoch_cls_loss / max(1, n_batches)

            val_f1 = -1.0
            val_acc = -1.0
            val_loss = float("inf")
            if val_mels is not None and val_y_known is not None:
                v_mask = val_y_known >= 0
                if np.any(v_mask):
                    v_out = self.predict(val_mels[v_mask])
                    val_f1 = _macro_f1(val_y_known[v_mask], v_out.predicted_class, self.num_classes)
                    val_acc = float(np.mean(v_out.predicted_class == val_y_known[v_mask]))
                    with torch.no_grad():
                        Xv = torch.from_numpy(val_mels[v_mask][:, None, :, :]).to(self._device)
                        yv = torch.from_numpy(val_y_known[v_mask].astype(np.int64)).to(self._device)
                        logits_v, _, _ = net(Xv)
                        val_loss = F.cross_entropy(logits_v, yv).item()

            current_lr = sched.get_last_lr()[0]
            log_entry = {
                "epoch": epoch + 1,
                "train_cls_loss": avg_cls,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_macro_f1": val_f1,
                "lr": current_lr,
            }
            self._training_log.append(log_entry)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_ctr = 0
                if checkpoint_path is not None:
                    self.save_checkpoint(checkpoint_path)
            else:
                patience_ctr += 1
                if patience > 0 and patience_ctr >= patience:
                    break

        if checkpoint_path is not None and checkpoint_path.exists():
            self.load_checkpoint(checkpoint_path)

        return {
            "training_log": self._training_log,
            "best_val_macro_f1": best_val_f1,
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _forward_batch(
        self, mels: np.ndarray, chunk_size: int = 64
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run batched inference; returns (logits, embeddings, bird_logits) as numpy arrays."""
        self._net.eval()
        device = next(self._net.parameters()).device
        n = mels.shape[0]
        if n == 0:
            return (
                np.zeros((0, self.num_classes), dtype=np.float32),
                np.zeros((0, self.embedding_dim), dtype=np.float32),
                np.zeros(0, dtype=np.float32),
            )
        logits_list = []
        embs_list = []
        bird_list = []
        for i in range(0, n, chunk_size):
            chunk = torch.from_numpy(mels[i : i + chunk_size, None, :, :]).to(device)
            logits_cls, logits_bird, emb = self._net(chunk)
            logits_list.append(logits_cls.cpu().numpy().astype(np.float32))
            embs_list.append(emb.cpu().numpy().astype(np.float32))
            bird_list.append(logits_bird.cpu().numpy().astype(np.float32))

        return (
            np.concatenate(logits_list, axis=0),
            np.concatenate(embs_list, axis=0),
            np.concatenate(bird_list, axis=0),
        )

    def forward_embeddings(self, mels: np.ndarray) -> np.ndarray:
        single = mels.ndim == 2
        batch = mels[None, :, :] if single else mels
        _, emb, _ = self._forward_batch(batch)
        return emb[0] if single else emb

    def predict(self, mels: np.ndarray, temperature: float = 1.0) -> ModelOutput:
        t0 = time.perf_counter()
        single = mels.ndim == 2
        batch = mels[None, :, :] if single else mels

        logits, emb, bird_logits = self._forward_batch(batch)
        probs = softmax(logits, temperature=temperature)
        preds = np.argmax(probs, axis=-1)
        bird_prob = sigmoid(bird_logits)
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

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def param_count(self) -> int:
        return int(sum(p.numel() for p in self._net.parameters()))

    def storage_bytes(self) -> int:
        total = 0
        for p in self._net.parameters():
            total += p.numel() * p.element_size()
        return total

    def save_checkpoint(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self._net.state_dict(),
                "config": {
                    "n_mels": self.n_mels,
                    "embedding_dim": self.embedding_dim,
                    "num_classes": self.num_classes,
                    "seed": self.seed,
                    "model_id": self.model_id,
                },
                "training_log": self._training_log,
            },
            path,
        )

    def load_checkpoint(self, path: Path) -> None:
        path = Path(path)
        ckpt = torch.load(path, map_location=self._device, weights_only=False)
        self._net.load_state_dict(ckpt["model_state_dict"])
        cfg = ckpt.get("config", {})
        self.model_id = cfg.get("model_id", self.model_id)
        self._training_log = ckpt.get("training_log", [])
        self._net.eval()


# ---------------------------------------------------------------------------
# Convenience: unified train_model() function
# ---------------------------------------------------------------------------

def train_model(
    model: "StudentCNN | ExpertTransformer",
    train_mels: np.ndarray,
    y_known: np.ndarray,
    y_bird: np.ndarray,
    val_mels: Optional[np.ndarray] = None,
    val_y_known: Optional[np.ndarray] = None,
    seeds: List[int] = None,
    checkpoint_dir: Optional[Path] = None,
    **fit_kwargs: Any,
) -> Dict[str, Any]:
    """
    Train ``model`` with one or more random seeds, returning the run with the
    best validation macro-F1.  When ``seeds`` has more than one element the
    function keeps the best checkpoint across seeds.
    """
    if seeds is None:
        seeds = [model.seed]

    best_f1 = -1.0
    best_result: Dict[str, Any] = {}
    best_seed = seeds[0]

    for seed in seeds:
        model.seed = seed
        ckpt_path = None
        if checkpoint_dir is not None:
            ckpt_path = Path(checkpoint_dir) / f"checkpoint_seed{seed}.pt"

        result = model.fit(
            train_mels,
            y_known,
            y_bird,
            val_mels=val_mels,
            val_y_known=val_y_known,
            checkpoint_path=ckpt_path,
            **fit_kwargs,
        )
        f1 = result.get("best_val_macro_f1", -1.0)
        if f1 > best_f1:
            best_f1 = f1
            best_result = result
            best_seed = seed

    # Reload the globally best checkpoint
    if checkpoint_dir is not None:
        best_ckpt = Path(checkpoint_dir) / f"checkpoint_seed{best_seed}.pt"
        if best_ckpt.exists():
            model.load_checkpoint(best_ckpt)
            # Also save as canonical "best" checkpoint
            best_canonical = Path(checkpoint_dir) / "best_checkpoint.pt"
            model.save_checkpoint(best_canonical)

    best_result["best_seed"] = best_seed
    best_result["all_seeds"] = seeds
    return best_result


# ---------------------------------------------------------------------------
# FLOP estimation
# ---------------------------------------------------------------------------

def estimate_flops(model: "StudentCNN | ExpertTransformer", input_shape: Tuple = (1, 64, 301)) -> int:
    """
    Rough FLOP count via fvcore (if available) or simple parameter-based heuristic.
    Returns integer FLOPs; -1 if estimation is not possible.
    """
    try:
        from fvcore.nn import FlopCountAnalysis
        _require_torch("estimate_flops")
        x = torch.zeros(1, 1, *input_shape[1:]).to(model._device)
        flops = FlopCountAnalysis(model._net, x)
        return int(flops.total())
    except Exception:
        # Fallback: very rough heuristic (2 × params × assumed reuse)
        return model.param_count() * 2

