"""
Edge Deployment, Dynamic INT8 Quantization, Post-Quantization Recalibration,
and Reference vs Raspberry Pi Zero 2 W Deployment Evaluation for EcoGate-Audio.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np

from src.calibration.calibrator import TemperatureScaler
from src.models.audio_models import LightweightStudentCNN, ModelOutput, softmax, sigmoid, silu

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def quantize_array_int8(arr: np.ndarray) -> Tuple[np.ndarray, float]:
    max_val = float(np.max(np.abs(arr))) + 1e-12
    scale = max_val / 127.0
    q = np.clip(np.round(arr / scale), -127, 127).astype(np.int8)
    return q, scale


def dequantize_array_int8(q: np.ndarray, scale: float) -> np.ndarray:
    return q.astype(np.float32) * scale


class TorchQuantizedStudentModel:
    """
    Genuine integer INT8 dynamic-quantized model for StudentCNN.
    Uses PyTorch integer quantization (torch.ao.quantization.quantize_dynamic) over
    linear projection and classification heads, executing genuine integer arithmetic
    (qint8 GEMM) on CPU at runtime.
    """

    def __init__(self, pytorch_student: Any) -> None:
        if not _TORCH_AVAILABLE:
            raise RuntimeError("TorchQuantizedStudentModel requires PyTorch.")
        self.model_id = f"{pytorch_student.model_id}_qint8_dynamic_quantized"
        self.num_classes = pytorch_student.num_classes
        self.recalibrated_temperature: float = 1.0
        self.runtime_compute_dtype: str = "qint8"

        import copy
        # Deepcopy to CPU so we do NOT mutate the original student model in-place!
        cpu_net = copy.deepcopy(pytorch_student._net).cpu().eval()
        # Apply dynamic integer quantization to Linear modules
        self._quantized_net = torch.ao.quantization.quantize_dynamic(
            cpu_net,
            {nn.Linear},
            dtype=torch.qint8,
        )
        self._quantized_net.eval()

    def param_count(self) -> int:
        return int(sum(p.numel() for p in self._quantized_net.parameters()))

    def storage_bytes(self) -> int:
        """Estimate actual storage bytes for quantized weights and fp32 conv layers."""
        total = 0
        for p in self._quantized_net.parameters():
            total += p.numel() * p.element_size()
        return total

    def recalibrate(self, cal_mels: np.ndarray, cal_y_known: np.ndarray) -> Dict[str, float]:
        out = self.predict(cal_mels, temperature=1.0)
        scaler = TemperatureScaler(init_temperature=1.0)
        stats = scaler.fit(out.logits, cal_y_known)
        self.recalibrated_temperature = scaler.temperature
        return stats

    @torch.no_grad()
    def _forward_batch(self, mels: np.ndarray, chunk_size: int = 128) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = mels.shape[0]
        if n == 0:
            return (
                np.zeros((0, self.num_classes), dtype=np.float32),
                np.zeros((0, 64), dtype=np.float32),
                np.zeros(0, dtype=np.float32),
            )
        logits_list = []
        embs_list = []
        bird_list = []
        for i in range(0, n, chunk_size):
            X = torch.from_numpy(mels[i : i + chunk_size, None, :, :].astype(np.float32))
            logits_cls, logits_bird, emb = self._quantized_net(X)
            logits_list.append(logits_cls.numpy().astype(np.float32))
            embs_list.append(emb.numpy().astype(np.float32))
            bird_list.append(logits_bird.numpy().astype(np.float32))
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

    def predict(self, mels: np.ndarray, temperature: Optional[float] = None) -> ModelOutput:
        t0 = time.perf_counter()
        temp = temperature if temperature is not None else self.recalibrated_temperature
        single = mels.ndim == 2
        batch = mels[None, :, :] if single else mels

        logits, emb, bird_logits = self._forward_batch(batch)
        probs = softmax(logits, temperature=temp)
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

    def export_quantized_artifact(self, path: Path) -> Dict[str, Any]:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self._quantized_net.state_dict(),
                "model_id": self.model_id,
                "recalibrated_temperature": self.recalibrated_temperature,
                "quantization_type": "dynamic_int8_linear",
                "runtime_compute_dtype": "qint8",
            },
            path,
        )
        return {
            "artifact_path": str(path),
            "size_bytes": path.stat().st_size,
            "weights_storage_dtype": "qint8",
            "runtime_compute_dtype": "qint8",
            "integer_matmul": True,
            "export_note": "Dynamic INT8 quantized PyTorch model with genuine integer linear operations.",
        }


class QuantizedStudentEdgeModel:
    """
    INT8-compressed weight storage with post-compression temperature recalibration.

    IMPORTANT: predict() dequantizes weights to float32 before matrix multiplication.
    This is storage compression, not integer inference or an INT8 compute-speedup path.
    """

    def __init__(self, fp32_student: LightweightStudentCNN) -> None:
        self.model_id = f"{fp32_student.model_id}_int8_weight_storage_fp32_compute"
        self.bn_mean = fp32_student.bn_mean.copy()
        self.bn_std = fp32_student.bn_std.copy()
        self.b_proj = fp32_student.b_proj.copy()
        self.b_res = fp32_student.b_res.copy()
        self.b_cls = fp32_student.b_cls.copy()
        self.b_bird = float(fp32_student.b_bird)

        self.W_proj_q, self.s_proj = quantize_array_int8(fp32_student.W_proj)
        self.W_res_q, self.s_res = quantize_array_int8(fp32_student.W_res)
        self.W_cls_q, self.s_cls = quantize_array_int8(fp32_student.W_cls)
        self.W_bird_q, self.s_bird = quantize_array_int8(fp32_student.W_bird)
        self._raw_extractor = fp32_student._extract_raw_conv_features
        self.recalibrated_temperature: float = 1.0

    def param_count(self) -> int:
        return int(self.W_proj_q.size + self.b_proj.size + self.W_res_q.size + self.b_res.size + self.W_cls_q.size + self.b_cls.size + self.W_bird_q.size + 1)

    def storage_bytes(self) -> int:
        """Actual stored ndarray bytes, including float32 biases and normalizers."""
        arrays = (self.W_proj_q, self.W_res_q, self.W_cls_q, self.W_bird_q,
                  self.b_proj, self.b_res, self.b_cls, self.bn_mean, self.bn_std)
        scalar_arrays = np.asarray(
            [self.s_proj, self.s_res, self.s_cls, self.s_bird,
             self.b_bird, self.recalibrated_temperature],
            dtype=np.float32,
        )
        return int(sum(array.nbytes for array in arrays) + scalar_arrays.nbytes)

    def recalibrate(self, cal_mels: np.ndarray, cal_y_known: np.ndarray) -> Dict[str, float]:
        out = self.predict(cal_mels, temperature=1.0)
        scaler = TemperatureScaler(init_temperature=1.0)
        stats = scaler.fit(out.logits, cal_y_known)
        self.recalibrated_temperature = scaler.temperature
        return stats

    def predict(self, mels: np.ndarray, temperature: Optional[float] = None) -> ModelOutput:
        t0 = time.perf_counter()
        temp = temperature if temperature is not None else self.recalibrated_temperature
        single = mels.ndim == 2
        batch_mels = mels[None, :, :] if single else mels

        raw = self._raw_extractor(batch_mels)
        normed = (raw - self.bn_mean) / (self.bn_std + 1e-5)

        W_proj = dequantize_array_int8(self.W_proj_q, self.s_proj)
        W_res = dequantize_array_int8(self.W_res_q, self.s_res)
        W_cls = dequantize_array_int8(self.W_cls_q, self.s_cls)
        W_bird = dequantize_array_int8(self.W_bird_q, self.s_bird)

        h1 = silu(normed @ W_proj + self.b_proj)
        h2 = silu(h1 + (h1 @ W_res + self.b_res) * 0.5)
        emb = (h2 / (np.linalg.norm(h2, axis=-1, keepdims=True) + 1e-8)).astype(np.float32)

        logits = (emb @ W_cls + self.b_cls) * 2.5
        probs = softmax(logits, temperature=temp)
        preds = np.argmax(probs, axis=-1)
        bird_prob = sigmoid((emb @ W_bird + self.b_bird) * 2.0)
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

    def export_int8_artifact(self, path: Path) -> Dict[str, Any]:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            W_proj_q=self.W_proj_q,
            s_proj=np.array([self.s_proj], dtype=np.float32),
            W_res_q=self.W_res_q,
            s_res=np.array([self.s_res], dtype=np.float32),
            W_cls_q=self.W_cls_q,
            s_cls=np.array([self.s_cls], dtype=np.float32),
            W_bird_q=self.W_bird_q,
            s_bird=np.array([self.s_bird], dtype=np.float32),
            b_proj=self.b_proj,
            b_res=self.b_res,
            b_cls=self.b_cls,
            b_bird=np.array([self.b_bird], dtype=np.float32),
            bn_mean=self.bn_mean,
            bn_std=self.bn_std,
            recalibrated_temperature=np.array([self.recalibrated_temperature], dtype=np.float32),
        )
        return {
            "artifact_path": str(path),
            "size_bytes": path.stat().st_size,
            "onnx_runtime_available": False,
            "weights_storage_dtype": "int8",
            "runtime_compute_dtype": "float32",
            "integer_matmul": False,
            "export_note": (
                "INT8 weight arrays are stored in NPZ, but predict() dequantizes them to float32 "
                "before matrix multiplication. Integer-only inference and ONNX Runtime acceleration "
                "are not implemented by this class."
            ),
        }
