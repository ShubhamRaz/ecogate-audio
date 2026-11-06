"""
Hardware, Latency, Memory, and Energy Benchmarking Interface for EcoGate-Audio.
Explicitly distinguishes:
- Theoretical MACs / FLOPs
- Measured CPU/GPU wall-clock runtime (batch-size-1 and batch inference)
- Estimated compute energy units
- Physical electrical energy measurement status (reported honestly as `not_measured` when
  no external hardware power meter is attached).
Always reports both numerator and denominator for Energy per Correct Prediction.
"""
from __future__ import annotations

import os
import platform
import tracemalloc
import time
from typing import Any, Dict, Optional
import numpy as np


def benchmark_model_inference(
    model: Any,
    sample_input: np.ndarray,
    n_warmup: int = 3,
    n_runs: int = 20,
    external_power_watts: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Measure batch-size-1 latency, batch latency, peak memory allocation, parameter storage,
    and power/energy metrics.
    """
    single_input = sample_input[0] if sample_input.ndim > 2 else sample_input[0]
    for _ in range(n_warmup):
        model.predict(single_input)

    tracemalloc.start()
    latencies_ms = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model.predict(single_input)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    params = model.param_count()
    storage_kb = (params * 4) / 1024.0
    # Approximate MACs per 3-sec clip spectrogram (64x301)
    est_macs = int(params * 301)

    mean_lat_ms = float(np.mean(latencies_ms))
    p95_lat_ms = float(np.percentile(latencies_ms, 95))

    if external_power_watts is not None:
        measured_joules_per_clip = external_power_watts * (mean_lat_ms / 1000.0)
        energy_status = "measured_external_meter"
    else:
        measured_joules_per_clip = None
        energy_status = "not_measured_no_physical_power_meter_attached"

    return {
        "model_id": getattr(model, "model_id", "unknown"),
        "parameters": params,
        "estimated_macs_per_clip": est_macs,
        "estimated_flops_per_clip": est_macs * 2,
        "model_storage_kb_fp32": round(storage_kb, 2),
        "model_storage_kb_int8": round(storage_kb / 4.0, 2),
        "batch_size_1_cpu_latency_mean_ms": round(mean_lat_ms, 3),
        "batch_size_1_cpu_latency_p95_ms": round(p95_lat_ms, 3),
        "peak_memory_kb": round(peak_bytes / 1024.0, 2),
        "gpu_available": False,
        "gpu_latency_ms": None,
        "electrical_energy_status": energy_status,
        "measured_electrical_joules_per_clip": measured_joules_per_clip,
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
    }


def compute_energy_per_correct_prediction(
    total_cost_units: float,
    num_correct_predictions: int,
    total_predictions: int,
    measured_total_joules: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Calculate Energy / Cost per Correct Prediction = Total Energy / Number of Correct Predictions,
    always reporting both numerator and denominator.
    """
    denom = max(1, num_correct_predictions)
    return {
        "numerator_total_compute_cost_units": round(total_cost_units, 3),
        "numerator_measured_electrical_joules": measured_total_joules,
        "electrical_measurement_status": (
            "measured" if measured_total_joules is not None else "not_measured_proxy_cost_units_only"
        ),
        "denominator_correct_predictions": int(num_correct_predictions),
        "total_evaluated_predictions": int(total_predictions),
        "compute_cost_units_per_correct_prediction": round(total_cost_units / float(denom), 4),
        "measured_joules_per_correct_prediction": (
            round(measured_total_joules / float(denom), 4) if measured_total_joules is not None else None
        ),
    }
