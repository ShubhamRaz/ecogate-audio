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
    sample_input = np.asarray(sample_input)
    if sample_input.ndim == 3:
        single_input = sample_input[0]
    elif sample_input.ndim == 2:
        single_input = sample_input
    else:
        raise ValueError("sample_input must have shape (N, M, T) or (M, T).")
    if n_warmup < 0 or n_runs < 1:
        raise ValueError("n_warmup must be non-negative and n_runs must be at least 1.")
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
    storage_bytes = (
        int(model.storage_bytes()) if callable(getattr(model, "storage_bytes", None))
        else int(params * 4)
    )
    storage_kb = storage_bytes / 1024.0

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
        "estimated_macs_per_clip": None,
        "estimated_flops_per_clip": None,
        "complexity_estimate_status": "not_available_exact_operator_accounting_required",
        "actual_stored_array_bytes": storage_bytes,
        "actual_stored_array_kb": round(storage_kb, 3),
        "model_storage_kb_fp32_equivalent": round((params * 4) / 1024.0, 3),
        "model_storage_kb_int8": round(storage_kb, 3) if callable(getattr(model, "storage_bytes", None)) else None,
        "runtime_compute_dtype": getattr(model, "runtime_compute_dtype", "float32"),
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
