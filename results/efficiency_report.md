# Hardware, Latency, Quantization & Energy Efficiency Report (`results/efficiency_report.md`)

## 1. Model Complexity & Measured CPU Latency (`Windows-10-10.0.26300-SP0`)

| Model Variant | Parameters | MACs / FLOPs | Actual Stored Arrays | Runtime Compute Dtype | Batch-1 CPU Latency (Mean / P95) | Peak Memory | Electrical Energy Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| Student 2D CNN (FP32 runtime) | 198325 | n/a | 774.707 KB | float32 | 8.795 ms / 10.131 ms | 4.23 KB | `not_measured_no_physical_power_meter_attached` |
| Student Dynamic Quantized (qint8 runtime) | 189744 | n/a | 741.188 KB | float32 | 17.709 ms / 21.3 ms | 79.04 KB | `not_measured_no_physical_power_meter_attached` |
| Expert Transformer (FP32 runtime) | 1285445 | n/a | 5021.27 KB | float32 | 13.519 ms / 15.913 ms | 4.66 KB | `not_measured_no_physical_power_meter_attached` |

## 2. Energy / Compute Cost per Correct Prediction (Proposed Cost-Aware Policy)
- **Numerator (Total Compute Cost Units)**: `1339.8`
- **Numerator (Measured Electrical Joules)**: `None` (`not_measured_proxy_cost_units_only`)
- **Denominator (Number of Correct Predictions)**: `587` out of `600` evaluated test clips
- **Compute Cost Units per Correct Prediction**: `2.2825`
- **Integer inference status**: `float32` runtime compute; PyTorch dynamic integer quantization (`torch.ao.quantization.quantize_dynamic`) executes integer linear GEMM operations on CPU. Calibrated temperature recalibration applied on held-out calibration split. Raspberry Pi Zero 2 W physical power-meter measurements pending.
- **Complexity accounting**: MAC/FLOP counts are unavailable until exact operator-level accounting is added; parameter count is not used as a MAC proxy.
