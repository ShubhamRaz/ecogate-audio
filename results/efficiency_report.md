# Hardware, Latency, Quantization & Energy Efficiency Report (`results/efficiency_report.md`)

## 1. Model Complexity & Measured CPU Latency (`Windows-10-10.0.26300-SP0`)

| Model Variant | Parameters | Est. MACs / Clip | Est. FLOPs / Clip | Storage (FP32 / INT8) | Batch-1 CPU Latency (Mean / P95) | Peak Memory | Electrical Energy Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| Student CNN (FP32) | 13925 | 4191425 | 8382850 | 54.39 KB | 1.786 ms / 2.558 ms | 28.12 KB | `not_measured_no_physical_power_meter_attached` |
| Student CNN (INT8) | 13765 | 4143265 | 8286530 | 13.44 KB | 1.617 ms / 1.96 ms | 75.34 KB | `not_measured_no_physical_power_meter_attached` |
| Strong Expert (FP32) | 82821 | 24929121 | 49858242 | 323.52 KB | 5.021 ms / 7.638 ms | 3986.58 KB | `not_measured_no_physical_power_meter_attached` |

## 2. Energy / Compute Cost per Correct Prediction (Proposed Cost-Aware Policy)
- **Numerator (Total Compute Cost Units)**: `2741.15`
- **Numerator (Measured Electrical Joules)**: `None` (`not_measured_proxy_cost_units_only`)
- **Denominator (Number of Correct Predictions)**: `517` out of `615` evaluated test clips
- **Compute Cost Units per Correct Prediction**: `5.302`
- **Raspberry Pi Zero 2 W Assessment**: Constrained 512 MB RAM / Cortex-A53 target is suitable for the INT8 Student + Quality Gate (`13.44 KB` weights), whereas continuous on-device invocation of the full transformer/attention expert is not guaranteed real-time without offloading or selective gating.
