# Experimental Protocol (`docs/research/experimental_protocol.md`)

## 1. Strict Partitioning & Zero-Leakage Protocol
- **Group-Safe Partitioning**: All clips sharing a decoded waveform SHA-256 hash (`duplicate_group_id`), ESC-50 source clip ID, or explicit recording session prefix are linked via Union-Find and placed in the same split: **Train (60%)**, **Validation (15%)**, **Calibration (10%)**, and **Test (15%)**.
- **Calibration Separation**: Temperature scaling parameters ($T_s, T_e$), open-set novelty thresholds, and routing budget thresholds $\tau^*$ are tuned exclusively on the Calibration split.
- **Pre-Expert Feature Isolation**: Automated unit tests (`test_06_leakage_prevention_routing_and_budget_constraints`) verify that passing any expert output feature raises an immediate `ValueError`.

## 2. Eleven Core Experiments
1. **Exp 1**: Dataset audit, missing-path verification, ESC-50 provenance check, and split contamination verification.
2. **Exp 2**: Classical MFCC logistic baseline vs. compact Student CNN comparison.
3. **Exp 3**: Student vs. Strong Expert uplift verification on held-out known species.
4. **Exp 4**: Six-method open-set comparison (`msp`, `temperature_scaled`, `entropy`, `margin`, `prototype_distance`, `energy`) and 4 leave-one-species-out rotations.
5. **Exp 5**: Unsupervised uncertainty routing baselines (`random`, `fixed_confidence`, `entropy`, `margin`, `prototype_distance`).
6. **Exp 6**: Proposed out-of-fold cost-aware benefit routing vs. cost-blind benefit routing and offline oracle upper bound.
7. **Exp 7**: Group-level bootstrap confidence intervals accounting for intra-session correlation.
8. **Exp 8**: Dynamic INT8 student quantization, post-quantization temperature recalibration, and edge profiling.
9. **Exp 9**: Risk-coverage curves and matched expert-call budget sweeps ($10\%$ to $50\%$).
10. **Exp 10**: Ablation study removing cost penalty, bird-activity gating, and learned benefit regression.
11. **Exp 11**: External unknown-species and provenance boundary analysis.
