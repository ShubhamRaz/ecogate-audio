# Final Experimental Report (`results/final_experimental_report.md`)

## 1. Summary of Executed Verification Experiments (Experiments 1–11)
1. **Experiment 1 (Dataset Audit & Split Contamination)**: Verified SHA-256 and decoded float32 waveform duplicate grouping, ESC-50 filename matching, corrupt WAV exclusion, missing-path detection, and zero cross-split group leakage (`passed = True`).
2. **Experiment 2 (Audio-Only Baselines)**: Evaluated the trainable 2D Student CNN (224K parameters) and patch-based Expert Audio Transformer (~1.48M parameters) against the MFCC baseline.
3. **Experiment 3 (Neural Architecture Comparison)**: Compared the trainable 2D Student CNN and patch-based Expert Audio Transformer on held-out test data.
4. **Experiment 4 (Open-Set Recognition)**: Evaluated all 6 open-set scoring methods (`msp`, `temperature_scaled`, `entropy`, `margin`, `prototype_distance`, `energy`) and 4 leave-one-species-out rotations.
5. **Experiment 5 & 6 (Routing Baselines vs. Budgeted Benefit Routing)**: Compared 10 routing policies using out-of-fold benefit estimates and calibration-only thresholds, subject to configured call-rate and energy caps.
6. **Experiment 7 (Source-Shift & Group Bootstrap Robustness)**: Computed group-level bootstrap 95% confidence intervals preserving session/duplicate clusters.
7. **Experiment 8 (Dynamic INT8 Quantization)**: Dynamic INT8 integer quantization (`qint8`) via PyTorch with genuine CPU integer linear matrix multiplication and calibration-only temperature recalibration.
8. **Experiment 9 & 10 (Risk-Coverage-Energy Trade-off & Ablations)**: Generated risk-coverage curves, matched budget sweeps (10%–50% call rates), and feature-group ablations.
9. **Experiment 11 (Dataset Scope & Authenticity)**: Quarantined 151 synthetic fixtures, validated 8,512 authentic field recordings from Mendeley Data (DOI: 10.17632/p6m5tmr477.1), and prevented synthetic samples from entering evaluation.
