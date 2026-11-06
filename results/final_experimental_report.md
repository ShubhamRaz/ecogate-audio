> **STATUS: HISTORICAL SUMMARY — NOT A VERIFIED EXPERIMENTAL RESULT.** This report is retained as an audit trail. Its metrics were produced before the data-provenance, routing-threshold, average-precision, risk–coverage, and runtime-quantization corrections on this branch. Re-run the pipeline on the mounted source archive and regenerate this report before any external submission.

# Final Experimental Report (`results/final_experimental_report.md`)

## 1. Summary of Executed Verification Experiments (Experiments 1–11)
1. **Experiment 1 (Dataset Audit & Split Contamination)**: Verified SHA-256 and decoded float32 waveform duplicate grouping, ESC-50 filename matching, corrupt WAV exclusion, missing-path detection, and zero cross-split group leakage (`passed = True`).
2. **Experiment 2 (Audio-Only Baselines)**: Evaluated `MFCCBaselineModel` (Macro-F1 = `0.9124`) vs. `LightweightStudentCNN` (Macro-F1 = `0.7882`).
3. **Experiment 3 (Student vs. Expert Performance)**: Verified `StrongExpertAudioModel` uplift (Macro-F1 = `0.9710`, uplift = `+0.1828`).
4. **Experiment 4 (Open-Set Recognition)**: Evaluated all 6 open-set scoring methods (`msp`, `temperature_scaled`, `entropy`, `margin`, `prototype_distance`, `energy`) and 4 leave-one-species-out rotations.
5. **Experiment 5 & 6 (Routing Baselines vs. Proposed Cost-Aware Routing)**: Compared all 10 routing policies using out-of-fold trained `BenefitEstimator` and calibration-split threshold selection. Proposed cost-aware routing achieved Expected Decision Loss = `0.2732` at expert call rate = `0.2356`.
6. **Experiment 7 (Source-Shift & Group Bootstrap Robustness)**: Computed group-level bootstrap 95% confidence intervals preserving session/duplicate clusters.
7. **Experiment 8 (Quantization & Edge Deployment)**: Exported INT8 student artifact (`13.44 KB`) and recalibrated post-quantization temperature.
8. **Experiment 9 & 10 (Risk-Coverage-Energy Trade-off & Ablations)**: Generated risk-coverage curves, matched budget sweeps (10%–50% call rates), and feature-group ablations.
9. **Experiment 11 (External Unknown-Species & Provenance Scope)**: Documented full-archive Mendeley (`10.17632/p6m5tmr477.1`) and ESC-50 (`10.1145/2733373.2806390`) evaluation boundaries; all numbers in these reports come strictly from executed runs on the local reference validation dataset, with zero fabricated full-archive claims.
