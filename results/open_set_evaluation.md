> **STATUS: HISTORICAL OUTPUT — RECOMPUTE BEFORE CITATION.** These numbers predate the tie-aware ROC/AP implementation now on the fix branch. They are preserved for traceability only; use fresh output from the corrected pipeline, and evaluate truly held-out taxa separately from generic background noise.

# Open-Set Recognition & Held-Out Species Evaluation (`results/open_set_evaluation.md`)

## 1. Score Direction & Evaluation Protocol
- **Score Direction**: Higher `novelty_score` indicates higher likelihood of an unfamiliar bird (`bird_unfamiliar`) or non-bird environmental sound (`nonbird`).
- **Downstream Open-Set Protocol**: Thresholds are calibrated exclusively on the independent calibration split (`target_known_recall = 0.90`). Pretraining exposure is not controlled when using external pretrained backbones, and `other_birds` is evaluated as an auxiliary open-set bird pool rather than claimed as biological discovery of unseen species.

## 2. Comparison of Six Open-Set Rejection Methods on Test Split

| Method | AUROC | AUPR | FPR95 | Unknown Recall | False Known-Species Rate | Calibrated ECE | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `msp` | 0.5554 | 0.5444 | 0.8748 | 0.1140 | 0.8860 | 0.0442 | 0.3277 |
| `temperature_scaled` | 0.5712 | 0.5579 | 0.8797 | 0.1023 | 0.8977 | 0.0442 | 0.3277 |
| `entropy` | 0.5743 | 0.5668 | 0.8780 | 0.1257 | 0.8743 | 0.0442 | 0.3277 |
| `margin` | 0.5456 | 0.5362 | 0.8797 | 0.1009 | 0.8991 | 0.0442 | 0.3277 |
| `prototype_distance` | 0.9478 | 0.9479 | 0.2520 | 0.8465 | 0.1535 | 0.0442 | 0.3277 |
| `energy` | 0.6586 | 0.6125 | 0.9089 | 0.1316 | 0.8684 | 0.0442 | 0.3277 |

## 3. Leave-One-Species-Out Downstream Rotations

| Held-Out Bird Species | Open-Set AUROC | Open-Set AUPR | FPR95 | Held-Out Species Rejection Rate | False Known-Species Assignment Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `common_waxbill` | 0.9196 | 0.9468 | 0.4297 | 0.4359 | 0.5641 |
| `red_billed_quelea` | 0.8832 | 0.9377 | 0.6439 | 0.2775 | 0.7225 |
| `village_weaver` | 0.8929 | 0.9380 | 0.5156 | 0.1257 | 0.8743 |
| `yellow_fronted_canary` | 0.9205 | 0.9499 | 0.4147 | 0.2071 | 0.7929 |
