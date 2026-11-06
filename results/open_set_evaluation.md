# Open-Set Recognition & Held-Out Species Evaluation (`results/open_set_evaluation.md`)

## 1. Score Direction & Evaluation Protocol
- **Score Direction**: Higher `novelty_score` indicates higher likelihood of an unfamiliar bird (`bird_unfamiliar`) or non-bird environmental sound (`nonbird`).
- **Downstream Open-Set Protocol**: Thresholds are calibrated exclusively on the independent calibration split (`target_known_recall = 0.90`). Pretraining exposure is not controlled when using external pretrained backbones, and `other_birds` is evaluated as an auxiliary open-set bird pool rather than claimed as biological discovery of unseen species.

## 2. Comparison of Six Open-Set Rejection Methods on Test Split

| Method | AUROC | AUPR | FPR95 | Unknown Recall | False Known-Species Rate | Calibrated ECE | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `msp` | 0.8458 | 0.8547 | 0.7417 | 0.6450 | 0.3550 | 0.0106 | 0.0370 |
| `temperature_scaled` | 0.8711 | 0.8667 | 0.6083 | 0.6568 | 0.3432 | 0.0106 | 0.0370 |
| `entropy` | 0.8221 | 0.8489 | 0.8067 | 0.6183 | 0.3817 | 0.0106 | 0.0370 |
| `margin` | 0.8679 | 0.8710 | 0.5983 | 0.6657 | 0.3343 | 0.0106 | 0.0370 |
| `prototype_distance` | 0.8949 | 0.8879 | 0.4600 | 0.6657 | 0.3343 | 0.0106 | 0.0370 |
| `energy` | 0.6252 | 0.7132 | 0.9817 | 0.4098 | 0.5902 | 0.0106 | 0.0370 |

## 3. Leave-One-Species-Out Downstream Rotations

| Held-Out Bird Species | Open-Set AUROC | Open-Set AUPR | FPR95 | Held-Out Species Rejection Rate | False Known-Species Assignment Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `common_waxbill` | 0.9244 | 0.9525 | 0.3326 | 0.4425 | 0.5575 |
| `red_billed_quelea` | 0.9137 | 0.9536 | 0.4818 | 0.4599 | 0.5401 |
| `village_weaver` | 0.8769 | 0.9290 | 0.5584 | 0.0675 | 0.9325 |
| `yellow_fronted_canary` | 0.9020 | 0.9415 | 0.4428 | 0.0365 | 0.9635 |
