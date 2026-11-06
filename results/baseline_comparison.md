# Audio-Only Baseline & Model Tier Comparison (`results/baseline_comparison.md`)

## 1. Held-Out Known-Species Classification Comparison

| Model Tier | Model ID | Accuracy | Macro-F1 | Weighted-F1 | Balanced Accuracy | Parameters |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Classical Baseline | `ecogate_mfcc_logreg_v1` | 0.9106 | 0.9124 | 0.9107 | 0.9140 | 605 |
| Lightweight Student (FP32) | `ecogate_student_cnn_v1` | 0.7919 | 0.7882 | 0.7936 | 0.7921 | 13925 |
| Quantized Student (INT8 Recalibrated) | `ecogate_student_cnn_v1_int8` | 0.7919 | 0.7886 | 0.7935 | 0.7927 | 13925 |
| Strong Expert Model | `ecogate_expert_multiscale_ast_v1` | 0.9707 | 0.9710 | 0.9707 | 0.9715 | 82821 |

## 2. Expert Uplift Verification Gate
- **Expert vs. Student Macro-F1 Uplift**: `+0.1828`
- **Minimum Required Uplift (config)**: `+0.0200`
- **Gate Status**: ✅ PASS — Expert demonstrates sufficient positive uplift over student; selective invocation is justified.
