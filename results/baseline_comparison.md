# Audio-Only Baseline & Model Tier Comparison (`results/baseline_comparison.md`)

## 1. Held-Out Known-Species Classification Comparison

| Model Tier | Model ID | Accuracy | Macro-F1 | Weighted-F1 | Balanced Accuracy | Parameters |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Classical Baseline | `ecogate_mfcc_logreg_v1` | 0.9333 | 0.9326 | 0.9331 | 0.9326 | 605 |
| Trainable Student 2D CNN (FP32 runtime) | `ecogate_student_cnn_pytorch_v1` | 0.9783 | 0.9790 | 0.9784 | 0.9807 | 198325 |
| Dynamic INT8 Quantized Student (qint8 runtime) | `ecogate_student_cnn_pytorch_v1_qint8_dynamic_quantized` | 0.9783 | 0.9790 | 0.9784 | 0.9807 | 189744 |
| Trainable Expert Audio Transformer (FP32 runtime) | `ecogate_expert_transformer_pytorch_v1` | 0.9167 | 0.9169 | 0.9168 | 0.9139 | 1285445 |

## 2. Descriptive Model Difference
- **Expert vs. Student Macro-F1 Difference**: `-0.0621`
- This is a descriptive result from one held-out partition, not a pre-registered acceptance gate or proof of generalization. Report per-group uncertainty and replicate on a source-independent archive before making superiority claims.
