# System Architecture (`docs/architecture/system_architecture.md`)

```
Audio Input (.wav / stream)
      │
      ▼
[1. Audio Quality Gate] ──(Corrupt / Clipped > 12% / SNR < -5 dB)──► Return `unresolved`
      │
      ▼
[2. Polyphase Resampler (32 kHz) + RMS Normalizer + 64-Band Log-Mel Extractor]
      │
      ▼
[3. Lightweight Student CNN (`ecogate_student_cnn_v1` / INT8)]
      ├──► Bird Activity Probability p_bird
      ├──► Known-Species Logits & Calibrated Probabilities
      ├──► 64-D L2-Normalized Embedding -> Prototype Distance & Open-Set Scores
      └──► Pre-Expert Feature Vector z (9 base + 3 interaction features)
      │
      ▼
[4. Cost-Aware Routing Mechanism (`CostAwareRouter` + `BudgetController`)]
      ├──► If p_bird < tau_bird (and low benefit): Return `nonbird`
      ├──► Estimate Expected Loss Reduction: Delta_hat(z) = BenefitEstimator(z)
      ├──► If Delta_hat(z) > tau*(lambda, Budget):
      │          Invoke Strong Expert (`ecogate_expert_multiscale_ast_v1`)
      │    Else:
      │          Accept Student Output
      │
      ▼
[5. Open-Set Novelty Gate (`OpenSetDetector`)]
      ├──► If NoveltyScore > tau_open_set: Return `bird_unfamiliar`
      └──► Else: Return `known_species` (with calibrated species prediction)
      │
      ▼
[6. Calibrated Final Prediction & Resource Consumption Report]
```
