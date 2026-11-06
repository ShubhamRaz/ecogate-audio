# Cost-Aware Expert Routing Comparison & Ablation Report (`results/routing_comparison.md`)

## 1. Ten-Policy Routing Comparison on Held-Out Test Split (Maximum Call-Rate Cap = 25%)

| Routing Policy | Expert Call Rate | Expected Decision Loss | Group Bootstrap 95% CI | Loss Reduction vs. Student | Improvement per Expert Call | Known-Species Macro-F1 | Avg. Compute Cost Units | Deployable |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `student_only` | 0.0000 | 0.3280 | [0.3036, 0.3497] | 0.0000 | 0.0000 | 0.9790 | 1.000x | `True` |
| `expert_only` | 1.0000 | 0.3556 | [0.3345, 0.3750] | -0.0276 | -0.0276 | 0.9169 | 4.500x | `True` |
| `random_routing` | 0.2500 | 0.3375 | [0.3114, 0.3591] | -0.0095 | -0.0378 | 0.9624 | 2.175x | `True` |
| `fixed_confidence_routing` | 0.2500 | 0.3461 | [0.3214, 0.3677] | -0.0181 | -0.0724 | 0.9813 | 2.175x | `True` |
| `entropy_routing` | 0.2445 | 0.3415 | [0.3165, 0.3621] | -0.0135 | -0.0550 | 0.9816 | 2.150x | `True` |
| `margin_routing` | 0.2500 | 0.3473 | [0.3208, 0.3693] | -0.0193 | -0.0770 | 0.9796 | 2.175x | `True` |
| `prototype_distance_routing` | 0.2500 | 0.3463 | [0.3202, 0.3677] | -0.0183 | -0.0731 | 0.9813 | 2.175x | `True` |
| `cost_blind_benefit_routing` | 0.0125 | 0.3291 | [0.3037, 0.3514] | -0.0011 | -0.0875 | 0.9769 | 1.106x | `True` |
| `proposed_cost_aware_routing` | 0.0000 | 0.3280 | [0.3036, 0.3497] | 0.0000 | 0.0000 | 0.9790 | 1.050x | `True` |
| `offline_oracle_upper_bound` | 0.2500 | 0.2924 | [0.2717, 0.3118] | 0.0356 | 0.1426 | 0.9869 | 2.175x | `False` |

*Notes*: `offline_oracle_upper_bound` uses ground-truth test loss reductions and is never deployable. Target call rate is a cap; realized rates can be lower. Expert cost is constant per call, so the cost multiplier changes the invocation threshold, not sample ranking. The proposed policy also applies a pre-expert bird-activity gate; do not interpret any difference as evidence of heterogeneous per-sample cost optimization.

## 2. Maximum Call-Rate Cap Sweep

| Maximum Expert Call-Rate Cap | Proposed Actual Call Rate | Fixed-Confidence Loss | Entropy Routing Loss | Budgeted Benefit Loss | Known-Species Macro-F1 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0.10 | 0.0000 | 0.3452 | 0.3350 | 0.3280 | 0.9790 |
| 0.20 | 0.0000 | 0.3481 | 0.3377 | 0.3280 | 0.9790 |
| 0.30 | 0.0000 | 0.3441 | 0.3438 | 0.3280 | 0.9790 |
| 0.40 | 0.0000 | 0.3434 | 0.3447 | 0.3280 | 0.9790 |
| 0.50 | 0.0000 | 0.3434 | 0.3447 | 0.3280 | 0.9790 |

## 3. Routing Feature & Component Ablation Study

| Ablation Variant | Expected Decision Loss | Known-Species Macro-F1 | Expert Call Rate |
| :--- | :---: | :---: | :---: |
| `Budgeted Benefit Router (Uncertainty + Prototype + Pre-Expert Bird Gate + SNR)` | 0.3280 | 0.9790 | 0.0000 |
| `Cost-Blind Benefit Router (No Cost Floor or Bird Gate)` | 0.3291 | 0.9769 | 0.0125 |
| `w/o Benefit Estimator (Entropy-Only Threshold Routing)` | 0.3415 | 0.9816 | 0.2445 |
| `w/o Expert Model (Student Only)` | 0.3280 | 0.9790 | 0.0000 |
