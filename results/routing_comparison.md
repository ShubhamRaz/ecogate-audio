# Cost-Aware Expert Routing Comparison & Ablation Report (`results/routing_comparison.md`)

## 1. Ten-Policy Routing Comparison on Held-Out Test Split (Matched Target Budget = 25%)

| Routing Policy | Expert Call Rate | Expected Decision Loss | Group Bootstrap 95% CI | Loss Reduction vs. Student | Improvement per Expert Call | Known-Species Macro-F1 | Avg. Compute Cost Units | Deployable |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `student_only` | 0.0000 | 0.2945 | [0.2757, 0.3124] | 0.0000 | 0.0000 | 0.7882 | 1.000x | `True` |
| `expert_only` | 1.0000 | 0.2219 | [0.2041, 0.2435] | 0.0726 | 0.0726 | 0.9710 | 5.500x | `True` |
| `random_routing` | 0.2679 | 0.2764 | [0.2586, 0.2936] | 0.0181 | 0.0677 | 0.8290 | 2.256x | `True` |
| `fixed_confidence_routing` | 0.2564 | 0.2796 | [0.2611, 0.3007] | 0.0150 | 0.0584 | 0.8882 | 2.204x | `True` |
| `entropy_routing` | 0.2764 | 0.2811 | [0.2608, 0.3008] | 0.0134 | 0.0487 | 0.8827 | 2.294x | `True` |
| `margin_routing` | 0.2602 | 0.2802 | [0.2604, 0.3014] | 0.0143 | 0.0551 | 0.8884 | 2.221x | `True` |
| `prototype_distance_routing` | 0.2602 | 0.3039 | [0.2856, 0.3223] | -0.0094 | -0.0362 | 0.7946 | 2.221x | `True` |
| `cost_blind_benefit_routing` | 0.2356 | 0.2732 | [0.2542, 0.2931] | 0.0213 | 0.0905 | 0.8382 | 2.110x | `True` |
| `proposed_cost_aware_routing` | 0.2356 | 0.2732 | [0.2542, 0.2931] | 0.0213 | 0.0905 | 0.8382 | 2.110x | `True` |
| `offline_oracle_upper_bound` | 0.2502 | 0.1832 | [0.1666, 0.1987] | 0.1113 | 0.4449 | 0.9749 | 2.176x | `False` |

*Note*: `offline_oracle_upper_bound` uses ground-truth test loss reductions and is included strictly as a theoretical lower-loss bound, never as a deployable router.

## 2. Matched Expert-Call Budget Sweep

| Target Expert Call Rate | Fixed-Confidence Loss | Entropy Routing Loss | Proposed Cost-Aware Loss | Proposed Known-Species Macro-F1 |
| :---: | :---: | :---: | :---: | :---: |
| 0.10 | 0.2867 | 0.2922 | 0.2857 | 0.8110 |
| 0.20 | 0.2836 | 0.2866 | 0.2781 | 0.8274 |
| 0.30 | 0.2781 | 0.2780 | 0.2700 | 0.8448 |
| 0.40 | 0.2744 | 0.2745 | 0.2585 | 0.8784 |
| 0.50 | 0.2698 | 0.2701 | 0.2420 | 0.9082 |

## 3. Routing Feature & Component Ablation Study

| Ablation Variant | Expected Decision Loss | Known-Species Macro-F1 | Expert Call Rate |
| :--- | :---: | :---: | :---: |
| `Full Proposed Cost-Aware Router (Uncertainty + Prototype + Bird Gate + SNR)` | 0.2732 | 0.8382 | 0.2356 |
| `w/o Cost Penalty & Bird Activity Gate (Cost-Blind Benefit Router)` | 0.2732 | 0.8382 | 0.2356 |
| `w/o Benefit Estimator (Entropy-Only Threshold Routing)` | 0.2811 | 0.8827 | 0.2764 |
| `w/o Expert Model (Student Only)` | 0.2945 | 0.7882 | 0.0000 |
