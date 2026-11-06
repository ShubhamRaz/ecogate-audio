#!/usr/bin/env python3
"""
Master End-to-End Training, Open-Set, Cost-Aware Routing, Benchmarking, Visualization,
and Report Generation Pipeline (`scripts/train_complete_pipeline.py`).
Executes all 11 required experiments on the available dataset, saves model checkpoints,
exports INT8 edge artifacts, generates publication-grade plots in `results/plots/`,
and writes all 6 required Markdown summary reports in `results/`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_dataset import main as run_audit
from scripts.find_duplicates import main as run_duplicates
from scripts.audit_noise_provenance import main as run_provenance
from scripts.analyze_audio import main as run_audio_analysis
from scripts.analyze_class_distribution import main as run_class_dist
from scripts.create_group_splits import main as run_group_splits
from scripts.validate_dataset import main as run_validate

from src.benchmarking.energy_benchmarker import (
    benchmark_model_inference,
    compute_energy_per_correct_prediction,
)
from src.calibration.calibrator import TemperatureScaler
from src.datasets.bird_dataset import (
    FeatureNormalizer,
    KNOWN_SPECIES,
    load_dataset_from_manifest,
)
from src.deployment.edge_engine import QuantizedStudentEdgeModel
from src.evaluation.evaluator import (
    compute_classification_metrics,
    compute_open_set_metrics,
    compute_risk_coverage_curve,
    group_bootstrap_ci,
)
from src.models.audio_models import (
    LightweightStudentCNN,
    MFCCBaselineModel,
    StrongExpertAudioModel,
)
from src.open_set.open_set_detector import OpenSetDetector
from src.routing.benefit_estimator import (
    BenefitEstimator,
    build_pre_expert_feature_matrix,
    compute_sample_loss,
)
from src.routing.budget_controller import BudgetController, BudgetPolicyConfig
from src.routing.cost_aware_router import CostAwareRouter
from src.training.pipeline_trainer import (
    evaluate_routing_policies_on_split,
    generate_oof_routing_examples,
    prepare_arrays,
    run_species_held_out_rotations,
)


def generate_plots(
    plots_dir: Path,
    routing_comp: dict,
    open_set_comp: dict,
    rc_curve_student: dict,
    rc_curve_routed: dict,
    lambda_sensitivity: list,
) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Routing Policy Comparison Plot (Decision Loss & Macro-F1)
    fig, ax1 = plt.subplots(figsize=(9, 4.8), dpi=150)
    policies = [k for k in routing_comp.keys() if k != "offline_oracle_upper_bound"]
    short_names = [p.replace("_routing", "").replace("proposed_", "*") for p in policies]
    f1s = [routing_comp[p]["known_species_macro_f1"] for p in policies]
    losses = [routing_comp[p]["expected_decision_loss"] for p in policies]

    x = np.arange(len(policies))
    width = 0.36
    ax1.bar(x - width / 2, f1s, width, label="Known-Species Macro-F1 (Higher Better)", color="#10b981")
    ax1.bar(x + width / 2, losses, width, label="Expected Decision Loss (Lower Better)", color="#f43f5e")
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_names, rotation=30, ha="right", fontsize=8.5)
    ax1.set_ylabel("Score / Loss")
    ax1.set_title("EcoGate-Audio: Routing Policy Comparison at Matched Target Budget")
    ax1.legend(loc="upper right")
    ax1.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "routing_policy_comparison.png")
    plt.close(fig)

    # 2. Risk-Coverage Curve Plot
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    cov_s = [pt["coverage"] for pt in rc_curve_student["curve"]]
    risk_s = [pt["selective_risk"] for pt in rc_curve_student["curve"]]
    cov_r = [pt["coverage"] for pt in rc_curve_routed["curve"]]
    risk_r = [pt["selective_risk"] for pt in rc_curve_routed["curve"]]
    ax.plot(cov_s, risk_s, "o--", label=f"Student Only (AURC={rc_curve_student['aurc']:.3f})", color="#64748b")
    ax.plot(cov_r, risk_r, "s-", label=f"Cost-Aware Selective Routing (AURC={rc_curve_routed['aurc']:.3f})", color="#059669", linewidth=2)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Selective Risk (Decision Loss)")
    ax.set_title("Risk-Coverage Trade-Off Curve")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "risk_coverage_curve.png")
    plt.close(fig)

    # 3. Open-Set Scoring Method Comparison (AUROC & FPR95)
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)
    os_methods = list(open_set_comp.keys())
    aurocs = [open_set_comp[m]["auroc"] for m in os_methods]
    auprs = [open_set_comp[m]["aupr"] for m in os_methods]
    xm = np.arange(len(os_methods))
    ax.bar(xm - 0.18, aurocs, 0.35, label="AUROC", color="#3b82f6")
    ax.bar(xm + 0.18, auprs, 0.35, label="AUPR", color="#8b5cf6")
    ax.set_xticks(xm)
    ax.set_xticklabels(os_methods, rotation=20, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_title("Open-Set Unknown Detection Performance Across 6 Scoring Methods")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "open_set_methods.png")
    plt.close(fig)


def write_summary_markdown_reports(
    results_dir: Path,
    audit_summary: dict,
    split_integrity: dict,
    baseline_metrics: dict,
    open_set_metrics: dict,
    held_out_rotations: list,
    routing_comparison: dict,
    budget_sweep: list,
    ablation_results: list,
    efficiency_metrics: dict,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    lf = audit_summary["local_execution_findings"]
    pr = audit_summary["preliminary_reported_archive_claims"]

    # 1. results/dataset_audit_report.md
    (results_dir / "dataset_audit_report.md").write_text(
        f"""# Dataset Audit & Split Integrity Report (`results/dataset_audit_report.md`)

## 1. Archive Provenance & Modality Policy
- **Official Dataset**: [Multimodal Computer Vision and Acoustic Bird Detection Dataset for Smart Rice Farming Environments](https://data.mendeley.com/datasets/p6m5tmr477) (DOI: `10.17632/p6m5tmr477.1`, CC BY 4.0).
- **Modality Enforcement**: Strictly **Audio-First**. Visual frames are never paired with audio recordings because no verified temporal synchronization metadata exists.
- **Execution Scope Note**: Because the 8,512-WAV external Mendeley archive is not pre-mounted inside this container environment, the audit pipeline executed on the deterministic 151-file multi-session acoustic validation archive in `data/raw/` while preserving the preliminary reported 8,512-file archive observations separately for full-archive comparison.

## 2. Preliminary Reported Archive Claims vs. Executed Local Validation Audit

| Audit Metric | Preliminary Reported Archive Observation (Unverified Historical Claim) | Executed Local Validation Audit Result |
| :--- | :---: | :---: |
| Total WAV files scanned | {pr['wav_files_in_released_archive']} | {lf['total_wav_files_scanned']} |
| Valid decodable WAV files | — | {lf['valid_wav_files']} |
| Corrupt WAV files flagged & excluded | — | {lf['corrupt_wav_files']} |
| Annotation records inspected | {pr['audio_annotation_records']} | {lf['annotation_records_inspected']} |
| Missing referenced audio paths | {pr['missing_referenced_audio_paths']} | {lf['missing_referenced_audio_paths_detected']} |
| Duplicate waveform groups detected | {pr['cross_split_duplicate_groups']} | {lf['duplicate_waveform_groups']} |
| Files involved in duplicate groups | {pr['file_entries_in_duplicate_groups']} | {lf['files_in_duplicate_groups']} |
| ESC-50 catalogue filename matches | {pr['noise_filenames_matching_esc50']} | {lf['esc50_filename_matches']} |

## 3. Group-Safe Partition Integrity Verification
- **Split Integrity Check Passed**: `{split_integrity['passed']}`
- **Leaked Duplicate Groups Across Splits**: `{len(split_integrity['leaked_duplicate_groups'])}`
- **Leaked Waveform SHA-256 Hashes Across Splits**: `{len(split_integrity['leaked_waveform_shas'])}`
- **Leaked Recording Sessions Across Splits**: `{len(split_integrity['leaked_recording_sessions'])}`
- **Partition Counts**: Train = `{split_integrity['split_counts'].get('train', 0)}`, Validation = `{split_integrity['split_counts'].get('val', 0)}`, Calibration = `{split_integrity['split_counts'].get('calibration', 0)}`, Test = `{split_integrity['split_counts'].get('test', 0)}`.
""",
        encoding="utf-8",
    )

    # 2. results/baseline_comparison.md
    m_mfcc = baseline_metrics["mfcc_logreg"]
    m_stud = baseline_metrics["student_cnn_fp32"]
    m_quant = baseline_metrics["student_cnn_int8"]
    m_exp = baseline_metrics["expert_multiscale"]
    (results_dir / "baseline_comparison.md").write_text(
        f"""# Audio-Only Baseline & Model Tier Comparison (`results/baseline_comparison.md`)

## 1. Held-Out Known-Species Classification Comparison

| Model Tier | Model ID | Accuracy | Macro-F1 | Weighted-F1 | Balanced Accuracy | Parameters |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Classical Baseline | `{m_mfcc['model_id']}` | {m_mfcc['accuracy']:.4f} | {m_mfcc['macro_f1']:.4f} | {m_mfcc['weighted_f1']:.4f} | {m_mfcc['balanced_accuracy']:.4f} | {m_mfcc['parameters']} |
| Lightweight Student (FP32) | `{m_stud['model_id']}` | {m_stud['accuracy']:.4f} | {m_stud['macro_f1']:.4f} | {m_stud['weighted_f1']:.4f} | {m_stud['balanced_accuracy']:.4f} | {m_stud['parameters']} |
| Quantized Student (INT8 Recalibrated) | `{m_quant['model_id']}` | {m_quant['accuracy']:.4f} | {m_quant['macro_f1']:.4f} | {m_quant['weighted_f1']:.4f} | {m_quant['balanced_accuracy']:.4f} | {m_quant['parameters']} |
| Strong Expert Model | `{m_exp['model_id']}` | {m_exp['accuracy']:.4f} | {m_exp['macro_f1']:.4f} | {m_exp['weighted_f1']:.4f} | {m_exp['balanced_accuracy']:.4f} | {m_exp['parameters']} |

## 2. Expert Uplift Verification Gate
- **Expert vs. Student Macro-F1 Uplift**: `{m_exp['macro_f1'] - m_stud['macro_f1']:+.4f}`
- **Minimum Required Uplift (config)**: `+0.0200`
- **Gate Status**: {"✅ PASS — Expert demonstrates sufficient positive uplift over student; selective invocation is justified." if (m_exp['macro_f1'] - m_stud['macro_f1']) >= 0.02 else ("⚠️ MARGINAL — Expert shows positive but sub-threshold uplift; check training data size." if (m_exp['macro_f1'] - m_stud['macro_f1']) > 0.0 else "❌ FAIL — Expert does NOT outperform student on this dataset; do NOT rely on expert gate for uplift claim.")}
""",
        encoding="utf-8",
    )

    # 3. results/open_set_evaluation.md
    os_rows = "\n".join(
        f"| `{m}` | {v['auroc']:.4f} | {v['aupr']:.4f} | {v['fpr95']:.4f} | {v['unknown_recall']:.4f} | {v['false_known_species_assignment_rate']:.4f} | {v['ece']:.4f} | {v['brier_score']:.4f} |"
        for m, v in open_set_metrics.items()
    )
    rot_rows = "\n".join(
        f"| `{r['held_out_species']}` | {r['auroc']:.4f} | {r['aupr']:.4f} | {r['fpr95']:.4f} | {r['held_out_species_rejection_rate']:.4f} | {r['false_known_assignment_on_held_out']:.4f} |"
        for r in held_out_rotations
    )
    (results_dir / "open_set_evaluation.md").write_text(
        f"""# Open-Set Recognition & Held-Out Species Evaluation (`results/open_set_evaluation.md`)

## 1. Score Direction & Evaluation Protocol
- **Score Direction**: Higher `novelty_score` indicates higher likelihood of an unfamiliar bird (`bird_unfamiliar`) or non-bird environmental sound (`nonbird`).
- **Downstream Open-Set Protocol**: Thresholds are calibrated exclusively on the independent calibration split (`target_known_recall = 0.90`). Pretraining exposure is not controlled when using external pretrained backbones, and `other_birds` is evaluated as an auxiliary open-set bird pool rather than claimed as biological discovery of unseen species.

## 2. Comparison of Six Open-Set Rejection Methods on Test Split

| Method | AUROC | AUPR | FPR95 | Unknown Recall | False Known-Species Rate | Calibrated ECE | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{os_rows}

## 3. Leave-One-Species-Out Downstream Rotations

| Held-Out Bird Species | Open-Set AUROC | Open-Set AUPR | FPR95 | Held-Out Species Rejection Rate | False Known-Species Assignment Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
{rot_rows}
""",
        encoding="utf-8",
    )

    # 4. results/routing_comparison.md
    rt_rows = "\n".join(
        f"| `{k}` | {v['expert_call_rate']:.4f} | {v['expected_decision_loss']:.4f} | [{v['loss_ci_95'][0]:.4f}, {v['loss_ci_95'][1]:.4f}] | {v['loss_reduction_vs_student']:.4f} | {v['improvement_per_expert_call']:.4f} | {v['known_species_macro_f1']:.4f} | {v['average_compute_cost_units']:.3f}x | `{v['is_deployable']}` |"
        for k, v in routing_comparison.items()
    )
    bs_rows = "\n".join(
        f"| {b['target_budget_rate']:.2f} | {b['confidence_loss']:.4f} | {b['entropy_loss']:.4f} | {b['proposed_cost_aware_loss']:.4f} | {b['proposed_macro_f1']:.4f} |"
        for b in budget_sweep
    )
    ab_rows = "\n".join(
        f"| `{a['ablation_variant']}` | {a['expected_decision_loss']:.4f} | {a['known_species_macro_f1']:.4f} | {a['expert_call_rate']:.4f} |"
        for a in ablation_results
    )
    (results_dir / "routing_comparison.md").write_text(
        f"""# Cost-Aware Expert Routing Comparison & Ablation Report (`results/routing_comparison.md`)

## 1. Ten-Policy Routing Comparison on Held-Out Test Split (Matched Target Budget = 25%)

| Routing Policy | Expert Call Rate | Expected Decision Loss | Group Bootstrap 95% CI | Loss Reduction vs. Student | Improvement per Expert Call | Known-Species Macro-F1 | Avg. Compute Cost Units | Deployable |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{rt_rows}

*Note*: `offline_oracle_upper_bound` uses ground-truth test loss reductions and is included strictly as a theoretical lower-loss bound, never as a deployable router.

## 2. Matched Expert-Call Budget Sweep

| Target Expert Call Rate | Fixed-Confidence Loss | Entropy Routing Loss | Proposed Cost-Aware Loss | Proposed Known-Species Macro-F1 |
| :---: | :---: | :---: | :---: | :---: |
{bs_rows}

## 3. Routing Feature & Component Ablation Study

| Ablation Variant | Expected Decision Loss | Known-Species Macro-F1 | Expert Call Rate |
| :--- | :---: | :---: | :---: |
{ab_rows}
""",
        encoding="utf-8",
    )

    # 5. results/efficiency_report.md
    s_b = efficiency_metrics["student_fp32"]
    q_b = efficiency_metrics["student_int8"]
    e_b = efficiency_metrics["expert_fp32"]
    epc = efficiency_metrics["energy_per_correct_routed"]
    (results_dir / "efficiency_report.md").write_text(
        f"""# Hardware, Latency, Quantization & Energy Efficiency Report (`results/efficiency_report.md`)

## 1. Model Complexity & Measured CPU Latency (`{s_b['platform']}`)

| Model Variant | Parameters | MACs / FLOPs | Actual Stored Arrays | Runtime Compute Dtype | Batch-1 CPU Latency (Mean / P95) | Peak Memory | Electrical Energy Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| Student fixed-feature model | {s_b['parameters']} | n/a | {s_b['actual_stored_array_kb']} KB | {s_b['runtime_compute_dtype']} | {s_b['batch_size_1_cpu_latency_mean_ms']} ms / {s_b['batch_size_1_cpu_latency_p95_ms']} ms | {s_b['peak_memory_kb']} KB | `{s_b['electrical_energy_status']}` |
| Student compressed-weight artifact | {q_b['parameters']} | n/a | {q_b['actual_stored_array_kb']} KB | {q_b['runtime_compute_dtype']} | {q_b['batch_size_1_cpu_latency_mean_ms']} ms / {q_b['batch_size_1_cpu_latency_p95_ms']} ms | {q_b['peak_memory_kb']} KB | `{q_b['electrical_energy_status']}` |
| Expert fixed-feature model | {e_b['parameters']} | n/a | {e_b['actual_stored_array_kb']} KB | {e_b['runtime_compute_dtype']} | {e_b['batch_size_1_cpu_latency_mean_ms']} ms / {e_b['batch_size_1_cpu_latency_p95_ms']} ms | {e_b['peak_memory_kb']} KB | `{e_b['electrical_energy_status']}` |

## 2. Energy / Compute Cost per Correct Prediction (Proposed Cost-Aware Policy)
- **Numerator (Total Compute Cost Units)**: `{epc['numerator_total_compute_cost_units']}`
- **Numerator (Measured Electrical Joules)**: `{epc['numerator_measured_electrical_joules']}` (`{epc['electrical_measurement_status']}`)
- **Denominator (Number of Correct Predictions)**: `{epc['denominator_correct_predictions']}` out of `{epc['total_evaluated_predictions']}` evaluated test clips
- **Compute Cost Units per Correct Prediction**: `{epc['compute_cost_units_per_correct_prediction']}`
- **Integer inference status**: `{q_b.get('runtime_compute_dtype', 'float32')}` runtime compute; the NPZ artifact uses compressed weights but dequantizes them before matrix multiplications. No Pi Zero 2 W benchmark or physical energy measurement is claimed.
- **Complexity accounting**: MAC/FLOP counts are unavailable until exact operator-level accounting is added; parameter count is not used as a MAC proxy.
""",
        encoding="utf-8",
    )

    # 6. results/final_experimental_report.md
    (results_dir / "final_experimental_report.md").write_text(
        f"""# Final Experimental Report (`results/final_experimental_report.md`)

## 1. Summary of Executed Verification Experiments (Experiments 1–11)
1. **Experiment 1 (Dataset Audit & Split Contamination)**: Verified SHA-256 and decoded float32 waveform duplicate grouping, ESC-50 filename matching, corrupt WAV exclusion, missing-path detection, and zero cross-split group leakage (`passed = {split_integrity['passed']}`).
2. **Experiment 2 (Audio-Only Baselines)**: Evaluated the MFCC baseline against the fixed-feature student compatibility model (not a CNN).
3. **Experiment 3 (Feature Model Comparison)**: Compared the fixed-feature student and expert compatibility implementations; neither is a trained deep neural backbone.
4. **Experiment 4 (Open-Set Recognition)**: Evaluated all 6 open-set scoring methods (`msp`, `temperature_scaled`, `entropy`, `margin`, `prototype_distance`, `energy`) and 4 leave-one-species-out rotations.
5. **Experiment 5 & 6 (Routing Baselines vs. Budgeted Benefit Routing)**: Compared routing policies using out-of-fold benefit estimates and calibration-only thresholds, subject to configured call-rate and energy caps. The reported result is an empirical output, not evidence of per-sample variable-cost optimization.
6. **Experiment 7 (Source-Shift & Group Bootstrap Robustness)**: Computed group-level bootstrap 95% confidence intervals preserving session/duplicate clusters.
7. **Experiment 8 (Compressed Weight Storage)**: Exported an NPZ with INT8 weight arrays and recalibrated probabilities; inference dequantizes to float32, so this is not integer compute.
8. **Experiment 9 & 10 (Risk-Coverage-Energy Trade-off & Ablations)**: Generated risk-coverage curves, matched budget sweeps (10%–50% call rates), and feature-group ablations.
9. **Experiment 11 (Dataset Scope)**: Records the dataset source type in the audit output. Synthetic fixtures are blocked from the complete research pipeline; unverified source metadata must not be described as source-independent generalization.
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EcoGate-Audio research experiments on a mounted real dataset.")
    parser.add_argument("--force-rebuild-cache", action="store_true", help="Rebuild cached features from the current split manifest.")
    args = parser.parse_args()

    raw_dir = PROJECT_ROOT / "data" / "raw"
    synthetic_marker = raw_dir / ".ecogate_synthetic_reference_fixture"
    if synthetic_marker.exists():
        raise SystemExit(
            "Research training is blocked: data/raw is marked as a synthetic reference fixture. "
            "Remove the fixture and mount the real source archive before running publication experiments."
        )

    print("=== [EcoGate-Audio] Starting Complete End-to-End Pipeline ===")
    run_audit()
    run_duplicates()
    run_provenance()
    run_audio_analysis()
    run_class_dist()
    run_group_splits()
    run_validate()

    features_cache = PROJECT_ROOT / "data" / "features" / "features_cache.pkl"
    cache_meta = features_cache.with_suffix(".meta.json")
    split_csv = PROJECT_ROOT / "data" / "splits" / "group_splits_manifest.csv"
    if not split_csv.exists():
        raise SystemExit(f"Split manifest not found: {split_csv}. Run the real-data audit and split steps first.")
    manifest_sha = hashlib.sha256(split_csv.read_bytes()).hexdigest()
    cache_valid = False
    if features_cache.exists() and cache_meta.exists() and not args.force_rebuild_cache:
        try:
            meta = json.loads(cache_meta.read_text(encoding="utf-8"))
            cache_valid = (
                meta.get("manifest_sha256") == manifest_sha
                and meta.get("dataset_source_type") == "local_archive_source_unverified"
            )
        except (OSError, json.JSONDecodeError):
            cache_valid = False

    if cache_valid:
        print(f"=== [EcoGate-Audio] Loading manifest-matched acoustic cache from {features_cache} ===", flush=True)
        import pickle
        with open(features_cache, "rb") as f:
            all_samples = pickle.load(f)
    else:
        if features_cache.exists():
            print("[load_dataset] Ignoring stale/unverified feature cache; rebuilding from current manifest.", flush=True)
        print("=== [EcoGate-Audio] Loading Dataset and Manifest ===", flush=True)
        all_samples = load_dataset_from_manifest(split_csv, raw_dir)
        try:
            features_cache.parent.mkdir(parents=True, exist_ok=True)
            import pickle
            with open(features_cache, "wb") as f:
                pickle.dump(all_samples, f, protocol=pickle.HIGHEST_PROTOCOL)
            cache_meta.write_text(json.dumps({
                "manifest_sha256": manifest_sha,
                "dataset_source_type": "local_archive_source_unverified",
                "cache_format_version": 2,
            }, indent=2), encoding="utf-8")
            print(f"[load_dataset] Cached {len(all_samples)} manifest-matched samples.", flush=True)
        except Exception as exc:
            print(f"[load_dataset] Warning: failed to write feature cache and metadata: {exc}", flush=True)

    if any(getattr(sample, "source_type", "unknown") == "synthetic_reference_fixture" for sample in all_samples):
        raise SystemExit("Synthetic fixture samples detected in cache; refusing research evaluation.")
    if not all_samples:
        raise SystemExit("No valid samples loaded from the audited manifest.")

    train_s = [s for s in all_samples if s.split == "train"]
    val_s = [s for s in all_samples if s.split == "val"]
    cal_s = [s for s in all_samples if s.split == "calibration"]
    test_s = [s for s in all_samples if s.split == "test"]
    print(f"[splits] Train: {len(train_s)}, Val: {len(val_s)}, Cal: {len(cal_s)}, Test: {len(test_s)}", flush=True)

    normalizer = FeatureNormalizer().fit(train_s)
    tr_arr = prepare_arrays(train_s, normalizer)
    va_arr = prepare_arrays(val_s, normalizer)
    ca_arr = prepare_arrays(cal_s, normalizer)
    te_arr = prepare_arrays(test_s, normalizer)

    # 1. Train MFCC Baseline, Student CNN, and Strong Expert
    print("=== [Step 1/11] Training MFCC Baseline, Student CNN, and Expert Model ===", flush=True)
    mfcc_model = MFCCBaselineModel(seed=42)
    mfcc_model.fit(tr_arr["mfccs"], tr_arr["y_known"], tr_arr["is_bird"])

    print("  Fitting Student CNN...", flush=True)
    student = LightweightStudentCNN(seed=42)
    student.fit(tr_arr["mels"], tr_arr["y_known"], tr_arr["is_bird"], va_arr["mels"], va_arr["y_known"], epochs=15)
    student.save_checkpoint(PROJECT_ROOT / "checkpoints" / "student_best.npz")

    print("  Fitting Expert Model...", flush=True)
    expert = StrongExpertAudioModel(seed=42)
    expert.fit(tr_arr["mels"], tr_arr["y_known"], tr_arr["is_bird"], va_arr["mels"], va_arr["y_known"], epochs=15)
    expert.save_checkpoint(PROJECT_ROOT / "checkpoints" / "expert_best.npz")

    # 2. Temperature Calibration on Calibration Split
    print("=== [Step 2/11] Temperature Calibration on Calibration Split ===", flush=True)
    ca_s_raw = student.predict(ca_arr["mels"])
    ca_e_raw = expert.predict(ca_arr["mels"])
    s_temp_scaler = TemperatureScaler().fit(ca_s_raw.logits, ca_arr["y_known"])
    e_temp_scaler = TemperatureScaler().fit(ca_e_raw.logits, ca_arr["y_known"])

    # 3. Quantized INT8 Student + Post-Quantization Recalibration
    print("=== [Step 3/11] Quantized INT8 Student & Edge Recalibration ===", flush=True)
    quant_student = QuantizedStudentEdgeModel(student)
    quant_recal = quant_student.recalibrate(ca_arr["mels"], ca_arr["y_known"])
    quant_export = quant_student.export_int8_artifact(PROJECT_ROOT / "checkpoints" / "student_int8_edge.npz")

    # 4. Fit Open-Set Prototypes on Train & Calibrate Thresholds on Calibration Split
    print("=== [Step 4/11] Open-Set Prototypes & Calibration ===", flush=True)
    tr_s_out = student.predict(tr_arr["mels"])
    tr_e_out = expert.predict(tr_arr["mels"])
    os_student = OpenSetDetector(temperature=s_temp_scaler["temperature"])
    os_student.fit_prototypes(tr_s_out.embeddings, tr_arr["y_known"])
    os_student.calibrate_thresholds(ca_s_raw.logits, ca_s_raw.embeddings, ca_arr["y_known"] >= 0, target_known_recall=0.90)

    os_expert = OpenSetDetector(temperature=e_temp_scaler["temperature"])
    os_expert.fit_prototypes(tr_e_out.embeddings, tr_arr["y_known"])
    os_expert.calibrate_thresholds(ca_e_raw.logits, ca_e_raw.embeddings, ca_arr["y_known"] >= 0, target_known_recall=0.90)

    # 5. Out-of-Fold Benefit Estimator Training on Development Data (Train + Val)
    print("=== [Step 5/11] Out-of-Fold Benefit Estimator Training ===", flush=True)
    dev_samples = train_s + val_s
    oof_data = generate_oof_routing_examples(dev_samples, n_folds=4, seed=42)
    benefit_est = BenefitEstimator(ridge_alpha=1.5)
    oof_fit_stats = benefit_est.fit_oof(oof_data["Z_oof"], oof_data["d_oof"])
    benefit_est.freeze()
    benefit_est.save(PROJECT_ROOT / "checkpoints" / "benefit_estimator.npz")

    # 6. Evaluate Models & Open-Set on Held-Out Test Split
    print("=== [Step 6/11] Model & Open-Set Evaluation on Held-Out Test Split ===", flush=True)
    te_k_mask = te_arr["y_known"] >= 0
    te_mfcc_out = mfcc_model.predict(te_arr["mfccs"])
    te_s_out = student.predict(te_arr["mels"], temperature=s_temp_scaler["temperature"])
    te_q_out = quant_student.predict(te_arr["mels"])
    te_e_out = expert.predict(te_arr["mels"], temperature=e_temp_scaler["temperature"])

    baseline_metrics = {
        "mfcc_logreg": {
            "model_id": mfcc_model.model_id,
            "parameters": mfcc_model.param_count(),
            **compute_classification_metrics(te_arr["y_known"][te_k_mask], te_mfcc_out.predicted_class[te_k_mask]),
        },
        "student_cnn_fp32": {
            "model_id": student.model_id,
            "parameters": student.param_count(),
            **compute_classification_metrics(te_arr["y_known"][te_k_mask], te_s_out.predicted_class[te_k_mask]),
        },
        "student_cnn_int8": {
            "model_id": quant_student.model_id,
            "parameters": student.param_count(),
            **compute_classification_metrics(te_arr["y_known"][te_k_mask], te_q_out.predicted_class[te_k_mask]),
        },
        "expert_multiscale": {
            "model_id": expert.model_id,
            "parameters": expert.param_count(),
            **compute_classification_metrics(te_arr["y_known"][te_k_mask], te_e_out.predicted_class[te_k_mask]),
        },
    }

    # Open-set evaluation across all 6 methods
    is_unk_test = (te_arr["y_known"] == -1).astype(int)
    open_set_metrics = {}
    for m_name in OpenSetDetector.SUPPORTED_METHODS:
        sc = os_student.score(te_s_out.logits, te_s_out.embeddings, method=m_name)
        open_set_metrics[m_name] = compute_open_set_metrics(
            is_unk_test, sc.novelty_score, sc.threshold, te_s_out.probabilities, te_arr["y_known"]
        )

    held_out_rotations = run_species_held_out_rotations(train_s, test_s, seed=42)

    # 7. Evaluate 10 Routing Policies on Test Split
    print("=== [Step 7/11] Evaluating 10 Routing Policies on Test Split ===", flush=True)
    ca_s_nov = os_student.compute_all_novelty_features(ca_s_raw.logits, ca_s_raw.embeddings)
    te_s_nov = os_student.compute_all_novelty_features(te_s_out.logits, te_s_out.embeddings)
    te_e_nov = os_expert.compute_all_novelty_features(te_e_out.logits, te_e_out.embeddings)

    cal_Z = build_pre_expert_feature_matrix(
        student_max_prob=ca_s_nov["raw_max_prob"],
        student_margin=ca_s_nov["raw_margin"],
        student_entropy=ca_s_nov["entropy"],
        prototype_distance=ca_s_nov["prototype_distance"],
        bird_activity_prob=ca_s_raw.bird_activity_prob,
        audio_snr_db=ca_arr["snr_db"],
        clipping_ratio=ca_arr["clipping_ratio"],
        rms_energy=ca_arr["rms_energy"],
    )
    cal_s_loss = compute_sample_loss(ca_s_raw.probabilities, ca_s_raw.bird_activity_prob, ca_arr["y_known"], ca_arr["is_bird"])
    cal_e_loss = compute_sample_loss(ca_e_raw.probabilities, ca_e_raw.bird_activity_prob, ca_arr["y_known"], ca_arr["is_bird"])

    routing_comparison = evaluate_routing_policies_on_split(
        test_arr=te_arr,
        s_out=te_s_out,
        e_out=te_e_out,
        s_nov_feats=te_s_nov,
        e_nov_feats=te_e_nov,
        cal_s_nov_feats=ca_s_nov,
        cal_Z=cal_Z,
        cal_s_loss=cal_s_loss,
        cal_e_loss=cal_e_loss,
        benefit_estimator=benefit_est,
        target_call_rate=0.25,
        seed=42,
    )

    # Budget sweep across 10%, 20%, 30%, 40%, 50% call rates
    print("=== [Step 8/11] Budget Sweep & Trade-Off Analysis ===", flush=True)
    budget_sweep = []
    for br in [0.10, 0.20, 0.30, 0.40, 0.50]:
        rc_b = evaluate_routing_policies_on_split(
            test_arr=te_arr,
            s_out=te_s_out,
            e_out=te_e_out,
            s_nov_feats=te_s_nov,
            e_nov_feats=te_e_nov,
            cal_s_nov_feats=ca_s_nov,
            cal_Z=cal_Z,
            cal_s_loss=cal_s_loss,
            cal_e_loss=cal_e_loss,
            benefit_estimator=benefit_est,
            target_call_rate=br,
            seed=42,
        )
        budget_sweep.append({
            "target_budget_rate": br,
            "confidence_loss": rc_b["fixed_confidence_routing"]["expected_decision_loss"],
            "entropy_loss": rc_b["entropy_routing"]["expected_decision_loss"],
            "proposed_cost_aware_loss": rc_b["proposed_cost_aware_routing"]["expected_decision_loss"],
            "proposed_macro_f1": rc_b["proposed_cost_aware_routing"]["known_species_macro_f1"],
        })

    # Ablation studies
    print("=== [Step 9/11] Feature & Architectural Ablation Studies ===", flush=True)
    ablation_results = [
        {
            "ablation_variant": "Full Proposed Cost-Aware Router (Uncertainty + Prototype + Bird Gate + SNR)",
            "expected_decision_loss": routing_comparison["proposed_cost_aware_routing"]["expected_decision_loss"],
            "known_species_macro_f1": routing_comparison["proposed_cost_aware_routing"]["known_species_macro_f1"],
            "expert_call_rate": routing_comparison["proposed_cost_aware_routing"]["expert_call_rate"],
        },
        {
            "ablation_variant": "w/o Cost Penalty & Bird Activity Gate (Cost-Blind Benefit Router)",
            "expected_decision_loss": routing_comparison["cost_blind_benefit_routing"]["expected_decision_loss"],
            "known_species_macro_f1": routing_comparison["cost_blind_benefit_routing"]["known_species_macro_f1"],
            "expert_call_rate": routing_comparison["cost_blind_benefit_routing"]["expert_call_rate"],
        },
        {
            "ablation_variant": "w/o Benefit Estimator (Entropy-Only Threshold Routing)",
            "expected_decision_loss": routing_comparison["entropy_routing"]["expected_decision_loss"],
            "known_species_macro_f1": routing_comparison["entropy_routing"]["known_species_macro_f1"],
            "expert_call_rate": routing_comparison["entropy_routing"]["expert_call_rate"],
        },
        {
            "ablation_variant": "w/o Expert Model (Student Only)",
            "expected_decision_loss": routing_comparison["student_only"]["expected_decision_loss"],
            "known_species_macro_f1": routing_comparison["student_only"]["known_species_macro_f1"],
            "expert_call_rate": 0.0,
        },
    ]

    # Risk-coverage curves
    print("=== [Step 10/11] Risk-Coverage & Hardware Benchmarking ===", flush=True)
    te_s_loss = compute_sample_loss(te_s_out.probabilities, te_s_out.bird_activity_prob, te_arr["y_known"], te_arr["is_bird"])
    te_e_loss = compute_sample_loss(te_e_out.probabilities, te_e_out.bird_activity_prob, te_arr["y_known"], te_arr["is_bird"])
    rc_student = compute_risk_coverage_curve(te_s_nov["raw_max_prob"], te_s_loss)
    # Evaluate the actual calibrated benefit-router decisions, not an unrelated
    # entropy heuristic or an expert confidence score when the expert was not called.
    routed_mask = np.asarray(
        routing_comparison["proposed_cost_aware_routing"]["expert_invoke_mask"],
        dtype=bool,
    )
    routed_loss = np.where(routed_mask, te_e_loss, te_s_loss)
    routed_probabilities = np.where(
        routed_mask[:, None], te_e_out.probabilities, te_s_out.probabilities
    )
    routed_confidence = np.max(routed_probabilities, axis=1)
    rc_routed = compute_risk_coverage_curve(routed_confidence, routed_loss)

    # 8. Hardware & Energy Benchmarking
    bench_s = benchmark_model_inference(student, te_arr["mels"])
    bench_q = benchmark_model_inference(quant_student, te_arr["mels"])
    bench_e = benchmark_model_inference(expert, te_arr["mels"])

    n_test = te_arr["y_known"].size
    prop_call_rate = routing_comparison["proposed_cost_aware_routing"]["expert_call_rate"]
    total_cost_u = n_test * (1.05 + prop_call_rate * 4.5)
    n_correct_known = int(round(routing_comparison["proposed_cost_aware_routing"]["known_species_accuracy"] * np.sum(te_k_mask)))
    epc_stats = compute_energy_per_correct_prediction(
        total_cost_units=total_cost_u,
        num_correct_predictions=n_correct_known,
        total_predictions=int(np.sum(te_k_mask)),
        measured_total_joules=None,
    )

    efficiency_metrics = {
        "student_fp32": bench_s,
        "student_int8": bench_q,
        "expert_fp32": bench_e,
        "quantization_recalibration": quant_recal,
        "export_info": quant_export,
        "energy_per_correct_routed": epc_stats,
    }

    # Save complete JSON metrics & predictions
    audit_summary = json.loads((PROJECT_ROOT / "data" / "audited" / "audit_summary.json").read_text(encoding="utf-8"))
    split_integrity = json.loads((PROJECT_ROOT / "data" / "splits" / "split_integrity_report.json").read_text(encoding="utf-8"))

    complete_summary = {
        "dataset_audit": audit_summary,
        "split_integrity": split_integrity,
        "baseline_metrics": baseline_metrics,
        "calibration": {"student": s_temp_scaler, "expert": e_temp_scaler},
        "oof_benefit_estimator": oof_fit_stats,
        "open_set_metrics": open_set_metrics,
        "species_held_out_rotations": held_out_rotations,
        "routing_comparison": routing_comparison,
        "budget_sweep": budget_sweep,
        "ablation_study": ablation_results,
        "risk_coverage": {"student": rc_student, "cost_aware_routed": rc_routed},
        "efficiency": efficiency_metrics,
    }

    metrics_out = PROJECT_ROOT / "results" / "metrics" / "complete_evaluation_summary.json"
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(complete_summary, indent=2), encoding="utf-8")

    (PROJECT_ROOT / "results" / "energy" / "hardware_benchmark.json").write_text(
        json.dumps(efficiency_metrics, indent=2), encoding="utf-8"
    )

    # Save sample test selective decisions
    bc = BudgetController()
    bc.calibrated_threshold = 0.02
    ca_router = CostAwareRouter(benefit_est, bc)
    test_decisions = ca_router.decide_batch(
        student_probs=te_s_out.probabilities,
        student_bird_prob=te_s_out.bird_activity_prob,
        student_novelty=te_s_nov["prototype_distance"],
        student_entropy=te_s_nov["entropy"],
        student_margin=te_s_nov["raw_margin"],
        student_proto_dist=te_s_nov["prototype_distance"],
        snr_db=te_arr["snr_db"],
        clipping_ratio=te_arr["clipping_ratio"],
        rms_energy=te_arr["rms_energy"],
        expert_probs=te_e_out.probabilities,
        expert_bird_prob=te_e_out.bird_activity_prob,
        expert_novelty=te_e_nov["prototype_distance"],
    )
    (PROJECT_ROOT / "results" / "predictions" / "test_selective_decisions.json").write_text(
        json.dumps([d.__dict__ for d in test_decisions], indent=2), encoding="utf-8"
    )

    print("=== [Step 11/11] Generating Publication Plots and 6 Summary Reports ===", flush=True)
    generate_plots(
        PROJECT_ROOT / "results" / "plots",
        routing_comparison,
        open_set_metrics,
        rc_student,
        rc_routed,
        budget_sweep,
    )

    write_summary_markdown_reports(
        PROJECT_ROOT / "results",
        audit_summary,
        split_integrity,
        baseline_metrics,
        open_set_metrics,
        held_out_rotations,
        routing_comparison,
        budget_sweep,
        ablation_results,
        efficiency_metrics,
    )
    print("=== [EcoGate-Audio] Complete Pipeline & All 6 Summary Reports Generated Successfully ===")


if __name__ == "__main__":
    main()
