"""
Complete Out-of-Fold (OOF) and End-to-End Training Pipeline for EcoGate-Audio.
Strictly prevents leakage between base-model training, OOF router training,
calibration threshold selection, and held-out test evaluation.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.calibration.calibrator import TemperatureScaler
from src.datasets.bird_dataset import FeatureNormalizer, FeatureSample, KNOWN_SPECIES
from src.evaluation.evaluator import (
    compute_classification_metrics,
    compute_open_set_metrics,
    compute_risk_coverage_curve,
    group_bootstrap_ci,
)
from src.models.audio_models import LightweightStudentCNN, MFCCBaselineModel, StrongExpertAudioModel
from src.open_set.open_set_detector import OpenSetDetector
from src.routing.benefit_estimator import (
    BenefitEstimator,
    build_pre_expert_feature_matrix,
    compute_sample_loss,
)
from src.routing.budget_controller import BudgetController, BudgetPolicyConfig
from src.routing.confidence_router import ConfidenceRouter
from src.routing.cost_aware_router import CostAwareRouter
from src.routing.entropy_router import EntropyRouter
from src.routing.margin_router import MarginRouter
from src.routing.prototype_router import PrototypeRouter


def _sample_group_ids(samples: List[FeatureSample]) -> List[str]:
    """Return connected-component IDs linking duplicate clips and known source/session groups."""
    n = len(samples)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    duplicate_owner: Dict[str, int] = {}
    source_owner: Dict[str, int] = {}
    for idx, sample in enumerate(samples):
        dup = sample.duplicate_group_id
        src = sample.source_id
        if dup and dup != "unknown":
            if dup in duplicate_owner:
                union(idx, duplicate_owner[dup])
            else:
                duplicate_owner[dup] = idx
        if src and src != "unknown":
            if src in source_owner:
                union(idx, source_owner[src])
            else:
                source_owner[src] = idx
    roots = [find(i) for i in range(n)]
    return [f"group_{root:08d}" for root in roots]


def prepare_arrays(
    samples: List[FeatureSample],
    normalizer: FeatureNormalizer,
) -> Dict[str, Any]:
    mels = np.stack([normalizer.transform_mel(s.log_mel) for s in samples], axis=0)
    mfccs = np.stack([normalizer.transform_mfcc(s.mfcc_vector) for s in samples], axis=0)
    y_known = np.array([s.known_class_idx for s in samples], dtype=int)
    is_bird = np.array([s.is_bird for s in samples], dtype=int)
    sem_targets = [s.semantic_target for s in samples]
    snr_db = np.array([s.snr_db for s in samples], dtype=np.float32)
    clip_r = np.array([s.clipping_ratio for s in samples], dtype=np.float32)
    rms = np.array([s.rms_energy for s in samples], dtype=np.float32)
    groups = _sample_group_ids(samples)
    sources = [s.source_id for s in samples]
    return {
        "mels": mels,
        "mfccs": mfccs,
        "y_known": y_known,
        "is_bird": is_bird,
        "semantic_targets": sem_targets,
        "snr_db": snr_db,
        "clipping_ratio": clip_r,
        "rms_energy": rms,
        "groups": groups,
        "sources": sources,
    }


def generate_oof_routing_examples(
    train_samples: List[FeatureSample],
    n_folds: int = 4,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Split training data by group into `n_folds` folds, train Student and Expert on K-1 folds,
    predict on held-out fold, and compute per-recording observed loss reduction
    d_i = L(f_s(x_i), y_i) - L(f_e(x_i), y_i) and pre-expert feature matrix Z_oof.
    """
    sample_groups = _sample_group_ids(train_samples)
    unique_groups = sorted(set(sample_groups))
    if len(unique_groups) < n_folds:
        raise ValueError(f"Need at least {n_folds} independent groups for OOF routing; found {len(unique_groups)}.")
    rng = np.random.RandomState(seed)
    shuffled_groups = [unique_groups[i] for i in rng.permutation(len(unique_groups))]
    group_to_fold = {g: (idx % n_folds) for idx, g in enumerate(shuffled_groups)}

    Z_list = []
    d_list = []
    s_loss_list = []
    e_loss_list = []

    for fold in range(n_folds):
        print(f"  [OOF] Training fold {fold + 1}/{n_folds}...", flush=True)
        fold_train = [s for s, gid in zip(train_samples, sample_groups) if group_to_fold[gid] != fold]
        fold_val = [s for s, gid in zip(train_samples, sample_groups) if group_to_fold[gid] == fold]
        if not fold_train or not fold_val:
            continue

        norm = FeatureNormalizer().fit(fold_train)
        tr_arr = prepare_arrays(fold_train, norm)
        va_arr = prepare_arrays(fold_val, norm)

        student = LightweightStudentCNN(seed=seed + fold)
        student.fit(tr_arr["mels"], tr_arr["y_known"], tr_arr["is_bird"], epochs=12)

        expert = StrongExpertAudioModel(seed=seed + fold)
        expert.fit(tr_arr["mels"], tr_arr["y_known"], tr_arr["is_bird"], epochs=12)

        tr_s_out = student.predict(tr_arr["mels"])
        os_det = OpenSetDetector()
        os_det.fit_prototypes(tr_s_out.embeddings, tr_arr["y_known"])

        s_out = student.predict(va_arr["mels"])
        e_out = expert.predict(va_arr["mels"])
        s_nov = os_det.compute_all_novelty_features(s_out.logits, s_out.embeddings)

        s_loss = compute_sample_loss(s_out.probabilities, s_out.bird_activity_prob, va_arr["y_known"], va_arr["is_bird"])
        e_loss = compute_sample_loss(e_out.probabilities, e_out.bird_activity_prob, va_arr["y_known"], va_arr["is_bird"])
        d_fold = s_loss - e_loss

        Z_fold = build_pre_expert_feature_matrix(
            student_max_prob=s_nov["raw_max_prob"],
            student_margin=s_nov["raw_margin"],
            student_entropy=s_nov["entropy"],
            prototype_distance=s_nov["prototype_distance"],
            bird_activity_prob=s_out.bird_activity_prob,
            audio_snr_db=va_arr["snr_db"],
            clipping_ratio=va_arr["clipping_ratio"],
            rms_energy=va_arr["rms_energy"],
        )
        Z_list.append(Z_fold)
        d_list.append(d_fold)
        s_loss_list.append(s_loss)
        e_loss_list.append(e_loss)

    return {
        "Z_oof": np.concatenate(Z_list, axis=0),
        "d_oof": np.concatenate(d_list, axis=0),
        "s_loss_oof": np.concatenate(s_loss_list, axis=0),
        "e_loss_oof": np.concatenate(e_loss_list, axis=0),
    }


def evaluate_routing_policies_on_split(
    test_arr: Dict[str, Any],
    s_out: Any,
    e_out: Any,
    s_nov_feats: Dict[str, np.ndarray],
    e_nov_feats: Dict[str, np.ndarray],
    cal_s_nov_feats: Dict[str, np.ndarray],
    cal_Z: np.ndarray,
    cal_s_loss: np.ndarray,
    cal_e_loss: np.ndarray,
    benefit_estimator: BenefitEstimator,
    target_call_rate: float = 0.25,
    lambda_cost: float = 0.01,
    bird_activity_threshold: float = 0.35,
    min_snr_db: float = -3.0,
    max_clipping_ratio: float = 0.08,
    budget_config: Optional[BudgetPolicyConfig] = None,
    seed: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """
    Compare 10 routing policies on held-out test data. target_call_rate is a maximum
    call-rate cap; cost and pre-expert quality gates may result in lower realized rates.
    All thresholds are calibrated strictly on the calibration split.
    """
    cfg = budget_config or BudgetPolicyConfig()
    energy_limited_rate = (
        cfg.max_average_energy_units - cfg.student_cost_units - cfg.router_overhead_units
    ) / max(cfg.expert_cost_units, 1e-12)
    if not 0.0 <= target_call_rate <= 1.0:
        raise ValueError("target_call_rate must be in [0, 1].")
    if not np.isfinite(lambda_cost) or lambda_cost < 0:
        raise ValueError("lambda_cost must be finite and non-negative.")
    effective_target_rate = min(
        float(target_call_rate),
        max(0.0, float(cfg.max_expert_call_rate)),
        max(0.0, float(energy_limited_rate)),
    )

    n = test_arr["y_known"].size
    s_loss = compute_sample_loss(s_out.probabilities, s_out.bird_activity_prob, test_arr["y_known"], test_arr["is_bird"])
    e_loss = compute_sample_loss(e_out.probabilities, e_out.bird_activity_prob, test_arr["y_known"], test_arr["is_bird"])

    Z_test = build_pre_expert_feature_matrix(
        student_max_prob=s_nov_feats["raw_max_prob"],
        student_margin=s_nov_feats["raw_margin"],
        student_entropy=s_nov_feats["entropy"],
        prototype_distance=s_nov_feats["prototype_distance"],
        bird_activity_prob=s_out.bird_activity_prob,
        audio_snr_db=test_arr["snr_db"],
        clipping_ratio=test_arr["clipping_ratio"],
        rms_energy=test_arr["rms_energy"],
    )
    est_b_test = benefit_estimator.predict_benefit(Z_test)
    est_b_cal = benefit_estimator.predict_benefit(cal_Z)

    # Fit thresholds on calibration set for target_call_rate
    conf_r = ConfidenceRouter()
    conf_r.fit_threshold_for_call_rate(cal_s_nov_feats["raw_max_prob"], effective_target_rate)

    ent_r = EntropyRouter()
    ent_r.fit_threshold_for_call_rate(cal_s_nov_feats["entropy"], effective_target_rate)

    mar_r = MarginRouter()
    mar_r.fit_threshold_for_call_rate(cal_s_nov_feats["raw_margin"], effective_target_rate)

    proto_r = PrototypeRouter()
    proto_r.fit_threshold_for_call_rate(cal_s_nov_feats["prototype_distance"], effective_target_rate)

    # Cost-blind learned benefit router (invokes whenever est_benefit > quantile on calibration)
    cb_bc = BudgetController(cfg)
    cb_bc.optimize_threshold_on_calibration(est_b_cal, cal_s_loss, cal_e_loss, lambda_cost=0.0, target_call_rate=target_call_rate)

    # Proposed cost-aware benefit router (incorporates lambda * expert_cost floor + budget cap)
    ca_bc = BudgetController(cfg)
    ca_bc.optimize_threshold_on_calibration(est_b_cal, cal_s_loss, cal_e_loss, lambda_cost=lambda_cost, target_call_rate=target_call_rate)

    rng = np.random.RandomState(seed)
    random_call_count = min(n, int(np.floor(effective_target_rate * n)))
    rand_mask = np.zeros(n, dtype=bool)
    if random_call_count:
        rand_mask[rng.choice(n, size=random_call_count, replace=False)] = True

    # Offline oracle (upper bound only - uses true test loss reduction).
    # Select no more than the effective budget and only when the expert reduces loss.
    true_d = s_loss - e_loss
    oracle_mask = np.zeros(n, dtype=bool)
    oracle_slots = min(n, int(np.floor(effective_target_rate * n)))
    positive_benefit_indices = np.flatnonzero(true_d > 0.0)
    if oracle_slots and positive_benefit_indices.size:
        oracle_order = positive_benefit_indices[
            np.argsort(-true_d[positive_benefit_indices], kind="mergesort")
        ]
        oracle_mask[oracle_order[:oracle_slots]] = True

    policies = {
        "student_only": np.zeros(n, dtype=bool),
        "expert_only": np.ones(n, dtype=bool),
        "random_routing": rand_mask,
        "fixed_confidence_routing": conf_r.route(s_nov_feats["raw_max_prob"]),
        "entropy_routing": ent_r.route(s_nov_feats["entropy"]),
        "margin_routing": mar_r.route(s_nov_feats["raw_margin"]),
        "prototype_distance_routing": proto_r.route(s_nov_feats["prototype_distance"]),
        "cost_blind_benefit_routing": est_b_test > cb_bc.calibrated_threshold,
        "proposed_cost_aware_routing": (
            (est_b_test > ca_bc.calibrated_threshold)
            & (s_out.bird_activity_prob > bird_activity_threshold)
            & (test_arr["snr_db"] >= min_snr_db)
            & (test_arr["clipping_ratio"] <= max_clipping_ratio)
            & (test_arr["rms_energy"] >= 0.0002)
        ),
        "offline_oracle_upper_bound": oracle_mask,
    }

    # Calibration-derived thresholds can exceed a configured cap under test-time
    # distribution shift. Enforce the same hard per-test cap on every budgeted
    # policy (the expert-only row remains an explicitly over-budget reference).
    max_test_calls = min(n, int(np.floor(effective_target_rate * n)))
    priorities = {
        "random_routing": np.zeros(n, dtype=np.float64),
        "fixed_confidence_routing": -np.asarray(s_nov_feats["raw_max_prob"]),
        "entropy_routing": np.asarray(s_nov_feats["entropy"]),
        "margin_routing": -np.asarray(s_nov_feats["raw_margin"]),
        "prototype_distance_routing": np.asarray(s_nov_feats["prototype_distance"]),
        "cost_blind_benefit_routing": np.asarray(est_b_test),
        "proposed_cost_aware_routing": np.asarray(est_b_test),
        "offline_oracle_upper_bound": np.asarray(true_d),
    }
    for policy_name, priority in priorities.items():
        candidate_indices = np.flatnonzero(policies[policy_name])
        if candidate_indices.size > max_test_calls:
            keep = candidate_indices[
                np.argsort(-priority[candidate_indices], kind="mergesort")[:max_test_calls]
            ]
            capped = np.zeros(n, dtype=bool)
            capped[keep] = True
            policies[policy_name] = capped

    known_mask = test_arr["y_known"] >= 0
    results: Dict[str, Dict[str, Any]] = {}

    for pname, mask in policies.items():
        call_rate = float(np.mean(mask))
        combined_loss = np.where(mask, e_loss, s_loss)
        mean_loss = float(np.mean(combined_loss))
        loss_reduction = float(np.mean(s_loss) - mean_loss)
        n_calls = int(np.sum(mask))
        imp_per_call = float(np.sum(s_loss - combined_loss) / max(1, n_calls)) if n_calls > 0 else 0.0

        # Known-species accuracy & Macro-F1 under routing policy
        s_preds = np.argmax(s_out.probabilities, axis=1)
        e_preds = np.argmax(e_out.probabilities, axis=1)
        routed_preds = np.where(mask, e_preds, s_preds)
        cls_m = compute_classification_metrics(test_arr["y_known"][known_mask], routed_preds[known_mask])

        if pname == "student_only":
            avg_cost_units = cfg.student_cost_units
        elif pname == "expert_only":
            avg_cost_units = cfg.expert_cost_units
        else:
            avg_cost_units = cfg.student_cost_units + cfg.router_overhead_units + call_rate * cfg.expert_cost_units
        ci_loss = group_bootstrap_ci(test_arr["groups"], combined_loss)

        results[pname] = {
            "expert_invoke_mask": mask.astype(bool).tolist(),
            "calibrated_threshold": (
                float(ca_bc.calibrated_threshold) if pname == "proposed_cost_aware_routing"
                else float(cb_bc.calibrated_threshold) if pname == "cost_blind_benefit_routing"
                else None
            ),
            "effective_max_call_rate": round(effective_target_rate, 6),
            "expert_call_rate": round(call_rate, 4),
            "expected_decision_loss": round(mean_loss, 4),
            "loss_ci_95": [ci_loss["ci_low"], ci_loss["ci_high"]],
            "loss_reduction_vs_student": round(loss_reduction, 4),
            "improvement_per_expert_call": round(imp_per_call, 4),
            "known_species_accuracy": cls_m["accuracy"],
            "known_species_macro_f1": cls_m["macro_f1"],
            "average_compute_cost_units": round(avg_cost_units, 3),
            "is_deployable": pname != "offline_oracle_upper_bound",
        }
    return results


def run_species_held_out_rotations(
    train_samples: List[FeatureSample],
    test_samples: List[FeatureSample],
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Downstream open-set evaluation rotation excluding one of the four named bird species
    at a time from supervised training and evaluating whether the held-out species is
    detected as unfamiliar rather than forced into the remaining 3 known species.
    """
    rotation_results = []
    for held_out_idx, held_out_name in enumerate(KNOWN_SPECIES):
        print(f"  [Held-Out Rotation] Evaluating held-out {held_out_name} ({held_out_idx + 1}/{len(KNOWN_SPECIES)})...", flush=True)
        # Filter training samples to exclude held_out_idx
        rot_train = [s for s in train_samples if s.known_class_idx != held_out_idx]
        if not rot_train:
            continue
        norm = FeatureNormalizer().fit(rot_train)

        # Remap remaining 3 species indices to 0..2
        remaining = [i for i in range(4) if i != held_out_idx]
        idx_map = {old_i: new_i for new_i, old_i in enumerate(remaining)}

        tr_mels = np.stack([norm.transform_mel(s.log_mel) for s in rot_train], axis=0)
        tr_yk = np.array([idx_map.get(s.known_class_idx, -1) for s in rot_train], dtype=int)
        tr_bird = np.array([s.is_bird for s in rot_train], dtype=int)

        student = LightweightStudentCNN(num_classes=3, seed=seed + held_out_idx)
        student.fit(tr_mels, tr_yk, tr_bird, epochs=10)
        tr_out = student.predict(tr_mels)

        os_det = OpenSetDetector(num_known_classes=3)
        os_det.fit_prototypes(tr_out.embeddings, tr_yk)
        os_det.calibrate_thresholds(tr_out.logits, tr_out.embeddings, tr_yk >= 0, target_known_recall=0.88)

        te_mels = np.stack([norm.transform_mel(s.log_mel) for s in test_samples], axis=0)
        te_out = student.predict(te_mels)
        # Held-out species + other_birds + noise are unknown relative to the 3 trained species
        is_unk = np.array([1 if (s.known_class_idx == held_out_idx or s.known_class_idx == -1) else 0 for s in test_samples])
        held_out_only_mask = np.array([s.known_class_idx == held_out_idx for s in test_samples])

        scores = os_det.score(te_out.logits, te_out.embeddings, method="prototype_distance")
        m_stats = compute_open_set_metrics(is_unk, scores.novelty_score, scores.threshold)
        held_out_reject_rate = (
            float(np.mean(scores.is_rejected_unknown[held_out_only_mask]))
            if np.any(held_out_only_mask)
            else 0.0
        )
        rotation_results.append({
            "held_out_species": held_out_name,
            "auroc": m_stats["auroc"],
            "aupr": m_stats["aupr"],
            "fpr95": m_stats["fpr95"],
            "held_out_species_rejection_rate": round(held_out_reject_rate, 4),
            "false_known_assignment_on_held_out": round(1.0 - held_out_reject_rate, 4),
        })
    return rotation_results
