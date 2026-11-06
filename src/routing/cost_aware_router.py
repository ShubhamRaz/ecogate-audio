"""
Cost-Aware Selective Inference & Decision Router for EcoGate-Audio.
Integrates:
1. Audio quality check -> `unresolved` if severe clipping or SNR below floor.
2. Bird activity gate -> `nonbird` if bird activity probability is below threshold.
3. Cost-aware expert routing -> invokes strong expert when Estimated_Benefit(z) > tau(lambda, B).
4. Open-set novelty gate -> `bird_unfamiliar` if novelty score exceeds calibrated open-set threshold.
5. Known species classification -> `known_species` (with species label and calibrated confidence).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
import numpy as np

from src.datasets.bird_dataset import INDEX_TO_LABEL
from src.routing.benefit_estimator import BenefitEstimator, build_pre_expert_feature_matrix
from src.routing.budget_controller import BudgetController


@dataclass
class SelectiveDecision:
    semantic_class: str          # known_species, bird_unfamiliar, nonbird, unresolved
    predicted_species: str       # species name if known_species else semantic_class
    known_class_idx: int         # 0..3 if known_species else -1
    confidence: float
    bird_activity_prob: float
    novelty_score: float
    estimated_benefit: float
    expert_invoked: bool
    model_used: str
    routing_policy: str
    inference_cost_units: float
    latency_ms: float


class CostAwareRouter:
    """
    Central cost-aware routing mechanism comparing:
    - student_only
    - expert_only
    - random_routing
    - confidence_routing
    - entropy_routing
    - margin_routing
    - prototype_routing
    - cost_blind_benefit_routing
    - cost_aware_benefit_routing (proposed)
    - offline_oracle_upper_bound (evaluation reference only; never deployable)
    """

    def __init__(
        self,
        benefit_estimator: BenefitEstimator,
        budget_controller: BudgetController,
        bird_activity_threshold: float = 0.38,
        unknown_novelty_threshold: float = 0.72,
        min_snr_db: float = -5.0,
        max_clipping_ratio: float = 0.12,
    ) -> None:
        self.benefit_estimator = benefit_estimator
        self.budget_controller = budget_controller
        self.bird_activity_threshold = bird_activity_threshold
        self.unknown_novelty_threshold = unknown_novelty_threshold
        self.min_snr_db = min_snr_db
        self.max_clipping_ratio = max_clipping_ratio

    def plan_expert_invocations(
        self,
        student_probs: np.ndarray,
        student_bird_prob: np.ndarray,
        student_entropy: np.ndarray,
        student_margin: np.ndarray,
        student_proto_dist: np.ndarray,
        snr_db: np.ndarray,
        clipping_ratio: np.ndarray,
        rms_energy: np.ndarray,
    ) -> np.ndarray:
        """
        Plan calls from student/audio features only. Call this BEFORE computing any
        expert outputs. The rolling cap limits cumulative calls across batches.
        """
        probs = np.asarray(student_probs, dtype=np.float64)
        if probs.ndim != 2 or probs.shape[1] < 2:
            raise ValueError("student_probs must have shape (N, C), with C >= 2.")
        n = probs.shape[0]
        arrays = [
            np.asarray(v, dtype=np.float64).reshape(-1)
            for v in (
                student_bird_prob, student_entropy, student_margin,
                student_proto_dist, snr_db, clipping_ratio, rms_energy,
            )
        ]
        if any(v.size != n for v in arrays):
            raise ValueError("All routing feature arrays must have one value per sample.")
        if not all(np.all(np.isfinite(v)) for v in [probs, *arrays]):
            raise ValueError("Routing inputs must contain only finite values.")
        bird_prob, entropy, margin, proto_dist, snr, clip, rms = arrays
        if np.any(probs < 0) or np.any(probs > 1) or not np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-3):
            raise ValueError("student_probs must contain normalized probabilities.")
        if n == 0:
            return np.zeros(0, dtype=bool)

        Z = build_pre_expert_feature_matrix(
            student_max_prob=np.max(probs, axis=1),
            student_margin=margin,
            student_entropy=entropy,
            prototype_distance=proto_dist,
            bird_activity_prob=bird_prob,
            audio_snr_db=snr,
            clipping_ratio=clip,
            rms_energy=rms,
        )
        benefits = np.asarray(self.benefit_estimator.predict_benefit(Z), dtype=np.float64).reshape(-1)
        eligible = (
            (benefits > self.budget_controller.calibrated_threshold)
            & (bird_prob >= self.bird_activity_threshold)
            & (clip <= self.max_clipping_ratio)
            & (snr >= self.min_snr_db)
            & (rms >= 0.0002)
        )

        cfg = self.budget_controller.cfg
        energy_rate = (
            cfg.max_average_energy_units - cfg.student_cost_units - cfg.router_overhead_units
        ) / max(cfg.expert_cost_units, 1e-12)
        cap_rate = min(max(0.0, cfg.max_expert_call_rate), max(0.0, energy_rate))
        cumulative_budget = int(np.floor(cap_rate * (self.budget_controller.total_queries + n) + 1e-12))
        slots = max(0, cumulative_budget - self.budget_controller.expert_calls)
        candidates = np.flatnonzero(eligible)
        if candidates.size > slots:
            # Stable descending-benefit selection; ties are deterministic by input order.
            selected = candidates[np.argsort(-benefits[candidates], kind="mergesort")[:slots]]
            mask = np.zeros(n, dtype=bool)
            mask[selected] = True
            return mask
        return eligible.astype(bool)

    def decide_batch(
        self,
        student_probs: np.ndarray,
        student_bird_prob: np.ndarray,
        student_novelty: np.ndarray,
        student_entropy: np.ndarray,
        student_margin: np.ndarray,
        student_proto_dist: np.ndarray,
        snr_db: np.ndarray,
        clipping_ratio: np.ndarray,
        rms_energy: np.ndarray,
        expert_probs: Optional[np.ndarray] = None,
        expert_bird_prob: Optional[np.ndarray] = None,
        expert_novelty: Optional[np.ndarray] = None,
        invoke_mask: Optional[np.ndarray] = None,
        policy_name: str = "cost_aware_benefit_routing",
        student_latency_ms: float = 1.2,
        expert_latency_ms: float = 5.4,
    ) -> List[SelectiveDecision]:
        """
        Produce calibrated final predictions and resource reports for a batch of clips.
        Note: `invoke_mask` is determined strictly from pre-expert quantities before reading
        any expert outputs.
        """
        n = student_probs.shape[0]
        s_max = np.max(student_probs, axis=1)
        Z = build_pre_expert_feature_matrix(
            student_max_prob=s_max,
            student_margin=student_margin,
            student_entropy=student_entropy,
            prototype_distance=student_proto_dist,
            bird_activity_prob=student_bird_prob,
            audio_snr_db=snr_db,
            clipping_ratio=clipping_ratio,
            rms_energy=rms_energy,
        )
        est_benefits = self.benefit_estimator.predict_benefit(Z)

        if invoke_mask is None:
            if expert_probs is not None:
                raise ValueError(
                    "Plan calls before expert inference: call plan_expert_invocations(), "
                    "run the expert only on selected samples, then pass invoke_mask explicitly."
                )
            invoke_mask = self.plan_expert_invocations(
                student_probs=student_probs,
                student_bird_prob=student_bird_prob,
                student_entropy=student_entropy,
                student_margin=student_margin,
                student_proto_dist=student_proto_dist,
                snr_db=snr_db,
                clipping_ratio=clipping_ratio,
                rms_energy=rms_energy,
            )
        else:
            invoke_mask = np.asarray(invoke_mask, dtype=bool).reshape(-1)
            if invoke_mask.size != n:
                raise ValueError("invoke_mask must have one value per sample.")

        # Apply quality and student bird-activity gates even when a caller supplies
        # a custom mask; invalid/no-bird clips should not consume expert calls.
        quality_ok = (
            (np.asarray(clipping_ratio) <= self.max_clipping_ratio)
            & (np.asarray(snr_db) >= self.min_snr_db)
            & (np.asarray(rms_energy) >= 0.0002)
        )
        bird_ok = np.asarray(student_bird_prob) >= self.bird_activity_threshold
        invoke_mask = np.asarray(invoke_mask, dtype=bool) & quality_ok & bird_ok

        # Expert results may be supplied either for the full batch or only for
        # indices selected by the pre-expert planner. Scatter sparse results into
        # student defaults so the expensive model need only run on selected clips.
        call_indices = np.flatnonzero(invoke_mask)
        expert_probs_full = None
        expert_bird_full = None
        expert_novelty_full = None
        if expert_probs is not None:
            exp_p = np.asarray(expert_probs, dtype=np.float64)
            if exp_p.ndim != 2 or exp_p.shape[1] != student_probs.shape[1]:
                raise ValueError("expert_probs must have shape (N, C) or (number_of_calls, C).")
            if exp_p.shape[0] == n:
                expert_probs_full = exp_p
            elif exp_p.shape[0] == call_indices.size:
                expert_probs_full = np.asarray(student_probs, dtype=np.float64).copy()
                expert_probs_full[call_indices] = exp_p
            else:
                raise ValueError("expert_probs row count must equal batch size or number of planned calls.")

        def _expand_expert_vector(values: Optional[np.ndarray], default: np.ndarray, name: str) -> np.ndarray:
            if values is None:
                return np.asarray(default, dtype=np.float64).copy()
            arr = np.asarray(values, dtype=np.float64).reshape(-1)
            if arr.size == n:
                return arr
            if arr.size == call_indices.size:
                full = np.asarray(default, dtype=np.float64).copy()
                full[call_indices] = arr
                return full
            raise ValueError(f"{name} must contain one value per sample or per planned expert call.")

        if expert_probs_full is not None:
            expert_bird_full = _expand_expert_vector(expert_bird_prob, student_bird_prob, "expert_bird_prob")
            expert_novelty_full = _expand_expert_vector(expert_novelty, student_novelty, "expert_novelty")

        decisions: List[SelectiveDecision] = []
        cfg = self.budget_controller.cfg

        for i in range(n):
            # Step 1: Audio quality gate
            if clipping_ratio[i] > self.max_clipping_ratio or snr_db[i] < self.min_snr_db or rms_energy[i] < 0.0002:
                decisions.append(
                    SelectiveDecision(
                        semantic_class="unresolved",
                        predicted_species="unresolved",
                        known_class_idx=-1,
                        confidence=1.0,
                        bird_activity_prob=float(student_bird_prob[i]),
                        novelty_score=float(student_novelty[i]),
                        estimated_benefit=float(est_benefits[i]),
                        expert_invoked=False,
                        model_used="quality_gate_student",
                        routing_policy=policy_name,
                        inference_cost_units=cfg.student_cost_units,
                        latency_ms=student_latency_ms,
                    )
                )
                continue

            use_exp = bool(invoke_mask[i]) and (expert_probs is not None)
            probs_i = expert_probs_full[i] if use_exp else student_probs[i]
            bird_p_i = float(expert_bird_full[i]) if use_exp else float(student_bird_prob[i])
            nov_i = float(expert_novelty_full[i]) if use_exp else float(student_novelty[i])
            cost_i = (
                cfg.student_cost_units + cfg.router_overhead_units + cfg.expert_cost_units
                if use_exp
                else cfg.student_cost_units + cfg.router_overhead_units
            )
            lat_i = student_latency_ms + (expert_latency_ms if use_exp else 0.05)
            model_tag = "expert_model" if use_exp else "student_model"

            # Step 2: Non-bird environmental sound gate
            if bird_p_i < self.bird_activity_threshold:
                decisions.append(
                    SelectiveDecision(
                        semantic_class="nonbird",
                        predicted_species="nonbird",
                        known_class_idx=-1,
                        confidence=round(1.0 - bird_p_i, 4),
                        bird_activity_prob=round(bird_p_i, 4),
                        novelty_score=round(nov_i, 4),
                        estimated_benefit=round(float(est_benefits[i]), 5),
                        expert_invoked=use_exp,
                        model_used=model_tag,
                        routing_policy=policy_name,
                        inference_cost_units=round(cost_i, 3),
                        latency_ms=round(lat_i, 3),
                    )
                )
                continue

            # Step 3: Unfamiliar bird vocalization rejection gate
            if nov_i > self.unknown_novelty_threshold:
                decisions.append(
                    SelectiveDecision(
                        semantic_class="bird_unfamiliar",
                        predicted_species="bird_unfamiliar",
                        known_class_idx=-1,
                        confidence=round(float(np.clip(nov_i, 0.0, 1.0)), 4),
                        bird_activity_prob=round(bird_p_i, 4),
                        novelty_score=round(nov_i, 4),
                        estimated_benefit=round(float(est_benefits[i]), 5),
                        expert_invoked=use_exp,
                        model_used=model_tag,
                        routing_policy=policy_name,
                        inference_cost_units=round(cost_i, 3),
                        latency_ms=round(lat_i, 3),
                    )
                )
                continue

            # Step 4: Known bird species classification
            cls_idx = int(np.argmax(probs_i))
            conf = float(probs_i[cls_idx])
            decisions.append(
                SelectiveDecision(
                    semantic_class="known_species",
                    predicted_species=INDEX_TO_LABEL[cls_idx],
                    known_class_idx=cls_idx,
                    confidence=round(conf, 4),
                    bird_activity_prob=round(bird_p_i, 4),
                    novelty_score=round(nov_i, 4),
                    estimated_benefit=round(float(est_benefits[i]), 5),
                    expert_invoked=use_exp,
                    model_used=model_tag,
                    routing_policy=policy_name,
                    inference_cost_units=round(cost_i, 3),
                    latency_ms=round(lat_i, 3),
                )
            )
        # Only count expert calls for which expert outputs were supplied and used.
        self.budget_controller.record_batch(
            np.asarray([decision.expert_invoked for decision in decisions], dtype=bool)
        )
        return decisions
