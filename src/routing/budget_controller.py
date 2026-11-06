"""
Calibration-only benefit thresholding and streaming call-budget accounting.

The decision rule is: invoke the expert when predicted loss reduction is greater
than lambda_cost times incremental expert cost. A separate hard cap is derived
from both the configured call-rate limit and the average-energy budget.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np


@dataclass
class BudgetPolicyConfig:
    max_expert_call_rate: float = 0.30
    max_average_energy_units: float = 2.40
    student_cost_units: float = 1.0
    expert_cost_units: float = 4.5
    router_overhead_units: float = 0.05
    # Retained for configuration compatibility; these quality constraints are
    # not enforced by threshold optimization because no validation labels are
    # passed to this method. They require a separate calibration/constraint audit.
    min_bird_recall: float = 0.80
    min_known_coverage: float = 0.70
    max_false_known_on_unknown: float = 0.20


class BudgetController:
    """Calibrate an expert-benefit threshold without reading held-out test labels."""

    def __init__(self, cfg: Optional[BudgetPolicyConfig] = None) -> None:
        self.cfg = cfg or BudgetPolicyConfig()
        self.calibrated_threshold = float("inf")
        self.total_queries = 0
        self.expert_calls = 0

    def reset_stream(self) -> None:
        self.total_queries = 0
        self.expert_calls = 0

    def remaining_budget_ratio(self) -> float:
        if self.total_queries == 0:
            return 1.0
        cap = float(self.cfg.max_expert_call_rate)
        if cap <= 0:
            return 0.0
        return float(np.clip(1.0 - (self.expert_calls / self.total_queries) / cap, 0.0, 1.0))

    def record_batch(self, invoke_mask: np.ndarray) -> None:
        """Record actual expert calls after a batch has been executed."""
        mask = np.asarray(invoke_mask, dtype=bool).reshape(-1)
        self.total_queries += int(mask.size)
        self.expert_calls += int(np.sum(mask))

    def optimize_threshold_on_calibration(
        self,
        cal_estimated_benefits: np.ndarray,
        cal_student_losses: np.ndarray,
        cal_expert_losses: np.ndarray,
        lambda_cost: float = 0.01,
        target_call_rate: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Fit a scalar threshold from calibration data only.

        target_call_rate is a maximum call-rate constraint, not an override of
        the configured or energy-derived constraints. The full cost floor is
        applied in all cases. With a constant per-call expert cost, cost-aware
        thresholding changes the overall call rate, not sample ranking.
        """
        benefits = np.asarray(cal_estimated_benefits, dtype=np.float64).reshape(-1)
        student_losses = np.asarray(cal_student_losses, dtype=np.float64).reshape(-1)
        expert_losses = np.asarray(cal_expert_losses, dtype=np.float64).reshape(-1)
        if benefits.size == 0:
            raise ValueError("Calibration arrays must not be empty.")
        if not (benefits.shape == student_losses.shape == expert_losses.shape):
            raise ValueError("Benefits and student/expert loss arrays must have identical shapes.")
        if not (np.all(np.isfinite(benefits)) and np.all(np.isfinite(student_losses)) and np.all(np.isfinite(expert_losses))):
            raise ValueError("Calibration arrays must contain only finite values.")
        if not np.isfinite(lambda_cost) or lambda_cost < 0:
            raise ValueError("lambda_cost must be finite and non-negative.")
        if target_call_rate is not None and not 0.0 <= target_call_rate <= 1.0:
            raise ValueError("target_call_rate must be between 0 and 1.")

        cfg = self.cfg
        if cfg.expert_cost_units <= 0 or cfg.student_cost_units < 0 or cfg.router_overhead_units < 0:
            raise ValueError("Configured inference costs must be non-negative and expert cost must be positive.")
        energy_rate = (
            cfg.max_average_energy_units - cfg.student_cost_units - cfg.router_overhead_units
        ) / cfg.expert_cost_units
        feasible_rate = min(float(cfg.max_expert_call_rate), max(0.0, float(energy_rate)))
        if target_call_rate is not None:
            feasible_rate = min(feasible_rate, float(target_call_rate))

        cost_floor = float(lambda_cost * cfg.expert_cost_units)
        tau = cost_floor
        if feasible_rate <= 0.0:
            tau = float("inf")
        elif feasible_rate < 1.0:
            # A conservative quantile threshold; strict greater-than means ties
            # reduce the call count rather than exceed the cap.
            cap_threshold = float(np.quantile(benefits, 1.0 - feasible_rate, method="higher"))
            tau = max(cost_floor, cap_threshold)

        invoke = benefits > tau
        actual_rate = float(np.mean(invoke))
        achieved_loss = float(np.mean(np.where(invoke, expert_losses, student_losses)))
        avg_energy = (
            cfg.student_cost_units + cfg.router_overhead_units
            + actual_rate * cfg.expert_cost_units
        )
        objective = achieved_loss + lambda_cost * cfg.expert_cost_units * actual_rate
        self.calibrated_threshold = tau
        return {
            "calibrated_threshold": float(tau),
            "cost_floor": round(cost_floor, 6),
            "effective_max_call_rate": round(feasible_rate, 6),
            "calibration_call_rate": round(actual_rate, 6),
            "calibration_expected_loss": round(achieved_loss, 6),
            "calibration_average_energy_units": round(avg_energy, 6),
            "calibration_cost_regularized_objective": round(float(objective), 6),
            "calibration_sample_count": int(benefits.size),
            "enforced_constraints": ["max_expert_call_rate", "max_average_energy_units", "target_call_rate_as_cap"],
            "not_enforced_constraints": ["min_bird_recall", "min_known_coverage", "max_false_known_on_unknown"],
        }
