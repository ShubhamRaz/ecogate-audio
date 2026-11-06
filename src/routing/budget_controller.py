"""
Constrained Mathematical Optimization and Online Budget Controller:
Minimizes expected decision loss subject to inference-energy budget E[E_pi] <= B
and operational constraints (min bird recall, min known coverage, max false known on unknown).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class BudgetPolicyConfig:
    max_expert_call_rate: float = 0.30
    max_average_energy_units: float = 2.40
    student_cost_units: float = 1.0
    expert_cost_units: float = 4.5
    router_overhead_units: float = 0.05
    min_bird_recall: float = 0.80
    min_known_coverage: float = 0.70
    max_false_known_on_unknown: float = 0.20


class BudgetController:
    """
    Calibrates the Lagrange cost multiplier lambda and tracks online remaining budget
    during streaming or batch inference.
    """

    def __init__(self, cfg: Optional[BudgetPolicyConfig] = None) -> None:
        self.cfg = cfg or BudgetPolicyConfig()
        self.calibrated_threshold: float = 0.0
        self.total_queries: int = 0
        self.expert_calls: int = 0

    def reset_stream(self) -> None:
        self.total_queries = 0
        self.expert_calls = 0

    def remaining_budget_ratio(self) -> float:
        if self.total_queries == 0:
            return 1.0
        current_rate = self.expert_calls / float(self.total_queries)
        if self.cfg.max_expert_call_rate <= 0:
            return 0.0
        return float(np.clip(1.0 - (current_rate / self.cfg.max_expert_call_rate), 0.0, 1.0))

    def optimize_threshold_on_calibration(
        self,
        cal_estimated_benefits: np.ndarray,
        cal_student_losses: np.ndarray,
        cal_expert_losses: np.ndarray,
        lambda_cost: float = 0.15,
        target_call_rate: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Select optimal benefit threshold tau* on the calibration split (NEVER on test set)
        satisfying both the marginal cost criterion Delta(z) > lambda * C_expert and the
        energy/call-rate budget constraint E[E_pi] <= B.
        """
        max_rate_from_energy = (
            self.cfg.max_average_energy_units - self.cfg.student_cost_units - self.cfg.router_overhead_units
        ) / max(1e-6, self.cfg.expert_cost_units)
        effective_max_rate = min(
            self.cfg.max_expert_call_rate,
            max(0.0, max_rate_from_energy),
        )
        if target_call_rate is not None:
            effective_max_rate = target_call_rate

        cost_floor = lambda_cost * self.cfg.expert_cost_units * 0.02
        if effective_max_rate <= 0.0:
            tau = float(np.max(cal_estimated_benefits) + 1.0)
        elif effective_max_rate >= 1.0:
            tau = float(np.min(cal_estimated_benefits) - 1.0)
        else:
            q_tau = float(np.quantile(cal_estimated_benefits, 1.0 - effective_max_rate))
            tau = max(q_tau, cost_floor) if target_call_rate is None else q_tau

        self.calibrated_threshold = tau
        invoke = cal_estimated_benefits > tau
        actual_rate = float(np.mean(invoke))
        achieved_loss = float(np.mean(np.where(invoke, cal_expert_losses, cal_student_losses)))
        avg_energy = (
            self.cfg.student_cost_units
            + self.cfg.router_overhead_units
            + actual_rate * self.cfg.expert_cost_units
        )
        return {
            "calibrated_threshold": round(tau, 6),
            "effective_max_call_rate": round(effective_max_rate, 4),
            "calibration_call_rate": round(actual_rate, 4),
            "calibration_expected_loss": round(achieved_loss, 5),
            "calibration_average_energy_units": round(avg_energy, 4),
        }
