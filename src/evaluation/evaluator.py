"""
Comprehensive Evaluation & Statistical Analysis Suite for EcoGate-Audio.
Computes:
1. Classification metrics: Accuracy, Precision, Recall, Macro-F1, Weighted-F1, Balanced Accuracy, Confusion Matrix.
2. Open-Set metrics: AUROC, AUPR, FPR95, Unknown Recall, False Known-Species Assignment Rate, ECE, Brier Score.
3. Selective Inference & Routing metrics: Risk-Coverage curves, Coverage at Fixed Risk, Abstention Rate,
   Expert Invocation Frequency, Improvement per Expert Invocation, Decision Quality at Matched Budgets.
4. Statistical Analysis: Group-level bootstrap confidence intervals (respecting recording/duplicate groups)
   and paired bootstrap significance tests.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.calibration.calibrator import compute_ece, compute_brier_score


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 4,
) -> Dict[str, Any]:
    """Compute multi-class accuracy, macro/weighted precision, recall, F1, balanced accuracy, and confusion matrix."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    n = y_true.size
    if n == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0, "balanced_accuracy": 0.0}

    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1

    precisions = np.zeros(num_classes, dtype=np.float64)
    recalls = np.zeros(num_classes, dtype=np.float64)
    f1s = np.zeros(num_classes, dtype=np.float64)
    supports = np.sum(cm, axis=1)

    for c in range(num_classes):
        tp = float(cm[c, c])
        fp = float(np.sum(cm[:, c]) - tp)
        fn = float(np.sum(cm[c, :]) - tp)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        precisions[c] = prec
        recalls[c] = rec
        f1s[c] = f1

    acc = float(np.trace(cm)) / float(max(1, np.sum(cm)))
    macro_prec = float(np.mean(precisions))
    macro_rec = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))
    total_sup = float(max(1, np.sum(supports)))
    weighted_f1 = float(np.sum(f1s * supports) / total_sup)

    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "balanced_accuracy": round(macro_rec, 4),
        "per_class_f1": [round(float(x), 4) for x in f1s],
        "confusion_matrix": cm.tolist(),
    }


def compute_binary_auroc_aupr_fpr95(
    y_positive: np.ndarray,
    scores: np.ndarray,
) -> Dict[str, float]:
    """
    Compute tie-aware ROC-AUC, non-interpolated average precision, and FPR95.
    All samples with equal scores are processed at the same threshold.
    """
    y = np.asarray(y_positive, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if y.shape != scores.shape:
        raise ValueError("y_positive and scores must have the same shape.")
    if y.size == 0:
        return {"auroc": 0.5, "aupr": 0.0, "fpr95": 1.0}
    if not np.all(np.isfinite(scores)) or not np.all(np.isin(y, [0, 1])):
        raise ValueError("Scores must be finite and labels must be binary (0/1).")
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return {"auroc": 0.5, "aupr": float(n_pos > 0), "fpr95": 1.0}

    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_y = y[order]
    group_ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), y.size - 1]
    tps = np.cumsum(sorted_y == 1)[group_ends].astype(np.float64)
    fps = np.cumsum(sorted_y == 0)[group_ends].astype(np.float64)
    tpr = np.r_[0.0, tps / n_pos, 1.0]
    fpr = np.r_[0.0, fps / n_neg, 1.0]
    auc_fn = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    auroc = float(auc_fn(tpr, fpr))

    precision = tps / np.maximum(1.0, tps + fps)
    recall = tps / n_pos
    aupr = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
    idx95 = int(np.searchsorted(tpr, 0.95, side="left"))
    fpr95 = float(fpr[min(idx95, fpr.size - 1)])
    return {
        "auroc": round(auroc, 4),
        "aupr": round(aupr, 4),
        "fpr95": round(fpr95, 4),
    }

def compute_open_set_metrics(
    is_unknown_true: np.ndarray,
    novelty_scores: np.ndarray,
    threshold: float,
    known_probs: Optional[np.ndarray] = None,
    known_labels: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Evaluate open-set recognition performance:
    - Positive direction: `is_unknown_true == 1` (unknown bird or non-bird), higher `novelty_scores` = more unknown.
    """
    curve_stats = compute_binary_auroc_aupr_fpr95(is_unknown_true, novelty_scores)
    pred_unknown = novelty_scores > threshold
    unk_mask = is_unknown_true.astype(bool)
    known_mask = ~unk_mask

    unk_recall = float(np.mean(pred_unknown[unk_mask])) if np.any(unk_mask) else 0.0
    false_known_rate = float(np.mean(~pred_unknown[unk_mask])) if np.any(unk_mask) else 0.0
    known_accept_rate = float(np.mean(~pred_unknown[known_mask])) if np.any(known_mask) else 0.0

    ece = 0.0
    brier = 0.0
    if known_probs is not None and known_labels is not None:
        valid_k = known_labels >= 0
        if np.any(valid_k):
            ece = compute_ece(known_probs[valid_k], known_labels[valid_k])
            brier = compute_brier_score(known_probs[valid_k], known_labels[valid_k])

    return {
        **curve_stats,
        "unknown_recall": round(unk_recall, 4),
        "false_known_species_assignment_rate": round(false_known_rate, 4),
        "known_species_acceptance_rate": round(known_accept_rate, 4),
        "ece": round(ece, 4),
        "brier_score": round(brier, 4),
    }


def compute_risk_coverage_curve(
    confidences: np.ndarray,
    losses: np.ndarray,
    coverages: List[float] = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
) -> Dict[str, Any]:
    """Compute selective risk among the highest-confidence retained predictions."""
    conf = np.asarray(confidences, dtype=np.float64).reshape(-1)
    loss = np.asarray(losses, dtype=np.float64).reshape(-1)
    if conf.shape != loss.shape:
        raise ValueError("confidences and losses must have identical shapes.")
    if conf.size == 0:
        return {"aurc": 0.0, "curve": []}
    if not np.all(np.isfinite(conf)) or not np.all(np.isfinite(loss)):
        raise ValueError("confidences and losses must contain only finite values.")
    if np.any(loss < 0):
        raise ValueError("losses must be non-negative.")
    if not coverages:
        raise ValueError("At least one coverage level is required.")

    order = np.argsort(-conf, kind="mergesort")
    sorted_losses = loss[order]
    n = loss.size
    points = []
    for coverage in coverages:
        coverage = float(coverage)
        if not 0.0 < coverage <= 1.0:
            raise ValueError("Each coverage must be in (0, 1].")
        k = min(n, max(1, int(np.ceil(coverage * n))))
        actual_coverage = k / n
        risk = float(np.mean(sorted_losses[:k]))
        points.append({
            "coverage": round(actual_coverage, 6),
            "selective_risk": round(risk, 6),
            "abstention_rate": round(1.0 - actual_coverage, 6),
        })
    cumulative_risk = np.cumsum(sorted_losses) / np.arange(1, n + 1)
    aurc = float(np.mean(cumulative_risk))
    return {"aurc": round(aurc, 6), "curve": points}

def group_bootstrap_ci(
    group_ids: List[str],
    sample_values: np.ndarray,
    n_boot: int = 200,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Compute cluster/group-level bootstrap mean and confidence interval so correlated clips
    from the same recording session or duplicate group are resampled together.
    """
    rng = np.random.RandomState(seed)
    group_to_idxs: Dict[str, List[int]] = {}
    for idx, gid in enumerate(group_ids):
        group_to_idxs.setdefault(gid, []).append(idx)

    unique_groups = sorted(group_to_idxs.keys())
    g_count = len(unique_groups)
    if g_count == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "std_err": 0.0}

    boot_means = np.zeros(n_boot, dtype=np.float64)
    for b in range(n_boot):
        chosen = rng.choice(g_count, size=g_count, replace=True)
        idxs = [i for g_i in chosen for i in group_to_idxs[unique_groups[g_i]]]
        boot_means[b] = float(np.mean(sample_values[idxs]))

    alpha = (1.0 - ci) / 2.0
    return {
        "mean": round(float(np.mean(sample_values)), 4),
        "ci_low": round(float(np.quantile(boot_means, alpha)), 4),
        "ci_high": round(float(np.quantile(boot_means, 1.0 - alpha)), 4),
        "std_err": round(float(np.std(boot_means)), 4),
    }
