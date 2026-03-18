"""Canonical paper-contract utilities for CalibraTeach.

This module encodes the paper-level formulas used for:
- six-signal confidence aggregation
- temperature scaling via validation-set NLL grid search
- selective prediction operating point and AUC-AC
- deployment guardrail checks
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# Paper-defined authority priors.
AUTHORITY_PRIORS: Dict[str, float] = {
    "peer-reviewed": 1.0,
    "textbook": 0.9,
    "official docs": 0.8,
    "lecture notes": 0.7,
    "stack overflow": 0.6,
    "so": 0.6,
    "blogs": 0.4,
    "blog": 0.4,
}

# Paper-defined learned aggregation weights in order:
# [rel, ent, div, agree, margin, auth]
SIGNAL_WEIGHTS: np.ndarray = np.asarray([0.18, 0.35, 0.10, 0.15, 0.10, 0.12], dtype=np.float64)

# Paper-defined temperature search grid.
TEMPERATURE_GRID: Tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)


@dataclass(frozen=True)
class PaperSignals:
    s_rel: float
    s_ent: float
    s_div: float
    s_agree: float
    s_margin: float
    s_auth: float

    def to_vector(self) -> np.ndarray:
        return np.asarray(
            [self.s_rel, self.s_ent, self.s_div, self.s_agree, self.s_margin, self.s_auth],
            dtype=np.float64,
        )


def _safe_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _normalize_source_type(source_type: Optional[str]) -> str:
    if not source_type:
        return ""
    return source_type.strip().lower()


def authority_prior_for_source(source_type: Optional[str], fallback: float = 0.4) -> float:
    st = _normalize_source_type(source_type)
    if st in AUTHORITY_PRIORS:
        return AUTHORITY_PRIORS[st]
    if "peer" in st and "review" in st:
        return AUTHORITY_PRIORS["peer-reviewed"]
    if "textbook" in st:
        return AUTHORITY_PRIORS["textbook"]
    if "official" in st or "docs" in st:
        return AUTHORITY_PRIORS["official docs"]
    if "lecture" in st:
        return AUTHORITY_PRIORS["lecture notes"]
    if "stack" in st or "overflow" in st:
        return AUTHORITY_PRIORS["stack overflow"]
    if "blog" in st:
        return AUTHORITY_PRIORS["blogs"]
    return float(fallback)


def compute_negative_mean_pairwise_similarity(similarity_matrix: np.ndarray) -> float:
    """S_div: negative mean pairwise similarity across the evidence set."""
    if similarity_matrix.size == 0:
        return 0.0
    matrix = np.asarray(similarity_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("similarity_matrix must be square")
    n = matrix.shape[0]
    if n < 2:
        return 0.0

    upper = matrix[np.triu_indices(n, k=1)]
    if upper.size == 0:
        return 0.0
    return float(-np.mean(upper))


def compute_signals(
    *,
    similarities: Sequence[float],
    entailment_probs: Sequence[float],
    voted_labels: Sequence[str],
    source_types: Sequence[Optional[str]],
    pairwise_similarity_matrix: Optional[np.ndarray] = None,
) -> PaperSignals:
    """Compute the six confidence signals exactly as defined in the paper."""
    rel = _safe_mean(similarities)
    ent = _safe_mean(entailment_probs)

    if pairwise_similarity_matrix is None:
        # If matrix is unavailable, estimate from scalar similarities around mean.
        sims = np.asarray(similarities, dtype=np.float64)
        if sims.size >= 2:
            centered = sims - np.mean(sims)
            approx = np.corrcoef(centered + 1e-12)
            div = compute_negative_mean_pairwise_similarity(np.asarray(approx, dtype=np.float64))
        else:
            div = 0.0
    else:
        div = compute_negative_mean_pairwise_similarity(pairwise_similarity_matrix)

    if voted_labels:
        label_counts: Dict[str, int] = {}
        for label in voted_labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        majority_label = max(label_counts.items(), key=lambda kv: kv[1])[0]
        agree = float(np.mean([1.0 if label == majority_label else 0.0 for label in voted_labels]))
    else:
        agree = 0.0

    if entailment_probs:
        ent_arr = np.asarray(entailment_probs, dtype=np.float64)
        margin = float(np.max(ent_arr) - np.min(ent_arr))
    else:
        margin = 0.0

    auth_values = [authority_prior_for_source(source_type) for source_type in source_types]
    auth = _safe_mean(auth_values)

    return PaperSignals(
        s_rel=float(np.clip(rel, 0.0, 1.0)),
        s_ent=float(np.clip(ent, 0.0, 1.0)),
        s_div=float(np.clip(div, -1.0, 1.0)),
        s_agree=float(np.clip(agree, 0.0, 1.0)),
        s_margin=float(np.clip(margin, 0.0, 1.0)),
        s_auth=float(np.clip(auth, 0.0, 1.0)),
    )


def aggregate_confidence(signals: PaperSignals, weights: np.ndarray = SIGNAL_WEIGHTS) -> float:
    vec = signals.to_vector()
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (6,):
        raise ValueError("weights must have shape (6,)")
    return float(np.clip(np.dot(vec, w), 0.0, 1.0))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def temperature_scale_logits(logits: Sequence[float], temperature: float) -> np.ndarray:
    t = float(temperature)
    if t <= 0.0:
        raise ValueError("temperature must be > 0")
    z = np.asarray(logits, dtype=np.float64)
    return sigmoid(z / t)


def nll_for_temperature(logits: Sequence[float], labels: Sequence[int], temperature: float) -> float:
    p = np.clip(temperature_scale_logits(logits, temperature), 1e-12, 1.0 - 1e-12)
    y = np.asarray(labels, dtype=np.float64)
    if y.shape != p.shape:
        raise ValueError("labels and logits must have same length")
    return float(-np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def fit_temperature_validation_nll(
    logits: Sequence[float],
    labels: Sequence[int],
    grid: Sequence[float] = TEMPERATURE_GRID,
) -> Dict[str, object]:
    losses: List[float] = []
    for t in grid:
        losses.append(nll_for_temperature(logits, labels, float(t)))
    best_idx = int(np.argmin(np.asarray(losses, dtype=np.float64)))
    best_t = float(grid[best_idx])
    return {
        "best_tau": best_t,
        "grid": [float(t) for t in grid],
        "nll": [float(v) for v in losses],
        "objective": "validation_nll",
    }


def confidence_from_probability(prob_supported: Sequence[float]) -> np.ndarray:
    p = np.asarray(prob_supported, dtype=np.float64)
    return np.maximum(p, 1.0 - p)


def selective_prediction_at_threshold(
    prob_supported: Sequence[float],
    gold_labels: Sequence[int],
    threshold: float,
) -> Dict[str, float]:
    p = np.asarray(prob_supported, dtype=np.float64)
    y = np.asarray(gold_labels, dtype=np.int64)
    conf = confidence_from_probability(p)

    keep = conf >= float(threshold)
    coverage = float(np.mean(keep)) if len(keep) else 0.0

    if np.any(keep):
        pred = (p >= 0.5).astype(np.int64)
        selective_acc = float(np.mean(pred[keep] == y[keep]))
        accepted = int(np.sum(keep))
    else:
        selective_acc = 0.0
        accepted = 0

    return {
        "threshold": float(threshold),
        "coverage": coverage,
        "selective_accuracy": selective_acc,
        "accepted": accepted,
        "n": int(len(p)),
    }


def sweep_selective_operating_points(
    prob_supported: Sequence[float],
    gold_labels: Sequence[int],
    start: float = 0.60,
    stop: float = 0.95,
    step: float = 0.01,
) -> Dict[str, object]:
    thresholds = np.arange(start, stop + 1e-9, step)
    points = [
        selective_prediction_at_threshold(prob_supported, gold_labels, float(t))
        for t in thresholds
    ]

    coverage = np.asarray([p["coverage"] for p in points], dtype=np.float64)
    accuracy = np.asarray([p["selective_accuracy"] for p in points], dtype=np.float64)

    order = np.argsort(coverage)
    auc_ac = float(np.trapezoid(accuracy[order], coverage[order]))

    return {
        "points": points,
        "auc_ac": auc_ac,
        "thresholds": [float(t) for t in thresholds],
    }


def expected_calibration_error(
    probs: Sequence[float],
    labels: Sequence[int],
    n_bins: int,
) -> float:
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if len(p) == 0:
        return 0.0
    conf = confidence_from_probability(p)
    pred = (p >= 0.5).astype(np.int64)
    correct = (pred == y).astype(np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(conf)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            in_bin = (conf >= lo) & (conf <= hi)
        else:
            in_bin = (conf >= lo) & (conf < hi)
        if not np.any(in_bin):
            continue
        weight = float(np.mean(in_bin))
        acc = float(np.mean(correct[in_bin]))
        c = float(np.mean(conf[in_bin]))
        ece += weight * abs(acc - c)
    return float(ece)


def adaptive_ece_equal_mass(
    probs: Sequence[float],
    labels: Sequence[int],
    n_bins: int = 10,
) -> float:
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if len(p) == 0:
        return 0.0
    conf = confidence_from_probability(p)
    pred = (p >= 0.5).astype(np.int64)
    correct = (pred == y).astype(np.float64)

    order = np.argsort(conf)
    conf = conf[order]
    correct = correct[order]

    splits = np.array_split(np.arange(len(conf)), n_bins)
    ece = 0.0
    n = len(conf)
    for idx in splits:
        if len(idx) == 0:
            continue
        acc = float(np.mean(correct[idx]))
        c = float(np.mean(conf[idx]))
        ece += (len(idx) / n) * abs(acc - c)
    return float(ece)


def run_guardrail_checks(
    *,
    calibration_split_name: str,
    train_ids: Iterable[str],
    val_ids: Iterable[str],
    test_ids: Iterable[str],
    temperature_star: float,
    val_ece: float,
    coverage_tau_090: float,
) -> List[str]:
    warnings: List[str] = []

    split_name = calibration_split_name.strip().lower()
    if "test" in split_name:
        warnings.append("Calibration split appears to be test; expected validation-only calibration.")

    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)

    if (train_set & val_set) or (train_set & test_set) or (val_set & test_set):
        warnings.append("Potential split leakage detected: overlapping IDs across train/val/test.")

    if not (0.5 <= float(temperature_star) <= 2.0):
        warnings.append("T* outside [0.5, 2.0]; calibration data may be unstable.")

    if float(val_ece) > 0.15:
        warnings.append("Validation ECE exceeds 0.15 before deployment.")

    if float(coverage_tau_090) < 0.70:
        warnings.append("Coverage at tau=0.90 fell below 70%.")

    return warnings
