"""Authoritative metric definitions for calibration and selective prediction.

This module is the single source of truth for:
- ECE (10 equal-width bins over confidence in [0,1])
- Accuracy-Coverage curve (confidence thresholding)
- AUC-AC (trapezoidal integration over coverage)

Confidence mode:
- predicted_class (recommended): confidence = max(p, 1-p)
- max_prob: alias of predicted_class for binary probabilities
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


ArrayLike = Union[np.ndarray, Sequence[float], Sequence[int]]


def _to_numpy_1d(values: ArrayLike, dtype: np.dtype) -> np.ndarray:
    arr = np.asarray(values, dtype=dtype)
    if arr.ndim != 1:
        raise ValueError("Expected 1D array")
    return arr


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _ensure_probs(probs_or_logits: ArrayLike) -> np.ndarray:
    x = _to_numpy_1d(probs_or_logits, np.float64)
    if np.any(np.isnan(x)):
        raise ValueError("Input contains NaN")
    if np.any((x < 0.0) | (x > 1.0)):
        x = _sigmoid(x)
    return np.clip(x, 0.0, 1.0)


def _confidence_from_probs(probs: np.ndarray, confidence_mode: str) -> np.ndarray:
    mode = confidence_mode.strip().lower()
    if mode in {"predicted_class", "max_prob"}:
        return np.maximum(probs, 1.0 - probs)
    raise ValueError(f"Unsupported confidence_mode: {confidence_mode}")


def _predictions_from_probs(probs: np.ndarray) -> np.ndarray:
    return (probs >= 0.5).astype(np.int64)


def _macro_f1_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    f1_values: List[float] = []
    for klass in (0, 1):
        tp = int(np.sum((y_pred == klass) & (y_true == klass)))
        fp = int(np.sum((y_pred == klass) & (y_true != klass)))
        fn = int(np.sum((y_pred != klass) & (y_true == klass)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            f1_values.append(0.0)
        else:
            f1_values.append(2.0 * precision * recall / (precision + recall))
    return float(np.mean(f1_values))


def compute_ece(
    y_true: ArrayLike,
    probs_or_logits: ArrayLike,
    n_bins: int = 10,
    scheme: str = "equal_width",
    confidence_mode: str = "predicted_class",
) -> Dict[str, object]:
    """Compute Expected Calibration Error (ECE).

    ECE = sum_k (n_k / N) * |acc_k - conf_k|

    Returns dict with:
    - ece: float
    - bins: per-bin statistics for reliability diagrams
    - confidence: list[float]
    - correctness: list[int]
    """
    if n_bins <= 0:
        raise ValueError("n_bins must be > 0")
    if scheme != "equal_width":
        raise ValueError("Only scheme='equal_width' is supported")

    y = _to_numpy_1d(y_true, np.int64)
    if not np.all(np.isin(y, [0, 1])):
        raise ValueError("y_true must be binary (0/1)")

    probs = _ensure_probs(probs_or_logits)
    if len(y) != len(probs):
        raise ValueError("y_true and probs_or_logits must have same length")

    conf = _confidence_from_probs(probs, confidence_mode)
    pred = _predictions_from_probs(probs)
    correct = (pred == y).astype(np.int64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n_total = len(y)
    ece = 0.0
    bins: List[Dict[str, object]] = []

    for i in range(n_bins):
        lo = edges[i]
        hi = edges[i + 1]
        if i == n_bins - 1:
            in_bin = (conf >= lo) & (conf <= hi)
        else:
            in_bin = (conf >= lo) & (conf < hi)

        count = int(np.sum(in_bin))
        if count == 0:
            bins.append(
                {
                    "bin_id": i,
                    "bin_lower": float(lo),
                    "bin_upper": float(hi),
                    "count": 0,
                    "accuracy": 0.0,
                    "confidence": 0.0,
                    "abs_difference": 0.0,
                }
            )
            continue

        acc_k = float(np.mean(correct[in_bin]))
        conf_k = float(np.mean(conf[in_bin]))
        abs_diff = abs(acc_k - conf_k)
        ece += (count / n_total) * abs_diff

        bins.append(
            {
                "bin_id": i,
                "bin_lower": float(lo),
                "bin_upper": float(hi),
                "count": count,
                "accuracy": acc_k,
                "confidence": conf_k,
                "abs_difference": float(abs_diff),
            }
        )

    return {
        "ece": float(ece),
        "bins": bins,
        "confidence": conf.tolist(),
        "correctness": correct.tolist(),
    }


def compute_accuracy_coverage_curve(
    y_true: ArrayLike,
    probs_or_logits: ArrayLike,
    confidence_mode: str = "predicted_class",
    thresholds: Union[str, ArrayLike] = "unique",
) -> Dict[str, List[float]]:
    """Compute accuracy-coverage curve via confidence thresholding.

    Coverage(tau) = fraction kept where confidence >= tau
    Accuracy(tau) = accuracy among kept predictions
    """
    y = _to_numpy_1d(y_true, np.int64)
    if not np.all(np.isin(y, [0, 1])):
        raise ValueError("y_true must be binary (0/1)")

    probs = _ensure_probs(probs_or_logits)
    if len(y) != len(probs):
        raise ValueError("y_true and probs_or_logits must have same length")

    conf = _confidence_from_probs(probs, confidence_mode)
    pred = _predictions_from_probs(probs)
    correct = (pred == y).astype(np.int64)

    if isinstance(thresholds, str):
        if thresholds != "unique":
            raise ValueError("thresholds must be 'unique' or array-like")
        unique = np.unique(conf)
        tau_values = np.concatenate(([1.0 + 1e-12], unique[::-1]))
    else:
        tau_values = _to_numpy_1d(thresholds, np.float64)

    coverage_list: List[float] = []
    accuracy_list: List[float] = []
    threshold_list: List[float] = []

    for tau in tau_values:
        keep = conf >= tau
        coverage = float(np.mean(keep))
        if np.any(keep):
            accuracy = float(np.mean(correct[keep]))
        else:
            accuracy = 1.0
        threshold_list.append(float(tau))
        coverage_list.append(coverage)
        accuracy_list.append(accuracy)

    order = np.argsort(np.asarray(coverage_list))
    coverage_sorted = np.asarray(coverage_list)[order]
    accuracy_sorted = np.asarray(accuracy_list)[order]
    threshold_sorted = np.asarray(threshold_list)[order]

    return {
        "thresholds": threshold_sorted.tolist(),
        "coverage": coverage_sorted.tolist(),
        "accuracy": accuracy_sorted.tolist(),
    }


def compute_auc_ac(coverage: ArrayLike, accuracy: ArrayLike) -> float:
    """Compute AUC-AC as trapezoidal integral of accuracy over coverage in [0,1]."""
    cov = _to_numpy_1d(coverage, np.float64)
    acc = _to_numpy_1d(accuracy, np.float64)
    if len(cov) != len(acc):
        raise ValueError("coverage and accuracy must have same length")

    order = np.argsort(cov)
    cov = cov[order]
    acc = acc[order]

    cov = np.clip(cov, 0.0, 1.0)
    acc = np.clip(acc, 0.0, 1.0)

    return float(np.trapezoid(acc, cov))


@dataclass
class MetricsComputer:
    """Compatibility wrapper exposing prior class-based API."""

    n_bins: int = 10

    def compute_ece(self, probabilities: ArrayLike, labels: ArrayLike, return_bins: bool = False) -> Dict[str, object]:
        result = compute_ece(
            y_true=labels,
            probs_or_logits=probabilities,
            n_bins=self.n_bins,
            scheme="equal_width",
            confidence_mode="predicted_class",
        )
        if not return_bins:
            return {"ece": result["ece"]}
        return {"ece": result["ece"], "bins": result["bins"]}

    def compute_accuracy_coverage_curve(
        self,
        confidences: ArrayLike,
        correctness: ArrayLike,
        thresholds: Optional[ArrayLike] = None,
    ) -> Dict[str, List[float]]:
        conf = _to_numpy_1d(confidences, np.float64)
        corr = _to_numpy_1d(correctness, np.int64)
        if len(conf) != len(corr):
            raise ValueError("confidences and correctness must have same length")

        tau_values = np.unique(conf)[::-1] if thresholds is None else _to_numpy_1d(thresholds, np.float64)
        cov_list: List[float] = []
        acc_list: List[float] = []
        thr_list: List[float] = []
        for tau in tau_values:
            keep = conf >= tau
            cov = float(np.mean(keep))
            acc = float(np.mean(corr[keep])) if np.any(keep) else 1.0
            thr_list.append(float(tau))
            cov_list.append(cov)
            acc_list.append(acc)

        order = np.argsort(np.asarray(cov_list))
        return {
            "thresholds": np.asarray(thr_list)[order].tolist(),
            "coverage": np.asarray(cov_list)[order].tolist(),
            "accuracy": np.asarray(acc_list)[order].tolist(),
        }

    def compute_auc_ac(self, coverage: ArrayLike, accuracy: ArrayLike, normalize: bool = True) -> float:
        return compute_auc_ac(coverage, accuracy)

    def compute_all_metrics(
        self,
        probabilities: ArrayLike,
        labels: ArrayLike,
        thresholds: Optional[ArrayLike] = None,
    ) -> Dict[str, object]:
        y = _to_numpy_1d(labels, np.int64)
        probs = _ensure_probs(probabilities)

        ece_result = compute_ece(
            y_true=y,
            probs_or_logits=probs,
            n_bins=self.n_bins,
            scheme="equal_width",
            confidence_mode="predicted_class",
        )
        curve = compute_accuracy_coverage_curve(
            y_true=y,
            probs_or_logits=probs,
            confidence_mode="predicted_class",
            thresholds="unique" if thresholds is None else thresholds,
        )

        pred = _predictions_from_probs(probs)
        correct = (pred == y).astype(np.int64)

        return {
            "accuracy": float(np.mean(correct)),
            "ece": float(ece_result["ece"]),
            "auc_ac": float(compute_auc_ac(curve["coverage"], curve["accuracy"])),
            "macro_f1": _macro_f1_binary(y, pred),
            "ece_bins": ece_result["bins"],
            "accuracy_coverage_curve": curve,
            "metadata": {
                "n_samples": int(len(y)),
                "ece_n_bins": int(self.n_bins),
                "ece_binning": "equal_width",
                "confidence_definition": "predicted_class",
            },
        }


def create_metrics_computer(n_bins: int = 10) -> MetricsComputer:
    return MetricsComputer(n_bins=n_bins)
