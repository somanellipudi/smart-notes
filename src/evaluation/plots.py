"""Plotting utilities for evaluation: reliability diagram, risk-coverage, confusion matrix, ablation charts.
"""
from __future__ import annotations

from typing import Sequence, Tuple, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from pathlib import Path


def reliability_diagram(confidences: Sequence[float], labels: Sequence[int], num_bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Compute bin centers and accuracies for a reliability diagram.

    Returns (bin_centers, accuracies)
    """
    confidences = np.asarray(confidences)
    labels = np.asarray(labels)
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    bin_idxs = np.digitize(confidences, bins) - 1
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    accuracies = np.zeros(num_bins)
    counts = np.zeros(num_bins)
    for i in range(num_bins):
        mask = bin_idxs == i
        counts[i] = mask.sum()
        if counts[i] > 0:
            accuracies[i] = labels[mask].mean()
        else:
            accuracies[i] = np.nan
    return bin_centers, accuracies


def plot_reliability_diagram(confidences: Sequence[float], labels: Sequence[int], outpath: str, num_bins: int = 10):
    bc, acc = reliability_diagram(confidences, labels, num_bins=num_bins)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(bc, acc, marker="o", label="Accuracy per bin")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Reliability Diagram")
    ax.legend()
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=150)
    fig.savefig(outpath.replace('.png', '.pdf'))
    plt.close(fig)


def compute_risk_coverage_curve(
    confidences: Sequence[float],
    predictions: Sequence[int],
    labels: Sequence[int],
    n_points: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute risk-coverage curve for selective prediction.

    Sweeps thresholds τ from 0 to 1, computing (coverage, accuracy, risk) at each point.
    - Coverage: Fraction of samples with conf >= τ
    - Risk: Error rate among predicted samples (1 - accuracy)
    - Accuracy: Fraction correct among predicted samples

    Returns:
        (thresholds, coverage, accuracy)
    """
    confidences = np.asarray(confidences)
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)

    # Sort by confidence (descending)
    sorted_idx = np.argsort(-confidences)
    sorted_preds = predictions[sorted_idx]
    sorted_labels = labels[sorted_idx]

    n = len(confidences)
    thresholds = np.linspace(0, 1, n_points)
    coverage = np.zeros(n_points)
    accuracy = np.zeros(n_points)

    for i, tau in enumerate(thresholds):
        mask = confidences >= tau
        if mask.sum() == 0:
            coverage[i] = 0.0
            accuracy[i] = 0.0
        else:
            coverage[i] = mask.sum() / n
            accuracy[i] = (predictions[mask] == labels[mask]).mean()

    return thresholds, coverage, accuracy


def plot_risk_coverage(
    confidences: Sequence[float],
    predictions: Sequence[int],
    labels: Sequence[int],
    outpath: str,
    title: str = "Risk-Coverage Curve (Selective Prediction)",
):
    """
    Plot risk-coverage curve: accuracy vs coverage as threshold varies.

    Args:
        confidences: Confidence scores [0, 1]
        predictions: Predicted labels
        labels: True labels
        outpath: Output path (PNG/PDF)
        title: Plot title
    """
    thresholds, coverage, accuracy = compute_risk_coverage_curve(
        confidences, predictions, labels, n_points=50
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(coverage, accuracy, marker='o', linewidth=2, markersize=6, label='Selective Prediction')
    ax.axhline(y=accuracy[0], color='gray', linestyle='--', alpha=0.5, label='Full Coverage Accuracy')
    ax.fill_between(coverage, accuracy, alpha=0.2)
    ax.set_xlabel('Coverage (fraction predicted)', fontsize=12)
    ax.set_ylabel('Accuracy (among predicted)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    pdf_path = outpath.replace('.png', '.pdf')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, labels: Sequence[str], outpath: str):
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(int(cm[i, j]), 'd'), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    fig.tight_layout()
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=150)
    fig.savefig(outpath.replace('.png', '.pdf'))
    plt.close(fig)


def plot_ablation_bar(metrics: dict, metric_name: str, outpath: str):
    # metrics: {ablation_name: value}
    names = list(metrics.keys())
    values = [metrics[n] for n in names]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, values)
    ax.set_ylabel(metric_name)
    ax.set_title(f'Ablation: {metric_name}')
    plt.xticks(rotation=45, ha='right')
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    fig.savefig(outpath.replace('.png', '.pdf'))
    plt.close(fig)


__all__ = [
    'reliability_diagram',
    'plot_reliability_diagram',
    'plot_confusion_matrix',
    'plot_risk_coverage',
    'compute_risk_coverage_curve',
    'plot_ablation_bar',
]
