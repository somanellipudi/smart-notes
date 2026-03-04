#!/usr/bin/env python3
"""Generate manuscript figures from computed metrics (single source of truth).

This script reads metrics_summary.json and the official per-example predictions,
recomputes ECE/AUC-AC via src.eval.metrics for safety, validates agreement, and
renders Figure 2/3 annotations from the validated values.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from src.eval.metrics import compute_accuracy_coverage_curve, compute_auc_ac, compute_ece


def _load_summary(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_simple_yaml(path: Path) -> Dict[str, str]:
    data: Dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip().strip('"').strip("'")
    return data


def _load_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    y_true = np.asarray(data["y_true"], dtype=np.int64)
    probs = np.asarray(data["probs"], dtype=np.float64)
    if y_true.ndim != 1 or probs.ndim != 1 or len(y_true) != len(probs):
        raise ValueError(f"Invalid prediction artifact: {path}")
    return y_true, np.clip(probs, 0.0, 1.0)


def _recompute_from_predictions(y_true: np.ndarray, probs: np.ndarray) -> dict:
    ece_result = compute_ece(
        y_true=y_true,
        probs_or_logits=probs,
        n_bins=10,
        scheme="equal_width",
        confidence_mode="predicted_class",
    )
    curve = compute_accuracy_coverage_curve(
        y_true=y_true,
        probs_or_logits=probs,
        confidence_mode="predicted_class",
        thresholds="unique",
    )
    auc_ac = compute_auc_ac(curve["coverage"], curve["accuracy"])
    return {
        "ece": float(ece_result["ece"]),
        "ece_bins": ece_result["bins"],
        "accuracy_coverage_curve": curve,
        "auc_ac": float(auc_ac),
    }


def _assert_summary_match(summary: dict, primary: str, recomputed: dict, tol: float = 1e-6) -> None:
    model = summary["models"][primary]
    diff_ece = abs(float(model["ece"]) - float(recomputed["ece"]))
    diff_auc = abs(float(model["auc_ac"]) - float(recomputed["auc_ac"]))
    if diff_ece > tol or diff_auc > tol:
        raise SystemExit(
            "Metric mismatch between metrics_summary.json and recomputation from predictions: "
            f"ECE diff={diff_ece:.10f}, AUC-AC diff={diff_auc:.10f}. "
            "Check split/run/configuration drift."
        )


def generate_reliability_diagram(primary: str, bins: list[dict], ece: float, output_path: Path) -> None:
    xs = np.array([(b["bin_lower"] + b["bin_upper"]) / 2.0 for b in bins], dtype=np.float64)
    ys = np.array([b["accuracy"] for b in bins], dtype=np.float64)
    counts = np.array([b["count"] for b in bins], dtype=np.float64)

    plt.figure(figsize=(7, 6), dpi=300)
    plt.plot([0, 1], [0, 1], "k--", linewidth=1.2, alpha=0.7, label="Perfect calibration")

    if np.max(counts) > 0:
        size = 40 + 260 * (counts / np.max(counts))
    else:
        size = np.full_like(counts, 40)

    plt.scatter(xs, ys, s=size, alpha=0.8, edgecolors="black", linewidth=0.6, label=primary)
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(alpha=0.25, linestyle="--")
    plt.title("Reliability Diagram")

    plt.text(
        0.98,
        0.03,
        f"ECE (10 bins) = {ece:.4f}",
        transform=plt.gca().transAxes,
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )
    plt.legend(loc="upper left")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def generate_accuracy_coverage_curve(primary: str, curve: dict, auc_ac: float, output_path: Path) -> None:
    coverage = np.asarray(curve["coverage"], dtype=np.float64)
    accuracy = np.asarray(curve["accuracy"], dtype=np.float64)

    order = np.argsort(coverage)
    coverage = coverage[order]
    accuracy = accuracy[order]

    plt.figure(figsize=(7, 6), dpi=300)
    plt.plot(coverage, accuracy, color="#1f77b4", linewidth=2.2, label=primary)
    plt.fill_between(coverage, 0.0, accuracy, color="#1f77b4", alpha=0.18)
    plt.xlabel("Coverage")
    plt.ylabel("Selective Accuracy")
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.25, linestyle="--")
    plt.title("Accuracy-Coverage Curve")

    plt.text(
        0.98,
        0.03,
        f"AUC-AC = {auc_ac:.4f}",
        transform=plt.gca().transAxes,
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )
    plt.legend(loc="lower left")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper figures from unified metrics")
    parser.add_argument("--config", type=Path, default=None, help="Path to configs/paper_run.yaml")
    parser.add_argument("--metrics_file", type=Path, default=Path("artifacts/metrics_summary.json"))
    parser.add_argument("--predictions_file", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=Path("submission_bundle/figures"))
    parser.add_argument("--primary_model", type=str, default=None)
    args = parser.parse_args()

    if args.config:
        cfg = _load_simple_yaml(args.config)
        args.metrics_file = Path(cfg.get("metrics_file", str(args.metrics_file)))
        args.output_dir = Path(cfg.get("figures_dir", str(args.output_dir)))
        if args.predictions_file is None and cfg.get("predictions_file"):
            args.predictions_file = Path(cfg["predictions_file"])
        if args.primary_model is None and cfg.get("model_name"):
            args.primary_model = cfg["model_name"]

    summary = _load_summary(args.metrics_file)
    primary = args.primary_model or summary["primary_model"]
    if primary not in summary["models"]:
        raise SystemExit(f"Primary model '{primary}' not found in {args.metrics_file}")

    if args.predictions_file is None:
        source = summary["models"][primary].get("source")
        if not source:
            raise SystemExit("predictions_file missing and no source field in metrics summary")
        args.predictions_file = Path(source)

    y_true, probs = _load_npz(args.predictions_file)
    recomputed = _recompute_from_predictions(y_true, probs)
    _assert_summary_match(summary, primary, recomputed, tol=1e-6)

    generate_reliability_diagram(
        primary=primary,
        bins=recomputed["ece_bins"],
        ece=recomputed["ece"],
        output_path=args.output_dir / "reliability_diagram_verified.pdf",
    )
    generate_accuracy_coverage_curve(
        primary=primary,
        curve=recomputed["accuracy_coverage_curve"],
        auc_ac=recomputed["auc_ac"],
        output_path=args.output_dir / "accuracy_coverage_verified.pdf",
    )

    print(f"[OK] Wrote {args.output_dir / 'reliability_diagram_verified.pdf'}")
    print(f"[OK] Wrote {args.output_dir / 'accuracy_coverage_verified.pdf'}")
    print(f"[OK] Annotation values: ECE={recomputed['ece']:.4f} AUC-AC={recomputed['auc_ac']:.4f}")


if __name__ == "__main__":
    main()
