#!/usr/bin/env python3
"""Verify paper consistency: metrics summary, recomputation, and figure freshness."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.eval.metrics import compute_accuracy_coverage_curve, compute_auc_ac, compute_ece


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
        raise ValueError(f"Invalid prediction arrays in {path}")
    return y_true, np.clip(probs, 0.0, 1.0)


def _compute_metrics(y_true: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
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
    return float(ece_result["ece"]), float(auc_ac)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify paper metric/figure consistency")
    parser.add_argument("--config", type=Path, default=Path("configs/paper_run.yaml"))
    args = parser.parse_args()

    cfg = _load_simple_yaml(args.config)
    metrics_file = Path(cfg.get("metrics_file", "artifacts/metrics_summary.json"))
    predictions_file = Path(cfg.get("predictions_file", "artifacts/preds/CalibraTeach.npz"))
    figures_dir = Path(cfg.get("figures_dir", "submission_bundle/figures"))
    model_name = cfg.get("model_name", "CalibraTeach")

    if not metrics_file.exists():
        raise SystemExit(f"metrics_file not found: {metrics_file}")
    if not predictions_file.exists():
        raise SystemExit(f"predictions_file not found: {predictions_file}")

    summary = json.loads(metrics_file.read_text(encoding="utf-8"))
    if model_name not in summary.get("models", {}):
        raise SystemExit(f"Model '{model_name}' not in metrics_summary.json")

    summary_ece = float(summary["models"][model_name]["ece"])
    summary_auc = float(summary["models"][model_name]["auc_ac"])

    y_true, probs = _load_npz(predictions_file)
    computed_ece, computed_auc = _compute_metrics(y_true, probs)

    if abs(computed_ece - summary_ece) > 1e-6:
        raise SystemExit(
            f"ECE mismatch: computed={computed_ece:.10f}, summary={summary_ece:.10f} (>1e-6)"
        )
    if abs(computed_auc - summary_auc) > 1e-6:
        raise SystemExit(
            f"AUC-AC mismatch: computed={computed_auc:.10f}, summary={summary_auc:.10f} (>1e-6)"
        )

    cmd = [
        sys.executable,
        "scripts/generate_paper_figures.py",
        "--config",
        str(args.config),
    ]
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(f"Figure regeneration failed with exit code {rc}")

    fig_rel = figures_dir / "reliability_diagram_verified.pdf"
    fig_acc = figures_dir / "accuracy_coverage_verified.pdf"
    if not fig_rel.exists() or not fig_acc.exists():
        raise SystemExit("Expected figure files missing after regeneration")

    metrics_mtime = metrics_file.stat().st_mtime
    rel_mtime = fig_rel.stat().st_mtime
    acc_mtime = fig_acc.stat().st_mtime
    if rel_mtime < metrics_mtime or acc_mtime < metrics_mtime:
        raise SystemExit(
            "Figure timestamp check failed: figures are older than metrics_summary.json"
        )

    print("[PASS] Paper consistency checks passed")
    print(f"[PASS] Model: {model_name}")
    print(f"[PASS] ECE: {computed_ece:.4f} (summary match within 1e-6)")
    print(f"[PASS] AUC-AC: {computed_auc:.4f} (summary match within 1e-6)")
    print(f"[PASS] Figures newer than metrics file: {fig_rel.name}, {fig_acc.name}")


if __name__ == "__main__":
    main()
