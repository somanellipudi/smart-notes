#!/usr/bin/env python
"""Run optimization/minimal-deployment ablations and summarize metrics."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def _run_profile(name: str, env_overrides: Dict[str, str], out_root: Path) -> Dict[str, float]:
    profile_out = out_root / name
    profile_out.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(env_overrides)

    cmd = [
        sys.executable,
        "-m",
        "src.evaluation.runner",
        "--mode",
        "verifiable_full",
        "--out",
        str(profile_out),
    ]
    subprocess.check_call(cmd, env=env)

    metrics_path = profile_out / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    row = {
        "profile": name,
        "accuracy": float(metrics.get("accuracy", 0.0)),
        "macro_f1": float(metrics.get("macro_f1", 0.0)),
        "ece": float(metrics.get("ece", 0.0)),
        "brier_score": float(metrics.get("brier_score", 0.0)),
        "risk_coverage_auc": float(metrics.get("risk_coverage", {}).get("auc", 0.0)),
    }
    return row


def _write_summary(rows: List[Dict[str, float]], out_root: Path) -> None:
    out_root.mkdir(parents=True, exist_ok=True)

    csv_path = out_root / "optimization_ablation_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = out_root / "optimization_ablation_summary.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    md_path = out_root / "optimization_ablation_summary.md"
    lines = [
        "# Optimization / Minimal-Deployment Ablation\n",
        "\n",
        "| Profile | Accuracy | Macro-F1 | ECE | Brier | Risk-Coverage AUC |\n",
        "|---|---:|---:|---:|---:|---:|\n",
    ]
    for r in rows:
        lines.append(
            f"| {r['profile']} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} | {r['ece']:.4f} | {r['brier_score']:.4f} | {r['risk_coverage_auc']:.4f} |\n"
        )
    md_path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/paper/optimization_ablation")
    args = parser.parse_args()

    out_root = Path(args.output_dir)

    profiles = [
        (
            "full_default",
            {
                "MINIMAL_DEPLOYMENT": "false",
                "ENABLE_OPTIMIZATION_ABLATION": "false",
            },
        ),
        (
            "minimal_deployment",
            {
                "MINIMAL_DEPLOYMENT": "true",
                "ENABLE_OPTIMIZATION_ABLATION": "false",
            },
        ),
        (
            "minimal_plus_opt_ablation",
            {
                "MINIMAL_DEPLOYMENT": "true",
                "ENABLE_OPTIMIZATION_ABLATION": "true",
            },
        ),
    ]

    rows = []
    for name, env_overrides in profiles:
        print(f"Running profile: {name}")
        rows.append(_run_profile(name, env_overrides, out_root))

    _write_summary(rows, out_root)
    print(f"Saved optimization ablation summary to: {out_root}")


if __name__ == "__main__":
    main()
