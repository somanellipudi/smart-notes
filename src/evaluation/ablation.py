"""Ablation runner that evaluates combinations of settings and writes per-run metrics.
"""
from __future__ import annotations

# add root to path for direct execution
import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import os
import csv
from pathlib import Path
from typing import List
from src.evaluation.runner import run


def run_ablations(output_base: str = "outputs/benchmark_results/ablations") -> List[dict]:
    out_base = Path(output_base)
    out_base.mkdir(parents=True, exist_ok=True)

    ablations = []

    # Toggle temperature scaling ON/OFF and min_entailing_sources
    for temp_on in [True, False]:
        for min_support in [1, 2, 3]:
            # set env vars used by VerificationConfig
            os.environ["TEMPERATURE_SCALING_ENABLED"] = "true" if temp_on else "false"
            os.environ["MIN_ENTAILING_SOURCES_FOR_VERIFIED"] = str(min_support)

            run_name = f"temp_{'on' if temp_on else 'off'}__minsrc_{min_support}"
            out_dir = out_base / run_name
            metrics = run(mode="verifiable_full", output_dir=str(out_dir))
            metrics["run_name"] = run_name
            metrics["temp_on"] = temp_on
            metrics["min_support"] = min_support
            ablations.append(metrics)

    # Save consolidated CSV
    csv_path = out_base / "ablations_summary.csv"
    keys = ["run_name", "mode", "n", "accuracy", "macro_f1", "ece", "brier_score", "temp_on", "min_support"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for a in ablations:
            row = {k: a.get(k, "") for k in keys}
            writer.writerow(row)

    return ablations


if __name__ == "__main__":
    # allow a custom output directory from command line (for reproducibility and paper results)
    import argparse

    parser = argparse.ArgumentParser(description="Run ablation experiments over verification settings.")
    parser.add_argument(
        "--output_base",
        type=str,
        default="outputs/benchmark_results/ablations",
        help="Base directory where per-run folders and summary CSV will be created",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="verifiable_full",
        help="Evaluation mode to run inside each ablation (passed to run()).",
    )

    args = parser.parse_args()
    print(f"Running ablations, results will be written under {args.output_base}")
    # we currently hardcode the grid inside run_ablations; if mode needs to be configurable we
    # could pass it through environment or modify run_ablations signature in future.
    run_ablations(output_base=args.output_base)
