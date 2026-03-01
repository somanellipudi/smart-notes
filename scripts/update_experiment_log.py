#!/usr/bin/env python3
"""Collect metrics/metadata from an evaluation run and append to a central experiment log.

Usage:
  python scripts/update_experiment_log.py --run_dir outputs/paper/verifiable_full --label verifiable_full

This will update `outputs/benchmark_results/experiment_log.json` with an entry summarizing
metrics.json and metadata.json found in the run directory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime


def load_json(p: Path):
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Append evaluation run to central experiment log.")
    parser.add_argument("--run_dir", type=str, required=True, help="Directory with metrics.json and metadata.json")
    parser.add_argument("--label", type=str, default=None, help="Short label for this run")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"Run directory does not exist: {run_dir}")

    metrics = load_json(run_dir / "metrics.json")
    metadata = load_json(run_dir / "metadata.json")

    entry = {
        "label": args.label or run_dir.name,
        "run_dir": str(run_dir),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "metrics": metrics,
        "metadata": metadata,
    }

    out_base = Path("outputs/benchmark_results")
    out_base.mkdir(parents=True, exist_ok=True)
    log_path = out_base / "experiment_log.json"

    # Load existing log
    if log_path.exists():
        try:
            data = json.loads(log_path.read_text())
            if not isinstance(data, list):
                data = []
        except Exception:
            data = []
    else:
        data = []

    data.append(entry)
    log_path.write_text(json.dumps(data, indent=2))
    print(f"Appended run '{entry['label']}' to {log_path}")


if __name__ == "__main__":
    main()
