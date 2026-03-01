#!/usr/bin/env python
"""Run parity + plotting end-to-end and produce checksums for artifacts.

Usage:
  python scripts/run_parity_all.py --results_glob evaluation/results/*_result.json --out_dir outputs/paper/calibration_parity_real --fig_dir figures/calibration_parity_real

This script is intended to be run inside the project virtualenv (e.g. .venv).
"""
import argparse
import subprocess
import sys
import glob
from pathlib import Path
import hashlib
import os


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def gather_files(base: Path):
    files = []
    for p in base.rglob("*"):
        if p.is_file():
            files.append(p)
    return sorted(files)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_glob", default="evaluation/results/*_result.json")
    parser.add_argument("--out_dir", default="outputs/paper/calibration_parity_real")
    parser.add_argument("--fig_dir", default="figures/calibration_parity_real")
    args = parser.parse_args()

    results = glob.glob(args.results_glob)
    if not results:
        print("No result JSONs found for pattern:", args.results_glob)
        sys.exit(1)

    # Run parity
    cmd_parity = [sys.executable, "scripts/calibration_parity.py", "--inputs"] + results + ["--output", args.out_dir]
    print("Running:", " ".join(cmd_parity))
    subprocess.check_call(cmd_parity)

    # Run plotting
    cmd_plot = [sys.executable, "scripts/generate_plots_from_calibrated.py", "--input_dir", args.out_dir, "--output_dir", args.fig_dir]
    print("Running:", " ".join(cmd_plot))
    subprocess.check_call(cmd_plot)

    # Produce checksums
    out_path = Path(args.out_dir)
    fig_path = Path(args.fig_dir)
    checksum_file = out_path / "ARTIFACT_CHECKSUMS.sha256"
    checksum_file.parent.mkdir(parents=True, exist_ok=True)

    with checksum_file.open("w", encoding="utf-8") as out:
        for f in gather_files(out_path):
            s = sha256_of_file(f)
            rel = os.path.relpath(str(f), str(Path.cwd()))
            out.write(f"{s}  {rel}\n")
        for f in gather_files(fig_path):
            s = sha256_of_file(f)
            rel = os.path.relpath(str(f), str(Path.cwd()))
            out.write(f"{s}  {rel}\n")

    print("Wrote checksums to", checksum_file)


if __name__ == "__main__":
    main()
