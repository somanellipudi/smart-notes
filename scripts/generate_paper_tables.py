#!/usr/bin/env python
"""Generate paper-ready tables from calibration parity artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def _collect_rows(parity_dir: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    tau_lookup: Dict[str, float] = {}
    summary_path = parity_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for system, rec in summary.items():
            tau = rec.get("tau_used")
            if tau is None and isinstance(rec.get("fit"), dict):
                tau = rec["fit"].get("best_tau")
            if tau is not None:
                tau_lookup[system] = float(tau)

    for metrics_file in sorted(parity_dir.rglob("metrics.json")):
        system = metrics_file.parent.name
        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        rows.append(
            {
                "system": system,
                "ece": float(metrics.get("ece", 0.0)),
                "brier_score": float(metrics.get("brier_score", 0.0)),
                "accuracy": float(metrics.get("accuracy", 0.0)),
                "temperature_tau": tau_lookup.get(system, None),
            }
        )

    return rows


def _write_csv(rows: List[Dict[str, object]], out_dir: Path) -> Path:
    path = out_dir / "calibration_parity_table.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["system", "ece", "brier_score", "accuracy", "temperature_tau"]
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_md(rows: List[Dict[str, object]], out_dir: Path) -> Path:
    path = out_dir / "calibration_parity_table.md"
    lines = [
        "# Calibration Parity Results\n\n",
        "| System | ECE | Brier | Accuracy | Temperature τ |\n",
        "|---|---:|---:|---:|---:|\n",
    ]
    for r in rows:
        tau = "-" if r["temperature_tau"] is None else f"{float(r['temperature_tau']):.4f}"
        lines.append(
            f"| {r['system']} | {float(r['ece']):.4f} | {float(r['brier_score']):.4f} | {float(r['accuracy']):.4f} | {tau} |\n"
        )
    path.write_text("".join(lines), encoding="utf-8")
    return path


def _write_tex(rows: List[Dict[str, object]], out_dir: Path) -> Path:
    path = out_dir / "calibration_parity_table.tex"
    lines = [
        "\\begin{table}[t]\n",
        "\\centering\n",
        "\\begin{tabular}{lrrrr}\n",
        "\\toprule\n",
        "System & ECE & Brier & Accuracy & Temperature $\\tau$ \\\\\n",
        "\\midrule\n",
    ]
    for r in rows:
        tau = "-" if r["temperature_tau"] is None else f"{float(r['temperature_tau']):.4f}"
        lines.append(
            f"{r['system']} & {float(r['ece']):.4f} & {float(r['brier_score']):.4f} & {float(r['accuracy']):.4f} & {tau} \\\\\n"
        )
    lines.extend(
        [
            "\\bottomrule\n",
            "\\end{tabular}\n",
            "\\caption{Calibration parity metrics across systems.}\n",
            "\\label{tab:calibration-parity}\n",
            "\\end{table}\n",
        ]
    )
    path.write_text("".join(lines), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parity-dir", default="outputs/paper/calibration_parity_real")
    parser.add_argument("--out-dir", default="outputs/paper/tables")
    args = parser.parse_args()

    parity_dir = Path(args.parity_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _collect_rows(parity_dir)
    if not rows:
        raise FileNotFoundError(f"No metrics.json files found under {parity_dir}")

    csv_path = _write_csv(rows, out_dir)
    md_path = _write_md(rows, out_dir)
    tex_path = _write_tex(rows, out_dir)

    print(f"Wrote table artifacts:\n- {csv_path}\n- {md_path}\n- {tex_path}")


if __name__ == "__main__":
    main()
