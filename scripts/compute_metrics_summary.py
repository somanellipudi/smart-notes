#!/usr/bin/env python3
"""Compute metrics summary for the official paper run using a config file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from scripts.verify_reported_metrics import _as_table_markdown, _write_metric_macros, build_summary


def _load_simple_yaml(path: Path) -> Dict[str, str]:
    data: Dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip().strip('"').strip("'")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute metrics summary from official paper run config")
    parser.add_argument("--config", type=Path, default=Path("configs/paper_run.yaml"))
    args = parser.parse_args()

    cfg = _load_simple_yaml(args.config)

    preds_dir = Path(cfg.get("preds_dir", "artifacts/preds"))
    output_dir = Path(cfg.get("output_dir", "artifacts"))
    metrics_file = Path(cfg.get("metrics_file", str(output_dir / "metrics_summary.json")))
    model_name = cfg.get("model_name")
    predictions_file = Path(cfg.get("predictions_file", "")) if cfg.get("predictions_file") else None

    if predictions_file and not predictions_file.exists():
        raise SystemExit(f"Configured predictions_file not found: {predictions_file}")

    output_dir.mkdir(parents=True, exist_ok=True)

    summary = build_summary(preds_dir=preds_dir, primary_model=model_name)
    summary["official_paper_run"] = {
        "config": str(args.config).replace("\\", "/"),
        "split_name": cfg.get("split_name"),
        "model_name": cfg.get("model_name"),
        "seed": cfg.get("seed"),
        "predictions_file": str(predictions_file).replace("\\", "/") if predictions_file else None,
    }

    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md_path = output_dir / "metrics_summary.md"
    md_path.write_text(_as_table_markdown(summary), encoding="utf-8")

    primary = summary["primary_model"]
    p = summary["models"][primary]
    _write_metric_macros(
        Path("submission_bundle/metrics_values.tex"),
        accuracy=p["accuracy"],
        ece=p["ece"],
        auc_ac=p["auc_ac"],
    )

    print(f"[OK] Wrote {metrics_file}")
    print(f"[OK] Wrote {md_path}")
    print("[OK] Wrote submission_bundle/metrics_values.tex")
    print(f"[OK] Official run: model={primary} split={cfg.get('split_name')} seed={cfg.get('seed')}")
    print(f"[OK] Values: ECE={p['ece']:.4f} AUC-AC={p['auc_ac']:.4f}")


if __name__ == "__main__":
    main()
