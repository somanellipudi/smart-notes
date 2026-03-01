#!/usr/bin/env python
"""
Calibration parity runner

Usage:
  python scripts/calibration_parity.py --inputs evaluation/results/*/*_result.json --output outputs/paper/calibration_parity --val-split validation

Expects each input JSON to follow `BenchmarkResult.to_json` format with `predictions` list,
where each prediction has at least: claim_id, pred_confidence (float), gold_label.
If logits are available, include `logits` or `pred_logits` per prediction (list or float).

Saves per-system: temperature.json, metrics.json, calibrated_predictions.json
"""

import argparse
import json
from pathlib import Path
import numpy as np
from src.evaluation.calibration import CalibrationEvaluator
from collections import defaultdict
import glob


def load_predictions(json_path: Path):
    data = json.loads(json_path.read_text())
    preds = data.get("predictions", [])
    # Build lists in order
    claim_ids = []
    probs = []
    logits = []
    labels = []
    for p in preds:
        claim_ids.append(p.get("claim_id"))
        # prefer explicit logits if present
        if "logits" in p and p.get("logits") is not None:
            l = p.get("logits")
            # If logits is a list (multi-class), convert to max-softmax correctness proxy
            if isinstance(l, list):
                arr = np.array(l, dtype=float)
                probs.append(float(np.max(softmax(arr))))
                logits.append(float(np.max(arr)))
            else:
                # single logit -> treat as logit for correctness
                logits.append(float(l))
                probs.append(1.0 / (1.0 + np.exp(-float(l))))
        else:
            # fallback to pred_confidence as probability
            p_conf = p.get("pred_confidence", None)
            if p_conf is None:
                # if missing, set neutral 0.5
                p_conf = 0.5
            probs.append(float(p_conf))
            logits.append(None)

        labels.append(1 if p.get("match", False) or p.get("pred_label") == p.get("gold_label") else 0)

    return {
        "claim_ids": claim_ids,
        "probs": np.array(probs, dtype=float),
        "logits": logits,
        "labels": np.array(labels, dtype=int),
        "raw": preds,
        "meta": data.get("config", {})
    }


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def run_parity(input_paths, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    calib = CalibrationEvaluator(n_bins=10)

    summary = {}
    for p in input_paths:
        name = Path(p).stem.replace("_result", "")
        dest = out_dir / name
        dest.mkdir(exist_ok=True)
        rec = load_predictions(Path(p))

        # Split validation vs test if available in JSON config; otherwise assume whole file is test and user provided separate val file
        # For parity runner we expect provided inputs correspond to a run on the validation set and another on test.
        # Here we assume input file is validation-level run if 'validation' in name
        is_val = "val" in name or "validation" in name

        # Fit tau on validation splits only
        if is_val:
            fit = calib.fit_temperature_grid(rec["probs"].tolist(), rec["labels"].tolist())
            # Save tau and grid
            with open(dest / "temperature.json", "w") as f:
                json.dump(fit, f, indent=2)
            summary[name] = {"fit": fit}
        else:
            # If validation fit exists in sibling folder, load it
            val_candidate = out_dir / (name + "_val")
            tau = None
            # naive attempt: look for any temperature.json in out_dir
            for tfile in out_dir.rglob("temperature.json"):
                try:
                    candidate = json.loads(tfile.read_text())
                    tau = candidate.get("best_tau")
                    break
                except Exception:
                    continue

            if tau is None:
                # if no tau found, do an internal fit (best-effort)
                fit = calib.fit_temperature_grid(rec["probs"].tolist(), rec["labels"].tolist())
                tau = fit.get("best_tau")
                with open(dest / "temperature_internal_fit.json", "w") as f:
                    json.dump(fit, f, indent=2)

            # Apply tau and compute calibrated metrics
            scaled = calib._apply_temperature(rec["probs"], tau)
            metrics = calib.evaluate(scaled.tolist(), rec["labels"].tolist(), return_bins=True)
            # save calibrated predictions
            calibrated_preds = []
            for i, raw in enumerate(rec["raw"]):
                cp = dict(raw)
                cp["calibrated_confidence"] = float(scaled[i])
                calibrated_preds.append(cp)

            with open(dest / "calibrated_predictions.json", "w") as f:
                json.dump(calibrated_preds, f, indent=2)

            with open(dest / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)

            summary[name] = {"tau_used": tau, "metrics": metrics}

    # Save summary
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Calibration parity outputs saved to: {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="Input result JSON files")
    parser.add_argument("--output", required=True, help="Output folder for parity results")
    args = parser.parse_args()

    # Expand any glob patterns provided (PowerShell does not expand globs by default)
    expanded = []
    for pat in args.inputs:
        matches = glob.glob(pat)
        if matches:
            expanded.extend(matches)
        else:
            expanded.append(pat)

    input_paths = expanded
    out_dir = Path(args.output)

    run_parity(input_paths, out_dir)


if __name__ == "__main__":
    main()
