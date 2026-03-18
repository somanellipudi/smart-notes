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
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.evaluation.paper_contract import (
    fit_temperature_validation_nll,
    temperature_scale_logits,
)


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


def _to_logits(probs_or_logits: np.ndarray) -> np.ndarray:
    arr = np.asarray(probs_or_logits, dtype=np.float64)
    if np.all((arr >= 0.0) & (arr <= 1.0)):
        p = np.clip(arr, 1e-12, 1.0 - 1e-12)
        return np.log(p / (1.0 - p))
    return arr


def _infer_model_and_split(path: str):
    stem = Path(path).stem.replace("_result", "")
    low = stem.lower()
    if low.endswith("_validation"):
        return stem[: -len("_validation")], "validation"
    if low.endswith("_val"):
        return stem[: -len("_val")], "validation"
    if low.endswith("_test"):
        return stem[: -len("_test")], "test"
    # Default fallback keeps file explicit and avoids accidental test fitting.
    return stem, "unknown"


def _ece_from_correctness(confidences: np.ndarray, correctness: np.ndarray, n_bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        if not np.any(mask):
            continue
        acc = float(np.mean(correctness[mask]))
        c = float(np.mean(confidences[mask]))
        ece += float(np.mean(mask)) * abs(acc - c)
    return float(ece)


def run_parity(input_paths, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}

    grouped = defaultdict(dict)
    for p in input_paths:
        model, split = _infer_model_and_split(p)
        grouped[model][split] = p

    for model_name, splits in grouped.items():
        model_dir = out_dir / model_name
        model_dir.mkdir(exist_ok=True)

        if "validation" not in splits:
            raise RuntimeError(
                f"Missing validation file for model '{model_name}'. "
                "Calibration parity requires validation-only fitting."
            )

        rec_val = load_predictions(Path(splits["validation"]))
        val_logits = _to_logits(rec_val["probs"])
        fit = fit_temperature_validation_nll(val_logits.tolist(), rec_val["labels"].tolist())

        with open(model_dir / "temperature.json", "w", encoding="utf-8") as f:
            json.dump(fit, f, indent=2)

        summary[model_name] = {
            "temperature": fit,
            "validation_file": splits["validation"],
        }

        # Always emit calibrated validation predictions for traceability.
        val_scaled = temperature_scale_logits(val_logits, float(fit["best_tau"]))
        val_calibrated = []
        for i, raw in enumerate(rec_val["raw"]):
            cp = dict(raw)
            cp["calibrated_confidence"] = float(val_scaled[i])
            val_calibrated.append(cp)
        with open(model_dir / "calibrated_validation_predictions.json", "w", encoding="utf-8") as f:
            json.dump(val_calibrated, f, indent=2)

        if "test" in splits:
            rec_test = load_predictions(Path(splits["test"]))
            test_logits = _to_logits(rec_test["probs"])
            test_scaled = temperature_scale_logits(test_logits, float(fit["best_tau"]))
            test_metrics = {
                "ece": _ece_from_correctness(np.asarray(test_scaled, dtype=np.float64), rec_test["labels"]),
                "accuracy": float(np.mean((np.asarray(test_scaled) >= 0.5).astype(int) == rec_test["labels"])),
                "n_samples": int(len(rec_test["labels"])),
            }

            calibrated_preds = []
            for i, raw in enumerate(rec_test["raw"]):
                cp = dict(raw)
                cp["calibrated_confidence"] = float(test_scaled[i])
                calibrated_preds.append(cp)

            with open(model_dir / "calibrated_predictions.json", "w", encoding="utf-8") as f:
                json.dump(calibrated_preds, f, indent=2)

            with open(model_dir / "metrics.json", "w", encoding="utf-8") as f:
                json.dump(test_metrics, f, indent=2)

            summary[model_name]["test_file"] = splits["test"]
            summary[model_name]["metrics"] = test_metrics
        else:
            summary[model_name]["warning"] = "No paired test file found; only validation calibration exported."

    # Save summary
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
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
