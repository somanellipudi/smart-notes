#!/usr/bin/env python3
"""Export canonical prediction artifacts to artifacts/preds/*.npz.

Supported inputs:
- outputs/benchmark_results/*_predictions.jsonl
  fields: pred_confidence, pred_label, gold_label
- outputs/paper/llm_comparison/predictions_*.jsonl
  fields: confidence, pred_label, gold_label
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


BINARY_POSITIVE = {"ENTAIL", "SUPPORTED", "VERIFIED", "SUPPORTS"}
BINARY_NEGATIVE = {"CONTRADICT", "REFUTED", "REJECTED", "REFUTES"}


def _norm_label(label: object) -> Optional[str]:
    if label is None:
        return None
    text = str(label).strip().upper()
    return text if text else None


def _label_to_binary(label: str) -> Optional[int]:
    if label in BINARY_POSITIVE:
        return 1
    if label in BINARY_NEGATIVE:
        return 0
    return None


def _load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _extract_prob(row: Dict) -> Optional[float]:
    if "pred_confidence" in row:
        value = float(row["pred_confidence"])
        return float(np.clip(value, 0.0, 1.0))

    if "confidence" in row and "pred_label" in row:
        conf = float(np.clip(float(row["confidence"]), 0.0, 1.0))
        pred_label = _norm_label(row.get("pred_label"))
        pred_binary = _label_to_binary(pred_label) if pred_label is not None else None
        if pred_binary is None:
            return None
        return conf if pred_binary == 1 else 1.0 - conf

    return None


def _extract_records(rows: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    pairs: List[Tuple[int, float]] = []
    for row in rows:
        gold_label = _norm_label(row.get("gold_label"))
        y_true = _label_to_binary(gold_label) if gold_label is not None else None
        if y_true is None:
            continue

        prob_supported = _extract_prob(row)
        if prob_supported is None:
            continue

        pairs.append((y_true, prob_supported))

    if not pairs:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

    y = np.asarray([p[0] for p in pairs], dtype=np.int64)
    probs = np.asarray([p[1] for p in pairs], dtype=np.float64)
    return y, probs


def export_one(input_path: Path, output_dir: Path, model_name: str) -> Optional[Path]:
    rows = _load_jsonl(input_path)
    y_true, probs = _extract_records(rows)
    if len(y_true) == 0:
        return None

    output_path = output_dir / f"{model_name}.npz"
    np.savez_compressed(
        output_path,
        y_true=y_true,
        probs=probs,
        model_name=model_name,
        split="test",
        source_file=str(input_path),
    )
    return output_path


def discover_inputs() -> List[Tuple[Path, str]]:
    discovered: List[Tuple[Path, str]] = []

    benchmark_files = sorted(Path("outputs/benchmark_results").glob("*_predictions.jsonl"))
    for index, path in enumerate(benchmark_files):
        name = "CalibraTeach" if index == 0 else f"CalibraTeach_{index+1}"
        discovered.append((path, name))

    llm_files = sorted(Path("outputs/paper/llm_comparison").glob("predictions_*.jsonl"))
    for path in llm_files:
        model = path.stem.replace("predictions_", "")
        discovered.append((path, model))

    return discovered


def main() -> None:
    parser = argparse.ArgumentParser(description="Export canonical NPZ prediction artifacts")
    parser.add_argument("--output_dir", type=Path, default=Path("artifacts/preds"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    exported: List[Path] = []
    for input_path, model_name in discover_inputs():
        result = export_one(input_path, args.output_dir, model_name)
        if result is not None:
            exported.append(result)
            print(f"[OK] {input_path} -> {result}")
        else:
            print(f"[SKIP] {input_path} (no binary rows/probabilities)")

    print(f"Exported {len(exported)} NPZ files to {args.output_dir}")


if __name__ == "__main__":
    main()
