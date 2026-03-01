#!/usr/bin/env python
"""Run CSClaimBench-Extended evaluation in a reproducible way.

This script normalizes CSClaimBench-Extended JSONL into the schema expected by
`scripts/run_cs_benchmark.py`, then runs the benchmark and writes artifacts.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

LABEL_MAP = {
    "SUPPORTED": "ENTAIL",
    "REFUTED": "CONTRADICT",
    "NEI": "NEUTRAL",
    "VERIFIED": "ENTAIL",
    "REJECTED": "CONTRADICT",
    "LOW_CONFIDENCE": "NEUTRAL",
    "ENTAIL": "ENTAIL",
    "CONTRADICT": "CONTRADICT",
    "NEUTRAL": "NEUTRAL",
}


def _normalize_record(rec: Dict[str, Any], idx: int) -> Dict[str, Any]:
    claim = rec.get("generated_claim") or rec.get("claim") or rec.get("text") or ""
    claim = str(claim).strip()
    if not claim:
        claim = f"Synthetic claim {idx}"

    doc_id = str(rec.get("doc_id") or rec.get("claim_id") or rec.get("id") or f"ext_{idx:04d}")
    domain = str(rec.get("domain_topic") or rec.get("domain") or "transfer.csclaimbench_extended")

    source_text = rec.get("source_text")
    if not source_text:
        source_text = rec.get("evidence") or rec.get("context") or claim

    gold_raw = str(rec.get("gold_label") or rec.get("label") or "NEI").upper()
    gold_label = LABEL_MAP.get(gold_raw, "NEUTRAL")

    return {
        "doc_id": doc_id,
        "domain_topic": domain,
        "source_text": str(source_text),
        "generated_claim": claim,
        "gold_label": gold_label,
        "evidence_span": rec.get("evidence_span", ""),
        "prediction": rec.get("prediction", ""),
    }


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def normalize_dataset(input_path: Path, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as out:
        for idx, rec in enumerate(_read_jsonl(input_path), start=1):
            norm = _normalize_record(rec, idx)
            out.write(json.dumps(norm, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="evaluation/cs_benchmark/csclaimbench_extended.jsonl",
        help="Input CSClaimBench-Extended JSONL",
    )
    parser.add_argument(
        "--normalized",
        default="evaluation/cs_benchmark/csclaimbench_extended_normalized.jsonl",
        help="Path for normalized benchmark JSONL",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation/results/csclaimbench_extended",
        help="Directory for evaluation artifacts",
    )
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    normalized = Path(args.normalized)
    n = normalize_dataset(input_path, normalized)
    print(f"Normalized {n} records -> {normalized}")

    cmd: List[str] = [
        sys.executable,
        "scripts/run_cs_benchmark.py",
        "--dataset",
        str(normalized),
        "--output-dir",
        args.output_dir,
        "--seed",
        str(args.seed),
    ]
    if args.sample_size is not None:
        cmd.extend(["--sample-size", str(args.sample_size)])

    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
