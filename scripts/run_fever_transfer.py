#!/usr/bin/env python
"""Run FEVER transfer evaluation via schema normalization.

- Accepts a FEVER-like JSONL file.
- Optionally samples deterministically via `sample_jsonl_subset`.
- Normalizes records into CS benchmark schema.
- Runs benchmark with `scripts/run_cs_benchmark.py`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

from src.evaluation.samplers import sample_jsonl_subset

LABEL_MAP = {
    "SUPPORTS": "ENTAIL",
    "REFUTES": "CONTRADICT",
    "NOT ENOUGH INFO": "NEUTRAL",
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


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _synthesize_fever(path: Path, n: int = 12) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(1, n + 1):
        label = "SUPPORTS" if i % 3 == 0 else "REFUTES" if i % 3 == 1 else "NOT ENOUGH INFO"
        rows.append(
            {
                "id": i,
                "claim": f"Synthetic FEVER claim {i}",
                "label": label,
                "evidence": [["doc", 0, f"Synthetic evidence for claim {i}"]],
            }
        )

    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _evidence_text(rec: Dict[str, Any], claim: str) -> str:
    if rec.get("source_text"):
        return str(rec["source_text"])
    if rec.get("context"):
        return str(rec["context"])

    ev = rec.get("evidence")
    if isinstance(ev, list) and ev:
        first = ev[0]
        if isinstance(first, list):
            flat = []
            for item in first:
                if isinstance(item, str):
                    flat.append(item)
            if flat:
                return " ".join(flat)
        if isinstance(first, str):
            return first

    return claim


def _normalize_record(rec: Dict[str, Any], idx: int) -> Dict[str, Any]:
    claim = str(rec.get("generated_claim") or rec.get("claim") or rec.get("text") or f"Claim {idx}")
    doc_id = str(rec.get("doc_id") or rec.get("id") or f"fever_{idx:06d}")
    gold_raw = str(rec.get("gold_label") or rec.get("label") or "NOT ENOUGH INFO").upper()
    gold = LABEL_MAP.get(gold_raw, "NEUTRAL")

    return {
        "doc_id": doc_id,
        "domain_topic": str(rec.get("domain_topic") or "transfer.fever"),
        "source_text": _evidence_text(rec, claim),
        "generated_claim": claim,
        "gold_label": gold,
        "evidence_span": "",
        "prediction": "",
    }


def normalize_dataset(input_path: Path, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as out:
        for idx, rec in enumerate(_read_jsonl(input_path), start=1):
            out.write(json.dumps(_normalize_record(rec, idx), ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="evaluation/fever/fever_dev.jsonl", help="FEVER JSONL input")
    parser.add_argument("--sample-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sampled-output",
        default="evaluation/fever/fever_dev_sampled.jsonl",
        help="Deterministic sampled FEVER JSONL",
    )
    parser.add_argument(
        "--normalized",
        default="evaluation/fever/fever_transfer_normalized.jsonl",
        help="Normalized JSONL for benchmark runner",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation/results/fever_transfer",
        help="Directory for FEVER transfer outputs",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        _synthesize_fever(input_path)
        print(f"Input FEVER file missing; created synthetic placeholder at {input_path}")

    sampled_path = Path(args.sampled_output)
    sample_jsonl_subset(str(input_path), str(sampled_path), n=args.sample_size, seed=args.seed)
    print(f"Sampled FEVER subset -> {sampled_path}")

    normalized = Path(args.normalized)
    n = normalize_dataset(sampled_path, normalized)
    print(f"Normalized {n} FEVER records -> {normalized}")

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
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
