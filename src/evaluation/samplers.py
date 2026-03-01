"""Deterministic sampling utilities for datasets (JSONL)."""
from pathlib import Path
import json
import random
from typing import List


def sample_jsonl_subset(input_path: str, output_path: str, n: int, seed: int = 42) -> List[str]:
    """
    Deterministically sample `n` records from a JSONL file and write to output_path.

    Returns list of written lines (as JSON strings).
    """
    p_in = Path(input_path)
    p_out = Path(output_path)
    p_out.parent.mkdir(parents=True, exist_ok=True)

    with p_in.open("r", encoding="utf-8") as f:
        lines = [l.rstrip("\n") for l in f if l.strip()]

    rnd = random.Random(seed)
    indices = list(range(len(lines)))
    rnd.shuffle(indices)
    selected = indices[:n]

    written = []
    with p_out.open("w", encoding="utf-8") as out:
        for i in selected:
            out.write(lines[i] + "\n")
            written.append(lines[i])

    return written
