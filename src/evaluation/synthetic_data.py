"""
Deterministic synthetic data generators for evaluation.

WARNING: All data generated here is SYNTHETIC PLACEHOLDER data for engineering
validation and rapid reproducibility checks. Do NOT use for scientific claims.

For authoritative evaluation, use real CSClaimBench or FEVER datasets.

Seeding: All generators use GLOBAL_RANDOM_SEED=42 by default for determinism.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Dict, Any
import numpy as np


def generate_synthetic_csclaimbench(
    n_samples: int = 300,
    seed: int = 42,
    outpath: str | Path | None = None,
) -> List[Dict[str, Any]]:
    """
    Generate deterministic synthetic CSClaimBench-like dataset.

    ⚠️ SYNTHETIC PLACEHOLDER: Engineering use only. Not representative of real data.

    Schema matches CSClaimBench with fields:
    - doc_id: Synthetic claim ID
    - domain_topic: One of (networks, databases, algorithms, os, dist_sys)
    - source_text: Synthetic evidence snippet
    - generated_claim: Synthetic CS claim
    - gold_label: VERIFIED, REJECTED, or LOW_CONFIDENCE
    - evidence_span: Empty (placeholder)
    - prediction: Empty (placeholder)

    Args:
        n_samples: Number of records to generate
        seed: Random seed for determinism
        outpath: Optional path to write JSONL

    Returns:
        List of synthetic records (also written to outpath if provided)
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    domains = ["networks", "databases", "algorithms", "os", "dist_sys"]
    labels = ["VERIFIED", "REJECTED", "LOW_CONFIDENCE"]

    # Synthetic claim templates per domain
    templates = {
        "networks": [
            "DNS translates domain names to IP addresses",
            "TCP ensures reliable delivery",
            "UDP is connectionless",
            "IP addresses are 32-bit",
            "Routing uses forwarding tables",
        ],
        "databases": [
            "SQL INSERT adds rows to table",
            "3NF eliminates transitive dependencies",
            "ACID ensures database consistency",
            "Foreign keys enforce referential integrity",
            "Indexes improve query performance",
        ],
        "algorithms": [
            "Dijkstra finds shortest paths",
            "Merge sort is O(n log n) worst case",
            "Binary search is O(log n)",
            "Hash tables provide O(1) lookup",
            "Quicksort average case is O(n log n)",
        ],
        "os": [
            "Context switching saves CPU state",
            "Virtual memory enables overcommitment",
            "Processes are isolated",
            "Threads share memory within a process",
            "Scheduling policies determine CPU allocation",
        ],
        "dist_sys": [
            "CAP theorem prevents 3 properties simultaneously",
            "Two-phase commit ensures atomicity",
            "Consensus requires quorum",
            "Byzantine fault tolerance handles malicious nodes",
            "Eventually consistent systems converge",
        ],
    }

    records = []
    for i in range(n_samples):
        domain = domains[i % len(domains)]
        label = labels[i % len(labels)]
        claim_template = templates[domain][i % len(templates[domain])]

        # Add minor perturbation per iteration for diversity
        claim = f"[SYNTH {i}] {claim_template}"
        evidence = f"Evidence snippet for synthetic claim {i} in {domain}"

        record = {
            "doc_id": f"synth_{i:06d}",
            "domain_topic": domain,
            "source_text": evidence,
            "generated_claim": claim,
            "gold_label": label,
            "evidence_span": "",
            "prediction": "",
            "_metadata": {
                "synthetic": True,
                "placeholder": True,
                "seed": seed,
                "generator": "generate_synthetic_csclaimbench",
            },
        }
        records.append(record)

    if outpath:
        p = Path(outpath)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return records


def generate_synthetic_calibration_data(
    n_samples: int = 100,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic calibration data (confidences + binary labels).

    ⚠️ SYNTHETIC PLACEHOLDER: For unit tests only.

    Returns:
        (confidences: [0, 1], labels: binary {0, 1})
    """
    rng = np.random.RandomState(seed)

    # Generate synthetic confidences
    confidences = rng.uniform(0, 1, n_samples)

    # Generate synthetic labels with correlation to confidence
    # High confidence → more likely correct
    # Low confidence → more likely incorrect
    probs = confidences  # Probability of label=1 increases with confidence
    labels = (rng.uniform(0, 1, n_samples) < probs).astype(int)

    return confidences, labels


def generate_synthetic_fever_like(
    n_samples: int = 200,
    seed: int = 42,
    outpath: str | Path | None = None,
) -> List[Dict[str, Any]]:
    """
    Generate FEVER-schema-like synthetic data for transfer evaluation.

    ⚠️ SYNTHETIC PLACEHOLDER: For reproducibility testing only.

    Schema:
    - id: Synthetic claim ID
    - claim: Synthetic claim text
    - label: SUPPORTS, REFUTES, or NOT ENOUGH INFO
    - evidence: List of [doc_title, line_num, evidence_text]

    Args:
        n_samples: Number of records
        seed: Random seed
        outpath: Optional output path (JSONL)

    Returns:
        List of synthetic FEVER-like records
    """
    rng = random.Random(seed)

    labels = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
    claims_by_type = {
        "SUPPORTS": [
            "Paris is the capital of France",
            "Water boils at 100°C at sea level",
            "The Earth orbits the Sun",
            "Python is a programming language",
            "COVID-19 is a viral infection",
        ],
        "REFUTES": [
            "Water freezes at 50°C",
            "The Earth is flat",
            "Gravity does not exist",
            "The Sun orbits the Earth",
            "Python is a reptile only",
        ],
        "NOT ENOUGH INFO": [
            "The average weight of a penguin is 5kg",
            "Most people prefer coffee over tea",
            "Climate change will cause 2 meter sea rise by 2100",
            "Artificial sweeteners are harmful",
            "Remote work improves productivity",
        ],
    }

    records = []
    for i in range(n_samples):
        label = labels[i % len(labels)]
        claim = rng.choice(claims_by_type[label])

        record = {
            "id": i,
            "claim": f"[SYNTH {i}] {claim}",
            "label": label,
            "evidence": [
                ["synthetic_doc", 0, f"Synthetic evidence for claim {i}"]
            ],
            "_metadata": {
                "synthetic": True,
                "placeholder": True,
                "seed": seed,
                "generator": "generate_synthetic_fever_like",
            },
        }
        records.append(record)

    if outpath:
        p = Path(outpath)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return records


def generate_synthetic_extended_csclaimbench(
    n_samples: int = 560,
    seed: int = 42,
    outpath: str | Path | None = None,
) -> List[Dict[str, Any]]:
    """
    Generate extended synthetic CSClaimBench (n=560) for scalability testing.

    ⚠️ SYNTHETIC PLACEHOLDER: Deterministic stress testing only.

    Args:
        n_samples: Number of records (default 560 for extended)
        seed: Random seed
        outpath: Optional output path

    Returns:
        List of synthetic CSClaimBench records
    """
    return generate_synthetic_csclaimbench(
        n_samples=n_samples,
        seed=seed,
        outpath=outpath,
    )


__all__ = [
    "generate_synthetic_csclaimbench",
    "generate_synthetic_calibration_data",
    "generate_synthetic_fever_like",
    "generate_synthetic_extended_csclaimbench",
]
