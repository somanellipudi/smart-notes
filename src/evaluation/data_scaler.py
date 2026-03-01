"""Large-scale dataset orchestration utilities for evaluation runs.

Provides:
- streaming JSONL ingestion
- label-space alignment to ENTAIL/CONTRADICT/NEUTRAL
- multi-dataset merge (e.g., CSClaimBench + FEVER + SciFact)
- stratified sampling for balanced large-scale testing
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DatasetSource:
    """Configuration for one dataset input source."""

    path: str
    name: str
    label_field: str = "gold_label"
    claim_field: str = "generated_claim"
    source_text_field: str = "source_text"
    doc_id_field: str = "doc_id"
    domain_topic: Optional[str] = None


class DataScaler:
    """Orchestrate large-scale, mixed-domain benchmark datasets."""

    LABEL_MAP: Dict[str, str] = {
        "ENTAIL": "ENTAIL",
        "SUPPORTS": "ENTAIL",
        "SUPPORTED": "ENTAIL",
        "VERIFIED": "ENTAIL",
        "TRUE": "ENTAIL",
        "CONTRADICT": "CONTRADICT",
        "REFUTES": "CONTRADICT",
        "REFUTED": "CONTRADICT",
        "REJECTED": "CONTRADICT",
        "FALSE": "CONTRADICT",
        "NEUTRAL": "NEUTRAL",
        "NEI": "NEUTRAL",
        "NOT ENOUGH INFO": "NEUTRAL",
        "LOW_CONFIDENCE": "NEUTRAL",
        "UNKNOWN": "NEUTRAL",
    }

    def __init__(self, batch_size: int = 256, seed: int = 42):
        self.batch_size = batch_size
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def stream_jsonl(self, path: str | Path) -> Iterator[Dict]:
        """Yield JSON records from a JSONL file; malformed lines are skipped."""
        p = Path(path)
        with p.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed JSON line %s in %s", line_no, p)

    def to_batches(self, records: Iterable[Dict], batch_size: Optional[int] = None) -> Iterator[List[Dict]]:
        """Batch any record iterator without materializing all rows."""
        size = batch_size or self.batch_size
        batch: List[Dict] = []
        for record in records:
            batch.append(record)
            if len(batch) >= size:
                yield batch
                batch = []
        if batch:
            yield batch

    def normalize_label(self, raw_label: str | None) -> str:
        """Map arbitrary source labels into ENTAIL/CONTRADICT/NEUTRAL."""
        if raw_label is None:
            return "NEUTRAL"
        return self.LABEL_MAP.get(str(raw_label).strip().upper(), "NEUTRAL")

    def normalize_record(self, rec: Dict, source: DatasetSource, index: int) -> Dict:
        """Normalize one source-specific record to benchmark schema."""
        claim = str(
            rec.get(source.claim_field)
            or rec.get("claim")
            or rec.get("generated_claim")
            or rec.get("text")
            or ""
        ).strip()
        if not claim:
            claim = f"Synthetic claim {index}"

        source_text = str(
            rec.get(source.source_text_field)
            or rec.get("source_text")
            or rec.get("context")
            or rec.get("evidence")
            or claim
        )

        doc_id = str(
            rec.get(source.doc_id_field)
            or rec.get("doc_id")
            or rec.get("id")
            or f"{source.name}_{index:08d}"
        )

        label_raw = rec.get(source.label_field) or rec.get("gold_label") or rec.get("label")
        gold_label = self.normalize_label(label_raw)

        domain = source.domain_topic or rec.get("domain_topic") or f"transfer.{source.name.lower()}"

        return {
            "doc_id": doc_id,
            "domain_topic": str(domain),
            "source_text": source_text,
            "generated_claim": claim,
            "gold_label": gold_label,
            "evidence_span": rec.get("evidence_span", ""),
            "prediction": rec.get("prediction", ""),
            "_source_dataset": source.name,
        }

    def merge_datasets(
        self,
        sources: List[DatasetSource],
        output_path: str | Path,
        limit_per_source: Optional[int] = None,
    ) -> Dict[str, int]:
        """Merge multiple datasets into one normalized JSONL output."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        counts: Dict[str, int] = {}
        with out.open("w", encoding="utf-8") as f:
            for source in sources:
                count = 0
                for idx, rec in enumerate(self.stream_jsonl(source.path), start=1):
                    if limit_per_source is not None and count >= limit_per_source:
                        break
                    norm = self.normalize_record(rec, source, idx)
                    f.write(json.dumps(norm, ensure_ascii=False) + "\n")
                    count += 1
                counts[source.name] = count

        logger.info("Merged dataset saved to %s | counts=%s", out, counts)
        return counts

    def stratified_sample(
        self,
        input_path: str | Path,
        output_path: str | Path,
        target_n: int,
        label_key: str = "gold_label",
    ) -> Dict[str, int]:
        """Create class-balanced stratified sample from normalized JSONL."""
        rows: List[Dict] = list(self.stream_jsonl(input_path))
        if not rows:
            raise ValueError(f"No rows found in {input_path}")

        strata: Dict[str, List[Dict]] = {"ENTAIL": [], "CONTRADICT": [], "NEUTRAL": []}
        for row in rows:
            label = self.normalize_label(row.get(label_key))
            strata[label].append(row)

        per_class = max(1, target_n // 3)
        sampled: List[Dict] = []
        for label in ("ENTAIL", "CONTRADICT", "NEUTRAL"):
            bucket = strata[label]
            if not bucket:
                continue
            if len(bucket) <= per_class:
                sampled.extend(bucket)
            else:
                idx = self._rng.choice(len(bucket), size=per_class, replace=False)
                sampled.extend([bucket[i] for i in idx])

        if len(sampled) < target_n:
            remaining = [r for r in rows if r not in sampled]
            need = min(target_n - len(sampled), len(remaining))
            if need > 0:
                idx = self._rng.choice(len(remaining), size=need, replace=False)
                sampled.extend([remaining[i] for i in idx])

        self._rng.shuffle(sampled)

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for row in sampled:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        label_counts = {"ENTAIL": 0, "CONTRADICT": 0, "NEUTRAL": 0}
        for row in sampled:
            label_counts[self.normalize_label(row.get(label_key))] += 1

        logger.info(
            "Stratified sample saved to %s | n=%s | counts=%s",
            out,
            len(sampled),
            label_counts,
        )
        return label_counts
