"""Deterministic keyword-overlap retrieval module for leakage scan reproducibility."""

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class Index:
    docs: Dict[str, str]
    token_sets: Dict[str, set]
    manifest: Dict


def _tokenize(text: str) -> List[str]:
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return [tok for tok in text.split() if tok]


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _extract_text(record: Dict) -> Optional[str]:
    for key in ("text", "passage", "source_text", "content", "body"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _load_corpus_records(corpus_path: Path) -> List[Dict]:
    suffix = corpus_path.suffix.lower()
    records: List[Dict] = []

    if suffix == ".jsonl":
        with corpus_path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    records.append(obj)
                elif isinstance(obj, str):
                    records.append({"text": obj})
    elif suffix == ".json":
        with corpus_path.open("r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    records.append(item)
                elif isinstance(item, str):
                    records.append({"text": item})
        elif isinstance(data, dict):
            for key in ("docs", "documents", "items", "data"):
                if isinstance(data.get(key), list):
                    for item in data[key]:
                        if isinstance(item, dict):
                            records.append(item)
                        elif isinstance(item, str):
                            records.append({"text": item})
                    break
            else:
                records.append(data)
    elif suffix == ".csv":
        with corpus_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                records.append(dict(row))
    elif suffix == ".txt":
        with corpus_path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append({"text": line})
    else:
        raise ValueError(f"Unsupported corpus format: {suffix}")

    return records


def build_index(corpus_path: str, outdir: str = "artifacts/retrieval_index") -> None:
    corpus_file = Path(corpus_path)
    if not corpus_file.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    records = _load_corpus_records(corpus_file)

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    docs_path = out_path / "docs.jsonl"
    token_sets_path = out_path / "token_sets.jsonl"
    manifest_path = out_path / "corpus_manifest.json"

    docs_written = 0
    seen_doc_ids: Dict[str, int] = {}

    with docs_path.open("w", encoding="utf-8") as docs_handle, token_sets_path.open("w", encoding="utf-8") as token_handle:
        for idx, record in enumerate(records):
            text = _extract_text(record)
            if not text:
                continue

            base_doc_id = str(record.get("doc_id") or f"doc_{idx:06d}")
            dup_count = seen_doc_ids.get(base_doc_id, 0)
            seen_doc_ids[base_doc_id] = dup_count + 1
            doc_id = base_doc_id if dup_count == 0 else f"{base_doc_id}__{dup_count}"

            source_type = str(record.get("source_type") or record.get("domain_topic") or "unknown")
            token_set = sorted(set(_tokenize(text)))

            docs_handle.write(json.dumps({
                "doc_id": doc_id,
                "text": text,
                "source_type": source_type,
            }, ensure_ascii=False) + "\n")
            token_handle.write(json.dumps({
                "doc_id": doc_id,
                "tokens": token_set,
            }, ensure_ascii=False) + "\n")
            docs_written += 1

    if docs_written == 0:
        raise ValueError("No valid text documents found in corpus")

    manifest = {
        "corpus_path": str(corpus_file.resolve()),
        "corpus_size_bytes": corpus_file.stat().st_size,
        "corpus_sha256": _sha256_file(corpus_file),
        "num_docs": docs_written,
        "index_version": 1,
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)


def load_index(outdir: str = "artifacts/retrieval_index") -> Index:
    out_path = Path(outdir)
    docs_path = out_path / "docs.jsonl"
    token_sets_path = out_path / "token_sets.jsonl"
    manifest_path = out_path / "corpus_manifest.json"

    missing = [str(p) for p in (docs_path, token_sets_path, manifest_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Retrieval index missing required files: " + ", ".join(missing)
        )

    docs: Dict[str, str] = {}
    with docs_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            docs[str(rec["doc_id"])] = str(rec["text"])

    token_sets: Dict[str, set] = {}
    with token_sets_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            token_sets[str(rec["doc_id"])] = set(rec.get("tokens", []))

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    if not docs:
        raise ValueError("Retrieval index is empty (no docs)")

    return Index(docs=docs, token_sets=token_sets, manifest=manifest)


def retrieve_passages(claim: str, k: int, index: Index) -> List[Dict]:
    if k <= 0:
        raise ValueError("k must be > 0")

    claim_tokens = set(_tokenize(claim))
    claim_len = max(1, len(claim_tokens))

    scored: List[Tuple[float, str]] = []
    for doc_id, doc_tokens in index.token_sets.items():
        overlap = len(claim_tokens & doc_tokens)
        overlap_recall = overlap / claim_len
        scored.append((overlap_recall, doc_id))

    scored.sort(key=lambda item: (-item[0], item[1]))

    results: List[Dict] = []
    for rank, (score, doc_id) in enumerate(scored[:k], start=1):
        results.append(
            {
                "doc_id": doc_id,
                "passage": index.docs.get(doc_id, ""),
                "score": float(round(score, 6)),
                "rank": rank,
            }
        )

    return results
