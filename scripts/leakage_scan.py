#!/usr/bin/env python3
"""
Automated Lexical Leakage Analysis across claims vs retrieved evidence passages.

Computes three lexical overlap metrics (LCO, LCS, SUBSTRING) for each claim against
its top-k retrieved passages, producing summary statistics and per-claim reports.

This addresses reviewer concern about limited manual leakage checks.

Supports three retrieval modes:
- smoke: Synthetic claims and passages (for testing)
- mock: Real claims with deterministic dummy passages (for unit tests)
- real: Real claims and real retrieved passages (for paper-usable results)
"""

import argparse
import json
import csv
import re
import sys
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
import statistics


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ============================================================================
# Tokenization
# ============================================================================

def tokenize(text: str) -> List[str]:
    """
    Deterministic tokenizer: lowercase, strip punctuation, split on whitespace.
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = text.split()
    return tokens


# ============================================================================
# Leakage Metrics
# ============================================================================

def longest_consecutive_overlap(claim_tokens: List[str], passage_tokens: List[str]) -> float:
    """
    LCO ratio: longest contiguous matching token sequence length / len(claim_tokens).
    """
    if not claim_tokens:
        raise ValueError("claim_tokens cannot be empty")
    
    max_overlap_len = 0
    
    for start in range(len(claim_tokens)):
        for end in range(start + 1, len(claim_tokens) + 1):
            subseq = claim_tokens[start:end]
            subseq_str = ' '.join(subseq)
            passage_str = ' '.join(passage_tokens)
            
            if subseq_str in passage_str:
                max_overlap_len = max(max_overlap_len, len(subseq))
    
    return max_overlap_len / len(claim_tokens)


def longest_common_subsequence_length(claim_tokens: List[str], passage_tokens: List[str]) -> int:
    """Compute the length of the longest common subsequence (LCS) using dynamic programming."""
    m, n = len(claim_tokens), len(passage_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if claim_tokens[i - 1] == passage_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    return dp[m][n]


def longest_common_subsequence_ratio(claim_tokens: List[str], passage_tokens: List[str]) -> float:
    """LCS ratio: LCS token length / len(claim_tokens)."""
    if not claim_tokens:
        raise ValueError("claim_tokens cannot be empty")
    
    lcs_len = longest_common_subsequence_length(claim_tokens, passage_tokens)
    return lcs_len / len(claim_tokens)


def longest_common_substring_ratio(claim_tokens: List[str], passage_tokens: List[str]) -> float:
    """SUBSTRING ratio: longest common contiguous token substring length / len(claim_tokens)."""
    if not claim_tokens:
        raise ValueError("claim_tokens cannot be empty")
    
    max_substring_len = 0
    
    for start in range(len(claim_tokens)):
        for end in range(start + 1, len(claim_tokens) + 1):
            subseq = claim_tokens[start:end]
            
            for p_start in range(len(passage_tokens) - len(subseq) + 1):
                if passage_tokens[p_start:p_start + len(subseq)] == subseq:
                    max_substring_len = max(max_substring_len, len(subseq))
                    break
    
    return max_substring_len / len(claim_tokens)


# ============================================================================
# Data Loading
# ============================================================================

def auto_detect_claims_path() -> Optional[str]:
    """
    Auto-detect claims file from known repo paths.
    
    Searches in order:
    - data/CSClaimBench/*.jsonl or *.json
    - data/claims/*.jsonl or *.json
    - artifacts/datasets/*claims*.jsonl
    - data/*.jsonl or *.json
    
    Returns the first file found, or None if none exist.
    """
    candidates = [
        ("data/CSClaimBench", ["claims"]),
        ("data/claims", []),
        ("artifacts/datasets", ["claims"]),
        ("data", ["claims"]),
    ]
    
    for cand_dir, keywords in candidates:
        cand_path = Path(cand_dir)
        if not cand_path.exists():
            continue
        
        # Look for JSONL first
        for jsonl_file in cand_path.glob("**/*.jsonl"):
            if not keywords or any(kw in jsonl_file.name.lower() for kw in keywords):
                return str(jsonl_file.resolve())
        
        # Then JSON
        for json_file in cand_path.glob("**/*.json"):
            if "report" not in json_file.name.lower():
                if not keywords or any(kw in json_file.name.lower() for kw in keywords):
                    return str(json_file.resolve())
    
    return None


def load_claims_from_file(filepath: str, file_format: str = "auto") -> List[Dict]:
    """
    Load claims from a file (JSONL, JSON, or CSV).
    
    Args:
        filepath: Path to claims file
        file_format: "auto" (auto-detect), "jsonl", "json", or "csv"
    
    Returns:
        List of claim dicts with at least "text" field
    
    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file format is unsupported
    """
    fpath = Path(filepath)
    
    if not fpath.exists():
        raise FileNotFoundError(f"Claims file not found: {filepath}")
    
    # Auto-detect format
    if file_format == "auto":
        ext = fpath.suffix.lower()
        if ext == ".jsonl":
            file_format = "jsonl"
        elif ext == ".json":
            file_format = "json"
        elif ext == ".csv":
            file_format = "csv"
        else:
            raise ValueError(f"Cannot auto-detect format for file: {filepath}")
    
    claims = []
    
    if file_format == "jsonl":
        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    claims.append(obj)
                except json.JSONDecodeError:
                    continue
    
    elif file_format == "json":
        with open(fpath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            claims = data
        elif isinstance(data, dict):
            for key in ["claims", "data", "items"]:
                if key in data and isinstance(data[key], list):
                    claims = data[key]
                    break
            if not claims:
                claims = [data]
        else:
            raise ValueError(f"JSON must be list or dict, got {type(data)}")
    
    elif file_format == "csv":
        import csv as csv_module
        with open(fpath, 'r', encoding='utf-8') as f:
            reader = csv_module.DictReader(f)
            for row in reader:
                claims.append(row)
    
    else:
        raise ValueError(f"Unsupported file format: {file_format}")
    
    # Normalize: ensure each claim has "text" field
    normalized = []
    for idx, claim in enumerate(claims):
        if isinstance(claim, str):
            normalized.append({"text": claim, "idx": idx})
        elif isinstance(claim, dict):
            if "text" not in claim and "claim" in claim:
                claim["text"] = claim["claim"]
            elif "text" not in claim and "title" in claim:
                claim["text"] = claim["title"]
            normalized.append(claim)
        else:
            continue
    
    return normalized


# ============================================================================
# Retrieval Modes
# ============================================================================

def get_synthetic_claims_passages() -> Tuple[List[Dict], List[Dict]]:
    """Synthetic claims and passages for --smoke mode."""
    claims = [
        {"id": "synthetic_0", "text": "Paris is the capital of France"},
        {"id": "synthetic_1", "text": "The Earth orbits the Sun"},
        {"id": "synthetic_2", "text": "Water boils at 100 degrees Celsius"},
    ]
    
    passages = [
        {"doc_id": "doc_1", "text": "Paris is a major city in France"},
        {"doc_id": "doc_2", "text": "The capital of France is Paris"},
        {"doc_id": "doc_3", "text": "Our planet Earth orbits the Sun in an elliptical path"},
        {"doc_id": "doc_4", "text": "Water freezes at 0 and boils at 100 degrees Celsius"},
        {"doc_id": "doc_5", "text": "Common knowledge about planets and celestial bodies"},
        {"doc_id": "doc_6", "text": "Properties of water and other liquids in chemistry"},
    ]
    
    return claims, passages


def get_mock_passages(n_claims: int, n_passages_per_claim: int = 6) -> List[Dict]:
    """Generate deterministic mock passages for testing (--retrieval_mode mock)."""
    mock_corpus = [
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks with multiple layers",
        "Natural language processing focuses on text analysis",
        "Computer vision enables image recognition and classification",
        "Reinforcement learning involves an agent learning from rewards",
        "Supervised learning requires labeled training data",
        "Unsupervised learning discovers patterns in unlabeled data",
        "Transfer learning reuses knowledge from one domain to another",
        "Overfitting occurs when a model memorizes training data",
        "Regularization techniques prevent overfitting in models",
    ]
    
    passages = []
    for i in range(n_passages_per_claim):
        passages.append({
            "doc_id": f"mock_doc_{i}",
            "text": mock_corpus[i % len(mock_corpus)]
        })
    
    return passages


def get_real_passages(claim_text: str, k_max: int, index: Any) -> List[Dict]:
    """Retrieve real passages for a single claim from a preloaded index."""
    try:
        from src.eval import retrieval_module
    except ImportError as exc:
        raise RuntimeError("[ERROR] Cannot import retrieval module for real mode") from exc

    rows = retrieval_module.retrieve_passages(claim=claim_text, k=k_max, index=index)
    passages: List[Dict] = []
    for row in rows:
        passages.append(
            {
                "doc_id": row.get("doc_id", "unknown"),
                "text": row.get("passage", ""),
            }
        )
    return passages


# ============================================================================
# Analysis
# ============================================================================

def generate_claim_id(claim: str, idx: int) -> str:
    """Deterministic claim ID generation using sha1 hash."""
    try:
        hash_val = hashlib.sha1(claim.encode('utf-8')).hexdigest()[:8]
        return f"claim_{idx}_{hash_val}"
    except Exception:
        return f"claim_{idx}"


def compute_leakage_metrics(claims: List[Dict], passages: List[Dict], k_values: List[int]) -> Tuple[Dict, List[Dict]]:
    """Compute leakage metrics for each claim against top-k passages."""
    per_claim = []
    metrics_by_name = defaultdict(list)
    claim_max_metrics = defaultdict(lambda: defaultdict(float))
    
    for claim_idx, claim in enumerate(claims):
        claim_text = claim.get("text", claim)
        claim_id = claim.get("id") or generate_claim_id(claim_text, claim_idx)
        claim_tokens = tokenize(claim_text)
        
        if not claim_tokens:
            continue
        
        for k in k_values:
            top_k_passages = passages[:k]
            
            max_lco = 0.0
            max_lcs = 0.0
            max_substring = 0.0
            max_source = None
            max_metric_type = None
            
            for rank, passage in enumerate(top_k_passages, start=1):
                passage_text = passage.get("text", passage)
                passage_tokens = tokenize(passage_text)
                
                try:
                    lco = longest_consecutive_overlap(claim_tokens, passage_tokens)
                    lcs = longest_common_subsequence_ratio(claim_tokens, passage_tokens)
                    substring = longest_common_substring_ratio(claim_tokens, passage_tokens)
                except ValueError:
                    continue
                
                if lco > max_lco:
                    max_lco = lco
                    max_metric_type = "lco"
                    max_source = {
                        "passage": passage_text[:200],
                        "doc_id": passage.get("doc_id", f"doc_{rank}"),
                        "rank": rank
                    }
                
                if lcs > max_lcs:
                    max_lcs = lcs
                    if max_metric_type != "lco":
                        max_metric_type = "lcs"
                        max_source = {
                            "passage": passage_text[:200],
                            "doc_id": passage.get("doc_id", f"doc_{rank}"),
                            "rank": rank
                        }
                
                if substring > max_substring:
                    max_substring = substring
                    if max_metric_type not in ["lco", "lcs"]:
                        max_metric_type = "substring"
                        max_source = {
                            "passage": passage_text[:200],
                            "doc_id": passage.get("doc_id", f"doc_{rank}"),
                            "rank": rank
                        }
            
            record = {
                "claim_id": claim_id,
                "claim": claim_text[:500],
                "k": k,
                "max_lco": round(max_lco, 6),
                "max_lcs": round(max_lcs, 6),
                "max_substring": round(max_substring, 6),
                "max_source": max_source or {"passage": "", "doc_id": "none", "rank": 0}
            }
            per_claim.append(record)
            
            metrics_by_name["lco"].append(max_lco)
            metrics_by_name["lcs"].append(max_lcs)
            metrics_by_name["substring"].append(max_substring)
            
            claim_max_metrics[claim_id]["lco"] = max(claim_max_metrics[claim_id]["lco"], max_lco)
            claim_max_metrics[claim_id]["lcs"] = max(claim_max_metrics[claim_id]["lcs"], max_lcs)
            claim_max_metrics[claim_id]["substring"] = max(claim_max_metrics[claim_id]["substring"], max_substring)
    
    per_claim.sort(key=lambda x: (x["claim_id"], x["k"]))
    
    claim_level_metrics = defaultdict(list)
    for claim_id, metrics in claim_max_metrics.items():
        claim_level_metrics["lco"].append(metrics["lco"])
        claim_level_metrics["lcs"].append(metrics["lcs"])
        claim_level_metrics["substring"].append(metrics["substring"])
    
    summary = {}
    for metric_name in ["lco", "lcs", "substring"]:
        row_values = metrics_by_name[metric_name]
        claim_values = claim_level_metrics[metric_name]
        
        if not row_values:
            summary[metric_name] = {
                "max": 0.0, "p95": 0.0, "mean": 0.0,
                "row_count_ge_0.10": 0, "row_count_ge_0.15": 0, "row_count_ge_0.20": 0,
                "claim_count_ge_0.10": 0, "claim_count_ge_0.15": 0, "claim_count_ge_0.20": 0
            }
        else:
            summary[metric_name] = {
                "max": round(max(row_values), 6),
                "p95": round(statistics.quantiles(row_values, n=20)[18] if len(row_values) > 1 else max(row_values), 6),
                "mean": round(statistics.mean(row_values), 6),
                "row_count_ge_0.10": sum(1 for v in row_values if v >= 0.10),
                "row_count_ge_0.15": sum(1 for v in row_values if v >= 0.15),
                "row_count_ge_0.20": sum(1 for v in row_values if v >= 0.20),
                "claim_count_ge_0.10": sum(1 for v in claim_values if v >= 0.10) if claim_values else 0,
                "claim_count_ge_0.15": sum(1 for v in claim_values if v >= 0.15) if claim_values else 0,
                "claim_count_ge_0.20": sum(1 for v in claim_values if v >= 0.20) if claim_values else 0,
            }
    
    return summary, per_claim


# ============================================================================
# Output
# ============================================================================

def write_json_report(output_file: Path, run_id: str, seed: int, k: int, k2: int,
                      claims_path: str, retrieval_mode: str, summary: Dict, per_claim: List[Dict],
                      corpus_path: Optional[str] = None, retrieval_manifest: Optional[Dict] = None) -> None:
    """Write JSON leakage report with canonical schema."""
    report = {
        "run_id": run_id,
        "seed": seed,
        "k": k,
        "k2": k2,
        "k_values": [k, k2],
        "claims_path": claims_path,
        "retrieval_mode": retrieval_mode,
        "corpus_path": corpus_path,
        "retrieval_manifest": retrieval_manifest,
        "n_claims_scanned": len(set(c["claim_id"] for c in per_claim)),
        "claim_count_definition": "max_over_k",
        "thresholds": [0.10, 0.15, 0.20],
        "summary": summary,
        "per_claim": per_claim
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def write_csv_report(output_file: Path, per_claim: List[Dict]) -> None:
    """Write CSV leakage report."""
    fieldnames = ["claim_id", "k", "max_lco", "max_lcs", "max_substring", "doc_id", "rank"]
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for record in per_claim:
            row = {
                "claim_id": record["claim_id"],
                "k": record["k"],
                "max_lco": record["max_lco"],
                "max_lcs": record["max_lcs"],
                "max_substring": record["max_substring"],
                "doc_id": record["max_source"].get("doc_id", ""),
                "rank": record["max_source"].get("rank", 0)
            }
            writer.writerow(row)


def write_readme(output_dir: Path) -> None:
    """Write README.md explaining leakage analysis."""
    readme_content = """# Lexical Leakage Analysis Report

## Overview

Automated lexical leakage scan across claims versus retrieved evidence passages. The analysis addresses reviewer concerns about limited manual leakage checks.

## Important Caveat

**Lexical Overlap ≠ Semantic Leakage**: Token-level analysis misses semantic overlap. High overlap may indicate common terminology, benign phrasing, or domain knowledge reuse.

## Metrics

Three lexical overlap metrics (range [0,1]):
- **LCO**: Longest Consecutive Overlap ratio
- **LCS**: Longest Common Subsequence ratio  
- **SUBSTRING**: Longest common contiguous token substring ratio

## Thresholds

Summary statistics report counts exceeding standard thresholds:
- **≥ 0.10**: Potential overlaps (10% of claim tokens)
- **≥ 0.15**: Moderate overlaps (15% of claim tokens)
- **≥ 0.20**: High overlaps (20% of claim tokens)

## Summary Statistics: Row vs Claim Counts

- **Row counts** (row_count_ge_*): Across all claim×k rows
- **Claim counts** (claim_count_ge_*): Across unique claims (max-over-k)

## Retrieval Modes

### smoke (Synthetic - Testing Only)
- Data: Embedded synthetic claims and passages
- Paper-usable: ❌ NO
- Command: `python scripts/leakage_scan.py --smoke`

### mock (Real Claims + Dummy Passages - Tests Only)
- Data: Real claims with deterministic mock passages
- Paper-usable: ❌ NO  
- Command: `python scripts/leakage_scan.py --claims <path> --retrieval_mode mock`
- Note: `retrieval_mode: "mock"` marked in output JSON

### real (Paper-Usable - Real Data + Real Passages)
- Data: Real claims and real retrieved passages
- Paper-usable: ✅ YES (if retrieval available)
- Command: `python scripts/leakage_scan.py --claims <path> --retrieval_mode real --corpus <path>`
- Prerequisite: Requires corpus index and retrieval module

### Retrieval Index Setup (Real Mode)

Build retrieval index explicitly:
```bash
python scripts/build_retrieval_index.py --corpus <path>
```

Then run real leakage scan:
```bash
python scripts/leakage_scan.py --claims <claims> --retrieval_mode real --outdir artifacts/leakage --k 5 --k2 15
```

Or build-on-run with `--corpus`:
```bash
python scripts/leakage_scan.py --claims <claims> --retrieval_mode real --corpus <path> --outdir artifacts/leakage --k 5 --k2 15
```

Fail-fast behavior: real mode exits with `[ERROR]` if no retrieval index exists and no `--corpus` is provided.

## Data Loading

Supported: JSONL, JSON, CSV with "claim" or "text" field.

Auto-detection searches:
- data/CSClaimBench/*.jsonl or *.json
- data/claims/*.jsonl or *.json
- artifacts/datasets/*claims*.jsonl
- data/*.jsonl or *.json

## Examples

**Mock mode (testing):**
```bash
python scripts/leakage_scan.py --claims data/my_claims.jsonl --retrieval_mode mock --max_claims 10
```

**Smoke mode (testing):**
```bash
python scripts/leakage_scan.py --smoke
```

**Real mode (paper-usable - requires retrieval setup):**
```bash
python scripts/leakage_scan.py --claims data/my_claims.jsonl --retrieval_mode real --corpus evaluation/cs_benchmark/csclaimbench_v1.jsonl --k 5 --k2 15
```

## Key Distinctions

| Aspect | smoke | mock | real |
|--------|-------|------|------|
| Data | Synthetic | Real claims + mock | Real claims + real |
| Paper-usable | ❌ NO | ❌ NO | ✅ YES |
| Use | Testing | Unit tests | Publication |

## Outputs

1. **leakage_report.json**: Complete results with `retrieval_mode` field
2. **leakage_report.csv**: Tabular format (one row per claim × k)
3. **README.md**: This file

## Limitations

- Lexical only (misses semantic overlap)
- Simple tokenization (lowercase + whitespace)
- Fixed top-k only
- Doesn't account for domain terminology
- Passages truncated to 200 chars in JSON

## Retrieval Method Note

Real-mode leakage scan uses deterministic keyword-overlap retrieval for reproducibility and stable auditing. Ranking quality is not the goal of this component.
"""
    
    readme_file = output_dir / "README.md"
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write(readme_content)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Automated lexical leakage analysis (smoke/mock/real modes)"
    )
    
    parser.add_argument("--claims", type=str, default=None,
                        help="Path to claims file (auto-detect if omitted; required for non-smoke if auto-detect fails)")
    parser.add_argument("--format", type=str, default="auto", choices=["auto", "jsonl", "json", "csv"],
                        help="Claims file format (default: auto-detect)")
    parser.add_argument("--k", type=int, default=5, help="Min top-k (default: 5)")
    parser.add_argument("--k2", type=int, default=15, help="Max top-k (default: 15)")
    parser.add_argument("--outdir", type=str, default="artifacts/leakage", help="Output directory")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument("--max_claims", type=int, default=None, help="Limit claims to scan")
    parser.add_argument("--smoke", action="store_true", default=False,
                        help="Run in smoke mode (synthetic data, testing only)")
    parser.add_argument("--retrieval_mode", type=str, default="real", choices=["smoke", "mock", "real"],
                        help="Retrieval mode: smoke (synthetic), mock (dummy), real (actual)")
    parser.add_argument("--corpus", type=str, default=None,
                        help="Path to corpus file (used to build retrieval index for --retrieval_mode real)")
    parser.add_argument("--fail_on_threshold", type=float, default=None,
                        help="Exit code 1 if max metric exceeds threshold")
    
    args = parser.parse_args()
    
    # Determine retrieval mode
    if args.smoke:
        retrieval_mode = "smoke"
    else:
        retrieval_mode = args.retrieval_mode
    
    corpus_path = None
    retrieval_manifest = None

    # Load claims and passages
    if args.smoke:
        claims, passages = get_synthetic_claims_passages()
        claims_path = "synthetic::smoke_mode"
    else:
        claims_path = args.claims or auto_detect_claims_path()
        
        if not claims_path:
            error_msg = """[ERROR] No claims file found and --claims not specified.

Searched paths:
  - data/CSClaimBench/*.jsonl or *.json
  - data/claims/*.jsonl or *.json
  - artifacts/datasets/*claims*.jsonl
  - data/*.jsonl or *.json

To proceed:
  python scripts/leakage_scan.py --claims <path> --outdir artifacts/leakage
or use smoke mode:
  python scripts/leakage_scan.py --smoke --outdir artifacts/leakage
"""
            print(error_msg, file=sys.stderr)
            return 2
        
        try:
            claims = load_claims_from_file(claims_path, file_format=args.format)
        except (FileNotFoundError, ValueError) as e:
            print(f"[ERROR] Failed to load claims: {e}", file=sys.stderr)
            return 2
        
        if not claims:
            print(f"[ERROR] No claims loaded from {claims_path}", file=sys.stderr)
            return 2

        if args.max_claims:
            claims = claims[:args.max_claims]
        
        # Load passages based on retrieval mode
        if retrieval_mode == "smoke":
            passages, _ = get_synthetic_claims_passages()
        elif retrieval_mode == "mock":
            passages = get_mock_passages(len(claims))
        elif retrieval_mode == "real":
            try:
                from src.eval import retrieval_module
            except ImportError:
                print("[ERROR] Cannot import src.eval.retrieval_module required for --retrieval_mode real", file=sys.stderr)
                return 2

            index_outdir = "artifacts/retrieval_index"
            if args.corpus:
                try:
                    retrieval_module.build_index(corpus_path=args.corpus, outdir=index_outdir)
                except Exception as e:
                    print(f"[ERROR] Failed to build retrieval index from --corpus: {e}", file=sys.stderr)
                    return 2

            try:
                index = retrieval_module.load_index(outdir=index_outdir)
            except Exception as e:
                print(
                    "[ERROR] Real retrieval requires an index at artifacts/retrieval_index or --corpus <path> to build one.\n"
                    f"Details: {e}",
                    file=sys.stderr,
                )
                return 2

            corpus_path = index.manifest.get("corpus_path")
            retrieval_manifest = index.manifest

            for claim in claims:
                claim_text = claim.get("text", "")
                claim["__retrieved_passages"] = get_real_passages(claim_text=claim_text, k_max=args.k2, index=index)

            passages = []
        else:
            print(f"[ERROR] Unknown retrieval mode: {retrieval_mode}", file=sys.stderr)
            return 2
    
    # Create output directory
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.max_claims and retrieval_mode == "smoke":
        claims = claims[:args.max_claims]
    
    k_values = list(range(args.k, args.k2 + 1))
    if retrieval_mode == "real":
        claim_specific_passages = [
            {
                "doc_id": c.get("id", f"doc_{idx}"),
                "text": c.get("text", "")
            }
            for idx, c in enumerate(claims)
        ]
        # compute_leakage_metrics expects a passage list, so process per claim explicitly.
        per_claim = []
        metrics_by_name = defaultdict(list)
        claim_max_metrics = defaultdict(lambda: defaultdict(float))

        for claim_idx, claim in enumerate(claims):
            claim_text = claim.get("text", claim)
            claim_id = claim.get("id") or generate_claim_id(claim_text, claim_idx)
            claim_tokens = tokenize(claim_text)

            if not claim_tokens:
                continue

            claim_passages = claim.get("__retrieved_passages", [])

            for k in k_values:
                top_k_passages = claim_passages[:k]
                max_lco = 0.0
                max_lcs = 0.0
                max_substring = 0.0
                max_source = None
                max_metric_type = None

                for rank, passage in enumerate(top_k_passages, start=1):
                    passage_text = passage.get("text", passage)
                    passage_tokens = tokenize(passage_text)

                    try:
                        lco = longest_consecutive_overlap(claim_tokens, passage_tokens)
                        lcs = longest_common_subsequence_ratio(claim_tokens, passage_tokens)
                        substring = longest_common_substring_ratio(claim_tokens, passage_tokens)
                    except ValueError:
                        continue

                    if lco > max_lco:
                        max_lco = lco
                        max_metric_type = "lco"
                        max_source = {
                            "passage": passage_text[:200],
                            "doc_id": passage.get("doc_id", f"doc_{rank}"),
                            "rank": rank
                        }

                    if lcs > max_lcs:
                        max_lcs = lcs
                        if max_metric_type != "lco":
                            max_metric_type = "lcs"
                            max_source = {
                                "passage": passage_text[:200],
                                "doc_id": passage.get("doc_id", f"doc_{rank}"),
                                "rank": rank
                            }

                    if substring > max_substring:
                        max_substring = substring
                        if max_metric_type not in ["lco", "lcs"]:
                            max_metric_type = "substring"
                            max_source = {
                                "passage": passage_text[:200],
                                "doc_id": passage.get("doc_id", f"doc_{rank}"),
                                "rank": rank
                            }

                record = {
                    "claim_id": claim_id,
                    "claim": claim_text[:500],
                    "k": k,
                    "max_lco": round(max_lco, 6),
                    "max_lcs": round(max_lcs, 6),
                    "max_substring": round(max_substring, 6),
                    "max_source": max_source or {"passage": "", "doc_id": "none", "rank": 0}
                }
                per_claim.append(record)

                metrics_by_name["lco"].append(max_lco)
                metrics_by_name["lcs"].append(max_lcs)
                metrics_by_name["substring"].append(max_substring)

                claim_max_metrics[claim_id]["lco"] = max(claim_max_metrics[claim_id]["lco"], max_lco)
                claim_max_metrics[claim_id]["lcs"] = max(claim_max_metrics[claim_id]["lcs"], max_lcs)
                claim_max_metrics[claim_id]["substring"] = max(claim_max_metrics[claim_id]["substring"], max_substring)

        per_claim.sort(key=lambda x: (x["claim_id"], x["k"]))

        claim_level_metrics = defaultdict(list)
        for _, metrics in claim_max_metrics.items():
            claim_level_metrics["lco"].append(metrics["lco"])
            claim_level_metrics["lcs"].append(metrics["lcs"])
            claim_level_metrics["substring"].append(metrics["substring"])

        summary = {}
        for metric_name in ["lco", "lcs", "substring"]:
            row_values = metrics_by_name[metric_name]
            claim_values = claim_level_metrics[metric_name]
            if not row_values:
                summary[metric_name] = {
                    "max": 0.0, "p95": 0.0, "mean": 0.0,
                    "row_count_ge_0.10": 0, "row_count_ge_0.15": 0, "row_count_ge_0.20": 0,
                    "claim_count_ge_0.10": 0, "claim_count_ge_0.15": 0, "claim_count_ge_0.20": 0
                }
            else:
                summary[metric_name] = {
                    "max": round(max(row_values), 6),
                    "p95": round(statistics.quantiles(row_values, n=20)[18] if len(row_values) > 1 else max(row_values), 6),
                    "mean": round(statistics.mean(row_values), 6),
                    "row_count_ge_0.10": sum(1 for v in row_values if v >= 0.10),
                    "row_count_ge_0.15": sum(1 for v in row_values if v >= 0.15),
                    "row_count_ge_0.20": sum(1 for v in row_values if v >= 0.20),
                    "claim_count_ge_0.10": sum(1 for v in claim_values if v >= 0.10) if claim_values else 0,
                    "claim_count_ge_0.15": sum(1 for v in claim_values if v >= 0.15) if claim_values else 0,
                    "claim_count_ge_0.20": sum(1 for v in claim_values if v >= 0.20) if claim_values else 0,
                }
    else:
        summary, per_claim = compute_leakage_metrics(claims, passages, k_values)
    
    json_file = outdir / "leakage_report.json"
    csv_file = outdir / "leakage_report.csv"
    
    write_json_report(
        json_file,
        run_id,
        args.seed,
        args.k,
        args.k2,
        claims_path,
        retrieval_mode,
        summary,
        per_claim,
        corpus_path=corpus_path,
        retrieval_manifest=retrieval_manifest,
    )
    write_csv_report(csv_file, per_claim)
    write_readme(outdir)
    
    print(f"\n[OK] Leakage analysis complete")
    print(f"[OK] JSON report: {json_file}")
    print(f"[OK] CSV report: {csv_file}")
    print(f"\n[OK] Summary:")
    print(f"  Claims scanned: {len(set(c['claim_id'] for c in per_claim))}")
    print(f"  K values: {k_values}")
    print(f"  Total records: {len(per_claim)}")
    print(f"  Retrieval mode: {retrieval_mode}")
    
    if args.fail_on_threshold:
        max_all = max(summary[m]["max"] for m in ["lco", "lcs", "substring"])
        if max_all > args.fail_on_threshold:
            print(f"\n[ERROR] Max metric {max_all:.4f} exceeds threshold {args.fail_on_threshold:.4f}")
            return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
