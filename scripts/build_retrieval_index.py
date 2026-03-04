#!/usr/bin/env python3
"""Build deterministic retrieval index for leakage scan real mode."""

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.retrieval_module import build_index, load_index


def main() -> int:
    parser = argparse.ArgumentParser(description="Build retrieval index for leakage scan")
    parser.add_argument("--corpus", type=str, required=True, help="Path to corpus file (jsonl/json/csv/txt)")
    parser.add_argument("--outdir", type=str, default="artifacts/retrieval_index", help="Output directory")
    args = parser.parse_args()

    try:
        build_index(corpus_path=args.corpus, outdir=args.outdir)
        index = load_index(outdir=args.outdir)
    except Exception as exc:
        print(f"[ERROR] Failed to build retrieval index: {exc}", file=sys.stderr)
        return 2

    print("[OK] Retrieval index built")
    print(f"[OK] Outdir: {args.outdir}")
    print(f"[OK] Corpus path: {index.manifest.get('corpus_path', args.corpus)}")
    print(f"[OK] Documents indexed: {index.manifest.get('num_docs', 0)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
