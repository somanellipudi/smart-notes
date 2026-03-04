"""
Remove Labeled Examples from Unlabeled Pool

Helper script to update the unlabeled pool by removing examples
that have been labeled in a previous round.

Usage:
    python scripts/remove_labeled.py \
        --unlabeled data/unlabeled.jsonl \
        --labeled data/newly_labeled.jsonl \
        --output data/unlabeled_updated.jsonl
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Set

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_jsonl(filepath: Path) -> List[Dict[str, Any]]:
    """Load JSONL file into list of dictionaries."""
    examples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                example = json.loads(line)
                examples.append(example)
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing line {line_num} in {filepath}: {e}")
    
    logger.info(f"Loaded {len(examples)} examples from {filepath}")
    return examples


def save_jsonl(examples: List[Dict[str, Any]], filepath: Path) -> None:
    """Save list of dictionaries to JSONL file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        for example in examples:
            json.dump(example, f, ensure_ascii=False)
            f.write('\n')
    
    logger.info(f"Saved {len(examples)} examples to {filepath}")


def get_doc_ids(examples: List[Dict[str, Any]]) -> Set[str]:
    """Extract set of doc_ids from examples."""
    return {ex.get('doc_id', '') for ex in examples if ex.get('doc_id')}


def main():
    parser = argparse.ArgumentParser(
        description="Remove labeled examples from unlabeled pool"
    )
    parser.add_argument(
        "--unlabeled",
        type=Path,
        required=True,
        help="Path to unlabeled pool (JSONL)"
    )
    parser.add_argument(
        "--labeled",
        type=Path,
        required=True,
        help="Path to newly labeled examples (JSONL)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to save updated unlabeled pool (JSONL)"
    )
    parser.add_argument(
        "--id-key",
        type=str,
        default="doc_id",
        help="Key for unique identifier (default: doc_id)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.unlabeled.exists():
        logger.error(f"Unlabeled file not found: {args.unlabeled}")
        return 1
    
    if not args.labeled.exists():
        logger.error(f"Labeled file not found: {args.labeled}")
        return 1
    
    # Load files
    logger.info("=" * 60)
    logger.info("REMOVING LABELED EXAMPLES FROM POOL")
    logger.info("=" * 60)
    
    unlabeled = load_jsonl(args.unlabeled)
    labeled = load_jsonl(args.labeled)
    
    # Get labeled IDs
    labeled_ids = get_doc_ids(labeled)
    logger.info(f"Found {len(labeled_ids)} unique labeled IDs")
    
    # Filter unlabeled pool
    remaining = [
        ex for ex in unlabeled
        if ex.get(args.id_key, '') not in labeled_ids
    ]
    
    removed_count = len(unlabeled) - len(remaining)
    
    logger.info(f"Removed {removed_count} examples from unlabeled pool")
    logger.info(f"Remaining in pool: {len(remaining)} examples")
    
    # Save updated pool
    save_jsonl(remaining, args.output)
    
    # Summary
    logger.info("=" * 60)
    logger.info("COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Original unlabeled: {len(unlabeled)}")
    logger.info(f"Newly labeled: {len(labeled)}")
    logger.info(f"Removed: {removed_count}")
    logger.info(f"Remaining: {len(remaining)}")
    logger.info(f"Output: {args.output}")
    
    return 0


if __name__ == "__main__":
    exit(main())
