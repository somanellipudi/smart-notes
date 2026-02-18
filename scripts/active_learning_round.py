"""
Active Learning Round Script for CSClaimBench Expansion

Runs one round of active learning to select informative examples for human labeling.

Usage:
    python scripts/active_learning_round.py \
        --unlabeled data/unlabeled.jsonl \
        --labeled data/labeled.jsonl \
        --output data/to_label.jsonl \
        --n-samples 50 \
        --strategy margin \
        --seed 42

Algorithm:
    1. Load unlabeled pool
    2. Run verifier to get prediction probabilities
    3. Compute uncertainty scores
    4. Apply diversity sampling by domain_topic
    5. Select top N examples
    6. Save to output file for human annotation

Author: Smart-Notes Active Learning Module
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.al_utils import (
    select_for_labeling,
    validate_predictions,
    compute_uncertainty
)

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


def run_mock_verifier(
    examples: List[Dict[str, Any]],
    random_state: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Mock verifier that generates random predictions.
    
    In production, replace this with actual NLI model inference.
    
    Args:
        examples: List of examples to predict
        random_state: Random seed for reproducibility
    
    Returns:
        Examples with added 'probabilities' field
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    logger.info("Running mock verifier (replace with actual NLI model in production)")
    
    examples_with_preds = []
    for ex in examples:
        # Generate random probabilities that sum to 1.0
        # Use Dirichlet distribution for valid probability vectors
        probs = np.random.dirichlet([1.0, 1.0, 1.0])
        
        ex_with_pred = ex.copy()
        ex_with_pred['probabilities'] = probs.tolist()
        ex_with_pred['predicted_label'] = ['ENTAIL', 'NEUTRAL', 'CONTRADICT'][np.argmax(probs)]
        examples_with_preds.append(ex_with_pred)
    
    return examples_with_preds


def run_actual_verifier(
    examples: List[Dict[str, Any]],
    model_name: str = "microsoft/deberta-v3-base"
) -> List[Dict[str, Any]]:
    """
    Run actual NLI verifier on examples.
    
    This function can be extended to use any NLI model (HuggingFace, API, etc.)
    
    Args:
        examples: List of examples to predict
        model_name: Name of NLI model to use
    
    Returns:
        Examples with added 'probabilities' field
    """
    logger.info(f"Running NLI verifier: {model_name}")
    
    try:
        from transformers import pipeline
        
        # Load NLI pipeline
        classifier = pipeline(
            "text-classification",
            model=model_name,
            return_all_scores=True,
            device=-1  # CPU, use device=0 for GPU
        )
        
        examples_with_preds = []
        
        # Process in batches for efficiency
        batch_size = 32
        for i in range(0, len(examples), batch_size):
            batch = examples[i:i+batch_size]
            
            # Format inputs as premise-hypothesis pairs
            inputs = [
                f"{ex['source_text']} [SEP] {ex['claim']}"
                for ex in batch
            ]
            
            # Get predictions
            predictions = classifier(inputs)
            
            # Parse predictions and add to examples
            for ex, pred in zip(batch, predictions):
                # pred is list of {"label": "ENTAILMENT", "score": 0.95}, etc.
                # Map labels to standard format
                label_map = {
                    "ENTAILMENT": 0,  # Index 0: ENTAIL
                    "NEUTRAL": 1,     # Index 1: NEUTRAL
                    "CONTRADICTION": 2  # Index 2: CONTRADICT
                }
                
                probs = np.zeros(3)
                for label_score in pred:
                    label = label_score["label"]
                    score = label_score["score"]
                    if label in label_map:
                        probs[label_map[label]] = score
                
                # Normalize to ensure sum=1.0
                probs = probs / np.sum(probs)
                
                ex_with_pred = ex.copy()
                ex_with_pred['probabilities'] = probs.tolist()
                ex_with_pred['predicted_label'] = ['ENTAIL', 'NEUTRAL', 'CONTRADICT'][np.argmax(probs)]
                examples_with_preds.append(ex_with_pred)
            
            logger.info(f"Processed {i+len(batch)}/{len(examples)} examples")
        
        return examples_with_preds
    
    except ImportError:
        logger.error(
            "transformers library not available. "
            "Install with: pip install transformers torch"
        )
        logger.info("Falling back to mock verifier")
        return run_mock_verifier(examples, random_state=42)
    except Exception as e:
        logger.error(f"Error running verifier: {e}")
        logger.info("Falling back to mock verifier")
        return run_mock_verifier(examples, random_state=42)


def main():
    parser = argparse.ArgumentParser(
        description="Run active learning round to select examples for labeling"
    )
    parser.add_argument(
        "--unlabeled",
        type=Path,
        required=True,
        help="Path to unlabeled examples (JSONL format)"
    )
    parser.add_argument(
        "--labeled",
        type=Path,
        help="Path to already labeled examples (JSONL format). Used for statistics."
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to save selected examples for labeling (JSONL format)"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=50,
        help="Number of examples to select (default: 50)"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="margin",
        choices=["least_confident", "margin", "entropy"],
        help="Uncertainty sampling strategy (default: margin)"
    )
    parser.add_argument(
        "--diversity-weight",
        type=float,
        default=0.3,
        help="Weight for diversity sampling 0-1 (default: 0.3)"
    )
    parser.add_argument(
        "--use-mock-verifier",
        action="store_true",
        help="Use mock verifier instead of actual NLI model"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="microsoft/deberta-v3-base",
        help="NLI model name for actual verifier (default: microsoft/deberta-v3-base)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only show statistics without running selection"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.unlabeled.exists():
        logger.error(f"Unlabeled file not found: {args.unlabeled}")
        sys.exit(1)
    
    # Load unlabeled examples
    logger.info("=" * 60)
    logger.info("ACTIVE LEARNING ROUND")
    logger.info("=" * 60)
    
    unlabeled = load_jsonl(args.unlabeled)
    
    if len(unlabeled) == 0:
        logger.error("No unlabeled examples found")
        sys.exit(1)
    
    # Load labeled examples (if provided) for statistics
    labeled = []
    if args.labeled and args.labeled.exists():
        labeled = load_jsonl(args.labeled)
    
    # Show statistics
    logger.info("-" * 60)
    logger.info("DATASET STATISTICS")
    logger.info("-" * 60)
    logger.info(f"Unlabeled pool: {len(unlabeled)} examples")
    logger.info(f"Already labeled: {len(labeled)} examples")
    logger.info(f"Total available: {len(unlabeled) + len(labeled)} examples")
    
    # Topic distribution in unlabeled pool
    from collections import Counter
    topic_counts = Counter([ex.get('domain_topic', 'unknown') for ex in unlabeled])
    logger.info(f"Unlabeled topic distribution: {dict(topic_counts)}")
    
    if args.stats_only:
        logger.info("Stats-only mode: exiting")
        return
    
    # Run verifier to get predictions
    logger.info("-" * 60)
    logger.info("RUNNING VERIFIER")
    logger.info("-" * 60)
    
    if args.use_mock_verifier:
        examples_with_preds = run_mock_verifier(unlabeled, random_state=args.seed)
    else:
        examples_with_preds = run_actual_verifier(unlabeled, model_name=args.model_name)
    
    # Validate predictions
    is_valid, errors = validate_predictions(examples_with_preds)
    if not is_valid:
        logger.error("Prediction validation failed:")
        for error in errors[:5]:  # Show first 5 errors
            logger.error(f"  - {error}")
        sys.exit(1)
    
    logger.info(f"✓ Predictions validated: {len(examples_with_preds)} examples")
    
    # Select examples for labeling
    logger.info("-" * 60)
    logger.info("SELECTING EXAMPLES FOR LABELING")
    logger.info("-" * 60)
    logger.info(f"Selection strategy: {args.strategy}")
    logger.info(f"Diversity weight: {args.diversity_weight}")
    logger.info(f"Target samples: {args.n_samples}")
    
    selected, stats = select_for_labeling(
        examples_with_preds,
        n_samples=args.n_samples,
        uncertainty_strategy=args.strategy,
        diversity_weight=args.diversity_weight,
        random_state=args.seed
    )
    
    # Save selected examples
    save_jsonl(selected, args.output)
    
    # Save selection statistics
    stats_path = args.output.with_suffix('.stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved selection statistics to {stats_path}")
    
    # Summary
    logger.info("=" * 60)
    logger.info("SELECTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Selected: {len(selected)} examples")
    logger.info(f"Output: {args.output}")
    logger.info(f"Topic distribution: {stats['topic_distribution']}")
    logger.info(f"Uncertainty range: [{stats['uncertainty_stats']['min']:.3f}, {stats['uncertainty_stats']['max']:.3f}]")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Review examples in to_label.jsonl")
    logger.info("  2. Add 'gold_label' field to each example")
    logger.info("  3. Move labeled examples to labeled.jsonl")
    logger.info("  4. Run next AL round with updated pool")


if __name__ == "__main__":
    main()
