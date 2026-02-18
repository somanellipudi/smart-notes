"""
Active Learning Utilities for CSClaimBench Expansion

Provides uncertainty scoring and diversity sampling functions for selecting
informative examples to label in each active learning round.

Key Functions:
- compute_uncertainty: Calculate uncertainty scores from model predictions
- diversity_sample: Topic-stratified sampling for diverse examples
- select_for_labeling: Combined uncertainty + diversity selection
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict, Counter
import logging

logger = logging.getLogger(__name__)


def compute_uncertainty_least_confident(probabilities: np.ndarray) -> float:
    """
    Compute uncertainty using least confident strategy.
    
    Uncertainty = 1 - max(probabilities)
    
    Higher values indicate more uncertain predictions.
    
    Args:
        probabilities: Array of class probabilities [P(entail), P(neutral), P(contradict)]
    
    Returns:
        Uncertainty score in [0, 1]
    
    Example:
        >>> probs = np.array([0.95, 0.03, 0.02])  # Very confident
        >>> compute_uncertainty_least_confident(probs)
        0.05
        >>> probs = np.array([0.4, 0.35, 0.25])  # Uncertain
        >>> compute_uncertainty_least_confident(probs)
        0.6
    """
    if len(probabilities) == 0:
        return 0.0
    return 1.0 - np.max(probabilities)


def compute_uncertainty_margin(probabilities: np.ndarray) -> float:
    """
    Compute uncertainty using margin strategy.
    
    Uncertainty = 1 - (P(top1) - P(top2))
    
    Measures the gap between top two predictions. Smaller gap = more uncertain.
    
    Args:
        probabilities: Array of class probabilities
    
    Returns:
        Uncertainty score in [0, 1]
    
    Example:
        >>> probs = np.array([0.95, 0.03, 0.02])  # Large margin
        >>> compute_uncertainty_margin(probs)
        0.08
        >>> probs = np.array([0.51, 0.49, 0.00])  # Small margin
        >>> compute_uncertainty_margin(probs)
        0.98
    """
    if len(probabilities) < 2:
        return 0.0
    
    sorted_probs = np.sort(probabilities)[::-1]  # Descending order
    margin = sorted_probs[0] - sorted_probs[1]
    return 1.0 - margin


def compute_uncertainty_entropy(probabilities: np.ndarray) -> float:
    """
    Compute uncertainty using entropy.
    
    Entropy = -sum(p * log(p)) for each class probability
    Normalized to [0, 1] by dividing by log(num_classes)
    
    Args:
        probabilities: Array of class probabilities
    
    Returns:
        Normalized entropy score in [0, 1]
    
    Example:
        >>> probs = np.array([1.0, 0.0, 0.0])  # Certain
        >>> compute_uncertainty_entropy(probs)
        0.0
        >>> probs = np.array([0.33, 0.33, 0.34])  # Maximum uncertainty
        >>> compute_uncertainty_entropy(probs)
        ~1.0
    """
    if len(probabilities) == 0:
        return 0.0
    
    # Avoid log(0) by adding small epsilon
    eps = 1e-10
    probs_safe = np.clip(probabilities, eps, 1.0)
    
    entropy = -np.sum(probs_safe * np.log(probs_safe))
    
    # Normalize by maximum possible entropy (uniform distribution)
    max_entropy = np.log(len(probabilities))
    if max_entropy > 0:
        return entropy / max_entropy
    return 0.0


def compute_uncertainty(
    probabilities: np.ndarray,
    strategy: str = "least_confident"
) -> float:
    """
    Compute uncertainty score using specified strategy.
    
    Args:
        probabilities: Array of class probabilities
        strategy: One of ["least_confident", "margin", "entropy"]
    
    Returns:
        Uncertainty score in [0, 1]
    
    Raises:
        ValueError: If unknown strategy specified
    """
    if strategy == "least_confident":
        return compute_uncertainty_least_confident(probabilities)
    elif strategy == "margin":
        return compute_uncertainty_margin(probabilities)
    elif strategy == "entropy":
        return compute_uncertainty_entropy(probabilities)
    else:
        raise ValueError(
            f"Unknown uncertainty strategy: {strategy}. "
            f"Must be one of: least_confident, margin, entropy"
        )


def diversity_sample_by_topic(
    examples: List[Dict[str, Any]],
    n_samples: int,
    topic_key: str = "domain_topic",
    random_state: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Sample examples with diversity across topics (stratified sampling).
    
    Ensures selected examples cover different topics/domains proportionally.
    
    Args:
        examples: List of example dictionaries
        n_samples: Number of examples to sample
        topic_key: Key in example dict containing topic/domain
        random_state: Random seed for reproducibility
    
    Returns:
        List of sampled examples (stratified by topic)
    
    Example:
        >>> examples = [
        ...     {"doc_id": "1", "domain_topic": "Algorithms", "uncertainty": 0.9},
        ...     {"doc_id": "2", "domain_topic": "Algorithms", "uncertainty": 0.8},
        ...     {"doc_id": "3", "domain_topic": "Networks", "uncertainty": 0.7},
        ... ]
        >>> diversity_sample_by_topic(examples, n_samples=2, random_state=42)
        # Returns 1 from Algorithms, 1 from Networks
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    if n_samples >= len(examples):
        return examples
    
    # Group by topic
    topic_groups = defaultdict(list)
    for ex in examples:
        topic = ex.get(topic_key, "unknown")
        topic_groups[topic].append(ex)
    
    # Calculate samples per topic (proportional to topic size)
    topic_counts = {topic: len(exs) for topic, exs in topic_groups.items()}
    total_examples = len(examples)
    
    samples_per_topic = {}
    remaining_samples = n_samples
    
    for topic in sorted(topic_groups.keys()):  # Sorted for determinism
        proportion = topic_counts[topic] / total_examples
        n_topic = max(1, int(np.round(proportion * n_samples)))
        n_topic = min(n_topic, len(topic_groups[topic]), remaining_samples)
        samples_per_topic[topic] = n_topic
        remaining_samples -= n_topic
    
    # Distribute any remaining samples
    while remaining_samples > 0:
        for topic in sorted(topic_groups.keys()):
            if remaining_samples == 0:
                break
            if samples_per_topic[topic] < len(topic_groups[topic]):
                samples_per_topic[topic] += 1
                remaining_samples -= 1
    
    # Sample from each topic
    selected = []
    for topic, n_topic in samples_per_topic.items():
        topic_examples = topic_groups[topic]
        if n_topic >= len(topic_examples):
            selected.extend(topic_examples)
        else:
            indices = np.random.choice(len(topic_examples), size=n_topic, replace=False)
            selected.extend([topic_examples[i] for i in indices])
    
    return selected


def select_for_labeling(
    examples_with_predictions: List[Dict[str, Any]],
    n_samples: int,
    uncertainty_strategy: str = "least_confident",
    diversity_weight: float = 0.3,
    topic_key: str = "domain_topic",
    prob_key: str = "probabilities",
    random_state: Optional[int] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Select examples for labeling using combined uncertainty + diversity.
    
    Algorithm:
    1. Compute uncertainty score for each example
    2. Select top uncertain examples (oversample by 2x)
    3. Apply diversity sampling to ensure topic coverage
    
    Args:
        examples_with_predictions: List of dicts with predictions and metadata
        n_samples: Number of examples to select
        uncertainty_strategy: Uncertainty computation method
        diversity_weight: Weight for diversity vs uncertainty (0=pure uncertain, 1=pure diverse)
        topic_key: Key for topic/domain field
        prob_key: Key for probability array field
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (selected_examples, selection_stats)
    
    Example:
        >>> examples = [
        ...     {
        ...         "doc_id": "algo_001",
        ...         "domain_topic": "Algorithms",
        ...         "probabilities": np.array([0.45, 0.30, 0.25])
        ...     },
        ...     # ... more examples
        ... ]
        >>> selected, stats = select_for_labeling(examples, n_samples=50)
        >>> print(stats["topic_distribution"])
        {'Algorithms': 12, 'Networks': 10, 'OS': 8, ...}
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    if n_samples >= len(examples_with_predictions):
        logger.warning(
            f"Requested {n_samples} samples but only {len(examples_with_predictions)} available. "
            "Returning all examples."
        )
        return examples_with_predictions, {
            "total_available": len(examples_with_predictions),
            "selected": len(examples_with_predictions),
            "uncertainty_strategy": uncertainty_strategy
        }
    
    # Step 1: Compute uncertainty for all examples
    examples_with_uncertainty = []
    for ex in examples_with_predictions:
        probs = ex.get(prob_key)
        if probs is None:
            logger.warning(f"Example {ex.get('doc_id', 'unknown')} missing '{prob_key}' field, skipping")
            continue
        
        if not isinstance(probs, np.ndarray):
            probs = np.array(probs)
        
        uncertainty = compute_uncertainty(probs, strategy=uncertainty_strategy)
        
        ex_with_unc = ex.copy()
        ex_with_unc["uncertainty"] = uncertainty
        examples_with_uncertainty.append(ex_with_unc)
    
    if len(examples_with_uncertainty) == 0:
        logger.error("No examples with valid predictions found")
        return [], {"error": "No valid examples"}
    
    # Step 2: Sort by uncertainty (descending)
    sorted_by_uncertainty = sorted(
        examples_with_uncertainty,
        key=lambda x: x["uncertainty"],
        reverse=True
    )
    
    # Step 3: Select top uncertain + apply diversity
    if diversity_weight > 0:
        # Oversample uncertain examples then diversify
        oversample_factor = 2.0
        n_uncertain = min(
            int(n_samples * oversample_factor),
            len(sorted_by_uncertainty)
        )
        top_uncertain = sorted_by_uncertainty[:n_uncertain]
        
        # Apply diversity sampling to top uncertain
        selected = diversity_sample_by_topic(
            top_uncertain,
            n_samples=n_samples,
            topic_key=topic_key,
            random_state=random_state
        )
    else:
        # Pure uncertainty sampling (no diversity)
        selected = sorted_by_uncertainty[:n_samples]
    
    # Compute selection statistics
    topic_distribution = Counter([ex.get(topic_key, "unknown") for ex in selected])
    uncertainty_stats = {
        "mean": float(np.mean([ex["uncertainty"] for ex in selected])),
        "median": float(np.median([ex["uncertainty"] for ex in selected])),
        "min": float(np.min([ex["uncertainty"] for ex in selected])),
        "max": float(np.max([ex["uncertainty"] for ex in selected]))
    }
    
    stats = {
        "total_available": len(examples_with_predictions),
        "selected": len(selected),
        "uncertainty_strategy": uncertainty_strategy,
        "diversity_weight": diversity_weight,
        "topic_distribution": dict(topic_distribution),
        "uncertainty_stats": uncertainty_stats
    }
    
    logger.info(f"Selected {len(selected)} examples for labeling")
    logger.info(f"Topic distribution: {dict(topic_distribution)}")
    logger.info(f"Uncertainty range: [{uncertainty_stats['min']:.3f}, {uncertainty_stats['max']:.3f}]")
    
    return selected, stats


def validate_predictions(
    examples_with_predictions: List[Dict[str, Any]],
    prob_key: str = "probabilities"
) -> Tuple[bool, List[str]]:
    """
    Validate that predictions have required format.
    
    Args:
        examples_with_predictions: List of dicts with predictions
        prob_key: Key for probability array
    
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    for i, ex in enumerate(examples_with_predictions):
        if prob_key not in ex:
            errors.append(f"Example {i} ({ex.get('doc_id', 'unknown')}) missing '{prob_key}' field")
            continue
        
        probs = ex[prob_key]
        if not isinstance(probs, (list, np.ndarray)):
            errors.append(f"Example {i} probabilities must be list or array, got {type(probs)}")
            continue
        
        probs_array = np.array(probs) if isinstance(probs, list) else probs
        
        if len(probs_array) != 3:
            errors.append(f"Example {i} must have 3 probabilities (entail/neutral/contradict), got {len(probs_array)}")
        
        if not np.allclose(np.sum(probs_array), 1.0, atol=0.01):
            errors.append(f"Example {i} probabilities must sum to 1.0, got {np.sum(probs_array):.3f}")
        
        if np.any(probs_array < 0) or np.any(probs_array > 1):
            errors.append(f"Example {i} probabilities must be in [0, 1]")
    
    return len(errors) == 0, errors
