"""
Random seed control for reproducible runs.

Sets seeds for:
- Python random module
- NumPy
- PyTorch (if available)
- FAISS (if available)
"""

import logging
import random
import os

logger = logging.getLogger(__name__)


def set_global_seed(seed: int) -> None:
    """
    Set global random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    logger.info(f"Setting global random seed: {seed}")
    
    # Python random
    random.seed(seed)
    
    # NumPy
    try:
        import numpy as np
        np.random.seed(seed)
        logger.debug("Set NumPy random seed")
    except ImportError:
        pass
    
    # PyTorch (if available)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # Make CUDA operations deterministic
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        logger.debug("Set PyTorch random seed")
    except ImportError:
        pass
    
    # FAISS (if available)
    try:
        import faiss
        faiss.set_random_seed(seed)
        logger.debug("Set FAISS random seed")
    except ImportError:
        pass
    
    # Set PYTHONHASHSEED for hash reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    logger.info(f"Global seed set to {seed}")


def get_deterministic_config() -> dict:
    """
    Get recommended config for deterministic execution.
    
    Returns:
        Dict of config recommendations
    """
    return {
        "GLOBAL_RANDOM_SEED": 42,
        "LLM_TEMPERATURE": 0.0,  # Greedy decoding
        "EMBEDDING_DEVICE": "cpu",  # Avoid GPU precision issues
        "ENABLE_RERANKER": False,  # Reranker may introduce nondeterminism
        "VERIFIABLE_CONSISTENCY_ENABLED": False,  # LLM-based checks are nondeterministic
    }
