"""
Safe OCR initialization with lazy loading and cloud detection.

Prevents crashes on Streamlit Cloud by:
- Not importing EasyOCR at module level
- Lazy loading on first use
- Catching all initialization errors
- Tracking model downloads
"""

import logging
import streamlit as st
from typing import Optional, Tuple, Dict, Any
import config

logger = logging.getLogger(__name__)

# Global OCR reader instance (lazily initialized)
_OCR_READER = None
_OCR_INIT_ATTEMPTED = False
_OCR_INIT_ERROR = None
_OCR_MODEL_DOWNLOADED = False


@st.cache_resource
def get_ocr_reader(gpu: bool = False) -> Optional[object]:
    """
    Get cached OCR reader instance.
    
    Uses st.cache_resource to persist across reruns.
    
    Args:
        gpu: Whether to use GPU acceleration
    
    Returns:
        EasyOCR Reader instance or None if unavailable
    """
    global _OCR_READER, _OCR_INIT_ATTEMPTED, _OCR_INIT_ERROR, _OCR_MODEL_DOWNLOADED
    
    # Check if OCR is disabled
    if not config.OCR_ENABLED:
        logger.info("OCR disabled via config")
        return None
    
    # Return cached reader if available
    if _OCR_READER is not None:
        return _OCR_READER
    
    # Don't retry if previous attempt failed
    if _OCR_INIT_ATTEMPTED and _OCR_INIT_ERROR:
        logger.debug(f"OCR init previously failed: {_OCR_INIT_ERROR}")
        return None
    
    _OCR_INIT_ATTEMPTED = True
    
    try:
        logger.info("Initializing EasyOCR...")
        import easyocr
        
        # Check if model needs to be downloaded
        # EasyOCR will download to ~/.EasyOCR/model/ on first use
        import os
        model_dir = os.path.expanduser("~/.EasyOCR/model/")
        model_exists_before = os.path.exists(model_dir) and len(os.listdir(model_dir)) > 0
        
        # Initialize reader
        _OCR_READER = easyocr.Reader(
            ['en'],
            gpu=gpu and config.OCR_ENABLED,
            verbose=False
        )
        
        # Check if model was downloaded
        model_exists_after = os.path.exists(model_dir) and len(os.listdir(model_dir)) > 0
        if model_exists_after and not model_exists_before:
            _OCR_MODEL_DOWNLOADED = True
            logger.info("EasyOCR model downloaded")
        
        device = "GPU" if gpu else "CPU"
        logger.info(f"EasyOCR initialized successfully on {device}")
        
        return _OCR_READER
    
    except ImportError:
        _OCR_INIT_ERROR = "easyocr not installed"
        logger.warning("EasyOCR not available: package not installed")
        return None
    
    except Exception as e:
        _OCR_INIT_ERROR = str(e)
        logger.error(f"Failed to initialize EasyOCR: {e}")
        return None


def is_ocr_available() -> bool:
    """
    Check if OCR is available without initializing.
    
    Returns:
        True if OCR can be used
    """
    if not config.OCR_ENABLED:
        return False
    
    # Try to import without initializing
    try:
        import easyocr
        return True
    except ImportError:
        return False


def get_ocr_device() -> str:
    """
    Determine which device OCR will use.
    
    Returns:
        "cuda" | "mps" | "cpu"
    """
    if not is_ocr_available():
        return "none"
    
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    except ImportError:
        return "cpu"


def ocr_extract_text(
    image,
    use_gpu: bool = False,
    detail_level: int = 0
) -> Tuple[str, Dict[str, Any]]:
    """
    Extract text from image using OCR.
    
    Safe wrapper that handles all errors gracefully.
    
    Args:
        image: PIL Image or numpy array
        use_gpu: Whether to use GPU
        detail_level: EasyOCR detail level (0-1)
    
    Returns:
        Tuple of (extracted_text, metadata_dict)
    """
    metadata = {
        "ocr_enabled": config.OCR_ENABLED,
        "ocr_available": False,
        "device": None,
        "model_downloaded": False,
        "error": None
    }
    
    # Check if OCR is available
    reader = get_ocr_reader(gpu=use_gpu)
    if reader is None:
        metadata["error"] = _OCR_INIT_ERROR or "OCR not available"
        return "", metadata
    
    metadata["ocr_available"] = True
    metadata["device"] = get_ocr_device()
    metadata["model_downloaded"] = _OCR_MODEL_DOWNLOADED
    
    try:
        # Extract text
        result = reader.readtext(image, detail=detail_level)
        
        if detail_level == 0:
            # Result is list of strings
            text = " ".join(result)
        else:
            # Result is list of (bbox, text, confidence) tuples
            text = " ".join([item[1] for item in result])
        
        return text, metadata
    
    except Exception as e:
        logger.error(f"OCR extraction failed: {e}")
        metadata["error"] = str(e)
        return "", metadata


def get_ocr_status() -> Dict[str, Any]:
    """
    Get current OCR status for display/tracking.
    
    Returns:
        Dict with OCR status information
    """
    return {
        "enabled": config.OCR_ENABLED,
        "available": is_ocr_available(),
        "device": get_ocr_device(),
        "initialized": _OCR_READER is not None,
        "init_attempted": _OCR_INIT_ATTEMPTED,
        "init_error": _OCR_INIT_ERROR,
        "model_downloaded": _OCR_MODEL_DOWNLOADED,
        "is_cloud": config.IS_STREAMLIT_CLOUD
    }
