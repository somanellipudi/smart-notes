"""
Image OCR module for extracting text from handwritten notes, blackboard, and whiteboard images.

This module provides OCR capabilities to convert images of notes into text
that can be processed by the reasoning pipeline.

Supports multiple OCR backends:
- EasyOCR (default, no API key needed)
- Tesseract OCR (fallback)
"""

from __future__ import annotations

import logging
import hashlib
import copy
from pathlib import Path
from typing import Optional, List, Dict, Any, TYPE_CHECKING
import tempfile
import importlib.util
import config

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = Any

def _check_easyocr():
    return importlib.util.find_spec("easyocr") is not None

def _check_pytesseract():
    return importlib.util.find_spec("pytesseract") is not None

EASYOCR_AVAILABLE = _check_easyocr()
PYTESSERACT_AVAILABLE = _check_pytesseract()

logger = logging.getLogger(__name__)

_OCR_RESULT_CACHE: Dict[str, Dict[str, Any]] = {}
_OCR_CACHE_MAX_ENTRIES = 64


class ImageOCR:
    """
    Extract text from images using OCR.
    
    Supports:
    - Handwritten notes
    - Blackboard/whiteboard photos
    - Printed text
    - Multiple languages (configurable)
    """
    
    def __init__(self):
        """
        Initialize OCR processor with error handling for Streamlit Cloud.
        """
        self.reader = None
        self.ocr_backend = None
        self.backend_error = None
        
        # Re-check backend in case it was just installed
        easyocr_ok = _check_easyocr()
        pytesseract_ok = _check_pytesseract()

        logger.info(f"OCR Init: easyocr={easyocr_ok}, pytesseract={pytesseract_ok}")

        if easyocr_ok:
            try:
                logger.info("Initializing EasyOCR...")
                import easyocr
                self.reader = easyocr.Reader(['en'], gpu=False, model_storage_directory='/tmp/easyocr')
                self.ocr_backend = "easyocr"
                logger.info("✓ EasyOCR initialized successfully")
                return
            except Exception as e:
                global EASYOCR_AVAILABLE
                EASYOCR_AVAILABLE = False
                self.backend_error = f"EasyOCR failed: {e}"
                logger.warning(f"EasyOCR unavailable, falling back: {e}")

        if pytesseract_ok:
            try:
                import pytesseract
                self.ocr_backend = "pytesseract"
                logger.info("✓ pytesseract initialized successfully")
                return
            except Exception as e:
                self.backend_error = f"pytesseract failed: {e}"
                logger.error(f"pytesseract initialization failed: {e}")

        error_detail = self.backend_error or "No OCR backend available"
        raise RuntimeError(
            "OCR system unavailable. "
            f"{error_detail}. "
            "Fix: pip install easyocr==1.7.1 python-bidi bidi, "
            "or install pytesseract plus the Tesseract OCR binary."
        )
    
    def extract_text_from_image(
        self,
        image_path: str,
        image_type: str = "notes"
    ) -> Dict[str, Any]:
        """
        Extract text from an image file.
        
        Args:
            image_path: Path to image file
            image_type: Type of image ("notes", "blackboard", "whiteboard")
        
        Returns:
            Dictionary with:
                - text: Extracted text
                - confidence: Average confidence score (0-1)
                - metadata: Additional processing info
        """
        try:
            # Read image
            if not PIL_AVAILABLE:
                raise ImportError("PIL/Pillow not available")
            
            image = Image.open(image_path)
            
            # Preprocess based on image type
            if image_type in ["blackboard", "whiteboard"]:
                image = self._preprocess_board_image(image)
            else:
                image = self._preprocess_notes_image(image)
            
            # Extract text
            if self.ocr_backend == "easyocr":
                result = self._extract_with_easyocr(image)
            else:
                result = self._extract_with_tesseract(image)
            
            logger.info(
                f"OCR extracted {len(result['text'])} characters "
                f"(confidence: {result['confidence']:.2f})"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return {
                "text": "",
                "confidence": 0.0,
                "metadata": {"error": str(e)}
            }
    
    def extract_text_from_bytes(
        self,
        image_bytes: bytes,
        image_type: str = "notes"
    ) -> Dict[str, Any]:
        """
        Extract text from image bytes (e.g., from file upload).
        
        Args:
            image_bytes: Image file bytes
            image_type: Type of image
        
        Returns:
            OCR result dictionary
        """
        if not image_bytes:
            return {
                "text": "",
                "confidence": 0.0,
                "metadata": {"error": "Empty image bytes"}
            }

        cache_key = hashlib.sha256(image_bytes).hexdigest()
        cached = _OCR_RESULT_CACHE.get(cache_key)
        if cached:
            cached_result = copy.deepcopy(cached)
            cached_result.setdefault("metadata", {})["from_cache"] = True
            return cached_result

        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name
        
        try:
            result = self.extract_text_from_image(tmp_path, image_type)
        finally:
            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)

        if result and "text" in result:
            if len(_OCR_RESULT_CACHE) >= _OCR_CACHE_MAX_ENTRIES:
                _OCR_RESULT_CACHE.clear()
            _OCR_RESULT_CACHE[cache_key] = copy.deepcopy(result)

        return result

    def _downscale_image(self, image: Image.Image) -> Image.Image:
        """Downscale large images to speed up OCR."""
        try:
            max_dim = int(getattr(config, "OCR_MAX_IMAGE_DIM", 1600))
        except Exception:
            max_dim = 1600

        if max_dim <= 0:
            return image

        width, height = image.size
        current_max = max(width, height)
        if current_max <= max_dim:
            return image

        scale = max_dim / float(current_max)
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        resample = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
        image = image.resize(new_size, resample=resample)
        logger.info("Downscaled image from %sx%s to %sx%s for OCR", width, height, new_size[0], new_size[1])
        return image
    
    def _preprocess_notes_image(self, image: Image.Image) -> Image.Image:
        """Preprocess handwritten notes image for better OCR accuracy."""
        try:
            from PIL import ImageEnhance, ImageFilter

            image = self._downscale_image(image)

            # Convert to grayscale
            image = image.convert('L')
            
            # Enhance contrast to make text darker and background lighter
            contrast_enhancer = ImageEnhance.Contrast(image)
            image = contrast_enhancer.enhance(2.0)
            
            # Enhance brightness if image is too dark
            brightness_enhancer = ImageEnhance.Brightness(image)
            image = brightness_enhancer.enhance(1.1)
            
            # Enhance sharpness to make text clearer
            sharpness_enhancer = ImageEnhance.Sharpness(image)
            image = sharpness_enhancer.enhance(1.5)
            
            # Apply slight blur to reduce noise (optional)
            image = image.filter(ImageFilter.MedianFilter(size=3))
            
            logger.info("✓ Image preprocessing: contrast+brightness+sharpness applied")
        except Exception as e:
            logger.warning(f"Image preprocessing warning: {e}")
        
        return image
    
    def _preprocess_board_image(self, image: Image.Image) -> Image.Image:
        """Preprocess blackboard/whiteboard image for better OCR accuracy."""
        try:
            from PIL import ImageEnhance, ImageOps, ImageFilter

            image = self._downscale_image(image)

            # Convert to grayscale
            image = image.convert('L')
            
            # For dark blackboards: invert to make text light
            # Calculate average brightness
            import numpy as np
            img_array = np.array(image)
            avg_brightness = np.mean(img_array)
            
            if avg_brightness < 128:  # Dark image, likely blackboard
                logger.info("Detected dark board, inverting colors...")
                image = ImageOps.invert(image)
            
            # Enhance contrast
            contrast_enhancer = ImageEnhance.Contrast(image)
            image = contrast_enhancer.enhance(2.0)
            
            # Enhance sharpness
            sharpness_enhancer = ImageEnhance.Sharpness(image)
            image = sharpness_enhancer.enhance(1.5)
            
            # Reduce noise
            image = image.filter(ImageFilter.MedianFilter(size=3))
            
            logger.info("✓ Blackboard preprocessing: invert+contrast+sharpness applied")
        except Exception as e:
            logger.warning(f"Board preprocessing warning: {e}")
        
        return image
    
    def _extract_with_easyocr(self, image: Image.Image) -> Dict[str, Any]:
        """Extract text using EasyOCR with improved preprocessing."""
        import numpy as np
        
        # Convert PIL Image to numpy array
        img_array = np.array(image)
        
        # Additional preprocessing: optional upscaling for small images
        try:
            import cv2
            min_dim = min(img_array.shape[0], img_array.shape[1])
            upscale_min_dim = int(getattr(config, "OCR_UPSCALE_MIN_DIM", 600))
            upscale_factor = float(getattr(config, "OCR_UPSCALE_FACTOR", 1.3))
            if min_dim < upscale_min_dim and upscale_factor > 1.0:
                new_size = (
                    int(img_array.shape[1] * upscale_factor),
                    int(img_array.shape[0] * upscale_factor)
                )
                img_array = cv2.resize(img_array, new_size, interpolation=cv2.INTER_CUBIC)
                logger.info("✓ Image upscaled by %sx for OCR", upscale_factor)
        except Exception as e:
            logger.debug(f"OpenCV upscaling skipped: {e}")
        
        # Run OCR
        results = self.reader.readtext(img_array)
        
        # Combine text and calculate average confidence
        text_parts = []
        confidences = []
        
        for (bbox, text, conf) in results:
            text_parts.append(text)
            confidences.append(conf)
        
        extracted_text = "\n".join(text_parts)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return {
            "text": extracted_text,
            "confidence": avg_confidence,
            "metadata": {
                "backend": "easyocr",
                "num_detections": len(results)
            }
        }

    def _extract_with_tesseract(self, image: Image.Image) -> Dict[str, Any]:
        """Extract text using pytesseract fallback."""
        try:
            import pytesseract
        except Exception as e:
            logger.error(f"pytesseract import failed: {e}")
            return {
                "text": "",
                "confidence": 0.0,
                "metadata": {"backend": "pytesseract", "error": str(e)}
            }

        try:
            text = pytesseract.image_to_string(image)
            return {
                "text": text.strip(),
                "confidence": 0.0,
                "metadata": {"backend": "pytesseract"}
            }
        except Exception as e:
            logger.error(f"pytesseract OCR failed: {e}")
            return {
                "text": "",
                "confidence": 0.0,
                "metadata": {"backend": "pytesseract", "error": str(e)}
            }
    
    
def process_images(
    image_files: List[Any],
    image_types: List[str] = None,
    correct_with_llm: bool = True,
    provider_type: str = "ollama",
    api_key: str = None,
    ollama_url: str = None,
    model: str = None
) -> str:
    """
    Process multiple images and combine extracted text.
    
    Args:
        image_files: List of image file objects or paths
        image_types: List of image types (same length as image_files)
        correct_with_llm: If True, use LLM to correct OCR text
        provider_type: LLM provider ("openai" or "ollama")
        api_key: API key for OpenAI (if applicable)
        ollama_url: URL for Ollama (if applicable)
        model: Model name to use
    
    Returns:
        Combined extracted text
    """
    ocr = ImageOCR()
    
    if image_types is None:
        image_types = ["notes"] * len(image_files)
    
    extracted_texts = []
    
    for img_file, img_type in zip(image_files, image_types):
        if isinstance(img_file, (str, Path)):
            result = ocr.extract_text_from_image(str(img_file), img_type)
        else:
            # Assume it's a file-like object with .read()
            result = ocr.extract_text_from_bytes(img_file.read(), img_type)
        
        if result['text']:
            extracted_texts.append(result['text'])
            logger.info(
                f"Extracted {len(result['text'])} chars from {img_type} "
                f"(confidence: {result['confidence']:.1%})"
            )
    
    combined_text = "\n\n---\n\n".join(extracted_texts)
    
    # Post-process with LLM to correct OCR errors
    if correct_with_llm and combined_text:
        logger.info("📝 Correcting OCR text with LLM...")
        combined_text = _correct_ocr_with_llm(
            combined_text,
            provider_type=provider_type,
            api_key=api_key,
            ollama_url=ollama_url,
            model=model
        )
    
    return combined_text


def _correct_ocr_with_llm(
    raw_ocr_text: str,
    provider_type: str = "ollama",
    api_key: str = None,
    ollama_url: str = None,
    model: str = None
) -> str:
    """
    Use LLM to correct garbled OCR text by reconstructing likely meaning.
    
    Args:
        raw_ocr_text: Raw OCR output (may be garbled)
        provider_type: LLM provider ("openai" or "ollama")
        api_key: API key for OpenAI (if applicable)
        ollama_url: URL for Ollama (if applicable)
        model: Model name to use
    
    Returns:
        Corrected text
    """
    try:
        from src.reasoning.pipeline import ReasoningPipeline
        import config
        
        pipeline = ReasoningPipeline(
            provider_type=provider_type,
            api_key=api_key or config.OPENAI_API_KEY,
            ollama_url=ollama_url or config.OLLAMA_URL,
            model=model or (config.OLLAMA_MODEL if provider_type == "ollama" else config.LLM_MODEL)
        )
        
        correction_prompt = f"""You are an expert at reading and correcting optical character recognition (OCR) errors.

You have received poorly extracted text from handwritten notes or photos. Your task is to:
1. Fix spelling and character recognition errors
2. Correct spacing and punctuation
3. Make the text coherent while preserving the original content and structure
4. Maintain all technical terms, formulas, and educational content
5. Return ONLY the corrected text, nothing else

RAW OCR TEXT:
{raw_ocr_text}

CORRECTED TEXT:"""
        
        response = pipeline._call_llm(
            prompt=correction_prompt,
            system_prompt="You are an expert OCR correction specialist. Fix the extracted text to make it readable and coherent while preserving all educational content."
        )
        
        logger.info("✓ OCR correction complete")
        return response.strip()
    
    except Exception as e:
        logger.warning(f"OCR correction failed: {e}. Using raw text.")
        return raw_ocr_text
