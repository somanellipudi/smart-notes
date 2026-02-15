"""
Robust PDF text extraction with multi-strategy approach and OCR fallback.

This module implements intelligent PDF ingestion that:
1. Tries PyMuPDF (fitz) extraction
2. Falls back to pdfplumber extraction
3. Uses OCR if extracted text quality is poor
4. Returns clean combined text with metadata
"""

import logging
import re
from typing import Optional, Tuple, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


def _count_letters(text: str) -> int:
    """Count alphabetic characters in text."""
    return sum(1 for c in text if c.isalpha())


def _count_words(text: str) -> int:
    """Count words in text (split by whitespace)."""
    return len(text.split())


def _compute_alphabetic_ratio(text: str) -> float:
    """Compute ratio of alphabetic characters to total length."""
    if not text:
        return 0.0
    letters = _count_letters(text)
    return letters / len(text)


def _assess_extraction_quality(text: str) -> Tuple[bool, str]:
    """
    Assess if extracted text quality is acceptable.
    
    Returns:
        (is_good: bool, reason: str)
    """
    text = text.strip()
    
    if not text:
        return False, "Empty text"
    
    letters = _count_letters(text)
    words = _count_words(text)
    alpha_ratio = _compute_alphabetic_ratio(text)
    
    # Quality thresholds
    if words < 80:
        return False, f"Too few words: {words} < 80"
    
    if letters < 400:
        return False, f"Too few letters: {letters} < 400"
    
    if alpha_ratio < 0.30:
        return False, f"Low alphabetic ratio: {alpha_ratio:.3f} < 0.30"
    
    return True, "Good quality"


def extract_pdf_text(uploaded_file, ocr: Optional[Any] = None) -> Tuple[str, Dict[str, Any]]:
    """
    Extract text from PDF using multi-strategy approach with OCR fallback.
    
    Args:
        uploaded_file: Streamlit UploadedFile object or file path
        ocr: Optional ImageOCR instance for OCR fallback
    
    Returns:
        Tuple of (extracted_text, metadata_dict)
    
    Metadata includes:
        - extraction_method_used: "pymupdf" | "pdfplumber" | "ocr"
        - pages: number of pages processed
        - letters: letter count
        - words: word count
        - alphabetic_ratio: alphabetic character ratio
        - quality_assessment: success reason or failure reason
    """
    metadata: Dict[str, Any] = {
        "extraction_method_used": None,
        "pages": 0,
        "letters": 0,
        "words": 0,
        "alphabetic_ratio": 0.0,
        "quality_assessment": ""
    }
    
    # Get file path
    if hasattr(uploaded_file, 'name'):
        # Streamlit UploadedFile
        file_path = uploaded_file.name
        # Use getvalue() to avoid file pointer issues, fallback to read() if needed
        if hasattr(uploaded_file, 'getvalue'):
            file_bytes = uploaded_file.getvalue()
        else:
            # Reset pointer and read for file-like objects
            if hasattr(uploaded_file, 'seek'):
                uploaded_file.seek(0)
            file_bytes = uploaded_file.read()
    else:
        # Assume it's a file path string
        file_path = str(uploaded_file)
        with open(file_path, 'rb') as f:
            file_bytes = f.read()
    
    logger.info(f"Extracting PDF from: {file_path}")
    
    # Strategy 1: Try PyMuPDF (fitz)
    logger.info("Strategy 1: Trying PyMuPDF extraction...")
    extracted_text = _extract_with_pymupdf(file_bytes)
    is_good, assessment = _assess_extraction_quality(extracted_text)
    
    if is_good:
        logger.info(f"✓ PyMuPDF extraction succeeded: {assessment}")
        metadata["extraction_method_used"] = "pymupdf"
        pages = _count_pdf_pages(file_bytes)
        metadata["pages"] = pages
        metadata["quality_assessment"] = assessment
    else:
        logger.warning(f"✗ PyMuPDF extraction failed: {assessment}")
        logger.info("Strategy 2: Trying pdfplumber extraction...")
        
        # Strategy 2: Try pdfplumber
        extracted_text = _extract_with_pdfplumber(file_bytes)
        is_good, assessment = _assess_extraction_quality(extracted_text)
        
        if is_good:
            logger.info(f"✓ pdfplumber extraction succeeded: {assessment}")
            metadata["extraction_method_used"] = "pdfplumber"
            pages = _count_pdf_pages(file_bytes)
            metadata["pages"] = pages
            metadata["quality_assessment"] = assessment
        else:
            logger.warning(f"✗ pdfplumber extraction failed: {assessment}")
            logger.info("Strategy 3: Falling back to OCR...")
            
            # Strategy 3: Use OCR as fallback
            if ocr is None:
                logger.error("OCR fallback requested but ocr instance not provided")
                # Return what we have with warning
                extracted_text = _clean_text(extracted_text)
                metadata["extraction_method_used"] = "fallback_text_only"
                metadata["quality_assessment"] = f"OCR unavailable; text quality: {assessment}"
            else:
                extracted_text = _extract_with_ocr(file_bytes, ocr)
                is_good, assessment = _assess_extraction_quality(extracted_text)
                
                if is_good:
                    logger.info(f"✓ OCR extraction succeeded: {assessment}")
                    metadata["extraction_method_used"] = "ocr"
                    pages = _count_pdf_pages(file_bytes)
                    metadata["pages"] = pages
                    metadata["quality_assessment"] = assessment
                else:
                    logger.warning(f"✗ OCR extraction also failed: {assessment}")
                    logger.warning("All extraction strategies failed, returning best effort text")
                    metadata["extraction_method_used"] = "fallback_text_only"
                    metadata["quality_assessment"] = f"All strategies failed; last error: {assessment}"
    
    # Clean text and compute final metrics
    extracted_text = _clean_text(extracted_text)
    letters = _count_letters(extracted_text)
    words = _count_words(extracted_text)
    alpha_ratio = _compute_alphabetic_ratio(extracted_text)
    
    metadata["letters"] = letters
    metadata["words"] = words
    metadata["alphabetic_ratio"] = alpha_ratio
    
    logger.info(
        f"Final extraction: {words} words, {letters} letters, "
        f"alpha_ratio={alpha_ratio:.3f}, method={metadata['extraction_method_used']}"
    )
    
    return extracted_text, metadata


def _extract_with_pymupdf(file_bytes: bytes) -> str:
    """Extract text using PyMuPDF (fitz)."""
    try:
        import fitz
        
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text_parts = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if text.strip():
                text_parts.append(f"[Page {page_num + 1}]\n{text}")
        
        doc.close()
        
        return "\n".join(text_parts)
    except ImportError:
        logger.debug("PyMuPDF (fitz) not available")
        return ""
    except Exception as e:
        logger.debug(f"PyMuPDF extraction error: {e}")
        return ""


def _extract_with_pdfplumber(file_bytes: bytes) -> str:
    """Extract text using pdfplumber."""
    try:
        import pdfplumber
        from io import BytesIO
        
        pdf_stream = BytesIO(file_bytes)
        text_parts = []
        
        with pdfplumber.open(pdf_stream) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and text.strip():
                    text_parts.append(f"[Page {page_num + 1}]\n{text}")
        
        return "\n".join(text_parts)
    except ImportError:
        logger.debug("pdfplumber not available")
        return ""
    except Exception as e:
        logger.debug(f"pdfplumber extraction error: {e}")
        return ""


def _extract_with_ocr(file_bytes: bytes, ocr) -> str:
    """Extract text using OCR on PDF pages."""
    try:
        from io import BytesIO
        
        # Convert PDF pages to images and OCR
        try:
            from pdf2image import convert_from_bytes
        except ImportError:
            logger.error("pdf2image not available for OCR fallback")
            return ""
        
        # Convert PDF to images (limit to first 5 pages for performance)
        try:
            images = convert_from_bytes(file_bytes, first_page=1, last_page=5)
        except Exception as e:
            logger.warning(f"PDF to image conversion failed: {e}")
            return ""
        
        text_parts = []
        for page_num, image in enumerate(images):
            logger.info(f"OCRing page {page_num + 1}...")
            
            try:
                # Use the existing ImageOCR to extract text
                text = ocr.extract_text_from_image(image)
                if text and text.strip():
                    text_parts.append(f"[Page {page_num + 1}]\n{text}")
            except Exception as e:
                logger.warning(f"OCR failed for page {page_num + 1}: {e}")
                continue
        
        return "\n".join(text_parts)
    except Exception as e:
        logger.error(f"OCR extraction failed: {e}")
        return ""


def _clean_text(text: str) -> str:
    """Clean and normalize extracted text."""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove common garbage patterns
    text = re.sub(r'\(cid:[0-9]+\)', '', text)  # CID glyphs
    text = re.sub(r'[^\x20-\x7E\n\t]', '', text)  # Non-printable chars (keep newline, tab)
    
    # Normalize newlines
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    return text.strip()


def _count_pdf_pages(file_bytes: bytes) -> int:
    """Count pages in PDF."""
    try:
        import fitz
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages = len(doc)
        doc.close()
        return pages
    except Exception:
        try:
            import pdfplumber
            from io import BytesIO
            with pdfplumber.open(BytesIO(file_bytes)) as pdf:
                return len(pdf.pages)
        except Exception:
            return 0
