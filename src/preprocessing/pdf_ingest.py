"""
Robust PDF text extraction with multi-strategy approach and OCR fallback.

This module implements intelligent PDF ingestion that:
1. Tries pdfplumber extraction
2. Uses OCR if extracted text quality is poor (PyMuPDF rendering + Tesseract/EasyOCR)
4. Returns clean combined text with metadata
5. Raises EvidenceIngestError if OCR is unavailable and text quality is poor
"""

import logging
import re
import tempfile
from io import BytesIO
from typing import Optional, Tuple, Dict, Any
from pathlib import Path

from src.exceptions import EvidenceIngestError
from src.preprocessing.text_cleaner import clean_extracted_text
import config

logger = logging.getLogger(__name__)

# Quality thresholds
MIN_CHARS_FOR_OCR = 300
"""Minimum characters required before skipping OCR fallback"""

MIN_WORDS_FOR_QUALITY = 20
"""Minimum words to consider extraction successful"""

MIN_ALPHA_RATIO = 0.30
"""Minimum alphabetic character ratio for quality text"""

MIN_UNIQUE_CHARS = 20
"""Minimum unique characters to consider extraction successful"""


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
    
    char_count = len(text)
    letters = _count_letters(text)
    words = _count_words(text)
    alpha_ratio = _compute_alphabetic_ratio(text)
    unique_chars = len(set(text))
    
    # Quality thresholds
    if char_count < MIN_CHARS_FOR_OCR:
        return False, f"Too few characters: {char_count} < {MIN_CHARS_FOR_OCR}"
    
    if words < MIN_WORDS_FOR_QUALITY:
        return False, f"Too few words: {words} < {MIN_WORDS_FOR_QUALITY}"
    
    if letters < MIN_CHARS_FOR_OCR:
        return False, f"Too few letters: {letters} < {MIN_CHARS_FOR_OCR}"
    
    if alpha_ratio < MIN_ALPHA_RATIO:
        return False, f"Low alphabetic ratio: {alpha_ratio:.3f} < {MIN_ALPHA_RATIO}"

    if unique_chars < MIN_UNIQUE_CHARS:
        return False, f"Too few unique characters: {unique_chars} < {MIN_UNIQUE_CHARS}"
    
    return True, "Good quality"


def extract_pdf_text(uploaded_file, ocr: Optional[Any] = None) -> Tuple[str, Dict[str, Any]]:
    """
    Extract text from PDF using multi-strategy approach with OCR fallback.
    
    Args:
        uploaded_file: Streamlit UploadedFile object or file path
        ocr: Optional ImageOCR instance for OCR fallback
    
    Returns:
        Tuple of (extracted_text, metadata_dict)
    
    Raises:
        EvidenceIngestError: If OCR is unavailable and text quality is insufficient
    
    Metadata includes:
        - extraction_method: "pdf_text" | "ocr_pymupdf_tesseract" | "ocr_easyocr"
        - method: same as extraction_method
        - diagnostics: structured extraction diagnostics
        - num_pages: number of pages processed
        - chars_extracted: character count
        - words: word count
        - alphabetic_ratio: alphabetic character ratio
        - quality_assessment: success/failure reason
    """
    metadata: Dict[str, Any] = {
        "extraction_method": None,
        "num_pages": 0,
        "chars_extracted": 0,
        "words": 0,
        "alphabetic_ratio": 0.0,
        "quality_assessment": "",
        "ocr_pages": 0,
        "diagnostics": {}
    }
    
    # Get file bytes
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
    
    # Count pages once
    num_pages = _count_pdf_pages(file_bytes)
    metadata["num_pages"] = num_pages
    
    # Strategy 1: Try pdfplumber extraction
    logger.info("Strategy 1: Trying pdfplumber extraction...")
    extracted_text = _extract_with_pdfplumber(file_bytes)
    is_good, assessment = _assess_extraction_quality(extracted_text)

    if is_good:
        logger.info(f"✓ pdfplumber extraction succeeded: {assessment}")
        metadata["extraction_method"] = "pdf_text"
        metadata["quality_assessment"] = assessment
    else:
        logger.warning(f"✗ pdfplumber extraction failed: {assessment}")
        logger.info("Strategy 2: Falling back to OCR...")

        if not config.ENABLE_OCR_FALLBACK:
            raise EvidenceIngestError(
                "OCR_DISABLED",
                "OCR fallback is disabled by configuration.",
                details={"chars_extracted": len(extracted_text), "chars_required": MIN_CHARS_FOR_OCR}
            )

        try:
            logger.info("Attempting OCR with PyMuPDF rendering...")
            extracted_text, ocr_method, ocr_pages = _extract_with_ocr_pymupdf(
                file_bytes,
                ocr,
                max_pages=config.OCR_MAX_PAGES,
                dpi=config.OCR_DPI
            )
            metadata["ocr_pages"] = ocr_pages
            is_good, assessment = _assess_extraction_quality(extracted_text)

            if is_good:
                logger.info(f"✓ OCR extraction succeeded: {assessment}")
                metadata["extraction_method"] = ocr_method
                metadata["quality_assessment"] = assessment
            else:
                logger.warning(f"✗ OCR extraction insufficient: {assessment}")
                metadata["extraction_method"] = ocr_method
                metadata["quality_assessment"] = f"OCR incomplete: {assessment}"
        except EvidenceIngestError:
            raise
        except Exception as e:
            logger.error(f"OCR extraction failed with error: {e}")
            raise EvidenceIngestError(
                "OCR_FAILED",
                f"OCR extraction failed: {str(e)}",
                details={"error": str(e)}
            )
    
    # Clean text and compute final metrics
    extracted_text, clean_diag = clean_extracted_text(extracted_text)
    chars = len(extracted_text)
    letters = _count_letters(extracted_text)
    words = _count_words(extracted_text)
    alpha_ratio = _compute_alphabetic_ratio(extracted_text)
    
    metadata["chars_extracted"] = chars
    metadata["words"] = words
    metadata["alphabetic_ratio"] = alpha_ratio
    metadata["method"] = metadata["extraction_method"]
    metadata["diagnostics"] = {
        "method": metadata["extraction_method"],
        "chars_extracted": chars,
        "word_count": words,
        "alphabetic_ratio": alpha_ratio,
        "unique_chars": len(set(extracted_text)),
        "ocr_pages": metadata.get("ocr_pages", 0),
        "quality_assessment": metadata["quality_assessment"],
        "removed_lines_count": clean_diag.removed_lines_count,
        "removed_by_regex": clean_diag.removed_by_regex,
        "removed_repeated_lines_count": clean_diag.removed_repeated_lines_count,
        "top_removed_lines": clean_diag.top_removed_lines,
        "repeat_threshold_used": clean_diag.repeat_threshold_used
    }
    metadata["result"] = {
        "text": extracted_text,
        "method": metadata["extraction_method"],
        "diagnostics": metadata["diagnostics"]
    }
    
    logger.info(
        f"Final extraction: {words} words, {chars} chars, "
        f"alpha_ratio={alpha_ratio:.3f}, method={metadata['extraction_method']}, "
        f"ocr_pages={metadata.get('ocr_pages', 0)}"
    )
    if clean_diag.removed_lines_count > 0:
        logger.info(
            "Text cleaning removed %s lines (repeated=%s).",
            clean_diag.removed_lines_count,
            clean_diag.removed_repeated_lines_count
        )
    
    # Backward compatibility: also include old metadata keys
    metadata["extraction_method_used"] = metadata["extraction_method"]
    metadata["pages"] = metadata["num_pages"]
    metadata["letters"] = letters
    
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


def _extract_with_ocr_pymupdf(
    file_bytes: bytes,
    ocr,
    max_pages: int = None,
    dpi: int = 200
) -> Tuple[str, str, int]:
    """
    Extract text using OCR on PDF pages rendered with PyMuPDF.
    
    This method uses PyMuPDF (fitz) to render PDF pages to images,
    avoiding the need for pdf2image and Poppler dependencies.
    
    Args:
        file_bytes: PDF file bytes
        ocr: ImageOCR instance
        max_pages: Maximum pages to process (None = all pages)
    
    Returns:
        Tuple of (extracted_text, method_name, pages_processed)
    """
    doc = None
    try:
        import fitz
        from PIL import Image
        try:
            import pytesseract
        except Exception:
            pytesseract = None

        if pytesseract is None and ocr is None:
            raise EvidenceIngestError(
                "OCR_UNAVAILABLE",
                "OCR fallback requested but neither pytesseract nor EasyOCR is available."
            )
        
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        total_pages = len(doc)
        pages_to_process = min(max_pages, total_pages) if max_pages else total_pages
        
        text_parts = []
        method_used = "ocr_pymupdf_tesseract" if pytesseract is not None else "ocr_easyocr"
        logger.info(f"OCR: Processing {pages_to_process} pages with PyMuPDF rendering...")
        
        for page_num in range(pages_to_process):
            page = doc[page_num]
            
            try:
                # Render page to pixmap at requested DPI for OCR quality
                zoom = dpi / 72
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                
                # Convert pixmap to PIL Image
                img_bytes = pix.tobytes("png")
                image = Image.open(BytesIO(img_bytes))
                
                # Save to temporary file for OCR when needed
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                    image.save(tmp, format="PNG")
                    tmp_path = tmp.name
                
                try:
                    logger.info(f"  OCRing page {page_num + 1}/{pages_to_process}...")
                    text = ""
                    method = None

                    if pytesseract is not None:
                        try:
                            text = pytesseract.image_to_string(image)
                            method = "ocr_pymupdf_tesseract"
                        except Exception as e:
                            logger.warning(f"    Tesseract OCR failed on page {page_num + 1}: {e}")

                    if (not text or not text.strip()) and ocr is not None:
                        result = ocr.extract_text_from_image(tmp_path, image_type="notes")
                        if isinstance(result, dict):
                            text = result.get("text", "")
                        else:
                            text = str(result)
                        method = "ocr_easyocr"
                    
                    if text and text.strip():
                        text_parts.append(f"[Page {page_num + 1}]\n{text.strip()}")
                        logger.info(f"    ✓ Extracted {len(text)} chars from page {page_num + 1}")
                        if method:
                            method_used = method
                    else:
                        logger.warning(f"    ⚠ No text found on page {page_num + 1}")
                        
                finally:
                    # Clean up temporary file
                    Path(tmp_path).unlink(missing_ok=True)
                    
            except Exception as e:
                logger.warning(f"OCR failed for page {page_num + 1}: {e}")
                continue
        
        extracted_text = "\n\n".join(text_parts)
        logger.info(f"OCR (PyMuPDF): Extracted {len(extracted_text)} chars from {len(text_parts)} pages")

        return extracted_text, method_used, pages_to_process
    except Exception as e:
        logger.error(f"OCR extraction failed: {e}")
        raise EvidenceIngestError(
            "OCR_FAILED",
            f"OCR extraction failed: {str(e)}",
            details={"error": str(e)}
        )
    finally:
        if doc is not None:
            try:
                doc.close()
            except Exception:
                pass


def _clean_text(text: str) -> str:
    """Deprecated: kept for backward compatibility."""
    cleaned, _ = clean_extracted_text(text)
    return cleaned


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
