"""
PDF ingestion module with OCR fallback support.

Handles PDF text extraction with quality assessment and optional OCR fallback
for corrupted PDFs (detected via CID glyphs).
"""

import logging
import io
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import config
from src.preprocessing.text_quality import compute_text_quality, log_quality_report

logger = logging.getLogger(__name__)


@dataclass
class PdfExtractionResult:
    """Result of PDF extraction attempt."""
    success: bool
    text: str
    method: str  # "pdfplumber", "ocr", or "failed"
    page_count: int
    quality_report: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class PdfIngestor:
    """
    Handles PDF ingestion with quality-aware text extraction and OCR fallback.
    
    Strategy:
    1. Try pdfplumber for fast text extraction
    2. Assess text quality
    3. If quality fails and CID ratio high → try OCR fallback
    4. Return best extraction result
    """
    
    def __init__(self, enable_ocr_fallback: bool = None, max_ocr_pages: int = None):
        """
        Initialize PDF ingestor.
        
        Args:
            enable_ocr_fallback: Enable OCR if text extraction fails (default: config)
            max_ocr_pages: Maximum pages to OCR (default: config)
        """
        self.enable_ocr_fallback = enable_ocr_fallback if enable_ocr_fallback is not None else config.ENABLE_OCR_FALLBACK
        self.max_ocr_pages = max_ocr_pages if max_ocr_pages is not None else config.OCR_MAX_PAGES
        self.ocr_available = self._check_ocr_availability()
        
        if self.enable_ocr_fallback and not self.ocr_available:
            logger.warning(
                "OCR fallback enabled but OCR not available. "
                "Install: pip install pdf2image pytesseract"
            )
    
    def _check_ocr_availability(self) -> bool:
        """Check if required OCR libraries are available."""
        try:
            import pdf2image  # noqa: F401
            import pytesseract  # noqa: F401
            return True
        except ImportError:
            return False
    
    def extract_text(self, pdf_file) -> PdfExtractionResult:
        """
        Extract text from PDF with quality assessment and OCR fallback.
        
        Args:
            pdf_file: File-like object or path to PDF
            
        Returns:
            PdfExtractionResult with extraction details
        """
        try:
            # Phase 1: Try standard text extraction
            result = self._extract_with_pdfplumber(pdf_file)
            
            if not result.success:
                logger.warning(f"pdfplumber extraction failed: {result.error}")
                if self.enable_ocr_fallback and self.ocr_available:
                    logger.info("Attempting OCR fallback...")
                    result = self._extract_with_ocr(pdf_file)
                return result
            
            # Phase 2: Assess extracted text quality
            quality = compute_text_quality(result.text)
            result.quality_report = {
                'alphabetic_ratio': quality.alphabetic_ratio,
                'cid_ratio': quality.cid_ratio,
                'printable_ratio': quality.printable_ratio,
                'passes_quality': quality.passes_quality,
                'failure_reasons': quality.failure_reasons
            }
            
            log_quality_report(quality, context="PDF extraction")
            
            # Phase 3: If quality fails and CID ratio high, try OCR fallback
            if not quality.passes_quality and quality.cid_ratio > config.MAX_CID_RATIO:
                if self.enable_ocr_fallback and self.ocr_available:
                    logger.info("Detected corrupted PDF (high CID ratio). Attempting OCR fallback...")
                    ocr_result = self._extract_with_ocr(pdf_file)
                    
                    # Compare results
                    if ocr_result.success:
                        ocr_quality = compute_text_quality(ocr_result.text)
                        if ocr_quality.passes_quality or len(ocr_result.text) > len(result.text):
                            logger.info("OCR extraction preferred over pdfplumber")
                            return ocr_result
            
            return result
            
        except Exception as e:
            logger.error(f"PDF extraction error: {str(e)}")
            return PdfExtractionResult(
                success=False,
                text="",
                method="failed",
                page_count=0,
                error=str(e)
            )
    
    def _extract_with_pdfplumber(self, pdf_file) -> PdfExtractionResult:
        """Extract text using pdfplumber."""
        try:
            import pdfplumber
            
            # Handle file-like objects
            if hasattr(pdf_file, 'read'):
                pdf_bytes = pdf_file.read()
                pdf_file = io.BytesIO(pdf_bytes)
            
            with pdfplumber.open(pdf_file) as pdf:
                text = ""
                for page_num, page in enumerate(pdf.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += f"--- Page {page_num + 1} ---\n{page_text}\n"
                    except Exception as e:
                        logger.warning(f"Failed to extract page {page_num + 1}: {str(e)}")
                
                if not text.strip():
                    return PdfExtractionResult(
                        success=False,
                        text="",
                        method="pdfplumber",
                        page_count=len(pdf.pages),
                        error="No text extracted"
                    )
                
                return PdfExtractionResult(
                    success=True,
                    text=text,
                    method="pdfplumber",
                    page_count=len(pdf.pages)
                )
        
        except ImportError:
            return PdfExtractionResult(
                success=False,
                text="",
                method="pdfplumber",
                page_count=0,
                error="pdfplumber not installed"
            )
        except Exception as e:
            return PdfExtractionResult(
                success=False,
                text="",
                method="pdfplumber",
                page_count=0,
                error=str(e)
            )
    
    def _extract_with_ocr(self, pdf_file) -> PdfExtractionResult:
        """Extract text using OCR (pdf2image + pytesseract)."""
        try:
            from pdf2image import convert_from_bytes, convert_from_path
            import pytesseract
            
            logger.info(f"Starting OCR extraction (max {self.max_ocr_pages} pages)...")
            
            # Convert PDF to images
            try:
                if hasattr(pdf_file, 'read'):
                    pdf_bytes = pdf_file.read()
                    images = convert_from_bytes(pdf_bytes)
                else:
                    images = convert_from_path(pdf_file)
            except Exception as e:
                return PdfExtractionResult(
                    success=False,
                    text="",
                    method="ocr",
                    page_count=0,
                    error=f"PDF to image conversion failed: {str(e)}"
                )
            
            # OCR each image (up to max_ocr_pages)
            text = ""
            for page_num, image in enumerate(images[:self.max_ocr_pages]):
                try:
                    logger.info(f"OCR processing page {page_num + 1}/{min(len(images), self.max_ocr_pages)}...")
                    page_text = pytesseract.image_to_string(image)
                    if page_text.strip():
                        text += f"--- Page {page_num + 1} (OCR) ---\n{page_text}\n"
                except Exception as e:
                    logger.warning(f"OCR failed for page {page_num + 1}: {str(e)}")
                    continue
            
            if not text.strip():
                return PdfExtractionResult(
                    success=False,
                    text="",
                    method="ocr",
                    page_count=len(images),
                    error="OCR extracted no text"
                )
            
            logger.info(f"OCR extraction complete: {len(text)} characters from {min(len(images), self.max_ocr_pages)} pages")
            
            return PdfExtractionResult(
                success=True,
                text=text,
                method="ocr",
                page_count=len(images)
            )
        
        except ImportError:
            return PdfExtractionResult(
                success=False,
                text="",
                method="ocr",
                page_count=0,
                error="pdf2image or pytesseract not installed. Install with: pip install pdf2image pytesseract"
            )
        except Exception as e:
            return PdfExtractionResult(
                success=False,
                text="",
                method="ocr",
                page_count=0,
                error=f"OCR extraction error: {str(e)}"
            )
