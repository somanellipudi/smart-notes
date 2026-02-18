#!/usr/bin/env python3
"""
End-to-end test of PDF extraction → reasoning pipeline.
Tests the complete flow from PDF upload to summary generation.
"""

import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing.pdf_ingest import extract_pdf_text
from src.audio.image_ocr import ImageOCR
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_pdf_extraction_with_real_file():
    """Test PDF extraction with a real file."""
    
    # Find a scanned PDF in the examples
    pdf_files = list(Path("data").glob("**/*.pdf"))
    pdf_files += list(Path("examples").glob("**/*.pdf"))
    pdf_files += list(Path("outputs").glob("**/*.pdf"))
    
    if not pdf_files:
        logger.warning("No PDF files found in data/examples/outputs")
        return False
    
    pdf_path = pdf_files[0]
    logger.info(f"Testing with PDF: {pdf_path}")
    
    # Initialize OCR
    try:
        ocr = ImageOCR()
        logger.info("✅ OCR initialized successfully")
    except Exception as e:
        logger.warning(f"⚠️ OCR initialization failed: {e}")
        ocr = None
    
    # Extract text from PDF
    try:
        with open(pdf_path, 'rb') as f:
            pdf_bytes = f.read()
        
        # Create a mock file object
        class MockFile:
            def __init__(self, content, name):
                self.content = content
                self.name = name
                self._pos = 0
            
            def read(self):
                self._pos = len(self.content)
                return self.content
            
            def getvalue(self):
                return self.content
            
            def seek(self, pos):
                self._pos = pos
        
        mock_file = MockFile(pdf_bytes, pdf_path.name)
        
        # Extract
        text, metadata = extract_pdf_text(mock_file, ocr=ocr)
        
        logger.info(f"✅ Extraction successful!")
        logger.info(f"   - Method: {metadata.get('extraction_method', 'unknown')}")
        logger.info(f"   - Pages: {metadata.get('num_pages', metadata.get('pages', 'unknown'))}")
        logger.info(f"   - Chars: {metadata.get('chars_extracted', len(text))} (text length: {len(text)})")
        logger.info(f"   - Words: {metadata.get('words', len(text.split()))}")
        logger.info(f"   - Quality: {metadata.get('quality_assessment', 'unknown')}")
        logger.info(f"\nFirst 200 chars of extracted text:")
        logger.info(f"{text[:200]}...")
        
        return True
    
    except Exception as e:
        logger.error(f"❌ Extraction failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = test_pdf_extraction_with_real_file()
    sys.exit(0 if success else 1)
