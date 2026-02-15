#!/usr/bin/env python3
"""
Practical demonstration of PDF and URL ingestion capabilities.

This script shows real-world usage of the new ingestion modules
and validates they work correctly.

Usage:
    python test_ingestion_practical.py --pdf path/to/file.pdf
    python test_ingestion_practical.py --url "https://youtube.com/watch?v=..."
    python test_ingestion_practical.py --demo
"""

import sys
import argparse
import logging
from pathlib import Path
from io import BytesIO

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing.pdf_ingest import extract_pdf_text, _assess_extraction_quality
from src.preprocessing.url_ingest import fetch_url_text, _is_youtube_url
import config


def test_pdf_file(pdf_path: str):
    """Test PDF extraction with a real file."""
    print("\n" + "="*70)
    print("PDF EXTRACTION TEST")
    print("="*70)
    
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        print(f"ERROR: File not found: {pdf_path}")
        return False
    
    print(f"File: {pdf_path.name}")
    print(f"Size: {pdf_path.stat().st_size / 1024:.1f} KB")
    
    try:
        # Open and extract
        with open(pdf_path, 'rb') as f:
            pdf_file = type('obj', (object,), {
                'read': f.read,
                'name': pdf_path.name,
                'type': 'application/pdf',
                'getvalue': lambda: f.read() if hasattr(f, 'read') else b''
            })()
        
        print("\n[*] Attempting extraction...")
        
        # Try extraction
        from src.preprocessing.pdf_ingest import extract_pdf_text as extract_func
        
        # Create a mock file object
        class MockFile:
            def __init__(self, path):
                self.path = path
                self.name = path.name
                self.type = "application/pdf"
            
            def read(self):
                with open(self.path, 'rb') as f:
                    return f.read()
            
            def getvalue(self):
                return self.read()
            
            def seek(self, pos):
                pass
        
        mock_file = MockFile(pdf_path)
        text, metadata = extract_func(mock_file, ocr=None)
        
        # Display results
        print(f"\nExtractionMethod: {metadata.get('extraction_method', 'unknown')}")
        print(f"Pages: {metadata.get('pages', 'unknown')}")
        print(f"Characters: {len(text)}")
        print(f"Words: {len(text.split())}")
        
        if text:
            # Assess quality
            is_good, reason = _assess_extraction_quality(text)
            print(f"Quality: {'PASS' if is_good else 'FAIL'}")
            print(f"Quality Reason: {reason}")
            
            # Show preview
            print(f"\nContent Preview (first 300 chars):")
            print("-" * 50)
            preview = text[:300]
            if len(text) > 300:
                preview += "...[truncated]"
            print(preview)
            print("-" * 50)
            
            success = is_good
        else:
            print("Quality: FAIL")
            print("Quality Reason: No text extracted")
            success = False
        
        return success
        
    except Exception as e:
        print(f"ERROR during extraction: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_url_ingestion(url: str):
    """Test URL ingestion (YouTube or article)."""
    print("\n" + "="*70)
    print("URL INGESTION TEST")
    print("="*70)
    
    print(f"URL: {url}")
    
    # Validate URL
    if _is_youtube_url(url):
        print("Detected: YouTube Video")
    else:
        print("Detected: Web Article")
    
    try:
        print("\n[*] Fetching content...")
        
        text, metadata = fetch_url_text(url)
        
        # Display results
        print(f"\nSource Type: {metadata.get('source_type', 'unknown')}")
        print(f"Title: {metadata.get('title', 'N/A')}")
        print(f"Extraction Method: {metadata.get('extraction_method', 'unknown')}")
        print(f"Characters: {len(text)}")
        print(f"Words: {len(text.split())}")
        
        if text:
            # Show preview
            print(f"\nContent Preview (first 300 chars):")
            print("-" * 50)
            preview = text[:300]
            if len(text) > 300:
                preview += "...[truncated]"
            print(preview)
            print("-" * 50)
            
            success = len(text) > 100
        else:
            print("\nNo content extracted")
            if "error" in metadata:
                print(f"Error: {metadata['error']}")
            success = False
        
        return success
        
    except Exception as e:
        print(f"ERROR during URL ingestion: {e}")
        import traceback
        traceback.print_exc()
        return False


def demo_modes():
    """Demonstrate quality assessment modes."""
    print("\n" + "="*70)
    print("QUALITY ASSESSMENT DEMO")
    print("="*70)
    
    test_cases = [
        ("Good educational content", 
         " ".join(["Calculus is the mathematical study of continuous change."] * 30),
         True),
        
        ("Corrupted PDF with CID glyphs",
         "(cid:1) (cid:2) (cid:3) " * 200,
         False),
        
        ("Too few words",
         " ".join(["word"] * 50),
         False),
        
        ("Too many numbers",
         " ".join(["123 456 789"] * 100),
         False),
        
        ("Good calculus content",
         """The derivative measures how a function changes as its input changes.
            For a function f(x), the derivative f'(x) represents the rate of change
            at each point. Integration is the reverse process of differentiation.
            It accumulates small changes to find total quantities like area under curves.
            These fundamental operations form the basis of calculus.""" * 5,
         True),
    ]
    
    for name, text, expected in test_cases:
        is_good, reason = _assess_extraction_quality(text)
        status = "PASS" if is_good == expected else "FAIL"
        result = "GOOD" if is_good else "REJECTED"
        
        print(f"\n[{status}] {name}")
        print(f"     Result: {result}")
        print(f"     Reason: {reason}")


def check_config():
    """Display relevant configuration."""
    print("\n" + "="*70)
    print("CONFIGURATION STATUS")
    print("="*70)
    
    print(f"ENABLE_URL_SOURCES: {config.ENABLE_URL_SOURCES}")
    print(f"ENABLE_OCR_FALLBACK: {config.ENABLE_OCR_FALLBACK}")
    print(f"MIN_INPUT_CHARS_ABSOLUTE: {config.MIN_INPUT_CHARS_ABSOLUTE}")
    print(f"MIN_INPUT_CHARS_FOR_VERIFICATION: {config.MIN_INPUT_CHARS_FOR_VERIFICATION}")
    
    print(f"\nPDF Quality Thresholds:")
    print(f"  MIN_WORDS: 80")
    print(f"  MIN_LETTERS: 400")
    print(f"  MIN_ALPHA_RATIO: 0.30 (30%)")


def main():
    parser = argparse.ArgumentParser(
        description="Test PDF and URL ingestion capabilities"
    )
    parser.add_argument("--pdf", help="Test PDF extraction from file")
    parser.add_argument("--url", help="Test URL ingestion")
    parser.add_argument("--demo", action="store_true", help="Run quality assessment demo")
    parser.add_argument("--config", action="store_true", help="Show configuration")
    
    args = parser.parse_args()
    
    # Show config first
    check_config()
    
    if args.pdf:
        success = test_pdf_file(args.pdf)
        exit(0 if success else 1)
    
    elif args.url:
        success = test_url_ingestion(args.url)
        exit(0 if success else 1)
    
    elif args.demo:
        demo_modes()
        exit(0)
    
    else:
        # Run all demos
        demo_modes()
        
        print("\n" + "="*70)
        print("USAGE EXAMPLES")
        print("="*70)
        print("\nTest with your own files:")
        print("  python test_ingestion_practical.py --pdf path/to/lecture.pdf")
        print("  python test_ingestion_practical.py --url 'https://youtube.com/watch?v=...'")
        print("  python test_ingestion_practical.py --url 'https://example.com/article'")
        print("\nRun quality assessment demo:")
        print("  python test_ingestion_practical.py --demo")


if __name__ == '__main__':
    main()
