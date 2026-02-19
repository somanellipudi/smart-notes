"""
Test script to verify Streamlit Cloud detection and OCR configuration.

Run this to check if OCR will be enabled in your environment:
    python test_ocr_config.py
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Test detection before importing config
print("=" * 60)
print("STREAMLIT CLOUD DETECTION TEST")
print("=" * 60)

# Check environment variables
print("\n1. Environment Variables:")
print(f"   STREAMLIT_SHARING: {os.getenv('STREAMLIT_SHARING', 'not set')}")
print(f"   STREAMLIT_CLOUD: {os.getenv('STREAMLIT_CLOUD', 'not set')}")
print(f"   OCR_ENABLED: {os.getenv('OCR_ENABLED', 'not set')}")

# Check home path
home = os.path.expanduser("~")
print(f"\n2. Home Path: {home}")
print(f"   Contains /home/appuser: {'/home/appuser' in home}")
print(f"   Contains \\appuser: {'\\\\appuser' in home}")

# Import config and check detection
import config

print(f"\n3. Detection Result:")
print(f"   IS_STREAMLIT_CLOUD: {config.IS_STREAMLIT_CLOUD}")
print(f"   OCR_ENABLED: {config.OCR_ENABLED}")
print(f"   ENABLE_OCR_FALLBACK: {config.ENABLE_OCR_FALLBACK}")

print("\n" + "=" * 60)
if config.IS_STREAMLIT_CLOUD:
    print("✓ Detected: STREAMLIT CLOUD environment")
    if config.OCR_ENABLED:
        print("✓ OCR is ENABLED (explicitly set via OCR_ENABLED=true)")
    else:
        print("✓ OCR is DISABLED (default for Cloud)")
else:
    print("✓ Detected: LOCAL environment")
    if config.OCR_ENABLED:
        print("✓ OCR is ENABLED (default for local)")
    else:
        print("✓ OCR is DISABLED (explicitly set via OCR_ENABLED=false)")

print("=" * 60)

# Check if OCR packages are available
print("\n4. OCR Package Availability:")
try:
    import easyocr
    print("   ✓ EasyOCR installed")
except ImportError:
    print("   ✗ EasyOCR not installed")

try:
    import pytesseract
    print("   ✓ Pytesseract installed")
except ImportError:
    print("   ✗ Pytesseract not installed")

try:
    from PIL import Image
    print("   ✓ PIL/Pillow installed")
except ImportError:
    print("   ✗ PIL/Pillow not installed")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
