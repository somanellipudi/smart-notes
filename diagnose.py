#!/usr/bin/env python
"""
Debug script to help diagnose Streamlit Cloud deployment issues.
Run locally to verify all dependencies and configurations.
"""

import sys
import os
from pathlib import Path

print("=" * 80)
print("Smart Notes - Deployment Diagnostics")
print("=" * 80)

# 1. Check Python version
print(f"\n✓ Python Version: {sys.version}")

# 2. Check critical dependencies
print("\nChecking critical dependencies...")
required_packages = [
    'streamlit',
    'openai',
    'pydantic',
    'easyocr',
    'torch',
    'PIL',
    'numpy',
    'nltk',
    'spacy',
]

failed = []
for pkg in required_packages:
    try:
        __import__(pkg.replace('-', '_'))
        print(f"  ✓ {pkg}")
    except ImportError as e:
        print(f"  ✗ {pkg} - {e}")
        failed.append(pkg)

# 3. Check environment variables
print("\nEnvironment Variables:")
env_vars = ['OPENAI_API_KEY', 'LANGCHAIN_API_KEY']
for var in env_vars:
    if var in os.environ:
        val = os.environ[var]
        masked = val[:10] + "..." if len(val) > 10 else val
        print(f"  ✓ {var}: {masked}")
    else:
        print(f"  ✗ {var}: NOT SET")

# 4. Check key files
print("\nChecking project files...")
files_to_check = [
    'app.py',
    'config.py',
    'requirements.txt',
    'packages.txt',
    '.streamlit/config.toml',
    'src/reasoning/pipeline.py',
    'src/audio/image_ocr.py',
]

for file in files_to_check:
    path = Path(file)
    if path.exists():
        print(f"  ✓ {file}")
    else:
        print(f"  ✗ {file} - NOT FOUND")

# 5. Try importing main modules
print("\nTesting module imports...")
sys.path.insert(0, str(Path(__file__).parent))

try:
    from src.audio.image_ocr import ImageOCR
    print("  ✓ src.audio.image_ocr")
except Exception as e:
    print(f"  ✗ src.audio.image_ocr - {e}")

try:
    from src.reasoning.pipeline import ReasoningPipeline
    print("  ✓ src.reasoning.pipeline")
except Exception as e:
    print(f"  ✗ src.reasoning.pipeline - {e}")

try:
    from src.preprocessing.text_processing import preprocess_classroom_content
    print("  ✓ src.preprocessing.text_processing")
except Exception as e:
    print(f"  ✗ src.preprocessing.text_processing - {e}")

# 6. Try OCR initialization
print("\nTesting OCR initialization...")
try:
    ocr = ImageOCR()
    print("  ✓ ImageOCR initialized successfully")
except Exception as e:
    print(f"  ✗ ImageOCR initialization failed: {e}")

# Summary
print("\n" + "=" * 80)
if failed:
    print(f"⚠️  ISSUES FOUND: {len(failed)} missing packages")
    print(f"   Install with: pip install {' '.join(failed)}")
else:
    print("✅ All checks passed! Ready for deployment.")
print("=" * 80)
