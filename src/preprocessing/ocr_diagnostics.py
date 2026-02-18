"""
CLI diagnostics for OCR backend availability.

Usage:
    python -m src.preprocessing.ocr_diagnostics
"""

import importlib.util
import sys


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> int:
    modules = {
        "easyocr": _has_module("easyocr"),
        "pytesseract": _has_module("pytesseract"),
        "bidi": _has_module("bidi"),
        "PIL": _has_module("PIL"),
        "torch": _has_module("torch"),
    }

    print("OCR diagnostics:")
    for name, available in modules.items():
        status = "OK" if available else "MISSING"
        print(f"- {name}: {status}")

    if modules["easyocr"]:
        print("EasyOCR backend: available")
    elif modules["pytesseract"]:
        print("EasyOCR backend: missing; pytesseract fallback available")
    else:
        print("No OCR backend available. Install easyocr or pytesseract.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
