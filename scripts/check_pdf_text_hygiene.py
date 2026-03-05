#!/usr/bin/env python3
"""
PDF Text Extraction Hygiene Checker.

Checks two things:
1) Replacement/hyphen artifacts in extracted text
2) Banned embedded strings in canonical architecture.pdf

Exit codes:
  0 = OK
  1 = FAIL (hygiene violation)
  2 = ERROR (I/O/tooling error)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
    HAS_PDFMINER = True
except ImportError:
    HAS_PDFMINER = False


REPLACEMENT_ARTIFACTS = {
    "U+FFFD replacement character": chr(0xFFFD),
    "U+FFFE replacement marker": chr(0xFFFE),
    "U+FFFF invalid marker": chr(0xFFFF),
    "soft hyphen U+00AD": chr(0x00AD),
    "double-quote artifact \"\"": '""',
}

COMPILED_PDF_BANNED = {
    "U+FFFD replacement character": chr(0xFFFD),
    "U+FFFE replacement marker": chr(0xFFFE),
    "U+FFFF invalid marker": chr(0xFFFF),
}

ARCH_BANNED = [
    "CalibraTeach:",
    "GPU:",
    "PyTorch",
    "CUDA",
    "Transformers",
]


def _safe(msg: str) -> str:
    return msg.encode("ascii", "backslashreplace").decode("ascii")


def _extract_text_pypdf(pdf_path: Path):
    try:
        reader = PdfReader(str(pdf_path))
        chunks = []
        for page in reader.pages:
            chunks.append(page.extract_text() or "")
        return "\n".join(chunks)
    except Exception as exc:  # pragma: no cover - exercised in CLI
        print(_safe(f"[ERROR] pypdf extraction failed for {pdf_path}: {exc}"))
        return None


def _extract_text_pdfminer(pdf_path: Path):
    try:
        return pdfminer_extract_text(str(pdf_path))
    except Exception as exc:  # pragma: no cover - exercised in CLI
        print(_safe(f"[ERROR] pdfminer extraction failed for {pdf_path}: {exc}"))
        return None


def extract_text_from_pdf(pdf_path: Path):
    if HAS_PYPDF:
        text = _extract_text_pypdf(pdf_path)
        if text is not None:
            return text

    if HAS_PDFMINER:
        text = _extract_text_pdfminer(pdf_path)
        if text is not None:
            return text

    print("[ERROR] No PDF extractor available. Install pypdf or pdfminer.six")
    return None


def find_artifacts(text: str):
    found = []
    for label, token in REPLACEMENT_ARTIFACTS.items():
        count = text.count(token)
        if count:
            found.append((label, count))
    return found


def find_banned_architecture_strings(text: str):
    found = []
    for token in ARCH_BANNED:
        count = text.count(token)
        if count:
            found.append((token, count))
    return found


def find_compiled_pdf_artifacts(text: str):
    found = []
    for label, token in COMPILED_PDF_BANNED.items():
        count = text.count(token)
        if count:
            found.append((label, count))
    return found


def canonical_architecture_path(repo_root: Path) -> Path:
    return repo_root / "paper" / "figures" / "architecture.pdf"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail-fast PDF hygiene checker for replacement artifacts and architecture banner/spec text."
    )
    parser.add_argument(
        "pdf_path",
        nargs="?",
        default="paper/main.pdf",
        help="PDF to scan (default: paper/main.pdf)",
    )
    parser.add_argument(
        "--check-architecture",
        action="store_true",
        help="Also scan canonical architecture.pdf for banned embedded strings.",
    )
    parser.add_argument(
        "--check-compiled",
        action="store_true",
        help="Scan compiled paper PDF for replacement glyph artifacts (\ufffe/\ufffd/etc).",
    )
    args = parser.parse_args()

    if args.check_architecture and args.check_compiled:
        print("[ERROR] Use exactly one mode: --check-architecture OR --check-compiled")
        return 2

    repo_root = Path(__file__).resolve().parent.parent
    target_pdf = Path(args.pdf_path)

    if not target_pdf.exists():
        print(_safe(f"[ERROR] PDF not found: {target_pdf}"))
        return 2

    target_text = extract_text_from_pdf(target_pdf)
    if target_text is None:
        print(_safe(f"[ERROR] Failed to extract text from: {target_pdf}"))
        return 2

    if args.check_compiled:
        compiled_hits = find_compiled_pdf_artifacts(target_text)
        if compiled_hits:
            print(_safe(f"[ERROR] Compiled PDF hygiene violations in {target_pdf}:"))
            for label, count in compiled_hits:
                print(_safe(f"  - {label} (count={count})"))
            return 1
        print(_safe(f"[OK] Compiled PDF is clean: {target_pdf}"))
        return 0

    artifacts = find_artifacts(target_text)
    if artifacts:
        print(_safe(f"[ERROR] Hygiene artifacts detected in {target_pdf}:"))
        for label, count in artifacts:
            print(_safe(f"  - {label} (count={count})"))
        return 1

    print(_safe(f"[OK] No replacement artifacts found in {target_pdf}"))

    if args.check_architecture:
        if target_pdf.name.lower() == "architecture.pdf":
            arch_pdf = target_pdf
        else:
            arch_pdf = canonical_architecture_path(repo_root)

        if not arch_pdf.exists():
            print(_safe(f"[ERROR] Canonical architecture PDF missing: {arch_pdf}"))
            return 1

        arch_text = extract_text_from_pdf(arch_pdf)
        if arch_text is None:
            print(_safe(f"[ERROR] Failed to extract text from architecture PDF: {arch_pdf}"))
            return 2

        banned_hits = find_banned_architecture_strings(arch_text)
        if banned_hits:
            print(_safe(f"[ERROR] Banned embedded strings detected in {arch_pdf}:"))
            for token, count in banned_hits:
                print(_safe(f"  - {token} (count={count})"))
            return 1

        arch_artifacts = find_artifacts(arch_text)
        if arch_artifacts:
            print(_safe(f"[ERROR] Hygiene artifacts detected in {arch_pdf}:"))
            for item, count in arch_artifacts:
                print(_safe(f"  - {item} (count={count})"))
            return 1

        print(_safe(f"[OK] Architecture PDF is clean: {arch_pdf}"))

    return 0


if __name__ == "__main__":
    sys.exit(main())
