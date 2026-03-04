#!/usr/bin/env python3
"""
PDF Text Extraction Verification for IEEE Access Submission
============================================================
Builds the manuscript PDF and verifies clean text extraction without Unicode
replacement characters ("￾", "�") or other artifacts.

Requirements:
    - pdflatex (TeX Live, MiKTeX, or Overleaf-compatible)
    - pdftotext (poppler-utils package)

Installation (Linux/Mac):
    sudo apt-get install poppler-utils  # Debian/Ubuntu
    brew install poppler                # macOS

Installation (Windows):
    Download poppler from: https://github.com/oschwartz10612/poppler-windows/releases
    Add bin/ folder to PATH

Usage:
    python scripts/verify_pdf_text.py                     # Build + verify main PDF
    python scripts/verify_pdf_text.py --pdf-only my.pdf   # Verify existing PDF (no build)
    python scripts/verify_pdf_text.py --keep-temp         # Keep intermediate .aux/.log files

Author: CalibraTeach Team
Date: March 4, 2026
"""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple, Optional


# ============================================================================
# BAD GLYPH PATTERNS (indicate PDF text extraction failures)
# ============================================================================
BAD_GLYPHS = [
    '\ufffd',  # � REPLACEMENT CHARACTER
    '\ufffe',  # Noncharacter
    '\uffff',  # Noncharacter
    '□',       # Empty box (missing glyph)
    '￾',       # Common PDF artifact from bad Unicode mapping
]

# Regex patterns for common broken words (soft hyphen artifacts)
BAD_PATTERNS = [
    r'of\ufffdten',      # "often" broken by soft hyphen
    r'in\ufffdformation', # "information"
    r're\ufffdfer',      # "refer"
    r'dif\ufffdferent',  # "different"
    r'per\ufffdformance', # "performance"
]

# Context window for reporting (characters before/after bad glyph)
CONTEXT_CHARS = 40


def check_dependencies() -> bool:
    """Check if required tools (pdflatex, pdftotext) are available."""
    missing = []
    
    if not shutil.which('pdflatex'):
        missing.append('pdflatex (install TeX Live, MiKTeX, or use Overleaf)')
    
    if not shutil.which('pdftotext'):
        missing.append('pdftotext (install poppler-utils)')
    
    if missing:
        print("ERROR: Missing required dependencies:", file=sys.stderr)
        for tool in missing:
            print(f"  - {tool}", file=sys.stderr)
        return False
    
    return True


def build_pdf(tex_file: Path, keep_temp: bool = False) -> Optional[Path]:
    """
    Build PDF using pdflatex (3 passes for cross-references).
    
    Returns:
        Path to generated PDF, or None if build failed
    """
    print(f"Building PDF from {tex_file}...")
    
    # Change to directory containing .tex file
    original_dir = Path.cwd()
    work_dir = tex_file.parent
    tex_name = tex_file.name
    
    try:
        # Run pdflatex 3 times (for cross-refs, TOC, bibliography)
        for pass_num in range(1, 4):
            print(f"  Pass {pass_num}/3: pdflatex {tex_name}")
            result = subprocess.run(
                ['pdflatex', '-interaction=nonstopmode', tex_name],
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode != 0:
                print(f"\nERROR: pdflatex failed (pass {pass_num}):", file=sys.stderr)
                # Extract errors from log
                log_file = work_dir / tex_file.with_suffix('.log').name
                if log_file.exists():
                    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                        log_content = f.read()
                        # Find error messages
                        errors = re.findall(r'! .*', log_content)
                        for err in errors[:5]:  # Show first 5 errors
                            print(f"  {err}", file=sys.stderr)
                return None
        
        pdf_path = work_dir / tex_file.with_suffix('.pdf').name
        
        if not pdf_path.exists():
            print("ERROR: PDF was not generated", file=sys.stderr)
            return None
        
        print(f"✓ PDF built successfully: {pdf_path}")
        
        # Clean up temp files unless --keep-temp
        if not keep_temp:
            for ext in ['.aux', '.log', '.out', '.toc', '.bbl', '.blg']:
                temp_file = work_dir / tex_file.with_suffix(ext).name
                if temp_file.exists():
                    temp_file.unlink()
        
        return pdf_path
    
    except subprocess.TimeoutExpired:
        print("ERROR: pdflatex timed out after 120s", file=sys.stderr)
        return None
    except Exception as e:
        print(f"ERROR: Build failed: {e}", file=sys.stderr)
        return None


def extract_text(pdf_path: Path) -> Optional[str]:
    """Extract text from PDF using pdftotext."""
    print(f"Extracting text from {pdf_path}...")
    
    try:
        result = subprocess.run(
            ['pdftotext', '-enc', 'UTF-8', str(pdf_path), '-'],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            print(f"ERROR: pdftotext failed: {result.stderr}", file=sys.stderr)
            return None
        
        text = result.stdout
        print(f"✓ Extracted {len(text)} characters from PDF")
        return text
    
    except subprocess.TimeoutExpired:
        print("ERROR: pdftotext timed out", file=sys.stderr)
        return None
    except FileNotFoundError:
        print("ERROR: pdftotext not found. Install poppler-utils.", file=sys.stderr)
        return None
    except Exception as e:
        print(f"ERROR: Text extraction failed: {e}", file=sys.stderr)
        return None


def find_bad_glyphs(text: str) -> List[Tuple[str, int, str]]:
    """
    Find bad glyphs and broken word patterns in extracted text.
    
    Returns:
        List of (glyph/pattern, position, context) tuples
    """
    issues = []
    
    # Check for individual bad glyphs
    for bad_glyph in BAD_GLYPHS:
        pos = 0
        while True:
            pos = text.find(bad_glyph, pos)
            if pos == -1:
                break
            
            # Extract context
            start = max(0, pos - CONTEXT_CHARS)
            end = min(len(text), pos + CONTEXT_CHARS)
            context = text[start:end]
            
            issues.append((bad_glyph, pos, context))
            pos += 1
    
    # Check for broken word patterns
    for pattern in BAD_PATTERNS:
        for match in re.finditer(pattern, text):
            pos = match.start()
            start = max(0, pos - CONTEXT_CHARS)
            end = min(len(text), pos + CONTEXT_CHARS + 20)
            context = text[start:end]
            
            issues.append((match.group(0), pos, context))
    
    return issues


def estimate_page_number(text: str, position: int) -> int:
    """
    Rough estimate of page number based on character position.
    Assumes ~3000 chars/page (typical for IEEE 2-column).
    """
    return (position // 3000) + 1


def main():
    parser = argparse.ArgumentParser(
        description='Build PDF and verify clean text extraction (no Unicode artifacts)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Build OVERLEAF_TEMPLATE.tex + verify
  %(prog)s --tex my_paper.tex           # Build specific .tex file
  %(prog)s --pdf-only existing.pdf      # Verify existing PDF (skip build)
  %(prog)s --keep-temp                  # Keep .aux/.log files after build
        """
    )
    parser.add_argument('--tex', type=Path, default='OVERLEAF_TEMPLATE.tex',
                        help='LaTeX source file to build (default: OVERLEAF_TEMPLATE.tex)')
    parser.add_argument('--pdf-only', type=Path,
                        help='Verify existing PDF without building (skip pdflatex)')
    parser.add_argument('--keep-temp', action='store_true',
                        help='Keep temporary LaTeX files (.aux, .log, etc.)')
    
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Determine PDF path
    if args.pdf_only:
        pdf_path = args.pdf_only
        if not pdf_path.exists():
            print(f"ERROR: PDF not found: {pdf_path}", file=sys.stderr)
            return 1
    else:
        # Build PDF
        tex_path = args.tex
        if not tex_path.exists():
            print(f"ERROR: TeX file not found: {tex_path}", file=sys.stderr)
            return 1
        
        pdf_path = build_pdf(tex_path, keep_temp=args.keep_temp)
        if pdf_path is None:
            return 1
    
    # Extract text
    text = extract_text(pdf_path)
    if text is None:
        return 1
    
    # Find bad glyphs
    print("\nChecking for bad glyphs and PDF artifacts...")
    issues = find_bad_glyphs(text)
    
    print(f"\n{'='*70}")
    if not issues:
        print("✓ PASS: PDF text extraction is clean!")
        print("  No replacement characters (￾, �) or broken words found.")
        print(f"\nPDF ready for IEEE Access submission: {pdf_path}")
        return 0
    else:
        print(f"✗ FAIL: Found {len(issues)} text extraction issues in PDF")
        print("\nOffending glyphs/patterns:\n")
        
        for glyph, pos, context in issues[:10]:  # Show first 10
            page_est = estimate_page_number(text, pos)
            # Make bad glyph visible
            display_glyph = repr(glyph) if len(glyph) == 1 and ord(glyph) > 127 else glyph
            print(f"  Page ~{page_est}, Char {pos}: {display_glyph} (U+{ord(glyph[0]):04X})")
            print(f"    Context: ...{repr(context)}...")
            print()
        
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more issues\n")
        
        print("DIAGNOSIS:")
        print("  - Check LaTeX preamble: \\input{glyphtounicode} + \\pdfgentounicode=1")
        print("  - Run: python scripts/sanitize_unicode.py --fix")
        print("  - Ensure T1 font encoding: \\usepackage[T1]{fontenc}")
        print("  - Rebuild PDF after fixes")
        return 1


if __name__ == '__main__':
    sys.exit(main())
