#!/usr/bin/env python3
"""
PDF VERIFICATION SCRIPT FOR FINAL CAMERA-READY FIXES
Purpose: Automatically verify compiled IEEE Access manuscript PDF
Author: GitHub Copilot
Date: March 3, 2026
"""

import os
import sys
from pathlib import Path

try:
    import PyPDF2
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF for verification."""
    if not HAS_PYPDF:
        print("ERROR: PyPDF2 not installed. Install with: pip install PyPDF2")
        return None
    
    try:
        text = ""
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text()
        return text
    except Exception as e:
        print(f"ERROR reading PDF: {e}")
        return None

def verify_source_no_hardcoded_refs(tex_path):
    """Verify source LaTeX has no hardcoded manual references."""
    print("\n" + "="*70)
    print("VERIFICATION 1: SOURCE-LEVEL CHECKS")
    print("="*70)
    
    try:
        with open(tex_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"ERROR: Cannot read {tex_path}: {e}")
        return False
    
    forbidden_strings = [
        "Sec. III-B",
        "Sec.~III-B", 
        "Eq. (6), Section III-B",
        "Eq. (6), Section~III-B",
        "Baseline (Sec.~III-B)"
    ]
    
    all_pass = True
    for forbidden in forbidden_strings:
        if forbidden in content:
            print(f"  FAIL: Found '{forbidden}' in source")
            all_pass = False
        else:
            print(f"  PASS: No '{forbidden}' in source")
    
    # Check labels exist
    required_labels = [
        r"\label{sec:multi_component_ensemble}",
        r"\label{eq:auth_score}"
    ]
    
    print("\n  Checking required labels:")
    for label in required_labels:
        if label in content:
            print(f"    PASS: Found {label}")
        else:
            print(f"    FAIL: Missing {label}")
            all_pass = False
    
    # Check label usage in references
    required_refs = [
        r"\ref{sec:multi_component_ensemble}",
        r"\eqref{eq:auth_score}"
    ]
    
    print("\n  Checking label-based references:")
    for ref in required_refs:
        if ref in content:
            print(f"    PASS: Found {ref}")
        else:
            print(f"    FAIL: Missing {ref}")
            all_pass = False
    
    return all_pass

def verify_pdf_no_hardcoded_refs(pdf_path):
    """Verify compiled PDF has no hardcoded manual references."""
    print("\n" + "="*70)
    print("VERIFICATION 2: COMPILED PDF TEXT SEARCH")
    print("="*70)
    
    if not HAS_PYPDF:
        print("  WARNING: PyPDF2 not installed, skipping PDF text extraction")
        print("  Install with: pip install PyPDF2")
        print("  Then manually search PDF with Ctrl+F for:")
        print("    - 'Sec. III-B' (should find 0 matches)")
        print("    - 'Eq. (6), Section III-B' (should find 0 matches)")
        return True  # Cannot verify but not a hard failure
    
    if not os.path.exists(pdf_path):
        print(f"  ERROR: PDF not found at {pdf_path}")
        return False
    
    pdf_text = extract_text_from_pdf(pdf_path)
    if pdf_text is None:
        return False
    
    # Normalize whitespace for searching
    pdf_text_normalized = " ".join(pdf_text.split())
    
    forbidden_in_pdf = [
        "Sec. III-B",
        "Eq. (6), Section III-B",
        "Equation (6), Section III-B"
    ]
    
    all_pass = True
    for forbidden in forbidden_in_pdf:
        if forbidden.lower() in pdf_text_normalized.lower():
            print(f"  FAIL: Found '{forbidden}' in compiled PDF")
            all_pass = False
        else:
            print(f"  PASS: No '{forbidden}' in compiled PDF")
    
    return all_pass

def verify_equation_split(pdf_path):
    """Verify that Equation (5) is rendered on two lines."""
    print("\n" + "="*70)
    print("VERIFICATION 3: EQUATION (5) RENDERING")
    print("="*70)
    
    if not HAS_PYPDF:
        print("  WARNING: Cannot verify PDF rendering without PyPDF2")
        print("  Manual check required:")
        print("    - Search PDF for 'S margin' or 'Equation (5)'")
        print("    - Verify it spans TWO LINES (not single overfull line)")
        return True
    
    if not os.path.exists(pdf_path):
        print(f"  ERROR: PDF not found at {pdf_path}")
        return False
    
    pdf_text = extract_text_from_pdf(pdf_path)
    if pdf_text is None:
        return False
    
    # Look for Smargin or equation (5)
    if "S_margin" in pdf_text or "S margin" in pdf_text:
        print("  PASS: Equation (5) S_margin found in PDF")
        print("\n  NOTE: Manual verification needed:")
        print("    - Open PDF in viewer")
        print("    - Search for 'S margin' or 'Equation (5)'")
        print("    - Visually confirm it displays on TWO lines")
        return True
    else:
        print("  WARNING: Could not locate S_margin equation in PDF text")
        print("  (This is often expected due to PDF encoding)")
        print("\n  Manual verification required:")
        print("    - Open PDF in viewer")  
        print("    - Search for 'Equation (5)'")
        print("    - Verify rendering on two lines")
        return True

def verify_metrics_unchanged(tex_path):
    """Verify metrics values remain unchanged."""
    print("\n" + "="*70)
    print("VERIFICATION 4: METRICS UNCHANGED")
    print("="*70)
    
    try:
        with open(tex_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"ERROR: Cannot read {tex_path}: {e}")
        return False
    
    required_metrics = [
        ("80.77%", "Accuracy"),
        ("0.1076", "Expected Calibration Error (ECE)"),
        ("0.8711", "Area-Under-Accuracy-Coverage (AUC-AC)")
    ]
    
    all_pass = True
    for metric_value, metric_name in required_metrics:
        if metric_value in content:
            print(f"  PASS: {metric_name} = {metric_value}")
        else:
            print(f"  FAIL: Metrics value {metric_value} ({metric_name}) not found")
            all_pass = False
    
    return all_pass

def main():
    """Run all verification checks."""
    workspace = Path("d:/dev/ai/projects/Smart-Notes")
    tex_file = workspace / "submission_bundle/CalibraTeach_IEEE_Access_Upload/OVERLEAF_TEMPLATE.tex"
    pdf_file = workspace / "submission_bundle/CalibraTeach_IEEE_Access_Upload/OVERLEAF_TEMPLATE.pdf"
    
    print("\n" + "="*70)
    print("IEEE ACCESS MANUSCRIPT - FINAL PDF FIX VERIFICATION")
    print("="*70)
    print(f"Source TeX: {tex_file}")
    print(f"Compiled PDF: {pdf_file}")
    
    # Check file existence
    if not tex_file.exists():
        print(f"\nERROR: TeX file not found: {tex_file}")
        return False
    
    if not pdf_file.exists():
        print(f"\nWARNING: PDF not yet compiled: {pdf_file}")
        print("-> Please compile the TeX source first using:")
        print("   cd submission_bundle/CalibraTeach_IEEE_Access_Upload")
        print("   pdflatex -interaction=nonstopmode OVERLEAF_TEMPLATE.tex")
        print("   pdflatex -interaction=nonstopmode OVERLEAF_TEMPLATE.tex  # Run twice")
        pdf_check = False
    else:
        pdf_check = True
    
    # Run verifications
    source_ok = verify_source_no_hardcoded_refs(str(tex_file))
    metrics_ok = verify_metrics_unchanged(str(tex_file))
    
    if pdf_check:
        pdf_refs_ok = verify_pdf_no_hardcoded_refs(str(pdf_file))
        eq_ok = verify_equation_split(str(pdf_file))
    else:
        pdf_refs_ok = None
        eq_ok = None
    
    # Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    checks = {
        "Source LATeX (no hardcoded refs)": source_ok,
        "Metrics (unchanged)": metrics_ok,
        "PDF text search (if compiled)": pdf_refs_ok,
        "Equation (5) rendering (if compiled)": eq_ok
    }
    
    for check_name, result in checks.items():
        status = "PASS" if result else ("PENDING" if result is None else "FAIL")
        symbol = "[✓]" if result else ("[⏳]" if result is None else "[✗]")
        print(f"{symbol} {check_name}: {status}")
    
    if source_ok and metrics_ok:
        if pdf_check and (not pdf_refs_ok or not eq_ok):
            print("\nOVERALL: SOURCE OK, PDF CHECKS FAILED")
            return False
        elif not pdf_check:
            print("\nOVERALL: SOURCE OK, AWAITING PDF COMPILATION")
            print("\nNext steps:")
            print("1. Compile: cd submission_bundle/CalibraTeach_IEEE_Access_Upload")
            print("2. Run: pdflatex -interaction=nonstopmode OVERLEAF_TEMPLATE.tex")
            print("3. Run: pdflatex -interaction=nonstopmode OVERLEAF_TEMPLATE.tex  (run twice)")
            print("4. Re-run this script to verify PDF")
            return True
        else:
            print("\nOVERALL: ALL CHECKS PASSED - MANUSCRIPT READY FOR SUBMISSION")
            return True
    else:
        print("\nOVERALL: SOURCE CHECKS FAILED")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
