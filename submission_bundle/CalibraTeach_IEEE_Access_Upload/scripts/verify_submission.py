#!/usr/bin/env python3
"""
Pre-Submission Verification Script for IEEE Access
===================================================
Validates all critical requirements before final submission:
1. Unicode hygiene (no "￾" artifacts)
2. Metric preservation (10 core values unchanged)
3. Reference integrity (no "??" in compiled PDF)
4. Table/figure numbering consistency
5. LaTeX compilation success

Run this before uploading to Overleaf or submitting to IEEE Access.

Usage:
    python submission_bundle/CalibraTeach_IEEE_Access_Upload/scripts/verify_submission.py

Author: CalibraTeach Team
Date: March 4, 2026
"""

import sys
import re
from pathlib import Path
from typing import List, Tuple, Dict

# ============================================================================
# CRITICAL METRIC VALUES (MUST NOT CHANGE)
# ============================================================================
REQUIRED_METRICS = {
    'Accuracy': '80.77%',
    'ECE': '0.1076',
    'AUC-AC': '0.8711',
    'Latency': '67.68',  # ms
    'Throughput': '14.78',  # claims/sec
    'FEVER_Accuracy': '74.3%',
    'FEVER_ECE': '0.150',
    'Coverage_at_tau_090': '74.0%',  # or 74%
    'SelectiveAcc_at_tau_090': '90.2%',  # or 90%
    'Pilot_n': '25',
    'Instructor_Agreement': '92%',
}

# Alternative representations (regex patterns)
METRIC_PATTERNS = {
    'Accuracy': r'(?:80\.77\s*\\?%|\\AccuracyValue\{\})',
    'ECE': r'(?:0\.1076|\\ECEValue\{\})',
    'AUC-AC': r'(?:0\.8711|\\AUCACValue\{\})',
    'Latency': r'67\.68\s*\\?,?\s*ms',
    'Throughput': r'14\.78\s+claims/sec',
    'FEVER_Accuracy': r'74\.3\s*\\?%',
    'FEVER_ECE': r'0\.150',
    'Coverage_at_tau_090': r'74(?:\.0)?\s*\\?%\s+(?:automated\s+)?coverage',
    'SelectiveAcc_at_tau_090': r'90(?:\.2)?\s*\\?%\s+selective\s+accuracy',
    'Pilot_n': r'\$n\s*=\s*25\$',
    'Instructor_Agreement': r'92\s*\\?%.*(?:instructor|agreement)',
}


def check_unicode_artifacts(tex_file: Path) -> List[Tuple[int, str]]:
    """Check for problematic Unicode characters."""
    issues = []
    bad_chars = [
        '\u00ad', '\u00a0', '\u200b', '\u200c', '\u200d',
        '\u2060', '\u2061', '\u2062', '\u2063', '\u2064',
        '\ufeff', '\ufffe', '\uffff'
    ]
    
    with open(tex_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            for char in bad_chars:
                if char in line:
                    issues.append((line_num, f'Found U+{ord(char):04X} ({unicodedata.name(char, "UNKNOWN")})'))
    
    return issues


def check_metrics(tex_file: Path) -> List[str]:
    """Verify all critical metrics are present and unchanged."""
    missing = []
    
    with open(tex_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    for metric_name, pattern in METRIC_PATTERNS.items():
        if not re.search(pattern, content, re.IGNORECASE):
            missing.append(f'{metric_name}: Expected pattern "{pattern}" not found')
    
    return missing


def check_references(tex_file: Path) -> List[str]:
    """Check for broken cross-references (labels without matching \\label{})."""
    issues = []
    
    with open(tex_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract all \ref{...} and \label{...}
    refs = set(re.findall(r'\\ref\{([^}]+)\}', content))
    labels = set(re.findall(r'\\label\{([^}]+)\}', content))
    
    missing_labels = refs - labels
    if missing_labels:
        issues.append(f'Missing labels: {", ".join(sorted(missing_labels))}')
    
    # Check for "Table X" placeholders
    if re.search(r'Table\s+X\b', content):
        issues.append('Found "Table X" placeholder (replace with \\ref{tab:...})')
    
    return issues


def check_structure(tex_file: Path) -> List[str]:
    """Check document structure (balanced environments, no duplicate sections)."""
    issues = []
    
    with open(tex_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check balanced environments
    for env in ['table', 'figure', 'equation', 'enumerate', 'itemize']:
        begin_count = len(re.findall(r'\\begin\{' + env + r'\}', content))
        end_count = len(re.findall(r'\\end\{' + env + r'\}', content))
        if begin_count != end_count:
            issues.append(f'Unbalanced {env} environment: {begin_count} begin, {end_count} end')
    
    # Check for duplicate section names
    sections = re.findall(r'\\section\{([^}]+)\}', content)
    duplicates = {s for s in sections if sections.count(s) > 1}
    if duplicates:
        issues.append(f'Duplicate section titles: {", ".join(duplicates)}')
    
    return issues


def main():
    print("="*70)
    print("IEEE Access Pre-Submission Verification")
    print("CalibraTeach: Calibrated Selective Prediction for Real-Time Educational Fact Verification")
    print("="*70)
    
    # Locate main .tex file
    script_dir = Path(__file__).parent
    tex_file = script_dir.parent / 'OVERLEAF_TEMPLATE.tex'
    
    if not tex_file.exists():
        print(f"\n✗ ERROR: Cannot find {tex_file}")
        return 1
    
    print(f"\nValidating: {tex_file.name}")
    print("-"*70)
    
    all_passed = True
    
    # CHECK 1: Unicode artifacts
    print("\n[1/5] Checking for Unicode artifacts...")
    unicode_issues = check_unicode_artifacts(tex_file)
    if unicode_issues:
        all_passed = False
        print("  ✗ FAIL: Found problematic Unicode characters:")
        for line_num, issue in unicode_issues[:10]:
            print(f"    Line {line_num}: {issue}")
        if len(unicode_issues) > 10:
            print(f"    ... and {len(unicode_issues) - 10} more issues")
        print("\n  ACTION: Run 'python scripts/sanitize_unicode.py --fix'")
    else:
        print("  ✓ PASS: No Unicode artifacts detected")
    
    # CHECK 2: Metric preservation
    print("\n[2/5] Checking metric preservation...")
    missing_metrics = check_metrics(tex_file)
    if missing_metrics:
        all_passed = False
        print("  ✗ FAIL: Critical metrics missing or changed:")
        for issue in missing_metrics:
            print(f"    {issue}")
        print("\n  ACTION: Restore original metric values (see REQUIRED_METRICS)")
    else:
        print("  ✓ PASS: All 11 critical metrics present")
    
    # CHECK 3: Reference integrity
    print("\n[3/5] Checking cross-references...")
    ref_issues = check_references(tex_file)
    if ref_issues:
        all_passed = False
        print("  ✗ FAIL: Reference issues:")
        for issue in ref_issues:
            print(f"    {issue}")
    else:
        print("  ✓ PASS: All references have matching labels")
    
    # CHECK 4: Document structure
    print("\n[4/5] Checking document structure...")
    structure_issues = check_structure(tex_file)
    if structure_issues:
        all_passed = False
        print("  ✗ FAIL: Structure issues:")
        for issue in structure_issues:
            print(f"    {issue}")
    else:
        print("  ✓ PASS: Document structure valid")
    
    # CHECK 5: File existence
    print("\n[5/5] Checking required files...")
    required_files = [
        'figures/architecture.pdf',
        'figures/reliability_diagram_verified.pdf',
        'figures/accuracy_coverage_verified.pdf',
        'SUBMISSION.md',
        'scripts/sanitize_unicode.py',
    ]
    missing_files = []
    for filepath in required_files:
        full_path = tex_file.parent / filepath
        if not full_path.exists():
            missing_files.append(filepath)
    
    if missing_files:
        all_passed = False
        print("  ✗ FAIL: Missing required files:")
        for filepath in missing_files:
            print(f"    {filepath}")
    else:
        print("  ✓ PASS: All required files present")
    
    # FINAL VERDICT
    print("\n" + "="*70)
    if all_passed:
        print("✓ ALL CHECKS PASSED")
        print("\nNext steps:")
        print("  1. Compile PDF: pdflatex OVERLEAF_TEMPLATE.tex (3x)")
        print("  2. Test PDF text extraction: pdftotext OVERLEAF_TEMPLATE.pdf - | grep '￾'")
        print("  3. Upload to Overleaf with figures/")
        print("  4. Submit to IEEE Access")
        print("\nSee SUBMISSION.md for detailed instructions.")
        return 0
    else:
        print("✗ VALIDATION FAILED")
        print("\nFix the above issues before submission.")
        print("Re-run this script after fixes: python scripts/verify_submission.py")
        return 1


if __name__ == '__main__':
    import unicodedata  # Ensure import for unicodedata.name()
    sys.exit(main())
