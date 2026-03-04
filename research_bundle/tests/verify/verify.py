#!/usr/bin/env python3
"""
Unified IEEE Access Submission Verification Pipeline
=====================================================
Runs all verification steps in sequence:
  1. Unicode sanitization (source files)
  2. Submission integrity checks (refs, metrics, structure)
  3. PDF build + text extraction verification

Usage:
    python verify.py              # Run all checks
    python verify.py --fix        # Auto-fix Unicode issues, then verify
    python verify.py --skip-build # Skip PDF build (faster, for source-only checks)

Exit Codes:
    0 = All checks passed (ready for submission)
    1 = One or more checks failed

Author: CalibraTeach Team  
Date: March 4, 2026
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_check(name: str, command: list, critical: bool = True) -> bool:
    """
    Run a verification check and report results.
    
    Args:
        name: Human-readable check name
        command: Command to execute (list of strings)
        critical: If True, fail pipeline on non-zero exit code
    
    Returns:
        True if check passed, False otherwise
    """
    print(f"\n{'='*70}")
    print(f"CHECK: {name}")
    print(f"{'='*70}")
    
    try:
        result = subprocess.run(
            command,
            cwd=Path(__file__).parent,
            timeout=180
        )
        
        if result.returncode == 0:
            print(f"✓ PASS: {name}")
            return True
        else:
            print(f"✗ FAIL: {name} (exit code {result.returncode})")
            if critical:
                print("  This is a CRITICAL check. Fix before submission.")
            return False
    
    except subprocess.TimeoutExpired:
        print(f"✗ FAIL: {name} (timeout)")
        return False
    except FileNotFoundError as e:
        print(f"✗ FAIL: {name} (command not found: {e})")
        return False
    except Exception as e:
        print(f"✗ FAIL: {name} (error: {e})")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Run all IEEE Access submission verification checks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Verification Pipeline:
  [1/4] Unicode Sanitization       - Check for invisible chars (U+00AD, U+200B, etc.)
  [2/4] Submission Integrity        - Check refs, metrics, structure
  [3/4] PDF Build                   - Compile PDF (pdflatex x3)
  [4/4] PDF Text Extraction         - Verify clean text (no ￾ � artifacts)

Examples:
  %(prog)s                 # Run all checks (recommended before submission)
  %(prog)s --fix           # Auto-fix Unicode issues, then run all checks
  %(prog)s --skip-build    # Skip PDF build (faster, for source-only checks)
        """
    )
    parser.add_argument('--fix', action='store_true',
                        help='Auto-fix Unicode issues before running checks')
    parser.add_argument('--skip-build', action='store_true',
                        help='Skip PDF build and text extraction checks')
    
    args = parser.parse_args()
    
    # Print header
    print("="*70)
    print("IEEE ACCESS SUBMISSION VERIFICATION PIPELINE")
    print("CalibraTeach: Calibrated Selective Prediction")
    print("="*70)
    
    results = []
    
    # [1/4] Unicode Sanitization
    if args.fix:
        print("\nAuto-fixing Unicode issues...")
        results.append(run_check(
            "Unicode Sanitization (FIX MODE)",
            [sys.executable, 'scripts/sanitize_unicode.py', '--fix'],
            critical=True
        ))
    else:
        results.append(run_check(
            "Unicode Sanitization (CHECK MODE)",
            [sys.executable, 'scripts/sanitize_unicode.py', '--check'],
            critical=True
        ))
    
    # [2/4] Submission Integrity
    results.append(run_check(
        "Submission Integrity (Refs, Metrics, Structure)",
        [sys.executable, 'scripts/verify_submission.py', 'OVERLEAF_TEMPLATE.tex'],
        critical=True
    ))
    
    # [3/4] PDF Build + [4/4] Text Extraction
    if not args.skip_build:
        results.append(run_check(
            "PDF Build + Text Extraction Verification",
            [sys.executable, 'scripts/verify_pdf_text.py'],
            critical=True
        ))
    else:
        print("\n" + "="*70)
        print("CHECK: PDF Build + Text Extraction (SKIPPED)")
        print("="*70)
        print("  Use --no-skip-build to enable PDF verification")
    
    # Summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✓ ALL CHECKS PASSED ({passed}/{total})")
        print("\n✅ READY FOR IEEE ACCESS SUBMISSION")
        print("\nNext steps:")
        print("  1. Upload OVERLEAF_TEMPLATE.tex + figures/ to Overleaf")
        print("  2. Compile in Overleaf (verify no issues in cloud environment)")
        print("  3. Submit to IEEE Access ScholarOne")
        print("\nSee SUBMISSION.md for detailed upload instructions.")
        return 0
    else:
        failed = total - passed
        print(f"✗ VERIFICATION FAILED ({failed}/{total} checks failed)")
        print("\n❌ NOT READY FOR SUBMISSION")
        print("\nFix the above issues, then run:")
        print("  python verify.py")
        return 1


if __name__ == '__main__':
    sys.exit(main())
