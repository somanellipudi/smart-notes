#!/usr/bin/env python3
"""
Unicode Sanitizer for LaTeX Manuscripts
========================================
Scans .tex, .bib, and .sty files for problematic Unicode characters that can
cause PDF copy/paste artifacts ("￾" replacement characters) or compilation issues.

Usage:
    python scripts/sanitize_unicode.py --check          # Report only (exit 1 if issues found)
    python scripts/sanitize_unicode.py --fix            # Auto-fix in-place (creates .bak backups)
    python scripts/sanitize_unicode.py --check --verbose  # Show all non-ASCII chars

Author: CalibraTeach Team
Date: March 4, 2026
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Set
import unicodedata

# ============================================================================
# PROBLEMATIC UNICODE CHARACTERS (known to cause PDF artifacts)
# ============================================================================
BAD_CHARS = {
    '\u00ad': 'SOFT HYPHEN',
    '\u00a0': 'NO-BREAK SPACE (use ~ or normal space in LaTeX)',
    '\u200b': 'ZERO WIDTH SPACE',
    '\u200c': 'ZERO WIDTH NON-JOINER',
    '\u200d': 'ZERO WIDTH JOINER',
    '\u2060': 'WORD JOINER',
    '\u2061': 'FUNCTION APPLICATION (invisible operator)',
    '\u2062': 'INVISIBLE TIMES',
    '\u2063': 'INVISIBLE SEPARATOR',
    '\u2064': 'INVISIBLE PLUS',
    '\ufeff': 'ZERO WIDTH NO-BREAK SPACE (BOM)',
    '\ufffe': 'NONCHARACTER (guaranteed to cause PDF artifacts)',
    '\uffff': 'NONCHARACTER',
}

# Characters that should be replaced (not just removed)
REPLACE_MAP = {
    '\u00a0': ' ',  # NBSP → normal space (LaTeX handles spacing via ~)
    '\u2013': '--',  # EN DASH → LaTeX ligature
    '\u2014': '---',  # EM DASH → LaTeX ligature
    '\u2018': '`',   # LEFT SINGLE QUOTATION MARK
    '\u2019': "'",   # RIGHT SINGLE QUOTATION MARK
    '\u201c': '``',  # LEFT DOUBLE QUOTATION MARK
    '\u201d': "''",  # RIGHT DOUBLE QUOTATION MARK
}

# Safe non-ASCII chars (commonly used in LaTeX, OK to keep)
SAFE_NON_ASCII = {
    '\u00e9',  # é (common in author names)
    '\u00e8',  # è
    '\u00c9',  # É
    '\u00fc',  # ü
    '\u00f1',  # ñ
    '\u00e1',  # á
    '\u00f3',  # ó
    '\u00ed',  # í
    '\u00fa',  # ú
    '\u00e7',  # ç
    # Add more as needed for author names
}


def scan_file(filepath: Path, verbose: bool = False) -> List[Tuple[int, int, str, str]]:
    """
    Scan a single file for problematic Unicode characters.
    
    Returns:
        List of (line_num, col_num, char, description) tuples
    """
    issues = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, start=1):
                for col_num, char in enumerate(line, start=1):
                    # Check for bad characters
                    if char in BAD_CHARS:
                        issues.append((line_num, col_num, char, BAD_CHARS[char]))
                    # Optionally report all non-ASCII
                    elif verbose and ord(char) > 127 and char not in SAFE_NON_ASCII:
                        char_name = unicodedata.name(char, 'UNKNOWN')
                        issues.append((line_num, col_num, char, f'Non-ASCII: {char_name} (U+{ord(char):04X})'))
    except UnicodeDecodeError as e:
        print(f"ERROR: Cannot decode {filepath}: {e}", file=sys.stderr)
        return []
    
    return issues


def fix_file(filepath: Path) -> int:
    """
    Fix problematic Unicode characters in-place (creates .bak backup).
    
    Returns:
        Number of characters replaced
    """
    backup_path = filepath.with_suffix(filepath.suffix + '.bak')
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Create backup
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Apply fixes
        fixed_content = content
        changes = 0
        
        # Remove bad characters
        for bad_char in BAD_CHARS:
            if bad_char in fixed_content:
                # Special handling: replace or remove
                if bad_char in REPLACE_MAP:
                    fixed_content = fixed_content.replace(bad_char, REPLACE_MAP[bad_char])
                else:
                    fixed_content = fixed_content.replace(bad_char, '')
                changes += content.count(bad_char)
        
        # Apply smart replacements (typographic quotes, dashes)
        for old_char, new_char in REPLACE_MAP.items():
            if old_char not in BAD_CHARS and old_char in fixed_content:
                fixed_content = fixed_content.replace(old_char, new_char)
                changes += content.count(old_char)
        
        # Write fixed content
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        
        return changes
    
    except Exception as e:
        print(f"ERROR: Cannot fix {filepath}: {e}", file=sys.stderr)
        return 0


def main():
    parser = argparse.ArgumentParser(
        description='Scan and fix problematic Unicode characters in LaTeX files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --check                    # Check all .tex/.bib/.sty files
  %(prog)s --check --verbose          # Show all non-ASCII characters
  %(prog)s --fix                      # Auto-fix (creates .bak backups)
  %(prog)s --check OVERLEAF_TEMPLATE.tex  # Check specific file
        """
    )
    parser.add_argument('files', nargs='*', help='Specific files to check (default: all .tex/.bib/.sty)')
    parser.add_argument('--check', action='store_true', default=True,
                        help='Check for issues (default mode)')
    parser.add_argument('--fix', action='store_true',
                        help='Fix issues in-place (creates .bak backups)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Report all non-ASCII characters, not just known-bad ones')
    
    args = parser.parse_args()
    
    # Determine files to scan
    if args.files:
        file_list = [Path(f) for f in args.files]
    else:
        # Scan submission_bundle directory for .tex, .bib, .sty
        root = Path(__file__).parent.parent
        file_list = []
        for ext in ['*.tex', '*.bib', '*.sty']:
            file_list.extend(root.glob(ext))
            file_list.extend(root.glob(f'**/{ext}'))
    
    if not file_list:
        print("No files found to scan.", file=sys.stderr)
        return 1
    
    # FIX MODE
    if args.fix:
        print(f"Fixing {len(file_list)} files...")
        total_changes = 0
        for filepath in file_list:
            if not filepath.exists():
                print(f"SKIP: {filepath} (not found)")
                continue
            changes = fix_file(filepath)
            if changes > 0:
                print(f"FIXED: {filepath} ({changes} characters replaced, backup: {filepath}.bak)")
                total_changes += changes
        
        if total_changes == 0:
            print("\n✓ No problematic characters found. All files clean!")
            return 0
        else:
            print(f"\n✓ Fixed {total_changes} characters across {len(file_list)} files.")
            print("  Backups created with .bak extension. Review changes before committing.")
            return 0
    
    # CHECK MODE (default)
    print(f"Scanning {len(file_list)} files for problematic Unicode characters...")
    total_issues = 0
    files_with_issues = 0
    
    for filepath in file_list:
        if not filepath.exists():
            continue
        
        issues = scan_file(filepath, verbose=args.verbose)
        if issues:
            files_with_issues += 1
            total_issues += len(issues)
            print(f"\n{filepath}:")
            for line_num, col_num, char, desc in issues[:10]:  # Limit output
                print(f"  Line {line_num}, Col {col_num}: {desc}")
                print(f"    Context: {repr(char)} (U+{ord(char):04X})")
            if len(issues) > 10:
                print(f"  ... and {len(issues) - 10} more issues")
    
    print(f"\n{'='*70}")
    if total_issues == 0:
        print("✓ PASS: No problematic Unicode characters found!")
        print("  All .tex/.bib/.sty files are clean for submission.")
        return 0
    else:
        print(f"✗ FAIL: Found {total_issues} problematic characters in {files_with_issues} files.")
        print(f"\nTo fix automatically:")
        print(f"  python {Path(__file__).name} --fix")
        print(f"\nOr manually replace the reported characters in your editor.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
