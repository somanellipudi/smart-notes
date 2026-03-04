#!/usr/bin/env python3
"""
Strip invisible Unicode characters and noncharacters from LaTeX files.

This script removes zero-width characters, soft hyphens, and invalid Unicode
that can cause PDF copy/paste artifacts and compilation warnings.

Target removal set: U+00AD (soft hyphen), U+200B (zero-width space),
U+2060 (word joiner), U+FEFF (BOM), U+FFFE (noncharacter), U+FFFF (noncharacter).

Usage:
    python strip_invisible_unicode.py <file.tex>
    python strip_invisible_unicode.py --all  # Process all .tex files recursively
"""

import os
import sys
import re
from pathlib import Path

# Target Unicode code points to remove
INVISIBLE_CHARS = {
    '\u00AD',  # U+00AD: soft hyphen
    '\u200B',  # U+200B: zero-width space
    '\u2060',  # U+2060: word joiner
    '\u2061',  # U+2061: function application (invisible)
    '\u2062',  # U+2062: invisible times
    '\u2063',  # U+2063: invisible separator
    '\uFEFF',  # U+FEFF: zero-width no-break space / BOM
    '\uFFFE',  # U+FFFE: noncharacter
    '\uFFFF',  # U+FFFF: noncharacter
}


def strip_file(filepath):
    """
    Read a file, remove invisible Unicode characters, and write back.
    Returns (num_chars_removed, content_changed).
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"ERROR reading {filepath}: {e}")
        return 0, False

    # Count and remove invisible chars
    original_len = len(content)
    cleaned = ''.join(
        char for char in content
        if char not in INVISIBLE_CHARS
    )
    removed_count = original_len - len(cleaned)

    if removed_count > 0:
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(cleaned)
            print(f"✓ {filepath}: removed {removed_count} invisible character(s)")
            return removed_count, True
        except Exception as e:
            print(f"ERROR writing {filepath}: {e}")
            return 0, False
    else:
        print(f"  {filepath}: clean (no invisible characters found)")
        return 0, False


def process_files(pattern=None):
    """
    Process .tex files matching pattern or recursively search project.
    Returns total count of characters removed.
    """
    total_removed = 0
    files_changed = 0

    if pattern:
        # Single file
        if os.path.isfile(pattern):
            removed, changed = strip_file(pattern)
            total_removed += removed
            if changed:
                files_changed += 1
        else:
            print(f"ERROR: File not found: {pattern}")
    else:
        # Recursive search
        root = Path('.')
        tex_files = sorted(root.rglob('*.tex'))
        if not tex_files:
            print("No .tex files found in current directory or subdirectories.")
            return 0, 0

        for tex_file in tex_files:
            removed, changed = strip_file(str(tex_file))
            total_removed += removed
            if changed:
                files_changed += 1

    print(f"\n{'='*60}")
    print(f"Summary: {total_removed} invisible character(s) removed from {files_changed} file(s)")
    print(f"{'='*60}")

    return total_removed, files_changed


if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == '--all':
            process_files()
        else:
            process_files(sys.argv[1])
    else:
        print(__doc__)
        sys.exit(1)
