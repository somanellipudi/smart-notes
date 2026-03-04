#!/usr/bin/env python
"""
CalibraTeach Figure Generation - Quick Reference
Run this file to regenerate all manuscript figures
"""

import subprocess
import sys
import os

# Ensure we're in the right directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("CalibraTeach Manuscript Figure Generation")
print("=" * 70)
print()

# Run all figures
result = subprocess.run(
    [sys.executable, 'scripts/make_all_figures.py'],
    cwd=os.path.dirname(os.path.abspath(__file__))
)

if result.returncode == 0:
    print()
    print("=" * 70)
    print("SUCCESS: All figures generated and ready for publication!")
    print("=" * 70)
    print()
    print("Next: Compile the manuscript with LaTeX")
    print("  cd submission_bundle")
    print("  pdflatex -interaction=nonstopmode OVERLEAF_TEMPLATE.tex")
    print()

sys.exit(result.returncode)
