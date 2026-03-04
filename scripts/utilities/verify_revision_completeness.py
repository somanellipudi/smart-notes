#!/usr/bin/env python3
"""Verify manuscript consistency after reviewer improvements."""

import re
from pathlib import Path

file = Path('submission_bundle/overleaf_upload_pack/OVERLEAF_TEMPLATE.tex')
content = file.read_text()

print('='*70)
print('MANUSCRIPT CONSISTENCY VERIFICATION')
print('='*70)
print()

# Check stale references
print('Checking for stale section references...')
stale_patterns = [r'\bV-F\b', r'\bV-G\b', r'Section~IV-E']
found_stale = False
for pattern in stale_patterns:
    matches = list(re.finditer(pattern, content))
    if matches:
        found_stale = True
        for m in matches:
            start = max(0, m.start()-50)
            end = min(len(content), m.end()+50)
            snippet = content[start:end].replace('\n', ' ')
            print(f'  WARN: Found "{m.group()}"')

if not found_stale:
    print('  ✓ No stale section references')
print()

# Check Acknowledgments removed
print('Checking Acknowledgments section...')
if 'section*{Acknowledgments}' in content:
    print('  ✗ Acknowledgments section still present')
else:
    print('  ✓ Acknowledgments section removed')
print()

# Check new appendix sections
print('Checking new appendix sections...')
appendix_sections = [
    'Calibration Robustness',
    'Abstention Threshold Stability',
    'Statistical Significance and Class Balance',
    'Authority Weights'
]
for section in appendix_sections:
    if section in content:
        print(f'  ✓ {section}')
    else:
        print(f'  ✗ {section} - NOT FOUND')
print()

# Check key metrics
print('Verifying key metrics preserved...')
metrics = {'ECE': '0.1076', 'Accuracy': '80.77', 'AUC-AC': '0.8711'}
all_found = True
for name, value in metrics.items():
    if value in content:
        print(f'  ✓ {name} = {value}')
    else:
        print(f'  ✗ {name} = {value} - NOT FOUND')
        all_found = False
print()

# Check formatting improvements
print('Checking formatting improvements...')
has_figure_star = 'begin{figure*}' in content
has_table_star = 'begin{table*}' in content
has_align = 'begin{align}' in content
no_motivation = 'subsection{Motivation}' not in content
no_contributions_subsection = content.count('subsection{Contributions}') == 0

print(f'  {"✓" if has_figure_star else "✗"} Figure spans two columns (figure*)')
print(f'  {"✓" if has_table_star else "✗"} Table 1 spans two columns (table*)')
print(f'  {"✓" if has_align else "✗"} Equation split with align environment')
print(f'  {"✓" if no_motivation else "✗"} Motivation subsection removed')
print(f'  {"✓" if no_contributions_subsection else "✗"} Contributions subsection removed')

print()
print('='*70)
print('✓ VERIFICATION COMPLETE')
print('='*70)
print()
print('SUMMARY:')
print(f'  Stale references: {"CLEAN" if not found_stale else "FOUND"}')
print(f'  Acks removed: {"YES" if "section*{Acknowledgments}" not in content else "NO"}')
print(f'  New appendices: {sum(1 for s in appendix_sections if s in content)}/4')
print(f'  Metrics intact: {"YES" if all_found else "NO"}')
print(f'  Formatting fixes: {sum([has_figure_star, has_table_star, has_align, no_motivation, no_contributions_subsection])}/5')
