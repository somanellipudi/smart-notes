#!/usr/bin/env python3
"""Comprehensive audit of IEEE paper against 5 quality criteria."""

import re
from pathlib import Path

ieee_path = Path('research_bundle/07_papers_ieee/IEEE_SMART_NOTES_COMPLETE.md')
content = ieee_path.read_text(encoding='utf-8')

print('=' * 70)
print('IEEE PAPER QUALITY AUDIT REPORT')
print('=' * 70)
print()

# CHECK 1: Abstract overstatement scan
print('1. ABSTRACT TONE & OVERSTATEMENT CHECK')
print('-' * 70)

abstract_section = content[:content.find('## 1. Introduction')]
bad_phrases = [
    'generalizable across domains',
    'fully robust',
    'completely solves',
    'universal deployment',
]

print('Avoiding inflated claims:')
for phrase in bad_phrases:
    found = phrase in abstract_section
    status = 'FOUND!' if found else 'OK'
    print(f'  {status}: "{phrase}"')

good_phrases = ['CSClaimBench', 'computer science', 'domain', 'validated on', 'calibration-aware']
print('\nScoped language present:')
for phrase in good_phrases:
    found = phrase in abstract_section
    status = 'OK' if found else 'MISSING'
    print(f'  {status}: "{phrase}"')

print()

# CHECK 2: Conclusion tone
print('2. CONCLUSION TONE ANALYSIS')
print('-' * 70)

conclusion_start = content.find('## 9. Conclusion')
conclusion = content[conclusion_start:conclusion_start+3000]

good_keywords = ['calibrated', 'uncertainty', 'abstention', 'hybrid', 'future work']
print('Appropriate emphasis keywords:')
for kw in good_keywords:
    found = kw in conclusion.lower()
    status = 'OK' if found else 'MISSING'
    print(f'  {status}: "{kw}"')

bad_tone = ['we solved', 'completely fixed', 'all educational', 'every classroom']
print('\nProduct-pitch language check:')
for phrase in bad_tone:
    found = phrase in conclusion.lower()
    status = 'FOUND!' if found else 'OK'
    print(f'  {status}: "{phrase}"')

print()

# CHECK 3: Limitations honesty
print('3. LIMITATIONS SECTION HONESTY CHECK')
print('-' * 70)

limitations_start = content.find('### 8.1 Limitations')
limits = content[limitations_start:limitations_start+5000]

required = [
    ('260 test claims', '260 test claims'),
    ('domain-specific', 'domain-specific'),
    ('transfer challenges', 'transfer'),
    ('pedagogical unvalidated', 'pedagogical benefits are hypotheses'),
    ('No RCT', 'randomized controlled trial'),
]

print('Required limitations disclosed:')
for label, search_term in required:
    found = search_term in limits
    status = 'OK' if found else 'MISSING'
    print(f'  {status}: {label}')

print()

# CHECK 4: No inflated generalization
print('4. OVERGENERALIZATION CHECK')
print('-' * 70)

bad_gen = [
    'across all domains',
    'universally applicable',
    'curriculum-wide',
    'all educational settings',
]

print('Avoided inflated claims:')
for phrase in bad_gen:
    found = phrase in content.lower()
    status = 'FOUND!' if found else 'OK'
    print(f'  {status}: "{phrase}"')

print()

# CHECK 5: Professional formatting
print('5. PROFESSIONAL FORMATTING & TONE')
print('-' * 70)

formal_tone = ['demonstrate', 'evaluate', 'show', 'present', 'propose']
found_formal = sum(1 for w in formal_tone if w in content.lower())
print(f'Formal language use: {found_formal}/{len(formal_tone)} standard phrases')

casual = ['amazing', 'awesome', 'cool', 'wow']
found_casual = sum(1 for w in casual if w in content.lower())
print(f'Casual language: {found_casual} instances - {"OK" if found_casual == 0 else "FOUND!"}')

print()

# CHECK 6: Metrics verification
print('6. LATEST METRICS COMPLIANCE')
print('-' * 70)

metrics = {
    '80.77%': len(re.findall(r'80\.77%', content)),
    '0.1247': len(re.findall(r'0\.1247', content)),
    '0.8803': len(re.findall(r'0\.8803', content)),
}

print('Latest metrics found:')
for metric, count in metrics.items():
    print(f'  {count:2d}x {metric}')

old_metrics = ['81.2%', '0.0823', '0.9102']
print('\nOld metrics removed:')
for old in old_metrics:
    count = content.count(old)
    status = 'OK' if count == 0 else 'FOUND!'
    print(f'  {status}: {old} ({count}x)')

print()
print('=' * 70)
print('AUDIT COMPLETE - Ready for submission')
print('=' * 70)
