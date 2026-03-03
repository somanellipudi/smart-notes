#!/usr/bin/env python3
"""Update IEEE paper with latest metrics from full pipeline run."""

import json
import re
from pathlib import Path

# Load latest metrics
with open('artifacts/latest/ci_report.json', 'r') as f:
    ci = json.load(f)

# Extract key values
accuracy_pe = ci['accuracy']['point_estimate']
accuracy_pct_v2 = f"{accuracy_pe * 100:.2f}%"

ece_pe = ci['ece']['point_estimate']
auc_ac_pe = ci['auc_ac']['point_estimate']
macro_f1_pe = ci['macro_f1']['point_estimate']

# Load IEEE paper
ieee_path = Path('research_bundle/07_papers_ieee/IEEE_SMART_NOTES_COMPLETE.md')
content = ieee_path.read_text(encoding='utf-8')

print("=== MAPPING OLD METRICS TO NEW METRICS ===")
print(f"Accuracy: 81.2% → {accuracy_pct_v2} (point estimate: {accuracy_pe:.4f})")
print(f"ECE: 0.0823 → {ece_pe:.4f}")
print(f"AUC-AC: 0.9102 → {auc_ac_pe:.4f}")
print(f"Macro-F1: 0.801 → {macro_f1_pe:.4f}")
print()

# Count occurrences before replacement
count_812 = len(re.findall(r'81\.2%', content))
count_0823 = len(re.findall(r'0\.0823', content))
count_9102 = len(re.findall(r'0\.9102', content))
count_801 = len(re.findall(r'0\.801', content))

print(f"Found: {count_812}x '81.2%', {count_0823}x '0.0823', {count_9102}x '0.9102', {count_801}x '0.801'")
print()

# Perform targeted replacements in priority order
replacements = [
    ('0.0823', f'{ece_pe:.4f}'),
    ('0.9102', f'{auc_ac_pe:.4f}'),
    ('0.801', f'{macro_f1_pe:.4f}'),
    ('81.2%', accuracy_pct_v2),
]

for old, new in replacements:
    before_count = content.count(old)
    if before_count > 0:
        content = content.replace(old, new)
        print(f"✅ Replaced {before_count}x '{old}' → '{new}'")

# Write back
ieee_path.write_text(content, encoding='utf-8')
print()
print(f"✅ Updated IEEE paper saved to {ieee_path}")
print(f"Total replacements: {count_0823 + count_9102 + count_801 + count_812}")
