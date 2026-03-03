import json
from pathlib import Path

print("=" * 80)
print("COMPREHENSIVE METRICS VERIFICATION REPORT")
print("=" * 80)

# Read metrics summary
with open("artifacts/metrics_summary.json") as f:
    metrics = json.load(f)

print("\n1. METRICS SUMMARY (Current State)")
print("-" * 80)
print(f"Primary Model: {metrics['primary_model']}")
print()

primary = metrics['models'].get(metrics['primary_model'], {})
print(f"CalibraTeach:")
print(f"  - Sample Size: {primary.get('n_samples')} samples")
print(f"  - Expected:   260 samples")
print(f"  - MATCH: {'✗ CRITICAL' if primary.get('n_samples') != 260 else '✓'}")
print()
print(f"  - Accuracy:   {primary.get('accuracy'):.4f} ({primary.get('accuracy')*100:.2f}%)")
print(f"  - Expected:   0.8077 (80.77%)")
print(f"  - MATCH: {'✗ CRITICAL' if abs(primary.get('accuracy', 0) - 0.8077) > 0.01 else '✓'}")
print()
print(f"  - ECE:        {primary.get('ece'):.4f}")
print(f"  - Expected:   0.1247")
print(f"  - MATCH: {'✗ CRITICAL' if abs(primary.get('ece', 0) - 0.1247) > 0.001 else '✓'}")
print()
print(f"  - AUC-AC:     {primary.get('auc_ac'):.4f}")
print(f"  - Expected:   0.8803")
print(f"  - MATCH: {'✗ CRITICAL' if abs(primary.get('auc_ac', 0) - 0.8803) > 0.001 else '✓'}")

print("\n2. MANUSCRIPT MACROS (Current State)")
print("-" * 80)
with open("submission_bundle/metrics_values.tex") as f:
    lines = f.readlines()
    for line in lines:
        if "newcommand" in line:
            print(f"  {line.strip()}")

print("\n3. FIGURE FILES CHECK")
print("-" * 80)
for fig_name in ["reliability_diagram_verified.pdf", "accuracy_coverage_verified.pdf"]:
    path = Path(f"figures/{fig_name}")
    exists = path.exists()
    size = path.stat().st_size if exists else 0
    print(f"  {fig_name}: {'✓ EXISTS' if exists else '✗ MISSING'} ({size:,} bytes)")

print("\n4. HARD-CODED VALUES IN MANUSCRIPT CHECK")
print("-" * 80)
import re
with open("submission_bundle/OVERLEAF_TEMPLATE.tex", encoding='utf-8') as f:
    content = f.read()
    
hard_coded = []
if "80.77" in content:
    count = content.count("80.77")
    matches = re.finditer(r'80\.77', content)
    for m in matches:
        line_num = content[:m.start()].count('\n') + 1
        hard_coded.append(f"Line {line_num}: '80.77' (appears {count} times)")

if "0.1247" in content:
    hard_coded.append("'0.1247' found (should use \\ECEValue{})")
    
if "0.8803" in content:
    hard_coded.append("'0.8803' found (should use \\AUCACValue{})")

if hard_coded:
    print("✗ Hard-coded metric values found in manuscript:")
    for item in hard_coded:
        print(f"   {item}")
else:
    print("✓ No obvious hard-coded metrics found")

print("\n5. RED FLAGS SUMMARY")
print("=" * 80)
flags = []
if primary.get('n_samples') != 260:
    flags.append(f"✗ CRITICAL: CalibraTeach.npz has {primary.get('n_samples')} samples, not 260 (18.5x too small!)")
if abs(primary.get('accuracy', 0) - 0.8077) > 0.01:
    flags.append(f"✗ CRITICAL: Accuracy mismatch ({primary.get('accuracy')*100:.2f}% vs 80.77%)")
if abs(primary.get('ece', 0) - 0.1247) > 0.001:
    flags.append(f"✗ CRITICAL: ECE mismatch ({primary.get('ece'):.4f} vs 0.1247)")
if abs(primary.get('auc_ac', 0) - 0.8803) > 0.001:
    flags.append(f"✗ CRITICAL: AUC-AC mismatch ({primary.get('auc_ac'):.4f} vs 0.8803)")

if hard_coded:
    flags.append("✗ WARNING: Hard-coded metric values still present in manuscript table description (line 330 & 546)")

for flag in flags:
    print(flag)

if not flags:
    print("✓ All metrics match expected values")
else:
    print(f"\nTotal Issues Found: {len(flags)}")
    print("\n6. ROOT CAUSE ANALYSIS")
    print("-" * 80)
    print("The core issue is that CalibraTeach.npz was exported from a tiny 14-sample")
    print("batch instead of the full 260-claim test set. The correct predictions file")
    print("needs to be identified and re-exported.")
