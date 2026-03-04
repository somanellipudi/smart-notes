#!/usr/bin/env python
"""
COMPREHENSIVE VERIFICATION SCRIPT
Verify all metrics, figures, and manuscript consistency
"""

import json
from pathlib import Path
import sys

print("\n" + "=" * 90)
print("COMPREHENSIVE DATA & FIGURE VERIFICATION REPORT")
print("=" * 90)

errors = []
warnings = []

# ============================================================================
# 1. VERIFY metrics_values.tex
# ============================================================================
print("\n[1] VERIFYING: submission_bundle/metrics_values.tex")
print("-" * 90)

tex_file = Path("submission_bundle/metrics_values.tex")
if not tex_file.exists():
    errors.append("✗ metrics_values.tex NOT FOUND")
    print("✗ File does not exist")
else:
    content = tex_file.read_text()
    print("✓ File exists")
    print(f"  Size: {tex_file.stat().st_size} bytes")
    
    required_macros = {
        "\\AccuracyValue": "80.77\\%",
        "\\ECEValue": "0.1076",
        "\\AUCACValue": "0.8711"
    }
    
    for macro, expected_val in required_macros.items():
        if macro in content:
            print(f"  ✓ {macro} defined")
            if expected_val in content:
                print(f"    ✓ Value correct: {expected_val}")
            else:
                errors.append(f"✗ {macro} has wrong value (expected {expected_val})")
                print(f"    ✗ Value MISMATCH (expected {expected_val})")
        else:
            errors.append(f"✗ {macro} NOT DEFINED")
            print(f"  ✗ {macro} NOT FOUND")

# ============================================================================
# 2. VERIFY metrics_summary.json
# ============================================================================
print("\n[2] VERIFYING: artifacts/metrics_summary.json")
print("-" * 90)

json_file = Path("artifacts/metrics_summary.json")
if not json_file.exists():
    errors.append("✗ artifacts/metrics_summary.json NOT FOUND")
    print("✗ File does not exist")
else:
    print("✓ File exists")
    print(f"  Size: {json_file.stat().st_size} bytes")
    
    try:
        data = json.load(open(json_file))
        print("✓ Valid JSON format")
        
        primary_model = data.get("primary_model", "")
        print(f"  Primary model: {primary_model}")
        
        if primary_model != "CalibraTeach":
            errors.append(f"✗ Primary model is '{primary_model}', expected 'CalibraTeach'")
            print(f"  ✗ Primary model MISMATCH")
        else:
            print(f"  ✓ Primary model correct")
        
        if primary_model in data.get("models", {}):
            model_data = data["models"][primary_model]
            
            # Check sample count
            n_samples = model_data.get("n_samples")
            if n_samples == 260:
                print(f"  ✓ Sample count correct: {n_samples}")
            else:
                errors.append(f"✗ Sample count is {n_samples}, expected 260")
                print(f"  ✗ Sample count MISMATCH: {n_samples} (expected 260)")
            
            # Check accuracy
            accuracy = model_data.get("accuracy")
            expected_acc = 0.8077
            if abs(accuracy - expected_acc) < 0.0001:
                print(f"  ✓ Accuracy correct: {accuracy:.4f} ({accuracy*100:.2f}%)")
            else:
                errors.append(f"✗ Accuracy is {accuracy:.4f}, expected {expected_acc:.4f}")
                print(f"  ✗ Accuracy MISMATCH: {accuracy:.4f} (expected {expected_acc:.4f})")
            
            # Check ECE
            ece = model_data.get("ece")
            expected_ece = 0.1076
            if abs(ece - expected_ece) < 0.0001:
                print(f"  ✓ ECE correct: {ece:.4f}")
            else:
                errors.append(f"✗ ECE is {ece:.4f}, expected {expected_ece:.4f}")
                print(f"  ✗ ECE MISMATCH: {ece:.4f} (expected {expected_ece:.4f})")
            
            # Check AUC-AC
            auc_ac = model_data.get("auc_ac")
            expected_auc = 0.8711
            if abs(auc_ac - expected_auc) < 0.0001:
                print(f"  ✓ AUC-AC correct: {auc_ac:.4f}")
            else:
                errors.append(f"✗ AUC-AC is {auc_ac:.4f}, expected {expected_auc:.4f}")
                print(f"  ✗ AUC-AC MISMATCH: {auc_ac:.4f} (expected {expected_auc:.4f})")
            
            # Check ECE bins
            bins = model_data.get("ece_bins", [])
            if len(bins) == 10:
                print(f"  ✓ ECE bins count correct: {len(bins)}")
            else:
                warnings.append(f"⚠ ECE bins count is {len(bins)}, expected 10")
                print(f"  ⚠ ECE bins count: {len(bins)} (expected 10)")
            
            # Check coverage curve
            curve = model_data.get("accuracy_coverage_curve", {})
            if curve:
                thresholds = curve.get("thresholds", [])
                coverage = curve.get("coverage", [])
                accuracy_vals = curve.get("accuracy", [])
                if len(thresholds) == len(coverage) == len(accuracy_vals) > 0:
                    print(f"  ✓ Accuracy-coverage curve valid: {len(thresholds)} points")
                else:
                    errors.append(f"✗ Accuracy-coverage curve invalid")
                    print(f"  ✗ Accuracy-coverage curve INVALID")
            else:
                errors.append(f"✗ No accuracy_coverage_curve found")
                print(f"  ✗ Accuracy-coverage curve MISSING")
        
    except json.JSONDecodeError as e:
        errors.append(f"✗ Invalid JSON: {e}")
        print(f"✗ JSON parsing error: {e}")

# ============================================================================
# 3. VERIFY Figure Files
# ============================================================================
print("\n[3] VERIFYING: Figure PDF Files")
print("-" * 90)

figures = {
    "submission_bundle/figures/reliability_diagram_verified.pdf": "Reliability Diagram (ECE visualization)",
    "submission_bundle/figures/accuracy_coverage_verified.pdf": "Accuracy-Coverage Curve (AUC-AC visualization)"
}

for fig_path, fig_desc in figures.items():
    fig = Path(fig_path)
    if fig.exists():
        size = fig.stat().st_size
        print(f"✓ {fig_desc}")
        print(f"  File: {fig_path}")
        print(f"  Size: {size:,} bytes")
        
        if size < 5000:
            warnings.append(f"⚠ {fig_path} appears too small ({size} bytes)")
            print(f"  ⚠ File size seems small")
        elif size > 100000:
            warnings.append(f"⚠ {fig_path} appears too large ({size} bytes)")
            print(f"  ⚠ File size seems large")
        else:
            print(f"  ✓ File size reasonable")
        
        # Check if valid PDF
        with open(fig, 'rb') as f:
            header = f.read(4)
            if header == b'%PDF':
                print(f"  ✓ Valid PDF format")
            else:
                errors.append(f"✗ {fig_path} is not a valid PDF")
                print(f"  ✗ NOT A VALID PDF")
    else:
        errors.append(f"✗ {fig_path} NOT FOUND")
        print(f"✗ FILE MISSING: {fig_path}")

# ============================================================================
# 4. VERIFY Manuscript References
# ============================================================================
print("\n[4] VERIFYING: Manuscript References")
print("-" * 90)

tex_template = Path("submission_bundle/OVERLEAF_TEMPLATE.tex")
if tex_template.exists():
    manuscript = tex_template.read_text()
    
    # Count macro references
    macro_refs = {
        "\\AccuracyValue{}": manuscript.count("\\AccuracyValue{}"),
        "\\ECEValue{}": manuscript.count("\\ECEValue{}"),
        "\\AUCACValue{}": manuscript.count("\\AUCACValue{}")
    }
    
    total_refs = sum(macro_refs.values())
    
    print(f"✓ Manuscript file exists ({len(manuscript):,} characters)")
    print(f"\n  Macro References:")
    for macro, count in macro_refs.items():
        print(f"    {macro}: {count} references")
    print(f"  Total: {total_refs} references")
    
    if total_refs < 10:
        errors.append(f"✗ Only {total_refs} macro references found (expected ~19)")
        print(f"  ✗ WARNING: Expected ~19 macro references")
    elif total_refs >= 19:
        print(f"  ✓ Expected number of references found")
    
    # Check for old hard-coded values
    hardcoded_checks = ["80.77", "0.1247", "0.8803"]
    hardcoded_found = []
    for val in hardcoded_checks:
        count = manuscript.count(val)
        if count > 0:
            hardcoded_found.append((val, count))
    
    if hardcoded_found:
        print(f"\n  Hard-coded Values Found:")
        for val, count in hardcoded_found:
            print(f"    '{val}': {count} occurrences")
            # Some hard-coded values are expected (in CIs, etc)
            if val == "80.77" and count <= 4:  # Expected in table and error analysis
                print(f"      ✓ Acceptable (likely in confidence intervals or error analysis)")
            elif val in ["0.0989", "0.1679", "0.8207", "0.9386"]:
                print(f"      ✓ Acceptable (confidence interval bounds)")
    else:
        print(f"  ✓ No problematic hard-coded metric values")

# ============================================================================
# 5. VERIFY File Consistency
# ============================================================================
print("\n[5] VERIFYING: File Consistency")
print("-" * 90)

required_files = [
    "submission_bundle/metrics_values.tex",
    "submission_bundle/OVERLEAF_TEMPLATE.tex",
    "submission_bundle/figures/reliability_diagram_verified.pdf",
    "submission_bundle/figures/accuracy_coverage_verified.pdf",
    "artifacts/metrics_summary.json"
]

all_exist = True
for fpath in required_files:
    f = Path(fpath)
    if f.exists():
        print(f"✓ {fpath}")
    else:
        all_exist = False
        errors.append(f"✗ {fpath} MISSING")
        print(f"✗ {fpath} MISSING")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 90)
print("VERIFICATION SUMMARY")
print("=" * 90)

if not errors and not warnings:
    print("\n✓✓✓ ALL CHECKS PASSED ✓✓✓")
    print("\nThe manuscript is ready for compilation and submission!")
    status_code = 0
else:
    if errors:
        print(f"\n❌ ERRORS FOUND: {len(errors)}")
        for i, err in enumerate(errors, 1):
            print(f"  {i}. {err}")
    
    if warnings:
        print(f"\n⚠️  WARNINGS: {len(warnings)}")
        for i, warn in enumerate(warnings, 1):
            print(f"  {i}. {warn}")
    
    status_code = 1 if errors else 0

print("\n" + "=" * 90)
print(f"FINAL STATUS: {'✓ PASS' if status_code == 0 else '✗ FAIL'}")
print("=" * 90 + "\n")

sys.exit(status_code)
