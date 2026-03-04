#!/usr/bin/env python3
"""
Verify paper artifacts for CalibraTeach reproducibility.

Validates:
- Required artifact directories exist
- Quickstart output exists and matches schema
- If metrics_summary.json exists, validate required keys
- Generates verification report
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple


REQUIRED_DIRS = [
    "artifacts",
    "artifacts/quickstart",
    "artifacts/verification",
]

REQUIRED_QUICKSTART_FIELDS = {
    "run_id": str,
    "smoke": bool,
    "n": int,
    "tau": float,
    "examples": list,
}

REQUIRED_EXAMPLE_FIELDS = {
    "claim": str,
    "pred_label": str,
    "confidence": (int, float),
    "abstained": bool,
    "top_evidence": list,
    "stage_latency_ms": dict,
}

REQUIRED_LATENCY_FIELDS = [
    "retrieval",
    "filtering",
    "nli",
    "aggregation",
    "calibration",
    "selective",
    "explanation",
    "total",
]

REQUIRED_METRICS_KEYS = ["accuracy", "ece", "auc_ac"]


def check_directories() -> Tuple[bool, List[str]]:
    """Check if required directories exist."""
    errors = []
    for dir_path in REQUIRED_DIRS:
        path = Path(dir_path)
        if not path.exists():
            # Create missing directories
            path.mkdir(parents=True, exist_ok=True)
            print(f"[OK] Created directory: {dir_path}")
    return True, errors


def validate_quickstart_schema(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate quickstart output matches required schema."""
    errors = []
    
    # Check top-level fields
    for field, field_type in REQUIRED_QUICKSTART_FIELDS.items():
        if field not in data:
            errors.append(f"Missing required field: {field}")
        elif not isinstance(data[field], field_type):
            errors.append(
                f"Field '{field}' has wrong type: "
                f"expected {field_type.__name__}, got {type(data[field]).__name__}"
            )
    
    if "examples" not in data or not isinstance(data["examples"], list):
        return False, errors
    
    # Check each example
    for i, example in enumerate(data["examples"]):
        for field, field_type in REQUIRED_EXAMPLE_FIELDS.items():
            if field not in example:
                errors.append(f"Example {i}: Missing field '{field}'")
            elif not isinstance(example[field], field_type):
                errors.append(
                    f"Example {i}: Field '{field}' has wrong type: "
                    f"expected {field_type}, got {type(example[field])}"
                )
        
        # Check stage_latency_ms fields
        if "stage_latency_ms" in example:
            latency = example["stage_latency_ms"]
            for lat_field in REQUIRED_LATENCY_FIELDS:
                if lat_field not in latency:
                    errors.append(
                        f"Example {i}: Missing latency field '{lat_field}'"
                    )
                elif not isinstance(latency[lat_field], (int, float)):
                    errors.append(
                        f"Example {i}: Latency field '{lat_field}' must be numeric"
                    )
        
        # Check pred_label values
        if "pred_label" in example:
            valid_labels = {"SUPPORTED", "REFUTED", "ABSTAIN"}
            if example["pred_label"] not in valid_labels:
                errors.append(
                    f"Example {i}: Invalid pred_label '{example['pred_label']}', "
                    f"must be one of {valid_labels}"
                )
    
    return len(errors) == 0, errors


def validate_metrics_summary() -> Tuple[bool, List[str], Dict[str, Any]]:
    """Validate metrics_summary.json if it exists."""
    metrics_path = Path("artifacts/metrics_summary.json")
    
    if not metrics_path.exists():
        return True, [], {}  # Not required, so no error
    
    errors = []
    metrics = {}
    
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Check if primary model exists
        if "primary_model" not in data:
            errors.append("metrics_summary.json: Missing 'primary_model' field")
            return False, errors, {}
        
        primary_model = data["primary_model"]
        
        # Check if models dict exists
        if "models" not in data or primary_model not in data["models"]:
            errors.append(
                f"metrics_summary.json: Primary model '{primary_model}' not found in 'models'"
            )
            return False, errors, {}
        
        model_data = data["models"][primary_model]
        
        # Check required metric keys
        for key in REQUIRED_METRICS_KEYS:
            if key not in model_data:
                errors.append(f"metrics_summary.json: Missing required metric '{key}'")
            else:
                metrics[key] = model_data[key]
        
        return len(errors) == 0, errors, metrics
        
    except json.JSONDecodeError as e:
        errors.append(f"metrics_summary.json: Invalid JSON: {e}")
        return False, errors, {}
    except Exception as e:
        errors.append(f"metrics_summary.json: Error reading file: {e}")
        return False, errors, {}


def generate_report(
    quickstart_valid: bool,
    quickstart_errors: List[str],
    metrics_valid: bool,
    metrics_errors: List[str],
    metrics: Dict[str, Any],
    quickstart_path: str,
) -> str:
    """Generate verification report markdown."""
    timestamp = Path(quickstart_path).exists()
    
    report = f"""# CalibraTeach Paper Artifacts Verification Report

**Generated**: {Path(quickstart_path).stat().st_mtime if timestamp else 'N/A'}  
**Status**: {'[PASS]' if quickstart_valid and metrics_valid else '[FAIL]'}

---

## Quickstart Demo Output

**Path**: `{quickstart_path}`  
**Status**: {'[VALID]' if quickstart_valid else '[INVALID]'}

"""
    
    if quickstart_errors:
        report += "**Errors**:\n"
        for error in quickstart_errors:
            report += f"- {error}\n"
        report += "\n"
    else:
        report += "All required fields present and valid.\n\n"
    
    report += "---\n\n## Metrics Summary\n\n"
    
    if metrics:
        report += f"**Path**: `artifacts/metrics_summary.json`  \n"
        report += f"**Status**: {'[VALID]' if metrics_valid else '[INVALID]'}\n\n"
        
        if metrics_valid:
            report += "**Primary Metrics**:\n"
            for key, value in metrics.items():
                if isinstance(value, float):
                    report += f"- `{key}`: {value:.4f}\n"
                else:
                    report += f"- `{key}`: {value}\n"
            report += "\n"
    else:
        report += "**Path**: `artifacts/metrics_summary.json`  \n"
        report += "**Status**: Not found (optional)\n\n"
    
    if metrics_errors:
        report += "**Errors**:\n"
        for error in metrics_errors:
            report += f"- {error}\n"
        report += "\n"
    
    report += "---\n\n## Schema Validation\n\n"
    report += "### Required Top-Level Fields\n"
    for field, field_type in REQUIRED_QUICKSTART_FIELDS.items():
        report += f"- `{field}`: {field_type.__name__}\n"
    
    report += "\n### Required Example Fields\n"
    for field, field_type in REQUIRED_EXAMPLE_FIELDS.items():
        type_name = field_type.__name__ if hasattr(field_type, '__name__') else str(field_type)
        report += f"- `{field}`: {type_name}\n"
    
    report += "\n### Required Latency Fields\n"
    for field in REQUIRED_LATENCY_FIELDS:
        report += f"- `{field}`\n"
    
    report += "\n---\n\n"
    report += f"**Overall Status**: {'[PASS] All checks passed' if quickstart_valid and metrics_valid else '[FAIL] Validation failed'}\n"
    
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Verify CalibraTeach paper artifacts"
    )
    parser.add_argument(
        "--quickstart",
        type=str,
        default="artifacts/quickstart/output.json",
        help="Path to quickstart output (default: artifacts/quickstart/output.json)",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="artifacts/verification/VerificationReport.md",
        help="Path to output report (default: artifacts/verification/VerificationReport.md)",
    )
    
    args = parser.parse_args()
    
    print("CalibraTeach Paper Artifacts Verification")
    print("=" * 50)
    print()
    
    # Check directories
    print("Checking required directories...")
    dirs_ok, dir_errors = check_directories()
    print()
    
    # Check quickstart output
    print(f"Validating quickstart output: {args.quickstart}")
    quickstart_path = Path(args.quickstart)
    
    if not quickstart_path.exists():
        print(f"[ERROR] Quickstart output not found at {args.quickstart}")
        print("   Run 'make quickstart' first to generate artifacts")
        return 1
    
    try:
        with open(quickstart_path, "r", encoding="utf-8") as f:
            quickstart_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"[ERROR] Invalid JSON in {args.quickstart}: {e}")
        return 1
    except Exception as e:
        print(f"[ERROR] reading {args.quickstart}: {e}")
        return 1
    
    quickstart_valid, quickstart_errors = validate_quickstart_schema(quickstart_data)
    
    if quickstart_valid:
        print(f"[OK] Quickstart output schema valid")
        print(f"  - Examples: {len(quickstart_data['examples'])}")
        print(f"  - Mode: {'smoke' if quickstart_data.get('smoke') else 'full'}")
    else:
        print(f"[ERROR] Quickstart output schema invalid:")
        for error in quickstart_errors:
            print(f"   - {error}")
    
    print()
    
    # Check metrics summary
    print("Validating metrics summary (optional)...")
    metrics_valid, metrics_errors, metrics = validate_metrics_summary()
    
    if metrics:
        if metrics_valid:
            print("[OK] Metrics summary valid")
            for key, value in metrics.items():
                formatted_value = f"{value:.4f}" if isinstance(value, float) else str(value)
                print(f"  - {key}: {formatted_value}")
        else:
            print("[ERROR] Metrics summary invalid:")
            for error in metrics_errors:
                print(f"   - {error}")
    else:
        print("  (metrics_summary.json not found - optional)")
    
    print()
    
    # Generate report
    print(f"Generating verification report: {args.report}")
    report = generate_report(
        quickstart_valid,
        quickstart_errors,
        metrics_valid,
        metrics_errors,
        metrics,
        args.quickstart,
    )
    
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"[OK] Report written to: {args.report}")
    print()
    
    # Final status
    overall_valid = quickstart_valid and (metrics_valid or not metrics)
    
    if overall_valid:
        print("[PASS] All validations passed!")
        return 0
    else:
        print("[ERROR] Validation failed - see errors above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
