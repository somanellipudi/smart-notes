#!/usr/bin/env python
"""
Comprehensive Evaluation Suite - Run All Tests & Benchmarks
Captures results for paper publication
"""

import json
import sys
import subprocess
from pathlib import Path
from datetime import datetime
import logging
import argparse
from src.evaluation.runner import run as run_eval
from src.config.verification_config import VerificationConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results" / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def run_command(cmd, description):
    """Run a command and log results."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Running: {description}")
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info('='*80)
    
    try:
        result = subprocess.run(cmd, capture_output=False, text=True, cwd=PROJECT_ROOT)
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Failed to run {description}: {e}")
        return False

def main():
    """Run all evaluations."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="verifiable_full", choices=["baseline_retriever","baseline_nli","baseline_rag_nli","verifiable_full","all"], help="Evaluation mode to run")
    args = parser.parse_args()

    cfg = VerificationConfig.from_env()

    logger.info(f"Smart Notes Evaluation Suite")
    logger.info(f"Results Directory: {RESULTS_DIR}")
    logger.info(f"Start Time: {datetime.now().isoformat()}")
    
    # Test 1: Run CS Benchmark Ablation Studies
    logger.info("\n" + "="*80)
    logger.info("PHASE 1: ABLATION STUDIES (Core Verification Pipeline)")
    logger.info("="*80)
    
    ablation_cmd = [
        sys.executable, "scripts/run_cs_benchmark.py",
        "--dataset", "evaluation/cs_benchmark/cs_benchmark_dataset.jsonl",
        "--output-dir", str(RESULTS_DIR / "ablation_core"),
        "--seed", "42",
        "--sample-size", "50"  # Use 50 for reasonably fast run while still comprehensive
    ]
    success_ablation = run_command(ablation_cmd, "Ablation Studies - Core Pipeline")
    
    # Test 2: Run Robustness Tests
    logger.info("\n" + "="*80)
    logger.info("PHASE 2: ROBUSTNESS EVALUATION (Noise & Edge Cases)")
    logger.info("="*80)
    
    robustness_cmd = [
        sys.executable, "examples/evaluate_robustness.py"
    ]
    # Note: This might need adjustment based on the actual script
    # success_robustness = run_command(robustness_cmd, "Robustness Evaluation")
    
    # Test 3: Run pytest on evaluation tests
    logger.info("\n" + "="*80)
    logger.info("PHASE 3: UNIT & INTEGRATION TESTS")
    logger.info("="*80)
    
    pytest_cmd = [
        sys.executable, "-m", "pytest",
        "tests/test_evaluation_comprehensive.py",
        "-v", "--tb=short",
        f"--junit-xml={RESULTS_DIR / 'test_results.xml'}"
    ]
    success_pytest = run_command(pytest_cmd, "Pytest Suite")
    
    # Test 4: Specific smoke tests
    logger.info("\n" + "="*80)
    logger.info("PHASE 4: SMOKE TESTS (Key Functionality)")
    logger.info("="*80)
    
    smoke_tests = [
        "tests/test_benchmark_runner_smoke.py",
        "tests/test_robustness_runner_smoke.py",
        "tests/test_ablation_runner_smoke.py"
    ]
    
    all_smoke_passed = True
    for test in smoke_tests:
        cmd = [sys.executable, "-m", "pytest", test, "-v"]
        passed = run_command(cmd, f"Smoke test: {test}")
        all_smoke_passed = all_smoke_passed and passed
    
    # Run selected evaluation mode(s)
    modes_to_run = [args.mode] if args.mode != "all" else ["baseline_retriever","baseline_nli","baseline_rag_nli","verifiable_full"]
    for m in modes_to_run:
        logger.info(f"Running evaluation mode: {m}")
        run_eval(mode=m, output_dir=str(Path("outputs") / "benchmark_results" / "latest"))

    # Generate Summary Report
    logger.info("\n" + "="*80)
    logger.info("GENERATING SUMMARY REPORT")
    logger.info("="*80)
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "results_directory": str(RESULTS_DIR),
        "evaluation_phases": {
            "ablation_studies": "PASS" if success_ablation else "FAIL",
            "robustness_evaluation": "PENDING",
            "unit_tests": "PASS" if success_pytest else "FAIL",
            "smoke_tests": "PASS" if all_smoke_passed else "FAIL"
        },
        "datasets_tested": [
            "CS Benchmark Core",
            "CS Benchmark Hard",
            "CS Benchmark Easy",
            "Adversarial"
        ],
        "metrics_captured": [
            "accuracy",
            "F1_verified",
            "precision_verified",
            "recall_verified",
            "ECE (calibration)",
            "Brier score",
            "inference_time",
            "memory_usage"
        ],
        "paper_usage": "Results stored in evaluation/results/eval_[timestamp] for paper publication"
    }
    
    # Save summary
    summary_path = RESULTS_DIR / "evaluation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\n✓ Summary saved to: {summary_path}")
    
    # Print Summary
    logger.info("\n" + "="*80)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*80)
    logger.info(f"Results Directory: {RESULTS_DIR}")
    logger.info(f"Ablation Studies: {summary['evaluation_phases']['ablation_studies']}")
    logger.info(f"Unit Tests: {summary['evaluation_phases']['unit_tests']}")
    logger.info(f"Smoke Tests: {summary['evaluation_phases']['smoke_tests']}")
    logger.info("\nKey outputs:")
    logger.info(f"  - Ablation results: {RESULTS_DIR / 'ablation_core' / 'results.csv'}")
    logger.info(f"  - Detailed metrics: {RESULTS_DIR / 'ablation_core' / 'detailed_results/'}")
    logger.info(f"  - Summary text: {RESULTS_DIR / 'ablation_core' / 'ablation_summary.md'}")
    logger.info(f"  - Test results: {RESULTS_DIR / 'test_results.xml'}")
    
    logger.info("\nFor paper use:")
    logger.info(f"  Reference location: evaluation/results/eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    logger.info("  Include results.csv and ablation_summary.md in manuscript")
    
    return 0 if (success_ablation and success_pytest and all_smoke_passed) else 1

if __name__ == "__main__":
    sys.exit(main())
