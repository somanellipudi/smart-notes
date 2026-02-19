"""
Demo: Selective Prediction and Conformal Guarantees

This script demonstrates how to use selective prediction and conformal
prediction to provide error guarantees for claim verification.

Usage:
    python examples/demo_selective_conformal.py
"""

import numpy as np
from src.evaluation.selective_prediction import (
    selective_prediction_analysis,
    format_selective_prediction_summary,
    plot_risk_coverage_curve
)
from src.evaluation.conformal import (
    conformal_prediction_calibration,
    format_conformal_summary,
    expected_calibration_error
)
from src.evaluation.integration_helpers import (
    combine_metrics,
    format_threshold_recommendation
)
from src.reporting.research_report import (
    ResearchReportBuilder,
    SessionMetadata,
    IngestionReport,
    VerificationSummary,
    ClaimEntry
)
from datetime import datetime


def generate_synthetic_verification_data(n=200, seed=42):
    """
    Generate synthetic claim verification data for demonstration.
    
    Simulates a realistic scenario where:
    - High confidence predictions are more likely correct
    - Low confidence predictions have more errors
    """
    np.random.seed(seed)
    
    # Generate confidence scores
    scores = np.random.beta(a=2, b=1, size=n)  # Skewed toward high confidence
    
    # Generate predictions (3 classes: ENTAIL=0, NEUTRAL=1, CONTRADICT=2)
    predictions = np.random.choice([0, 1, 2], size=n, p=[0.5, 0.3, 0.2])
    
    # Generate targets based on confidence
    # Higher confidence → more likely correct
    targets = predictions.copy()
    for i in range(n):
        error_prob = 0.4 * (1 - scores[i])  # Error probability inversely proportional to confidence
        if np.random.rand() < error_prob:
            # Introduce error
            targets[i] = np.random.choice([x for x in [0, 1, 2] if x != predictions[i]])
    
    return scores, predictions, targets


def demo_selective_prediction():
    """Demonstrate selective prediction (risk-coverage tradeoff)."""
    print("=" * 80)
    print("DEMO 1: SELECTIVE PREDICTION (Risk-Coverage Tradeoff)")
    print("=" * 80)
    print()
    
    # Generate data
    scores, predictions, targets = generate_synthetic_verification_data(n=200)
    
    print(f"Dataset: {len(scores)} claims")
    print(f"Overall accuracy: {np.mean(predictions == targets):.1%}")
    print()
    
    # Run selective prediction analysis
    result = selective_prediction_analysis(
        scores, predictions, targets,
        target_risk=0.1,  # Target: ≤10% error rate
        num_thresholds=50
    )
    
    # Print summary
    print(format_selective_prediction_summary(result))
    
    # Show what happens at different thresholds
    print("\nRisk-Coverage at Different Thresholds:")
    print("-" * 60)
    print(f"{'Threshold':<12} {'Coverage':<12} {'Risk':<12}")
    print("-" * 60)
    
    curve = result.risk_coverage_curve
    # Show 5 points across the curve
    indices = [0, len(curve)//4, len(curve)//2, 3*len(curve)//4, len(curve)-1]
    for idx in indices:
        point = curve[idx]
        print(f"{point.threshold:<12.3f} {point.coverage:<12.1%} {point.risk:<12.1%}")
    
    print()
    return result


def demo_conformal_prediction():
    """Demonstrate conformal prediction (distribution-free guarantees)."""
    print("=" * 80)
    print("DEMO 2: CONFORMAL PREDICTION (Distribution-Free Guarantees)")
    print("=" * 80)
    print()
    
    # Generate data
    scores, predictions, targets = generate_synthetic_verification_data(n=200)
    
    # Split into calibration and test (50/50)
    n_cal = 100
    cal_scores = scores[:n_cal]
    cal_preds = predictions[:n_cal]
    cal_targets = targets[:n_cal]
    
    test_scores = scores[n_cal:]
    test_preds = predictions[n_cal:]
    test_targets = targets[n_cal:]
    
    print(f"Calibration set: {n_cal} claims")
    print(f"Test set: {len(test_scores)} claims")
    print()
    
    # Run conformal calibration
    result = conformal_prediction_calibration(
        cal_scores, cal_preds, cal_targets,
        alpha=0.1  # 90% confidence
    )
    
    # Print summary
    print(format_conformal_summary(result))
    
    # Validate on test set
    from src.evaluation.conformal import validate_conformal_coverage
    test_correct = (test_preds == test_targets).astype(int)
    error_rate, coverage, num_accepted, num_errors = validate_conformal_coverage(
        test_scores, test_correct, result.threshold
    )
    
    print("\nTest Set Validation:")
    print("-" * 60)
    print(f"Predictions accepted: {num_accepted}/{len(test_scores)} ({coverage:.1%})")
    print(f"Errors on accepted: {num_errors}/{num_accepted}")
    print(f"Error rate: {error_rate:.1%}")
    print(f"Guaranteed bound: ≤{result.alpha:.1%} with {result.coverage_guarantee:.1%} confidence")
    
    if error_rate <= result.alpha:
        print("✅ Guarantee SATISFIED on test set!")
    else:
        print("⚠️ Guarantee exceeded (can happen with 10% probability)")
    
    print()
    return result


def demo_calibration_quality():
    """Demonstrate calibration quality measurement."""
    print("=" * 80)
    print("DEMO 3: CALIBRATION QUALITY (Expected Calibration Error)")
    print("=" * 80)
    print()
    
    # Generate well-calibrated data
    scores, predictions, targets = generate_synthetic_verification_data(n=200)
    is_correct = (predictions == targets).astype(int)
    
    # Compute ECE
    ece = expected_calibration_error(scores, is_correct, num_bins=10)
    
    print(f"Expected Calibration Error (ECE): {ece:.4f}")
    print()
    print("Interpretation:")
    if ece < 0.05:
        print("  ✅ EXCELLENT: Model is well-calibrated (ECE < 0.05)")
    elif ece < 0.10:
        print("  ✅ GOOD: Model is reasonably calibrated (ECE < 0.10)")
    elif ece < 0.15:
        print("  ⚠️ MODERATE: Model has some calibration error (ECE < 0.15)")
    else:
        print("  ❌ POOR: Model is poorly calibrated (ECE ≥ 0.15)")
    
    print()
    print("ECE measures how well confidence scores match actual accuracy.")
    print("Lower ECE = better calibration (closer to perfect reliability).")
    print()


def demo_combined_analysis():
    """Demonstrate combined selective + conformal analysis."""
    print("=" * 80)
    print("DEMO 4: COMBINED ANALYSIS (Selective + Conformal)")
    print("=" * 80)
    print()
    
    # Generate data
    scores, predictions, targets = generate_synthetic_verification_data(n=300)
    
    # Combine both approaches
    sp_metrics, cp_metrics = combine_metrics(
        scores.tolist(), predictions.tolist(), targets.tolist(),
        target_risk=0.05,  # 5% target error
        alpha=0.1,  # 90% confidence
        calibration_split=0.5
    )
    
    # Print recommendation
    print(format_threshold_recommendation(sp_metrics, cp_metrics))
    
    # Show comparison
    print("\nComparison:")
    print("-" * 60)
    print(f"{'Approach':<25} {'Threshold':<15} {'Basis':<40}")
    print("-" * 60)
    print(f"{'Selective Prediction':<25} {sp_metrics['optimal_threshold']:<15.3f} {'Minimize risk for target coverage':<40}")
    print(f"{'Conformal Prediction':<25} {cp_metrics['threshold']:<15.3f} {'Distribution-free guarantee':<40}")
    print(f"{'Combined (max)':<25} {max(sp_metrics['optimal_threshold'], cp_metrics['threshold']):<15.3f} {'Conservative threshold':<40}")
    print()


def demo_report_integration():
    """Demonstrate integration with report generation."""
    print("=" * 80)
    print("DEMO 5: REPORT INTEGRATION")
    print("=" * 80)
    print()
    
    # Generate verification data
    scores, predictions, targets = generate_synthetic_verification_data(n=150)
    is_correct = (predictions == targets)
    
    # Compute metrics
    sp_metrics, cp_metrics = combine_metrics(
        scores.tolist(), predictions.tolist(), targets.tolist(),
        target_risk=0.1, alpha=0.1
    )
    
    # Create report components
    session_metadata = SessionMetadata(
        session_id="demo_selective_conformal",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        seed=42,
        language_model="gpt-4",
        embedding_model="text-embedding-ada-002",
        nli_model="cross-encoder/nli-deberta-v3-base",
        inputs_used=["demo_lecture.pdf"]
    )
    
    ingestion_report = IngestionReport(
        total_pages=50,
        pages_ocr=45,
        headers_removed=50,
        footers_removed=50,
        watermarks_removed=0,
        chunks_total_all_sources=200,
        avg_chunk_size_all_sources=512,
        extraction_methods=["pdf_text", "pdf_ocr"]
    )
    
    # Compute ECE
    ece = expected_calibration_error(scores, is_correct.astype(int))
    
    verification_summary = VerificationSummary(
        total_claims=len(scores),
        verified_count=int(np.sum(is_correct)),
        low_confidence_count=int(np.sum((scores >= 0.5) & (scores < 0.8))),
        rejected_count=int(np.sum(~is_correct)),
        avg_confidence=float(np.mean(scores)),
        top_rejection_reasons=[("Low evidence", 10), ("Contradicted", 5)],
        calibration_metrics={"ece": ece, "brier": 0.15},
        selective_prediction=sp_metrics,
        conformal_prediction=cp_metrics
    )
    
    # Create sample claims
    claims = [
        ClaimEntry(
            claim_text="Machine learning models require training data",
            status="VERIFIED",
            confidence=0.95,
            evidence_count=3,
            top_evidence="ML models learn patterns from data...",
            page_num=12
        ),
        ClaimEntry(
            claim_text="Neural networks have no hyperparameters",
            status="REJECTED",
            confidence=0.15,
            evidence_count=0,
            top_evidence="No supporting evidence found",
            page_num=None
        )
    ]
    
    # Build report
    builder = ResearchReportBuilder()
    builder.add_session_metadata(session_metadata)
    builder.add_ingestion_report(ingestion_report)
    builder.add_verification_summary(verification_summary)
    builder.add_claims(claims)
    
    markdown_report = builder.build_markdown()
    
    # Show selective prediction section
    lines = markdown_report.split('\n')
    in_section = False
    section_lines = []
    
    for line in lines:
        if "## Confidence Guarantees" in line:
            in_section = True
        elif in_section and line.startswith("##"):
            break
        elif in_section:
            section_lines.append(line)
    
    print("Report section generated:")
    print("-" * 60)
    print('\n'.join(section_lines[:40]))  # First 40 lines
    print()
    print("✅ Full report can be exported to MD/HTML/JSON formats")
    print()


def main():
    """Run all demos."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "SELECTIVE PREDICTION & CONFORMAL GUARANTEES DEMO" + " " * 15 + "║")
    print("╚" + "=" * 78 + "╝")
    print("\n")
    
    # Run demos
    demo_selective_prediction()
    input("Press Enter to continue to next demo...")
    print("\n")
    
    demo_conformal_prediction()
    input("Press Enter to continue to next demo...")
    print("\n")
    
    demo_calibration_quality()
    input("Press Enter to continue to next demo...")
    print("\n")
    
    demo_combined_analysis()
    input("Press Enter to continue to next demo...")
    print("\n")
    
    demo_report_integration()
    
    print("=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print()
    print("Key Takeaways:")
    print("  1. Selective Prediction: Control error rate by rejecting uncertain predictions")
    print("  2. Conformal Prediction: Distribution-free confidence guarantees")
    print("  3. ECE: Measure calibration quality")
    print("  4. Combined: Use max threshold for conservative error control")
    print("  5. Integration: Seamlessly integrates with existing reports")
    print()


if __name__ == "__main__":
    main()
