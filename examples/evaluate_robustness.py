"""
Example: Evaluate robustness to document ingestion noise.

This script demonstrates how to use the CSBenchmarkRunner to evaluate
the verification pipeline's robustness to common document ingestion issues:
- Headers/footers from scanned documents
- OCR character substitution errors
- Two-column layout interleaving
"""

import json
from pathlib import Path
from src.evaluation.cs_benchmark_runner import CSBenchmarkRunner


def main():
    """Run robustness evaluation on CSClaimBench v1.0."""
    
    # Configuration
    benchmark_path = Path("evaluation/cs_benchmark/csclaimbench_v1.jsonl")
    output_dir = Path("outputs/robustness_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Robustness Evaluation: Ingestion Noise Impact")
    print("=" * 80)
    
    # Initialize runner
    print("\n1. Initializing benchmark runner...")
    runner = CSBenchmarkRunner(
        dataset_path=str(benchmark_path),
        batch_size=4,
        device="cpu",
        seed=42,
        log_predictions=False
    )
    print(f"   ✓ Loaded {len(runner.dataset)} examples")
    
    # Run robustness evaluation
    print("\n2. Running robustness evaluation...")
    print("   Testing clean baseline and 4 noise variants:")
    print("   - Clean (baseline)")
    print("   - Headers/Footers (simulated scanned document headers)")
    print("   - OCR Typos (character substitution errors: l↔1, O↔0, rn↔m, I↔l)")
    print("   - Column Shuffle (two-column layout interleaving)")
    print("   - All (combined noise)")
    print()
    
    results = runner.run_ingestion_robustness_eval(
        sample_size=15,  # Use 15 examples for demo
        noise_types=["headers_footers", "ocr_typos", "column_shuffle", "all"]
    )
    
    # Display results
    print("=" * 80)
    print("ROBUSTNESS EVALUATION RESULTS")
    print("=" * 80)
    
    # Clean baseline
    clean_metrics = results["clean"].metrics
    print(f"\n📊 BASELINE (Clean Text):")
    print(f"   Accuracy:    {clean_metrics.accuracy:.3f}")
    print(f"   ECE:         {clean_metrics.ece:.3f} (calibration)")
    print(f"   Brier:       {clean_metrics.brier_score:.3f} (confidence MSE)")
    print(f"   F1-ENTAIL:       {clean_metrics.F1_verified:.3f}")
    print(f"   F1-CONTRADICT:   {clean_metrics.F1_rejected:.3f}")
    
    # Degradation analysis
    print(f"\n📉 NOISE ROBUSTNESS DEGRADATION:")
    print(f"    {'-' * 75}")
    print(f"    {'Noise Type':<20} {'Acc Drop':<12} {'ECE Δ':<12} {'Brier Δ':<15}")
    print(f"    {'-' * 75}")
    
    for noise_type in ["headers_footers", "ocr_typos", "column_shuffle", "all"]:
        noisy_result = results[noise_type]
        noisy_metrics = noisy_result.metrics
        
        # Get degradation metrics
        if noisy_metrics.ingestion_noise_results:
            degradation = noisy_metrics.ingestion_noise_results[noise_type]
            
            acc_drop = degradation["accuracy_drop"]
            ece_increase = degradation["ece_increase"]
            brier_increase = degradation["brier_increase"]
            
            # Visual indicator for severity
            severity = ""
            if abs(acc_drop) > 0.2:
                severity = " ⚠️  HIGH"
            elif abs(acc_drop) > 0.1:
                severity = " ⚡ MEDIUM"
            else:
                severity = " ✓ LOW"
            
            print(f"    {noise_type:<20} {acc_drop:+.3f}      {ece_increase:+.3f}      {brier_increase:+.3f}{severity}")
    
    # Per-type analysis
    print(f"\n📋 DETAILED PER-NOISE-TYPE RESULTS:")
    print()
    
    for noise_type in ["headers_footers", "ocr_typos", "column_shuffle", "all"]:
        noisy_result = results[noise_type]
        noisy_metrics = noisy_result.metrics
        
        print(f"   {noise_type.upper().replace('_', ' ')}:")
        print(f"   {'-' * 70}")
        print(f"      Accuracy:      {clean_metrics.accuracy:.3f} → {noisy_metrics.accuracy:.3f}")
        print(f"      Calibration:   {clean_metrics.ece:.3f} → {noisy_metrics.ece:.3f}")
        print(f"      Evidence Cov:  {clean_metrics.evidence_coverage_rate:.1%} → {noisy_metrics.evidence_coverage_rate:.1%}")
        
        if noisy_metrics.ingestion_noise_results:
            degradation = noisy_metrics.ingestion_noise_results[noise_type]
            print(f"      => Robust?  {('No ❌' if abs(degradation['accuracy_drop']) > 0.15 else 'Reasonably ⚡' if abs(degradation['accuracy_drop']) > 0.05 else 'Yes ✓')}")
        print()
    
    # Save detailed results
    print(f"\n💾 Saving detailed results...")
    for noise_type, result in results.items():
        json_path = output_dir / f"robustness_{noise_type}_metrics.json"
        result.to_json(json_path)
        print(f"    ✓ {json_path.name}")
    
    # Create summary report
    summary_path = output_dir / "robustness_summary.json"
    summary = {}
    
    for noise_type in ["headers_footers", "ocr_typos", "column_shuffle", "all"]:
        if noise_type == "clean":
            continue
        result = results[noise_type]
        metrics = result.metrics
        
        if metrics.ingestion_noise_results:
            degradation = metrics.ingestion_noise_results[noise_type]
            summary[noise_type] = {
                "accuracy_drop": degradation["accuracy_drop"],
                "ece_increase": degradation["ece_increase"],
                "brier_increase": degradation["brier_increase"],
                "evidence_coverage_drop": degradation["evidence_coverage_drop"],
                "baseline": {
                    "accuracy": degradation["clean_accuracy"],
                    "ece": clean_metrics.ece,
                    "brier": clean_metrics.brier_score
                },
                "under_noise": {
                    "accuracy": degradation["noisy_accuracy"],
                    "ece": metrics.ece,
                    "brier": metrics.brier_score
                }
            }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"    ✓ {summary_path.name}")
    
    print("\n" + "=" * 80)
    print("Robustness evaluation complete!")
    print(f"Results saved to: {output_dir.absolute()}")
    print("=" * 80)
    
    # Key insights
    print("\n💡 KEY INSIGHTS:")
    print()
    for noise_type in ["headers_footers", "ocr_typos", "column_shuffle", "all"]:
        if noise_type == "clean":
            continue
        degradation = results[noise_type].metrics.ingestion_noise_results.get(noise_type, {})
        acc_drop = degradation.get("accuracy_drop", 0)
        
        if abs(acc_drop) > 0.2:
            print(f"   ⚠️  {noise_type}: System is VULNERABLE to this noise type")
        elif abs(acc_drop) > 0.1:
            print(f"   ⚡ {noise_type}: System is MODERATELY affected by this noise")
        else:
            print(f"   ✓ {noise_type}: System is ROBUST to this noise")
    
    print()


if __name__ == "__main__":
    main()
