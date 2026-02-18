"""
Example script to run the CS benchmark and generate output artifacts.

This script demonstrates:
1. Loading CSClaimBench v1.0 dataset
2. Running verification pipeline on all examples
3. Computing evaluation metrics (accuracy, F1, ECE, Brier, etc.)
4. Saving output artifacts (predictions.jsonl, metrics.json, results.csv)
"""

import json
from pathlib import Path
from src.evaluation.cs_benchmark_runner import CSBenchmarkRunner

def main():
    # Configuration
    benchmark_path = Path("evaluation/cs_benchmark/csclaimbench_v1.jsonl")
    output_dir = Path("outputs/benchmark_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("CS Benchmark Evaluation - CSClaimBench v1.0")
    print("=" * 70)
    
    # Initialize runner
    print("\n1. Initializing benchmark runner...")
    runner = CSBenchmarkRunner(
        dataset_path=str(benchmark_path),
        batch_size=8,
        device="cpu",  # Change to "cuda" if GPU available
        seed=42,
        log_predictions=True
    )
    print(f"   ✓ Loaded {len(runner.dataset)} examples from dataset")
    
    # Run benchmark
    print("\n2. Running verification pipeline...")
    print("   (This may take a few minutes for the full dataset)")
    
    # For quick demo, use sample_size=20; remove for full dataset
    result = runner.run(
        config={
            "use_retrieval": True,
            "use_nli": True,
            "use_ensemble": False,
            "verify_threshold": 0.55,
            "low_conf_threshold": 0.35,
            "enable_contradiction_gate": True,
            "contradiction_threshold": 0.6
        },
        sample_size=20  # Remove this line to run on full dataset
    )
    
    print(f"   ✓ Processed {result.metrics.total_claims} claims")
    
    # Display summary metrics
    print("\n3. Evaluation Metrics Summary:")
    print("-" * 70)
    print(f"   Accuracy:              {result.metrics.accuracy:.3f}")
    print(f"   \n   Per-label F1 Scores:")
    print(f"   - ENTAIL:              {result.metrics.F1_verified:.3f}")
    print(f"   - CONTRADICT:          {result.metrics.F1_rejected:.3f}")
    print(f"   - NEUTRAL:             {result.metrics.F1_low_conf:.3f}")
    print(f"   \n   Calibration Metrics:")
    print(f"   - ECE (Expected Cal. Error): {result.metrics.ece:.3f}")
    print(f"   - Brier Score:         {result.metrics.brier_score:.3f}")
    print(f"   \n   Efficiency:")
    print(f"   - Avg time per claim:  {result.metrics.avg_time_per_claim:.3f}s")
    print(f"   - Total time:          {result.metrics.total_time:.1f}s")
    print(f"   \n   Coverage:")
    print(f"   - Evidence coverage:   {result.metrics.evidence_coverage_rate:.1%}")
    print(f"   - Avg evidence/claim:  {result.metrics.avg_evidence_count:.2f}")
    
    # Save artifacts
    print("\n4. Saving output artifacts...")
    
    # Save predictions as JSONL
    predictions_path = output_dir / f"{result.run_id}_predictions.jsonl"
    with open(predictions_path, 'w') as f:
        for pred in result.predictions:
            f.write(json.dumps(pred) + '\n')
    print(f"   ✓ Predictions saved to: {predictions_path}")
    
    # Save metrics as JSON
    metrics_path = output_dir / f"{result.run_id}_metrics.json"
    result.to_json(metrics_path)
    print(f"   ✓ Metrics JSON saved to: {metrics_path}")
    
    # Save summary as CSV
    csv_path = output_dir / f"{result.run_id}_results.csv"
    result.to_csv(csv_path)
    print(f"   ✓ Results CSV saved to: {csv_path}")
    
    # Display label distribution
    print("\n5. Label Distribution:")
    print("-" * 70)
    for label, count in result.metrics.label_distribution.items():
        pct = count / result.metrics.total_claims * 100
        print(f"   {label:12s}: {count:3d} ({pct:5.1f}%)")
    
    # Display confusion matrix (simplified)
    print("\n6. Prediction Analysis:")
    print("-" * 70)
    correct = sum(1 for p in result.predictions if p["match"])
    print(f"   Correct predictions:   {correct}/{result.metrics.total_claims} "
          f"({correct/result.metrics.total_claims:.1%})")
    
    # Show some example predictions
    print("\n7. Sample Predictions:")
    print("-" * 70)
    for i, pred in enumerate(result.predictions[:3], 1):
        status_icon = "✓" if pred["match"] else "✗"
        print(f"   {status_icon} Example {i}: {pred['claim_id']}")
        print(f"      Predicted: {pred['pred_label']:12s} (conf: {pred['pred_confidence']:.3f})")
        print(f"      Gold:      {pred['gold_label']:12s}")
        print(f"      Evidence:  {pred['evidence_count']} items")
        print()
    
    print("=" * 70)
    print("Benchmark evaluation complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
