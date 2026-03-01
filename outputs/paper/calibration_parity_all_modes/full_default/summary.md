# Calibration Parity Report: full_default

[WARNING] SYNTHETIC PLACEHOLDER DATA (seed=42, n=300)

## Metrics

- **Accuracy**: 0.7467
- **ECE**: 0.0587
- **Brier Score**: 0.1652
- **AUC-AC**: -0.9950
- **Samples**: 300

## Configuration

```json
{
  "verified_confidence_threshold": 0.7,
  "rejected_confidence_threshold": 0.3,
  "low_confidence_range": [
    0.3,
    0.7
  ],
  "min_entailing_sources_for_verified": 2,
  "top_k_retrieval": 20,
  "top_k_rerank": 5,
  "mmr_lambda": 0.5,
  "temperature_scaling_enabled": true,
  "temperature_init": 1.0,
  "temperature_grid_min": 0.8,
  "temperature_grid_max": 2.0,
  "temperature_grid_steps": 100,
  "calibration_split": "validation",
  "random_seed": 42,
  "retriever_threshold": 0.7,
  "nli_positive_threshold": 0.6,
  "nli_negative_threshold": 0.4,
  "rag_positive_threshold": 0.65,
  "rag_negative_threshold": 0.35,
  "deployment_mode": "full_default",
  "enable_result_cache": true,
  "enable_quality_screening": true,
  "enable_query_expansion": true,
  "enable_evidence_ranker": true,
  "enable_type_classifier": true,
  "enable_semantic_deduplicator": true,
  "enable_adaptive_depth": true,
  "enable_priority_scorer": true
}
```

## Plots

- Reliability Diagram: `reliability_diagram.png`
- Risk-Coverage Curve: `risk_coverage.png`

## Notes

- This evaluation uses SYNTHETIC DATA for engineering validation.
- For authoritative results, use real CSClaimBench or FEVER datasets.
- Reproducibility verified: same seed -> identical outputs.
