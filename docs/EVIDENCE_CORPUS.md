# Evidence Corpus and Source Provenance

## Overview

Smart Notes uses a synthetic evidence corpus for quick evaluation and reproducibility testing. This document describes the corpus structure, authority scoring, and data sources used.

## Synthetic Evidence Corpus (Evaluation)

For reproducible evaluation, the system uses a **synthetic dataset composed of 300 automatically generated claims** paired with simulated retrieval scores and NLI probabilities.

### Dataset Structure

- **n_claims**: 300 claims per evaluation run
- **classes**: 3-way classification (NEI = 0, REFUTED = 1, SUPPORTED = 2)
- **class distribution**: approximate 30% NEI, 35% REFUTED, 35% SUPPORTED
- **random seed**: controlled via `GLOBAL_RANDOM_SEED` config (default 42), ensuring reproducibility across runs
- **split**: synthetic data is not formally split (train/val/test) for the quick evaluation mode; full evaluation should use a real annotated dataset

### Source Authority Scoring

In the real pipeline, sources are scored for authority using:

1. **Citation count**: normalized count of references to a source document
2. **Recency**: publication year or last-updated timestamp (older sources penalized slightly)
3. **Domain reputation**: curated list of high-authority domains (e.g., IEEE Xplore for CS, PubMed for biomedical)
4. **Consensus**: agreement across multiple classifiers (higher agreement → higher authority)

For the synthetic evaluation, authority is simulated uniformly as a mean score across all sources.

### Data License and Usage

- **Synthetic data**: generated on-the-fly; no license restrictions.
- **Real corpus** (future work, e.g., CSClaimBench): to be annotated with expert labels and released under CC-BY-4.0 or CC-BY-SA-4.0 for educational and research use.

## Integration with Evaluation Runner

The evaluation runner (`src/evaluation/runner.py`) generates synthetic data with deterministic seeds:

```python
def _synthetic_dataset(n=200, seed=42):
    _seed_everything(seed)
    labels = np.random.choice([0, 1, 2], size=n, p=[0.3, 0.35, 0.35])
    claims = [f"Synthetic claim {i}" for i in range(n)]
    return claims, labels
```

This ensures identical dataset across all runs when the seed is fixed.

## Reproducibility Notes

- To reproduce a specific run, use `GLOBAL_RANDOM_SEED=<seed>` environment variable.
- All metrics and metadata are logged to `outputs/benchmark_results/experiment_log.json`.
- Per-run logs include git commit hash, seed, and platform info for full traceability.

