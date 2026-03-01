# Evaluation Protocol

This document describes the standardized evaluation protocol for the Smart Notes system, including dataset splits, baseline definitions, metrics, and ablation suite.

## Dataset and Splits

### Synthetic Evaluation Set

- **Size**: 300 examples per mode/ablation run
- **Generation**: deterministic using `np.random.RandomState(seed)` with seed from config
- **Class distribution**: 30% NEI, 35% REFUTED, 35% SUPPORTED
- **Randomness**: seeded by `GLOBAL_RANDOM_SEED` (default 42), ensuring reproducible draws

### Real Evaluation Set (Future)

When moving to a real dataset (e.g., CSClaimBench or FEVER on CS domain):
- **train**: ~60% of labeled examples, used for component weight learning and calibration parameter tuning
- **calibration/validation**: ~20%, used for temperature scaling calibration and threshold learning
- **test**: ~20%, held-out test set for final reporting

## Baseline Definitions

Four evaluation modes are supported via `--mode` flag:

### 1. `baseline_retriever`
- **Pipeline**: semantic retrieval + heuristic threshold
- **Decision rule**: predict SUPPORTED if top retrieval score > `retriever_threshold` (default 0.70), else NEI
- **Confidence**: raw retrieval similarity score
- **Thresholds**: configurable via `RETRIEVER_THRESHOLD` env var

### 2. `baseline_nli`
- **Pipeline**: NLI classifier on top-1 evidence (no retrieval)
- **Decision rule**: predict SUPPORTED if NLI entailment prob > `nli_positive_threshold` (0.6), REFUTED if < `nli_negative_threshold` (0.4), else NEI
- **Confidence**: NLI entailment probability

### 3. `baseline_rag_nli`
- **Pipeline**: retrieval + NLI ensemble (no component calibration)
- **Decision rule**: combined score = 0.5 × sim_score + 0.5 × nli_entail_prob; thresholds at 0.65 (SUPPORTED) and 0.35 (REFUTED)
- **Confidence**: combined score
- **Note**: no temperature scaling or multi-source consensus

### 4. `verifiable_full`
- **Pipeline**: full 7-stage pipeline with all 6 confidence components, learned weights, temperature scaling, and multi-source consensus
- **Decision rule**: predict SUPPORTED if `calibrated_confidence > verified_confidence_threshold` AND `count_entailing_sources >= min_entailing_sources_for_verified`; REFUTED if `calibrated_confidence < rejected_confidence_threshold`; else NEI
- **Confidence**: post-temperature-scaling probability calibrated for binary correctness

## Ablation Study Grid

The ablation runner (`src/evaluation/ablation.py`) tests combinations of:

- **Temperature scaling**: ON / OFF (env var `TEMPERATURE_SCALING_ENABLED`)
- **Min entailing sources**: {1, 2, 3} (env var `MIN_ENTAILING_SOURCES_FOR_VERIFIED`)

This produces 6 runs (2 × 3 grid). Each run computes full metrics and saves to individual folders.

**Future ablations** (not yet implemented):
- Remove individual confidence components (semantic, entailment, diversity, agreement, margin, authority)
- Toggle MMR ON/OFF
- Vary retrieval top_k

## Metrics Computed

For each run, the following metrics are computed and saved to `metrics.json`:

### Classification Metrics
- **Accuracy**: fraction correct
- **Macro-F1**: unweighted F1 across all classes
- **Per-class Precision, Recall, F1**: for each of {NEI, REFUTED, SUPPORTED}
- **Confusion matrix**: 3×3 matrix for error analysis

### Calibration Metrics
- **ECE (Expected Calibration Error)**: binned calibration error (10 bins, default)
- **Brier Score**: mean squared difference between predicted prob and correctness

### Selective Prediction
- **Risk-coverage curve**: cumulative accuracy vs. fraction of examples predicted
- **AUC-RC**: area under risk-coverage curve (1.0 = perfect selective prediction)

### Retrieval Metrics
- **Recall@k**: fraction of examples where ground-truth rank ≤ k
- **MRR (Mean Reciprocal Rank)**: 1/mean_rank of predictions

### Plots Generated
- **Reliability diagram**: confidence bins vs. observed accuracy (PNG)
- **Confusion matrix heatmap**: per-class error breakdown (PNG)
- **Risk-coverage curve**: coverage vs. accuracy (PNG)

## Running Evaluations

### Quick evaluation (synthetic, 300 examples):
```bash
python src/evaluation/runner.py --mode baseline_retriever --out outputs/quick/baseline_ret
python src/evaluation/runner.py --mode verifiable_full --out outputs/quick/verifiable_full
```

### Ablation suite:
```bash
python src/evaluation/ablation.py --output_base outputs/ablations
```

### Consolidate runs:
```bash
python scripts/update_experiment_log.py --run_dir outputs/quick/baseline_ret --label quick_retriever
```

## Environment Variable Configuration

All thresholds and parameters can be overridden via environment variables:

```bash
export VERIFIED_CONFIDENCE_THRESHOLD=0.75
export REJECTED_CONFIDENCE_THRESHOLD=0.25
export MIN_ENTAILING_SOURCES_FOR_VERIFIED=2
export TOP_K_RETRIEVAL=20
export RETRIEVER_THRESHOLD=0.65
export NLI_POSITIVE_THRESHOLD=0.6
export NLI_NEGATIVE_THRESHOLD=0.4
export TEMPERATURE_SCALING_ENABLED=true
export GLOBAL_RANDOM_SEED=42
```
