# CS Benchmark Datasets

This directory contains multiple benchmark datasets for research evaluation of claim verification and evidence retrieval.

## Available Datasets

### 1. `cs_benchmark_dataset.jsonl` (Core Dataset)
- **Purpose**: General-purpose CS domain benchmark
- **Size**: 21 examples
- **Domains**: 14 computer science topics
- **Labels**: VERIFIED, REJECTED, LOW_CONFIDENCE
- **Use Case**: Primary ablation study dataset

### 2. `cs_benchmark_hard.jsonl` (Challenging Cases)
- **Purpose**: Difficult verification cases
- **Size**: 20 examples
- **Challenge Types**: 
  - Subtle logical errors
  - Partial contradictions
  - Complex evidence context
  - Multi-hop reasoning required
- **Use Case**: Robustness evaluation

### 3. `cs_benchmark_easy.jsonl` (Simple Cases)
- **Purpose**: Baseline accuracy validation
- **Size**: 15 examples
- **Characteristics**:
  - Clear evidence-claim pairs
  - Unambiguous labels
  - Single-source evidence
- **Use Case**: Sanity checks, regression testing

### 4. `cs_benchmark_multilingual.jsonl` (Language Diversity)
- **Purpose**: Multilingual evaluation
- **Languages**: English, Spanish, French, German, Chinese
- **Size**: 25 examples (5 per language)
- **Use Case**: Cross-lingual robustness

### 5. `cs_benchmark_adversarial.jsonl` (Adversarial)
- **Purpose**: Test adversarial robustness
- **Size**: 20 examples
- **Perturbations**:
  - Paraphrased claims
  - Out-of-order evidence
  - Injected noise
- **Use Case**: Robustness + adversarial evaluation

### 6. `cs_benchmark_long_context.jsonl` (Long Documents)
- **Purpose**: Long document understanding
- **Size**: 15 examples
- **Characteristics**:
  - Long source texts (5000+ chars)
  - Evidence spans deep in context
  - Multi-paragraph reasoning
- **Use Case**: Scale testing, efficiency evaluation

### 7. `cs_benchmark_domain_specific.jsonl` (Domain Splits)
- **Purpose**: Per-domain performance analysis
- **Size**: 24 examples (3 per domain, 8 domains)
- **Domains**: algorithms, datastructures, complexity, networking, security, databases, ml, compilers
- **Use Case**: Granular error analysis

### 8. `cs_benchmark_recent_2026.jsonl` (Current Knowledge)
- **Purpose**: Recent advances in CS
- **Size**: 18 examples
- **Topics**: 
  - LLMs and transformers
  - Diffusion models
  - Quantum computing progress
  - Recent papers (2024-2026)
- **Use Case**: Knowledge currency validation

## Dataset Schema

All datasets follow this JSONL schema:

```json
{
  "doc_id": "string",
  "domain_topic": "string (category.subcategory)",
  "source_text": "string",
  "generated_claim": "string",
  "gold_label": "VERIFIED|REJECTED|LOW_CONFIDENCE",
  "evidence_span": "string or empty",
  "difficulty": "easy|medium|hard",
  "source_type": "textbook|paper|wikipedia|doc|tutorial",
  "claim_type": "factual|definition|performance|relationship",
  "reasoning_type": "direct|implicit|multi_hop|arithmetic",
  "metadata": {
    "creation_date": "YYYY-MM-DD",
    "verifier": "string",
    "notes": "string"
  }
}
```

## Usage

### Quick Start
```python
from src.evaluation.cs_benchmark_runner import CSBenchmarkRunner
import json

runner = CSBenchmarkRunner()

# Load any dataset
dataset_path = "evaluation/cs_benchmark/cs_benchmark_hard.jsonl"
examples = [json.loads(line) for line in open(dataset_path)]

# Run benchmark
result = runner.run(config={}, noise_types=[], sample_size=len(examples))
```

### Run All Suites
```bash
# Run on all datasets
for dataset in evaluation/cs_benchmark/*.jsonl; do
    python scripts/run_cs_benchmark.py \
        --dataset "$dataset" \
        --output-dir "evaluation/results/$(basename $dataset .jsonl)"
done
```

## Metrics Captured Per Dataset

- **Accuracy**: Overall correctness
- **Per-Label F1**: VERIFIED, REJECTED, LOW_CONFIDENCE
- **Calibration**: ECE, MCE, Brier score
- **Robustness**: Noise injection effects
- **Efficiency**: Time per claim, throughput
- **Domain Analysis**: Per-category performance

## Contributing

To add a new dataset:

1. Create `cs_benchmark_<name>.jsonl` in this directory
2. Validate with: `python tests/test_benchmark_format_validation.py`
3. Update this README with dataset description
4. Document in paper/ablation results

## Citation

If using these datasets, please cite:

```bibtex
@dataset{smarten_notes_cs_benchmark_2026,
  title={CS Benchmark: Multimodal Claim Verification Evaluation},
  author={Smart Notes Research},
  year={2026},
  url={https://github.com/yourrepo/smart-notes}
}
```
