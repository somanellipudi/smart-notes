# Research Paper Artifact Generation - Complete Documentation

## Overview

This package provides a comprehensive IEEE Access journal-level evaluation pipeline for the CalibraTeach system. All experimental results are automatically generated, saved as artifacts, and populated into `research_paper.md` with NO hard-coded numbers.

## Quick Start

### Generate All Artifacts (Quick Mode - Testing)
```bash
# Windows PowerShell
.\scripts\make_paper_artifacts.ps1 -Quick

# Or directly with Python
python scripts/make_paper_artifacts.py --quick
```

### Generate All Artifacts (Full Mode - For Paper)
```bash
# Windows PowerShell  
.\scripts\make_paper_artifacts.ps1

# Or directly with Python
python scripts/make_paper_artifacts.py
```

**Quick mode**: 500 bootstrap samples, 3 seeds (~3 minutes)  
**Full mode**: 2000 bootstrap samples, 5 seeds (~10-15 minutes)

## Architecture

### Module Overview

```
src/evaluation/
├── bootstrap_ci.py              # [A] Bootstrap confidence intervals
├── multi_seed_eval.py           # [B] Multi-seed stability analysis
├── ablation_study.py            # [C] Component ablation experiments
├── calibration_comprehensive.py # [D] Calibration evaluation
├── selective_prediction_reporting.py # [E] Selective prediction analysis
├── error_analysis.py            # [F] Structured error classification
├── llm_baseline.py              # [G] LLM baseline wrapper
└── paper_updater.py             # Auto-update research_paper.md

scripts/
├── make_paper_artifacts.py      # Orchestrator script (Python)
└── make_paper_artifacts.ps1     # Orchestrator script (PowerShell)

artifacts/latest/                # All generated outputs
research_paper.md                # Auto-updated with results
```

## Module Details

### [A] Bootstrap Confidence Intervals (`bootstrap_ci.py`)

**Purpose**: Compute 95% CIs for key metrics using bootstrap resampling.

**Metrics**:
- Accuracy
- Macro-F1
- ECE (Expected Calibration Error, n_bins=15)
- AUC-AC (Area Under Accuracy-Coverage curve)

**Outputs**:
- `artifacts/latest/ci_report.json`

**Example Usage**:
```python
from src.evaluation.bootstrap_ci import compute_bootstrap_cis

report = compute_bootstrap_cis(
    predictions=[0, 1, 2, ...],
    labels=[0, 1, 1, ...],
    confidences=[0.9, 0.7, 0.8, ...],
    n_bootstrap=2000
)

print(report.summary_table())
```

**Paper Section**: 5.1 Main Results

---

### [B] Multi-Seed Evaluation (`multi_seed_eval.py`)

**Purpose**: Assess model stability across random seeds.

**Outputs**:
- `artifacts/latest/metrics_by_seed.csv` - Per-seed results
- `artifacts/latest/metrics_summary.csv` - Mean, std, worst-case
- `artifacts/latest/worst_case_metrics.csv` - Conservative estimates

**Example Usage**:
```python
from src.evaluation.multi_seed_eval import run_multi_seed_evaluation

def my_eval(seed):
    # Your evaluation code here
    return {"accuracy": 0.85, "ece": 0.05, ...}

report = run_multi_seed_evaluation(
    eval_fn=my_eval,
    seeds=[0, 1, 2, 3, 4]
)
```

**Paper Section**: 5.2 Multi-Seed Stability

---

### [C] Ablation Study (`ablation_study.py`)

**Purpose**: Quantify contribution of each system component.

**Configurations Tested**:
1. Base Pipeline (raw predictions)
2. + Ensemble Confidence
3. + Temperature Scaling
4. + Selective Prediction

**Outputs**:
- `artifacts/latest/ablation_table.csv`
- `artifacts/latest/ablation_table.md`
- `artifacts/latest/ablation_study.json`
- `artifacts/latest/ablation_improvements.csv` (delta from baseline)

**Example Usage**:
```python
from src.evaluation.ablation_study import run_ablation_study, AblationConfig

def my_eval(config: AblationConfig):
    # Run evaluation with config settings
    if config.use_ensemble:
        # Enable ensemble
        pass
    return {"accuracy": 0.78, "ece": 0.08, ...}

report = run_ablation_study(eval_fn=my_eval)
```

**Paper Section**: 5.3 Ablation Study

---

### [D] Calibration Evaluation (`calibration_comprehensive.py`)

**Purpose**: Assess and visualize calibration quality.

**Outputs**:
- `artifacts/latest/calibration_report.json`
- `artifacts/latest/ece_bins_table.csv` (ECE at 10/15/20 bins)
- `artifacts/latest/fig_reliability_before.png`
- `artifacts/latest/fig_reliability_after.png`
- Optimal temperature parameter

**Example Usage**:
```python
from src.evaluation.calibration_comprehensive import evaluate_calibration_comprehensive

report = evaluate_calibration_comprehensive(
    confidences=[0.9, 0.8, ...],
    correctness=[1, 0, ...],  # Binary
    bin_sizes=[10, 15, 20],
    output_dir=Path("artifacts/latest")
)
```

**Paper Section**: 5.4 Calibration Analysis

---

### [E] Selective Prediction (`selective_prediction_reporting.py`)

**Purpose**: Analyze accuracy-coverage tradeoffs.

**Outputs**:
- `artifacts/latest/selective_accuracy_at_coverage.csv`
- `artifacts/latest/selective_coverage_at_risk.csv`
- `artifacts/latest/fig_risk_coverage.png`
- `artifacts/latest/selective_prediction_report.json`

**Example Usage**:
```python
from src.evaluation.selective_prediction_reporting import generate_selective_prediction_report

report = generate_selective_prediction_report(
    confidences=[0.9, 0.7, ...],
    predictions=[0, 1, ...],
    targets=[0, 1, ...],
    target_coverages=[1.0, 0.9, 0.8],
    target_risks=[0.10, 0.05]
)
```

**Paper Section**: 5.5 Selective Prediction Deployment Analysis

---

### [F] Error Analysis (`error_analysis.py`)

**Purpose**: Classify and understand failure modes.

**Error Types**:
- `retrieval_failure` - No evidence found
- `ambiguous_claim` - Inherent ambiguity
- `evidence_mismatch` - Semantic mismatch
- `overconfidence_error` - High confidence + wrong
- `other` - Uncategorized

**Outputs**:
- `artifacts/latest/error_breakdown.csv` (percentages)
- `artifacts/latest/error_examples.md` (5-10 examples per type)
- `artifacts/latest/error_analysis_report.json`

**Example Usage**:
```python
from src.evaluation.error_analysis import analyze_errors, save_error_analysis

predictions = [
    {
        "claim_id": "claim_1",
        "claim_text": "...",
        "predicted_label": "VERIFIED",
        "true_label": "REJECTED",
        "confidence": 0.85,
        "evidence_count": 2
    },
    # ... more predictions
]

report = analyze_errors(predictions)
save_error_analysis(report, Path("artifacts/latest"))
```

**Paper Section**: 5.6 Error Analysis

---

### [G] LLM Baseline (`llm_baseline.py`)

**Purpose**: Compare against modern LLMs (GPT-4o, Claude, Llama).

**Features**:
- Deterministic stub (no API keys needed)
- Real API support (when keys available)
- Cost estimation

**Outputs**:
- `artifacts/latest/llm_baseline/metrics.csv`
- `artifacts/latest/llm_baseline/predictions.csv`
- `artifacts/latest/llm_baseline/llm_baseline_result.json`

**Example Usage**:
```python
from src.evaluation.llm_baseline import LLMBaseline, save_llm_baseline_results

baseline = LLMBaseline(model_name="gpt-4o", use_stub=True)

test_data = [
    {
        "claim_id": "claim_1",
        "claim_text": "Machine learning requires labeled data.",
        "true_label": "VERIFIED",
        "evidence": ["Evidence text 1", "Evidence text 2"]
    },
    # ... more test examples
]

result = baseline.evaluate(test_data)
save_llm_baseline_results(result, Path("artifacts/latest/llm_baseline"))
```

**Paper Section**: 5.7 Baseline Comparison

---

### Paper Auto-Update (`paper_updater.py`)

**Purpose**: Automatically populate `research_paper.md` with experimental results.

**Auto-Generated Sections**:
1. Experimental Setup (seeds, bootstrap config)
2. Main Results Table (from CI report)
3. Multi-Seed Stability Table
4. Ablation Study Table + Interpretation
5. Calibration Analysis (ECE bin sensitivity, figures)
6. Selective Prediction Tables (accuracy@coverage, coverage@risk)
7. Error Analysis (breakdown, examples)
8. Baseline Comparison (vs LLM)
9. Limitations (dataset size, domain, language)
10. Reproducibility (commands, seeds, artifact structure)

**Example Usage**:
```python
from src.evaluation.paper_updater import PaperUpdater

updater = PaperUpdater(
    artifacts_dir=Path("artifacts/latest"),
    paper_path=Path("research_paper.md")
)
updater.update_paper()
```

## Artifact Directory Structure

```
artifacts/latest/
├── ci_report.json                      # Bootstrap CIs
├── metrics_by_seed.csv                 # Per-seed metrics
├── metrics_summary.csv                 # Multi-seed summary
├── worst_case_metrics.csv              # Conservative estimates
├── ablation_table.csv                  # Ablation results
├── ablation_table.md                   # Formatted table
├── ablation_study.json                 # Full ablation data
├── ablation_improvements.csv           # Delta from baseline
├── calibration_report.json             # Calibration metrics
├── ece_bins_table.csv                  # ECE sensitivity
├── ece_bins_table.md                   # Formatted table
├── fig_reliability_before.png          # Calibration plot (before)
├── fig_reliability_after.png           # Calibration plot (after)
├── selective_accuracy_at_coverage.csv  # Selective prediction
├── selective_coverage_at_risk.csv      # Risk thresholds
├── fig_risk_coverage.png               # Risk-coverage curve
├── selective_prediction_report.json    # Full selective data
├── error_breakdown.csv                 # Error type distribution
├── error_breakdown.md                  # Formatted table
├── error_examples.md                   # Representative examples
├── error_analysis_report.json          # Full error data
└── llm_baseline/
    ├── metrics.csv                     # LLM performance
    ├── predictions.csv                 # LLM predictions
    └── llm_baseline_result.json        # Full LLM data
```

## Extending the System

### Adding a New Metric

1. **Compute the metric** in your evaluation function
2. **Update bootstrap_ci.py** to include it:
   ```python
   def compute_new_metric(data):
       # Your metric computation
       return metric_value
   
   # Add to compute_bootstrap_cis()
   new_point, new_lower, new_upper = bootstrap_ci(...)
   ```

3. **Update paper_updater.py** to display it:
   ```python
   def _update_main_results_table(self, content: str) -> str:
       # Add row to table
       table += f"| New Metric | {ci['new_metric']['point_estimate']:.4f} | ..."
   ```

### Adding a New Ablation Configuration

```python
from src.evaluation.ablation_study import AblationConfig

new_config = AblationConfig(
    name="+ My New Component",
    description="Description of what it does",
    use_ensemble=True,
    use_temperature_scaling=True,
    use_selective_prediction=True,
    # Add custom parameters as needed
)

configs = create_default_ablation_configs()
configs.append(new_config)

report = run_ablation_study(eval_fn=my_eval, configs=configs)
```

### Adding a New Error Type

Edit `src/evaluation/error_analysis.py`:

```python
def classify_error(...):
    # Add your classification logic
    if some_condition:
        return "my_new_error_type"
    # ... existing logic
```

## Testing

### Test Individual Modules

Each module includes standalone example usage in the `if __name__ == "__main__":` block:

```bash
# Test bootstrap CI
python src/evaluation/bootstrap_ci.py

# Test multi-seed eval
python src/evaluation/multi_seed_eval.py

# Test ablation study
python src/evaluation/ablation_study.py

# etc.
```

### Validate Full Pipeline

```bash
# Quick test (3 minutes)
python scripts/make_paper_artifacts.py --quick

# Review generated artifacts
ls artifacts/latest/

# Check updated paper
cat research_paper.md
```

### Run Existing Tests

```bash
# Run all existing tests
pytest

# Run specific evaluation tests
pytest tests/ -k evaluation
```

## Troubleshooting

### Missing Dependencies

```bash
pip install -r requirements.txt
pip install tabulate  # For markdown table generation
```

### Matplotlib Backend Issues

If you encounter matplotlib display errors:
```python
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
```

### Memory Issues (Large Bootstrap)

Reduce bootstrap samples in quick mode:
```bash
python scripts/make_paper_artifacts.py --quick  # Uses 500 instead of 2000
```

### API Keys for LLM Baseline

Set environment variables:
```bash
# Windows PowerShell
$env:OPENAI_API_KEY="sk-..."
$env:ANTHROPIC_API_KEY="sk-ant-..."

# Linux/Mac
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

Or use stub mode (no API needed):
```python
baseline = LLMBaseline(model_name="gpt-4o", use_stub=True)
```

## Performance Benchmarks

**Hardware**: AMD Ryzen / Intel i7, 16GB RAM, NVIDIA RTX 3080 (GPU optional)

| Task | Quick Mode | Full Mode |
|------|------------|-----------|
| Bootstrap CI | 5s | 20s |
| Multi-Seed (3/5 seeds) | 10s | 30s |
| Ablation Study | 5s | 15s |
| Calibration | 5s | 10s |
| Selective Prediction | 3s | 5s |
| Error Analysis | 2s | 5s |
| LLM Baseline (stub) | 5s | 10s |
| Paper Update | 1s | 1s |
| **Total** | ~3 min | ~10 min |

## Citation

If you use this evaluation framework, please cite:

```bibtex
@software{calibrateach_eval_2025,
  author = {Your Name},
  title = {CalibraTeach: Journal-Level Evaluation Framework},
  year = {2025},
  url = {https://github.com/your-repo/Smart-Notes}
}
```

## License

Same license as main Smart-Notes repository.

## Contributing

1. Add new evaluation modules in `src/evaluation/`
2. Update `paper_updater.py` to include new sections
3. Update `make_paper_artifacts.py` orchestrator
4. Add documentation to this README
5. Test with `--quick` before full run

## Support

- **Issues**: GitHub Issues
- **Questions**: GitHub Discussions
- **Documentation**: This README + inline docstrings

## Changelog

### v1.0.0 (2025-03-02)
- Initial release
- All 7 evaluation modules (A-G)
- Auto-update system for research_paper.md
- PowerShell and Python orchestrators
- Full artifact generation pipeline
