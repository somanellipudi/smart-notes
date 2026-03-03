# Supplementary Materials Guide

**Manuscript**: CalibraTeach: Calibrated Selective Prediction for Real-Time Educational Fact Verification

---

## Overview

Supplementary materials are organized in separate files for ease of navigation. All materials support reproducibility and provide detailed results referenced in the main paper.

---

## A. Code and Data Repository

**Location**: https://github.com/somanellipudi/smart-notes  
**License**: MIT (code), CC-BY-4.0 (CSClaimBench dataset)

### Repository Contents

```
smart-notes/
├── src/
│   ├── evaluation/          # Core evaluation module
│   │   ├── llm_baseline.py          # LLM baseline implementation
│   │   ├── paper_updater.py         # Automated paper generation
│   │   └── calibration.py           # Temperature scaling
│   ├── pipeline/            # Fact verification pipeline
│   ├── optimization/        # ML optimization layer (8 models)
│   └── utils/               # Helper functions
├── data/
│   ├── CSClaimBench/        # 1,045 annotated claims
│   ├── FEVER_transfer/      # Transfer test subset
│   └── synthetic/           # 20,000 synthetic claims for scaling tests
├── scripts/
│   ├── make_paper_artifacts.py      # Reproducible artifact generation
│   ├── make_paper_artifacts.sh      # Shell orchestrator
│   └── make_paper_artifacts.ps1     # PowerShell version
├── docs/
│   ├── TECHNICAL_DOCS.md    # Full technical specifications
│   ├── THREATS_TO_VALIDITY.md       # Limitations and threats analysis
│   ├── DOMAIN_ADAPTATION.md         # Re-calibration protocol
│   └── REPRODUCIBILITY.md           # Full reproducibility guide
├── artifacts/
│   └── latest/              # 31 auto-generated YAML/CSV/JSON artifacts
├── Dockerfile               # Reproducible environment
├── requirements.txt         # Python dependencies (pinned versions)
└── README.md                # Installation and usage instructions
```

### Quick Start

```bash
# Clone repository
git clone https://github.com/somanellipudi/smart-notes.git
cd smart-notes

# Install dependencies
pip install -r requirements.txt

# Full pipeline (10 min on A100)
python scripts/make_paper_artifacts.py

# Quick mode (3 min)
python scripts/make_paper_artifacts.py --quick

# Generate paper with latest artifacts
python scripts/make_paper_artifacts.py --update-paper
```

**Reproduction Time**: 20 minutes from scratch to full paper generation

---

## B. Artifact Files (31 Total)

Located in `artifacts/latest/`:

### Evaluation Metrics (6 files)
- `ci_report.json` — Confidence intervals for all metrics (2000 bootstrap)
- `multiseed_report.json` — Multi-seed stability (5 seeds: [0,1,2,3,4])
- `ablation_study.json` — Complete ablation results with removed components
- `calibration_report.json` — ECE, MCE, Brier score, and bin analysis
- `error_analysis_report.json` — Error categorization and statistics
- `latency_summary.json` — Latency breakdown and throughput

### Tabular Results (8 CSV + 8 Markdown equivalents)
- `baseline_comparison_table.{csv,md}` — Main baseline results table
- `latency_breakdown.{csv,md}` — Stage-wise latency decomposition
- `ece_bins_table.{csv,md}` — ECE binning for calibration analysis
- `error_breakdown.{csv,md}` — Error analysis by category
- `ablation_table.{csv,md}` — Component ablation results
- `ablation_improvements.csv` — Sequential component additions

### Baseline Metadata (2 files)
- `baseline_comparison_metadata.json` — Baseline configuration and notes
- `baseline_comparison_table.md` — Professional tabular summary

### Visualization Data (6 files)
- `reliability_diagram_data.json` — ECE bin coordinates for plotting
- `risk_coverage_data.json` — AUC-AC curve points
- `error_analysis.md` — Error pattern visualization metadata
- `per_domain_accuracy.csv` — Per-domain performance breakdown
- `confidence_distribution.json` — LLM baseline confidence histogram
- `latency_correlation.json` — Latency vs accuracy correlations

### Report Files (3 summary files)
- `full_pipeline_report.txt` — Complete pipeline execution log
- `evaluation_summary.md` — Results summary with interpretations
- `reproducibility_checklist.md` — Verification of determinism

---

## C. Extended Results (Appendices A-H)

### Appendix E.1: Calibration Analysis
- File: `APPENDIX_E1_CALIBRATION_ANALYSIS.md`
- Content: 10-bin ECE breakdown, calibration curves, temperature scaling sensitivity
- Reference: Paper Section 5.1.2

### Appendix E.2: Error Analysis Details
- File: `APPENDIX_E2_ERROR_ANALYSIS.md`
- Content: Error categorization taxonomy, example errors, per-domain error patterns
- Reference: Paper Section 5.6

### Appendix E.3: Ablation Study Detailed
- File: `APPENDIX_E3_ABLATION_STUDY.md`
- Content: Component-by-component analysis, sequential additions, importance ranking
- Reference: Paper Section 5.7

### Appendix E.4: Domain Adaptation Protocol
- File: `APPENDIX_E4_DOMAIN_ADAPTATION.md`
- Content: Step-by-step re-calibration procedure, transfer learning guide, case studies
- Reference: Paper Section 8.1, Limitation 2

### Appendix E.5: Infrastructure Scaling Tests
- File: `APPENDIX_E5_INFRASTRUCTURE_SCALING.md`
- Content: 20,000 claim evaluation results, memory usage, GPU utilization
- Reference: Paper Abstract

### Appendix E.6: LLM Baseline Details
- File: `APPENDIX_E6_LLM_BASELINE.md`
- Content: RAG implementation, confidence calibration, stub mode documentation
- Reference: Paper Section 5.1.3

### Appendix E.7: Ethical Framework
- File: `APPENDIX_E7_ETHICAL_FRAMEWORK.md`
- Content: Fairness audit, bias analysis by domain, deployment safety guidelines
- Reference: Paper Section 8.2

### Appendix E.8: Pilot Study Details
- File: `APPENDIX_E8_PILOT_STUDY.md`
- Content: Participant demographics, study protocol, statistical analysis, full results
- Reference: Paper Section 5.8-5.9

---

## D. Technical Documentation

### TECHNICAL_DOCS.md
- Complete 7-stage pipeline pseudocode
- Component specifications with mathematical definitions
- Hyperparameter configurations
- Implementation details and edge cases

### THREATS_TO_VALIDITY.md
- Comprehensive validity threat analysis
- Internal validity concerns and controls
- External validity limitations
- Construct and statistical validity issues
- Reproducibility threats and mitigations

### DOMAIN_ADAPTATION.md
- Mandatory re-calibration protocol for new domains
- Step-by-step procedure with example
- Evidence corpus considerations
- Performance validation checklist
- Case study: CSClaimBench → History domain

### REPRODUCIBILITY.md
- Docker setup instructions
- Cross-GPU validation procedure
- Random seed documentation
- Determinism verification
- Software version pinning details

---

## E. Research Artifacts

### Dataset: CSClaimBench
- **Location**: `data/CSClaimBench/`
- **Claims**: 1,045 total (524 train, 261 val, 260 test)
- **Format**: YAML with annotations and inter-rater agreement
- **License**: CC-BY-4.0
- **Splits**: Stratified by domain (200 per domain)

### Transfer Dataset: FEVER Subset
- **Location**: `data/FEVER_transfer/`
- **Claims**: 200 claims from FEVER dataset
- **Purpose**: Transfer learning evaluation
- **Format**: Consistent with FEVER original

### Synthetic Data: Infrastructure Test
- **Location**: `data/synthetic/`
- **Claims**: 20,000 synthetic claims for scaling tests
- **Purpose**: Infrastructure validation and latency analysis

---

## F. Reproducibility Verification

### Deterministic Verification Script
```bash
python scripts/verify_determinism.py --seeds 0 1 2 3 4
```

**Expected Output**: All 5 runs produce identical predictions and confidence scores

### Artifact Hashing
```bash
python scripts/verify_artifacts.py --artifacts artifacts/latest/
```

**File**: `artifacts/latest/SHA256_HASHES.txt`

---

## G. Literature and References

### Key References Provided
- `docs/REFERENCES.bibtex` — BibTeX file with all 65 references
- `docs/LITERATURE_SUMMARY.md` — Related work detailed summaries
- `docs/BENCHMARKS_SUMMARY.md` — Comparison with relevant benchmarks

---

## H. Deployment Materials

### Docker Container
```bash
docker run -it ghcr.io/somanellipudi/calibrateach:latest
python scripts/make_paper_artifacts.py
```

### Kubernetes/Cloud Deployment
- `deployment/kubernetes/` — Kubernetes manifests
- `deployment/aws/` — AWS CloudFormation templates
- `deployment/azure/` — Azure Resource Manager templates

### Installation Guides
- `docs/INSTALLATION.md` — Local CPU/GPU setup
- `docs/CLOUD_DEPLOYMENT.md` — AWS SageMaker, Azure ML, GCP

---

## I. Extended Results Tables

All tables also available in:
1. **CSV format** — For data analysis and re-analysis
2. **Markdown format** — For easy reading and embedding
3. **YAML format** — For reproducible configuration
4. **JSON format** — For programmatic access

---

## J. How to Access Supplementary Materials

### During Review
- All materials available in submission system's supplementary files section
- GitHub repository: https://github.com/somanellipudi/smart-notes
- Docker container: https://ghcr.io/somanellipudi/calibrateach

### After Publication
- All materials permanently archived on GitHub (MIT + CC-BY-4.0)
- Zenodo archive: [DOI - assigned upon publication]
- Paper supplementary materials page on IEEE Xplore

---

## K. Reproducibility Timeline

| Task | Time | System |
|------|------|--------|
| Clone repo | 1 min | Any |
| Install dependencies | 3 min | Any |
| Run full evaluation | 10 min | A100 GPU |
| Generate paper | 2 min | Any |
| Verify determinism | 5 min | Any |
| **Total** | **~20 min** | **A100 GPU** |

---

## L. Support and Questions

**GitHub Issues**: Report bugs or ask questions  
https://github.com/somanellipudi/smart-notes/issues

**Email**: she4@kennesaw.edu  
(Selena He, corresponding author)

---

Last Updated: March 2, 2026
