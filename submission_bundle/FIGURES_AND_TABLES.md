# Figures and Tables List

**Manuscript**: CalibraTeach: Calibrated Selective Prediction for Real-Time Educational Fact Verification

---

## Main Paper Figures (6 total)

### Figure 1: CalibraTeach System Architecture
- **Location**: Section 3.1
- **Description**: 7-stage verification pipeline with component visualization
- **File**: figure_1_architecture.png (or similar)
- **Format**: Diagram/architecture
- **Caption**: "7-stage CalibraTeach verification pipeline. Components: (1) Retrieval, (2) Ranking, (3) NLI scoring, (4) Ensemble aggregation, (5) Temperature scaling, (6) Selective decision, (7) Output generation."

### Figure 2: Ensemble Component Weighting
- **Location**: Section 3.2
- **Description**: Bar chart showing 6 component weights [0.18, 0.35, 0.10, 0.15, 0.10, 0.12]
- **File**: figure_2_ensemble_weights.png
- **Format**: Bar chart
- **Caption**: "Learned ensemble component weights optimized for calibration on validation set. Components: S1 (semantic), S2 (entailment), S3 (diversity), S4 (agreement), S5 (margin), S6 (authority)."

### Figure 3: Latency Breakdown
- **Location**: Section 5.2
- **Description**: Stacked bar chart showing stage-wise latency components
- **File**: figure_3_latency_breakdown.png
- **Format**: Stacked bar chart with error bars
- **Caption**: "Stage-wise mean latency (67.68ms total). Retrieval dominates (38.6±5.2ms), followed by LLM inference (22.5±4.1ms). Optimization targets retrieval and caching."

### Figure 4: Reliability Diagram
- **Location**: Section 5.1.2
- **Description**: ECE calibration curve comparing CalibraTeach vs FEVER baseline
- **File**: figure_4_reliability_diagram.png
- **Format**: XY plot with confidence bins
- **Caption**: "10-bin reliability diagram. CalibraTeach (ECE=0.1247) tracks diagonal more closely than FEVER baseline (ECE=0.1847), indicating superior calibration particularly in 0.5-0.8 confidence range."

### Figure 5: Risk-Coverage Curve
- **Location**: Section 5.3
- **Description**: AUC-AC curve showing accuracy vs coverage at different thresholds
- **File**: figure_5_risk_coverage.png
- **Format**: XY curve plot
- **Caption**: "Risk-coverage curve (AUC-AC=0.8803). At 74% coverage (1-0.26 deferral rate), system maintains 90.2%+ precision. Operating point demonstrates practical hybrid workflow viability."

### Figure 6: Transfer Learning Degradation
- **Location**: Section 5.5
- **Description**: Accuracy and ECE comparison across datasets (CSClaimBench vs FEVER)
- **File**: figure_6_transfer_analysis.png
- **Format**: Grouped bar chart
- **Caption**: "Transfer learning analysis. CSClaimBench (270 test): 80.8% acc, 0.1247 ECE. FEVER transfer (200 claims): 74.3% acc, 0.150 ECE, showing graceful degradation with domain shift."

---

## Main Paper Tables (8 total)

### Table 1: Related Work Comparison
- **Location**: Section 2
- **Content**: Fact verification systems comparison matrix
- **Rows**: FEVER, SciFact, ExpertQA, CalibraTeach
- **Columns**: Accuracy, Calibration (ECE), Selective Prediction (AUC-AC), Cross-Domain Robustness, Novelty
- **Reference**: [See IEEE paper Section 2]

### Table 2: Dataset Composition (CSClaimBench)
- **Location**: Section 4.1
- **Content**: Dataset statistics
- Rows**: Networks, Databases, Algorithms, OS, Dist Sys
- **Columns**: Domain, Train Claims, Val Claims, Test Claims, Total Claims, Inter-rater κ
- **Reference**: [Section 4.1]

### Table 3: Baseline Hyperparameter Tuning
- **Location**: Section 4.2
- **Content**: Fair comparison protocol for all baselines
- **Rows**: FEVER, SciFact, BERT-base
- **Columns**: Learning rate, Batch size, Epochs, Validation ECE, Test ECE, Temperature t
- **Note**: All systems undergo identical temperature-scaling treatment for calibration parity

### Table 4: Baseline Comparison (Main Results)
- **Location**: Section 5.1.1
- **Content**: System performance comparison
- **Rows**: FEVER, SciFact, BERT-base, CalibraTeach
- **Columns**: Accuracy, Macro-F1, ECE, AUC-AC, Latency (ms), Notes
- **Key**: CalibraTeach shows best ECE and AUC-AC despite lower accuracy than some

### Table 5: Confidence Intervals
- **Location**: Section 5.3
- **Content**: Bootstrap CIs on primary metrics
- **Rows**: Accuracy, Macro-F1, ECE, AUC-AC, Precision@74% Coverage
- **Columns**: Point Estimate, 95% CI Lower, 95% CI Upper, CI Width, Bootstrap Samples
- **Note**: 2000 bootstrap resamples for all metrics

### Table 6: Multi-Seed Stability
- **Location**: Section 5.4
- **Content**: Reproducibility across 5 deterministic seeds
- **Rows**: Accuracy, Macro-F1, ECE, AUC-AC
- **Columns**: Seed 0, Seed 1, Seed 2, Seed 3, Seed 4, Mean, Std Dev
- **Interpretation**: Small ±0.007 variance indicates reproducible results

### Table 7: Per-Domain Accuracy Analysis
- **Location**: Section 5.3.1
- **Content**: Per-domain performance validation
- **Rows**: Networks, Databases, Algorithms, OS, Dist Sys, Overall
- **Columns**: Accuracy (%), 95% CI Lower, 95% CI Upper, N Claims, Stability Assessment
- **Key Finding**: Variance 0.9pp across domains indicates equitable performance

### Table 8: Selective Prediction Operating Points
- **Location**: Section 5.3
- **Content**: Coverage-precision tradeoffs
- **Rows**: 50% coverage, 60%, 70%, 74% (selected), 80%, 90%, 100%
- **Columns**: Coverage, Deferral Rate (%), Accuracy on Accepted, Precision, Num Claims
- **Note**: 74% coverage selected for practical balance

---

## Appendix Figures (12 total)

### Appendix E.1: ECE Bin-by-Bin Breakdown
- Description: Detailed 10-bin table with predicted confidence ranges
- Location: Appendix E.1

### Appendix E.2: Error Analysis Detailed Patterns
- Description: Categorized error cases with examples
- Location: Appendix E.2

### Appendix E.3: Component Ablation Curves
- Description: Ablation study showing impact of removing each ensemble component
- Location: Appendix E.3

### Appendix E.4: Domain Adaptation Protocol
- Description: Re-calibration procedure for new domains with visualization
- Location: Appendix E.4

### Appendix E.5: Infrastructure Scaling Tests
- Description: Latency/accuracy curves for 20,000 claim evaluation
- Location: Appendix E.5

### Appendix E.6: LLM Baseline Confidence Distribution
- Description: Histogram of LLM-generated confidence scores vs calibrated
- Location: Appendix E.6

### Appendix E.7: Ethical Fairness Audit
- Description: Per-domain and cross-domain performance disparities
- Location: Appendix E.7

### Appendix E.8: Pilot Study Results
- Description: Student-instructor agreement visualizations
- Location: Appendix E.8

### (6 more appendix figures)

---

## Appendix Tables (15 total)

### Table E.1.1: ECE Bin-by-Bin Analysis
- Content: Confidence bin ranges, predicted confidence, accuracy, ECE contribution
- Location: Appendix E.1

### Table E.2.1: Error Categories Breakdown
- Content: Error type taxonomy with counts and percentages
- Location: Appendix E.2

### Table E.3.1: Component Ablation Study
- Content: Accuracy/ECE impact of removing each component
- Location: Appendix E.3

### Table E.3.2: Ablation Improvements Table
- Content: % improvement from adding each component sequentially
- Location: Appendix E.3

### Table E.4.1: Domain Adaptation Protocol Steps
- Content: Step-by-step re-calibration procedure with examples
- Location: Appendix E.4

### Table E.5.1: Infrastructure Scaling Results
- Content: Performance metrics at 1K, 5K, 10K, 20K claim scales
- Location: Appendix E.5

### Table E.6.1: LLM Baseline Metrics
- Content: Accuracy, ECE, AUC-AC for RAG-style LLM baseline
- Location: Appendix E.6

### Table E.7.1: Fairness Audit Per-Domain
- Content: Accuracy, error rate, precision, recall by domain
- Location: Appendix E.7

### Table E.7.2: Fairness Audit Cross-Domain
- Content: False positive/negative rates by domain comparison
- Location: Appendix E.7

### Table E.8.1: Pilot Study Participant Data
- Content: Student demographics, experience levels, demographics
- Location: Appendix E.8

### Table E.8.2: Pilot Study Results
- Content: Trust correlation, instructor agreement, error detection
- Location: Appendix E.8

### (4 more appendix tables)

---

## Artifact Files (from artifacts/latest/)

### Data Files Referenced
- `ci_report.json`: Confidence intervals for all metrics
- `multiseed_report.json`: Multi-seed stability data
- `baseline_comparison_table.csv/.md`: Main baseline results
- `latency_breakdown.csv`: Stage-wise latency data
- `ece_bins_table.csv`: Detailed ECE binning
- `error_breakdown.csv`: Error analysis categorization

---

## Digital Assets Provided

All figures available in both:
1. **High-Resolution Format**: 300 DPI PNG/PDF (for printing)
2. **Vector Format**: PDF/EPS (for editing if needed)
3. **Source Files**: YAML specifications for reproducible figure generation

---

## Supplementary Materials

All data, code artifacts, and extended tables available in:
- **GitHub Repository**: https://github.com/somanellipudi/smart-notes
- **Docker Container**: [Container registry link]
- **CSClaimBench Dataset**: [Dataset landing page link]

---

Last Updated: March 2, 2026
