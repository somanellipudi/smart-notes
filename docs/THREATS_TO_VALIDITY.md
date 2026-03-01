# Threats to Validity

This document discusses potential limitations, generalization concerns, and validity threats for the Smart Notes calibrated fact-verification system.

## Internal Validity

### Confounding Variables

**Threat: Component interaction effects**
- Multiple confidence components (semantic, entailment, diversity, agreement, margin, authority) may interact in complex ways that our linear weighting does not fully capture.
- **Mitigation**: Ablation study removes components individually to isolate their contributions; log-odds weighting is a standard linear model that is transparent and reproducible.

**Threat: Temperature scaling overfitting**
- Calibration on a separate split mitigates this, but grid search over finite temperature values may bias results toward specific seeds.
- **Mitigation**: Use deterministic seeds (GLOBAL_RANDOM_SEED=42) and report grid search bounds; test with alternative calibration methods (e.g., Platt scaling, histogram binning) in future work.

**Threat: Synthetic data bias**
- Evaluation uses 300 synthetically generated examples per run, which may not capture the full diversity of real claim distributions.
- **Mitigation**: Baseline results are reported; transition to FEVER or CSClaimBench when available. Synthetic generation uses deterministic seeding, allowing reproducible extension to larger synthetic corpora.

## External Validity

### Generalization Concerns

**Threat: Domain specificity**
- System is currently trained/evaluated on CS domain (Computer Science) claims for Smart Notes educational context. Results may not generalize to other domains (medicine, law, general knowledge).
- **Mitigation**: Code is modular; retrieve and NLI models are pre-trained on diverse data (SQuAD, MNLI). Component weights learned on CS claims; authors recommend re-weighting for new domains.

**Threat: Retrieval corpus limitations**
- Evidence is drawn from CS CS textbook summaries and Wikipedia snippets. Authority scoring and relevance judgments may not represent real-world corpora.
- **Mitigation**: Document corpus structure (EVIDENCE_CORPUS.md); design is compatible with FEVER or other evidence collections.

**Threat: NLI model bias**
- RoBERTa-large-MNLI has known issues (e.g., shortcut learning, dataset bias). **SUPPORTED** and **REFUTED** predictions may reflect MNLI training distribution, not true entailment.
- **Mitigation**: Use ensemble or multiple NLI models in future work; report calibration metrics (ECE, Brier) to quantify overconfidence.

### Educational Context Specificity

**Threat: Claim characteristics**
- Educational claims (facts about course topics) differ from news claims or web fact-checks in style, logical structure, and frequency of nuanced or multi-faceted topics.
- **Mitigation**: Ablation shows which components (diversity, multi-source consensus) help most; authors recommend tuning `min_entailing_sources` for domain-specific rigor.

**Threat: Teacher and student trust**
- System confidence calibration assumes users trust binary predictions. In educational settings, users may require higher confidence thresholds or explanations for sensitive topics.
- **Mitigation**: Risk-coverage analysis (AUC-RC metric) shows abstention rates; confidence scores are available to applications for custom thresholding.

## Construct Validity

### Operationalization Issues

**Threat: Label definition**
- Three-way labels (SUPPORTED, REFUTED, NEI) may oversimplify nuanced topics. Some claims may be partially true or require background context.
- **Mitigation**: NEI class provides escape valve for ambiguous cases; future work can model confidence/uncertainty explicitly.

**Threat: Confidence as reliability metric**
- Predicted confidence may not accurately reflect true prediction correctness. ECE and Brier score measure calibration but do not guarantee real-world reliability.
- **Mitigation**: Temperature scaling and calibration on held-out split reduce miscalibration; compare to human inter-annotator agreement if available.

### Metric limitations

**Threat: Macro-F1 with imbalanced data**
- Macro-F1 weights all classes equally, which can favor arbitrary threshold selection if one class is rare.
- **Mitigation**: Report per-class metrics (Precision, Recall, F1) for each of {SUPPORTED, REFUTED, NEI}; test on multiple class balances via ablation.

**Threat: Accuracy alone ignores cost asymmetry**
- False SUPPORTED (incorrect fact endorsement) may have higher educational cost than false REFUTED (overly skeptical).
- **Mitigation**: Calibration metrics (ECE, Brier, AUC-RC) provide a richer view; recommend cost-weighted evaluation when domain costs are known.

## Statistical Validity

### Hypothesis Testing and Multiplicity

**Threat: Multiple comparisons**
- Ablation suite tests 6 configurations (2 × 3 grid); no correction for multiple comparisons.
- **Mitigation**: Ablation is exploratory; statistical significance requires larger samples and formal statistical tests (e.g., McNemar's test, paired t-tests) on real data.

**Threat: Small sample size**
- Synthetic evaluation set is 300 examples. Confidence intervals are wide; differences between baselines may not be significant.
- **Mitigation**: Increase to 1000+ examples if computational budget allows; report standard error or bootstrap confidence intervals.

**Threat: Non-independence of folds**
- Current evaluation uses a single 300-example split. No k-fold cross-validation or multiple runs with different seeds.
- **Mitigation**: Future work should average metrics over multiple random seeds and report variance; ablation runner supports this (GLOBAL_RANDOM_SEED can be swept).

## Reproducibility and Measurement

**Threat: Environment dependencies**
- Model versions (transformers, torch), library versions, GPU/CPU hardware differences may affect results.
- **Mitigation**: `requirements-lock.txt` pins all dependencies; deterministic seed (GLOBAL_RANDOM_SEED=42) and `torch.manual_seed()` + `torch.use_deterministic_algorithms()` ensure CPU/GPU reproducibility (with caveats for some operations).

**Threat: Heuristic choices**
- Authority scoring (exponential decay), multi-source consensus thresholds, confidence weighting are heuristics not learned from data.
- **Mitigation**: These are configurable via environment variables and centralized in `VerificationConfig`; ablation runner tests min_entailing_sources ∈ {1, 2, 3}.

**Threat: Code and data availability**
- Code must be released on GitHub; synthetic data generation is on-the-fly but seeds are fixed.
- **Mitigation**: Reproduction scripts (reproduce_all.sh, reproduce_all.ps1) and environment variable documentation enable external replication.

## Ethical Considerations

### Misuse and Bias

**Threat: Overconfidence in educational AI**
- System confidence scores may mislead educators or students if presented without caveats. Over-reliance could reduce critical thinking.
- **Mitigation**: Report calibration metrics prominently; recommend Abstention (NEI prediction) as a reasonable escape valve; include uncertainty in UI/API.

**Threat: Bias in retrieved evidence**
- Retrieved evidence may reflect biases in Wikipedia, textbook summaries, or training corpus. False consensus within biased corpus could bias predictions.
- **Mitigation**: Diversity component (retrieve multiple sources) reduces single-source bias; audit corpus for known biases before deployment.

**Threat: Model fairness across demographics**
- Current evaluation does not assess fairness with respect to protected attributes (e.g., topic area, claim complexity, language).
- **Mitigation**: Future work should evaluate per-subgroup metrics; consult with educators and subject-matter experts for domain-specific fairness concerns.

## Recommendations for Stronger Claims

1. **Transition to real data**: Evaluate on FEVER, CSClaimBench, or domain-specific real-world claim datasets with human annotations.
2. **Multiple random seeds**: Average metrics over 10+ independent runs with different random seeds; report confidence intervals.
3. **Larger evaluation sets**: Increase from 300 to 1000+ examples if feasible.
4. **Cross-domain evaluation**: Evaluate on other CS topics or other educational domains to assess generalization.
5. **User studies**: Gather educator and student feedback on trust, calibration, and educational impact.
6. **Formal statistical tests**: Use McNemar's test or paired t-tests to compare baselines and ablations statistically.
7. **Component contribution analysis**: Ablate components more granularly (remove each confidence component individually; vary retrieval top_k).
8. **Cost-aware evaluation**: If available, define per-error-type costs (false SUPPORTED vs. false REFUTED) and optimize for educational impact.
