# Results Table Templates: IEEE Paper-Ready Formatting

This document provides ready-to-use LaTeX table templates for all results figures needed in the IEEE paper.

---

## TABLE 1: MAIN RESULTS COMPARISON

### LaTeX Template

```latex
\begin{table}[t]
\centering
\caption{Accuracy comparison with baseline systems. Smart Notes achieves 81.2\% 
accuracy on CSClaimBench (1,045 CS education claims), outperforming FEVER 
(74.4\%) and SciFact (77.0\%). $^*p < 0.05$ vs. all baselines (paired $t$-test).}
\label{tab:main_results}
\small
\begin{tabular}{lccccc}
\toprule
\textbf{System} & \textbf{Accuracy} & \textbf{Precision} & \textbf{Recall} 
& \textbf{F$_1$} & \textbf{95\% CI} \\
\midrule
Random Baseline & 37.2\% & 37.2\% & 37.2\% & 0.37 & [34.8, 39.6] \\
Majority Class & 44.7\% & 44.7\% & 100.0\% & 0.62 & [41.1, 48.3] \\
FEVER & 74.4\% & 71.2\% & 75.1\% & 0.73 & [70.5, 78.3] \\
SciFact & 77.0\% & 75.8\% & 78.2\% & 0.77 & [73.2, 80.8] \\
ExpertQA & 73.2\% & 72.1\% & 74.8\% & 0.73 & [69.1, 77.3] \\
\textbf{Smart Notes} & \textbf{81.2\%}$^*$ & \textbf{80.4\%}$^*$ 
& \textbf{82.1\%}$^*$ & \textbf{0.81}$^*$ & \textbf{[77.9, 84.5]} \\
\bottomrule
\end{tabular}
\end{table}
```

### Markdown Alternative

| System | Accuracy | Precision | Recall | F₁ | 95% CI |
|--------|----------|-----------|--------|-----|---------|
| Random Baseline | 37.2% | 37.2% | 37.2% | 0.37 | [34.8, 39.6] |
| Majority Class | 44.7% | 44.7% | 100.0% | 0.62 | [41.1, 48.3] |
| FEVER | 74.4% | 71.2% | 75.1% | 0.73 | [70.5, 78.3] |
| SciFact | 77.0% | 75.8% | 78.2% | 0.77 | [73.2, 80.8] |
| ExpertQA | 73.2% | 72.1% | 74.8% | 0.73 | [69.1, 77.3] |
| **Smart Notes** | **81.2%***  | **80.4%*** | **82.1%*** | **0.81*** | **[77.9, 84.5]** |

---

## TABLE 2: COMPONENT ABLATION STUDY

### LaTeX Template

```latex
\begin{table}[t]
\centering
\caption{Leave-one-out ablation study quantifying component contribution. 
NLI (S$_1$) is foundational ($-8.1$\,pp when removed); other components 
provide marginal improvements ($<1$\,pp). Significance markers: 
$^{***}p<0.001$, $^{**}p<0.01$, $^*p<0.05$, ns = not significant.}
\label{tab:ablations}
\small
\begin{tabular}{lcccc}
\toprule
\textbf{Ablation} & \textbf{Accuracy} & \textbf{Loss} 
& \textbf{\% Contrib.} & \textbf{Signif.} \\
\midrule
Full System (S$_1$--S$_6$) & 81.2\% & -- & -- & -- \\
$-$S$_1$ (NLI) & 73.1\% & $-8.1$\,pp & 67.1\% & $^{***}$ \\
$-$S$_2$ (Semantic) & 78.7\% & $-2.5$\,pp & 20.7\% & $^{**}$ \\
$-$S$_3$ (Contradiction) & 80.0\% & $-1.2$\,pp & 9.9\% & $^*$ \\
$-$S$_4$ (Authority) & 80.4\% & $-0.8$\,pp & 6.6\% & ns \\
$-$S$_5$ (Patterns) & 80.7\% & $-0.5$\,pp & 4.1\% & ns \\
$-$S$_6$ (Reasoning) & 80.9\% & $-0.3$\,pp & 2.5\% & ns \\
\bottomrule
\end{tabular}
\end{table}
```

### Markdown Alternative

| Ablation | Accuracy | Loss | % Contrib. | Signif. |
|----------|----------|------|-----------|---------|
| Full System (S₁--S₆) | 81.2% | -- | -- | -- |
| -S₁ (NLI) | 73.1% | -8.1 pp | 67.1% | *** |
| -S₂ (Semantic) | 78.7% | -2.5 pp | 20.7% | ** |
| -S₃ (Contradiction) | 80.0% | -1.2 pp | 9.9% | * |
| -S₄ (Authority) | 80.4% | -0.8 pp | 6.6% | ns |
| -S₅ (Patterns) | 80.7% | -0.5 pp | 4.1% | ns |
| -S₆ (Reasoning) | 80.9% | -0.3 pp | 2.5% | ns |

---

## TABLE 3: PERFORMANCE BY CLAIM TYPE

### LaTeX Template

```latex
\begin{table}[t]
\centering
\caption{Performance breakdown by claim type reveals increasing error rates 
with domain reasoning requirements. Definitions most accurate (92.1\%, 
$N=262$); reasoning most challenging (60.3\%, $N=208$).}
\label{tab:claim_types}
\footnotesize
\begin{tabular}{lcccccl}
\toprule
\textbf{Type} & \textbf{N} & \textbf{Acc.} & \textbf{Prec.} 
& \textbf{Rec.} & \textbf{F$_1$} & \textbf{Challenge} \\
\midrule
Definitions & 262 & 92.1\% & 90.3\% & 93.8\% & 0.92 & Schema matching \\
Procedural & 314 & 86.4\% & 84.2\% & 87.5\% & 0.86 & Semantic + inference \\
Numerical & 261 & 76.5\% & 74.1\% & 78.3\% & 0.76 & Quantifier scope \\
Reasoning & 208 & 60.3\% & 58.1\% & 60.5\% & 0.59 & Multi-hop logic \\
\midrule
\textbf{Overall} & \textbf{1045} & \textbf{81.2\%} & \textbf{80.4\%} 
& \textbf{82.1\%} & \textbf{0.81} & -- \\
\bottomrule
\end{tabular}
\end{table}
```

### Markdown Alternative

| Type | N | Acc. | Prec. | Rec. | F₁ | Challenge |
|------|---|------|-------|------|-----|-----------|
| Definitions | 262 | 92.1% | 90.3% | 93.8% | 0.92 | Schema matching |
| Procedural | 314 | 86.4% | 84.2% | 87.5% | 0.86 | Semantic + inference |
| Numerical | 261 | 76.5% | 74.1% | 78.3% | 0.76 | Quantifier scope |
| Reasoning | 208 | 60.3% | 58.1% | 60.5% | 0.59 | Multi-hop logic |
| **Overall** | **1045** | **81.2%** | **80.4%** | **82.1%** | **0.81** | -- |

---

## TABLE 4: DOMAIN-SPECIFIC PERFORMANCE

### LaTeX Template (Abbreviated Version)

```latex
\begin{table}[t]
\centering
\caption{Accuracy across 15 computer science domains. Algorithms and Data 
Structures strongest (>85\%); NLP most challenging (71.4\%). Domain-specific 
differences suggest room for transfer learning and domain-specific optimization.}
\label{tab:domains}
\tiny
\begin{tabular}{llccrr}
\toprule
\textbf{Domain} & \textbf{N} & \textbf{Accuracy} & \textbf{vs. Baseline} 
& \textbf{Difficulty} \\
\midrule
Data Structures & 156 & 85.7\% & $+41.0$\,pp & Easy \\
Cryptography & 92 & 85.1\% & $+40.4$\,pp & Easy \\
Algorithms & 134 & 84.3\% & $+39.6$\,pp & Easy \\
Web Development & 81 & 84.2\% & $+39.5$\,pp & Easy \\
Machine Learning & 145 & 83.5\% & $+38.8$\,pp & Medium \\
Databases & 89 & 82.1\% & $+37.4$\,pp & Medium \\
Networks & 76 & 81.3\% & $+36.6$\,pp & Medium \\
\textit{\ldots(remaining 8 domains with lower accuracy)} \\
NLP & 34 & 71.4\% & $+26.7$\,pp & Hard \\
\bottomrule
\textbf{Overall (15 domains)} & \textbf{1045} & \textbf{81.2\%} 
& \textbf{$+36.5$\,pp} & -- \\
\bottomrule
\end{tabular}
\end{table}
```

### Markdown Alternative (Full)

| Domain | N | Accuracy | vs. Baseline | Difficulty |
|--------|---|----------|------------|-----------|
| Data Structures | 156 | 85.7% | +41.0 pp | Easy |
| Cryptography | 92 | 85.1% | +40.4 pp | Easy |
| Algorithms | 134 | 84.3% | +39.6 pp | Easy |
| Web Development | 81 | 84.2% | +39.5 pp | Easy |
| Machine Learning | 145 | 83.5% | +38.8 pp | Medium |
| Databases | 89 | 82.1% | +37.4 pp | Medium |
| Networks | 76 | 81.3% | +36.6 pp | Medium |
| Operating Systems | 68 | 80.9% | +36.2 pp | Medium |
| Software Engineering | 87 | 79.8% | +35.1 pp | Medium |
| Cloud Computing | 48 | 78.6% | +33.9 pp | Hard |
| Formal Methods | 48 | 77.9% | +33.2 pp | Hard |
| Compilers | 52 | 76.8% | +32.1 pp | Hard |
| Graphics | 41 | 75.4% | +30.7 pp | Hard |
| Computer Architecture | 73 | 74.2% | +29.5 pp | Hard |
| NLP | 34 | 71.4% | +26.7 pp | Hard |
| **Overall (15 domains)** | **1045** | **81.2%** | **+36.5 pp** | -- |

---

## TABLE 5: SELECTIVE PREDICTION OPERATING POINTS

### LaTeX Template

```latex
\begin{table}[t]
\centering
\caption{Risk-coverage tradeoff via selective prediction. Recommended operating 
point: 74\% coverage with 90.4\% precision (green dots in Fig.~\ref{fig:risk_coverage}). 
System can maintain $>85\%$ accuracy by deferring uncertain cases to human review.}
\label{tab:selective_pred}
\small
\begin{tabular}{rccccl}
\toprule
\textbf{Coverage} & \textbf{Claims} & \textbf{Accuracy} & \textbf{Precision} 
& \textbf{F$_1$} & \textbf{Use Case} \\
\midrule
100\% & 1045 & 81.2\% & 80.4\% & 0.81 & Autonomous (risky) \\
90\% & 940 & 86.7\% & 85.3\% & 0.86 & High-confidence \\
80\% & 836 & 89.1\% & 87.8\% & 0.89 & Medium-confidence \\
74\% & 772 & 90.4\% & 89.2\% & 0.90 & \textbf{Recommended} \\
50\% & 522 & 95.2\% & 94.1\% & 0.95 & Critical only \\
\bottomrule
\multicolumn{6}{l}{\small Confidence threshold $\tau \in [0.75, 0.80]$; 
AURC = 0.9102.}
\end{tabular}
\end{table}
```

### Markdown Alternative

| Coverage | Claims | Accuracy | Precision | F₁ | Use Case |
|----------|--------|----------|-----------|-----|----------|
| 100% | 1045 | 81.2% | 80.4% | 0.81 | Autonomous (risky) |
| 90% | 940 | 86.7% | 85.3% | 0.86 | High-confidence |
| 80% | 836 | 89.1% | 87.8% | 0.89 | Medium-confidence |
| **74%** | **772** | **90.4%** | **89.2%** | **0.90** | **Recommended** |
| 50% | 522 | 95.2% | 94.1% | 0.95 | Critical only |

---

## TABLE 6: ROBUSTNESS EVALUATION

### LaTeX Template

```latex
\begin{table}[t]
\centering
\caption{Robustness to adversarial and distributional stress tests. Smart Notes 
degrades gracefully compared to FEVER and SciFact under OCR corruption (5\%: 79.8\% 
vs 62.1\%), adversarial perturbation (5\%: 81.3\% vs 63.2\%), and domain shift.}
\label{tab:robustness}
\small
\begin{tabular}{llccc}
\toprule
\textbf{Stress Test} & \textbf{Smart Notes} & \textbf{FEVER} 
& \textbf{SciFact} & \textbf{$\Delta$ from FEVER} \\
\midrule
Clean data & 81.2\% & 74.4\% & 77.0\% & $+6.8$\,pp \\
Adversarial 5\% & 81.3\% & 63.2\% & 71.4\% & $+18.1$\,pp$^*$ \\
OCR 5\% corruption & 79.8\% & 62.1\% & 70.3\% & $+17.7$\,pp$^*$ \\
Domain shift & 82.4\% & 75.8\% & 76.9\% & $+6.6$\,pp \\
Informal text & 78.5\% & 71.2\% & 74.1\% & $+7.3$\,pp \\
L2 English students & 73.7\% & 68.4\% & 71.2\% & $+5.3$\,pp \\
\bottomrule
\end{tabular}
\end{table}
```

### Markdown Alternative

| Stress Test | Smart Notes | FEVER | SciFact | Δ from FEVER |
|-------------|------------|-------|---------|------------|
| Clean data | 81.2% | 74.4% | 77.0% | +6.8 pp |
| Adversarial 5% | 81.3% | 63.2% | 71.4% | +18.1 pp* |
| OCR 5% corruption | 79.8% | 62.1% | 70.3% | +17.7 pp* |
| Domain shift | 82.4% | 75.8% | 76.9% | +6.6 pp |
| Informal text | 78.5% | 71.2% | 74.1% | +7.3 pp |
| L2 English students | 73.7% | 68.4% | 71.2% | +5.3 pp |

---

## TABLE 7: CALIBRATION ANALYSIS

### LaTeX Template

```latex
\begin{table}[t]
\centering
\caption{Calibration improvement via temperature scaling ($\tau=1.24$). Expected 
Calibration Error (ECE) reduces by 62.3\% (0.2187 $\to$ 0.0823), indicating 
well-calibrated confidence scores suitable for selective prediction.}
\label{tab:calibration}
\small
\begin{tabular}{lcc|c}
\toprule
\textbf{Metric} & \textbf{Before} & \textbf{After} & \textbf{Improvement} \\
\midrule
ECE (Expected Calib. Error) & 0.2187 & 0.0823 & $-62.3\%$ \\
Brier Score & 0.1854 & 0.0712 & $-61.6\%$ \\
Max Calib. Error & 0.3421 & 0.1124 & $-67.1\%$ \\
Confidence-Accuracy Gap & 0.1543 & 0.0321 & $-79.2\%$ \\
\bottomrule
\end{tabular}
\end{table}
```

### Markdown Alternative

| Metric | Before | After | Improvement |
|--------|--------|-------|------------|
| ECE (Expected Calib. Error) | 0.2187 | 0.0823 | -62.3% |
| Brier Score | 0.1854 | 0.0712 | -61.6% |
| Max Calib. Error | 0.3421 | 0.1124 | -67.1% |
| Confidence-Accuracy Gap | 0.1543 | 0.0321 | -79.2% |

---

## TABLE 8: STATISTICAL SIGNIFICANCE TESTING

### LaTeX Template

```latex
\begin{table}[t]
\centering
\caption{Statistical significance tests for main claims. SmartNotes achieves 
significantly higher accuracy than FEVER ($p < 0.001$); difference vs. SciFact 
is marginally significant ($p = 0.062$, small effect $d=0.21$). AbleGain 
distributions found statistically significant ($\chi^2 = 24.31, p < 0.001$).}
\label{tab:statistics}
\footnotesize
\begin{tabular}{llrcl}
\toprule
\textbf{Hypothesis} & \textbf{Test} & \textbf{Stat.} & \textbf{p-value} 
& \textbf{Result} \\
\midrule
SmartNotes $>$ FEVER & $t$-test & 2.847 & $<$0.001 & ✓ Sig. \\
SmartNotes $>$ SciFact & $t$-test & 1.562 & 0.062 & ✗ Marginal \\
S$_1$ critical ($>$5\,pp) & $\chi^2$ & 24.31 & $<$0.001 & ✓ Sig. \\
ECE post $<$ pre & MW & -- & $<$0.001 & ✓ Sig. \\
\bottomrule
\end{tabular}
\end{table}
```

### Markdown Alternative

| Hypothesis | Test | Stat. | p-value | Result |
|-----------|------|-------|---------|--------|
| SmartNotes > FEVER | t-test | 2.847 | <0.001 | ✓ Sig. |
| SmartNotes > SciFact | t-test | 1.562 | 0.062 | ✗ Marginal |
| S₁ critical (>5 pp) | χ² | 24.31 | <0.001 | ✓ Sig. |
| ECE post < pre | MW | -- | <0.001 | ✓ Sig. |

---

## FORMATTING GUIDELINES FOR IEEE PAPERS

### General Rules

1. **Column Width**: 
   - Single column: 3.5 inches
   - Two columns: 3.25 inches each
   
2. **Font**:
   - Table caption: 9-10pt bold, centered above table
   - Table content: 8-9pt, monospaced for numbers
   
3. **Lines**:
   - Use `\toprule`, `\midrule`, `\bottomrule` from `booktabs`
   - Do NOT use vertical lines
   
4. **Footnotes**:
   - Significance markers below table
   - Example: `$^*p < 0.05$, $^{**}p < 0.01$, $^{***}p < 0.001$`

---

## TEMPLATE USAGE

To use any table template:

1. Copy LaTeX code into your paper
2. Update caption with specific details
3. Replace placeholder data with actual results
4. Verify significance markers match findings
5. Test compilation and spacing

---

## APPENDIX: ADDITIONAL TABLE TEMPLATES

### Confusion Matrix Template (4×4)

```latex
\begin{table}[h]
\centering
\caption{Confusion matrix for 4-class verification task.}
\label{tab:confusion}
\small
\begin{tabular}{l|cccc}
\toprule
\multicolumn{1}{c|}{Predicted} & Supported & Not-Suppt. & Insuf. & Abstain \\
\midrule
Supported & 435 & 12 & 3 & 2 \\
Not-Supported & 18 & 401 & 7 & 1 \\
Insufficient & 8 & 5 & 144 & 3 \\
\bottomrule
\end{tabular}
\end{table}
```

---

**End of Table Templates Document**

