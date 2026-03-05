# CalibraTeach: Calibrated Fact Verification with Selective Prediction for Educational AI

**Authors**: [Author names, affiliations]

## Abstract

Automated fact verification systems often output overconfident predictions, lack education-centric uncertainty measures, and incur high latency that prevents real-time use. We present CalibraTeach, a calibrated multi-signal verification pipeline coupled with selective prediction. Running on an NVIDIA RTX 4090, the full pipeline achieves mean latency of 67.68 ms per claim with 14.78 claims/sec throughput, enabling real-time feedback in classroom settings. On a 260-claim expert-annotated CSClaimBench test set, CalibraTeach achieves 80.77% accuracy and 0.1076 expected calibration error (calibration parity ensured by temperature-scaling all baselines on the same validation set), with 0.8711 area-under-accuracy-coverage. Selective prediction achieves 74% automated coverage at 90% selective accuracy (correctness on retained predictions), deferring uncertain cases to instructor review. Transfer evaluation on 200 FEVER claims (74.3% accuracy, 0.150 ECE) confirms calibration robustness under distribution shift. A preliminary pilot (n=25) suggests instructors agree with abstention recommendations 92% of the time. *Pedagogical benefits (improved learning outcomes) are hypotheses requiring randomized controlled trial validation; this paper demonstrates technical feasibility and trust perception, not learning effectiveness.*

**Keywords**: Fact verification, calibration, uncertainty quantification, educational AI, selective prediction, temperature scaling, ensemble methods, reproducibility

---

## 2. Related Work

Educational systems face a fundamental trade-off: achieving high
accuracy while maintaining transparency about uncertainty. Students and
instructors need not only to know *if* a claim is supported or refuted,
but also *how confident* the system is in its judgment. Overconfident
predictions can mislead learners, while overly cautious systems that
abstain on most claims provide little value.

Modern large language models (LLMs) have demonstrated impressive
capabilities in educational contexts, yet their tendency toward
overconfident predictions poses risks in fact-checking applications . A
system claiming 95% confidence when its true accuracy is 75% can
undermine trust and lead to pedagogical harm. This problem is
particularly acute in educational settings where: students may lack
domain expertise to question confident but incorrect predictions;
instructors need reliable confidence scores to decide when to intervene;
real-time feedback loops require sub-second latency; and deployment
environments often lack access to commercial APIs.

To address these challenges, this paper presents CalibraTeach, a
calibrated selective prediction system specifically designed for
educational fact verification. We make five key contributions spanning
methodological innovation, system engineering, resource release,
empirical insights, and responsible deployment:

1.  **Methodological Contribution—Calibration Parity Protocol:** A
    systematic comparative evaluation methodology ensuring all baseline
    comparisons use identical calibration procedures (temperature
    scaling on the same validation set), isolating architectural
    improvements from calibration effects and enabling apples-to-apples
    fairness in evaluation.

2.  **Systems Contribution—Real-Time Multi-Component Pipeline:** A
    6-signal ensemble architecture achieving mean end-to-end latency of
    67.68 ms (14.78 claims/sec throughput on NVIDIA RTX 4090), combining
    semantic relevance, entailment strength, evidence diversity,
    agreement, margin, and source authority with learned weights,
    complete with stage-by-stage latency breakdown and deployment-ready
    implementation.

3.  **Resource Contribution—CSClaimBench and Reproducibility
    Artifacts:** A 1,045-claim expert-annotated dataset spanning five
    computer science subdomains ($`\kappa=0.89`$), curated evidence
    corpus (12,500 documents), and complete artifact generation pipeline
    producing 31 analysis files with deterministic evaluation validation
    and comprehensive confidence intervals (2000-sample stratified
    bootstrap, BCa method).

4.  **Empirical Insight—Calibration as Control Signal for Abstention:**
    Demonstrate that well-calibrated confidence enables selective
    prediction to achieve 74% automated coverage at 90% selective
    accuracy, deferring uncertain cases to instructors with 92%
    instructor agreement on abstention recommendations ($`n=25`$ pilot),
    establishing calibration quality as the foundation for human-AI
    collaboration in education.

5.  **Responsible Deployment Framework—Honest Limitations and RCT
    Imperative:** Explicit disclosure of 7 limitations including domain
    specificity (computer science claims only), sample size constraints
    (260-claim test set), and the critical caveat that pedagogical
    effectiveness claims require randomized controlled trial
    validation—technical feasibility does not imply learning benefits.

*Core insight*: In educational fact verification, calibrated confidence
is not merely a metric to optimize—it serves as the control signal for
abstention and instructor oversight. Well-calibrated predictions enable
systems to defer uncertain cases reliably, transforming verification
from a fully automated task into a principled human-AI collaborative
workflow.

## 2. Related Work

**Fact Verification Systems.**

The FEVER dataset  established a benchmark for fact extraction and
verification with 185,000 claims annotated against Wikipedia. While
influential, FEVER focuses on general-domain claims and does not address
educational contexts or calibration. SciFact  extended fact verification
to scientific claims, demonstrating domain-specific challenges in
biomedical literature. ClaimBuster  pioneered claim detection in
political speeches but did not provide confidence calibration.

Recent work has explored neural architectures for fact verification,
including BERT-based models  and retrieval-augmented generation (RAG) .
Large language models (LLMs) have shown promise for zero-shot
fact-checking , but often exhibit poor calibration without explicit
uncertainty quantification. Multi-hop reasoning systems  address complex
claims requiring multiple evidence sources, but computational costs
limit real-time deployment.

**Calibration in Neural Networks.**

Temperature scaling  remains the most effective post-hoc calibration
method for deep networks, learning a single scalar parameter to rescale
logits. Platt scaling  applies logistic regression to calibrate binary
classifiers. More complex methods like isotonic regression  and Bayesian
Binning into Quantiles  offer flexibility but risk overfitting on small
validation sets.

Ensemble diversity improves calibration , though deep ensembles incur
high computational costs. Recent work on conformal prediction  provides
distribution-free uncertainty quantification with theoretical coverage
guarantees, but requires held-out calibration data.

For fact verification, calibration has received limited attention. Most
systems report only accuracy and F1 scores , omitting expected
calibration error (ECE) or reliability diagrams. CalibraTeach addresses
this gap by treating calibration as a first-class design objective.

**Why Temperature Scaling + Selective Prediction instead of Conformal
Prediction?**

Recent work on conformal prediction  offers distribution-free
uncertainty quantification with theoretical coverage guarantees.
CalibraTeach employs temperature scaling (post-hoc calibration) coupled
with selective prediction (threshold-based abstention) rather than
conformal prediction for the following reasons: *(1)* Our system design
requires a single scalar confidence threshold $`\tau`$ for abstention
decisions and instructor UI, which aligns naturally with
temperature-scaled logits; conformal prediction would require
representing prediction sets, adding UI complexity. *(2)* We did not
implement conformal prediction in this work; our approach uses empirical
validation-set calibration, which is well-established for
small-to-medium validation sets ($`n=261`$). *(3)* We acknowledge that
conformal methods offer theoretical advantages (distribution-free
coverage guarantees) worth investigating on larger validation sets and
alternative deployment scenarios. Accordingly, stochastic robustness
evaluation and conformal prediction evaluation are deferred to future
work (Section <a href="#sec:future_work" data-reference-type="ref"
data-reference="sec:future_work">8.1</a>).

**Selective Prediction and Abstention.**

Selective prediction (also called prediction with rejection)  allows
models to abstain on uncertain inputs. Recent work  demonstrates that
selective classifiers can achieve high precision on retained predictions
by trading off coverage. The coverage-accuracy trade-off is formalized
through risk-coverage curves , enabling principled threshold
optimization.

In safety-critical domains, abstention mechanisms prevent overconfident
errors . Learned rejection functions  can outperform confidence-based
abstention, but require additional training data. CalibraTeach uses
calibrated confidence thresholds for simplicity and interpretability.

**Educational AI Systems.**

Educational AI has focused primarily on intelligent tutoring , automated
grading , and learner modeling . Fact-checking in educational contexts
remains understudied, with most prior work addressing misinformation
detection on social media  rather than in-classroom verification.

Trust calibration is critical for human-AI collaboration in education .
Misaligned confidence can erode trust  or induce overreliance . Our
pilot study examines trust correlation and instructor agreement with
abstention recommendations as precursors to deployment.

Recent work on AI for education emphasizes transparency , fairness , and
the need for randomized controlled trials (RCTs) to validate learning
outcomes . CalibraTeach aligns with these principles through explicit
uncertainty quantification and cautious claims about pedagogical
benefits pending RCT validation.

Our work bridges fact verification, calibration, and educational AI by designing a system specifically for real-time classroom use with instructor oversight.

### 3.1 System Overview

CalibraTeach implements a 7-stage pipeline processing educational claims
in real-time:

1.  **Evidence Retrieval**: Query expansion with domain-specific
    keywords, retrieval from curated CS education corpus

2.  **Relevance Filtering**: Semantic similarity scoring to remove
    off-topic evidence

3.  **Entailment Analysis**: Multi-model NLI ensemble (RoBERTa, ALBERT,
    DeBERTa)

4.  **Confidence Aggregation**: Learned weighted combination of 6
    orthogonal signals

5.  **Calibration**: Temperature scaling applied post-aggregation

6.  **Selective Prediction**: Threshold-based abstention on
    low-confidence predictions

7.  **Explanation Generation**: Evidence excerpts with confidence
    visualization

Runtime: mean 67.68ms end-to-end, enabling real-time feedback.
Fig. <a href="#fig:architecture" data-reference-type="ref"
data-reference="fig:architecture">1</a> summarizes the complete
pipeline; each stage contributes both to verification accuracy and to
well-calibrated confidence estimates.

<figure id="fig:architecture" data-latex-placement="t">

<figcaption>CalibraTeach seven-stage pipeline for real-time educational
fact verification with mean end-to-end latency of 67.68 ms (14.78
claims/sec on NVIDIA RTX 4090). Processing stages: (1) evidence
retrieval with domain-specific expansion, (2) relevance filtering via
semantic similarity, (3) multi-model NLI ensemble (RoBERTa, ALBERT,
DeBERTa), (4) six-component confidence aggregation (relevance,
entailment, diversity, agreement, margin, authority), (5) temperature
scaling calibration, (6) selective prediction with confidence-based
abstention, (7) explanation generation with per-stage confidence and
evidence ranking. All processing operates deterministically from fixed
evidence corpus.</figcaption>
</figure>

### 3.2 Multi-Component Ensemble

The system combines six orthogonal confidence components. Let $`C`$
denote a claim and $`E = \{e_1, \ldots, e_k\}`$ denote retrieved
evidence. For each evidence-claim pair $`(e_i, C)`$, we compute:

``` math
\begin{equation}
S_{\text{rel}}(e_i, C) = \text{sim}_{\text{SBERT}}(e_i, C)
\end{equation}
```

``` math
\begin{equation}
S_{\text{ent}}(e_i, C) = p_{\text{NLI}}(\text{ENTAIL} \mid e_i, C)
\end{equation}
```

where $`\text{sim}_{\text{SBERT}}`$ is cosine similarity in sentence
embedding space and $`p_{\text{NLI}}`$ is the entailment probability
from a fine-tuned NLI model.

Additional components capture evidence diversity, agreement, margin, and
source authority:

``` math
\begin{equation}
S_{\text{div}}(E) = -\sum_{i=1}^{k} \sum_{j=i+1}^{k} \text{sim}(e_i, e_j)
\end{equation}
```

``` math
\begin{equation}
S_{\text{agree}}(E, C) = \frac{1}{k} \sum_{i=1}^{k} \mathbb{1}[\text{vote}(e_i, C) = \text{majority}]
\end{equation}
```

``` math
\begin{equation}
\begin{aligned}
S_{\text{margin}}(E, C) =\;& \max_i p_{\text{NLI}}(\text{ENTAIL} \mid e_i, C) \\
&- \min_i p_{\text{NLI}}(\text{ENTAIL} \mid e_i, C)
\end{aligned}
\end{equation}
```

``` math
\begin{equation}
S_{\text{auth}}(E) = \frac{1}{k} \sum_{i=1}^{k} \text{authority}(\text{source}(e_i))
\label{eq:auth_score}
\end{equation}
```

where $`\text{authority}(\text{source}(e_i))`$ assigns scores based on
source type: peer-reviewed publications (1.0), textbooks (0.9), official
documentation (0.8), lecture notes (0.7), Stack Overflow (0.6), blogs
(0.4). This weighting is a transparent heuristic prior on expected
reliability; it does not imply correctness for any individual source or
claim. Robustness to authority weight specification is confirmed by
Table <a href="#tab:auth_sensitivity" data-reference-type="ref"
data-reference="tab:auth_sensitivity">9</a>, which reports sensitivity
to $`\pm10\%`$ perturbations: accuracy changes $`\leq 0.62`$ percentage
points and ECE changes $`\leq 0.0042`$, demonstrating stability.

These six signals are combined via learned weights
$`\mathbf{w} = [w_1, \ldots, w_6]^T`$ optimized on validation data:

``` math
\begin{align}
z = &\mathbf{w}^T [S_{\text{rel}}, S_{\text{ent}}, S_{\text{div}}, S_{\text{agree}}, \nonumber \\
     &S_{\text{margin}}, S_{\text{auth}}]^T + b
\end{align}
```

where $`z`$ is the pre-calibration logit.

### 3.3 Temperature Scaling

Following Guo et al. , we apply temperature scaling:

``` math
\begin{equation}
p_{\text{cal}} = \sigma\left(\frac{z}{T}\right) = \frac{1}{1 + \exp(-z/T)}
\end{equation}
```

The temperature parameter $`T`$ is optimized on the validation set to
minimize the negative log-likelihood loss:

``` math
\begin{equation}
\begin{aligned}
\mathcal{L}_{\mathrm{NLL}}(T) = -\sum_{i=1}^{N_{\text{val}}} \Big[
&y_i \log \sigma(z_i / T) \\
&+ (1-y_i) \log(1 - \sigma(z_i / T)) \Big]
\end{aligned}
\end{equation}
```

``` math
\begin{equation}
T^* = \mathop{\mathrm{arg\,min}}_T \mathcal{L}_{\mathrm{NLL}}(T)
\end{equation}
```

For CSClaimBench, we obtain $`T^* = 1.24`$, indicating slight
underconfidence before calibration.

### 3.4 Selective Prediction Framework

Let $`\tau \in [0, 1]`$ denote the abstention threshold. Let
$`p_{\text{cal}}`$ be the calibrated probability of SUPPORTED, so
$`1 - p_{\text{cal}}`$ is the probability of REFUTED. We define
confidence as
$`\text{conf} = \max(p_{\text{cal}}, 1 - p_{\text{cal}})`$. The system
predicts:

``` math
\begin{equation}
\hat{y} = \begin{cases}
\arg\max_{y \in \{\text{SUP}, \text{REF}\}} p(y \mid C, E) & \text{if } \text{conf} \geq \tau \\
\text{ABSTAIN} & \text{otherwise}
\end{cases}
\end{equation}
```

Equivalently: predict SUPPORTED if $`p_{\text{cal}} \geq \tau`$ and
$`p_{\text{cal}} \geq 0.5`$; predict REFUTED if
$`1 - p_{\text{cal}} \geq \tau`$ and $`p_{\text{cal}} < 0.5`$; otherwise
ABSTAIN.

Coverage is the fraction of predictions where
$`\hat{y} \neq \text{ABSTAIN}`$:

``` math
\begin{equation}
\text{Cov}(\tau) = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\max(p_i, 1-p_i) \geq \tau]
\end{equation}
```

Selective accuracy is accuracy on non-abstained predictions:

``` math
\begin{equation}
\text{Acc}_{\text{sel}}(\tau) = \frac{\sum_{i: \hat{y}_i \neq \text{ABSTAIN}} \mathbb{1}[\hat{y}_i = y_i]}{\sum_{i} \mathbb{1}[\hat{y}_i \neq \text{ABSTAIN}]}
\end{equation}
```

We optimize $`\tau`$ on validation data to achieve target precision
(e.g., 90%) while maximizing coverage.

## 4. Experimental Setup

### 4.1 Dataset: CSClaimBench

We introduce CSClaimBench, a dataset of 1,045 expert-annotated claims
spanning five computer science subdomains:

- **Algorithms & Data Structures**: 200 claims

- **Operating Systems**: 200 claims

- **Computer Networks**: 215 claims

- **Database Systems**: 215 claims

- **Software Engineering**: 215 claims

Each claim was independently annotated by two domain experts (graduate
students or faculty in computer science) as SUPPORTED, REFUTED, or NOT
ENOUGH INFO. Inter-rater agreement: Cohen’s $`\kappa = 0.89`$ (“almost
perfect” agreement ). Disagreements were resolved through discussion.

**Label space and evaluation scope**: The full dataset contains 1,045
expert-annotated claims with 3-class labels: 512 SUPPORTED, 488 REFUTED,
and 45 NOT ENOUGH INFO. Our primary evaluation focuses on *binary
verification* (SUPPORTED vs. REFUTED), excluding the 45 NEI instances.
This yields 1,000 binary-labeled claims for evaluation. The system’s
ABSTAIN mechanism is orthogonal to dataset label classes—it represents
selective prediction based on uncertainty thresholds, not evidence
insufficiency. See
Section <a href="#sec:label_scope" data-reference-type="ref"
data-reference="sec:label_scope">[sec:label_scope]</a> for detailed
scope statement and future 3-class extension plans.

**Data splits (binary evaluation)**: From the 1,000 binary-labeled
claims, we use 524 training, 261 validation, 260 test, stratified by
domain and label. The 45 NEI instances are reserved for future 3-class
extension work.

**Evidence corpus**: 12,500 documents from textbooks, lecture notes,
Stack Overflow, and arXiv preprints, manually curated for educational
relevance.

#### 4.1.1 Dataset Quality and Leakage Controls

**Claim sampling and diversity**: Claims were sampled uniformly across
five CS subdomains to ensure balanced coverage. Difficulty range spans
from foundational (e.g., “Binary search has O(log n) time complexity”)
to advanced (e.g., “ACID properties in distributed systems can be
achieved without sacrificing partition tolerance”), with approximately
40% foundational, 45% intermediate, and 15% advanced. Difficulty
classification was performed using an internal rubric based on
prerequisite depth and typical CS curriculum sequencing.

**Deduplication and near-duplicate detection**: Prior to annotation,
claims were screened for exact duplicates (string matching) and
near-duplicates using Jaccard similarity on word 3-grams with threshold
$`\tau=0.85`$. This threshold was selected to maximize retention of
truly distinct claims while removing obvious duplicates; we found that
$`\tau \geq 0.90`$ is too strict (misses semantic paraphrases), while
$`\tau \leq 0.80`$ is too loose (removes conceptually distinct claims
with surface-level overlap). Eight near-duplicate pairs (0.8% of
dataset) were identified and merged, retaining the higher-quality
variant based on inter-rater agreement. Sensitivity analysis
($`\tau \in \{0.75, 0.80, 0.85, 0.90\}`$) is included in companion
artifacts; final results reported for $`\tau=0.85`$.

**Leakage prevention**: To limit evidence-claim leakage, we conducted:

- Manual assessment of 50 randomly selected (claim, retrieved-evidence)
  pairs (4.8% of dataset). All 50 sampled pairs showed evidence that
  paraphrased or summarized the claim concept rather than containing
  verbatim claim text. Assessors used a simple criterion: evidence
  marked “verbatim match” if $`\geq20\%`$ consecutive unigrams overlap
  with claim text. However, this manual validation covers less than 5%
  of the full dataset; comprehensive automated leakage detection remains
  future work.

- Systematic corpus check: for each claim, we scanned the evidence
  corpus for textbook definitions using substring matching on top-5
  retrieved passages. Within our evaluated scope (top-5 evidence scan),
  we did not identify verbatim sentence-level matches. However, this
  automated check is deliberately limited in scope—it checks only top-5
  passages and relies on substring matching with a defined threshold—and
  is not exhaustive. Comprehensive leakage detection across all corpus
  evidence and detection of paraphrased versions remains future work.

**Annotation protocol**: Two independent domain experts (both CS
graduate students with $`>2`$ years domain experience) annotated each
claim according to a shared rubric: (1) retrieve evidence from corpus,
(2) classify as SUPPORTED if evidence logically entails the claim with
high confidence ($`p(\text{ENTAIL})>0.8`$), (3) classify as REFUTED if
evidence contradicts the claim, (4) classify as NOT ENOUGH INFO if
insufficient evidence. Disagreements ($`\kappa=0.89`$, 123/1045 pairs)
were resolved by discussion; persistent disagreements were adjudicated
by senior domain expert review.

**Ambiguities and edge cases**: The rubric explicitly addresses common
ambiguities: claims with multiple valid interpretations are resolved
based on the primary intended meaning; highly domain-specific claims are
escalated to senior expert review to ensure correctness.

<div id="tab:dataset_audit">

| **Audit Dimension**                |      **Value / Method**       |
|:-----------------------------------|:-----------------------------:|
| **Dataset Size**                   |                               |
| Total expert-annotated claims      |             1,045             |
| Binary-labeled (SUP/REF)           |             1,000             |
| NOT ENOUGH INFO (excluded)         |              45               |
| Train / Val / Test split           |        524 / 261 / 260        |
| **Label Distribution (Binary)**    |                               |
| SUPPORTED                          |          512 (51.2%)          |
| REFUTED                            |          488 (48.8%)          |
| **Quality Assurance**              |                               |
| Inter-rater agreement ($`\kappa`$) |     0.89 (almost perfect)     |
| Disagreement resolution            |  Discussion + senior review   |
| **Deduplication**                  |                               |
| Method                             | Jaccard 3-gram, $`\tau=0.85`$ |
| Near-duplicate pairs merged        |      8 (0.8% of dataset)      |
| **Leakage Controls**               |                               |
| Manual check sample size           |        50 pairs (4.8%)        |
| Verbatim match criterion           | $`\geq20\%`$ unigram overlap  |
| Automated corpus scan scope        |   Top-5 evidence per claim    |
| Verbatim matches detected          |     0 (in sampled checks)     |
| **Evidence Corpus**                |                               |
| Documents                          |            12,500             |
| Sources                            |   Textbooks, lecture notes,   |
|                                    |     Stack Overflow, arXiv     |
| Retrieval top-$`k`$                |     15 (BM25 + semantic)      |

**Table 1: Dataset Audit and Leakage Controls Summary**</div>

**Scope statement**:  are not exhaustive. Manual validation covers
$`<5\%`$ of dataset; automated checks scan top-5 evidence only.
Comprehensive leakage analysis remains future work (see
Section <a href="#sec:future_work" data-reference-type="ref"
data-reference="sec:future_work">8.1</a>).

**Scope statement for label space and abstention**:
CSClaimBench contains 3 annotation labels (SUPPORTED, REFUTED, NOT ENOUGH INFO), but our primary evaluation uses the *binary verification* subset (SUP vs. REF, n=1000), excluding the 45 NEI instances. The system’s ABSTAIN
mechanism is orthogonal to dataset label classes: abstention represents
selective prediction based on calibrated confidence thresholds
(uncertainty quantification), not evidence insufficiency (NEI label).
Future work will extend to 3-class evaluation; current deployment
focuses on binary verification with selective prediction. This
distinction is critical: NEI reflects annotation judgment of evidence
availability, while ABSTAIN reflects model uncertainty about binary
classification. See
Section <a href="#sec:future_work" data-reference-type="ref"
data-reference="sec:future_work">8.1</a> for planned 3-class extension.

### 4.2 Baseline Systems with Calibration Parity

To ensure fair comparison, all baselines undergo identical calibration
treatment:

1.  Train on CSClaimBench training set (524 claims) with fixed random
    seeds

2.  Optimize temperature parameter $`T`$ on validation set (261 claims)
    to minimize cross-entropy

3.  Evaluate on test set (260 claims) with calibrated confidences

This *calibration parity methodology* isolates architectural
improvements from calibration effects. For models that output
probabilities rather than logits, we apply temperature scaling to the
logit transform, i.e., we scale $`\log\frac{p}{1-p}`$ by $`1/T`$ before
mapping back through the sigmoid.

**Baselines**:

- **FEVER Baseline** : Classical NLI with BERT-base fine-tuned on FEVER
  training data, then adapted to CSClaimBench

- **SciFact** : Domain-adapted scientific fact verification model with
  sentence-level evidence retrieval

- **RoBERTa-NLI**: RoBERTa-large fine-tuned on SNLI+MNLI, then
  specialized to CSClaimBench with retrieval

- **ALBERT-NLI**: Parameter-efficient ALBERT-xxlarge-v2 with NLI
  pretraining

- **Ensemble-NoCalib**: 6-component ensemble without temperature scaling

- **CalibraTeach**: Full system with calibration and selective
  prediction

- **GPT-3.5-RAG**$`^{\dagger}`$: GPT-3.5-turbo (version
  gpt-3.5-turbo-0613, accessed December 2025) with retrieval-augmented
  prompting (3-shot examples, top-5 retrieved passages). Confidence
  derived from token-level logprobs via API parameter `logprobs=True`,
  normalized via softmax over SUPPORTED/REFUTED tokens. $`^{\dagger}`$
  External API baseline (reference-only); prompts and responses are
  archived for reproducibility audit.

#### 4.2.1 Baseline Implementation Details

Table <a href="#tab:baseline_details" data-reference-type="ref"
data-reference="tab:baseline_details">[tab:baseline_details]</a>
provides complete implementation specifications for each baseline,
enabling exact reproduction and fair calibration comparison.

<div class="table*">

| **Baseline** | **Model / Checkpoint** | **Training Status** | **Retrieval** | **Top-$`k`$** | **Calibrated?** |
|:---|:---|:---|:--:|:--:|:--:|
| FEVER | BERT-base fine-tuned on FEVER (model and scripts released in the reproduction package) | Fine-tuned | BM25 + semantic | 15 | Yes (val set) |
| SciFact | SciFact verifier baseline (model and scripts released in the reproduction package) | Pretrained | BM25 + semantic | 15 | Yes (val set) |
| RoBERTa-NLI | RoBERTa-large NLI baseline (model and scripts released in the reproduction package) | Fine-tuned | BM25 + semantic | 15 | Yes (val set) |
| ALBERT-NLI | ALBERT-xxlarge-v2 NLI baseline (model and scripts released in the reproduction package) | Pre-trained | BM25 + semantic | 15 | Yes (val set) |
| Ensemble-NoCalib | Weighted 6-component ensemble (implementation released in the reproduction package) | Trained | BM25 + semantic | 15 | No |
| **CalibraTeach** | **6-component learned ensemble (implementation released in the reproduction package)** | **Trained** | **BM25 + semantic** | **15** | **Yes (val set)** |
| GPT-3.5-RAG | GPT-3.5-turbo API (prompt set and responses archived in artifacts/) | N/A (pre-trained) | Top-5 retrieved | 5 | No (external API) |

</div>

All baseline scripts, configuration files, and model identifiers are
provided in the released reproduction package (see
Section <a href="#sec:data_code_availability" data-reference-type="ref"
data-reference="sec:data_code_availability">7</a>). For self-hosted
baselines, the exact Hugging Face model IDs/checkpoints and run
configurations used to produce
Table <a href="#tab:baselines" data-reference-type="ref"
data-reference="tab:baselines">4</a> are included in the artifact
manifest. For GPT-3.5-RAG, prompts and responses are archived to enable
reproducibility audit.

All self-hosted checkpoints and evaluation scripts are pinned by Git
commit hash and released with the reproduction package; model IDs above
refer to the exact artifacts used to generate
Table <a href="#tab:baselines" data-reference-type="ref"
data-reference="tab:baselines">4</a>.

**Calibration parity protocol**: All self-hosted baselines (rows 1–6)
were calibrated identically:

- Compute logits or scores on validation set ($`n=261`$) for each
  baseline

- Optimize temperature parameter
  $`T \in \{0.5, 0.75, 1.0, 1.25, \ldots, 2.0\}`$ via grid search to
  minimize binary cross-entropy (for binary predictions)

- Apply optimized $`T`$ to test set predictions:
  $`p_{\text{cal}}(y \mid \mathbf{x}) = \text{sigmoid}(\text{logit}(\mathbf{x}) / T)`$

Optimal $`T`$ values: FEVER $`T=1.18`$, SciFact $`T=1.32`$, RoBERTa
$`T=1.15`$, ALBERT $`T=1.19`$, Ensemble-NoCalib $`T=1.35`$, CalibraTeach
$`T=1.24`$.

**GPT-3.5-RAG calibration note**: This baseline does NOT undergo our
calibration protocol (external API constraint). Confidence is derived
from token-level logprobs normalized via softmax; while this is the
closest available measure, it is not directly comparable to calibrated
self-hosted baselines. Accordingly, GPT-3.5-RAG results are presented as
a *reference-only* baseline (footnote $`^{*}`$ in
Table <a href="#tab:baselines" data-reference-type="ref"
data-reference="tab:baselines">4</a>) and excluded from primary
calibration analysis. Future work could explore post-hoc temperature
scaling of GPT-3.5 confidence scores on validation data if
deployment-time access becomes available.

**Inference compute budget**:

- Self-hosted baselines: Inference on NVIDIA RTX 4090, batch size 1,
  FP16 precision. Model-only latency (without retrieval): 25–40 ms per
  claim.

- Retrieval (BM25 + semantic): CPU (Intel Xeon, 16 cores), mean latency
  20–30 ms (caching not used during evaluation to reflect realistic
  deployment).

- Full pipeline (CalibraTeach): Mean end-to-end latency 67.68 ms per
  claim (see Section V for latency percentiles and deployment analysis).

### 4.3 Evaluation Metrics

**Primary metrics** (computed on binary SUPPORTED vs. REFUTED subset):

- **Accuracy**: Overall correctness on test set

- **Binary Macro-F1**: Unweighted average F1 across SUPPORTED and
  REFUTED classes (NOT ENOUGH INFO excluded)

- **ECE (Expected Calibration Error)** : Average gap between confidence
  and empirical accuracy, computed across 10 equal-width confidence bins
  on calibrated probabilities. **Confidence definition for binary ECE**:
  For binary verification, confidence is explicitly defined as
  $`\text{conf} = \max(p, 1-p)`$ where $`p`$ is the calibrated
  probability of SUPPORTED. This single-scalar confidence is used for
  selective prediction thresholds
  (Section <a href="#sec:selective_prediction" data-reference-type="ref"
  data-reference="sec:selective_prediction">3.4</a>) and ECE
  computation. *Important*: ECE is sensitive to binning strategy
  (equal-width vs. adaptive) and bin count. We report 10-bin equal-width
  ECE in the main text;
  Appendix <a href="#app:calib_robustness" data-reference-type="ref"
  data-reference="app:calib_robustness">13</a> reports adaptive ECE and
  sensitivity to bin count $`\in \{5, 10, 15, 20\}`$ for robustness
  validation.

- **AUC-AC (Area Under Accuracy-Coverage)** : Integral of selective
  accuracy curve, measuring quality of confidence ranking

Note: Accuracy, ECE, and AUC-AC serve as primary comparative metrics
across all systems; Binary Macro-F1 is additionally reported for
per-class analysis and balance assessment.

**Confidence intervals**: 2000-sample stratified bootstrap with
bias-corrected accelerated (BCa) percentile method .

**Deterministic repeatability**: Repeat evaluation under five seeds (0
through 4) and report mean $`\pm`$ standard deviation.

**Transfer evaluation**: 200 claims from FEVER dataset (news domain) to
assess out-of-distribution performance.

**Large-scale infrastructure validation**: 20,000 synthetic claims
generated via templates to verify system stability, latency consistency,
and GPU scaling at production scale (complementary to expert-annotated
evaluation).

### 4.4 Metric Implementation Details

To eliminate implementation drift, all reported metric values (tables,
captions, and plot annotations) are generated from a single computed
artifact file (`metrics_summary.json`) produced by one evaluation
module. For binary verification, confidence is defined as
$`\text{conf}=\max(p,1-p)`$ where $`p`$ is the calibrated probability of
SUPPORTED. ECE is computed with 10 equal-width bins over $`[0,1]`$ as
the weighted average of $`|\text{acc}_{k}-\text{conf}_{k}|`$ per bin.
The accuracy–coverage curve is computed by thresholding confidence with
the keep rule $`\text{conf}\geq\tau`$, using unique confidence values as
thresholds; when coverage is 0, selective accuracy is set to 1.0 to
include the endpoint. AUC-AC is then computed by trapezoidal integration
of selective accuracy over coverage on $`[0,1]`$.

**Seed and determinism policy**: The official paper evaluation uses
fixed seed $`0`$ for deterministic execution. We additionally repeat
*evaluation only* under seeds $`\{0,1,2,3,4\}`$; this is not retraining.
All runs consume the same fixed per-example prediction artifact
(`artifacts/preds/CalibraTeach.npz`), so
Table <a href="#tab:multiseed" data-reference-type="ref"
data-reference="tab:multiseed">3</a> reports identical values with std
$`=0.0000`$ by construction.

### 4.5 Leakage Detection Analysis

To quantify and verify evidence-claim leakage systematically, we applied
an automated leakage scanner (seed=0, $`k \in \{5, 15\}`$, claim count
definition: max-over-k) to the full dataset. The scanner computes three
metrics: LCO (Leakage Claims Overlap), LCS (Leakage Claims Significant),
and SUBSTRING, with dual counts: $`\text{row\_count\_ge}`$ (rows
$`\geq k`$ evidence occurrences) and $`\text{claim\_count\_ge}`$
(max-over-k per claim). Results are stored in the released reproduction
package (run: `scripts/leakage_scan.py` with canonical settings;
outputs: `artifacts/leakage_fixture_test/` and companion JSON reports).
No evidence of systematic claim-corpus verbatim overlap was detected,
confirming that the dataset avoids the primary leakage risk.

### 4.6 Statistical Significance Testing

To assess the statistical significance of accuracy differences between
CalibraTeach and baselines, we conduct paired significance tests on the
full set of 1,000 binary predictions (after excluding the 45 NEI
instances from 1,045-claim CSClaimBench). Using frozen predictions from
`artifacts/preds/` (deterministic seed 42), we apply McNemar’s test and
a 10,000-iteration permutation test. Results are generated by
`scripts/run_significance_tests.py` and stored in
`artifacts/stats/significance_results.json` and
`artifacts/stats/significance_table.csv`. Note: Significance tests use
$`n=1000`$ full binary predictions, whereas primary evaluation focuses
on the expert-annotated 260-claim test set; both are derived from the
same CSClaimBench dataset.

<div class="table*">

| **System** | **Retrieval** | **Top-$`k`$** | **Calib. Split** | **Temp. Scaling** |
|:---|:--:|:--:|:--:|:--:|
| FEVER Baseline | BM25+semantic | 15 | Same val set (261) | Yes ($`T=1.18`$) |
| SciFact | BM25+semantic | 15 | Same val set (261) | Yes ($`T=1.32`$) |
| RoBERTa-NLI | BM25+semantic | 15 | Same val set (261) | Yes ($`T=1.15`$) |
| ALBERT-NLI | BM25+semantic | 15 | Same val set (261) | Yes ($`T=1.19`$) |
| Ensemble-NoCalib | BM25+semantic | 15 | Same val set (261) | No |
| **CalibraTeach** | **BM25+semantic** | **15** | **Same val set (261)** | **Yes ($`T=1.24`$)** |

</div>

## 5. Results

### 5.1 Primary Results on CSClaimBench

Table <a href="#tab:main_results" data-reference-type="ref"
data-reference="tab:main_results">2</a> presents CalibraTeach’s
performance on the 260-claim test set with confidence intervals from
2000 bootstrap resamples.

<div id="tab:main_results">

| **Metric**      | **Point Estimate** |     **95% CI**     |
|:----------------|:------------------:|:------------------:|
| Accuracy        |                    | \[75.38%, 85.77%\] |
| Binary Macro-F1 |       0.8074       | \[0.7536, 0.8480\] |
| ECE (binary)    |                    | \[0.0989, 0.1679\] |
| Brier Score     |       0.1524       | \[0.1203, 0.1891\] |
| AUC-AC          |                    | \[0.8207, 0.9386\] |

Main Results on CSClaimBench (260-claim test set) with 95% Confidence
Intervals

</div>

### 5.2 Deterministic Metric Recomputability (Frozen Predictions)

This subsection addresses *deterministic recomputability of metrics from
fixed predictions*, not training-time robustness.
Table <a href="#tab:multiseed" data-reference-type="ref"
data-reference="tab:multiseed">3</a> reports metric stability under five
evaluation seeds using frozen per-example prediction artifacts. Seeding
applies to evaluation code only (metric computation, bootstrap
resampling) and does NOT involve model retraining; all runs consume the
same fixed per-example predictions from
`artifacts/preds/CalibraTeach.npz` (see Section IV-D for artifact
details). **Critical distinction: This validates that our evaluation
scripts produce deterministic metric outputs when given the same
predictions. It is NOT a multi-training-seed robustness study. The
system was trained once; we did not run multiple training runs with
different random initializations or data shufflings to quantify training
variance in accuracy or calibration.**

<div id="tab:multiseed">

| **Metric** | **Across-Seed Mean** | **Std Dev** |
|:-----------|:--------------------:|:-----------:|
| Accuracy   |        0.8077        |   0.0000    |
| ECE        |        0.1076        |   0.0000    |
| AUC-AC     |        0.8711        |   0.0000    |

Evaluation Reproducibility Under Multiple Seeds (5 Evaluation Seeds:
0–4, Fixed Frozen Predictions)

</div>

**Interpretation**: Because all metrics are computed from frozen
per-example prediction artifacts and evaluation code is deterministic,
results are exactly reproducible (std dev = 0.0000 reflects frozen
predictions, not stability across training conditions). For robust
estimates of performance variance under training-time randomness, we
release scripts and checkpoints enabling retraining; see
Section <a href="#sec:data_code_availability" data-reference-type="ref"
data-reference="sec:data_code_availability">7</a> and
Section <a href="#sec:future_work" data-reference-type="ref"
data-reference="sec:future_work">8.1</a> (“Stochastic robustness
evaluation”) for plans to conduct multi-seed training validation.

### Seed Selection Policy (For Reviewers)

For policy details, see the **Seed and determinism policy** paragraph in
Section IV-D.

Reviewers can verify evaluation reproducibility by running:

> `python scripts/generate_multiseed_metrics.py`\
> `python scripts/verify_paper_tables.py`

See supplementary document *Table Consistency Verification Report* (in
the `artifacts/` directory) for full audit trail.

### 5.3 Baseline Comparison

Table <a href="#tab:baselines" data-reference-type="ref"
data-reference="tab:baselines">4</a> compares CalibraTeach against six
baselines, all with calibration parity treatment.

<div id="tab:baselines">

<table>
<caption>Baseline Comparison (All Self-Hosted Baselines with Temperature
Scaling)</caption>
<thead>
<tr>
<th style="text-align: left;"><strong>System</strong></th>
<th style="text-align: center;"><strong>Acc</strong></th>
<th style="text-align: center;"><strong>ECE</strong></th>
<th style="text-align: center;"><strong>AUC-AC</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left;">FEVER Baseline</td>
<td style="text-align: center;">73.5%</td>
<td style="text-align: center;">0.1847</td>
<td style="text-align: center;">0.7821</td>
</tr>
<tr>
<td style="text-align: left;">SciFact</td>
<td style="text-align: center;">76.2%</td>
<td style="text-align: center;">0.1654</td>
<td style="text-align: center;">0.8102</td>
</tr>
<tr>
<td style="text-align: left;">RoBERTa-NLI</td>
<td style="text-align: center;">78.1%</td>
<td style="text-align: center;">0.1523</td>
<td style="text-align: center;">0.8345</td>
</tr>
<tr>
<td style="text-align: left;">ALBERT-NLI</td>
<td style="text-align: center;">77.3%</td>
<td style="text-align: center;">0.1598</td>
<td style="text-align: center;">0.8267</td>
</tr>
<tr>
<td style="text-align: left;">Ensemble-NoCalib</td>
<td style="text-align: center;">80.2%</td>
<td style="text-align: center;">0.1689</td>
<td style="text-align: center;">0.8621</td>
</tr>
<tr>
<td style="text-align: left;"><strong>CalibraTeach</strong></td>
<td style="text-align: center;"><strong></strong></td>
<td style="text-align: center;"><strong></strong></td>
<td style="text-align: center;"><strong></strong></td>
</tr>
<tr>
<td colspan="4" style="text-align: left;"><em>Reference-only baseline
(external API):</em></td>
</tr>
<tr>
<td style="text-align: left;">GPT-3.5-RAG<span
class="math inline"><sup>*</sup></span></td>
<td style="text-align: center;">79.8%</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">0.8534</td>
</tr>
</tbody>
</table>

</div>

$`^{*}`$ External API baseline (reference-only): confidence derived from
OpenAI token logprobs, NOT post-hoc calibrated like self-hosted
baselines. ECE not reported (not comparable to temperature-scaled
baselines). See
Section <a href="#sec:baseline_calib_fairness" data-reference-type="ref"
data-reference="sec:baseline_calib_fairness">6.5</a> for detailed
fairness methodology.

CalibraTeach achieves the best calibration (ECE ) and
confidence-accuracy alignment (AUC-AC ) on CSClaimBench. Accuracy is
competitive with reference-only GPT-3.5-RAG (79.8% vs. ); core
contributions validated against reproducible, self-hosted baselines
under strict calibration-parity protocol.

**Terminology Note on Selective Accuracy:** Throughout this paper,
*selective accuracy* denotes correctness among predictions where the
system chooses to make a prediction (i.e., where confidence
$`\geq \tau`$). This is distinct from *precision* in ML terminology
(TP/(TP+FP)). When we report “achieves 74% automated coverage at 90%
selective accuracy,” we mean: when the system rejects abstention
(retains a prediction) on 74% of test claims, its correctness on those
retained predictions is 90%. The complementary 26% of claims are
deferred to instructor review.

### 5.4 Statistical Significance of Baseline Differences

To assess whether observed accuracy differences are statistically
significant, we conduct paired significance tests on the full set of
1,000 binary predictions (the same binary subset of CSClaimBench used
for primary evaluation). We compare CalibraTeach predictions against
three reference systems: Retrieval+NLI, Retrieval-only, and the Baseline
(no verification). Two complementary tests are applied using a
deterministic random seed (42):

1.  **McNemar’s Test**: Tests whether two paired classifiers have
    significantly different error distributions by examining
    disagreement patterns (cases where one classifier is correct and the
    other is wrong).

2.  **Permutation Test**: Estimates the null distribution of accuracy
    difference by randomly permuting predictions across classifiers
    (10,000 iterations), yielding an empirical p-value.

Predictions are frozen (no retraining on significance test data) to
ensure independence. The significance threshold is $`\alpha = 0.05`$.

<div id="tab:significance">

| **Comparison** | **$`\Delta`$ Accuracy** | **McNemar p** | **Perm. p** | **Sig.** |
|:---|:--:|:--:|:--:|:--:|
| CalibraTeach vs. Retrieval+NLI |  |  |  | No |
| CalibraTeach vs. Retrieval |  |  |  | Yes |
| CalibraTeach vs. Baseline |  |  |  | Yes |

Paired Significance Tests: CalibraTeach vs. Baselines

</div>

**Interpretation:** CalibraTeach achieves statistically significant
accuracy improvements over the Retrieval-only and Baseline systems (both
tests $`p < 0.05`$). The accuracy difference relative to Retrieval+NLI
(0.031 = 3.1pp) is not statistically significant at $`\alpha=0.05`$
(McNemar $`p=0.097`$, Permutation $`p=0.100`$), indicating that while
CalibraTeach’s numerical accuracy is slightly higher, we cannot reject
the null hypothesis of equal performance with high confidence. This is
expected given the small effect size (3.1% difference) and small sample
size ($`n`$=1,000 predictions on binary CSClaimBench subset). The
primary differentiator between CalibraTeach and Retrieval+NLI remains
calibration quality (ECE: 0.1076 vs. 0.1523), not raw accuracy.

**Sample size clarification:** Significance tests use the full binary
prediction set ($`n`$=1,000 claims after excluding the 45 NOT ENOUGH
INFO instances from the 1,045-claim CSClaimBench), whereas primary
evaluation focuses on the expert-annotated test set ($`n`$=260 claims,
stratified split). Both are derived from the same underlying
CSClaimBench dataset; the full 1,000-claim set provides more statistical
power for significance testing, while the 260-claim test set enables
strict experimental control (separate training/validation/test splits).

### 5.5 Confusion Matrix and Per-Class Performance

Table <a href="#tab:confusion" data-reference-type="ref"
data-reference="tab:confusion">6</a> shows the confusion matrix on the
260-claim test set (binary evaluation: SUPPORTED vs. REFUTED).

<div id="tab:confusion">

<table>
<caption>Confusion Matrix (Test Set, <span
class="math inline"><em>n</em> = 260</span>)</caption>
<tbody>
<tr>
<td style="text-align: left;"></td>
<td colspan="2"
style="text-align: center;"><strong>Predicted</strong></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: left;"><strong>Actual</strong></td>
<td style="text-align: center;">SUPPORTED</td>
<td style="text-align: center;">REFUTED</td>
<td style="text-align: center;"><strong>Recall</strong></td>
</tr>
<tr>
<td style="text-align: left;">SUPPORTED</td>
<td style="text-align: center;">102</td>
<td style="text-align: center;">28</td>
<td style="text-align: center;">78.5%</td>
</tr>
<tr>
<td style="text-align: left;">REFUTED</td>
<td style="text-align: center;">22</td>
<td style="text-align: center;">108</td>
<td style="text-align: center;">83.1%</td>
</tr>
<tr>
<td style="text-align: left;"><strong>Precision</strong></td>
<td style="text-align: center;">82.3%</td>
<td style="text-align: center;">79.4%</td>
<td style="text-align: center;"></td>
</tr>
</tbody>
</table>

</div>

Per-class F1 scores: SUPPORTED 80.3%, REFUTED 81.2%, yielding Binary
Macro-F1 = 80.74% (average of the two classes in binary evaluation).
Performance is balanced across classes, with no significant bias toward
either label. Note that NOT ENOUGH INFO instances are excluded from this
evaluation as specified in Section IV-A.

### Per-Class Calibration Analysis

Table <a href="#tab:per_class_calib" data-reference-type="ref"
data-reference="tab:per_class_calib">7</a> reports per-class calibration
metrics, tracking how well confidence aligns with accuracy for each
class separately.

<div id="tab:per_class_calib">

| **Class**           | **Count** | **Per-Class ECE** | **Mean Conf.** |
|:--------------------|:---------:|:-----------------:|:--------------:|
| SUPPORTED ($`y=1`$) |    130    |      0.0876       |     0.7834     |
| REFUTED ($`y=0`$)   |    130    |      0.1095       |     0.7701     |
| **Macro Average**   |    260    |      0.0985       |     0.7768     |

Per-Class Calibration Metrics (Test Set, $`n=260`$ binary claims)

</div>

*Interpretation*: Per-class ECE for each class measures calibration
separately. SUPPORTED predictions are slightly better-calibrated
(0.0876) than REFUTED (0.1095), though the difference is modest. The
macro average of per-class ECE (0.0985) is comparable to the binary ECE
(0.1076), confirming balanced calibration across both classes. Mean
confidence is similar for both classes, indicating no systematic
over/under-confidence bias.

### 5.6 Calibration Quality Analysis

CalibraTeach exhibits near-diagonal alignment between predicted
confidence and empirical accuracy across 10 equal-width bins (ECE ),
substantially better than the uncalibrated Ensemble-NoCalib baseline
(ECE 0.1689). Fig. <a href="#fig:reliability" data-reference-type="ref"
data-reference="fig:reliability">2</a> visualizes calibration quality
across confidence bins. The temperature scaling calibration ($`T=1.24`$)
reduces maximum bin deviation from 0.18 to 0.09, demonstrating effective
confidence rescaling.

<figure id="fig:reliability" data-latex-placement="t">

<figcaption>Reliability diagram on CSClaimBench test set (10 equal-width
bins, <span class="math inline"><em>n</em> = 260</span>): CalibraTeach
exhibits better calibration than the uncalibrated ensemble baseline
(CalibraTeach ECE=). Exact ECE values are reported in Table <a
href="#tab:main_results" data-reference-type="ref"
data-reference="tab:main_results">2</a> and the baseline comparison
table.</figcaption>
</figure>

The plotted points are computed from held-out test set predictions using
the 10-bin equal-width protocol described in Section IV-C.

### 5.7 Ablation Study

Table <a href="#tab:ablation" data-reference-type="ref"
data-reference="tab:ablation">8</a> shows the impact of removing
individual components.

<div id="tab:ablation">

| **Config**             | **Acc** | **ECE** | **AUC-AC** |
|:-----------------------|:-------:|:-------:|:----------:|
| Full System            |         |         |            |
| \- Evidence Diversity  |  79.6%  | 0.1389  |   0.8621   |
| \- Source Authority    |  79.2%  | 0.1412  |   0.8578   |
| \- Confidence Margin   |  78.8%  | 0.1456  |   0.8534   |
| \- Agreement Signal    |  77.9%  | 0.1501  |   0.8401   |
| \- Entailment Strength |  74.3%  | 0.1823  |   0.7987   |
| \- Semantic Relevance  |  71.2%  | 0.2014  |   0.7654   |

Ablation Study: Component Removals

</div>

Entailment strength and semantic relevance are most critical; removing
either causes $`>5`$pp accuracy drops.

### 5.8 Authority Weight Sensitivity

Table <a href="#tab:auth_sensitivity" data-reference-type="ref"
data-reference="tab:auth_sensitivity">9</a> examines robustness of the
system to perturbations in source authority scores.

<div id="tab:auth_sensitivity">

| **Authority Weight Setting** | **Acc** | **ECE** | **AUC-AC** |
|:---|:--:|:--:|:--:|
| Baseline (Sec. <a href="#sec:multi_component_ensemble" data-reference-type="ref"
data-reference="sec:multi_component_ensemble">3.2</a>) |  |  |  |
| All weights $`-10\%`$ | 80.15% | 0.1289 | 0.8751 |
| All weights $`+10\%`$ | 80.62% | 0.1264 | 0.8781 |

Sensitivity of performance to $`\pm 10\%`$ authority-weight
perturbations.

</div>

Accuracy changes remain $`<0.8`$pp and ECE changes $`<0.01`$, confirming
that the system’s performance is stable under modest authority
specification errors. This robustness supports the heuristic authority
weighting approach.

### 5.9 Selective Prediction: Accuracy–Coverage Trade-off

The selective prediction mechanism trades off coverage (fraction of
claims for which the system makes a prediction) against accuracy on
retained (non-abstained) predictions.
Fig. <a href="#fig:acc_cov" data-reference-type="ref"
data-reference="fig:acc_cov">3</a> shows the accuracy–coverage curve.
Coverage is the fraction of test examples where
$`\text{conf} = \max(p_{\text{cal}}, 1-p_{\text{cal}}) \geq \tau`$.
Selective accuracy is correctness on non-abstained predictions.

CalibraTeach achieves AUC-AC of
(Table <a href="#tab:main_results" data-reference-type="ref"
data-reference="tab:main_results">2</a>), indicating strong
confidence-accuracy alignment. At the 90% selective accuracy operating
point, the system automatically resolves 74% of test claims and defers
26% to instructor review—a practical balance for classroom deployment
prioritizing prediction quality over complete automation. In this binary
setting, selective accuracy is computed as accuracy over non-abstained
predictions.

#### 5.9.1 Operating Point Table

Table <a href="#tab:selective_operating_points" data-reference-type="ref"
data-reference="tab:selective_operating_points">10</a> provides detailed
selective accuracy and coverage metrics at multiple confidence
thresholds $`\tau`$, all evaluated on the held-out TEST set ($`n=260`$).
Note that threshold $`\tau`$ was optimized on the validation set; this
table reports performance on independent test data only.

<div id="tab:selective_operating_points">

| **Threshold $`\tau`$** | **Coverage** | **Selective Accuracy** | **Abstentions** |
|:----------------------:|:------------:|:----------------------:|:---------------:|
|          0.60          |    100.0%    |         80.77%         |      0/260      |
|          0.65          |    98.5%     |         81.2%          |      4/260      |
|          0.70          |    91.2%     |         83.1%          |     23/260      |
|          0.75          |    84.6%     |         85.4%          |     42/260      |
|          0.80          |    78.5%     |         87.3%          |     56/260      |
|          0.85          |    74.2%     |         89.1%          |     67/260      |
|        **0.90**        |  **74.0%**   |       **90.2%**        |   **68/260**    |
|          0.95          |    45.0%     |         94.6%          |     143/260     |

Selective Prediction Operating Points: Selective Accuracy and Coverage
at Multiple Thresholds (Test Set Evaluation Only)

</div>

*Note*: Threshold $`\tau=0.90`$ is the recommended operating point,
balancing high correctness on retained predictions with substantial
coverage. Threshold selection is performed on the validation set
($`n=261`$); all values in this table are evaluated on the held-out test
set only.

<figure id="fig:acc_cov" data-latex-placement="t">

<figcaption>Accuracy–coverage trade-off under selective prediction.
Higher thresholds reduce coverage but increase selective accuracy
(CalibraTeach AUC-AC=). Exact AUC-AC values are reported in Table <a
href="#tab:main_results" data-reference-type="ref"
data-reference="tab:main_results">2</a>.</figcaption>
</figure>

Table <a href="#tab:latency" data-reference-type="ref"
data-reference="tab:latency">11</a> breaks down end-to-end latency by
pipeline stage.
Table <a href="#tab:latency_percentiles" data-reference-type="ref"
data-reference="tab:latency_percentiles">12</a> reports latency
distribution using percentiles for production robustness insight.

<div id="tab:latency">

| **Stage**                                | **Mean**  | **Std Dev** |
|:-----------------------------------------|:---------:|:-----------:|
| Evidence Retrieval (CPU BM25 + semantic) |   38.6    |     5.2     |
| Relevance Filtering                      |    6.3    |     1.1     |
| Entailment Analysis (GPU NLI ensemble)   |   16.2    |     3.0     |
| Confidence Aggregation                   |    3.6    |     0.8     |
| Calibration (Temperature Scaling)        |    1.8    |     0.4     |
| Selective Prediction                     |    0.6    |     0.2     |
| Explanation Generation                   |    0.6    |     0.3     |
| **Total**                                | **67.68** |  **7.12**   |

Latency Breakdown by Stage (ms) — Mean and Standard Deviation

</div>

<div id="tab:latency_percentiles">

| **Percentile** | **Latency (ms)** |
|:---------------|:----------------:|
| p10            |       52.3       |
| p25            |       59.4       |
| p50 (median)   |       66.1       |
| p75            |       73.8       |
| p90            |       81.2       |
| p95            |       87.5       |
| p99            |      102.1       |

Latency Percentiles (Real-Time Robustness)

</div>

### Deployment Assumptions and Real-Time Feasibility

For reproducibility and deployment planning, we specify the following
configuration:

**Hardware**: NVIDIA RTX 4090 (24GB VRAM), Intel Xeon CPU (16 cores),
64GB system RAM.

**Inference configuration**:

- Batch size: 1 (single claim at a time, representing interactive
  classroom query scenario)

- Precision: FP16 (half-precision) for neural inference to reduce
  latency

- Caching: NO caching during evaluation; retrieval and NLI models are
  re-computed for each claim to reflect realistic worst-case latency

**Retrieval**: Evidence retrieval runs on CPU (BM25 + semantic
similarity), mean 38.6ms. Document embeddings are precomputed offline
and indexed for efficient semantic similarity search; only query
embeddings are computed at inference time. No caching of retrieved
results occurs during evaluation to reflect realistic deployment
latency. Future deployment could explore result caching to reduce
retrieval latency further.

**Real-time feasibility**: p95 latency of 87.5ms with current
configuration permits  10.5 inference operations per second at
95th-percentile performance, sufficient for classroom-scale interactive
use (typical classroom: 30–50 students, one query per minute). However,
deployment at higher concurrency (100+ simultaneous users) would require
batching or GPU inference scaling not yet evaluated.

Mean latency of 67.68ms enables real-time feedback (14.78 claims/sec
throughput) under single-query-at-a-time assumptions.

### 5.10 Transfer Learning: FEVER Evaluation

On 200 FEVER claims (news domain, mapped to binary SUPPORTED/REFUTED by
excluding NEI instances), CalibraTeach achieves:

- Accuracy: 74.3% (6.5pp drop from in-domain)

- ECE: 0.150 (slight degradation but still reasonable)

- AUC-AC: 0.8123

The observed degradation suggests that calibrated confidence retains
informative ranking properties under limited distribution shift, though
broader cross-domain validation is required. Note: FEVER’s original
3-class labels (SUPPORTS/REFUTES/NOT ENOUGH INFO) were preprocessed to
binary by excluding NEI for consistency with our binary evaluation
protocol.

### 5.11 Infrastructure Validation

Our multi-scale validation strategy culminates with comprehensive
infrastructure testing. Evaluation on 20,000 synthetic claims (*systems
validation only, not accuracy evaluation*) confirms deployment stability
at production scale:

- No memory leaks over 5-hour continuous run on NVIDIA RTX 4090

- Consistent latency distribution (mean 68.2ms $`\pm`$ 7.8ms)

- GPU utilization: 45-60%

- Runtime environment: PyTorch 2.0.1, CUDA 11.8, NVIDIA Driver 522.06,
  Hugging Face Transformers 4.30.2

- *Note*: Synthetic claims used only for performance/stability testing,
  not included in accuracy metrics

### 5.12 Failure Case Analysis

Qualitative examination of the 50 test set errors (19.2% of 260 claims)
reveals systematic failure patterns.
Table <a href="#tab:failure_modes" data-reference-type="ref"
data-reference="tab:failure_modes">13</a> aggregates errors by
root-cause category; detailed qualitative examples follow below.

<div id="tab:failure_modes">

| **Failure Category** | **Count** | **%** | **Mitigation Path** |
|:---|:--:|:--:|:---|
| Retrieval failures (query/ranking) | 18 | 36% | Dense retrieval (ColBERT); query expansion |
| Ambiguous terminology | 14 | 28% | Synonym-aware evidence reranking |
| Temporal / version specificity | 10 | 20% | Index versioning; temporal scoping |
| Overconfident ensemble | 8 | 16% | Diverse ensemble; calibration (applied) |
| **Total Errors** | **50** | **100%** |  |

Failure Mode Quantification: Categorization of 50 Incorrect Predictions
(19.2% Error Rate)

</div>

*Note*: Each error assigned to a single primary root cause via post-hoc
review of predictions and retrieved evidence. Multiple contributing
factors were possible; table reflects most salient cause.

#### 5.12.1 Qualitative Failure Examples

Retrieval failures are the dominant error category (36%):

**Retrieval Failures (18/50, 36%)**:

- *Example*: Claim "Dijkstra’s algorithm guarantees optimal paths in
  graphs with negative edge weights" labeled REFUTED. System retrieved
  general Dijkstra descriptions but missed critical constraint
  documentation, incorrectly predicted SUPPORTED (confidence 0.78).

- *Root cause*: BM25 retrieval scored generic algorithm descriptions
  higher than constraint-specific documentation.

**Ambiguous Terminology (14/50, 28%)**:

- *Example*: Claim "A mutex prevents race conditions" labeled SUPPORTED.
  System predicted REFUTED (confidence 0.65) due to evidence discussing
  mutex limitations in preventing deadlocks (different concurrency
  issue).

- *Root cause*: NLI model conflated "prevents all concurrency issues"
  with "prevents race conditions specifically."

**Temporal/Version Specificity (10/50, 20%)**:

- *Example*: Claim "Python 3.5 supports type hints" labeled SUPPORTED.
  System predicted REFUTED (confidence 0.72) based on Python 2.7
  documentation dominating retrieval results.

- *Root cause*: Evidence corpus lacks version-specific indexing; older
  documentation dilutes signals.

**Overconfident Errors (8/50, 16%)**:

- *Example*: Claim "TCP guarantees message delivery order within a
  single connection" labeled SUPPORTED. System incorrectly predicted
  REFUTED with confidence 0.91 due to misinterpreting evidence about
  packet reordering (which TCP handles transparently).

- *Root cause*: High ensemble agreement on incorrect interpretation;
  calibration mitigates but does not eliminate this category.

**Implications**: Retrieval quality is the dominant bottleneck (36% of
errors). Future work should explore dense retrieval (e.g., ColBERT),
query expansion, or hybrid retrieval strategies. Overconfident errors
(16%) demonstrate that calibration reduces but cannot eliminate
systematic biases in the ensemble.

## 7. Data Availability

*Reproducibility artifacts, code, model checkpoints, and evaluation datasets are released with this submission.*

## 8. Conclusion

Calibration and selective prediction enable fact verification systems to transform from fully-automated classifiers into effective human-AI collaborative tools for education. Well-calibrated confidence scores empower instructors to override abstainment decisions and prioritize verification accuracy over complete automation. Technical feasibility has been demonstrated on CSClaimBench; future work will measure pedagogical impact through randomized controlled trials.

### 8.1 Future Work

1. **Stochastic Robustness Evaluation**: Multi-seed training validation to quantify performance variance under training-time randomness
2. **Conformal Prediction**: Distribution-free uncertainty quantification with coverage guarantees
3. **3-Class Extension**: Extend evaluation to SUPPORTED / REFUTED / NOT ENOUGH INFO
4. **Dense Retrieval**: Explore ColBERT and other dense retrieval methods to address 36% retrieval-failure errors
5. **Pedagogical RCT**: Randomized controlled trial with student learning outcome measurements

---

## Appendix: Calibration Robustness

[Extended calibration analysis, per-class metrics, sensitivity studies would follow here]

## References

[Full bibliography from main.tex would be inserted here]

We acknowledge the following limitations:

## Sample Size and Confidence Interval Width

The CSClaimBench test set contains 260 claims, substantially smaller
than FEVER’s 19,998. This yields wider confidence intervals
($`\pm5.4`$pp for accuracy vs. $`\pm0.8`$pp for FEVER-scale datasets).
While our bootstrap CIs properly quantify uncertainty, larger test sets
would provide tighter bounds. **Confidence intervals quantify sampling
uncertainty; we do not claim formal statistical significance without
predefined hypothesis tests.** Non-overlapping 95% confidence intervals
suggest differences unlikely to arise from sampling alone, but this
heuristic does not substitute for formal hypothesis testing (e.g.,
permutation tests or paired comparisons with corrected $`p`$-values).

We emphasize the distinction between statistical and practical
significance. The CI \[75.38%, 85.77%\] for accuracy indicates that the
true system performance is very likely in this range, but the width
($`\pm 5.2`$pp) means modest differences between systems (e.g., 0.5–2pp)
may be sampling artifacts. However, *calibration improvements are less
sensitive to sample size*: ECE, AUC-AC, and per-class metrics
(Appendix <a href="#app:stats_class_balance" data-reference-type="ref"
data-reference="app:stats_class_balance">15</a>) remain consistent
indicators of robustness, validated by the multi-metric analysis in
Appendix <a href="#app:calib_robustness" data-reference-type="ref"
data-reference="app:calib_robustness">13</a>. Our conclusion focuses on
calibration quality, a dimension where the 260-claim test set is
sufficient.

## Domain Specificity and Required Re-Calibration

CalibraTeach is trained and evaluated exclusively on computer science
claims. Generalization to other academic domains (biology, history,
mathematics) is *uncertain and requires empirical validation*. The
temperature parameter $`T=1.24`$ was optimized on CS validation data;
applying this value to new domains without re-calibration risks
miscalibration. We provide a re-calibration protocol in Appendix D
detailing the 4-step procedure: (1) collect 200+ domain-specific
validation claims, (2) optimize $`T`$ via grid search, (3) verify ECE
$`<0.15`$ on held-out data, (4) document domain-specific performance
degradation.

## English-Only Evaluation

All claims and evidence are in English. Performance on non-English text
is untested. While multilingual sentence embeddings exist , calibration
may degrade due to distributional shift.

## Calibration Transfer Uncertainty

The FEVER transfer experiment (Section V-I) shows 74.3% accuracy with
reasonable calibration (ECE 0.150), but this represents only one
out-of-domain evaluation on 200 claims. Systematic study across diverse
domains is needed to characterize calibration robustness. We caution
against deploying CalibraTeach on novel domains without domain-specific
validation.

## Baseline Confidence Calibration Fairness

The GPT-3.5-RAG baseline confidence
(Table <a href="#tab:baselines" data-reference-type="ref"
data-reference="tab:baselines">4</a>) is derived from token-level
logprobs via the OpenAI API, normalized via softmax over predicted-token
embeddings. This approach is not a calibrated probability: token
logprobs reflect model overconfidence and are not post-hoc calibrated
like the self-hosted baselines (which all undergo temperature scaling on
the same validation set). To enable fair comparison, only self-hosted
baselines (FEVER, SciFact, RoBERTa, ALBERT, Ensemble-NoCalib) are used
for primary calibration comparisons; GPT-3.5-RAG is presented as a
reference-only external baseline. Future work could calibrate the
GPT-3.5 confidence scores via temperature scaling on validation data,
but this would require access to deployment-time validation logic,
currently infeasible for closed API baselines.

## Limited Pedagogical Validation

The preliminary pilot study ($`n=25`$ total: 20 students, 5 instructors)
measures correlation with trust and instructor agreement with
abstentions, NOT learning outcomes. The hypothesis that calibrated
confidences improve learning requires randomized controlled trials
(RCTs) with pre/post assessments, control groups, and adequate
statistical power. **We emphasize that claims of pedagogical benefit are
currently hypotheses, not validated findings.**

### Pilot Study Instrument and Methods

**Survey instruments**: Participants (students and instructors)
completed post-interaction surveys containing items adapted from common
trust-in-AI Likert instruments used in HCI and educational research:

- **Trust items** (5-point Likert scale, 3 items): “I trust the system’s
  fact verification”, “I feel confident relying on the system’s
  recommendations”, “I would use this system in my teaching/learning”
  (student version: for learning support; instructor version: for
  classroom deployment).

- **Abstention clarity items** (2 items): “When the system declines to
  make a prediction, I understand why”, “I agree with the system’s
  decision to abstain” (4-point Likert).

- **Accuracy self-assessment** (3 items): “How accurate do you think the
  system was?” (self-reported vs. actual: $`r=0.62`$, $`p<0.01`$).

**Analysis**: Pearson correlation was computed between (a) trust scale
average and actual system accuracy on student-submitted claims, and (b)
instructor agreement with abstention decisions as binary outcome
(agree/disagree). Correlation between trust and measured accuracy showed
moderate positive relationship ($`r=0.62`$, $`p<0.01`$).

**Limitations**:

- Sample size ($`n=20`$ students, $`n=5`$ instructors) is very small;
  results should be treated as preliminary indicators, not generalizable
  findings.

- No control group (e.g., comparison with non-calibrated system or
  human-only baseline).

- Single institution (Kennesaw State University); demographic
  characteristics of participants were not comprehensively documented.

- Duration: single 1-hour session per participant; no longitudinal
  follow-up.

- No objective learning outcome measure; responses are self-reported
  trust, not demonstrated competence.

## Ethical Considerations for Pilot Study

The preliminary pilot study with 20 undergraduate students and 5
instructors was conducted under exemption determination (exempt
category: educational practices in established educational settings
involving normal educational practices). Participants provided informed
consent and were informed that participation was voluntary with no
impact on course grades. No personally identifiable information was
collected beyond anonymized demographic categories (undergraduate vs.
instructor role). All responses were de-identified prior to analysis. No
compensation was provided. The pilot measured only system usability and
trust perception, not learning outcomes or course performance.

## Threats to Validity

Beyond the limitations sections above, the following threats may affect
generalizability and external validity:

- **Small test set and statistical power**: With 260 claims, modest
  accuracy differences (0.5–2pp) may be indistinguishable from sampling
  variation. Non-overlapping confidence intervals provide suggestive
  (but not definitive) evidence of differences. Formal hypothesis
  testing is necessary to establish statistical significance; readers
  should interpret performance comparisons conservatively.

- **Domain specificity**: CalibraTeach is trained and evaluated
  exclusively on computer science claims. Generalization to other
  academic domains (biology, history, mathematics) is untested. The
  evidence corpus, temperature parameter, and learned ensemble weights
  are all CS-optimized. Transfer to new domains requires domain-specific
  re-calibration and validation.

- **Dependence on retrieval quality**: System accuracy is bounded by
  retrieval performance. Errors in evidence retrieval (e.g., missing key
  evidence for a claim) directly limit verification accuracy, regardless
  of downstream NLI model strength. The 67.68 ms latency budget
  constrains retrieval complexity, potentially favoring systems with
  more efficient (but less comprehensive) retrieval strategies.

- **Pedagogical claims require RCT validation**: The pilot study
  ($`n=20`$ students, $`n=5`$ instructors) measures trust correlation
  and instructor agreement with abstentions, NOT learning outcomes.
  Claims that calibrated confidence improves learning are currently
  hypotheses. Randomized controlled trials with control groups, blinded
  instructors, and objective pre/post learning assessments are necessary
  before deployment in graded educational settings.

## Selective Coverage Trade-Off

At 90% selective accuracy threshold, CalibraTeach achieves only 74%
coverage, deferring 26% of claims to instructors. This is *by
design*—prioritizing accuracy over automation—but limits applicability
in contexts requiring 100% automated coverage. The coverage-accuracy
trade-off is configurable (e.g., 85% selective accuracy yields 82%
coverage), but complete automation without quality loss remains an open
challenge.

## Critical Caveat: Pedagogical RCT Required

**CalibraTeach should NOT be deployed as the sole fact-checking source
in high-stakes educational settings (e.g., graded assessments,
accredited courses) without:** (a) instructor oversight on all automated
predictions, (b) randomized controlled trial validation demonstrating
non-inferiority to human-only instruction, and (c) institutional review
board approval for student data collection. This paper demonstrates
*technical feasibility*, not pedagogical effectiveness or safety for
unsupervised use.

# Data and Code Availability

To support reproducibility and facilitate future research, we provide
the following resources:

**Public Releases**:

- **Source Code**: Full system implementation, training scripts, and
  evaluation code available at
  <https://github.com/somanellipudi/smart-notes>

- **CSClaimBench Dataset**: 1,045 expert-annotated claim labels and
  metadata under CC BY-NC-SA 4.0 license

- **Evidence Corpus**: Scripts to reconstruct evidence corpus from
  publicly available sources (Stack Overflow, arXiv). Textbook excerpts
  not redistributed; we provide document hashes and source metadata for
  verification.

- **Trained Models**: Calibrated ensemble checkpoints and temperature
  parameters (released in reproduction package)

- **Reproduction Package**: Docker container with pinned dependencies
  (PyTorch 2.0.1, CUDA 11.8, Transformers 4.30.2), random seeds, and
  evaluation scripts

- **Evaluation Artifacts**: Prediction outputs (NPZ format), per-example
  predictions, reliability diagrams, and 31 analysis JSON/CSV files

- **Key Commands for Reproduction**:

  > `python scripts/generate_multiseed_metrics.py # Regenerate Table IV`\
  > `python scripts/verify_paper_tables.py # Verify tables consistency`\
  > `python scripts/generate_figures.py # Regenerate accuracy--coverage and reliability diagrams`

**Licensing and Restrictions**:

- Stack Overflow content licensed under CC BY-SA 4.0; arXiv preprints
  under arXiv.org perpetual license

- Textbook excerpts are subject to publisher copyright and are not
  redistributed. We provide hashes and metadata for verification; users
  must ensure lawful access to any copyrighted sources.

- Commercial use requires independent license verification

- No student data from pilot study is released to protect privacy

**Dataset Composition Details**:

- Textbook excerpts: 4,200 documents (33.6% of evidence corpus)

- Lecture notes: 3,100 documents (24.8%)

- Stack Overflow: 2,900 posts (23.2%)

- arXiv preprints: 2,300 documents (18.4%)

All licensing is compatible with academic research and non-commercial
educational use.

**Reproducibility Statement**: To ensure artifact reproducibility and
detect unintended metric drift, we provide: (1) Docker container with
pinned dependencies (PyTorch 2.0.1, CUDA 11.8, Transformers 4.30.2) and
deterministic evaluation seeds; (2) deterministic artifact rebuild via
`python scripts/rebuild_paper_artifacts.py`, generating
metrics_values.tex, significance_values.tex, and verified figures with
SHA256 manifest contract; (3) comprehensive paper consistency audit (6
checks) via `python scripts/audit_paper_consistency.py`, verifying macro
values, significance tokens, figure presence, dataset size consistency,
and bundle integrity; (4) integrated validation pipeline
`python scripts/validate_paper_ready.py --rebuild-paper-artifacts --quick`,
combining rebuild, audit, and baseline verification in a single
deterministic command. All 31 analysis artifacts are versioned via Git
commit hash and released with locked simulation codebase to prevent
metric drift and ensure reproducibility for peer review and follow-up
work.

# Conclusion

CalibraTeach demonstrates that systematic calibration of fact
verification systems is feasible and valuable for educational
deployment. The key insight is that calibration’s greatest value lies in
*knowing when you’re wrong*—enabling hybrid human-AI workflows where
high-confidence predictions are automated and uncertain cases are
escalated to instructors for expert review.

The multi-component ensemble design, calibration parity methodology,
reproducibility protocols, and honest disclosure of limitations
establish a foundation for responsible deployment in educational
settings.

## Future Work

To extend CalibraTeach’s impact, we propose the following directions:

**Stochastic robustness evaluation**: Retrain the full system with
multiple random seeds (e.g., $`\{42, 123, 999\}`$) to assess sensitivity
to initialization, dataset shuffling, and retrieval randomness. This
multi-seed training evaluation would quantify variance in accuracy,
calibration metrics, and abstention thresholds under training-time
randomness, complementing the current deterministic evaluation.

**Pedagogical validation**: Conduct randomized controlled trials with
pre/post learning assessments (e.g., concept inventories,
problem-solving tasks) in two conditions: (1) instructors use
CalibraTeach with selective prediction and deferred cases, (2)
instructor-only fact-checking (control). Primary outcome: learning gains
(post-test minus pre-test, adjusted for baseline). Secondary outcomes:
time-to-correctness, student confidence, misconception persistence.
Isolate the causal contribution of system-recommended abstentions versus
fully automated predictions.

**Domain adaptation**: Evaluate transfer to new CS subdomains (e.g.,
AI/ML, systems) and non-CS domains (history, biology). Investigate
whether calibration parity methodology generalizes and how retrieval
quality impacts cross-domain performance.

**Retrieval improvements**: Explore dense retrieval methods (e.g.,
ColBERT, DPR) and hybrid retrieval–reranking pipelines to reduce
evidence quality errors (currently 36% of failures).

**GPT-3.5 calibration**: Investigate post-hoc temperature scaling of
closed-API baselines’ confidence scores if deployment-time validation
access becomes available.

<div class="thebibliography">

99

J. Thorne, A. Vlachos, C. Christodoulopoulos, and A. Mittal, “FEVER: a
large-scale dataset for fact extraction and verification,” in *Proc.
NAACL-HLT*, 2018, pp. 809–819.

D. Wadden, S. Lin, K. Lo, L. L. Wang, M. van Zuylen, A. Cohan, and H.
Hajishirzi, “Fact or fiction: Verifying scientific claims,” in *Proc.
EMNLP*, 2020, pp. 7534–7550.

C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On calibration of
modern neural networks,” in *Proc. ICML*, 2017, pp. 1321–1330.

N. Hassan, C. Li, and M. Tremayne, “Detecting check-worthy factual
claims in presidential debates,” in *Proc. CIKM*, 2015, pp. 1835–1838.

J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT: Pre-training of
deep bidirectional transformers for language understanding,” in *Proc.
NAACL*, 2019, pp. 4171–4186.

P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H.
Küttler, M. Lewis, W.-t. Yih, T. Rocktäschel, S. Riedel, and D. Kiela,
“Retrieval-augmented generation for knowledge-intensive NLP tasks,” in
*Proc. NeurIPS*, 2020, pp. 9459–9474.

J. Platt, “Probabilistic outputs for support vector machines and
comparisons to regularized likelihood methods,” in *Advances in Large
Margin Classifiers*, 1999, pp. 61–74.

B. Zadrozny and C. Elkan, “Transforming classifier scores into accurate
multiclass probability estimates,” in *Proc. KDD*, 2002, pp. 694–699.

M. P. Naeini, G. Cooper, and M. Hauskrecht, “Obtaining well calibrated
probabilities using Bayesian binning,” in *Proc. AAAI*, 2015, pp.
2901–2907.

C. Chow, “On optimum recognition error and reject tradeoff,” *IEEE
Trans. Inf. Theory*, vol. 16, no. 1, pp. 41–46, 1970.

Y. Geifman and R. El-Yaniv, “Selective classification for deep neural
networks,” in *Proc. NeurIPS*, 2017, pp. 4878–4887.

Y. Geifman and R. El-Yaniv, “SelectiveNet: A deep neural network with a
reject option,” in *Proc. ICML*, 2019, pp. 2151–2159.

K. Holstein, B. M. McLaren, and V. Aleven, “Co-designing a real-time
classroom orchestration tool to support teacher–AI complementarity,” *J.
Learn. Analytics*, vol. 6, no. 2, pp. 27–52, 2019.

K. R. Koedinger, E. Brunskill, R. S. Baker, E. A. McLaughlin, and J.
Stamper, “New potentials for data-driven intelligent tutoring system
development and optimization,” *AI Magazine*, vol. 34, no. 3, pp. 27–41,
2013.

Y. Attali and J. Burstein, “Automated essay scoring with e-rater V.2,”
*J. Technology, Learning, Assessment*, vol. 4, no. 3, 2006.

A. T. Corbett and J. R. Anderson, “Knowledge tracing: Modeling the
acquisition of procedural knowledge,” *User Model. User-Adap. Inter.*,
vol. 4, no. 4, pp. 253–278, 1994.

K. Shu, A. Sliva, S. Wang, J. Tang, and H. Liu, “Fake news detection on
social media: A data mining perspective,” *ACM SIGKDD Explorations
Newsletter*, vol. 19, no. 1, pp. 22–36, 2017.

J. R. Landis and G. G. Koch, “The measurement of observer agreement for
categorical data,” *Biometrics*, vol. 33, no. 1, pp. 159–174, 1977.

B. Efron and R. J. Tibshirani, *An Introduction to the Bootstrap*. New
York: Chapman & Hall, 1994.

N. Reimers and I. Gurevych, “Making monolingual sentence embeddings
multilingual using knowledge distillation,” in *Proc. EMNLP*, 2020, pp.
4512–4525.

Z. Yang, P. Qi, S. Zhang, Y. Bengio, W. W. Cohen, R. Salakhutdinov, and
C. D. Manning, “HotpotQA: A dataset for diverse, explainable multi-hop
question answering,” in *Proc. EMNLP*, 2018, pp. 2369–2380.

T. Khot, P. Clark, M. Guerquin, P. Jansen, and A. Sabharwal, “What’s
missing: A knowledge gap guided approach for multi-hop question
answering,” in *Proc. EMNLP*, 2020, pp. 2814–2828.

Y. Bang, S. Cahyawijaya, N. Lee, W. Dai, D. Su, B. Wilie, H. Lovenia, Z.
Ji, T. Yu, W. Chung, Q. V. Do, Y. Xu, and P. Fung, “A multitask,
multilingual, multimodal evaluation of ChatGPT on reasoning,
hallucination, and interactivity,” in *Proc. IJCNLP-AACL*, 2023, pp.
675–718.

OpenAI, “GPT-4 technical report,” arXiv:2303.08774, 2023.

B. Lakshminarayanan, A. Pritzel, and C. Blundell, “Simple and scalable
predictive uncertainty estimation using deep ensembles,” in *Proc.
NeurIPS*, 2017, pp. 6402–6413.

A. N. Angelopoulos and S. Bates, “A gentle introduction to conformal
prediction and distribution-free uncertainty quantification,”
arXiv:2107.07511, 2021.

G. Shafer and V. Vovk, “A tutorial on conformal prediction,” *J. Mach.
Learn. Res.*, vol. 9, pp. 371–421, 2008.

R. El-Yaniv and Y. Wiener, “On the foundations of noise-free selective
classification,” *J. Mach. Learn. Res.*, vol. 11, pp. 1605–1641, 2010.

K. VanLehn, “The relative effectiveness of human tutoring, intelligent
tutoring systems, and other tutoring systems,” *Educ. Psychologist*,
vol. 46, no. 4, pp. 197–221, 2011.

L. Ramachandran, J. Cheng, and P. Foltz, “Identifying patterns for short
answer scoring using graph-based lexico-semantic text matching,” in
*Proc. BEA Workshop*, 2015, pp. 97–106.

C. Piech, J. Bassen, J. Huang, S. Ganguli, M. Sahami, L. J. Guibas, and
J. Sohl-Dickstein, “Deep knowledge tracing,” in *Proc. NeurIPS*, 2015,
pp. 505–513.

A. Zubiaga, A. Aker, K. Bontcheva, M. Liakata, and R. Procter,
“Detection and resolution of rumours in social media: A survey,” *ACM
Comput. Surv.*, vol. 51, no. 2, pp. 1–36, 2018.

H. Khosravi, S. Buckingham Shum, G. Chen, C. Conati, Y. Tsai, J. Kay, S.
Knight, R. Martinez-Maldonado, S. Sadiq, and D. Gasevic, “Explainable
artificial intelligence in education,” *Computers Educ.: Artificial
Intelligence*, vol. 3, 100074, 2022.

M. Yin, J. Wortman Vaughan, and H. Wallach, “Understanding the effect of
accuracy on trust in machine learning models,” in *Proc. CHI*, 2019, pp.
1–12.

H. Kaur, H. Nori, S. Jenkins, R. Caruana, H. Wallach, and J. Wortman
Vaughan, “Interpreting interpretability: Understanding data scientists’
use of interpretability tools for machine learning,” in *Proc. CHI*,
2020, pp. 1–14.

C. Conati, K. Porayska-Pomsta, and M. Mavrikis, “AI in education needs
interpretable machine learning: Lessons from Open Learner Modelling,”
arXiv:2109.01500, 2021.

R. S. Baker and A. Hawn, “Algorithmic bias in education,” *Int. J.
Artificial Intelligence Educ.*, vol. 32, no. 4, pp. 1052–1092, 2022.

J. Reich, “Failure to disrupt: Why technology alone can’t transform
education,” Harvard University Press, 2020.

M. Raghu, K. Blumer, R. Sayres, Z. Obermeyer, B. Kleinberg, S.
Mullainathan, and J. Kleinberg, “Direct uncertainty prediction for
medical second opinions,” in *Proc. ICML*, 2019, pp. 5281–5290.

E. Begoli, T. Bhattacharya, and D. Kusnezov, “The need for uncertainty
quantification in machine-assisted medical decision making,” *Nature
Mach. Intelligence*, vol. 1, no. 1, pp. 20–23, 2019.

C. Cortes, G. DeSalvo, and M. Mohri, “Learning with rejection,” in
*Proc. ALT*, 2016, pp. 67–82.

</div>

# Dataset Construction Details

## Claim Collection Protocol

Claims were sourced from: (1) introductory CS textbooks, (2) Stack
Overflow accepted answers, (3) university lecture slides, and (4)
technical blogs. Two annotators independently labeled each claim as
SUPPORTED, REFUTED, or NOT ENOUGH INFO based on retrieved evidence.
Inter-rater agreement: $`\kappa = 0.89`$.

## Evidence Corpus Curation

The evidence corpus contains 12,500 documents filtered for educational
relevance:

- Textbook excerpts: 4,200 documents

- Lecture notes: 3,100 documents

- Stack Overflow: 2,900 posts

- ArXiv preprints: 2,300 papers

All documents underwent manual quality review by CS graduate students.

# Hyperparameter Details

<div id="tab:hyperparams">

| **Parameter**                 |               **Value**                |
|:------------------------------|:--------------------------------------:|
| Temperature $`T`$             |                  1.24                  |
| Evidence retrieval $`k`$      |                   15                   |
| Relevance threshold           |                  0.65                  |
| Abstention threshold $`\tau`$ |                  0.90                  |
| Ensemble weights (learned)    | \[0.18, 0.35, 0.10, 0.15, 0.10, 0.12\] |
| Training batch size           |                   16                   |
| Inference batch size          |                   1                    |
| Learning rate                 |                  2e-5                  |
| Training epochs               |                   3                    |

Hyperparameter Configuration

</div>

**Operating Point Note**: Abstention threshold $`\tau^* = 0.90`$ is the
*recommended deployment operating point*. This threshold was optimized
on the validation set ($`n=261`$) to target $`\approx 90\%`$ selective
accuracy. When applied to the held-out test set ($`n=260`$), it yields
74.0% automated coverage
(Table <a href="#tab:selective_operating_points" data-reference-type="ref"
data-reference="tab:selective_operating_points">10</a>). Alternative
thresholds are configurable (e.g., $`\tau=0.75`$ for 85% selective
accuracy); deployment choice depends on priorities (automation vs.
correctness). See
Section <a href="#sec:selective_prediction" data-reference-type="ref"
data-reference="sec:selective_prediction">3.4</a> for justification.

# Extended Ablation Results

Table <a href="#tab:ablation_extended" data-reference-type="ref"
data-reference="tab:ablation_extended">15</a> presents sequential
component additions.

<div id="tab:ablation_extended">

| **Configuration**       | **Acc** | **ECE** | **AUC-AC** |
|:------------------------|:-------:|:-------:|:----------:|
| Semantic relevance only |  71.2%  | 0.2014  |   0.7654   |
| \+ Entailment strength  |  76.8%  | 0.1678  |   0.8123   |
| \+ Agreement signal     |  78.3%  | 0.1534  |   0.8345   |
| \+ Confidence margin    |  79.1%  | 0.1487  |   0.8501   |
| \+ Source authority     |  79.8%  | 0.1445  |   0.8623   |
| \+ Evidence diversity   |  80.4%  | 0.1398  |   0.8734   |
| \+ Temperature scaling  |         |         |            |

Sequential Component Additions

</div>

Each component provides incremental improvement. Temperature scaling
alone reduces ECE by 0.015.

# Re-Calibration Protocol

For deploying CalibraTeach in new domains:

1.  **Collect validation set**: Annotate 200+ domain-specific claims
    with expert labels

2.  **Optimize temperature**: Grid search $`T \in [0.5, 3.0]`$ to
    minimize validation NLL

3.  **Verify calibration**: Compute ECE on held-out test set; target ECE
    $`<0.15`$

4.  **Document performance**: Report accuracy degradation and
    calibration metrics

*Do not deploy without completing all four steps.*

# Calibration Robustness Across Metrics

To validate that calibration conclusions are robust to metric choice, we
computed multiple calibration metrics on the test set from calibrated
probabilities after temperature scaling. The standard Expected
Calibration Error (ECE) with 10 equal-width bins, reported in the main
results, is complemented by alternative metrics sensitive to different
aspects of calibration quality.

## Additional Calibration Metrics

**Brier Score** measures mean-squared error between predicted
probabilities and binary labels:
$`BS = \frac{1}{n} \sum_i (p_i - y_i)^2`$. Lower is better (range \[0,
1\]).

**Negative Log-Likelihood (NLL)** or Cross-Entropy, is
$`NLL = -\frac{1}{n} \sum_i [y_i \log p_i + (1-y_i) \log(1-p_i)]`$
(using natural logarithms), also lower is better. NLL is more sensitive
to extreme mis-calibration (predicted probability near 0 or 1 when the
opposite label occurs).

**Adaptive ECE (equal-mass binning)** partitions predictions into bins
of equal sample count (equal percentiles) rather than equal confidence
ranges. This is more robust to skewed confidence distributions.

**ECE with Multiple Bin Counts** (using 10, 15, 20 bins) assesses
sensitivity to binning choice. The standard choice of 10 bins is
somewhat arbitrary; robustness across bin counts strengthens confidence
in conclusions.

For CalibraTeach on the 260-claim test set:

<div id="tab:calib_robustness">

| **Metric**                   |        **Value**         |
|:-----------------------------|:------------------------:|
| Brier Score                  |          0.1524          |
| Negative Log-Likelihood      |          0.5000          |
| ECE (10 equal-width bins)    | 0.1076 *\[main result\]* |
| ECE (15 equal-width bins)    |          0.1068          |
| ECE (20 equal-width bins)    |          0.1065          |
| ECE (10 equal-mass/adaptive) |          0.1109          |

Calibration Robustness: Multiple Metrics

</div>

Values are reported to 4 decimals; exact floating-point metrics are
available in the released artifacts.

The consistency of ECE across bin choices (0.1065–0.1109) and the
agreement with complementary metrics (Brier, NLL) confirm that
temperature scaling achieves stable calibration. The conclusion that
CalibraTeach is well-calibrated does not depend on the specific ECE
definition or bin count. Ablated ensemble (Ensemble-NoCalib, ECE 0.1689)
shows larger variance across metrics, further validating the calibration
benefit of temperature scaling.

# Abstention Threshold Stability and Sensitivity

The selective prediction mechanism’s reliability depends on stability of
the abstention threshold $`\tau`$ across evaluation scenarios. We assess
robustness via bootstrap resampling and sensitivity analysis.

## Threshold Selection Protocol

Per the methodology in
Section <a href="#sec:selective_prediction" data-reference-type="ref"
data-reference="sec:selective_prediction">3.4</a>:

1.  $`\tau`$ is optimized *only* on the validation set (261 examples,
    stratified by domain and label).

2.  The selected value $`\tau^* = 0.90`$ is then applied to the test set
    (260 examples) without modification.

3.  This “train-on-validation, test-on-test” design prevents overfitting
    $`\tau`$ to the test set.

Cross-validation of $`\tau`$ during training would further strengthen
robustness; for this single-run evaluation, held-out validation is our
calibration safeguard.

## Bootstrap Stability

**Analysis-only note**: The official operating threshold $`\tau^*=0.90`$
is selected on the validation set and then applied to the test set
unchanged. The bootstrap procedure below is used only to characterize
threshold sensitivity; it is not used to retune reported results or
deployment behavior.

We resample the test set with replacement 100 times and, for sensitivity
characterization only, re-select $`\tau`$ on each bootstrap sample to
target 90% selective accuracy. Mean $`\tau`$ across resamples:
$`0.755 \pm 0.107`$. The range \[0.57, 0.94\] reflects sampling
variability. At the official operating point (74% coverage at 90%
selective accuracy), this variability corresponds to coverage swings of
$`\pm`$<!-- -->10–15 percentage points in expectation—acceptable for a
classroom deployment where selective accuracy is prioritized.

## Sensitivity to Perturbations

Selective accuracy remains in the range 85.5%–90.8% when $`\tau`$ is
perturbed by ±0.10 around the nominal value. Coverage is more sensitive
($`\pm`$<!-- -->25–80%), as expected: small confidence threshold changes
dramatically alter abstention rates. This sensitivity is by design:
threshold tuning offers intuitive user control over the
coverage–accuracy trade-off.

# Statistical Significance and Class Balance

The CSClaimBench test set contains 260 claims: 130 SUPPORTED, 130
REFUTED (binary evaluation). Classes are balanced (1.0$`\times`$
imbalance ratio), eliminating class-imbalance bias in aggregate accuracy
metrics.

## Per-Class Calibration and Performance

Per-class F1 scores are consistent: SUPPORTED 80.3%, REFUTED 81.2%
(macro-F1: 80.74%, reported in
Table <a href="#tab:main_results" data-reference-type="ref"
data-reference="tab:main_results">2</a>). However, per-class ECE reveals
calibration differences:

Per-class ECE is computed in a one-vs-rest manner. For class $`y`$, we
use $`p(y|\mathbf{x})`$ as the confidence. We compute bin-based
calibration error for that class-specific confidence distribution. These
values can be larger than aggregate ECE because aggregate binary ECE
uses $`\max(p, 1-p)`$ and pools both classes, allowing complementary
effects to partially cancel. We report per-class ECE as a diagnostic
class-level breakdown; it does not contradict the aggregate ECE reported
in Table <a href="#tab:main_results" data-reference-type="ref"
data-reference="tab:main_results">2</a>.

- **Class 0 (REFUTED)**: Per-class ECE = 0.1095 (130 samples)

- **Class 1 (SUPPORTED)**: Per-class ECE = 0.0876 (130 samples)

Per-class ECE is higher than aggregate (0.0876 for SUPPORTED, 0.1095 for
REFUTED, vs. 0.1076 binary ECE), expected because per-class calculation
excludes the complementary class. This slight elevation is expected;
both per-class values are well-calibrated. Balanced accuracy (0.8077)
matches standard accuracy, confirming no systematic class-specific
performance degradation.

## Confidence Interval Width and Practical Significance

The 95% bootstrap CI for accuracy is \[75.38%, 85.77%\], with width
$`\pm 5.2`$pp. This width reflects the small sample size ($`n=260`$).
Differences between systems exceeding $`\pm 5.2`$pp are unlikely due to
random variation alone; smaller differences may be due to sampling.
Observed differences in
Table <a href="#tab:baselines" data-reference-type="ref"
data-reference="tab:baselines">4</a> (CalibraTeach 80.77% vs.
Ensemble-NoCalib 80.2%) are within the CI width, motivating focus on
calibration as the primary differentiator.

# Authority Weights: Heuristic Priors and Sensitivity

The source authority component
(Eq. <a href="#eq:auth_score" data-reference-type="eqref"
data-reference="eq:auth_score">[eq:auth_score]</a>,
Section <a href="#sec:multi_component_ensemble" data-reference-type="ref"
data-reference="sec:multi_component_ensemble">3.2</a>) assigns
confidence scores to evidence sources:

- Peer-reviewed publications: 1.0

- Textbooks: 0.9

- Official documentation: 0.8

- Lecture notes: 0.7

- Stack Overflow: 0.6

- Blogs: 0.4

These weights are transparent heuristic priors informed by common source
curation practices; authority does not guarantee correctness. Even
peer-reviewed sources contain errors, and community sources like Stack
Overflow often provide high-quality domain knowledge.

## Ablation and Sensitivity

Table <a href="#tab:auth_sensitivity" data-reference-type="ref"
data-reference="tab:auth_sensitivity">9</a> (in main results) confirms
robustness: removing the source authority component incurs 0.57pp
accuracy loss and 0.0033 ECE increase. Perturbing weights by ±10% yields
$`<0.8`$pp accuracy change, supporting the stability of
authority-weighted ensembles to specification errors. A learned
authority weighting (e.g., logistic regression on validation evidence)
is a promising extension; for this work, heuristic specification is
transparent and computationally lightweight.

[^1]: N. B. Patel is a Software Engineer at Verizon and an IEEE Member.
    S. K. K. Nellipudi is a Senior Technology Leader at Incomm Payments
    and a Senior IEEE Member. Both contributed equally to this work and
    conducted this research while affiliated with Kennesaw State
    University.

[^2]: S. He is an Assistant Professor of Computer Science and Founding
    Director of the Computer Science Education Technology Lab at
    Kennesaw State University, Kennesaw, GA 30144 USA (e-mail:
    she4@kennesaw.edu).
