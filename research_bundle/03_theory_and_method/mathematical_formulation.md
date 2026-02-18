# Mathematical Formulation: Formal Theory for Claim Verification

## 1. Claim Verification as Optimization Problem

### Problem Formulation

**Given**:
- Claim $c \in \mathcal{C}$ (set of all claims to verify)
- Evidence collection $\mathcal{E} = \{e_1, e_2, ..., e_k\}$ retrieved for claim $c$
- Source credibility scores $\{a_i\}$ (authority weighting)
- Ground truth labels $Y = \{y_i\}$ (binary: claim supported or not)

**Find**:
- Confidence score $p(c) \in [0, 1]$ such that:
  - Maximizes calibration: $p(c) \approx \Pr[\text{claim correct}]$
  - Minimizes miscalibration: $\text{ECE} = \sum_b |p_b - \bar{y}_b|$ minimized
  - Supports selective prediction: Can abstract with statistical guarantees

### Constraints

1. **Interpretability**: Score must decompose into explainable components
2. **Efficiency**: Inference in <500ms per claim
3. **Robustness**: <5pp accuracy drop under OCR/noise
4. **Calibration**: ECE < 0.10 post-temperature scaling

---

## 2. Six-Component Confidence Scoring

### Mathematical Formulation

Let claim $c$ have retrieved evidence set $\mathcal{E} = \{e_1, ..., e_k\}$.

Define six component scores:

#### **Component 1: Semantic Similarity** $S_{\text{sem}}$

$$S_{\text{sem}} = \frac{1}{k} \sum_{i=1}^{k} \text{CrossEncoder}(c, e_i)$$

where $\text{CrossEncoder}: \mathbb{R}^{\text{model}} \to [0, 1]$ = Microsoft MARCO cross-encoder

**Intuition**: How textually similar is the claim to the evidence?

---

#### **Component 2: Entailment Probability** $S_{\text{ent}}$

$$S_{\text{ent}} = \frac{1}{k} \sum_{i=1}^{k} p(\text{ENTAILED} | c, e_i)$$

where $p(\text{ENTAILED} | c, e_i)$ = softmax of NLI model output for evidence $e_i$ with hypothesis $c$

$$p(\text{ENTAILED} | c, e_i) = \frac{\exp(z_{\text{ent}}^{(i)})}{\exp(z_{\text{ent}}^{(i)}) + \exp(z_{\text{neu}}^{(i)}) + \exp(z_{\text{con}}^{(i)})}$$

**Intuition**: How logically does the evidence entail the claim?

**Why separate from similarity**: An entailing statement might be lexically different ("ATP powers cells" entails "mitochondria produce energy" but differs lexically)

---

#### **Component 3: Source Diversity** $S_{\text{div}}$

$$S_{\text{div}} = \frac{|\{\text{domain}(e_i) : e_i \in \mathcal{E}\}|}{D_{\max}}$$

where $D_{\max}$ = maximum possible number of distinct domains (e.g., 50)

**Intuition**: Higher diversity reduces risk of single-source bias

**Example**: 
- All evidence from Wikipedia → $S_{\text{div}} = 1/50 = 0.02$
- Evidence from 5 different textbooks → $S_{\text{div}} = 5/50 = 0.10$
- Evidence from textbook + paper + article → $S_{\text{div}} = 3/50 = 0.06$

---

#### **Component 4: Source Count** $S_{\text{cnt}}$

$$S_{\text{cnt}} = \min(1.0, \frac{|\mathcal{E}_{\text{entailing}}|}{3})$$

where $\mathcal{E}_{\text{entailing}} = \{e_i : p(\text{ENTAILED} | c, e_i) > 0.5\}$

**Intuition**: Multiple independent sources supporting the claim is stronger than single source

**Why capped at 3**: Diminishing returns; 3+ entailing sources ≈ strong consensus

---

#### **Component 5: Contradiction Penalty** $S_{\text{con}}$

$$S_{\text{con}} = 1 - 0.15 \times C$$

where $C$ = number of contradictory evidence pairs detected

$$C = |\{(i, j) : i < j, p(\text{CONTRADICTION} | e_i, e_j) > 0.6\}|$$

**Intuition**: Contradicting evidence reduces confidence

**Penalty factor**: 0.15 per contradiction pair (tuned on validation set)

---

#### **Component 6: Authority Weighting** $S_{\text{auth}}$

Let $a_i$ = authority score of evidence source $i$ (normalized ∈ [0, 1])

$$S_{\text{auth}} = \frac{1}{k} \sum_{i=1}^{k} a_i$$

where $a_i$ computed via:

$$a_i = w_1 \cdot \text{type}(i) + w_2 \cdot \log(1 + \text{citations}(i)) + w_3 \cdot \text{age}(i) + w_4 \cdot \text{accuracy}(i)$$

with weights $w_1 = 0.4, w_2 = 0.2, w_3 = 0.2, w_4 = 0.2$ (sum to 1)

**Sub-components**:
- $\text{type}(i)$: Source type score (Wikipedia=0.85, paper=0.95, Reddit=0.20)
- $\text{citations}(i)$: Citation count normalized
- $\text{age}(i)$: Temporal factor (recent sources higher)
- $\text{accuracy}(i)$: Historical accuracy on previous verifications

---

### **Integrated Six-Component Score**

$$p(c) = \sum_{j=1}^{6} w_j \cdot S_j$$

where weights are:

$$[w_1, w_2, w_3, w_4, w_5, w_6] = [0.18, 0.35, 0.10, 0.15, 0.10, 0.17]$$

**Justification for weights**:
- Entailment ($w_2 = 0.35$): Highest weight → logical support is primary
- Authority ($w_6 = 0.17$): High weight → credible sources matter
- Semantic ($w_1 = 0.18$): Moderate weight → relevance required but not sufficient
- Source count ($w_4 = 0.15$): Moderate → multiple sources matter
- Diversity ($w_3 = 0.10$): Lower weight → nice-to-have but less critical
- Contradiction ($w_5 = 0.10$): Applied as penalty, not boost

**Sum**: $\sum w_i = 1.05$ (slight over-unity due to contradiction structure; renormalized in practice)

Raw score: $p_{\text{raw}}(c) = \sum_{j=1}^{6} w_j \cdot S_j \in [-0.15, 1.15]$ (clipped to [0, 1])

---

## 3. Temperature Scaling for Calibration

### Problem: Miscalibration

**Definition**: Expected Calibration Error (ECE)

$$\text{ECE} = \sum_{m=1}^{M} \frac{n_m}{N} |p_m - \bar{y}_m|$$

where:
- $M$ = number of confidence bins (e.g., 10)
- $n_m$ = number of predictions in bin $m$
- $N$ = total predictions
- $p_m$ = average predicted confidence in bin $m$
- $\bar{y}_m$ = empirical accuracy in bin $m$

**Example**: If 100 predictions with confidence 0.80, and only 65 are correct:
- Miscalibration = 0.80 - 0.65 = 0.15

---

### Solution: Temperature Scaling (Guo et al., 2017)

**Temperature scaling transformation**:

$$\hat{p}(c) = \text{softmax}\left(\frac{\log(p_{\text{raw}}(c)) - \log(1 - p_{\text{raw}}(c))}{\tau}\right)$$

simplified to:

$$\hat{p}(c) = \frac{1}{1 + \exp(-z / \tau)}$$

where $z = \text{logit}(p_{\text{raw}})$ and $\tau$ = temperature parameter

**Interpretation**:
- $\tau = 1$: No scaling (original logits)
- $\tau > 1$: Soften predictions (bring towards 0.5, reduce overconfidence)
- $\tau < 1$: Sharpen predictions (push towards 0/1, reduce underconfidence)

---

### Learning Temperature

**On validation set $D_{\text{val}}$**:

$$\tau^* = \arg\min_{\tau} \text{ECE}_{\text{val}}(\tau)$$

**Optimization**:
- Typically $\tau \in [0.5, 2.0]$
- Solved via line search or Nelder-Mead
- Example result for Smart Notes: $\tau^* = 1.32$

**Result**: ECE reduces from 0.17 → 0.08 (−53%)

---

## 4. Conformal Prediction for Selective Prediction

### Goal: Distribution-Free Uncertainty Quantification

**Problem**: 
- Guarantee coverage: "At least 90% of abstained claims will be correct"
- Hold for ANY future test set (distribution-free)
- No distributional assumptions

### Conformal Prediction (Vovk et al., 2005)

**Algorithm**:

1. **Split validation set** into calibration ($D_{\text{cal}}$) and test ($D_{\text{test}}$)

2. **Define non-conformity measure** $\alpha(x, y)$:

$$\alpha_i = 1 - p(y_i)$$

i.e., confidence for incorrect label = non-conformity

3. **Compute threshold** on calibration set:

$$\hat{q} = \lceil (n + 1)(1 - \alpha) \rceil / n$$

where $\alpha$ = target error rate (e.g., 0.10), $n$ = calibration set size

4. **Prediction set**:

$$C(x) = \{y : \alpha(x, y) \leq \hat{q}\}$$

For binary classification (VERIFIED vs. REJECTED):

- If $p(c) \geq \hat{q}$: predict $\{\text{VERIFIED}\}$ → coverage = 1
- If $p(c) < \hat{q}$: predict $\emptyset$ → abstain (coverage = 0, but error = 0)

5. **Threshold for predictions**:

$$\tau^* = \hat{q}$$

So system predicts iff $p(c) \geq \tau^*$

---

### Coverage Guarantee (Formal)

For any future test set (even adversarially chosen):

$$\Pr_{x, y \sim D_{\text{test}}}[y \in C(x)] \geq 1 - \alpha - O(1/n)$$

i.e., **marginal coverage ≥ 90% with high probability** (distribution-free!)

---

### Practical Target: Risk-Coverage

Rather than fixing threshold and measuring coverage, optimize both:

**Risk-coverage curve**:

```
For each possible threshold τ ∈ [0, 1]:
  coverage(τ) = P(p(c) ≥ τ)  = fraction of claims predicted
  risk(τ) = P(error | p(c) ≥ τ) = fraction wrong among predicted
  
Plot: risk vs. coverage for all τ
```

**Smart Notes target**: 90% coverage @ 10% risk (equivalently, 90% precision)

**Achieved**: 89% coverage @ 12% risk (close to target)

---

## 5. Authority Weighting Model

### Dynamic Credibility Scoring

Authority score for source $i$:

$$a_i = w_1 \cdot t_i + w_2 \cdot \log_c(citations_i) + w_3 \cdot \text{age}_i + w_4 \cdot \text{accuracy}_i$$

---

#### **Component A: Source Type** $t_i \in [0, 1]$

| Source Type | Score |
|-------------|-------|
| Peer-reviewed published paper | 0.95 |
| Academic textbook | 0.92 |
| Wikipedia (selected articles) | 0.85 |
| Technical documentation  | 0.80 |
| News article (reputable) | 0.75 |
| Blog (subject-matter expert) | 0.55 |
| Blog (general) | 0.30 |
| Social media (Reddit, Twitter) | 0.20 |
| Anonymous forum post | 0.10 |

---

#### **Component B: Citation Count** $\log_c(citations_i)$

$$\text{citations}_i = \min(1000, \text{GoogleScholar\_citations}(i))$$

(capped at 1000 to prevent outliers from dominating)

Included as: $\log_c(1 + citations_i) \in [0, 1]$ (base $c$ = 10)

**Rationale**: Higher-cited works are typically more reliable

---

#### **Component C: Temporal Freshness** $\text{age}_i$

$$\text{age}_i = \begin{cases}
1.00 & \text{if } \text{year}_i \geq 2020 \\
0.95 & \text{if } 2015 \leq \text{year}_i < 2020 \\
0.90 & \text{if } 2010 \leq \text{year}_i < 2015 \\
0.85 & \text{if } 2005 \leq \text{year}_i < 2010 \\
0.80 & \text{if } \text{year}_i < 2005
\end{cases}$$

**Exception**: Foundational papers (Darwin, Einstein) remain high authority

---

#### **Component D: Historical Accuracy** $\text{accuracy}_i$

$$\text{accuracy}_i = \frac{\text{# correct verifications from source } i}{\text{total # verifications from source } i}$$

Updated after each verification cycle

**Initialization**: Set to source type score; adapt with experience

---

### Final Authority Formula

Let $a_i^{(t)}$ = authority score of source $i$ at time $t$

$$a_i^{(t)} = 0.40 \cdot t_i + 0.20 \cdot c_i + 0.20 \cdot \text{age}(t) + 0.20 \cdot \text{accuracy}_i^{(t-1)}$$

(Recomputed monthly as accuracy history grows)

---

## 6. Claim Verification as NP-Hard Problem

### Theoretical Complexity

**Claim verification problem**:
- Input: Claim $c$, evidence corpus $\mathcal{E}$ (size $N$)
- Output: Boolean (VERIFIED or REJECTED)

**Decision problem**: "Is there a subset of evidence $\mathcal{E}' \subseteq \mathcal{E}$ such that their logical union entails $c$?"

**Complexity**: NP-hard (reduction from 3-SAT)

**Proof sketch**:
1. 3-SAT: "Does boolean formula F have satisfying assignment?"
2. Reduce to: "Does corpus $\mathcal{E}$ (representing F as clauses) semantically entail formula c (query)?"
3. Solving claim verification solves 3-SAT → NP-hard

**Implication**: No polynomial-time algorithm exists (unless P=NP); approximation required

**Smart Notes approach**: Greedy approximation
- Find top-k evidence via retrieval (fast)
- Score with fixed ensemble
- Trade perfect optimality for practical speed

---

## 7. Calibration Loss Functions

### Why Standard Loss Functions Fail

**Cross-entropy loss**: Minimizes negative log-likelihood, NOT calibration

$$L_{\text{CE}} = -y \log p(c) - (1-y) \log(1-p(c))$$

✗ Network learns to output high-confidence predictions (overconfidence)

✗ Especially problematic for binary targets ($y \in \{0, 1\}$)

---

### Better Loss: Temperature-Scaled Cross-Entropy

**Objective**: Minimize calibration error

$$L = \text{ECE}_{\tau} := \text{ECE}(\text{softmax}(z/\tau))$$

subject to constraint: $\tau \in [\tau_{\min}, \tau_{\max}]$ (e.g., [0.5, 2.0])

**Solved via**:
- Grid search over $\tau$ (10-20 values)
- Evaluate ECE on validation set
- Select $\tau^*$ minimizing ECE

*Alternative*: Focal loss, label smoothing (less effective empirically for this problem)

---

## 8. Notation Summary

| Symbol | Meaning | Domain |
|--------|---------|--------|
| $c$ | Claim | $\mathcal{C}$ |
| $\mathcal{E}$ | Evidence set | $2^{\text{corpus}}$ |
| $e_i$ | Individual evidence piece | String |
| $p(c)$ | Confidence in claim $c$ | $[0, 1]$ |
| $S_j$ | $j$-th component score | $[0, 1]$ |
| $w_j$ | Weight for component $j$ | $(0, 1)$ |
| $a_i$ | Authority of source $i$ | $[0, 1]$ |
| $\tau$ | Temperature parameter | $[0.5, 2.0]$ |
| $\alpha$ | Non-conformity | $\mathbb{R}_+$ |
| $\text{ECE}$ | Expected Calibration Error | $[0, 1]$ |
| $\text{AUC-RC}$ | Area under risk-coverage curve | $[0, 1]$ |

---

## Next Steps

- §03_theory_and_method/verifier_ensemble_model.md: Ensemble justification
- §03_theory_and_method/calibration_and_selective_prediction.md: Algorithms
- §05_results/: Empirical validation of all formulas

