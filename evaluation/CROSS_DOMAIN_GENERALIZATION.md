# Cross-Domain Generalization Analysis

## Executive Summary

SmartNotes is validated on **Computer Science educational claims only**. This document analyzes expected performance on other domains and provides guidance for deployment.

---

## Current Scope: Computer Science Domain

**Validated Performance**:
- Real-world accuracy: **94.2%** (14,322 CS educational claims)
- Confidence interval: 95% CI [93.8%, 94.6%]
- Domain expertise: CS professors (faculty verification)
- Evidence sources: Textbooks, lecture notes, published papers

**Key Characteristics of CS Domain**:
- Highly technical terminology (algorithms, data structures, complexity theory)
- Well-defined correct answers (objective verification possible)
- Abundant formal text evidence (textbooks, academic papers)
- Active research community (recent papers regularly available)
- Relatively homogeneous audience (university CS students)

---

## Expected Transfer Performance

### Transfer Principle (Ben-David et al., 2010)

Machine learning systems have limited generalization across domains. When deploying to new domains:

$$\text{Error}_{target} \approx \text{Error}_{source} + \text{DomainDistance}$$

where DomainDistance reflects differences in:
- Vocabulary and terminology
- Evidence availability
- Claim verification difficulty
- Source trustworthiness distribution

### Predicted Accuracy by Domain Type

| Domain | Relationship | Predicted Accuracy | Degradation | Confidence |
|--------|-------------|-------------------|-------------|-----------|
| **Computer Science** | Baseline | **94.2%** | — | HIGH ✓ |
| **Physics/Math** | Very Similar | 88-92% | -2 to -6pp | MEDIUM |
| **Chemistry/Biology** | Similar | 82-88% | -6 to -12pp | MEDIUM |
| **Medicine** | Different | 70-80% | -14 to -24pp | LOW |
| **Law** | Very Different | 70-80% | -14 to -24pp | LOW |
| **Finance** | Very Different | 65-75% | -19 to -29pp | LOW |
| **News** | Adversarial | 40-60% | -34 to -54pp | VERY LOW |
| **Social Media** | Adversarial | 30-50% | -44 to -64pp | VERY LOW |

---

## Domain Transfer Analysis

### ✅ Similar Domains (Physics, Mathematics, Engineering)

**Expected**: 88-92% accuracy (-2 to -6pp degradation)

**Why limited degradation**:
- Overlapping terminology (algorithms, complexity classes exist in math)
- Similar evidence sources (textbooks, academic papers)
- Objective verification (well-defined truths)
- Strong entailment signals (NLI should transfer well)

**Transfer procedure**:
1. Collect 50-100 labeled physics claims
2. Fine-tune on combined CS + Physics data
3. Test on separate physics validation set
4. Expected result: ~90% accuracy

**GPU cost**: 1-2 hours

---

### ⚠️ Different Domains (Medicine, Law, Finance)

**Expected**: 70-80% accuracy (-14 to -24pp degradation)

**Why significant degradation**:
- Different terminology (medical jargon vs CS jargon)
- Different evidence sources (medical abstracts, legal precedents)
- Subjective verification (multiple valid interpretations possible)
- Weak entailment signals (NLI trained on general text)
- Domain-specific reasoning (medical contraindications, legal interpretations)

**Example - Medicine**:
```
Claim: "Statins reduce cardiovascular risk by 30% in all patients"
Issue: Medical claims often probabilistic and patient-dependent
       NLI may fail to capture: "reduces risk in many but not all patients"
Expected accuracy drop: ~15-20pp
```

**Example - Law**:
```
Claim: "Under the First Amendment, all speech is protected"
Issue: Legal claims highly context-dependent and precedent-based
       NLI may fail to capture: "except narrow exceptions (fraud, incitement)"
Expected accuracy drop: ~10-20pp
```

**Transfer procedure**:
1. Collect 100-200 labeled medical claims (clinically verified)
2. Fine-tune component weights on medical data
3. Retrain NLI models on medical claim-evidence pairs
4. Test on separate validation set
5. Expected result: ~75-80% accuracy

**GPU cost**: 4-8 hours (requires NLI fine-tuning)

---

### ❌ Adversarial Domains (News, Social Media)

**Expected**: 40-60% accuracy (-34 to -54pp degradation)

**Why severe degradation**:
- Adversarial language (deliberate misinformation, ambiguity)
- Low-quality evidence (personal opinions, unverified claims)
- Temporal dynamics (claims become outdated quickly)
- Social dynamics (claims spread despite being false)
- Weak entailment signals amplified (NLI fails on sarcasm, metaphor)

**Example - News Misinformation**:
```
Claim: "Vaccines cause autism"
Issue: False claim widely repeated; some weak "evidence" exists
       Entailment signals weak (evidence is low-quality)
       System likely confused by volume of supporting "evidence"
Expected accuracy: ~50% (no better than random chance)
```

**Example - Social Media**:
```
Claim: "This celebrity said X"
Issue: Often misquoted; out-of-context; sarcasm not flagged
       NLI cannot distinguish sarcasm from literal truth
       Evidence often contradictory and unreliable
Expected accuracy: ~40% (worse than random)
```

**Transfer procedure**: 
Not recommended without significant architectural changes. Would require:
1. Sarcasm/metaphor detection layer
2. Source credibility scoring (Twitter account reputation, etc.)
3. Temporal dynamics modeling
4. Adversarial robustness training

**GPU cost**: Weeks of development

---

## Component-Level Transfer Analysis

### Which Components Transfer Well?

| Component | Transfer Quality | Notes |
|-----------|-----------------|-------|
| **S1: Semantic Similarity** | ✅ Good | Embeddings fairly domain-independent |
| **S2: Entailment (NLI)** | ⚠️ Mixed | Transfers but drops in ambiguous domains |
| **S3: Diversity** | ✅ Good | Works across domains |
| **S4: Consensus** | ✅ Good | Universal principle (agreement = credible) |
| **S5: Contradiction** | ⚠️ Mixed | Fails on sarcasm, metaphor (news/social) |
| **S6: Authority** | ❌ Poor | Source credibility is domain-specific |

**Implication**: Component weights will likely need adjustment per domain.

---

## Recommended Cross-Domain Deployment Strategy

### Phase 1: Validation on Similar Domain (Physics)
**Timeline**: 1-2 weeks  
**Effort**: Collect ~100 claims + fine-tune  
**Expected**: 88-92% accuracy  
**Purpose**: Validate transfer learning works; build confidence

### Phase 2: Validation on Different Domain (Medicine)  
**Timeline**: 3-4 weeks  
**Effort**: Collect ~200 claims + fine-tune + NLI adaptation  
**Expected**: 75-80% accuracy  
**Purpose**: Understand multi-domain architecture needs

### Phase 3: Decide on Adversarial Domains  
**Timeline**: Evaluation only (decide whether to pursue)  
**Effort**: Depends on architectural changes needed  
**Decision point**: Is system worth maintaining for news/social media?

---

## Deployment Recommendations by Domain

### Physics/Math Education ✅ RECOMMENDED NOW
```
Collect: 50-100 labeled examples
Timeline: 1-2 weeks  
Expected: 88-92% accuracy
Process:
  1. Gather physics claims from textbooks
  2. Get expert verification (physics professor)
  3. Fine-tune using scripts/reproduce_weights.py
  4. Validate on held-out set
Cost: Low ($100-200 GPU)
```

### Medicine/Law ⚠️ FEASIBLE, PLAN AHEAD
```
Collect: 100-200 labeled examples
Timeline: 3-4 weeks
Expected: 75-80% accuracy
Process:
  1. Gather medical/legal claims from authoritative sources
  2. Get expert verification (doctor/lawyer)
  3. Fine-tune NLI models on domain data
  4. Re-optimize component weights
  5. Validate on held-out set
Cost: Moderate ($1-2K GPU)
```

### News/Social Media ❌ NOT RECOMMENDED NOW
```
Timeline: 1-2 months (minimum)
Expected: 40-60% accuracy (worse than human baseline)
Process:
  1. Implement adversarial robustness measures
  2. Add credibility scoring layer
  3. Retrain on news-specific claim-evidence pairs
  4. Address sarcasm/metaphor detection
Cost: High ($5K+ GPU + engineering time)
Recommendation: Wait for Specialized Architecture
  → Consider FEVER or other news-specific systems instead
```

---

## Technical Path for New Domains

### Step 1: Collect Domain-Specific Data
```bash
# Gather ~100-200 claims in target domain
# Requirements:
#  - Objective ground truth (expert verification)
#  - Aligned with deployment use case
#  - Similar audience/education level to source
```

### Step 2: Fine-Tune Component Weights
```bash
# Use reproducible weight optimization
python scripts/reproduce_weights.py \
    --dataset data/new_domain_claims_labeled.jsonl \
    --cv-folds 5 \
    --output-path models/weights_new_domain.json
```

### Step 3: Optional: Fine-Tune NLI Models
```bash
# Only needed for significantly different domains
python src/evaluation/fine_tune_nli_for_domain.py \
    --train-data data/new_domain_train.jsonl \
    --val-data data/new_domain_val.jsonl \
    --output-model models/roberta_mnli_new_domain.pt
```

### Step 4: Evaluate and Validate
```bash
# Comprehensive evaluation
python evaluation/real_world_validation.py \
    --dataset data/new_domain_test.jsonl \
    --weights models/weights_new_domain.json \
    --output evaluation/results_new_domain.json
```

### Step 5: Report Results
Create: `evaluation/CROSS_DOMAIN_RESULTS_new_domain.md`
Document:
- Accuracy achieved
- Confidence interval
- Per-component contribution (ablations)
- Failure analysis
- Recommendations for further improvement

---

## Expected Challenges by Domain

### Physics Field
**Challenges**:
- Overlapping terminology can be misleading (velocity in CS ≠ physics)
- Mathematical notation differences
- Experimental vs theoretical claims

**Mitigation**: Collect diverse physics subdomains (mechanics, thermodynamics, optics)

### Medicine Field
**Challenges**:
- Individual variation (drug interactions patient-specific)
- Temporal evolution (treatment guidelines change)
- Probabilistic evidence (rare side effects)

**Mitigation**: Focus on well-established medical facts, not bleeding-edge research

### Law Field  
**Challenges**:
- Jurisdiction variation (law differs by country/state)
- Precedent interpretation (same law, different courts)
- Adversarial language (precise wording matters enormously)

**Mitigation**: Focus on single jurisdiction, collect guidance documents

### News Field
**Challenges**:
- Active misinformation (false claims widely supported)
- Temporal shifts (old claims become irrelevant)
- Out-of-context quotes

**Mitigation**: Focus on fact-checkable claims, use date information, integrate credibility scores

---

## Performance Expectations: Research vs Production

| Scenario | Expected Accuracy | Confidence | Use-Case |
|----------|------------------|------------|----------|
| CS education (current) | 94.2% | HIGH | ✅ Production ready |
| Physics education (projected) | 88-92% | MEDIUM | ⚠️ Needs validation |
| Medicine (projected) | 75-80% | LOW | ❌ Requires study |
| Law (projected) | 70-75% | LOW | ❌ Requires study |
| News (projected) | 40-60% | VERY LOW | ❌ Not recommended |

---

## Honest Limitations

1. **No Cross-Domain Validation Yet**: Projections based on transfer learning literature, not empirical measurement
2. **Component Weights Are Domain-Specific**: Optimal weights for Physics may differ from CS  
3. **NLI Models Have Limitations**: RoBERTa-MNLI trained on general text; may fail on specialized vocabularies
4. **Authority Scoring Is Brittle**: Source credibility heuristics may not work in new domains
5. **Evidence Quality Varies**: News has lower-quality evidence than academic sources

---

## Future Work

### Phase 1 (Next Quarter)
- [ ] Validate on Physics domain (50-100 claims)
- [ ] Document transfer learning empirically
- [ ] Publish cross-domain results

### Phase 2 (Next Year)  
- [ ] Validate on Medicine (100-200 claims)
- [ ] Fine-tune NLI on medical claim-evidence pairs
- [ ] Analyze component weight changes

### Phase 3 (Year 2+)
- [ ] Decide on news/social media investment
- [ ] Explore domain-adaptive architectures
- [ ] Build domain-specific variant systems

---

## Conclusion

SmartNotes achieves excellent performance on Computer Science educational claims (94.2%) but is **not validated** on other domains. Deployment to new domains requires:

1. Collection of 50-200 representative labeled examples
2. Fine-tuning of component weights (always)
3. Optional NLI model fine-tuning (for very different domains)
4. Empirical validation on held-out test set

Predicted performance ranges from **88-92% on similar domains** (physics) to **40-60% on adversarial domains** (social media). Choose new domains strategically based on deployment use case and available resources.

For production deployment to new domains, recommend starting with physics/mathematics (similar to CS) before attempting medicine/law/news.

