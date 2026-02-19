# Domain-Specific Deployment Guidelines

**Quick Reference**: Checklist and instructions for deploying SmartNotes to new domains.

---

## Quick Decision Tree

```
"I want to use SmartNotes for [NEW DOMAIN]"

1. Is it Computer Science education?
   └─ YES → Use existing system (94.2% accuracy)
   └─ NO → Go to question 2

2. Is it similar to CS (Physics, Math, Engineering)?
   └─ YES → Go to "Similar Domain" (expect 88-92%)
   └─ NO → Go to question 3

3. Is it professional (Medicine, Law, Finance)?
   └─ YES → Go to "Different Domain" (expect 70-80%)
   └─ NO → Go to question 4

4. Is it news, social media, or user-generated content?
   └─ YES → See "Adversarial Domain" (expect <60%, not ready)
   └─ NO → Contact support; unknown territory
```

---

## Similar Domain (Physics, Math, Engineering)

### Timeline: 1-2 weeks | Cost: Low ($100-200)

### Step 1: Gather Data (3-5 days)
```
Target: 50-100 claims with expert verification

Sources:
  - Textbook problems (calculus, linear algebra, physics)
  - University exams (with answer keys)
  - Online courses (MIT OCW, Coursera with verified tracks)
  
Quality criteria:
  ✓ Ground truth from expert (professor or advanced student)
  ✓ Claims are clear and unambiguous
  ✓ Evidence is available (textbooks, papers)
  ✓ Claims cover multiple subdomains (doesn't matter if scattered)

File format: data/new_domain_claims.jsonl (one JSON per line)
Example:
  {
    "doc_id": "phys_001",
    "domain_topic": "physics.mechanics",
    "source_text": "A ball dropped from 10m takes 1.4s to reach ground",
    "generated_claim": "Free fall from 10m takes 1.4 seconds on Earth",
    "gold_label": "VERIFIED",
    "evidence_span": "Using h = 0.5*g*t², t = sqrt(2*h/g) ≈ 1.43s"
  }
```

### Step 2: Fine-Tune Weights (2-3 hours)
```bash
# Run weight optimization on your domain
python scripts/reproduce_weights.py \
    --dataset data/new_domain_claims.jsonl \
    --output-path models/weights_new_domain.json \
    --cv-folds 5 \
    --random-seed 42

# Result: models/weights_new_domain.json (717 bytes)
# Expected accuracy: Similar to source (88-92%)
```

### Step 3: Evaluate (1-2 hours)
```bash
# Comprehensive validation
python evaluation/real_world_validation.py \
    --dataset data/new_domain_claims.jsonl \
    --output-dir evaluation/

# Check results
cat evaluation/cross_validation_results.json
# Expected: mean accuracy 88-92% across 5 folds
```

### Step 4: Document Results (1-2 days)
Create: `evaluation/NEW_DOMAIN_RESULTS.md`

Sections:
- Dataset summary (n=X claims, subdomains covered)
- Accuracy results (mean ± std from 5-fold CV)
- Per-domain breakdown (if applicable)
- Comparison to CS results (88-92% vs 94.2% baseline)
- Recommendations for further improvement

### Sign-Off Criteria: ✅ READY FOR PRODUCTION
- [ ] Accuracy ≥ 85% (or within 10pp of CS baseline)
- [ ] Confidence interval computed
- [ ] Cross-validation stable across folds
- [ ] Results documented with evidence links

---

## Different Domain (Medicine, Law, Finance)

### Timeline: 3-4 weeks | Cost: Medium ($1-2K)

### Step 1: Gather Data (1 week)
```
Target: 100-200 claims with expert verification

Key difference from Similar Domain:
  - Larger sample (100-200 vs 50-100)
  - More rigorous verification (use multiple experts)
  - Diverse subcategories (if domain has subtypes)

Example (Medicine):
  - Cardiology: 40 claims
  - Oncology: 40 claims
  - Psychiatry: 40 claims
  - General medicine: 40 claims
  Total: 160 claims

Quality: Each claim verified by:
  - Primary care physician (or specialist)
  - Medical literature search (PubMed)
  - Current clinical guidelines
```

### Step 2: Fine-Tune Component Weights (1-2 weeks)
```bash
# Same as Similar Domain, but repeated iterations

python scripts/reproduce_weights.py \
    --dataset data/new_domain_claims.jsonl \
    --output-path models/weights_new_domain_v1.json \
    --cv-folds 5

# Check initial results
# Expected: 70-80% accuracy (degradation expected)
```

### Step 3: Optional - Fine-Tune NLI Model (1-2 weeks)
```bash
# Only needed if accuracy < 75%

# Note: This script is a reference; full implementation needed
python src/evaluation/fine_tune_nli_for_domain.py \
    --train-data data/new_domain_train.jsonl \
    --val-data data/new_domain_val.jsonl \
    --model-name roberta-base-mnli \
    --output-model models/roberta_mnli_new_domain.pt \
    --epochs 3 \
    --batch-size 32 \
    --learning-rate 2e-5

# Re-optimize weights with NLI-fine-tuned model
python scripts/reproduce_weights.py \
    --dataset data/new_domain_claims.jsonl \
    --nli-model models/roberta_mnli_new_domain.pt \
    --output-path models/weights_new_domain_v2.json
```

### Step 4: Evaluate (1-2 days)
```bash
# Full evaluation pipeline
python evaluation/real_world_validation.py \
    --dataset data/new_domain_claims.jsonl \
    --weights models/weights_new_domain_v2.json

# Ablation analysis (which components matter?)
python evaluation/error_analysis_by_domain.py \
    --dataset data/new_domain_claims.jsonl \
    --output evaluation/error_analysis_new_domain.json
```

### Step 5: Document Results (1-2 days)
Create: `evaluation/new_domain_RESULTS.md`

Include:
- Dataset composition (subdomains, verification process)
- Accuracy by subcategory (does it perform better on some types?)
- Component ablations (which signals help in new domain?)
- Failure analysis (where does system struggle?)
- Comparison to CS baseline
- Resource cost (GPU hours, data collection effort)

### Sign-Off Criteria: ✅ READY FOR PILOTING
- [ ] Accuracy ≥ 70% (reduced threshold due to domain difficulty)
- [ ] Confidence interval computed
- [ ] Ablations show interpretable component contributions
- [ ] Failure modes documented
- [ ] Cost-benefit analyzed (is 75% useful for this domain?)

---

## Adversarial Domain (News, Social Media)

### Status: NOT RECOMMENDED YET ❌

**Why**:
- Projected accuracy 40-60% (worse than random/baseline)
- Requires architectural changes (sarcasm detection, credibility scoring)
- No clear deployment path without significant engineering

### If You Must Deploy to News/Social Media:

**Recommended Instead**: Use specialized fact-checking systems:
- FEVER (Wikipedia-based)
- ClaimBuster (news claim detection)
- Fact-checking APIs (Google Fact Check API, etc.)

**If Building Custom Solution** (~2-3 month timeline):

1. Add adversarial robustness training
2. Implement credibility scoring for low-quality sources
3. Add temporal dynamics (claims change over time)
4. Integrate sarcasm/metaphor detection
5. Build misinformation dataset (10K+ adversarial examples)

**Expected investment**: $10-20K in research + engineering

---

## Choosing Your Next Domain

| Domain | Difficulty | Timeline | Cost | Recommendation |
|--------|-----------|----------|------|-----------------|
| Physics Education | Easy | 1-2 weeks | Low | ✅ START HERE |
| Mathematics Ed | Easy | 1-2 weeks | Low | ✅ TRY NEXT |
| Engineering Ed | Easy | 2 weeks | Low | ✅ TRY NEXT |
| Medicine | Hard | 3-4 weeks | Medium | ⚠️ PLAN AHEAD |
| Law | Hard | 3-4 weeks | Medium | ⚠️ PLAN AHEAD |
| Finance | Hard | 3-4 weeks | Medium | ⚠️ PLAN AHEAD |
| News | Very Hard | 2-3 months | High | ❌ USE DIFFERENT SYSTEM |
| Social Media | Very Hard | 2-3 months | High | ❌ USE DIFFERENT SYSTEM |

**Recommendation**: Start with Physics, then Medicine, then evaluate news at end.

---

## Troubleshooting Common Issues

### Issue: Accuracy Drops > 10pp from CS Baseline
**Possible Causes**:
1. Different terminology (NLI trained on general English)
2. Different evidence source quality (lower quality than textbooks)
3. Domain-specific reasoning NLI lacks

**Solutions** (in order):
1. Collect more training data (100→200 claims)
2. Fine-tune NLI model specifically
3. Add domain-specific preprocessing (extract technical terms)
4. Consult with domain expert on unusual errors

### Issue: Accuracy Fluctuates Widely Across Folds
**Possible Causes**:
1. Small dataset (100 claims not enough for stable k-fold)
2. Domain has natural variance (medicine has probabilistic claims)
3. Imbalanced label distribution (too many VERIFIED, few REJECTED)

**Solutions**:
1. Collect more data (move to 200+ claims)
2. Use stratified k-fold (already done by default)
3. Over-sample minority class or weight by class

### Issue: Component Weights Very Different from CS
**Possible Causes**:
1. Domain genuinely different (entailment more/less important)
2. Evidence sources different (credibility signals differ)
3. Not enough data to stabilize optimization

**Solutions**:
1. Collect more data (200+ claims minimum)
2. Compare component contributions via ablations
3. Consult domain expert on whether new weights make sense

---

## Checklist: Before Going Live

### Data Quality
- [ ] Claims from authoritative sources
- [ ] Multiple expert verification
- [ ] Unambiguous ground truth
- [ ] Diverse coverage of domain subtypes
- [ ] ≥100 claims for evaluation

### Model Performance
- [ ] Accuracy ≥ 85% (or justified threshold for your domain)
- [ ] Confidence intervals computed and documented
- [ ] Cross-validation stable (std < mean × 0.15)
- [ ] Ablations show interpretable component contributions
- [ ] Error analysis shows failures are acceptable

### Documentation
- [ ] Results documented in evaluation/DOMAIN_RESULTS.md
- [ ] Weights saved in models/weights_DOMAIN.json
- [ ] Evidence links provided
- [ ] Limitations clearly stated
- [ ] Reproducibility steps documented

### Deployment
- [ ] System tested on held-out data
- [ ] Integration with application tested
- [ ] Error handling implemented
- [ ] User expectations set (accuracy ± confidence interval)
- [ ] Human review process in place (optional)

---

## Support: When to Contact

**If you get stuck** on any step:

1. **Data collection**: Contact domain expert network
2. **Weight optimization**: Check scripts/reproduce_weights.py README
3. **Accuracy too low**: Try Fine-tune NLI (Step 3 above)
4. **Deployment questions**: See RESEARCH_INTEGRITY_COMPLETION_SUMMARY.md

---

## Success Stories

After deploying to new domain, document in:
`evaluation/CROSS_DOMAIN_SUCCESS_new_domain.md`

Include:
- What domain?
- How long did it take?
- What was the final accuracy?
- What surprised you?
- Any tips for others?

Share back to improve guidelines for future domains!

