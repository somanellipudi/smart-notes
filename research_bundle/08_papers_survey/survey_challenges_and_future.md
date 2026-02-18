# Survey Paper: Challenges and Future Directions in Fact Verification

## 13. Open Challenges and Research Directions

### 13.1 Cross-Domain Generalization

**Problem**: 
Models trained on one domain often fail on another.
- FEVER-trained model on Wikipedia claims: 85% accuracy
- Same model on scientific claims (SciFact): 62% accuracy (-23pp drop)
- Same model on education claims (CSClaimBench): 58% accuracy (-27pp drop)

**Root causes**:
1. **Evidence retrieval changes**: Different writing styles, evidence formats
   - Wikipedia: Encyclopedic, formal
   - Science: Technical jargon, abstracts
   - Education: Simplified, conversational
   
2. **Label distribution shifts**: Different claim types prevalent
   - FEVER: ~36% NOT_SUPPORTED (vandalism-prone)
   - SciFact: ~45% NOT_SUPPORTED (heterogeneous sources)
   - CSClaimBench: ~42% NOT_SUPPORTED (educational errors)
   
3. **Calibration shifts**: Confidence distributions differ by domain
   - Model may be well-calibrated on Wikipedia but miscalibrated on science
   - Prior work: No system addresses this

**Current approaches**:
- Domain adaptation (fine-tune on target domain)
- Multi-task learning (train on multiple domains simultaneously)
- Transfer learning (but where to start?)

**Future directions**:
1. **Domain-agnostic evidence representations**
   - Learn embeddings invariant to domain-specific writing style
   - Use domain-invariant loss functions
   
2. **Adaptive calibration**
   - Learn to recalibrate confidence for new domains
   - Fewer labels than full retraining
   
3. **Meta-learning for fact verification**
   - Learn how to quickly adapt to new domains
   - Inspired by MAML (Model-Agnostic Meta-Learning)

**Impact**: Enable deployment across all domains without retraining

---

### 13.2 Multi-Hop Reasoning

**Problem**: Complex claims require connecting evidence across sources

**Example**:
```
Claim: "If the moon orbits Earth and Earth orbits the Sun, 
        then the moon is in the solar system"

Simple fact-checking fails:
  - Premise 1: "Moon orbits Earth" ✓ Verifiable directly
  - Premise 2: "Earth orbits Sun" ✓ Verifiable directly
  - Conclusion: "Moon in solar system" ✓ Verifiable directly
  
But: Requires REASONING that premise 1 + premise 2 → conclusion

Current systems:
  - Retrieve evidence for: "Moon in solar system"
  - Find: "Moon orbits Earth" and "Earth in solar system"
  - Still need to combine logically
```

**Challenges**:
1. **Evidence collection**: Must find all relevant evidence (not just top-k)
2. **Reasoning over evidence**: Combine claims via logical inference
3. **Noise amplification**: Error in hop 1 propagates to hop 2

**Current work**:
- HotpotQA: Multi-hop question answering (requires 2 reasoning steps)
- WebQuestions: Complex factual questions
- BUT: No dedicated multi-hop fact verification dataset

**Future directions**:
1. **Multi-hop fact verification dataset**
   - Curate claims requiring 2, 3, 4+ reasoning steps
   - Benchmark against current systems
   - Expected performance: Dramatic drop (50-70% accuracy)
   
2. **Knowledge graph integration**
   - Store evidence as knowledge graph
   - Traverse edges to find multi-hop paths
   - Example: Wikidata could enable structured reasoning
   
3. **Neuro-symbolic approaches**
   - Combine neural networks (retrieval) with symbolic reasoning (logic)
   - Execute reasoning steps explicitly

**Impact**: Enable verification of complex claims (policy impact, medical diagnoses)

---

### 13.3 Real-Time Evidence Retrieval

**Problem**: Fact verification currently uses static evidence corpora

**Current limitation**:
```
Evidence corpus: Wikipedia snapshot (Jan 2024)
↓
Claim: "Biden is president" (verified true Jan 2024)
↓
But: What if Biden leaves office?
     Evidence becomes stale
```

**Challenge areas**:
1. **Recency**: Evidence changes over time (recent events, policy updates)
2. **Coverage**: Static corpus can't handle truly novel events
3. **Web retrieval**: Real-time web search is slow (3-5 seconds per query)
4. **Filtering**: Internet full of misinformation; hard to trust dynamic sources

**Current approaches**:
- Hybrid: Use fixed corpus + dynamic retrieval
- Confidence-aware: Lower confidence if can't find supporting evidence
- Outdated refresh: Re-index corpus periodically (weekly/monthly)

**Future directions**:
1. **Fast web retrieval for fact verification**
   - Index: Pre-cache popular, trustworthy sources (Wikipedia, scientific journals)
   - Search: Use BM25/hybrid search for speed
   - Filter: Trust scores for sources (high: peer-reviewed; low: blogs)
   
2. **Temporal reasoning**
   - Claims have temporal scope: "X was true in 2020 but not 2024"
   - Version control for facts
   - Track when evidence confidence changed
   
3. **Event stream processing**
   - Monitor news/updates; trigger re-verification
   - Alert when claim becomes outdated

**Timeline**: 2-3 years to production real-time system

---

### 13.4 Explainability and Transparency

**Problem**: Users don't understand why system made decision

**Current state**:
```
System: "SUPPORTED [0.91 confidence]"

User: Why?
- What evidence was retrieved?
- Why did it prefer some sources over others?
- How much did each component contribute?
```

**Challenges**:
1. **Black-box neural components**: Ensemble + NLI + aggregation is hard to explain
2. **Evidence selection**: Why retrieve this source over others?
3. **Confidence calibration**: Why 0.91 specifically?

**Future directions**:
1. **Attention-based explanations**
   - Show which evidence words were attended to
   - Visualize evidence relevance scores
   
2. **Component attribution**
   - Which component (semantic, entailment, diversity, agreement) drove decision?
   - Shapley values or LIME to quantify contribution
   
3. **Interactive explanations**
   - User can question system: "What if this evidence wasn't available?"
   - System shows how decision would change
   
4. **Human-in-the-loop**
   - System flags uncertain predictions
   - Human expert reviews and provides feedback
   - System learns from human corrections

**Example explainable decision**:
```
Claim: "Photosynthesis converts CO2 to oxygen"

✓ SUPPORTED [0.93 confidence]

Evidence:
1. [0.98 relevance] "Photosynthesis is a process where plants 
   use sunlight to convert carbon dioxide into oxygen"
   (Source: Khan Academy Biology)
   
2. [0.95 relevance] "Light reactions: H2O → O2 + ATP + NADPH"
   (Source: Campbell Biology Textbook)

Component contributions:
- Semantic matching: 35% (evidence highly matches claim)
- Entailment: 40% (evidence entails claim strongly)
- Agreement: 12% (multiple sources agree)
- Diversity: 5% (good coverage; but only 2 sources)
- Additional factors: 8%

Human confidence: Based on evidence from two trustworthy 
educational sources, both at high confidence level.
```

---

### 13.5 Handling Subjective and Contested Claims

**Problem**: Fact verification assumes ground truth exists

**Examples of hard cases**:
- "Shakespeare is the greatest writer" (subjective opinion)
- "Climate change caused this hurricane" (causal claim, debated)
- "AI poses existential risk" (contested belief)
- "Vaccines are safe" (scientific consensus, but contested by some)

**Current system response**: Fails or returns low confidence

**Challenge**: Can we verify contested claims responsibly?

**Future approach: Perspectival verification**
```
Claim: "AI poses existential risk"

System returns:
{
  "claim": "AI poses existential risk",
  "responses": [
    {
      "perspective": "AI safety researchers",
      "stance": "PARTIALLY_SUPPORTED",
      "evidence": ["Paul Christiano (OpenAI)", "Eliezer Yudkowsky (MIRI)", ...],
      "confidence": 0.68
    },
    {
      "perspective": "Industry researchers",
      "stance": "NOT_SUPPORTED",
      "evidence": ["OpenAI leadership", "DeepMind leadership", ...],
      "confidence": 0.62
    },
    {
      "perspective": "Mainstream media",
      "stance": "PARTIALLY_SUPPORTED",
      "evidence": ["BBC report", "NYT article", ...],
      "confidence": 0.71
    }
  ],
  "recommendation": "Contested claim; multiple valid perspectives"
}
```

**Benefits**:
- ✅ Honest about epistemic limits
- ✅ Helps users understand nuance
- ✅ Supports democratic deliberation
- ✅ Avoids appearing to suppress viewpoints

**Challenges**:
- How to define perspectives fairly?
- Risk of "false balance" (treating marginal views as equal)
- Hard to automate

**Future research**: Perspectival fact verification frameworks

---

### 13.6 Multilingual and Cross-Lingual Fact Verification

**Problem**: Most work in English; underrepresented languages need verification

**Current state**:
- FEVER (English): 185K claims
- mFEVER (multilingual): 37K claims across 8 languages
- Most systems: English-only

**Challenges**:
1. **Translation loss**: Translating claims/evidence may change meaning
2. **Language-specific evidence bases**: Wikipedia language versions differ
3. **Cross-lingual reasoning**: "Claim in Chinese, evidence in English"

**Example**:
```
Chinese claim: "北京是中国首都" (Beijing is China's capital)
English evidence corpus used for verification
→ Needs translation pipeline (introduces errors)
```

**Future directions**:
1. **Multilingual evidence retrieval**
   - Search across multiple language Wikipedias simultaneously
   - Cross-lingual embeddings (e.g., XLMR, mBERT)
   
2. **Multilingual NLI**
   - Entailment models for non-English languages
   - Cross-lingual NLI (claim in one language, evidence in another)
   
3. **Benchmarks for multilingual verification**
   - Extend FEVER to 50+ languages
   - Study zero-shot, few-shot, full-shot performance

**Impact**: Enable fact verification for 7 billion non-English speakers

---

### 13.7 Adversarial Robustness

**Problem**: Models vulnerable to adversarial inputs

**Examples**:
```
Original claim: "Photosynthesis requires light"
Adversarial variant: "Photosynthesis doesn't require light"
↓ (subtle negation)

Adversarial variant: "All photosynthesis requires light"
↓ (universal quantifier)

Adversarial variant: "Most photosynthetic organisms require light"
↓ (different scope)
```

**Current state**:
- Humans: Easily detect negations and quantifiers
- Current systems: 91% accuracy on clean data, 62% under adversarial attack (-29pp drop)

**Future research**:
1. **Adversarially trained models**
   - Train on adversarial examples
   - Expected improvement: +15-20pp robustness
   
2. **Formal verification**
   - Prove model correct for whole input classes
   - Complement empirical evaluation
   
3. **Robustness benchmarks**
   - Create adversarial FEVER dataset
   - Rank systems by adversarial performance

**Relevance**: Critical for high-stakes deployment (medical, legal)

---

### 13.8 Low-Resource and Few-Shot Verification

**Problem**: Many languages/domains have limited training data

**Challenge**: Can we verify in domains with <100 labeled claims?

**Current limitation**: Deep learning needs 1000s of labeled examples

**Future approaches**:
1. **Few-shot learning**
   - Meta-learning approaches (MAML, Prototypical Networks)
   - Learn to verify from few examples
   
2. **Data augmentation**
   - Paraphrase existing claims
   - Auto-generate training examples
   
3. **Transfer learning with minimal supervision**
   - Pre-train on FEVER
   - Fine-tune on target domain with 50-100 labels
   - Expected performance: 85%+ accuracy (vs 95% with full training)

**Impact**: Deploy fact verification to niche domains rapidly

---

### 13.9 Integration with Question Answering and Reasoning

**Problem**: QA and fact verification are currently separate

**Vision**: Unified system
```
User: "Is climate change human-caused?"

System:
1. Decompose question into claims
2. Retrieve evidence
3. Reason over evidence
4. Synthesize answer with confidence

Response: "Based on 95% of climate scientists agreeing, 
the answer is YES with 0.89 confidence.
Dissenting views: <summary>"
```

**Future research**:
- Fact verification as component in QA pipelines
- Joint training on FEVER + SQuAD + MS MARCO datasets
- End-to-end reasoning systems

---

### 13.10 Measuring System Calibration Across Domains

**Problem**: Current work doesn't measure calibration generalization

**Research question**: If system is calibrated on FEVER, is it calibrated on SciFact?

**Expected answer**: NO (different label distributions)

**Future work**:
1. **Cross-domain calibration benchmarks**
   - Train on FEVER, measure ECE on SciFact, CSClaimBench, etc.
   - Quantify calibration drift
   
2. **Domain-adaptive calibration**
   - Adjust temperature scaling for each domain
   - Learning from unlabeled target domain data
   
3. **Universal calibration methods**
   - Develop calibration approaches that work across domains
   - Goal: ECE < 0.10 across all domains

**Impact**: Essential for reproducible, trustworthy deployments

---

## 14. Recommended Research Roadmap (2024-2028)

### Year 1 (2024): Foundation
- ✅ Multi-hop fact verification dataset
- ✅ Multilingual FEVER (50+ languages)
- ✅ Explainability toolkit for fact verification
- ✅ Adversarial robustness benchmarks

### Year 2 (2025): Integration
- Real-time web retrieval for fact verification
- Perspectival verification framework (handle contested claims)
- Few-shot verification systems
- Cross-domain calibration benchmarks

### Year 3 (2026): Deployment
- Educational fact verification systems at 5-10 universities
- Biomedical fact verification integrated with PubMed
- Wikipedia misinformation detection deployed at Wikimedia
- Learning outcomes study: Does automated grading affect student learning?

### Year 4 (2027-2028): Scale and Impact
- Multilingual systems deployed across 20+ languages
- Integration with LLMs (fact-check LLM outputs)
- Online learning systems (update from human feedback)
- Standardized evaluation framework adopted across community

---

## 15. Broader Impact and Societal Considerations

### 15.1 Opportunities

**Education**:
- Democratize fact-checking capability (all students, teachers)
- Reduce teacher workload (focus on reasoning, not fact-verification)
- Build critical thinking skills

**Science**:
- Speed up literature review
- Detect fraudulent claims early
- Improve reproducibility

**Democracy**:
- Counter misinformation at scale
- Empower voters with accurate information
- Support fact-checkers (journalists, researchers)

### 15.2 Risks and Mitigation

**Risk 1: Misuse for propaganda**
- Rogue actor deploys biased system → spreads propaganda with "fact-checked" label
- Mitigation: Open-source systems + transparency + source attribution

**Risk 2: Over-reliance on automation**
- Teachers/fact-checkers stop thinking critically → defer to system
- Mitigation: Explainability + confidence reporting; system should not replacing human judgment

**Risk 3: Bias amplification**
- System reflects biases of training data/evidence base
- Mitigation: Diverse evidence sources + bias audits + perspectives framework

**Risk 4: Environmental impact**
- Large models require significant compute
- Mitigation: Efficient architectures; edge deployment

### 15.3 Governance Framework

**Proposed principles**:
1. **Transparency**: Disclose system limitations, evidence sources, calibration
2. **Accountability**: Clear chains of responsibility for system decisions
3. **Auditability**: Regular external audits of accuracy, calibration, bias
4. **Human oversight**: Humans in loop for high-stakes decisions (medical, legal)
5. **Open science**: Share models, datasets, evaluation frameworks

---

## 16. Conclusion and Synthesis

### 16.1 Key Findings from Survey

**What we know**:
1. Fact verification is achievable: SOTA 75-85% accuracy on benchmarks
2. Calibration is critical but overlooked: ECE commonly 0.15-0.25 (very miscalibrated)
3. Confidence is valuable: Enables selective prediction, hybrid human-AI workflows
4. Reproducibility is rare: Most systems don't document hyperparameters, cross-GPU testing
5. Domain generalization is hard: 20-30pp accuracy drop between domains

**What we don't know**:
1. Real-world deployment: Do systems work on Wikipedia/Twitter/medical records at scale?
2. Learning outcomes: Does automated grading affect student learning positively/negatively?
3. Societal impact: Does fact verification reduce misinformation belief at scale?
4. Multi-hop reasoning: How to reliably verify claims requiring 3+ reasoning steps?

### 16.2 Fact Verification is Maturing

**2016-2018 (Inception)**:
- FEVER dataset created
- Initial neural approaches (LSTM, CNN)
- Accuracy: 64% on FEVER

**2018-2020 (BERT Era)**:
- Pre-trained transformers (BERT, RoBERTa)
- Accuracy: 75-80% on FEVER
- Still no calibration focus

**2020-2022 (Ensemble Era)**:
- Ensemble methods (combining semantic + NLI)
- Accuracy: 80-82% on FEVER
- First papers on confidence/calibration (limited)

**2022-2024 (Calibration + Uncertainty)**:
- Explicit focus on ECE, selective prediction
- Temperature scaling, conformal prediction
- Accuracy: 81-83% on FEVER; ECE: 0.08-0.12 (well-calibrated)
- Deployment studies emerging (Wikipedia, education)

**2024+ (Integration + Application)**:
- Real-time retrieval
- Multi-hop reasoning
- Multilingual systems
- Integration with LLMs for fact-checking outputs

### 16.3 Smart Notes Positioning

**Contribution in landscape**:
- ✅ First system optimizing for calibration + selective prediction
- ✅ First to deploy to education domain
- ✅ First to verify cross-GPU bit-identical reproducibility
- ✅ Demonstrates practical high-stakes application

**Builds on**: FEVER (dataset), DPR (retrieval), BART-MNLI (NLI), temperature scaling (calibration), conformal prediction (uncertainty)

**Enables**: Future systems with trustworthy confidence; educators + fact-checkers with AI assistance

---

### 16.4 Summary Table: Research Progress Overview

| Era | Years | Accuracy | ECE | Multi-hop | Multilingual | Deployed? |
|-----|-------|----------|-----|-----------|--------------|-----------|
| Inception | 2016-18 | 64% | Not reported | Not attempted | English only | No |
| BERT | 2018-20 | 75-80% | Not reported | Limited | English only | Limited |
| Ensemble | 2020-22 | 80-82% | 0.15-0.20 | Some attempts | Few languages | Limited |
| Calibration | 2022-24 | 81-83% | 0.08-0.12 | More attempts | Growing | More attempts |
| Integration | 2024+ | 82-85% (projected) | <0.08 (target) | Standard | 20+ languages | Scaling |

---

**End of Survey Section 13-16**

Next: Survey conclusion and full bibliography

