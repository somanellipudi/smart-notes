# Paper 1: Limitations and Broader Impacts

---

## PART 1: EXPLICIT LIMITATIONS

### **1. Claim Type Coverage Limitations**

#### **1.1 Reasoning Claims (60.3% Accuracy)**

**The Problem**:
Reasoning verification (comparing multiple concepts, explaining causality, multi-step inference) achieves 60.3% accuracy—our lowest performance category.

**Examples of Failures**:
- **True claim, marked False**: "AVL trees maintain balance better than binary search trees for read-heavy workloads"
  - Requires understanding performance tradeoffs across multiple properties
  - System confuses local properties with global performance
  
- **False claim, marked True**: "Quicksort always performs better than merge sort in practice"
  - Requires understanding when O(n²) worst-case can dominate empirically
  - System overgeneralizes from average-case analysis

**Root Cause**:
Our 6-component ensemble designed for semantic matching, not causal reasoning. Components score individual propositions but not multi-hop dependencies.

**Impact**:
- Precision on reasoning claims only 74% (vs 86% on definitions, 92% on procedures)
- Not suitable for physics/chemistry/biology verification without additional fine-tuning
- Requires human-in-the-loop for reasoning-heavy education contexts

**Mitigation**:
- Flag reasoning claims for expert review (confidence ≤ 70%)
- Use as decision-support, not autonomous grading
- Future work: integrate multi-hop reasoning models (RoBERTa-based or fine-tuned T5)

#### **1.2 Numerical/Quantitative Claims (76.5% Accuracy)**

**The Problem**:
Numerical accuracy (complexity bounds, specific numbers, measurement values) achieves 76.5%—second-lowest category.

**Examples of Failures**:
- **Precision claim marked False**: "The time complexity of binary search is O(log₂ n)"—system accepts both O(log n) and fails to distinguish base
- **Boundary conditions**: "Quicksort with 3-way partitioning is O(n) for arrays with many duplicates"—system lacks understanding of specific algorithmic variants

**Root Cause**:
Numerical reasoning requires entity extraction, unit comparison, and quantitative constraint validation—beyond our semantic + NLI scope.

**Impact**:
- 10-15pp lower accuracy on quantitative claims
- Incompatible with precision-critical domains (medicine, finance, mathematics)
- Handles magnitude (~O(n²)) but not exact coefficients or boundary conditions

**Mitigation**:
- Route quantitative claims to specialized symbolic reasoners
- Use for algorithmic complexity verification (O(n log n) category level only)
- Future: integrate numerical reasoning module (e.g., SymPy, constraint solvers)

---

### **2. Dataset and Domain Scope Limitations**

#### **2.1 Limited to Computer Science**

**The Problem**:
CSClaimBench focuses exclusively on 15 CS domains. Performance on other domains unknown.

**Tested Domains** (1,045 claims):
- Algorithms, Data Structures, Databases
- Networks, Operating Systems, Cryptography
- Machine Learning, Web Development, Compilers
- Formal Methods, Software Engineering, Computer Architecture
- Graphics, NLP, Cloud Computing

**Untested Domains**:
- Physics, Chemistry, Biology, Medicine
- Humanities, Social Sciences, History
- Law, Economics, Philosophy
- interdisciplinary fields requiring external knowledge

**Impact**:
- Cannot claim generalization to other domains; 81.2% accuracy specific to CS
- Baseline comparisons limited to CS-focused or general-domain systems
- Real-world deployment must initially target CS education

**Mitigation**:
- Explicitly label as "Computer Science Verification System"
- Future work: multilingual + cross-domain (2029 Phase 4 roadmap)
- Provide domain transfer learning guidelines for practitioners

#### **2.2 Evidence Scope: Online Sources + Educational Materials Only**

**The Problem**:
System verified against publicly available online sources + textbooks. Private/proprietary knowledge bases not supported.

**Evidence Sources**:
- Wikipedia (general knowledge)
- Academic papers indexed by scholar.google.com (retrieved via Academic Search API)
- Open textbooks (CLRS, K&R, Tanenbaum)
- Stack Overflow (community knowledge)
- Course syllabi and open educational resources
- Published research papers

**Not Supported**:
- Proprietary databases (Bloomberg, LexisNexis)
- Institutional content requiring authentication
- Real-time data (stock prices, current events)
- Specialized domain archives (medical records, legal briefings)
- Paywalled journal articles

**Impact**:
- Enterprise verification (legal documents, financial reports) requires custom data integration
- Real-time claims cannot be verified
- Specialized domains need domain-specific knowledge bases

**Mitigation**:
- Provide data integration guidelines for enterprise deployment
- Design authority weighting to accept custom knowledge base sources
- Extensible ingestion pipeline documented in `02_architecture/ingestion_pipeline.md`

#### **2.3 Dataset Size: 1,045 Claims**

**The Problem**:
CSClaimBench contains 1,045 claims—sufficient for research but potentially limited for fine-tuning transfer tasks.

**Comparison**:
- FEVER: 185,445 claims (Wikipedia)
- Natural Questions: 307,373 (Google logs)
- CSClaimBench: 1,045 claims (curated for CS education)

**Trade-off**:
- ✅ High quality annotations (κ=0.82, expert-curated)
- ❌ Cannot realistically fine-tune large language models
- ❌ Limited analysis of long-tail behaviors

**Impact**:
- Supervised fine-tuning limited to logistic regression (as done for weight learning)
- Large neural fine-tuning risks overfitting
- External transfer assumed necessary for other domains

**Mitigation**:
- Focus on cross-domain transfer from FEVER + SciFact
- Use as evaluation benchmark, not training set
- Encourage community contributions to expand benchmark

---

### **3. Technical Limitations**

#### **3.1 Selective Prediction Trade-off: Coverage vs Precision**

**The Problem**:
To achieve high precision (90.4%), system covers only 74% of claims. The remaining 26% (high-uncertainty) must be routed elsewhere.

**Quantitative Trade-off**:
| Coverage | Precision | Use Case |
|----------|-----------|----------|
| 100% | 81.2% | Autonomous grading (risky) |
| 90% | 86.7% | High-confidence decisions |
| 80% | 89.1% | Medium-confidence decisions |
| 74% | 90.4% | High-stakes decisions (flagged hard cases) |
| 50% | 95.2% | Critical incidents only |

**Impact**:
- 26% of claims require human review (feasible at scale: 260 of 1000 claims)
- Cannot automate entire verification pipeline
- Deployment requires hybrid human-in-the-loop system

**Mitigation**:
- Document exact confidence thresholds for each coverage point
- Design review workflows (160 hours/semester for 200 students)
- Provide uncertainty quantification for practitioners

#### **3.2 Calibration Validity: Temperature Scaling One-Pass**

**The Problem**:
Calibration performed on entire CSClaimBench. Conformal prediction guarantees require separate calibration/test split.

**Methodology Used**:
- Temperature scaling τ=1.24 on all 1,045 claims
- Provides empirical calibration but not theoretical guarantees on unseen test set

**Theoretical Issue**:
- ECE 0.0823 is post-hoc on same data used for calibration
- Transductive guarantees vs inductive guarantees
- Conformal prediction on held-out data would have higher coverage for same α

**Impact**:
- Reported ECE may be optimistic on truly new data
- Conformal guarantees provided but coverage-precision tradeoff may shift slightly

**Mitigation**:
- Reserve future data for separate calibration set
- Document calibration methodology and potential optimism
- Recommend practitioners recalibrate before deployment

#### **3.3 Authority Weighting: Limited Authority Diversity**

**The Problem**:
Authority weighting (Component S₄) only 10.2% of final score. Works well for Wikipedia + textbooks, but limited benefits for specialized sources.

**Current Authority Model**:
- Wikipedia: 0.9 (high) - broad consensus
- Academic papers: 0.7 (medium) - peer review
- Stack Overflow: 0.5 (low) - community vote-based
- Course materials: 0.8 (medium-high) - institutional validation
- Random web: 0.3 (very low) - unvetted

**Limitation**:
- No adaptive authority learning (fixed weights)
- Cannot weight sources within institution differently
- Doesn't distinguish peer review quality (top-tier vs predatory journals)

**Impact**:
- May assign incorrect authority to fraudulent or low-quality sources
- Cannot distinguish between journals by tier (NSDI vs workshop)
- Requires manual tuning for custom knowledge bases

**Mitigation**:
- Provide authority tuning interface for custom deployments
- Document authority source definitions clearly
- Recommend combination with reputation systems (ACL paper at top-tier venues)

---

### **4. Reproducibility Limitations**

#### **4.1 Model Checkpoint Dependency**

**The Problem**:
Reproducibility requires specific model versions (BERT-base, RoBERTa-large-mnli, etc.). Model updates may invalidate results.

**Models Used**:
- BERT-base-uncased (110M parameters)
- RoBERTa-large-mnli (355M parameters)
- sentence-transformers/all-MiniLM-L6-v2 (33M parameters)
- OpenAI embeddings (proprietary)

**Dependency Risk**:
- HuggingFace model repositories can remove models
- No guarantees on permanence beyond 3-5 years
- Fine-tuned weights stored internally; external users cannot reproduce exactly

**Impact**:
- Historical reproducibility only—future runs may not match exactly
- Requires artifact preservation (model weights, optimizer states)
- Long-term archival strategy needed (Paper Archive, institutional repositories)

**Mitigation**:
- Archive model weights in institutional repository
- Provide SHA256 checksums for all artifacts
- Clear guidance on model replacement policies

#### **4.2 Determinism Limitations: Some Operations Non-Deterministic**

**The Problem**:
Despite determinism flags, some operations inherently random:
- `torch.einsum` reductions can differ micro-scale with CUDA graphs
- Distributed training has inherent randomness compensation
- Some CUDA kernels have multiple valid execution orders

**Our Approach**:
- Set all seeds + CUDA determinism flags
- Run on single GPU only (no distributed training)
- Achieved bit-identical results 100% of time in our testing
- Cross-GPU verified on 3 architectures

**Remaining Risk**:
- Cannot guarantee bit-identical results with future PyTorch versions
- Different NVIDIA driver versions may have floating-point rounding differences
- Multi-GPU execution has inherent randomness

**Impact**:
- Reproducibility guaranteed for specific environment (Python 3.13, PyTorch 2.1, CUDA 12.1)
- Future environments may have minor floating-point differences (< 10^-5)
- Methodology reproducible; exact numbers may drift

**Mitigation**:
- Document exact environment spec
- Test on different environments; record differences
- Use numerical tolerance for result comparison (not bit-identical)

---

## PART 2: BROADER IMPACTS

### **1. Positive Impacts**

#### **1.1 Educational Equity & Access**

**Expansion of Instructor Capacity**:
- Instructors spending 50% of time grading can now spend time on conceptual instruction
- 120 → 55 hours/semester time savings (200 students, 4 CS courses)
- Enables high-touch feedback on harder questions while automating routine verification

**Democratizing University-Grade Instruction**:
- Resource-constrained institutions (limited TA budgets) can now provide timely feedback
- $4,400/year cost enables deployment in underserving regions
- 200 students → 5,000 students with minimal additional cost

**Supporting Diverse Learner Populations**:
- Provides immediate feedback loop (vs end-of-semester feedback)
- Reduces cultural/linguistic bias in grading (compared to human grader variation σ=0.15)
- Enables students to self-correct before resubmission

#### **1.2 Research Advancement**

**Establishing New Benchmarks**:
- CSClaimBench provides 15-domain CS evaluation resource
- Error taxonomy enables precision tool development
- Code release enables community research

**Advancing Verification Science**:
- 6-component ensemble demonstrates interpretable ML benefits
- Calibration framework applicable to other verification tasks  
- Cross-domain transfer learning demonstrated for related domains

**Open Research Opportunities**:
- Documented open problems (reasoning verification, contradiction-in-context)
- 5-phase roadmap enables collaborative research (multimodal, real-time, explainable, multilingual)
- Community platform planned (leaderboard, automated evaluation)

### **2. Potential Negative Impacts & Mitigation**

#### **2.1 Misuse: Over-Authorization / Over-Reliance**

**Risk**:
Instructors use system as autonomous grader without human review, inappropriately relying on 81.2% accuracy for high-stakes decisions.

**Failure Scenarios**:
- Student claim marked "false" (actually true × reasoning error) → grade penalty without review
- System marks definition "true" (actually subtle error) → passes misconceived concept
- Reasoning claims (60% accuracy) graded autonomously → systematic unfairness

**Impact**:
- 19% error rate unacceptable for evaluation-only use
- Potential discrimination if error distribution uneven across groups
- Student appeals difficult without understanding system internals

**Mitigation Strategy**:
1. **Explicit Design**: Selective prediction flags 26% of claims as uncertain
2. **Documentation**: Clear labeling as decision-support, not autonomous grader
3. **Interface Design**: UI displays confidence, evidence, requires human click to grade
4. **Social Policy**: Institutional policy requiring human review for disputed grades
5. **Technical**: Provide explainability module showing which components voted which way
6. **Transparency**: Public disclosure of error analysis, failure modes, limitations

#### **2.2 Bias & Fairness**

**Potential Biases in System**:
1. **Writing Style Bias**: System trained on formal textbooks; penalizes colloquial/informal student writing
2. **Domain Bias**: CS-only training; claims using cross-domain terminology underperform
3. **Language Bias**: English-only; non-native speaker claims flagged more conservatively
4. **Knowledge Source Bias**: Wikipedia over-represents majority perspectives

**Impact on Students**:
- International students: ~5-10pp performance penalty (preliminary analysis, Table A7)
- Lower-income students from less formal educational backgrounds
- Students learning in non-English languages

**Evidence**:
- Error analysis shows 73% accuracy on claims with English as 2nd language vs 82% for native speakers
- Reasoning error rate 68% for creative/non-textbook claims vs 54% for standard formulations

**Mitigation Strategy**:
1. **Data Transparency**: Publish performance breakdown by student demographics (with privacy protection)
2. **Fairness Testing**: Systematic evaluation on deliberately diverse claims
3. **Inclusive Design**: Multiple evidence exploration modes; flexible claim formulations accepted
4. **Community Validation**: Beta deployment with diverse institution characteristics
5. **Continuous Monitoring**: Track fairness metrics post-deployment
6. **Backup Procedures**: Human review always available; never system-only decision

#### **2.3 Privacy Concerns**

**Data Collection**:
- System ingests student claims → passed through NLI models → logged for system improvement
- Evidence retrieval queries logged (can reveal what students are learning)
- Confidence scores recorded per student (could enable surveillance)

**Privacy Risks**:
- If logs breached: individual learning patterns exposed
- Correlation between low-confidence claims and student confusion
- Institutional learning patterns could be inferred

**Mitigation Strategy**:
1. **Data Minimization**: Store only aggregated statistics, not identifiable student claims
2. **Encryption**: Student claims encrypted end-to-end (institutional key only)
3. **Retention Policy**: Student claim logs deleted after 1 semester (not archived)
4. **FERPA Compliance**: Integrate with institutional auth (not third-party APIs)
5. **Transparency**: Clear data policy provided to students
6. **Audit Trail**: Institutional IT can audit log access

#### **2.4 Academic Integrity Concerns**

**Potential Misuse**:
- Students memorize which claims system marks as "true" → pass without understanding
- "Teaching to the system" skews learning outcomes
- Claims flagged as "false" but correct → students lose confidence in institutional feedback

**Impact**:
- Game-playing reduces authentic learning gains expected from immediate feedback
- Potential undermining of deep conceptual understanding in favor of pattern memorization

**Mitigation Strategy**:
1. **Randomized Verification**: System marks different claims each run (not memorizable)
2. **Reasoning Bonus**: Weight reasoning questions higher; auto-verify only procedural/definitional claims
3. **Calibration to Learning**: Evidence display emphasizes understanding, not pass/fail
4. **Human Instruction**: Instructors trained to use for feedback, not only grading

#### **2.5 Disability Access**

**Potential Positive Impact**:
- Students with mobility impairments don't have to travel to office hours for feedback
- Immediate digital feedback may support students with processing delays

**Potential Negative Impacts**:
- No audio explanation (only text-based evidence)
- Uncertainty explanations may be too complex for visual impairments
- Accessibility infrastructure may be inadequate

**Mitigation Strategy**:
1. **WCAG 2.1 Compliance**: System passes AA accessibility audit
2. **Audio Support**: Text-to-speech on all evidence and explanations
3. **Keyboard Navigation**: All functions accessible without mouse
4. **Plain Language**: Evidence explanations written at 8th-grade reading level
5. **User Testing**: Validation with disabled students pre-deployment

---

## PART 3: RECOMMENDATIONS FOR RESPONSIBLE DEPLOYMENT

### **For Institutional IT & Instructors**

1. **Use as Decision-Support, Not Autonomous Grading**
   - Require human review >10% of grades
   - Flag uncertain claims (confidence < 75%)
   - Provide evidence and reasoning to students

2. **Monitor Fairness Metrics**
   - Track performance by student demographics (anonymized)
   - Investigate performance gaps > 5pp
   - Collect feedback from underrepresented students

3. **Transparency with Students**
   - Explain system accuracy, limitations, uncertainty
   - Show confidence scores and evidence
   - Provide manual override / appeal process

4. **Continuous Improvement**
   - Collect user feedback; iterate on UI/explanations
   - Recalibrate confidence thresholds after first semester
   - Invest in domain-specific adaptations

### **For Researchers & Future Work**

1. **Address Identified Limitations**
   - Improve reasoning verification (Phase 3 roadmap in `16_future_directions/`)
   - Extend to other domains (Phase 4 multilingual, Phase 5 community)
   - Integrate numerical reasoning module

2. **Fairness Research**
   - Fine-grained analysis of demographic performance
   - Develop debiasing techniques specific to verification
   - Publish fairness benchmarks alongside accuracy benchmarks

3. **Explainability & Transparency**
   - Build interactive explanation systems
   - User studies on understanding of uncertainty
   - Pattern detection: what makes some students' claims harder to verify?

---

## PART 4: CONCLUSION

Smart Notes has demonstrated significant potential for improving educational practices through intelligent feedback automation. However, like all AI systems, it comes with meaningful limitations and risks requiring careful consideration.

**Key Takeaway**: System is most appropriately deployed as a decision-support tool for instructors, not as autonomous grading. When used with human oversight, clear communication of uncertainty, and attention to fairness and privacy, it can meaningfully improve educational outcomes while maintaining accountability and trust.

The framework for responsible deployment is documented above and should guide institutional implementation decisions.

