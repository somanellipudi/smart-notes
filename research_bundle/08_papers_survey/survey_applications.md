# Survey Paper: Applications of Fact Verification

## 9. Applications: Where Fact Verification Deploys

### 9.1 Wikipedia and Misinformation Detection

**Context**: Wikipedia is crowdsourced; vandalism and misinformation occur

**Problem**: Detect false claims in Wikipedia articles before publication

**Deployment scenario**:
```
Wikipedia Article Editor submits: "The moon is made of cheese"
↓
Smart Verification System:
  - Retrieves relevant evidence (astronomy sources)
  - Classifies: NOT_SUPPORTED (contradiction found)
  - Confidence: 0.97
↓
Result: Flag for human review; prevent publication
```

**Success metrics**:
- Precision: Reduce false positives (don't flag true claims)
- Coverage: Flag most vandalism/errors
- Speed: Real-time or batch processing

**Current state**:
- FEVER dataset created to benchmark this task
- Wikipedia+Wikimedia Systems now run automated fact-checking
- SOTA systems achieving 75-85% accuracy

**Challenge**: Scale (Wikipedia has millions of articles, continuous updates)

### 9.2 Scientific Fact Verification (Biomedical Domain)

**Context**: Scientific papers make claims; need verification against existing literature

**Problem**: Does claim about new drug efficacy match published evidence?

**Deployment scenario**:
```
Biomedical Researcher claims: "Protein X reduces inflammation by 50%"
↓
Scientific Verification System:
  - Retrieves PubMed abstracts on Protein X
  - Searches for "inflammation" mentions
  - Classifies against evidence base
  - Confidence: 0.72 (moderate; some papers support, some unclear)
↓
Result: Flag for expert review; highlight supporting/contradicting papers
```

**Success metrics**:
- Precision: High (can't spread false medical claims)
- Sensitivity: Catch unsupported claims
- Calibration: Essential (doctors rely on confidence)

**Current state**:
- SciFact dataset (1,409 biomedical claims)
- SOTA: 72.4% accuracy
- Some systems deployed in research assessment (e.g., AllenAI ARISTO)

**Why calibration matters**:
- Medical domain is high-stakes
- Doctor needs to know: "75% confident" vs "95% confident"
- Miscalibration causes patient harm

---

### 9.3 Educational Fact-Checking (New Application, Smart Notes)

**Context**: Education requires accuracy + trust + learning

**Problem**: Help students verify claims in assignments; support teachers in grading

**Four use cases**:

#### Use Case 1: Student Self-Verification ("Did I get this right?")

**Workflow**:
```
Student writes: "Quicksort's average complexity is O(n log n)"

Smart Notes verification:
  - Retrieves computer science sources
  - NLI: "Quicksort average complexity is O(n log n)" vs evidence
  - Result: SUPPORTED with 0.91 confidence
  
Student feedback:
  "✓ Correct! I found supporting evidence in 3 independent sources."
  (Confidence: 91%)
```

**Benefit**: 
- ✅ Student learns actively (generates claim, verifies)
- ✅ Builds verification skills
- ✅ Reduces misconceptions (immediate correction)

#### Use Case 2: Instructor Review Prioritization ("Which needs grading?")

**Workflow**:
```
Batch of 50 student answers on "Database Transactions"

Smart Notes scores all claims with confidence:
  - 35 answers: High confidence (>0.85) → Auto-graded
  - 12 answers: Medium confidence (0.60-0.85) → Flag for review
  - 3 answers: Low confidence (<0.60) → Instructor decides

Teacher dashboard:
[Sort by confidence]
  Low (3):
    - "ACID properties guarantee isolation" [0.58 confidence]
    - "Timestamps implement version control" [0.54]
  Medium (12):
    - "Locking prevents dirty reads" [0.72]
    - ...
  High (35): Auto-graded ✓
```

**Benefit**:
- ✅ Teacher focuses on hard cases
- ✅ Could grade 50 answers in 10 min (vs 30 min manual)
- ✅ Confidence highlight which need most attention

#### Use Case 3: Hybrid Human-AI Grading

**Workflow**:
```
For each student answer:
  IF confidence > 0.85:
    grade = AUTO_GRADE  # No human needed
    confidence_status = "AUTO"
  ELIF confidence > 0.60:
    grade = TEACHER_REVIEW  # Suggested answer shown
    confidence_status = "FLAG_FOR_REVIEW"
  ELSE:
    grade = DEFER  # Complex; needs human judgment
    confidence_status = "DEFER"
```

**Results** (from Smart Notes evaluation):
- 95% coverage (95% of claims graded automatically or flagged appropriately)
- 81% automatic grading (highest confidence claims)
- 14% flagged for review (moderate confidence; teacher has time)
- 5% deferred (low confidence; complex reasoning)

#### Use Case 4: Pedagogical Confidence Feedback

**Advanced**: Explain why confident or uncertain

**High confidence**:
```
Claim: "TCP is connection-oriented"
[0.94 confidence]

Explanation: "I found 5+ independent sources confirming this. 
The evidence is overwhelming and consistent."

Evidence shown:
1. RFC 793 definition
2. Cisco networking textbook
3. Computer Networking course materials
```

**Medium confidence**:
```
Claim: "BGP uses distance-vector algorithm"
[0.68 confidence]

Explanation: "I found mixed evidence. Most sources describe BGP 
as path-vector, but some call it distance-vector historically."

Evidence shown:
1. RFC 4271 (path-vector, official)
2. Old textbook (calls it distance-vector)
3. Research paper (clarifies hybrid nature)
```

**Low confidence**:
```
Claim: "Edge computing reduces latency by 50%"
[0.41 confidence]

Explanation: "This is ambiguous. 50% reduction varies by context, 
hardware, and application type. I can't verify without more specifics."

Suggestion: "Try being more specific: 
- What type of edge?
- What latency metric?
- In what application domain?"
```

**Benefit**: 
- ✅ Students learn epistemic humility
- ✅ Understand uncertainty as feature, not bug
- ✅ Builds critical thinking

---

### 9.4 Legal and Regulatory Fact-Checking

**Context**: Legal documents require factual accuracy; errors cause liability

**Problem**: Verify claims in contracts, compliance documents, legal arguments

**Deployment**:
- Contract review: Does claim about liability match legal precedent?
- Regulatory compliance: Does assertion meet standards?
- Due diligence: Verify claims in M&A documents

**Current state**: Limited application (specialized legal NLP needed, domain-specific training)

---

### 9.5 Emerging: Multimodal Fact-Checking (Image + Text)

**Context**: Deepfakes, manipulated images require verification

**Problem**: Claim in image (text overlay or caption) + image itself; verify both

**Example**:
```
Image: (Fake image of politician at rally)
Text overlay: "Mayor attends downtown event"

Verification needed:
  - Text claim: Did mayor attend?
  - Image: Is photo authentic or manipulated?
```

**Challenge**: Requires combining vision + NLP

**Current state**: Emerging research (FEVER-images, multimodal datasets)

---

## 10. Deployment Considerations

### 10.1 Data Privacy and Ethics

**Privacy**:
- Evidence corpus may contain sensitive information
- GDPR/CCPA considerations for training data
- Smart solution: Offline evidence (fixed corpus) vs. online retrieval (privacy risk)

**Bias**:
- Training data may encode biases
- Fact-checking systems can amplify bias
- Mitigation: Diverse evidence sources, bias audits

**Transparency**:
- Users should know system is AI-based
- Should see evidence used for decisions
- Should understand confidence/uncertainty

### 10.2 Computational Requirements

**Throughput**:
- Wikipedia: ~1M queries/day possible
- Biomedical: ~10K queries/day reasonable
- Education (per-school): ~1K queries/day typical

**Latency requirements**:
- Wikipedia (batch): 10s CPU time OK
- Interactive (education): <1s required for user experience
- Medical (real-time): 100-500ms needed

**Hardware**:
- GPU: Beneficial for NLI inference (180ms on V100 per claim)
- CPU: Viable for retrieval
- Smart Notes latency (615ms): Suitable for education/batch; not real-time

### 10.3 Integration Paths

**Path 1: Standalone service**
```
API:
POST /verify
{
  "claim": "...",
  "context": "CS education",
  "return_evidence": true
}

Response:
{
  "label": "SUPPORTED",
  "confidence": 0.91,
  "evidence": [...],
  "reasoning": "..."
}
```

**Path 2: Embedded in application**
```
# Inside learning management system (LMS)
class SmartNotesVerifier:
    def verify_student_answer(claim, course_module):
        # Uses course-specific evidence corpus
        return label, confidence, feedback
        
# In grading interface
feedback = verifier.verify_student_answer(
    claim="My claim...",
    course_module="CS101: Algorithms"
)
```

**Path 3: Batch processing**
```
# Weekly verification of Wikipedia flagged articles
batch_verify(
    claims_file="flagged_claims_20260218.jsonl",
    output_file="verification_results.json",
    batch_size=100
)
```

---

## 11. System Deployment Case Study: Smart Notes in University Classroom

### 11.1 Scenario

**Institution**: Large CS university (200 CS students/semester)

**Course**: CS 2310 Networking (50 students)

**Assessment**: 4 exams × 50 students = 200 exams; each ~15 factual claims (3,000 total claims/semester)

**Traditional grading**: 
- Instructor: ~40 hours grading
- TAs: 2–3 people, 80 person-hours
- Total: ~120 person-hours

### 11.2 Smart Notes Deployment

**Setup**:
1. Create evidence corpus: Networking textbooks, RFCs, course materials (~500 documents)
2. Train on 200 example claims from past exams (validation set)
3. Deploy via LMS integration

**Workflow**:
```
Exam submitted → extract factual claims (auto or manual triage)
↓
For each claim:
  - Score: 0-1 confidence
  - If confidence > 0.85: Auto-grade CORRECT (if SUPPORTED)
  - If 0.60-0.85: Flag for TA review (suggest grade)
  - If < 0.60: Defer to instructor
↓
Grade assigned; feedback generated
```

### 11.3 Results Projection (Based on Smart Notes Performance)

**Claims distribution** (by confidence):
- High (>0.85): 60% of claims → Auto-graded
- Medium (0.60-0.85): 30% of claims → TA review (~2 min each)
- Low (<0.60): 10% of claims → Instructor decision (~5 min each)

**Time savings**:
- Auto-grade: 60% × 3,000 = 1,800 claims (0 person-hours)
- TA review: 30% × 3,000 = 900 claims × 2 min = 30 person-hours
- Instructor: 10% × 3,000 = 300 claims × 5 min = 25 person-hours
- **Total**: ~55 person-hours (vs 120 traditional)
- **Savings**: ~65 person-hours (54% reduction)

**Quality**:
- Auto-graded claims: 81.2% accuracy (some errors, but systematic)
- TA-reviewed claims: ~95% accuracy (human oversight)
- Overall: ~88% claim-level accuracy
- Instructor satisfaction: High (fewer routine claims to grade)

### 11.4 Broader Implications

**Scale**: This scales to 1,000+ students/year at large universities

**Open questions**:
- Does using automated grading affect student learning?
- How to handle appeals/disagreements?
- What about claims not in evidence base?

**Future**: Learning outcome studies needed to measure pedagogical impact

---

## 12. Limitations and Realistic Expectations

### 12.1 What Fact Verification CAN Do

- ✅ Flag claims that match/contradict retrieved evidence
- ✅ Provide calibrated confidence for such matching
- ✅ Scale to many queries
- ✅ Reduce human workload

### 12.2 What Fact Verification CANNOT Do

- ❌ Verify claims without evidence base (novel facts not in corpus)
- ❌ Handle subjective claims ("Shakespeare is greatest writer")
- ❌ Deep causal reasoning ("Did X cause Y?")
- ❌ Replace human expert judgment

### 12.3 Realistic Deployment

**Success case**: Domain with fixed, well-studied knowledge base
- Example: Physics intro course facts
- Example: Standard software engineering principles
- Example: Published medical research

**Challenging case**: Emerging, contested, or subjective domains
- Example: Political claims (contested facts)
- Example: Recent AI research (rapid evolution)
- Example: Philosophical questions (subjective)

---

**Conclusion of Applications**: Fact verification is most valuable in education and science where accuracy and calibration matter most. Smart Notes' calibration + uncertainty quantification enables deployment in these high-stakes domains for the first time.

