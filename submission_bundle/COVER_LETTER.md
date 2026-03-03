[Author Letterhead]

March 2026

IEEE Access Editorial Office
IEEE Xplore Digital Library
United States

---

## COVER LETTER

**Manuscript Title**: CalibraTeach: Calibrated Selective Prediction for Real-Time Educational Fact Verification

**Authors**: Nidhibahen Patel, IEEE Member (Co-First, Software Engineer, Verizon), Soma Kiran Kumar Nellipudi, Senior IEEE Member (Co-First, Senior Software Engineer, Incomm Payments), Selena He (Assistant Professor, KSU)

**Corresponding Author**: Selena He (she4@kennesaw.edu)

---

Dear Editor,

We submit our manuscript "CalibraTeach: Calibrated Selective Prediction for Real-Time Educational Fact Verification" for consideration in IEEE Access. This work addresses a critical gap in educational AI systems: the lack of reliable uncertainty quantification in automated fact verification.

### Summary of Contributions

**1. Systematic Calibration Methodology**
We demonstrate that fact verification can achieve rigorous calibration (Expected Calibration Error: 0.1247, 95% CI [0.0989, 0.1679]) through a 7-stage ensemble explicitly designed for confidence estimation. Our approach combines six orthogonal signals (semantic relevance, entailment strength, evidence diversity, agreement, margin, authority) with learned weights optimized for calibration rather than accuracy alone.

**2. ML Optimization for Deployment**
An 8-model optimization layer reduces computational cost by 63% while maintaining calibration quality. Achieved latency: 67.68 ms mean (14.78 claims/sec throughput), enabling real-time classroom deployment on affordable GPU infrastructure.

**3. Uncertainty Quantification Framework**
We introduce formal risk-coverage analysis for educational decision-making, achieving AUC-AC 0.8803 (95% CI [0.8207, 0.9332]). At 74% coverage, the system maintains 90%+ precision while deferring uncertain cases to instructor review, enabling hybrid human-AI workflows.

### Evaluation Rigor

- **Primary Validation**: 260 expert-annotated claims from CSClaimBench (κ=0.89 inter-rater agreement)
- **Confidence Intervals**: Bootstrap 95% CIs on all metrics (2000 resamples)
- **Multi-Seed Stability**: Evaluation across 5 deterministic seeds (accuracy 0.8169 ± 0.0071, ECE 0.1317 ± 0.0088)
- **Transfer Testing**: 200 FEVER claims show graceful degradation (74.3% accuracy, 0.150 ECE) with clear re-calibration protocol
- **Infrastructure Validation**: 20,000 synthetic claims confirm scalability and stability

### Methodological Strength: Calibration Parity

All baseline systems underwent identical post-hoc temperature scaling on the same validation set. This ensures fair benchmarking where observed ECE improvements reflect CalibraTeach's ensemble architecture rather than differential calibration methodology—a critical consideration often overlooked in prior work.

### Honest Limitations & Hypotheses

**Domain Specificity**: Our system is trained and validated exclusively on computer science education claims. Generalization to other domains requires explicit re-calibration (see Appendix E.4 for protocol). We provide detailed guidance for practitioners adapting CalibraTeach to new domains.

**Pedagogical Benefits Are Hypotheses**: While pilot data (n=20 students, n=5 instructors) suggests calibrated confidences improve trust alignment (r=0.62, p<0.001) and instructors agree with abstention recommendations 92% of the time, we make NO claims about actual learning outcomes. This remains an open research question requiring randomized controlled trials. Our paper demonstrates **technical feasibility**, not pedagogical effectiveness. This distinction is critical to maintain community trust and prevent reviewer disappointment.

**Sample Size**: The primary test set contains 260 claims—smaller than large-scale benchmarks like FEVER (19,998 claims)—but fully expert-annotated with high inter-rater agreement, trading scale for quality.

### Reproducibility & Open Science

- **Code & Data**: All source code, CSClaimBench (1,045 annotated claims, CC-BY-4.0 licensed), and reproducibility scripts available on GitHub
- **Deterministic Protocols**: Cross-GPU reproducibility validated on A100, V100, RTX 4090 with fixed random seeds [0,1,2,3,4]
- **Reproduction Time**: 20 minutes on A100 GPU to regenerate all figures and tables
- **Docker Container**: Complete environment for reproducible deployment

### Relevance to IEEE Access

This work aligns with IEEE Access's scope by:
1. Advancing educational AI through rigorous calibration methodology
2. Demonstrating practical deployment feasibility (sub-100ms latency)
3. Providing open-source tools and benchmarks for the community
4. Addressing real pedagogical needs (honest uncertainty + hybrid workflows)
5. Establishing reproducibility best practices for ML in education

### No Conflicts of Interest

All authors have no financial or personal relationships with third parties that could inappropriately influence this work. The research was conducted independently without external funding beyond institutional support.

### Suggested Reviewers

1. Dr. [Expert in Calibration/Neural Network Uncertainty] — Known for work on ECE and temperature scaling
2. Dr. [Expert in Educational AI] — Specializes in intelligent tutoring systems and learning sciences
3. Dr. [Expert in Fact Verification] — Author of FEVER or comparable verification work
4. Dr. [Expert in Reproducibility in ML] — Strong track record advocating for reproducibility standards

### Request

We herewith submit our manuscript for peer review in IEEE Access. The paper has not been previously published or submitted elsewhere. All figures and tables are original works. We believe this work makes significant contributions to calibrated fact verification for educational deployment and would be of interest to the IEEE Access readership.

We are available to address any questions or provide additional information as needed.

---

Sincerely,

**Nidhibahen Patel** (Co-First Author)  
Software Engineer, Verizon  
IEEE Member  
(Conducted research while affiliated with Computer Science Education Technology Lab, Kennesaw State University)

**Soma Kiran Kumar Nellipudi** (Co-First Author)  
Senior Software Engineer, Incomm Payments  
Senior IEEE Member  
(Conducted research while affiliated with Computer Science Education Technology Lab, Kennesaw State University)

**Selena He** (Corresponding Author)  
Assistant Professor of Computer Science  
Founding Director, Computer Science Education Technology Lab  
Kennesaw State University, GA USA  
Email: she4@kennesaw.edu  
Phone: [Contact]  

---

**Manuscript Details**:
- Total pages: ~40 (with appendices)
- Main text: 15 pages
- Appendices: 8 sections (E.1-E.8)
- Figures: 6 main + 12 appendix
- Tables: 8 main + 15 appendix
- References: 65 total
