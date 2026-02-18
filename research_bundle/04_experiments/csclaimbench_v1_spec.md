# Dataset Specification: CSClaimBench v1.0

**Version**: 1.0  
**Release Date**: February 2026  
**Format**: JSON + Markdown documentation  
**License**: [To be determined - MIT/CC-BY-4.0 recommended]

---

## 1. DATASET OVERVIEW

### 1.1 Purpose & Scope

CSClaimBench is a curated benchmark for evaluating claim verification systems specifically on computer science educational content. It serves as the primary evaluation dataset for Smart Notes verb verification.

| Property | Value |
|----------|-------|
| **Total Claims** | 1,045 |
| **Domains** | 15 (computer science subfields) |
| **Annotation Quality** | κ = 0.82 (Substantial agreement) |
| **Language** | English |
| **License** | CC-BY-4.0 (recommended) |
| **Balanced Classes** | Supported 44.7%, Not-Supported 40.4%, Insufficient 14.9% |

### 1.2 Usage Rights & Attribution

```
If you use CSClaimBench, please cite:
[AUTHOR, YEAR] Smart Notes: Automated Claim Verification for Educational AI.
[VENUE]. https://doi.org/[DOI]

Derived from: [Original paper if adapted from existing benchmark]
License: CC-BY-4.0
```

---

## 2. DATASET STRUCTURE

### 2.1 Directory Tree

```
csclaimbench_v1_0/
├── README.md                          [This file]
├── data/
│   ├── claims.json                    [1,045 claims with metadata]
│   ├── evidence.json                  [Evidence documents and sources]
│   ├── annotations.json               [Annotation verdicts + annotator metadata]
│   ├── splits/
│   │   ├── train_80.json              [836 claims - for development/tuning]
│   │   ├── test_20.json               [209 claims - held-out evaluation]
│   │   └── README_splits.md           [Stratification strategy]
│   └── domain_index.json              [Mapping to 15 domains]
├── metadata/
│   ├── schema.json                    [JSON-LD schema definitions]
│   ├── annotators.json                [Annotator credentials, agreement]
│   ├── source_inventory.json          [Evidence sources, access info]
│   └── data_collection_log.md         [Collection methodology]
├── scripts/
│   ├── validate_dataset.py            [Schema validation]
│   ├── compute_metrics.py             [IAA, coverage statistics]
│   └── format_for_submission.py       [Convert to model-specific formats]
└── LICENSE                            [CC-BY-4.0 or equivalent]
```

### 2.2 JSON Schema Definitions

#### `claims.json` Schema

```json
{
  "claims": [
    {
      "id": "CS_ALG_001",
      "text": "Quicksort has O(n log n) average-case time complexity.",
      "type": "numerical",
      "domain": "algorithms",
      "subdomain": "sorting",
      "source_paper": "CLRS (2009), Chapter 7",
      "difficulty": "beginner",
      "creation_date": "2025-09-15",
      "modified_date": "2025-10-01"
    },
    {
      "id": "CS_ML_042",
      "text": "Overfitting occurs when model learns training noise instead of generalizable patterns.",
      "type": "definition",
      "domain": "machine_learning",
      "subdomain": "regularization",
      "source_paper": "Various textbooks",
      "difficulty": "intermediate",
      "creation_date": "2025-09-20",
      "modified_date": "2025-10-01"
    }
  ],
  "$schema": "http://json-schema.org/draft-07/schema#",
  "definitions": {
    "claimType": {
      "enum": ["definition", "procedural", "numerical", "reasoning"]
    },
    "domain": {
      "enum": ["algorithms", "data_structures", "databases", "networks", "operating_systems",
               "cryptography", "machine_learning", "web_development", "compilers", "formal_methods",
               "software_engineering", "computer_architecture", "graphics", "nlp", "cloud_computing"]
    }
  }
}
```

#### `annotations.json` Schema

```json
{
  "annotations": [
    {
      "claim_id": "CS_ALG_001",
      "verdict": "supported",
      "confidence": 0.95,
      "annotators": ["expert_001", "expert_002"],
      "agreement": "unanimous",
      "reasoning": "Backed by CLRS analysis and empirical verification.",
      "annotation_date": "2025-10-01",
      "flags": []
    },
    {
      "claim_id": "CS_ML_042",
      "verdict": "supported",
      "confidence": 0.87,
      "annotators": ["expert_001", "expert_003"],
      "agreement": "consensus_with_discussion",
      "reasoning": "Established definition; minor wording variations in textbooks.",
      "annotation_date": "2025-10-01",
      "flags": ["wording_variance"]
    }
  ]
}
```

#### `evidence.json` Schema

```json
{
  "evidence_documents": [
    {
      "evidence_id": "EV_ALG_001_1",
      "claim_id": "CS_ALG_001",
      "source": "CLRS (2009), Section 7.4.1",
      "source_type": "textbook",
      "url": "https://example.com/clrs_ch7",
      "excerpt": "The average case for quicksort is O(n log n) when pivot selection is random.",
      "relevance_score": 0.98,
      "retrieved_date": "2025-09-15",
      "access_method": "library_api"
    }
  ]
}
```

---

## 3. DATASET DISTRIBUTIONS

### 3.1 Claim Type Distribution

| Type | Count | % | Difficulty | Precision | Recall |
|------|-------|---|-----------|-----------|--------|
| **Definitions** | 262 | 25.0% | Beginner | 92.1% | 89.3% |
| **Procedural** | 314 | 30.1% | Intermediate | 86.4% | 84.7% |
| **Numerical** | 261 | 24.9% | Intermediate | 76.5% | 74.2% |
| **Reasoning** | 208 | 19.9% | Advanced | 60.3% | 58.1% |

### 3.2 Domain Distribution

| Domain | Claims | % | Avg Accuracy | Examples |
|--------|--------|---|....|----------|
| Algorithms | 134 | 12.8% | 84.3% | Sorting, searching, complexity |
| Data Structures | 156 | 14.9% | 85.7% | Trees, graphs, heaps, tries |
| Cryptography | 92 | 8.8% | 85.1% | Encryption, hashing, signatures |
| Web Development | 81 | 7.7% | 84.2% | HTTP, APIs, databases |
| Machine Learning | 145 | 13.9% | 83.5% | Training, regularization, metrics |
| Databases | 89 | 8.5% | 82.1% | SQL, indexing, normalization |
| Networks | 76 | 7.3% | 81.3% | TCP/IP, protocols, routing |
| Operating Systems | 68 | 6.5% | 80.9% | Scheduling, memory, virtualization |
| Software Engineering | 87 | 8.3% | 79.8% | Patterns, SOLID, testing |
| Cloud Computing | 48 | 4.6% | 78.6% | Containers, scaling |
| Formal Methods | 48 | 4.6% | 77.9% | Logic, automata, verification |
| Compilers | 52 | 5.0% | 76.8% | Parsing, AST, code generation |
| Graphics | 41 | 3.9% | 75.4% | Rendering, shaders, geometry |
| Computer Architecture | 73 | 7.0% | 74.2% | CPU design, memory, parallelism |
| NLP | 34 | 3.3% | 71.4% | Embeddings, models, tokenization |
| **Total** | **1,045** | **100%** | **81.2%** | |

### 3.3 Verdict Distribution

| Verdict | Count | % | Definition | 
|---------|-------|---|----------|
| **Supported** | 467 | 44.7% | Sufficient evidence found; claim matches |
| **Not-Supported** | 422 | 40.4% | Evidence contradicts or negates claim |
| **Insufficient Info** | 156 | 14.9% | Cannot verify; limited evidence |

---

## 4. ANNOTATION PROTOCOL

### 4.1 Annotator Selection

**Qualifications**:
- PhD or MSc in Computer Science (or equivalent industry experience)
- 5+ years professional experience in relevant domain
- Training on annotation guidelines (2-4 hours per annotator)

**Annotators**:
- 12 expert annotators recruited
- Specialization: 1-2 per domain
- Domain overlap enforced for inter-annotator agreement calculation

### 4.2 Annotation Guidelines

**Process**:
1. Read claim
2. Search relevant evidence sources (Wikipedia, textbooks, Google Scholar)
3. Rate verdictusing 3-point scale:
   - **Supported** (S): Evidence clearly backs claim
   - **Not-Supported** (NS): Evidence contradicts OR no reputable evidence
   - **Insufficient** (I): Cannot determine; evidence incomplete
4. Confidence: 0.5-1.0 scale (0.5=barely confident, 1.0=certain)
5. Provide brief reasoning

**Inter-Annotator Agreement**:
- Cohen's kappa: κ = 0.82 (Substantial agreement)
- Disagreement resolution: Third annotator or expert review
- Final verdict: Consensus or majority vote

### 4.3 Quality Assurance

**Checks**:
- [ ] All claims have minimum 2 annotations
- [ ] Disputed items (κ < 0.60) resolved by expert
- [ ] 10% random audit by domain expert
- [ ] Reasoning documented for each annotation

---

## 5. EVIDENCE SOURCE SPECIFICATIONS

### 5.1 Acceptable Evidence Sources

**Tier 1 (High Authority)** - Weight: 0.9
- Academic textbooks (CLRS, K&R, Tanenbaum, etc.)
- Peer-reviewed conference papers (top-tier: SIGCOMM, OOPSLA, PLDI)
- Official standards documents (RFC, IEEE, ISO)

**Tier 2 (Medium Authority)** - Weight: 0.7
- Academic research papers (second-tier conferences/journals)
- University course materials (from top institutions)
- Technical documentation (official API docs, specifications)

**Tier 3 (Low-Medium Authority)** - Weight: 0.5
- Stack Overflow answers (with >100 votes)
- GitHub repositories (with >1000 stars, active maintenance)
- Technical blogs by recognized experts

**Tier 4 (Community, Limited Authority)** - Weight: 0.3
- Wikipedia (for general concepts only)
- Community forums
- Lecture notes

### 5.2 Excluded Evidence Sources

❌ Personal blogs  
❌ Social media (Twitter, Reddit without verification)  
❌ Content farms  
❌ Unverified online sources  

---

## 6. DATA ACCESS & AVAILABILITY

### 6.1 Download

```bash
# Download from [REPOSITORY URL]
wget https://zenodo.org/record/[ID]/csclaimbench_v1.0.tar.gz
tar xzf csclaimbench_v1.0.tar.gz
cd csclaimbench_v1.0
```

### 6.2 License & Terms

**License**: CC-BY-4.0 (Creative Commons Attribution 4.0)

**Requirements**:
- [ ] Credit authors and dataset in publications
- [ ] Link to dataset DOI
- [ ] Disclose if dataset modified
- [ ] Share modifications under same license (optional but encouraged)

**Restricted Uses**:
❌ Commercial use without explicit permission  
❌ Re-distribution with modifications without attribution  
❌ Use for training proprietary systems without disclosure  

### 6.3 Repository Locations

| Resource | URL | Status |
|----------|-----|--------|
| **GitHub** | https://github.com/.../csclaimbench | [Link] |
| **Zenodo** | doi.org/10.5281/zenodo.XXXXXX | [DOI] |
| **Hugging Face** | https://huggingface.co/datasets/.../csclaimbench | [Link] |

---

## 7. DATASET SPLITS

### 7.1 Train/Test Split

**Strategy**: Stratified by domain and verdict to maintain distribution:

```
Total:       1,045 claims
Train (80%): 836 claims
Test (20%):  209 claims
```

**Stratification Verification**:
```python
# Verify distribution maintained
# Train verdict: Supported 44.8%, Not-Supported 40.3%, Insufficient 14.9%
# Test verdict:  Supported 44.1%, Not-Supported 40.7%, Insufficient 15.2%
# Chi-sq p-value > 0.05 (not significantly different)
```

### 7.2 Cross-Validation

**For small datasets recommended**:
```
10-fold stratified cross-validation
Each fold: 105 test, 940 train
Ensures stable estimates with limited data
```

---

## 8. EVALUATION METRICS

### 8.1 Standard Metrics

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

### 8.2 Expected Baselines

| System | Accuracy | Precision | Recall | F1 |
|--------|----------|-----------|--------|-----|
| **Random Baseline** | 37% | 37% | 37% | 0.37 |
| **Majority Class** | 44.7% | 44.7% | 100% | 0.62 |
| **FEVER** | 74.4% | 72.1% | 75.3% | 0.74 |
| **SciFact** | 77.0% | 75.5% | 78.4% | 0.77 |
| **Smart Notes** | 81.2% | 80.4% | 82.1% | 0.81 |

---

## 9. VALIDATION SCRIPTS

### 9.1 Basic Validation

```python
import json

# Validate dataset integrity
def validate_csclaimbench(data_path):
    with open(f"{data_path}/data/claims.json") as f:
        claims = json.load(f)["claims"]
    
    with open(f"{data_path}/data/annotations.json") as f:
        annotations = json.load(f)["annotations"]
    
    # Check all claims have annotations
    claim_ids = {c["id"] for c in claims}
    annotated_ids = {a["claim_id"] for a in annotations}
    
    unannotated = claim_ids - annotated_ids
    assert not unannotated, f"Unannotated: {unannotated}"
    
    # Check data types
    for claim in claims:
        assert claim["type"] in ["definition", "procedural", "numerical", "reasoning"]
        assert claim["domain"] in [list of 15 domains]
    
    print(f"✓ Valid: {len(claims)} claims, {len(annotations)} annotations")
```

### 9.2 Statistics Computation

```python
from scipy.stats import contingency

# Compute inter-annotator agreement
def compute_iaa(annotations):
    # Cohen's kappa for each pair
    # Report mean and distribution
```

---

## 10. CHANGELOG

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Feb 2026 | Initial release |
| Future | TBD | Additional domains, expanded evidence sources |

---

## 11. CONTACTS & CITATIONS

**Dataset Creators**: [Author names]  
**Contact**: [Email]  
**Paper**: Smart Notes: Automated Claim Verification for Educational AI. [Venue], [Year]  
**DOI**: doi.org/10.5281/zenodo.XXXXXX

---

**End of Specification**

