# CS Benchmark Dataset

**Description**: Synthetic, non-copyright computer science benchmark for verifying claim validation performance.

**Format**: JSONL (JSON Lines), one claim per line

**Schema**:
```json
{
  "doc_id": "algo_001",
  "domain_topic": "algorithms.sorting",
  "source_text": "...",
  "generated_claim": "...",
  "gold_label": "VERIFIED|LOW_CONFIDENCE|REJECTED",
  "evidence_span": "..."
}
```

**Fields**:
- `doc_id`: Unique document identifier (domain_XXX pattern)
- `domain_topic`: CS domain and topic (e.g., "algorithms.sorting", "datastructures.hashtable")
- `source_text`: Synthetic source material (what evidence retriever would find)
- `generated_claim`: AI-generated claim to verify
- `gold_label`: Ground truth verification status
  - `VERIFIED`: Claim is supported by evidence
  - `LOW_CONFIDENCE`: Claim is partially supported or ambiguous
  - `REJECTED`: Claim is contradicted or unsupported by evidence
- `evidence_span`: Relevant text snippet from source_text (empty if rejected)

**Coverage**:
- **Algorithms**: sorting, search, dynamic programming (7 examples)
- **Data Structures**: hash tables, trees, graphs (5 examples)
- **Complexity Theory**: NP-hardness, relationships (2 examples)
- **Networking**: TCP/UDP protocols (2 examples)
- **Security**: encryption, hashing (2 examples)
- **Databases**: indexing, SQL queries (2 examples)
- **Machine Learning**: optimization, regression (2 examples)
- **Compilers**: parsing, optimization (2 examples)

**Total**: 20 claims (balanced: 11 VERIFIED, 4 LOW_CONFIDENCE, 5 REJECTED)

**Usage**:

```python
from src.evaluation.cs_benchmark_runner import CSBenchmarkRunner

runner = CSBenchmarkRunner(
    dataset_path="evaluation/cs_benchmark/cs_benchmark_dataset.jsonl"
)
results = runner.run()
print(results.to_csv())
```

**Extensions**:
- Add more domains (networking, security, databases, ML)
- Create difficulty stratified subsets (beginner, intermediate, expert)
- Generate adversarial claims (near-misses, ambiguous cases)
- Add source document snippets for realistic detection scenarios
