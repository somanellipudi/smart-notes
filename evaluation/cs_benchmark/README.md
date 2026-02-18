# CS Claim Verification Benchmark (CSClaimBench v1.0)

## Overview

CSClaimBench is a synthetic benchmark dataset for evaluating claim verification systems in computer science domains. It contains 185 examples covering 7 major CS topics with ground-truth labels for entailment, contradiction, and neutral relationships.

## Dataset Statistics

- **Total Examples**: 185
- **Domains**: 7 (Algorithms, Data Structures, Operating Systems, Databases, Distributed Systems, Networks, Compilers)
- **Examples per Domain**: Algorithms (35), Others (25 each)
- **Label Distribution**:
  - ENTAIL: ~103 (56%)
  - CONTRADICT: ~62 (34%)
  - NEUTRAL: ~20 (10%)

## Format

Each line in `csclaimbench_v1.jsonl` contains one JSON object with the following fields:

```json
{
  "doc_id": "unique_identifier",
  "domain_topic": "CS_domain_name",
  "source_text": "Lecture-style educational content",
  "claim": "Statement to verify against source",
  "gold_label": "ENTAIL|CONTRADICT|NEUTRAL",
  "evidence_span": {"start": int, "end": int}  // optional
}
```

### Field Descriptions

- **doc_id**: Unique identifier (domain prefix + sequential number)
- **domain_topic**: One of: Algorithms, DataStructures, OS, DB, Distributed, Networks, Compilers
- **source_text**: Educational paragraph describing CS concepts (100-300 words)
- **claim**: A statement that may be supported, contradicted, or unrelated to source
- **gold_label**:
  - `ENTAIL`: Claim is directly supported by source_text
  - `CONTRADICT`: Claim is contradicted by source_text
  - `NEUTRAL`: Claim is neither supported nor contradicted (not mentioned)
- **evidence_span** (optional): Character offsets indicating evidence location in source_text

## Domain Coverage

### Algorithms (35 examples)
- Graph traversal (BFS, DFS)
- Sorting algorithms (Quicksort, Mergesort, Heapsort, Counting sort, Radix sort)
- Graph algorithms (Dijkstra, Bellman-Ford, Kruskal, Prim)
- Search algorithms (Binary search)
- Algorithmic paradigms (Dynamic programming, Greedy algorithms)
- Topological sorting

### Data Structures (25 examples)
- Basic structures (Stack, Queue)
- Trees (BST, AVL, Red-Black, Trie)
- Heaps (Min-heap, Max-heap)
- Hash tables
- Advanced structures (Segment tree, Bloom filter, Union-find)

### Operating Systems (25 examples)
- Process scheduling (Round-robin, SJF)
- Deadlock (Necessary conditions, prevention)
- Memory management (Paging, Virtual memory)
- Page replacement (LRU)
- Synchronization (Semaphores, Monitors)
- Context switching
- Threads
- File systems

### Databases (25 examples)
- ACID properties
- Normalization (1NF, 2NF, 3NF)
- Indexing (B-tree, Hash indexes)
- SQL operations (Joins)
- Transaction isolation levels
- Concurrency control (2PL)
- NoSQL databases
- Replication and sharding

### Distributed Systems (25 examples)
- CAP theorem
- Consensus algorithms (Paxos, Raft)
- Two-phase commit
- Vector clocks
- Eventual consistency
- Consistent hashing
- Byzantine fault tolerance
- Gossip protocols
- MapReduce

### Networks (25 examples)
- Transport protocols (TCP, UDP)
- IP addressing (IPv4, IPv6)
- DNS
- HTTP/HTTPS
- Routing protocols (RIP, OSPF, BGP)
- Subnetting
- Firewalls and VPNs
- CDNs
- OSI model
- Network protocols (ARP, DHCP)

### Compilers (25 examples)
- Lexical analysis
- Syntax analysis (Parsing, LL, LR)
- Semantic analysis
- Intermediate code generation
- Code optimization
- Register allocation
- Type systems
- Garbage collection
- Abstract syntax trees

## Generation Methodology

### Synthetic Data Creation

This dataset was **synthetically generated** using templated educational content to avoid copyright concerns. The generation process involved:

1. **Source Text Creation**: Wrote concise educational paragraphs (lecture-style) explaining CS concepts
2. **Claim Generation**:
   - **ENTAIL**: Extracted factual statements directly from source text
   - **CONTRADICT**: Modified key facts (negation, swapped terms, wrong complexity classes)
   - **NEUTRAL**: Created related but unmentioned statements about the topic
3. **Evidence Annotation**: Identified character spans for entailed/contradicted claims

### Design Principles

- **Educational Realism**: Source texts mimic lecture notes or textbook explanations
- **Factual Accuracy**: Content reflects standard CS curriculum knowledge
- **Label Clarity**: Labels are unambiguous with clear evidence or contradiction
- **Difficulty Balance**: Mix of simple definitional claims and nuanced technical claims
- **Domain Diversity**: Broad coverage across theoretical and practical CS areas

### Quality Controls

- All source texts fact-checked against standard CS references
- Claims verified for logical consistency with labels
- Evidence spans manually validated for accuracy
- No overlapping examples between domains

## Intended Use

### Primary Applications

1. **Claim Verification Evaluation**: Benchmark verification systems on technical content
2. **Evidence Retrieval Testing**: Assess evidence retrieval with ground-truth spans
3. **Domain Transfer Analysis**: Test generalization across CS subdomains
4. **Educational AI Tools**: Validate fact-checking in learning applications

### Limitations

- **Synthetic Nature**: Does not capture natural language variation in real student questions
- **Domain Scope**: Limited to 7 CS areas; does not cover all computer science
- **Complexity**: Focuses on fundamental concepts; limited coverage of advanced topics
- **Language**: English only
- **Label Imbalance**: More ENTAIL examples than CONTRADICT/NEUTRAL (reflects typical verification scenarios)

## Citation

If you use this benchmark in your research, please cite:

```
Smart-Notes CS Claim Verification Benchmark (CSClaimBench v1.0)
Generated: 2026
Purpose: Evaluating claim verification systems on computer science educational content
```

## License

This dataset is released for research and educational purposes. The synthetic content is original and not subject to copyright restrictions.

## Version History

- **v1.0** (2026-02-08): Initial release with 525 examples across 7 CS domains

## Contact

For questions, corrections, or suggestions, please open an issue in the Smart-Notes repository.
