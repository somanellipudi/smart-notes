# Smart Notes Architecture Flow Map

**Repository**: https://github.com/somanellipudi/smart-notes  
**Purpose**: Educational AI system with multi-modal ingestion and evidence-grounded verification  
**Last Updated**: 2026-02-17

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Call Flow Diagram](#call-flow-diagram)
3. [Detailed Call Chain](#detailed-call-chain)
4. [Data Objects Map](#data-objects-map)
5. [Nondeterminism Sources](#nondeterminism-sources)
6. [Ingestion Failure Modes](#ingestion-failure-modes)
7. [Integration Points](#integration-points)
8. [Evaluation & Research](#evaluation--research)

---

## System Overview

Smart Notes is a research-grade educational AI system that processes classroom materials (PDFs, audio, text) and generates verifiable study guides. The system supports two modes:

- **Standard Mode**: Traditional LLM-based study guide generation
- **Verifiable Mode**: Evidence-grounded claim generation with explicit verification

The verification pipeline treats AI outputs as testable hypotheses, retrieving evidence and scoring confidence to detect hallucinations.

---

## Call Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          USER INPUT (app.py)                            │
│  - Text, PDF uploads, Audio files, YouTube URLs, Web articles          │
└──────────────────────────┬──────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      INGESTION & PREPROCESSING                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                 │
│  │ PDF Ingest   │  │ Audio Trans  │  │ URL Scraper  │                 │
│  │ pdf_ingest.py│  │ transcription│  │ url_ingest.py│                 │
│  │ + OCR fallbak│  │ .py          │  │              │                 │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                 │
│         │                  │                  │                         │
│         └──────────────────┴──────────────────┘                         │
│                           │                                             │
│                           ▼                                             │
│              ┌─────────────────────────┐                                │
│              │ Text Cleaning/Chunking  │                                │
│              │ text_cleaner.py         │                                │
│              │ text_processing.py      │                                │
│              └────────────┬────────────┘                                │
└───────────────────────────┼─────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│               SESSION PROCESSING (process_session)                      │
│  app.py:368 → Creates session ID, combines inputs                       │
└──────────────────────────┬──────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│          VERIFIABLE PIPELINE WRAPPER (verifiable_pipeline.py)           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Standard Mode?                                                  │   │
│  │   → ReasoningPipeline.process()                                 │   │
│  │   → Returns ClassSessionOutput                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Verifiable Mode? → _process_verifiable()                        │   │
│  │                                                                  │   │
│  │  1. URL Ingestion (if URLs provided)                            │   │
│  │     url_ingest.py:ingest_urls()                                 │   │
│  │     → Scrapes YouTube/articles                                  │   │
│  │     → chunk_url_sources()                                       │   │
│  │                                                                  │   │
│  │  2. Text Quality Assessment                                     │   │
│  │     text_quality.py:compute_text_quality()                      │   │
│  │     → Warns if input too short/low quality                      │   │
│  │                                                                  │   │
│  │  3. Build Evidence Store                                        │   │
│  │     evidence_builder.py:build_session_evidence_store()          │   │
│  │     → evidence_store.py:EvidenceStore()                         │   │
│  │     → Chunks all inputs (session + URLs)                        │   │
│  │     → Generates embeddings (embedding_provider.py)              │   │
│  │     → Builds FAISS index                                        │   │
│  │                                                                  │   │
│  │  4. Run Standard Pipeline (baseline)                            │   │
│  │     pipeline.py:ReasoningPipeline.process()                     │   │
│  │     → LLM generates ClassSessionOutput                          │   │
│  │                                                                  │   │
│  │  5. Extract Claims                                              │   │
│  │     extractor.py:ClaimExtractor.extract()                       │   │
│  │     → Converts ClassSessionOutput → LearningClaim objects       │   │
│  │     → ClaimType: DEFINITION, EQUATION, EXAMPLE, etc.            │   │
│  │                                                                  │   │
│  │  6. Retrieve Evidence                                           │   │
│  │     claim_rag.py:retrieve_evidence_for_claim()                  │   │
│  │     → For each claim: query EvidenceStore                       │   │
│  │     → Compute similarity scores                                 │   │
│  │     → Attach EvidenceItem objects                               │   │
│  │                                                                  │   │
│  │  7. Validate Claims                                             │   │
│  │     validator.py:ClaimValidator.validate_collection()           │   │
│  │     → Check evidence sufficiency                                │   │
│  │     → Compute consistency scores (NLI checks via Ollama)        │   │
│  │     → Set VerificationStatus: VERIFIED/LOW_CONFIDENCE/REJECTED  │   │
│  │     → Tag RejectionReason if rejected                           │   │
│  │                                                                  │   │
│  │  8. Build Claim Graph                                           │   │
│  │     claim_graph.py:ClaimGraph()                                 │   │
│  │     → NetworkX graph: claims + evidence nodes                   │   │
│  │     → Compute GraphMetrics (redundancy, diversity, conflicts)   │   │
│  │                                                                  │   │
│  │  9. Calculate Metrics                                           │   │
│  │     verifiability_metrics.py:calculate_metrics()                │   │
│  │     → Rejection rates, traceability, confidence stats           │   │
│  │                                                                  │   │
│  │  10. Return Output + Metadata                                   │   │
│  │     → ClassSessionOutput (filtered by rejected claims)          │   │
│  │     → verifiable_metadata dict (claims, graph, metrics)         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└──────────────────────────┬──────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      UI DISPLAY & EXPORT (app.py)                       │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Notes Tab:                                                       │   │
│  │   - Display ClassSessionOutput sections                         │   │
│  │   - Export as Markdown                                          │   │
│  │                                                                  │   │
│  │ Verification Tab: (if verifiable_mode)                          │   │
│  │   - Verification summary (verified/rejected counts)             │   │
│  │   - Evidence statistics                                         │   │
│  │   - Rejection reasons breakdown                                 │   │
│  │   - Claim-evidence table                                        │   │
│  │   - Export as JSON/PDF (report_exporter.py)                     │   │
│  │                                                                  │   │
│  │ Diagnostics Expander:                                           │   │
│  │   - Extraction methods, char counts, retrieval stats            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Detailed Call Chain

### Entry Point: `app.py`

**Main Functions:**
- `app.py:process_session()` (line 368)
  - Combines text inputs, file uploads, audio
  - Creates session ID
  - Calls `VerifiablePipelineWrapper.process()`

**Handler Functions:**
- `generate_notes_handler()`: Triggered by "Generate Notes" button
- `run_verification_handler()`: Triggered by "Run Verification" button (can work independently of notes generation)

### Ingestion Pipeline

**PDF Ingestion**: `src/preprocessing/pdf_ingest.py`
- `extract_pdf_text(uploaded_file, ocr)` → `(text, metadata)`
  1. Try `pdfplumber` extraction
  2. Assess quality: `_assess_extraction_quality()`
  3. If poor quality → OCR fallback: `_extract_with_ocr_pymupdf()`
     - Render pages with PyMuPDF (`fitz`)
     - OCR with Tesseract or EasyOCR (`image_ocr.py`)
  4. Clean text: `text_cleaner.clean_extracted_text()`

**Audio Transcription**: `src/audio/transcription.py`
- `transcribe_audio(audio_file, language, model_size)` → `text`
  - Uses Whisper model via `openai-whisper` package
  - Fallback to mock transcription if Whisper unavailable

**URL Ingestion**: `src/retrieval/url_ingest.py`
- `ingest_urls(urls)` → `List[URLSource]`
  - YouTube: extract transcript via `youtube_transcript_api`
  - Articles: scrape with `BeautifulSoup` + `requests`
- `chunk_url_sources(sources, chunk_size=500, overlap=50)` → `List[TextChunk]`

**Text Cleaning**: `src/preprocessing/text_cleaner.py`
- `clean_extracted_text(raw_text)` → `(cleaned_text, CleanDiagnostics)`
  - Removes boilerplate: unit/chapter headers, scan watermarks, page numbers
  - Detects repeated lines across pages (configurable threshold)
  - Protects code-like lines (regex patterns for algorithms, Big-O)

### Reasoning Pipeline

**Standard Mode**: `src/reasoning/pipeline.py`
- `ReasoningPipeline.process(combined_content, equations, external_context, session_id, output_filters)` → `ClassSessionOutput`
  - Builds prompt from templates
  - Calls LLM (OpenAI/Ollama via `llm_provider.py`)
  - Parses structured JSON output into `ClassSessionOutput`

**Verifiable Mode**: `src/reasoning/verifiable_pipeline.py`
- `VerifiablePipelineWrapper._process_verifiable()` → `(ClassSessionOutput, verifiable_metadata)`

#### Step-by-Step Flow:

**Step 0**: URL Ingestion (if URLs provided)
- `url_ingest.ingest_urls(urls)` → `url_sources`
- `url_ingest.chunk_url_sources(url_sources)` → `url_chunks`

**Step 0.25**: Text Quality Assessment
- `text_quality.compute_text_quality(combined_content)` → `QualityReport`
- Checks: char count, word count, unique words, avg word length
- Warns if insufficient input

**Step 1**: Build Evidence Store
- `evidence_builder.build_session_evidence_store(session_id, combined_content, url_chunks, embedding_provider)` → `EvidenceStore`
  - `evidence_store.py:EvidenceStore.__init__(session_id, embedding_dim=384)`
  - Chunks session content (chunk_size=200, overlap=50)
  - Adds URL chunks
  - Generates embeddings: `embedding_provider.py:embed_texts(chunks)` → `np.ndarray`
    - Uses `sentence-transformers` (default: `all-MiniLM-L6-v2`)
  - Builds FAISS index: `EvidenceStore.build_index(embeddings)`

**Step 2**: Run Standard Pipeline (baseline)
- `pipeline.py:ReasoningPipeline.process()` → `baseline_output`
- Generates full `ClassSessionOutput` with topics, concepts, equations, etc.

**Step 3**: Extract Claims
- `extractor.py:ClaimExtractor.extract(baseline_output, session_id)` → `ClaimCollection`
  - Converts each concept → `LearningClaim(claim_type=DEFINITION)`
  - Converts equations → `LearningClaim(claim_type=EQUATION)`
  - Converts examples → `LearningClaim(claim_type=EXAMPLE)`
  - Converts misconceptions → `LearningClaim(claim_type=MISCONCEPTION)`
  - Initial status: `VerificationStatus.REJECTED` (awaiting evidence)

**Step 4**: Retrieve Evidence
- `claim_rag.py:ClaimRAG.retrieve_evidence_for_claim(claim, evidence_store)` → `List[EvidenceItem]`
  - Embeds claim text: `embedding_provider.embed_texts([claim.claim_text])`
  - Queries FAISS index: `evidence_store.search(claim_embedding, top_k=20)`
  - Filters by similarity threshold (τ = 0.2)
  - Applies reranking: `reranker.rerank(claim, candidates)` (if enabled)
  - Attaches `EvidenceItem` objects to claim

**Step 5**: Validate Claims
- `validator.py:ClaimValidator.validate_collection(claim_collection)` → `ClaimCollection`
  - For each claim: `validate_claim(claim)` → `VerificationStatus`
    - Check evidence count: `>= min_evidence_count`
    - Check consistency: `consistency_score >= rejected_threshold`
      - NLI consistency via `concept_agent.py:ConceptAgent` (uses Ollama by default)
    - Compute confidence: weighted score from similarity + consistency + graph metrics
    - Set status:
      - `confidence >= 0.7` → `VERIFIED`
      - `0.3 <= confidence < 0.7` → `LOW_CONFIDENCE`
      - `confidence < 0.3` → `REJECTED`
    - Tag rejection reason: `NO_EVIDENCE`, `LOW_SIMILARITY`, `INSUFFICIENT_CONFIDENCE`, etc.

**Step 6**: Build Claim Graph
- `claim_graph.py:ClaimGraph(claims)` → `ClaimGraph`
  - Builds NetworkX `DiGraph`
  - Nodes: claims + evidence
  - Edges: claim → evidence (support relationship)
  - Computes `GraphMetrics`:
    - `redundancy`: avg evidence per claim
    - `diversity`: proportion of different source types
    - `support_depth`: avg path length from evidence
    - `conflict_count`: contradictory evidence pairs

**Step 7**: Calculate Metrics
- `verifiability_metrics.py:VerifiabilityMetrics.calculate_metrics(claim_collection, graph_metrics)` → `Dict`
  - Rejection rate, verification rate, confidence stats
  - Evidence sufficiency metrics
  - Traceability rate (% claims with evidence)
  - Rejection reason histogram

**Step 8**: Filter Output
- Creates filtered `ClassSessionOutput` with only verified/low-confidence claims
- Rejected claims removed from final output

**Step 9**: Return Results
- Returns `(ClassSessionOutput, verifiable_metadata)`
- `verifiable_metadata` contains:
  - `claim_collection`: full ClaimCollection object
  - `graph_metrics`: GraphMetrics dict
  - `verifiability_metrics`: evaluation metrics
  - `evidence_stats`: retrieval statistics
  - `url_ingestion_summary`: URL ingestion results

### Export Pipeline

**Markdown Export**: `app.py`
- Serializes `ClassSessionOutput` to Markdown format
- Sections: Summary, Topics, Concepts, Examples, FAQs, etc.

**JSON Export**: `src/exporters/report_exporter.py`
- `export_report_json(verifiable_metadata)` → `json_string`
- Includes claims, evidence, metrics, graph data

**PDF Export**: `src/exporters/report_exporter.py`
- `export_report_pdf(verifiable_metadata)` → `pdf_bytes`
- Uses PyMuPDF (`fitz`) to render claims table, evidence, metrics

---

## Data Objects Map

### Core Schemas

#### `ClassSessionOutput` (`src/schema/output_schema.py`)

**Purpose**: Structured study guide output (standard format)

**Fields**:
```python
session_id: str                              # Unique session identifier
class_summary: str                           # Brief overview
topics: List[Topic]                          # Major topics covered
key_concepts: List[Concept]                  # Key concepts
equation_explanations: List[EquationExplanation]
worked_examples: List[WorkedExample]
common_mistakes: List[Misconception]
faqs: List[FAQ]
real_world_connections: List[RealWorldConnection]
metadata: Dict[str, Any]                     # Session metadata
```

**Nested Objects**:
- `Topic`: `{name, summary, subtopics, learning_objectives, timestamp_range}`
- `Concept`: `{name, definition, prerequisites, difficulty_level}`
- `EquationExplanation`: `{equation, explanation, variables, applications}`
- `WorkedExample`: `{problem, solution, key_concepts, common_mistakes}`
- `Misconception`: `{misconception, explanation, correct_understanding, related_concepts}`
- `FAQ`: `{question, answer, related_concepts, difficulty}`
- `RealWorldConnection`: `{concept, application, example, impact}`

---

#### `LearningClaim` (`src/claims/schema.py`)

**Purpose**: Evidence-grounded factual assertion for verifiable mode

**Fields**:
```python
claim_id: str                                # Unique claim ID (UUID)
claim_type: ClaimType                        # DEFINITION | EQUATION | EXAMPLE | MISCONCEPTION
claim_text: str                              # The factual statement
evidence_ids: List[str]                      # IDs of supporting evidence
evidence_objects: List[EvidenceItem]         # Full evidence objects
confidence: float                            # 0.0-1.0 confidence score
status: VerificationStatus                   # VERIFIED | LOW_CONFIDENCE | REJECTED
rejection_reason: Optional[RejectionReason]  # Coded rejection reason
dependency_requests: List[DependencyRequest] # Undefined prerequisites
metadata: Dict[str, Any]                     # Additional claim metadata
created_at: datetime
validated_at: Optional[datetime]
```

**Enums**:
- `ClaimType`: DEFINITION, EQUATION, EXAMPLE, MISCONCEPTION, ALGORITHM_STEP, COMPLEXITY, INVARIANT
- `VerificationStatus`: VERIFIED, LOW_CONFIDENCE, REJECTED
- `RejectionReason`: NO_EVIDENCE, LOW_SIMILARITY, INSUFFICIENT_CONFIDENCE, CONFLICT, LOW_CONSISTENCY, etc.

---

#### `EvidenceItem` (`src/claims/schema.py`)

**Purpose**: Single piece of evidence supporting a claim

**Fields**:
```python
evidence_id: str                             # Unique evidence ID
source_id: str                               # Source identifier (filename, URL)
source_type: str                             # "notes" | "transcript" | "youtube" | "article"
snippet: str                                 # Text span (min 15 chars)
span_metadata: Dict[str, Any]                # {page, line, start_char, end_char}
similarity: float                            # Cosine similarity to claim (0-1)
reliability_prior: float                     # Source reliability (0-1)
timestamp: datetime
```

---

#### `EvidenceStore` (`src/retrieval/evidence_store.py`)

**Purpose**: Centralized evidence storage with FAISS indexing

**Internal Structure**:
```python
session_id: str
embedding_dim: int                           # Default: 384
evidence: List[Evidence]                     # All evidence chunks
evidence_by_id: Dict[str, Evidence]          # ID → Evidence mapping
faiss_index: faiss.Index                     # FAISS index for retrieval
index_built: bool
source_counts: Dict[str, int]                # Source type counts
total_chars: int
```

**`Evidence` dataclass**:
```python
evidence_id: str
source_id: str
source_type: str
text: str
chunk_index: int
char_start: int
char_end: int
metadata: Dict[str, Any]
embedding: Optional[np.ndarray]
```

---

#### `ClaimCollection` (`src/claims/schema.py`)

**Purpose**: Container for all claims in a session

**Fields**:
```python
session_id: str
claims: List[LearningClaim]
metadata: Dict[str, Any]
created_at: datetime
```

**Methods**:
- `calculate_statistics()` → rejection rates, avg confidence
- `filter_by_status(status)` → claims with given status
- `get_verified_claims()` → VERIFIED claims only

---

#### `GraphMetrics` (`src/claims/schema.py`)

**Purpose**: Graph-based metrics from claim-evidence network

**Fields**:
```python
redundancy: float                            # Avg evidence per claim
diversity: float                             # Proportion of different source types
support_depth: float                         # Avg path length from evidence
conflict_count: int                          # Contradictory evidence pairs
connected_components: int                    # Number of disconnected subgraphs
avg_claim_degree: float                      # Avg edges per claim node
```

---

#### `CleanDiagnostics` (`src/preprocessing/text_cleaner.py`)

**Purpose**: Text cleaning metadata

**Fields**:
```python
removed_lines_count: int
removed_by_regex: Dict[str, int]            # Regex name → count
removed_repeated_lines_count: int
top_removed_lines: List[str]
kept_lines_count: int
repeat_threshold_used: int
```

---

#### `QualityReport` (`src/preprocessing/text_quality.py`)

**Purpose**: Text quality assessment for input validation

**Fields**:
```python
char_count: int
word_count: int
unique_words: int
avg_word_length: float
quality_flags: List[str]                     # ["too_short", "repetitive", ...]
is_sufficient: bool
warnings: List[str]
```

---

### Metadata Flow

**Session Metadata** (`ClassSessionOutput.metadata`):
```python
{
    "extraction_method": "pdf_text" | "ocr_pymupdf_tesseract" | "ocr_easyocr",
    "num_pages": int,
    "chars_extracted": int,
    "cleaned_lines_removed": int,
    "cleaning_diagnostics": CleanDiagnostics,
    "llm_model": str,
    "llm_provider": "openai" | "ollama",
    "temperature": float,
    "fallback": bool
}
```

**Verifiable Metadata** (returned from verifiable pipeline):
```python
{
    "claim_collection": ClaimCollection,
    "graph_metrics": GraphMetrics,
    "verifiability_metrics": {
        "rejection_rate": float,
        "verification_rate": float,
        "avg_confidence": float,
        "rejection_reasons": Dict[str, int],
        "traceability_metrics": {...},
        "evidence_metrics": {...}
    },
    "evidence_stats": {
        "total_evidence": int,
        "source_counts": Dict[str, int],
        "avg_similarity": float,
        "min_similarity": float,
        "max_similarity": float
    },
    "url_ingestion_summary": {
        "total_urls": int,
        "successful": int,
        "failed": int,
        "total_chars": int,
        "urls": List[Dict]
    },
    "step_timings": Dict[str, float]
}
```

---

## Nondeterminism Sources

### 1. **Text Chunking**

**Location**: `src/retrieval/semantic_retriever.py`, `src/retrieval/evidence_builder.py`

**Source of Nondeterminism**:
- Chunk boundaries depend on `chunk_size` and `chunk_overlap` parameters
- Fixed-length chunking (char-based, not semantic)
- No sentence boundary awareness → chunks may split mid-sentence

**Parameters**:
- `chunk_size`: 200 characters (default for session content), 500 (for URL content)
- `chunk_overlap`: 50 characters

**Impact**: Different chunking → different evidence spans → different similarity scores

**Mitigation**:
- Use consistent `chunk_size`/`chunk_overlap` across runs
- Consider semantic chunking (sentence-based) for deterministic boundaries

---

### 2. **Embedding Generation**

**Location**: `src/retrieval/embedding_provider.py`

**Source of Nondeterminism**:
- Model initialization: random weight initialization (if training)
- GPU vs CPU execution: different floating-point precision
- Batch encoding: order of inputs may affect caching

**Model**: `sentence-transformers/all-MiniLM-L6-v2` (default)

**Impact**: Slight variations in embedding vectors → FAISS retrieval order changes

**Mitigation**:
- Use frozen pre-trained models (no training)
- Pin model version in `requirements.txt`
- Use deterministic CUDA operations (if GPU)

---

### 3. **FAISS Retrieval**

**Location**: `src/retrieval/evidence_store.py`

**Source of Nondeterminism**:
- FAISS `IndexFlatIP` (inner product): deterministic
- FAISS `IndexIVFFlat` (approximate): nondeterministic (random clustering)
- Tie-breaking: if multiple evidence have same similarity, order is arbitrary

**Current Index**: `IndexFlatIP` (deterministic, brute-force)

**Impact**: With approximate indexes, top-k results may vary across runs

**Mitigation**:
- Use flat indexes for reproducibility
- Set FAISS seed: `faiss.random.seed(42)` (newer API) or `faiss.set_random_seed(42)` (legacy)
- Sort results by (similarity, evidence_id) for tie-breaking

---

### 4. **LLM Sampling**

**Location**: `src/llm_provider.py`, `src/reasoning/pipeline.py`

**Source of Nondeterminism**:
- Temperature > 0: stochastic sampling
- OpenAI API: no seed parameter support (as of GPT-4)
- Ollama: seed parameter available but may not guarantee determinism across platforms

**Parameters**:
- `temperature`: default from `config.LLM_TEMPERATURE` (typically 0.3-0.7)
- `top_p`, `top_k`, `presence_penalty`, `frequency_penalty`

**Impact**: Different LLM outputs → different claims → different verification results

**Mitigation**:
- Set `temperature=0` for greedy decoding (reduces but doesn't eliminate nondeterminism)
- Use fixed prompt templates
- Log LLM responses for reproducibility audits

---

### 5. **Reranking Models**

**Location**: `src/retrieval/reranker.py` (if enabled)

**Source of Nondeterminism**:
- Cross-encoder models: GPU/CPU precision differences
- Batch processing: order of inputs may affect caching

**Model**: Configurable (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`)

**Impact**: Reranked evidence order affects top-k selection

**Mitigation**:
- Disable reranking for reproducibility (`RERANKER_ENABLED=false`)
- Pin reranker model version
- Use deterministic hardware (CPU-only)

---

### 6. **NLI Consistency Checks**

**Location**: `src/agents/concept_agent.py`, `src/claims/validator.py`

**Source of Nondeterminism**:
- Ollama/OpenAI calls for consistency scoring
- LLM sampling (temperature > 0)
- Prompt variations

**Impact**: Different consistency scores → different verification statuses

**Mitigation**:
- Use rule-based consistency checks (regex matching) instead of LLM
- Cache NLI results by (claim, evidence) pair
- Set `VERIFIABLE_CONSISTENCY_ENABLED=false` to skip LLM-based checks

---

### 7. **Graph Metrics Computation**

**Location**: `src/graph/claim_graph.py`

**Source of Nondeterminism**:
- NetworkX algorithms: some use randomization (e.g., layout algorithms)
- Tie-breaking in graph traversal (if multiple paths have same weight)

**Impact**: Minor variations in graph metrics (diversity, connected components)

**Mitigation**:
- Use deterministic algorithms (shortest path, degree centrality)
- Set NetworkX seed: `nx.set_random_state(42)`
- Sort nodes/edges before iteration

---

### 8. **Multi-Column PDF Layout**

**Location**: `src/preprocessing/pdf_ingest.py`

**Source of Nondeterminism**:
- PyMuPDF/pdfplumber column detection: heuristic-based
- OCR text block ordering: varies by OCR engine

**Impact**: Text extraction order differs → chunks have different content

**Mitigation**:
- Use layout-aware extraction (detect columns before chunking)
- Add robust column detection heuristics
- Manual column ordering hints in config

---

### 9. **Timestamp-Based IDs**

**Location**: Session IDs, evidence IDs

**Source of Nondeterminism**:
- UUIDs: random generation
- Timestamps: vary across runs

**Impact**: IDs differ across runs → cannot compare sessions directly

**Mitigation**:
- Use content-based hashing for deterministic IDs
- Seed UUID generation: `uuid.uuid5(namespace, name)`

---

## Ingestion Failure Modes

### 1. **Headers and Footers**

**Problem**:
- Repeated university names, course codes, page numbers
- Dominate evidence retrieval (high repetition → high chunk count)

**Current Handling**:
- `text_cleaner.py:repeated_line_detector()`: removes lines appearing on >30% of pages
- Regex patterns for unit/chapter/module headers (`BOILERPLATE_REGEX_RULES`)

**Failure Cases**:
- Custom header formats not in regex rules
- Headers with dynamic content (changing dates, pages)
- Multi-line headers

**Future Enhancement**:
- Statistical outlier detection (TF-IDF low-scoring patterns)
- User-configurable header/footer regions (top 5/bottom 5 lines)
- OCR confidence-weighted removal

---

### 2. **CamScanner and Scan Watermarks**

**Problem**:
- "Scanned with CamScanner", "Adobe Scan", "Genius Scan" text
- Embedded in every page → pollutes evidence

**Current Handling**:
- Regex removal: `r"\b(scanned\s+with|camscanner|adobe\s+scan)\b"` (line 109, config.py)

**Failure Cases**:
- Watermarks in image layer (not text layer) → OCR extracts them
- Non-English watermarks
- App-specific formats ("Scanned by John's iPhone")

**Future Enhancement**:
- OCR preprocessing: detect watermark regions by bounding box analysis
- Language-agnostic pattern matching
- Crowdsourced watermark database

---

### 3. **Multi-Column Layouts**

**Problem**:
- PDFs with 2+ columns (journals, textbooks)
- Text extraction reads left-to-right, top-to-bottom → columns interleaved
- Example: "Derivatives are... [column 2] ...used in physics"

**Current Handling**:
- None (assumes single-column layout)

**Failure Cases**:
- All multi-column PDFs
- Mixed layouts (some single, some multi)

**Future Enhancement**:
- Column detection: analyze text bounding boxes in PyMuPDF
- Re-order extraction by column boundaries
- Layout analysis with `pdfplumber.extract_words()` + clustering

---

### 4. **CID (Character Identifier) Text**

**Problem**:
- Garbled text in PDFs with embedded fonts: "(cid:123) (cid:456)"
- Unreadable → useless evidence

**Current Handling**:
- None (passed through as-is)

**Failure Cases**:
- PDFs with font encoding issues
- Scanned PDFs with OCR errors

**Detection**:
- Regex: `r"\(cid:\d+\)"`

**Future Enhancement**:
- Detect CID patterns → force OCR fallback
- Character replacement heuristics (map CIDs to Unicode)
- Warn user: "PDF contains garbled text, using OCR"

---

### 5. **Low-Contrast Scans**

**Problem**:
- Faded text, low-resolution scans, coffee stains
- OCR confidence drops → gibberish output

**Current Handling**:
- Quality assessment: `_assess_extraction_quality()` checks char count, word count, alpha ratio
- Falls back to OCR if extraction quality poor

**Failure Cases**:
- OCR also fails on low-contrast images
- No preprocessing (contrast enhancement)

**Future Enhancement**:
- Image preprocessing: contrast normalization, binarization (OpenCV)
- Multi-pass OCR: try Tesseract, EasyOCR, cloud OCR (Google Vision)
- User warning: "Low-quality scan detected, results may be incomplete"

---

### 6. **Handwritten Notes**

**Problem**:
- OCR models trained on printed text
- Handwriting recognition requires specialized models

**Current Handling**:
- Tesseract/EasyOCR: limited handwriting support

**Failure Cases**:
- Cursive handwriting
- Mixed print/handwriting
- Sloppy notes

**Future Enhancement**:
- Detect handwriting: image classification (CNN)
- Route to specialized OCR: TrOCR, Google Vision API
- Prompt user: "Handwritten notes detected, results may vary"

---

### 7. **Equation Rendering Issues**

**Problem**:
- LaTeX equations as images → OCR extracts "∫" as "J"
- MathML/MathType → not extractable by `pdfplumber`

**Current Handling**:
- Equations passed as-is (if extractable)
- OCR attempts to recognize symbols

**Failure Cases**:
- Complex equations with fractions, integrals, summations
- Misrecognized: "∫_0^∞ e^(-x) dx" → "Jo e(-x) dx"

**Future Enhancement**:
- Equation detection: detect LaTeX images by bounding box analysis
- Specialized OCR: `pix2tex`, `mathpix` API
- User-provided equation input (manual override)

---

### 8. **Non-English Content**

**Problem**:
- OCR/cleaning optimized for English
- Non-ASCII characters, diacritics → removed or garbled

**Current Handling**:
- Text cleaning: preserves Unicode (no ASCII-only enforcement)
- OCR: language detection via `pytesseract` (config: `lang='eng'`)

**Failure Cases**:
- Mixed-language documents (e.g., English + Hindi)
- Non-Latin scripts (Chinese, Arabic)

**Future Enhancement**:
- Auto-detect document language: `langdetect` library
- Multi-language OCR: `lang='eng+hin+chi_sim'`
- Language-specific cleaning rules

---

### 9. **Corrupted or Password-Protected PDFs**

**Problem**:
- Corrupted files → extraction crashes
- Password-protected → locked

**Current Handling**:
- Exception handling in `extract_pdf_text()` → raises `EvidenceIngestError`

**Failure Cases**:
- Crashes app if exception not caught
- No user-friendly error message

**Future Enhancement**:
- PDF repair: `pikepdf.open(allow_overwriting_input=True)`
- Password prompt: ask user for PDF password
- Fallback: "PDF could not be processed, please provide text manually"

---

### 10. **Large Files (Memory Exhaustion)**

**Problem**:
- 100+ page PDFs → memory overflow during OCR
- FAISS index: large embeddings → >4GB RAM

**Current Handling**:
- No pagination (loads entire PDF into memory)

**Failure Cases**:
- OOM errors on resource-constrained systems
- Streamlit crashes

**Future Enhancement**:
- Batch processing: extract 10 pages at a time
- Streaming embeddings: process chunks on-the-fly
- Warn user: "Large file detected, processing may take several minutes"

---

## Integration Points

### 1. **Artifact Store Integration**

**Purpose**: Persist evidence, claims, and verification results for cross-session comparisons

**Current State**: Ephemeral (session-only storage in `st.session_state`)

**Integration Needed**:
- **Database**: SQL (PostgreSQL) or NoSQL (MongoDB)
  - Tables: `sessions`, `claims`, `evidence`, `claim_evidence_links`, `graphs`
- **File locations**:
  - `src/storage/artifact_store.py`
  - `src/storage/db_schema.py`
- **Methods**:
  ```python
  ArtifactStore.save_session(session_id, output, verifiable_metadata)
  ArtifactStore.load_session(session_id) → (output, verifiable_metadata)
  ArtifactStore.query_claims(filters) → List[LearningClaim]
  ```

**Use Cases**:
- Cumulative study guides across lectures
- Claim deduplication (avoid regenerating identical concepts)
- Historical trend analysis (rejection rate over time)

---

### 2. **Online Authority Retrieval**

**Purpose**: Query external knowledge bases (Wikipedia, arXiv, Wolfram Alpha) for evidence

**Current State**: Offline-only (session content + URLs)

**Integration Needed**:
- **APIs**:
  - Wikipedia: `wikipedia-api` Python package
  - arXiv: `arxiv` Python package
  - Wolfram Alpha: `wolframalpha` API client
  - Google Scholar: `scholarly` package (unofficial)
- **File locations**:
  - `src/retrieval/external_retrieval.py`
  - Add to `url_ingest.py` as additional source type
- **Methods**:
  ```python
  ExternalRetriever.query_wikipedia(concept) → List[EvidenceItem]
  ExternalRetriever.query_arxiv(topic) → List[EvidenceItem]
  ExternalRetriever.query_wolfram(equation) → EvidenceItem
  ```
- **Evidence metadata**:
  - `source_type`: "wikipedia", "arxiv", "wolfram"
  - `reliability_prior`: 0.9 (Wikipedia), 0.95 (arXiv), 0.99 (Wolfram)

**Use Cases**:
- Verify definitions against Wikipedia
- Cross-check equations with Wolfram Alpha
- Find research papers supporting advanced topics

---

### 3. **CS-Aware Verifier Integration**

**Purpose**: Domain-specific verification for algorithms, complexity, invariants

**Current State**: Generic NLI consistency checks (via Ollama LLM)

**Integration Needed**:
- **Specialized validators**:
  - `Algorithm Verifier`: parse pseudocode, check loop invariants
  - `Complexity Analyzer`: verify Big-O claims against code patterns
  - `Proof Checker`: validate mathematical proofs step-by-step
- **File locations**:
  - `src/verification/cs_verifier.py`
  - `src/verification/complexity_analyzer.py`
  - `src/verification/proof_checker.py`
- **Methods**:
  ```python
  CSVerifier.verify_algorithm_step(claim, evidence) → (is_valid, explanation)
  ComplexityAnalyzer.verify_bigo_claim(claim, code_snippet) → (is_valid, explanation)
  ProofChecker.verify_proof_step(claim, previous_steps) → (is_valid, errors)
  ```

**Use Cases**:
- Detect incorrect Big-O claims (e.g., "Bubble sort is O(n)" → REJECTED)
- Validate loop invariants in algorithms
- Check proof correctness in discrete math

---

### 4. **Citation Tracking Integration**

**Purpose**: Track provenance from claims → evidence → source documents → page/line numbers

**Current State**: Partial (evidence has `span_metadata`, but not exposed in UI)

**Integration Needed**:
- **UI enhancements**:
  - Clickable citations: "See evidence on page 5, line 120"
  - Highlight spans in PDF viewer (PDF.js integration)
- **Export formats**:
  - APA/MLA citations: "Smith, J. (2026). Lecture Notes. Page 5."
  - BibTeX export for LaTeX documents
- **File locations**:
  - `src/citations/citation_formatter.py`
  - Update `report_exporter.py` to include citations in PDF
- **Methods**:
  ```python
  CitationFormatter.format_citation(evidence, style="APA") → str
  CitationFormatter.generate_bibliography(claims) → List[str]
  ```

**Use Cases**:
- Academic integrity: cite sources in study guides
- Debugging: trace claim back to original page/line
- Student reference: "Where did this definition come from?"

---

### 5. **Report Builder Integration**

**Purpose**: Customizable export templates (PDF, HTML, LaTeX) with branding

**Current State**: Basic PDF export with fixed layout (`report_exporter.py`)

**Integration Needed**:
- **Template system**:
  - Jinja2 templates for HTML/LaTeX
  - ReportLab or WeasyPrint for advanced PDF rendering
- **File locations**:
  - `src/exporters/template_engine.py`
  - `templates/report.html.jinja`
  - `templates/report.tex.jinja`
- **Features**:
  - Custom headers/footers (university logo, course name)
  - Section toggling (include/exclude topics, FAQs, etc.)
  - Graph visualizations embedded (claim-evidence graphs)
- **Methods**:
  ```python
  ReportBuilder.render_html(verifiable_metadata, template="default") → html_string
  ReportBuilder.render_latex(verifiable_metadata) → latex_string
  ReportBuilder.compile_pdf(latex_string) → pdf_bytes
  ```

**Use Cases**:
- Branded study guides for institutions
- LaTeX export for thesis/paper integration
- HTML export for web publishing

---

### 6. **Feedback Loop Integration**

**Purpose**: Collect user feedback on claim correctness to improve verification

**Current State**: No feedback mechanism

**Integration Needed**:
- **UI components**:
  - Thumbs up/down buttons per claim
  - "Report error" form with free text
- **Backend**:
  - Store feedback in database: `feedback` table
  - Link to `claim_id` and `session_id`
- **File locations**:
  - `src/feedback/collector.py`
  - Add to `app.py` UI
- **Methods**:
  ```python
  FeedbackCollector.record_feedback(claim_id, rating, comment, user_id)
  FeedbackCollector.get_disputed_claims() → List[LearningClaim]
  ```
- **Analytics**:
  - Identify low-quality claims (high rejection rate + negative feedback)
  - Retrain models on corrected data

**Use Cases**:
- Continuous improvement: flag hallucinations for review
- User trust: show correction transparency
- Model fine-tuning: supervised learning on feedback

---

### 7. **Multi-Session Aggregation**

**Purpose**: Combine claims from multiple lectures into cumulative study guide

**Current State**: Single-session processing only

**Integration Needed**:
- **Session graph**:
  - Link claims across sessions by topic/concept
  - Detect duplicates (same concept, different wording)
  - Merge evidence from multiple sessions
- **File locations**:
  - `src/study_book/session_aggregator.py`
  - Update `session_manager.py` for multi-session queries
- **Methods**:
  ```python
  SessionAggregator.merge_sessions(session_ids) → ClassSessionOutput
  SessionAggregator.deduplicate_claims(claims) → List[LearningClaim]
  SessionAggregator.build_concept_graph(sessions) → ConceptGraph
  ```

**Use Cases**:
- Final exam study guide (all lectures)
- Prerequisite tracking (Lecture 5 requires Lecture 2 concepts)
- Progressive learning paths

---

### 8. **Real-Time Verification API**

**Purpose**: Expose verification pipeline as REST API for external tools

**Current State**: Streamlit-only (no API)

**Integration Needed**:
- **Framework**: FastAPI or Flask
- **Endpoints**:
  - `POST /verify`: Upload PDF, return verification report
  - `GET /session/{session_id}`: Retrieve past session
  - `POST /claims`: Submit claims for verification
- **File locations**:
  - `api/main.py` (FastAPI app)
  - `api/routes.py`
- **Authentication**:
  - API keys: `API_KEY` header
  - Rate limiting: 10 requests/minute (Redis-backed)
- **Methods**:
  ```python
  @app.post("/verify")
  async def verify_content(file: UploadFile, mode: str) -> VerificationReport
  ```

**Use Cases**:
- Integration with note-taking apps (Notion, Obsidian)
- Jupyter Notebook plugin for live verification
- Browser extension for web article verification

---

### 9. **Graph Database Integration**

**Purpose**: Store claim-evidence graph in graph database (Neo4j, Amazon Neptune) for advanced queries

**Current State**: NetworkX in-memory graph (ephemeral)

**Integration Needed**:
- **Database**: Neo4j Community Edition (local) or Neo4j Aura (cloud)
- **Schema**:
  - Nodes: `Claim`, `Evidence`, `Source`, `Concept`
  - Edges: `SUPPORTS` (Evidence → Claim), `REQUIRES` (Claim → Concept), `CITED_IN` (Evidence → Source)
- **File locations**:
  - `src/graph/neo4j_connector.py`
  - Update `claim_graph.py` to export to Neo4j
- **Methods**:
  ```python
  Neo4jConnector.save_graph(graph: ClaimGraph, session_id: str)
  Neo4jConnector.query_subgraph(concept: str) → ClaimGraph
  Neo4jConnector.find_shortest_path(claim_id_1, claim_id_2) → List[str]
  ```

**Use Cases**:
- Query: "Find all claims supported by Wikipedia"
- Detect circular dependencies (Concept A requires B, B requires A)
- Visualize knowledge graph in Neo4j Bloom

---

### 10. **Active Learning for Claim Generation**

**Purpose**: Prioritize generating claims for high-uncertainty regions

**Current State**: Generate claims for all output sections (concepts, equations, etc.)

**Integration Needed**:
- **Uncertainty estimation**:
  - Compute entropy of LLM output distribution
  - Identify low-confidence concepts (e.g., `confidence < 0.5`)
- **Selective generation**:
  - Generate additional evidence for low-confidence claims
  - Query user for clarification ("Is X correct?")
- **File locations**:
  - `src/active_learning/uncertainty_estimator.py`
  - Update `verifiable_pipeline.py` to trigger selective generation
- **Methods**:
  ```python
  UncertaintyEstimator.compute_entropy(logits) → float
  ActiveLearner.prioritize_claims(claims) → List[LearningClaim]
  ActiveLearner.request_user_feedback(claim) → (is_correct, explanation)
  ```

**Use Cases**:
- Focus verification on uncertain claims (reduce cost)
- Interactive mode: ask user to resolve ambiguities
- Iterative refinement: regenerate claims with user corrections

---

## Evaluation & Research

### 8.1 Research Framework

Smart Notes includes production-grade research infrastructure for evaluating verification accuracy and calibration:

**Goal**: Measure claim verification performance, confidence calibration, and component ablations

**Scope**: Educational AI system evaluation (research vs. production)

### 8.2 Benchmark Dataset

**Location**: `evaluation/cs_benchmark/cs_benchmark_dataset.jsonl`

**Format**: JSONL, 20 synthetic examples

**Schema**:
```json
{
  "doc_id": "algo_001",
  "domain_topic": "algorithms.sorting",
  "source_text": "Course material...",
  "generated_claim": "The claim to verify",
  "gold_label": "VERIFIED|LOW_CONFIDENCE|REJECTED",
  "evidence_span": "Supporting text from source_text"
}
```

**Coverage**: 8 CS domains
- Algorithms (7 examples) - sorting, search, DP
- Data structures (5) - hash tables, trees, graphs
- Complexity theory (2) - NP-hardness, relationships
- Networking (2) - TCP/UDP protocols
- Security (2) - encryption, hashing
- Databases (2) - indexing, SQL
- ML (2) - optimization, regression
- Compilers (2) - parsing, optimization

**Label Distribution**:
- VERIFIED: 11/20 (fully supported)
- LOW_CONFIDENCE: 4/20 (partially supported/ambiguous)
- REJECTED: 5/20 (contradicted/unsupported)

**Rationale for Synthetic Data**:
- Avoids copyright issues
- Deterministic and reproducible
- Easily extensible to new domains
- Controlled difficulty/ambiguity

### 8.3 Benchmark Runner

**Module**: `src/evaluation/cs_benchmark_runner.py`

**Purpose**: Evaluate verification pipeline on benchmark dataset

**Key Components**:

#### 8.3.1 CSBenchmarkRunner

```python
runner = CSBenchmarkRunner(
    dataset_path="evaluation/cs_benchmark/cs_benchmark_dataset.jsonl",
    embedding_provider=None,  # Default: all-MiniLM-L6-v2
    nli_verifier=None,        # Default: roberta-large-mnli
    batch_size=8,
    device="cpu",
    seed=42
)

result = runner.run(
    config={                   # Configuration dict
        "use_retrieval": True,
        "use_nli": True,
        "use_batch_nli": True,
        # ... toggles for ablations
    },
    noise_types=["typo", "paraphrase"],  # Robustness testing
    sample_size=5  # For CI smoke tests
)
```

**Output**: `BenchmarkResult` with metrics and predictions

#### 8.3.2 Computed Metrics

**Classification**:
- Accuracy: Overall correctness
- Precision/Recall per label (VERIFIED, etc.)
- F1 score (harmonic mean)

**Calibration**:
- ECE (Expected Calibration Error): Gap between confidence and accuracy
  - Formula: $ECE = \sum_{b} \frac{|S_b|}{n} | \text{acc}(S_b) - \text{conf}(S_b) |$
  - Target: ECE < 0.1 (well-calibrated)
- Brier Score: Mean squared error of confidence
  - Formula: $\text{Brier} = \frac{1}{n} \sum_i (\text{conf}_i - y_i)^2$

**Robustness**:
- Noise injection (typo, paraphrase, negation)
- Accuracy degradation per noise type
- Overall robustness score

**Efficiency**:
- Average, median, p95 time per claim
- Total inference time
- Memory usage (evidence store size)

**Coverage**:
- Claims with evidence (%)
- Average evidence per claim
- Evidence source distribution

### 8.4 Ablation Study Runner

**Module**: `scripts/run_cs_benchmark.py`

**Purpose**: Compare pipeline configurations systematically

**Configurations Tested**:

| Config | Retrieval | NLI | Ensemble | Description |
|--------|-----------|-----|----------|---|
| 00 | ✗ | ✗ | ✗ | Baseline (no verification) |
| 1a | ✓ | ✗ | ✗ | Retrieval-only (similarity) |
| 1b | ✓ | ✓ | ✗ | Full pipeline (main) |
| 1c | ✓ | ✓ | ✓ | Ensemble verifier |
| +Features | ✓ | ✓ | - | Toggles for cleaning, batch NLI, etc. |

**Feature Toggles**:
- `use_cleaning`: Text preprocessing (boilerplate removal)
- `use_artifact_persistence`: Cache/load intermediate results
- `use_batch_nli`: Batch processing for efficiency
- `use_online_authority`: Query external knowledge bases

**Execution**:

```bash
python scripts/run_cs_benchmark.py \
    --dataset evaluation/cs_benchmark/cs_benchmark_dataset.jsonl \
    --output-dir evaluation/results \
    --sample-size 20 \  # Full benchmark (10 for CI smoke)
    --seed 42 \
    --noise-injection  # Test robustness
```

**Outputs**:
- `results.csv`: Metrics table (one row per config)
- `ablation_summary.md`: Markdown report with findings
- `detailed_results/`: Per-config JSON result files

### 8.5 Research Documentation

#### 8.5.1 RESEARCH_RESULTS.md

**Purpose**: Template for publishing benchmark results

**Contents**:
- Executive summary
- Problem statement & hypotheses
- Methodology section
- Results tables and analysis
- Findings and interpretation
- Limitations and generalization

**Structure**:
```
1. Executive Summary
2. Problem Statement
3. Methodology
   3.1 Dataset
   3.2 Verification Pipeline
   3.3 Metrics
4. Results
   4.1 Main Results
   4.2 Ablation Results
   4.3 Calibration Analysis
   4.4 Robustness Analysis
5. Analysis & Discussion
6. Reproducibility
7. Appendix
```

**Fillables**: `[??]` placeholders for actual results

#### 8.5.2 PAPER_OUTLINE.md

**Purpose**: Structure for academic paper submission

**Contents**:
- Title and abstract options
- Full outline (10 sections)
- Contribution statements
- Related work
- Method descriptions with equations
- Results tables and figures
- Discussion and future work
- Reproducibility checklist

**Target Venues**: ACL 2026, AAAI 2026, ICML 2026

### 8.6 Test Suite

#### 8.6.1 test_benchmark_format_validation.py

**Purpose**: Validate benchmark dataset schema and quality

**Tests**:
- Dataset exists and is well-formed
- Required fields present in all examples
- Unique doc_ids and claims
- Valid label and domain values
- Text field lengths reasonable
- Evidence span consistency
- Label distribution balanced
- No Unicode errors

**Run**: `pytest tests/test_benchmark_format_validation.py -v`

#### 8.6.2 test_ablation_runner_smoke.py

**Purpose**: CI-friendly smoke tests for ablation infrastructure

**Tests**:
- Runner initialization
- Ablation configs available
- Single config runs without error
- CSV and markdown outputs created
- Output format validation
- Reproducibility with seed

**Dependencies**: Runs on small sample (size=3) for speed

**Run**: `pytest tests/test_ablation_runner_smoke.py -v`

### 8.7 Reproducibility

**Determinism Requirements**:

1. **Dataset**: Fixed JSONL file (deterministic loading)
2. **Seed**: Set via `seed=42` in runner initialization
3. **Models**: Pre-trained frozen models (no training)
4. **Hardware**: CPU-only (for reproducibility)
5. **Randomness**: All random seeds fixed

**Verification**:

```bash
# Run 1
python scripts/run_cs_benchmark.py --seed 42 --sample-size 5

# Run 2 (same command)
python scripts/run_cs_benchmark.py --seed 42 --sample-size 5

# Results should be identical (bit-for-bit matching for main metrics)
```

**Computational Cost**:
- Time: ~5 minutes for full benchmark (20 claims × 8 configs)
- Memory: 8GB+ (for model loading)
- GPU: Optional (uses CPU by default)
- Models: Auto-downloaded (~2GB total)

### 8.8 Integration with CI/CD

**GitHub Actions Workflow** (example):

```yaml
name: Benchmark Smoke Test

on: [push, pull_request]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - run: pip install -r requirements.txt
      - run: pytest tests/test_benchmark_format_validation.py -v
      - run: pytest tests/test_ablation_runner_smoke.py -v
      - run: python scripts/run_cs_benchmark.py --sample-size 5 --seed 42
      - uses: actions/upload-artifact@v2
        with:
          name: benchmark-results
          path: evaluation/results/
```

**Expected Duration**: < 5 minutes on CI

### 8.9 Extension Points

**Adding New Domains**:

1. Create examples in `cs_benchmark_dataset.jsonl` with new `domain_topic`
2. Update `VALID_DOMAINS` in test file
3. Re-run benchmark

**Custom Embeddings**:

```python
from sentence_transformers import SentenceTransformer
custom_embedder = SentenceTransformer("model-name")
provider = EmbeddingProvider(model=custom_embedder)
runner = CSBenchmarkRunner(embedding_provider=provider)
```

**Custom NLI Model**:

```python
verifier = NLIVerifier(model_name="deberta-base-mnli")
runner = CSBenchmarkRunner(nli_verifier=verifier)
```

**Real Data Integration**:

Replace synthetic dataset with real lecture notes:

```python
# Load real notes
runner = CSBenchmarkRunner(dataset_path="my_course/benchmark.jsonl")
```

---

## Summary

This architecture map documents the **current implementation** of Smart Notes as of 2026-02-17. Key takeaways:

1. **Hybrid pipeline**: Standard mode (fast LLM generation) + Verifiable mode (evidence-grounded claims)
2. **Evidence-first approach**: All claims must retrieve evidence before verification
3. **Explicit rejection**: Claims without sufficient evidence are marked REJECTED (not hallucinated into output)
4. **Multi-modal ingestion**: PDFs (with OCR), audio (Whisper), YouTube/web articles
5. **Nondeterminism sources**: Chunking, embeddings, FAISS, LLM sampling, reranking, NLI checks
6. **Failure modes**: Headers/footers, CamScanner watermarks, multi-column layouts, CID text, low-contrast scans
7. **Research evaluation**: Synthetic benchmark, ablation study, calibration analysis for EB1A/NSF-grade research
8. **Integration points**: 10 identified areas for future enhancement (artifact store, online retrieval, CS-aware verification, citations, etc.)

For implementation details, see:
- [TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md)
- [VERIFIABILITY_CONTRACT.md](VERIFIABILITY_CONTRACT.md)
- [GRAPH_METRICS_VERIFICATION.md](GRAPH_METRICS_VERIFICATION.md)
- [RESEARCH_RESULTS.md](RESEARCH_RESULTS.md) - Research results template
- [PAPER_OUTLINE.md](PAPER_OUTLINE.md) - Paper structure template

---

**Contributors**: Smart Notes Team  
**License**: MIT  
**Contact**: https://github.com/somanellipudi/smart-notes/issues
