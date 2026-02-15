# Smart Notes: File Structure & Organization Guide

**Version**: 2.0 (Production-Ready)  
**Date**: February 12, 2026  
**Purpose**: Complete visual guide to file organization and module dependencies

---

## Table of Contents

1. [Complete File Tree](#complete-file-tree)
2. [Directory Organization](#directory-organization)
3. [Module Dependency Graph](#module-dependency-graph)
4. [File Categories & Purposes](#file-categories--purposes)
5. [Import Patterns](#import-patterns)
6. [Configuration & Secrets](#configuration--secrets)
7. [Cache & Output Directories](#cache--output-directories)
8. [Test Organization](#test-organization)
9. [Documentation Files](#documentation-files)

---

## Complete File Tree

```
Smart-Notes/
│
│═══════════════════════════════════════════════════════════════════════════
│  ROOT-LEVEL FILES (Configuration & Entry Points)
│═══════════════════════════════════════════════════════════════════════════
│
├── 📄 app.py                                  ⭐ PRIMARY ENTRY POINT
│   │  Lines: 1600+
│   │  Purpose: Streamlit UI, main application
│   │  Dependencies: streamlit, src.*, config
│   │  Exports: Web interface on localhost:8501
│   │  Status: Production-ready (v2.0)
│   │
│
├── 🔑 config.py                               ⚙️ CONFIGURATION
│   │  Lines: ~150
│   │  Purpose: Global parameters and thresholds
│   │  Key vars: VERIFIED_THRESHOLD, MODEL_NAMES, CACHE_DIR
│   │  Used by: All src/* modules
│   │  Status: Production-ready
│   │
│
├── 📋 requirements.txt                        📦 DEPENDENCIES
│   │  Lines: 25+
│   │  Contains: Python packages + versions
│   │  Install: pip install -r requirements.txt
│   │  Categories:
│   │    - Core: streamlit, pandas, numpy
│   │    - ML: transformers, torch, sentence-transformers
│   │    - Graph: networkx, pyvis
│   │    - Optional: pyarrow, matplotlib, easycr
│   │
│
├── 🔐 .env                                    🔒 SECRETS (GITIGNORE)
│   │  Created by user from .env.example
│   │  Contains: API_KEY, LLM_PROVIDER, etc.
│   │  Status: Never committed to git
│   │
│
├── 📄 .env.example                            📄 SECRETS TEMPLATE
│   │  Template for creating .env file
│   │  Shows required environment variables
│   │
│
├── .gitignore                                 📝 GIT EXCLUSIONS
│   │  Excludes: .env, __pycache__, *.pyc, .venv
│   │
│
├── README.md                                  📖 USER DOCUMENTATION
│   │  Lines: 1585+
│   │  Audience: End users, researchers
│   │  Contents: Quick start, installation, usage
│   │  Status: Updated Feb 2025
│   │
│
├── LICENSE                                    📜 MIT LICENSE
│   │
│
└── validate_fixes.py                          🔍 VALIDATION SCRIPT ✨ NEW
    │  Lines: ~200
    │  Purpose: Test environment and verify all fixes work
    │  Usage: python validate_fixes.py
    │  Status: Comprehensive validation (Feb 2025)
    │
│
│═══════════════════════════════════════════════════════════════════════════
│  src/ DIRECTORY (8,000+ lines of application code)
│═══════════════════════════════════════════════════════════════════════════
│
├── 📁 src/
│   │
│   ├── 📄 __init__.py                         (Empty Python package marker)
│   │
│   ├── 📄 llm_provider.py                     🤖 LLM ABSTRACTION LAYER
│   │   │  Lines: ~200
│   │   │  Purpose: Unified interface for OpenAI/Ollama
│   │   │  Classes:
│   │   │    - LLMProvider (base interface)
│   │   │    - OpenAIProvider
│   │   │    - OllamaProvider
│   │   │  Key methods: generate(), generate_json()
│   │   │  Used by: reasoning/pipeline.py
│   │   │  Status: Production-ready
│   │   │
│   │
│   ├── 📄 logging_config.py                   📊 LOGGING SETUP
│   │   │  Lines: ~50
│   │   │  Purpose: Configure logging to files + console
│   │   │  Output files:
│   │   │    - logs/app.log (all messages)
│   │   │    - logs/errors.log (errors only)
│   │   │  Used by: All modules (logger = logging.getLogger(__name__))
│   │   │  Status: Production-ready
│   │   │
│   │
│   ├── 📄 output_formatter.py                 📤 EXPORT FORMATTING
│   │   │  Lines: ~300
│   │   │  Purpose: Format results into JSON/CSV/Markdown
│   │   │  Classes: OutputFormatter
│   │   │  Methods:
│   │   │    - to_json(results) → JSON bytes
│   │   │    - to_csv(results) → CSV bytes
│   │   │    - to_markdown(results) → Markdown bytes
│   │   │  Used by: app.py, reasoning/pipeline.py
│   │   │  Status: Production-ready
│   │   │
│   │
│   ├── 📄 streamlit_display.py               🎨 UI UTILITIES
│   │   │  Lines: ~150
│   │   │  Purpose: Streamlit-specific display utilities
│   │   │  Functions:
│   │   │    - display_metrics_dashboard(results)
│   │   │    - display_claims_table(claims)
│   │   │    - display_graph(graph)
│   │   │  Used by: app.py
│   │   │  Status: Production-ready
│   │   │
│   │
│   │
│   ├── 📁 claims/                            ✨ CLAIMS MANAGEMENT
│   │   │  Purpose: Extract, verify, and analyze claims
│   │   │
│   │   ├── 📄 __init__.py                     (Package marker)
│   │   │
│   │   ├── 📄 schema.py                       📊 DATA MODELS
│   │   │   │  Lines: 493
│   │   │   │  Classes:
│   │   │   │    - VerificationStatus (enum)
│   │   │   │    - ClaimType (enum)
│   │   │   │    - EvidenceItem (Pydantic model)
│   │   │   │    - LearningClaim (Pydantic model)
│   │   │   │    - GraphMetrics (Pydantic model)
│   │   │   │        ├─ .to_dict() method ✨ NEW
│   │   │   │        └─ .get() method ✨ NEW
│   │   │   │    - SessionResult (Pydantic model)
│   │   │   │  Used by: All claims/* modules
│   │   │   │  Status: Production-ready (Feb 2025)
│   │   │   │
│   │   │
│   │   ├── 📄 extractor.py                    🔍 CLAIM EXTRACTION
│   │   │   │  Lines: ~200
│   │   │   │  Purpose: Parse LLM output into discrete claims
│   │   │   │  Classes: ClaimExtractor
│   │   │   │  Methods: extract(json_output) → List[LearningClaim]
│   │   │   │  Pattern: Regex + NLP tokenization
│   │   │   │  Used by: reasoning/pipeline.py, reasoning/verifiable_pipeline.py
│   │   │   │  Status: Production-ready
│   │   │   │
│   │   │
│   │   ├── 📄 validator.py                    ✅ STATUS ASSIGNMENT
│   │   │   │  Lines: ~150
│   │   │   │  Purpose: Assign verification status based on confidence
│   │   │   │  Thresholds:
│   │   │   │    - confidence ≥ 0.7 → VERIFIED
│   │   │   │    - 0.3 ≤ confidence < 0.7 → LOW_CONFIDENCE
│   │   │   │    - confidence < 0.3 → REJECTED
│   │   │   │  Classes: ClaimValidator
│   │   │   │  Used by: reasoning/verifiable_pipeline.py
│   │   │   │  Status: Production-ready
│   │   │   │
│   │   │
│   │   ├── 📄 nli_verifier.py                 🧠 NLI CLASSIFICATION ✨ NEW
│   │   │   │  Lines: 266
│   │   │   │  Purpose: Verify entailment using BART-MNLI
│   │   │   │  Classes: NLIVerifier
│   │   │   │  Models:
│   │   │   │    - facebook/bart-large-mnli (1.6GB)
│   │   │   │    - Alternative: roberta-large-mnli
│   │   │   │  Methods:
│   │   │   │    - verify_entailment(claim, evidence)
│   │   │   │    - multi_source_consensus(claim, evidence_list)
│   │   │   │  Output:
│   │   │   │    - label ∈ {ENTAILMENT, CONTRADICTION, NEUTRAL}
│   │   │   │    - probabilities for each label
│   │   │   │  Used by: reasoning/verifiable_pipeline.py
│   │   │   │  Status: Production-ready (Feb 2025)
│   │   │   │  Performance: ~200ms per (claim, evidence) pair
│   │   │   │
│   │   │
│   │   └── 📄 confidence.py                    📊 CONFIDENCE SCORING ✨ NEW
│       │  Lines: 304
│       │  Purpose: Calculate multi-factor confidence scores
│       │  Classes: ConfidenceCalculator
│       │  Components:
│       │    - Semantic similarity (25% weight)
│       │    - Entailment probability (35%)
│       │    - Source diversity (10%)
│       │    - Evidence count (15%)
│       │    - Contradiction penalty (-10%)
│       │    - Graph centrality (5%)
│       │  Methods:
│       │    - compute_confidence() → float ∈ [0, 1]
│       │    - fit_temperature() (calibration)
│       │  Used by: reasoning/verifiable_pipeline.py
│       │  Status: Production-ready (Feb 2025)
│       │
│
│   ├── 📁 retrieval/                         🔎 EVIDENCE RETRIEVAL
│   │   │  Purpose: Search and rank evidence from source material
│   │   │
│   │   ├── 📄 __init__.py                     (Package marker)
│   │   │
│   │   ├── 📄 claim_rag.py                    🔍 KEYWORD RETRIEVAL (DEPRECATED)
│   │   │   │  Lines: ~150
│   │   │   │  Purpose: Legacy keyword-based search (Jaccard similarity)
│   │   │   │  Status: Deprecated (use semantic_retriever.py)
│   │   │   │  Kept for: Backward compatibility
│   │   │   │
│   │   │
│   │   └── 📄 semantic_retriever.py           🧠 SEMANTIC RETRIEVAL ✨ NEW
│       │  Lines: 296
│       │  Purpose: Dense retrieval with FAISS + cross-encoder re-ranking
│       │  Models:
│       │    - intfloat/e5-base-v2 (400MB) for embeddings
│       │    - cross-encoder/ms-marco-MiniLM-L-6-v2 (80MB) for ranking
│       │  Classes: SemanticRetriever
│       │  Methods:
│       │    - index_corpus(documents) → build FAISS index
│       │    - retrieve(query) → List[EvidenceItem]
│       │  Performance:
│       │    - FAISS search: ~100ms (k=10)
│       │    - Cross-encoder re-rank: ~400ms (n=5)
│       │    - Per-claim: ~500ms
│       │  Used by: reasoning/verifiable_pipeline.py
│       │  Status: Production-ready (Feb 2025)
│       │
│
│   ├── 📁 reasoning/                         💡 LLM PIPELINES
│   │   │  Purpose: Orchestrate LLM generation and verification
│   │   │
│   │   ├── 📄 __init__.py                     (Package marker)
│   │   │
│   │   ├── 📄 pipeline.py                     🔄 BASELINE PIPELINE
│   │   │   │  Lines: ~300
│   │   │   │  Purpose: Fast generation without verification
│   │   │   │  Pipeline:
│   │   │   │    1. Call LLM with 7-stage prompt
│   │   │   │    2. Parse JSON output
│   │   │   │    3. Extract claims
│   │   │   │    4. Format output
│   │   │   │  Classes: BaselinePipeline
│   │   │   │  Speed: 20-40 seconds for ~300 claims
│   │   │   │  Used by: app.py, reasoning/verifiable_pipeline.py
│   │   │   │  Status: Production-ready
│   │   │   │
│   │   │
│   │   ├── 📄 verifiable_pipeline.py          🔬 VERIFIABLE PIPELINE ✨ NEW
│   │   │   │  Lines: ~400
│   │   │   │  Purpose: Generate + verify claims against source
│   │   │   │  Pipeline:
│   │   │   │    1. Generate claims (BaselinePipeline)
│   │   │   │    2. Index source corpus (SemanticRetriever)
│   │   │   │    3. Retrieve evidence per claim (SemanticRetriever)
│   │   │   │    4. Verify entailment (NLIVerifier)
│   │   │   │    5. Calculate confidence (ConfidenceCalculator)
│   │   │   │    6. Assign status (ClaimValidator)
│   │   │   │    7. Build graph (ClaimGraph)
│   │   │   │    8. Compute metrics (GraphMetrics)
│   │   │   │  Classes: VerifiablePipeline
│   │   │   │  Speed: 60-120 seconds for ~300 claims
│   │   │   │  Used by: app.py
│   │   │   │  Status: Production-ready (Feb 2025)
│   │   │   │
│   │   │
│   │   └── 📄 prompts.py                      📝 LLM PROMPTS
│       │  Lines: ~200
│       │  Purpose: Store and manage LLM prompt templates
│       │  Prompts:
│       │    - 7-stage generation pipeline
│       │    - Extraction pattern prompts
│       │  Used by: reasoning/pipeline.py
│       │  Status: Production-ready
│       │
│
│   ├── 📁 graph/                             📈 GRAPH ANALYSIS
│   │   │  Purpose: Build, analyze, and export knowledge graphs
│   │   │
│   │   ├── 📄 __init__.py                     (Package marker)
│   │   │
│   │   ├── 📄 claim_graph.py                  🔗 GRAPH CONSTRUCTION
│   │   │   │  Lines: 727
│   │   │   │  Purpose: Build NetworkX DiGraph from claims
│   │   │   │  Classes: ClaimGraph
│   │   │   │  Graph structure:
│   │   │   │    - Nodes: Claims (blue) + Evidence (green)
│   │   │   │    - Edges: claim → evidence (weight = similarity)
│   │   │   │    - Attributes: status, confidence, snippet, etc.
│   │   │   │  Methods:
│   │   │   │    - compute_metrics() → GraphMetrics
│   │   │   │    - export_graphml() → bytes
│   │   │   │    - export_adjacency_json() → str
│   │   │   │    - visualize() → Image (PNG if matplotlib)
│   │   │   │  Metrics computed:
│   │   │   │    - Redundancy (evidence per claim)
│   │   │   │    - Diversity (source variety)
│   │   │   │    - Support depth (max path length)
│   │   │   │    - Conflict count (contradictions)
│   │   │   │    - Centrality (claim importance)
│   │   │   │  Used by: reasoning/verifiable_pipeline.py, app.py
│   │   │   │  Status: Production-ready
│   │   │   │
│   │   │
│   │   └── 📄 graph_sanitize.py               🧹 SANITIZATION ✨ NEW (Feb 2025)
│       │  Lines: 166
│       │  Purpose: Sanitize graph for GraphML export (handles bytes/enums/Pydantic)
│       │  Classes: None (functions only)
│       │  Functions:
│       │    - _sanitize_value() → GraphML-safe string
│       │    - sanitize_graph_for_graphml() → sanitized copy
│       │    - export_graphml_string() → XML string
│       │    - export_graphml_bytes() → UTF-8 bytes
│       │  Handles:
│       │    - bytes → UTF-8 decode or base64
│       │    - enums → string values
│       │    - Pydantic models → JSON
│       │    - dicts/lists → JSON strings
│       │    - long strings → truncated to 500 chars
│       │  Fixes: Graph export "bytes" TypeError (Issue #3)
│       │  Used by: graph/claim_graph.py, app.py
│       │  Status: Fully tested (18 unit tests)
│       │
│
│   ├── 📁 preprocessing/                     ⚙️ INPUT PROCESSING
│   │   │  Purpose: Clean and normalize input text
│   │   │
│   │   ├── 📄 __init__.py                     (Package marker)
│   │   │
│   │   ├── 📄 text_processor.py               📝 TEXT CLEANING
│   │   │   │  Lines: ~100
│   │   │   │  Purpose: Clean, normalize, and tokenize text
│   │   │   │  Classes: TextProcessor
│   │   │   │  Methods:
│   │   │   │    - clean(text) → normalized text
│   │   │   │    - split_sentences(text) → List[str]
│   │   │   │    - tokenize(text) → List[str]
│   │   │   │  Used by: app.py, reasoning/verifiable_pipeline.py
│   │   │   │  Status: Production-ready
│   │   │   │
│   │   │
│   │   └── 📄 ocr_processor.py                📷 OCR EXTRACTION
│       │  Lines: ~150
│       │  Purpose: Extract text from images using EasyOCR
│       │  Classes: OCRProcessor
│       │  Models:
│       │    - EasyOCR (language-specific, ~200MB per language)
│       │  Methods:
│       │    - extract_text(image) → str
│       │    - extract_with_confidence(image) → List[(text, confidence)]
│       │  Quality: 70-90% accuracy depending on image
│       │  Used by: app.py
│       │  Status: Production-ready
│       │
│
│   ├── 📁 audio/                             🎵 AUDIO PROCESSING
│   │   │  Purpose: Transcribe audio to text
│   │   │
│   │   ├── 📄 __init__.py                     (Package marker)
│   │   │
│   │   └── 📄 whisper_transcriber.py          🎙️ SPEECH-TO-TEXT
│       │  Lines: ~100
│       │  Purpose: Transcribe audio using OpenAI Whisper
│       │  Classes: WhisperTranscriber
│       │  Models:
│       │    - Whisper (base/small/medium/large, ~1-3GB)
│       │  Methods:
│       │    - transcribe(audio_file) → str
│       │    - transcribe_with_timestamps(audio_file) → List[(text, timestamp)]
│       │  Used by: app.py
│       │  Status: Production-ready
│       │
│
│   ├── 📁 evaluation/                        📊 ANALYSIS & METRICS
│   │   │  Purpose: Evaluate and calibrate verification system
│   │   │
│   │   ├── 📄 __init__.py                     (Package marker)
│   │   │
│   │   ├── 📄 verifiability_metrics.py        📈 BASIC METRICS
│   │   │   │  Lines: ~100
│   │   │   │  Purpose: Compute verification rates
│   │   │   │  Metrics:
│   │   │   │    - Rejection rate (% REJECTED)
│   │   │   │    - Verification rate (% VERIFIED)
│   │   │   │    - Uncertainty rate (% LOW_CONFIDENCE)
│   │   │   │  Used by: app.py, evaluation/compare_modes.py
│   │   │   │  Status: Production-ready
│   │   │   │
│   │   │
│   │   └── 📄 calibration.py                  🎯 CALIBRATION METRICS ✨ NEW
│       │  Lines: 400+
│       │  Purpose: Assess and improve confidence calibration
│       │  Classes: CalibrationAnalyzer
│       │  Metrics:
│       │    - ECE (Expected Calibration Error)
│       │    - Brier score
│       │    - Accuracy metrics
│       │    - Confidence-accuracy bins
│       │  Methods:
│       │    - compute_ece(predictions, labels) → float
│       │    - compute_brier_score(predictions, labels) → float
│       │    - plot_reliability_diagram() → Image
│       │  Used by: app.py, evaluation/compare_modes.py
│       │  Status: Production-ready (Feb 2025)
│       │
│
│   ├── 📁 display/                           🎨 UI COMPONENTS
│   │   │  Purpose: Streamlit UI rendering
│   │   │
│   │   ├── 📄 __init__.py                     (Package marker)
│   │   │
│   │   ├── 📄 interactive_claims.py           📋 CLAIMS TABLE
│   │   │   │  Lines: ~200
│   │   │   │  Purpose: Render claims in interactive Streamlit table
│   │   │   │  Functions:
│   │   │   │    - display_claims_interactive(claims)
│   │   │   │    - filter_by_status(claims, status)
│   │   │   │    - sort_by_confidence(claims)
│   │   │   │  Features:
│   │   │   │    - Search across claims
│   │   │   │    - Filter by status
│   │   │   │    - Sort by confidence
│   │   │   │  Used by: app.py
│   │   │   │  Status: Production-ready
│   │   │   │
│   │   │
│   │   ├── 📄 research_assessment_ui.py       📊 METRICS DASHBOARD
│   │   │   │  Lines: ~300
│   │   │   │  Purpose: Render calibration metrics + plots
│   │   │   │  Functions:
│   │   │   │    - display_metrics_summary()
│   │   │   │    - display_reliability_diagram()
│   │   │   │    - display_confidence_distribution()
│   │   │   │  Used by: app.py
│   │   │   │  Status: Production-ready
│   │   │   │
│   │   │
│   │   └── 📄 streamlit_display.py            🎨 DISPLAY UTILS
│       │  Lines: ~150
│       │  Purpose: General Streamlit display utilities
│       │  Functions:
│       │    - show_metric_cards(metrics)
│       │    - show_graph(graph)
│       │    - show_download_buttons(results)
│       │  Used by: app.py
│       │  Status: Production-ready
│       │
│
│   ├── 📁 study_book/                        📚 SESSION AGGREGATION
│   │   │  Purpose: Combine results from multiple sessions
│   │   │
│   │   ├── 📄 __init__.py                     (Package marker)
│   │   │
│   │   └── 📄 aggregator.py                   📖 STUDY GUIDE GENERATION
│       │  Lines: ~200
│       │  Purpose: Aggregate claims from multiple sessions into study guide
│       │  Classes: StudyBookAggregator
│       │  Methods:
│       │    - aggregate_sessions(sessions) → StudyGuide
│       │    - dedup_claims() → unique claims
│       │    - organize_by_topic() → topic hierarchy
│       │  Used by: app.py (advanced features)
│       │  Status: Production-ready
│       │
│
│   └── 📁 schema/                            📊 SCHEMA DEFINITIONS
       │  Purpose: Shared schema definitions
       │
       └── 📄 __init__.py                     (Package marker)
          │
          └── (Most schemas in claims/schema.py)
          │
│
│═══════════════════════════════════════════════════════════════════════════
│  evaluation/ DIRECTORY (Standalone Analysis Tools)
│═══════════════════════════════════════════════════════════════════════════
│
├── 📁 evaluation/
│   │
│   ├── 📄 __init__.py                         (Package marker)
│   │
│   ├── 📄 compare_modes.py                    🔀 MODE COMPARISON ✨ NEW
│   │   │  Lines: 500+
│   │   │  Purpose: Compare Baseline vs Verifiable mode performance
│   │   │  Classes: ModeComparator
│   │   │  Analysis:
│   │   │    - Speed comparison
│   │   │    - Accuracy comparison
│   │   │    - Resource usage
│   │   │  Output: Side-by-side report
│   │   │  Status: Production-ready (Feb 2025)
│   │   │
│   │
│   └── 📄 benchmark.py                       ⚡ PERFORMANCE TESTS
       │  Lines: ~200
       │  Purpose: Benchmark system performance
       │  Tests:
       │    - Generation speed
       │    - Verification speed
       │    - Memory usage
       │  Used by: Developers, performance tuning
       │  Status: Production-ready
       │
│
│═══════════════════════════════════════════════════════════════════════════
│  tests/ DIRECTORY (Test Suite - 21/21 PASSING)
│═══════════════════════════════════════════════════════════════════════════
│
├── 📁 tests/
│   │
│   ├── 📄 __init__.py                         (Package marker)
│   │
│   ├── 📄 test_graph_sanitize.py              ✅ GRAPH TESTS ✨ NEW (Feb 2025)
│   │   │  Lines: 378
│   │   │  Tests: 18 unit tests
│   │   │  Coverage:
│   │   │    - Sanitization (bytes, enums, Pydantic, dicts, strings)
│   │   │    - GraphML export (string, bytes)
│   │   │    - Graph attribute handling
│   │   │    - GraphMetrics backward compatibility
│   │   │  Status: ✅ ALL PASSING (0.29s)
│   │   │
│   │
│   ├── 📄 test_integration_graph_fixes.py     ✅ INTEGRATION TESTS ✨ NEW (Feb 2025)
│   │   │  Lines: 258
│   │   │  Tests: 4 end-to-end tests
│   │   │  Coverage:
│   │   │    - GraphMetrics.get() compatibility
│   │   │    - GraphML export with complex attributes
│   │   │    - ClaimGraph integration
│   │   │    - Pydantic model sanitization
│   │   │  Status: ✅ ALL PASSING
│   │   │
│   │
│   └── 📄 pytest.ini                         ⚙️ PYTEST CONFIG
       │  Configuration for test discovery
       │  Settings: testpaths, python_files, etc.
       │
│
│═══════════════════════════════════════════════════════════════════════════
│  docs/ DIRECTORY (Technical Documentation)
│═══════════════════════════════════════════════════════════════════════════
│
├── 📁 docs/
│   │
│   ├── 📄 README.md                          📖 Technical README
│   │   │  Technical overview and getting started
│   │   │
│   │
│   ├── 📄 ARCHITECTURE.md                    🏗️ ARCHITECTURE (via TECHNICAL_DOCUMENTATION.md)
│   │   │  System architecture and design
│   │   │
│   │
│   ├── 📄 CHANGELOG_FEB2025.md                📋 CHANGELOG ✨ NEW
│   │   │  Lines: ~600
│   │   │  Detailed changelog of Feb 2025 fixes
│   │   │  Sections:
│   │   │    - Breaking changes (none)
│   │   │    - New features
│   │   │    - Bug fixes (7 critical issues)
│   │   │    - Migration guide
│   │   │  Status: Complete (Feb 2025)
│   │   │
│   │
│   ├── 📄 IMPLEMENTATION_SUMMARY.md           📄 IMPLEMENTATION ✨ NEW
│   │   │  Lines: ~1000
│   │   │  Detailed implementation guide
│   │   │  Covers:
│   │   │    - Architecture and components
│   │   │    - Data flow
│   │   │    - Module specifications
│   │   │    - Integration points
│   │   │  Status: Complete (Feb 2025)
│   │   │
│   │
│   ├── 📄 COMPLETION_REPORT.md                ✅ COMPLETION ✨ NEW
│   │   │  Lines: ~600
│   │   │  Final project status report
│   │   │  Contents:
│   │   │    - All tasks completed (8/8)
│   │   │    - All tests passing (21/21)
│   │   │    - Performance metrics
│   │   │    - Production status
│   │   │  Status: Complete (Feb 2025)
│   │   │
│   │
│   ├── 📄 TECHNICAL_DOCUMENTATION.md          🔧 TECHNICAL DOCS ✨ NEW
│   │   │  Lines: 3000+
│   │   │  Complete technical documentation
│   │   │  Includes:
│   │   │    - System architecture
│   │   │    - Data flow diagrams
│   │   │    - File structure
│   │   │    - Module specs
│   │   │    - Component interactions
│   │   │    - Data models
│   │   │    - APIs
│   │   │    - Algorithms
│   │   │    - Configuration
│   │   │    - Error handling
│   │   │  Status: Complete (Feb 2025)
│   │   │
│   │
│   ├── 📄 FILE_STRUCTURE.md                   📁 FILE STRUCTURE ✨ NEW
│   │   │  Lines: 1000+
│   │   │  Complete file structure documentation
│   │   │  Includes:
│   │   │    - File tree
│   │   │    - Directory organization
│   │   │    - Module dependencies
│   │   │    - File categories
│   │   │    - Import patterns
│   │   │    - Cache/output structure
│   │   │  Status: This file!
│   │   │
│   │
│   └── 📄 API.md                             🔌 API REFERENCE
       │  API specifications and usage examples
       │
│
│═══════════════════════════════════════════════════════════════════════════
│  examples/ DIRECTORY (Usage Examples & Demo Data)
│═══════════════════════════════════════════════════════════════════════════
│
├── 📁 examples/
│   │
│   ├── 📄 __init__.py                         (Package marker)
│   │
│   ├── 📄 demo_usage.py                       💡 USAGE EXAMPLES
│   │   │  Lines: ~200
│   │   │  Purpose: Demonstrate API usage
│   │   │  Examples:
│   │   │    - Baseline pipeline
│   │   │    - Verifiable pipeline
│   │   │    - Graph visualization
│   │   │    - Export formats
│   │   │  Status: Production-ready
│   │   │
│   │
│   ├── 📄 verifiable_mode_demo.py             📊 VERIFIABLE DEMO ✨ NEW
│   │   │  Lines: ~300
│   │   │  Purpose: Demonstrate verifiable mode in detail
│   │   │  Includes:
│   │   │    - Step-by-step verification
│   │   │    - Confidence scoring
│   │   │    - Graph analysis
│   │   │  Status: Production-ready (Feb 2025)
│   │   │
│   │
│   ├── 📄 sample_input.json                   📋 SAMPLE INPUT
│   │   │  Example JSON input for testing
│   │   │
│   │
│   ├── 📄 README_EXAMPLES.md                  📖 EXAMPLES README
│   │   │  Usage guide for examples/
│   │   │
│   │
│   ├── 📁 audio/                             🎵 AUDIO SAMPLES
│   │   │  Audio files for testing
│   │   │  (If used in development)
│   │   │
│   │
│   ├── 📁 inputs/                            📥 TEST INPUTS
│   │   │
│   │   ├── 📄 example1.json                  Sample input 1
│   │   └── 📄 example2.json                  Sample input 2
│   │
│   └── 📁 notes/                             📝 SAMPLE NOTES
       │
       ├── 📄 notes1.txt                      Sample notes 1
       └── 📄 notes2.txt                      Sample notes 2
│
│═══════════════════════════════════════════════════════════════════════════
│  DATA DIRECTORIES (Auto-created at runtime)
│═══════════════════════════════════════════════════════════════════════════
│
├── 📁 outputs/                               📤 USER OUTPUTS (Auto-created)
│   │
│   ├── 📁 sessions/                          💾 SAVED SESSIONS
│   │   │  JSON files of processed sessions
│   │   │
│   │   ├── 📄 session_20260131_163827.json   Session result (JSON)
│   │   ├── 📄 session_20260131_163906.json
│   │   ├── 📄 session_20260201_103004.json
│   │   ├── 📄 session_20260209_175725.json
│   │   └── ... (30+ session files)
│   │
│   │  Each contains:
│   │    - All claims with confidence + status
│   │    - Evidence for each claim
│   │    - Graph metrics
│   │    - Timestamp
│   │
│   └── 📁 evaluation/                        📊 ANALYSIS OUTPUTS
       │  Calibration plots and reports
       │
       ├── 📄 calibration_metrics.json        ECE, Brier score
       ├── 📄 reliability_diagram.png         Calibration plot
       └── 📄 mode_comparison.html            Baseline vs Verifiable
│
├── 📁 cache/                                 💾 LOCAL CACHING
│   │
│   ├── 📄 ocr_cache.json                    ⚡ OCR Results Cache
│   │   │  Caches EasyOCR results to avoid re-processing
│   │   │  Format: {image_hash: {text, confidence, timestamp}}
│   │   │
│   │
│   ├── 📁 faiss_index/                      🔍 FAISS INDEXES
│   │   │
│   │   ├── 📄 index_session1.faiss          Per-session FAISS index
│   │   ├── 📄 index_session2.faiss
│   │   └── 📄 metadata_session1.json        Index metadata
│   │
│   │  Enables fast evidence retrieval
│   │  Built from source corpus
│   │
│   └── 📁 api_responses/                    🤖 API RESPONSE CACHE
       │  Cache LLM responses to avoid re-querying
       │  Reduces API costs
       │
       ├── 📄 openai_response_hash1.json
       └── 📄 openai_response_hash2.json
│
└── 📁 logs/                                  📊 LOGGING OUTPUT
    │
    ├── 📄 app.log                            📝 ALL LOGS
    │   │  All log messages (DEBUG, INFO, WARNING)
    │   │  Format: timestamp - module - level - message
    │   │
    │
    └── 📄 errors.log                         ⚠️ ERROR LOGS
        │  Errors only (ERROR, CRITICAL)
        │  For quick error diagnosis
        │
```

---

## Directory Organization

### Source Code Organization (src/)

```
src/
├── Core Infrastructure (utility + abstraction)
│   ├── llm_provider.py       ─→ LLM abstraction
│   ├── logging_config.py     ─→ Logging setup
│   ├── output_formatter.py   ─→ Export formatting
│   └── streamlit_display.py  ─→ UI utilities
│
├── Claim Processing (claims/)
│   ├── schema.py             ─→ Data models
│   ├── extractor.py          ─→ Extract claims
│   ├── validator.py          ─→ Assign status
│   ├── nli_verifier.py       ─→ Verify entailment
│   └── confidence.py         ─→ Score confidence
│
├── Evidence Retrieval (retrieval/)
│   ├── claim_rag.py          ─→ Legacy keyword search
│   └── semantic_retriever.py ─→ Dense + re-rank
│
├── LLM Pipelines (reasoning/)
│   ├── pipeline.py           ─→ Baseline generation
│   ├── verifiable_pipeline.py ─→ Verification orchestration
│   └── prompts.py            ─→ LLM prompts
│
├── Graph Analysis (graph/)
│   ├── claim_graph.py        ─→ Build + analyze
│   └── graph_sanitize.py     ─→ Export sanitization
│
├── Input Processing
│   ├── preprocessing/
│   │   ├── text_processor.py ─→ Text cleaning
│   │   └── ocr_processor.py  ─→ OCR extraction
│   │
│   └── audio/
│       └── whisper_transcriber.py ─→ Speech-to-text
│
├── Analysis (evaluation/)
│   ├── verifiability_metrics.py ─→ Basic metrics
│   └── calibration.py        ─→ Calibration analysis
│
├── UI Components (display/)
│   ├── interactive_claims.py ─→ Claims table
│   ├── research_assessment_ui.py ─→ Metrics dashboard
│   └── streamlit_display.py  ─→ General utilities
│
├── Advanced Features
│   ├── study_book/
│   │   └── aggregator.py     ─→ Multi-session aggregation
│   │
│   └── schema/               ─→ (Mostly in claims/schema.py)
```

### Dependency Hierarchy

```
Top Level (User-facing):
  app.py (Streamlit UI)
    ↓
  reasoning/ (LLM Pipelines)
    ├─→ pipeline.py (Baseline generation)
    └─→ verifiable_pipeline.py (Orchestration)
         ├─→ claims/ (Claim processing)
         │    ├─→ extractor.py
         │    ├─→ nli_verifier.py
         │    ├─→ confidence.py
         │    └─→ validator.py
         ├─→ retrieval/ (Evidence search)
         │    └─→ semantic_retriever.py
         └─→ graph/ (Graph analysis)
              ├─→ claim_graph.py
              └─→ graph_sanitize.py

Support Layers:
  preprocessing/ (Text/image/audio processing)
  evaluation/ (Metrics + calibration)
  display/ (UI components)
  output_formatter.py (Export formatting)
  llm_provider.py (LLM abstraction)
  logging_config.py (Logging)

Data:
  claims/schema.py (Data models)
  graph_sanitize.py (Type conversion)
```

---

## Module Dependency Graph

```
                          ┌─────────────┐
                          │   app.py    │ (Streamlit UI)
                          │  (Main)     │
                          └──────┬──────┘
                                 │
           ┌─────────────────────┼─────────────────────┐
           │                     │                     │
           ↓                     ↓                     ↓
      ┌─────────┐        ┌──────────────┐     ┌────────────────┐
      │ Baseline│        │  Verifiable  │     │  Display/Export│
      │Pipeline │        │  Pipeline    │     │  Components    │
      │ (Fast)  │        │  (Detailed)  │     │                │
      └────┬────┘        └──────┬───────┘     └────────┬───────┘
           │                    │                      │
           │             ┌──────┴──────────────┐      │
           │             │                     │      │
           ↓             ↓                     ↓      ↓
      ┌────────────────────────────────────────────────────┐
      │        Claim Processing (claims/)                  │
      │  ├─ schema.py (Data models)                        │
      │  ├─ extractor.py (Extract claims)                  │
      │  ├─ nli_verifier.py (Verify entailment)            │
      │  ├─ confidence.py (Score confidence)               │
      │  └─ validator.py (Assign status)                   │
      └────────────────┬───────────────────────────────────┘
                       │
         ┌─────────────┼─────────────┐
         │             │             │
         ↓             ↓             ↓
    ┌────────┐  ┌──────────┐  ┌──────────────┐
    │ Graph  │  │ Evidence │  │ Evaluation   │
    │ (graph)│  │ Search   │  │ (evaluation) │
    │        │  │(retrieval)│  │              │
    └────────┘  └──────────┘  └──────────────┘
         │             │             │
         ↓             ↓             ↓
  ┌─────────────────────────────────────────┐
  │        Utilities & Infrastructure       │
  │  ├─ preprocessing/ (Input cleaning)     │
  │  ├─ audio/ (Speech-to-text)             │
  │  ├─ output_formatter.py (Export)        │
  │  ├─ llm_provider.py (LLM abstraction)   │
  │  ├─ logging_config.py (Logging)         │
  │  └─ streamlit_display.py (UI utils)     │
  └─────────────────────────────────────────┘
```

---

## File Categories & Purposes

### By Functional Area

#### **LLM Integration**
- `src/llm_provider.py` - Abstraction for OpenAI/Ollama
- `src/reasoning/pipeline.py` - Baseline generation
- `src/reasoning/verifiable_pipeline.py` - Verification orchestration
- `src/reasoning/prompts.py` - LLM prompts

#### **Data Models**
- `src/claims/schema.py` - Pydantic models (LearningClaim, EvidenceItem, etc.)

#### **Verification Pipeline**
- `src/retrieval/semantic_retriever.py` - Evidence retrieval (FAISS)
- `src/claims/nli_verifier.py` - Entailment verification (NLI)
- `src/claims/confidence.py` - Confidence scoring
- `src/claims/validator.py` - Status assignment

#### **Input Processing**
- `src/preprocessing/text_processor.py` - Text cleaning
- `src/preprocessing/ocr_processor.py` - Image OCR
- `src/audio/whisper_transcriber.py` - Speech-to-text

#### **Graph Analysis**
- `src/graph/claim_graph.py` - Graph construction
- `src/graph/graph_sanitize.py` - Export sanitization

#### **Analysis & Metrics**
- `src/evaluation/verifiability_metrics.py` - Rejection/verification rates
- `src/evaluation/calibration.py` - ECE and calibration

#### **UI & Display**
- `app.py` - Main Streamlit application
- `src/display/interactive_claims.py` - Claims table
- `src/display/research_assessment_ui.py` - Metrics dashboard
- `src/streamlit_display.py` - Display utilities

#### **Export & Formatting**
- `src/output_formatter.py` - JSON/CSV/Markdown export
- `src/graph/graph_sanitize.py` - GraphML export

#### **Configuration**
- `config.py` - Global parameters
- `.env` - Secrets (user-created)

#### **Infrastructure**
- `src/logging_config.py` - Logging setup
- `src/llm_provider.py` - LLM provider abstraction

---

## Import Patterns

### Standard Imports (Production)

```python
# Standard library
import os
import json
import logging
from typing import List, Dict, Optional
from datetime import datetime
import uuid

# Third-party
import numpy as np
import pandas as pd
import streamlit as st
import networkx as nx
from pydantic import BaseModel, Field

# From transformers (NLP models)
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import faiss

# Smart Notes modules
from src.claims.schema import LearningClaim, EvidenceItem, GraphMetrics
from src.retrieval.semantic_retriever import SemanticRetriever
from src.claims.nli_verifier import NLIVerifier
from src.claims.confidence import ConfidenceCalculator
from src.graph.claim_graph import ClaimGraph
from src.output_formatter import OutputFormatter
from config import VERIFIED_THRESHOLD, NLI_MODEL
```

### Common Import Blocks by Module

#### **Claim Processing** (src/claims/*.py)
```python
from src.claims.schema import LearningClaim, EvidenceItem, VerificationStatus
from config import VERIFIED_THRESHOLD, REJECT_THRESHOLD
import logging
logger = logging.getLogger(__name__)
```

#### **Graph Operations** (src/graph/*.py)
```python
import networkx as nx
from src.claims.schema import LearningClaim, GraphMetrics
from src.graph.graph_sanitize import sanitize_graph_for_graphml
```

#### **LLM Pipelines** (src/reasoning/*.py)
```python
from src.llm_provider import LLMProvider
from src.claims.extractor import ClaimExtractor
from src.reasoning.prompts import GENERATION_PROMPT
```

#### **Verification Pipeline** (src/reasoning/verifiable_pipeline.py)
```python
from src.retrieval.semantic_retriever import SemanticRetriever
from src.claims.nli_verifier import NLIVerifier
from src.claims.confidence import ConfidenceCalculator
from src.claims.validator import ClaimValidator
from src.graph.claim_graph import ClaimGraph
```

---

## Configuration & Secrets

### Configuration Files

#### **config.py** (Production Parameters)
```python
# Model selection
EMBEDDING_MODEL = "intfloat/e5-base-v2"
NLI_MODEL = "facebook/bart-large-mnli"
LLM_PROVIDER = "openai"  # or "ollama"

# Thresholds
VERIFIED_THRESHOLD = 0.7
REJECT_THRESHOLD = 0.3

# Retrieval
SEMANTIC_TOP_K = 10
RERANK_TOP_N = 5

# Confidence weights (must sum ≈ 1.0)
CONFIDENCE_WEIGHTS = {
    'similarity': 0.25,
    'entailment': 0.35,
    'diversity': 0.10,
    'count': 0.15,
    'contradiction': -0.10,
    'graph': 0.05
}
```

#### **.env** (Secrets, User-created)
```
# API Keys
OPENAI_API_KEY=sk-...
OLLAMA_BASE_URL=http://localhost:11434

# LLM Provider
LLM_PROVIDER=openai
LLM_MODEL=gpt-4

# Storage
CACHE_DIR=./cache
OUTPUT_DIR=./outputs
LOG_DIR=./logs

# Feature Flags
ENABLE_NLI=true
ENABLE_CALIBRATION=false
```

#### **.env.example** (Template)
```
# Example environment file
# Copy to .env and fill in your values

OPENAI_API_KEY=sk-your-key-here
OLLAMA_BASE_URL=http://localhost:11434
LLM_PROVIDER=openai
LLM_MODEL=gpt-4
```

---

## Cache & Output Directories

### Cache Structure (Auto-created)

```
cache/
├── ocr_cache.json                     ⚡ EasyOCR results
│   {
│       "image_hash_abc123": {
│           "text": "extracted text...",
│           "confidence": 0.85,
│           "timestamp": "2025-02-12T10:30:00"
│       }
│   }
│
├── faiss_index/
│   ├── index_session_id_1.faiss       FAISS binary index
│   ├── index_session_id_1_metadata.json
│   │   {
│   │       "num_docs": 1000,
│   │       "embedding_dim": 768,
│   │       "model": "intfloat/e5-base-v2",
│   │       "created": "2025-02-12T10:30:00"
│   │   }
│   └── ...
│
└── api_responses/
    ├── openai_hash_abc123.json        LLM response cache
    │   {
    │       "prompt": "Generate study notes...",
    │       "response": "Here are the key concepts...",
    │       "model": "gpt-4",
    │       "created": "2025-02-12T10:30:00",
    │       "cost_usd": 0.05
    │   }
    └── ...
```

### Output Structure (Auto-created)

```
outputs/
├── sessions/                          💾 All session results
│   ├── session_20260131_163827.json  (Full result as JSON)
│   │   {
│   │       "session_id": "uuid-1234",
│   │       "timestamp": "2025-01-31T16:38:27",
│   │       "mode": "verifiable",
│   │       "input_type": "text",
│   │       "claims": [...],
│   │       "metrics": {
│   │           "total_claims": 287,
│   │           "verified": 245,
│   │           "rejected": 32,
│   │           "avg_confidence": 0.72
│   │       },
│   │       "processing_time_seconds": 95.3
│   │   }
│   ├── session_20260201_103004.json
│   └── ... (30+ sessions)
│
└── evaluation/                        📊 Analysis outputs
    ├── calibration_metrics.json       (ECE, Brier score)
    ├── reliability_diagram.png        (Matplotlib plot)
    ├── confidence_distribution.png
    └── mode_comparison_report.html
```

---

## Test Organization

### Test Structure

```
tests/
├── __init__.py
├── pytest.ini                         Pytest configuration
│   [pytest]
│   testpaths = tests
│   python_files = test_*.py
│   python_classes = Test*
│   python_functions = test_*
│
├── test_graph_sanitize.py             ✨ GRAPH SANITIZATION (NEW - Feb 2025)
│   │
│   ├── class TestSanitizeValue
│   │   ├── test_sanitize_string()
│   │   ├── test_sanitize_bytes_utf8()
│   │   ├── test_sanitize_bytes_binary()
│   │   ├── test_sanitize_enum()
│   │   ├── test_sanitize_dict()
│   │   ├── test_sanitize_list()
│   │   ├── test_sanitize_pydantic_model()
│   │   ├── test_sanitize_datetime()
│   │   ├── test_sanitize_none()
│   │   ├── test_sanitize_long_string()
│   │   └── test_sanitize_nested_structures()
│   │
│   ├── class TestSanitizeGraphForGraphML
│   │   ├── test_graph_node_attributes()
│   │   ├── test_graph_edge_attributes()
│   │   ├── test_graph_with_mixed_types()
│   │   └── test_graph_preserves_structure()
│   │
│   ├── class TestExportGraphML
│   │   ├── test_export_graphml_string()
│   │   ├── test_export_graphml_bytes()
│   │   └── test_graphml_valid_xml()
│   │
│   └── class TestGraphMetricsCompatibility
│       ├── test_graphmetrics_to_dict()
│       └── test_graphmetrics_get_method()
│
│   Status: ✅ 18 tests, 100% passing (0.29s)
│
│
└── test_integration_graph_fixes.py    ✨ INTEGRATION TESTS (NEW - Feb 2025)
    │
    ├── class TestGraphFixes
    │   ├── test_graph_metrics_get()
    │   ├── test_graphml_export_with_complex_attributes()
    │   ├── test_claim_graph_integration()
    │   └── test_pydantic_model_in_graph()
    │
    Status: ✅ 4 tests, 100% passing
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_graph_sanitize.py

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=src

# Run single test
pytest tests/test_graph_sanitize.py::TestSanitizeValue::test_sanitize_enum
```

---

## Documentation Files

### Documentation Hierarchy

```
Root README.md (1585 lines)
├─ Quick Start
├─ Installation
├─ Usage Examples
├─ System Status ✅ Production Ready
├─ Performance Benchmarks
└─ Troubleshooting

docs/
├── README.md                          Technical README
├── TECHNICAL_DOCUMENTATION.md         🔧 COMPLETE TECHNICAL GUIDE
│   └─ Architecture, data flow, modules, algorithms
├── FILE_STRUCTURE.md                  📁 FILE STRUCTURE (THIS FILE)
│   └─ Complete file tree and organization
├── CHANGELOG_FEB2025.md               📋 CHANGELOG
│   └─ All changes in February 2025
├── IMPLEMENTATION_SUMMARY.md          📄 IMPLEMENTATION
│   └─ Detailed implementation guide
├── COMPLETION_REPORT.md               ✅ COMPLETION
│   └─ Final project status
└── API.md                             🔌 API REFERENCE
    └─ Public API specifications

examples/
├── demo_usage.py                      💡 USAGE EXAMPLES
├── verifiable_mode_demo.py            📊 VERIFIABLE DEMO
└── README_EXAMPLES.md                 📖 EXAMPLES README
```

---

**End of File Structure Documentation**

For navigation:
- Architecture & algorithms → [TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md)
- Changes & updates → [CHANGELOG_FEB2025.md](docs/CHANGELOG_FEB2025.md)
- API usage → [README.md](README.md) or examples/
- Project status → [COMPLETION_REPORT.md](docs/COMPLETION_REPORT.md)
