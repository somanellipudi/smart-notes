"""Schema validation for run history entries."""

REQUIRED_KEYS = {
    "run_id",
    "session_id",
    "timestamp",
    "domain_profile",
    "llm_model",
    "embedding_model",
    "nli_model",
    "inputs_used",
    "ingestion_stats",
    "verification_stats",
    "artifact_paths",
}


def test_run_history_schema():
    sample = {
        "run_id": "run_20260218_123456_abcd1234",
        "session_id": "session_001",
        "timestamp": "2026-02-18T12:34:56",
        "domain_profile": "Computer Science",
        "llm_model": "gpt-4",
        "embedding_model": "intfloat/e5-base-v2",
        "nli_model": "cross-encoder/qnli",
        "inputs_used": ["pdf", "text"],
        "ingestion_stats": {"pages": 10, "ocr_pages": 2},
        "verification_stats": {
            "total_claims": 5,
            "verified": 3,
            "rejected": 1,
            "low_conf": 1,
            "avg_conf": 0.72,
        },
        "artifact_paths": {
            "report_md": "artifacts/session_001/run_1/research_report.md",
            "report_html": "artifacts/session_001/run_1/research_report.html",
            "audit_json": "artifacts/session_001/run_1/research_report_audit.json",
            "metrics_json": "artifacts/session_001/run_1/metrics.json",
            "graphml": "artifacts/session_001/run_1/claim_graph.graphml",
        },
    }

    missing = REQUIRED_KEYS - set(sample.keys())
    assert not missing
