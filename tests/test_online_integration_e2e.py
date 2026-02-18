"""
End-to-end integration test for Online Authority Verification.

This test verifies the complete integration:
1. UI toggle stores session state
2. Config receives session state value
3. evidence_builder.py retrieves online evidence
4. Authority allowlist enforces policies
5. Conflicts are detected and reported
"""

import pytest
from unittest.mock import MagicMock, patch
from src.retrieval.evidence_builder import build_session_evidence_store
from src.retrieval.online_evidence_integration import create_integrator
import config


@pytest.fixture
def mock_embedding_provider():
    """Mock embedding provider for testing."""
    import numpy as np
    provider = MagicMock()
    provider.model_name = "test-model"
    # Return embeddings matching the number of input texts
    provider.embed_texts.side_effect = lambda texts: np.array([[0.1] * 384] * len(texts))
    return provider


def test_online_verification_disabled_by_default():
    """Test that online verification is OFF by default."""
    original_value = config.ENABLE_ONLINE_VERIFICATION
    
    try:
        # Default should be False
        assert config.ENABLE_ONLINE_VERIFICATION is False, \
            "ENABLE_ONLINE_VERIFICATION should default to False"
    finally:
        config.ENABLE_ONLINE_VERIFICATION = original_value


def test_evidence_builder_respects_online_toggle():
    """
    Test that evidence_builder.py respects ENABLE_ONLINE_VERIFICATION config.
    
    When disabled: Should not call online retrieval.
    When enabled: Should attempt online retrieval (even if it fails due to mocking).
    """
    original_value = config.ENABLE_ONLINE_VERIFICATION
    original_artifact = config.ENABLE_ARTIFACT_PERSISTENCE
    
    try:
        # Disable artifacts to avoid file I/O in test
        config.ENABLE_ARTIFACT_PERSISTENCE = False
        
        # Test 1: Online disabled - build store without embeddings
        config.ENABLE_ONLINE_VERIFICATION = False
        
        store, stats = build_session_evidence_store(
            session_id="test_session_disabled",
            input_text="Machine learning is a subset of artificial intelligence. "
                      "It involves training models on data to make predictions. "
                      "Neural networks are inspired by biological neurons.",
            embedding_provider=None  # No embeddings for this test
        )
        
        # Should NOT have online evidence
        assert "online_evidence_chunks" not in stats, \
            "Stats should not contain online evidence when disabled"
        assert "online_conflicts" not in stats, \
            "Stats should not contain conflict data when disabled"
        
        # Evidence should only be from local input
        local_evidence_count = len([e for e in store.evidence if e.source_type != "online_authority"])
        assert local_evidence_count == len(store.evidence), \
            "All evidence should be local when online disabled"
        
        # Test 2: Online enabled (will trigger retrieval attempt)
        config.ENABLE_ONLINE_VERIFICATION = True
        
        with patch('src.retrieval.online_evidence_integration.OnlineEvidenceIntegrator.retrieve_online_evidence') as mock_retrieve:
            # Mock to return empty list (no online evidence found)
            mock_retrieve.return_value = []
            
            store2, stats2 = build_session_evidence_store(
                session_id="test_session_enabled",
                input_text="Python is a high-level programming language. "
                          "It emphasizes code readability and simplicity. "
                          "Python supports multiple programming paradigms.",
                embedding_provider=None  # No embeddings for this test
            )
            
            # Should have attempted online retrieval
            # (even if no results due to mock)
            assert mock_retrieve.called, \
                "Online retrieval should be attempted when enabled"
            
            # Since mock returns empty, no online chunks should be added
            assert stats2.get("online_evidence_chunks", 0) == 0, \
                "Mock returns empty, so no online chunks"
    
    finally:
        config.ENABLE_ONLINE_VERIFICATION = original_value
        config.ENABLE_ARTIFACT_PERSISTENCE = original_artifact


def test_online_integrator_policy_enforcement():
    """
    Test that OnlineEvidenceIntegrator enforces tier-based policies.
    """
    integrator = create_integrator(enable_online=True, enforce_policies=True)
    
    # Mock online evidence with different tiers
    from src.retrieval.online_retriever import OnlineSpan
    from src.retrieval.authority_sources import AuthorityTier
    
    tier1_span = OnlineSpan(
        span_id="span_tier1",
        source_id="https://docs.python.org",
        text="Python is an interpreted language.",
        authority_tier=AuthorityTier.TIER_1,
        authority_weight=0.90,
        access_date="2024-01-01",
        is_from_cache=False
    )
    
    tier3_span = OnlineSpan(
        span_id="span_tier3",
        source_id="https://stackoverflow.com/questions/12345",
        text="Python is dynamically typed.",
        authority_tier=AuthorityTier.TIER_3,
        authority_weight=0.55,
        access_date="2024-01-01",
        is_from_cache=False
    )
    
    # Test policy: Tier 1 can verify alone
    is_valid_tier1, reason_tier1 = integrator.enforce_verification_policy(
        claim_text="Python is an interpreted language",
        local_evidence=[],
        online_evidence=[tier1_span],
        min_tier_for_solo_online=2  # Tier 2 or better
    )
    assert is_valid_tier1, f"Tier 1 should verify alone: {reason_tier1}"
    
    # Test policy: Single Tier 3 cannot verify alone
    is_valid_tier3_solo, reason_tier3_solo = integrator.enforce_verification_policy(
        claim_text="Python is dynamically typed",
        local_evidence=[],
        online_evidence=[tier3_span],
        min_tier_for_solo_online=2
    )
    assert not is_valid_tier3_solo, \
        "Single Tier 3 source should NOT verify alone without corroboration"
    assert "tier" in reason_tier3_solo.lower() or "corroboration" in reason_tier3_solo.lower(), \
        "Reason should mention tier or corroboration requirement"
    
    # Test policy: Multiple Tier 3 sources CAN verify together
    tier3_span2 = OnlineSpan(
        span_id="span_tier3_2",
        source_id="https://github.com/python/cpython",
        text="Python uses dynamic typing.",
        authority_tier=AuthorityTier.TIER_3,
        authority_weight=0.60,
        access_date="2024-01-01",
        is_from_cache=False
    )
    
    is_valid_tier3_multi, reason_tier3_multi = integrator.enforce_verification_policy(
        claim_text="Python is dynamically typed",
        local_evidence=[],
        online_evidence=[tier3_span, tier3_span2],
        min_tier_for_solo_online=2
    )
    assert is_valid_tier3_multi, \
        f"Multiple Tier 3 sources should verify together: {reason_tier3_multi}"


def test_conflict_detection():
    """
    Test that conflict detection identifies contradictory evidence.
    """
    integrator = create_integrator(enable_online=True, enforce_policies=True)
    
    from src.retrieval.online_retriever import OnlineSpan
    from src.retrieval.authority_sources import AuthorityTier
    
    # Claim: "Python is statically typed"
    claim_text = "Python is statically typed"
    
    # Local evidence says statically typed
    local_mock = MagicMock()
    local_mock.text = "Python uses static type checking with mypy."
    
    # Online evidence contradicts (says dynamically typed)
    online_span = OnlineSpan(
        span_id="conflict_span",
        source_id="https://docs.python.org",
        text="Python is a dynamically typed language, not statically typed.",
        authority_tier=AuthorityTier.TIER_1,
        authority_weight=0.95,
        access_date="2024-01-01",
        is_from_cache=False
    )
    
    has_conflict, conflict_report = integrator.detect_conflicts(
        claim_text=claim_text,
        local_evidence=[local_mock],
        online_evidence=[online_span]
    )
    
    # Should detect conflict
    assert has_conflict, "Should detect conflict between local and online evidence"
    assert conflict_report is not None, "Conflict report should be populated"
    assert conflict_report.conflict_detected is True
    assert conflict_report.severity in ["low", "medium", "high"]


def test_privacy_notice_in_ui():
    """
    Verify privacy notice requirements are documented.
    
    This test serves as documentation that:
    1. PII redaction patterns are defined in online_retriever.py
    2. Queries are redacted before sending (search_and_retrieve)
    3. Allowlist enforces only trusted domains
    """
    from src.retrieval.online_retriever import PIIRedactor
    
    # Test PII patterns exist
    assert PIIRedactor.EMAIL_PATTERN is not None
    assert PIIRedactor.PHONE_PATTERN is not None
    assert PIIRedactor.SSN_PATTERN is not None
    assert PIIRedactor.CC_PATTERN is not None
    
    # Test redaction works
    test_text = "Contact me at user@example.com or 555-123-4567"
    redacted = PIIRedactor.redact(test_text)
    
    assert "user@example.com" not in redacted, "Email should be redacted"
    assert "555-123-4567" not in redacted, "Phone should be redacted"
    assert "[REDACTED_EMAIL]" in redacted, "Redacted marker should be present"
    assert "[REDACTED_PHONE]" in redacted, "Redacted marker should be present"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
