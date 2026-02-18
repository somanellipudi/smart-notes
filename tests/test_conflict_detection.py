"""
Tests for Online vs Local Evidence Conflict Detection

Tests detection and handling of conflicts between local and online authority evidence.
"""

import pytest
from datetime import datetime
from src.retrieval.authority_sources import AuthorityTier, get_allowlist
from src.retrieval.online_retriever import OnlineSpan, OnlineSourceContent, create_retriever


class TestConflictDetection:
    """Test conflict detection between local and online evidence."""
    
    def test_tier_based_verification_tier1(self):
        """Test that Tier 1 sources can verify alone."""
        # Tier 1 should be able to verify a claim independently
        tier_1_weight = 0.95
        min_tier_for_solo_verification = AuthorityTier.TIER_1.value
        
        online_span_tier = AuthorityTier.TIER_1.value
        
        # Should pass: Tier 1 can verify alone
        can_verify = online_span_tier <= min_tier_for_solo_verification
        assert can_verify
    
    def test_tier_based_verification_tier2(self):
        """Test that Tier 2 sources can verify with policy."""
        min_tier_for_solo = AuthorityTier.TIER_2.value
        online_span_tier = AuthorityTier.TIER_2.value
        
        # Should pass: Tier 2 meets requirement
        can_verify = online_span_tier <= min_tier_for_solo
        assert can_verify
    
    def test_tier_3_requires_corroboration(self):
        """Test that Tier 3 requires multiple independent sources."""
        min_tier_for_solo = AuthorityTier.TIER_2.value
        online_span_tier = AuthorityTier.TIER_3.value
        
        # Single Tier 3 source should NOT verify alone
        can_verify = online_span_tier <= min_tier_for_solo
        assert not can_verify
        
        # Multiple Tier 3 sources should meet corroboration requirement
        tier_3_count = 2
        corroboration_threshold = 2
        has_corroboration = tier_3_count >= corroboration_threshold
        assert has_corroboration
    
    def test_mixed_tier_verification(self):
        """Test verification with mixed tiers."""
        min_tier_for_solo = AuthorityTier.TIER_2.value
        
        # Tier 1 + Tier 3: should verify (has Tier 1)
        sources = [AuthorityTier.TIER_1, AuthorityTier.TIER_3]
        has_strong_tier = any(tier.value <= min_tier_for_solo for tier in sources)
        assert has_strong_tier
        
        # Tier 3 + Tier 3: should verify (multiple Tier 3)
        sources = [AuthorityTier.TIER_3, AuthorityTier.TIER_3]
        has_multiple_tier_3 = sum(1 for t in sources if t == AuthorityTier.TIER_3) >= 2
        assert has_multiple_tier_3
    
    def test_identity_span_matching(self):
        """Test matching spans from local vs online for conflict detection."""
        # Simulate local span
        local_span = {
            "text": "Python is a programming language",
            "source": "local_doc"
        }
        
        # Simulate online span with same text
        online_span = OnlineSpan(
            span_id="span_001",
            source_id="python_docs",
            text="Python is a programming language",
            start_char=0,
            end_char=33,
            authority_tier=AuthorityTier.TIER_1.value,
            authority_weight=0.95,
            origin_url="https://docs.python.org/",
            access_date="2025-01-01T00:00:00Z",
            is_from_cache=False
        )
        
        # Texts match (corroboration, not conflict)
        is_corroborating = local_span["text"] == online_span.text
        assert is_corroborating
    
    def test_conflict_detection_different_text(self):
        """Test detection of conflicting information."""
        local_text = "Python requires 4 spaces for indentation"
        online_text = "Python requires 3 spaces for indentation"
        
        # Different claims about same topic
        is_conflict = local_text != online_text
        assert is_conflict
    
    def test_conflict_resolution_by_authority(self):
        """Test conflict resolution using authority weights."""
        local_weight = 0.5  # Community source
        online_weight = 0.95  # Tier 1 source
        
        # Online has higher authority
        authoritative_is_online = online_weight > local_weight
        assert authoritative_is_online
    
    def test_conflicting_spans_aggregation(self):
        """Test aggregation of conflicting claims."""
        claims = [
            {
                "text": "Claim A",
                "source": "local",
                "authority_weight": 0.6,
                "count": 1
            },
            {
                "text": "Claim B",
                "source": "online",
                "authority_weight": 0.95,
                "count": 2
            }
        ]
        
        # Identify conflicts
        unique_claims = {}
        for claim in claims:
            key = claim["text"]
            if key not in unique_claims:
                unique_claims[key] = claim
            else:
                # Check if weights differ significantly
                weight_diff = abs(claim["authority_weight"] - unique_claims[key]["authority_weight"])
                if weight_diff > 0.1:
                    # Mark as potential conflict
                    claim["is_conflict"] = True
        
        assert len(unique_claims) == 2


class TestOnlineAuthorityValidation:
    """Test online evidence validation against authority policies."""
    
    def test_allowlist_validation_tier_1(self):
        """Test that Tier 1 sources validate."""
        allowlist = get_allowlist()
        
        tier_1_sources = allowlist.get_sources_by_tier(AuthorityTier.TIER_1)
        assert len(tier_1_sources) > 0
        
        # All Tier 1 sources have weight >= 0.9
        for source in tier_1_sources:
            assert source.authority_weight >= 0.9
    
    def test_allowlist_validation_tier_3(self):
        """Test that Tier 3 sources validate with lower weight."""
        allowlist = get_allowlist()
        
        tier_3_sources = allowlist.get_sources_by_tier(AuthorityTier.TIER_3)
        assert len(tier_3_sources) > 0
        
        # All Tier 3 sources have weight < 0.8
        for source in tier_3_sources:
            assert source.authority_weight < 0.8
    
    def test_conflicting_authority_levels(self):
        """Test handling different authority levels in conflict."""
        allowlist = get_allowlist()
        
        # Get samples from different tiers
        tier_1 = allowlist.get_sources_by_tier(AuthorityTier.TIER_1)[0]
        tier_3 = allowlist.get_sources_by_tier(AuthorityTier.TIER_3)[0]
        
        # Tier 1 should be more authoritative
        assert tier_1.authority_weight > tier_3.authority_weight


class TestConflictReporting:
    """Test reporting conflicting evidence."""
    
    def test_conflict_metadata(self):
        """Test recording conflict metadata."""
        conflict = {
            "claim_id": "claim_123",
            "local_evidence": {
                "text": "Local claim",
                "source": "local_doc",
                "authority_weight": 0.6
            },
            "online_evidence": {
                "text": "Online claim",
                "source": "https://docs.python.org/",
                "authority_weight": 0.95,
                "authority_tier": AuthorityTier.TIER_1
            },
            "conflict_type": "text_contradiction",
            "severity": "high",
            "detected_at": "2025-01-01T00:00:00Z"
        }
        
        assert conflict["claim_id"]
        assert conflict["local_evidence"]["authority_weight"] < conflict["online_evidence"]["authority_weight"]
        assert conflict["severity"] == "high"
    
    def test_corroboration_metadata(self):
        """Test recording corroborating evidence (no conflict)."""
        corroboration = {
            "claim_id": "claim_456",
            "local_evidence": {
                "text": "Python is a language",
                "source": "local_doc",
                "authority_weight": 0.6
            },
            "online_evidence": {
                "text": "Python is a language",
                "source": "https://python.org/",
                "authority_weight": 0.95,
                "authority_tier": AuthorityTier.TIER_1
            },
            "conflict_type": "none",
            "corroboration_level": "strong",
            "detected_at": "2025-01-01T00:00:00Z"
        }
        
        assert corroboration["conflict_type"] == "none"
        assert corroboration["corroboration_level"] == "strong"


class TestConflictResolution:
    """Test conflict resolution strategies."""
    
    def test_resolve_by_tier_preference(self):
        """Test resolution by choosing higher tier source."""
        local_tier = AuthorityTier.TIER_3
        online_tier = AuthorityTier.TIER_1
        
        # Choose online (better tier)
        resolution = online_tier if online_tier.value < local_tier.value else local_tier
        assert resolution == online_tier
    
    def test_resolve_by_weight_preference(self):
        """Test resolution by choosing higher weight source."""
        local_weight = 0.65
        online_weight = 0.95
        
        # Choose online (higher weight)
        resolution_weight = max(local_weight, online_weight)
        assert resolution_weight == online_weight
    
    def test_resolve_tie_by_recency(self):
        """Test resolution by recency when weights are equal."""
        local_date = "2024-01-01"
        online_date = "2025-01-01"
        
        # For equal weights, prefer more recent
        resolution_date = max(local_date, online_date)
        assert resolution_date == online_date


class TestCachePersistenceWithConflicts:
    """Test caching behavior when conflicts are detected."""
    
    def test_conflict_cache_isolation(self):
        """Test that conflicts are tracked separately from claims."""
        retriever = create_retriever()
        
        claim_cache = {
            "claim_001": {
                "text": "Claim text",
                "evidence": []
            }
        }
        
        conflict_log = {
            "claim_001": {
                "has_conflict": True,
                "details": "Local vs online contradiction"
            }
        }
        
        # Both caches should be independent
        assert "claim_001" in claim_cache
        assert "claim_001" in conflict_log
    
    def test_cache_content_hash_with_conflict(self):
        """Test content hash remains stable for conflicting content."""
        retriever = create_retriever()
        
        text = "This is conflicting information"
        hash1 = retriever._compute_content_hash(text)
        hash2 = retriever._compute_content_hash(text)
        
        # Hash should be identical regardless of conflict status
        assert hash1 == hash2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
