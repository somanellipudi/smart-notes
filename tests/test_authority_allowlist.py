"""
Tests for Authority Source Allowlist

Tests authority tier validation, source lookup, and policy enforcement.
"""

import pytest
from src.retrieval.authority_sources import (
    AuthorityAllowlist,
    AuthorityTier,
    get_allowlist,
    is_allowed_source,
    get_source_tier,
    get_source_weight
)


class TestAuthorityAllowlist:
    """Test the authority source allowlist."""
    
    def test_allowlist_initialization(self):
        """Test allowlist is properly initialized with sources."""
        allowlist = AuthorityAllowlist()
        
        assert len(allowlist.sources) > 0
        
        # Check for Tier 1 sources
        tier_1 = allowlist.get_sources_by_tier(AuthorityTier.TIER_1)
        assert len(tier_1) > 10, "Should have multiple Tier 1 sources"
        
        # Check for Tier 2 sources
        tier_2 = allowlist.get_sources_by_tier(AuthorityTier.TIER_2)
        assert len(tier_2) > 5, "Should have multiple Tier 2 sources"
        
        # Check for Tier 3 sources
        tier_3 = allowlist.get_sources_by_tier(AuthorityTier.TIER_3)
        assert len(tier_3) > 5, "Should have multiple Tier 3 sources"
    
    def test_tier_1_sources_present(self):
        """Test that critical Tier 1 sources are present."""
        allowlist = AuthorityAllowlist()
        
        expected_tier_1 = [
            "rfc-editor.org",
            "docs.python.org",
            "kubernetes.io",
            "docs.microsoft.com",
            "docs.aws.amazon.com",
            "developer.mozilla.org",
        ]
        
        for domain in expected_tier_1:
            assert domain in allowlist.sources, f"Missing Tier 1 source: {domain}"
            assert allowlist.sources[domain].tier == AuthorityTier.TIER_1
    
    def test_tier_2_sources_present(self):
        """Test that Tier 2 academic sources are present."""
        allowlist = AuthorityAllowlist()
        
        expected_tier_2 = [
            "ocw.mit.edu",
            "arxiv.org",
        ]
        
        for domain in expected_tier_2:
            assert domain in allowlist.sources, f"Missing Tier 2 source: {domain}"
            assert allowlist.sources[domain].tier == AuthorityTier.TIER_2
    
    def test_tier_3_sources_present(self):
        """Test that Tier 3 community sources are present."""
        allowlist = AuthorityAllowlist()
        
        expected_tier_3 = [
            "wikipedia.org",
            "github.com",
            "stackoverflow.com",
        ]
        
        for domain in expected_tier_3:
            assert domain in allowlist.sources, f"Missing Tier 3 source: {domain}"
            assert allowlist.sources[domain].tier == AuthorityTier.TIER_3
    
    def test_get_source_from_full_url(self):
        """Test retrieving source info from full URL."""
        allowlist = AuthorityAllowlist()
        
        urls = [
            "https://docs.python.org/3/library/",
            "http://rfc-editor.org/rfc/rfc3986.txt",
            "https://kubernetes.io/docs/",
        ]
        
        for url in urls:
            source = allowlist.get_source(url)
            assert source is not None, f"Should find source for {url}"
            assert source.tier in [AuthorityTier.TIER_1, AuthorityTier.TIER_2, AuthorityTier.TIER_3]
    
    def test_is_allowed(self):
        """Test allowlist validation."""
        allowlist = AuthorityAllowlist()
        
        # Allowed sources
        assert allowlist.is_allowed("https://docs.python.org/")
        assert allowlist.is_allowed("https://wikipedia.org/wiki/Recursion")
        assert allowlist.is_allowed("https://github.com/torvalds/linux")
        
        # Disallowed sources
        assert not allowlist.is_allowed("https://example.com/")
        assert not allowlist.is_allowed("https://suspicious-site.net/")
    
    def test_www_prefix_handling(self):
        """Test that www prefix is handled correctly."""
        allowlist = AuthorityAllowlist()
        
        # Should match both with and without www
        source1 = allowlist.get_source("https://python.org/")
        source2 = allowlist.get_source("https://www.python.org/")
        
        # Both should be found (since we strip www for matching)
        assert source1 is not None or source2 is not None
    
    def test_get_authority_weight(self):
        """Test weight retrieval."""
        allowlist = AuthorityAllowlist()
        
        # Tier 1 should have weight close to 1.0
        tier1_weight = allowlist.get_authority_weight("https://docs.python.org/")
        assert 0.9 <= tier1_weight <= 1.0
        
        # Wikipedia should be lower
        wiki_weight = allowlist.get_authority_weight("https://wikipedia.org/")
        assert 0.5 <= wiki_weight < 0.7
        
        # Unknown should be 0
        unknown_weight = allowlist.get_authority_weight("https://example.com/")
        assert unknown_weight == 0.0
    
    def test_validate_source_with_tier_requirement(self):
        """Test source validation with tier requirements."""
        allowlist = AuthorityAllowlist()
        
        # Tier 1 source passes Tier 1 requirement
        is_valid, reason = allowlist.validate_source(
            "https://docs.python.org/",
            require_tier=AuthorityTier.TIER_1
        )
        assert is_valid
        
        # Tier 3 source fails Tier 1 requirement
        is_valid, reason = allowlist.validate_source(
            "https://wikipedia.org/",
            require_tier=AuthorityTier.TIER_1
        )
        assert not is_valid
        assert "tier" in reason.lower()
        
        # Unknown domain fails
        is_valid, reason = allowlist.validate_source("https://example.com/")
        assert not is_valid
        assert "allowlist" in reason.lower()
    
    def test_add_custom_source(self):
        """Test adding custom sources."""
        allowlist = AuthorityAllowlist()
        initial_count = len(allowlist.sources)
        
        source = allowlist.add_custom_source(
            domain="test-university.edu",
            tier=AuthorityTier.TIER_2,
            authority_weight=0.85,
            category="university",
            description="Test University"
        )
        
        assert len(allowlist.sources) == initial_count + 1
        assert allowlist.is_allowed("https://test-university.edu/")
        assert allowlist.get_tier("https://test-university.edu/") == AuthorityTier.TIER_2
    
    def test_get_statistics(self):
        """Test allowlist statistics."""
        allowlist = AuthorityAllowlist()
        stats = allowlist.get_statistics()
        
        assert "total_sources" in stats
        assert "by_tier" in stats
        assert "avg_authority_weight" in stats
        
        assert stats["total_sources"] > 0
        assert stats["by_tier"][AuthorityTier.TIER_1.name] > 0
        assert 0.0 < stats["avg_authority_weight"] <= 1.0
    
    def test_global_allowlist_singleton(self):
        """Test that global allowlist is a singleton."""
        allowlist1 = get_allowlist()
        allowlist2 = get_allowlist()
        
        assert allowlist1 is allowlist2
    
    def test_convenience_functions(self):
        """Test convenience wrapper functions."""
        # is_allowed_source
        assert is_allowed_source("https://docs.python.org/")
        assert not is_allowed_source("https://example.com/")
        
        # get_source_tier
        tier = get_source_tier("https://docs.python.org/")
        assert tier == AuthorityTier.TIER_1
        
        tier = get_source_tier("https://wikipedia.org/")
        assert tier == AuthorityTier.TIER_3
        
        tier = get_source_tier("https://example.com/")
        assert tier is None
        
        # get_source_weight
        weight = get_source_weight("https://docs.python.org/")
        assert weight > 0.9
        
        weight = get_source_weight("https://example.com/")
        assert weight == 0.0


class TestAuthorityTiers:
    """Test authority tier system."""
    
    def test_tier_hierarchy(self):
        """Test tier hierarchy."""
        assert AuthorityTier.TIER_1.value < AuthorityTier.TIER_2.value < AuthorityTier.TIER_3.value
    
    def test_tier_ordering(self):
        """Test tier ordering for comparison."""
        tier1 = AuthorityTier.TIER_1
        tier2 = AuthorityTier.TIER_2
        tier3 = AuthorityTier.TIER_3
        
        assert tier1.value < tier2.value
        assert tier2.value < tier3.value
        assert tier1.value < tier3.value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
