"""
Unit tests for domain profiles and configuration.

Tests cover:
- Domain profile structure
- Domain-specific validation rules
- Profile selection and defaults
"""

import pytest
import config
from config import DomainProfile, get_domain_profile


class TestDomainProfiles:
    """Test domain profile definitions."""
    
    def test_physics_profile_exists(self):
        """Physics profile should be defined."""
        profile = get_domain_profile("physics")
        assert profile.name == "physics"
        assert profile.display_name == "Physics"
        assert profile.require_units is True
        assert profile.require_equations is True
        assert "equation" in profile.allowed_claim_types
    
    def test_discrete_math_profile_exists(self):
        """Discrete math profile should be defined."""
        profile = get_domain_profile("discrete_math")
        assert profile.name == "discrete_math"
        assert profile.display_name == "Discrete Mathematics"
        assert profile.require_proof_steps is True
        assert profile.strict_dependencies is True
    
    def test_algorithms_profile_exists(self):
        """Algorithms profile should be defined."""
        profile = get_domain_profile("algorithms")
        assert profile.name == "algorithms"
        assert profile.display_name == "Algorithms & Data Structures"
        assert profile.require_pseudocode is True
        assert "equation" in profile.allowed_claim_types
    
    def test_default_profile(self):
        """Default profile should be physics."""
        profile = get_domain_profile(None)
        assert profile.name == config.DEFAULT_DOMAIN_PROFILE
    
    def test_invalid_domain_raises_error(self):
        """Invalid domain name should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid domain"):
            get_domain_profile("invalid_domain_name")
    
    def test_all_profiles_have_required_fields(self):
        """All domain profiles should have required fields."""
        for domain_name in ["physics", "discrete_math", "algorithms"]:
            profile = get_domain_profile(domain_name)
            assert profile.name
            assert profile.display_name
            assert profile.description
            assert isinstance(profile.allowed_claim_types, list)
            assert isinstance(profile.evidence_type_expectations, dict)
            assert isinstance(profile.require_units, bool)
            assert isinstance(profile.require_proof_steps, bool)
            assert isinstance(profile.require_pseudocode, bool)
            assert isinstance(profile.require_equations, bool)
            assert isinstance(profile.strict_dependencies, bool)
    
    def test_evidence_expectations_valid(self):
        """Evidence type expectations should be valid for all claim types."""
        for domain_name in ["physics", "discrete_math", "algorithms"]:
            profile = get_domain_profile(domain_name)
            
            # Check that all allowed claim types have evidence expectations
            for claim_type in profile.allowed_claim_types:
                if claim_type in profile.evidence_type_expectations:
                    expectations = profile.evidence_type_expectations[claim_type]
                    assert isinstance(expectations, list)
                    assert len(expectations) > 0


class TestDomainSpecificRules:
    """Test domain-specific validation rules."""
    
    def test_physics_requires_units(self):
        """Physics profile should require unit checking."""
        physics = get_domain_profile("physics")
        discrete_math = get_domain_profile("discrete_math")
        
        assert physics.require_units is True
        assert discrete_math.require_units is False
    
    def test_discrete_math_strict_dependencies(self):
        """Discrete math should enforce strict dependencies."""
        discrete_math = get_domain_profile("discrete_math")
        physics = get_domain_profile("physics")
        
        assert discrete_math.strict_dependencies is True
        assert physics.strict_dependencies is False
    
    def test_algorithms_requires_pseudocode(self):
        """Algorithms profile should require pseudocode checks."""
        algorithms = get_domain_profile("algorithms")
        physics = get_domain_profile("physics")
        
        assert algorithms.require_pseudocode is True
        assert physics.require_pseudocode is False


class TestConfigIntegration:
    """Test config integration with policies."""
    
    def test_granularity_config(self):
        """Granularity policy config should be present."""
        assert hasattr(config, 'MAX_PROPOSITIONS_PER_CLAIM')
        assert config.MAX_PROPOSITIONS_PER_CLAIM == 1
    
    def test_evidence_sufficiency_config(self):
        """Evidence sufficiency config should be present."""
        assert hasattr(config, 'MIN_ENTAILMENT_PROB')
        assert hasattr(config, 'MIN_SUPPORTING_SOURCES')
        assert hasattr(config, 'MAX_CONTRADICTION_PROB')
        
        assert 0.0 <= config.MIN_ENTAILMENT_PROB <= 1.0
        assert config.MIN_SUPPORTING_SOURCES >= 1
        assert 0.0 <= config.MAX_CONTRADICTION_PROB <= 1.0
    
    def test_dependency_checking_config(self):
        """Dependency checking config should be present."""
        assert hasattr(config, 'ENABLE_DEPENDENCY_WARNINGS')
        assert hasattr(config, 'STRICT_DEPENDENCY_ENFORCEMENT')
        
        assert isinstance(config.ENABLE_DEPENDENCY_WARNINGS, bool)
        assert isinstance(config.STRICT_DEPENDENCY_ENFORCEMENT, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
