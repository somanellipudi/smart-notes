"""
Integration test for backward compatibility.

Verifies that:
1. Standard mode (non-verifiable) still works
2. Existing verifiable mode still works
3. New features are opt-in and don't break existing functionality
"""

import pytest
from unittest.mock import Mock, patch
from src.reasoning.verifiable_pipeline import VerifiablePipelineWrapper
from src.schema.output_schema import ClassSessionOutput


class TestBackwardCompatibility:
    """Test that existing functionality still works."""
    
    def test_pipeline_initialization_without_domain(self):
        """Pipeline should initialize without domain profile (backward compat)."""
        # Should work with or without domain_profile parameter
        pipeline1 = VerifiablePipelineWrapper(
            provider_type="ollama",
            model="mistral"
        )
        assert pipeline1 is not None
        assert pipeline1.domain_profile is not None  # Should use default
        
        pipeline2 = VerifiablePipelineWrapper(
            provider_type="ollama",
            model="mistral",
            domain_profile="physics"
        )
        assert pipeline2 is not None
        assert pipeline2.domain_profile.name == "physics"
    
    def test_default_domain_profile(self):
        """Default domain profile should be applied when none specified."""
        pipeline = VerifiablePipelineWrapper(
            provider_type="ollama",
            model="mistral",
            domain_profile=None
        )
        
        # Should have default profile
        assert pipeline.domain_profile is not None
        assert pipeline.domain_profile.name in ["physics", "discrete_math", "algorithms"]
    
    @patch('src.reasoning.verifiable_pipeline.ReasoningPipeline')
    @patch('src.reasoning.verifiable_pipeline.ClaimExtractor')
    @patch('src.reasoning.verifiable_pipeline.ClaimRAG')
    def test_standard_mode_unchanged(self, mock_rag, mock_extractor, mock_pipeline):
        """Standard mode (verifiable_mode=False) should work unchanged."""
        # Mock the standard pipeline
        mock_standard = Mock()
        mock_output = ClassSessionOutput(
            session_id="test",
            class_summary="Test summary",
            topics=[],
            key_concepts=[]
        )
        mock_standard.process.return_value = mock_output
        mock_pipeline.return_value = mock_standard
        
        pipeline = VerifiablePipelineWrapper(
            provider_type="ollama",
            model="mistral"
        )
        pipeline.standard_pipeline = mock_standard
        
        # Process in standard mode
        output, metadata = pipeline.process(
            combined_content="Test content",
            equations=[],
            external_context="",
            session_id="test",
            verifiable_mode=False
        )
        
        # Should return output without verifiable metadata
        assert output is not None
        assert metadata is None
        mock_standard.process.assert_called_once()


class TestNewFeaturesOptIn:
    """Test that new features are opt-in and don't affect existing code."""
    
    def test_granularity_policy_optional(self):
        """Granularity enforcement should only apply in verifiable mode."""
        from src.policies.granularity_policy import enforce_granularity
        from src.claims.schema import LearningClaim, ClaimType
        
        # Create a compound claim
        claim = LearningClaim(
            claim_type=ClaimType.DEFINITION,
            claim_text="A is X. B is Y.",
            confidence=0.8
        )
        
        # Enforce granularity
        result = enforce_granularity([claim], max_propositions=1)
        
        # Should split into atomic claims
        assert len(result) >= 1
        # Original claim should be preserved (or split)
        assert all(isinstance(c, LearningClaim) for c in result)
    
    def test_evidence_policy_standalone(self):
        """Evidence policy should work standalone without breaking existing code."""
        from src.policies.evidence_policy import evaluate_evidence_sufficiency
        from src.claims.schema import LearningClaim, EvidenceItem, ClaimType
        
        claim = LearningClaim(
            claim_type=ClaimType.DEFINITION,
            claim_text="Test claim",
            confidence=0.0
        )
        
        evidence = [
            EvidenceItem(
                source_id="test",
                source_type="transcript",
                snippet="Evidence text",
                similarity=0.8
            )
        ]
        
        # Should evaluate without breaking
        decision = evaluate_evidence_sufficiency(claim, evidence)
        assert decision is not None
        assert hasattr(decision, 'status_override')
        assert hasattr(decision, 'confidence_score')
    
    def test_dependency_checker_optional(self):
        """Dependency checker should be optional and not break existing code."""
        from src.verification.dependency_checker import check_dependencies
        from src.claims.schema import LearningClaim, ClaimType
        
        claims = [
            LearningClaim(
                claim_type=ClaimType.DEFINITION,
                claim_text="Test claim",
                confidence=0.8
            )
        ]
        
        # Should work with empty or single claim list
        warnings = check_dependencies(claims, strict_mode=False)
        assert isinstance(warnings, list)
    
    def test_threat_model_read_only(self):
        """Threat model should be read-only and not affect runtime."""
        from src.policies.threat_model import (
            get_in_scope_threats,
            get_out_of_scope_threats,
            get_threat_model_summary
        )
        
        # Should retrieve threat information without side effects
        in_scope = get_in_scope_threats()
        out_of_scope = get_out_of_scope_threats()
        summary = get_threat_model_summary()
        
        assert isinstance(in_scope, list)
        assert isinstance(out_of_scope, list)
        assert isinstance(summary, dict)


class TestConfigDefaults:
    """Test that config defaults are sensible and backward compatible."""
    
    def test_config_has_defaults(self):
        """Config should have sensible defaults for all new settings."""
        import config
        
        # Domain profile defaults
        assert hasattr(config, 'DEFAULT_DOMAIN_PROFILE')
        assert config.DEFAULT_DOMAIN_PROFILE in ["physics", "discrete_math", "algorithms"]
        
        # Granularity defaults
        assert hasattr(config, 'MAX_PROPOSITIONS_PER_CLAIM')
        assert config.MAX_PROPOSITIONS_PER_CLAIM >= 1
        
        # Evidence sufficiency defaults
        assert hasattr(config, 'MIN_ENTAILMENT_PROB')
        assert 0.0 <= config.MIN_ENTAILMENT_PROB <= 1.0
        
        assert hasattr(config, 'MIN_SUPPORTING_SOURCES')
        assert config.MIN_SUPPORTING_SOURCES >= 1
        
        # Dependency checking defaults
        assert hasattr(config, 'ENABLE_DEPENDENCY_WARNINGS')
        assert hasattr(config, 'STRICT_DEPENDENCY_ENFORCEMENT')
    
    def test_existing_config_unchanged(self):
        """Existing config values should remain unchanged."""
        import config
        
        # Verify critical existing config still present
        assert hasattr(config, 'VERIFIABLE_VERIFIED_THRESHOLD')
        assert hasattr(config, 'VERIFIABLE_REJECTED_THRESHOLD')
        assert hasattr(config, 'VERIFIABLE_MIN_EVIDENCE')
        assert hasattr(config, 'ENABLE_VERIFIABLE_MODE')


class TestAPICompatibility:
    """Test that public API remains compatible."""
    
    def test_learning_claim_fields_unchanged(self):
        """LearningClaim schema should maintain existing fields."""
        from src.claims.schema import LearningClaim, ClaimType
        
        claim = LearningClaim(
            claim_type=ClaimType.DEFINITION,
            claim_text="Test",
            confidence=0.8
        )
        
        # Existing fields should still be present
        assert hasattr(claim, 'claim_id')
        assert hasattr(claim, 'claim_type')
        assert hasattr(claim, 'claim_text')
        assert hasattr(claim, 'evidence_ids')
        assert hasattr(claim, 'evidence_objects')
        assert hasattr(claim, 'confidence')
        assert hasattr(claim, 'status')
        assert hasattr(claim, 'metadata')
    
    def test_verification_status_unchanged(self):
        """VerificationStatus enum should maintain existing values."""
        from src.claims.schema import VerificationStatus
        
        # Existing statuses should still be present
        assert hasattr(VerificationStatus, 'VERIFIED')
        assert hasattr(VerificationStatus, 'LOW_CONFIDENCE')
        assert hasattr(VerificationStatus, 'REJECTED')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
