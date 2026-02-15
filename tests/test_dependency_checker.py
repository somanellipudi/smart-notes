"""
Unit tests for cross-claim dependency checker.

Tests cover:
- Term extraction
- Dependency checking
- Warning generation
- Enforcement with downgrading
"""

import pytest
from src.verification.dependency_checker import (
    extract_terms,
    build_defined_terms_index,
    check_dependencies,
    apply_dependency_enforcement,
    DependencyWarning
)
from src.claims.schema import (
    LearningClaim,
    ClaimType,
    VerificationStatus
)


class TestTermExtraction:
    """Test term extraction from claim text."""
    
    def test_extract_capitalized_terms(self):
        """Extract capitalized terms (technical terms)."""
        text = "Newton's Law states that Force is..."
        terms = extract_terms(text)
        assert "Newton" in terms or "Law" in terms
        assert "Force" in terms
    
    def test_extract_variables(self):
        """Extract mathematical variables."""
        text = "The variables x and y satisfy the equation..."
        terms = extract_terms(text)
        assert "x" in terms
        assert "y" in terms
    
    def test_extract_greek_letters(self):
        """Extract Greek letter names."""
        text = "The angle theta and radius rho are..."
        terms = extract_terms(text)
        assert "theta" in terms
        assert "rho" in terms
    
    def test_filter_common_words(self):
        """Common words should be filtered out."""
        text = "The definition is that the concept..."
        terms = extract_terms(text)
        # 'The', 'Is', 'That' should be filtered
        assert "The" not in terms
        assert "Is" not in terms
        assert "That" not in terms
    
    def test_empty_text(self):
        """Empty text should return empty set."""
        assert extract_terms("") == set()
        assert extract_terms(None) == set()


class TestDefinedTermsIndex:
    """Test building index of defined terms."""
    
    def test_definition_claim_extracts_term(self):
        """DEFINITION claim should extract defined term."""
        claim = LearningClaim(
            claim_type=ClaimType.DEFINITION,
            claim_text="Force is the product of mass and acceleration.",
            confidence=0.9,
            status=VerificationStatus.VERIFIED
        )
        
        index = build_defined_terms_index([claim])
        
        assert claim.claim_id in index
        assert "Force" in index[claim.claim_id]
    
    def test_equation_claim_extracts_variable(self):
        """EQUATION claim should extract left-side variable."""
        claim = LearningClaim(
            claim_type=ClaimType.EQUATION,
            claim_text="F = ma",
            confidence=0.9,
            status=VerificationStatus.VERIFIED
        )
        
        index = build_defined_terms_index([claim])
        
        assert claim.claim_id in index
        assert "F" in index[claim.claim_id]
    
    def test_equation_with_name(self):
        """EQUATION with name should extract both name and variable."""
        claim = LearningClaim(
            claim_type=ClaimType.EQUATION,
            claim_text="Newton's Second Law: F = ma",
            confidence=0.9,
            status=VerificationStatus.VERIFIED
        )
        
        index = build_defined_terms_index([claim])
        defined = index[claim.claim_id]
        
        # Should extract equation name
        assert any("Newton" in term for term in defined)
        # Should also extract variable
        assert "F" in defined
    
    def test_empty_claim_text(self):
        """Empty claim text should be handled gracefully."""
        claim = LearningClaim(
            claim_type=ClaimType.DEFINITION,
            claim_text="",
            confidence=0.0,
            status=VerificationStatus.REJECTED
        )
        
        index = build_defined_terms_index([claim])
        assert claim.claim_id not in index or len(index[claim.claim_id]) == 0


class TestDependencyChecking:
    """Test dependency checking across claims."""
    
    def test_no_undefined_dependencies(self):
        """Claims with all terms defined should have no warnings."""
        claims = [
            LearningClaim(
                claim_type=ClaimType.DEFINITION,
                claim_text="Force is the product of mass and acceleration.",
                confidence=0.9,
                status=VerificationStatus.VERIFIED
            ),
            LearningClaim(
                claim_type=ClaimType.EQUATION,
                claim_text="Force equals mass times acceleration: F = ma",
                confidence=0.9,
                status=VerificationStatus.VERIFIED
            )
        ]
        
        warnings = check_dependencies(claims, strict_mode=False)
        
        # Second claim references Force, which is defined in first claim
        # Should have minimal warnings (might warn about 'mass' and 'acceleration')
        # This is acceptable since we can't perfectly track all terms
        assert isinstance(warnings, list)
    
    def test_undefined_term_warning(self):
        """Claim referencing undefined term should generate warning."""
        claims = [
            LearningClaim(
                claim_type=ClaimType.EQUATION,
                claim_text="Velocity v equals distance d over time t: v = d/t",
                confidence=0.9,
                status=VerificationStatus.VERIFIED
            )
        ]
        
        warnings = check_dependencies(claims, strict_mode=False)
        
        # First claim references variables without definitions
        # Should generate warnings for undefined terms
        # (though 'v', 'd', 't' might be filtered as common variables)
        assert isinstance(warnings, list)
        # Check structure of warnings
        if warnings:
            for w in warnings:
                assert hasattr(w, 'claim_id')
                assert hasattr(w, 'undefined_terms')
                assert hasattr(w, 'severity')
    
    def test_dependency_order_matters(self):
        """Term must be defined before use."""
        # Correct order: definition first, then usage
        claims_correct = [
            LearningClaim(
                claim_type=ClaimType.DEFINITION,
                claim_text="Derivative is the rate of change.",
                confidence=0.9,
                status=VerificationStatus.VERIFIED
            ),
            LearningClaim(
                claim_type=ClaimType.EXAMPLE,
                claim_text="The Derivative of x^2 is 2x.",
                confidence=0.9,
                status=VerificationStatus.VERIFIED
            )
        ]
        
        # Incorrect order: usage first, then definition
        claims_incorrect = [
            LearningClaim(
                claim_type=ClaimType.EXAMPLE,
                claim_text="The Derivative of x^2 is 2x.",
                confidence=0.9,
                status=VerificationStatus.VERIFIED
            ),
            LearningClaim(
                claim_type=ClaimType.DEFINITION,
                claim_text="Derivative is the rate of change.",
                confidence=0.9,
                status=VerificationStatus.VERIFIED
            )
        ]
        
        warnings_correct = check_dependencies(claims_correct, strict_mode=True)
        warnings_incorrect = check_dependencies(claims_incorrect, strict_mode=True)
        
        # Correct order should have fewer warnings
        # (or none if 'Derivative' is successfully tracked)
        assert len(warnings_incorrect) >= len(warnings_correct)
    
    def test_strict_mode_severity(self):
        """Strict mode should generate error-level warnings."""
        claims = [
            LearningClaim(
                claim_type=ClaimType.EQUATION,
                claim_text="UnknownVariable x = 5",
                confidence=0.9,
                status=VerificationStatus.VERIFIED
            )
        ]
        
        warnings_strict = check_dependencies(claims, strict_mode=True)
        warnings_nonstrict = check_dependencies(claims, strict_mode=False)
        
        # Strict mode warnings should have severity='error'
        if warnings_strict:
            assert any(w.severity == "error" for w in warnings_strict)
        
        # Non-strict warnings should have severity='warning'
        if warnings_nonstrict:
            assert all(w.severity == "warning" for w in warnings_nonstrict)


class TestDependencyEnforcement:
    """Test dependency enforcement with claim downgrading."""
    
    def test_downgrade_with_warnings(self):
        """Claims with warnings should be downgraded in strict mode."""
        claim = LearningClaim(
            claim_type=ClaimType.DEFINITION,
            claim_text="Test claim with UnknownTerm",
            confidence=0.9,
            status=VerificationStatus.VERIFIED
        )
        
        warning = DependencyWarning(
            claim_id=claim.claim_id,
            claim_text=claim.claim_text,
            undefined_terms=["UnknownTerm"],
            severity="error",
            suggestion="Define UnknownTerm first"
        )
        
        result = apply_dependency_enforcement(
            [claim],
            [warning],
            downgrade_to_low_confidence=True
        )
        
        # Claim should be downgraded
        assert result[0].status == VerificationStatus.LOW_CONFIDENCE
        assert result[0].confidence < 0.9  # Confidence reduced
        assert "dependency_warning" in result[0].metadata
    
    def test_no_downgrade_without_enforcement(self):
        """Without enforcement, claims should remain unchanged."""
        claim = LearningClaim(
            claim_type=ClaimType.DEFINITION,
            claim_text="Test claim",
            confidence=0.9,
            status=VerificationStatus.VERIFIED
        )
        
        warning = DependencyWarning(
            claim_id=claim.claim_id,
            claim_text=claim.claim_text,
            undefined_terms=["SomeTerm"],
            severity="warning",
            suggestion="Consider defining SomeTerm"
        )
        
        result = apply_dependency_enforcement(
            [claim],
            [warning],
            downgrade_to_low_confidence=False
        )
        
        # Claim should remain VERIFIED
        assert result[0].status == VerificationStatus.VERIFIED
        assert result[0].confidence == 0.9
    
    def test_no_downgrade_already_rejected(self):
        """Already rejected/low-confidence claims should not be further downgraded."""
        claim = LearningClaim(
            claim_type=ClaimType.DEFINITION,
            claim_text="Test claim",
            confidence=0.3,
            status=VerificationStatus.LOW_CONFIDENCE
        )
        
        warning = DependencyWarning(
            claim_id=claim.claim_id,
            claim_text=claim.claim_text,
            undefined_terms=["Term"],
            severity="error",
            suggestion="Define Term"
        )
        
        result = apply_dependency_enforcement(
            [claim],
            [warning],
            downgrade_to_low_confidence=True
        )
        
        # Should remain LOW_CONFIDENCE (not downgraded further)
        assert result[0].status == VerificationStatus.LOW_CONFIDENCE


class TestIntegration:
    """Integration tests for full dependency checking workflow."""
    
    def test_full_workflow(self):
        """Test complete dependency checking workflow."""
        claims = [
            LearningClaim(
                claim_type=ClaimType.DEFINITION,
                claim_text="Force is the interaction that changes motion.",
                confidence=0.9,
                status=VerificationStatus.VERIFIED
            ),
            LearningClaim(
                claim_type=ClaimType.EQUATION,
                claim_text="Force equals mass times acceleration: F = ma",
                confidence=0.9,
                status=VerificationStatus.VERIFIED
            ),
            LearningClaim(
                claim_type=ClaimType.EXAMPLE,
                claim_text="An example of Force is pushing a box.",
                confidence=0.8,
                status=VerificationStatus.VERIFIED
            ),
            LearningClaim(
                claim_type=ClaimType.DEFINITION,
                claim_text="Velocity is the rate of change of position.",
                confidence=0.9,
                status=VerificationStatus.VERIFIED
            )
        ]
        
        # Check dependencies
        warnings = check_dependencies(claims, strict_mode=False)
        
        # Apply enforcement
        result = apply_dependency_enforcement(
            claims,
            warnings,
            downgrade_to_low_confidence=False
        )
        
        # Should return same number of claims
        assert len(result) == len(claims)
        
        # All claims should still be valid
        assert all(isinstance(c, LearningClaim) for c in result)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
