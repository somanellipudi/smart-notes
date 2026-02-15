"""
Unit tests for threat model documentation.

Tests cover:
- Threat model structure
- In-scope and out-of-scope threat definitions
- Threat model formatting and export
"""

import pytest
from src.policies.threat_model import (
    THREAT_MODEL,
    ThreatCategory,
    get_in_scope_threats,
    get_out_of_scope_threats,
    format_threat_model,
    get_threat_model_summary
)


class TestThreatModelStructure:
    """Test threat model structure and completeness."""
    
    def test_threat_model_not_empty(self):
        """Threat model should contain threats."""
        assert len(THREAT_MODEL) > 0
    
    def test_all_threats_have_required_fields(self):
        """All threats should have required fields."""
        for threat_name, threat in THREAT_MODEL.items():
            assert threat.name
            assert threat.description
            assert threat.category in [ThreatCategory.IN_SCOPE, ThreatCategory.OUT_OF_SCOPE]
            
            if threat.category == ThreatCategory.IN_SCOPE:
                assert threat.mitigation, f"In-scope threat {threat_name} missing mitigation"
            
            if threat.category == ThreatCategory.OUT_OF_SCOPE:
                assert threat.rationale, f"Out-of-scope threat {threat_name} missing rationale"
    
    def test_in_scope_threats_exist(self):
        """Should have in-scope threats defined."""
        in_scope = get_in_scope_threats()
        assert len(in_scope) > 0
    
    def test_out_of_scope_threats_exist(self):
        """Should have out-of-scope threats defined."""
        out_of_scope = get_out_of_scope_threats()
        assert len(out_of_scope) > 0
    
    def test_specific_in_scope_threats(self):
        """Verify specific in-scope threats are present."""
        in_scope_names = [t.name for t in get_in_scope_threats()]
        
        # These are the key threats from requirements
        expected = [
            "Unsupported Claims (Hallucinations)",
            "Scope Creep and Overgeneralization",
            "Misinterpreted Equations and Units",
            "Contradiction Across Sources",
            "Circular or Missing Dependencies"
        ]
        
        for threat_name in expected:
            assert any(threat_name in name for name in in_scope_names), \
                f"Expected in-scope threat not found: {threat_name}"
    
    def test_specific_out_of_scope_threats(self):
        """Verify specific out-of-scope threats are present."""
        out_of_scope_names = [t.name for t in get_out_of_scope_threats()]
        
        expected = [
            "OCR Noise",
            "Transcription Errors",
            "Domain Ambiguity"
        ]
        
        for phrase in expected:
            assert any(phrase in name for name in out_of_scope_names), \
                f"Expected out-of-scope threat not found: {phrase}"


class TestThreatModelFormatting:
    """Test threat model formatting functions."""
    
    def test_format_markdown(self):
        """Format as Markdown should produce valid output."""
        output = format_threat_model(markdown=True)
        
        assert "# Threat Model" in output
        assert "## In-Scope Threats" in output
        assert "## Out-of-Scope Threats" in output
        assert "**Description:**" in output
        assert "**Mitigation:**" in output or "**Rationale:**" in output
    
    def test_format_plain_text(self):
        """Format as plain text should produce valid output."""
        output = format_threat_model(markdown=False)
        
        assert "THREAT MODEL" in output
        assert "IN-SCOPE THREATS" in output
        assert "OUT-OF-SCOPE THREATS" in output
        assert "Description:" in output
        assert "Mitigation:" in output or "Rationale:" in output
    
    def test_summary_export(self):
        """Threat model summary should be exportable as dict."""
        summary = get_threat_model_summary()
        
        assert isinstance(summary, dict)
        assert "in_scope_count" in summary
        assert "out_of_scope_count" in summary
        assert "in_scope_threats" in summary
        assert "out_of_scope_threats" in summary
        assert "version" in summary
        
        assert summary["in_scope_count"] == len(get_in_scope_threats())
        assert summary["out_of_scope_count"] == len(get_out_of_scope_threats())
        
        assert isinstance(summary["in_scope_threats"], list)
        assert isinstance(summary["out_of_scope_threats"], list)


class TestThreatMitigations:
    """Test that in-scope threats have proper mitigations."""
    
    def test_hallucination_mitigation(self):
        """Unsupported claims should have evidence-first mitigation."""
        threat = THREAT_MODEL["unsupported_claims"]
        assert threat.category == ThreatCategory.IN_SCOPE
        assert "evidence-first" in threat.mitigation.lower()
        assert "sufficiency" in threat.mitigation.lower()
    
    def test_equation_mitigation(self):
        """Equation misinterpretation should mention domain profiles."""
        threat = THREAT_MODEL["misinterpreted_equations_units"]
        assert threat.category == ThreatCategory.IN_SCOPE
        assert "domain" in threat.mitigation.lower() or "profile" in threat.mitigation.lower()
    
    def test_contradiction_mitigation(self):
        """Contradictions should mention conflict detection."""
        threat = THREAT_MODEL["contradiction_across_sources"]
        assert threat.category == ThreatCategory.IN_SCOPE
        assert "conflict" in threat.mitigation.lower() or "contradiction" in threat.mitigation.lower()


class TestThreatRationales:
    """Test that out-of-scope threats have proper rationales."""
    
    def test_ocr_rationale(self):
        """OCR noise should be explained as preprocessing concern."""
        threat = THREAT_MODEL["ocr_noise"]
        assert threat.category == ThreatCategory.OUT_OF_SCOPE
        assert "preprocessing" in threat.rationale.lower() or "ocr" in threat.rationale.lower()
    
    def test_transcription_rationale(self):
        """Transcription errors should be explained as preprocessing concern."""
        threat = THREAT_MODEL["transcription_errors"]
        assert threat.category == ThreatCategory.OUT_OF_SCOPE
        assert "preprocessing" in threat.rationale.lower() or "transcription" in threat.rationale.lower()
    
    def test_domain_ambiguity_rationale(self):
        """Domain ambiguity should mention user responsibility."""
        threat = THREAT_MODEL["domain_ambiguity"]
        assert threat.category == ThreatCategory.OUT_OF_SCOPE
        assert "user" in threat.rationale.lower() or "domain" in threat.rationale.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
