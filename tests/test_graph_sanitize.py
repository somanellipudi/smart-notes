"""
Tests for GraphML sanitization utilities.

Tests ensure that complex Python objects (bytes, dicts, Pydantic models, etc.)
are properly converted to GraphML-compatible types.
"""

import pytest
import networkx as nx
from pydantic import BaseModel, Field
from typing import Dict, Any

from src.graph.graph_sanitize import (
    _sanitize_value,
    sanitize_graph_for_graphml,
    export_graphml_string,
    export_graphml_bytes,
)


class MockPydanticModel(BaseModel):
    """Mock Pydantic model for testing."""
    name: str = Field(..., description="Name field")
    value: int = Field(..., description="Value field")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TestSanitizeValue:
    """Test _sanitize_value function."""
    
    def test_none_returns_empty_string(self):
        """None should convert to empty string."""
        assert _sanitize_value(None) == ""
    
    def test_simple_types_unchanged(self):
        """Simple types (str, int, float, bool) should pass through."""
        assert _sanitize_value("hello") == "hello"
        assert _sanitize_value(42) == 42
        assert _sanitize_value(3.14) == 3.14
        assert _sanitize_value(True) is True
        assert _sanitize_value(False) is False
    
    def test_long_string_truncated(self):
        """Long strings should be truncated to max_str_length."""
        long_str = "x" * 600
        result = _sanitize_value(long_str, max_str_length=500)
        assert len(result) == 503  # 500 + "..."
        assert result.endswith("...")
    
    def test_bytes_decoded(self):
        """Bytes should decode to UTF-8."""
        data = b"hello world"
        result = _sanitize_value(data)
        assert result == "hello world"
        assert isinstance(result, str)
    
    def test_bytes_with_errors(self):
        """Bytes with decode errors should use replacement characters."""
        data = b"\xff\xfe invalid utf-8"
        result = _sanitize_value(data)
        assert isinstance(result, str)
        # Should not raise exception
    
    def test_dict_to_json(self):
        """Dicts should serialize to JSON strings."""
        data = {"key": "value", "number": 42}
        result = _sanitize_value(data)
        assert isinstance(result, str)
        assert "key" in result
        assert "value" in result
        assert "42" in result
    
    def test_list_to_json(self):
        """Lists should serialize to JSON strings."""
        data = [1, 2, "three", {"nested": True}]
        result = _sanitize_value(data)
        assert isinstance(result, str)
        assert "three" in result
        assert "nested" in result
    
    def test_pydantic_model_dumps(self):
        """Pydantic models should use model_dump() then JSON."""
        model = MockPydanticModel(name="test", value=100, metadata={"foo": "bar"})
        result = _sanitize_value(model)
        assert isinstance(result, str)
        assert "test" in result
        assert "100" in result
        assert "foo" in result
    
    def test_nested_structure_truncated(self):
        """Large nested structures should be truncated."""
        data = {"key": "x" * 600}
        result = _sanitize_value(data, max_str_length=100)
        assert len(result) == 103  # 100 + "..."
        assert result.endswith("...")


class TestSanitizeGraphForGraphML:
    """Test sanitize_graph_for_graphml function."""
    
    def test_empty_graph(self):
        """Empty graph should sanitize without errors."""
        G = nx.DiGraph()
        sanitized = sanitize_graph_for_graphml(G)
        assert sanitized.number_of_nodes() == 0
        assert sanitized.number_of_edges() == 0
    
    def test_node_attributes_sanitized(self):
        """Complex node attributes should be sanitized."""
        G = nx.DiGraph()
        G.add_node("claim1", 
                   status="VERIFIED",
                   confidence=0.85,
                   metadata={"key": "value"},
                   model=MockPydanticModel(name="test", value=42),
                   binary_data=b"hello",
                   none_value=None)
        
        sanitized = sanitize_graph_for_graphml(G)
        
        # Check node exists
        assert sanitized.has_node("claim1")
        attrs = sanitized.nodes["claim1"]
        
        # Check simple types unchanged
        assert attrs["status"] == "VERIFIED"
        assert attrs["confidence"] == 0.85
        
        # Check complex types converted
        assert isinstance(attrs["metadata"], str)
        assert "key" in attrs["metadata"]
        
        assert isinstance(attrs["model"], str)
        assert "test" in attrs["model"]
        
        assert isinstance(attrs["binary_data"], str)
        assert "hello" in attrs["binary_data"]
        
        assert attrs["none_value"] == ""
    
    def test_edge_attributes_sanitized(self):
        """Complex edge attributes should be sanitized."""
        G = nx.DiGraph()
        G.add_edge("claim1", "evidence1",
                   weight=0.75,
                   evidence={"snippet": "supporting text"},
                   scores=[0.8, 0.7, 0.9])
        
        sanitized = sanitize_graph_for_graphml(G)
        
        # Check edge exists
        assert sanitized.has_edge("claim1", "evidence1")
        attrs = sanitized.edges["claim1", "evidence1"]
        
        # Check simple type unchanged
        assert attrs["weight"] == 0.75
        
        # Check complex types converted
        assert isinstance(attrs["evidence"], str)
        assert "snippet" in attrs["evidence"]
        
        assert isinstance(attrs["scores"], str)
        assert "0.8" in attrs["scores"]
    
    def test_original_graph_unchanged(self):
        """Sanitization should not modify original graph."""
        G = nx.DiGraph()
        G.add_node("node1", data={"key": "value"})
        original_data = G.nodes["node1"]["data"]
        
        sanitized = sanitize_graph_for_graphml(G)
        
        # Original should still have dict
        assert isinstance(G.nodes["node1"]["data"], dict)
        assert G.nodes["node1"]["data"] is original_data
        
        # Sanitized should have string
        assert isinstance(sanitized.nodes["node1"]["data"], str)


class TestExportGraphML:
    """Test GraphML export functions."""
    
    def test_export_graphml_string_returns_string(self):
        """export_graphml_string should return string."""
        G = nx.DiGraph()
        G.add_node("node1", label="Test Node")
        G.add_edge("node1", "node2", weight=1.0)
        
        result = export_graphml_string(G)
        
        assert isinstance(result, str)
        assert "<?xml" in result  # GraphML XML header
        assert "graphml" in result.lower()
        assert "node1" in result
    
    def test_export_graphml_bytes_returns_bytes(self):
        """export_graphml_bytes should return UTF-8 bytes."""
        G = nx.DiGraph()
        G.add_node("node1", label="Test Node")
        
        result = export_graphml_bytes(G)
        
        assert isinstance(result, bytes)
        # Should be valid UTF-8
        decoded = result.decode('utf-8')
        assert "<?xml" in decoded
        assert "node1" in decoded
    
    def test_export_with_complex_graph(self):
        """Export should handle complex graph with many attributes."""
        G = nx.DiGraph()
        
        # Add multiple claims with various attributes
        for i in range(5):
            G.add_node(f"claim_{i}",
                       node_type="claim",
                       status="VERIFIED" if i % 2 == 0 else "REJECTED",
                       confidence=0.5 + (i * 0.1),
                       metadata={"index": i, "details": "some text"})
        
        # Add evidence nodes
        for i in range(3):
            G.add_node(f"evidence_{i}",
                       node_type="evidence",
                       snippet=f"Evidence text {i}",
                       source_type="transcript")
        
        # Add edges
        G.add_edge("claim_0", "evidence_0", weight=0.85, similarity=0.92)
        G.add_edge("claim_1", "evidence_0", weight=0.65, similarity=0.71)
        G.add_edge("claim_2", "evidence_1", weight=0.95, similarity=0.98)
        
        # Should export without errors
        graphml_bytes = export_graphml_bytes(G)
        assert len(graphml_bytes) > 0
        
        # Should be valid XML
        graphml_str = graphml_bytes.decode('utf-8')
        assert "<?xml" in graphml_str
        assert "claim_0" in graphml_str
        assert "evidence_0" in graphml_str


class TestGraphMetricsCompatibility:
    """Test that GraphMetrics objects work with sanitization."""
    
    def test_graph_metrics_dict_method(self):
        """GraphMetrics.to_dict() should work with sanitization."""
        from src.claims.schema import GraphMetrics
        
        metrics = GraphMetrics(
            avg_support_depth=1.5,
            avg_redundancy=2.3,
            avg_diversity=0.75,
            conflict_count=2,
            total_claims=25,
            total_evidence=50
        )
        
        # to_dict() should return plain dict
        metrics_dict = metrics.to_dict()
        assert isinstance(metrics_dict, dict)
        assert metrics_dict["total_claims"] == 25
        assert metrics_dict["total_evidence"] == 50
        
        # Should sanitize correctly
        sanitized = _sanitize_value(metrics)
        assert isinstance(sanitized, str)
        assert "total_claims" in sanitized
    
    def test_graph_metrics_get_method(self):
        """GraphMetrics.get() should work like dict.get()."""
        from src.claims.schema import GraphMetrics
        
        metrics = GraphMetrics(
            avg_support_depth=1.5,
            avg_redundancy=2.3,
            avg_diversity=0.75,
            conflict_count=2,
            total_claims=25,
            total_evidence=50
        )
        
        # .get() should work
        assert metrics.get("total_claims") == 25
        assert metrics.get("total_evidence") == 50
        assert metrics.get("nonexistent_field", "default") == "default"
        
        # Alias should work
        assert metrics.get("evidence_nodes") == 50  # Alias for total_evidence


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
