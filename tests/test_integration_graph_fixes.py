"""
Integration test to verify all GraphML and graph fixes.

Run this script to test:
1. GraphMetrics .get() compatibility
2. GraphML export with complex attributes
3. Graph sanitization
4. Export functions
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.claims.schema import GraphMetrics, LearningClaim, EvidenceItem, ClaimType, VerificationStatus
from src.graph.claim_graph import ClaimGraph
from src.graph.graph_sanitize import (
    sanitize_graph_for_graphml,
    export_graphml_bytes,
)
import networkx as nx


def test_graph_metrics_get():
    """Test that GraphMetrics.get() works like dict."""
    print("\n" + "=" * 60)
    print("TEST 1: GraphMetrics .get() compatibility")
    print("=" * 60)
    
    metrics = GraphMetrics(
        avg_support_depth=1.5,
        avg_redundancy=2.3,
        avg_diversity=0.75,
        conflict_count=2,
        total_claims=25,
        total_evidence=50
    )
    
    # Test .get() method
    print(f"metrics.get('total_claims'): {metrics.get('total_claims')}")
    print(f"metrics.get('total_evidence'): {metrics.get('total_evidence')}")
    print(f"metrics.get('nonexistent', 'default'): {metrics.get('nonexistent', 'default')}")
    
    # Test alias
    print(f"metrics.get('evidence_nodes'): {metrics.get('evidence_nodes')} (alias for total_evidence)")
    
    # Test to_dict()
    metrics_dict = metrics.to_dict()
    print(f"\nmetrics.to_dict() keys: {list(metrics_dict.keys())}")
    print(f"metrics_dict['total_claims']: {metrics_dict['total_claims']}")
    
    assert metrics.get('total_claims') == 25
    assert metrics.get('total_evidence') == 50
    assert metrics.get('evidence_nodes') == 50  # Alias
    assert metrics.get('nonexistent', 'default') == 'default'
    
    print("✅ GraphMetrics .get() compatibility: PASSED")


def test_graphml_export_with_complex_attributes():
    """Test GraphML export with bytes, dicts, Pydantic models."""
    print("\n" + "=" * 60)
    print("TEST 2: GraphML export with complex attributes")
    print("=" * 60)
    
    G = nx.DiGraph()
    
    # Add claim node with complex attributes
    G.add_node("claim1",
               node_type="claim",
               status="VERIFIED",
               confidence=0.85,
               claim_type="definition",
               # Complex attributes that need sanitization
               metadata={"key": "value", "nested": {"deep": "data"}},
               binary_data=b"some binary content",
               none_value=None,
               long_text="x" * 600)  # Long string
    
    # Add evidence node
    G.add_node("evidence1",
               node_type="evidence",
               snippet="Supporting evidence text",
               source_type="transcript",
               scores=[0.8, 0.9, 0.7])  # List attribute
    
    # Add edge with attributes
    G.add_edge("claim1", "evidence1",
               weight=0.85,
               similarity=0.92,
               entailment={"label": "ENTAILMENT", "prob": 0.95})
    
    # Sanitize graph
    sanitized = sanitize_graph_for_graphml(G)
    print(f"Original nodes: {G.number_of_nodes()}")
    print(f"Sanitized nodes: {sanitized.number_of_nodes()}")
    
    # Check sanitization
    claim_attrs = sanitized.nodes["claim1"]
    print(f"\nSanitized claim1 attributes:")
    for key, value in claim_attrs.items():
        print(f"  {key}: {type(value).__name__} = {str(value)[:50]}...")
    
    # Test export to GraphML bytes
    try:
        graphml_bytes = export_graphml_bytes(G)
        print(f"\n✅ GraphML export successful: {len(graphml_bytes)} bytes")
        
        # Verify it's valid UTF-8
        graphml_str = graphml_bytes.decode('utf-8')
        assert "<?xml" in graphml_str
        assert "claim1" in graphml_str
        print("✅ GraphML is valid XML")
        
    except Exception as e:
        print(f"❌ GraphML export failed: {e}")
        raise
    
    print("✅ GraphML export with complex attributes: PASSED")


def test_claim_graph_integration():
    """Test ClaimGraph with real claim objects."""
    print("\n" + "=" * 60)
    print("TEST 3: ClaimGraph integration")
    print("=" * 60)
    
    # Create mock claims
    claims = []
    for i in range(3):
        evidence_list = [
            EvidenceItem(
                evidence_id=f"ev_{i}_0",
                source_id=f"source_{i}",
                snippet=f"Evidence text for claim {i}",
                source_type="transcript",
                similarity=0.8 + (i * 0.05),
                reliability_prior=0.9
            )
        ]
        
        claim = LearningClaim(
            claim_id=f"claim_{i}",
            claim_text=f"This is claim {i}",
            claim_type=ClaimType.DEFINITION,
            status=VerificationStatus.VERIFIED if i % 2 == 0 else VerificationStatus.REJECTED,
            confidence=0.7 + (i * 0.1),
            evidence_ids=[f"ev_{i}_0"],
            evidence_objects=evidence_list
        )
        claims.append(claim)
    
    # Build graph
    graph = ClaimGraph(claims)
    
    print(f"Graph nodes: {graph.graph.number_of_nodes()}")
    print(f"Graph edges: {graph.graph.number_of_edges()}")
    
    # Test export methods
    try:
        graphml_bytes = graph.get_graphml_bytes()
        print(f"✅ ClaimGraph.get_graphml_bytes(): {len(graphml_bytes)} bytes")
    except Exception as e:
        print(f"❌ get_graphml_bytes() failed: {e}")
        raise
    
    try:
        adjacency_json = graph.export_adjacency_json()
        print(f"✅ ClaimGraph.export_adjacency_json(): {len(adjacency_json)} chars")
    except Exception as e:
        print(f"❌ export_adjacency_json() failed: {e}")
        raise
    
    # Test metrics
    try:
        metrics = graph.compute_metrics()
        print(f"\n✅ Graph metrics computed:")
        print(f"  Total claims: {metrics.total_claims}")
        print(f"  Total evidence: {metrics.total_evidence}")
        print(f"  Avg redundancy: {metrics.avg_redundancy}")
        
        # Test metrics.get()
        print(f"\n✅ Testing metrics.get():")
        print(f"  metrics.get('total_claims'): {metrics.get('total_claims')}")
        print(f"  metrics.get('evidence_nodes'): {metrics.get('evidence_nodes')}")
    except Exception as e:
        print(f"❌ Metrics computation failed: {e}")
        raise
    
    print("✅ ClaimGraph integration: PASSED")


def test_pydantic_model_in_graph():
    """Test that Pydantic models in graph attributes are sanitized."""
    print("\n" + "=" * 60)
    print("TEST 4: Pydantic models in graph attributes")
    print("=" * 60)
    
    metrics = GraphMetrics(
        avg_support_depth=1.5,
        avg_redundancy=2.3,
        avg_diversity=0.75,
        conflict_count=2,
        total_claims=25,
        total_evidence=50
    )
    
    G = nx.DiGraph()
    G.add_node("node1", metrics=metrics, status="VERIFIED")
    
    # Sanitize
    try:
        sanitized = sanitize_graph_for_graphml(G)
        attrs = sanitized.nodes["node1"]
        
        print(f"Original metrics type: {type(metrics)}")
        print(f"Sanitized metrics type: {type(attrs['metrics'])}")
        print(f"Sanitized metrics value: {attrs['metrics'][:100]}...")
        
        assert isinstance(attrs['metrics'], str)
        assert "total_claims" in attrs['metrics']
        
        # Test GraphML export
        graphml_bytes = export_graphml_bytes(G)
        print(f"\n✅ GraphML export with Pydantic model: {len(graphml_bytes)} bytes")
        
    except Exception as e:
        print(f"❌ Pydantic model sanitization failed: {e}")
        raise
    
    print("✅ Pydantic models in graph attributes: PASSED")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("SMART NOTES: GRAPH & GRAPHML FIX VERIFICATION")
    print("=" * 80)
    
    try:
        test_graph_metrics_get()
        test_graphml_export_with_complex_attributes()
        test_claim_graph_integration()
        test_pydantic_model_in_graph()
        
        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED")
        print("=" * 80)
        print("\nFixes verified:")
        print("  1. ✅ GraphMetrics .get() works correctly")
        print("  2. ✅ GraphML export handles complex attributes")
        print("  3. ✅ Graph sanitization converts bytes/dicts/Pydantic models")
        print("  4. ✅ ClaimGraph exports work reliably")
        print("\nReady for production use!")
        
        return 0
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ TEST FAILED")
        print("=" * 80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
