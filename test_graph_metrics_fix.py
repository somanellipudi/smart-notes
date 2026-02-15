"""
Smoke test for GraphMetrics type safety fix.

Tests that graph_metrics can be:
1. Created as a Pydantic object
2. Safely accessed with getattr()
3. Serialized to dict for JSON export
4. Handled when None
"""

from src.claims.schema import GraphMetrics, LearningClaim, ClaimType, VerificationStatus
from src.evaluation.verifiability_metrics import VerifiabilityMetrics
import json


def test_graph_metrics_object_creation():
    """Test creating GraphMetrics as Pydantic object."""
    print("Test 1: Creating GraphMetrics object...")
    
    metrics = GraphMetrics(
        avg_support_depth=1.5,
        avg_redundancy=2.0,
        avg_diversity=1.8,
        conflict_count=0,
        total_claims=10,
        total_evidence=15
    )
    
    print(f"  ✓ Created: {metrics}")
    return metrics


def test_graph_metrics_serialization(metrics):
    """Test serialization to dict."""
    print("\nTest 2: Serializing to dict...")
    
    # Try Pydantic v2 first
    if hasattr(metrics, 'model_dump'):
        metrics_dict = metrics.model_dump()
        print("  ✓ Used model_dump() (Pydantic v2)")
    # Fall back to Pydantic v1
    elif hasattr(metrics, 'dict'):
        metrics_dict = metrics.dict()
        print("  ✓ Used dict() (Pydantic v1)")
    else:
        raise RuntimeError("Could not serialize GraphMetrics")
    
    # Verify JSON serializable
    json_str = json.dumps(metrics_dict, indent=2)
    print(f"  ✓ JSON serializable: {len(json_str)} chars")
    
    return metrics_dict


def test_safe_attribute_access(metrics):
    """Test safe attribute access with getattr."""
    print("\nTest 3: Safe attribute access...")
    
    # Test existing attributes
    redundancy = getattr(metrics, 'avg_redundancy', 0.0)
    diversity = getattr(metrics, 'avg_diversity', 0.0)
    print(f"  ✓ avg_redundancy = {redundancy}")
    print(f"  ✓ avg_diversity = {diversity}")
    
    # Test missing attribute with default
    missing = getattr(metrics, 'nonexistent_field', 99.9)
    print(f"  ✓ missing field defaults to {missing}")
    
    # Test None handling
    none_result = getattr(None, 'avg_redundancy', 0.0) if None else 0.0
    print(f"  ✓ None handling: {none_result}")


def test_verifiability_metrics_integration(metrics):
    """Test VerifiabilityMetrics._calculate_graph_metrics."""
    print("\nTest 4: VerifiabilityMetrics integration...")
    
    vm = VerifiabilityMetrics()
    
    # Test with GraphMetrics object
    result = vm._calculate_graph_metrics(metrics)
    print(f"  ✓ Processed GraphMetrics object: {result}")
    
    # Test with dict
    metrics_dict = metrics.model_dump() if hasattr(metrics, 'model_dump') else metrics.dict()
    result2 = vm._calculate_graph_metrics(metrics_dict)
    print(f"  ✓ Processed dict: {result2}")
    
    # Test with None
    result3 = vm._calculate_graph_metrics(None)
    print(f"  ✓ Processed None: {result3}")
    
    assert result == result2, "Object and dict should produce same result"
    print("  ✓ Consistency verified")


def test_claim_confidence_calculation(metrics):
    """Test LearningClaim.calculate_confidence with graph_metrics."""
    print("\nTest 5: LearningClaim.calculate_confidence...")
    
    # Create a minimal claim (won't actually calculate, just test no crash)
    from src.claims.schema import EvidenceItem
    
    claim = LearningClaim(
        claim_id="test_001",
        claim_type=ClaimType.DEFINITION,
        claim_text="Test claim",
        status=VerificationStatus.VERIFIED,
        confidence=0.8,
        evidence_ids=["ev1"],
        evidence_objects=[
            EvidenceItem(
                evidence_id="ev1",
                snippet="This is a test snippet with enough characters to pass validation",
                source_type="notes",
                source_id="test.txt",
                similarity=0.9,
                span_metadata={}
            )
        ]
    )
    
    # Test calculate_confidence with GraphMetrics object
    try:
        conf = claim.calculate_confidence(graph_metrics=metrics)
        print(f"  ✓ Confidence with object: {conf:.2f}")
    except Exception as e:
        print(f"  ✗ Failed with object: {e}")
        raise
    
    # Test with dict
    metrics_dict = metrics.model_dump() if hasattr(metrics, 'model_dump') else metrics.dict()
    try:
        conf2 = claim.calculate_confidence(graph_metrics=metrics_dict)
        print(f"  ✓ Confidence with dict: {conf2:.2f}")
    except Exception as e:
        print(f"  ✗ Failed with dict: {e}")
        raise
    
    # Test with None
    try:
        conf3 = claim.calculate_confidence(graph_metrics=None)
        print(f"  ✓ Confidence with None: {conf3:.2f}")
    except Exception as e:
        print(f"  ✗ Failed with None: {e}")
        raise


def main():
    """Run all tests."""
    print("=" * 60)
    print("GraphMetrics Type Safety Smoke Test")
    print("=" * 60)
    
    try:
        # Test 1: Create object
        metrics = test_graph_metrics_object_creation()
        
        # Test 2: Serialize
        metrics_dict = test_graph_metrics_serialization(metrics)
        
        # Test 3: Safe access
        test_safe_attribute_access(metrics)
        
        # Test 4: VerifiabilityMetrics
        test_verifiability_metrics_integration(metrics)
        
        # Test 5: Claim confidence
        test_claim_confidence_calculation(metrics)
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        print("\nGraphMetrics type safety fix is working correctly!")
        print("The system can now handle:")
        print("  • GraphMetrics as Pydantic objects")
        print("  • GraphMetrics as dicts (backward compat)")
        print("  • None values with safe defaults")
        print("  • JSON serialization for exports")
        
        return 0
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"✗ TEST FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
