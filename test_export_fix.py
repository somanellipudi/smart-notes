"""
Test export functionality for GraphMetrics and ClaimCollection.
"""

import json
from src.claims.schema import ClaimCollection, LearningClaim, ClaimType, VerificationStatus, EvidenceItem

def test_claim_collection_export():
    """Test that ClaimCollection can be exported to JSON."""
    print("Testing ClaimCollection export...")
    
    # Create a collection with a claim
    collection = ClaimCollection(session_id="test_session")
    
    claim = LearningClaim(
        claim_id="test_001",
        claim_type=ClaimType.DEFINITION,
        claim_text="Test claim for export",
        status=VerificationStatus.VERIFIED,
        confidence=0.85,
        evidence_ids=["ev1"],
        evidence_objects=[
            EvidenceItem(
                evidence_id="ev1",
                snippet="This is evidence for the test claim with sufficient length",
                source_type="notes",
                source_id="test.txt",
                similarity=0.9,
                span_metadata={}
            )
        ]
    )
    
    collection.add_claim(claim)
    
    # Export to dict
    collection_dict = collection.to_dict()
    print(f"  ✓ Collection exported to dict: {len(collection_dict)} keys")
    
    # Verify JSON serializable
    json_str = json.dumps(collection_dict, indent=2)
    print(f"  ✓ JSON serializable: {len(json_str)} chars")
    
    # Parse back
    parsed = json.loads(json_str)
    print(f"  ✓ JSON parsed successfully")
    print(f"  ✓ Session ID: {parsed['session_id']}")
    print(f"  ✓ Claims count: {len(parsed['claims'])}")
    
    return True


def test_claim_export():
    """Test that individual claims can be exported."""
    print("\nTesting LearningClaim export...")
    
    claim = LearningClaim(
        claim_id="test_002",
        claim_type=ClaimType.EQUATION,
        claim_text="f'(x) = 2x for f(x) = x²",
        status=VerificationStatus.VERIFIED,
        confidence=0.92,
        evidence_ids=["ev1", "ev2"],
        evidence_objects=[]
    )
    
    # Export to dict
    claim_dict = claim.to_dict()
    print(f"  ✓ Claim exported to dict: {len(claim_dict)} keys")
    
    # Verify JSON serializable
    json_str = json.dumps(claim_dict, indent=2)
    print(f"  ✓ JSON serializable: {len(json_str)} chars")
    
    # Parse back
    parsed = json.loads(json_str)
    print(f"  ✓ JSON parsed successfully")
    print(f"  ✓ Claim ID: {parsed['claim_id']}")
    print(f"  ✓ Confidence: {parsed['confidence']}")
    
    return True


def test_nested_export():
    """Test complex nested structure export."""
    print("\nTesting nested structure export...")
    
    collection = ClaimCollection(session_id="nested_test")
    
    # Add multiple claims with different statuses
    for i in range(3):
        claim = LearningClaim(
            claim_id=f"claim_{i:03d}",
            claim_type=ClaimType.DEFINITION,
            claim_text=f"Test claim {i}",
            status=VerificationStatus.VERIFIED if i < 2 else VerificationStatus.REJECTED,
            confidence=0.8 - (i * 0.2),
            evidence_ids=[f"ev_{i}"],
            evidence_objects=[]
        )
        collection.add_claim(claim)
    
    # Create a nested structure similar to what's exported
    export_data = {
        "session_id": collection.session_id,
        "collection": collection.to_dict(),
        "statistics": collection.calculate_statistics(),
        "rejection_breakdown": collection.get_rejection_breakdown()
    }
    
    # Verify JSON serializable
    json_str = json.dumps(export_data, indent=2, default=str)
    print(f"  ✓ Nested structure serializable: {len(json_str)} chars")
    
    # Parse back
    parsed = json.loads(json_str)
    print(f"  ✓ JSON parsed successfully")
    print(f"  ✓ Total claims: {parsed['statistics']['total_claims']}")
    print(f"  ✓ Verified: {parsed['statistics']['verified_count']}")
    print(f"  ✓ Rejected: {parsed['statistics']['rejected_count']}")
    
    return True


def main():
    """Run all export tests."""
    print("=" * 60)
    print("Export Functionality Tests")
    print("=" * 60)
    
    try:
        test_claim_export()
        test_claim_collection_export()
        test_nested_export()
        
        print("\n" + "=" * 60)
        print("✓ ALL EXPORT TESTS PASSED")
        print("=" * 60)
        print("\nExport functionality is working correctly!")
        print("JSON serialization handles:")
        print("  • Individual claims with all fields")
        print("  • Collections with statistics")
        print("  • Nested structures with default=str fallback")
        
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
