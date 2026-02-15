"""
Example demonstrating Verifiable Mode usage.

This script shows how to use Smart Notes in Verifiable Mode,
including claim extraction, evidence retrieval, validation,
and metrics calculation.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.reasoning.verifiable_pipeline import VerifiablePipelineWrapper


def example_basic_usage():
    """Basic example of verifiable mode."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Verifiable Mode Usage")
    print("=" * 60)
    
    # Sample content
    lecture_content = """
    LECTURE TRANSCRIPT:
    Today we're going to learn about derivatives. A derivative represents
    the instantaneous rate of change of a function. Think of it as measuring
    how fast something is changing at a specific point.
    
    For example, if you have a function f(x) = x², the derivative f'(x) = 2x
    tells you the slope of the tangent line at any point x.
    
    HANDWRITTEN NOTES:
    - Derivative = rate of change
    - f(x) = x² → f'(x) = 2x
    - Used in physics for velocity, acceleration
    """
    
    equations = ["f(x) = x²", "f'(x) = 2x"]
    
    external_context = """
    From calculus textbook: The derivative of a function measures how the
    function value changes as its input changes. It is a fundamental concept
    in calculus with applications throughout mathematics and science.
    """
    
    # Initialize pipeline
    print("\n1. Initializing pipeline...")
    pipeline = VerifiablePipelineWrapper()
    
    # Process with verifiable mode
    print("2. Processing with verifiable_mode=True...")
    output, verifiable_metadata = pipeline.process(
        combined_content=lecture_content,
        equations=equations,
        external_context=external_context,
        session_id="demo_derivatives",
        verifiable_mode=True
    )
    
    # Access results
    print("\n3. Results:")
    print(f"   Session ID: {output.session_id}")
    
    if verifiable_metadata:
        metrics = verifiable_metadata["metrics"]
        print(f"\n   METRICS:")
        print(f"   - Total claims: {metrics['total_claims']}")
        print(f"   - Verified claims: {metrics['verified_claims']}")
        print(f"   - Rejected claims: {metrics['rejected_claims']}")
        print(f"   - Rejection rate: {metrics['rejection_rate']:.1%}")
        print(f"   - Avg confidence: {metrics['avg_confidence']:.2f}")
        
        # Show verified claims
        verified = verifiable_metadata["verified_collection"]
        print(f"\n   VERIFIED CLAIMS ({len(verified.claims)}):")
        for i, claim in enumerate(verified.claims[:3], 1):  # Show first 3
            print(f"\n   {i}. {claim.claim_text[:100]}...")
            print(f"      Type: {claim.claim_type.value}")
            print(f"      Confidence: {claim.confidence:.2f}")
            print(f"      Evidence count: {len(claim.evidence_objects)}")
            
            # Show evidence
            if claim.evidence_objects:
                evidence = claim.evidence_objects[0]
                print(f"      Evidence: \"{evidence.content[:80]}...\"")
                print(f"      Source: {evidence.source_type}")
        
        # Show rejected claims (if any)
        rejected = verifiable_metadata["claim_collection"].get_rejected_claims()
        if rejected:
            print(f"\n   REJECTED CLAIMS ({len(rejected)}):")
            for i, claim in enumerate(rejected[:2], 1):  # Show first 2
                print(f"   {i}. {claim.claim_text[:80]}...")
                print(f"      Reason: No evidence or low confidence ({claim.confidence:.2f})")


def example_comparison():
    """Compare standard mode vs verifiable mode."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Standard vs Verifiable Mode Comparison")
    print("=" * 60)
    
    # Sample content
    content = """
    The Pythagorean theorem states that in a right triangle,
    the square of the hypotenuse equals the sum of squares of
    the other two sides: a² + b² = c²
    """
    
    equations = ["a² + b² = c²"]
    
    pipeline = VerifiablePipelineWrapper()
    
    # Standard mode
    print("\n1. Running STANDARD mode...")
    output_standard, _ = pipeline.process(
        combined_content=content,
        equations=equations,
        external_context="",
        session_id="pythagoras_standard",
        verifiable_mode=False
    )
    
    print(f"   Standard output:")
    print(f"   - Topics: {len(output_standard.topics)}")
    print(f"   - Concepts: {len(output_standard.key_concepts)}")
    print(f"   - Equations: {len(output_standard.equation_explanations)}")
    
    # Verifiable mode
    print("\n2. Running VERIFIABLE mode...")
    output_verifiable, metadata = pipeline.process(
        combined_content=content,
        equations=equations,
        external_context="",
        session_id="pythagoras_verifiable",
        verifiable_mode=True
    )
    
    if metadata:
        metrics = metadata["metrics"]
        print(f"   Verifiable output:")
        print(f"   - Total claims: {metrics['total_claims']}")
        print(f"   - Verified: {metrics['verified_claims']}")
        print(f"   - Rejected: {metrics['rejected_claims']}")
        print(f"   - Rejection rate: {metrics['rejection_rate']:.1%}")
        
        # Baseline comparison
        if "baseline_comparison" in metrics:
            comp = metrics["baseline_comparison"]
            print(f"\n   COMPARISON:")
            print(f"   - Baseline items: {comp['baseline_total_items']}")
            print(f"   - Verified items: {comp['verifiable_verified_claims']}")
            print(f"   - Reduction: {comp['reduction_rate']:.1%}")


def example_evidence_analysis():
    """Analyze evidence quality."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Evidence Quality Analysis")
    print("=" * 60)
    
    # Rich content with good evidence
    content = """
    TRANSCRIPT:
    Newton's Second Law states that force equals mass times acceleration.
    This can be written as F = ma. This fundamental law explains how
    objects move when forces are applied to them.
    
    NOTES:
    - F = ma (Newton's 2nd Law)
    - Force causes acceleration
    - More mass = less acceleration for same force
    """
    
    equations = ["F = ma"]
    
    context = """
    Newton's Second Law: The acceleration of an object is directly
    proportional to the net force acting on it and inversely proportional
    to its mass. This is one of three laws of motion formulated by Isaac Newton.
    """
    
    pipeline = VerifiablePipelineWrapper()
    
    print("\n1. Processing with rich evidence sources...")
    output, metadata = pipeline.process(
        combined_content=content,
        equations=equations,
        external_context=context,
        session_id="newton_evidence",
        verifiable_mode=True
    )
    
    if metadata:
        # Evidence metrics
        ev_metrics = metadata["metrics"]["evidence_metrics"]
        print(f"\n   EVIDENCE QUALITY:")
        print(f"   - Avg evidence per claim: {ev_metrics['avg_evidence_per_claim']:.1f}")
        print(f"   - Unsupported claims: {ev_metrics['claims_without_evidence']}")
        print(f"   - Unsupported rate: {ev_metrics['unsupported_rate']:.1%}")
        print(f"   - Avg evidence quality: {ev_metrics['avg_evidence_quality']:.2f}")
        
        # Show a claim with its evidence
        verified = metadata["verified_collection"]
        if verified.claims:
            print(f"\n   EXAMPLE CLAIM WITH EVIDENCE:")
            claim = verified.claims[0]
            print(f"   Claim: {claim.claim_text}")
            print(f"   Confidence: {claim.confidence:.2f}")
            print(f"\n   Evidence ({len(claim.evidence_objects)} items):")
            for i, evidence in enumerate(claim.evidence_objects, 1):
                print(f"   {i}. Source: {evidence.source_type}")
                print(f"      Quote: \"{evidence.content[:100]}...\"")
                print(f"      Relevance: {evidence.relevance_score:.2f}")
                print()


def example_graph_analysis():
    """Analyze claim-evidence graph."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Claim-Evidence Graph Analysis")
    print("=" * 60)
    
    content = """
    Photosynthesis is the process by which plants convert light energy
    into chemical energy. The equation is: 6CO₂ + 6H₂O → C₆H₁₂O₆ + 6O₂
    This process occurs in chloroplasts and requires sunlight, water, and CO₂.
    """
    
    pipeline = VerifiablePipelineWrapper()
    
    print("\n1. Building claim-evidence graph...")
    output, metadata = pipeline.process(
        combined_content=content,
        equations=["6CO₂ + 6H₂O → C₆H₁₂O₆ + 6O₂"],
        external_context="",
        session_id="photosynthesis_graph",
        verifiable_mode=True
    )
    
    if metadata and "graph" in metadata:
        graph = metadata["graph"]
        stats = graph.get_statistics()
        
        print(f"\n   GRAPH STATISTICS:")
        print(f"   - Total nodes: {stats['nodes']['total']}")
        print(f"   - Claim nodes: {stats['nodes']['claims']}")
        print(f"   - Evidence nodes: {stats['nodes']['evidence']}")
        print(f"   - Support edges: {stats['edges']['supports']}")
        print(f"   - Unsupported claims: {stats['metrics']['unsupported_claims']}")
        print(f"   - Avg evidence/claim: {stats['metrics']['avg_evidence_per_claim']:.1f}")
        print(f"   - Graph density: {stats['metrics']['graph_density']:.3f}")
        
        # Find claims with most evidence
        collection = metadata["claim_collection"]
        claims_by_evidence = sorted(
            collection.claims,
            key=lambda c: len(c.evidence_objects),
            reverse=True
        )
        
        print(f"\n   TOP CLAIMS BY EVIDENCE:")
        for i, claim in enumerate(claims_by_evidence[:3], 1):
            print(f"   {i}. {len(claim.evidence_objects)} evidence items")
            print(f"      {claim.claim_text[:80]}...")


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "    SMART NOTES - VERIFIABLE MODE EXAMPLES".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    
    try:
        # Run examples
        example_basic_usage()
        example_comparison()
        example_evidence_analysis()
        example_graph_analysis()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        print("\nFor more information:")
        print("  - Full docs: docs/VERIFIABLE_MODE.md")
        print("  - Quick ref: docs/QUICK_REFERENCE.md")
        print("  - Implementation: docs/IMPLEMENTATION_SUMMARY.md")
        print()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
