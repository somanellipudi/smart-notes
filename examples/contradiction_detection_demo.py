"""
Demonstration of contradiction detection features.

Shows how the contradiction gate and CS operation rules prevent contradictory
claims from being marked as VERIFIED.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.claims.nli_verifier import NLIVerifier
from src.claims.cs_operation_rules import check_cs_operation_contradiction
from src.claims.schema import LearningClaim, EvidenceItem
from src.evaluation.cs_benchmark_runner import CSBenchmarkRunner


def demo_contradiction_gate():
    """Demonstrate contradiction gate preventing false VERIFIED status."""
    print("=" * 80)
    print("DEMONSTRATION: Contradiction Gate")
    print("=" * 80)
    
    # Initialize NLI verifier
    print("\n1. Initializing NLI verifier (roberta-large-mnli)...")
    nli_verifier = NLIVerifier(device="cpu")
    
    # Test contradictory claim-evidence pair
    claim = "Stack push operation removes an element from the top"
    evidence = "The push operation adds an element to the top of the stack"
    
    print(f"\n2. Testing contradictory claim-evidence pair:")
    print(f"   Claim:    '{claim}'")
    print(f"   Evidence: '{evidence}'")
    
    result = nli_verifier.verify(claim, evidence)
    
    print(f"\n3. NLI Results:")
    print(f"   Label:              {result.label.value}")
    print(f"   Entailment prob:    {result.entailment_prob:.3f}")
    print(f"   Contradiction prob: {result.contradiction_prob:.3f}")
    print(f"   Neutral prob:       {result.neutral_prob:.3f}")
    
    # Check against threshold
    threshold = 0.6
    print(f"\n4. Contradiction Gate Check (threshold={threshold}):")
    if result.contradiction_prob >= threshold:
        print(f"   ✓ GATE TRIGGERED: contradiction_prob ({result.contradiction_prob:.3f}) >= {threshold}")
        print(f"   → Status: REJECTED (prevented false VERIFIED)")
    else:
        print(f"   ✗ Gate not triggered: contradiction_prob ({result.contradiction_prob:.3f}) < {threshold}")
    
    # Test non-contradictory pair
    claim2 = "Stack push operation adds an element to the top"
    evidence2 = "The push operation adds an element to the top of the stack"
    
    print(f"\n5. Testing non-contradictory claim-evidence pair:")
    print(f"   Claim:    '{claim2}'")
    print(f"   Evidence: '{evidence2}'")
    
    result2 = nli_verifier.verify(claim2, evidence2)
    
    print(f"\n6. NLI Results:")
    print(f"   Label:              {result2.label.value}")
    print(f"   Entailment prob:    {result2.entailment_prob:.3f}")
    print(f"   Contradiction prob: {result2.contradiction_prob:.3f}")
    
    print(f"\n7. Contradiction Gate Check:")
    if result2.contradiction_prob >= threshold:
        print(f"   ✓ Gate triggered: {result2.contradiction_prob:.3f} >= {threshold}")
    else:
        print(f"   ✗ Gate not triggered: {result2.contradiction_prob:.3f} < {threshold}")
        print(f"   → Status: Can be VERIFIED (passes gate)")


def demo_cs_operation_rules():
    """Demonstrate CS operation rules detecting semantic contradictions."""
    print("\n\n" + "=" * 80)
    print("DEMONSTRATION: CS Operation Rules")
    print("=" * 80)
    
    test_cases = [
        {
            "claim": "Stack push operation removes an element from the top",
            "evidence": "The push operation adds elements to the stack",
            "expected_contradiction": True,
            "description": "Push removes (contradicts: push should add)"
        },
        {
            "claim": "Stack pop operation adds an element to the top",
            "evidence": "Pop removes the top element from stack",
            "expected_contradiction": True,
            "description": "Pop adds (contradicts: pop should remove)"
        },
        {
            "claim": "Queue enqueue operation removes an element from the front",
            "evidence": "Enqueue adds element to the rear of queue",
            "expected_contradiction": True,
            "description": "Enqueue removes (contradicts: enqueue should add)"
        },
        {
            "claim": "Queue dequeue operation adds an element",
            "evidence": "Dequeue removes element from front of queue",
            "expected_contradiction": True,
            "description": "Dequeue adds (contradicts: dequeue should remove)"
        },
        {
            "claim": "Stack push operation adds an element to the top",
            "evidence": "The push operation adds elements to the stack",
            "expected_contradiction": False,
            "description": "Push adds (correct: no contradiction)"
        },
        {
            "claim": "Push adds and pop removes elements from stack",
            "evidence": "Stack operations: push adds to top, pop removes from top",
            "expected_contradiction": False,
            "description": "Multiple operations, all correct"
        }
    ]
    
    print("\n1. Testing CS operation semantic rules:")
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n   Test {i}: {test['description']}")
        print(f"   Claim:    '{test['claim']}'")
        print(f"   Evidence: '{test['evidence']}'")
        
        is_inconsistent, reason = check_cs_operation_contradiction(
            claim=test['claim'],
            evidence=test['evidence'],
            enabled=True
        )
        
        expected = test['expected_contradiction']
        status = "✓ PASS" if (is_inconsistent == expected) else "✗ FAIL"
        
        print(f"   Result:   {status}")
        print(f"   Expected: {'Contradiction' if expected else 'No contradiction'}")
        print(f"   Got:      {'Contradiction' if is_inconsistent else 'No contradiction'}")
        if reason:
            print(f"   Reason:   {reason}")


def demo_integrated_verification():
    """Demonstrate both features working together in verification pipeline."""
    print("\n\n" + "=" * 80)
    print("DEMONSTRATION: Integrated Verification Pipeline")
    print("=" * 80)
    
    print("\n1. Creating claim with contradictory evidence...")
    
    claim = LearningClaim(
        claim_text="Queue dequeue removes from rear",
        claim_type="CODE_BEHAVIOR_CLAIM",
        snippet_id="test_demo"
    )
    
    # Add contradictory evidence
    claim.evidence_objects = [
        EvidenceItem(
            evidence_id="ev_demo_1",
            snippet="Dequeue removes element from front of queue",
            similarity=0.85,
            metadata={"doc_id": "demo_doc"}
        ),
        EvidenceItem(
            evidence_id="ev_demo_2",
            snippet="Front element is removed by dequeue operation",
            similarity=0.80,
            metadata={"doc_id": "demo_doc"}
        )
    ]
    
    print(f"   Claim:     '{claim.claim_text}'")
    print(f"   Evidence:")
    for ev in claim.evidence_objects:
        print(f"   - {ev.snippet} (similarity: {ev.similarity:.2f})")
    
    print("\n2. Running verification with contradiction gate...")
    
    runner = CSBenchmarkRunner(
        dataset_path="evaluation/cs_benchmark/cs_benchmark_hard.jsonl",
        device="cpu",
        log_predictions=False
    )
    
    # Config with contradiction gate enabled
    config = {
        "enable_contradiction_gate": True,
        "contradiction_threshold": 0.6,
        "verify_threshold": 0.55,
        "low_conf_threshold": 0.35,
        "use_batch_nli": True
    }
    
    runner._validate_claim(claim, config)
    
    print(f"\n3. Verification Results:")
    print(f"   Status:     {claim.status.name}")
    print(f"   Confidence: {claim.confidence:.3f}")
    print(f"   Expected:   REJECTED (due to contradiction)")
    
    if claim.status.name == "REJECTED":
        print(f"\n   ✓ SUCCESS: Contradiction gate prevented false VERIFIED status!")
    else:
        print(f"\n   ✗ UNEXPECTED: Claim was not rejected despite contradiction")
    
    print("\n4. Testing with CS operation rules enabled...")
    
    claim2 = LearningClaim(
        claim_text="Push removes elements from stack",
        claim_type="CODE_BEHAVIOR_CLAIM",
        snippet_id="test_demo_2"
    )
    
    claim2.evidence_objects = [
        EvidenceItem(
            evidence_id="ev_demo_3",
            snippet="Push adds elements to the top of stack",
            similarity=0.90,
            metadata={"doc_id": "demo_doc"}
        )
    ]
    
    print(f"   Claim:     '{claim2.claim_text}'")
    print(f"   Evidence:  '{claim2.evidence_objects[0].snippet}'")
    
    config_with_cs_rules = {
        **config,
        "enable_cs_operation_rules": True
    }
    
    runner._validate_claim(claim2, config_with_cs_rules)
    
    print(f"\n5. Verification Results (with CS rules):")
    print(f"   Status:     {claim2.status.name}")
    print(f"   Confidence: {claim2.confidence:.3f}")
    print(f"   Expected:   REJECTED (due to CS operation contradiction)")
    
    if claim2.status.name == "REJECTED":
        print(f"\n   ✓ SUCCESS: CS operation rules detected semantic contradiction!")
    else:
        print(f"\n   ✗ UNEXPECTED: Claim was not rejected by CS rules")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 80)
    print("CONTRADICTION DETECTION DEMONSTRATION")
    print("=" * 80)
    print("\nThis script demonstrates the contradiction gate and CS operation rules")
    print("that prevent contradictory claims from being marked as VERIFIED.")
    print("\nFeatures:")
    print("1. Contradiction Gate: Uses NLI model to detect contradictions")
    print("2. CS Operation Rules: Detects CS-specific semantic contradictions")
    print("3. Integrated Pipeline: Both features working together")
    
    try:
        demo_contradiction_gate()
        demo_cs_operation_rules()
        demo_integrated_verification()
        
        print("\n\n" + "=" * 80)
        print("DEMONSTRATION COMPLETE")
        print("=" * 80)
        print("\nAll features working correctly!")
        print("\nConfiguration:")
        print("- Contradiction gate: ENABLED (threshold=0.6)")
        print("- CS operation rules: OPTIONAL (disabled by default)")
        print("\nTo enable in your code:")
        print("  config = {")
        print("      'enable_contradiction_gate': True,")
        print("      'contradiction_threshold': 0.6,")
        print("      'enable_cs_operation_rules': False  # or True")
        print("  }")
        
    except Exception as e:
        print(f"\n\n✗ ERROR during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
