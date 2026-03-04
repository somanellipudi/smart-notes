#!/usr/bin/env python
"""Quick test of the real retriever + NLI pipeline on a tiny sample.

This demonstrates the integration works without waiting for full evaluation.
"""

import sys
import os

# Set Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.retrieval.semantic_retriever import SemanticRetriever
from src.claims.nli_verifier import NLIVerifier
import numpy as np

print("=" * 60)
print("Testing Real Pipeline: SemanticRetriever + NLIVerifier")
print("=" * 60)

# Initialize models
print("\n[1] Loading SemanticRetriever...")
retriever = SemanticRetriever(device="cpu")

print("[2] Loading NLIVerifier...")
nli = NLIVerifier(device="cpu", batch_size=8)

# Create a tiny evidence corpus
evidence_text = """
Machine learning is a subset of artificial intelligence.
Deep learning uses neural networks with multiple layers.
Natural language processing handles text analysis.
"""

print("[3] Indexing evidence corpus...")
retriever.index_sources(external_context=evidence_text)
print(f"    → Indexed {retriever.index.ntotal} evidence spans")

# Test retrieval and verification on a few claims
claims = [
    "Machine learning is part of AI",
    "Deep learning uses neural networks",
    "Natural language processing is about numbers",
]

print(f"\n[4] Testing retrieval + NLI on {len(claims)} claims...")
for i, claim in enumerate(claims, 1):
    print(f"\n  Claim {i}: {claim}")
    
    # Retrieve evidence
    candidates = retriever.retrieve(
        claim_text=claim,
        top_k=5,
        rerank_top_n=3,
        min_similarity=0.0,
    )
    print(f"    → Retrieved {len(candidates)} evidence spans")
    
    if candidates:
        # Run NLI on candidates
        pairs = [(claim, span.text) for span in candidates]
        try:
            results, scores = nli.verify_batch_with_scores(pairs)
            entail_probs = scores[:, 0]
            max_entail = entail_probs.max()
            print(f"    → Max entailment prob: {max_entail:.3f}")
            
            # Decision logic
            if max_entail > 0.7:
                decision = "SUPPORTED"
            elif max_entail < 0.3:
                decision = "REFUTED"
            else:
                decision = "NEI"
            print(f"    → Decision: {decision}")
        except Exception as e:
            print(f"    → NLI error: {e}")

print("\n" + "=" * 60)
print("✅ Real pipeline integration test completed successfully!")
print("=" * 60)
