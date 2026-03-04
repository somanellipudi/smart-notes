#!/usr/bin/env python
"""
Latency profiler for Smart Notes pipeline.

Measures stage-wise timing (retrieval, NLI, confidence aggregation) and caching impact.
"""

import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

# Mock components for profiling without loading real models
class MockRetriever:
    def retrieve(self, query, top_k=20):
        """Simulate retrieval delay (typically 10-100ms per query)."""
        time.sleep(0.05)  # 50ms
        return [(f"doc_{i}", 0.9 - i * 0.01) for i in range(top_k)]

class MockNLIModel:
    def predict(self, premise, hypothesis):
        """Simulate NLI inference delay (typically 50-150ms per pair)."""
        time.sleep(0.08)  # 80ms
        return {"entailment": 0.7, "neutral": 0.2, "contradiction": 0.1}

class PipelineProfiler:
    def __init__(self, n_claims: int = 100, top_k: int = 20, use_cache: bool = True):
        self.n_claims = n_claims
        self.top_k = top_k
        self.use_cache = use_cache
        self.retriever = MockRetriever()
        self.nli = MockNLIModel()
        self.cache = {}
        self.timings = {
            "retrieval": [],
            "nli_inference": [],
            "confidence_aggregation": [],
            "total_per_claim": [],
        }
    
    def profile_retrieval(self, claim: str) -> List[tuple]:
        """Profile retrieval stage."""
        cache_key = f"retrieval:{claim}"
        if self.use_cache and cache_key in self.cache:
            return self.cache[cache_key]
        
        start = time.perf_counter()
        docs = self.retriever.retrieve(claim, top_k=self.top_k)
        elapsed = time.perf_counter() - start
        
        self.timings["retrieval"].append(elapsed)
        if self.use_cache:
            self.cache[cache_key] = docs
        return docs
    
    def profile_nli_inference(self, claim: str, docs: List[tuple]) -> List[Dict]:
        """Profile NLI inference stage."""
        results = []
        for doc_id, score in docs[:3]:  # NLI on top-3 docs
            cache_key = f"nli:{claim}:{doc_id}"
            if self.use_cache and cache_key in self.cache:
                results.append(self.cache[cache_key])
                continue
            
            start = time.perf_counter()
            pred = self.nli.predict(doc_id, claim)
            elapsed = time.perf_counter() - start
            
            self.timings["nli_inference"].append(elapsed)
            if self.use_cache:
                self.cache[cache_key] = pred
            results.append(pred)
        return results
    
    def profile_confidence_aggregation(self, claim_idx: int, docs, nli_preds) -> float:
        """Profile confidence aggregation stage."""
        start = time.perf_counter()
        # Simulate lightweight aggregation
        confidences = [pred["entailment"] for pred in nli_preds]
        final_conf = np.mean(confidences)
        elapsed = time.perf_counter() - start
        self.timings["confidence_aggregation"].append(elapsed)
        return final_conf
    
    def profile_full_pipeline(self):
        """Profile end-to-end pipeline on synthetic claims."""
        for i in range(self.n_claims):
            claim = f"Claim {i}: This is a test claim about topic {i % 10}."
            
            start = time.perf_counter()
            
            # Stage 1: Retrieval
            docs = self.profile_retrieval(claim)
            
            # Stage 2: NLI Inference
            nli_preds = self.profile_nli_inference(claim, docs)
            
            # Stage 3: Confidence Aggregation
            conf = self.profile_confidence_aggregation(i, docs, nli_preds)
            
            elapsed = time.perf_counter() - start
            self.timings["total_per_claim"].append(elapsed)
        
        return self._summarize_timings()
    
    def _summarize_timings(self) -> Dict[str, Any]:
        """Summarize timing statistics."""
        summary = {}
        for stage, times in self.timings.items():
            if times:
                times_arr = np.array(times)
                summary[stage] = {
                    "mean_ms": float(times_arr.mean() * 1000),
                    "std_ms": float(times_arr.std() * 1000),
                    "min_ms": float(times_arr.min() * 1000),
                    "max_ms": float(times_arr.max() * 1000),
                    "median_ms": float(np.median(times_arr) * 1000),
                    "count": int(len(times)),
                }
        
        # Add cache stats
        summary["cache"] = {
            "enabled": self.use_cache,
            "size": len(self.cache),
            "estimated_hits": max(0, (self.n_claims - 1) * 3),  # Approximate repeated retrievals
        }
        
        # Add throughput
        total_time = sum(self.timings["total_per_claim"])
        summary["throughput"] = {
            "claims_per_second": float(self.n_claims / total_time) if total_time > 0 else 0,
            "total_time_seconds": float(total_time),
            "estimated_latency_per_claim_ms": float(np.mean(self.timings["total_per_claim"]) * 1000),
        }
        
        return summary

def main():
    parser = argparse.ArgumentParser(description="Profile Smart Notes pipeline latency")
    parser.add_argument("--n_claims", type=int, default=100, help="Number of claims to profile")
    parser.add_argument("--top_k", type=int, default=20, help="Top-k retrieval results")
    parser.add_argument("--no_cache", action="store_true", help="Disable caching")
    parser.add_argument("--output", type=str, default="outputs/profiling/latency_profile.json",
                        help="Output path for profiling results")
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Run profiling
    print(f"Profiling pipeline with {args.n_claims} claims (caching={'disabled' if args.no_cache else 'enabled'})...")
    profiler = PipelineProfiler(
        n_claims=args.n_claims,
        top_k=args.top_k,
        use_cache=not args.no_cache
    )
    results = profiler.profile_full_pipeline()
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("LATENCY PROFILE SUMMARY")
    print(f"{'='*60}")
    for stage, stats in results.items():
        if stage == "cache":
            print(f"\nCache Statistics:")
            print(f"  Enabled: {stats['enabled']}")
            print(f"  Entries: {stats['size']}")
            print(f"  Estimated hits: {stats['estimated_hits']}")
        elif stage == "throughput":
            print(f"\nThroughput:")
            print(f"  Claims/sec: {stats['claims_per_second']:.1f}")
            print(f"  Total time: {stats['total_time_seconds']:.2f}s")
            print(f"  Avg latency per claim: {stats['estimated_latency_per_claim_ms']:.1f}ms")
        else:
            print(f"\n{stage}:")
            print(f"  Mean: {stats['mean_ms']:.1f}ms (±{stats['std_ms']:.1f})")
            print(f"  Range: {stats['min_ms']:.1f}ms - {stats['max_ms']:.1f}ms")
            print(f"  Median: {stats['median_ms']:.1f}ms")
    
    print(f"\nResults saved to: {args.output}")

if __name__ == "__main__":
    main()
