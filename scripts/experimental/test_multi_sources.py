"""
Quick test to verify multiple sources are being used
Run this in Python to verify the online evidence search is finding multiple sources
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.retrieval.online_evidence_search import OnlineEvidenceSearcher

def test_multi_source_search():
    """Test that multiple sources are being searched."""
    
    searcher = OnlineEvidenceSearcher(
        max_results_per_query=10,
        max_urls_to_fetch=5
    )
    
    # Test query
    test_queries = [
        "What is a stack data structure",
        "Python list functions",
        "Web API documentation",
        "machine learning algorithms"
    ]
    
    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"Query: {query}")
        print(f"{'='*70}")
        
        # Get search results
        results = searcher.search_duckduckgo(query)
        
        if results:
            print(f"✅ Found {len(results)} sources:\n")
            for i, result in enumerate(results, 1):
                print(f"{i}. {result.title}")
                print(f"   URL: {result.url[:80]}...")
                print(f"   Relevance: {result.relevance_score:.2f}")
                print()
        else:
            print("❌ No results found\n")

if __name__ == "__main__":
    print("Testing Multi-Source Evidence Search")
    print("=" * 70)
    test_multi_source_search()
    print("\n" + "="*70)
    print("✅ Test Complete")
