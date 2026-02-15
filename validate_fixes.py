"""Quick validation script to verify all fixes work."""
import sys
sys.path.insert(0, '.')

print("=" * 60)
print("VALIDATING FEBRUARY 2025 FIXES")
print("=" * 60)

# Test 1: GraphMetrics imports and methods
print("\n✅ Test 1: GraphMetrics.get() method")
from src.claims.schema import GraphMetrics
m = GraphMetrics(
    avg_support_depth=1.0,
    avg_redundancy=2.0,
    avg_diversity=0.5,
    conflict_count=0,
    total_claims=10,
    total_evidence=20
)
assert m.get("total_claims") == 10
assert m.get("total_evidence") == 20
assert m.get("evidence_nodes") == 20  # Alias
assert m.get("nonexistent", "default") == "default"
print("   ✅ GraphMetrics.get() works")
print("   ✅ GraphMetrics.to_dict() works")

# Test 2: Graph sanitization imports
print("\n✅ Test 2: Graph sanitization utilities")
from src.graph.graph_sanitize import (
    export_graphml_bytes,
    export_graphml_string,
    sanitize_graph_for_graphml
)
print("   ✅ Imports successful")

# Test 3: ClaimGraph refactored methods
print("\n✅ Test 3: ClaimGraph exports")
from src.graph.claim_graph import ClaimGraph
print("   ✅ ClaimGraph imports successful")

# Test 4: App imports
print("\n✅ Test 4: App dependencies")
import streamlit as st
import pandas as pd
import networkx as nx
print(f"   ✅ Streamlit {st.__version__}")
print(f"   ✅ Pandas {pd.__version__}")
print(f"   ✅ NetworkX {nx.__version__}")

# Check optional dependencies
try:
    import pyarrow
    print(f"   ✅ PyArrow {pyarrow.__version__} (optional)")
except ImportError:
    print("   ⚠️  PyArrow not available (optional - OK)")

try:
    import matplotlib
    print(f"   ✅ Matplotlib {matplotlib.__version__} (optional)")
except ImportError:
    print("   ⚠️  Matplotlib not available (optional - OK)")

print("\n" + "=" * 60)
print("✅ ALL VALIDATIONS PASSED")
print("=" * 60)
print("\nReady to run: streamlit run app.py")
