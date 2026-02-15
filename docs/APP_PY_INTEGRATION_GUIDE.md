"""
Quick Integration Guide: Adding ResearchAssessmentUI to app.py

This file shows exactly where and how to integrate the new research assessment UI
into your existing Streamlit application.

File: app.py
Section: display_output() function, verifiable mode block
"""

# ============================================================================
# STEP 1: ADD IMPORT AT TOP OF app.py
# ============================================================================

# Add this import with your other display imports:
from src.display.research_assessment_ui import ResearchAssessmentUI


# ============================================================================
# STEP 2: LOCATE display_output() FUNCTION
# ============================================================================

# Find this function in app.py (around line 570):
def display_output(result: dict, verifiable_metadata: Optional[Dict[str, Any]] = None):
    """
    Display structured output using new streaming display system.
    """
    output = result["output"]
    evaluation = result["evaluation"]
    
    # ========================================================================
    # STEP 3: REPLACE OLD VERIFIABLE MODE SECTION
    # ========================================================================
    
    # Find this section (around line 587):
    if verifiable_metadata:
        st.header("🔬 Verifiable Mode Metrics")
        
        # OLD CODE HERE (lines 587-850) - DELETE OR COMMENT OUT
        # ... (the old visualization code)
        
        # REPLACE WITH THIS:
        # ====================================================================
        
        # NEW: Single call to research assessment UI
        try:
            # Calculate baseline count if available
            baseline_count = None
            if output and hasattr(output, 'key_concepts'):
                baseline_count = (
                    len(output.key_concepts or []) +
                    len(output.equation_explanations or []) +
                    len(output.worked_examples or []) +
                    len(output.common_mistakes or []) +
                    len(output.faqs or []) +
                    len(output.real_world_connections or [])
                )
            
            # Render full assessment UI
            ResearchAssessmentUI.render_full_assessment(
                verifiable_metadata=verifiable_metadata,
                baseline_claims_count=baseline_count
            )
        except Exception as e:
            st.error(f"Error rendering assessment: {e}")
            import traceback
            st.error(traceback.format_exc())
        
        # ====================================================================
        st.divider()


# ============================================================================
# STEP 4: VERIFY DEPENDENCIES
# ============================================================================

# Make sure these are already imported in app.py:
# - streamlit as st
# - Dict, Any, List, Optional from typing
# - json, csv, io
# - matplotlib.pyplot (should already be there for graph viz)
# - networkx (should already be there)

# If not imported, add to top of app.py:
import json
import csv
from io import StringIO, BytesIO
from pathlib import Path


# ============================================================================
# STEP 5: TEST
# ============================================================================

# To test:
# 1. Enable Verifiable Mode checkbox
# 2. Upload content
# 3. Click Generate
# 4. Scroll to "🔬 Interactive Verifiability Assessment" section
# 5. Verify:
#    - Summary cards appear (6 columns)
#    - Baseline comparison shows (if baseline available)
#    - Rejection reason breakdown visible
#    - Traceability metrics shown
#    - Claim table filters work
#    - Drill-down expandable
#    - Graph renders
#    - Downloads work


# ============================================================================
# COMPLETE REPLACEMENT: display_output() VERIFIABLE MODE SECTION
# ============================================================================

"""
Full replacement code for the verifiable_metadata block in display_output():

    if verifiable_metadata:
        # ===================================================================
        # Research-Grade Interactive Verifiability Assessment
        # ===================================================================
        
        try:
            # Calculate baseline count for comparison
            baseline_count = None
            if output and hasattr(output, 'key_concepts'):
                baseline_count = (
                    len(output.key_concepts or []) +
                    len(output.equation_explanations or []) +
                    len(output.worked_examples or []) +
                    len(output.common_mistakes or []) +
                    len(output.faqs or []) +
                    len(output.real_world_connections or [])
                )
            
            # Single unified call to render all components
            ResearchAssessmentUI.render_full_assessment(
                verifiable_metadata=verifiable_metadata,
                baseline_claims_count=baseline_count
            )
        
        except Exception as e:
            st.error(f"Error rendering assessment: {e}")
            import traceback
            st.error(traceback.format_exc())
        
        st.divider()
"""


# ============================================================================
# IF YOU WANT INCREMENTAL INTEGRATION
# ============================================================================

"""
If you prefer to add components gradually instead of replacing all at once:

Step 1: Add summary cards
    InteractiveClaimDisplay.display_summary_metrics(
        claims,
        metrics
    )

Step 2: Add baseline comparison
    if baseline_claims_count:
        InteractiveClaimDisplay.display_baseline_vs_verifiable(
            baseline_claims_count,
            claims
        )

Step 3: Add filterable table
    filtered_claims = InteractiveClaimDisplay.display_claim_table_with_filters(
        claims
    )

Step 4: Add drill-down
    for claim in filtered_claims:
        InteractiveClaimDisplay.display_claim_drill_down(claim)

Step 5: Add graph
    ResearchAssessmentUI._render_graph_section(claim_graph, claims)

Step 6: Add downloads
    ResearchAssessmentUI._render_downloads_section(
        claims=claims,
        claim_graph=claim_graph,
        session_id=session_id,
        metrics=metrics,
        baseline_count=baseline_claims_count,
        is_negative_control=False
    )
"""


# ============================================================================
# ENVIRONMENT SETUP (if needed)
# ============================================================================

"""
Ensure your .env file has these (or use defaults):

# Verifiable mode thresholds
VERIFIABLE_VERIFIED_THRESHOLD=0.5
VERIFIABLE_REJECTED_THRESHOLD=0.2
VERIFIABLE_MIN_EVIDENCE=1

# Input sufficiency (for warnings)
VERIFIABLE_MIN_INPUT_TOKENS=100
VERIFIABLE_MIN_INPUT_CHUNKS=2

# Graph export settings
VERIFIABLE_GRAPH_DPI=150
VERIFIABLE_GRAPH_FIGSIZE_WIDTH=14
VERIFIABLE_GRAPH_FIGSIZE_HEIGHT=10

# High rejection warning
VERIFIABLE_HIGH_REJECTION_THRESHOLD=0.7

# Traceability expectations
VERIFIABLE_MIN_TRACEABILITY_FOR_GOOD_QUALITY=0.7
"""
