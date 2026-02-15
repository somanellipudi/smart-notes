"""
Research-Grade Interactive Verifiability Assessment UI.

Integrates all components for reviewer-ready outputs:
- Summary cards and comparison panels
- Filterable claim tables with drill-down
- Graph visualization and exports
- Downloadable research artifacts
- Negative control detection and handling
"""

import streamlit as st
from typing import Dict, Any, List, Optional
import json
from io import BytesIO
from pathlib import Path

from src.claims.schema import LearningClaim, ClaimCollection
from src.display.interactive_claims import InteractiveClaimDisplay
from src.graph.claim_graph import ClaimGraph
from src.evaluation.verifiability_metrics import VerifiabilityMetrics
import config


class ResearchAssessmentUI:
    """Unified UI for research-grade verifiability assessment."""
    
    @staticmethod
    def render_full_assessment(
        verifiable_metadata: Dict[str, Any],
        baseline_claims_count: Optional[int] = None
    ) -> None:
        """
        Render complete interactive verifiability assessment section.
        
        Args:
            verifiable_metadata: Output from verifiable pipeline
            baseline_claims_count: Number of claims in baseline mode (optional)
        """
        if not verifiable_metadata:
            st.error("No verifiable metadata available")
            return
        
        # Extract components
        claim_collection: ClaimCollection = verifiable_metadata.get("claim_collection")
        claim_graph: ClaimGraph = verifiable_metadata.get("claim_graph")
        metrics: Dict[str, Any] = verifiable_metadata.get("metrics", {})
        
        if not claim_collection:
            st.error("No claim collection available")
            return
        
        claims = claim_collection.claims
        session_id = claim_collection.session_id
        
        # =====================================================================
        # 1. SUMMARY METRICS CARDS
        # =====================================================================
        
        st.header("🔬 Interactive Verifiability Assessment (Research)")
        
        st.subheader("📊 Summary Metrics")
        InteractiveClaimDisplay.display_summary_metrics(claims, metrics)
        
        st.divider()
        
        # =====================================================================
        # 2. BASELINE VS VERIFIABLE COMPARISON
        # =====================================================================
        
        if baseline_claims_count:
            InteractiveClaimDisplay.display_baseline_vs_verifiable(
                baseline_claims_count,
                claims
            )
            st.divider()
        
        # =====================================================================
        # 3. INPUT SUFFICIENCY WARNING (Pre-check)
        # =====================================================================
        
        input_sufficiency = metrics.get("input_sufficiency")
        if input_sufficiency and not input_sufficiency.get("is_sufficient"):
            with st.expander("⚠️ Input Sufficiency Warning", expanded=True):
                for warning in input_sufficiency.get("warnings", []):
                    st.warning(warning)
                st.info(input_sufficiency.get("recommendation", ""))
        
        # =====================================================================
        # 4. REJECTION REASON BREAKDOWN
        # =====================================================================
        
        rejection_reasons = metrics.get("rejection_reasons", {})
        if any(rejection_reasons.values()):
            with st.expander("🏷️ Rejection Reason Breakdown", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                for i, (reason, count) in enumerate(sorted(rejection_reasons.items())):
                    if count > 0:
                        col_idx = i % 3
                        if col_idx == 0:
                            with col1:
                                st.metric(reason, count)
                        elif col_idx == 1:
                            with col2:
                                st.metric(reason, count)
                        else:
                            with col3:
                                st.metric(reason, count)
        
        # =====================================================================
        # 5. TRACEABILITY METRICS
        # =====================================================================
        
        traceability = metrics.get("traceability_metrics", {})
        if traceability:
            with st.expander("🔗 Traceability Metrics", expanded=False):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Traceability Rate",
                        f"{traceability.get('traceability_rate', 0):.0%}"
                    )
                with col2:
                    st.metric(
                        "With Evidence",
                        traceability.get("claims_with_evidence", 0)
                    )
                with col3:
                    st.metric(
                        "Without Evidence",
                        traceability.get("claims_without_evidence", 0)
                    )
                with col4:
                    st.metric(
                        "Multi-Source Rate",
                        f"{traceability.get('multi_source_rate', 0):.0%}"
                    )
        
        st.divider()
        
        # =====================================================================
        # 6. NEGATIVE CONTROL DETECTION
        # =====================================================================
        
        negative_control_flag = metrics.get("negative_control", False)
        negative_control_details = metrics.get("negative_control_details", {})
        
        if negative_control_flag:
            explanation = negative_control_details.get("explanation", "No evidence found in sources.")
            trigger_reasons = negative_control_details.get("trigger_reasons", [])
            
            st.warning(
                f"🔴 **Negative Control Detected**\n\n"
                f"{explanation}\n\n"
                f"**Triggers**: {', '.join(trigger_reasons) if trigger_reasons else 'Unknown'}\n\n"
                "**This is correct behavior** - the system refused to hallucinate unsupported claims."
            )
        
        st.divider()
        
        st.subheader("📋 Claims Table (Sortable & Filterable)")
        
        filtered_claims = InteractiveClaimDisplay.display_claim_table_with_filters(claims)
        
        st.divider()
        
        # =====================================================================
        # 8. CLAIM DRILL-DOWN (Expandable Details)
        # =====================================================================
        
        st.subheader("🔍 Claim Details (Click to Expand)")
        
        for claim in filtered_claims:
            InteractiveClaimDisplay.display_claim_drill_down(claim)
        
        st.divider()
        
        # =====================================================================
        # 9. GRAPH VISUALIZATION
        # =====================================================================
        
        st.subheader("📊 Claim-Evidence Network Graph")
        
        ResearchAssessmentUI._render_graph_section(claim_graph, claims)
        
        st.divider()
        
        # =====================================================================
        # 10. DOWNLOADABLE ARTIFACTS
        # =====================================================================
        
        st.subheader("📥 Downloads (Research Artifacts)")
        
        ResearchAssessmentUI._render_downloads_section(
            claims=claims,
            claim_graph=claim_graph,
            session_id=session_id,
            metrics=metrics,
            baseline_count=baseline_claims_count,
            is_negative_control=negative_control_flag
        )
    
    @staticmethod
    def _render_graph_section(claim_graph: Optional[ClaimGraph], claims: List[LearningClaim]) -> None:
        """
        Render graph visualization section.
        
        Args:
            claim_graph: The claim-evidence graph
            claims: List of claims
        """
        if not claim_graph or not hasattr(claim_graph, 'graph'):
            # Fallback: create minimal graph from claims
            st.info("📋 No evidence nodes found. This run is a Negative Control (No Evidence).")
            
            # Still show claim-only visualization
            try:
                import matplotlib.pyplot as plt
                import matplotlib.patches as mpatches
                import networkx as nx
                
                G = nx.DiGraph()
                for c in claims:
                    G.add_node(
                        c.claim_id,
                        claim_type=getattr(c.claim_type, 'value', ''),
                        status=getattr(c.status, 'value', ''),
                        confidence=c.confidence
                    )
                
                if G.number_of_nodes() > 0:
                    fig, ax = plt.subplots(figsize=(12, 8))
                    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
                    
                    status_colors = {
                        "verified": "#28a745",
                        "low_confidence": "#ffc107",
                        "rejected": "#dc3545"
                    }
                    
                    colors = [
                        status_colors.get(G.nodes[n].get('status', '').lower(), '#6c757d')
                        for n in G.nodes()
                    ]
                    
                    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=1500, ax=ax)
                    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
                    
                    ax.set_title("Claims Overview (No Evidence Sources)", fontsize=12, fontweight='bold')
                    ax.axis('off')
                    plt.tight_layout()
                    
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
            except Exception as e:
                st.warning(f"Could not visualize: {e}")
            return
        
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
            
            G = claim_graph.graph
            
            if G.number_of_nodes() == 0:
                st.info("No graph data available")
                return
            
            # Full graph visualization
            fig, ax = plt.subplots(figsize=(14, 10))
            fig.patch.set_facecolor('#f8f9fa')
            ax.set_facecolor('#ffffff')
            
            claim_nodes = [node for node, attr in G.nodes(data=True) if attr.get("node_type") == "claim"]
            evidence_nodes = [node for node, attr in G.nodes(data=True) if attr.get("node_type") == "evidence"]
            
            if evidence_nodes:
                pos = {}
                for i, node in enumerate(claim_nodes):
                    pos[node] = (i * 2, 1)
                for i, node in enumerate(evidence_nodes):
                    pos[node] = (i * 1.5, 0)
            else:
                pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
            
            status_colors = {"verified": "#28a745", "low_confidence": "#ffc107", "rejected": "#dc3545"}
            claim_colors = [
                status_colors.get(
                    str(G.nodes[n].get('status', 'rejected')).lower(),
                    '#6c757d'
                )
                for n in claim_nodes
            ]
            
            if claim_nodes:
                nx.draw_networkx_nodes(
                    G, pos, nodelist=claim_nodes,
                    node_color=claim_colors,
                    node_size=2000,
                    edgecolors='#333',
                    linewidths=2,
                    ax=ax
                )
            
            if evidence_nodes:
                nx.draw_networkx_nodes(
                    G, pos, nodelist=evidence_nodes,
                    node_color="#17a2b8",
                    node_size=1200,
                    node_shape='s',
                    edgecolors='#333',
                    linewidths=2,
                    ax=ax,
                    alpha=0.8
                )
            
            if G.number_of_edges() > 0:
                edge_weights = [G[u][v].get('weight', 1.0) for u, v in G.edges()]
                nx.draw_networkx_edges(
                    G, pos,
                    edge_color='#666',
                    arrows=True,
                    arrowsize=15,
                    width=[w * 2 for w in edge_weights],
                    ax=ax,
                    alpha=0.6,
                    connectionstyle='arc3,rad=0.1'
                )
            
            claim_labels = {
                n: f"{G.nodes[n].get('claim_type', 'claim')}\n{G.nodes[n].get('confidence', 0):.0%}"
                for n in claim_nodes
            }
            nx.draw_networkx_labels(G, pos, claim_labels, font_size=9, font_weight='bold', ax=ax)
            
            title = f"Claim-Evidence Network: {len(claim_nodes)} Claims, {len(evidence_nodes)} Sources"
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            
            import matplotlib.patches as mpatches
            legend = [
                mpatches.Patch(color='#28a745', label='✓ Verified'),
                mpatches.Patch(color='#ffc107', label='⚠ Low Confidence'),
                mpatches.Patch(color='#dc3545', label='✗ Rejected')
            ]
            if evidence_nodes:
                legend.append(mpatches.Patch(color='#17a2b8', label='📄 Evidence'))
            
            ax.legend(handles=legend, loc='upper left', fontsize=10)
            ax.axis('off')
            plt.tight_layout()
            
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            
        except Exception as e:
            st.error(f"Graph visualization error: {e}")
    
    @staticmethod
    def _render_downloads_section(
        claims: List[LearningClaim],
        claim_graph: Optional[ClaimGraph],
        session_id: str,
        metrics: Dict[str, Any],
        baseline_count: Optional[int] = None,
        is_negative_control: bool = False
    ) -> None:
        """
        Render downloads expander with research artifacts.
        
        Args:
            claims: List of claims
            claim_graph: Claim-evidence graph
            session_id: Session identifier
            metrics: Pre-computed metrics
            baseline_count: Baseline claims count
            is_negative_control: Whether this is negative control
        """
        with st.expander("📥 Download Research Artifacts", expanded=False):
            
            col1, col2 = st.columns(2)
            
            # === JSON Audit Report ===
            with col1:
                st.write("**JSON Audit Report**")
                report = InteractiveClaimDisplay.create_claim_authenticity_report(
                    claims,
                    session_id,
                    baseline_count=baseline_count,
                    metrics=metrics,
                    is_negative_control=is_negative_control
                )
                
                report_json = json.dumps(report, indent=2, default=str)
                st.download_button(
                    label="📄 Download JSON Report",
                    data=report_json,
                    file_name=f"audit_report_{session_id[:8]}.json",
                    mime="application/json"
                )
            
            # === Claims CSV ===
            with col2:
                st.write("**Claims CSV**")
                claims_csv = InteractiveClaimDisplay.create_claims_csv(claims)
                st.download_button(
                    label="📊 Download Claims CSV",
                    data=claims_csv,
                    file_name=f"claims_{session_id[:8]}.csv",
                    mime="text/csv"
                )
            
            st.divider()
            
            col1, col2 = st.columns(2)
            
            # === Evidence CSV ===
            with col1:
                st.write("**Evidence CSV**")
                evidence_csv = InteractiveClaimDisplay.create_evidence_csv(claims)
                st.download_button(
                    label="📋 Download Evidence CSV",
                    data=evidence_csv,
                    file_name=f"evidence_{session_id[:8]}.csv",
                    mime="text/csv"
                )
            
            # === Graph Artifacts ===
            with col2:
                st.write("**Graph Artifacts**")
                
                if claim_graph and hasattr(claim_graph, 'export_adjacency_json'):
                    # GraphML
                    graphml_data = ""
                    try:
                        import tempfile
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.graphml', delete=False) as f:
                            if claim_graph.export_graphml(f.name):
                                with open(f.name, 'r') as gf:
                                    graphml_data = gf.read()
                        
                        if graphml_data:
                            st.download_button(
                                label="🔗 Download GraphML",
                                data=graphml_data,
                                file_name=f"graph_{session_id[:8]}.graphml",
                                mime="application/xml"
                            )
                    except:
                        pass
                    
                    # Adjacency JSON
                    adj_json = claim_graph.export_adjacency_json()
                    st.download_button(
                        label="📊 Download Adjacency JSON",
                        data=adj_json,
                        file_name=f"graph_adjacency_{session_id[:8]}.json",
                        mime="application/json"
                    )
                    
                    # PNG
                    try:
                        import tempfile
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.png', delete=False) as f:
                            if claim_graph.render_graph_png(f.name):
                                with open(f.name, 'rb') as gf:
                                    png_data = gf.read()
                                st.download_button(
                                    label="🖼️ Download Graph PNG",
                                    data=png_data,
                                    file_name=f"graph_{session_id[:8]}.png",
                                    mime="image/png"
                                )
                    except:
                        pass
            
            st.divider()
            
            # === Full Session JSON ===
            st.write("**Full Session Export**")
            
            full_session = {
                "session_id": session_id,
                "is_negative_control": is_negative_control,
                "timestamp": str(__import__('datetime').datetime.now()),
                "claims_count": len(claims),
                "verified_count": sum(1 for c in claims if getattr(c.status, 'value', '') == 'verified'),
                "rejected_count": sum(1 for c in claims if getattr(c.status, 'value', '') == 'rejected'),
                "metrics": metrics,
                "claims": [
                    {
                        "id": c.claim_id,
                        "type": getattr(c.claim_type, 'value', ''),
                        "status": getattr(c.status, 'value', ''),
                        "confidence": c.confidence,
                        "evidence_count": len(c.evidence_ids),
                        "rejection_reason": c.rejection_reason.value if c.rejection_reason else None,
                        "text": c.claim_text or c.metadata.get('ui_display', '')
                    }
                    for c in claims
                ]
            }
            
            session_json = json.dumps(full_session, indent=2, default=str)
            st.download_button(
                label="💾 Download Full Session JSON",
                data=session_json,
                file_name=f"session_export_{session_id[:8]}.json",
                mime="application/json"
            )
