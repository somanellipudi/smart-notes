"""
Interactive claim display with hover tooltips for verifiability assessment.

Enables researchers to inspect AI-generated content with confidence scores,
evidence sources, and authenticity indicators.

Research-Grade Features:
- Summary metric cards (Total, Verified, Rejected, Refusal Rate, Traceability, Conflict)
- Sortable/filterable claim table with drill-down
- Baseline vs Verifiable comparison panel
- Claim-evidence graph visualization with exports
- Downloadable artifacts (JSON/CSV/GraphML/PNG)
- Negative control handling and explanation
"""

import streamlit as st
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from src.claims.schema import LearningClaim, ClaimType, VerificationStatus
import json
import csv
from io import StringIO, BytesIO


class InteractiveClaimDisplay:
    """Display claims with interactive hover tooltips and authenticity indicators."""
    
    @staticmethod
    def display_claim_with_tooltip(
        claim: LearningClaim,
        content_text: str = None,
        show_metadata: bool = True
    ) -> str:
        """
        Display content with interactive claim details.
        
        Args:
            claim: The LearningClaim object
            content_text: Text to display (uses ui_display or draft_text if None)
            show_metadata: Whether to show metadata inline
        
        Returns:
            HTML for display
        """
        if content_text is None:
            content_text = (
                claim.metadata.get("ui_display", "") or
                claim.claim_text or
                claim.metadata.get("draft_text", "")
            )
        
        status_value = getattr(claim.status, "value", str(claim.status))
        claim_type = getattr(claim.claim_type, "value", str(claim.claim_type))
        
        # Determine trust color based on confidence
        confidence = claim.confidence
        if confidence >= 0.8:
            trust_color = "#2ecc71"  # Green - verified
            trust_label = "✅ Verified"
        elif confidence >= 0.6:
            trust_color = "#f39c12"  # Orange - likely trustworthy
            trust_label = "⚠️ Moderate"
        else:
            trust_color = "#e74c3c"  # Red - low confidence
            trust_label = "❌ Low Trust"
        
        # Generate unique ID for tooltip
        claim_id_short = claim.claim_id[:8]
        
        # Create interactive HTML with tooltip
        html = f"""
        <div class="claim-container" data-claim-id="{claim_id_short}" 
             style="border-left: 4px solid {trust_color}; padding: 8px 12px; margin: 8px 0; background-color: #f9f9f9; border-radius: 4px;">
            
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 16px; font-weight: 500; color: #333;">
                    {content_text[:100]}{'...' if len(content_text) > 100 else ''}
                </span>
                <span style="font-size: 12px; color: {trust_color}; font-weight: bold;">
                    {trust_label}
                </span>
            </div>
            
            <div style="margin-top: 8px; font-size: 12px; color: #666;">
                <strong>AI-Generated Content</strong> • Type: {claim_type} • Confidence: {confidence:.1%}
            </div>
            
            <div style="margin-top: 6px; font-size: 11px; color: #999;">
                Claim ID: {claim_id_short} | Status: {status_value} | Evidence: {len(claim.evidence_ids)}
            </div>
        </div>
        """
        
        return html
    
    @staticmethod
    def display_claim_detail_modal(claim: LearningClaim) -> None:
        """
        Display detailed claim information in a modal-like expander.
        
        Args:
            claim: The claim to display
        """
        status_value = getattr(claim.status, "value", str(claim.status))
        claim_type = getattr(claim.claim_type, "value", str(claim.claim_type))
        
        # Confidence color coding
        if claim.confidence >= 0.8:
            confidence_color = "🟢"
        elif claim.confidence >= 0.6:
            confidence_color = "🟡"
        else:
            confidence_color = "🔴"
        
        # Determine if verified or answered
        is_verified = status_value.lower() == "verified"
        is_answered = status_value.lower() == "answered_with_citations"
        claim_type_val = getattr(claim.claim_type, "value", str(claim.claim_type))
        is_question = claim_type_val == "question"
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            title = (
                claim.metadata.get("ui_display", "") or
                claim.claim_text or
                claim.metadata.get("draft_text", "Claim")
            )
            # Add question mark emoji for questions
            prefix = "❓ " if is_question else ""
            st.write(f"**{prefix}{title[:150]}**")
        
        with col2:
            # Show different metrics for questions vs facts
            if is_question:
                st.metric(
                    "Answer Quality",
                    f"{claim.confidence:.0%}",
                    delta="Answered" if is_answered else "Pending"
                )
            else:
                st.metric(
                    "Confidence",
                    f"{claim.confidence:.0%}",
                    delta="Verified" if is_verified else "Pending"
                )
        
        with col3:
            st.metric(
                "Evidence",
                len(claim.evidence_ids),
                delta=f"{len(claim.evidence_objects)} attached"
            )
        
        # Expandable details
        with st.expander("🔍 Full Claim Details & Sources", expanded=False):
            tabs = st.tabs(["Overview", "Evidence", "Metadata", "Assessment"])
            
            # Tab 1: Overview
            with tabs[0]:
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.write("**Claim Properties**")
                    st.info(
                        f"• **Type**: {claim_type}\n"
                        f"• **Status**: {status_value}\n"
                        f"• **ID**: {claim.claim_id[:16]}...\n"
                        f"• **Created**: {claim.metadata.get('created_at', 'N/A')[:10]}"
                    )
                    
                    # Show answer for questions
                    if is_question and claim.answer_text:
                        st.write("**Answer:**")
                        st.success(claim.answer_text)
                
                with col_b:
                    st.write("**Confidence Analysis**")
                    st.metric("Overall Confidence", f"{claim.confidence:.3f}")
                    
                    # Confidence breakdown
                    if claim.confidence >= 0.7:
                        st.success("✅ High confidence - backed by evidence")
                    elif claim.confidence >= 0.4:
                        st.warning("⚠️ Moderate confidence - limited evidence")
                    else:
                        st.error("❌ Low confidence - insufficient evidence")
            
            # Tab 2: Evidence Sources
            with tabs[1]:
                st.write("**Evidence Sources & Relevance**")
                
                if claim.evidence_objects:
                    # Import citation display helper
                    try:
                        from src.display.citation_display import (
                            CitationInfo,
                            render_citation_inline,
                            get_source_icon
                        )
                        use_new_citation_display = True
                    except ImportError:
                        use_new_citation_display = False
                    
                    for i, evidence in enumerate(claim.evidence_objects, 1):
                        # Get provenance fields
                        source_type = evidence.source_type or "unknown"
                        origin = getattr(evidence, "origin", None)
                        page_num = getattr(evidence, "page_num", None)
                        timestamp_range = getattr(evidence, "timestamp_range", None)
                        
                        # Source badge
                        if use_new_citation_display:
                            source_icon = get_source_icon(source_type)
                        else:
                            source_colors = {
                                "transcript": "🎙️",
                                "notes": "📝",
                                "external_context": "🌐",
                                "equations": "📐"
                            }
                            source_icon = source_colors.get(source_type, "📄")
                        
                        similarity = getattr(evidence, "similarity", None)
                        similarity_display = f"{similarity:.0%}" if isinstance(similarity, (int, float)) else "N/A"
                        snippet = getattr(evidence, "snippet", "") or ""
                        source_id = getattr(evidence, "source_id", None)
                        span_meta = getattr(evidence, "span_metadata", {}) or {}
                        location = span_meta.get("location") or span_meta.get("line") or span_meta.get("page")

                        with st.expander(
                            f"{source_icon} Source {i}: {source_type} "
                            f"(similarity: {similarity_display})",
                            expanded=(i == 1)
                        ):
                            # Show citation with provenance
                            if use_new_citation_display and (origin or page_num or timestamp_range):
                                citation_info = CitationInfo(
                                    source_type=source_type,
                                    origin=origin,
                                    page_num=page_num,
                                    timestamp_range=timestamp_range,
                                    snippet=snippet,
                                    confidence=similarity if isinstance(similarity, (int, float)) else 1.0
                                )
                                render_citation_inline(citation_info, max_length=80)
                                st.divider()
                            
                            # Show snippet
                            st.write(f"**Snippet**: {snippet}")
                            
                            # Show metadata
                            st.caption(
                                f"Source ID: {source_id or 'N/A'} | "
                                f"Location: {location or page_num or 'N/A'} | "
                                f"Similarity: {similarity_display}"
                            )
                            
                            # Show clickable URL if available
                            if origin and origin.startswith("http"):
                                st.link_button("🔗 Open Source", origin)
                else:
                    st.warning("⚠️ No evidence attached to this claim yet")
            
            # Tab 3: Metadata
            with tabs[2]:
                st.write("**Metadata**")
                
                # Show key metadata
                if claim.metadata:
                    for key, value in list(claim.metadata.items())[:8]:
                        if isinstance(value, (str, int, float, bool)):
                            st.write(f"**{key}**: {value}")
                        elif isinstance(value, list) and len(value) <= 5:
                            st.write(f"**{key}**: {', '.join(map(str, value))}")
            
            # Tab 4: Verifiability Assessment
            with tabs[3]:
                st.write("**Verifiability Assessment for Research**")
                
                assessment = f"""
                ### AI Content Authenticity Evaluation
                
                **Claim Type**: {claim_type}  
                **Verifiability Score**: {claim.confidence:.1%}  
                **Evidence Count**: {len(claim.evidence_ids)}  
                **Status**: {status_value}
                
                #### Key Indicators:
                • **Evidence-Backed**: {'✅ Yes' if len(claim.evidence_ids) > 0 else '❌ No'}
                • **Multi-Source**: {'✅ Yes' if len(set(e.source_type for e in claim.evidence_objects)) > 1 else '⚠️ Single source'}
                • **Conflict Detected**: {'⚠️ Yes - Review evidence' if claim.metadata.get('has_conflicts') else '✅ No conflicts'}
                • **Consistency Score**: {claim.metadata.get('consistency_score', 'N/A')}
                
                #### Verifiability Recommendation:
                """
                
                if claim.confidence >= 0.8:
                    assessment += "🟢 **HIGH** - Content is well-supported by evidence from multiple sources"
                elif claim.confidence >= 0.6:
                    assessment += "🟡 **MODERATE** - Content has some evidence support but may need additional sources"
                else:
                    assessment += "🔴 **LOW** - Content lacks sufficient evidence and should be used with caution"
                
                st.markdown(assessment)
    
    @staticmethod
    def display_questions_with_answers(
        questions: List[LearningClaim],
        show_confidence: bool = True
    ) -> None:
        """
        Display questions separately with their answers and citations.
        
        Args:
            questions: List of question claims (ClaimType.QUESTION)
            show_confidence: Whether to show answer confidence
        """
        if not questions:
            return
        
        st.write("### ❓ Study Questions")
        st.caption(f"{len(questions)} questions with evidence-based answers")
        
        for i, q in enumerate(questions, 1):
            status = getattr(q.status, "value", str(q.status))
            
            # Question display
            with st.expander(f"**Q{i}:** {q.claim_text}", expanded=False):
                # Show answer if available
                if q.answer_text and status == "answered_with_citations":
                    st.markdown("**Answer:**")
                    st.info(q.answer_text)
                    
                    if show_confidence and q.confidence > 0:
                        confidence_color = "🟢" if q.confidence > 0.7 else "🟡" if q.confidence > 0.5 else "🟠"
                        st.caption(f"{confidence_color} Answer confidence: {q.confidence:.0%}")
                    
                    # Show sources if available
                    if q.evidence_objects:
                        st.write("**Sources:**")
                        for idx, ev in enumerate(q.evidence_objects[:3], 1):
                            source = ev.metadata.get("source", "Unknown")
                            st.caption(f"[{idx}] {source}: {ev.text[:100]}...")
                else:
                    st.warning("⚠️ No answer available yet")
                
                # Show metadata in collapsed section
                with st.expander("📊 Question Details", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Status:** {status}")
                        st.write(f"**Confidence:** {q.confidence:.0%}")
                    with col2:
                        st.write(f"**Evidence Count:** {len(q.evidence_ids)}")
                        st.write(f"**ID:** {q.claim_id[:8]}")
    
    @staticmethod
    def display_claims_summary_with_verifiability(
        claims: List[LearningClaim],
        show_ai_badge: bool = True,
        separate_questions: bool = True
    ) -> None:
        """
        Display summary of all claims with verifiability indicators.
        Separates questions from factual claims for clarity.
        
        Args:
            claims: List of claims to display
            show_ai_badge: Whether to show "AI-Generated" badge
            separate_questions: Whether to show questions in separate section
        """
        if not claims:
            st.info("No claims to display")
            return
        
        # Separate questions from fact claims
        questions = []
        fact_claims = []
        
        for c in claims:
            claim_type = getattr(c.claim_type, "value", str(c.claim_type))
            if separate_questions and claim_type == "question":
                questions.append(c)
            else:
                fact_claims.append(c)
        
        # Summary metrics (for fact claims)
        display_claims = fact_claims if separate_questions else claims
        
        if display_claims:
            col1, col2, col3, col4 = st.columns(4)
            
            verified_count = sum(1 for c in display_claims if getattr(c.status, "value", "") == "verified")
            avg_confidence = sum(c.confidence for c in display_claims) / len(display_claims) if display_claims else 0
            evidence_count = sum(len(c.evidence_ids) for c in display_claims)
            
            with col1:
                st.metric("Fact Claims", len(display_claims))
            
            with col2:
                st.metric("Verified", verified_count, delta=f"{verified_count/len(display_claims)*100:.0f}%" if display_claims else "0%")
            
            with col3:
                st.metric("Avg Confidence", f"{avg_confidence:.1%}")
            
            with col4:
                st.metric("Total Evidence", evidence_count)
            
            st.divider()
        
        # Display questions separately
        if separate_questions and questions:
            InteractiveClaimDisplay.display_questions_with_answers(questions)
            st.divider()
        
        # Interactive fact claims list
        if display_claims:
            st.write("### 🔬 Verified Facts Review")
            
            for claim in display_claims:
                InteractiveClaimDisplay.display_claim_detail_modal(claim)
                st.divider()
    
    @staticmethod
    def display_summary_metrics(
        claims: List[LearningClaim],
        metrics: Optional[Dict[str, Any]] = None,
        show_questions: bool = True
    ) -> None:
        """
        Display research-grade summary metric cards.
        
        Shows:
        - Total Claims
        - Verified Claims
        - Rejected Claims
        - Refusal Rate (% rejected)
        - Traceability (% with evidence)
        - Conflict Rate (if applicable)
        - Questions (if show_questions=True)
        
        Args:
            claims: List of claims to summarize
            metrics: Optional pre-computed metrics (includes conflict_count, etc.)
            show_questions: Whether to show question count separately
        """
        if not claims:
            st.info("No claims to display")
            return
        
        # Separate questions from fact claims
        questions = [c for c in claims if getattr(c.claim_type, "value", "") == "question"]
        fact_claims = [c for c in claims if getattr(c.claim_type, "value", "") != "question"]
        
        verified = sum(1 for c in fact_claims if getattr(c.status, "value", "") == "verified")
        rejected = sum(1 for c in fact_claims if getattr(c.status, "value", "") == "rejected")
        low_conf = sum(1 for c in fact_claims if getattr(c.status, "value", "") == "low_confidence")
        answered = sum(1 for c in questions if getattr(c.status, "value", "") == "answered_with_citations")
        
        total = len(claims)
        fact_total = len(fact_claims)
        refusal_rate = rejected / fact_total if fact_total > 0 else 0
        traceability = sum(1 for c in claims if len(c.evidence_ids) > 0) / total if total > 0 else 0
        conflict_count = metrics.get("graph_metrics", {}).get("conflict_count", 0) if metrics else 0
        
        # Display with or without questions
        if show_questions and questions:
            col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
            
            with col1:
                st.metric("📊 Total", total)
            with col2:
                st.metric("✅ Verified", verified, delta=f"{verified/fact_total*100:.0f}%" if fact_total > 0 else "0%")
            with col3:
                st.metric("❓ Questions", len(questions), delta=f"{answered} answered")
            with col4:
                st.metric("❌ Rejected", rejected, delta=f"{rejected/fact_total*100:.0f}%" if fact_total > 0 else "0%")
            with col5:
                st.metric("⚠️ Low Conf", low_conf)
            with col6:
                st.metric("🔗 Evidence", f"{traceability:.0%}")
            with col7:
                st.metric("⚡ Conflicts", conflict_count)
        else:
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            
            with col1:
                st.metric("📊 Total Claims", total)
            with col2:
                st.metric("✅ Verified", verified, delta=f"{verified/total*100:.0f}%")
            with col3:
                st.metric("❌ Rejected", rejected, delta=f"{rejected/total*100:.0f}%")
            with col4:
                st.metric("⚠️ Low Conf", low_conf, delta=f"{low_conf/total*100:.0f}%")
            with col5:
                st.metric("🔗 Traceability", f"{traceability:.0%}")
            with col6:
                st.metric("⚡ Conflicts", conflict_count)
        
        # High rejection warning
        if refusal_rate > 0.7:
            st.info(
                "🔬 **High Rejection Indicates Correct Abstention**: "
                "The system is rejecting unsupported claims as designed. "
                "This is expected when source material is insufficient."
            )
    
    @staticmethod
    def display_baseline_vs_verifiable(
        baseline_claims_count: int,
        verifiable_claims: List[LearningClaim]
    ) -> None:
        """
        Display baseline vs verifiable mode comparison.
        
        Args:
            baseline_claims_count: Number of claims in baseline (standard mode)
            verifiable_claims: Claims in verifiable mode
        """
        verified = sum(1 for c in verifiable_claims if getattr(c.status, "value", "") == "verified")
        rejected = sum(1 for c in verifiable_claims if getattr(c.status, "value", "") == "rejected")
        low_conf = sum(1 for c in verifiable_claims if getattr(c.status, "value", "") == "low_confidence")
        
        with st.expander("📊 Baseline vs Verifiable Comparison", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Baseline Mode (Standard)**")
                st.write(f"• Claims Generated: {baseline_claims_count}")
                st.write("• Evidence Checking: ❌ NO")
                st.write("• Rejection: 0%")
                st.write("• All claims accepted as-is")
            
            with col2:
                st.write("**Verifiable Mode (Evidence-First)**")
                st.write(f"• Claims Extracted: {len(verifiable_claims)}")
                st.write(f"• Verified: {verified} ✅")
                st.write(f"• Low Confidence: {low_conf} ⚠️")
                st.write(f"• Rejected: {rejected} ❌")
                st.write("• Evidence Checking: ✅ YES")
    
    @staticmethod
    def display_claim_table_with_filters(
        claims: List[LearningClaim]
    ) -> List[LearningClaim]:
        """
        Display sortable/filterable claim table.
        
        Returns:
            Filtered claims list
        """
        if not claims:
            st.info("No claims to display")
            return []
        
        st.write("### Claim Table (Filterable)")
        
        # Filters
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status_filter = st.multiselect(
                "Status",
                ["verified", "low_confidence", "rejected"],
                default=["verified", "low_confidence", "rejected"]
            )
        
        with col2:
            type_filter = st.multiselect(
                "Claim Type",
                ["definition", "equation", "example", "misconception"],
                default=["definition", "equation", "example", "misconception"]
            )
        
        with col3:
            conf_range = st.slider(
                "Confidence Range",
                0.0, 1.0, (0.0, 1.0),
                step=0.05
            )
        
        with col4:
            search_text = st.text_input("Search claim text")
        
        # Apply filters
        filtered = []
        for c in claims:
            status_val = getattr(c.status, "value", "").lower()
            claim_type_val = getattr(c.claim_type, "value", "").lower()
            
            # Status filter
            if status_val not in status_filter:
                continue
            
            # Type filter
            if claim_type_val not in type_filter:
                continue
            
            # Confidence filter
            if not (conf_range[0] <= c.confidence <= conf_range[1]):
                continue
            
            # Search filter
            display_text = (
                c.metadata.get("ui_display", "") or 
                c.claim_text or 
                c.metadata.get("draft_text", "")
            ).lower()
            if search_text and search_text.lower() not in display_text:
                continue
            
            filtered.append(c)
        
        # Display table
        table_data = []
        for c in filtered:
            status_val = getattr(c.status, "value", "").replace("_", " ").title()
            claim_type_val = getattr(c.claim_type, "value", "").replace("_", " ").title()
            display_text = (
                c.metadata.get("ui_display", "") or 
                c.claim_text or 
                c.metadata.get("draft_text", "[No text]")
            )[:80]
            rejection_reason = c.rejection_reason.value if c.rejection_reason else ""
            
            table_data.append({
                "ID": c.claim_id[:6],
                "Type": claim_type_val,
                "Status": status_val,
                "Confidence": f"{c.confidence:.2f}",
                "Evidence": len(c.evidence_ids),
                "Rejection": rejection_reason,
                "Text": display_text
            })
        
        if table_data:
            df = pd.DataFrame(table_data)
            has_pyarrow = True
            try:
                import pyarrow  # noqa: F401
            except Exception:
                has_pyarrow = False

            if has_pyarrow:
                try:
                    st.dataframe(
                        df,
                        use_container_width=True,
                        height=400
                    )
                except Exception:
                    st.warning("Dataframe rendering failed. Falling back to table-lite.")
                    try:
                        st.table(df.head(50))
                    except Exception:
                        st.json(table_data[:50])
            else:
                st.warning("pyarrow missing; install with pip install pyarrow. Falling back to table-lite.")
                try:
                    st.table(df.head(50))
                except Exception:
                    st.json(table_data[:50])
        else:
            st.info("No claims match filter criteria")
        
        return filtered
    
    @staticmethod
    def display_claim_drill_down(claim: LearningClaim) -> None:
        """
        Display drill-down panel with tabs for single claim.
        
        Tabs:
        - Overview: status, confidence, rejection reason
        - Evidence: evidence snippets with metadata
        - Assessment: natural language explanation
        
        Args:
            claim: Claim to display
        """
        with st.expander(f"🔍 Claim Details: {claim.claim_id[:8]}", expanded=False):
            tabs = st.tabs(["Overview", "Evidence", "Assessment"])
            
            # Tab 1: Overview
            with tabs[0]:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Status", getattr(claim.status, "value", "").replace("_", " ").title())
                with col2:
                    st.metric("Confidence", f"{claim.confidence:.2f}")
                with col3:
                    st.metric("Evidence Count", len(claim.evidence_ids))
                
                st.write(f"**Type**: {getattr(claim.claim_type, 'value', '').replace('_', ' ').title()}")
                
                display_text = (
                    claim.metadata.get("ui_display", "") or 
                    claim.claim_text or 
                    claim.metadata.get("draft_text", "[No claim text]")
                )
                st.write(f"**Claim Text**:\n{display_text}")
                
                if claim.rejection_reason:
                    st.warning(f"**Rejection Reason**: {claim.rejection_reason.value}")
                
                if claim.metadata.get("dependency_flag"):
                    st.info("ℹ️ This claim has dependencies on other concepts")
            
            # Tab 2: Evidence
            with tabs[1]:
                if claim.evidence_objects:
                    for i, evidence in enumerate(claim.evidence_objects, 1):
                        source_type = evidence.source_type or "unknown"
                        source_id = evidence.source_id or "N/A"
                        similarity = getattr(evidence, "similarity", 0.0)
                        snippet = evidence.snippet or ""
                        span_meta = getattr(evidence, "span_metadata", {}) or {}
                        location = span_meta.get("location") or span_meta.get("line") or "N/A"
                        
                        with st.expander(
                            f"📄 Source {i}: {source_type} (similarity: {similarity:.0%})",
                            expanded=(i == 1)
                        ):
                            st.write(f"**Snippet**:\n{snippet}")
                            st.caption(
                                f"Source: {source_id} | Location: {location} | "
                                f"Similarity: {similarity:.2f}"
                            )
                else:
                    st.warning("⚠️ No evidence attached to this claim")
            
            # Tab 3: Assessment
            with tabs[2]:
                status_val = getattr(claim.status, "value", "").lower()
                
                if status_val == "verified":
                    st.success(
                        f"✅ **VERIFIED**\n\n"
                        f"This claim is supported by {len(claim.evidence_ids)} evidence source(s) "
                        f"with confidence {claim.confidence:.0%}."
                    )
                elif status_val == "low_confidence":
                    st.warning(
                        f"⚠️ **LOW CONFIDENCE**\n\n"
                        f"This claim has limited evidence support ({len(claim.evidence_ids)} source(s), "
                        f"confidence {claim.confidence:.0%}). Use with caution."
                    )
                else:
                    st.error(
                        f"❌ **REJECTED**\n\n"
                        f"Reason: {claim.rejection_reason.value if claim.rejection_reason else 'Insufficient evidence'}\n\n"
                        f"This claim was rejected because it lacks sufficient evidence in source materials. "
                        f"This is correct behavior preventing hallucination."
                    )
    
    @staticmethod
    def create_claims_csv(claims: List[LearningClaim]) -> str:
        """
        Export claims to CSV format.
        
        Columns: claim_id, type, status, confidence, evidence_count, 
                 rejection_reason, claim_text, dependency_flag
        
        Args:
            claims: List of claims
        
        Returns:
            CSV string
        """
        output = StringIO()
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "claim_id", "claim_type", "status", "confidence",
                "evidence_count", "rejection_reason", "claim_text", "dependency_flag"
            ]
        )
        writer.writeheader()
        
        for c in claims:
            writer.writerow({
                "claim_id": c.claim_id,
                "claim_type": getattr(c.claim_type, "value", ""),
                "status": getattr(c.status, "value", ""),
                "confidence": f"{c.confidence:.4f}",
                "evidence_count": len(c.evidence_ids),
                "rejection_reason": c.rejection_reason.value if c.rejection_reason else "",
                "claim_text": c.claim_text or "",
                "dependency_flag": str(c.metadata.get("dependency_flag", False))
            })
        
        return output.getvalue()
    
    @staticmethod
    def create_evidence_csv(claims: List[LearningClaim]) -> str:
        """
        Export evidence to CSV format.
        
        Columns: evidence_id, source_id, source_type, location, snippet,
                 similarity, reliability, linked_claim_ids
        
        Args:
            claims: List of claims with evidence
        
        Returns:
            CSV string
        """
        output = StringIO()
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "evidence_id", "source_id", "source_type", "location",
                "snippet", "similarity", "reliability_prior", "linked_claim_ids"
            ]
        )
        writer.writeheader()
        
        for c in claims:
            if hasattr(c, 'evidence_objects') and c.evidence_objects:
                for e in c.evidence_objects:
                    span_meta = getattr(e, "span_metadata", {}) or {}
                    location = span_meta.get("location") or span_meta.get("line") or "N/A"
                    
                    writer.writerow({
                        "evidence_id": e.evidence_id,
                        "source_id": e.source_id or "",
                        "source_type": e.source_type or "",
                        "location": location,
                        "snippet": (e.snippet or "")[:100],
                        "similarity": f"{getattr(e, 'similarity', 0.0):.4f}",
                        "reliability_prior": f"{getattr(e, 'reliability_prior', 0.8):.4f}",
                        "linked_claim_ids": c.claim_id
                    })
        
        return output.getvalue()
    
    @staticmethod
    def create_claim_authenticity_report(
        claims: List[LearningClaim],
        session_id: str,
        baseline_count: Optional[int] = None,
        metrics: Optional[Dict[str, Any]] = None,
        is_negative_control: bool = False
    ) -> Dict[str, Any]:
        """
        Generate research-grade authenticity report.
        
        Args:
            claims: List of claims
            session_id: Session identifier
            baseline_count: Number of baseline (standard mode) claims
            metrics: Pre-computed metrics (rejection reasons, etc.)
            is_negative_control: True if this is negative control (no evidence)
        
        Returns:
            Report dictionary
        """
        verified = sum(1 for c in claims if getattr(c.status, "value", "") == "verified")
        rejected = sum(1 for c in claims if getattr(c.status, "value", "") == "rejected")
        low_conf = sum(1 for c in claims if getattr(c.status, "value", "") == "low_confidence")
        
        report = {
            "session_id": session_id,
            "timestamp": str(__import__('datetime').datetime.now()),
            "negative_control": is_negative_control,
            
            # Claim statistics
            "total_claims": len(claims),
            "verified_claims": verified,
            "low_confidence_claims": low_conf,
            "rejected_claims": rejected,
            
            # Rates
            "rejection_rate": rejected / len(claims) if claims else 0,
            "verification_rate": verified / len(claims) if claims else 0,
            "traceability_rate": sum(1 for c in claims if c.evidence_ids) / len(claims) if claims else 0,
            
            # Confidence
            "average_confidence": sum(c.confidence for c in claims) / len(claims) if claims else 0,
            
            # Claims breakdown
            "claims_by_type": {
                claim_type: sum(1 for c in claims 
                               if getattr(c.claim_type, "value", "") == claim_type)
                for claim_type in ["definition", "equation", "example", "misconception"]
            },
            
            # Evidence stats
            "evidence_stats": {
                "total_evidence": sum(len(c.evidence_ids) for c in claims),
                "avg_evidence_per_claim": sum(len(c.evidence_ids) for c in claims) / len(claims) if claims else 0,
                "claims_without_evidence": sum(1 for c in claims if len(c.evidence_ids) == 0)
            },
            
            # Confidence distribution
            "confidence_distribution": {
                "high_08_plus": sum(1 for c in claims if c.confidence >= 0.8),
                "moderate_06_to_08": sum(1 for c in claims if 0.6 <= c.confidence < 0.8),
                "low_below_06": sum(1 for c in claims if c.confidence < 0.6)
            },
            
            # Rejection reasons
            "rejection_reasons": {
                reason: sum(1 for c in claims if c.rejection_reason and c.rejection_reason.value == reason)
                for reason in ["NO_EVIDENCE", "LOW_SIMILARITY", "INSUFFICIENT_SOURCES", 
                              "LOW_CONSISTENCY", "CONFLICT", "INSUFFICIENT_CONFIDENCE"]
            },
            
            # Baseline comparison
            "baseline_comparison": {
                "baseline_items": baseline_count or 0,
                "verifiable_verified": verified,
                "hallucination_reduction_estimate": (rejected / baseline_count) if baseline_count else 0
            } if baseline_count else None,
            
            # Pre-computed metrics
            "metrics": metrics or {}
        }
        
        return report
