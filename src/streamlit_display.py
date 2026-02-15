"""
Streamlit-specific streaming handler for real-time output display.
"""

import streamlit as st
from typing import Dict, Any, Optional


class StreamlitProgressDisplay:
    """Display processing progress and results in real-time on Streamlit."""
    
    def __init__(self):
        self.stages = [
            ("summary", "🧠 Analyzing content", "Summary"),
            ("topics", "📚 Identifying topics", "Topics"),
            ("concepts", "💡 Extracting concepts", "Concepts"),
            ("equations", "📐 Finding equations", "Equations"),
            ("misconceptions", "⚠️ Identifying misconceptions", "Misconceptions"),
            ("faqs", "❓ Generating FAQ", "FAQs"),
            ("worked_examples", "🎯 Creating examples", "Worked Examples"),
        ]
        
        # Create placeholder containers
        self.progress_container = None
        self.results_container = None
        self.status_display = {}
    
    def setup_display(self):
        """Setup progress and results display areas."""
        # Progress/status area
        self.progress_container = st.container()
        st.divider()
        # Results area
        self.results_container = st.container()
    
    def update_stage_status(self, stage: str, status: str = "processing"):
        """Update status for a stage."""
        with self.progress_container:
            for stage_key, emoji, label in self.stages:
                if stage_key == stage:
                    if status == "complete":
                        st.success(f"✅ {emoji} {label}")
                    elif status == "error":
                        st.error(f"❌ {emoji} {label} - No data")
                    else:
                        st.info(f"⏳ {emoji} {label}...")
    
    def display_streaming_output(self, output_data: Dict[str, Any]):
        """Display output progressively in the results container."""
        with self.results_container:
            # Summary (usually generated first)
            if output_data.get("summary"):
                with st.expander("📝 Summary", expanded=True):
                    st.markdown(output_data["summary"])
            
            # Topics
            if output_data.get("topics"):
                with st.expander(f"📚 Main Topics ({len(output_data['topics'])})", expanded=True):
                    for i, topic in enumerate(output_data["topics"], 1):
                        st.write(f"**{i}. {topic.get('title', 'Topic')}**")
                        st.write(topic.get("description", ""))
            
            # Concepts
            if output_data.get("concepts"):
                with st.expander(f"💡 Key Concepts ({len(output_data['concepts'])})", expanded=False):
                    for i, concept in enumerate(output_data["concepts"], 1):
                        st.write(f"**{i}. {concept.get('name', 'Concept')}**")
                        st.write(f"*{concept.get('definition', '')}*")
                        if concept.get("importance"):
                            st.caption(f"Importance: {concept['importance']}")
            
            # Equations
            if output_data.get("equations"):
                with st.expander(f"📐 Key Equations ({len(output_data['equations'])})", expanded=False):
                    for i, eq in enumerate(output_data["equations"], 1):
                        st.write(f"**Equation {i}:**")
                        st.code(eq.get("equation", ""), language="text")
                        if eq.get("explanation"):
                            st.write(eq["explanation"])
            
            # Misconceptions
            if output_data.get("misconceptions"):
                with st.expander(f"⚠️ Common Misconceptions ({len(output_data['misconceptions'])})", expanded=False):
                    for i, misc in enumerate(output_data["misconceptions"], 1):
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            st.error(f"❌ **Wrong:** {misc.get('misconception', '')}")
                        with col2:
                            st.success(f"✅ **Correct:** {misc.get('correct_understanding', '')}")
            
            # FAQs
            if output_data.get("faqs"):
                with st.expander(f"❓ Common Questions ({len(output_data['faqs'])})", expanded=False):
                    for i, faq in enumerate(output_data["faqs"], 1):
                        difficulty = faq.get("difficulty", "medium")
                        emoji = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}.get(difficulty, "⚪")
                        st.write(f"**Q{i} [{emoji}]: {faq.get('question', '')}**")
                        st.write(f"*{faq.get('answer', '')}*")
            
            # Worked Examples
            if output_data.get("worked_examples"):
                with st.expander(f"🎯 Worked Examples ({len(output_data['worked_examples'])})", expanded=False):
                    for i, example in enumerate(output_data["worked_examples"], 1):
                        st.write(f"**Example {i}: {example.get('problem', '')}**")
                        if example.get("solution"):
                            st.code(example["solution"], language="text")
                        if example.get("explanation"):
                            st.write(f"*Why: {example['explanation']}*")
    
    def show_empty_field_warning(self, missing_fields: list):
        """Show warning for missing/empty fields."""
        if missing_fields:
            st.warning(
                f"⚠️ Some sections are minimal: {', '.join(missing_fields)}\n\n"
                "Try:\n"
                "- Adding more detailed notes\n"
                "- Including specific concepts or questions\n"
                "- Uploading more images with content"
            )


class QuickExportButtons:
    """Quick export buttons for students."""
    
    @staticmethod
    def show_export_buttons(output_data: Dict[str, Any], session_id: str):
        """Show export options."""
        st.divider()
        st.subheader("📥 Export")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            import json
            json_str = json.dumps(output_data, indent=2, ensure_ascii=False, default=str)
            st.download_button(
                label="📋 Download JSON",
                data=json_str,
                file_name=f"notes_{session_id}.json",
                mime="application/json"
            )
        
        with col2:
            md_str = QuickExportButtons._to_markdown(output_data)
            st.download_button(
                label="📝 Download Markdown",
                data=md_str,
                file_name=f"notes_{session_id}.md",
                mime="text/markdown"
            )
        
        with col3:
            if st.button("💾 Save Session"):
                st.success(f"✓ Session saved as {session_id}")
    
    @staticmethod
    def _to_markdown(data: Dict[str, Any]) -> str:
        """Convert output to markdown format."""
        md = "# Study Notes\n\n"
        
        summary = data.get("summary") or data.get("class_summary")
        if summary:
            md += f"## Summary\n\n{summary}\n\n"
        
        topics = data.get("topics") or []
        if topics:
            md += "## Topics\n\n"
            for topic in topics:
                title = topic.get("title") or topic.get("name") or ""
                desc = topic.get("description") or topic.get("summary") or ""
                md += f"- **{title}**: {desc}\n"
            md += "\n"
        
        key_concepts = data.get("key_concepts") or data.get("concepts") or []
        if key_concepts:
            md += "## Key Concepts\n\n"
            for concept in key_concepts:
                name = concept.get("concept_name") or concept.get("name") or ""
                definition = concept.get("definition") or ""
                md += f"### {name}\n\n"
                md += f"{definition}\n\n"
                example = concept.get("example") or concept.get("examples")
                if example:
                    md += f"**Example**: {example}\n\n"
        
        equations = data.get("equation_explanations") or data.get("equations") or []
        if equations:
            md += "## Equations\n\n"
            for eq in equations:
                md += f"### {eq.get('equation', '')}\n\n"
                md += f"{eq.get('explanation', '')}\n\n"
                if eq.get('application'):
                    md += f"**Application**: {eq['application']}\n\n"
        
        worked_examples = data.get("worked_examples") or data.get("examples") or []
        if worked_examples:
            md += "## Worked Examples\n\n"
            for i, example in enumerate(worked_examples, 1):
                problem = example.get("problem_statement") or example.get("problem") or ""
                md += f"### Example {i}: {problem}\n\n"
                solution_steps = example.get("solution_steps")
                if solution_steps:
                    md += "**Solution Steps**:\n\n"
                    for j, step in enumerate(solution_steps, 1):
                        md += f"{j}. {step}\n"
                    md += "\n"
                solution = example.get("solution")
                if solution and not solution_steps:
                    md += f"**Solution**: {solution}\n\n"
                key_insight = example.get("key_insight") or example.get("explanation")
                if key_insight:
                    md += f"**Key Insight**: {key_insight}\n\n"
        
        common_mistakes = data.get("common_mistakes") or data.get("misconceptions") or []
        if common_mistakes:
            md += "## Common Mistakes\n\n"
            for mistake in common_mistakes:
                label = mistake.get("mistake") or mistake.get("misconception") or ""
                md += f"### ❌ {label}\n\n"
                explanation = mistake.get("explanation") or ""
                correct = mistake.get("correction") or mistake.get("correct_understanding")
                if explanation:
                    md += f"{explanation}\n\n"
                if correct:
                    md += f"**✓ Correction**: {correct}\n\n"
        
        if data.get("faqs"):
            md += "## FAQs\n\n"
            for faq in data["faqs"]:
                md += f"**Q: {faq.get('question', '')}**\n\n"
                md += f"A: {faq.get('answer', '')}\n\n"
        
        connections = data.get("real_world_connections") or data.get("connections") or []
        if connections:
            md += "## Real-World Connections\n\n"
            for conn in connections:
                field = conn.get("field") or conn.get("connection") or ""
                application = conn.get("application") or conn.get("relevance") or ""
                md += f"### {field}\n\n"
                md += f"{application}\n\n"
                if conn.get('example'):
                    md += f"**Example**: {conn['example']}\n\n"
        
        return md
