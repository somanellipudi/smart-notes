"""
Streaming output handler for real-time results display in Streamlit.
Shows results as they're being generated instead of waiting for completion.
"""

import streamlit as st
import json
from typing import Dict, Any, Generator, Optional

class OutputFormatter:
    """Format ClassSessionOutput to various display formats."""
    
    def format_to_markdown(self, output) -> str:
        """Convert ClassSessionOutput to markdown format.
        
        Args:
            output: ClassSessionOutput object with session results
            
        Returns:
            Formatted markdown string
        """
        md = []
        
        # Session info
        if hasattr(output, 'session_id'):
            md.append(f"# Study Notes - {output.session_id}\n")
        else:
            md.append("# Study Notes\n")
        
        # Summary
        if hasattr(output, 'class_summary') and output.class_summary:
            md.append("## 📋 Summary\n")
            md.append(f"{output.class_summary}\n")
        
        # Topics
        if hasattr(output, 'topics') and output.topics:
            md.append(f"\n## 📚 Topics ({len(output.topics)})\n")
            for i, topic in enumerate(output.topics, 1):
                name = topic.name if hasattr(topic, 'name') else str(topic)
                md.append(f"\n### {i}. {name}\n")
                if hasattr(topic, 'summary'):
                    md.append(f"{topic.summary}\n")
        
        # Concepts
        if hasattr(output, 'key_concepts') and output.key_concepts:
            md.append(f"\n## 💡 Key Concepts ({len(output.key_concepts)})\n")
            for i, concept in enumerate(output.key_concepts, 1):
                name = concept.name if hasattr(concept, 'name') else str(concept)
                md.append(f"\n### {i}. {name}\n")
                if hasattr(concept, 'definition'):
                    md.append(f"{concept.definition}\n")
        
        # Equations
        if hasattr(output, 'equation_explanations') and output.equation_explanations:
            md.append(f"\n## 📐 Equations ({len(output.equation_explanations)})\n")
            for i, eq in enumerate(output.equation_explanations, 1):
                equation = eq.equation if hasattr(eq, 'equation') else str(eq)
                md.append(f"\n### Equation {i}: {equation}\n")
                if hasattr(eq, 'explanation'):
                    md.append(f"{eq.explanation}\n")
        
        # Worked Examples
        if hasattr(output, 'worked_examples') and output.worked_examples:
            md.append(f"\n## 🎯 Worked Examples ({len(output.worked_examples)})\n")
            for i, ex in enumerate(output.worked_examples, 1):
                md.append(f"\n### Example {i}\n")
                if hasattr(ex, 'problem'):
                    md.append(f"**Problem:** {ex.problem}\n")
                if hasattr(ex, 'solution'):
                    md.append(f"\n**Solution:** {ex.solution}\n")
        
        # FAQs
        if hasattr(output, 'faqs') and output.faqs:
            md.append(f"\n## ❓ FAQs ({len(output.faqs)})\n")
            for i, faq in enumerate(output.faqs, 1):
                question = faq.question if hasattr(faq, 'question') else str(faq)
                md.append(f"\n### Q{i}: {question}\n")
                if hasattr(faq, 'answer'):
                    md.append(f"**A:** {faq.answer}\n")
        
        return "\n".join(md)

class StudentFriendlyFormatter:
    """Format output in a way that's intuitive for students."""
    
    @staticmethod
    def format_topics(topics: list) -> str:
        """Format topics for student view."""
        if not topics:
            return "No main topics identified yet. Try adding more detailed notes."
        
        formatted = "## 📚 Main Topics\n\n"
        for i, topic in enumerate(topics, 1):
            title = topic.get("title", "Unknown Topic")
            description = topic.get("description", "")
            
            formatted += f"### {i}. {title}\n"
            if description:
                formatted += f"{description}\n"
            formatted += "\n"
        
        return formatted
    
    @staticmethod
    def format_concepts(concepts: list) -> str:
        """Format concepts for student view."""
        if not concepts:
            return "No key concepts identified yet. Check your notes!"
        
        formatted = "## 💡 Key Concepts\n\n"
        for i, concept in enumerate(concepts, 1):
            name = concept.get("name", "Concept")
            definition = concept.get("definition", "")
            importance = concept.get("importance", "medium")
            
            formatted += f"### {i}. {name}\n"
            formatted += f"**Importance:** {importance}\n\n"
            if definition:
                formatted += f"**Definition:** {definition}\n"
            formatted += "\n"
        
        return formatted
    
    @staticmethod
    def format_faqs(faqs: list) -> str:
        """Format FAQs for student view."""
        if not faqs:
            return "No common questions identified yet."
        
        formatted = "## ❓ Common Questions\n\n"
        for i, faq in enumerate(faqs, 1):
            question = faq.get("question", "?")
            answer = faq.get("answer", "")
            
            formatted += f"### Q{i}: {question}\n"
            if answer:
                formatted += f"**A:** {answer}\n"
            formatted += "\n"
        
        return formatted
    
    @staticmethod
    def format_equations(equations: list) -> str:
        """Format equations for student view."""
        if not equations:
            return "No equations found in your notes."
        
        formatted = "## 📐 Key Equations\n\n"
        for i, eq in enumerate(equations, 1):
            equation = eq.get("equation", "")
            explanation = eq.get("explanation", "")
            
            formatted += f"### Equation {i}\n"
            formatted += f"```\n{equation}\n```\n"
            if explanation:
                formatted += f"**Explanation:** {explanation}\n"
            formatted += "\n"
        
        return formatted
    
    @staticmethod
    def format_examples(examples: list) -> str:
        """Format worked examples for student view."""
        if not examples:
            return "No worked examples generated yet."
        
        formatted = "## 🎯 Worked Examples\n\n"
        for i, example in enumerate(examples, 1):
            problem = example.get("problem", "")
            solution = example.get("solution", "")
            explanation = example.get("explanation", "")
            
            formatted += f"### Example {i}\n"
            formatted += f"**Problem:** {problem}\n\n"
            if solution:
                formatted += f"**Solution:**\n```\n{solution}\n```\n\n"
            if explanation:
                formatted += f"**Why this works:** {explanation}\n"
            formatted += "\n"
        
        return formatted
    
    @staticmethod
    def format_misconceptions(misconceptions: list) -> str:
        """Format misconceptions for student view."""
        if not misconceptions:
            return "No common misconceptions identified."
        
        formatted = "## ⚠️ Common Misconceptions\n\n"
        for i, misc in enumerate(misconceptions, 1):
            misconception = misc.get("misconception", "")
            correction = misc.get("correction", "")
            
            formatted += f"### Misconception {i}\n"
            formatted += f"❌ **Wrong:** {misconception}\n\n"
            if correction:
                formatted += f"✅ **Correct:** {correction}\n"
            formatted += "\n"
        
        return formatted
    
    @staticmethod
    def format_summary(summary: str) -> str:
        """Format executive summary for student view."""
        if not summary:
            return "Summary not available yet."
        
        return f"## 📝 Summary\n\n{summary}\n"


class StreamingOutputDisplay:
    """Handle real-time streaming output display in Streamlit."""
    
    def __init__(self):
        self.formatter = StudentFriendlyFormatter()
    
    def display_streaming_result(
        self,
        output_data: Dict[str, Any],
        container: Optional[Any] = None
    ):
        """
        Display results in a student-friendly format.
        
        Args:
            output_data: Generated output data
            container: Streamlit container (if None, uses st)
        """
        if container is None:
            container = st
        
        display = container.container()
        
        # Summary
        if output_data.get("summary"):
            with display.expander("📝 Summary", expanded=True):
                display.markdown(self.formatter.format_summary(output_data["summary"]))
        
        # Topics
        if output_data.get("topics"):
            with display.expander("📚 Main Topics", expanded=True):
                display.markdown(self.formatter.format_topics(output_data["topics"]))
        
        # Concepts
        if output_data.get("concepts"):
            with display.expander("💡 Key Concepts", expanded=True):
                display.markdown(self.formatter.format_concepts(output_data["concepts"]))
        
        # FAQs
        if output_data.get("faqs"):
            with display.expander("❓ Common Questions"):
                display.markdown(self.formatter.format_faqs(output_data["faqs"]))
        
        # Equations
        if output_data.get("equations"):
            with display.expander("📐 Key Equations"):
                display.markdown(self.formatter.format_equations(output_data["equations"]))
        
        # Worked Examples
        if output_data.get("worked_examples"):
            with display.expander("🎯 Worked Examples"):
                display.markdown(self.formatter.format_examples(output_data["worked_examples"]))
        
        # Misconceptions
        if output_data.get("misconceptions"):
            with display.expander("⚠️ Common Misconceptions"):
                display.markdown(self.formatter.format_misconceptions(output_data["misconceptions"]))
    
    def display_summary_only(self, summary: str, container: Optional[Any] = None):
        """Display just the summary while processing."""
        if container is None:
            container = st
        
        if summary:
            container.markdown("## 📝 Summary\n\n" + summary)
    
    def display_progress(self, stage: str, status: str, container: Optional[Any] = None):
        """Display processing progress."""
        if container is None:
            container = st
        
        stages = {
            "summary": "🧠 Analyzing content",
            "topics": "📚 Identifying topics",
            "concepts": "💡 Extracting concepts",
            "equations": "📐 Finding equations",
            "examples": "🎯 Creating examples",
            "misconceptions": "⚠️ Identifying misconceptions",
            "faqs": "❓ Generating FAQs",
        }
        
        display_text = stages.get(stage, stage)
        container.info(f"{display_text}... {status}")
