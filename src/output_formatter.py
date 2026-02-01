"""
Streaming output handler for real-time results display in Streamlit.
Shows results as they're being generated instead of waiting for completion.
"""

import streamlit as st
import json
from typing import Dict, Any, Generator, Optional


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
