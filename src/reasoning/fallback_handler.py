"""
Enhanced streaming and fallback logic for the reasoning pipeline.
Handles empty outputs with intelligent retries and generates default content.
"""

import logging
from typing import Dict, List, Any, Optional, Callable

logger = logging.getLogger(__name__)


class FallbackGenerator:
    """Generate sensible defaults when LLM output is empty."""
    
    @staticmethod
    def generate_default_topics(text: str, count: int = 3) -> List[Dict[str, Any]]:
        """Generate default topics from text content."""
        # Extract potential topics by looking for headers or key phrases
        topics = []
        sentences = text.split('.')[:count]
        
        for i, sentence in enumerate(sentences, 1):
            clean_sent = sentence.strip()
            if len(clean_sent) > 10:
                title = clean_sent[:50] + "..." if len(clean_sent) > 50 else clean_sent
                topics.append({
                    "title": title,
                    "description": clean_sent,
                    "importance": "high" if i == 1 else "medium"
                })
        
        return topics if topics else [{
            "title": "Main Topic",
            "description": "Core content from the notes",
            "importance": "high"
        }]
    
    @staticmethod
    def generate_default_concepts(topics: List[Dict[str, Any]], count: int = 5) -> List[Dict[str, Any]]:
        """Generate default concepts from topics."""
        concepts = []
        for topic in topics[:3]:
            title = topic.get("title", "Concept")[:40]
            concepts.append({
                "name": f"Key Concept: {title}",
                "definition": topic.get("description", "Important concept related to the topic"),
                "importance": "high",
                "related_topics": [topic.get("title", "")]
            })
        
        return concepts
    
    @staticmethod
    def generate_default_faqs(text: str, count: int = 3) -> List[Dict[str, Any]]:
        """Generate default FAQs from content."""
        faqs = []
        
        questions = [
            "What is the main idea?",
            "Why is this important?",
            "How does this connect to previous concepts?",
            "What are the key applications?",
            "How can I remember this?"
        ]
        
        for i, question in enumerate(questions[:count]):
            faqs.append({
                "question": question,
                "answer": "Refer to your notes for detailed explanation. Key points are highlighted above.",
                "difficulty": ["easy", "medium", "hard"][i % 3],
                "related_concepts": []
            })
        
        return faqs
    
    @staticmethod
    def generate_default_examples(text: str, count: int = 2) -> List[Dict[str, Any]]:
        """Generate placeholder examples."""
        examples = []
        
        for i in range(count):
            examples.append({
                "problem": f"Example {i+1}: Apply the concepts from the notes",
                "solution": "Work through this step by step using the principles described",
                "explanation": "This demonstrates the key concepts from your notes",
                "key_concepts": ["Main concept"],
                "common_mistakes": ["Common mistake: Not reading the problem carefully"]
            })
        
        return examples


class StreamingPipeline:
    """Handle streaming output with progressive generation."""
    
    def __init__(self, callback: Optional[Callable] = None):
        """
        Initialize streaming pipeline.
        
        Args:
            callback: Optional callback to receive streaming updates
        """
        self.callback = callback
        self.output_data = {}
    
    def emit(self, stage: str, data: Any):
        """Emit streaming update."""
        self.output_data[stage] = data
        
        if self.callback:
            self.callback(stage, data)
        
        logger.info(f"✓ Generated {stage}: {len(data) if isinstance(data, list) else 'content'} items")
    
    def emit_summary(self, summary: str):
        """Emit summary (generated first)."""
        self.output_data["summary"] = summary
        
        if self.callback:
            self.callback("summary", summary)
        
        logger.info(f"✓ Generated summary: {len(summary)} characters")


class PipelineEnhancer:
    """Enhance pipeline with fallback and retry logic."""
    
    @staticmethod
    def ensure_minimum_output(
        output_data: Dict[str, Any],
        source_text: str,
        fallback: FallbackGenerator = None
    ) -> Dict[str, Any]:
        """
        Ensure output has minimum viable content using fallbacks.
        
        Args:
            output_data: Generated output (may have empty fields)
            source_text: Original input text for fallback generation
            fallback: Fallback generator instance
        
        Returns:
            Enhanced output with populated fields
        """
        if fallback is None:
            fallback = FallbackGenerator()
        
        # Ensure topics
        if not output_data.get("topics") or len(output_data["topics"]) == 0:
            logger.warning("Topics empty, using fallback generation")
            output_data["topics"] = fallback.generate_default_topics(source_text)
        
        # Ensure concepts
        if not output_data.get("concepts") or len(output_data["concepts"]) == 0:
            logger.warning("Concepts empty, generating from topics")
            output_data["concepts"] = fallback.generate_default_concepts(
                output_data.get("topics", [])
            )
        
        # Ensure FAQs
        if not output_data.get("faqs") or len(output_data["faqs"]) == 0:
            logger.warning("FAQs empty, generating defaults")
            output_data["faqs"] = fallback.generate_default_faqs(source_text)
        
        # Ensure worked examples
        if not output_data.get("worked_examples") or len(output_data["worked_examples"]) == 0:
            logger.warning("Worked examples empty, generating placeholders")
            output_data["worked_examples"] = fallback.generate_default_examples(source_text)
        
        # Ensure misconceptions
        if not output_data.get("misconceptions") or len(output_data["misconceptions"]) == 0:
            logger.warning("Misconceptions empty, using minimal set")
            output_data["misconceptions"] = [{
                "misconception": "The content is just memorization",
                "explanation": "Understanding the concepts is more important than memorization",
                "correct_understanding": "Focus on understanding relationships and applications",
                "related_concepts": []
            }]
        
        return output_data
    
    @staticmethod
    def add_retry_context(
        previous_output: Dict[str, Any],
        summary: str
    ) -> str:
        """
        Create enhanced context for retry generation using successful summary.
        
        Args:
            previous_output: Output from first pass (may have empty fields)
            summary: Successfully generated summary
        
        Returns:
            Enhanced prompt context
        """
        context = f"""
You have already generated this summary successfully:

SUMMARY:
{summary}

Now, based on this summary and the original content, extract detailed information for:
- Topics: Main topics covered
- Concepts: Key concepts and definitions
- FAQs: Common questions students ask
- Worked Examples: Practical examples
- Misconceptions: Common student mistakes

Ensure EVERY field has detailed, meaningful content.
"""
        return context


class ParallelProcessingOptimizer:
    """Optimize processing with parallelization."""
    
    @staticmethod
    def get_optimized_stages(depth: str = "Balanced") -> Dict[str, bool]:
        """
        Get which stages to process based on depth setting.
        
        Args:
            depth: "Fast", "Balanced", or "Thorough"
        
        Returns:
            Dictionary mapping stage names to boolean (process or skip)
        """
        stages = {
            "summary": True,
            "topics": True,
            "concepts": True,
            "equations": True,
            "misconceptions": True,
            "faqs": True,
            "worked_examples": True,
            "real_world_connections": True,
        }
        
        if depth == "Fast":
            # Skip slow stages
            stages["real_world_connections"] = False
            stages["worked_examples"] = False
            return stages
        
        elif depth == "Thorough":
            # Process everything
            return stages
        
        else:  # Balanced (default)
            return stages
    
    @staticmethod
    def get_prompt_optimization(model_type: str = "ollama") -> Dict[str, str]:
        """
        Get optimized prompts based on LLM type.
        
        Args:
            model_type: "openai" or "ollama"
        
        Returns:
            Dictionary of optimized prompts
        """
        if model_type == "ollama":
            # Shorter, more direct prompts for local LLMs
            return {
                "topics": "List 3-5 main topics in JSON format. Be concise.",
                "concepts": "Extract 5 key concepts with definitions in JSON format.",
                "faqs": "Generate 5 FAQ pairs in JSON format.",
                "examples": "Create 2-3 worked examples in JSON format.",
            }
        
        else:  # OpenAI
            # Can use longer, more detailed prompts
            return {
                "topics": "Identify and describe all major topics covered",
                "concepts": "Extract comprehensive key concepts with detailed definitions",
                "faqs": "Generate comprehensive FAQs addressing common student questions",
                "examples": "Create detailed worked examples with full explanations",
            }
