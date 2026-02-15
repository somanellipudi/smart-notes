"""
Multi-stage GenAI reasoning pipeline for educational content extraction.

This module implements a sophisticated reasoning pipeline that processes classroom
content through multiple specialized stages, NOT using a single monolithic prompt.
Each stage focuses on a specific aspect of educational knowledge extraction.
"""

import logging
import json
import re
from typing import Dict, List, Any, Optional
from datetime import datetime

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

import config
from src.llm_provider import LLMProviderFactory
from src.schema.output_schema import (
    ClassSessionOutput,
    Topic,
    Concept,
    WorkedExample,
    EquationExplanation,
    FAQ,
    Misconception,
    RealWorldConnection
)

logging.basicConfig(level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)


class ReasoningPipeline:
    """
    Multi-stage reasoning pipeline for educational content extraction.
    
    This pipeline implements a research-focused approach where each reasoning
    stage is independent and specialized, enabling fine-grained control and
    evaluation of each component.
    
    Pipeline stages:
        1. Topic Identification
        2. Concept Extraction
        3. Equation Interpretation
        4. Misconception Detection
        5. FAQ Generation
        6. Real-World Connections
        7. Worked Example Analysis
    """
    
    def __init__(
        self,
        model: str = None,
        temperature: float = None,
        provider_type: str = "openai",
        api_key: str = None,
        ollama_url: str = None
    ):
        """
        Initialize the reasoning pipeline.
        
        Args:
            model: LLM model identifier (default from config)
            temperature: Sampling temperature (default from config)
        """
        self.model = model or config.LLM_MODEL
        self.temperature = temperature or config.LLM_TEMPERATURE
        self.provider_type = provider_type or "openai"
        self.provider = None
        
        if self.provider_type == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI SDK not installed. Install with: pip install openai")
            api_key = api_key or config.OPENAI_API_KEY
            if not api_key:
                raise ValueError("OpenAI API key not set. Please set OPENAI_API_KEY in .env")
        
        self.provider = LLMProviderFactory.create_provider(
            provider_type=self.provider_type,
            api_key=api_key or config.OPENAI_API_KEY,
            ollama_url=ollama_url or config.OLLAMA_URL,
            model=self.model
        )
        
        logger.info(
            f"ReasoningPipeline initialized: provider={self.provider_type}, "
            f"model={self.model}, temperature={self.temperature}"
        )
    
    def _call_llm(self, prompt: str, system_prompt: str = None, response_format: Dict[str, Any] = None) -> str:
        """
        Call LLM with given prompt.
        
        Args:
            prompt: User prompt
            system_prompt: System instruction
        
        Returns:
            LLM response text
        """
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.provider.chat_completion(
                messages=messages,
                temperature=self.temperature,
                max_tokens=2000,
                response_format=response_format
            )
            
            return response.strip()
        
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise

    def _load_json_list(self, response: str, label: str) -> List[Dict[str, Any]]:
        """Parse a JSON list from model output, tolerating extra text."""
        def _ensure_list(data: Any) -> List[Dict[str, Any]]:
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                for key in [
                    "topics",
                    "concepts",
                    "equations",
                    "misconceptions",
                    "faqs",
                    "connections",
                    "examples",
                    "items",
                ]:
                    if key in data and isinstance(data[key], list):
                        return data[key]
                return [data]
            return []

        try:
            return _ensure_list(json.loads(response))
        except Exception:
            try:
                match = re.search(r"\[.*\]", response, re.S)
                if match:
                    return _ensure_list(json.loads(match.group(0)))
            except Exception:
                pass

        logger.error(f"Failed to parse {label}: not valid JSON list")
        return []

    def _call_llm_json_list(self, user_prompt: str, system_prompt: str, label: str) -> List[Dict[str, Any]]:
        """Call LLM and ensure JSON list output, retrying with strict JSON-only prompt if needed."""
        response = self._call_llm(user_prompt, system_prompt)
        data = self._load_json_list(response, label)
        if data:
            return data

        strict_system = (
            "Return ONLY valid JSON. No markdown, no prose, no commentary. "
            "If you cannot comply, return an empty JSON array []."
        )
        strict_prompt = f"""{user_prompt}

IMPORTANT:
Return ONLY a JSON array. Do not include any extra text.
"""
        response = self._call_llm(strict_prompt, strict_system)
        data = self._load_json_list(response, label)
        if data:
            return data

        # Final attempt: enforce JSON object with response_format
        json_system = "Return a JSON object only. No extra text."
        json_prompt = f"""{user_prompt}

Return a JSON object with a single key \"items\" that contains the array.
Example: {{\"items\": [{{...}}, {{...}}]}}"""
        response = self._call_llm(
            json_prompt,
            json_system,
            response_format={"type": "json_object"}
        )
        return self._load_json_list(response, label)
    
    def stage1_identify_topics(self, content: str, external_context: str = "") -> List[Topic]:
        """
        Stage 1: Identify major topics covered in the class.
        
        Args:
            content: Combined classroom content
            external_context: Optional textbook/reference context
        
        Returns:
            List of Topic objects
        """
        logger.info("Stage 1: Topic Identification")
        
        system_prompt = (
            "You are an expert educational content analyzer. "
            "Identify the major topics covered in this classroom content. "
            "Focus on high-level themes and learning areas."
        )
        
        user_prompt = f"""
Analyze this classroom content and identify the major topics covered.

CONTENT:
{content[:2000]}

EXTERNAL CONTEXT (if available):
{external_context[:500] if external_context else "N/A"}

Return a JSON array of topics with the following structure:
[
  {{
    "name": "Topic Name",
    "summary": "1-2 sentence overview",
    "subtopics": ["subtopic1", "subtopic2"],
    "learning_objectives": ["objective1", "objective2"]
  }}
]

Limit to {config.MAX_TOPICS} most important topics.
"""
        
        try:
            topics_data = self._call_llm_json_list(user_prompt, system_prompt, "topics")
            topics = [Topic(**topic) for topic in topics_data]
            logger.info(f"Identified {len(topics)} topics")
            return topics[:config.MAX_TOPICS]
        except Exception as e:
            logger.error(f"Failed to parse topics: {e}")
            return []
    
    def stage2_extract_concepts(
        self,
        content: str,
        topics: List[Topic],
        external_context: str = ""
    ) -> List[Concept]:
        """
        Stage 2: Extract key concepts from identified topics.
        
        Args:
            content: Combined classroom content
            topics: Topics identified in stage 1
            external_context: Optional reference context
        
        Returns:
            List of Concept objects
        """
        logger.info("Stage 2: Concept Extraction")
        
        topic_names = [t.name for t in topics]
        
        system_prompt = (
            "You are an expert at identifying and explaining educational concepts. "
            "Extract key concepts that students need to understand."
        )
        
        user_prompt = f"""
Extract key concepts from this classroom content for the following topics:
{", ".join(topic_names)}

CONTENT:
{content[:2000]}

EXTERNAL CONTEXT:
{external_context[:500] if external_context else "N/A"}

Return a JSON array of concepts:
[
  {{
    "name": "Concept Name",
    "definition": "Clear explanation suitable for students",
    "prerequisites": ["prerequisite1", "prerequisite2"],
    "difficulty_level": 3
  }}
]

Focus on concepts that are:
1. Central to understanding the topics
2. Clearly defined in the content
3. Buildable from prerequisites

Limit to {config.MAX_CONCEPTS_PER_TOPIC * len(topics)} concepts.
"""
        
        try:
            concepts_data = self._call_llm_json_list(user_prompt, system_prompt, "concepts")
            concepts = [Concept(**concept) for concept in concepts_data]
            logger.info(f"Extracted {len(concepts)} concepts")
            return concepts
        except Exception as e:
            logger.error(f"Failed to parse concepts: {e}")
            return []
    
    def stage3_interpret_equations(
        self,
        equations: List[str],
        content: str,
        concepts: List[Concept]
    ) -> List[EquationExplanation]:
        """
        Stage 3: Provide plain-language interpretations of equations.
        
        Args:
            equations: List of equation strings
            content: Classroom content for context
            concepts: Extracted concepts for linking
        
        Returns:
            List of EquationExplanation objects
        """
        logger.info("Stage 3: Equation Interpretation")
        
        if not equations:
            return []
        
        concept_names = [c.name for c in concepts]
        
        system_prompt = (
            "You are an expert at explaining mathematical equations in plain language. "
            "Make equations accessible to students learning the material."
        )
        
        user_prompt = f"""
Interpret these equations in plain language:

EQUATIONS:
{chr(10).join(f"{i+1}. {eq}" for i, eq in enumerate(equations))}

RELEVANT CONCEPTS:
{", ".join(concept_names)}

CONTEXT:
{content[:1000]}

For each equation, provide:
1. Plain-language explanation of what it means
2. Description of each variable
3. Where/how this equation is applied

Return a JSON array:
[
  {{
    "equation": "original equation",
    "explanation": "what it means in plain language",
    "variables": {{"var1": "description", "var2": "description"}},
    "applications": ["application1", "application2"]
  }}
]
"""
        
        try:
            explanations_data = self._call_llm_json_list(user_prompt, system_prompt, "equations")
            explanations = [EquationExplanation(**exp) for exp in explanations_data]
            logger.info(f"Interpreted {len(explanations)} equations")
            return explanations
        except Exception as e:
            logger.error(f"Failed to parse equation explanations: {e}")
            return []
    
    def stage4_detect_misconceptions(
        self,
        content: str,
        concepts: List[Concept],
        external_context: str = ""
    ) -> List[Misconception]:
        """
        Stage 4: Identify common misconceptions and mistakes.
        
        Args:
            content: Classroom content
            concepts: Extracted concepts
            external_context: Reference material
        
        Returns:
            List of Misconception objects
        """
        logger.info("Stage 4: Misconception Detection")
        
        concept_names = [c.name for c in concepts]
        
        system_prompt = (
            "You are an expert educator who understands common student misconceptions. "
            "Identify typical mistakes and incorrect understandings."
        )
        
        user_prompt = f"""
Based on this classroom content and these concepts, identify common misconceptions
or mistakes students typically make:

CONCEPTS:
{", ".join(concept_names)}

CONTENT:
{content[:1500]}

Return a JSON array:
[
  {{
    "misconception": "The incorrect belief or common mistake",
    "explanation": "Why this is wrong",
    "correct_understanding": "What students should understand instead",
    "related_concepts": ["concept1", "concept2"]
  }}
]

Focus on misconceptions that:
1. Are pedagogically important
2. Can be addressed with this content
3. Help deepen understanding
"""
        
        try:
            misconceptions_data = self._call_llm_json_list(user_prompt, system_prompt, "misconceptions")
            misconceptions = [Misconception(**m) for m in misconceptions_data]
            logger.info(f"Detected {len(misconceptions)} misconceptions")
            return misconceptions
        except Exception as e:
            logger.error(f"Failed to parse misconceptions: {e}")
            return []
    
    def stage5_generate_faqs(
        self,
        content: str,
        concepts: List[Concept],
        topics: List[Topic]
    ) -> List[FAQ]:
        """
        Stage 5: Generate frequently asked questions and answers.
        
        Args:
            content: Classroom content
            concepts: Extracted concepts
            topics: Identified topics
        
        Returns:
            List of FAQ objects
        """
        logger.info("Stage 5: FAQ Generation")
        
        system_prompt = (
            "You are an experienced teacher who anticipates student questions. "
            "Generate questions students commonly ask about this material."
        )
        
        user_prompt = f"""
Generate frequently asked questions (FAQs) that students would have about this content:

TOPICS:
{", ".join(t.name for t in topics)}

KEY CONCEPTS:
{", ".join(c.name for c in concepts[:10])}

CONTENT:
{content[:1500]}

Return a JSON array of FAQs:
[
  {{
    "question": "Student question",
    "answer": "Clear, pedagogical answer",
    "related_concepts": ["concept1", "concept2"],
    "difficulty": "easy|medium|hard"
  }}
]

Generate {config.MAX_FAQS} questions covering different difficulty levels and aspects.
"""
        
        try:
            faqs_data = self._call_llm_json_list(user_prompt, system_prompt, "faqs")
            faqs = [FAQ(**faq) for faq in faqs_data]
            logger.info(f"Generated {len(faqs)} FAQs")
            return faqs[:config.MAX_FAQS]
        except Exception as e:
            logger.error(f"Failed to parse FAQs: {e}")
            return []
    
    def stage6_real_world_connections(
        self,
        concepts: List[Concept],
        content: str
    ) -> List[RealWorldConnection]:
        """
        Stage 6: Identify real-world applications and connections.
        
        Args:
            concepts: Extracted concepts
            content: Classroom content
        
        Returns:
            List of RealWorldConnection objects
        """
        logger.info("Stage 6: Real-World Connections")
        
        system_prompt = (
            "You are an expert at connecting abstract concepts to real-world applications. "
            "Help students see the relevance and practical value of what they're learning."
        )
        
        user_prompt = f"""
Identify real-world applications and connections for these concepts:

CONCEPTS:
{chr(10).join(f"- {c.name}: {c.definition}" for c in concepts[:10])}

CONTENT CONTEXT:
{content[:1000]}

Return a JSON array:
[
  {{
    "concept": "Concept name",
    "application": "Real-world application area",
    "description": "How the concept is applied",
    "relevance": "Why students should care"
  }}
]

Focus on connections that:
1. Are authentic and specific
2. Motivate student learning
3. Span diverse fields (technology, science, daily life)
"""
        
        try:
            connections_data = self._call_llm_json_list(user_prompt, system_prompt, "connections")
            connections = [RealWorldConnection(**conn) for conn in connections_data]
            logger.info(f"Identified {len(connections)} real-world connections")
            return connections
        except Exception as e:
            logger.error(f"Failed to parse connections: {e}")
            return []
    
    def stage7_analyze_examples(
        self,
        content: str,
        concepts: List[Concept]
    ) -> List[WorkedExample]:
        """
        Stage 7: Extract and analyze worked examples from class.
        
        Args:
            content: Classroom content
            concepts: Extracted concepts
        
        Returns:
            List of WorkedExample objects
        """
        logger.info("Stage 7: Worked Example Analysis")
        
        system_prompt = (
            "You are an expert at analyzing worked examples in educational content. "
            "Extract and structure problem-solution pairs."
        )
        
        user_prompt = f"""
Extract worked examples (problems and solutions) from this content:

CONTENT:
{content[:2000]}

RELEVANT CONCEPTS:
{", ".join(c.name for c in concepts[:10])}

Return a JSON array:
[
  {{
    "problem": "Problem statement",
    "solution": "Step-by-step solution",
    "key_concepts": ["concept1", "concept2"],
    "common_mistakes": ["mistake1", "mistake2"]
  }}
]

Extract up to {config.MAX_WORKED_EXAMPLES} most illustrative examples.
"""
        
        try:
            examples_data = self._call_llm_json_list(user_prompt, system_prompt, "examples")
            examples = [WorkedExample(**ex) for ex in examples_data]
            logger.info(f"Extracted {len(examples)} worked examples")
            return examples[:config.MAX_WORKED_EXAMPLES]
        except Exception as e:
            logger.error(f"Failed to parse worked examples: {e}")
            return []
    
    def generate_class_summary(
        self,
        content: str,
        topics: List[Topic],
        concepts: List[Concept]
    ) -> str:
        """
        Generate comprehensive class summary.
        
        Args:
            content: Combined content
            topics: Identified topics
            concepts: Extracted concepts
        
        Returns:
            Summary string (2-3 paragraphs)
        """
        logger.info("Generating class summary")
        
        system_prompt = (
            "You are an expert at summarizing educational content. "
            "Create a comprehensive yet concise summary."
        )
        
        # Fast path: if no topics/concepts provided (filtered out), generate summary directly from content
        if not topics and not concepts:
            logger.info("Fast summary mode (no topics/concepts)")
            user_prompt = f"""
Create a comprehensive 2-3 paragraph summary of this educational content:

CONTENT:
{content[:3000]}

The summary should:
1. Provide a high-level overview of what was covered
2. Highlight the most important points and key takeaways
3. Be clear, concise, and suitable for study purposes

Return only the summary text, no extra formatting.
"""
        else:
            # Standard mode with topics/concepts
            user_prompt = f"""
Create a comprehensive 2-3 paragraph summary of this class session:

TOPICS COVERED:
{chr(10).join(f"- {t.name}: {t.summary}" for t in topics)}

KEY CONCEPTS:
{chr(10).join(f"- {c.name}" for c in concepts[:8])}

CONTENT:
{content[:2000]}

The summary should:
1. Provide a high-level overview of what was covered
2. Highlight the most important concepts
3. Connect topics in a coherent narrative
4. Be suitable for study guide purposes
"""
        
        response = self._call_llm(user_prompt, system_prompt)
        return response.strip()
    
    def process(
        self,
        combined_content: str,
        equations: List[str],
        external_context: str = "",
        session_id: str = None,
        output_filters: Dict[str, bool] = None
    ) -> ClassSessionOutput:
        """
        Execute full multi-stage reasoning pipeline.
        
        This is the main entry point that orchestrates all reasoning stages
        in sequence, building up the complete structured output.
        
        Args:
            combined_content: Preprocessed classroom content
            equations: List of equations to interpret
            external_context: Optional reference material
            session_id: Unique session identifier
            output_filters: Dict of section names to booleans to skip unnecessary processing
        
        Returns:
            ClassSessionOutput with all extracted knowledge
        """
        logger.info("=" * 60)
        logger.info("Starting multi-stage reasoning pipeline")
        if output_filters:
            logger.info(f"Output filters: {output_filters}")
        logger.info("=" * 60)
        
        if output_filters is None:
            output_filters = {
                'summary': True,
                'topics': True,
                'concepts': True,
                'equations': True,
                'misconceptions': True,
                'faqs': True,
                'worked_examples': True,
                'real_world': True
            }
        
        logger.info(f"Stage execution plan: summary={output_filters.get('summary', True)}, topics={output_filters.get('topics', True)}, concepts={output_filters.get('concepts', True)}, equations={output_filters.get('equations', True)}, misconceptions={output_filters.get('misconceptions', True)}, faqs={output_filters.get('faqs', True)}, worked_examples={output_filters.get('worked_examples', True)}, real_world={output_filters.get('real_world', True)}")
        
        # Stage 1: Topic Identification
        if output_filters.get('topics', True):
            logger.info(">>> Executing Stage 1: Topic Identification")
            topics = self.stage1_identify_topics(combined_content, external_context)
        else:
            logger.info(">>> Skipping Stage 1: Topic Identification (filtered)")
            topics = []
        
        # Stage 2: Concept Extraction
        if output_filters.get('concepts', True):
            logger.info(">>> Executing Stage 2: Concept Extraction")
            concepts = self.stage2_extract_concepts(combined_content, topics, external_context)
        else:
            logger.info(">>> Skipping Stage 2: Concept Extraction (filtered)")
            concepts = []
        
        # Stage 3: Equation Interpretation
        if output_filters.get('equations', True):
            logger.info(">>> Executing Stage 3: Equation Interpretation")
            equation_explanations = self.stage3_interpret_equations(
                equations, combined_content, concepts
            )
        else:
            logger.info(">>> Skipping Stage 3: Equation Interpretation (filtered)")
            equation_explanations = []
        
        # Stage 4: Misconception Detection
        if output_filters.get('misconceptions', True):
            logger.info(">>> Executing Stage 4: Misconception Detection")
            misconceptions = self.stage4_detect_misconceptions(
                combined_content, concepts, external_context
            )
        else:
            logger.info(">>> Skipping Stage 4: Misconception Detection (filtered)")
            misconceptions = []
        
        # Stage 5: FAQ Generation
        if output_filters.get('faqs', True):
            logger.info(">>> Executing Stage 5: FAQ Generation")
            faqs = self.stage5_generate_faqs(combined_content, concepts, topics)
        else:
            logger.info(">>> Skipping Stage 5: FAQ Generation (filtered)")
            faqs = []
        
        # Stage 6: Real-World Connections
        if output_filters.get('real_world', True):
            logger.info(">>> Executing Stage 6: Real-World Connections")
            real_world_connections = self.stage6_real_world_connections(
                concepts, combined_content
            )
        else:
            logger.info(">>> Skipping Stage 6: Real-World Connections (filtered)")
            real_world_connections = []
        
        # Stage 7: Worked Examples
        if output_filters.get('worked_examples', True):
            logger.info(">>> Executing Stage 7: Worked Examples")
            worked_examples = self.stage7_analyze_examples(combined_content, concepts)
        else:
            logger.info(">>> Skipping Stage 7: Worked Examples (filtered)")
            worked_examples = []
        
        # Generate summary
        if output_filters.get('summary', True):
            logger.info(">>> Executing Summary Generation")
            class_summary = self.generate_class_summary(combined_content, topics, concepts)
        else:
            logger.info(">>> Skipping Summary Generation (filtered)")
            class_summary = ""

        # If structured outputs are empty, retry stages using the summary as context
        if (
            class_summary
            and not topics
            and not concepts
            and not equation_explanations
            and not worked_examples
            and not misconceptions
            and not faqs
            and not real_world_connections
        ):
            enriched_content = f"SUMMARY:\n{class_summary}\n\nSOURCE CONTENT:\n{combined_content}"
            if output_filters.get('topics', True):
                topics = self.stage1_identify_topics(enriched_content, external_context)
            if output_filters.get('concepts', True):
                concepts = self.stage2_extract_concepts(enriched_content, topics, external_context)
            if output_filters.get('equations', True):
                equation_explanations = self.stage3_interpret_equations(
                    equations, enriched_content, concepts
                )
            if output_filters.get('misconceptions', True):
                misconceptions = self.stage4_detect_misconceptions(
                    enriched_content, concepts, external_context
                )
            if output_filters.get('faqs', True):
                faqs = self.stage5_generate_faqs(enriched_content, concepts, topics)
            if output_filters.get('real_world', True):
                real_world_connections = self.stage6_real_world_connections(
                    concepts, enriched_content
                )
            if output_filters.get('worked_examples', True):
                worked_examples = self.stage7_analyze_examples(enriched_content, concepts)
        
        # Assemble output
        output = ClassSessionOutput(
            session_id=session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(),
            class_summary=class_summary,
            topics=topics,
            key_concepts=concepts,
            equation_explanations=equation_explanations,
            worked_examples=worked_examples,
            common_mistakes=misconceptions,
            faqs=faqs,
            real_world_connections=real_world_connections,
            metadata={
                "model": self.model,
                "temperature": self.temperature,
                "num_topics": len(topics),
                "num_concepts": len(concepts),
                "num_equations": len(equation_explanations),
                "num_examples": len(worked_examples),
                "filters_applied": output_filters
            }
        )
        
        logger.info("=" * 60)
        logger.info("Pipeline complete")
        logger.info(f"Generated: {len(topics)} topics, {len(concepts)} concepts, "
                   f"{len(faqs)} FAQs, {len(worked_examples)} examples")
        logger.info("=" * 60)
        
        return output
