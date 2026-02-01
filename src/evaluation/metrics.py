"""
Evaluation framework for assessing quality of extracted educational content.

This module provides metrics and rubrics for evaluating the reasoning pipeline's
output across multiple dimensions: correctness, structure, hallucinations, and
educational usefulness.
"""

import logging
from typing import Dict, List, Optional, Tuple
import re

from src.schema.output_schema import ClassSessionOutput, EvaluationResult
import config

logging.basicConfig(level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)


class ContentEvaluator:
    """
    Evaluates educational content quality across multiple dimensions.
    
    This class implements evaluation metrics suitable for research assessment,
    including both automated heuristics and LLM-assisted evaluation.
    """
    
    def __init__(self):
        """Initialize the content evaluator."""
        pass
    
    def evaluate_reasoning_correctness(
        self,
        output: ClassSessionOutput,
        ground_truth: Dict = None,
        reference_content: str = None
    ) -> float:
        """
        Evaluate correctness of extracted reasoning.
        
        Assesses how accurate and logically sound the extracted educational
        content is compared to ground truth or reference material.
        
        Args:
            output: Session output to evaluate
            ground_truth: Optional ground truth annotations
            reference_content: Optional reference material for comparison
        
        Returns:
            Score between 0 and 1 (1 = perfectly correct)
        
        Note:
            In a full research implementation, this would use:
            - Expert annotations
            - LLM-based evaluation
            - Semantic similarity to reference material
        """
        score = 0.0
        checks = 0
        
        # Check 1: Concepts have definitions
        if output.key_concepts:
            defined_concepts = sum(
                1 for c in output.key_concepts
                if c.definition and len(c.definition) > 20
            )
            score += defined_concepts / len(output.key_concepts)
            checks += 1
        
        # Check 2: Topics have substantial summaries
        if output.topics:
            substantial_topics = sum(
                1 for t in output.topics
                if t.summary and len(t.summary) > 30
            )
            score += substantial_topics / len(output.topics)
            checks += 1
        
        # Check 3: Worked examples have solutions
        if output.worked_examples:
            complete_examples = sum(
                1 for ex in output.worked_examples
                if ex.solution and len(ex.solution) > 50
            )
            score += complete_examples / len(output.worked_examples)
            checks += 1
        
        # Check 4: Equation explanations are comprehensive
        if output.equation_explanations:
            explained_equations = sum(
                1 for eq in output.equation_explanations
                if eq.explanation and eq.variables
            )
            score += explained_equations / len(output.equation_explanations)
            checks += 1
        
        # Check 5: FAQs have substantial answers
        if output.faqs:
            answered_faqs = sum(
                1 for faq in output.faqs
                if faq.answer and len(faq.answer) > 30
            )
            score += answered_faqs / len(output.faqs)
            checks += 1
        
        final_score = score / checks if checks > 0 else 0.5
        
        logger.info(f"Reasoning correctness: {final_score:.3f}")
        return final_score
    
    def evaluate_structural_accuracy(self, output: ClassSessionOutput) -> float:
        """
        Evaluate structural quality and completeness of output.
        
        Checks that the output contains all expected components and that
        each component is well-formed according to the schema.
        
        Args:
            output: Session output to evaluate
        
        Returns:
            Score between 0 and 1 (1 = perfectly structured)
        """
        score = 0.0
        total_checks = 7
        
        # Check 1: Has class summary
        if output.class_summary and len(output.class_summary) >= 50:
            score += 1.0
        
        # Check 2: Has topics
        if output.topics and len(output.topics) > 0:
            score += 1.0
        
        # Check 3: Has concepts
        if output.key_concepts and len(output.key_concepts) > 0:
            score += 1.0
        
        # Check 4: Topics have required fields
        if output.topics:
            complete_topics = sum(
                1 for t in output.topics
                if t.name and t.summary and t.learning_objectives
            )
            score += (complete_topics / len(output.topics))
        
        # Check 5: Concepts have prerequisites
        if output.key_concepts:
            concepts_with_prereqs = sum(
                1 for c in output.key_concepts
                if c.prerequisites  # Not empty
            )
            score += min(1.0, concepts_with_prereqs / max(1, len(output.key_concepts) / 2))
        
        # Check 6: Has diverse content types
        content_types = sum([
            bool(output.topics),
            bool(output.key_concepts),
            bool(output.worked_examples),
            bool(output.faqs),
            bool(output.common_mistakes),
            bool(output.real_world_connections)
        ])
        score += (content_types / 6.0)
        
        # Check 7: Metadata present
        if output.metadata and len(output.metadata) > 0:
            score += 1.0
        
        final_score = score / total_checks
        
        logger.info(f"Structural accuracy: {final_score:.3f}")
        return final_score
    
    def estimate_hallucination_rate(
        self,
        output: ClassSessionOutput,
        source_content: str,
        external_context: str = ""
    ) -> float:
        """
        Estimate rate of hallucinated (unsupported) content.
        
        Uses heuristics to detect content that doesn't appear grounded
        in the source material. Lower scores are better.
        
        Args:
            output: Session output to evaluate
            source_content: Original classroom content
            external_context: External reference material
        
        Returns:
            Estimated hallucination rate (0 = no hallucinations, 1 = all hallucinated)
        
        Note:
            This is a heuristic approximation. Full implementation would use:
            - Citation checking
            - Semantic entailment models
            - Expert validation
        """
        combined_source = source_content.lower()
        if external_context:
            combined_source += " " + external_context.lower()
        
        hallucination_indicators = 0
        total_checks = 0
        
        # Check concepts against source
        for concept in output.key_concepts:
            total_checks += 1
            # Check if concept name appears in source
            if concept.name.lower() not in combined_source:
                # Check if any word from concept name appears
                words = concept.name.lower().split()
                if not any(word in combined_source for word in words if len(word) > 3):
                    hallucination_indicators += 1
        
        # Check topics against source
        for topic in output.topics:
            total_checks += 1
            if topic.name.lower() not in combined_source:
                words = topic.name.lower().split()
                if not any(word in combined_source for word in words if len(word) > 3):
                    hallucination_indicators += 0.5  # Topics can be more abstract
        
        # Check worked examples
        for example in output.worked_examples:
            total_checks += 1
            # Check if problem has any overlap with source
            problem_words = set(example.problem.lower().split())
            source_words = set(combined_source.split())
            overlap = len(problem_words & source_words) / len(problem_words) if problem_words else 0
            if overlap < 0.2:  # Less than 20% word overlap
                hallucination_indicators += 1
        
        # Estimate rate
        if total_checks == 0:
            hallucination_rate = 0.5  # Uncertain
        else:
            hallucination_rate = hallucination_indicators / total_checks
        
        # Cap at 1.0
        hallucination_rate = min(1.0, hallucination_rate)
        
        logger.info(f"Estimated hallucination rate: {hallucination_rate:.3f}")
        return hallucination_rate
    
    def assess_educational_usefulness(self, output: ClassSessionOutput) -> float:
        """
        Assess pedagogical value using a rubric.
        
        Evaluates how useful the output would be for student learning,
        considering factors like clarity, depth, and practical value.
        
        Args:
            output: Session output to evaluate
        
        Returns:
            Score between 1 and 5 (5 = highly useful)
        
        Rubric:
            5 = Exceptional: Comprehensive, clear, actionable
            4 = Strong: Good coverage with minor gaps
            3 = Adequate: Useful but limited depth
            2 = Weak: Significant gaps or clarity issues
            1 = Poor: Minimal educational value
        """
        score = 1.0  # Start at minimum
        
        # Criterion 1: Comprehensiveness (0-1.5 points)
        content_variety = sum([
            bool(output.topics),
            bool(output.key_concepts),
            bool(output.worked_examples),
            bool(output.faqs),
            bool(output.common_mistakes),
            bool(output.real_world_connections)
        ])
        score += (content_variety / 6.0) * 1.5
        
        # Criterion 2: Depth (0-1.0 points)
        avg_concept_detail = 0
        if output.key_concepts:
            avg_concept_detail = sum(
                len(c.definition) for c in output.key_concepts
            ) / len(output.key_concepts)
            # Award points for detailed definitions
            score += min(1.0, avg_concept_detail / 100.0)
        
        # Criterion 3: Practical value (0-1.0 points)
        practical_elements = sum([
            len(output.worked_examples),
            len(output.real_world_connections),
            len(output.faqs)
        ])
        score += min(1.0, practical_elements / 10.0)
        
        # Criterion 4: Organization and clarity (0-0.5 points)
        if output.class_summary and len(output.class_summary) > 100:
            score += 0.3
        if output.topics and all(t.learning_objectives for t in output.topics):
            score += 0.2
        
        # Ensure score is in valid range
        final_score = min(5.0, max(1.0, score))
        
        logger.info(f"Educational usefulness: {final_score:.2f}/5.0")
        return final_score
    
    def evaluate(
        self,
        output: ClassSessionOutput,
        source_content: str,
        external_context: str = "",
        ground_truth: Dict = None
    ) -> EvaluationResult:
        """
        Perform comprehensive evaluation across all dimensions.
        
        This is the main evaluation entry point that computes all metrics
        and returns a structured evaluation result.
        
        Args:
            output: Session output to evaluate
            source_content: Original classroom content
            external_context: External reference material
            ground_truth: Optional ground truth for validation
        
        Returns:
            EvaluationResult with all metrics
        """
        logger.info("=" * 60)
        logger.info(f"Evaluating session: {output.session_id}")
        logger.info("=" * 60)
        
        reasoning_correctness = self.evaluate_reasoning_correctness(
            output, ground_truth, source_content
        )
        
        structural_accuracy = self.evaluate_structural_accuracy(output)
        
        hallucination_rate = self.estimate_hallucination_rate(
            output, source_content, external_context
        )
        
        educational_usefulness = self.assess_educational_usefulness(output)
        
        # Detailed feedback
        detailed_feedback = {
            "num_topics": len(output.topics),
            "num_concepts": len(output.key_concepts),
            "num_examples": len(output.worked_examples),
            "num_faqs": len(output.faqs),
            "num_misconceptions": len(output.common_mistakes),
            "summary_length": len(output.class_summary),
            "has_equations": len(output.equation_explanations) > 0,
            "has_real_world": len(output.real_world_connections) > 0
        }
        
        result = EvaluationResult(
            session_id=output.session_id,
            reasoning_correctness=reasoning_correctness,
            structural_accuracy=structural_accuracy,
            hallucination_rate=hallucination_rate,
            educational_usefulness=educational_usefulness,
            detailed_feedback=detailed_feedback
        )
        
        # Check against thresholds
        passes = result.passes_thresholds()
        
        logger.info("=" * 60)
        logger.info(f"Evaluation complete: {'PASS' if passes else 'FAIL'}")
        logger.info(f"  Reasoning: {reasoning_correctness:.3f} (min: {config.MIN_REASONING_CORRECTNESS})")
        logger.info(f"  Structure: {structural_accuracy:.3f} (min: {config.MIN_STRUCTURAL_ACCURACY})")
        logger.info(f"  Hallucination: {hallucination_rate:.3f} (max: {config.MAX_HALLUCINATION_RATE})")
        logger.info(f"  Usefulness: {educational_usefulness:.2f}/5.0 (min: {config.MIN_EDUCATIONAL_USEFULNESS})")
        logger.info("=" * 60)
        
        return result


def evaluate_session_output(
    output: ClassSessionOutput,
    source_content: str,
    external_context: str = ""
) -> EvaluationResult:
    """
    Convenience function for evaluating session output.
    
    Args:
        output: Session output to evaluate
        source_content: Original content
        external_context: Reference material
    
    Returns:
        EvaluationResult
    """
    evaluator = ContentEvaluator()
    return evaluator.evaluate(output, source_content, external_context)
