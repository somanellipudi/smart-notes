"""
Claim Extractor: Convert Smart Notes outputs to evidence-grounded learning claims.

This module implements the evidence-first paradigm: extract claims from baseline
outputs with EMPTY claim_text initially, then populate via RAG retrieval and
role-constrained agents.

Key Design: Claims start without text, forcing evidence-first generation.
See docs/VERIFIABILITY_CONTRACT.md for details.

Includes automatic question detection to prevent questions from being treated as
factual claims.
"""

import logging
import re
from typing import List, Dict, Any
import uuid

from .schema import LearningClaim, ClaimType, VerificationStatus
from .claim_type_detector import detect_claim_type

logger = logging.getLogger(__name__)


class ClaimExtractor:
    """Extract learning claims from Smart Notes structured outputs (evidence-first)."""

    @staticmethod
    def extract_from_session(session_output: Any) -> "ClaimCollection":
        """
        Extract claims from a ClassSessionOutput or dict.

        Args:
            session_output: ClassSessionOutput instance or dict

        Returns:
            ClaimCollection with extracted claims
        """
        from .schema import ClaimCollection

        if hasattr(session_output, "to_dict"):
            output_dict = session_output.to_dict()
        elif isinstance(session_output, dict):
            output_dict = session_output
        else:
            raise TypeError("session_output must be ClassSessionOutput or dict")

        logger.info(f"Extracting claims from output with keys: {list(output_dict.keys())}")
        claims = ClaimExtractor.extract_all_from_structured_output(output_dict)

        session_id = output_dict.get("session_id", "unknown_session")
        collection = ClaimCollection(session_id=session_id, claims=claims)
        collection.metadata.update({"source": "session_output"})
        logger.info(f"Created ClaimCollection with {len(claims)} claims for session {session_id}")
        return collection
    
    @staticmethod
    def extract_from_concepts(concepts: List[Dict[str, Any]]) -> List[LearningClaim]:
        """
        Extract definition claims from concepts.
        
        Args:
            concepts: List of concept dicts with 'name', 'definition', 'explanation'
        
        Returns:
            List of LearningClaim objects with empty claim_text (evidence-first)
        """
        claims = []
        for i, concept in enumerate(concepts):
            if not concept.get("name"):
                continue
            
            claim = LearningClaim(
                claim_id=f"claim_{uuid.uuid4().hex[:8]}",
                claim_type=ClaimType.DEFINITION,
                claim_text="",  # Evidence-first: empty until retrieval + agent generation
                status=VerificationStatus.REJECTED,  # Start rejected until verified
                metadata={
                    "source": "concepts",
                    "concept_name": concept.get("name"),
                    "definition": concept.get("definition", ""),
                    "explanation": concept.get("explanation", ""),
                    "prerequisites": concept.get("prerequisites", []),
                    "difficulty_level": concept.get("difficulty_level", "intermediate"),
                    "draft_text": concept.get("name", ""),  # For RAG retrieval
                    "ui_display": f"Definition: {concept.get('name', 'Unknown')}"
                }
            )
            claims.append(claim)
        
        return claims
    
    @staticmethod
    def extract_from_equations(equations: List[Dict[str, Any]]) -> List[LearningClaim]:
        """
        Extract equation claims.
        
        Args:
            equations: List of equation dicts with 'equation', 'explanation'
        
        Returns:
            List of LearningClaim objects (equations)
        """
        claims = []
        for i, eq in enumerate(equations):
            if not eq.get("equation"):
                continue
            
            claim = LearningClaim(
                claim_id=f"claim_{uuid.uuid4().hex[:8]}",
                claim_type=ClaimType.EQUATION,
                claim_text="",  # Evidence-first
                status=VerificationStatus.REJECTED,
                metadata={
                    "source": "equations",
                    "equation": eq.get("equation"),
                    "explanation": eq.get("explanation", ""),
                    "variables": eq.get("variables", []),
                    "applications": eq.get("applications", []),
                    "draft_text": eq.get("equation", ""),  # For RAG retrieval
                    "ui_display": f"Equation: {eq.get('equation', 'Unknown')}"
                }
            )
            claims.append(claim)
        
        return claims
    
    @staticmethod
    def extract_from_examples(examples: List[Dict[str, Any]]) -> List[LearningClaim]:
        """
        Extract example claims.
        
        Args:
            examples: List of example dicts with 'problem', 'solution'
        
        Returns:
            List of LearningClaim objects (examples)
        """
        claims = []
        for i, ex in enumerate(examples):
            if not ex.get("problem"):
                continue
            
            claim = LearningClaim(
                claim_id=f"claim_{uuid.uuid4().hex[:8]}",
                claim_type=ClaimType.EXAMPLE,
                claim_text="",  # Evidence-first
                status=VerificationStatus.REJECTED,
                metadata={
                    "source": "examples",
                    "problem": ex.get("problem"),
                    "solution": ex.get("solution", ""),
                    "explanation": ex.get("explanation", ""),
                    "key_concepts": ex.get("key_concepts", []),
                    "common_mistakes": ex.get("common_mistakes", []),
                    "draft_text": ex.get("problem", "")[:100],  # For RAG retrieval
                    "ui_display": f"Example: {ex.get('problem', 'Unknown')[:50]}..."
                }
            )
            claims.append(claim)
        
        return claims
    
    @staticmethod
    def extract_from_misconceptions(misconceptions: List[Dict[str, Any]]) -> List[LearningClaim]:
        """
        Extract misconception claims.
        
        Args:
            misconceptions: List of misconception dicts with 'misconception', 'clarification'
        
        Returns:
            List of LearningClaim objects (misconceptions)
        """
        claims = []
        for i, misc in enumerate(misconceptions):
            if not misc.get("misconception"):
                continue
            
            claim = LearningClaim(
                claim_id=f"claim_{uuid.uuid4().hex[:8]}",
                claim_type=ClaimType.MISCONCEPTION,
                claim_text="",  # Evidence-first
                status=VerificationStatus.REJECTED,
                metadata={
                    "source": "misconceptions",
                    "misconception": misc.get("misconception"),
                    "correct_understanding": misc.get("correct_understanding", ""),
                    "explanation": misc.get("explanation", ""),
                    "related_concepts": misc.get("related_concepts", []),
                    "draft_text": misc.get("misconception", ""),  # For RAG retrieval
                    "ui_display": f"Misconception: {misc.get('misconception', 'Unknown')[:50]}..."
                }
            )
            claims.append(claim)
        
        return claims
    
    @staticmethod
    def extract_from_questions(questions: List[Dict[str, Any]]) -> List[LearningClaim]:
        """
        Extract question claims from FAQ or study questions.
        
        Args:
            questions: List of question dicts with 'question', optionally 'answer'
        
        Returns:
            List of LearningClaim objects with type=QUESTION
        """
        claims = []
        for i, q in enumerate(questions):
            question_text = q.get("question") or q.get("text") or ""
            if not question_text:
                continue
            
            # Auto-detect if this is really a question
            detected_type = detect_claim_type(question_text)
            
            claim = LearningClaim(
                claim_id=f"claim_{uuid.uuid4().hex[:8]}",
                claim_type=detected_type,
                claim_text=question_text,  # Questions have text from the start
                status=VerificationStatus.REJECTED,
                metadata={
                    "source": "questions",
                    "original_answer": q.get("answer", ""),
                    "difficulty": q.get("difficulty", "unknown"),
                    "draft_text": question_text,
                    "ui_display": f"Q: {question_text[:60]}...",
                    "requires_answer": True,
                    "skip_verification": True  # Questions don't get verified
                }
            )
            claims.append(claim)
        
        logger.info(f"Extracted {len(claims)} question claims")
        return claims
    
    @staticmethod
    def extract_with_auto_detection(text: str, source: str = "unknown") -> LearningClaim:
        """
        Extract a single claim with automatic type detection.
        
        Args:
            text: Text content
            source: Source identifier
        
        Returns:
            LearningClaim with auto-detected type
        """
        detected_type = detect_claim_type(text)
        
        # Adjust status based on type
        if detected_type == ClaimType.QUESTION:
            initial_status = VerificationStatus.REJECTED  # Needs answering
        elif detected_type == ClaimType.MISCONCEPTION:
            initial_status = VerificationStatus.NEEDS_FRAMING  # Needs framing check
        else:
            initial_status = VerificationStatus.REJECTED  # Needs verification
        
        claim = LearningClaim(
            claim_id=f"claim_{uuid.uuid4().hex[:8]}",
            claim_type=detected_type,
            claim_text=text if detected_type == ClaimType.QUESTION else "",  # Questions have text
            status=initial_status,
            metadata={
                "source": source,
                "draft_text": text,
                "detected_type": detected_type.value,
                "ui_display": text[:70] + "..." if len(text) > 70 else text,
                "requires_answer": (detected_type == ClaimType.QUESTION),
                "skip_verification": (detected_type == ClaimType.QUESTION)
            }
        )
        
        return claim
    
    @staticmethod
    def extract_from_summary(summary: str) -> List[LearningClaim]:
        """
        Extract high-level summary claims when only summary section is available.
        
        Args:
            summary: Summary text
        
        Returns:
            List of LearningClaim objects extracted from summary
        """
        if not summary or not summary.strip():
            return []
        
        claims = []
        
        # Split summary into sentences and create a summary claim
        sentences = [s.strip() for s in summary.split(".") if s.strip()]
        
        if sentences:
            # Create one comprehensive summary claim
            claim = LearningClaim(
                claim_id=f"claim_{uuid.uuid4().hex[:8]}",
                claim_type=ClaimType.DEFINITION,
                claim_text="",  # Evidence-first
                status=VerificationStatus.REJECTED,
                metadata={
                    "source": "summary",
                    "full_summary": summary,
                    "sentence_count": len(sentences),
                    "draft_text": summary[:200],  # For RAG retrieval
                    "ui_display": f"Summary: {summary[:100].rstrip()}..."
                }
            )
            claims.append(claim)
            logger.info(f"Extracted 1 summary claim from {len(sentences)} sentences")
        
        return claims

    @staticmethod
    def extract_atomic_claims(text: str, domain: str = "cs") -> List[LearningClaim]:
        """
        Extract simple atomic claims from raw text.

        Uses sentence splitting and emits definition-style claims as a fallback
        when structured outputs are unavailable.
        """
        if not text or not text.strip():
            return []

        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        claims = []
        for sentence in sentences:
            if len(sentence) < 20:
                continue
            claim = LearningClaim(
                claim_id=f"claim_{uuid.uuid4().hex[:8]}",
                claim_type=ClaimType.DEFINITION,
                claim_text=sentence,
                status=VerificationStatus.REJECTED,
                metadata={
                    "source": "raw_text",
                    "domain": domain,
                    "draft_text": sentence
                }
            )
            claims.append(claim)

        return claims
    
    @staticmethod
    def extract_all_from_structured_output(output: Dict[str, Any]) -> List[LearningClaim]:
        """
        Extract claims from all structured output fields.
        
        Args:
            output: Smart Notes structured output dict with keys like
                   'key_concepts', 'equation_explanations', 'worked_examples', 
                   'common_mistakes', 'questions', 'faqs', 'class_summary'
        
        Returns:
            List of all extracted LearningClaim objects (empty claim_text for facts, evidence-first)
        """
        all_claims = []
        
        # Extract from each source (prioritized by reliability)
        if output.get("key_concepts"):
            all_claims.extend(ClaimExtractor.extract_from_concepts(output["key_concepts"]))
        
        if output.get("equation_explanations"):
            all_claims.extend(ClaimExtractor.extract_from_equations(output["equation_explanations"]))
        
        if output.get("worked_examples"):
            all_claims.extend(ClaimExtractor.extract_from_examples(output["worked_examples"]))
        
        if output.get("common_mistakes"):
            all_claims.extend(ClaimExtractor.extract_from_misconceptions(output["common_mistakes"]))
        
        # Extract questions/FAQs
        if output.get("questions"):
            all_claims.extend(ClaimExtractor.extract_from_questions(output["questions"]))
        
        if output.get("faqs"):
            all_claims.extend(ClaimExtractor.extract_from_questions(output["faqs"]))
        
        if output.get("study_questions"):
            all_claims.extend(ClaimExtractor.extract_from_questions(output["study_questions"]))
        
        # If no claims were extracted but summary exists, extract from summary (summary-only mode)
        if not all_claims:
            # Try both 'summary' and 'class_summary' field names
            summary_text = output.get("summary") or output.get("class_summary")
            if summary_text:
                logger.info("No claims from structured sections, extracting from summary (summary-only mode)")
                all_claims.extend(ClaimExtractor.extract_from_summary(summary_text))
        
        logger.info(f"Extracted {len(all_claims)} claims from structured output")
        
        return all_claims
