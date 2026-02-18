"""
Question Answering with Citations.

Handles QUESTION type claims by:
1. Retrieving relevant evidence
2. Generating concise answer with citations
3. Marking status as ANSWERED_WITH_CITATIONS

Questions skip the standard claim verification pipeline.
"""

import logging
from typing import List, Optional, Any
from .schema import LearningClaim, VerificationStatus, EvidenceItem

logger = logging.getLogger(__name__)


class QuestionAnswerer:
    """
    Answer questions with evidence-based citations.
    
    Questions are NOT verified like fact claims. Instead, we:
    1. Retrieve top-k evidence snippets
    2. Generate concise answer citing evidence
    3. Mark as ANSWERED_WITH_CITATIONS
    """
    
    def __init__(
        self,
        evidence_store: Optional[Any] = None,
        embedding_provider: Optional[Any] = None,
        top_k: int = 5
    ):
        """
        Initialize question answerer.
        
        Args:
            evidence_store: EvidenceStore instance for retrieval
            embedding_provider: EmbeddingProvider for query encoding
            top_k: Number of evidence snippets to retrieve (default: 5)
        """
        self.evidence_store = evidence_store
        self.embedding_provider = embedding_provider
        self.top_k = top_k
    
    def answer_question(
        self,
        claim: LearningClaim,
        generate_answer: bool = True
    ) -> LearningClaim:
        """
        Answer a question claim with citations.
        
        Args:
            claim: LearningClaim with claim_type=QUESTION
            generate_answer: If True, generate answer text from evidence
        
        Returns:
            Updated claim with answer_text and status=ANSWERED_WITH_CITATIONS
        """
        from .schema import ClaimType
        
        if claim.claim_type != ClaimType.QUESTION:
            logger.warning(
                f"answer_question called on non-question claim: {claim.claim_type}"
            )
            return claim
        
        question_text = claim.claim_text or claim.metadata.get("draft_text", "")
        
        if not question_text:
            logger.warning(f"Question claim {claim.claim_id} has no text")
            claim.status = VerificationStatus.REJECTED
            claim.confidence = 0.0
            return claim
        
        # Step 1: Retrieve evidence
        logger.info(f"Retrieving evidence for question: '{question_text[:50]}...'")
        evidence = self._retrieve_evidence(question_text, claim)
        
        if not evidence:
            logger.warning(f"No evidence found for question: '{question_text[:50]}...'")
            claim.status = VerificationStatus.REJECTED
            claim.confidence = 0.0
            claim.answer_text = "No relevant information found in source material."
            return claim
        
        # Add evidence to claim
        claim.evidence_objects = evidence
        claim.evidence_ids = [e.evidence_id for e in evidence]
        
        # Step 2: Generate answer with citations
        if generate_answer:
            answer = self._generate_answer(question_text, evidence, claim)
            claim.answer_text = answer
        
        # Step 3: Set status and confidence
        claim.status = VerificationStatus.ANSWERED_WITH_CITATIONS
        claim.confidence = self._compute_answer_confidence(evidence)
        claim.validated_at = datetime.now()
        
        logger.info(
            f"Answered question with {len(evidence)} citations, "
            f"confidence={claim.confidence:.2f}"
        )
        
        return claim
    
    def _retrieve_evidence(
        self,
        question: str,
        claim: LearningClaim
    ) -> List[EvidenceItem]:
        """
        Retrieve evidence snippets for question.
        
        Args:
            question: Question text
            claim: LearningClaim object
        
        Returns:
            List of EvidenceItem objects
        """
        if not self.evidence_store or not self.embedding_provider:
            logger.warning("Evidence store or embedding provider not configured")
            return []
        
        try:
            # Encode question as query
            query_embedding = self.embedding_provider.embed_queries([question])[0]
            
            # Search evidence store
            results = self.evidence_store.search(
                query_embedding=query_embedding,
                top_k=self.top_k,
                min_similarity=0.2  # Lower threshold for questions
            )
            
            # Convert to EvidenceItem objects
            evidence = []
            for i, (evidence_id, similarity, metadata) in enumerate(results):
                snippet = metadata.get("snippet", "")
                if not snippet:
                    snippet = metadata.get("text", "")
                
                evidence_item = EvidenceItem(
                    evidence_id=evidence_id,
                    source_id=metadata.get("doc_id", "unknown"),
                    source_type=metadata.get("source_type", "notes"),
                    snippet=snippet,
                    span_metadata=metadata.get("span_metadata", {}),
                    similarity=float(similarity),
                    reliability_prior=0.8  # Default reliability
                )
                evidence.append(evidence_item)
            
            logger.debug(f"Retrieved {len(evidence)} evidence items for question")
            return evidence
        
        except Exception as e:
            logger.error(f"Error retrieving evidence for question: {e}")
            return []
    
    def _generate_answer(
        self,
        question: str,
        evidence: List[EvidenceItem],
        claim: LearningClaim
    ) -> str:
        """
        Generate concise answer with citations from evidence.
        
        Args:
            question: Question text
            evidence: List of evidence items
            claim: LearningClaim object
        
        Returns:
            Answer text with inline citations [1], [2], etc.
        """
        if not evidence:
            return "No relevant information found."
        
        # For now, simple extractive answer from top evidence
        # TODO: Use LLM to synthesize answer from multiple evidence items
        
        # Take top 3 evidence snippets
        top_evidence = evidence[:3]
        
        # Simple concatenation with citations
        answer_parts = []
        for i, ev in enumerate(top_evidence, 1):
            snippet = ev.snippet.strip()
            # Truncate long snippets
            if len(snippet) > 150:
                snippet = snippet[:147] + "..."
            answer_parts.append(f"{snippet} [{i}]")
        
        answer = " ".join(answer_parts)
        
        # Add source references
        answer += "\n\nSources:\n"
        for i, ev in enumerate(top_evidence, 1):
            source_id = ev.source_id
            answer += f"[{i}] {source_id}\n"
        
        return answer
    
    def _compute_answer_confidence(self, evidence: List[EvidenceItem]) -> float:
        """
        Compute confidence score for answer based on evidence quality.
        
        Args:
            evidence: List of evidence items
        
        Returns:
            Confidence score (0.0-1.0)
        """
        if not evidence:
            return 0.0
        
        # Average similarity of top-3 evidence
        top_3 = evidence[:3]
        avg_similarity = sum(e.similarity for e in top_3) / len(top_3)
        
        # Adjust based on evidence count
        count_factor = min(len(evidence) / 3.0, 1.0)  # Max boost at 3+ evidence
        
        confidence = avg_similarity * 0.7 + count_factor * 0.3
        return min(confidence, 0.95)  # Cap at 0.95


from datetime import datetime
from typing import Any


def answer_question_simple(
    claim: LearningClaim,
    evidence_store: Any,
    embedding_provider: Any,
    top_k: int = 5
) -> LearningClaim:
    """
    Convenience function to answer a question claim.
    
    Args:
        claim: LearningClaim with type=QUESTION
        evidence_store: EvidenceStore instance
        embedding_provider: EmbeddingProvider instance
        top_k: Number of evidence items to retrieve
    
    Returns:
        Updated claim with answer and ANSWERED_WITH_CITATIONS status
    """
    answerer = QuestionAnswerer(
        evidence_store=evidence_store,
        embedding_provider=embedding_provider,
        top_k=top_k
    )
    return answerer.answer_question(claim)
