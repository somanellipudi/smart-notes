"""
Base Agent for evidence-first claim generation.

Abstract base class that enforces evidence requirements and
refusal to generate without sufficient supporting evidence.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

from src.claims.schema import LearningClaim, EvidenceItem, VerificationStatus
from src.llm_provider import LLMProviderFactory

logger = logging.getLogger(__name__)


class AgentRefusalError(Exception):
    """Raised when agent refuses to generate due to insufficient evidence."""
    pass


class BaseAgent(ABC):
    """
    Abstract base agent for evidence-first claim generation.
    
    Enforces:
    1. Evidence requirement before generation
    2. Refusal to generate without sufficient evidence
    3. Clear reasoning for refusal
    
    Subclasses must implement the generate() method.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize agent with configuration.
        
        Args:
            config: Dict with keys:
                - min_evidence_count: Minimum evidence required (default: 1)
                - min_similarity: Minimum evidence similarity (default: 0.2)
                - llm_provider: "ollama" or "openai" (default: "ollama")
                - model: Model name
                - api_key: API key for OpenAI (if using)
                - ollama_url: Ollama URL (if using)
        """
        self.config = config or {}
        self.min_evidence_count = self.config.get("min_evidence_count", 1)
        self.min_similarity = self.config.get("min_similarity", 0.2)
        self.llm_provider = self.config.get("llm_provider", "ollama")
        self.model = self.config.get("model", "mistral")
        self.api_key = self.config.get("api_key", "")
        self.ollama_url = self.config.get("ollama_url", "http://localhost:11434")
        
        logger.info(
            f"BaseAgent initialized: "
            f"min_evidence={self.min_evidence_count}, "
            f"min_similarity={self.min_similarity}, "
            f"provider={self.llm_provider}"
        )
    
    def validate_evidence(
        self,
        evidence: List[EvidenceItem],
        claim_type: str = "definition"
    ) -> bool:
        """
        Validate that evidence meets minimum requirements.
        
        Args:
            evidence: List of evidence items
            claim_type: Type of claim being generated
        
        Returns:
            True if evidence is sufficient, False otherwise
        """
        if not evidence:
            logger.debug(f"Evidence validation failed: no evidence provided")
            return False
        
        if len(evidence) < self.min_evidence_count:
            logger.debug(
                f"Evidence validation failed: "
                f"only {len(evidence)} evidence items, "
                f"minimum {self.min_evidence_count} required"
            )
            return False
        
        avg_similarity = sum(e.similarity for e in evidence) / len(evidence)
        if avg_similarity < self.min_similarity:
            logger.debug(
                f"Evidence validation failed: "
                f"average similarity {avg_similarity:.2f} < {self.min_similarity}"
            )
            return False
        
        logger.debug(
            f"Evidence validation passed: "
            f"{len(evidence)} items, avg_similarity={avg_similarity:.2f}"
        )
        return True
    
    @abstractmethod
    def generate(
        self,
        claim: LearningClaim,
        evidence: List[EvidenceItem]
    ) -> str:
        """
        Generate claim text from evidence.
        
        Must be implemented by subclasses.
        
        Args:
            claim: LearningClaim with empty claim_text
            evidence: List of supporting EvidenceItem objects
        
        Returns:
            Generated claim text
        
        Raises:
            AgentRefusalError: If evidence is insufficient
        """
        pass
    
    def process_claim(
        self,
        claim: LearningClaim,
        evidence: List[EvidenceItem]
    ) -> LearningClaim:
        """
        Template method: validate evidence → generate text → return updated claim.
        
        This is the main entry point for processing claims.
        
        Args:
            claim: LearningClaim with empty text
            evidence: List of supporting evidence
        
        Returns:
            Updated LearningClaim with generated text (or unmodified if evidence insufficient)
        
        Raises:
            AgentRefusalError: If evidence validation fails
        """
        # Step 1: Validate evidence
        if not self.validate_evidence(evidence, claim.claim_type):
            error_msg = (
                f"Agent refuses to generate {claim.claim_type} claim: "
                f"insufficient evidence (need {self.min_evidence_count}, "
                f"have {len(evidence)})"
            )
            logger.warning(error_msg)
            raise AgentRefusalError(error_msg)
        
        # Step 2: Generate text
        try:
            generated_text = self.generate(claim, evidence)
            
            # Step 3: Update claim
            claim.claim_text = generated_text
            claim.evidence_ids = [f"evidence_{i}" for i in range(len(evidence))]
            claim.evidence_objects = evidence
            
            logger.info(
                f"Successfully generated {claim.claim_type} claim "
                f"({len(generated_text)} chars, from {len(evidence)} evidence items)"
            )
            
            return claim
        
        except Exception as e:
            error_msg = f"Generation failed for {claim.claim_type} claim: {str(e)}"
            logger.error(error_msg)
            raise AgentRefusalError(error_msg)
    
    def _get_llm_provider(self):
        """Get configured LLM provider."""
        return LLMProviderFactory.create_provider(
            provider_type=self.llm_provider,
            api_key=self.api_key,
            ollama_url=self.ollama_url,
            model=self.model
        )
    
    def _call_llm(self, prompt: str, max_tokens: int = 200) -> str:
        """
        Call configured LLM with prompt.
        
        Args:
            prompt: Text prompt
            max_tokens: Maximum tokens in response
        
        Returns:
            LLM response text
        """
        try:
            provider = self._get_llm_provider()
            response = provider.complete(prompt, max_tokens=max_tokens)
            return response.strip()
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise
