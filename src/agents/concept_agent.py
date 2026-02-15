"""
Concept Agent for generating definition claims from evidence.

Specializes in DEFINITION claims and synthesizes evidence into
clear, pedagogically sound definitions.
"""

import logging
from typing import List, Optional, Dict, Any

from src.claims.schema import LearningClaim, EvidenceItem
from src.agents.base import BaseAgent, AgentRefusalError

logger = logging.getLogger(__name__)


class ConceptAgent(BaseAgent):
    """
    Agent for generating DEFINITION claims from evidence.
    
    Synthesizes multiple evidence sources into concise definitions
    that are clear and pedagogically appropriate for the education level.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ConceptAgent.
        
        Args:
            config: Configuration dict (see BaseAgent)
        """
        super().__init__(config)
        self.max_definition_length = self.config.get("max_definition_length", 200)
    
    def generate(
        self,
        claim: LearningClaim,
        evidence: List[EvidenceItem]
    ) -> str:
        """
        Generate a definition claim from evidence.
        
        Args:
            claim: LearningClaim with empty claim_text
            evidence: List of supporting EvidenceItem objects
        
        Returns:
            Generated definition text
        
        Raises:
            AgentRefusalError: If evidence is insufficient
        """
        # Validate evidence
        if not self.validate_evidence(evidence):
            raise AgentRefusalError(
                f"Cannot generate definition: insufficient evidence "
                f"({len(evidence)} items, min {self.min_evidence_count})"
            )
        
        # Get claim metadata
        draft_text = claim.metadata.get("draft_text", "")
        concept_name = claim.metadata.get("concept_name", draft_text)
        difficulty_level = claim.metadata.get("difficulty_level", "intermediate")
        
        # Format evidence for prompt
        evidence_text = self._format_evidence(evidence)
        
        # Build prompt
        prompt = self._build_prompt(concept_name, evidence_text, difficulty_level)
        
        # Call LLM
        definition = self._call_llm(prompt, max_tokens=self.max_definition_length)
        
        if not definition or len(definition) < 10:
            raise AgentRefusalError("Generated definition too short or empty")
        
        logger.info(
            f"Generated definition for '{concept_name}': "
            f"{len(definition)} chars"
        )
        
        return definition
    
    def _format_evidence(self, evidence: List[EvidenceItem]) -> str:
        """
        Format evidence items into readable text.
        
        Args:
            evidence: List of EvidenceItem objects
        
        Returns:
            Formatted evidence text
        """
        formatted = []
        for i, item in enumerate(evidence, 1):
            snippet = item.snippet[:150].strip()  # Truncate long snippets
            similarity = f"{item.similarity:.1%}"
            formatted.append(f"{i}. [{similarity}] {snippet}")
        
        return "\n".join(formatted)
    
    def _build_prompt(
        self,
        concept_name: str,
        evidence_text: str,
        difficulty_level: str
    ) -> str:
        """
        Build prompt for definition generation.
        
        Args:
            concept_name: Name of concept to define
            evidence_text: Formatted evidence
            difficulty_level: Educational level (beginner/intermediate/advanced)
        
        Returns:
            Prompt text
        """
        prompt = f"""You are an expert educational content creator generating clear, 
concise definitions for students.

Concept: {concept_name}
Difficulty Level: {difficulty_level}

Supporting Evidence:
{evidence_text}

Generate a clear, concise definition of "{concept_name}" that:
- Uses plain language appropriate for {difficulty_level} learners
- Synthesizes the provided evidence
- Is 1-3 sentences (max {self.max_definition_length} characters)
- Avoids repetition and jargon where possible
- Is pedagogically sound

Definition of {concept_name}:"""
        
        return prompt
    
    def generate_batch(
        self,
        claims: List[LearningClaim],
        evidence_map: Dict[str, List[EvidenceItem]]
    ) -> List[LearningClaim]:
        """
        Generate definitions for multiple claims.
        
        Args:
            claims: List of LearningClaim objects
            evidence_map: Dict mapping claim_id to List[EvidenceItem]
        
        Returns:
            List of updated claims with generated text
        """
        updated_claims = []
        
        for claim in claims:
            evidence = evidence_map.get(claim.claim_id, [])
            
            try:
                updated_claim = self.process_claim(claim, evidence)
                updated_claims.append(updated_claim)
            except AgentRefusalError as e:
                logger.warning(f"Skipping claim {claim.claim_id}: {e}")
                updated_claims.append(claim)  # Keep original claim
            except Exception as e:
                logger.error(f"Error generating claim {claim.claim_id}: {e}")
                updated_claims.append(claim)
        
        return updated_claims
