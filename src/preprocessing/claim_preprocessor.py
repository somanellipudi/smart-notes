"""
Data preprocessing and mining for claim validation.

This module optimizes data before sending to LLMs by:
- Deduplicating similar claims
- Filtering low-value information
- Extracting key entities (dates, names, numbers)
- Segmenting content for targeted evidence retrieval
"""

import logging
import re
from typing import List, Dict, Any, Set
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


class ClaimPreprocessor:
    """Preprocesses claims and source material for efficient validation."""
    
    def __init__(self, similarity_threshold: float = 0.85):
        """
        Initialize claim preprocessor.
        
        Args:
            similarity_threshold: Threshold for considering claims duplicates (0-1)
        """
        self.similarity_threshold = similarity_threshold
    
    def deduplicate_claims(self, claims_text_list: List[str]) -> List[str]:
        """
        Remove duplicate or near-duplicate claims.
        
        Args:
            claims_text_list: List of claim texts
        
        Returns:
            Filtered list of unique claims
        """
        if not claims_text_list:
            return []
        
        unique_claims = []
        for claim in claims_text_list:
            is_duplicate = False
            for existing in unique_claims:
                similarity = self._text_similarity(claim, existing)
                if similarity >= self.similarity_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_claims.append(claim)
        
        removed = len(claims_text_list) - len(unique_claims)
        if removed > 0:
            logger.info(f"Deduplicated {removed} claims, {len(unique_claims)} unique remain")
        
        return unique_claims
    
    def extract_key_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities and key information from text.
        
        Args:
            text: Text to extract from
        
        Returns:
            Dict with entity types as keys and lists as values
        """
        entities = {
            "numbers": [],
            "equations": [],
            "definitions": [],
            "measurements": []
        }
        
        # Extract numbers
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
        entities["numbers"] = list(set(numbers))[:10]  # Limit to top 10
        
        # Extract measurement-like patterns (number + unit)
        measurements = re.findall(r'\b(\d+(?:\.\d+)?)\s*([a-zA-Z°/%]+)\b', text)
        entities["measurements"] = [f"{n}{u}" for n, u in measurements][:10]
        
        # Extract definitions (text after "is", "means", "defined as")
        definitions = re.findall(
            r'(?:is|means|defined as|called|refers to)\s+([^.!?]+)',
            text,
            re.IGNORECASE
        )
        entities["definitions"] = [d.strip() for d in definitions][:5]
        
        # Extract equation-like patterns
        equations = re.findall(r'[a-zA-Z0-9]+\s*=\s*[a-zA-Z0-9\s+\-*/()]+', text)
        entities["equations"] = list(set(equations))[:5]
        
        return entities
    
    def segment_content(self, content: str, max_segment_length: int = 300) -> List[Dict[str, Any]]:
        """
        Segment long content into manageable chunks with metadata.
        
        Args:
            content: Text to segment
            max_segment_length: Maximum characters per segment
        
        Returns:
            List of segments with position info
        """
        segments = []
        sentences = re.split(r'[.!?]\s+', content)
        
        current_segment = ""
        start_pos = 0
        
        for sentence in sentences:
            if not sentence.strip():
                continue
            
            # Add period back
            sentence = sentence + "."
            
            # Check if adding this sentence exceeds limit
            if len(current_segment) + len(sentence) > max_segment_length and current_segment:
                segments.append({
                    "text": current_segment.strip(),
                    "start": start_pos,
                    "length": len(current_segment),
                    "sentence_count": current_segment.count(".")
                })
                current_segment = sentence
                start_pos += len(current_segment)
            else:
                current_segment += " " + sentence if current_segment else sentence
        
        # Add remaining segment
        if current_segment.strip():
            segments.append({
                "text": current_segment.strip(),
                "start": start_pos,
                "length": len(current_segment),
                "sentence_count": current_segment.count(".")
            })
        
        logger.info(f"Segmented content into {len(segments)} chunks")
        return segments
    
    def filter_noise(self, text: str) -> str:
        """
        Remove noise/low-value content from text.
        
        Args:
            text: Text to clean
        
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common filler phrases
        fillers = [
            r'\b(?:um|uh|uhh|err|erm|hmm)\b',
            r'\b(?:like|you know|kind of|sort of)\b',
            r'\b(?:okay|right|good|bad)\b(?=\W|$)',
        ]
        
        for filler in fillers:
            text = re.sub(filler, '', text, flags=re.IGNORECASE)
        
        # Remove repeated characters (more than 2)
        text = re.sub(r'(\w)\1{2,}', r'\1\1', text)
        
        return text.strip()
    
    def rank_evidence_quality(self, evidence_text: str) -> float:
        """
        Rank evidence quality based on characteristics.
        
        Args:
            evidence_text: Text of the evidence
        
        Returns:
            Quality score 0-1
        """
        score = 0.5  # Base score
        
        # Longer evidence is better
        if len(evidence_text) > 100:
            score += 0.1
        if len(evidence_text) > 300:
            score += 0.1
        
        # Evidence with specific entities is better
        has_numbers = bool(re.search(r'\d', evidence_text))
        if has_numbers:
            score += 0.1
        
        # Evidence with causation/explanation is better
        causal_words = ['because', 'therefore', 'thus', 'hence', 'results in', 'causes']
        has_causation = any(w in evidence_text.lower() for w in causal_words)
        if has_causation:
            score += 0.1
        
        # Evidence with examples is better
        has_example = bool(re.search(r'\b(?:example|for instance|such as|like)\b', evidence_text, re.IGNORECASE))
        if has_example:
            score += 0.1
        
        # Cap at 1.0
        return min(score, 1.0)
    
    @staticmethod
    def _text_similarity(text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts (0-1).
        
        Args:
            text1: First text
            text2: Second text
        
        Returns:
            Similarity score 0-1
        """
        # Normalize texts
        t1 = text1.lower().strip()
        t2 = text2.lower().strip()
        
        # Use SequenceMatcher for similarity
        matcher = SequenceMatcher(None, t1, t2)
        return matcher.ratio()
    
    def prepare_for_llm(
        self,
        claim_text: str,
        available_evidence: List[str],
        max_evidence_tokens: int = 500
    ) -> Dict[str, Any]:
        """
        Prepare minimal data for LLM processing.
        
        Args:
            claim_text: The claim to process
            available_evidence: List of evidence snippets
            max_evidence_tokens: Maximum tokens to include
        
        Returns:
            Optimized data structure for LLM
        """
        # Rank evidence
        ranked_evidence = sorted(
            [(e, self.rank_evidence_quality(e)) for e in available_evidence],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Select best evidence (prioritize quality and diverse sources)
        selected_evidence = []
        total_chars = 0
        
        for evidence, quality in ranked_evidence:
            if total_chars + len(evidence) > max_evidence_tokens * 4:  # Approx 4 chars per token
                break
            selected_evidence.append(evidence)
            total_chars += len(evidence)
        
        return {
            "claim": claim_text,
            "evidence": selected_evidence,
            "evidence_count": len(selected_evidence),
            "total_chars": total_chars,
            "average_quality": sum(q for _, q in ranked_evidence[:len(selected_evidence)]) / max(1, len(selected_evidence))
        }
