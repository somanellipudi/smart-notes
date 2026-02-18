"""
CS-Aware Claim Feature Extraction

Extracts specialized tokens and features from computer science claims for verification.
Includes numeric parsing, complexity analysis notation, code pattern detection, and negation handling.
"""

import logging
import re
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class NumericToken:
    """Represents a numeric value extracted from text."""
    value: float
    unit: Optional[str] = None
    context: Optional[str] = None  # e.g., "worst-case", "average"


@dataclass
class ComplexityToken:
    """Represents a complexity notation (Big-O, Big-Theta, Big-Omega)."""
    notation: str  # e.g., "O", "Θ", "Ω"
    expression: str  # e.g., "n log n", "n^2", "1"
    full_form: str  # e.g., "O(n log n)"


@dataclass
class CodeToken:
    """Represents a code pattern or concept."""
    pattern: str
    category: str  # "loop", "recursion", "concurrency", "data_structure", "algorithm", "consensus"
    confidence: float  # 0.0-1.0


class CSClaimFeatureExtractor:
    """Extract CS-specific features from claims and evidence."""
    
    # Numeric patterns
    NUMERIC_PATTERNS = [
        r'\b\d+\b',  # Integers
        r'\b\d+\.\d+\b',  # Decimals
        r'\binfinit(e|y)\b',  # Infinity
    ]
    
    # Complexity notation patterns
    COMPLEXITY_NOTATIONS = {
        'O': r'\bO\s*\(',
        'Θ': r'\bΘ\s*\(|\bTheta\s*\(',
        'Ω': r'\bΩ\s*\(|\bOmega\s*\(',
    }
    
    # Complexity expression patterns
    COMPLEXITY_EXPRESSIONS = [
        r'1\b|const(?:ant)?',
        r'\blog\s*n\b|\bln\s*n\b',
        r'\bn\b(?!\w)',
        r'\bn\s*\*\s*\blog\s*n\b|\bn\s*log\s*n\b|\bn\s*ln\s*n\b',
        r'\bn\s*\*\s*m\b|\bn\s*m\b|\bn\s*x\s*m\b',
        r'\bn\s*\^\s*2\b|\bn\s*squared\b',
        r'\bn\s*\^\s*3\b|\bn\s*cubed\b',
        r'\b2\s*\^\s*n\b|exp(?:onential)?\s*n\b',
        r'\bn!\b|factorial.*n\b',
        r'\bV\s*\+\s*E\b|vertices.*edges\b',
    ]
    
    # Code pattern keywords
    CODE_PATTERNS = {
        'loop': [
            r'\b(for|while|foreach|iterate|iteration)\b',
            r'\bloop\b',
            r'\bcyclic\b',
        ],
        'recursion': [
            r'\b(recursive|recursion|recur)\b',
            r'\bcall.*itself\b',
            r'\bstack\s+(overflow|trace)\b',
        ],
        'concurrency': [
            r'\b(mutex|lock|semaphore|monitor)\b',
            r'\b(thread|process|coroutine|async|await)\b',
            r'\b(race|condition|deadlock|livelock)\b',
            r'\b(critical\s+section|atomic|synchronized)\b',
            r'\bThread-safe|concurrent\b',
        ],
        'data_structure': [
            r'\b(array|list|linked\s+list|tree|graph|heap|stack|queue|hash|trie)\b',
            r'\b(binary\s+search\s+tree|AVL|red-black|B-tree)\b',
            r'\b(priority\s+queue|deque|set|map|dictionary)\b',
        ],
        'algorithm': [
            r'\b(binary\s+search|merge\s+sort|quick\s+sort|bubble\s+sort|heap\s+sort)\b',
            r'\b(BFS|DFS|breadth.*first|depth.*first)\b',
            r'\b(Dijkstra|Bellman-Ford|Floyd-Warshall|Prim|Kruskal)\b',
            r'\b(dynamic\s+programming|memoization|greedy)\b',
            r'\b(divide\s+and\s+conquer|backtracking)\b',
        ],
        'consensus': [
            r'\b(ACID|BASE|CAP\s+theorem|consistency|availability|partition\s+tolerance)\b',
            r'\b(consensus|Paxos|Raft|Byzantine)\b',
            r'\b(eventual\s+consistency|strong\s+consistency)\b',
        ],
    }
    
    # Negation indicators
    NEGATION_MARKERS = [
        r'\b(not|no|never|cannot|can\'t|shouldn\'t|don\'t|doesn\'t)\b',
        r'\bdoes\s+not\b',
        r'\bfails?\b',
        r'\binvalid\b',
        r'\bassumes?\s+no\b',
        r'\bwithout\s+',
        r'\bunless\b',
    ]
    
    # Anchor terms for evidence sufficiency
    ANCHOR_TERMS = {
        'complexity': [
            'worst-case', 'best-case', 'average-case', 'amortized',
            'tight', 'bound', 'asymptotic', 'lower bound', 'upper bound',
            'optimal', 'time complexity', 'space complexity',
        ],
        'definition': [
            'defined as', 'definition', 'is a', 'means', 'refers to',
            'specifically', 'formally', 'iff', 'if and only if', 'equivalently',
            'i.e.', 'that is', 'namely', 'in other words',
        ],
        'code': [
            'invariant', 'precondition', 'postcondition', 'loop invariant',
            'correctness', 'soundness', 'completeness', 'correctness proof',
            'implementation', 'pseudocode', 'algorithm',
        ],
    }
    
    @staticmethod
    def extract_numeric_tokens(text: str) -> List[NumericToken]:
        """
        Extract numeric tokens from text.
        
        Args:
            text: Text to analyze
        
        Returns:
            List of NumericToken objects
        """
        tokens = []
        
        for pattern in CSClaimFeatureExtractor.NUMERIC_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                value_str = match.group(0)
                try:
                    if value_str.lower() in ['infinity', 'infinite']:
                        value = float('inf')
                    else:
                        value = float(value_str)
                    
                    # Extract context (surrounding words)
                    start = max(0, match.start() - 50)
                    end = min(len(text), match.end() + 50)
                    context = text[start:end]
                    
                    # Detect context markers
                    context_type = None
                    if any(marker in context.lower() for marker in ['worst', 'maximum', 'upper']):
                        context_type = 'worst-case'
                    elif any(marker in context.lower() for marker in ['best', 'minimum', 'lower']):
                        context_type = 'best-case'
                    elif 'average' in context.lower():
                        context_type = 'average-case'
                    elif 'amortized' in context.lower():
                        context_type = 'amortized'
                    
                    tokens.append(NumericToken(value=value, context=context_type))
                except ValueError:
                    continue
        
        return tokens
    
    @staticmethod
    def extract_complexity_tokens(text: str) -> List[ComplexityToken]:
        """
        Extract Big-O/Theta/Omega complexity notations from text.
        
        Args:
            text: Text to analyze
        
        Returns:
            List of ComplexityToken objects
        """
        tokens = []
        
        # Find all complexity notations
        for notation, pattern in CSClaimFeatureExtractor.COMPLEXITY_NOTATIONS.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                # Extract the full notation including the expression inside parentheses
                start = match.start()
                paren_start = text.find('(', start)
                if paren_start == -1:
                    continue
                
                # Find matching closing parenthesis
                paren_count = 1
                pos = paren_start + 1
                while pos < len(text) and paren_count > 0:
                    if text[pos] == '(':
                        paren_count += 1
                    elif text[pos] == ')':
                        paren_count -= 1
                    pos += 1
                
                if paren_count == 0:
                    paren_end = pos - 1
                    expression = text[paren_start+1:paren_end].strip()
                    full_form = text[start:paren_end+1]
                    
                    tokens.append(ComplexityToken(
                        notation=notation,
                        expression=expression,
                        full_form=full_form
                    ))
        
        return tokens
    
    @staticmethod
    def extract_code_tokens(text: str) -> List[CodeToken]:
        """
        Extract code patterns and concepts from text.
        
        Args:
            text: Text to analyze
        
        Returns:
            List of CodeToken objects
        """
        tokens = []
        text_lower = text.lower()
        
        for category, patterns in CSClaimFeatureExtractor.CODE_PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    tokens.append(CodeToken(
                        pattern=match.group(0),
                        category=category,
                        confidence=0.8  # Default confidence for pattern matching
                    ))
        
        return tokens
    
    @staticmethod
    def detect_negation(text: str) -> bool:
        """
        Detect if text contains negation markers.
        
        Args:
            text: Text to analyze
        
        Returns:
            True if negation is detected
        """
        for pattern in CSClaimFeatureExtractor.NEGATION_MARKERS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    @staticmethod
    def count_negations(text: str) -> int:
        """
        Count negation markers in text.
        
        Args:
            text: Text to analyze
        
        Returns:
            Count of negation markers
        """
        count = 0
        for pattern in CSClaimFeatureExtractor.NEGATION_MARKERS:
            count += len(re.findall(pattern, text, re.IGNORECASE))
        return count
    
    @staticmethod
    def find_anchor_terms(text: str, anchor_type: str) -> List[str]:
        """
        Find anchor terms in text for a specific claim type.
        
        Args:
            text: Text to analyze
            anchor_type: Type of anchor terms (complexity, definition, code)
        
        Returns:
            List of found anchor terms
        """
        if anchor_type not in CSClaimFeatureExtractor.ANCHOR_TERMS:
            return []
        
        found_terms = []
        text_lower = text.lower()
        
        for term in CSClaimFeatureExtractor.ANCHOR_TERMS[anchor_type]:
            if term in text_lower:
                found_terms.append(term)
        
        return found_terms
    
    @staticmethod
    def extract_all_features(text: str) -> Dict:
        """
        Extract all CS-specific features from text.
        
        Args:
            text: Text to analyze
        
        Returns:
            Dictionary with extracted features
        """
        return {
            'numeric_tokens': CSClaimFeatureExtractor.extract_numeric_tokens(text),
            'complexity_tokens': CSClaimFeatureExtractor.extract_complexity_tokens(text),
            'code_tokens': CSClaimFeatureExtractor.extract_code_tokens(text),
            'has_negation': CSClaimFeatureExtractor.detect_negation(text),
            'negation_count': CSClaimFeatureExtractor.count_negations(text),
            'complexity_anchors': CSClaimFeatureExtractor.find_anchor_terms(text, 'complexity'),
            'definition_anchors': CSClaimFeatureExtractor.find_anchor_terms(text, 'definition'),
            'code_anchors': CSClaimFeatureExtractor.find_anchor_terms(text, 'code'),
        }


# Convenience functions
def extract_numeric_tokens(text: str) -> List[NumericToken]:
    """Extract numeric tokens from text."""
    return CSClaimFeatureExtractor.extract_numeric_tokens(text)


def extract_complexity_tokens(text: str) -> List[ComplexityToken]:
    """Extract complexity notations from text."""
    return CSClaimFeatureExtractor.extract_complexity_tokens(text)


def extract_code_tokens(text: str) -> List[CodeToken]:
    """Extract code pattern tokens from text."""
    return CSClaimFeatureExtractor.extract_code_tokens(text)


def detect_negation(text: str) -> bool:
    """Detect negation in text."""
    return CSClaimFeatureExtractor.detect_negation(text)


def find_anchor_terms(text: str, anchor_type: str) -> List[str]:
    """Find anchor terms for a specific claim type."""
    return CSClaimFeatureExtractor.find_anchor_terms(text, anchor_type)


def extract_all_features(text: str) -> Dict:
    """Extract all CS-specific features from text."""
    return CSClaimFeatureExtractor.extract_all_features(text)
