"""
Test suite for citation-based generation with authoritative sources.

Tests that:
1. LLM receives actual source URLs in context
2. Citations are extracted with real URLs (not topic names)
3. URLs are from authoritative sources
4. Citation display shows clickable links
"""

import pytest
import re
from unittest.mock import MagicMock, patch
from src.reasoning.cited_pipeline import CitedGenerationPipeline
from src.schema.output_schema import Topic, Concept


class TestCitedGeneration:
    """Test citation-based generation with authoritative sources."""
    
    @pytest.fixture
    def pipeline(self):
        """Create a CitedGenerationPipeline instance for testing."""
        with patch('src.reasoning.cited_pipeline.LLMProviderFactory'):
            with patch('src.reasoning.cited_pipeline.EmbeddingProvider'):
                pipeline = CitedGenerationPipeline(model="gpt-4")
                # Mock the provider
                pipeline.provider = MagicMock()
                return pipeline
    
    def test_build_source_library(self, pipeline):
        """Test that source library is correctly built with real URLs."""
        topics = [
            Topic(name="Data Structures", summary="Data structure concepts"),
            Topic(name="Computer Science", summary="CS fundamentals")
        ]
        concepts = [
            Concept(name="Stack", definition="LIFO structure"),
            Concept(name="Queue", definition="FIFO structure"),
            Concept(name="Tree", definition="Hierarchical data structure")
        ]
        
        library = pipeline._build_source_library(topics, concepts)
        
        # Verify library contains documented sources
        assert "Python Official Documentation" in library
        assert "Stack Overflow" in library
        assert "GeeksforGeeks" in library
        assert "Khan Academy" in library
        assert "MDN Web Docs" in library
        
        # Verify URLs are included
        assert "https://docs.python.org/3/" in library
        assert "https://stackoverflow.com/" in library
        assert "https://www.geeksforgeeks.org/" in library
        assert "https://www.khanacademy.org/" in library
        
        # Verify format
        assert "[Source:" in library
        assert library.count("https://") > 10  # At least 10+ URLs
    
    def test_extract_citations_with_urls(self, pipeline):
        """Test extraction of citations with real URLs."""
        # Sample LLM response with citations in standard format
        sample_response = """
        Stack is a LIFO data structure [Source: GeeksforGeeks https://www.geeksforgeeks.org/stack-data-structure/]
        used in many algorithms [Source: Python Docs https://docs.python.org/3/tutorial/datastructures.html].
        
        Common operations include push and pop [Source: StackOverflow https://stackoverflow.com/questions/tagged/stack].
        
        Queues are FIFO structures [Source: Khan Academy https://www.khanacademy.org/computing].
        """
        
        citations = pipeline._extract_citations_with_urls(sample_response)
        
        # Should extract 4 citations
        assert len(citations) == 4
        
        # Verify citation structure
        for cite in citations:
            assert "resource_name" in cite
            assert "url" in cite
            assert "position" in cite
            assert "verified" in cite
            assert "source_type" in cite
            # URL should be valid
            assert cite["url"].startswith("https://")
        
        # Verify specific citations
        geeks_cite = next((c for c in citations if "GeeksforGeeks" in c["resource_name"]), None)
        assert geeks_cite is not None
        assert "geeksforgeeks.org" in geeks_cite["url"]
        assert geeks_cite["source_type"] == "Computer Science Tutorial"
        
        python_cite = next((c for c in citations if "Python" in c["resource_name"]), None)
        assert python_cite is not None
        assert "docs.python.org" in python_cite["url"]
        assert python_cite["source_type"] == "Official Documentation"
        
        stackoverflow_cite = next((c for c in citations if "StackOverflow" in c["resource_name"]), None)
        assert stackoverflow_cite is not None
        assert "stackoverflow.com" in stackoverflow_cite["url"]
        assert stackoverflow_cite["source_type"] == "Community Q&A"
    
    def test_classify_source(self, pipeline):
        """Test that sources are correctly classified by type."""
        test_cases = [
            ("https://docs.python.org/3/", "Official Documentation"),
            ("https://stackoverflow.com/", "Community Q&A"),
            ("https://github.com/", "Code Repository"),
            ("https://en.wikipedia.org/", "Encyclopedia"),
            ("https://www.coursera.org/", "Educational Platform"),
            ("https://arxiv.org/", "Academic Research"),
            ("https://www.geeksforgeeks.org/", "Computer Science Tutorial"),
            ("https://mathworld.wolfram.com/", "Online Resource"),  # Default fallback
        ]
        
        for url, expected_type in test_cases:
            result_type = pipeline._classify_source(url)
            assert result_type == expected_type, f"Failed for {url}: got {result_type}, expected {expected_type}"
    
    def test_verify_and_enrich_citations(self, pipeline):
        """Test that citations are verified against authoritative sources."""
        citations = [
            {"url": "https://docs.python.org/3/", "resource_name": "Python", "source_type": "Official"},
            {"url": "https://stackoverflow.com/", "resource_name": "Stack Overflow", "source_type": "Q&A"},
            {"url": "https://example-random-site.com/", "resource_name": "Random Site", "source_type": "Blog"},
            {"url": "https://www.geeksforgeeks.org/", "resource_name": "GeeksforGeeks", "source_type": "Tutorial"},
        ]
        
        evidence_map = {
            "Stack": [{"source_url": "https://stackoverflow.com/", "title": "Stack Q&A"}],
            "Python": [{"source_url": "https://docs.python.org/3/", "title": "Python Docs"}]
        }
        
        enriched = pipeline._verify_and_enrich_citations(citations, evidence_map)
        
        # Python Docs should be verified (authoritative domain)
        python_cite = enriched[0]
        assert python_cite["verified"] is True
        
        # Stack Overflow should be verified
        so_cite = enriched[1]
        assert so_cite["verified"] is True
        
        # Random site should NOT be verified
        random_cite = enriched[2]
        assert random_cite["verified"] is False
        
        # GeeksforGeeks should be verified
        geeks_cite = enriched[3]
        assert geeks_cite["verified"] is True
    
    def test_citation_url_format(self, pipeline):
        """Test that citations match expected URL format pattern."""
        # Pattern we're looking for: [Source: Name https://url]
        pattern = r'\[Source:\s*([^\]]+?)\s+(https?://[^\s\]]+)\]'
        
        test_text = "[Source: Python Docs https://docs.python.org/3/]"
        matches = list(re.finditer(pattern, test_text))
        
        assert len(matches) == 1
        assert matches[0].group(1).strip() == "Python Docs"
        assert matches[0].group(2).strip() == "https://docs.python.org/3/"
    
    def test_no_duplicate_citations(self, pipeline):
        """Test that duplicate URLs are removed, keeping first occurrence."""
        sample_response = """
        First mention [Source: Stack Overflow https://stackoverflow.com/questions/1]
        Second mention [Source: StackOverflow https://stackoverflow.com/questions/2]
        Third mention [Source: Stack Overflow https://stackoverflow.com/questions/1]
        """
        
        citations = pipeline._extract_citations_with_urls(sample_response)
        
        # Should only have unique URLs
        unique_urls = {c["url"] for c in citations}
        assert len(citations) == len(unique_urls)
        
        # stackoverflow.com URLs should appear only once (first unique)
        so_citations = [c for c in citations if "stackoverflow.com" in c["url"]]
        assert len(so_citations) <= 2  # Two different URLs
    
    def test_source_library_includes_all_domains(self, pipeline):
        """Test that source library covers the authoritative domains."""
        topics = [Topic(name="Test", summary="Test topic")]
        concepts = [Concept(name="Test", definition="Test concept")]
        
        library = pipeline._build_source_library(topics, concepts)
        
        # Check for key domains
        required_domains = [
            "python.org",
            "stackoverflow.com",
            "github.com",
            "wikipedia.org",
            "khanacademy.org",
            "coursera.org",
            "mdn",
            "arxiv.org",
            "geeksforgeeks.org",
            "mathworld.wolfram.com",
            "ocw.mit.edu",
            "leetcode.com"
        ]
        
        for domain in required_domains:
            assert domain in library.lower(), f"Domain {domain} not in source library"
    
    def test_citation_extraction_robustness(self, pipeline):
        """Test citation extraction with edge cases."""
        test_cases = [
            # Normal case
            ("[Source: Python Docs https://docs.python.org/3/]", 1),
            # Multiple citations
            ("[Source: A https://a.com/] text [Source: B https://b.com/]", 2),
            # URL with parameters
            ("[Source: Search https://stackoverflow.com/search?q=python]", 1),
            # No citations
            ("Some text without citations", 0),
            # Malformed (missing URL)
            ("[Source: No URL here]", 0),
            # Malformed (missing https)
            ("[Source: Missing Protocol www.example.com]", 0),
        ]
        
        for text, expected_count in test_cases:
            citations = pipeline._extract_citations_with_urls(text)
            assert len(citations) == expected_count, f"Failed for: {text}"


class TestCitationIntegration:
    """Integration tests for citation-based generation pipeline."""
    
    @pytest.fixture
    def mock_llm_response(self):
        """Sample LLM response with citations from authoritative sources."""
        return """
        ## Data Structures

        ### Stack
        A stack is a Last-In-First-Out (LIFO) data structure [Source: GeeksforGeeks https://www.geeksforgeeks.org/stack-data-structure/]
        where elements are added and removed from the same end [Source: Python Docs https://docs.python.org/3/tutorial/datastructures.html].
        
        Common operations include push and pop [Source: Stack Overflow https://stackoverflow.com/questions/tagged/stack],
        which can be implemented using lists or linked structures [Source: Khan Academy https://www.khanacademy.org/computing].
        
        #### Real-world Applications
        Stacks are used in function calls [Source: MDN Web Docs https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Statements/return],
        in web browsers for back button navigation [Source: Wikipedia https://en.wikipedia.org/wiki/Stack_(abstract_data_type)],
        and in parsing expressions [Source: ArXiv https://arxiv.org/search/?query=expression+parsing].
        """
    
    def test_full_citation_flow(self, mock_llm_response):
        """Test the complete citation extraction and verification flow."""
        from src.reasoning.cited_pipeline import CitedGenerationPipeline
        
        with patch('src.reasoning.cited_pipeline.LLMProviderFactory'):
            with patch('src.reasoning.cited_pipeline.EmbeddingProvider'):
                pipeline = CitedGenerationPipeline(model="gpt-4")
        
        # Extract citations from mock response
        citations = pipeline._extract_citations_with_urls(mock_llm_response)
        
        # Should extract all citations from the response
        assert len(citations) > 0, "Should extract at least one citation"
        
        # All citations should have required fields
        for cite in citations:
            assert cite["resource_name"], "Citation should have resource name"
            assert cite["url"], "Citation should have URL"
            assert cite["url"].startswith("https://"), "Citation URL should use https"
        
        # Verify we have citations from different sources
        domains = {c["url"].split("/")[2] for c in citations}
        assert len(domains) >= 5, f"Should have citations from at least 5 different domains, got {domains}"
        
        # Verify citation types are correctly assigned
        source_types = {c["source_type"] for c in citations}
        assert len(source_types) > 1, "Should have multiple source types"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
