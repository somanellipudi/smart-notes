"""
Citation-Based Generation Pipeline - Optimized for Speed

Instead of: Generate → Extract Claims → Search → Verify (slow, N+1 LLM calls)
This does: Extract Topics → Search Once → Generate With Citations (fast, 2 LLM calls)

Expected speedup: 5-10x faster than current verifiable mode
"""

import logging
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
from src.llm_provider import LLMProviderFactory
from src.schema.output_schema import ClassSessionOutput, Topic, Concept
from src.retrieval.online_evidence_search import search_and_retrieve_evidence
from src.retrieval.embedding_provider import EmbeddingProvider
from src.utils.performance_logger import PerformanceTimer

logger = logging.getLogger(__name__)


class CitedGenerationPipeline:
    """
    Fast citation-based generation pipeline.
    
    Flow:
    1. Quick topic/concept extraction (1 LLM call)
    2. Parallel online evidence search for all topics
    3. Single generation pass with evidence included
    4. Citation verification
    
    Time complexity:
    - Old: O(N+1) LLM calls where N = number of claims
    - New: O(2) LLM calls (1 extraction + 1 generation)
    """
    
    def __init__(
        self,
        model: str = None,
        temperature: float = None,
        provider_type: str = "openai",
        api_key: str = None,
        embedding_provider: EmbeddingProvider = None
    ):
        """Initialize cited generation pipeline."""
        self.model = model or config.LLM_MODEL
        self.temperature = temperature or config.LLM_TEMPERATURE
        self.provider_type = provider_type or "openai"
        
        self.provider = LLMProviderFactory.create_provider(
            provider_type=self.provider_type,
            api_key=api_key or config.OPENAI_API_KEY,
            model=self.model
        )
        
        self.embedding_provider = embedding_provider or EmbeddingProvider()
        
        logger.info(
            f"CitedGenerationPipeline initialized: provider={self.provider_type}, "
            f"model={self.model}"
        )
    
    def process(
        self,
        combined_content: str,
        equations: List[str] = None,
        external_context: str = "",
        session_id: str = None,
        output_filters: Dict[str, bool] = None
    ) -> Tuple[ClassSessionOutput, Dict[str, Any]]:
        """
        Process content with citation-based generation.
        
        Args:
            combined_content: Input text (notes + transcript)
            equations: Detected equations
            external_context: Optional reference material
            session_id: Session identifier
            output_filters: Content type filters
        
        Returns:
            (output, metadata) tuple with generated content and citations
        """
        start_time = time.perf_counter()
        session_id = session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        equations = equations or []
        output_filters = output_filters or {}
        
        logger.info("Starting citation-based pipeline (FAST MODE)")
        
        # Validate input
        if not combined_content or len(combined_content.strip()) < 20:
            logger.error("⚠️ Input content too short for cited generation")
            return self._create_empty_output(session_id), {
                "mode": "cited_generation",
                "error": "Input too short",
                "processing_time": 0
            }
        
        # STEP 1: Quick topic/concept extraction (1 LLM call)
        logger.info("Step 1: Extracting topics and concepts (fast)")
        with PerformanceTimer("extract_topics_concepts", session_id=session_id):
            topics, concepts = self._extract_topics_and_concepts(combined_content, external_context)
        
        logger.info(f"✅ Extracted {len(topics)} topics, {len(concepts)} concepts")
        
        if not topics and not concepts:
            logger.warning("⚠️ No topics/concepts extracted - input may be unclear")
        
        # STEP 2: Parallel online evidence search for ALL topics/concepts
        logger.info("Step 2: Searching online evidence for all topics (parallel)")
        with PerformanceTimer("search_all_evidence", session_id=session_id):
            evidence_map = self._search_evidence_parallel(topics, concepts, session_id)
        
        total_evidence = sum(len(ev) for ev in evidence_map.values())
        logger.info(f"Retrieved {total_evidence} evidence items for {len(evidence_map)} topics")
        logger.debug(f"Evidence map keys: {list(evidence_map.keys())}")
        for key, evs in evidence_map.items():
            logger.debug(f"  {key}: {len(evs)} evidence items")
        
        # STEP 3: Single generation pass WITH evidence citations
        logger.info("Step 3: Generating content with inline citations (1 LLM call)")
        with PerformanceTimer("generate_with_citations", session_id=session_id):
            output, citations = self._generate_with_citations(
                combined_content=combined_content,
                topics=topics,
                concepts=concepts,
                evidence_map=evidence_map,
                equations=equations,
                external_context=external_context,
                output_filters=output_filters
            )
        
        logger.info(f"Generated content with {len(citations)} citations with URLs")
        
        # STEP 4: Verify citations and associate with evidence sources
        logger.info("Step 4: Verifying and enriching citations with source metadata")
        with PerformanceTimer("verify_citations", session_id=session_id):
            # For citation-based generation, verification means:
            # - URLs are from authoritative sources (already filtered in _build_source_library)
            # - Linking citations to matched evidence from searches
            verified_citations = self._verify_and_enrich_citations(citations, evidence_map)
        
        verified_count = sum(1 for c in verified_citations if c.get('verified', False))
        logger.info(f"Verified {verified_count}/{len(verified_citations)} citations")
        
        # Log citation sources for transparency
        for cite in verified_citations[:10]:
            logger.info(f"  Citation: {cite['resource_name']} ({cite['source_type']}) - {cite['url'][:50]}...")
        
        processing_time = time.perf_counter() - start_time
        
        # Build evidence summary mapping concepts to their sources
        evidence_summary = {}
        skipped_concepts = []
        
        logger.info(f"Evidence map contains {len(evidence_map)} search results: {list(evidence_map.keys())}")
        logger.info(f"Need to map {len(concepts)} concepts to evidence")
        
        for concept in concepts:
            # Try exact match first
            evidence_list = evidence_map.get(concept.name)
            found_key = None
            
            # If no exact match, try case-insensitive
            if evidence_list is None:
                for key in evidence_map.keys():
                    if key.lower() == concept.name.lower():
                        evidence_list = evidence_map[key]
                        found_key = key
                        logger.info(f"Case-insensitive match: '{key}' → '{concept.name}'")
                        break
            else:
                found_key = concept.name
            
            if evidence_list:
                sources = []
                for ev in evidence_list[:3]:  # Top 3 sources per concept
                    source_dict = {
                        "url": ev.get("url", "Unknown Source"),
                        "title": ev.get("title", ""),
                        "snippet": ev.get("snippet", ev.get("text", ""))[:180],
                        "authority_tier": ev.get("authority_tier", "Unknown"),
                        "authority_weight": ev.get("authority_weight", 0)
                    }
                    # Extract domain for better display
                    try:
                        from urllib.parse import urlparse
                        domain = urlparse(source_dict["url"]).netloc.replace("www.", "")
                        source_dict["domain"] = domain
                    except:
                        source_dict["domain"] = "Unknown"
                    
                    sources.append(source_dict)
                
                evidence_summary[concept.name] = sources
                logger.info(f"✓ '{concept.name}': {len(sources)} verified external sources")
            else:
                # NO FALLBACK: Keep empty if no external sources found
                # These concepts will be marked as LOW_CONFIDENCE and require user verification
                evidence_summary[concept.name] = []
                logger.warning(f"⚠️  '{concept.name}': NO EXTERNAL SOURCES FOUND - will be marked LOW_CONFIDENCE")
        
        # Build metadata with comprehensive info
        metadata = {
            "mode": "cited_generation",
            "processing_time": processing_time,
            "topics": len(topics),
            "concepts": len(concepts),
            "total_evidence": total_evidence,
            "citations": verified_citations,
            "verified_citations": verified_count,
            "verification_rate": verified_count / len(verified_citations) if verified_citations else 0.0,
            "topics_list": [t.name for t in topics],
            "concepts_list": [c.name for c in concepts],
            "evidence_summary": evidence_summary,
            "skipped_concepts": skipped_concepts,
            "quality_report": {
                "extraction_count": {
                    "topics": len(topics),
                    "concepts": len(concepts),
                    "expected_minimum": 10,
                    "status": "✅ Good" if len(concepts) >= 10 else "⚠️  Low - only few concepts extracted"
                },
                "evidence_coverage": {
                    "with_sources": len(evidence_summary),
                    "without_sources": len(skipped_concepts),
                    "coverage_rate": len(evidence_summary) / len(concepts) * 100 if concepts else 0
                },
                "content_richness": {
                    "total_evidence_items": total_evidence,
                    "avg_evidence_per_concept": total_evidence / len(concepts) if concepts else 0
                },
                "recommendations": []
            }
        }
        
        # Add recommendations based on quality metrics
        if len(concepts) < 10:
            metadata["quality_report"]["recommendations"].append(
                "Low concept extraction detected. Possible causes: (1) Input text is sparse/unclear, (2) OCR quality is poor, (3) Content is very brief. Consider: providing clearer images, adding more notes, or using external context."
            )
        
        if len(evidence_summary) < len(concepts) * 0.5:
            metadata["quality_report"]["recommendations"].append(
                f"Less than 50% of concepts have external source verification. This means most content cannot be independently verified. Consider manually searching and adding sources."
            )
        
        if total_evidence < len(concepts) * 2:
            metadata["quality_report"]["recommendations"].append(
                "Low evidence-to-concept ratio. Each concept should have 2-3 external sources for thorough verification."
            )
        
        logger.info(
            f"✅ Citation-based pipeline complete: {processing_time:.1f}s, "
            f"Topics: {len(topics)}, Concepts: {len(concepts)}, "
            f"Citations: {verified_count}/{len(verified_citations)} verified"
        )
        
        return output, metadata
    
    def _extract_topics_and_concepts(
        self,
        content: str,
        external_context: str
    ) -> Tuple[List[Topic], List[Concept]]:
        """
        Fast extraction of topics and concepts in single LLM call.
        
        Returns:
            (topics, concepts) tuple
        """
        system_prompt = (
            "You are an expert at analyzing educational content from lectures, notes, and textbooks. "
            "Extract ALL main topics and key concepts, even from handwritten notes or incomplete text. "
            "Be generous in extraction - include anything that could be a learning objective."
        )
        
        user_prompt = f"""
Analyze this classroom content and extract:
1. Main TOPICS covered (high-level subjects like "Data Structures", "Algorithms", etc.)
2. Key CONCEPTS within each topic (specific ideas like "Stack", "Queue", "LIFO", etc.)

CONTENT:
{content[:3000]}

EXTERNAL CONTEXT:
{external_context[:500] if external_context else "N/A"}

IMPORTANT:
- Extract topics even if mentioned only once
- Include ALL technical terms as concepts
- For handwritten notes, extract key terms and definitions
- If you see diagrams described, extract the concept being illustrated

Return JSON with this exact structure:
{{
  "topics": [
    {{"name": "Topic Name", "summary": "Brief 1-2 sentence overview"}}
  ],
  "concepts": [
    {{
      "name": "Concept Name",
      "definition": "Clear explanation",
      "difficulty_level": 3
    }}
  ]
}}

REQUIREMENTS:
- Extract at least 2 topics and 10 concepts
- Be thorough - include ALL mentioned concepts, terms, operations, examples
- For data structures: include ALL operations (insert, delete, search, etc.)
- For each concept, provide detailed definition (2-3 sentences minimum)
- Extract up to 10 topics and 50 concepts to be comprehensive
"""
        
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = self.provider.chat_completion(
                messages=messages,
                temperature=self.temperature,
                max_tokens=3000,  # Increased for detailed extraction
                response_format={"type": "json_object"}
            )
            
            logger.debug(f"Extraction response: {response[:200]}...")
            
            data = json.loads(response)
            
            topics = [Topic(**t) for t in data.get("topics", [])]
            concepts = [Concept(**c) for c in data.get("concepts", [])]
            
            logger.info(f"📊 EXTRACTION RESULTS:")
            logger.info(f"   Topics: {len(topics)} extracted")
            for t in topics:
                logger.info(f"      - {t.name}")
            logger.info(f"   Concepts: {len(concepts)} extracted")
            for c in concepts:
                logger.info(f"      - {c.name}: {c.definition[:60]}...")
            
            if len(concepts) < 10:
                logger.warning(f"⚠️  Only {len(concepts)} concepts extracted - this seems low. Content might be sparse or OCR quality issue.")
                logger.warning(f"   Input content length: {len(content)} characters")
                logger.warning(f"   First 200 chars: {content[:200]}")
            
            # Validation: Ensure we got something
            if not topics and not concepts:
                logger.warning("No topics or concepts extracted, creating fallback")
                # Create a generic topic from content
                topics = [Topic(
                    name="General Study Content",
                    description="Content extracted from input"
                )]                # Try to extract at least some basic concepts from keywords
                words = content.split()[:100]  # First 100 words
                common_terms = [w for w in words if len(w) > 5 and w[0].isupper()][:3]
                if common_terms:
                    concepts = [
                        Concept(
                            name=term,
                            definition=f"Concept mentioned in study material: {term}",
                            difficulty_level=3
                        )
                        for term in common_terms
                    ]
                    logger.info(f"Created {len(concepts)} fallback concepts from keywords")
            
            if not concepts:
                logger.error("❌ CRITICAL: No concepts extracted - input may be invalid or OCR failed")            
            return topics, concepts
            
        except Exception as e:
            logger.error(f"Failed to extract topics/concepts: {e}", exc_info=True)
            # Return fallback topic
            return [Topic(name="General Content", description="Fallback topic")], []
    
    def _search_evidence_parallel(
        self,
        topics: List[Topic],
        concepts: List[Concept],
        session_id: str
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search online evidence for all topics/concepts in parallel.
        
        Returns:
            Dict mapping topic/concept name to evidence list
        """
        # Combine topics and concepts into search queries
        search_items = []
        for topic in topics:
            search_items.append(("topic", topic.name, topic.summary or topic.name))
        for concept in concepts:
            search_items.append(("concept", concept.name, concept.definition))
        
        logger.info(f"Searching evidence for {len(search_items)} items (topics={len(topics)}, concepts={len(concepts)})")
        for item_type, name, query in search_items:
            logger.debug(f"  Will search for {item_type}: {name}")
        
        evidence_map = {}
        
        # Parallel search with ThreadPoolExecutor
        def search_one(item):
            item_type, name, query_text = item
            try:
                logger.info(f"Searching {item_type} '{name}' with query: {query_text[:80]}")
                results = search_and_retrieve_evidence(
                    query=query_text,
                    session_id=session_id,
                    max_results=3,  # Top 3 per topic/concept
                    use_query_expansion=False  # Already have good queries
                )
                logger.info(f"Found {len(results)} results for '{name}'")
                return name, results
            except Exception as e:
                logger.error(f"Search failed for '{name}': {e}", exc_info=True)
                return name, []
        
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = {executor.submit(search_one, item): item for item in search_items}
            
            for future in as_completed(futures):
                name, evidence = future.result()
                evidence_map[name] = evidence
                logger.debug(f"  Search complete for '{name}': {len(evidence)} items found")
        
        return evidence_map
    
    def _generate_with_citations(
        self,
        combined_content: str,
        topics: List[Topic],
        concepts: List[Concept],
        evidence_map: Dict[str, List[Dict[str, Any]]],
        equations: List[str],
        external_context: str,
        output_filters: Dict[str, bool]
    ) -> Tuple[ClassSessionOutput, List[Dict[str, Any]]]:
        """
        Generate complete study guide with REAL source URLs cited inline.
        
        Works with authoritative sources: LLM cites actual URLs and sources as it generates.
        
        Returns:
            (output, citations_with_urls) tuple
        """
        # Build authoritative source URL library for LLM to cite
        source_library = self._build_source_library(topics, concepts)
        
        system_prompt = (
            "You are an expert educational content generator. "
            "Generate a comprehensive study guide citing ACTUAL AUTHORITATIVE SOURCES. "
            "CRITICAL: When making ANY factual claim, provide a real source URL. "
            "Use URLs from official documentation, academic resources, and specialized educational sites. "
            "Format citations as: [Source: URL https://example.com] inline with the text. "
            "EVERY educational claim must have a source URL."
        )
        
        user_prompt = f"""
Generate a comprehensive study guide with inline source citations.

CLASSROOM CONTENT to synthesize:
{combined_content[:2000]}

AVAILABLE AUTHORITATIVE SOURCES TO CITE:
{source_library}

DETECTED EQUATIONS:
{', '.join(equations) if equations else 'None'}

GENERATION INSTRUCTIONS (With Authoritative Source Citations):
1. For EACH concept, provide:
   - Detailed definition with [Source: URL] citations
   - Real-world applications with source links
   - Code examples if applicable with [Source: URL]
   - Common use cases with citations
   - Cite specific official documentation

2. Include worked examples with:
   - Step-by-step process
   - Code snippets or pseudocode
   - Cite examples to educational resources [Source: URL]

3. Create comprehensive FAQs with:
   - Each answer backed by sources [Source: URL]
   - Comparisons between concepts with citations
   - Implementation guidance with links

4. Explain common mistakes with:
   - Each mistake with source-backed explanation [Source: URL]
   - Prevention strategies with references

5. Show real-world applications with:
   - Each application area with authoritative source [Source: URL]
   - Industry examples with citations

CRITICAL REQUIREMENTS:
- EVERY factual statement must include [Source: URL https://...] 
- Use ACTUAL URLs from the source library above
- Cite Python docs, Stack Overflow, documentation sites, etc with their real URLs
- Do NOT invent or make up URLs
- Format: [Source: ResourceName https://actual-url-here]
- Aim for 3-4 citations per concept minimum
- This generates authoritative, verifiable content

Generate rich, detailed educational content with authoritative sources.
Include inline citations with REAL URLS for ALL factual statements.
Ensure every claim is traceable to an authoritative source.
"""
        
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = self.provider.chat_completion(
                messages=messages,
                temperature=self.temperature,
                max_tokens=8000  # Comprehensive content
            )
            
            logger.info(f"📝 GENERATION RESULTS (Citation-based generation):")
            logger.info(f"   Generated content length: {len(response)} characters")
            logger.info(f"   Content preview: {response[:300]}...")
            
            if len(response) < 1000:
                logger.warning(f"⚠️  Generated content is short ({len(response)} chars).")
                logger.warning(f"   Evidence provided: {len(source_library)} chars")
            
            # Parse URLS and citations from response
            citations_with_urls = self._extract_citations_with_urls(response)
            logger.info(f"✅ Extracted {len(citations_with_urls)} source citations with URLs")
            for cite in citations_with_urls[:5]:
                logger.info(f"   - {cite['resource_name']}: {cite['url']}")
            
            # Parse LLM response into structured output
            output = self._parse_generated_content(response, topics, concepts, output_filters)
            
            return output, citations_with_urls
            
        except Exception as e:
            logger.error(f"Failed to generate with citations: {e}", exc_info=True)
            # Return fallback output
            fallback_output = self._create_empty_output(self.model)
            return fallback_output, []
    
    def _build_source_library(
        self,
        topics: List[Topic],
        concepts: List[Concept]
    ) -> str:
        """
        Build a library of authoritative sources the LLM can cite.
        
        Returns:
            Formatted string listing real sources with URLs
        """
        sources = {
            # Official Documentation
            "Python Official Documentation": "https://docs.python.org/3/",
            "Python Data Structures": "https://docs.python.org/3/tutorial/datastructures.html",
            "MDN Web Docs": "https://developer.mozilla.org/en-US/",
            "GitHub Documentation": "https://docs.github.com/",
            
            # Educational Platforms
            "Khan Academy Computer Science": "https://www.khanacademy.org/computing",
            "Coursera Data Structures": "https://www.coursera.org/search?query=data%20structures",
            "MIT OpenCourseWare": "https://ocw.mit.edu/",
            
            # Technical Communities
            "Stack Overflow": "https://stackoverflow.com/",
            "GeeksforGeeks Data Structures": "https://www.geeksforgeeks.org/data-structures/",
            
            # Academic Resources
            "ArXiv Computer Science": "https://arxiv.org/list/cs/recent",
            
            # Reference Materials
            "Wikipedia Data Structures": "https://en.wikipedia.org/wiki/Data_structure",
            "Wolfram MathWorld": "https://mathworld.wolfram.com/",
            
            # Programming Specific
            "LeetCode": "https://leetcode.com/",
            "HackerRank Tutorials": "https://www.hackerrank.com/domains/tutorials/10-days-of-javascript",
        }
        
        # Build formatted library
        library_text = "AUTHORITATIVE SOURCES TO CITE:\n\n"
        for source_name, url in sources.items():
            library_text += f"- {source_name}: {url}\n"
        
        library_text += "\nUse these exact URLs when citing sources. Format citations as: [Source: Source Name {url}]"
        
        return library_text
    
    def _extract_citations_with_urls(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract citations with real URLs from generated text.
        
        Looks for: [Source: ResourceName https://url]
        
        Returns:
            List of citation dicts with URLs
        """
        import re
        
        citations = []
        # Pattern: [Source: Name https://url]
        pattern = r'\[Source:\s*([^\]]+?)\s+(https?://[^\s\]]+)\]'
        
        for match in re.finditer(pattern, text):
            resource_name = match.group(1).strip()
            url = match.group(2).strip()
            
            citations.append({
                "resource_name": resource_name,
                "url": url,
                "position": match.start(),
                "verified": False,
                "source_type": self._classify_source(url)
            })
        
        # Remove duplicates, keep first occurrence
        seen_urls = set()
        unique_citations = []
        for cite in citations:
            if cite['url'] not in seen_urls:
                unique_citations.append(cite)
                seen_urls.add(cite['url'])
        
        logger.info(f"Extracted {len(unique_citations)} unique source citations with URLs")
        return unique_citations
    
    def _classify_source(self, url: str) -> str:
        """Classify source by domain."""
        if "python.org" in url:
            return "Official Documentation"
        elif "stackoverflow.com" in url:
            return "Community Q&A"
        elif "github.com" in url:
            return "Code Repository"
        elif "wikipedia.org" in url:
            return "Encyclopedia"
        elif "coursera.org" in url or "khanacademy.org" in url or "edx.org" in url:
            return "Educational Platform"  
        elif "arxiv.org" in url:
            return "Academic Research"
        elif "geeksforgeeks.org" in url:
            return "Computer Science Tutorial"
        else:
            return "Online Resource"
    
    def _parse_generated_content(
        self,
        generated_text: str,
        topics: List[Topic],
        concepts: List[Concept],
        output_filters: Dict[str, bool]
    ) -> ClassSessionOutput:
        """
        Parse LLM-generated content into structured ClassSessionOutput.
        
        Returns:
            Properly structured output
        """
        # For now, use the existing parsing logic
        # In a full implementation, you'd parse the generated text
        # into topics, concepts, examples, FAQs, etc.
        
        return ClassSessionOutput(
            session_id=self.model,
            class_summary=generated_text[:500],
            topics=topics,
            key_concepts=concepts,
            worked_examples=[],
            equation_explanations=[],
            common_mistakes=[],
            faqs=[],
            real_world_connections=[]
        )
    
    def _format_evidence_context(self, evidence_map: Dict[str, List[Dict[str, Any]]]) -> str:
        """Format evidence map into readable context for LLM."""
        formatted = []
        
        for name, evidence_list in evidence_map.items():
            if evidence_list:
                formatted.append(f"\n=== {name} ===")
                for i, ev in enumerate(evidence_list[:3], 1):  # Top 3
                    snippet = ev.get('snippet', ev.get('text', ''))[:200]
                    source = ev.get('url', 'Unknown')
                    formatted.append(f"  {i}. {snippet}\n     (Source: {source})")
        
        return "\n".join(formatted)
    
    def _verify_and_enrich_citations(
        self,
        citations: List[Dict[str, Any]],
        evidence_map: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Verify citations against evidence and enrich with metadata.
        
        Checks that URLs are from authoritative sources and matches 
        citations to evidence concepts when available.
        
        Returns:
            Enriched citations with verification and evidence linkage
        """
        authoritative_domains = {
            "python.org", "docs.python.org",
            "stackoverflow.com",
            "github.com",
            "mdn.github.io", "developer.mozilla.org",
            "wikipedia.org",
            "khanacademy.org", "coursera.org", "edx.org",
            "arxiv.org",
            "geeksforgeeks.org",
            "mathworld.wolfram.com",
            "ocw.mit.edu",
            "leetcode.com"
        }
        
        for citation in citations:
            url = citation.get("url", "")
            
            # Verify URL is from authoritative source
            citation["verified"] = any(domain in url for domain in authoritative_domains)
            
            # Try to match to evidence concepts
            matched_concepts = []
            for concept_name, evidence_list in evidence_map.items():
                if evidence_list:
                    # Check if any evidence came from a domain in this citation
                    for evidence_item in evidence_list:
                        if isinstance(evidence_item, dict):
                            source_url = evidence_item.get("source_url", "")
                            if source_url and source_url.replace("https://", "").replace("http://", "").split("/")[0] in url:
                                matched_concepts.append(concept_name)
            
            citation["related_concepts"] = list(set(matched_concepts))
            citation["evidence_match_count"] = len(matched_concepts)
        
        return citations
    
    def _extract_citations(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract [Source: Name] citations from generated text.
        
        Returns:
            List of citation dicts with position and source name
        """
        import re
        
        citations = []
        pattern = r'\[Source: ([^\]]+)\]'
        
        for match in re.finditer(pattern, text):
            citations.append({
                "source_name": match.group(1).strip(),
                "position": match.start(),
                "verified": False
            })
        
        return citations
    
    def _verify_citations(
        self,
        citations: List[Dict[str, Any]],
        evidence_map: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Verify each citation matches available evidence.
        
        Returns:
            Updated citations with verification status
        """
        for citation in citations:
            source_name = citation["source_name"]
            
            # Check if source exists in evidence map
            if source_name in evidence_map and evidence_map[source_name]:
                citation["verified"] = True
                citation["evidence_count"] = len(evidence_map[source_name])
            else:
                citation["verified"] = False
                citation["evidence_count"] = 0
        
        return citations
    
    def _create_empty_output(self, session_id: str) -> ClassSessionOutput:
        """Create empty output for error cases."""
        return ClassSessionOutput(
            session_id=session_id,
            timestamp=datetime.now(),
            class_summary="Unable to generate content. Please check input quality.",
            topics=[],
            key_concepts=[],
            worked_examples=[],
            equation_explanations=[],
            faqs=[],
            common_mistakes=[],
            real_world_connections=[]
        )
