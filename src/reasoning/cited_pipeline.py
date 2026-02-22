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
        
        logger.info(f"Generated content with {len(citations)} citations")
        
        # STEP 4: Verify citations match evidence
        logger.info("Step 4: Verifying citations")
        with PerformanceTimer("verify_citations", session_id=session_id):
            verified_citations = self._verify_citations(citations, evidence_map)
        
        verified_count = sum(1 for c in verified_citations if c['verified'])
        logger.info(f"Verified {verified_count}/{len(verified_citations)} citations")
        
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
        Generate complete study guide with inline citations.
        
        Returns:
            (output, citations) tuple
        """
        # Format evidence for prompt
        evidence_context = self._format_evidence_context(evidence_map)
        
        system_prompt = (
            "You are an expert educational content generator. "
            "Generate a comprehensive study guide using ONLY the provided evidence sources. "
            "IMPORTANT: Cite sources inline using [Source: Topic/Concept Name] format. "
            "Do NOT make claims without citing the relevant source."
        )
        
        user_prompt = f"""
Generate a comprehensive study guide based on this classroom content.

CLASSROOM CONTENT:
{combined_content[:2000]}

VERIFIED EVIDENCE SOURCES:
{evidence_context}

DETECTED EQUATIONS:
{', '.join(equations) if equations else 'None'}

INSTRUCTIONS:
1. For EACH concept extracted, provide:
   - Complete detailed definition (3-4 sentences minimum)
   - Real-world applications and examples
   - Common use cases and scenarios
   - Important properties or characteristics
   - Cite sources: [Source: ConceptName]

2. Include DETAILED worked examples showing:
   - Step-by-step implementation
   - Code snippets or pseudocode where applicable
   - Expected output and edge cases

3. Create comprehensive FAQs covering:
   - Common questions students ask
   - Differences between similar concepts
   - When to use which approach

4. Explain common mistakes and how to avoid them

5. Show real-world connections and applications

6. CRITICAL: Every factual claim MUST have [Source: SourceName] citation
7. This is a comprehensive study guide - aim for depth and completeness
8. Generate at least 3-4 paragraphs per concept

Generate rich, detailed educational content organized by topics and concepts.
Include inline citations for ALL factual statements.
"""
        
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = self.provider.chat_completion(
                messages=messages,
                temperature=self.temperature,
                max_tokens=8000  # Significantly increased for comprehensive content
            )
            
            logger.info(f"📝 GENERATION RESULTS:")
            logger.info(f"   Generated content length: {len(response)} characters")
            logger.info(f"   Content preview: {response[:300]}...")
            
            if len(response) < 1000:
                logger.warning(f"⚠️  Generated content is very short ({len(response)} chars). This will result in poor quality study guide.")
                logger.warning(f"   Evidence provided: {len(evidence_context)} chars, Concepts: {len(concepts)}")
            
            # Parse citations from response
            citations = self._extract_citations(response)
            logger.info(f"   Citations found: {len(citations)}")
            
            # Build ClassSessionOutput with correct field names
            output = ClassSessionOutput(
                session_id=f"cited_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                timestamp=datetime.now(),
                class_summary=response[:500] if len(response) >= 50 else response + " (Generated with cited sources)",
                topics=topics,
                key_concepts=concepts,
                worked_examples=[],
                equation_explanations=[],
                faqs=[],
                common_mistakes=[],
                real_world_connections=[]
            )
            
            return output, citations
            
        except Exception as e:
            logger.error(f"Failed to generate with citations: {e}")
            # Return empty output with valid class_summary
            return ClassSessionOutput(
                session_id=f"cited_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                timestamp=datetime.now(),
                class_summary="Error generating content with citations. Please try again.",
                topics=[],
                key_concepts=[],
                worked_examples=[],
                equation_explanations=[],
                faqs=[],
                common_mistakes=[],
                real_world_connections=[]
            ), []
    
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
