"""
Tests for online evidence search and store building.

These tests focus on the online evidence pipeline without making network calls.
"""

import numpy as np

from src.claims.schema import LearningClaim, ClaimType
from src.retrieval.evidence_store import Evidence
from src.retrieval.online_evidence_search import OnlineEvidenceSearcher, SearchResult, build_online_evidence_store
from src.retrieval.online_retriever import OnlineSpan


def test_build_online_evidence_store_uses_embed_texts(monkeypatch):
    """Ensure online evidence store uses EmbeddingProvider.embed_texts."""

    class DummySearcher:
        def __init__(self, *args, **kwargs):
            pass

        def search_and_retrieve_evidence(self, claim_text, session_id=None):
            return [
                Evidence(
                    evidence_id="",
                    source_id="https://en.wikipedia.org/wiki/Stack_(abstract_data_type)",
                    source_type="online_article",
                    text="Stacks are LIFO data structures.",
                    chunk_index=0,
                    char_start=0,
                    char_end=35,
                    metadata={"origin_url": "https://en.wikipedia.org/wiki/Stack_(abstract_data_type)"},
                    origin="https://en.wikipedia.org/wiki/Stack_(abstract_data_type)"
                )
            ]

    class DummyEmbeddingProvider:
        def __init__(self):
            self.called = False

        def embed_texts(self, texts):
            self.called = True
            return np.ones((len(texts), 3), dtype="float32")

    monkeypatch.setattr(
        "src.retrieval.online_evidence_search.OnlineEvidenceSearcher",
        DummySearcher
    )

    claims = [
        LearningClaim(
            claim_id="c1",
            claim_type=ClaimType.DEFINITION,
            claim_text="A stack is a LIFO data structure."
        )
    ]

    provider = DummyEmbeddingProvider()
    store, stats = build_online_evidence_store(
        session_id="test_session",
        claims=claims,
        embedding_provider=provider,
        max_evidence_per_claim=5
    )

    assert provider.called is True
    assert store.index_built is True
    assert stats["evidence_source"] == "online_only"
    assert len(store.evidence) == 1


def test_search_and_retrieve_filters_allowlist(monkeypatch):
    """Only allowlisted URLs should be fetched."""

    class DummyAllowlist:
        def validate_source(self, url):
            if "wikipedia.org" in url:
                return True, None
            return False, "blocked"

    class DummyRetriever:
        def __init__(self):
            self.last_urls = None

        def search_and_retrieve(self, query, urls):
            self.last_urls = urls
            return [
                OnlineSpan(
                    span_id="s1",
                    source_id=urls[0],
                    text="Stacks are LIFO.",
                    start_char=0,
                    end_char=18,
                    origin_url=urls[0]
                )
            ]

    searcher = OnlineEvidenceSearcher()
    searcher.allowlist = DummyAllowlist()
    searcher.retriever = DummyRetriever()

    def fake_search(_query):
        return [
            SearchResult(title="Wiki", url="https://en.wikipedia.org/wiki/Stack", snippet="..."),
            SearchResult(title="Blocked", url="https://blocked.example.com/stack", snippet="...")
        ]

    monkeypatch.setattr(searcher, "search_duckduckgo", fake_search)

    evidence = searcher.search_and_retrieve_evidence("stack data structure", session_id="s1")

    assert len(evidence) == 1
    assert searcher.retriever.last_urls == ["https://en.wikipedia.org/wiki/Stack"]


def test_search_and_retrieve_no_results_short_circuits(monkeypatch):
    """No search results should return empty evidence and skip retriever."""

    class DummyRetriever:
        def __init__(self):
            self.called = False

        def search_and_retrieve(self, query, urls):
            self.called = True
            return []

    searcher = OnlineEvidenceSearcher()
    searcher.retriever = DummyRetriever()

    monkeypatch.setattr(searcher, "search_duckduckgo", lambda _q: [])

    evidence = searcher.search_and_retrieve_evidence("stack data structure", session_id="s1")

    assert evidence == []
    assert searcher.retriever.called is False


def test_build_online_evidence_store_skips_empty_claims(monkeypatch):
    """Empty claims should be skipped before search."""

    class DummySearcher:
        def __init__(self, *args, **kwargs):
            self.calls = []

        def search_and_retrieve_evidence(self, claim_text, session_id=None):
            self.calls.append(claim_text)
            return []

    class DummyEmbeddingProvider:
        def embed_texts(self, texts):
            return np.zeros((len(texts), 3), dtype="float32")

    monkeypatch.setattr(
        "src.retrieval.online_evidence_search.OnlineEvidenceSearcher",
        DummySearcher
    )

    claims = [
        LearningClaim(
            claim_id="c1",
            claim_type=ClaimType.DEFINITION,
            claim_text=""
        ),
        LearningClaim(
            claim_id="c2",
            claim_type=ClaimType.DEFINITION,
            claim_text="Stacks are LIFO data structures."
        )
    ]

    provider = DummyEmbeddingProvider()
    store, stats = build_online_evidence_store(
        session_id="test_session",
        claims=claims,
        embedding_provider=provider,
        max_evidence_per_claim=5
    )

    assert stats["num_claims_skipped"] == 1
    assert stats["num_claims_searched"] == 2
    assert store is not None
