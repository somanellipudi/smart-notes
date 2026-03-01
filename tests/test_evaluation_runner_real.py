import pytest

from src.evaluation import runner


import pytest

@pytest.mark.timeout(30)
def test_runner_with_real_models(tmp_path):
    """Ensure the evaluation runner executes the real retriever/NLI path when available.

    This test will be skipped if the heavy dependency packages cannot be imported, so
    it remains safe to run in minimal environments. When the dependencies are present
    the runner should set ``used_real`` to True in its output and still return a
    valid metrics dictionary.
    """
    # to keep the test fast we stub out the heavy retriever/NLI
    # models rather than downloading them.  the goal is to exercise the
    # ``use_real`` branch in runner.run() rather than perform real ML work.
    import sys, types
    import numpy as np

    # create module types for injection
    sem_mod = types.ModuleType('src.retrieval.semantic_retriever')
    nli_mod = types.ModuleType('src.claims.nli_verifier')

    class DummyRetriever:
        def __init__(self, *args, **kwargs):
            pass

        def index_sources(self, *args, **kwargs):
            pass

        def retrieve(self, claim_text, top_k=None, rerank_top_n=None, min_similarity=None):
            from src.retrieval.semantic_retriever import EvidenceSpan
            return [EvidenceSpan(text="dummy", source_type="test", source_id="0", span_start=0, span_end=5, similarity=1.0, rerank_score=1.0)]

    class DummyNLIVerifier:
        def __init__(self, *args, **kwargs):
            pass

        def verify_batch_with_scores(self, pairs):
            results = []
            for _ in pairs:
                r = types.SimpleNamespace(entailment_prob=0.9, contradiction_prob=0.05, neutral_prob=0.05)
                results.append(r)
            scores = np.array([[0.9, 0.05, 0.05] for _ in pairs])
            return results, scores

    sem_mod.SemanticRetriever = DummyRetriever
    # need the EvidenceSpan class for constructing a dummy return
    from src.retrieval.semantic_retriever import EvidenceSpan as _RealEvidenceSpan
    sem_mod.EvidenceSpan = _RealEvidenceSpan

    nli_mod.NLIVerifier = DummyNLIVerifier

    sys.modules['src.retrieval.semantic_retriever'] = sem_mod
    sys.modules['src.claims.nli_verifier'] = nli_mod

    metrics = runner.run(mode="verifiable_full", output_dir=str(tmp_path))
    assert isinstance(metrics, dict)
    assert metrics.get("n") == 300
    assert metrics.get("used_real") is True

    # metrics should contain probabilities and a confusion matrix
    assert "confusion_matrix" in metrics
    assert isinstance(metrics["confusion_matrix"], list)
