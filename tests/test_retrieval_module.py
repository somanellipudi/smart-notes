import json
import pytest
from pathlib import Path

from src.eval.retrieval_module import build_index, load_index, retrieve_passages


@pytest.mark.paper
def test_retrieval_module_build_load_retrieve_deterministic(tmp_path):
    corpus = tmp_path / "tiny_corpus.jsonl"
    corpus.write_text(
        "\n".join(
            [
                json.dumps({"doc_id": "doc_b", "text": "alpha beta gamma"}),
                json.dumps({"doc_id": "doc_a", "text": "alpha beta delta"}),
                json.dumps({"doc_id": "doc_c", "text": "alpha"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    outdir = tmp_path / "retrieval_index"
    build_index(str(corpus), outdir=str(outdir))
    index = load_index(outdir=str(outdir))

    assert index.manifest["num_docs"] == 3

    results = retrieve_passages("alpha beta", k=3, index=index)

    assert [r["doc_id"] for r in results] == ["doc_a", "doc_b", "doc_c"]
    assert [r["rank"] for r in results] == [1, 2, 3]
    assert results[0]["score"] == 1.0
    assert results[1]["score"] == 1.0
    assert results[2]["score"] == 0.5

    results_again = retrieve_passages("alpha beta", k=3, index=index)
    assert results == results_again
