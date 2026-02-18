import numpy as np

from src.retrieval.evidence_store import EvidenceStore, Evidence


def _vectorize(text: str) -> np.ndarray:
    keywords = ["stack", "lifo", "queue", "fifo"]
    text_lower = text.lower()
    return np.array([text_lower.count(word) for word in keywords], dtype="float32")


def _build_store(chunks: list[str]) -> EvidenceStore:
    store = EvidenceStore(session_id="test_dense_retrieval", embedding_dim=4)
    for idx, chunk in enumerate(chunks):
        ev = Evidence(
            evidence_id="",
            source_id="session_input",
            source_type="notes",
            text=chunk,
            chunk_index=idx,
            char_start=0,
            char_end=len(chunk),
            metadata={"session_id": "test_dense_retrieval"}
        )
        store.add_evidence(ev)

    embeddings = np.vstack([_vectorize(ev.text) for ev in store.evidence])
    for i, ev in enumerate(store.evidence):
        ev.embedding = embeddings[i]
    store.build_index(embeddings=embeddings, embedding_dim=embeddings.shape[1])
    return store


def test_dense_retrieval_finds_stack_lifo():
    chunks = [
        "A stack follows last-in, first-out ordering.",
        "A queue follows first-in, first-out ordering.",
        "Stacks support push and pop operations."
    ]
    store = _build_store(chunks)

    query_embedding = _vectorize("Stack follows LIFO").astype("float32")
    results = store.search(query_embedding=query_embedding, top_k=2, min_similarity=0.0)

    assert results, "Expected non-empty retrieval results"
    top_text = results[0][0].text.lower()
    assert "stack" in top_text
    assert "last-in" in top_text or "lifo" in top_text


def test_dense_retrieval_finds_queue_fifo():
    chunks = [
        "Queues process elements in FIFO order.",
        "Stacks process elements in LIFO order.",
        "Queues allow enqueue and dequeue."
    ]
    store = _build_store(chunks)

    query_embedding = _vectorize("Queue uses FIFO").astype("float32")
    results = store.search(query_embedding=query_embedding, top_k=2, min_similarity=0.0)

    assert results, "Expected non-empty retrieval results"
    top_text = results[0][0].text.lower()
    assert "queue" in top_text
    assert "fifo" in top_text
