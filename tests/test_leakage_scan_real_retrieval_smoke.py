import json
import pytest
import subprocess
from pathlib import Path


@pytest.mark.paper
def test_leakage_scan_real_retrieval_with_tmp_corpus(tmp_path):
    repo_root = Path(__file__).parent.parent

    claims_file = tmp_path / "tmp_claims.jsonl"
    claims_file.write_text(
        "\n".join(
            [
                json.dumps({"id": "c1", "claim": "Binary search runs in O(log n) time on sorted arrays."}),
                json.dumps({"id": "c2", "claim": "TCP guarantees in-order delivery."}),
                json.dumps({"id": "c3", "claim": "Dijkstra supports negative edge weights."}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    corpus_file = tmp_path / "tmp_corpus.jsonl"
    corpus_file.write_text(
        "\n".join(
            [
                json.dumps({"doc_id": "d1", "source_text": "Binary search runs in O log n time on sorted arrays."}),
                json.dumps({"doc_id": "d2", "source_text": "TCP provides reliable and ordered byte-stream delivery."}),
                json.dumps({"doc_id": "d3", "source_text": "Dijkstra algorithm does not support negative weight edges."}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    outdir = tmp_path / "leakage_out"

    result = subprocess.run(
        [
            "python",
            "scripts/leakage_scan.py",
            "--claims",
            str(claims_file),
            "--retrieval_mode",
            "real",
            "--corpus",
            str(corpus_file),
            "--outdir",
            str(outdir),
            "--k",
            "5",
            "--k2",
            "15",
            "--max_claims",
            "3",
        ],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )

    assert result.returncode == 0, f"real retrieval run failed: {result.stderr}\n{result.stdout}"

    report_path = outdir / "leakage_report.json"
    assert report_path.exists()

    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert data["retrieval_mode"] == "real"
    assert Path(data["corpus_path"]).resolve() == corpus_file.resolve()
    assert data["n_claims_scanned"] == 3
    assert len(data["per_claim"]) == 33
