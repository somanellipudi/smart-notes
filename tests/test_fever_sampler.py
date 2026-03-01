import json
from src.evaluation.samplers import sample_jsonl_subset


def test_sample_jsonl_subset_deterministic(tmp_path):
    # Create a small JSONL file with deterministic content
    lines = [json.dumps({"id": i, "claim": f"Claim {i}"}) for i in range(20)]
    in_file = tmp_path / "fever_dev.jsonl"
    out_file1 = tmp_path / "sample1.jsonl"
    out_file2 = tmp_path / "sample2.jsonl"

    in_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    written1 = sample_jsonl_subset(str(in_file), str(out_file1), n=5, seed=42)
    written2 = sample_jsonl_subset(str(in_file), str(out_file2), n=5, seed=42)

    # The two outputs must be identical (deterministic sampling)
    assert written1 == written2
    # Also ensure the output file exists and has 5 lines
    assert out_file1.exists()
    assert len(out_file1.read_text(encoding="utf-8").strip().splitlines()) == 5
