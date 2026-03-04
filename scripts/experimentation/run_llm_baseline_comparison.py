"""Run LLM baseline comparison on CSClaimBench.

Evaluates GPT-4o, Claude, and Llama (Ollama) on claim verification labels:
VERIFIED, REJECTED, LOW_CONFIDENCE.

Outputs:
- outputs/paper/llm_comparison/llm_baseline_results.json
- outputs/paper/llm_comparison/llm_baseline_predictions.jsonl
- outputs/paper/llm_comparison/llm_baseline_table.md
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import f1_score

import config
from src.evaluation.conformal import expected_calibration_error
from src.evaluation.selective_prediction import compute_auc_risk_coverage, compute_risk_coverage_curve


LABELS = ["VERIFIED", "REJECTED", "LOW_CONFIDENCE"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}


@dataclass
class ModelSpec:
    name: str
    provider: str
    model: str


def build_prompt(claim: str, source_text: str) -> str:
    return (
        "You are evaluating a computer-science factual claim against a provided source text.\n"
        "Return ONLY JSON with keys: label, confidence, rationale.\n"
        "Allowed labels: VERIFIED, REJECTED, LOW_CONFIDENCE.\n"
        "confidence must be a float in [0,1] for confidence that your chosen label is correct.\n"
        "Keep rationale <= 25 words.\n\n"
        f"CLAIM: {claim}\n\n"
        f"SOURCE_TEXT: {source_text}\n"
    )


def extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return None
    return None


def normalize_label(value: Any) -> str:
    if not isinstance(value, str):
        return "LOW_CONFIDENCE"
    v = value.strip().upper().replace(" ", "_").replace("-", "_")
    if v in LABEL_TO_ID:
        return v
    aliases = {
        "SUPPORTED": "VERIFIED",
        "ENTAILMENT": "VERIFIED",
        "ENTAIL": "VERIFIED",
        "REFUTED": "REJECTED",
        "CONTRADICTION": "REJECTED",
        "CONTRADICT": "REJECTED",
        "INSUFFICIENT_EVIDENCE": "LOW_CONFIDENCE",
        "NOT_ENOUGH_INFO": "LOW_CONFIDENCE",
        "NEI": "LOW_CONFIDENCE",
        "NEUTRAL": "LOW_CONFIDENCE",
        "UNCERTAIN": "LOW_CONFIDENCE",
    }
    return aliases.get(v, "LOW_CONFIDENCE")


def normalize_confidence(value: Any) -> float:
    try:
        c = float(value)
    except Exception:
        return 0.5
    return max(0.0, min(1.0, c))


def call_openai(model: str, prompt: str) -> Tuple[str, float, Dict[str, Any]]:
    from openai import OpenAI

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": "You are a strict JSON response engine."},
            {"role": "user", "content": prompt},
        ],
    )
    latency_ms = (time.perf_counter() - t0) * 1000.0
    text = resp.choices[0].message.content or "{}"
    usage = {
        "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(resp.usage, "completion_tokens", 0) or 0,
    }
    return text, latency_ms, usage


def call_anthropic(model: str, prompt: str) -> Tuple[str, float, Dict[str, Any]]:
    import anthropic

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
    t0 = time.perf_counter()
    resp = client.messages.create(
        model=model,
        max_tokens=200,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
    )
    latency_ms = (time.perf_counter() - t0) * 1000.0
    text = ""
    if getattr(resp, "content", None):
        chunks = []
        for item in resp.content:
            chunk_text = getattr(item, "text", "")
            if chunk_text:
                chunks.append(chunk_text)
        text = "\n".join(chunks)
    usage = {
        "prompt_tokens": getattr(resp.usage, "input_tokens", 0) or 0,
        "completion_tokens": getattr(resp.usage, "output_tokens", 0) or 0,
    }
    return text, latency_ms, usage


def call_ollama(model: str, prompt: str) -> Tuple[str, float, Dict[str, Any]]:
    import requests

    url = config.OLLAMA_URL.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a strict JSON response engine."},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": 0},
    }
    t0 = time.perf_counter()
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    latency_ms = (time.perf_counter() - t0) * 1000.0
    text = data.get("message", {}).get("content", "{}")
    usage = {
        "prompt_tokens": int(data.get("prompt_eval_count", 0) or 0),
        "completion_tokens": int(data.get("eval_count", 0) or 0),
    }
    return text, latency_ms, usage


def provider_available(spec: ModelSpec) -> Tuple[bool, str]:
    if spec.provider == "openai":
        if not config.OPENAI_API_KEY:
            return False, "OPENAI_API_KEY missing"
        return True, "ok"
    if spec.provider == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY", ""):
            return False, "ANTHROPIC_API_KEY missing"
        try:
            import anthropic  # noqa: F401
        except Exception:
            return False, "anthropic package missing"
        return True, "ok"
    if spec.provider == "ollama":
        try:
            import requests

            url = config.OLLAMA_URL.rstrip("/") + "/api/tags"
            r = requests.get(url, timeout=3)
            if r.status_code != 200:
                return False, f"Ollama unavailable: HTTP {r.status_code}"
            models = [m.get("name", "") for m in r.json().get("models", [])]
            if spec.model not in models:
                return False, f"Ollama model not found: {spec.model}"
            return True, "ok"
        except Exception as exc:
            return False, f"Ollama unavailable: {exc}"
    return False, f"Unknown provider: {spec.provider}"


def call_model(spec: ModelSpec, prompt: str) -> Tuple[str, float, Dict[str, Any]]:
    if spec.provider == "openai":
        return call_openai(spec.model, prompt)
    if spec.provider == "anthropic":
        return call_anthropic(spec.model, prompt)
    if spec.provider == "ollama":
        return call_ollama(spec.model, prompt)
    raise ValueError(f"Unsupported provider {spec.provider}")


def compute_cost_usd(spec: ModelSpec, prompt_tokens: int, completion_tokens: int) -> float:
    rates = {
        "gpt-4o": (5.0 / 1_000_000, 15.0 / 1_000_000),
        "claude-3-5-sonnet-20241022": (3.0 / 1_000_000, 15.0 / 1_000_000),
    }
    if spec.provider == "ollama":
        return 0.0
    input_rate, output_rate = rates.get(spec.model, (0.0, 0.0))
    return prompt_tokens * input_rate + completion_tokens * output_rate


def load_dataset(path: Path, max_examples: Optional[int]) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    if max_examples is not None:
        return data[:max_examples]
    return data


def evaluate_model(spec: ModelSpec, dataset: List[Dict[str, Any]], out_dir: Path) -> Dict[str, Any]:
    available, reason = provider_available(spec)
    if not available:
        return {
            "model": spec.model,
            "provider": spec.provider,
            "status": "skipped",
            "skip_reason": reason,
        }

    predictions_path = out_dir / f"predictions_{spec.provider}_{spec.model.replace(':', '_')}.jsonl"
    rows: List[Dict[str, Any]] = []
    y_true: List[int] = []
    y_pred: List[int] = []
    conf: List[float] = []
    is_correct: List[int] = []
    latencies: List[float] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost_usd = 0.0
    error_count = 0
    first_error_type: Optional[str] = None
    fatal_error_types = {"NotFoundError", "AuthenticationError", "PermissionDeniedError"}

    for idx, item in enumerate(dataset, start=1):
        claim_text = item.get("generated_claim") or item.get("claim") or ""
        source_text = item.get("source_text") or item.get("source") or ""
        prompt = build_prompt(claim_text, source_text)
        try:
            raw_text, latency_ms, usage = call_model(spec, prompt)
            parsed = extract_json_block(raw_text) or {}
            pred_label = normalize_label(parsed.get("label"))
            confidence = normalize_confidence(parsed.get("confidence", 0.5))
            rationale = str(parsed.get("rationale", "")).strip()
        except Exception as exc:
            pred_label = "LOW_CONFIDENCE"
            confidence = 0.5
            err_type = type(exc).__name__
            rationale = f"ERROR: {err_type}"
            latency_ms = math.nan
            usage = {"prompt_tokens": 0, "completion_tokens": 0}
            error_count += 1
            if first_error_type is None:
                first_error_type = err_type
            if err_type in fatal_error_types:
                rows.append(
                    {
                        "index": idx,
                        "doc_id": item.get("doc_id"),
                        "domain_topic": item.get("domain_topic"),
                        "gold_label": normalize_label(item.get("gold_label", "LOW_CONFIDENCE")),
                        "pred_label": pred_label,
                        "confidence": confidence,
                        "correct": 0,
                        "latency_ms": None,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "cost_usd": 0.0,
                        "rationale": rationale,
                    }
                )
                break

        gold_label = normalize_label(item.get("gold_label", "LOW_CONFIDENCE"))
        pt = int(usage.get("prompt_tokens", 0) or 0)
        ct = int(usage.get("completion_tokens", 0) or 0)
        cost = compute_cost_usd(spec, pt, ct)

        total_prompt_tokens += pt
        total_completion_tokens += ct
        total_cost_usd += cost

        correct = int(pred_label == gold_label)
        y_true.append(LABEL_TO_ID[gold_label])
        y_pred.append(LABEL_TO_ID[pred_label])
        conf.append(confidence)
        is_correct.append(correct)
        if not math.isnan(latency_ms):
            latencies.append(latency_ms)

        row = {
            "index": idx,
            "doc_id": item.get("doc_id"),
            "domain_topic": item.get("domain_topic"),
            "gold_label": gold_label,
            "pred_label": pred_label,
            "confidence": confidence,
            "correct": correct,
            "latency_ms": None if math.isnan(latency_ms) else latency_ms,
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "cost_usd": cost,
            "rationale": rationale,
        }
        rows.append(row)

    with predictions_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if dataset and error_count == len(dataset):
        return {
            "model": spec.model,
            "provider": spec.provider,
            "status": "failed",
            "skip_reason": f"All API calls failed ({first_error_type or 'UnknownError'})",
            "n": len(dataset),
            "predictions_path": str(predictions_path),
        }

    accuracy = float(np.mean(np.array(y_pred) == np.array(y_true))) if y_true else 0.0
    macro_f1 = float(f1_score(y_true, y_pred, average="macro")) if y_true else 0.0
    ece = expected_calibration_error(np.array(conf), np.array(is_correct), num_bins=10) if conf else 0.0
    brier = float(np.mean((np.array(conf) - np.array(is_correct)) ** 2)) if conf else 0.0
    rc_curve = compute_risk_coverage_curve(np.array(conf), np.array(y_pred), np.array(y_true), num_thresholds=100)
    auc_rc = compute_auc_risk_coverage(rc_curve)
    auc_ac = 1.0 - auc_rc

    avg_latency_ms = float(statistics.mean(latencies)) if latencies else None
    p95_latency_ms = float(np.percentile(np.array(latencies), 95)) if latencies else None
    avg_cost_per_claim = total_cost_usd / len(dataset) if dataset else 0.0

    return {
        "model": spec.model,
        "provider": spec.provider,
        "status": "ok",
        "n": len(dataset),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "ece": float(ece),
        "brier": brier,
        "auc_ac": float(auc_ac),
        "avg_latency_ms": avg_latency_ms,
        "p95_latency_ms": p95_latency_ms,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_cost_usd": total_cost_usd,
        "avg_cost_usd_per_claim": avg_cost_per_claim,
        "predictions_path": str(predictions_path),
    }


def build_markdown_table(results: List[Dict[str, Any]]) -> str:
    lines = []
    lines.append("| System | Accuracy | Macro-F1 | ECE | AUC-AC | Avg Latency (ms) | Avg Cost / Claim (USD) | Status |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for r in results:
        label = f"{r.get('model','')} ({r.get('provider','')})"
        if r.get("status") != "ok":
            lines.append(f"| {label} | — | — | — | — | — | — | {r.get('status')}: {r.get('skip_reason','')} |")
            continue
        latency_value = r.get("avg_latency_ms")
        latency_text = f"{latency_value:.1f}" if latency_value is not None else "n/a"
        cost_text = f"${r.get('avg_cost_usd_per_claim', 0.0):.5f}"
        lines.append(
            f"| {label} | {100*r['accuracy']:.1f}% | {r['macro_f1']:.3f} | {r['ece']:.4f} | {r['auc_ac']:.4f} | "
            f"{latency_text} | {cost_text} | ok |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM baseline comparison")
    parser.add_argument("--dataset", default="evaluation/cs_benchmark/csclaimbench_v1.jsonl")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--openai-model", default="gpt-4o")
    parser.add_argument("--anthropic-model", default="claude-sonnet-4-20250514")
    parser.add_argument("--ollama-model", default="llama3.2:3b")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    out_dir = Path("outputs/paper/llm_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(dataset_path, args.max_examples)

    specs = [
        ModelSpec(name="GPT-4o", provider="openai", model=args.openai_model),
        ModelSpec(name="Claude", provider="anthropic", model=args.anthropic_model),
        ModelSpec(name="Llama", provider="ollama", model=args.ollama_model),
    ]

    results = []
    for spec in specs:
        print(f"Running {spec.provider}:{spec.model} on n={len(dataset)}")
        result = evaluate_model(spec, dataset, out_dir)
        results.append(result)
        if result.get("status") == "ok":
            lat_text = f"{result['avg_latency_ms']:.1f}ms" if result.get('avg_latency_ms') is not None else "n/a"
            print(
                f"  accuracy={result['accuracy']:.3f}, ece={result['ece']:.4f}, "
                f"auc_ac={result['auc_ac']:.4f}, latency={lat_text}"
            )
        else:
            print(f"  skipped: {result.get('skip_reason')}")

    summary = {
        "dataset": str(dataset_path),
        "n": len(dataset),
        "timestamp_unix": time.time(),
        "results": results,
    }

    summary_path = out_dir / "llm_baseline_results.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    table_md = build_markdown_table(results)
    table_path = out_dir / "llm_baseline_table.md"
    table_path.write_text(table_md + "\n", encoding="utf-8")

    print("\nLLM baseline comparison table:")
    print(table_md)
    print(f"\nSaved: {summary_path}")
    print(f"Saved: {table_path}")


if __name__ == "__main__":
    main()
