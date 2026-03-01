"""Lightweight evaluation runner for baselines and verifiable_full mode.

This module provides a reproducible, deterministic evaluation on a small
synthetic dataset and writes metrics and figures to outputs/benchmark_results/latest/.
"""
from __future__ import annotations

# ensure workspace root is on PYTHONPATH when executed directly
import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import json
import os
import random
from pathlib import Path
from typing import List

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
from sklearn.metrics import brier_score_loss

from src.evaluation import plots
from src.config.verification_config import VerificationConfig


def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def _synthetic_dataset(n=200, seed=42):
    _seed_everything(seed)
    # Labels: 0=NEI,1=REFUTED,2=SUPPORTED (simulate balanced)
    labels = np.random.choice([0, 1, 2], size=n, p=[0.3, 0.35, 0.35])
    claims = [f"Synthetic claim {i}" for i in range(n)]
    return claims, labels


def _simulate_predictions(labels: np.ndarray, mode: str, cfg: VerificationConfig):
    n = len(labels)
    rng = np.random.RandomState(cfg.random_seed)
    # Simulate retrieval score and NLI entailment probabilities
    sim_scores = rng.rand(n)
    nli_probs = rng.rand(n)

    preds = []
    confidences = []
    for i in range(n):
        if mode == "baseline_retriever":
            pred = 2 if sim_scores[i] > cfg.retriever_threshold else 0
            conf = sim_scores[i]
        elif mode == "baseline_nli":
            pred = 2 if nli_probs[i] > cfg.nli_positive_threshold else (1 if nli_probs[i] < cfg.nli_negative_threshold else 0)
            conf = nli_probs[i]
        elif mode == "baseline_rag_nli":
            # combine sim and nli
            score = 0.5 * sim_scores[i] + 0.5 * nli_probs[i]
            pred = 2 if score > cfg.rag_positive_threshold else (1 if score < cfg.rag_negative_threshold else 0)
            conf = score
        else:  # verifiable_full
            # emulate ensemble + temperature scaling
            raw = 0.4 * sim_scores[i] + 0.6 * nli_probs[i]
            # temperature
            tau = cfg.temperature_init
            calibrated = 1.0 / (1.0 + np.exp(- (np.log(raw + 1e-9) / tau))) if raw > 0 else raw
            pred = 2 if calibrated > cfg.verified_confidence_threshold else (1 if calibrated < cfg.rejected_confidence_threshold else 0)
            conf = float(calibrated)
        preds.append(int(pred))
        confidences.append(float(conf))
    return np.array(preds), np.array(confidences)


def compute_ece(confidences: np.ndarray, correct: np.ndarray, num_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    bin_idxs = np.digitize(confidences, bins) - 1
    ece = 0.0
    for i in range(num_bins):
        mask = bin_idxs == i
        if mask.sum() == 0:
            continue
        conf_bin = confidences[mask].mean()
        acc_bin = correct[mask].mean()
        ece += (mask.sum() / len(confidences)) * abs(acc_bin - conf_bin)
    return float(ece)


def run(mode: str = "verifiable_full", output_dir: str = "outputs/benchmark_results/latest") -> dict:
    cfg = VerificationConfig.from_env()
    _seed_everything(cfg.random_seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    claims, labels = _synthetic_dataset(n=300, seed=cfg.random_seed)

    # For simulated modes, skip expensive model imports.
    # For verifiable_full, try to use real pipeline.
    use_real = False
    if mode in ["baseline_retriever", "baseline_nli", "baseline_rag_nli"]:
        # Use fast simulated predictions
        preds, confidences = _simulate_predictions(labels, mode, cfg)
    else:
        # Try to wire real retrieval + NLI pipeline for verifiable_full
        try:
            from src.retrieval.semantic_retriever import SemanticRetriever
            from src.claims.nli_verifier import NLIVerifier

            retriever = SemanticRetriever(seed=cfg.random_seed)
            combined = "\n\n".join([f"Evidence for claim {i}: {c}" for i, c in enumerate(claims)])
            retriever.index_sources(external_context=combined)
            nli = NLIVerifier()
            use_real = True
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Real retriever/NLI unavailable ({e}); falling back to simulated predictions")

    if not use_real:
        preds, confidences = _simulate_predictions(labels, mode, cfg)
    else:
        preds = []
        confidences = []
        for claim in claims:
            candidates = retriever.retrieve(
                claim_text=claim,
                top_k=cfg.top_k_retrieval,
                rerank_top_n=cfg.top_k_rerank,
                min_similarity=0.0,
            )

            if not candidates:
                preds.append(0)
                confidences.append(0.0)
                continue

            # Run NLI on all candidates
            pairs = [(claim, span.text) for span in candidates]
            try:
                results, scores = nli.verify_batch_with_scores(pairs)
            except Exception:
                # Fallback to verify_batch if method missing
                results = nli.verify_batch(pairs)
                scores = np.array([[r.entailment_prob, r.contradiction_prob, r.neutral_prob] for r in results])

            entail_probs = scores[:, 0]
            max_entail = float(entail_probs.max())

            # Aggregate decision using thresholds from config
            if max_entail > cfg.verified_confidence_threshold:
                pred = 2
            elif max_entail < cfg.rejected_confidence_threshold:
                pred = 1
            else:
                pred = 0

            preds.append(int(pred))
            confidences.append(float(max_entail))

        preds = np.array(preds)
        confidences = np.array(confidences)

    # classification metrics
    acc = float(accuracy_score(labels, preds))
    macro_f1 = float(f1_score(labels, preds, average="macro"))
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average=None, zero_division=0)
    cm = confusion_matrix(labels, preds)

    correct = (labels == preds).astype(float)
    ece = compute_ece(confidences, correct)
    brier = float(np.mean((confidences - correct) ** 2))

    # risk-coverage: sort by confidence descending, compute cumulative accuracy
    sorted_idx = np.argsort(-confidences)
    sorted_correct = correct[sorted_idx]
    coverage_vals = np.arange(1, len(sorted_correct) + 1) / len(sorted_correct)
    accuracy_vals = np.cumsum(sorted_correct) / np.arange(1, len(sorted_correct) + 1)
    auc_rc = float(np.trapz(accuracy_vals, coverage_vals))

    # Retrieval metrics (simulated): assume higher sim_scores -> better rank
    # compute recall@k and MRR@k using deterministic mapping from sim_scores
    def compute_retrieval_metrics(sim_scores, top_k):
        # map sim_score in [0,1] to a rank in [1, top_k]
        ranks = (1 + (1.0 - sim_scores) * (top_k - 1)).astype(int)
        # ensure bounds
        ranks = np.clip(ranks, 1, top_k)
        recall_at = {}
        for k in [1, min(5, top_k), top_k]:
            recall_at[f"recall@{k}"] = float(np.mean(ranks <= k))
        # MRR (mean reciprocal rank)
        rr = 1.0 / ranks
        mrr_at_k = float(np.mean(rr))
        return recall_at, mrr_at_k

    # if simulated mode, we have confidences as proxy for sim_scores; compute retrieval metrics
    recall_at_k = {}
    mrr_at_k = None
    try:
        recall_at_k, mrr_at_k = compute_retrieval_metrics(confidences, cfg.top_k_retrieval)
    except Exception:
        recall_at_k = {}
        mrr_at_k = None

    metrics = {
        "mode": mode,
        "n": len(labels),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class_precision": prec.tolist(),
        "per_class_recall": rec.tolist(),
        "per_class_f1": f1.tolist(),
        "confusion_matrix": cm.tolist(),
        "ece": ece,
        "brier_score": brier,
        # risk-coverage data for selective prediction analysis
        "risk_coverage": {
            "coverage": coverage_vals.tolist(),
            "accuracy": accuracy_vals.tolist(),
            "auc": auc_rc,
        },
        "retrieval": {"recall_at": recall_at_k, "mrr": mrr_at_k},
        "used_real": use_real,
    }

    # save metrics
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # save markdown summary
    md = [f"# Evaluation summary ({mode})\n", "\n"]
    md.append(f"- examples: {len(labels)}\n")
    md.append(f"- accuracy: {acc:.4f}\n")
    md.append(f"- macro_f1: {macro_f1:.4f}\n")
    md.append(f"- ECE: {ece:.4f}\n")
    with open(out / "metrics.md", "w") as f:
        f.writelines(md)

    # plots
    plots.plot_reliability_diagram(confidences, (labels == preds).astype(int), str(out / "figures" / "reliability.png"))
    plots.plot_confusion_matrix(cm, ["NEI", "REFUTED", "SUPPORTED"], str(out / "figures" / "confusion.png"))
    # generate risk-coverage plot
    plots.plot_risk_coverage(coverage_vals, accuracy_vals, str(out / "figures" / "risk_coverage.png"))

    # record metadata
    metadata = {
        "mode": mode,
        "git_commit": os.getenv("GIT_COMMIT", "unknown"),
        "seed": cfg.random_seed,
        "used_real": use_real,
    }
    with open(out / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="verifiable_full", choices=["baseline_retriever", "baseline_nli", "baseline_rag_nli", "verifiable_full"])
    parser.add_argument("--out", default="outputs/benchmark_results/latest")
    args = parser.parse_args()
    run(mode=args.mode, output_dir=args.out)
