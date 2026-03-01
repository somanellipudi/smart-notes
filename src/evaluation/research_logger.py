"""Research-grade inference logging and persistence.

Features:
- Non-blocking queue-based logging to JSONL + SQLite
- Per-claim persistence for reproducibility and auditability
- Summary export with Accuracy, ECE, and AUC-AC
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
import json
import logging
from pathlib import Path
import queue
import sqlite3
import threading
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class InferenceLogRecord:
    claim_id: str
    raw_prediction: str
    calibrated_confidence: float
    bin_assignment: str
    latency_ms: float
    hardware_id: str
    gold_label: Optional[str] = None
    is_correct: Optional[int] = None
    domain: str = ""
    dataset: str = ""
    error: str = ""


class ResearchLogger:
    """Asynchronous persistent logger for benchmark inferences."""

    def __init__(
        self,
        output_dir: str,
        run_id: str,
        flush_every: int = 128,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.flush_every = flush_every

        self.jsonl_path = self.output_dir / f"{run_id}_inference.jsonl"
        self.sqlite_path = self.output_dir / "research_logs.sqlite"
        self.summary_csv_path = self.output_dir / f"{run_id}_summary.csv"

        self._queue: queue.Queue = queue.Queue(maxsize=20000)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._thread.start()

    def log_inference(self, record: InferenceLogRecord) -> None:
        """Queue one inference log event without blocking pipeline execution."""
        payload = {
            "timestamp": datetime.now().isoformat(),
            "run_id": self.run_id,
            **record.__dict__,
        }
        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            logger.warning("ResearchLogger queue full; dropping record claim_id=%s", record.claim_id)

    def flush(self, timeout_seconds: float = 10.0) -> None:
        """Wait for queued records to be written."""
        deadline = datetime.now().timestamp() + timeout_seconds
        while not self._queue.empty() and datetime.now().timestamp() < deadline:
            pass

    def close(self) -> None:
        """Stop background writer and flush all logs."""
        self._stop_event.set()
        self._thread.join(timeout=15)

    def export_summary_report(self, csv_path: Optional[str] = None) -> Dict[str, float]:
        """Export headline metrics to CSV for current run_id."""
        conn = sqlite3.connect(self.sqlite_path)
        try:
            rows = conn.execute(
                """
                SELECT calibrated_confidence, is_correct
                FROM inference_logs
                WHERE run_id = ? AND calibrated_confidence IS NOT NULL AND is_correct IS NOT NULL
                """,
                (self.run_id,),
            ).fetchall()
        finally:
            conn.close()

        if not rows:
            summary = {"run_id": self.run_id, "n": 0, "accuracy": 0.0, "ece": 0.0, "auc_ac": 0.0}
        else:
            conf = np.array([float(r[0]) for r in rows], dtype=float)
            corr = np.array([int(r[1]) for r in rows], dtype=float)
            summary = {
                "run_id": self.run_id,
                "n": int(len(rows)),
                "accuracy": float(np.mean(corr)),
                "ece": float(self._compute_ece(corr, conf, n_bins=10)),
                "auc_ac": float(self._compute_auc_ac(corr, conf)),
            }

        target = Path(csv_path) if csv_path else self.summary_csv_path
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["run_id", "n", "accuracy", "ece", "auc_ac"])
            writer.writeheader()
            writer.writerow(summary)

        return summary

    def _writer_loop(self) -> None:
        conn = sqlite3.connect(self.sqlite_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        self._ensure_schema(conn)

        with self.jsonl_path.open("a", encoding="utf-8") as jsonl_file:
            batch: List[Dict] = []
            while not self._stop_event.is_set() or not self._queue.empty():
                try:
                    item = self._queue.get(timeout=0.1)
                    batch.append(item)
                    if len(batch) >= self.flush_every:
                        self._flush_batch(batch, conn, jsonl_file)
                        batch = []
                except queue.Empty:
                    if batch:
                        self._flush_batch(batch, conn, jsonl_file)
                        batch = []

            if batch:
                self._flush_batch(batch, conn, jsonl_file)

        conn.close()

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS inference_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                run_id TEXT NOT NULL,
                claim_id TEXT NOT NULL,
                raw_prediction TEXT,
                gold_label TEXT,
                calibrated_confidence REAL,
                bin_assignment TEXT,
                latency_ms REAL,
                hardware_id TEXT,
                is_correct INTEGER,
                domain TEXT,
                dataset TEXT,
                error TEXT
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_inference_logs_run_id ON inference_logs(run_id)")
        conn.commit()

    def _flush_batch(self, batch: List[Dict], conn: sqlite3.Connection, jsonl_file) -> None:
        for row in batch:
            jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")
        jsonl_file.flush()

        conn.executemany(
            """
            INSERT INTO inference_logs (
                timestamp, run_id, claim_id, raw_prediction, gold_label,
                calibrated_confidence, bin_assignment, latency_ms, hardware_id,
                is_correct, domain, dataset, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    r.get("timestamp"),
                    r.get("run_id"),
                    r.get("claim_id"),
                    r.get("raw_prediction"),
                    r.get("gold_label"),
                    r.get("calibrated_confidence"),
                    r.get("bin_assignment"),
                    r.get("latency_ms"),
                    r.get("hardware_id"),
                    r.get("is_correct"),
                    r.get("domain"),
                    r.get("dataset"),
                    r.get("error"),
                )
                for r in batch
            ],
        )
        conn.commit()

    @staticmethod
    def _compute_ece(matches: np.ndarray, confidences: np.ndarray, n_bins: int = 10) -> float:
        ece = 0.0
        total = len(matches)
        boundaries = np.linspace(0.0, 1.0, n_bins + 1)
        for i in range(n_bins):
            lo, hi = boundaries[i], boundaries[i + 1]
            mask = (confidences >= lo) & (confidences < hi)
            if np.any(mask):
                acc = np.mean(matches[mask])
                conf = np.mean(confidences[mask])
                ece += (np.sum(mask) / total) * np.abs(acc - conf)
        return float(ece)

    @staticmethod
    def _compute_auc_ac(matches: np.ndarray, confidences: np.ndarray) -> float:
        if len(matches) == 0:
            return 0.0
        order = np.argsort(-confidences)
        sorted_matches = matches[order]
        cumulative_correct = np.cumsum(sorted_matches)
        coverage = np.arange(1, len(sorted_matches) + 1) / len(sorted_matches)
        accuracy_curve = cumulative_correct / np.arange(1, len(sorted_matches) + 1)
        return float(np.trapz(accuracy_curve, coverage))
