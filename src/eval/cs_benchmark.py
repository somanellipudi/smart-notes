"""Importable CS benchmark ablation runner used by CLI wrappers and tests."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.evaluation.cs_benchmark_runner import CSBenchmarkRunner

logger = logging.getLogger(__name__)


class AblationRunner:
    """Run ablation studies for the CS benchmark pipeline."""

    def __init__(
        self,
        dataset_path: str = "evaluation/cs_benchmark/cs_benchmark_dataset.jsonl",
        output_dir: str = "evaluation/results",
        sample_size: Optional[int] = None,
        seed: int = 42,
        verify_threshold: Optional[float] = None,
        low_conf_threshold: Optional[float] = None,
        device: str = "cpu",
        batch_size: int = 32,
    ) -> None:
        self.dataset_path = dataset_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.detailed_dir = self.output_dir / "detailed_results"
        self.detailed_dir.mkdir(parents=True, exist_ok=True)

        self.sample_size = sample_size
        self.seed = seed
        self.verify_threshold = verify_threshold
        self.low_conf_threshold = low_conf_threshold
        self.device = device
        self.batch_size = batch_size
        self.results: List[Dict[str, Any]] = []

    def get_ablation_configs(self) -> Dict[str, Dict[str, Any]]:
        configs: Dict[str, Dict[str, Any]] = {
            "00_no_verification": {
                "use_retrieval": False,
                "use_nli": False,
                "use_ensemble": False,
                "use_cleaning": False,
                "use_artifact_persistence": False,
                "use_batch_nli": True,
                "use_online_authority": False,
            },
            "01a_retrieval_only": {
                "use_retrieval": True,
                "use_nli": False,
                "use_ensemble": False,
                "use_cleaning": True,
                "use_artifact_persistence": False,
                "use_batch_nli": False,
                "use_online_authority": False,
            },
            "01b_retrieval_nli": {
                "use_retrieval": True,
                "use_nli": True,
                "use_ensemble": False,
                "use_cleaning": True,
                "use_artifact_persistence": False,
                "use_batch_nli": True,
                "use_online_authority": False,
            },
            "01c_ensemble": {
                "use_retrieval": True,
                "use_nli": True,
                "use_ensemble": True,
                "use_cleaning": True,
                "use_artifact_persistence": False,
                "use_batch_nli": True,
                "use_online_authority": False,
                "domain_agnostic_weighting": True,
                "quality_predictor_enabled": True,
                "non_cs_entail_weight": 0.85,
                "non_cs_similarity_weight": 0.15,
                "enable_calibration_checkpoint": True,
                "calibration_target_ece": 0.10,
                "enable_research_logging": True,
                "research_log_dir": "evaluation/results/research_logs",
            },
            "02_no_cleaning": {
                "use_retrieval": True,
                "use_nli": True,
                "use_ensemble": False,
                "use_cleaning": False,
                "use_artifact_persistence": False,
                "use_batch_nli": True,
                "use_online_authority": False,
            },
            "03_with_artifact_persistence": {
                "use_retrieval": True,
                "use_nli": True,
                "use_ensemble": False,
                "use_cleaning": True,
                "use_artifact_persistence": True,
                "use_batch_nli": True,
                "use_online_authority": False,
            },
            "04_no_batch_nli": {
                "use_retrieval": True,
                "use_nli": True,
                "use_ensemble": False,
                "use_cleaning": True,
                "use_artifact_persistence": False,
                "use_batch_nli": False,
                "use_online_authority": False,
            },
            "05_with_online_authority": {
                "use_retrieval": True,
                "use_nli": True,
                "use_ensemble": False,
                "use_cleaning": True,
                "use_artifact_persistence": False,
                "use_batch_nli": True,
                "use_online_authority": True,
            },
        }

        if self.verify_threshold is not None or self.low_conf_threshold is not None:
            for config in configs.values():
                if self.verify_threshold is not None:
                    config["verify_threshold"] = self.verify_threshold
                if self.low_conf_threshold is not None:
                    config["low_conf_threshold"] = self.low_conf_threshold

        return configs

    def run_ablations(self, noise_injection: bool = False, resume: bool = True) -> pd.DataFrame:
        configs = self.get_ablation_configs()
        config_order = list(configs.keys())

        existing_by_config: Dict[str, Dict[str, Any]] = {}
        if resume:
            existing_csv = self.output_dir / "results.csv"
            if existing_csv.exists():
                try:
                    existing_df = pd.read_csv(existing_csv)
                    if "config_name" in existing_df.columns:
                        for _, row in existing_df.iterrows():
                            cfg_name = row.get("config_name")
                            if isinstance(cfg_name, str) and cfg_name:
                                existing_by_config[cfg_name] = row.to_dict()
                except Exception as exc:
                    logger.warning("Could not read existing results for resume: %s", exc)

        self.results = [existing_by_config[name] for name in config_order if name in existing_by_config]

        for name, config in configs.items():
            if name in existing_by_config:
                continue
            try:
                runner = CSBenchmarkRunner(
                    dataset_path=self.dataset_path,
                    seed=self.seed,
                    device=self.device,
                    batch_size=self.batch_size,
                )
                result = runner.run(
                    config=config,
                    noise_types=["typo", "paraphrase"] if noise_injection else [],
                    sample_size=self.sample_size,
                )
                result.to_json(self.detailed_dir / f"{name}_result.json")
                result.to_csv(self.detailed_dir / f"{name}_metrics.csv")

                metrics_dict = result.metrics.to_csv_row()
                metrics_dict["config_name"] = name
                metrics_dict["timestamp"] = result.timestamp
                self.results.append(metrics_dict)
            except Exception as exc:
                logger.error("Config %s failed: %s", name, exc)
                self.results.append({"config_name": name, "accuracy": 0.0, "error": str(exc)})

        df = pd.DataFrame(self.results)
        if not df.empty and "config_name" in df.columns:
            df["_order"] = df["config_name"].apply(lambda x: config_order.index(x) if x in config_order else 999)
            df = df.sort_values("_order").drop(columns=["_order"])

        df.to_csv(self.output_dir / "results.csv", index=False)
        self.generate_summary(df)
        return df

    def generate_summary(self, df: pd.DataFrame) -> None:
        summary_path = self.output_dir / "ablation_summary.md"
        lines = [
            "# Ablation Study Results",
            "",
            f"**Run Date**: {datetime.now().isoformat()}",
            f"**Dataset**: {self.dataset_path}",
            f"**Sample Size**: {self.sample_size if self.sample_size else 'Full'}",
            "",
            "## Results",
            "",
            "| Config | Accuracy | F1 (Verified) | ECE | Avg Time (s) |",
            "|--------|----------|---------------|-----|--------------|",
        ]
        for _, row in df.iterrows():
            lines.append(
                f"| {row.get('config_name', '')} | "
                f"{float(row.get('accuracy', 0.0)):.3f} | "
                f"{float(row.get('F1_verified', 0.0)):.3f} | "
                f"{float(row.get('ece', 0.0)):.4f} | "
                f"{float(row.get('avg_time_per_claim', 0.0)):.4f} |"
            )

        summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_cs_benchmark(
    dataset_path: str = "evaluation/cs_benchmark/cs_benchmark_dataset.jsonl",
    output_dir: str = "evaluation/results",
    sample_size: Optional[int] = None,
    seed: int = 42,
    device: str = "cpu",
    batch_size: int = 32,
    noise_injection: bool = False,
    smoke_mode: bool = False,
) -> pd.DataFrame:
    """Run CS benchmark ablations and return results DataFrame."""
    if smoke_mode:
        device = "cpu"
        if sample_size is None:
            sample_size = 5

    runner = AblationRunner(
        dataset_path=dataset_path,
        output_dir=output_dir,
        sample_size=sample_size,
        seed=seed,
        device=device,
        batch_size=batch_size,
    )
    return runner.run_ablations(noise_injection=noise_injection, resume=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CS benchmark ablations")
    parser.add_argument("--dataset", default="evaluation/cs_benchmark/cs_benchmark_dataset.jsonl", help="Path to benchmark dataset")
    parser.add_argument("--output-dir", default="evaluation/results", help="Output directory")
    parser.add_argument("--sample-size", type=int, default=None, help="Number of examples to run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--noise-injection", action="store_true", help="Enable noise injection robustness mode")
    parser.add_argument("--smoke", action="store_true", help="Run smoke mode (CPU, small sample)")
    args = parser.parse_args()

    run_cs_benchmark(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        seed=args.seed,
        device=args.device,
        batch_size=args.batch_size,
        noise_injection=args.noise_injection,
        smoke_mode=args.smoke,
    )
    return 0
