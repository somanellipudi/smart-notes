"""
Performance profiling for verifiable mode.

Measures performance of each stage in the verification pipeline:
- Claim extraction
- Evidence retrieval
- Embedding computation
- NLI verification
- Graph construction
- Filtering and post-processing

Output: Breakdown of time spent in each stage.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Any
import json
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class StageProfile:
    """Profile metrics for a single pipeline stage."""
    stage_name: str
    start_time: float
    end_time: float
    num_items_processed: int
    memory_mb: float = 0.0
    
    @property
    def elapsed_seconds(self) -> float:
        """Elapsed time in seconds."""
        return self.end_time - self.start_time
    
    @property
    def throughput(self) -> float:
        """Items processed per second."""
        if self.elapsed_seconds == 0:
            return 0.0
        return self.num_items_processed / self.elapsed_seconds


class VerifiableModeProfiler:
    """Profile verifiable mode performance."""
    
    def __init__(self, output_dir: str = "profiling"):
        """
        Initialize profiler.
        
        Args:
            output_dir: Directory for profiling results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.stages: Dict[str, StageProfile] = {}
        self.start_time = time.time()
    
    def start_stage(self, stage_name: str):
        """Start profiling a stage."""
        if stage_name not in self.stages:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            self.stages[stage_name] = {
                'start': time.time(),
                'start_memory': memory_mb,
                'items': 0
            }
            logger.info(f"Starting profiling: {stage_name} (Memory: {memory_mb:.1f}MB)")
    
    def end_stage(self, stage_name: str, num_items: int = 0):
        """End profiling a stage."""
        if stage_name not in self.stages:
            logger.warning(f"Stage {stage_name} was not started")
            return
        
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        stage_data = self.stages[stage_name]
        elapsed = time.time() - stage_data['start']
        memory_delta = memory_mb - stage_data['start_memory']
        
        logger.info(
            f"Completed {stage_name}: "
            f"{elapsed:.2f}s, "
            f"{num_items} items, "
            f"Memory: +{memory_delta:.1f}MB → {memory_mb:.1f}MB"
        )
        
        stage_data['end'] = time.time()
        stage_data['end_memory'] = memory_mb
        stage_data['items'] = num_items
    
    def print_summary(self):
        """Print profiling summary."""
        import psutil
        process = psutil.Process()
        final_memory = process.memory_info().rss / 1024 / 1024
        total_elapsed = time.time() - self.start_time
        
        logger.info("\n" + "="*80)
        logger.info("VERIFIABLE MODE PERFORMANCE PROFILE")
        logger.info("="*80)
        logger.info(f"Total elapsed: {total_elapsed:.2f}s")
        logger.info(f"Final memory: {final_memory:.1f}MB\n")
        
        logger.info(f"{'Stage':<30} {'Time (s)':<12} {'Items':<10} {'Throughput':<15}")
        logger.info("-" * 80)
        
        total_time = 0.0
        for stage_name, stage_data in sorted(self.stages.items()):
            if 'end' not in stage_data:
                continue
            
            elapsed = stage_data['end'] - stage_data['start']
            items = stage_data['items']
            throughput = items / elapsed if elapsed > 0 else 0
            total_time += elapsed
            
            throughput_str = f"{throughput:.1f} items/s" if throughput > 0 else "N/A"
            
            logger.info(
                f"{stage_name:<30} {elapsed:<12.2f} {items:<10} {throughput_str:<15}"
            )
        
        logger.info("-" * 80)
        logger.info(f"{'TOTAL':<30} {total_time:<12.2f}\n")
        
        # Save to JSON
        self._save_profile_json(total_elapsed, final_memory, total_time)
    
    def _save_profile_json(self, total_elapsed: float, final_memory: float, total_time: float):
        """Save profiling data to JSON."""
        output_file = self.output_dir / f"profiling_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        profile_data = {
            "timestamp": datetime.now().isoformat(),
            "total_elapsed_seconds": total_elapsed,
            "final_memory_mb": final_memory,
            "total_processing_time": total_time,
            "stages": {}
        }
        
        for stage_name, stage_data in self.stages.items():
            if 'end' not in stage_data:
                continue
            
            elapsed = stage_data['end'] - stage_data['start']
            items = stage_data['items']
            throughput = items / elapsed if elapsed > 0 else 0
            memory_delta = stage_data['end_memory'] - stage_data['start_memory']
            
            profile_data["stages"][stage_name] = {
                "elapsed_seconds": elapsed,
                "items_processed": items,
                "throughput_items_per_second": throughput,
                "start_memory_mb": stage_data['start_memory'],
                "end_memory_mb": stage_data['end_memory'],
                "memory_delta_mb": memory_delta
            }
        
        with open(output_file, 'w') as f:
            json.dump(profile_data, f, indent=2)
        
        logger.info(f"\nProfile saved to: {output_file}\n")


def profile_verifiable_pipeline_demo():
    """Demonstrate profiling with mock data."""
    
    import sys
    from src.reasoning.verifiable_pipeline import VerifiablePipelineWrapper
    import config
    
    # Disable profiling unless explicitly enabled
    if not config.ENABLE_PROFILING:
        logger.warning("ENABLE_PROFILING is False. Enable in config to use profiling.")
        return
    
    profiler = VerifiableModeProfiler(config.PROFILING_ARTIFACTS_DIR)
    
    # Initialize pipeline
    try:
        profiler.start_stage("pipeline_initialization")
        pipeline = VerifiablePipelineWrapper()
        profiler.end_stage("pipeline_initialization", 1)
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        return
    
    # Simulate profiling with sample input
    sample_input = {
        "topic": "Machine Learning Basics",
        "content": " ".join(["Machine learning is a subset of artificial intelligence."] * 50)
    }
    
    try:
        from src.claims.extractor import ClaimExtractor
        
        profiler.start_stage("claim_extraction")
        extractor = ClaimExtractor()
        # This would normally extract claims from content
        profiler.end_stage("claim_extraction", 0)  # Placeholder
        
    except Exception as e:
        logger.error(f"Profiling error: {e}")
    
    # Print summary
    profiler.print_summary()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    profile_verifiable_pipeline_demo()
