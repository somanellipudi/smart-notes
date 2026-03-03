#!/usr/bin/env python3
"""
Generate Paper Figures - Regenerate figures with verified metrics.

This script regenerates the reliability diagram and accuracy-coverage curve
figures using the verified metrics from metrics_summary.json to ensure
consistency across all paper visualizations.

Usage:
    python scripts/generate_paper_figures.py [--metrics_file artifacts/metrics_summary.json]

Produces:
    - figures/reliability_diagram_verified.pdf
    - figures/accuracy_coverage_verified.pdf
    - With auto-filled annotations showing verified metric values
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_metrics_summary(metrics_file: Path) -> Dict[str, Any]:
    """Load verified metrics from JSON."""
    with open(metrics_file, 'r') as f:
        return json.load(f)


def generate_reliability_diagram(
    summary: Dict[str, Any],
    output_path: Path,
    dpi: int = 300,
    figsize: tuple = (8, 6)
):
    """
    Generate reliability diagram showing calibration analysis.
    
    Uses verified ECE value from metrics_summary.json
    """
    logger.info("Generating reliability diagram...")
    
    ece = summary['reported_metrics']['ece']
    bin_stats = summary['bin_statistics']
    
    # Extract bin data
    confidences = []
    accuracies = []
    bin_sizes = []
    
    for bin_info in bin_stats:
        confidences.append(bin_info['confidence'])
        accuracies.append(bin_info['accuracy'])
        bin_sizes.append(bin_info['count'])
    
    confidences = np.array(confidences)
    accuracies = np.array(accuracies)
    bin_sizes = np.array(bin_sizes)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Plot reliability diagram
    # Diagonal line (perfect calibration)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfect Calibration', alpha=0.5)
    
    # Plot bins
    bin_width = 0.1  # 10 bins
    for i, (conf, acc, size) in enumerate(zip(confidences, accuracies, bin_sizes)):
        # Size of marker proportional to bin size
        marker_size = max(50, (size / len(bin_sizes)) * 300)
        ax.scatter(conf, acc, s=marker_size, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # Plot ECE as shaded area
    ax.fill_between([0, 1], 0, 1, alpha=0.05, color='red', label=f'ECE = {ece:.4f}')
    
    # Formatting
    ax.set_xlabel('Confidence (predicted probability)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Reliability Diagram - CalibraTeach System', fontsize=13, fontweight='bold')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add ECE annotation
    textstr = f'ECE (10 bins) = {ece:.4f}'
    ax.text(0.98, 0.05, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.legend(loc='upper left', fontsize=10)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    logger.info(f"✓ Reliability diagram saved to {output_path}")
    plt.close()


def generate_accuracy_coverage_curve(
    summary: Dict[str, Any],
    output_path: Path,
    dpi: int = 300,
    figsize: tuple = (8, 6)
):
    """
    Generate accuracy-coverage curve showing selective prediction performance.
    
    Uses verified AUC-AC value from metrics_summary.json
    """
    logger.info("Generating accuracy-coverage curve...")
    
    auc_ac = summary['reported_metrics']['auc_ac']
    curve_data = summary['accuracy_coverage_curve']
    
    thresholds = np.array(curve_data['thresholds'])
    coverage = np.array(curve_data['coverage'])
    accuracy = np.array(curve_data['accuracy'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Plot ACC-AUC curve
    ax.plot(coverage, accuracy, 'b-', linewidth=2.5, marker='o', markersize=6, label='CalibraTeach')
    
    # Plot random baseline (50% accuracy)
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, label='Random Baseline (50%)', alpha=0.7)
    
    # Fill area under curve
    ax.fill_between(coverage, 0.5, accuracy, alpha=0.2, color='blue')
    
    # Formatting
    ax.set_xlabel('Coverage (fraction of predictions)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Selective Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy-Coverage Curve - Selective Prediction', fontsize=13, fontweight='bold')
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0.4, 1.05])
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add AUC-AC annotation
    textstr = f'AUC-AC = {auc_ac:.4f}'
    ax.text(0.98, 0.05, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    ax.legend(loc='lower left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    logger.info(f"✓ Accuracy-coverage curve saved to {output_path}")
    plt.close()


def create_metrics_comparison_table(
    summary: Dict[str, Any],
    output_path: Path
):
    """Create comparison table showing verified vs paper-reported metrics."""
    logger.info("Creating metrics comparison table...")
    
    computed = summary['reported_metrics']
    reported = summary['paper_reported_values']
    ci = summary['confidence_intervals']
    
    table_data = {
        'Metric': ['Accuracy', 'ECE (10 bins)', 'AUC-AC', 'Macro-F1'],
        'Computed': [
            f"{computed['accuracy']:.4f}",
            f"{computed['ece']:.4f}",
            f"{computed['auc_ac']:.4f}",
            f"{computed['macro_f1']:.4f}",
        ],
        'Paper Reported': [
            f"{reported['accuracy']:.4f}",
            f"{reported['ece']:.4f}",
            f"{reported['auc_ac']:.4f}",
            f"{reported['macro_f1']:.4f}",
        ],
        'Difference': [
            f"{abs(computed['accuracy'] - reported['accuracy']):.4f}",
            f"{abs(computed['ece'] - reported['ece']):.4f}",
            f"{abs(computed['auc_ac'] - reported['auc_ac']):.4f}",
            f"{abs(computed['macro_f1'] - reported['macro_f1']):.4f}",
        ],
        '95% CI': [
            f"[{ci['accuracy_ci_lower']:.4f}, {ci['accuracy_ci_upper']:.4f}]",
            f"[{ci['ece_ci_lower']:.4f}, {ci['ece_ci_upper']:.4f}]",
            f"[{ci['auc_ac_ci_lower']:.4f}, {ci['auc_ac_ci_upper']:.4f}]",
            "—",
        ]
    }
    
    # Create markdown table
    md_content = "# Metrics Comparison Table\n\n"
    md_content += "Comparison of verified metrics (MetricsComputer) vs paper-reported values.\n\n"
    md_content += "| " + " | ".join(table_data.keys()) + " |\n"
    md_content +=  "|" + "|".join(["---" for _ in table_data.keys()]) + "|\n"
    
    for i in range(len(table_data['Metric'])):
        row = " | ".join([table_data[k][i] for k in table_data.keys()])
        md_content += f"| {row} |\n"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    logger.info(f"✓ Metrics comparison table saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate paper figures with verified metrics"
    )
    parser.add_argument(
        "--metrics_file",
        type=Path,
        default=Path("artifacts/metrics_summary.json"),
        help="Path to verified metrics JSON file"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("figures"),
        help="Output directory for generated figures"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for figure output"
    )
    
    args = parser.parse_args()
    
    # Verify metrics file exists
    if not args.metrics_file.exists():
        logger.error(f"Metrics file not found: {args.metrics_file}")
        logger.info("Run: python scripts/verify_reported_metrics.py first")
        sys.exit(1)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metrics
    logger.info(f"Loading metrics from {args.metrics_file}...")
    summary = load_metrics_summary(args.metrics_file)
    
    logger.info("\n" + "=" * 80)
    logger.info("GENERATING PAPER FIGURES WITH VERIFIED METRICS")
    logger.info("=" * 80)
    
    # Generate figures
    generate_reliability_diagram(
        summary,
        args.output_dir / "reliability_diagram_verified.pdf",
        dpi=args.dpi
    )
    
    generate_accuracy_coverage_curve(
        summary,
        args.output_dir / "accuracy_coverage_verified.pdf",
        dpi=args.dpi
    )
    
    # Create comparison table
    create_metrics_comparison_table(
        summary,
        args.output_dir / "metrics_comparison.md"
    )
    
    logger.info("\n" + "=" * 80)
    logger.info("FIGURE GENERATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"✓ Reliability diagram: {args.output_dir / 'reliability_diagram_verified.pdf'}")
    logger.info(f"✓ Accuracy-coverage curve: {args.output_dir / 'accuracy_coverage_verified.pdf'}")
    logger.info(f"✓ Metrics comparison: {args.output_dir / 'metrics_comparison.md'}")
    logger.info(f"\nAll figures use verified metrics from {args.metrics_file}")


if __name__ == "__main__":
    main()
