#!/usr/bin/env python3
"""
Regenerate architecture.pdf with clean diagram (NO embedded text).

Creates a simple 7-stage pipeline diagram showing:
1. Evidence Retrieval
2. Relevance Filtering
3. Entailment Analysis
4. Confidence Aggregation
5. Calibration
6. Selective Prediction
7. Explanation Generation

All text is kept MINIMAL to serve as visual reference. Full descriptions belong in LaTeX captions.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Suppress matplotlib warnings
import warnings
warnings.filterwarnings("ignore")


def create_architecture_diagram(output_path):
    """Create and save a clean architecture diagram."""
    
    # Create figure - optimized for two-column IEEE layout
    # Width: ~5-6 inches (typical \textwidth for IEEE two-column)
    # Height: compact to leave room for other figures
    fig, ax = plt.subplots(figsize=(12, 2.2), dpi=100)
    
    # Define stages (7 stages)
    stages = [
        "Retrieval",
        "Filtering",
        "NLI\nEnsemble",
        "Aggregation",
        "Calibrate",
        "Selective",
        "Explanation"
    ]
    
    # Position stages
    box_width = 1.6
    box_height = 0.8
    arrow_length = 0.3
    y_center = 1.5
    
    # Colors
    color_box = "#E8F4F8"
    color_border = "#0077BB"
    color_arrow = "#0077BB"
    
    x_positions = []
    for i, stage in enumerate(stages):
        x = i * (box_width + arrow_length)
        x_positions.append(x)
        
        # Draw box
        box = FancyBboxPatch(
            (x, y_center - box_height / 2),
            box_width,
            box_height,
            boxstyle="round,pad=0.05",
            edgecolor=color_border,
            facecolor=color_box,
            linewidth=2,
            zorder=2
        )
        ax.add_patch(box)
        
        # Add stage label (small font, minimal text)
        ax.text(
            x + box_width / 2,
            y_center,
            stage,
            ha="center",
            va="center",
            fontsize=8,
            weight="bold",
            zorder=3
        )
        
        # Draw arrow to next stage
        if i < len(stages) - 1:
            arrow = FancyArrowPatch(
                (x + box_width, y_center),
                (x + box_width + arrow_length, y_center),
                arrowstyle="->",
                mutation_scale=20,
                linewidth=2,
                color=color_arrow,
                zorder=1
            )
            ax.add_patch(arrow)
    
    # Set axis properties
    ax.set_xlim(-0.5, x_positions[-1] + box_width + 0.5)
    ax.set_ylim(0.2, 2.8)
    ax.set_aspect("equal")
    ax.axis("off")  # Hide axes
    
    # Tight layout to remove margins
    fig.tight_layout(pad=0.2)
    
    # Save as PDF (NO TITLE, NO EMBEDDED TEXT BEYOND STAGE NAMES)
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Use tight bounding box with minimal padding to remove whitespace margins
        fig.savefig(
            output_path,
            format="pdf",
            bbox_inches="tight",
            pad_inches=0.02,  # Minimal padding for tight figure fit
            dpi=300
        )
        print(f"[OK] Architecture diagram saved to {output_path}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save {output_path}: {e}", file=sys.stderr)
        return False
    finally:
        plt.close(fig)


if __name__ == "__main__":
    repo_root = Path(__file__).parent.parent
    
    # CANONICAL PATH (single source of truth)
    # Matches: paper/main.tex \includegraphics{figures/architecture.pdf}
    # Which resolves to: paper/figures/architecture.pdf (relative to paper/ dir)
    canonical_output = repo_root / "paper" / "figures" / "architecture.pdf"
    
    print(f"[INFO] Regenerating architecture diagram at canonical path:")
    print(f"       {canonical_output}")
    
    if create_architecture_diagram(str(canonical_output)):
        print(f"[OK] Canonical architecture.pdf regenerated successfully")
        sys.exit(0)
    else:
        print(f"[ERROR] Failed to regenerate canonical architecture.pdf", file=sys.stderr)
        sys.exit(1)
