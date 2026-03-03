"""
Generate architecture diagram for CalibraTeach system.
Outputs: figures/architecture.pdf
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

def create_architecture_diagram():
    """Create a clean architecture diagram showing the 7-stage pipeline."""
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Define stages
    stages = [
        "Claim\nInput",
        "Evidence\nRetrieval",
        "Relevance\nFiltering",
        "NLI\nEnsemble",
        "6-Signal\nAggregation",
        "Temperature\nScaling",
        "Selective\nPrediction",
        "Explanation\nGeneration"
    ]
    
    box_width = 1.4
    box_height = 1.2
    y_pos = 2.4
    
    # Draw boxes and labels
    x_positions = []
    for i, stage in enumerate(stages):
        x_pos = 0.5 + i * 1.7
        x_positions.append(x_pos)
        
        # Draw box
        box = FancyBboxPatch(
            (x_pos - box_width/2, y_pos - box_height/2),
            box_width, box_height,
            boxstyle="round,pad=0.1",
            edgecolor='#2c3e50',
            facecolor='#ecf0f1',
            linewidth=2
        )
        ax.add_patch(box)
        
        # Add text
        ax.text(x_pos, y_pos, stage, ha='center', va='center',
                fontsize=9, fontweight='bold', color='#2c3e50')
    
    # Draw arrows between stages
    for i in range(len(x_positions) - 1):
        x1 = x_positions[i] + box_width/2
        x2 = x_positions[i+1] - box_width/2
        
        arrow = FancyArrowPatch(
            (x1, y_pos), (x2, y_pos),
            arrowstyle='->', mutation_scale=20,
            linewidth=2, color='#34495e'
        )
        ax.add_patch(arrow)
    
    # Add title
    ax.text(7, 5.3, "CalibraTeach: 7-Stage Real-Time Fact Verification Pipeline",
            ha='center', va='top', fontsize=13, fontweight='bold', color='#2c3e50')
    
    # Add footer with hardware specs
    ax.text(7, 0.5, "GPU: NVIDIA RTX 4090 (24GB, FP16) | Batch=1 | Mean Latency: 67.68ms (14.78 claims/sec)",
            ha='center', va='bottom', fontsize=9, color='#7f8c8d', style='italic')
    
    # Add runtime environment note
    ax.text(7, 0.1, "PyTorch 2.0.1 | CUDA 11.8 | Transformers 4.30.2",
            ha='center', va='bottom', fontsize=7, color='#95a5a6')
    
    plt.tight_layout()
    
    # Ensure figures directory exists
    os.makedirs('figures', exist_ok=True)
    
    # Save to PDF
    pdf_path = 'figures/architecture.pdf'
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"[OK] Saved: {pdf_path}")
    
    plt.close()

if __name__ == '__main__':
    create_architecture_diagram()
