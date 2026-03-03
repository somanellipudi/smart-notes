"""
Generate accuracy--coverage curve for selective prediction.
Inputs: research_bundle/07_papers_ieee/risk_coverage_curve.csv
Outputs: figures/acc_coverage.pdf
"""

import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

def compute_auc_ac(df, coverage_col, accuracy_col):
    """
    Compute Area Under Accuracy-Coverage curve.
    Normalized so that AUC-AC ranges from 0 to 1.
    """
    # Convert to decimal if needed
    coverage = df[coverage_col].values
    accuracy = df[accuracy_col].values
    
    # Ensure coverage is percentage decimal (0-1)
    if coverage.max() > 1:
        coverage = coverage / 100.0
    if accuracy.max() > 1:
        accuracy = accuracy / 100.0
    
    # Sort by coverage (should already be)
    sort_idx = np.argsort(coverage)
    coverage = coverage[sort_idx]
    accuracy = accuracy[sort_idx]
    
    # Compute AUC using trapezoidal rule
    auc = np.trapz(accuracy, coverage)
    
    return auc

def create_acc_coverage_curve():
    """Create accuracy--coverage trade-off curve for selective prediction."""
    
    # Try to load data
    csv_path = 'research_bundle/07_papers_ieee/risk_coverage_curve.csv'
    
    if not os.path.exists(csv_path):
        print(f"[WARN] {csv_path} not found. Skipping accuracy--coverage curve.")
        return
    
    df = pd.read_csv(csv_path)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Extract data
    coverage = df['coverage_pct'].values / 100.0  # Convert to decimal 0-1
    accuracy = df['accuracy_pct'].values / 100.0  # Convert to decimal 0-1
    threshold = df['threshold'].values
    
    # Find operating point at ~90% accuracy precision
    # Based on paper: "74% coverage at 90% precision operating point"
    # Looking for accuracy near 90%
    operating_idx = np.argmin(np.abs(accuracy - 0.90))
    op_coverage = coverage[operating_idx]
    op_accuracy = accuracy[operating_idx]
    op_threshold = threshold[operating_idx]
    
    # Compute AUC-AC
    auc_ac = compute_auc_ac(df, 'coverage_pct', 'accuracy_pct')
    
    # Plot curve
    ax.plot(coverage * 100, accuracy * 100, 'o-', color='#3498db', linewidth=3,
           markersize=8, label='Accuracy--Coverage Curve', zorder=3)
    
    # Highlight operating point
    ax.scatter([op_coverage * 100], [op_accuracy * 100], s=300, color='#e74c3c',
              edgecolors='#c0392b', linewidths=2, marker='*', zorder=4,
              label=f'Operating Point: {op_coverage*100:.1f}% Coverage @ {op_accuracy*100:.1f}% Accuracy (τ={op_threshold:.2f})')
    
    # Annotate operating point
    ax.annotate(f'{op_coverage*100:.1f}%, {op_accuracy*100:.1f}%',
               xy=(op_coverage * 100, op_accuracy * 100),
               xytext=(op_coverage * 100 + 5, op_accuracy * 100 - 3),
               fontsize=10, fontweight='bold', color='#c0392b',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='#fadbd8', alpha=0.8))
    
    # Labels and formatting
    ax.set_xlabel('Coverage (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Selective Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy--Coverage Trade-off: Selective Prediction\n(Varying Abstention Threshold τ)',
                fontsize=13, fontweight='bold', pad=20)
    
    # Set limits and grid
    ax.set_xlim(20, 102)
    ax.set_ylim(78, 100)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Ticks
    ax.set_xticks(np.arange(20, 110, 10))
    ax.set_yticks(np.arange(80, 100.5, 2))
    ax.tick_params(labelsize=11)
    
    # Legend
    ax.legend(loc='lower left', fontsize=11, framealpha=0.95)
    
    # Add AUC-AC display
    ax.text(0.98, 0.05, f'AUC-AC = {auc_ac:.4f}',
           ha='right', va='bottom', fontsize=11, color='#34495e', fontweight='bold',
           transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='#d5f4e6', alpha=0.8))
    
    # Add interpretation text
    ax.text(0.5, 0.02, 'Higher AUC-AC indicates better alignment between confidence and accuracy',
           ha='center', va='bottom', fontsize=9, color='#7f8c8d', style='italic',
           transform=ax.transAxes)
    
    plt.tight_layout()
    
    # Ensure figures directory exists
    os.makedirs('figures', exist_ok=True)
    
    # Save to PDF
    pdf_path = 'figures/acc_coverage.pdf'
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"[OK] Saved: {pdf_path}")
    
    plt.close()

if __name__ == '__main__':
    create_acc_coverage_curve()
