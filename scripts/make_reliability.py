"""
Generate reliability diagram for CalibraTeach calibration quality.
Inputs: research_bundle/07_papers_ieee/calibration_bins_ece_correctness.csv
Outputs: figures/reliability.pdf
"""

import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

def compute_ece(df, accuracy_col, confidence_col):
    """Compute Expected Calibration Error from bins."""
    n_total = df['bin_count'].sum() if 'bin_count' in df.columns else len(df)
    ece = 0.0
    for _, row in df.iterrows():
        acc = row[accuracy_col] / 100.0  # Convert from percentage
        conf = row[confidence_col] / 100.0
        n_k = row['bin_count'] if 'bin_count' in df.columns else 1
        ece += (n_k / n_total) * abs(acc - conf)
    return ece

def create_reliability_diagram():
    """Create reliability diagram showing calibration quality (10 equal-width bins)."""
    
    # Try to load data
    csv_path = 'research_bundle/07_papers_ieee/calibration_bins_ece_correctness.csv'
    
    if not os.path.exists(csv_path):
        print(f"[WARN] {csv_path} not found. Skipping reliability diagram.")
        return
    
    df = pd.read_csv(csv_path)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Extract data
    confidence = df['bin_confidence_pct'].values / 100.0
    accuracy_calibrated = df['bin_accuracy_pct'].values / 100.0
    bin_counts = df['bin_count'].values if 'bin_count' in df.columns else np.ones(len(df))
    
    # Compute ECE for CalibraTeach
    ece_calibrated = compute_ece(df, 'bin_accuracy_pct', 'bin_confidence_pct')
    
    # Plot diagonal reference line (perfect calibration)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration (y=x)', alpha=0.5)
    
    # Plot CalibraTeach (calibrated, main result)
    ax.scatter(confidence, accuracy_calibrated, s=bin_counts*3, alpha=0.6,
              color='#3498db', edgecolors='#2980b9', linewidths=2,
              marker='o', label=f'CalibraTeach (ECE={ece_calibrated:.4f})', zorder=3)
    
    # Connect points with line
    ax.plot(confidence, accuracy_calibrated, color='#3498db', linewidth=2, alpha=0.4, zorder=2)
    
    # Labels and formatting
    ax.set_xlabel('Predicted Confidence', fontsize=12, fontweight='bold')
    ax.set_ylabel('Empirical Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Reliability Diagram: Predicted Confidence vs. Actual Accuracy\n(10 Equal-Width Bins, Test Set n=260)',
                fontsize=13, fontweight='bold', pad=20)
    
    # Set limits and grid
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal')
    
    # Ticks
    ax.set_xticks(np.linspace(0, 1, 11))
    ax.set_yticks(np.linspace(0, 1, 11))
    ax.tick_params(labelsize=11)
    
    # Legend
    ax.legend(loc='lower right', fontsize=11, framealpha=0.95)
    
    # Add ECE interpretation text
    ax.text(0.5, 0.05, f'ECE = {ece_calibrated:.4f} (lower is better, 0 = perfect)',
           ha='center', fontsize=10, color='#34495e', style='italic',
           transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.7))
    
    plt.tight_layout()
    
    # Ensure figures directory exists
    os.makedirs('figures', exist_ok=True)
    
    # Save to PDF
    pdf_path = 'figures/reliability.pdf'
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"[OK] Saved: {pdf_path}")
    
    plt.close()

if __name__ == '__main__':
    create_reliability_diagram()
