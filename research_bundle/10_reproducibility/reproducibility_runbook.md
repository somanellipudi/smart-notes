# Reproducibility Runbook: Complete End-to-End Reproduction Guide

**Purpose**: Single comprehensive reference to reproduce all Smart Notes results bit-identically  
**Duration**: 2-3 hours (including model downloads)  
**Difficulty**: Intermediate (requires command-line proficiency)

---

## QUICK START (5 minutes)

```bash
# Clone repository
git clone https://github.com/[REPO]/smart-notes.git
cd smart-notes

# Run complete reproduction (installs deps, downloads models, runs full pipeline, generates all results)
bash scripts/reproduce_all.sh

# Verify results match paper (runs consistency checks)
python scripts/verify_results.py
```

If successful, you'll see: ✓ Results reproducible  

---

## PART 1: ENVIRONMENT SETUP (30 minutes)

### Step 1.1: System Requirements

**Hardware Minimum**:
- GPU: 40GB+ VRAM (A100 recommended; tested on V100, RTX 4090)
- CPU: 8 cores (16 recommended)
- RAM: 128GB (64GB min)
- Storage: 100GB free (50GB for models, 30GB for datasets, 20GB for outputs)

**Software Requirements**:
- Linux/Mac (Windows via WSL2 supported but NOT tested for bit-identical reproducibility)
- CUDA 12.1 + cuDNN 8.9
- Python 3.13.0 (EXACT version, not 3.13.1 or 3.12.x)

### Step 1.2: CUDA & cuDNN Installation

```bash
# Verify CUDA 12.1 installed
nvcc --version
# Expected output: Cuda compilation tools, release 12.1, ...

# Check cuDNN (if installed to /usr/local/cuda)
ls /usr/local/cuda/include/cudnn.h
# Should exist

# If not installed, follow NVIDIA official guide:
# https://docs.nvidia.com/cuda/cuda-installation-guide-linux/
```

### Step 1.3: Python Environment Setup

**Option A: Conda (Recommended)**

```bash
# Create environment
conda create -n smart-notes python=3.13.0 pytorch::pytorch pytorch::pytorch-cuda=12.1 -c pytorch -c nvidia

# Activate
conda activate smart-notes

# Verify
python --version  # Output: Python 3.13.0
```

**Option B: venv (Manual)**

```bash
# Create venv
python3.13 -m venv venv_smart_notes
source venv_smart_notes/bin/activate  # Linux/Mac
# OR: venv_smart_notes\Scripts\activate  (Windows)

# Install PyTorch + CUDA
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Verify
python -c "import torch; print(torch.cuda.is_available())"  # Should print: True
```

### Step 1.4: Install Smart Notes Dependencies

```bash
# Clone repository
git clone https://github.com/[REPO]/smart-notes.git
cd smart-notes

# Install package requirements
pip install -r requirements.txt

# Verify installation
python -c "import src; print('Smart Notes installed')"
```

**Expected requirements.txt**:
```
transformers==4.35.0
torch==2.1.0
numpy==1.24.3
scipy==1.11.2
pandas==2.1.0
scikit-learn==1.3.1
tqdm==4.66.1
pyyaml==6.0
```

---

## PART 2: DETERMINISM CONFIGURATION (5 minutes)

### Step 2.1: Random Seed Management

**Python Environment Setup**:

```python
# File: src/utils/determinism.py (to be created/verified)

import torch
import numpy as np
import random
import os

def set_reproducibility_seeds(seed=42):
    """Set all random seeds for reproducibility."""
    # Python's random
    random.seed(seed)
    
    # NumPy's random
    np.random.seed(seed)
    
    # PyTorch's random
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If multi-GPU
    
    # PyTorch determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # HuggingFace transformers
    os.environ['PYTORCH_DETERMINISTIC'] = '1'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'  # or ':32:8'
    
    print(f"✓ Reproducibility seeds set (seed={seed})")

# Call in main script
if __name__ == "__main__":
    set_reproducibility_seeds(42)
```

### Step 2.2: CUDA Determinism Configuration

```bash
# Set CUDA environment variable (required for bit-identical results)
export CUBLAS_WORKSPACE_CONFIG=:16:8

# Verify
echo $CUBLAS_WORKSPACE_CONFIG  # Output: :16:8

# Make permanent (add to ~/.bashrc or ~/.zshrc)
echo "export CUBLAS_WORKSPACE_CONFIG=:16:8" >> ~/.bashrc
source ~/.bashrc
```

**Note**: `:16:8` allocates 16 bytes per thread (conservative); `:32:8` uses 32 bytes (faster but uses more memory).

### Step 2.3: PyTorch Version Verification

```bash
# Check PyTorch version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
# Expected: PyTorch: 2.1.0

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
# Expected: CUDA available: True

# Check GPU name (for cross-GPU comparison)
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
# Expected output: A100-SXM4-40GB (or your GPU)
```

---

## PART 3: DATASET PREPARATION (15 minutes)

### Step 3.1: Download CSClaimBench Dataset

```bash
# Create data directory
mkdir -p data/csclaimbench

# Download from repository
wget https://zenodo.org/record/[ID]/csclaimbench_v1.0.tar.gz -O data/csclaimbench_v1.0.tar.gz

# Extract
tar xzf data/csclaimbench_v1.0.tar.gz -C data/csclaimbench

# Verify contents
ls data/csclaimbench/
# Expected: claims.json, annotations.json, evidence.json, splits/

# Verify checksums (prevent corruption)
sha256sum data/csclaimbench/data/claims.json
# Expected: [CHECKSUM from paper/release notes]
```

### Step 3.2: Preprocess Dataset

```bash
# Run preprocessing script (validates format, creates train/test splits)
python scripts/prepare_data.py \
  --input data/csclaimbench/data \
  --output data/processed \
  --seed 42 \
  --split 0.8 \
  --verbose

# Verify outputs
ls data/processed/
# Expected: train.pkl, test.pkl, metadata.json
```

---

## PART 4: MODEL DOWNLOAD & INITIALIZATION (30 minutes)

### Step 4.1: Download Pretrained Models

The system uses 3 pretrained models (total ~1GB):

```bash
# Create model directory
mkdir -p models/pretrained

# NLI Component (RoBERTa-large-mnli)
python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
           model='roberta-large-mnli'; \
           tokenizer = AutoTokenizer.from_pretrained(model); \
           transformer = AutoModelForSequenceClassification.from_pretrained(model); \
           print(f'✓ Downloaded {model}')"

# Semantic Similarity (sentence-transformers)
python -c "from sentence_transformers import SentenceTransformer; \
           model='all-MiniLM-L6-v2'; \
           encoder = SentenceTransformer(model); \
           print(f'✓ Downloaded {model}')"

# Verify cache location
ls ~/.cache/huggingface/hub/
# Should show: models--roberta-large-mnli, models--sentence-transformers--all-MiniLM-L6-v2
```

### Step 4.2: Copy Models to Local Cache

```bash
# Set HuggingFace cache directory (recommended for reproducibility)
export HF_HOME=$(pwd)/models/.huggingface_cache

# Verify settings
echo $HF_HOME  # Output: /full/path/to/smart-notes/models/.huggingface_cache
```

---

## PART 5: RUN EXPERIMENTS (45 minutes)

### Step 5.1: Run Main Verification Pipeline

```bash
# Main experiment pipeline
python scripts/run_experiments.py \
  --config configs/main_experiment.yaml \
  --seed 42 \
  --gpu 0 \
  --output outputs/exp_main_seed42 \
  --verbose

# Expected output files:
# - outputs/exp_main_seed42/predictions.json
# - outputs/exp_main_seed42/metrics.json
# - outputs/exp_main_seed42/logs.txt
```

**Expected Runtime**: ~45 minutes (processing 1,045 claims on A100)

### Step 5.2: Run Ablation Studies

```bash
# Single command for all ablations
python scripts/run_ablations.py \
  --config configs/ablations.yaml \
  --seed 42 \
  --output outputs/ablation_studies \
  --verbose

# Or individual ablations
for component in S1 S2 S3 S4 S5 S6; do
  python scripts/evaluate_model.py \
    --model_config configs/model_${component}.yaml \
    --data data/processed/test.pkl \
    --output outputs/ablation_${component}
done
```

### Step 5.3: Run Robustness Analysis

```bash
# Adversarial perturbations
python scripts/robustness_adversarial.py \
  --input data/processed/test.pkl \
  --output outputs/robustness_adversarial \
  --seed 42

# OCR noise injection
python scripts/robustness_ocr.py \
  --input data/processed/test.pkl \
  --corruption_levels 0 1 2 5 10 \
  --output outputs/robustness_ocr \
  --seed 42

# Domain shift evaluation
python scripts/evaluate_domain_shift.py \
  --input data/processed/test.pkl \
  --hold_out_domain NLP \
  --output outputs/robustness_domain_shift
```

---

## PART 6: COMPUTED RESULTS GENERATION (20 minutes)

### Step 6.1: Compute Metrics & Statistics

```bash
# Aggregate all results
python scripts/compute_metrics.py \
  --experiment_dir outputs/exp_main_seed42 \
  --output results/ \
  --confidence_level 0.95 \
  --verbose

# Expected output:
# - results/accuracy_report.json
# - results/calibration_analysis.json
# - results/statistical_tests.json
```

### Step 6.2: Generate Paper Tables

```bash
# Generate all tables in Markdown/LaTeX format
python scripts/generate_paper_tables.py \
  --metrics_file results/accuracy_report.json \
  --ablation_dir outputs/ablation_studies \
  --output paper_tables/ \
  --format latex  # or markdown

# Expected LaTeX output: paper_tables/table_*.tex (references ready for \input{})
```

### Step 6.3: Generate Figures

```bash
# Generate all figures (PNG + PDF)
python scripts/generate_figures.py \
  --metrics_dir results/ \
  --robustness_dir outputs/robustness_* \
  --output figures/ \
  --format pdf

# Expected: figures/fig_{01-10}.pdf
```

---

## PART 7: VERIFICATION & VALIDATION (10 minutes)

### Step 7.1: Verify Results Match Paper

```bash
# Run consistency checker
python scripts/verify_results.py \
  --results_dir results/ \
  --expected_accuracy 0.812 \
  --tolerance 0.001  # Allow ±0.1pp variance

# Expected: ✓ All metrics match within tolerance
```

**Key Metrics to Verify**:

```python
expected_metrics = {
    'overall_accuracy': 0.812,
    'overall_precision': 0.804,
    'overall_recall': 0.821,
    'f1_score': 0.81,
    'ece_pre_calibration': 0.2187,
    'ece_post_calibration': 0.0823,
    'selective_pred_precision_74cov': 0.904,
    'ablation_s1_loss': 0.081,
}
```

### Step 7.2: Cross-GPU Reproducibility Check

**Procedure**: Run pipeline on 2-3 different GPU types, verify bit-identical results.

```bash
# GPU Type 1: A100
python scripts/run_experiments.py --seed 42 --gpu 0 --output outputs/gpu_a100

# GPU Type 2: V100
python scripts/run_experiments.py --seed 42 --gpu 1 --output outputs/gpu_v100

# GPU Type 3: RTX4090
python scripts/run_experiments.py --seed 42 --gpu 2 --output outputs/gpu_rtx4090

# Compare outputs (should be byte-identical)
python scripts/compare_outputs.py \
  --output1 outputs/gpu_a100/predictions.json \
  --output2 outputs/gpu_v100/predictions.json \
  --output3 outputs/gpu_rtx4090/predictions.json

# Expected: All predictions identical (no floating-point differences)
```

### Step 7.3: Review Logs for Errors

```bash
# Check main experiment logs
tail -100 outputs/exp_main_seed42/logs.txt

# Check for warnings/errors
grep -i "error\|warning\|failed" outputs/exp_main_seed42/logs.txt
# Expected: No critical errors (warnings OK)

# Verify seed logging
grep -i "seed" outputs/exp_main_seed42/logs.txt
# Expected: Seed set to 42 in all modules
```

---

## PART 8: AUTOMATION SCRIPTS

### Script 8.1: reproduce_all.sh (One-Command Full Reproduction)

```bash
#!/bin/bash
set -e  # Exit on error

echo "Smart Notes Full Reproduction Runbook"
echo "======================================"

# Part 1: Environment
echo "[1/8] Setting up environment..."
export CUBLAS_WORKSPACE_CONFIG=:16:8
export HF_HOME=$(pwd)/models/.huggingface_cache

# Part 2: Verify dependencies
echo "[2/8] Verifying dependencies..."
python scripts/verify_environment.py

# Part 3: Prepare data
echo "[3/8] Preparing dataset..."
python scripts/prepare_data.py --seed 42 --split 0.8

# Part 4: Download models
echo "[4/8] Downloading pretrained models..."
python scripts/download_models.py

# Part 5: Run main experiments
echo "[5/8] Running main experiments (this may take 45min on A100)..."
python scripts/run_experiments.py --seed 42 --gpu 0

# Part 6: Run ablations
echo "[6/8] Running ablation studies..."
python scripts/run_ablations.py --seed 42

# Part 7: Generate results
echo "[7/8] Computing metrics and generating tables..."
python scripts/compute_metrics.py
python scripts/generate_paper_tables.py

# Part 8: Verification
echo "[8/8] Verifying results..."
python scripts/verify_results.py

echo ""
echo "✓ REPRODUCIBILITY VERIFICATION COMPLETE"
echo "Check results/ directory for all outputs"
```

**Usage**:
```bash
chmod +x scripts/reproduce_all.sh
./scripts/reproduce_all.sh
```

---

## PART 9: TROUBLESHOOTING

### Issue: CUDA Out of Memory

```bash
# Reduce batch size in config
vim configs/main_experiment.yaml
# Change: batch_size: 32 → batch_size: 8

# Or use different GPU
python scripts/run_experiments.py --gpu 1  # Switch to GPU 1
```

### Issue: Determinism Flag Error

```bash
# Error: "CUBLAS_WORKSPACE_CONFIG environment variable is not set"
export CUBLAS_WORKSPACE_CONFIG=:16:8

# Make permanent
echo "export CUBLAS_WORKSPACE_CONFIG=:16:8" >> ~/.bashrc
```

### Issue: Model Download Errors

```bash
# Clear HuggingFace cache and retry
rm -rf models/.huggingface_cache
python scripts/download_models.py  # Will re-download

# Or manually download
wget https://huggingface.co/.../pytorch_model.bin
```

### Issue: Results Don't Match Paper

```bash
# Run verification with debug output
python scripts/verify_results.py --verbose --show_diffs

# Check if random seeds are being used
grep -i "seed\|random" output logs/*.txt

# Verify CUDA determinism is enabled
python -c "import torch; print(torch.backends.cudnn.deterministic)"
# Should print: True
```

---

## PART 10: EXPECTED OUTPUT STRUCTURE

```
outputs/
├── exp_main_seed42/
│   ├── logs.txt
│   ├── config.yaml (recorded configuration)
│   ├── predictions.json (1,045 predictions)
│   └── metrics.json
├── ablation_studies/
│   ├── ablation_s1/metrics.json
│   ├── ablation_s2/metrics.json
│   ├── ... (S3-S6)
├── robustness_adversarial/
│   ├── adversarial_1pct.json
│   ├── adversarial_5pct.json
│   └── adversarial_10pct.json
├── robustness_ocr/
│   ├── ocr_corruption_1pct.json
│   ├── ... (2, 5, 10%)
└── robustness_domain_shift/
    └── held_out_NLP.json

results/
├── accuracy_report.json
├── calibration_analysis.json
├── statistical_tests.json
└── error_distribution.json

paper_tables/
├── table_01_main_results.tex
├── table_02_ablations.tex
├── ... (tables 3-8)

figures/
├── fig_01_architecture.pdf
├── fig_02_accuracy_by_type.pdf
├── ... (figs 3-10)
```

---

## PART 11: VALIDATION CHECKLIST

Before declaring reproducibility successful:

- [ ] Python 3.13.0 configured (verified via `python --version`)
- [ ] PyTorch 2.1.0 installed with CUDA 12.1 support
- [ ] CUBLAS_WORKSPACE_CONFIG environment variable set
- [ ] All random seeds set to 42 across all modules
- [ ] Dataset checksums verified (no corruption)
- [ ] Models downloaded and verified
- [ ] Main experiment runs without errors
- [ ] Accuracy reported: 81.2% (±0.1pp tolerance)
- [ ] ECE post-calibration: 0.0823 (±0.005)
- [ ] All ablations complete, results as expected
- [ ] Robustness tests pass
- [ ] Cross-GPU verification (bit-identical on 2+ GPU types)
- [ ] Paper tables generated successfully
- [ ] Figures generated without errors

**Success Criterion**: ✓ All items checked = Reproducibility verified

---

## FINAL VERIFICATION COMMAND

```bash
# One-line verification
python -c "
import json
with open('results/accuracy_report.json') as f:
    metrics = json.load(f)
    acc = metrics['overall_accuracy']
    assert abs(acc - 0.812) < 0.001, f'Accuracy {acc} != 0.812'
    print('✓ REPRODUCIBILITY VERIFIED: accuracy={:.3f}'.format(acc))
"
```

---

**End of Reproducibility Runbook**

If you encounter issues not covered here, please open an issue on GitHub with:
1. Environment details (GPU, CUDA, Python versions)
2. Error message and logs
3. Steps to reproduce the error
4. Output of `python scripts/verify_environment.py`

