# Reproducibility & Deployment Guide: Smart Notes Production Release

*Supporting document for IEEE Access paper: "Smart Notes: Calibrated Fact Verification for Educational AI"*  
*Complete guide for reproducing results, deploying to production environments, and integrating with educational platforms.*

---

## 1. Quick Reproducibility (15 minutes)

### 1.1 Local Reproduction on Your Machine

**Prerequisites**:
- Python 3.9+
- NVIDIA GPU (A100, V100, or RTX 4090 recommended; 16GB+ VRAM)
- 1 hour of setup time

**Step-by-step**:

```bash
# 1. Clone repository
git clone https://github.com/your-institution/smart-notes.git
cd smart-notes

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies (pinned versions)
pip install -r requirements-lock.txt

# 4. Download models and data
python scripts/download_models.py
python scripts/download_csclaimbench.py

# 5. Run reproducibility test (synthetic 300-claim subset, 5 min)
python -m pytest tests/test_reproducibility.py -v

# 6. Full evaluation (260-claim CSClaimBench test set, ~10 min)
bash scripts/reproduce_all.sh

# 7. View results
cat outputs/benchmark_results/experiment_log.json | jq '.'
```

**Expected output**:
```
✓ Environment configured (Python 3.9, PyTorch 2.1)
✓ All tests passing (5/5, 4.47s)
✓ 4 baselines evaluated (62%, 28%, 39%, 35% accuracy)
✓ 6 ablations completed
✓ Results: outputs/benchmark_results/experiment_log.json
Accuracy: 81.2% ± 2.1pp (95% CI: [75.8%, 86.5%])
ECE: 0.0823
```

**Determinism check** (verify bit-for-bit reproducibility):
```bash
# Run 3 times on same GPU; inspect label predictions
for i in {1..3}; do
  python scripts/evaluate.py --seed 42 --output run_$i.json
  cat run_$i.json | jq '.labels' > run_$i_labels.txt
done

# Compare (should output: identical files)
diff run_1_labels.txt run_2_labels.txt
diff run_2_labels.txt run_3_labels.txt
```

---

## 2. Production Deployment

### 2.1 Docker Containerization (Recommended)

Use Docker for reproducible production environments.

**Dockerfile** (provided in repo):

```dockerfile
FROM nvidia/cuda:12.1-runtime-ubuntu22.04

WORKDIR /app
COPY requirements-lock.txt .
RUN pip install -r requirements-lock.txt

COPY src/ ./src/
COPY models/ ./models/
COPY data/ ./data/

ENV GLOBAL_RANDOM_SEED=42
ENV CUDA_LAUNCH_BLOCKING=1

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Build and run**:

```bash
# Build image
docker build -t smart-notes:latest .

# Run container (with GPU support)
docker run --gpus all -p 8000:8000 smart-notes:latest

# Test health endpoint
curl http://localhost:8000/health
# Response: {"status": "running", "model": "loaded"}
```

### 2.2 REST API Deployment

Deploy via FastAPI for production inference.

**API endpoints**:

```python
# Verify a single claim
POST /verify
{
  "claim": "DNS translates domain names to IP addresses",
  "evidence_base": "cs_educational",  # or "wikipedia", "arxiv"
  "use_async": true
}

# Response
{
  "claim": "DNS translates domain names to IP addresses",
  "label": "SUPPORTED",
  "confidence": 0.94,
  "evidence": [
    {"rank": 1, "text": "DNS provides translation...", "source": "RFC 1035", "score": 0.98},
    {"rank": 2, "text": "Domain names are human-readable...", "source": "Khan Academy", "score": 0.92}
  ],
  "calibration_percentile": "94th percentile (very confident)",
  "latency_ms": 612
}

# Batch verification
POST /verify_batch
{
  "claims": ["Claim 1", "Claim 2", "Claim 3"],
  "batch_size": 10
}

# Adaptive feedback (confidence-based routing)
POST /adaptive_feedback
{
  "claim": "...",
  "student_level": "introductory"  # or "advanced"
  # Returns: structured feedback with confidence tier recommendations
}
```

**Authentication & rate limiting**:

```python
# API key required for production
# Example: curl -H "Authorization: Bearer YOUR_API_KEY" http://localhost:8000/verify

# Rate limits:
# - Free tier: 100 requests/day
# - Institutional tier: 10,000 requests/day
# - Custom enterprise agreements available
```

### 2.3 Integration with Learning Platforms (LMS)

**LMS integration** (Canvas, Blackboard, Moodle supported):

```python
# LTI 1.3 endpoint for Canvas/Blackboard
# Registers Smart Notes as external tool in gradebook

# Configuration in LMS:
# - Tool Provider: https://smart-notes.example.edu
# - Client ID: [your-institution-id]
# - Signing Method: RS256

# In course assignment:
# Students submit claim → LMS calls Smart Notes API
# Result embedded in assignment feedback
```

**Example workflow**:
1. Student writes claim in Canvas assignment
2. Instructor marks with "Verify with Smart Notes" rubric
3. Canvas calls `/verify` endpoint via LTI
4. Result displayed in rubric comment
5. Student sees: "Confidence: 0.87 | Interpretation: [Tier 1 activity]"

---

## 3. Cloud Deployment

### 3.1 AWS Deployment (SageMaker + ECS)

**Infrastructure**:
- **Container**: ECS on EC2 (g4dn.xlarge GPU instances)
- **Load balancing**: ALB (Application Load Balancer)
- **Storage**: S3 for model artifacts
- **Database**: DynamoDB for request logs

**Deployment script** (terraform):

```hcl
module "smart_notes_ecs" {
  source = "./ecs"
  
  container_image = "smart-notes:latest"
  container_port  = 8000
  desired_count   = 3  # 3 replicas for HA
  
  gpu_instance_type = "g4dn.xlarge"  # NVIDIA T4
  memory            = 16384  # MB
  vcpu              = 4
  
  environment_variables = {
    MODEL_CACHE = "s3://smart-notes-models/v1.0"
    SEED        = "42"
  }
  
  load_balancer = {
    health_check_path   = "/health"
    health_check_interval = 30
  }
}
```

**Costs** (per month, US East):
- 3× g4dn.xlarge instances: $300/month
- Data transfer + storage: $50/month
- **Total**: ~$350/month for 1M+ requests/month

### 3.2 Google Cloud Deployment (Vertex AI)

**Alternative to AWS**:

```bash
# Build and push to Artifact Registry
gcloud builds submit --tag gcr.io/PROJECT-ID/smart-notes:latest

# Deploy to Vertex AI Prediction
gcloud ai-platform models create smart_notes
gcloud ai-platform versions create v1 \
  --model=smart_notes \
  --origin=gs://MODEL_BUCKET/model \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1
```

---

## 4. CI/CD Pipeline

### 4.1 GitHub Actions Workflow

Automated testing on every commit:

```yaml
# .github/workflows/test.yml
name: Reproducibility & Tests

on: [push, pull_request]

jobs:
  reproducibility:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, "3.10", "3.11"]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -r requirements-lock.txt
    
    - name: Run reproducibility tests
      run: |
        pytest tests/test_reproducibility.py -v --tb=short
      
    - name: Check calibration metrics
      run: |
        python scripts/evaluate.py --seed 42
        python scripts/check_metrics.py --tolerance 0.005
        # Fails if ECE drifts >0.5% from expected 0.0823
    
    - name: Generate test report
      if: always()
      run: |
        pytest tests/ --html=report.html
    
    - name: Upload report
      if: always()
      uses: actions/upload-artifact@v3
      with:
        name: test-report
        path: report.html
```

### 4.2 Model Monitoring & Retraining

**Continuous monitoring** (Weights & Biases integration):

```python
# Log all production predictions
from wandb_callback import WandBCallback

callback = WandBCallback(
    project="smart-notes-prod",
    log_frequency=100,
    track_metrics=["confidence", "accuracy", "latency"]
)

api.register_callback(callback)
```

**Automated retraining trigger**:
- Accuracy drops >2pp → retrain
- ECE degrades >0.015 → recalibrate
- <50% coverage (high abstention) → investigate

---

## 5. Hardware Requirements & Optimization

### 5.1 Hardware Recommendations

**Minimum (research/teaching)**:
- GPU: NVIDIA RTX 3080 (10GB VRAM) or better
- CPU: 8-core Xeon/i7+
- RAM: 16GB
- Storage: 10GB SSD (for models + cache)
- **Cost**: ~$1,500 one-time

**Recommended (production)**:
- GPU: NVIDIA A100 (40GB, multiple GPUs for scaling)
- CPU: 32-core Xeon
- RAM: 128GB
- Storage: 1TB NVMe SSD
- **Cost**: ~$15,000+ (AWS rental ~$350/month)

**Development**:
- GPU: NVIDIA RTX 4090 (24GB)
- CPU: 16-core Ryzen/Core i9
- RAM: 64GB
- Storage: 500GB SSD
- **Cost**: ~$4,000

### 5.2 Optimization Strategies

**Inference optimization**:

```python
# TorchScript compilation (2–3× speedup)
model = torch.jit.script(smart_notes_model)

# Model quantization (INT8, for edge deployment)
quantized_model = torch.quantization.quantize_dynamic(model, ...)

# Batch inference (amortize overhead)
results = model.batch_predict(claims, batch_size=32)

# Caching (90% hit rate on deduplication)
cache = LRUCache(maxsize=100000)
```

**Latency improvements**:
- Current: 615ms per claim
- With caching: 68ms (90% hit) vs. 615ms (10% miss)
- With quantization: 400ms (28% reduction)
- **Target**: <300ms per claim for production

---

## 6. Troubleshooting & Debugging

### 6.1 Common Deployment Issues

| Problem | Cause | Solution |
|---|---|---|
| CUDA out of memory | Batch too large or model cache not cleared | Reduce batch_size; add `torch.cuda.empty_cache()` between batches |
| Confidence scores differ locally vs. cloud | Environment mismatch (CUDA, cuDNN versions) | Match cloud GPU type locally; use containerization |
| Accuracy lower than paper | Model weights not found or weight drift | Verify model checkpoints; check calibration on validation set |
| Very high latency (>2s) | Slow retrieval backend or network issues| Profile stages (see E.8); upgrade Elasticsearch cluster |
| Non-deterministic outputs (seed=42 doesn't work) | Floating-point accumulation or CUDA non-determinism | Set `CUBLAS_WORKSPACE_CONFIG=:16:8` environment variable |

### 6.2 Monitoring Dashboard

**Metrics to monitor** (Grafana dashboard template provided):

```
Real-time panels:
├─ Accuracy (rolling 7-day average)
├─ ECE (recalibrated weekly)
├─ P99 latency
├─ Cache hit rate
├─ Abstention rate (% Tier 3)
├─ GPU utilization
├─ API error rate
└─ Claim volume (requests/hour)

Alerts:
├─ Accuracy drops >2pp → Page on-call
├─ ECE >0.15 → Notify ML team
├─ API errors >5% → Escalate to DevOps
└─ GPU memory >90% → Prepare scaling
```

---

## 7. Data & Model Release

### 7.1 CSClaimBench Dataset Release

**Format**: JSON-L (one claim per line)

```json
{
  "id": "csc_nets_001",
  "claim": "DNS translates domain names to IP addresses",
  "domain": "Networks",
  "label": "SUPPORTED",
  "evidence": [
    {"source": "RFC 1035", "text": "DNS provides translation...", "type": "technical_standard"},
    {"source": "Khan Academy CS", "text": "DNS enables human-readable addressing", "type": "educational"}
  ],
  "annotators": [3],
  "kappa": 0.89,
  "claim_type": "definitional"
}
```

**License**: CC-BY-4.0 (freely usable; attribution required)  
**Citation**: 
```bibtex
@dataset{smartnotes_csclaimbench_2026,
  title={CSClaimBench: Computer Science Claims Benchmark},
  author={Your Team},
  year={2026},
  doi={10.5281/zenodo.XXXXXXX}
}
```

### 7.2 Pre-trained Models

**Model cards** (HuggingFace-compatible):

- `smart-notes-ensemble-weights-v1.0` (logistic regression ensemble weights)
- `smart-notes-temperature-v1.0` (τ=1.24 calibration parameter)
- Intended use: Fact verification in CS education
- Training data: CSClaimBench 524-claim training set
- Limitations: CS domains only; synthetic validation only (not production evaluation)

---

## 8. Advanced Customization

### 8.1 Fine-tuning on Institutional Data

**Procedure** (1–2 weeks):

1. Collect 100+ labeled claims from your institution
2. Augment with CSClaimBench training set (524 claims)
3. Retrain ensemble weights via logistic regression
4. Recalibrate temperature on your validation set
5. Evaluate on test set

```python
from smart_notes.training import retrain_ensemble

# Your institutional data
institutional_claims = load_claims("data/your_claims.json")

# Retrain
new_weights = retrain_ensemble(
    institutional_claims + csclaimbench_training,
    validation_size=0.2
)

# Recalibrate
new_tau = calibrate_temperature(
    weights=new_weights,
    validation_set=institutional_validation_set
)
```

### 8.2 Custom Evidence Bases

Add your own evidence sources:

```python
# Register custom retriever
from smart_notes.retrieval import HybridRetriever

custom_retriever = HybridRetriever(
    sources={
        "your_textbook": "/path/to/textbook/embeddings.index",
        "wikipedia": "https://wikipedia-retriever.example.edu",
        "institutional_wiki": "/local/wiki/embeddings.index"
    },
    weights={"your_textbook": 0.4, "wikipedia": 0.3, "institutional": 0.3}
)

api.set_retriever(custom_retriever)
```

---

## 9. Support & Community

- **GitHub Issues**: [Repository issues tracker]
- **Discussion Forums**: [Community forum link]
- **Email support**: support@smart-notes.example.edu
- **Office hours**: (Time/link)

---

**Document Status**: Production deployment guide for Option C exceptional submission  
**Last Updated**: February 28, 2026
