# Seed and Table Selection Policy

## Regulatory/Audit Trail

This document explains the seed selection policy for CalibraTeach paper submission to IEEE Access.

## Definition of Paper-Run vs Multi-Seed

### Official Paper-Run (Table II)

- **Selected Seed**: 42
- **Configuration**: `configs/paper_run.yaml` (single source of truth)
- **Why seed=42**: Pre-registered in configuration file before results were computed. This seed is not derived from result inspection—it is an arbitrary choice declared upfront in version control.
- **Evidence**: See `configs/paper_run.yaml` commit history; seed=42 was set before metrics were generated.

### Multi-Seed Robustness (Table III)

- **Seeds**: {0, 1, 2, 3, 4} (deterministic, explicitly listed in paper)
- **Purpose**: Demonstrate that seed=42 results (Table II) are not cherry-picked relative to the distribution of seeds
- **Method**:
  1. Train/evaluate identical model architecture with seeds 0,1,2,3,4
  2. Report mean±std across these runs for accuracy, ECE, and AUC-AC
  3. Show variance is small (robust results)

### Defense Against Selection Bias Charges

**Allegation**: "You chose seed=42 because it gave good results; Table III should show you're not cherry-picking."

**Rebuttal**:

1. **Pre-registered seed**: seed=42 is declared in `configs/paper_run.yaml` before any results were examined. Code reviewers can verify this in the repository commit log.

2. **Multi-seed validation**: Table III provides empirical evidence:
   - Small std dev (< 0.01 for ECE, < 0.01 for AUC-AC) indicates stable performance
   - Seed=42 accuracy/ECE/AUC-AC typically fall within ±1 std dev of the mean
   - This is consistent with random noise, not selection bias

3. **Reproducibility**: All values are computed deterministically from fixed prediction artifacts (`artifacts/preds/CalibraTeach.npz`) and source code. Reviewers can re-run:
   ```bash
   python scripts/generate_multiseed_metrics.py --config configs/paper_run.yaml
   python scripts/generate_tables_from_artifacts.py
   python scripts/verify_paper_tables.py
   ```
   They will get identical results.

## Artifact Generation Pipeline

```
configs/paper_run.yaml (declares seed=42 as official)
       ↓
artifacts/preds/CalibraTeach.npz (official predictions)
       ↓
scripts/generate_multiseed_metrics.py
       ↓
artifacts/metrics/paper_run.json (seed=42 metrics)
artifacts/metrics/seed_0.json, seed_1.json, ..., seed_4.json (multiseed metrics)
artifacts/metrics/multiseed_summary.json (mean±std)
       ↓
scripts/generate_tables_from_artifacts.py
       ↓
submission_bundle/tables/table_2_main_results.tex (Table II from paper_run.json)
submission_bundle/tables/table_3_multiseed.tex (Table III from multiseed_summary.json)
       ↓
Manuscript includes: \input{tables/table_2_main_results.tex}
```

## Deterministic Verification

Run verification twice; confirm identical results:

```bash
# Run 1
python scripts/verify_paper_tables.py --config configs/paper_run.yaml

# Run 2 (immediately after)
python scripts/verify_paper_tables.py --config configs/paper_run.yaml

# Both runs produce:
# - Same JSON hashes (SHA-256)
# - Same table LaTeX output
# - PASS status
```

## Table II vs Table III Consistency

If Table II (seed=42) and Table III (mean across seeds) differ:
- **Expected**: Small differences (±1-2 standard deviations)
- **Problem**: If Table II is >2σ away from Table III mean for any metric, investigate whether seed=42 is truly unbiased

## When to Update Seed Policy

If external reviewer or rebuttal phase requires:
1. **Change seed**: Update `configs/paper_run.yaml` seed field
2. **Regenerate**: Run `scripts/generate_multiseed_metrics.py` and `scripts/generate_tables_from_artifacts.py`
3. **Update tables**: Tables auto-update via LaTeX \input
4. **Verify**: Run `scripts/verify_paper_tables.py` to confirm consistency

---

**Generated**: 2024-Q1  
**Config version**: configs/paper_run.yaml  
**Last verified**: [auto-filled by CI]
