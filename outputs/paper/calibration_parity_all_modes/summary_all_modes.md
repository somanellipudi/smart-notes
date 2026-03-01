# Calibration Parity: All Modes

[WARNING] **SYNTHETIC EVALUATION** (seed=42, n=300)

## Results

| Mode | Accuracy | ECE | AUC-AC | Directory |
|------|----------|-----|--------|----------|
| full_default | 0.7467 | 0.0587 | -0.9950 | outputs\paper\calibration_parity_all_modes\full_default\ |
| minimal_deployment | 0.7467 | 0.0587 | -0.9950 | outputs\paper\calibration_parity_all_modes\minimal_deployment\ |
| verifiable | 0.7467 | 0.0587 | -0.9950 | outputs\paper\calibration_parity_all_modes\verifiable\ |

## Determinism Verification

- [OK] Same seed (42) used for all modes
- [OK] Predictions deterministic across runs
- [OK] Metrics stable (ECE computed consistently)
- [OK] Plots generated (reliability + risk-coverage)

## Running Reproducibility Test

```bash
python scripts/run_calibration_parity.py \
    --seed 42 --n-samples 300 --output-dir outputs/paper
```

## Notes

- **SYNTHETIC DATA**: All outputs are placeholder for reproducibility testing.
- **Real Evaluation**: Use CSClaimBench or FEVER for scientific claims.
- **Determinism**: Running twice with same seed produces identical discrete labels.
