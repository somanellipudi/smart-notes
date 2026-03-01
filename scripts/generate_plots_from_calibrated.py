"""Generate reliability and confidence plots from calibrated_predictions.json files.

Usage:
  python scripts/generate_plots_from_calibrated.py --input_dir outputs/paper/calibration_parity --output_dir figures/calibration_parity
"""
import argparse
import json
from pathlib import Path
from src.evaluation.calibration import CalibrationEvaluator


def find_calibrated_files(input_dir: Path):
    return list(input_dir.rglob("calibrated_predictions.json"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    evalr = CalibrationEvaluator(n_bins=10)

    files = find_calibrated_files(input_dir)
    for f in files:
        dest_dir = output_dir / f.parent.name
        dest_dir.mkdir(parents=True, exist_ok=True)

        data = json.loads(f.read_text(encoding="utf-8"))
        probs = [float(rec.get("calibrated_confidence", rec.get("pred_confidence", 0.5))) for rec in data]
        labels = [1 if rec.get("pred_label") == rec.get("gold_label") else 0 for rec in data]

        reli_path = dest_dir / "reliability_diagram.png"
        conf_path = dest_dir / "confidence_by_correctness.png"

        evalr.plot_reliability_diagram(probs, labels, save_path=str(reli_path))
        evalr.plot_confidence_by_correctness(probs, labels, save_path=str(conf_path))

        print(f"Saved plots to {dest_dir}")


if __name__ == "__main__":
    main()
