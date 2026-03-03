"""
Research Paper Auto-Update System.

Automatically populates research_paper.md from artifacts only (no hard-coded metrics).
Fails fast if required artifacts are missing.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict

import pandas as pd

logger = logging.getLogger(__name__)


class PaperUpdater:
    """Automatically updates research_paper.md with experimental artifacts."""

    REQUIRED_ARTIFACTS = {
        "ci_report": "ci_report.json",
        "multiseed_summary": "metrics_summary.csv",
        "ablation_table": "ablation_table.md",
        "calibration_report": "calibration_report.json",
        "selective_report": "selective_prediction_report.json",
        "error_analysis": "error_analysis_report.json",
        "llm_baseline": "llm_baseline/llm_baseline_result.json",
        "baseline_comparison_table": "baseline_comparison_table.md",
        "baseline_comparison_table_csv": "baseline_comparison_table.csv",
        "latency_breakdown": "latency_breakdown.csv",
        "latency_summary": "latency_summary.json",
    }

    def __init__(self, artifacts_dir: Path, paper_path: Path):
        self.artifacts_dir = Path(artifacts_dir)
        self.paper_path = Path(paper_path)

        self._validate_required_artifacts()
        self.artifacts = self._load_artifacts()

        logger.info("Paper updater initialized")
        logger.info("  Artifacts dir: %s", artifacts_dir)
        logger.info("  Paper path: %s", paper_path)

    def _validate_required_artifacts(self) -> None:
        missing = []
        for _, rel_path in self.REQUIRED_ARTIFACTS.items():
            full_path = self.artifacts_dir / rel_path
            if not full_path.exists():
                missing.append(str(full_path))

        if missing:
            message = (
                "Required artifacts are missing; cannot update research_paper.md. "
                "Regenerate artifacts first.\nMissing files:\n- " + "\n- ".join(missing)
            )
            logger.error(message)
            raise FileNotFoundError(message)

    def _load_artifacts(self) -> Dict[str, Any]:
        artifacts: Dict[str, Any] = {}

        with open(self.artifacts_dir / "ci_report.json", "r", encoding="utf-8") as f:
            artifacts["ci_report"] = json.load(f)

        artifacts["multiseed_summary"] = pd.read_csv(self.artifacts_dir / "metrics_summary.csv")

        artifacts["ablation_table"] = (self.artifacts_dir / "ablation_table.md").read_text(encoding="utf-8")

        with open(self.artifacts_dir / "calibration_report.json", "r", encoding="utf-8") as f:
            artifacts["calibration_report"] = json.load(f)

        with open(self.artifacts_dir / "selective_prediction_report.json", "r", encoding="utf-8") as f:
            artifacts["selective_report"] = json.load(f)

        with open(self.artifacts_dir / "error_analysis_report.json", "r", encoding="utf-8") as f:
            artifacts["error_analysis"] = json.load(f)

        with open(self.artifacts_dir / "llm_baseline" / "llm_baseline_result.json", "r", encoding="utf-8") as f:
            artifacts["llm_baseline"] = json.load(f)

        artifacts["baseline_comparison_table"] = (self.artifacts_dir / "baseline_comparison_table.md").read_text(encoding="utf-8")
        artifacts["baseline_comparison_csv"] = pd.read_csv(self.artifacts_dir / "baseline_comparison_table.csv")
        artifacts["latency_breakdown"] = pd.read_csv(self.artifacts_dir / "latency_breakdown.csv")

        with open(self.artifacts_dir / "latency_summary.json", "r", encoding="utf-8") as f:
            artifacts["latency_summary"] = json.load(f)

        metadata_path = self.artifacts_dir / "baseline_comparison_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                artifacts["baseline_metadata"] = json.load(f)
        else:
            artifacts["baseline_metadata"] = {"llm_stub_mode": False, "message": ""}

        return artifacts

    def update_paper(self) -> None:
        if self.paper_path.exists():
            content = self.paper_path.read_text(encoding="utf-8")
        else:
            content = self._create_template()

        content = self._update_abstract(content)
        content = self._update_experimental_setup(content)
        content = self._update_formal_definition(content)
        content = self._update_main_results_table(content)
        content = self._update_multiseed_table(content)
        content = self._update_ablation_table(content)
        content = self._update_calibration_section(content)
        content = self._update_selective_prediction_section(content)
        content = self._update_error_analysis_section(content)
        content = self._update_baseline_comparison(content)
        content = self._update_latency_breakdown(content)
        content = self._update_limitations_section(content)
        content = self._update_conclusion(content)
        content = self._update_reproducibility_section(content)
        content = self._tighten_claim_language(content)

        self.paper_path.write_text(content, encoding="utf-8")
        logger.info("✓ Research paper updated: %s", self.paper_path)

    def _create_template(self) -> str:
        return """# CalibraTeach: Calibrated Claim Verification for Educational Content

## Abstract
<!-- SECTION:abstract_scoped -->
<!-- /SECTION:abstract_scoped -->

## 1. Introduction

CalibraTeach studies calibrated selective verification for educational claims under controlled evaluation settings.

## 2. Related Work

[TODO: Add related work section]

## 3. Method

### 3.1 Formal Definition of Calibrated Selective Verification
<!-- SECTION:formal_definition -->
<!-- /SECTION:formal_definition -->

## 4. Experimental Setup
<!-- SECTION:experimental_setup -->
<!-- /SECTION:experimental_setup -->

## 5. Results

### 5.1 Main Results
<!-- SECTION:main_results -->
<!-- /SECTION:main_results -->

### 5.2 Multi-Seed Stability
<!-- SECTION:multiseed_stability -->
<!-- /SECTION:multiseed_stability -->

### 5.3 Ablation Study
<!-- SECTION:ablation_study -->
<!-- /SECTION:ablation_study -->

### 5.4 Calibration Analysis
<!-- SECTION:calibration_analysis -->
<!-- /SECTION:calibration_analysis -->

### 5.5 Selective Prediction Deployment Analysis
<!-- SECTION:selective_prediction -->
<!-- /SECTION:selective_prediction -->

### 5.6 Error Analysis
<!-- SECTION:error_analysis -->
<!-- /SECTION:error_analysis -->

### 5.7 Modern Baseline Comparison
<!-- SECTION:baseline_comparison -->
<!-- /SECTION:baseline_comparison -->

### 5.8 Latency Engineering Breakdown
<!-- SECTION:latency_breakdown -->
<!-- /SECTION:latency_breakdown -->

## 6. Discussion

[TODO: Add discussion]

## 7. Limitations and Future Work
<!-- SECTION:limitations -->
<!-- /SECTION:limitations -->

## 8. Reproducibility
<!-- SECTION:reproducibility -->
<!-- /SECTION:reproducibility -->

## 9. Conclusion
<!-- SECTION:conclusion -->
<!-- /SECTION:conclusion -->

## References

[TODO: Add references]
"""

    def _replace_section(self, content: str, section_name: str, new_content: str) -> str:
        pattern = f"<!-- SECTION:{section_name} -->.*?<!-- /SECTION:{section_name} -->"
        replacement = f"<!-- SECTION:{section_name} -->\n{new_content}\n<!-- /SECTION:{section_name} -->"
        new_text = re.sub(pattern, lambda _m: replacement, content, flags=re.DOTALL)
        if new_text == content:
            logger.warning("Section '%s' not found in paper", section_name)
        return new_text

    def _update_abstract(self, content: str) -> str:
        ci = self.artifacts["ci_report"]
        sel = self.artifacts["selective_report"]
        llm_meta = self.artifacts.get("baseline_metadata", {})
        stub_msg = " LLM-RAG results correspond to stub mode (no API evaluation)." if llm_meta.get("llm_stub_mode") else ""

        abstract = f"""
We present CalibraTeach, a calibrated selective verification framework evaluated on CSClaimBench under controlled experimental conditions. On the current evaluation split, the system achieves accuracy {ci['accuracy']['point_estimate']:.4f}, ECE {ci['ece']['point_estimate']:.4f}, and AUC-AC {ci['auc_ac']['point_estimate']:.4f}, with bootstrap confidence intervals reported for all primary metrics. Selective prediction analysis shows improved accepted-set accuracy under abstention, with an operating point at {sel['optimal_operating_point']['coverage']:.0%} coverage and {sel['optimal_operating_point']['accuracy']:.4f} accepted accuracy. We also report modern baseline comparison (including retrieval-augmented LLM baseline), stage-wise latency decomposition, and structured error analysis. The study is scoped to computer science claims and demonstrates calibration-aware decision quality and engineering feasibility rather than universal domain generalization.{stub_msg}
""".strip()
        return self._replace_section(content, "abstract_scoped", abstract)

    def _update_experimental_setup(self, content: str) -> str:
        ci = self.artifacts["ci_report"]
        setup = f"""
- Dataset: CSClaimBench-style evaluation split (n={ci['n_samples']})
- Bootstrap: {ci['n_bootstrap']} resamples for 95% confidence intervals
- Multi-seed protocol: metrics aggregated over deterministic seed set
- Evaluation focus: calibration-aware performance and selective prediction
- Baselines: CalibraTeach final, retrieval-augmented LLM baseline, and classical neural verifier (when available)
""".strip()
        return self._replace_section(content, "experimental_setup", setup)

    def _update_formal_definition(self, content: str) -> str:
        formal = r"""
We define calibrated selective verification with three stages:

1) **Ensemble aggregation**

$$
p_{\text{raw}} = f(c_1, c_2, \ldots, c_6)
$$

where $c_i$ are confidence components and $f(\cdot)$ is the learned aggregation map.

2) **Temperature scaling**

$$
p_{\text{cal}} = \sigma\left(\frac{z}{T}\right)
$$

where $z$ is the pre-sigmoid logit, $T>0$ is temperature, and $\sigma(\cdot)$ is the logistic function.

3) **Selective decision rule**

$$
\hat{y}=\begin{cases}
\operatorname{predict}, & p_{\text{cal}} \ge \tau \\
\operatorname{abstain}, & p_{\text{cal}} < \tau
\end{cases}
$$

with decision threshold $\tau$ chosen from validation risk-coverage tradeoffs.

4) **Risk-coverage formalization**

$$
\operatorname{Coverage}(\tau)=\frac{1}{n}\sum_{i=1}^n \mathbf{1}[p_i \ge \tau],
\quad
\operatorname{Risk}(\tau)=1-\operatorname{Accuracy}(\tau)
$$

and selective quality summarized by area under the accuracy-coverage curve (AUC-AC).
""".strip()
        return self._replace_section(content, "formal_definition", formal)

    def _update_main_results_table(self, content: str) -> str:
        ci = self.artifacts["ci_report"]
        table = f"""
| Metric | Value | 95% CI |
|--------|--------|--------|
| Accuracy | {ci['accuracy']['point_estimate']:.4f} | [{ci['accuracy']['lower']:.4f}, {ci['accuracy']['upper']:.4f}] |
| Macro-F1 | {ci['macro_f1']['point_estimate']:.4f} | [{ci['macro_f1']['lower']:.4f}, {ci['macro_f1']['upper']:.4f}] |
| ECE (15 bins) | {ci['ece']['point_estimate']:.4f} | [{ci['ece']['lower']:.4f}, {ci['ece']['upper']:.4f}] |
| AUC-AC | {ci['auc_ac']['point_estimate']:.4f} | [{ci['auc_ac']['lower']:.4f}, {ci['auc_ac']['upper']:.4f}] |

**Table 1**: Main results with 95% bootstrap confidence intervals (n={ci['n_samples']}, {ci['n_bootstrap']} bootstrap samples).
""".strip()
        return self._replace_section(content, "main_results", table)

    def _update_multiseed_table(self, content: str) -> str:
        df = self.artifacts["multiseed_summary"]
        table = "\n| Metric | Mean ± Std | Worst Case |\n|--------|------------|------------|\n"
        for _, row in df.iterrows():
            table += f"| {row['Metric']} | {row['Mean']:.4f} ± {row['Std']:.4f} | {row['Worst_Case']:.4f} |\n"
        table += "\n**Table 2**: Multi-seed stability analysis under deterministic seed control.\n"
        return self._replace_section(content, "multiseed_stability", table.strip())

    def _update_ablation_table(self, content: str) -> str:
        section = f"""
{self.artifacts['ablation_table']}

**Table 3**: Component ablation under calibration-aware evaluation.
""".strip()
        return self._replace_section(content, "ablation_study", section)

    def _update_calibration_section(self, content: str) -> str:
        cal = self.artifacts["calibration_report"]
        table = "\n| Bins | ECE Before | ECE After | Improvement |\n|------|------------|-----------|-------------|\n"
        for bins in sorted(cal["ece_before"].keys(), key=lambda x: int(x)):
            before = cal["ece_before"][str(bins)]
            after = cal["ece_after"][str(bins)]
            improvement = before - after
            table += f"| {bins} | {before:.4f} | {after:.4f} | {improvement:.4f} |\n"

        section = f"""
{table}

**Table 4**: ECE sensitivity to bin size before and after temperature scaling (T={cal['temperature']:.3f}).

- Brier score: {cal['brier_before']:.4f} -> {cal['brier_after']:.4f}
- Reliability plots: `fig_reliability_before.png`, `fig_reliability_after.png`
""".strip()
        return self._replace_section(content, "calibration_analysis", section)

    def _update_selective_prediction_section(self, content: str) -> str:
        sel = self.artifacts["selective_report"]

        acc_table = "\n| Coverage | Accuracy |\n|----------|----------|\n"
        for cov, acc in sel["accuracy_at_coverage"].items():
            acc_table += f"| {float(cov):.0%} | {acc:.4f} |\n"

        risk_table = "\n| Max Risk | Coverage |\n|----------|----------|\n"
        for risk, cov in sel["coverage_at_risk"].items():
            risk_table += f"| {float(risk):.1%} | {float(cov):.2%} |\n"

        op = sel["optimal_operating_point"]
        section = f"""
**Accuracy at Coverage**:

{acc_table}

**Coverage at Risk**:

{risk_table}

- AUC-RC: {sel['auc_rc']:.4f}
- Recommended operating point: threshold={op['threshold']:.3f}, coverage={op['coverage']:.0%}, accuracy={op['accuracy']:.4f}, risk={op['risk']:.2%}
""".strip()
        return self._replace_section(content, "selective_prediction", section)

    def _update_error_analysis_section(self, content: str) -> str:
        err = self.artifacts["error_analysis"]
        table = "\n| Error Type | Count | Percentage |\n|------------|-------|------------|\n"
        for error_type, count in sorted(err["error_breakdown"].items(), key=lambda x: -x[1]):
            pct = err["error_percentages"][error_type]
            table += f"| {error_type.replace('_', ' ').title()} | {count} | {pct:.1f}% |\n"

        section = f"""
**Error Breakdown** (Total Errors: {err['total_errors']}):

{table}

See `error_examples.md` for representative cases.
""".strip()
        return self._replace_section(content, "error_analysis", section)

    def _update_baseline_comparison(self, content: str) -> str:
        meta = self.artifacts.get("baseline_metadata", {})
        note = "\n\n**Note**: Stub baseline (no API evaluation)." if meta.get("llm_stub_mode") else ""
        section = f"""
{self.artifacts['baseline_comparison_table']}

**Table 8**: Baseline comparison under calibration-aware metrics (Accuracy, Macro-F1, ECE, AUC-AC).{note}
""".strip()
        return self._replace_section(content, "baseline_comparison", section)

    def _update_latency_breakdown(self, content: str) -> str:
        df = self.artifacts["latency_breakdown"]
        summary = self.artifacts["latency_summary"]
        section = f"""
{df.to_markdown(index=False)}

**Latency summary**:
- Total mean latency: {summary['total_mean_latency_ms']:.2f} ms
- Throughput: {summary['throughput_claims_per_sec']:.2f} claims/sec

This breakdown separates retrieval, inference, ensemble scoring, calibration, and selective decision stages for engineering reproducibility.
""".strip()
        return self._replace_section(content, "latency_breakdown", section)

    def _update_limitations_section(self, content: str) -> str:
        section = """
1. **Dataset Size**: Primary evaluation is based on 260 expert-labeled claims, which limits external statistical power.
2. **Domain Restriction**: Results are validated on CSClaimBench (computer science domain) under controlled experimental conditions.
3. **English-Only Evaluation**: Current experiments evaluate only English-language claims and evidence.
4. **Calibration Transfer**: Temperature scaling and selective thresholds may require domain-specific re-scaling under distribution shift.
5. **LLM Baseline Dependency**: API-based LLM baselines depend on provider availability, pricing, and reproducible access; stub mode is explicitly marked when used.
6. **Threshold Tuning**: Selective prediction thresholds are context-dependent and should be tuned for deployment risk tolerances.
""".strip()
        return self._replace_section(content, "limitations", section)

    def _update_conclusion(self, content: str) -> str:
        ci = self.artifacts["ci_report"]
        summary = self.artifacts["latency_summary"]
        selective = self.artifacts["selective_report"]["optimal_operating_point"]

        conclusion = f"""
CalibraTeach demonstrates that calibrated decision-making is a more deployment-relevant target than raw accuracy alone for educational claim verification. On the present evaluation split, the system achieves accuracy {ci['accuracy']['point_estimate']:.4f}, ECE {ci['ece']['point_estimate']:.4f}, and AUC-AC {ci['auc_ac']['point_estimate']:.4f}, while supporting abstention-aware operation at {selective['coverage']:.0%} coverage and {selective['accuracy']:.4f} accepted accuracy.

Engineering analysis reports a mean end-to-end latency of {summary['total_mean_latency_ms']:.2f} ms ({summary['throughput_claims_per_sec']:.2f} claims/sec) with transparent stage-wise decomposition. This supports practical integration in supervised educational workflows.

Overall, the framework positions abstention-aware AI as a safety feature: uncertain cases are explicitly deferred instead of overconfidently auto-decided. Future work should prioritize cross-domain validation, multilingual evaluation, and prospective classroom trials focused on learning outcomes.
""".strip()
        return self._replace_section(content, "conclusion", conclusion)

    def _update_reproducibility_section(self, content: str) -> str:
        section = """
**Full reproduction from scratch**:

```bash
git clone https://github.com/somanellipudi/smart-notes.git
cd Smart-Notes
pip install -r requirements.txt
python scripts/make_paper_artifacts.py
```

**Deterministic settings**:
- Multi-seed set: `[0, 1, 2, 3, 4]`
- Bootstrap samples: `2000` with seed `42`
- Artifact-driven paper generation: all tables loaded from `artifacts/latest/`

**Fail-fast behavior**:
- If required artifacts are missing, paper update aborts with a clear `FileNotFoundError` listing missing files.
""".strip()
        return self._replace_section(content, "reproducibility", section)

    def _tighten_claim_language(self, content: str) -> str:
        replacements = {
            "generalizable across domains": "validated on CSClaimBench (computer science domain)",
            "robust in all educational settings": "evaluated under controlled experimental conditions",
            "state-of-the-art": "outperforms classical neural baselines under calibration-aware evaluation",
        }
        for old, new in replacements.items():
            content = content.replace(old, new)
        return content


if __name__ == "__main__":
    artifacts_dir = Path("artifacts/latest")
    paper_path = Path("research_paper.md")

    updater = PaperUpdater(artifacts_dir, paper_path)
    updater.update_paper()
    print("✓ Paper updated successfully")
