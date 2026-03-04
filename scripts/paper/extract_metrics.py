#!/usr/bin/env python3
"""
Quick metric extraction utility for paper writing.
Extracts specific evaluation results for copy-paste into manuscripts.

Usage Examples:
    python extract_metrics.py accuracy
    python extract_metrics.py ablation
    python extract_metrics.py calibration --format latex
    python extract_metrics.py domain-performance --format csv
    python extract_metrics.py all --format markdown
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List


class MetricExtractor:
    def __init__(self, results_file: str = "evaluation_results.json"):
        self.results_path = Path(results_file)
        if not self.results_path.exists():
            raise FileNotFoundError(f"Results file not found: {results_file}")
        
        with open(self.results_path) as f:
            self.data = json.load(f)
    
    def extract_core_results(self, fmt: str = "text") -> str:
        """Extract overall accuracy, F1, calibration metrics."""
        core = self.data["core_results"]
        
        if fmt == "text":
            return f"""
CORE RESULTS
============
Accuracy:               {core['overall_accuracy']:.1%}
F1 Score:               {core['overall_f1_score']:.3f}
Calibration (ECE):      {core['calibration_ece']:.4f}
Brier Score:            {core['brier_score']:.3f}
Inference Time:         {core['inference_time_ms']}ms/claim
Improvement:            +{core['improvement_over_baseline_pp']:.1f}pp vs baseline
"""
        
        elif fmt == "latex":
            return f"""
\\begin{{table}}
\\centering
\\begin{{tabular}}{{lr}}
\\toprule
\\textbf{{Metric}} & \\textbf{{Value}} \\\\
\\midrule
Accuracy & {core['overall_accuracy']:.1%} \\\\
F1 Score & {core['overall_f1_score']:.3f} \\\\
ECE (Calibration) & {core['calibration_ece']:.4f} \\\\
Improvement over Baseline & +{core['improvement_over_baseline_pp']:.1f}pp \\\\
\\bottomrule
\\end{{tabular}}
\\caption{{Overall Smart Notes Performance on CSClaimBench v1.0}}
\\end{{table}}
"""
        
        elif fmt == "csv":
            return "Metric,Value\n" + \
                   f"Accuracy,{core['overall_accuracy']:.1%}\n" + \
                   f"F1 Score,{core['overall_f1_score']:.3f}\n" + \
                   f"ECE,{core['calibration_ece']:.4f}\n" + \
                   f"Brier Score,{core['brier_score']:.3f}\n"
        
        elif fmt == "json":
            return json.dumps(core, indent=2)
    
    def extract_ablation(self, fmt: str = "text") -> str:
        """Extract ablation study results."""
        ablations = self.data["ablation_results"]
        
        if fmt == "text":
            lines = ["ABLATION STUDY RESULTS", "=" * 80]
            for ab in ablations:
                lines.append(f"\n{ab['config_name']}: {ab['description']}")
                lines.append(f"  Accuracy:     {ab['accuracy']:.1%}")
                lines.append(f"  F1 Score:     {ab['f1_verified']:.3f}")
                lines.append(f"  ECE:          {ab['ece']:.4f}")
                lines.append(f"  Inference:    {ab['avg_time_per_claim_ms']}ms")
                if 'delta_from_baseline_pp' in ab:
                    lines.append(f"  Δ Baseline:   +{ab['delta_from_baseline_pp']:.1f}pp")
            return "\n".join(lines) + "\n"
        
        elif fmt == "latex":
            lines = [
                "\\begin{table}",
                "\\centering",
                "\\begin{tabular}{lrrrr}",
                "\\toprule",
                "\\textbf{Configuration} & \\textbf{Accuracy} & \\textbf{F1} & \\textbf{ECE} & \\textbf{Time (ms)} \\\\",
                "\\midrule"
            ]
            for ab in ablations:
                lines.append(
                    f"{ab['config_name']} & {ab['accuracy']:.1%} & {ab['f1_verified']:.3f} & {ab['ece']:.4f} & {ab['avg_time_per_claim_ms']} \\\\"
                )
            lines.extend([
                "\\bottomrule",
                "\\end{tabular}",
                "\\caption{Ablation Study: Component Contribution to Accuracy}",
                "\\end{table}"
            ])
            return "\n".join(lines)
        
        elif fmt == "csv":
            lines = ["Configuration,Accuracy,F1,ECE,Time_ms"]
            for ab in ablations:
                lines.append(
                    f"{ab['config_name']},{ab['accuracy']:.3f},{ab['f1_verified']:.3f},{ab['ece']:.4f},{ab['avg_time_per_claim_ms']}"
                )
            return "\n".join(lines)
        
        elif fmt == "json":
            return json.dumps(ablations, indent=2)
    
    def extract_domain_performance(self, fmt: str = "text") -> str:
        """Extract per-domain performance breakdown."""
        domains = self.data["domain_performance"]["results"]
        
        if fmt == "text":
            lines = ["DOMAIN PERFORMANCE BREAKDOWN", "=" * 80]
            for domain in domains:
                lines.append(
                    f"\n{domain['domain']:25} (n={domain['num_claims']:3}): "
                    f"{domain['accuracy']:.1%} accuracy, F1: {domain['f1_score']:.3f}"
                )
            return "\n".join(lines) + "\n"
        
        elif fmt == "latex":
            lines = [
                "\\begin{table}",
                "\\centering",
                "\\begin{tabular}{lrrr}",
                "\\toprule",
                "\\textbf{Domain} & \\textbf{N Claims} & \\textbf{Accuracy} & \\textbf{F1} \\\\",
                "\\midrule"
            ]
            for domain in domains:
                lines.append(
                    f"{domain['domain']} & {domain['num_claims']} & {domain['accuracy']:.1%} & {domain['f1_score']:.3f} \\\\"
                )
            lines.extend([
                "\\bottomrule",
                "\\end{tabular}",
                "\\caption{Accuracy by Computer Science Domain}",
                "\\end{table}"
            ])
            return "\n".join(lines)
        
        elif fmt == "csv":
            lines = ["Domain,N_Claims,Accuracy,F1,Difficulty"]
            for domain in domains:
                lines.append(
                    f"{domain['domain']},{domain['num_claims']},{domain['accuracy']:.3f},"
                    f"{domain['f1_score']:.3f},{domain['difficulty']}"
                )
            return "\n".join(lines)
        
        elif fmt == "json":
            return json.dumps(domains, indent=2)
    
    def extract_calibration(self, fmt: str = "text") -> str:
        """Extract calibration analysis."""
        cal = self.data["calibration_analysis"]
        
        if fmt == "text":
            pre = cal["pre_calibration"]
            post = cal["post_calibration"]
            
            return f"""
CALIBRATION ANALYSIS
====================
Pre-Calibration:
  ECE:              {pre['ece']:.4f}
  Avg Confidence:   {pre['avg_confidence']:.2f}
  Actual Accuracy:  {pre['actual_accuracy']:.1%}
  Issue:            {pre['issue']}

Post-Calibration (Temperature Scaling T=1.2):
  ECE:              {post['ece']:.4f}
  Avg Confidence:   {post['avg_confidence']:.2f}
  Actual Accuracy:  {post['actual_accuracy']:.1%}
  Improvement:      {post['improvement_percent']:.0f}% ECE reduction
  Result:           {post['interpretation']}
"""
        
        elif fmt == "latex":
            pre = cal["pre_calibration"]
            post = cal["post_calibration"]
            return f"""
\\begin{{table}}
\\centering
\\begin{{tabular}}{{lrr}}
\\toprule
\\textbf{{Metric}} & \\textbf{{Pre-Cal}} & \\textbf{{Post-Cal}} \\\\
\\midrule
ECE & {pre['ece']:.4f} & {post['ece']:.4f} \\\\
Avg Confidence & {pre['avg_confidence']:.2f} & {post['avg_confidence']:.2f} \\\\
Actual Accuracy & {pre['actual_accuracy']:.1%} & {post['actual_accuracy']:.1%} \\\\
\\bottomrule
\\end{{tabular}}
\\caption{{Calibration Improvement with Temperature Scaling (T=1.2)}}
\\end{{table}}
"""
        elif fmt == "json":
            return json.dumps(cal, indent=2)
    
    def extract_deployment(self, fmt: str = "text") -> str:
        """Extract real-world deployment results."""
        deploy = self.data["real_world_deployment"]
        acc = deploy["accuracy_assessment"]
        time_save = deploy["time_savings"]
        learn_out = deploy["learning_outcomes"]
        
        if fmt == "text":
            return f"""
REAL-WORLD DEPLOYMENT RESULTS
=============================
Scope: {deploy['students']} students in {deploy['scope']}
Submissions: {deploy['total_submissions']}, Total Claims: {deploy['total_claims_verified']:,}

Accuracy Assessment (Faculty Review):
  Verified Verdicts Correct:    {acc['verified_verdicts_correct_percent']:.1%}
  Contradicted Verdicts Correct: {acc['contradicted_verdicts_correct_percent']:.1%}
  Faculty Confidence with System: {acc['faculty_confidence_with_system_percent']}% (vs {acc['faculty_confidence_before_percent']}% before)

Time Savings:
  Grading Time (Before):  {time_save['grading_time_before_minutes']} minutes
  Grading Time (After):   {time_save['grading_time_after_minutes']} minutes
  Efficiency Improvement: {time_save['efficiency_improvement_percent']}%
  Faculty Time Released:  {time_save['faculty_time_released_per_semester_hours']} hours/semester

Learning Outcomes:
  Quiz Score Improvement:         +{learn_out['quiz_improvement_pp']:.1f}pp
  Unsupported Claims Reduction:   -{learn_out['unsupported_claims_reduction_percent']}%
  Well-Cited Claims Increase:     +{learn_out['well_cited_claims_increase_percent']}%
  Student Satisfaction:           {learn_out['satisfaction_percent']}%
"""
        elif fmt == "json":
            return json.dumps(deploy, indent=2)
    
    def extract_statistical_significance(self, fmt: str = "text") -> str:
        """Extract statistical significance test results."""
        sig = self.data["statistical_significance"]
        improvement = sig["improvement_over_baseline"]
        
        if fmt == "text":
            return f"""
STATISTICAL SIGNIFICANCE
========================
Improvement Over Baseline:
  Point Estimate:   +{improvement['point_estimate_pp']:.1f}pp
  95% CI:           +{improvement['ci_lower_pp']:.1f}pp to +{improvement['ci_upper_pp']:.1f}pp
  p-value:          {improvement['p_value']}
  Significance:     {improvement['significance'].upper()}

Ablation F-Tests:
"""  + "\n".join([
                f"  {test['component']:20} F={test['f_statistic']:6.1f}, p={test['p_value']}, {test['significance'].upper()}"
                for test in sig["ablation_f_tests"]
            ])
        
        elif fmt == "latex":
            return f"""
The improvement over baseline is highly significant ($\\Delta = {improvement['point_estimate_pp']:.1f}$pp, 
95\\% CI: [{improvement['ci_lower_pp']:.1f}, {improvement['ci_upper_pp']:.1f}]pp, $p < 0.001$).
"""
        elif fmt == "json":
            return json.dumps(sig, indent=2)
    
    def extract_all(self, fmt: str = "text") -> str:
        """Extract all metrics."""
        sections = [
            self.extract_core_results(fmt),
            self.extract_ablation(fmt),
            self.extract_domain_performance(fmt),
            self.extract_calibration(fmt),
            self.extract_deployment(fmt),
            self.extract_statistical_significance(fmt),
        ]
        return "\n".join(sections)


def main():
    parser = argparse.ArgumentParser(
        description="Extract evaluation metrics for paper writing"
    )
    parser.add_argument(
        "metrics",
        choices=[
            "accuracy", "core",
            "ablation", "abln",
            "domain", "domains",
            "calibration", "calib",
            "deployment", "deploy", "real-world",
            "significance", "stat", "stats",
            "all"
        ],
        help="Metrics to extract"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["text", "latex", "csv", "json"],
        default="text",
        help="Output format"
    )
    parser.add_argument(
        "--file",
        default="evaluation_results.json",
        help="Results file path"
    )
    
    args = parser.parse_args()
    
    try:
        extractor = MetricExtractor(args.file)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    
    # Map aliases
    metric_map = {
        "accuracy": "core",
        "core": "core",
        "ablation": "ablation",
        "abln": "ablation",
        "domain": "domain",
        "domains": "domain",
        "calibration": "calibration",
        "calib": "calibration",
        "deployment": "deployment",
        "deploy": "deployment",
        "real-world": "deployment",
        "significance": "significance",
        "stat": "significance",
        "stats": "significance",
        "all": "all",
    }
    
    metric = metric_map[args.metrics]
    
    # Extract
    if metric == "core":
        output = extractor.extract_core_results(args.format)
    elif metric == "ablation":
        output = extractor.extract_ablation(args.format)
    elif metric == "domain":
        output = extractor.extract_domain_performance(args.format)
    elif metric == "calibration":
        output = extractor.extract_calibration(args.format)
    elif metric == "deployment":
        output = extractor.extract_deployment(args.format)
    elif metric == "significance":
        output = extractor.extract_statistical_significance(args.format)
    elif metric == "all":
        output = extractor.extract_all(args.format)
    
    print(output)
    return 0


if __name__ == "__main__":
    exit(main())
