"""
Error Analysis Module.

Classifies and analyzes prediction errors to understand failure modes:
- retrieval_failure: No relevant evidence retrieved
- ambiguous_claim: Claim is inherently ambiguous
- evidence_mismatch: Evidence doesn't match claim semantics
- overconfidence_error: High confidence but incorrect
- other: Other error types

Generates:
- Error breakdown table (percentages)
- Representative error examples for qualitative analysis
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass
class ErrorExample:
    """Single error example with diagnostic information."""
    claim_id: str
    claim_text: str
    predicted_label: str
    true_label: str
    confidence: float
    error_type: str
    explanation: str
    evidence_summary: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ErrorAnalysisReport:
    """Complete error analysis report."""
    total_errors: int
    error_breakdown: Dict[str, int]  # error_type -> count
    error_percentages: Dict[str, float]  # error_type -> percentage
    examples_by_type: Dict[str, List[ErrorExample]]
    
    def to_dict(self) -> Dict:
        return {
            "total_errors": self.total_errors,
            "error_breakdown": self.error_breakdown,
            "error_percentages": self.error_percentages,
            "examples_by_type": {
                error_type: [ex.to_dict() for ex in examples]
                for error_type, examples in self.examples_by_type.items()
            }
        }


def classify_error(
    claim_text: str,
    predicted_label: str,
    true_label: str,
    confidence: float,
    evidence_count: int = 0,
    evidence_texts: Optional[List[str]] = None
) -> str:
    """
    Classify an error into one of the predefined categories.
    
    Args:
        claim_text: The claim text
        predicted_label: Predicted label
        true_label: True label
        confidence: Prediction confidence
        evidence_count: Number of evidence items retrieved
        evidence_texts: Optional evidence texts for analysis
    
    Returns:
        Error type string
    """
    # Overconfidence error: high confidence but wrong
    if confidence >= 0.8:
        return "overconfidence_error"
    
    # Retrieval failure: no evidence found
    if evidence_count == 0:
        return "retrieval_failure"
    
    # Ambiguous claim: contains hedging language
    hedging_words = ["might", "may", "could", "possibly", "perhaps", "sometimes", 
                     "usually", "often", "generally", "typically"]
    if any(word in claim_text.lower() for word in hedging_words):
        return "ambiguous_claim"
    
    # Evidence mismatch: has evidence but still wrong (and not overconfident)
    if evidence_count > 0 and confidence < 0.8:
        return "evidence_mismatch"
    
    # Default: other
    return "other"


def analyze_errors(
    predictions: List[Dict[str, Any]],
    max_examples_per_type: int = 5
) -> ErrorAnalysisReport:
    """
    Analyze prediction errors and generate report.
    
    Args:
        predictions: List of prediction dicts containing:
            - claim_id: Unique identifier
            - claim_text: Claim text
            - predicted_label: Predicted label
            - true_label: True label
            - confidence: Prediction confidence
            - evidence_count: Number of evidence items (optional)
            - evidence_texts: Evidence texts (optional)
        max_examples_per_type: Maximum examples to save per error type
    
    Returns:
        ErrorAnalysisReport
    """
    logger.info("Analyzing prediction errors...")
    
    # Find errors
    errors = []
    for pred in predictions:
        if pred["predicted_label"] != pred["true_label"]:
            errors.append(pred)
    
    total_errors = len(errors)
    logger.info(f"Found {total_errors} errors out of {len(predictions)} predictions")
    
    if total_errors == 0:
        # No errors - return empty report
        return ErrorAnalysisReport(
            total_errors=0,
            error_breakdown={},
            error_percentages={},
            examples_by_type={}
        )
    
    # Classify errors
    error_classifications = []
    for error in errors:
        error_type = classify_error(
            claim_text=error.get("claim_text", ""),
            predicted_label=error["predicted_label"],
            true_label=error["true_label"],
            confidence=error.get("confidence", 0.5),
            evidence_count=error.get("evidence_count", 0),
            evidence_texts=error.get("evidence_texts", [])
        )
        error_classifications.append(error_type)
        error["error_type"] = error_type
    
    # Count error types
    error_counts = Counter(error_classifications)
    error_breakdown = dict(error_counts)
    
    # Compute percentages
    error_percentages = {
        error_type: (count / total_errors) * 100
        for error_type, count in error_breakdown.items()
    }
    
    # Log breakdown
    logger.info("Error breakdown:")
    for error_type, count in sorted(error_breakdown.items(), key=lambda x: -x[1]):
        pct = error_percentages[error_type]
        logger.info(f"  {error_type}: {count} ({pct:.1f}%)")
    
    # Select representative examples for each type
    examples_by_type = {}
    for error_type in error_breakdown.keys():
        # Get all errors of this type
        errors_of_type = [e for e in errors if e["error_type"] == error_type]
        
        # Sort by confidence (descending) to get most confident errors
        errors_of_type.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)
        
        # Take top N examples
        selected = errors_of_type[:max_examples_per_type]
        
        # Create ErrorExample objects
        examples = []
        for err in selected:
            evidence_summary = None
            if "evidence_texts" in err and err["evidence_texts"]:
                # Truncate evidence for readability
                evidence_summary = " | ".join([
                    (text[:100] + "...") if len(text) > 100 else text
                    for text in err["evidence_texts"][:2]  # Max 2 evidence items
                ])
            
            example = ErrorExample(
                claim_id=err.get("claim_id", "unknown"),
                claim_text=err.get("claim_text", "")[:200],  # Truncate long claims
                predicted_label=err["predicted_label"],
                true_label=err["true_label"],
                confidence=err.get("confidence", 0.0),
                error_type=error_type,
                explanation=generate_error_explanation(err, error_type),
                evidence_summary=evidence_summary
            )
            examples.append(example)
        
        examples_by_type[error_type] = examples
    
    # Create report
    report = ErrorAnalysisReport(
        total_errors=total_errors,
        error_breakdown=error_breakdown,
        error_percentages=error_percentages,
        examples_by_type=examples_by_type
    )
    
    return report


def generate_error_explanation(error: Dict, error_type: str) -> str:
    """Generate human-readable explanation for an error."""
    explanations = {
        "retrieval_failure": "No relevant evidence was retrieved for this claim.",
        "ambiguous_claim": "Claim contains hedging language making it inherently ambiguous.",
        "evidence_mismatch": "Evidence was retrieved but didn't semantically match the claim.",
        "overconfidence_error": f"Model was highly confident ({error.get('confidence', 0):.2f}) but incorrect.",
        "other": "Error does not fit into predefined categories."
    }
    return explanations.get(error_type, "Unknown error type")


def save_error_analysis(
    report: ErrorAnalysisReport,
    output_dir: Path
):
    """
    Save error analysis report to files.
    
    Args:
        report: ErrorAnalysisReport
        output_dir: Directory to save outputs
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save breakdown table (CSV)
    df_breakdown = pd.DataFrame([
        {
            "Error Type": error_type,
            "Count": count,
            "Percentage": f"{report.error_percentages[error_type]:.1f}%"
        }
        for error_type, count in sorted(
            report.error_breakdown.items(),
            key=lambda x: -x[1]
        )
    ])
    df_breakdown.to_csv(output_dir / "error_breakdown.csv", index=False)
    
    # Save as markdown
    with open(output_dir / "error_breakdown.md", 'w') as f:
        f.write(df_breakdown.to_markdown(index=False))
    
    # Save error examples (markdown)
    with open(output_dir / "error_examples.md", 'w') as f:
        f.write("# Error Analysis Examples\n\n")
        f.write(f"**Total Errors**: {report.total_errors}\n\n")
        
        for error_type, examples in report.examples_by_type.items():
            count = report.error_breakdown[error_type]
            pct = report.error_percentages[error_type]
            
            f.write(f"## {error_type.replace('_', ' ').title()}\n\n")
            f.write(f"**Count**: {count} ({pct:.1f}%)\n\n")
            
            for i, example in enumerate(examples, 1):
                f.write(f"### Example {i}\n\n")
                f.write(f"- **Claim**: {example.claim_text}\n")
                f.write(f"- **Predicted**: {example.predicted_label}\n")
                f.write(f"- **True**: {example.true_label}\n")
                f.write(f"- **Confidence**: {example.confidence:.3f}\n")
                f.write(f"- **Explanation**: {example.explanation}\n")
                
                if example.evidence_summary:
                    f.write(f"- **Evidence**: {example.evidence_summary}\n")
                
                f.write("\n")
            
            f.write("---\n\n")
    
    # Save JSON report
    with open(output_dir / "error_analysis_report.json", 'w') as f:
        json.dump(report.to_dict(), f, indent=2)
    
    logger.info(f"Error analysis saved to {output_dir}")


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Generate synthetic predictions with errors
    predictions = []
    error_types = ["retrieval_failure", "ambiguous_claim", "evidence_mismatch", 
                   "overconfidence_error", "other"]
    
    for i in range(100):
        # Simulate correct and incorrect predictions
        is_error = np.random.rand() < 0.2  # 20% error rate
        
        if is_error:
            # Generate error
            error_type = np.random.choice(error_types)
            
            if error_type == "retrieval_failure":
                evidence_count = 0
                confidence = np.random.uniform(0.5, 0.7)
            elif error_type == "overconfidence_error":
                evidence_count = 3
                confidence = np.random.uniform(0.8, 0.95)
            else:
                evidence_count = np.random.randint(1, 5)
                confidence = np.random.uniform(0.5, 0.8)
            
            pred = {
                "claim_id": f"claim_{i}",
                "claim_text": f"Example claim {i} that may or may not be correct",
                "predicted_label": "VERIFIED",
                "true_label": "REJECTED",
                "confidence": confidence,
                "evidence_count": evidence_count,
                "evidence_texts": [f"Evidence {j}" for j in range(evidence_count)]
            }
        else:
            # Correct prediction
            pred = {
                "claim_id": f"claim_{i}",
                "claim_text": f"Example claim {i}",
                "predicted_label": "VERIFIED",
                "true_label": "VERIFIED",
                "confidence": np.random.uniform(0.7, 0.95),
                "evidence_count": np.random.randint(1, 5),
                "evidence_texts": [f"Evidence {j}" for j in range(np.random.randint(1, 5))]
            }
        
        predictions.append(pred)
    
    # Analyze errors
    report = analyze_errors(predictions, max_examples_per_type=3)
    
    # Save report
    save_error_analysis(report, Path("artifacts/error_analysis_test"))
    
    print(f"\nError Analysis Complete:")
    print(f"Total Errors: {report.total_errors}")
    print("\nBreakdown:")
    for error_type, count in report.error_breakdown.items():
        print(f"  {error_type}: {count} ({report.error_percentages[error_type]:.1f}%)")
