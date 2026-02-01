"""
Example usage of the Structured Understanding of Classroom Content framework.

This script demonstrates the complete pipeline from raw classroom data to
structured educational output, evaluation, and study guide generation.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.audio.transcription import transcribe_audio
from src.preprocessing.text_processing import preprocess_classroom_content
from src.reasoning.pipeline import ReasoningPipeline
from src.evaluation.metrics import evaluate_session_output
from src.study_book.session_manager import SessionManager


def load_sample_input(filepath: str = None) -> dict:
    """
    Load sample input data from JSON file.
    
    Args:
        filepath: Path to input JSON (default: examples/sample_input.json)
    
    Returns:
        Input data dictionary
    """
    if filepath is None:
        filepath = Path(__file__).parent / "sample_input.json"
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✓ Loaded input data: {filepath}")
    return data


def process_classroom_session(input_data: dict) -> dict:
    """
    Process a complete classroom session through the full pipeline.
    
    This function demonstrates the end-to-end workflow:
    1. Audio transcription
    2. Text preprocessing
    3. Multi-stage reasoning
    4. Quality evaluation
    5. Session storage
    
    Args:
        input_data: Dictionary with classroom data
    Returns:
        Dictionary containing:
            - output: ClassSessionOutput
            - evaluation: EvaluationResult
            - session_path: Path to saved session
    """
    print("\n" + "=" * 70)
    print("STRUCTURED UNDERSTANDING OF CLASSROOM CONTENT")
    print("=" * 70)
    
    session_id = input_data.get("session_id", "demo_session")
    
    # Step 1: Audio Transcription
    print("\n[1/6] Audio Transcription")
    print("-" * 70)
    
    audio_path = input_data.get("lecture_audio_path", "")
    
    # Require a valid audio file if provided
    if audio_path and Path(audio_path).exists():
        print(f"Transcribing: {audio_path}")
        transcription = transcribe_audio(audio_path)
    else:
        raise FileNotFoundError("Audio file not found. Please provide a valid lecture audio path.")
    
    transcript = transcription["transcript"]
    print(f"✓ Transcription complete: {len(transcript)} characters")
    print(f"  Sample: {transcript[:150]}...")
    
    # Step 2: Preprocessing
    print("\n[2/6] Text Preprocessing")
    print("-" * 70)
    
    handwritten_notes = input_data.get("handwritten_notes", "")
    equations = input_data.get("equations", [])
    external_context = input_data.get("external_context", "")
    
    preprocessed = preprocess_classroom_content(
        handwritten_notes=handwritten_notes,
        transcript=transcript,
        equations=equations
    )
    
    combined_text = preprocessed["combined_text"]
    print(f"✓ Preprocessing complete")
    print(f"  Notes: {preprocessed['metadata']['notes_length']} chars")
    print(f"  Transcript: {preprocessed['metadata']['transcript_length']} chars")
    print(f"  Equations: {preprocessed['metadata']['num_equations']}")
    print(f"  Segments: {preprocessed['metadata']['num_segments']}")
    
    # Step 3: Multi-Stage Reasoning
    print("\n[3/6] Multi-Stage Reasoning Pipeline")
    print("-" * 70)
    
    pipeline = ReasoningPipeline()
    
    output = pipeline.process(
        combined_content=combined_text,
        equations=equations,
        external_context=external_context,
        session_id=session_id
    )
    
    print(f"✓ Reasoning complete")
    print(f"  Topics: {len(output.topics)}")
    print(f"  Concepts: {len(output.key_concepts)}")
    print(f"  Equations explained: {len(output.equation_explanations)}")
    print(f"  Worked examples: {len(output.worked_examples)}")
    print(f"  FAQs: {len(output.faqs)}")
    print(f"  Misconceptions: {len(output.common_mistakes)}")
    print(f"  Real-world connections: {len(output.real_world_connections)}")
    
    # Step 4: Evaluation
    print("\n[4/6] Quality Evaluation")
    print("-" * 70)
    
    evaluation = evaluate_session_output(
        output=output,
        source_content=combined_text,
        external_context=external_context
    )
    
    print(f"✓ Evaluation complete")
    print(f"  Reasoning Correctness: {evaluation.reasoning_correctness:.3f}")
    print(f"  Structural Accuracy: {evaluation.structural_accuracy:.3f}")
    print(f"  Hallucination Rate: {evaluation.hallucination_rate:.3f}")
    print(f"  Educational Usefulness: {evaluation.educational_usefulness:.2f}/5.0")
    print(f"  Passes Thresholds: {'✓ PASS' if evaluation.passes_thresholds() else '✗ FAIL'}")
    
    # Step 5: Session Storage
    print("\n[5/6] Session Storage")
    print("-" * 70)
    
    session_manager = SessionManager()
    session_path = session_manager.save_session(output, overwrite=True)
    
    print(f"✓ Session saved: {session_path}")
    
    # Step 6: Display Results
    print("\n[6/6] Results Summary")
    print("-" * 70)
    print("\nClass Summary:")
    print(output.class_summary[:300] + "..." if len(output.class_summary) > 300 else output.class_summary)
    
    if output.topics:
        print(f"\nTopics ({len(output.topics)}):")
        for i, topic in enumerate(output.topics[:3], 1):
            print(f"  {i}. {topic.name}")
            print(f"     {topic.summary}")
    
    if output.key_concepts:
        print(f"\nKey Concepts ({len(output.key_concepts)}):")
        for i, concept in enumerate(output.key_concepts[:3], 1):
            print(f"  {i}. {concept.name}")
            print(f"     {concept.definition[:80]}...")
    
    if output.faqs:
        print(f"\nSample FAQ:")
        faq = output.faqs[0]
        print(f"  Q: {faq.question}")
        print(f"  A: {faq.answer[:100]}...")
    
    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)
    
    return {
        "output": output,
        "evaluation": evaluation,
        "session_path": session_path
    }


def demonstrate_study_guide_generation():
    """
    Demonstrate cumulative study guide generation from multiple sessions.
    """
    print("\n" + "=" * 70)
    print("CUMULATIVE STUDY GUIDE GENERATION")
    print("=" * 70)
    
    session_manager = SessionManager()
    
    # List available sessions
    sessions = session_manager.list_sessions()
    print(f"\nFound {len(sessions)} saved session(s)")
    
    if not sessions:
        print("No sessions available. Process at least one session first.")
        return
    
    # Build study guide
    print("\nBuilding cumulative study guide...")
    study_guide = session_manager.build_cumulative_study_guide()
    
    # Save study guide
    guide_path = session_manager.save_study_guide(study_guide)
    print(f"✓ Study guide saved: {guide_path}")
    
    # Display summary
    print("\nStudy Guide Contents:")
    print(f"  Sessions covered: {study_guide['metadata']['num_sessions']}")
    print(f"  Total topics: {study_guide['statistics']['total_topics']}")
    print(f"  Total concepts: {study_guide['statistics']['total_concepts']}")
    print(f"  Total examples: {study_guide['statistics']['total_examples']}")
    print(f"  Total FAQs: {study_guide['statistics']['total_faqs']}")
    
    # Show most frequent concepts
    if study_guide['concepts']:
        print("\nMost Frequently Covered Concepts:")
        for i, concept in enumerate(study_guide['concepts'][:5], 1):
            print(f"  {i}. {concept['name']} (appeared in {concept['frequency']} session(s))")
    
    print("\n" + "=" * 70)


def main():
    """Main demonstration function."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  Structured Understanding of Classroom Content - Demo".center(68) + "║")
    print("║" + "  Research Prototype".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "═" * 68 + "╝")
    
    # Load sample input
    input_data = load_sample_input()
    
    # Process session
    result = process_classroom_session(input_data)
    
    # Generate study guide
    demonstrate_study_guide_generation()
    
    # Export structured output as JSON
    print("\n" + "=" * 70)
    print("EXPORTING STRUCTURED OUTPUT")
    print("=" * 70)
    
    output_path = config.OUTPUT_DIR / "demo_output.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(result["output"].to_json())
    
    print(f"✓ Full structured output exported: {output_path}")
    
    print("\n" + "=" * 70)
    print("✓ DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nTo view the full output:")
    print(f"  cat {output_path}")
    print("\nTo view the study guide:")
    print(f"  cat {config.OUTPUT_DIR}/sessions/cumulative_study_guide.json")
    print("\n")


if __name__ == "__main__":
    main()
