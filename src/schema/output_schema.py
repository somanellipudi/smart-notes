"""
Structured output schema definitions using Pydantic.

This module defines strict data models for educational content extraction,
ensuring consistent and validated JSON output from the reasoning pipeline.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator


class Concept(BaseModel):
    """
    Represents a key educational concept.
    
    Attributes:
        name: Concept identifier
        definition: Plain-language explanation
        prerequisites: Required prior knowledge
        difficulty_level: Estimated difficulty (1-5)
    """
    name: str = Field(..., description="Concept name or term")
    definition: str = Field(..., description="Clear, concise explanation")
    prerequisites: List[str] = Field(
        default_factory=list,
        description="Concepts that should be understood first"
    )
    difficulty_level: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Difficulty rating (1=basic, 5=advanced)"
    )


class WorkedExample(BaseModel):
    """
    Represents a solved problem or example from class.
    
    Attributes:
        problem: Problem statement
        solution: Step-by-step solution
        key_concepts: Concepts demonstrated
        common_mistakes: Typical errors to avoid
    """
    problem: str = Field(..., description="Problem statement or question")
    solution: str = Field(..., description="Step-by-step solution explanation")
    key_concepts: List[str] = Field(
        default_factory=list,
        description="Concepts applied in this example"
    )
    common_mistakes: List[str] = Field(
        default_factory=list,
        description="Typical errors students make"
    )


class EquationExplanation(BaseModel):
    """
    Plain-language interpretation of mathematical equations.
    
    Attributes:
        equation: Original equation or formula
        explanation: What the equation means
        variables: Description of each variable
        applications: Where this equation is used
    """
    equation: str = Field(..., description="The mathematical equation")
    explanation: str = Field(
        ...,
        description="Plain-language interpretation"
    )
    variables: Dict[str, str] = Field(
        default_factory=dict,
        description="Variable name -> description mapping"
    )
    applications: List[str] = Field(
        default_factory=list,
        description="Practical applications or use cases"
    )


class Topic(BaseModel):
    """
    Represents a major topic covered in class.
    
    Attributes:
        name: Topic name
        summary: Brief overview
        subtopics: Related subtopics
        learning_objectives: What students should learn
        timestamp_range: When discussed (for video referencing)
    """
    name: str = Field(..., description="Topic name")
    summary: str = Field(..., description="1-2 sentence overview")
    subtopics: List[str] = Field(
        default_factory=list,
        description="Related subtopics or themes"
    )
    learning_objectives: List[str] = Field(
        default_factory=list,
        description="Learning goals for this topic"
    )
    timestamp_range: Optional[Dict[str, float]] = Field(
        None,
        description="Start and end timestamps from lecture"
    )


class FAQ(BaseModel):
    """
    Frequently asked question or common student doubt.
    
    Attributes:
        question: The question or doubt
        answer: Clear, pedagogical answer
        related_concepts: Concepts this question touches on
        difficulty: How advanced the question is
    """
    question: str = Field(..., description="Student question or doubt")
    answer: str = Field(..., description="Clear, helpful answer")
    related_concepts: List[str] = Field(
        default_factory=list,
        description="Concepts related to this question"
    )
    difficulty: str = Field(
        default="medium",
        description="Question difficulty: easy, medium, hard"
    )
    
    @validator('difficulty')
    def validate_difficulty(cls, v):
        """Ensure difficulty is one of allowed values."""
        allowed = {'easy', 'medium', 'hard'}
        if v.lower() not in allowed:
            raise ValueError(f"Difficulty must be one of {allowed}")
        return v.lower()


class Misconception(BaseModel):
    """
    Common misconception or mistake students make.
    
    Attributes:
        misconception: The incorrect belief or approach
        explanation: Why it's wrong
        correct_understanding: What the correct understanding should be
        related_concepts: Concepts this misconception affects
    """
    misconception: str = Field(
        ...,
        description="The incorrect belief or common mistake"
    )
    explanation: str = Field(
        ...,
        description="Why this is incorrect"
    )
    correct_understanding: str = Field(
        ...,
        description="What students should understand instead"
    )
    related_concepts: List[str] = Field(
        default_factory=list,
        description="Concepts involved in this misconception"
    )


class RealWorldConnection(BaseModel):
    """
    Connection between classroom content and real-world applications.
    
    Attributes:
        concept: The concept being connected
        application: Real-world application
        description: How the concept applies
        relevance: Why this matters to students
    """
    concept: str = Field(..., description="Classroom concept")
    application: str = Field(..., description="Real-world application area")
    description: str = Field(
        ...,
        description="How the concept is applied"
    )
    relevance: str = Field(
        ...,
        description="Why students should care about this connection"
    )


class ClassSessionOutput(BaseModel):
    """
    Complete structured output for a single classroom session.
    
    This is the main output schema returned by the reasoning pipeline,
    containing all extracted educational knowledge in a strictly validated format.
    
    Attributes:
        session_id: Unique identifier for this class session
        timestamp: When this session was processed
        class_summary: High-level overview of the class
        topics: Major topics covered
        key_concepts: Important concepts taught
        equation_explanations: Interpretations of equations
        worked_examples: Solved problems from class
        common_mistakes: Misconceptions and errors to avoid
        faqs: Frequently asked questions
        real_world_connections: Practical applications
        metadata: Processing metadata
    """
    session_id: str = Field(
        ...,
        description="Unique identifier for this session"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Processing timestamp"
    )
    class_summary: str = Field(
        ...,
        description="Comprehensive 2-3 paragraph summary of the class"
    )
    topics: List[Topic] = Field(
        default_factory=list,
        description="Major topics covered in class"
    )
    key_concepts: List[Concept] = Field(
        default_factory=list,
        description="Key concepts students should understand"
    )
    equation_explanations: List[EquationExplanation] = Field(
        default_factory=list,
        description="Plain-language interpretations of equations"
    )
    worked_examples: List[WorkedExample] = Field(
        default_factory=list,
        description="Solved problems demonstrated in class"
    )
    common_mistakes: List[Misconception] = Field(
        default_factory=list,
        description="Common misconceptions and errors"
    )
    faqs: List[FAQ] = Field(
        default_factory=list,
        description="Frequently asked questions"
    )
    real_world_connections: List[RealWorldConnection] = Field(
        default_factory=list,
        description="Practical applications of concepts"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Processing metadata (model used, duration, etc.)"
    )
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "session_id": "calc101_2026-01-31_lecture12",
                "class_summary": "Today's lecture introduced the concept of derivatives...",
                "topics": [
                    {
                        "name": "Introduction to Derivatives",
                        "summary": "The derivative as a rate of change.",
                        "subtopics": ["Limit definition", "Notation"],
                        "learning_objectives": [
                            "Understand the limit definition of derivative"
                        ]
                    }
                ],
                "key_concepts": [
                    {
                        "name": "Derivative",
                        "definition": "The rate of change of a function at a point",
                        "prerequisites": ["Functions", "Limits"],
                        "difficulty_level": 3
                    }
                ]
            }
        }
    
    @validator('class_summary')
    def validate_summary_length(cls, v):
        """Ensure summary is substantial."""
        if len(v.strip()) < 50:
            raise ValueError("Class summary must be at least 50 characters")
        return v
    
    def to_json(self, **kwargs) -> str:
        """Export to JSON string."""
        return self.model_dump_json(indent=2, **kwargs)
    
    def to_dict(self) -> Dict:
        """Export to dictionary."""
        return self.model_dump()


class EvaluationResult(BaseModel):
    """
    Evaluation metrics for a session output.
    
    Attributes:
        session_id: Reference to evaluated session
        reasoning_correctness: How accurate the reasoning is (0-1)
        structural_accuracy: How well-formed the output is (0-1)
        hallucination_rate: Estimated rate of hallucinated content (0-1)
        educational_usefulness: Rubric-based score (1-5)
        detailed_feedback: Specific evaluation notes
    """
    session_id: str = Field(..., description="Session being evaluated")
    reasoning_correctness: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Correctness of extracted reasoning (0-1)"
    )
    structural_accuracy: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Well-formedness of output structure (0-1)"
    )
    hallucination_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Estimated hallucination rate (0-1, lower is better)"
    )
    educational_usefulness: float = Field(
        ...,
        ge=1.0,
        le=5.0,
        description="Pedagogical value score (1-5)"
    )
    detailed_feedback: Dict[str, Any] = Field(
        default_factory=dict,
        description="Detailed evaluation notes and breakdown"
    )
    
    def passes_thresholds(self) -> bool:
        """Check if evaluation meets minimum quality thresholds."""
        import config
        return (
            self.reasoning_correctness >= config.MIN_REASONING_CORRECTNESS
            and self.structural_accuracy >= config.MIN_STRUCTURAL_ACCURACY
            and self.hallucination_rate <= config.MAX_HALLUCINATION_RATE
            and self.educational_usefulness >= config.MIN_EDUCATIONAL_USEFULNESS
        )
