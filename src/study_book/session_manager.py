"""
Session management for Smart Study Book.

This module handles saving, loading, and aggregating classroom session outputs
to build a cumulative knowledge base across multiple class sessions.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from collections import defaultdict

from src.schema.output_schema import ClassSessionOutput, Concept, Topic
import config

logging.basicConfig(level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)


class SessionManager:
    """
    Manages classroom session storage and aggregation.
    
    This class provides functionality for:
    - Saving individual session outputs
    - Loading previous sessions
    - Aggregating knowledge across multiple sessions
    - Building cumulative study materials
    """
    
    def __init__(self, output_dir: Path = None):
        """
        Initialize session manager.
        
        Args:
            output_dir: Directory for storing session outputs (default: config.SESSIONS_DIR)
        """
        self.output_dir = output_dir or config.SESSIONS_DIR
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"SessionManager initialized: {self.output_dir}")
    
    def save_session(
        self,
        output: ClassSessionOutput,
        overwrite: bool = False
    ) -> Path:
        """
        Save session output to disk.
        
        Args:
            output: Session output to save
            overwrite: Whether to overwrite existing file
        
        Returns:
            Path to saved file
        
        Raises:
            FileExistsError: If file exists and overwrite=False
        """
        filename = f"{output.session_id}.json"
        filepath = self.output_dir / filename
        
        if filepath.exists() and not overwrite:
            raise FileExistsError(
                f"Session file already exists: {filepath}. "
                f"Set overwrite=True to replace."
            )
        
        # Save to JSON
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(output.to_json())
        
        logger.info(f"Session saved: {filepath}")
        return filepath
    
    def load_session(self, session_id: str) -> ClassSessionOutput:
        """
        Load session output from disk.
        
        Args:
            session_id: Unique session identifier
        
        Returns:
            ClassSessionOutput object
        
        Raises:
            FileNotFoundError: If session file doesn't exist
        """
        filename = f"{session_id}.json"
        filepath = self.output_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Session not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Reconstruct from dict
        output = ClassSessionOutput(**data)
        
        logger.info(f"Session loaded: {session_id}")
        return output
    
    def list_sessions(self) -> List[str]:
        """
        List all saved session IDs.
        
        Returns:
            List of session IDs
        """
        session_files = list(self.output_dir.glob("*.json"))
        session_ids = [f.stem for f in session_files]
        
        logger.info(f"Found {len(session_ids)} sessions")
        return sorted(session_ids)
    
    def load_all_sessions(self) -> List[ClassSessionOutput]:
        """
        Load all saved sessions.
        
        Returns:
            List of ClassSessionOutput objects
        """
        session_ids = self.list_sessions()
        sessions = []
        
        for session_id in session_ids:
            try:
                session = self.load_session(session_id)
                sessions.append(session)
            except Exception as e:
                logger.error(f"Failed to load session {session_id}: {e}")
        
        logger.info(f"Loaded {len(sessions)} sessions")
        return sessions
    
    def aggregate_concepts(
        self,
        sessions: List[ClassSessionOutput]
    ) -> List[Dict]:
        """
        Aggregate concepts across multiple sessions.
        
        Combines and deduplicates concepts from multiple classes,
        tracking which sessions each concept appeared in.
        
        Args:
            sessions: List of session outputs
        
        Returns:
            List of aggregated concept dictionaries with session references
        """
        concept_map = {}  # name -> concept data
        
        for session in sessions:
            for concept in session.key_concepts:
                name = concept.name.lower()
                
                if name not in concept_map:
                    concept_map[name] = {
                        "name": concept.name,
                        "definitions": [concept.definition],
                        "prerequisites": set(concept.prerequisites),
                        "sessions": [session.session_id],
                        "difficulty_level": concept.difficulty_level
                    }
                else:
                    # Merge information
                    concept_map[name]["definitions"].append(concept.definition)
                    concept_map[name]["prerequisites"].update(concept.prerequisites)
                    concept_map[name]["sessions"].append(session.session_id)
                    # Take max difficulty
                    concept_map[name]["difficulty_level"] = max(
                        concept_map[name]["difficulty_level"],
                        concept.difficulty_level
                    )
        
        # Convert to list and clean up
        aggregated = []
        for name, data in concept_map.items():
            aggregated.append({
                "name": data["name"],
                "definition": data["definitions"][0],  # Take first (or could merge)
                "alternative_definitions": data["definitions"][1:] if len(data["definitions"]) > 1 else [],
                "prerequisites": sorted(list(data["prerequisites"])),
                "sessions": data["sessions"],
                "frequency": len(data["sessions"]),
                "difficulty_level": data["difficulty_level"]
            })
        
        # Sort by frequency (most common first)
        aggregated.sort(key=lambda x: x["frequency"], reverse=True)
        
        logger.info(f"Aggregated {len(aggregated)} unique concepts")
        return aggregated
    
    def aggregate_topics(
        self,
        sessions: List[ClassSessionOutput]
    ) -> List[Dict]:
        """
        Aggregate topics across sessions.
        
        Args:
            sessions: List of session outputs
        
        Returns:
            List of aggregated topic dictionaries
        """
        topic_map = {}
        
        for session in sessions:
            for topic in session.topics:
                name = topic.name.lower()
                
                if name not in topic_map:
                    topic_map[name] = {
                        "name": topic.name,
                        "summaries": [topic.summary],
                        "subtopics": set(topic.subtopics),
                        "learning_objectives": set(topic.learning_objectives),
                        "sessions": [session.session_id]
                    }
                else:
                    topic_map[name]["summaries"].append(topic.summary)
                    topic_map[name]["subtopics"].update(topic.subtopics)
                    topic_map[name]["learning_objectives"].update(topic.learning_objectives)
                    topic_map[name]["sessions"].append(session.session_id)
        
        aggregated = []
        for name, data in topic_map.items():
            aggregated.append({
                "name": data["name"],
                "summary": data["summaries"][0],
                "subtopics": sorted(list(data["subtopics"])),
                "learning_objectives": sorted(list(data["learning_objectives"])),
                "sessions": data["sessions"],
                "frequency": len(data["sessions"])
            })
        
        aggregated.sort(key=lambda x: x["frequency"], reverse=True)
        
        logger.info(f"Aggregated {len(aggregated)} unique topics")
        return aggregated
    
    def build_cumulative_study_guide(
        self,
        session_ids: List[str] = None
    ) -> Dict:
        """
        Build cumulative study guide from multiple sessions.
        
        Creates a comprehensive study guide that combines knowledge
        from multiple class sessions.
        
        Args:
            session_ids: Specific sessions to include (default: all)
        
        Returns:
            Dictionary containing aggregated study materials
        """
        logger.info("Building cumulative study guide")
        
        # Load sessions
        if session_ids:
            sessions = [self.load_session(sid) for sid in session_ids]
        else:
            sessions = self.load_all_sessions()
        
        if not sessions:
            logger.warning("No sessions to aggregate")
            return {}
        
        # Aggregate components
        concepts = self.aggregate_concepts(sessions)
        topics = self.aggregate_topics(sessions)
        
        # Collect all FAQs
        all_faqs = []
        for session in sessions:
            for faq in session.faqs:
                all_faqs.append({
                    "question": faq.question,
                    "answer": faq.answer,
                    "difficulty": faq.difficulty,
                    "session": session.session_id
                })
        
        # Collect all worked examples
        all_examples = []
        for session in sessions:
            for example in session.worked_examples:
                all_examples.append({
                    "problem": example.problem,
                    "solution": example.solution,
                    "key_concepts": example.key_concepts,
                    "session": session.session_id
                })
        
        # Collect all misconceptions
        all_misconceptions = []
        for session in sessions:
            for misc in session.common_mistakes:
                all_misconceptions.append({
                    "misconception": misc.misconception,
                    "explanation": misc.explanation,
                    "correct_understanding": misc.correct_understanding,
                    "session": session.session_id
                })
        
        # Build study guide
        study_guide = {
            "metadata": {
                "title": "Cumulative Study Guide",
                "generated": datetime.now().isoformat(),
                "num_sessions": len(sessions),
                "session_ids": [s.session_id for s in sessions],
                "date_range": {
                    "first": min(s.timestamp for s in sessions).isoformat(),
                    "last": max(s.timestamp for s in sessions).isoformat()
                }
            },
            "topics": topics,
            "concepts": concepts,
            "worked_examples": all_examples,
            "faqs": all_faqs,
            "common_mistakes": all_misconceptions,
            "statistics": {
                "total_topics": len(topics),
                "total_concepts": len(concepts),
                "total_examples": len(all_examples),
                "total_faqs": len(all_faqs),
                "total_misconceptions": len(all_misconceptions)
            }
        }
        
        logger.info(
            f"Study guide built: {len(topics)} topics, {len(concepts)} concepts, "
            f"{len(all_examples)} examples"
        )
        
        return study_guide
    
    def save_study_guide(
        self,
        study_guide: Dict,
        filename: str = "cumulative_study_guide.json"
    ) -> Path:
        """
        Save cumulative study guide to disk.
        
        Args:
            study_guide: Study guide dictionary
            filename: Output filename
        
        Returns:
            Path to saved file
        """
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(study_guide, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Study guide saved: {filepath}")
        return filepath
    
    def get_concept_progression(self, concept_name: str) -> List[Dict]:
        """
        Track how a concept evolved across sessions.
        
        Args:
            concept_name: Name of concept to track
        
        Returns:
            List of concept appearances across sessions
        """
        sessions = self.load_all_sessions()
        progression = []
        
        concept_lower = concept_name.lower()
        
        for session in sessions:
            for concept in session.key_concepts:
                if concept.name.lower() == concept_lower:
                    progression.append({
                        "session_id": session.session_id,
                        "timestamp": session.timestamp.isoformat(),
                        "definition": concept.definition,
                        "prerequisites": concept.prerequisites,
                        "difficulty_level": concept.difficulty_level
                    })
        
        # Sort by timestamp
        progression.sort(key=lambda x: x["timestamp"])
        
        logger.info(f"Concept '{concept_name}' appeared in {len(progression)} sessions")
        return progression


def create_session_manager(output_dir: Path = None) -> SessionManager:
    """
    Convenience function to create session manager.
    
    Args:
        output_dir: Output directory for sessions
    
    Returns:
        SessionManager instance
    """
    return SessionManager(output_dir)
