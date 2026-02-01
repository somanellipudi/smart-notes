"""
Audio transcription module using OpenAI Whisper.

This module handles loading and transcribing classroom lecture audio files,
providing clean text transcripts for downstream educational content extraction.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Union
import logging

try:
    import whisper
    import torch
    import librosa
    import soundfile as sf
except ImportError:
    whisper = None
    torch = None
    librosa = None
    sf = None

import config

logging.basicConfig(level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)


class AudioTranscriber:
    """
    Transcribes lecture audio files to text using OpenAI Whisper ASR model.
    
    This class provides automatic speech recognition capabilities optimized for
    educational content, with support for various audio formats and quality levels.
    
    Attributes:
        model: Loaded Whisper model instance
        device: Computation device (cuda/cpu)
        model_size: Whisper model variant (tiny/base/small/medium/large)
    """
    
    def __init__(self, model_size: str = None):
        """
        Initialize the audio transcriber with specified Whisper model.
        
        Args:
            model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
                       Defaults to config.WHISPER_MODEL_SIZE
        """
        if whisper is None:
            raise ImportError(
                "Whisper is not installed. Install with: pip install openai-whisper"
            )
        
        self.model_size = model_size or config.WHISPER_MODEL_SIZE
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Loading Whisper model '{self.model_size}' on {self.device}")
        self.model = whisper.load_model(self.model_size, device=self.device)
        logger.info("Whisper model loaded successfully")
    
    def load_audio(self, audio_path: Union[str, Path]) -> Dict:
        """
        Load and validate audio file.
        
        Args:
            audio_path: Path to audio file (.wav, .mp3, .m4a, .flac, etc.)
        
        Returns:
            Dictionary containing audio metadata
        
        Raises:
            FileNotFoundError: If audio file doesn't exist
            ValueError: If audio duration exceeds maximum allowed
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Get audio duration without loading entire file
        duration = librosa.get_duration(path=str(audio_path))
        
        if duration > config.MAX_AUDIO_DURATION:
            logger.warning(
                f"Audio duration ({duration:.1f}s) exceeds maximum "
                f"({config.MAX_AUDIO_DURATION}s)"
            )
        
        logger.info(f"Audio file loaded: {audio_path.name} (duration: {duration:.1f}s)")
        
        return {
            "path": str(audio_path),
            "duration": duration,
            "format": audio_path.suffix
        }
    
    def transcribe(
        self,
        audio_path: Union[str, Path],
        language: str = "en",
        return_timestamps: bool = True
    ) -> Dict:
        """
        Transcribe audio file to text with optional timestamps.
        
        This method uses Whisper to perform automatic speech recognition,
        returning both the full transcript and segmented text with timestamps
        for downstream topic boundary detection.
        
        Args:
            audio_path: Path to lecture audio file
            language: ISO language code (default: 'en' for English)
            return_timestamps: Whether to include word-level timestamps
        
        Returns:
            Dictionary containing:
                - transcript: Full cleaned transcript text
                - segments: List of segments with timestamps and text
                - language: Detected/specified language
                - duration: Audio duration in seconds
        
        Example:
            >>> transcriber = AudioTranscriber()
            >>> result = transcriber.transcribe("lecture.wav")
            >>> print(result["transcript"])
            "Today we'll discuss derivatives..."
        """
        audio_path = Path(audio_path)
        
        # Validate audio file
        audio_info = self.load_audio(audio_path)
        
        logger.info(f"Starting transcription of {audio_path.name}")
        
        # Transcribe using Whisper
        result = self.model.transcribe(
            str(audio_path),
            language=language,
            verbose=False if not config.ENABLE_DEBUG else True,
            word_timestamps=return_timestamps
        )
        
        # Extract clean transcript
        transcript = result["text"].strip()
        
        # Extract segments with timestamps
        segments = []
        if "segments" in result:
            for seg in result["segments"]:
                segments.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"].strip(),
                    "confidence": seg.get("confidence", None)
                })
        
        logger.info(
            f"Transcription complete: {len(transcript)} characters, "
            f"{len(segments)} segments"
        )
        
        return {
            "transcript": transcript,
            "segments": segments,
            "language": result.get("language", language),
            "duration": audio_info["duration"]
        }
    
    def transcribe_batch(self, audio_paths: list) -> list:
        """
        Transcribe multiple audio files in batch.
        
        Args:
            audio_paths: List of paths to audio files
        
        Returns:
            List of transcription results (same format as transcribe())
        """
        results = []
        for audio_path in audio_paths:
            try:
                result = self.transcribe(audio_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to transcribe {audio_path}: {e}")
                results.append(None)
        
        return results


# Convenience function for simple use cases
def transcribe_audio(audio_path: Union[str, Path]) -> Dict:
    """
    Convenience function to transcribe audio without instantiating class.
    
    Args:
        audio_path: Path to audio file
    
    Returns:
        Transcription result dictionary
    """
    if whisper is None:
        raise ImportError("Whisper is not installed. Install with: pip install openai-whisper")
    
    transcriber = AudioTranscriber()
    return transcriber.transcribe(audio_path)
