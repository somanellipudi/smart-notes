"""Audio processing module for classroom lecture transcription."""

from .transcription import AudioTranscriber, transcribe_audio

__all__ = ["AudioTranscriber", "transcribe_audio"]
