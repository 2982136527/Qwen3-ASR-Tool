"""ASR backend abstraction layer.

Each backend implements a minimal protocol so ModelManager can swap models
at runtime without the rest of the application caring which engine is loaded.
"""
from .base import ASRResult, TimestampItem, TimestampResult
from .qwen_backend import QwenBackend
from .whisper_backend import WhisperBackend
from .sensevoice_backend import SenseVoiceBackend

__all__ = [
    "ASRResult", "TimestampItem", "TimestampResult",
    "QwenBackend", "WhisperBackend", "SenseVoiceBackend",
]
